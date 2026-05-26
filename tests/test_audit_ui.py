"""Tests for the audit UI (live region, event queue, non-TTY fallback)."""

import io
import threading
import time

from rich.console import Console

from swival import audit_ui, fmt


def _swap_console(monkeypatch, *, force_terminal: bool):
    buf = io.StringIO()
    console = Console(
        file=buf,
        force_terminal=force_terminal,
        no_color=True,
        width=120,
        record=True,
    )
    monkeypatch.setattr(fmt, "_console", console)
    return buf, console


class TestNonTtyFallback:
    """Non-TTY: output must stay byte-identical to historical fmt.info/warning
    behavior. The pipeline owns its own legacy phase banners and per-batch
    progress lines via ``if not ui.is_live``; AuditUI itself must not add
    anything extra."""

    def test_phase_does_not_print_anything(self, monkeypatch):
        buf, _ = _swap_console(monkeypatch, force_terminal=False)
        with audit_ui.AuditUI(
            run_id="abcd1234",
            branch="main",
            commit="deadbeefcafe",
            workers=2,
            total_files=10,
        ) as ui:
            assert ui.is_live is False
            ph = ui.phase("Phase X · Demo", total=3, color="cyan")
            ph.advance()
            ph.advance()
            ph.advance()
            ph.complete("done")
        assert buf.getvalue() == ""

    def test_finding_is_silent(self, monkeypatch):
        buf, _ = _swap_console(monkeypatch, force_terminal=False)
        with audit_ui.AuditUI(
            run_id="r1",
            branch="main",
            commit="c",
            workers=1,
            total_files=1,
        ) as ui:
            ui.finding("high", "SSRF in fetch", "swival/fetch.py")
        assert buf.getvalue() == ""

    def test_summary_is_silent(self, monkeypatch):
        buf, _ = _swap_console(monkeypatch, force_terminal=False)
        with audit_ui.AuditUI(
            run_id="r1",
            branch="main",
            commit="c",
            workers=1,
            total_files=1,
        ) as ui:
            ui.tally(verified=2, discarded=1, failed=0)
            ui.summary(artifact_dir="audit-findings", written=2)
        assert buf.getvalue() == ""

    def test_disabled_mode_does_not_accumulate_queue_events(self, monkeypatch):
        """No render thread runs in non-TTY mode, so PhaseHandle.advance()
        and friends must drop events rather than letting the queue grow
        unboundedly for the lifetime of the audit run."""
        _swap_console(monkeypatch, force_terminal=False)
        with audit_ui.AuditUI(
            run_id="r1",
            branch="main",
            commit="c",
            workers=2,
            total_files=1,
        ) as ui:
            assert ui.is_live is False
            ph = ui.phase("Phase X · Demo", total=100, color="cyan")
            for _ in range(100):
                ph.advance(current="x")
            ph.complete("done")
            ui.worker_started(1, "x")
            ui.worker_ended(1)
            ui.finding("high", "Bug", "x.py")
            ui.summary(artifact_dir="audit-findings", written=0)
        assert ui._queue.qsize() == 0

    def test_worker_events_are_silent(self, monkeypatch):
        buf, _ = _swap_console(monkeypatch, force_terminal=False)
        with audit_ui.AuditUI(
            run_id="r1",
            branch="main",
            commit="c",
            workers=2,
            total_files=1,
        ) as ui:
            ui.worker_started(1, "a.py")
            ui.worker_ended(1)
        assert buf.getvalue() == ""

    def test_scrollback_routes_to_fmt_info(self, monkeypatch):
        buf, _ = _swap_console(monkeypatch, force_terminal=False)
        with audit_ui.AuditUI(
            run_id="r1",
            branch="main",
            commit="c",
            workers=1,
            total_files=1,
        ) as ui:
            ui.scrollback("  triage progress: 5/10")
        assert "triage progress: 5/10" in buf.getvalue()

    def test_warning_routes_to_fmt_warning(self, monkeypatch):
        buf, _ = _swap_console(monkeypatch, force_terminal=False)
        with audit_ui.AuditUI(
            run_id="r1",
            branch="main",
            commit="c",
            workers=1,
            total_files=1,
        ) as ui:
            ui.warning("config load failed")
        out = buf.getvalue()
        assert "Warning" in out
        assert "config load failed" in out


class TestTtyMode:
    def test_live_region_starts_and_stops(self, monkeypatch):
        buf, console = _swap_console(monkeypatch, force_terminal=True)
        with audit_ui.AuditUI(
            run_id="r1",
            branch="main",
            commit="c0ffee",
            workers=2,
            total_files=4,
        ) as ui:
            assert ui.is_live is True
            ph = ui.phase("Phase 2 · Triage", total=4, color="blue")
            ph.advance(current="a.py")
            ph.advance(current="b.py")
            ui.worker_started(1, "a.py")
            ui.worker_started(2, "b.py")
            ui.finding("critical", "Auth bypass", "swival/auth.py")
            ui.worker_ended(1)
            ui.worker_ended(2)
            ph.complete("2/4 done")
            time.sleep(0.2)  # let the render thread tick
        # After exit, console.export_text() reflects what was rendered.
        captured = console.export_text(clear=False)
        assert "Phase 2 · Triage" in captured
        assert "Auth bypass" in captured

    def test_summary_panel_renders_severity(self, monkeypatch):
        """The summary's "By severity" row counts verified findings only,
        sourced from severity-tagged tally() calls. Ticker-only ui.finding()
        events (proposal-time or "VERIFIED:" echoes) do not affect it."""
        buf, console = _swap_console(monkeypatch, force_terminal=True)
        with audit_ui.AuditUI(
            run_id="r1",
            branch="main",
            commit="c",
            workers=1,
            total_files=1,
        ) as ui:
            ui.finding("high", "Bug A (proposal)", "x.py")
            ui.finding("high", "Bug A (proposal duplicate)", "x.py")
            ui.tally(verified=1, severity="high")
            ui.finding("high", "VERIFIED: Bug A", "x.py")
            ui.finding("critical", "Bug B (proposal)", "y.py")
            ui.tally(verified=1, severity="critical")
            ui.finding("critical", "VERIFIED: Bug B", "y.py")
            ui.summary(artifact_dir="audit-findings", written=2)
        out = console.export_text(clear=False)
        assert "Audit complete" in out
        assert "2 verified" in out
        assert "audit-findings" in out
        assert "high: 1" in out
        assert "critical: 1" in out

    def test_summary_blocks_until_rendered(self, monkeypatch):
        """summary() must not return until the render thread has emitted
        the panel. Otherwise __exit__ could tear down Live before the panel
        ever shows up."""
        buf, console = _swap_console(monkeypatch, force_terminal=True)
        with audit_ui.AuditUI(
            run_id="r1",
            branch="main",
            commit="c",
            workers=1,
            total_files=1,
        ) as ui:
            ui.summary(artifact_dir="audit-findings", written=0)
            captured_at_return = console.export_text(clear=False)
        assert "Audit complete" in captured_at_return


class TestPause:
    """pause() must be reference-counted so concurrent verifier workers
    don't restart Live while a sibling worker's child agent loop is still
    writing to stderr."""

    def test_overlapping_pause_keeps_paused_until_last_exits(self, monkeypatch):
        _swap_console(monkeypatch, force_terminal=True)
        with audit_ui.AuditUI(
            run_id="r1",
            branch="main",
            commit="c",
            workers=2,
            total_files=2,
        ) as ui:
            assert ui._paused is False
            outer = ui.pause()
            outer.__enter__()
            assert ui._paused is True
            inner = ui.pause()
            inner.__enter__()
            assert ui._paused is True
            # First entrant exits while second is still inside the context.
            outer.__exit__(None, None, None)
            assert ui._paused is True, (
                "Live must stay paused while a nested pause is still active"
            )
            inner.__exit__(None, None, None)
            assert ui._paused is False
            assert ui._pause_depth == 0

    def test_concurrent_pause_serializes_stop_start(self, monkeypatch):
        _swap_console(monkeypatch, force_terminal=True)
        with audit_ui.AuditUI(
            run_id="r1",
            branch="main",
            commit="c",
            workers=4,
            total_files=4,
        ) as ui:
            start_barrier = threading.Barrier(4)
            mid_barrier = threading.Barrier(4)
            errors: list[BaseException] = []
            paused_observations: list[bool] = []
            lock = threading.Lock()

            def _worker():
                try:
                    start_barrier.wait()
                    with ui.pause():
                        mid_barrier.wait()
                        with lock:
                            paused_observations.append(ui._paused)
                except BaseException as e:
                    with lock:
                        errors.append(e)

            threads = [threading.Thread(target=_worker) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert errors == []
            assert paused_observations == [True, True, True, True]
            assert ui._paused is False
            assert ui._pause_depth == 0


class TestPauseHeartbeat:
    """While Live is paused, a daemon thread must keep printing per-worker
    liveness to stderr. Without it the panel disappears and verifier child
    agent loops (which run with verbose=False) leave the terminal frozen."""

    def test_heartbeat_prints_while_paused(self, monkeypatch):
        buf, _ = _swap_console(monkeypatch, force_terminal=True)
        with audit_ui.AuditUI(
            run_id="r1",
            branch="main",
            commit="c",
            workers=2,
            total_files=2,
        ) as ui:
            ui._heartbeat_interval = 0.05
            ui.worker_started(1, "finding A")
            ui.worker_started(2, "finding B")
            time.sleep(0.1)
            with ui.pause():
                time.sleep(0.2)
            time.sleep(0.05)
        out = buf.getvalue()
        assert "verifier worker 1" in out
        assert "verifier worker 2" in out
        assert "still running" in out
        assert "finding A" in out
        assert "finding B" in out

    def test_heartbeat_stops_after_last_pause_exits(self, monkeypatch):
        _swap_console(monkeypatch, force_terminal=True)
        with audit_ui.AuditUI(
            run_id="r1",
            branch="main",
            commit="c",
            workers=1,
            total_files=1,
        ) as ui:
            ui._heartbeat_interval = 0.05
            with ui.pause():
                assert ui._heartbeat_thread is not None
            assert ui._heartbeat_thread is None


class TestConcurrency:
    """Workers must be safe to call from many threads at once."""

    def test_many_threads_no_exceptions(self, monkeypatch):
        _swap_console(monkeypatch, force_terminal=True)
        errors: list[BaseException] = []
        with audit_ui.AuditUI(
            run_id="r1",
            branch="main",
            commit="c",
            workers=8,
            total_files=80,
        ) as ui:
            ph = ui.phase("Stress · Triage", total=80, color="blue")

            def _hammer(slot: int) -> None:
                try:
                    for i in range(10):
                        ui.worker_started(slot, f"item-{slot}-{i}")
                        ph.advance(current=f"item-{slot}-{i}")
                        if i % 3 == 0:
                            ui.finding("low", f"finding {slot}-{i}", "p.py")
                        if i % 5 == 0:
                            ui.warning(f"flap {slot}-{i}")
                        ui.worker_ended(slot)
                except BaseException as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=_hammer, args=(slot,)) for slot in range(1, 9)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            ph.complete("done")
        assert errors == []

"""Tests for the REPL front end: output routing, queueing, interrupts,
suspension and the status region."""

import contextlib
import io
import queue
import signal
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from prompt_toolkit.document import Document
from rich.console import Console
from rich.text import Text

from conftest import styled_console
from swival import fmt
from swival.repl_ui import (
    ReplEvent,
    ReplUI,
    _CommandSlot,
    _interrupt_main_thread,
    _is_exit_command,
    _OutputRouter,
    _render_ansi,
    _ShrinkWrap,
    _SpinnerSlot,
    _StreamSlot,
    _with_right_hint,
)


class _ImmediateLoop:
    """Stand-in event loop that runs callbacks on the calling thread."""

    def __init__(self):
        self.calls = 0

    def call_soon_threadsafe(self, fn, *args, context=None):
        self.calls += 1
        fn(*args)


class _FakeApp:
    def __init__(self):
        self.loop = _ImmediateLoop()
        self.results = queue.Queue()

    @property
    def is_done(self):
        return False

    def exit(self, *, result=None, exception=None, style=""):
        self.results.put(result)

    def invalidate(self):
        pass


class _FakeSession:
    """Scripted prompt session.

    Each script entry is a string to return, an exception instance to raise,
    or ``"<block>"`` to behave like a real prompt: call ``pre_run`` and wait
    until ``app.exit()`` supplies a result.
    """

    def __init__(self, script):
        self.script = list(script)
        self.app = _FakeApp()
        self.default_buffer = SimpleNamespace(text="", document=Document(""))
        self.calls = []

    def prompt(self, message=None, *, default="", pre_run=None):
        self.calls.append(default)
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, BaseException):
                raise item
            if item != "<block>":
                return item
        if pre_run is not None:
            pre_run()
        return self.app.results.get(timeout=5)


def _make_ui(session, *, conversation=False):
    return ReplUI(session, has_conversation=lambda: conversation)


# ---------------------------------------------------------------------------
# Output router
# ---------------------------------------------------------------------------


class TestOutputRouter:
    def test_direct_write_when_no_prompt(self):
        router = _OutputRouter()
        err = io.StringIO()
        router.set_target("stderr", err)
        router.write("stderr", "hello\n")
        router.write("stderr", "partial")
        assert err.getvalue() == "hello\npartial"

    def test_partial_lines_wait_while_prompt_active(self):
        router = _OutputRouter()
        err = io.StringIO()
        router.set_target("stderr", err)
        loop = _ImmediateLoop()
        router.prompt_started(object(), loop, None)
        router.write("stderr", "abc")
        assert err.getvalue() == ""
        router.write("stderr", "d\nef")
        assert err.getvalue() == "abcd\n"
        router.prompt_finished()
        assert err.getvalue() == "abcd\nef"

    def test_order_preserved_across_channels(self):
        router = _OutputRouter()
        out, err = io.StringIO(), io.StringIO()
        router.set_target("stdout", out)
        router.set_target("stderr", err)
        router.prompt_started(object(), _ImmediateLoop(), None)
        router.write("stderr", "one\n")
        router.write("stdout", "two\n")
        router.write("stderr", "three\n")
        assert err.getvalue() == "one\nthree\n"
        assert out.getvalue() == "two\n"

    def test_flush_partial_pushes_incomplete_line(self):
        router = _OutputRouter()
        err = io.StringIO()
        router.set_target("stderr", err)
        router.prompt_started(object(), _ImmediateLoop(), None)
        router.write("stderr", "no newline")
        router.flush_partial()
        assert err.getvalue() == "no newline"

    def test_closed_loop_does_not_lose_output(self):
        class _ClosedLoop:
            def call_soon_threadsafe(self, fn, *args, context=None):
                raise RuntimeError("loop closed")

        router = _OutputRouter()
        err = io.StringIO()
        router.set_target("stderr", err)
        router.prompt_started(object(), _ClosedLoop(), None)
        router.write("stderr", "late\n")
        router.prompt_finished()
        assert err.getvalue() == "late\n"


# ---------------------------------------------------------------------------
# Queueing and recall
# ---------------------------------------------------------------------------


class TestQueueing:
    def test_idle_submission_is_not_marked_queued(self):
        ui = _make_ui(_FakeSession([]))
        ui._submit("hello")
        assert ui.queued == []
        event = ui.next_event(timeout=0)
        assert event == ReplEvent("line", "hello")

    def test_busy_submission_is_queued_until_consumed(self):
        ui = _make_ui(_FakeSession([]))
        ui.set_busy(True)
        ui._submit("later")
        assert ui.queued == ["later"]
        assert ui.live_visible()
        ui.set_busy(False)
        assert ui.next_event(timeout=0).text == "later"
        assert ui.queued == []

    def test_blank_submission_ignored(self):
        ui = _make_ui(_FakeSession([]))
        ui._submit("   \n ")
        assert ui.next_event(timeout=0) is None

    def test_recall_moves_busy_lines_to_editor_and_keeps_exit(self):
        ui = _make_ui(_FakeSession([]))
        ui._submit("typed while idle")
        ui.set_busy(True)
        ui._submit("follow-up one")
        ui._submit("follow-up two")
        ui._submit("/exit")
        ui.recall_queued()
        texts = []
        while True:
            event = ui.next_event(timeout=0)
            if event is None:
                break
            texts.append(event.text)
        assert texts == ["typed while idle", "/exit"]
        assert ui._saved_document == "follow-up one\n\nfollow-up two"
        assert ui.queued == []

    def test_recall_with_nothing_queued_is_noop(self):
        ui = _make_ui(_FakeSession([]))
        ui.recall_queued()
        assert ui._saved_document is None
        assert ui.flash_text() is None

    def test_next_event_timeout(self):
        ui = _make_ui(_FakeSession([]))
        started = time.monotonic()
        assert ui.next_event(timeout=0.05) is None
        assert time.monotonic() - started < 1

    def test_exit_command_detection(self):
        assert _is_exit_command("/exit")
        assert _is_exit_command("  /QUIT  ")
        assert _is_exit_command("/exit\nmore")
        assert _is_exit_command("/exit now")
        assert not _is_exit_command("hello")
        assert not _is_exit_command("/copy")


# ---------------------------------------------------------------------------
# Ctrl-C at an idle prompt
# ---------------------------------------------------------------------------


class TestIdleInterrupt:
    def test_clears_pending_text_without_arming(self):
        session = _FakeSession([])
        session.default_buffer.text = "half typed"
        ui = _make_ui(session, conversation=True)
        assert ui._handle_idle_interrupt() is False
        assert ui._ctrl_c_armed is False

    def test_arms_then_exits_with_conversation(self, capsys):
        ui = _make_ui(_FakeSession([]), conversation=True)
        assert ui._handle_idle_interrupt() is False
        assert "Press Ctrl-C again" in capsys.readouterr().err
        assert ui._handle_idle_interrupt() is True

    def test_exits_immediately_without_conversation(self):
        ui = _make_ui(_FakeSession([]), conversation=False)
        assert ui._handle_idle_interrupt() is True

    def test_interrupt_turn_is_debounced(self, monkeypatch):
        sent = []
        monkeypatch.setattr(
            "swival.repl_ui._interrupt_main_thread", lambda: sent.append(1)
        )
        ui = _make_ui(_FakeSession([]))
        ui.set_busy(True)
        ui.interrupt_turn()
        ui.interrupt_turn()
        assert sent == [1]
        assert ui.flash_text() == "interrupting…"


# ---------------------------------------------------------------------------
# UI thread
# ---------------------------------------------------------------------------


class TestUiThread:
    def _collect(self, ui, count, timeout=5):
        events = []
        deadline = time.monotonic() + timeout
        while len(events) < count and time.monotonic() < deadline:
            event = ui.next_event(timeout=0.1)
            if event is not None:
                events.append(event)
        return events

    def test_scripted_lines_become_events_then_exit(self):
        session = _FakeSession(["hello", "world", EOFError()])
        ui = _make_ui(session)
        original_err = sys.stderr
        ui.start()
        try:
            assert sys.stderr is not original_err
            events = self._collect(ui, 3)
        finally:
            ui.stop()
        assert sys.stderr is original_err
        assert [e.kind for e in events] == ["line", "line", "exit"]
        assert [e.text for e in events[:2]] == ["hello", "world"]

    def test_exit_command_stops_prompting(self):
        session = _FakeSession(["/exit", "never read"])
        ui = _make_ui(session)
        ui.start()
        try:
            events = self._collect(ui, 2)
        finally:
            ui.stop()
        assert events[0].text == "/exit"
        assert events[1].kind == "exit"
        assert session.script == ["never read"]

    def test_double_ctrl_c_exits(self):
        session = _FakeSession([KeyboardInterrupt(), KeyboardInterrupt()])
        ui = _make_ui(session, conversation=True)
        ui.start()
        try:
            events = self._collect(ui, 1)
        finally:
            ui.stop()
        assert events[0].kind == "exit"
        assert len(session.calls) == 2

    def test_eof_while_busy_is_ignored(self):
        session = _FakeSession([EOFError(), "after"])
        ui = _make_ui(session)
        ui.set_busy(True)
        ui.start()
        try:
            events = self._collect(ui, 1)
        finally:
            ui.stop()
        assert events[0].text == "after"

    def test_exhausted_script_is_treated_as_eof(self):
        session = _FakeSession([StopIteration()])
        ui = _make_ui(session)
        ui.start()
        try:
            events = self._collect(ui, 1)
        finally:
            ui.stop()
        assert events[0].kind == "exit"

    def test_stop_unblocks_a_waiting_prompt(self):
        session = _FakeSession(["<block>"])
        ui = _make_ui(session)
        ui.start()
        time.sleep(0.2)
        ui.stop()
        assert not ui._thread.is_alive()

    def test_suspension_parks_prompt_and_restores_draft(self):
        session = _FakeSession(["<block>", "<block>"])
        ui = _make_ui(session)
        ui.start()
        try:
            time.sleep(0.2)
            assert ui._router.app is not None
            session.default_buffer.text = "draft"
            session.default_buffer.document = Document("draft", 3)
            real_err = ui._original_streams["stderr"]
            with ui.suspended():
                assert ui._parked.is_set()
                assert sys.stderr is real_err
                assert ui._router.app is None
            deadline = time.monotonic() + 5
            while len(session.calls) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert len(session.calls) == 2
            restored = session.calls[1]
            assert isinstance(restored, Document)
            assert restored.text == "draft"
            assert restored.cursor_position == 3
        finally:
            ui.stop()

    def test_nested_suspension_resumes_once(self):
        session = _FakeSession(["<block>", "<block>"])
        ui = _make_ui(session)
        ui.start()
        try:
            time.sleep(0.2)
            with ui.suspended():
                with ui.suspended():
                    assert ui._suspend_count == 2
                assert ui._parked.is_set()
                assert ui._suspend_count == 1
            deadline = time.monotonic() + 5
            while len(session.calls) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert len(session.calls) == 2
        finally:
            ui.stop()

    def test_stuck_ui_thread_is_reported_not_waited_for(self, monkeypatch):
        class _StuckSession(_FakeSession):
            def prompt(self, message=None, *, default="", pre_run=None):
                self.calls.append(default)
                pre_run()
                time.sleep(1.0)
                return "late"

        monkeypatch.setattr("swival.repl_ui._PARK_TIMEOUT", 0.2)
        session = _StuckSession([])
        ui = _make_ui(session)
        real_err = io.StringIO()
        ui.start()
        try:
            time.sleep(0.1)
            ui._original_streams["stderr"] = real_err
            started = time.monotonic()
            with ui.suspended():
                pass
            assert time.monotonic() - started < 0.9
            assert "did not release the terminal" in real_err.getvalue()
        finally:
            ui.stop()
            time.sleep(1.2)

    def test_output_during_prompt_goes_above_it(self):
        session = _FakeSession(["<block>"])
        ui = _make_ui(session)
        captured = io.StringIO()
        ui.start()
        try:
            time.sleep(0.2)
            ui._router.set_target("stderr", captured)
            print("above the prompt", file=sys.stderr)
            assert captured.getvalue() == "above the prompt\n"
            assert session.app.loop.calls >= 1
        finally:
            ui.stop()


# ---------------------------------------------------------------------------
# Status region rendering
# ---------------------------------------------------------------------------


class TestRendering:
    def test_spinner_slot_renders_phase_and_suffix(self):
        slot = _SpinnerSlot("Thinking (turn 2/5)")
        text = _render_ansi(slot.render(80, 5), 80)
        assert "Thinking (turn 2/5)" in text

    def test_spinner_slots_advance_frames_over_time(self):
        # Each redraw builds a fresh Rich Spinner, whose animation clock
        # starts at its first render, so the glyph must follow the slot's
        # own elapsed time or the spinner freezes on frame zero.
        for slot in (_SpinnerSlot("Thinking"), _CommandSlot("ls", None)):
            frames = set()
            for elapsed in (0.0, 0.1, 0.2, 0.3):
                slot.started = time.monotonic() - elapsed
                frames.add(_render_ansi(slot.render(80, 5), 80)[:4])
            assert len(frames) > 1, type(slot).__name__

    def test_command_slot_shows_bar_with_timeout(self):
        slot = _CommandSlot("sleep 5", 30)
        text = _render_ansi(slot.render(100, 5), 100)
        assert "Running sleep 5" in text
        assert "of timeout" in text

    def test_command_slot_without_timeout(self):
        slot = _CommandSlot("x" * 80, None)
        text = _render_ansi(slot.render(100, 5), 100)
        assert "of timeout" not in text
        assert "…" in text

    def test_stream_slot_tails_answer(self):
        slot = _StreamSlot()
        assert slot.render(80, 3) is None
        slot.update(answer="\n".join(f"line {i}" for i in range(10)))
        text = _render_ansi(slot.render(80, 3), 80)
        assert "line 9" in text
        assert "line 0" not in text

    def test_live_fragments_include_queued_lines(self):
        ui = _make_ui(_FakeSession([]))
        ui.set_busy(True)
        ui._submit("first queued")
        ui._submit("second queued")
        fragments = ui.live_fragments(80, 24, 4)
        text = "".join(t for _s, t in fragments)
        assert "queued" in text
        assert "first queued" in text
        assert "second queued" in text

    def test_live_fragments_tolerate_slot_failure(self):
        class _Broken(_SpinnerSlot):
            def render(self, width, budget):
                raise RuntimeError("boom")

        ui = _make_ui(_FakeSession([]))
        with ui._slot(_Broken("x")):
            assert ui.live_fragments(80, 24, 4) == []

    def test_toolbar_flash_overrides_fragments(self):
        ui = ReplUI(
            _FakeSession([]),
            has_conversation=lambda: False,
            toolbar_fragments=lambda: [("class:bottom-toolbar", "ctx 3%")],
        )
        assert ("class:bottom-toolbar", "ctx 3%") in ui.toolbar()
        ui.flash("queued")
        assert any("queued" in t for _s, t in ui.toolbar())

    def test_right_hint_needs_room(self):
        base = [("", "ctx 3%")]
        assert _with_right_hint(base, "hint", None) == base
        assert _with_right_hint(base, "hint", 12) == base
        wide = _with_right_hint(base, "hint", 40)
        assert "".join(t for _s, t in wide).endswith("hint")
        assert len("".join(t for _s, t in wide)) == 38

    def test_placeholder_and_prompt_follow_busy_state(self, monkeypatch):
        monkeypatch.setattr(fmt, "animations_enabled", lambda: True)
        ui = _make_ui(_FakeSession([]))
        assert "Ask anything" in ui.placeholder()
        assert ui.prompt_style() == "class:prompt"
        assert ui.border_phase() is None
        assert ui._border_style() == "class:frame.border"
        ui.set_busy(True)
        assert "queues" in ui.placeholder()
        assert ui.prompt_style() == "class:prompt.busy"
        assert 0 <= ui.border_phase() < 1
        assert ui._border_style().startswith("class:frame.border class:frame.border.s")

    def test_border_stays_still_without_animations(self, monkeypatch):
        monkeypatch.setattr(fmt, "animations_enabled", lambda: False)
        ui = _make_ui(_FakeSession([]))
        ui.set_busy(True)
        assert ui.border_phase() is None
        assert ui._border_style() == "class:frame.border"
        assert ui._animating() is False

    def test_shrink_wrap_caps_height_at_preferred(self):
        from prompt_toolkit.layout import Window
        from prompt_toolkit.layout.controls import FormattedTextControl

        window = Window(FormattedTextControl(lambda: [("", "a\nb\nc")]))
        wrapped = _ShrinkWrap(window)
        dim = wrapped.preferred_height(80, 40)
        assert dim.preferred == 3
        assert dim.max == 3

    def test_flash_expires(self):
        ui = _make_ui(_FakeSession([]))
        ui.flash("blink", seconds=0.05)
        assert ui.flash_text() == "blink"
        time.sleep(0.1)
        assert ui.flash_text() is None


# ---------------------------------------------------------------------------
# fmt integration
# ---------------------------------------------------------------------------


class _RecordingBackend:
    def __init__(self):
        self.calls = []

    def _cm(self, name, handle="handle"):
        @contextlib.contextmanager
        def cm(*args):
            self.calls.append((name, args))
            yield handle

        return cm

    def __getattr__(self, name):
        return self._cm(name)


class TestFmtBackend:
    @pytest.fixture
    def tty_console(self, monkeypatch):
        console = styled_console(io.StringIO())
        monkeypatch.setattr(fmt, "_console", console)
        yield console
        fmt.set_live_backend(None)

    def test_live_displays_delegate_to_backend(self, tty_console):
        backend = _RecordingBackend()
        fmt.set_live_backend(backend)
        with fmt.llm_spinner("Thinking (turn 1/2)") as dismiss:
            assert dismiss == "handle"
        with fmt.command_spinner("ls", 30):
            pass
        with fmt.step_spinner("step"):
            pass
        with fmt.input_marquee("text"):
            pass
        with fmt.input_marquee_then_spinner("text", "Thinking"):
            pass
        with fmt.stream_raw():
            pass
        with fmt.stream_channels():
            pass
        names = [name for name, _ in backend.calls]
        assert names == [
            "llm_spinner",
            "command_spinner",
            "step_spinner",
            "input_marquee",
            "input_marquee_then_spinner",
            "stream_raw",
            "stream_channels",
        ]
        assert backend.calls[1][1] == ("ls", 30)
        assert backend.calls[4][1] == ("text", "Thinking", 4.0)

    def test_suspend_live_parks_backend(self, tty_console):
        backend = _RecordingBackend()
        fmt.set_live_backend(backend)
        with fmt.suspend_live():
            pass
        assert backend.calls == [("suspend", ())]

    def test_no_backend_off_terminal(self, monkeypatch):
        console = Console(file=io.StringIO(), force_terminal=False, width=80)
        monkeypatch.setattr(fmt, "_console", console)
        backend = _RecordingBackend()
        fmt.set_live_backend(backend)
        try:
            with fmt.llm_spinner("x") as dismiss:
                dismiss()
        finally:
            fmt.set_live_backend(None)
        assert backend.calls == []

    def test_turn_header_static_with_backend(self, tty_console, monkeypatch):
        monkeypatch.setattr(fmt, "animations_enabled", lambda: True)
        animated = []
        monkeypatch.setattr(
            fmt, "_animate_turn_rule", lambda title: animated.append(title)
        )
        fmt.set_live_backend(_RecordingBackend())
        fmt.turn_header(1, 5, 100)
        assert animated == []
        assert "Turn 1/5" in tty_console.file.getvalue()

    def test_prompt_echo_indents_continuation_lines(self, tty_console):
        fmt.repl_prompt_echo("first\nsecond")
        out = Text.from_ansi(tty_console.file.getvalue()).plain
        assert "❯ first" in out
        assert "\n  second" in out


# ---------------------------------------------------------------------------
# Main-thread interrupt delivery
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="needs pthread_kill")
def test_interrupt_main_thread_breaks_blocking_wait():
    gate = threading.Event()
    interrupted = False
    threading.Timer(0.1, _interrupt_main_thread).start()
    try:
        gate.wait(timeout=5)
    except KeyboardInterrupt:
        interrupted = True
    assert interrupted

"""Live terminal UI for the /audit pipeline.

All worker threads talk to ``AuditUI`` through an event queue. The render
thread is the sole consumer that touches Rich primitives. On a non-TTY
console the UI degrades to plain ``fmt.info`` / ``fmt.warning`` lines so
piped runs stay byte-identical to the historical output.
"""

from __future__ import annotations

import queue
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import fmt


@dataclass
class _PhaseAdvance:
    phase_id: int
    n: int = 1
    current: Optional[str] = None


@dataclass
class _PhaseAdd:
    phase_id: int
    title: str
    total: Optional[int]
    color: str


@dataclass
class _PhaseComplete:
    phase_id: int
    summary: Optional[str]


@dataclass
class _WorkerStart:
    slot: int
    label: str
    started_at: float


@dataclass
class _WorkerEnd:
    slot: int


@dataclass
class _Finding:
    severity: str
    title: str
    path: Optional[str]


@dataclass
class _Warning:
    text: str


@dataclass
class _Scrollback:
    text: str


@dataclass
class _Shutdown:
    pass


@dataclass
class _Summary:
    artifact_dir: Optional[str]
    written: int
    done: threading.Event


@dataclass
class _Incomplete:
    message: str
    done: threading.Event


@dataclass
class _PhaseState:
    phase_id: int
    title: str
    color: str
    total: Optional[int]
    completed: int = 0
    current: Optional[str] = None
    task_id: Optional[int] = None
    started_at: float = field(default_factory=time.monotonic)
    finished: bool = False


class PhaseHandle:
    """Caller-side handle for a phase. All methods enqueue events."""

    def __init__(self, ui: "AuditUI", phase_id: int):
        self._ui = ui
        self._id = phase_id
        self._completed_local = 0
        self._total_local: Optional[int] = None

    def advance(self, n: int = 1, *, current: Optional[str] = None) -> None:
        self._completed_local += n
        self._ui._enqueue(_PhaseAdvance(self._id, n, current))

    def set_current(self, label: str) -> None:
        self._ui._enqueue(_PhaseAdvance(self._id, 0, label))

    def complete(self, summary: Optional[str] = None) -> None:
        self._ui._enqueue(_PhaseComplete(self._id, summary))


class AuditUI:
    """Single live region for an audit run. Thread-safe via an event queue."""

    def __init__(
        self,
        *,
        run_id: str,
        branch: str,
        commit: str,
        workers: int,
        total_files: int,
    ):
        self.run_id = run_id
        self.branch = branch
        self.commit = (commit or "")[:8]
        self.workers = max(1, workers)
        self.total_files = total_files

        self._console = fmt.get_console()
        self._enabled = bool(self._console.is_terminal)

        self._queue: "queue.Queue[object]" = queue.Queue()
        self._render_thread: Optional[threading.Thread] = None
        self._live: Optional[Live] = None
        self._started_at = time.monotonic()

        self._phases: dict[int, _PhaseState] = {}
        self._phase_order: list[int] = []
        self._next_phase_id = 0
        self._workers: dict[int, tuple[str, float]] = {}
        self._tally_verified = 0
        self._tally_discarded = 0
        self._tally_failed = 0
        self._findings_seen: list[tuple[str, str, Optional[str]]] = []
        self._warnings_buffer: list[str] = []
        self._warnings_count = 0
        self._spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._tick = 0
        self._pause_depth = 0
        self._pause_lock = threading.Lock()
        self._paused = False

        self._progress = None  # set when enabled in __enter__

    @property
    def is_live(self) -> bool:
        """True when a Rich Live region is rendering on stderr."""
        return self._enabled

    def __enter__(self) -> "AuditUI":
        if not self._enabled:
            return self

        self._progress = fmt.bar_progress(transient=False)

        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=10,
            auto_refresh=False,
            transient=False,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self._live.start()
        self._render_thread = threading.Thread(
            target=self._render_loop, name="audit-ui-render", daemon=True
        )
        self._render_thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if not self._enabled:
            return
        self._enqueue(_Shutdown())
        if self._render_thread is not None:
            self._render_thread.join(timeout=2.0)
            if self._render_thread.is_alive():
                fmt.warning("audit-ui: render thread did not exit within 2s")
        if self._live is not None:
            try:
                self._live.stop()
            except Exception as e:
                fmt.warning(f"audit-ui: failed to stop live region: {e}")

    @contextmanager
    def pause(self) -> Iterator[None]:
        """Temporarily stop the Live region so external code can write to stderr.

        Reference-counted: concurrent callers (e.g., parallel verifier workers
        each wrapping their own ``run_agent_loop``) only stop Live on the first
        entrant and only restart it after the last one exits, so Live stays
        suspended for as long as any caller is still inside the context.
        """
        if not self._enabled or self._live is None:
            yield
            return
        with self._pause_lock:
            self._pause_depth += 1
            first = self._pause_depth == 1
            if first:
                self._paused = True
                try:
                    self._live.stop()
                except Exception as e:
                    fmt.warning(f"audit-ui: pause failed to stop live region: {e}")
        try:
            yield
        finally:
            with self._pause_lock:
                self._pause_depth -= 1
                last = self._pause_depth == 0
                if last:
                    try:
                        self._live.start()
                    except Exception as e:
                        fmt.warning(
                            f"audit-ui: pause failed to restart live region: {e}"
                        )
                    self._paused = False

    # ------------------------------------------------------------------
    # Public API (thread-safe)
    # ------------------------------------------------------------------

    def phase(
        self,
        title: str,
        *,
        total: Optional[int] = None,
        color: str = "cyan",
    ) -> PhaseHandle:
        phase_id = self._next_phase_id
        self._next_phase_id += 1
        if not self._enabled:
            # Non-TTY: the pipeline owns its own legacy "phase N: ..." line via
            # `if not ui.is_live`. AuditUI must not emit anything extra here
            # or piped output would no longer be byte-identical to history.
            return PhaseHandle(self, phase_id)
        self._enqueue(_PhaseAdd(phase_id, title, total, color))
        return PhaseHandle(self, phase_id)

    def worker_started(self, slot: int, label: str) -> None:
        if not self._enabled:
            return
        self._enqueue(_WorkerStart(slot, label, time.monotonic()))

    def worker_ended(self, slot: int) -> None:
        if not self._enabled:
            return
        self._enqueue(_WorkerEnd(slot))

    def finding(self, severity: str, title: str, path: Optional[str] = None) -> None:
        if not self._enabled:
            # Non-TTY: the pipeline's existing "[N/M] verified: ..." line is
            # the legacy producer. The findings ticker is a TTY-only addition.
            return
        self._enqueue(_Finding(severity, title, path))

    def warning(self, text: str) -> None:
        if not self._enabled:
            self._warnings_count += 1
            fmt.warning(text)
            return
        self._warnings_buffer.append(text)
        self._enqueue(_Warning(text))

    def scrollback(self, text: str) -> None:
        if not self._enabled:
            fmt.info(text)
            return
        self._enqueue(_Scrollback(text))

    def tally(self, *, verified: int = 0, discarded: int = 0, failed: int = 0) -> None:
        """Adjust the running outcome counters."""
        self._tally_verified += verified
        self._tally_discarded += discarded
        self._tally_failed += failed

    def summary(self, *, artifact_dir: Optional[str], written: int) -> None:
        """Print the final summary panel.

        TTY: enqueued as a render-thread event so it runs after all prior
        events have been drained — severity counts are consistent and the
        panel is printed from the same thread that owns the Live region.
        Non-TTY: no-op. The pipeline's existing "Audit complete." return
        string is the only legacy producer; emitting a new plain block here
        would diverge from historical output.

        Pass ``artifact_dir=None`` when no artifact directory was created
        (e.g., zero verified findings); the artifacts row is omitted.
        """
        if not self._enabled:
            return
        done = threading.Event()
        self._enqueue(_Summary(artifact_dir, written, done))
        done.wait(timeout=2.0)

    def incomplete(self, message: str) -> None:
        """Print a final 'Audit incomplete' panel and block until rendered."""
        if not self._enabled:
            return
        done = threading.Event()
        self._enqueue(_Incomplete(message, done))
        done.wait(timeout=2.0)

    def _build_outcome_table(self) -> Table:
        elapsed = time.monotonic() - self._started_at
        by_sev: dict[str, int] = {}
        for sev, _, _ in self._findings_seen:
            key = (sev or "unknown").lower()
            by_sev[key] = by_sev.get(key, 0) + 1

        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold")
        table.add_column()
        table.add_row("Run", f"{self.run_id} · {self.branch} @ {self.commit}")
        table.add_row("Elapsed", _fmt_duration(elapsed))

        outcomes = Text()
        outcomes.append(f"{self._tally_verified} verified", style="green")
        outcomes.append("  ")
        outcomes.append(f"{self._tally_discarded} discarded", style="yellow")
        outcomes.append("  ")
        outcomes.append(f"{self._tally_failed} failed", style="red")
        table.add_row("Findings", outcomes)

        if by_sev:
            sev_line = Text()
            first = True
            for sev in ("critical", "high", "medium", "low", "unknown"):
                n = by_sev.get(sev, 0)
                if not n:
                    continue
                if not first:
                    sev_line.append("  ")
                first = False
                sev_line.append(f"{sev}: {n}", style=fmt.severity_style(sev))
            table.add_row("By severity", sev_line)
        return table

    def _render_summary_panel(self, artifact_dir: Optional[str], written: int) -> None:
        """Build and print the summary panel. Render-thread only."""
        table = self._build_outcome_table()
        if artifact_dir is not None:
            table.add_row(
                "Artifacts",
                Text(f"{written} written to {artifact_dir}/", style="cyan"),
            )
        if self._warnings_buffer:
            table.add_row(
                "Warnings",
                Text(f"{len(self._warnings_buffer)} non-fatal", style="yellow"),
            )

        panel = Panel(
            table,
            title=Text("Audit complete", style="bold green"),
            border_style="green",
            padding=(1, 2),
        )
        self._print_above(Text(""))
        self._print_above(panel)

    def _render_incomplete_panel(self, message: str) -> None:
        """Build and print the incomplete-audit panel. Render-thread only."""
        table = self._build_outcome_table()
        table.add_row("Status", Text(message, style="yellow"))
        if self._warnings_buffer:
            table.add_row(
                "Warnings",
                Text(f"{len(self._warnings_buffer)} non-fatal", style="yellow"),
            )

        panel = Panel(
            table,
            title=Text("Audit incomplete", style="bold yellow"),
            border_style="yellow",
            padding=(1, 2),
        )
        self._print_above(Text(""))
        self._print_above(panel)

    # ------------------------------------------------------------------
    # Event plumbing (render thread only past this line touches Rich)
    # ------------------------------------------------------------------

    def _enqueue(self, event: object) -> None:
        if not self._enabled:
            # No render thread runs in non-TTY mode. Dropping the event
            # avoids unbounded queue growth from PhaseHandle.advance()/
            # complete() calls that the pipeline still issues uniformly
            # for both modes.
            if isinstance(event, _Summary):
                event.done.set()
            return
        self._queue.put(event)

    def _render_loop(self) -> None:
        last_refresh = 0.0
        while True:
            try:
                event = self._queue.get(timeout=0.1)
            except queue.Empty:
                event = None

            if isinstance(event, _Shutdown):
                self._apply_pending_quick()
                self._refresh()
                return

            if event is not None:
                self._apply(event)
                self._apply_pending_quick()

            now = time.monotonic()
            if now - last_refresh >= 0.1:
                self._tick += 1
                self._refresh()
                last_refresh = now

    def _apply_pending_quick(self) -> None:
        """Drain any remaining events without blocking."""
        drained = 0
        while drained < 64:
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(event, _Shutdown):
                self._queue.put(event)
                return
            self._apply(event)
            drained += 1

    def _apply(self, event: object) -> None:
        if isinstance(event, _PhaseAdd):
            state = _PhaseState(
                phase_id=event.phase_id,
                title=event.title,
                color=event.color,
                total=event.total,
            )
            if self._progress is not None:
                desc = Text(event.title, style=f"bold {event.color}")
                state.task_id = self._progress.add_task(
                    desc, total=event.total if event.total else None
                )
            self._phases[event.phase_id] = state
            self._phase_order.append(event.phase_id)
            self._print_above(_phase_banner_text(event.title, event.color))
        elif isinstance(event, _PhaseAdvance):
            state = self._phases.get(event.phase_id)
            if state is None:
                return
            state.completed += event.n
            if event.current is not None:
                state.current = event.current
            if self._progress is not None and state.task_id is not None:
                kwargs: dict = {}
                if event.n:
                    kwargs["advance"] = event.n
                if event.current is not None:
                    kwargs["description"] = Text(
                        f"{state.title} · {event.current}",
                        style=f"bold {state.color}",
                    )
                if kwargs:
                    self._progress.update(state.task_id, **kwargs)
        elif isinstance(event, _PhaseComplete):
            state = self._phases.get(event.phase_id)
            if state is None:
                return
            state.finished = True
            if self._progress is not None and state.task_id is not None:
                if state.total:
                    self._progress.update(state.task_id, completed=state.total)
                self._progress.remove_task(state.task_id)
                state.task_id = None
            line = Text("  ✓ ", style=f"bold {state.color}")
            line.append(state.title, style=f"bold {state.color}")
            if event.summary:
                line.append(" — ", style="dim")
                line.append(event.summary, style="dim")
            self._print_above(line)
        elif isinstance(event, _WorkerStart):
            self._workers[event.slot] = (event.label, event.started_at)
        elif isinstance(event, _WorkerEnd):
            self._workers.pop(event.slot, None)
        elif isinstance(event, _Finding):
            self._findings_seen.append((event.severity, event.title, event.path))
            style = fmt.severity_style(event.severity)
            line = Text("  ! ", style=style)
            line.append(f"[{(event.severity or 'unknown').lower()}] ", style=style)
            line.append(event.title, style="bold")
            if event.path:
                line.append("  ", style="dim")
                line.append(event.path, style="dim")
            self._print_above(line)
        elif isinstance(event, _Warning):
            line = Text("  ⚠ ", style="yellow")
            line.append(event.text, style="yellow")
            self._print_above(line)
        elif isinstance(event, _Scrollback):
            self._print_above(Text(f"  {event.text}", style="dim"))
        elif isinstance(event, _Summary):
            try:
                self._render_summary_panel(event.artifact_dir, event.written)
            finally:
                event.done.set()
        elif isinstance(event, _Incomplete):
            try:
                self._render_incomplete_panel(event.message)
            finally:
                event.done.set()

    def _print_above(self, renderable) -> None:
        """Print scrollback above the live region without tearing it."""
        if self._live is not None:
            self._live.console.print(renderable)
        else:
            self._console.print(renderable)

    def _refresh(self) -> None:
        if self._live is None or self._paused:
            return
        self._live.update(self._render(), refresh=True)

    def _render(self):
        elapsed = time.monotonic() - self._started_at
        header = Text()
        header.append("audit ", style="bold cyan")
        header.append(self.run_id, style="bold white")
        header.append("  ")
        header.append(self.branch, style="green")
        header.append(" @ ", style="dim")
        header.append(self.commit, style="dim")
        header.append("  ·  ", style="dim")
        active = next(
            (
                p
                for p in self._phases.values()
                if not p.finished and p.task_id is not None
            ),
            None,
        )
        if active is not None:
            header.append(active.title, style=f"bold {active.color}")
        else:
            header.append("idle", style="dim")
        header.append("  ·  ", style="dim")
        header.append(_fmt_duration(elapsed), style="cyan")

        rows = [header]

        if self._progress is not None:
            rows.append(self._progress)

        if self._workers:
            spin = self._spinner_frames[self._tick % len(self._spinner_frames)]
            for slot in sorted(self._workers):
                label, started = self._workers[slot]
                age = time.monotonic() - started
                line = Text(overflow="ellipsis", no_wrap=True)
                line.append(f"  {spin} ", style="cyan")
                line.append(f"worker {slot}", style="bold")
                line.append(" · ", style="dim")
                line.append(label, style="white")
                line.append(f"  {_fmt_duration(age)}", style="dim")
                rows.append(line)
        elif active is not None:
            rows.append(Text("  (workers idle)", style="dim"))

        tally = Text()
        tally.append("  ")
        tally.append(f"{self._tally_verified} verified", style="green")
        tally.append("   ")
        tally.append(f"{self._tally_discarded} discarded", style="yellow")
        tally.append("   ")
        tally.append(f"{self._tally_failed} failed", style="red")
        if self._findings_seen:
            tally.append("   ")
            tally.append(
                f"{len(self._findings_seen)} findings seen", style="bold magenta"
            )
        rows.append(tally)

        panel = Panel(
            Group(*rows),
            border_style="blue",
            padding=(0, 1),
        )
        return panel


def _fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:0.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _phase_banner_text(title: str, color: str) -> Text:
    line = Text()
    line.append("── ", style=color)
    line.append(title, style=f"bold {color}")
    line.append(" ──", style=color)
    return line


def make_label_for_path() -> Callable[[str], str]:
    return lambda p: p


def make_label_for_finding(prefix: str = "") -> Callable[[object], str]:
    def _label(item) -> str:
        try:
            _key, finding = item
            title = getattr(finding, "title", str(finding))
        except (TypeError, ValueError):
            title = getattr(item, "title", str(item))
        return f"{prefix}{title}" if prefix else title

    return _label

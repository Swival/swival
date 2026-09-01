"""Interactive REPL front end.

The prompt lives on its own thread so the user can type the next message while
a turn is running on the main thread. Everything the agent prints during the
turn is routed above the prompt, and the transient displays (spinner, marquee,
streaming viewport, command progress) render inside the prompt layout instead
of a Rich Live region, which could not share the screen with a prompt below it.

The agent turn deliberately stays on the main thread: Python only runs signal
handlers there, and every Ctrl-C path in the agent relies on a real
KeyboardInterrupt landing in the middle of a blocking call.
"""

from __future__ import annotations

import collections
import contextlib
import contextvars
import io
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable

from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.formatted_text.utils import fragment_list_width
from prompt_toolkit.layout.containers import Container
from prompt_toolkit.layout.dimension import Dimension
from rich.console import Console
from rich.progress_bar import ProgressBar
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from . import fmt
from ._msg import _trim_for_marquee
from .input_commands import EXIT_COMMANDS
from .input_dispatch import parse_input_line

_TICK_SECONDS = 1 / 12
_FLASH_SECONDS = 3.0
_INTERRUPT_DEBOUNCE = 0.5
_STREAM_MIN_ROWS = 3
_QUEUE_PREVIEW_CHARS = 60
# Grace period for the UI thread to hand the terminal back. Only a thread
# stuck inside prompt_toolkit can exhaust it; one that exits releases the wait.
_PARK_TIMEOUT = 10.0

_IDLE_PLACEHOLDER = "Ask anything.  / commands  @ files  ! shell  Shift+Enter newline"
_BUSY_PLACEHOLDER = "Type the next message, Enter queues it.  Ctrl-C interrupts."
_CONTINUATION = "  "
_BORDER_STEPS = 24
_BORDER_CYCLE_SECONDS = 3.0

_REPL_SHIFT_ENTER_SEQUENCES = ("\x1b[27;2;13~", "\x1b[13;2u")


def _border_palette() -> dict:
    """One style class per step of the border's breathing cycle.

    Keeping every color in a single constant Style lets the border animate by
    swapping class names on the border windows, which prompt_toolkit diffs
    cell by cell. Swapping Style objects instead would invalidate its screen
    cache and repaint the whole prompt every frame.
    """
    palette = {}
    for step in range(_BORDER_STEPS):
        r, g, b = fmt._lerp_color(fmt._TURN_GRADIENT, step / (_BORDER_STEPS - 1))
        palette[f"frame.border.s{step}"] = f"#{r:02x}{g:02x}{b:02x}"
    return palette


_BASE_STYLE = {
    "": "",
    "prompt": "bold ansigreen",
    "prompt.busy": "bold ansiyellow",
    "placeholder": "italic #6c7086",
    "frame.border": "#585b70",
    "bottom-toolbar": "noreverse",
    "bottom-toolbar.text": "#6c7086",
    "bottom-toolbar.key": "bold ansiyellow",
    "bottom-toolbar.model": "ansicyan",
    "bottom-toolbar.tip": "italic #6c7086",
    "bottom-toolbar.flash": "bold ansicyan",
    "bottom-toolbar.sep": "#45475a",
    "queued.label": "bold ansiyellow",
    "queued.text": "#a6adc8",
    "completion-menu": "bg:#1e1e2e #cdd6f4",
    "completion-menu.completion.current": "bg:#585b70 #ffffff bold",
    "completion-menu.meta.completion": "bg:#1e1e2e #6c7086",
    "completion-menu.meta.completion.current": "bg:#585b70 #cdd6f4",
    "scrollbar.background": "bg:#313244",
    "scrollbar.button": "bg:#585b70",
    **_border_palette(),
}


def _repl_key_bindings(ui: "ReplUI | None" = None):
    """Enter submits, Shift+Enter and Ctrl-J insert a newline, Ctrl-C depends
    on whether a turn is running. Without a ``ui`` (tests) Ctrl-C aborts."""
    from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys

    for sequence in _REPL_SHIFT_ENTER_SEQUENCES:
        ANSI_SEQUENCES[sequence] = Keys.ControlM

    kb = KeyBindings()

    @kb.add("c-j")
    def _insert_newline(event):
        event.current_buffer.insert_text("\n")

    @kb.add("c-m")
    def _insert_newline_or_accept(event):
        if event.data in _REPL_SHIFT_ENTER_SEQUENCES:
            event.current_buffer.insert_text("\n")
        else:
            event.current_buffer.validate_and_handle()

    @kb.add("c-c")
    def _interrupt_or_abort(event):
        if ui is not None and ui.busy:
            ui.interrupt_turn()
        else:
            event.app.exit(exception=KeyboardInterrupt(), style="class:aborting")

    return kb


class _Leave:
    """Result the prompt returns when the main thread asked it to step aside,
    either for good (stop) or until a nested prompt is done (suspend)."""


@dataclass
class ReplEvent:
    """Something the user did that the main loop must act on."""

    kind: str  # "line" or "exit"
    text: str = ""
    # Submitted while a turn was running: shown above the input until it
    # runs, handed back to the editor if the turn is interrupted.
    queued: bool = False


def _interrupt_main_thread() -> None:
    """Deliver SIGINT to the main thread, the way the tty used to."""
    main = threading.main_thread()
    pthread_kill = getattr(signal, "pthread_kill", None)
    if pthread_kill is not None and main.ident is not None:
        pthread_kill(main.ident, signal.SIGINT)
        return
    import _thread

    _thread.interrupt_main()


def _is_tty(stream) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _is_exit_command(text: str) -> bool:
    return parse_input_line(text).cmd in EXIT_COMMANDS


def _preview(text: str, limit: int = _QUEUE_PREVIEW_CHARS) -> str:
    return _trim_for_marquee(" ".join(text.split()), limit)


class _StreamProxy:
    """File-like stand-in for stdout/stderr while the prompt is on screen."""

    def __init__(self, router: "_OutputRouter", channel: str, target) -> None:
        self._router = router
        self._channel = channel
        self._target = target

    def write(self, data) -> int:
        if not isinstance(data, str):
            data = str(data)
        self._router.write(self._channel, data)
        return len(data)

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        self._router.nudge()

    def isatty(self) -> bool:
        return _is_tty(self._target)

    def fileno(self) -> int:
        return self._target.fileno()

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    @property
    def closed(self) -> bool:
        return False

    @property
    def encoding(self) -> str:
        return getattr(self._target, "encoding", "utf-8")

    @property
    def errors(self) -> str:
        return getattr(self._target, "errors", "strict")

    @property
    def line_buffering(self) -> bool:
        return True


class _OutputRouter:
    """Collects writes from any thread and emits them above the prompt.

    Complete lines are printed through prompt_toolkit's ``run_in_terminal``
    while a prompt is active, so the prompt is erased, the text is written and
    the prompt is drawn again below it. A line without its newline waits,
    because a redraw starting in the middle of a line would misplace the
    prompt. When no prompt is active the text goes straight to the real
    streams, under the same lock the prompt takes before its first draw, so
    the two can never interleave.

    prompt_toolkit's own ``patch_stdout`` is not used because it keeps one
    flush thread per stream (stdout and stderr could reorder), flushes on a
    fixed 0.2s cadence, and gives the rest of the front end no handle on the
    running application's loop and context, which this class also provides.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._targets: dict[str, object] = {}
        self._pending: list[tuple[str, str]] = []
        self._partial: dict[str, str] = {}
        self._app = None
        self._loop = None
        self._context: contextvars.Context | None = None
        self._scheduled = False

    def set_target(self, channel: str, stream) -> None:
        with self._lock:
            self._targets[channel] = stream

    @property
    def app(self):
        """The prompt_toolkit application currently on screen, if any."""
        return self._app

    def prompt_started(self, app, loop, context) -> None:
        with self._lock:
            self._app = app
            self._loop = loop
            self._context = context

    def prompt_finished(self) -> None:
        with self._lock:
            self._app = None
            self._loop = None
            self._context = None
            self._scheduled = False
            self._release_partials_locked()
            self._drain_locked()

    def call_in_prompt(self, fn) -> bool:
        """Run ``fn`` on the prompt's event loop. False when no prompt is up."""
        with self._lock:
            loop, context = self._loop, self._context
        if loop is None:
            return False
        try:
            loop.call_soon_threadsafe(fn, context=context)
        except RuntimeError:
            return False
        return True

    def write(self, channel: str, data: str) -> None:
        if not data:
            return
        with self._lock:
            buffered = self._partial.get(channel, "") + data
            if self._app is None:
                # No prompt below the cursor, so partial lines can go out as is.
                self._pending.append((channel, buffered))
                self._partial[channel] = ""
                self._drain_locked()
                return
            head, sep, tail = buffered.rpartition("\n")
            if sep:
                self._pending.append((channel, head + sep))
                self._partial[channel] = tail
            else:
                self._partial[channel] = buffered
            if not self._pending:
                return
        self.nudge()

    def nudge(self) -> None:
        with self._lock:
            if not self._pending:
                return
            if self._app is None:
                self._drain_locked()
                return
            if self._scheduled:
                return
            self._scheduled = True
        if not self.call_in_prompt(self._drain_in_loop):
            # The loop closed under us; prompt_finished() drains what is left.
            with self._lock:
                self._scheduled = False

    def _drain_in_loop(self) -> None:
        from prompt_toolkit.application import run_in_terminal
        from prompt_toolkit.application.current import get_app_or_none

        with self._lock:
            self._scheduled = False
        app = get_app_or_none()
        if app is None or not app.is_running:
            self._drain()
            return
        run_in_terminal(self._drain, in_executor=False)

    def flush_partial(self) -> None:
        """Push incomplete lines out, for hand-offs where nothing more comes."""
        with self._lock:
            self._release_partials_locked()
            self._drain_locked()

    def _release_partials_locked(self) -> None:
        for channel, text in list(self._partial.items()):
            if text:
                self._pending.append((channel, text))
            self._partial[channel] = ""

    def _drain(self) -> None:
        with self._lock:
            self._drain_locked()

    def _drain_locked(self) -> None:
        if not self._pending:
            return
        chunks, self._pending = self._pending, []
        touched = []
        for channel, text in chunks:
            stream = self._targets.get(channel)
            if stream is None:
                continue
            try:
                stream.write(text)
            except Exception:
                continue
            if stream not in touched:
                touched.append(stream)
        for stream in touched:
            try:
                stream.flush()
            except Exception:
                pass


def _spinner_text(name: str, style: str, desc: str, elapsed: float) -> Text:
    text = Text("  ")
    text.append_text(Spinner(name, style=style, speed=1.5).render(elapsed))
    text.append(f"  {desc}", style=style)
    text.append(f"  {fmt.format_duration(elapsed)}", style="dim")
    return text


class _Slot:
    """One transient display shown in the status region above the input."""

    def __init__(self) -> None:
        self.started = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def render(self, width: int, budget: int):
        raise NotImplementedError


class _SpinnerSlot(_Slot):
    """Spinner with the phase verbs of ``fmt.llm_spinner``."""

    def __init__(self, label: str) -> None:
        super().__init__()
        self.initial_desc, self.suffix = fmt.split_spinner_label(label)

    def render(self, width: int, budget: int):
        index, name, style, verb = fmt.spinner_phase(self.elapsed)
        desc = self.initial_desc if index == 0 else f"{verb}{self.suffix}"
        return _spinner_text(name, style, desc, self.elapsed)


class _StepSlot(_Slot):
    """Spinner whose label can be replaced from worker threads."""

    def __init__(self, label: str) -> None:
        super().__init__()
        self.label = label

    def update(self, text: str) -> None:
        self.label = text

    def render(self, width: int, budget: int):
        return _spinner_text("dots", "cyan", self.label, self.elapsed)


class _CommandSlot(_Slot):
    """Command progress: a bar filling toward the timeout, or a plain spinner."""

    def __init__(self, label: str, timeout: float | None) -> None:
        super().__init__()
        self.label = fmt.command_label(label)
        self.timeout = timeout if timeout and timeout > 0 else None

    def render(self, width: int, budget: int):
        grid = Table.grid(padding=(0, 1))
        cells = [
            Text("  ").append_text(
                Spinner("dots", style="cyan", speed=1.5).render(self.elapsed)
            ),
            Text(f"Running {self.label}", style="cyan"),
        ]
        if self.timeout is not None:
            done = min(self.elapsed, self.timeout)
            cells.append(ProgressBar(total=self.timeout, completed=done, width=30))
            cells.append(
                Text(f"{int(done * 100 / self.timeout):3d}% of timeout", style="dim")
            )
        cells.append(Text(fmt.format_duration(self.elapsed), style="dim"))
        for _ in cells:
            grid.add_column(no_wrap=True)
        grid.add_row(*cells)
        return grid


class _MarqueeSlot(_Slot):
    """Scrolling tail of the prompt, then the phase spinner after ``delay``."""

    def __init__(self, text: str, spinner_label: str | None, delay: float) -> None:
        super().__init__()
        self.text = text
        self.spinner_label = spinner_label
        self.delay = delay
        self._spinner: _SpinnerSlot | None = None

    def render(self, width: int, budget: int):
        if self.spinner_label is not None and self.elapsed >= self.delay:
            if self._spinner is None:
                self._spinner = _SpinnerSlot(self.spinner_label)
            return self._spinner.render(width, budget)
        offset = int(self.elapsed / fmt.MARQUEE_FRAME_SECONDS)
        return fmt._input_marquee_text(self.text, offset, width)


class _StreamSlot(_Slot):
    """Live tail of the streamed reply, sized to the space above the input."""

    def __init__(self) -> None:
        super().__init__()
        self.reasoning = ""
        self.answer = ""
        self.activity = ""

    def update(self, reasoning: str = "", answer: str = "", activity: str = "") -> None:
        self.reasoning = reasoning
        self.answer = answer
        self.activity = activity

    def render(self, width: int, budget: int):
        if not (self.reasoning.strip() or self.answer.strip() or self.activity.strip()):
            return None
        return fmt.render_stream_channels(
            self.reasoning,
            self.answer,
            self.activity,
            max(width - 2, 1),
            max(budget, _STREAM_MIN_ROWS),
        )


class ReplUI:
    """Owns the terminal for an interactive session.

    Build the prompt with :meth:`build_session` (or pass a scripted stand-in
    as ``session``), call :meth:`start`, then pull :class:`ReplEvent` objects
    with :meth:`next_event` from the main thread and run each turn inside
    :meth:`turn` so Ctrl-C and queueing know a turn is running.

    The instance doubles as the live backend for ``fmt``: the spinner,
    marquee, stream and command displays become slots in the status region.
    """

    def __init__(
        self,
        session=None,
        *,
        has_conversation: Callable[[], bool],
        toolbar_fragments: Callable[[], list] | None = None,
    ) -> None:
        self._session = session
        self._has_conversation = has_conversation
        self._toolbar_fragments = toolbar_fragments

        self._router = _OutputRouter()
        self._original_streams: dict[str, object] = {}

        self._inbox: collections.deque[ReplEvent] = collections.deque()
        self._inbox_cv = threading.Condition()

        self._lock = threading.Lock()
        self._slots: list[_Slot] = []
        self._busy = False
        self._busy_since = 0.0
        self._flash: tuple[str, float] | None = None
        self._ctrl_c_armed = False
        self._last_interrupt = 0.0

        self._stopping = False
        self._suspend_count = 0
        self._park_cv = threading.Condition()
        self._parked = threading.Event()
        self._saved_document = None

        self._thread: threading.Thread | None = None
        self._ticker: threading.Thread | None = None
        self._wake = threading.Event()

    def build_session(self, *, history_path: str, completer) -> None:
        """Create the prompt session and graft the status region onto it.

        The prompt_toolkit pieces are imported here rather than at module
        level so tests can substitute scripted stand-ins for them.
        """
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.styles import Style
        from prompt_toolkit.widgets.base import Border

        class _SafeFileHistory(FileHistory):
            def store_string(self, string: str) -> None:
                os.makedirs(os.path.dirname(self.filename), exist_ok=True)
                super().store_string(string)

        os.makedirs(os.path.dirname(history_path), exist_ok=True)

        saved = (
            Border.HORIZONTAL,
            Border.VERTICAL,
            Border.TOP_LEFT,
            Border.TOP_RIGHT,
            Border.BOTTOM_LEFT,
            Border.BOTTOM_RIGHT,
        )
        Border.HORIZONTAL = "─"
        Border.VERTICAL = "│"
        Border.TOP_LEFT = "╭"
        Border.TOP_RIGHT = "╮"
        Border.BOTTOM_LEFT = "╰"
        Border.BOTTOM_RIGHT = "╯"
        try:
            session = PromptSession(
                history=_SafeFileHistory(history_path),
                enable_history_search=True,
                style=Style.from_dict(_BASE_STYLE),
                completer=completer,
                complete_while_typing=False,
                key_bindings=_repl_key_bindings(self),
                prompt_continuation=_CONTINUATION,
                mouse_support=False,
                show_frame=True,
                erase_when_done=True,
                placeholder=self._placeholder_fragments,
                bottom_toolbar=self.toolbar,
            )
        finally:
            (
                Border.HORIZONTAL,
                Border.VERTICAL,
                Border.TOP_LEFT,
                Border.TOP_RIGHT,
                Border.BOTTOM_LEFT,
                Border.BOTTOM_RIGHT,
            ) = saved

        self._session = session
        self._graft_status_region(session)

    def _graft_status_region(self, session) -> None:
        """Put the live status window above the framed input of a real app,
        and let the frame's border windows pick their style per frame."""
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, Window
        from prompt_toolkit.layout.layout import walk
        from prompt_toolkit.layout.controls import FormattedTextControl

        app = getattr(session, "app", None)
        layout = getattr(app, "layout", None)
        if not isinstance(layout, Layout):
            return
        for container in walk(layout.container):
            if (
                isinstance(container, Window)
                and container.style == "class:frame.border"
            ):
                container.style = self._border_style
        status = Window(
            FormattedTextControl(self._status_fragments),
            dont_extend_height=True,
            wrap_lines=False,
        )
        root = HSplit(
            [
                ConditionalContainer(status, Condition(self.live_visible)),
                _ShrinkWrap(layout.container),
            ]
        )
        app.layout = Layout(root, focused_element=layout.current_window)
        app.min_redraw_interval = 0.02

    def _app(self):
        return getattr(self._session, "app", None)

    def _buffer(self):
        return getattr(self._session, "default_buffer", None)

    def _terminal_size(self) -> tuple[int, int] | None:
        try:
            size = self._app().output.get_size()
            return int(size.columns), int(size.rows)
        except Exception:
            return None

    def _border_style(self) -> str:
        phase = self.border_phase()
        if phase is None:
            return "class:frame.border"
        # Triangle wave, so the color breathes instead of snapping back.
        position = phase * 2
        if position > 1:
            position = 2 - position
        step = min(int(position * (_BORDER_STEPS - 1)), _BORDER_STEPS - 1)
        return f"class:frame.border class:frame.border.s{step}"

    def _placeholder_fragments(self) -> list:
        return [("class:placeholder", self.placeholder())]

    def prompt_message(self) -> list:
        return [(self.prompt_style(), fmt.PROMPT_SYMBOL)]

    def _status_fragments(self) -> list:
        width, rows = self._terminal_size() or (80, 24)
        try:
            input_rows = self._buffer().document.line_count + 3
        except Exception:
            input_rows = 4
        return self.live_fragments(width, rows, input_rows)

    def toolbar(self) -> list:
        """Bottom toolbar: a flash message when one is active, otherwise the
        contextual fragments supplied by the caller, with key hints on the
        right when there is room for them."""
        flash = self.flash_text()
        if flash is not None:
            fragments = [("class:bottom-toolbar.flash", f"  {flash}")]
        elif self._toolbar_fragments is None:
            fragments = []
        else:
            try:
                fragments = list(self._toolbar_fragments())
            except Exception:
                fragments = []
        hint = (
            "Enter queues · ^C interrupts"
            if self._busy
            else "^R history · Tab complete"
        )
        size = self._terminal_size()
        return _with_right_hint(fragments, hint, size[0] if size else None)

    def start(self) -> None:
        self._install_proxies()
        fmt.set_live_backend(self)
        self._thread = threading.Thread(
            target=self._run, name="swival-repl-ui", daemon=True
        )
        self._thread.start()
        self._ticker = threading.Thread(
            target=self._tick, name="swival-repl-tick", daemon=True
        )
        self._ticker.start()

    def stop(self) -> None:
        self._stopping = True
        with self._park_cv:
            self._park_cv.notify_all()
        self._park_ui_thread("did not stop")
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._ticker is not None:
            self._ticker.join(timeout=1)
        fmt.set_live_backend(None)

    def _park_ui_thread(self, what: str) -> None:
        """Make the UI thread leave the prompt and hand the terminal back."""
        self._exit_prompt()
        if self._thread is not None and self._thread.is_alive():
            if not self._parked.wait(timeout=_PARK_TIMEOUT):
                self._warn_stuck(what)
        self._uninstall_proxies()

    def _warn_stuck(self, what: str) -> None:
        """Say so on the real stderr when the UI thread ignores a request.

        The alternative, waiting forever, would leave a process nobody can
        interrupt: the terminal is in raw mode and the thread that reads keys
        is the one that is stuck.
        """
        stream = self._original_streams.get("stderr") or sys.__stderr__
        try:
            stream.write(
                f"\nwarning: the REPL front end {what} within "
                f"{int(_PARK_TIMEOUT)}s; continuing without it\n"
            )
            stream.flush()
        except Exception:
            pass

    def _install_proxies(self) -> None:
        if self._original_streams:
            return
        stderr = sys.stderr
        self._original_streams = {"stderr": stderr}
        self._router.set_target("stderr", stderr)
        sys.stderr = _StreamProxy(self._router, "stderr", stderr)
        stdout = sys.stdout
        if _is_tty(stdout):
            self._original_streams["stdout"] = stdout
            self._router.set_target("stdout", stdout)
            sys.stdout = _StreamProxy(self._router, "stdout", stdout)

    def _uninstall_proxies(self) -> None:
        if not self._original_streams:
            return
        self._router.flush_partial()
        sys.stderr = self._original_streams["stderr"]
        if "stdout" in self._original_streams:
            sys.stdout = self._original_streams["stdout"]
        self._original_streams = {}

    def next_event(self, timeout: float | None = None) -> ReplEvent | None:
        """Block for the next event; ``None`` when ``timeout`` elapses first."""
        with self._inbox_cv:
            if not self._inbox:
                self._inbox_cv.wait(timeout=timeout)
            if not self._inbox:
                return None
            event = self._inbox.popleft()
        # Echo a queued line right above its own turn's output.
        if event.queued and fmt._console.is_terminal:
            fmt.repl_prompt_echo(event.text)
        return event

    @contextlib.contextmanager
    def turn(self):
        """Run one turn on the main thread.

        Marks the UI busy for its duration and turns a Ctrl-C that landed
        outside the agent loop's own handlers (a state command, cleanup)
        into a warning plus a recall of the queued lines.
        """
        self.set_busy(True)
        try:
            yield
        except KeyboardInterrupt:
            fmt.warning("interrupted.")
            self.recall_queued()
        finally:
            self.set_busy(False)

    def recall_queued(self) -> None:
        """Move lines queued during an interrupted turn back into the editor.

        A follow-up typed while the model was on the wrong track should not
        run unattended after the interrupt. An exit command stays queued:
        that one the user wants honored regardless.
        """
        with self._inbox_cv:
            recalled = [
                e for e in self._inbox if e.queued and not _is_exit_command(e.text)
            ]
            for event in recalled:
                self._inbox.remove(event)
        if not recalled:
            return
        self._prepend_to_editor("\n\n".join(e.text for e in recalled))
        self.flash("queued messages moved back to the editor")

    def _prepend_to_editor(self, text: str) -> None:
        buffer = self._buffer()

        def _apply() -> None:
            try:
                current = buffer.text
                buffer.text = text if not current else text + "\n\n" + current
                buffer.cursor_position = len(buffer.text)
            except Exception:
                pass

        if not self._router.call_in_prompt(_apply):
            self._saved_document = text

    def set_busy(self, busy: bool) -> None:
        with self._lock:
            if busy == self._busy:
                return
            self._busy = busy
            self._busy_since = time.monotonic()
        self._wake.set()
        self.refresh()

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def queued(self) -> list[str]:
        with self._inbox_cv:
            return [e.text for e in self._inbox if e.queued]

    def flash(self, text: str, seconds: float = _FLASH_SECONDS) -> None:
        with self._lock:
            self._flash = (text, time.monotonic() + seconds)
        self._wake.set()
        self.refresh()

    def flash_text(self) -> str | None:
        with self._lock:
            if self._flash is None:
                return None
            text, until = self._flash
            if time.monotonic() >= until:
                self._flash = None
                return None
            return text

    @contextlib.contextmanager
    def _slot(self, slot: _Slot, handle=None):
        with self._lock:
            self._slots.append(slot)
        self._wake.set()
        self.refresh()

        def dismiss() -> None:
            with self._lock:
                if slot in self._slots:
                    self._slots.remove(slot)
            self.refresh()

        try:
            yield dismiss if handle is None else handle
        finally:
            dismiss()

    def llm_spinner(self, label: str):
        return self._slot(_SpinnerSlot(label))

    def command_spinner(self, label: str, timeout: float | None):
        return self._slot(_CommandSlot(label, timeout))

    def step_spinner(self, label: str):
        slot = _StepSlot(label)
        return self._slot(slot, slot.update)

    def input_marquee(self, text: str):
        return self._slot(_MarqueeSlot(text, None, 0.0))

    def input_marquee_then_spinner(self, text: str, spinner_label: str, delay: float):
        return self._slot(_MarqueeSlot(text, spinner_label, delay))

    def stream_raw(self):
        slot = _StreamSlot()
        return self._slot(slot, lambda text: slot.update(answer=text))

    def stream_channels(self):
        slot = _StreamSlot()
        return self._slot(slot, slot.update)

    def suspend(self):
        return self.suspended()

    def slots(self) -> list[_Slot]:
        with self._lock:
            return list(self._slots)

    def refresh(self) -> None:
        app = self._router.app
        if app is None:
            return
        try:
            app.invalidate()
        except Exception:
            pass

    def _animating(self) -> bool:
        with self._lock:
            if self._slots or self._flash is not None:
                return True
            return self._busy and fmt.animations_enabled()

    def _tick(self) -> None:
        while not self._stopping:
            if self._animating():
                self.refresh()
                time.sleep(_TICK_SECONDS)
            else:
                self._wake.wait(timeout=1.0)
                self._wake.clear()

    @contextlib.contextmanager
    def suspended(self):
        """Give the raw terminal to the main thread for a nested prompt."""
        with self._park_cv:
            self._suspend_count += 1
            first = self._suspend_count == 1
        if first:
            self._park_ui_thread("did not release the terminal")
        try:
            yield
        finally:
            with self._park_cv:
                self._suspend_count -= 1
                if self._suspend_count == 0:
                    self._install_proxies()
                    self._park_cv.notify_all()

    def _exit_prompt(self) -> None:
        app = self._router.app
        if app is None:
            return

        def _exit() -> None:
            try:
                if not app.is_done:
                    app.exit(result=_Leave)
            except Exception:
                pass

        self._router.call_in_prompt(_exit)

    def interrupt_turn(self) -> None:
        """Ctrl-C while a turn is running."""
        now = time.monotonic()
        if now - self._last_interrupt < _INTERRUPT_DEBOUNCE:
            return
        self._last_interrupt = now
        self.flash("interrupting…")
        _interrupt_main_thread()

    def _handle_idle_interrupt(self) -> bool:
        """Ctrl-C at an idle prompt. Returns True when the session should end."""
        text = getattr(self._buffer(), "text", "") or ""
        if text:
            self._ctrl_c_armed = False
            return False
        if not self._ctrl_c_armed and self._has_conversation():
            self._ctrl_c_armed = True
            fmt.info("Press Ctrl-C again or Ctrl-D to exit")
            return False
        return True

    def _run(self) -> None:
        try:
            while self._wait_unparked():
                default = self._saved_document
                self._saved_document = None
                try:
                    result = self._session.prompt(
                        self.prompt_message,
                        default=default if default is not None else "",
                        pre_run=self._on_pre_run,
                    )
                except KeyboardInterrupt:
                    self._router.prompt_finished()
                    if self._handle_idle_interrupt():
                        break
                    continue
                except EOFError:
                    self._router.prompt_finished()
                    if self._busy:
                        self.flash("a turn is running: Ctrl-C interrupts it")
                        continue
                    break
                except BaseException as exc:
                    self._router.prompt_finished()
                    if not isinstance(exc, StopIteration):
                        fmt.error(f"prompt failed: {type(exc).__name__}: {exc}")
                    break
                self._router.prompt_finished()
                if result is _Leave:
                    self._save_document()
                    continue
                self._ctrl_c_armed = False
                text = result if isinstance(result, str) else ""
                self._submit(text)
                if _is_exit_command(text):
                    # The main loop exits once the current turn is over.
                    break
        finally:
            self._parked.set()
            with self._inbox_cv:
                self._inbox.append(ReplEvent("exit"))
                self._inbox_cv.notify()

    def _wait_unparked(self) -> bool:
        with self._park_cv:
            while self._suspend_count > 0 and not self._stopping:
                self._parked.set()
                self._park_cv.wait()
            self._parked.clear()
            return not self._stopping

    def _on_pre_run(self) -> None:
        app = self._session.app
        self._router.prompt_started(app, app.loop, contextvars.copy_context())
        # A stop or suspend requested between two prompts had no app to exit.
        if self._stopping or self._suspend_count > 0:
            app.exit(result=_Leave)

    def _save_document(self) -> None:
        document = getattr(self._buffer(), "document", None)
        if document is not None and document.text:
            self._saved_document = document

    def _submit(self, text: str) -> None:
        if not text.strip():
            return
        event = ReplEvent("line", text, queued=self._busy)
        if not event.queued and fmt._console.is_terminal:
            fmt.repl_prompt_echo(text)
        with self._inbox_cv:
            self._inbox.append(event)
            self._inbox_cv.notify()
        if event.queued:
            self.flash("queued, runs after the current turn")
        self.refresh()

    def live_visible(self) -> bool:
        with self._lock:
            if self._slots:
                return True
        return bool(self.queued)

    def live_fragments(self, width: int, rows: int, input_rows: int) -> list:
        """Fragments for the status region: active slots, then queued lines."""
        slots = self.slots()
        queued = self.queued
        budget = max(rows - input_rows - len(queued) - 4, _STREAM_MIN_ROWS)
        fragments: list = []
        for slot in slots:
            try:
                renderable = slot.render(width, budget)
            except Exception:
                continue
            if renderable is None:
                continue
            text = _render_ansi(renderable, width)
            if not text:
                continue
            if fragments:
                fragments.append(("", "\n"))
            fragments.extend(ANSI(text).__pt_formatted_text__())
        for index, line in enumerate(queued):
            if fragments:
                fragments.append(("", "\n"))
            label = "queued" if index == 0 else "      "
            fragments.append(("class:queued.label", f"  {label} "))
            fragments.append(("class:queued.text", _preview(line, width - 12)))
        return fragments

    def placeholder(self) -> str:
        return _BUSY_PLACEHOLDER if self._busy else _IDLE_PLACEHOLDER

    def prompt_style(self) -> str:
        return "class:prompt.busy" if self._busy else "class:prompt"

    def border_phase(self) -> float | None:
        """Position in the breathing cycle while busy, ``None`` when idle or
        when decorative animations are turned off."""
        if not self._busy or not fmt.animations_enabled():
            return None
        t = (time.monotonic() - self._busy_since) / _BORDER_CYCLE_SECONDS
        return t - int(t)


def _with_right_hint(fragments: list, hint: str, width: int | None) -> list:
    """Append ``hint`` right-aligned when the toolbar has room for it."""
    if width is None:
        return fragments
    gap = width - fragment_list_width(fragments) - len(hint) - 2
    if gap < 4:
        return fragments
    return fragments + [("", " " * gap), ("class:bottom-toolbar.tip", hint)]


class _ShrinkWrap(Container):
    """Give a container exactly its preferred height.

    The renderer hands the root layout every row below the cursor, and the
    framed input would grow to fill them. Capping the height at the preferred
    value keeps the box tight around the text while completion menus still
    get the space they reserve.
    """

    def __init__(self, content) -> None:
        self.content = content

    def reset(self) -> None:
        self.content.reset()

    def preferred_width(self, max_available_width: int):
        return self.content.preferred_width(max_available_width)

    def preferred_height(self, width: int, max_available_height: int):
        d = self.content.preferred_height(width, max_available_height)
        preferred = max(d.preferred, d.min)
        return Dimension(min=d.min, max=preferred, preferred=preferred)

    def write_to_screen(self, *args, **kwargs) -> None:
        self.content.write_to_screen(*args, **kwargs)

    def is_modal(self) -> bool:
        return self.content.is_modal()

    def get_key_bindings(self):
        return self.content.get_key_bindings()

    def get_children(self):
        return [self.content]


def _render_ansi(renderable, width: int) -> str:
    """Render a Rich renderable to an ANSI string at the given width."""
    real = fmt._console
    console = Console(
        file=io.StringIO(),
        force_terminal=True,
        color_system=real.color_system,
        no_color=real.no_color,
        width=max(width, 1),
        legacy_windows=False,
        highlight=False,
        markup=False,
        emoji=False,
    )
    console.print(renderable, end="")
    return console.file.getvalue().rstrip("\n")

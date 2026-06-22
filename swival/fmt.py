"""ANSI-formatted stderr output using Rich."""

import contextlib
import difflib
import math
import threading
import time

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.spinner import Spinner
from rich.segment import Segment
from rich.style import Style
from rich.text import Text

_console = Console(stderr=True)
_stdout_console = Console(stderr=False)

_think_count = 0

_active_live_suspend = None


def _noop(*_args, **_kwargs) -> None:
    return


@contextlib.contextmanager
def suspend_live():
    """Temporarily stop the active live display so interactive prompts render cleanly.

    No-op when no live display is running. The display resumes (with its
    timing reset) when the block exits.
    """
    suspend = _active_live_suspend
    if suspend is None:
        yield
        return
    resume = suspend()
    try:
        yield
    finally:
        resume()


def reset_state() -> None:
    """Reset all module-level rendering state (think tree counter, etc.)."""
    global _think_count
    _think_count = 0


def init(*, color: bool = False, no_color: bool = False) -> None:
    """Reconfigure the module-level console from CLI flags.

    Call once at startup, before any output.
    """
    global _console, _stdout_console
    kwargs: dict = {"stderr": True}
    stdout_kwargs: dict = {"stderr": False}
    if color:
        kwargs["force_terminal"] = True
        kwargs["no_color"] = False
    if no_color:
        kwargs["no_color"] = True
        stdout_kwargs["no_color"] = True
    _console = Console(**kwargs)
    _stdout_console = Console(**stdout_kwargs)


# -- Turn structure ----------------------------------------------------------


_TURN_GRADIENT = [
    (0, 180, 220),  # cyan
    (60, 120, 220),  # blue
    (160, 80, 200),  # magenta
]


class _GradientRule:
    """A horizontal rule with a gradient color ramp and centered title."""

    def __init__(self, title: str):
        self.title = title

    def __rich_console__(self, console, options):
        width = options.max_width
        title = f" {self.title} "
        side = max((width - len(title)) // 2, 0)
        text = Text()
        for i in range(side):
            t = i / max(width - 1, 1)
            r, g, b = _lerp_color(_TURN_GRADIENT, t)
            text.append("─", style=Style(color=f"rgb({r},{g},{b})"))
        text.append(title, style=Style(bold=True, color="white"))
        for i in range(side + len(title), width):
            t = i / max(width - 1, 1)
            r, g, b = _lerp_color(_TURN_GRADIENT, t)
            text.append("─", style=Style(color=f"rgb({r},{g},{b})"))
        yield from text.__rich_console__(console, options)


def _turn_title(n: int, max_n: int, token_est: int, context_length: int | None) -> str:
    if context_length:
        pct = token_est * 100 // context_length
        return f"Turn {n}/{max_n} (~{token_est:,} / {context_length:,} tokens, {pct}%)"
    return f"Turn {n}/{max_n} (~{token_est:,} tokens)"


def turn_header(
    n: int, max_n: int, token_est: int, context_length: int | None = None
) -> None:
    reset_state()
    _console.print()
    title = _turn_title(n, max_n, token_est, context_length)
    if _console.is_terminal:
        _console.print(_GradientRule(title))
    else:
        _console.print(Rule(title, style="cyan"))


def llm_timing(elapsed: float, finish_reason: str) -> None:
    style = "green" if finish_reason == "stop" else "yellow"
    text = Text()
    text.append(f"  LLM responded in {elapsed:.1f}s", style=style)
    text.append(f"  finish_reason={escape(str(finish_reason))}", style=style)
    _console.print(text)


_SPINNER_PHASES: list[tuple[float, str, str, str]] = [
    # (min_seconds, spinner_name, style, verb)
    (0, "dots", "cyan", "Thinking"),
    (3, "dots2", "cyan", "Reasoning"),
    (8, "dots3", "blue", "Composing"),
    (15, "dots", "magenta", "Elaborating"),
    (25, "dots2", "blue", "Refining"),
    (40, "dots3", "cyan", "Polishing"),
]


@contextlib.contextmanager
def llm_spinner(label: str = "Thinking"):
    """Context manager showing a phase-cycling spinner on stderr.

    The spinner style and label evolve over time to give the perception
    of progress through distinct work stages. Yields a ``dismiss()``
    callable that early-stops the display so a different live region
    can take over.
    """
    if not _console.is_terminal:
        yield _noop
        return

    suffix = ""
    if "(" in label:
        idx = label.index("(")
        suffix = " " + label[idx:].strip()
        initial_desc = f"{_SPINNER_PHASES[0][3]}{suffix}"
    else:
        initial_desc = label

    spinner_col = SpinnerColumn("dots", style="cyan", speed=1.5)
    progress = Progress(
        spinner_col,
        TextColumn("  {task.description}"),
        TimeElapsedColumn(),
        console=_console,
        transient=True,
        refresh_per_second=16,
    )

    stop = threading.Event()
    dismissed = threading.Event()

    def _cycle_phases(task_id):
        t0 = time.monotonic()
        phase_idx = 0
        while not stop.wait(0.3):
            elapsed = time.monotonic() - t0
            new_idx = phase_idx
            for i, (threshold, _, _, _) in enumerate(_SPINNER_PHASES):
                if elapsed >= threshold:
                    new_idx = i
            if new_idx != phase_idx:
                phase_idx = new_idx
                _, name, style, verb = _SPINNER_PHASES[phase_idx]
                spinner_col.spinner = Spinner(name, style=style, speed=1.5)
                progress.update(task_id, description=f"{verb}{suffix}")

    progress.start()
    tid = progress.add_task(initial_desc, total=None)
    t = threading.Thread(target=_cycle_phases, args=(tid,), daemon=True)
    t.start()

    def dismiss() -> None:
        if dismissed.is_set():
            return
        dismissed.set()
        stop.set()
        t.join(timeout=1)
        progress.stop()

    try:
        yield dismiss
    finally:
        dismiss()


@contextlib.contextmanager
def command_spinner(label: str, timeout: float | None = None):
    """Progress display on stderr while a shell command runs.

    With a positive *timeout*, shows a bar filling toward the timeout deadline
    plus an elapsed timer; without one, falls back to an indeterminate spinner.
    Transient: the line is wiped on exit so the command's captured output prints
    cleanly afterwards. No-op when stderr is not a terminal. Yields a
    ``dismiss()`` callable that early-stops the display.

    Only one Rich live display may be active at a time, so callers must not
    start this while another live display (e.g. the LLM spinner) is running.
    """
    if not _console.is_terminal:
        yield _noop
        return

    label = " ".join((label or "command").split())
    if len(label) > 50:
        label = label[:49] + "…"

    determinate = bool(timeout and timeout > 0)
    columns = [
        SpinnerColumn("dots", style="cyan", speed=1.5),
        TextColumn("  Running {task.description}"),
    ]
    if determinate:
        columns += [
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            TextColumn("[dim]of timeout[/dim]"),
        ]
    columns.append(TimeElapsedColumn())

    progress = Progress(
        *columns,
        console=_console,
        transient=True,
        refresh_per_second=12,
    )

    stop = threading.Event()
    dismissed = threading.Event()
    progress.start()
    task_id = progress.add_task(
        escape(label), total=timeout if determinate else None
    )

    advancer: threading.Thread | None = None
    if determinate:
        (task,) = progress.tasks

        def _advance():
            while not stop.wait(0.1):
                progress.update(task_id, completed=min(task.elapsed, timeout))

        advancer = threading.Thread(target=_advance, daemon=True)
        advancer.start()

    def _suspend():
        progress.stop()

        def _resume():
            if dismissed.is_set():
                return
            progress.reset(task_id, total=timeout if determinate else None)
            progress.start()

        return _resume

    global _active_live_suspend
    _active_live_suspend = _suspend

    def dismiss() -> None:
        global _active_live_suspend
        if dismissed.is_set():
            return
        dismissed.set()
        if _active_live_suspend is _suspend:
            _active_live_suspend = None
        stop.set()
        if advancer is not None:
            advancer.join(timeout=1)
        progress.stop()

    try:
        yield dismiss
    finally:
        dismiss()


_INPUT_MARQUEE_PREFIX = "  > "
_INPUT_MARQUEE_SEPARATOR = "   ·   "


def _input_marquee_text(text: str, offset: int, width: int) -> Text:
    flat = " ".join(text.split()) or "Thinking"
    base = flat + _INPUT_MARQUEE_SEPARATOR
    content_width = max(width - len(_INPUT_MARQUEE_PREFIX), 20)
    start = offset % len(base)
    repeats = math.ceil((start + content_width) / len(base)) + 1
    tiled = base * repeats

    line = Text(_INPUT_MARQUEE_PREFIX, style="bold cyan")
    line.append(tiled[start : start + content_width], style="cyan")
    return line


@contextlib.contextmanager
def input_marquee(text: str):
    """Context manager showing a scrolling marquee of ``text`` on stderr.

    Used by the REPL as visual feedback while the LLM is processing the
    user's prompt. The line is cleared as soon as the context exits.
    """
    if not _console.is_terminal:
        yield _noop
        return

    def _frame(offset: int) -> Text:
        return _input_marquee_text(text, offset, _console.width)

    stop = threading.Event()
    dismissed = threading.Event()
    live = Live(
        _frame(0),
        console=_console,
        auto_refresh=False,
        transient=True,
    )

    def _scroll():
        offset = 0
        while not stop.wait(0.06):
            offset += 1
            try:
                live.update(_frame(offset), refresh=True)
            except Exception:
                break

    live.start()
    t = threading.Thread(target=_scroll, daemon=True)
    t.start()

    def dismiss() -> None:
        if dismissed.is_set():
            return
        dismissed.set()
        stop.set()
        t.join(timeout=1)
        live.stop()

    try:
        yield dismiss
    finally:
        dismiss()


@contextlib.contextmanager
def input_marquee_then_spinner(text: str, spinner_label: str, delay: float = 4.0):
    """Show the input marquee, then swap to the labeled spinner after ``delay``.

    Gives instant feedback (the scrolling tail of what was sent to the model)
    while keeping the turn-state spinner available for slower prefills. The
    yielded ``dismiss()`` stops whichever indicator is currently active and
    cancels any pending transition.
    """
    if not _console.is_terminal:
        yield _noop
        return

    marquee_cm = input_marquee(text)
    marquee_dismiss = marquee_cm.__enter__()

    lock = threading.Lock()
    state = {"phase": "marquee", "dismissed": False}
    spinner_cm: dict = {"cm": None, "dismiss": None}
    stop_timer = threading.Event()

    def _transition():
        if stop_timer.wait(delay):
            return
        try:
            with lock:
                if state["dismissed"] or state["phase"] != "marquee":
                    return
                state["phase"] = "transitioning"

            marquee_dismiss()
            cm = llm_spinner(spinner_label)
            d = cm.__enter__()

            with lock:
                spinner_cm["cm"] = cm
                spinner_cm["dismiss"] = d
                if state["dismissed"]:
                    state["phase"] = "spinner"
                    needs_dismiss = True
                else:
                    state["phase"] = "spinner"
                    needs_dismiss = False
            if needs_dismiss:
                d()
        except Exception as exc:
            with lock:
                state["phase"] = "failed"
            warning(f"prefill spinner transition failed: {exc!r}")

    t = threading.Thread(target=_transition, daemon=True)
    t.start()

    def dismiss() -> None:
        with lock:
            if state["dismissed"]:
                return
            state["dismissed"] = True
            stop_timer.set()
            phase = state["phase"]
            sd = spinner_cm["dismiss"]
        if phase == "marquee":
            marquee_dismiss()
        elif phase == "spinner" and sd is not None:
            sd()
        # phase == "transitioning": the transition thread will observe
        # state["dismissed"] after entering the spinner and dismiss it.
        # phase == "failed": both CMs are already torn down.

    try:
        yield dismiss
    finally:
        dismiss()
        t.join(timeout=0.5)
        marquee_cm.__exit__(None, None, None)
        if spinner_cm["cm"] is not None:
            spinner_cm["cm"].__exit__(None, None, None)


def _wrap_to_rows(line: str, width: int) -> list[str]:
    """Split a single source line into the visual rows it occupies at *width*,
    measured in display cells. An empty line is one (empty) row."""
    from rich.cells import cell_len

    if not line:
        return [""]
    rows: list[str] = []
    cur: list[str] = []
    cur_w = 0
    for ch in line:
        w = cell_len(ch)
        if cur and cur_w + w > width:
            rows.append("".join(cur))
            cur, cur_w = [ch], w
        else:
            cur.append(ch)
            cur_w += w
    rows.append("".join(cur))
    return rows


def _collapse_blank_rows(rows: list[str]) -> list[str]:
    """Collapse runs of blank rows to a single blank and drop trailing blanks.

    Models routinely emit long runs of empty lines. Left alone they fill the
    streaming viewport with nothing, pushing real content off the top and
    sometimes leaving the region visually empty. A row counts as blank when it
    has no non-whitespace content, which also catches whitespace-only wrapped
    rows. Paragraph separation is preserved by keeping one blank per run.
    """
    out: list[str] = []
    prev_blank = False
    for row in rows:
        blank = not row.strip()
        if blank and prev_blank:
            continue
        out.append(row)
        prev_blank = blank
    while out and not out[-1].strip():
        out.pop()
    return out


def _wrap_and_tail(text: str, width: int, height: int) -> list[str]:
    """Return the last *height* visual rows of *text*, wrapped at *width*.

    Tails by visual rows rather than source lines, so a single paragraph that
    wraps to more than *height* rows is trimmed to its final rows instead of
    overflowing. This keeps the newest streamed output on screen; Rich's own
    overflow handling would crop from the top instead. Blank runs are collapsed
    before tailing so the visible window stays dense with real content.

    This is the wrapping/tailing half of the streaming display, kept free of
    styling so per-channel renderers can apply their own styles to the rows.
    """
    width = max(width, 1)
    rows: list[str] = []
    for line in text.split("\n"):
        rows.extend(_wrap_to_rows(line, width))
    rows = _collapse_blank_rows(rows)
    return rows[-max(height, 1) :]


def _tail_to_viewport(text: str, width: int, height: int) -> Text:
    """Dim-styled tail of *text* for the legacy single-channel stream view."""
    return Text("\n".join(_wrap_and_tail(text, width, height)), style="dim")


# When the answer has started streaming, the reasoning channel is demoted to a
# small tail above the answer so a long chain of thought can't crowd out the
# reply the user is actually waiting for.
_THINK_TAIL_ROWS = 3


def render_stream_channels(
    reasoning: str, answer: str, activity: str, width: int, height: int
) -> Text:
    """Compose the three streaming channels into a single viewport-sized Text.

    Layout, top to bottom: a dim/italic ``thinking…`` block, the normal-weight
    answer, then a dim activity tail for streamed tool-call metadata. Row budget
    is allocated by state so the answer keeps priority once it begins:

    - no reasoning and no activity: identical to the legacy single-stream view;
    - reasoning only (answer not started): reasoning fills the viewport;
    - answer present: the answer takes most rows, reasoning keeps a short tail.
    """
    width = max(width, 1)
    height = max(height, 1)
    reasoning = reasoning or ""
    answer = answer or ""
    activity = activity or ""
    has_r = bool(reasoning.strip())
    has_a = bool(answer.strip())
    has_v = bool(activity.strip())

    if not has_r and not has_v:
        return _tail_to_viewport(answer, width, height)

    activity_rows = _wrap_and_tail(activity, width, min(2, height)) if has_v else []
    body = max(height - len(activity_rows), 1)

    if has_r:
        header = 1
        if has_a:
            think_budget = min(_THINK_TAIL_ROWS, max(body - header - 1, 1))
        else:
            think_budget = max(body - header, 1)
        think_rows = _wrap_and_tail(reasoning, width, think_budget)
    else:
        think_rows = []

    answer_budget = body - len(think_rows) - (1 if has_r else 0)
    answer_rows = _wrap_and_tail(answer, width, max(answer_budget, 1)) if has_a else []

    text = Text()
    first = True

    def _emit(row: str, style: str) -> None:
        nonlocal first
        if not first:
            text.append("\n")
        text.append(row, style=style)
        first = False

    if has_r:
        _emit("thinking…", "dim italic")
        for row in think_rows:
            _emit(row, "dim italic")
    for row in answer_rows:
        _emit(row, "")
    for row in activity_rows:
        _emit(row, "dim")
    return text


@contextlib.contextmanager
def stream_raw():
    """Context manager that displays streamed LLM output as plain text on stderr.

    Yields an ``update(text)`` callable that redraws the tail of the accumulated
    text as it arrives. The text is shown unformatted; the live region is
    transient, so it is wiped on exit and the caller can re-print the finished
    response with formatting.
    """
    if not _console.is_terminal:
        yield _noop
        return

    with Live(
        console=_console,
        transient=True,
        auto_refresh=False,
        vertical_overflow="crop",
    ) as live:

        def update(text: str) -> None:
            height = max(_console.height - 2, 1)
            live.update(
                _tail_to_viewport(text, max(_console.width, 1), height),
                refresh=True,
            )

        yield update


@contextlib.contextmanager
def stream_channels():
    """Channel-aware variant of :func:`stream_raw`.

    Yields an ``update(reasoning, answer, activity)`` callable that redraws the
    viewport with thinking, answer, and tool-call activity styled distinctly.
    Like ``stream_raw`` the region is transient and a no-op off a terminal.
    """
    if not _console.is_terminal:
        yield _noop
        return

    with Live(
        console=_console,
        transient=True,
        auto_refresh=False,
        vertical_overflow="crop",
    ) as live:

        def update(reasoning: str = "", answer: str = "", activity: str = "") -> None:
            height = max(_console.height - 2, 1)
            live.update(
                render_stream_channels(
                    reasoning, answer, activity, max(_console.width, 1), height
                ),
                refresh=True,
            )

        yield update


def thinking_block(text: str) -> None:
    """Print retained reasoning to stderr after a transient stream is wiped.

    Used by the opt-in ``--show-thinking`` retention so the thinking the user
    watched scroll by survives in scrollback. Dim/italic, stderr only — never
    the final-answer stdout channel.
    """
    if not text or not text.strip():
        return
    _console.print(Text("thinking", style="bold dim"))
    for line in text.splitlines():
        _console.print(Text(line, style="dim italic"))


def thinking_summary(lines: int, tokens: int) -> None:
    """Print a one-line collapsed note that reasoning was streamed and hidden."""
    _console.print(
        Text(f"  thinking: {lines} lines / ~{tokens} tokens, hidden", style="dim")
    )


def completion(turns: int, exit_code: str) -> None:
    if exit_code == "ok":
        _console.print(
            Text(f"  \u2713 Agent finished: {turns} turns", style="bold green")
        )
    else:
        _console.print(
            Text(f"  Agent finished: {turns} turns, exit={exit_code}", style="bold red")
        )


# -- Tool calls --------------------------------------------------------------


def tool_call(name: str, args_json: str) -> None:
    header = Text()
    header.append("  \u25b6 ", style="bold magenta")
    header.append(name, style="bold magenta")
    _console.print(header)
    if args_json:
        for line in args_json.splitlines():
            _console.print(Text(f"    {line}", style="dim"))


def tool_result(name: str, elapsed: float, preview: str) -> None:
    header = Text()
    header.append(f"  \u2713 {name}", style="green")
    header.append(f"  {elapsed:.1f}s", style="green")
    _console.print(header)
    if preview:
        _console.print(Text(f"    {preview}", style="dim"))


_DIFF_MAX_LINES = 50
_DIFF_MAX_BYTES = 4096

_FRAME_RESERVE = 8


def _sanitize_title(title, *, max_width: int) -> Text:
    """Return a single-line Text safe to use as a Panel title.

    Strips newlines, drops markup interpretation, and truncates to roughly
    ``max_width`` characters with an ellipsis if needed.
    """
    if isinstance(title, Text):
        plain = title.plain
        styled = title
    else:
        plain = str(title)
        styled = Text(plain)

    if "\n" in plain or "\r" in plain:
        plain = plain.replace("\r", " ").replace("\n", " ")
        styled = Text(plain)

    if max_width > 1 and len(plain) > max_width:
        styled = Text(plain[: max_width - 1] + "…")

    return styled


def _framed(
    body,
    *,
    title,
    subtitle=None,
    border_style: str = "dim",
) -> bool:
    """Print ``body`` inside a Rich Panel with sanitized title/subtitle.

    Returns True if a panel was printed, False when stderr is not a TTY
    (callers should then emit their own plain-text fallback).
    """
    if not _console.is_terminal:
        return False

    subtitle_obj = subtitle
    subtitle_width = 0
    if subtitle is not None:
        subtitle_obj = subtitle if isinstance(subtitle, Text) else Text(str(subtitle))
        subtitle_width = len(subtitle_obj.plain)

    available = max(_console.width - _FRAME_RESERVE - subtitle_width, 8)
    safe_title = _sanitize_title(title, max_width=available)

    panel = Panel(
        body,
        title=safe_title,
        title_align="left",
        subtitle=subtitle_obj,
        subtitle_align="right",
        border_style=border_style,
        padding=(0, 1),
    )
    _console.print(panel)
    return True


def tool_diff(file_path: str, old: str, new: str) -> None:
    """Print a colored unified diff of an edit to stderr."""
    diff_lines = list(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=file_path,
            tofile=file_path,
        )
    )
    if not diff_lines:
        return

    additions = sum(
        1 for dl in diff_lines if dl.startswith("+") and not dl.startswith("+++")
    )
    deletions = sum(
        1 for dl in diff_lines if dl.startswith("-") and not dl.startswith("---")
    )

    is_tty = _console.is_terminal
    output = Text()
    total_bytes = 0
    shown = 0
    for line in diff_lines:
        if shown >= _DIFF_MAX_LINES or total_bytes >= _DIFF_MAX_BYTES:
            remaining = len(diff_lines) - shown
            output.append(f"... {remaining} more lines\n", style="dim")
            break
        if line.startswith("---") or line.startswith("+++"):
            style = "bold"
        elif line.startswith("@@"):
            style = "cyan"
        elif line.startswith("-"):
            style = "red"
        elif line.startswith("+"):
            style = "green"
        else:
            style = "dim"
        encoded = line.encode("utf-8")
        budget = _DIFF_MAX_BYTES - total_bytes
        if len(encoded) > budget:
            encoded = encoded[:budget]
            line = encoded.decode("utf-8", errors="ignore")
        display = line if is_tty else f"    {line}"
        output.append(display, style=style)
        if not display.endswith("\n"):
            output.append("\n")
        total_bytes += len(encoded)
        shown += 1

    subtitle = Text()
    subtitle.append(f"+{additions}", style="green")
    subtitle.append(" / ", style="dim")
    subtitle.append(f"-{deletions}", style="red")

    if not _framed(output, title=file_path, subtitle=subtitle):
        _console.print(output, end="")


_FETCH_MAX_LINES = 30
_FETCH_MAX_BYTES = 4096


def tool_fetch(result) -> None:
    """Print a framed preview of a fetch_url result to stderr.

    Body matches the string returned to the model (success content,
    save-to-disk notice, or error). Subtitle carries status, raw size, and
    content-type when known.
    """
    if not _console.is_terminal:
        return

    body_text = result.body
    head = body_text[: _FETCH_MAX_BYTES * 2]
    lines = head.splitlines(keepends=True)
    has_tail = len(head) < len(body_text)
    output = Text()
    total_bytes = 0
    shown = 0
    truncated_inline = False
    for line in lines:
        if shown >= _FETCH_MAX_LINES or total_bytes >= _FETCH_MAX_BYTES:
            remaining = len(lines) - shown + (1 if has_tail else 0)
            output.append(f"... {remaining} more lines\n", style="dim")
            break
        encoded = line.encode("utf-8")
        budget = _FETCH_MAX_BYTES - total_bytes
        if len(encoded) > budget:
            truncated_inline = True
            encoded = encoded[:budget]
            line = encoded.decode("utf-8", errors="ignore")
        output.append(line)
        if not line.endswith("\n"):
            output.append("\n")
        total_bytes += len(encoded)
        shown += 1
    else:
        if truncated_inline or has_tail:
            output.append(f"... truncated at {_FETCH_MAX_BYTES} bytes\n", style="dim")

    parts = []
    if result.status is not None:
        parts.append(str(result.status))
    if result.raw_bytes:
        parts.append(f"{result.raw_bytes} B")
    if result.content_type:
        parts.append(result.content_type.split(";", 1)[0].strip())
    if result.saved_path:
        parts.append(f"saved {result.saved_path}")
    subtitle = Text(" · ".join(parts), style="dim") if parts else None

    is_error = result.body.startswith("error:")
    border = "red" if is_error else "dim"

    _framed(output, title=result.final_url, subtitle=subtitle, border_style=border)


def tool_error(name: str, msg: str) -> None:
    header = Text()
    header.append(f"  \u2717 {name}", style="bold red")
    header.append(f"  {msg}", style="red")
    _console.print(header)


def tool_repair(name: str, repairs: list[dict]) -> None:
    for r in repairs:
        line = Text()
        line.append(f"  ~ {name}", style="bold yellow")
        line.append(f"  repaired: {r['type']} on {r.get('field', '?')}", style="yellow")
        _console.print(line)


def truncation_repair(tool_name: str, notes: list[str]) -> None:
    line = Text()
    line.append(f"  ~ {tool_name}", style="bold yellow")
    detail = "; ".join(notes) if notes else "repaired truncated args"
    line.append(f"  truncation repair: {detail}", style="yellow")
    _console.print(line)


def scavenged_call(tool_name: str, source: str) -> None:
    line = Text()
    line.append(f"  + {tool_name}", style="bold cyan")
    line.append(f"  scavenged from {source}", style="cyan")
    _console.print(line)


def storm_suppression(tool_name: str, count: int, reason: str) -> None:
    line = Text()
    line.append("  ⚠ Storm guard: ", style="bold yellow")
    line.append(
        f"suppressed {tool_name} (call #{count}). {reason}",
        style="yellow",
    )
    _console.print(line)


def guardrail(tool_name: str, count: int, error: str) -> None:
    line = Text()
    line.append("  \u26a0 Guardrail: ", style="bold yellow")
    line.append(
        f"{tool_name} repeated the same error {count} times. Last error: {error}",
        style="yellow",
    )
    _console.print(line)


# -- Think steps -------------------------------------------------------------


def think_step(
    number: int,
    total: int,
    text: str,
    *,
    is_revision: bool = False,
    revises_thought: int | None = None,
    branch_id: str | None = None,
    branch_from_thought: int | None = None,
) -> None:
    global _think_count

    if _think_count == 0:
        _console.print(Text("  [think]", style="yellow"))
    _think_count += 1

    line = Text()
    if is_revision and revises_thought is not None:
        line.append("  \u2502  \u2514\u2500 ", style="yellow")
        line.append(f"rev: {text}", style="dim italic")
    elif branch_id is not None and branch_from_thought is not None:
        line.append("  \u251c\u2500 ", style="yellow")
        line.append(f"[branch:{branch_id}] ", style="yellow")
        line.append(text, style="dim italic")
    else:
        line.append("  \u251c\u2500 ", style="yellow")
        line.append(text, style="dim italic")
    _console.print(line)


# -- Todo updates ------------------------------------------------------------


def todo_update(action: str, detail: str) -> None:
    prefix_map = {"add": "+1", "done": "\u2713", "remove": "-1", "cleared": "cleared"}
    tag = prefix_map.get(action, action)
    line = Text()
    line.append(f"  [todo {tag}]", style="yellow")
    line.append(f" {detail}", style="dim italic")
    _console.print(line)


def todo_list(
    items: list,
    action: str | None = None,
    changed_task: str | None = None,
    note: str | None = None,
) -> None:
    """Render the full todo checklist with an optional action annotation."""
    remaining = sum(1 for i in items if not i.done)
    header = Text()
    header.append("  [todo]", style="yellow")
    header.append(f" {remaining} remaining", style="dim")
    if note:
        header.append(f"  ({note})", style="dim italic")
    _console.print(header)
    for item in items:
        line = Text()
        is_changed = changed_task is not None and item.text == changed_task
        if item.done:
            line.append("  \u2611 ", style="dim")
            line.append(item.text, style="bold dim" if is_changed else "dim")
        else:
            line.append("  \u2610 ", style="")
            line.append(item.text, style="bold" if is_changed else "")
        _console.print(line)


# -- Assistant text ----------------------------------------------------------

_ASSISTANT_MAX_LINES = 100


class _LeftBar:
    """Renders a child renderable with a blue left-border bar (│)."""

    def __init__(self, renderable):
        self.renderable = renderable

    def __rich_console__(self, console, options):
        inner_width = max(options.max_width - 4, 20)
        inner_options = options.update_width(inner_width)
        lines = console.render_lines(self.renderable, inner_options, pad=False)
        bar = Segment("  │ ", Style(color="blue"))
        newline = Segment("\n")
        for line in lines:
            yield bar
            yield from line
            yield newline


def assistant_text(text: str) -> None:
    src_lines = text.split("\n")
    if len(src_lines) > _ASSISTANT_MAX_LINES:
        remaining = len(src_lines) - _ASSISTANT_MAX_LINES
        text = "\n".join(src_lines[:_ASSISTANT_MAX_LINES])
        md = Markdown(text)
        _console.print(_LeftBar(md), end="")
        _console.print(
            Text(f"  │ ... {remaining} more lines (truncated)", style="blue dim")
        )
    else:
        md = Markdown(text)
        _console.print(_LeftBar(md), end="")


def repl_answer(text: str) -> None:
    """Print a REPL answer to stdout, with syntax highlighting when on a TTY."""
    if _stdout_console.is_terminal and not _stdout_console.no_color:
        from rich.syntax import Syntax

        highlighted = Syntax(
            text,
            "markdown",
            theme="ansi_dark",
            background_color="default",
            word_wrap=True,
        )
        _stdout_console.print(highlighted)
    else:
        print(text)


# -- Reviewer feedback -------------------------------------------------------


def review_feedback(review_round: int, text: str) -> None:
    title = Text(f"review round {review_round}", style="bold magenta")
    body = Text(text.rstrip("\n"), style="magenta")

    if not _framed(body, title=title, border_style="magenta"):
        header = Text()
        header.append(f"  [review round {review_round}] ", style="bold magenta")
        header.append("Reviewer requested changes:", style="magenta")
        _console.print(header)
        for line in text.splitlines():
            _console.print(Text(f"    {line}", style="magenta"))


def review_sending(review_round: int) -> None:
    _console.print(
        Text(
            f"  ▶ Review round {review_round}: sending answer to reviewer",
            style="bold cyan",
        )
    )


def review_accepted(review_round: int) -> None:
    _console.print(
        Text(
            f"  \u2713 Reviewer accepted the answer (round {review_round})",
            style="bold green",
        )
    )


# -- Diagnostics -------------------------------------------------------------


def info(msg: str) -> None:
    _console.print(Text(f"  {msg}", style="dim"))


model_info = info


def context_stats(label: str, tokens: int) -> None:
    _console.print(Text(f"  {label}: ~{tokens} tokens", style="dim"))


def think_summary(line: str) -> None:
    _console.print(Text(f"  {line}", style="dim"))


todo_summary = think_summary


def warning(msg: str) -> None:
    line = Text()
    line.append("  \u26a0 Warning: ", style="yellow")
    line.append(msg, style="yellow")
    _console.print(line)


def error(msg: str) -> None:
    line = Text()
    line.append("Error: ", style="bold red")
    line.append(msg, style="red")
    _console.print(line)


sandbox_hint = info


def quick_shell(cmd: str, returncode: int, output: str) -> None:
    title = Text(f"$ {cmd}", style="bold")
    subtitle = Text()
    if returncode == 0:
        subtitle.append("exit 0", style="green")
    else:
        subtitle.append(f"exit {returncode}", style="red")
    border_style = "dim" if returncode == 0 else "red"

    body = Text(output.rstrip("\n"))

    if not _framed(body, title=title, subtitle=subtitle, border_style=border_style):
        header = Text()
        header.append(f"  $ {cmd}", style="bold dim")
        _console.print(header)
        if output:
            _console.print(output)
        if returncode != 0:
            _console.print(Text(f"  exit {returncode}", style="red dim"))


_PHASE_COLORS: dict[str, str] = {
    "inventory": "cyan",
    "triage": "blue",
    "deep_review": "magenta",
    "verification": "yellow",
    "artifacts": "green",
}

_SEVERITY_STYLES: dict[str, str] = {
    "critical": "bold bright_red",
    "high": "bold red",
    "medium": "yellow",
    "low": "dim white",
}


def phase_color(phase_key: str) -> str:
    return _PHASE_COLORS.get(phase_key, "cyan")


def severity_style(sev: str) -> str:
    return _SEVERITY_STYLES.get((sev or "").lower(), "white")


def phase_banner(title: str, *, color: str = "cyan") -> None:
    """Print a gradient rule with a centered title in the given color."""
    _console.print()
    if _console.is_terminal:
        _console.print(_GradientRule(title))
    else:
        _console.print(Rule(title, style=color))


def bar_progress(*, transient: bool = False) -> Progress:
    """Configured Progress with spinner, bar, count, elapsed, and ETA."""
    return Progress(
        SpinnerColumn("dots", style="cyan", speed=1.2),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=_console,
        transient=transient,
        refresh_per_second=10,
        auto_refresh=False,
        expand=True,
    )


def get_console() -> Console:
    return _console


def repl_banner() -> None:
    _console.print(
        Text(
            "Type /exit or Ctrl-D to quit. Ctrl-C to interrupt; /continue to continue.",
            style="dim",
        )
    )


_LOGO = r"""
 ███████╗██╗    ██╗██╗██╗   ██╗ █████╗ ██╗
 ██╔════╝██║    ██║██║██║   ██║██╔══██╗██║
 ███████╗██║ █╗ ██║██║██║   ██║███████║██║
 ╚════██║██║███╗██║██║╚██╗ ██╔╝██╔══██║██║
 ███████║╚███╔███╔╝██║ ╚████╔╝ ██║  ██║███████╗
 ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝
""".strip("\n")

_GRADIENT_STOPS = [
    (32, 252, 214),  # mint neon
    (0, 177, 255),  # sky laser
    (77, 96, 255),  # cobalt
    (176, 79, 255),  # ultraviolet
    (255, 65, 184),  # plasma pink
    (255, 204, 92),  # sunlit amber
]


def _lerp_color(stops: list[tuple[int, int, int]], t: float) -> tuple[int, int, int]:
    """Interpolate between color stops at position t in [0, 1]."""
    t = max(0.0, min(1.0, t))
    n = len(stops) - 1
    idx = min(int(t * n), n - 1)
    local_t = (t * n) - idx
    r0, g0, b0 = stops[idx]
    r1, g1, b1 = stops[idx + 1]
    return (
        int(r0 + (r1 - r0) * local_t),
        int(g0 + (g1 - g0) * local_t),
        int(b0 + (b1 - b0) * local_t),
    )


def repl_splash(
    model: str = "",
    provider: str = "",
    workspace: str = "",
) -> None:
    """Print a colorful startup splash banner to stderr."""
    if not _console.is_terminal:
        return

    logo_lines = _LOGO.split("\n")
    max_len = max(len(ln) for ln in logo_lines)
    text = Text()
    row_count = len(logo_lines)
    for row_idx, row in enumerate(logo_lines):
        padded = row.ljust(max_len)
        for col_idx, ch in enumerate(padded):
            col_t = col_idx / max(max_len - 1, 1)
            row_t = row_idx / max(row_count - 1, 1)
            t = (col_t * 0.82) + (row_t * 0.18)
            r, g, b = _lerp_color(_GRADIENT_STOPS, t)
            text.append(ch, style=Style(color=f"rgb({r},{g},{b})", bold=True))
        text.append("\n")

    _console.print()
    _console.print(text, end="")
    try:
        from importlib import metadata

        _version = metadata.version("swival")
    except Exception:
        _version = ""
    banner_line = Text("  https://swival.dev", style="dim")
    if _version:
        banner_line.append(f"  v{_version}", style="dim")
    _console.print(banner_line)

    if model or provider or workspace:
        info_line = Text()
        if model:
            info_line.append(f"  model: {model}", style="dim")
        if provider:
            if model:
                info_line.append(" · ", style="dim")
            info_line.append(f"provider: {provider}", style="dim")
        if workspace:
            if model or provider:
                info_line.append(" · ", style="dim")
            info_line.append(f"workspace: {workspace}", style="dim")
        _console.print(info_line)

    grad_rule = Text()
    width = _console.width or 80
    for i in range(width):
        t = i / max(width - 1, 1)
        r, g, b = _lerp_color(_GRADIENT_STOPS, t)
        grad_rule.append("─", style=Style(color=f"rgb({r},{g},{b})"))
    _console.print(grad_rule)


# -- External servers (MCP / A2A) --------------------------------------------


def _server_start(kind: str, name: str, tool_count: int) -> None:
    line = Text()
    line.append(f"  {kind} {name}", style="cyan")
    line.append(f"  {tool_count} tool(s)", style="dim")
    _console.print(line)


def _server_error(kind: str, name: str, error: str) -> None:
    line = Text()
    line.append(f"  {kind} {name}", style="bold red")
    line.append(f"  {error}", style="red")
    _console.print(line)


def mcp_server_start(name: str, tool_count: int) -> None:
    _server_start("MCP", name, tool_count)


def mcp_server_error(name: str, error: str) -> None:
    _server_error("MCP", name, error)


def a2a_server_start(name: str, tool_count: int) -> None:
    _server_start("A2A", name, tool_count)


def a2a_server_error(name: str, error: str) -> None:
    _server_error("A2A", name, error)

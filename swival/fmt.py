"""ANSI-formatted stderr output using Rich."""

import contextlib
import difflib
import threading
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.spinner import Spinner
from rich.segment import Segment
from rich.style import Style
from rich.text import Text

_console = Console(stderr=True)
_stdout_console = Console(stderr=False)

_think_count = 0


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


def turn_header(n: int, max_n: int, token_est: int) -> None:
    reset_state()
    _console.print()
    title = f"Turn {n}/{max_n} (~{token_est} tokens)"
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
    of progress through distinct work stages.
    """
    if not _console.is_terminal:
        yield
        return

    # Extract the parenthetical suffix from the label, e.g. "(turn 2/5)"
    # If present, we cycle verbs with the suffix appended.
    # If not, use the label verbatim as the initial description.
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

    with progress:
        tid = progress.add_task(initial_desc, total=None)
        t = threading.Thread(target=_cycle_phases, args=(tid,), daemon=True)
        t.start()
        try:
            yield
        finally:
            stop.set()
            t.join(timeout=1)


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


def repl_banner() -> None:
    _console.print(Text("Interactive mode. Type /exit or Ctrl-D to quit.", style="dim"))


_LOGO = r"""
 ███ █   █ █ █   █ ███ █
 █   █   █ █ █   █ █ █ █
 ███ █ █ █ █  █ █  ███ █
   █ ██ ██ █  █ █  █ █ █
 ███ █   █ █   █   █ █ ███
""".strip("\n")

_GRADIENT_STOPS = [
    (0, 80, 220),  # cyan
    (120, 60, 220),  # blue-purple
    (200, 50, 200),  # magenta
    (220, 160, 40),  # yellow
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
    for row_idx, row in enumerate(logo_lines):
        padded = row.ljust(max_len)
        for col_idx, ch in enumerate(padded):
            t = col_idx / max(max_len - 1, 1)
            r, g, b = _lerp_color(_GRADIENT_STOPS, t)
            text.append(ch, style=Style(color=f"rgb({r},{g},{b})", bold=True))
        text.append("\n")

    _console.print()
    _console.print(text, end="")
    _console.print(Text("  https://swival.dev", style="dim"))

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

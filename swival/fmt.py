"""ANSI-formatted stderr output using Rich."""

from rich.console import Console
from rich.markup import escape
from rich.rule import Rule
from rich.text import Text

_console = Console(stderr=True)


def init(*, color: bool = False, no_color: bool = False) -> None:
    """Reconfigure the module-level console from CLI flags.

    Call once at startup, before any output.
    """
    global _console
    kwargs: dict = {"stderr": True}
    if color:
        kwargs["force_terminal"] = True
        kwargs["no_color"] = False
    if no_color:
        kwargs["no_color"] = True
    _console = Console(**kwargs)


# -- Turn structure ----------------------------------------------------------


def turn_header(n: int, max_n: int, token_est: int) -> None:
    title = f"Turn {n}/{max_n} (~{token_est} tokens)"
    _console.print(Rule(title, style="cyan"))


def llm_timing(elapsed: float, finish_reason: str) -> None:
    style = "green" if finish_reason == "stop" else "yellow"
    text = Text()
    text.append(f"  LLM responded in {elapsed:.1f}s", style=style)
    text.append(f"  finish_reason={escape(str(finish_reason))}", style=style)
    _console.print(text)


def llm_spinner(label: str = "Waiting for LLM"):
    """Return a Rich Status context manager that spins on stderr."""
    return _console.status(f"  {label}", spinner="dots")


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


def tool_error(name: str, msg: str) -> None:
    header = Text()
    header.append(f"  \u2717 {name}", style="bold red")
    header.append(f"  {msg}", style="red")
    _console.print(header)


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
    prefix = f"[think {number}/{total}"
    if is_revision and revises_thought is not None:
        prefix += f" rev:{revises_thought}"
    if branch_id is not None and branch_from_thought is not None:
        prefix += f" branch:{branch_id} from:{branch_from_thought}"
    prefix += "]"

    line = Text()
    line.append(f"  {prefix}", style="yellow")
    line.append(f" {text}", style="dim italic")
    _console.print(line)


# -- Todo updates ------------------------------------------------------------


def todo_update(action: str, detail: str) -> None:
    prefix_map = {"add": "+1", "done": "\u2713", "remove": "-1", "cleared": "cleared"}
    tag = prefix_map.get(action, action)
    line = Text()
    line.append(f"  [todo {tag}]", style="yellow")
    line.append(f" {detail}", style="dim italic")
    _console.print(line)


# -- Assistant text ----------------------------------------------------------


def assistant_text(text: str) -> None:
    line = Text()
    line.append("  [assistant] ", style="blue")
    line.append(text)
    _console.print(line)


# -- Reviewer feedback -------------------------------------------------------


def review_feedback(review_round: int, text: str) -> None:
    header = Text()
    header.append(f"  [review round {review_round}] ", style="bold magenta")
    header.append("Reviewer requested changes:", style="magenta")
    _console.print(header)
    for line in text.splitlines():
        _console.print(Text(f"    {line}", style="magenta"))


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


def repl_banner() -> None:
    _console.print(Text("Interactive mode. Type /exit or Ctrl-D to quit.", style="dim"))


# -- MCP servers -------------------------------------------------------------


def mcp_server_start(name: str, tool_count: int) -> None:
    line = Text()
    line.append(f"  MCP {name}", style="cyan")
    line.append(f"  {tool_count} tool(s)", style="dim")
    _console.print(line)


def mcp_server_error(name: str, error: str) -> None:
    line = Text()
    line.append(f"  MCP {name}", style="bold red")
    line.append(f"  {error}", style="red")
    _console.print(line)

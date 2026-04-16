"""Single registry of input commands (slash and bang)."""

from typing import NamedTuple


class CommandInfo(NamedTuple):
    """Metadata for an input command."""

    desc: str
    arg: str | None = None
    arg_type: str | None = None
    kind: str = "state_change"
    modes: tuple[str, ...] = ("repl", "oneshot")
    options: tuple[tuple[str, str], ...] | None = None


INPUT_COMMANDS: dict[str, CommandInfo] = {
    "!!": CommandInfo(
        desc="Run a shell command and print output (no LLM)",
        arg="<command>",
        kind="state_change",
        modes=("repl",),
    ),
    "/audit": CommandInfo(
        desc="Run a staged security audit over tracked committed code",
        arg="[path|glob]",
        kind="agent_turn",
        modes=("repl",),
        options=(
            ("--resume", "Resume a previous audit run from its last checkpoint"),
            ("--regen", "Regenerate reports and patches for a completed audit run"),
            ("--workers N", "Number of parallel verification workers (default: 4)"),
        ),
    ),
    "/add-dir": CommandInfo(
        desc="Grant read+write access to a directory",
        arg="<path>",
        arg_type="dir_path",
        kind="state_change",
    ),
    "/add-dir-ro": CommandInfo(
        desc="Grant read-only access to a directory",
        arg="<path>",
        arg_type="dir_path",
        kind="state_change",
    ),
    "/clear": CommandInfo(
        desc="Reset conversation to initial state",
        kind="state_change",
    ),
    "/compact": CommandInfo(
        desc="Compress conversation context",
        kind="state_change",
        options=(("--drop", "Also remove middle turns for more aggressive reduction"),),
    ),
    "/continue": CommandInfo(
        desc="Reset turn counter and continue the agent loop",
        kind="agent_turn",
        modes=("repl",),
    ),
    "/copy": CommandInfo(
        desc="Copy last output to clipboard",
        kind="flow_control",
        modes=("repl",),
    ),
    "/exit": CommandInfo(
        desc="Exit the REPL",
        kind="flow_control",
    ),
    "/extend": CommandInfo(
        desc="Double max turns, or set to N",
        arg="[N]",
        kind="state_change",
    ),
    "/help": CommandInfo(
        desc="Show this help message",
        kind="info",
    ),
    "/init": CommandInfo(
        desc="Scan project for build/test/lint workflow and conventions, write AGENTS.md",
        kind="agent_turn",
    ),
    "/learn": CommandInfo(
        desc="Review session for mistakes and persist to memory",
        kind="agent_turn",
    ),
    "/new": CommandInfo(
        desc="Reset conversation to initial state",
        kind="state_change",
    ),
    "/profile": CommandInfo(
        desc="Switch LLM profile (no arg = list, - = revert)",
        arg="[name]",
        kind="state_change",
    ),
    "/quit": CommandInfo(
        desc="Exit the REPL",
        kind="flow_control",
    ),
    "/remember": CommandInfo(
        desc="Add a durable project fact to AGENTS.md",
        arg="<text>",
        kind="state_change",
    ),
    "/restore": CommandInfo(
        desc="Summarize & collapse since checkpoint",
        kind="state_change",
    ),
    "/save": CommandInfo(
        desc="Set a context checkpoint",
        arg="[label]",
        kind="state_change",
    ),
    "/simplify": CommandInfo(
        desc="Simplify codebase (optionally scoped to focus area)",
        arg="[focus]",
        kind="agent_turn",
    ),
    "/status": CommandInfo(
        desc="Show session stats (model, context, turns, state)",
        kind="info",
    ),
    "/tools": CommandInfo(
        desc="List all available tools",
        kind="info",
    ),
    "/unsave": CommandInfo(
        desc="Cancel active checkpoint",
        kind="state_change",
    ),
}

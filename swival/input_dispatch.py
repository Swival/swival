"""Shared command parsing, execution, and script running for REPL and one-shot modes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from .snapshot import SnapshotState
    from .thinking import ThinkingState
    from .todo import TodoState
    from .tracker import FileAccessTracker


@dataclass
class InputContext:
    """Mutable session state shared by the command executor."""

    messages: list
    tools: list
    base_dir: str
    turn_state: dict
    thinking_state: "ThinkingState"
    todo_state: "TodoState"
    snapshot_state: "SnapshotState | None"
    file_tracker: "FileAccessTracker | None"
    no_history: bool
    continue_here: bool
    verbose: bool
    # Provider / loop kwargs passed through to run_agent_loop.
    loop_kwargs: dict
    # Profile state.
    current_profile: str | None = None
    profiles: dict = field(default_factory=dict)
    startup_profile: str | None = None
    raw_llm_baseline: dict = field(default_factory=dict)
    pre_profile_baseline: dict = field(default_factory=dict)
    # External managers.
    mcp_manager: object = None
    a2a_manager: object = None
    subagent_manager: object = None
    subagent_holder: list | None = None
    # Misc.
    start_dir: "Path | None" = None
    extra_write_roots: list = field(default_factory=list)
    skill_read_roots: list = field(default_factory=list)
    skills_catalog: dict = field(default_factory=dict)
    is_subagent: bool = False


@dataclass
class ParsedInput:
    """Result of parsing a single input line."""

    raw: str
    cmd: str | None = None
    cmd_arg: str = ""
    is_command: bool = False
    is_custom_command: bool = False


@dataclass
class StepResult:
    """Outcome of executing a single input line."""

    kind: str  # "info", "agent_turn", "state_change", "flow_control"
    text: str | None = None
    stop: bool = False
    exhausted: bool = False
    is_error: bool = False


def parse_input_line(line: str) -> ParsedInput:
    """Parse a single input line into a structured form."""
    line = line.strip()
    if not line:
        return ParsedInput(raw=line)

    # Custom bang commands: !foo (but not "! foo" with a leading space).
    if line.startswith("!") and len(line) > 1 and not line[1:].startswith(" "):
        return ParsedInput(raw=line, is_custom_command=True)

    # Slash commands.
    if line.startswith("/"):
        parts = line.split(None, 1)
        cmd = parts[0].lower()
        cmd_arg = parts[1] if len(parts) > 1 else ""
        return ParsedInput(raw=line, cmd=cmd, cmd_arg=cmd_arg, is_command=True)

    # Plain text.
    return ParsedInput(raw=line)


def is_command_script(text: str) -> bool:
    """Decide whether one-shot input should be treated as a command script.

    A command script is detected when the first non-blank line is a known
    slash command or a custom bang command.

    Intentional sharp edge: a natural-language prompt whose first line
    happens to be a known command name will be treated as a command script.
    This is acceptable because backward compatibility is not a goal.

    The ``! `` (bang-space) exclusion is deliberate so that ``! foo`` remains
    ordinary text instead of being mistaken for a custom command.
    """
    for raw_line in text.splitlines():
        parsed = parse_input_line(raw_line)
        if not parsed.raw:
            continue
        if parsed.is_custom_command:
            return True
        if parsed.is_command:
            return True
        return False
    return False

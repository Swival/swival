"""Shared test fixtures."""

import shutil
import types
from contextlib import contextmanager
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from swival.snapshot import SnapshotState
from swival.thinking import ThinkingState
from swival.todo import TodoState
from swival.tools import _execute_command_call

_FAKE_TTY_ENV = {"TERM": "xterm-256color"}


def styled_console(buf: StringIO) -> Console:
    """Build a Rich console that always renders styled TTY output.

    Forces terminal mode, truecolor, and a non-dumb TERM so tests can assert
    Swival's framing/coloring without depending on the host shell.
    """
    return Console(
        file=buf,
        force_terminal=True,
        color_system="truecolor",
        no_color=False,
        width=80,
        _environ=_FAKE_TTY_ENV,
    )


def capture_styled(func, *args, **kwargs) -> str:
    """Call a fmt function with a forced-TTY console and return ANSI output."""
    from swival import fmt

    buf = StringIO()
    old = fmt._console
    fmt._console = styled_console(buf)
    fmt.reset_state()
    try:
        func(*args, **kwargs)
    finally:
        fmt.reset_state()
        fmt._console = old
    return buf.getvalue()


@contextmanager
def plain_console(width=200):
    """Swap fmt._console for a plain no-color console; yields the buffer."""
    from swival import fmt

    buf = StringIO()
    old = fmt._console
    fmt._console = Console(file=buf, no_color=True, width=width)
    try:
        yield buf
    finally:
        fmt._console = old


def fake_tool_result(name="read_file", tc_id="tc1", content="ok"):
    """Build the (tool message, stats) pair handle_tool_call returns."""
    return (
        {"role": "tool", "tool_call_id": tc_id, "content": content},
        {"name": name, "arguments": {}, "elapsed": 0.0, "succeeded": True},
    )


def run_command(
    command,
    base_dir,
    resolved_commands,
    timeout=30,
    unrestricted=False,
    scratch_dir=None,
):
    """Convenience wrapper around _execute_command_call with prefer_shell=False."""
    return _execute_command_call(
        command,
        prefer_shell=False,
        base_dir=base_dir,
        resolved_commands=resolved_commands,
        timeout=timeout,
        unrestricted=unrestricted,
        scratch_dir=scratch_dir,
    )


def which_or_skip(name: str) -> str:
    """Resolve a command name to its absolute path, skip test if not found."""
    path = shutil.which(name)
    if path is None:
        pytest.skip(f"{name!r} not found on PATH")
    return str(Path(path).resolve())


def init_git(tmp_path: Path) -> None:
    """Initialize a git repository with a test identity."""
    import subprocess

    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )


def commit_file(tmp_path: Path, rel_path: str, content: str, msg: str = "c") -> str:
    """Write and commit a file; returns the resulting HEAD commit."""
    import subprocess

    fp = tmp_path / rel_path
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(content)
    subprocess.run(
        ["git", "add", rel_path], cwd=tmp_path, capture_output=True, check=True
    )
    subprocess.run(
        ["git", "commit", "-m", msg], cwd=tmp_path, capture_output=True, check=True
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture(autouse=True)
def _fresh_model_catalog_cache():
    """The model-catalog cache is module-global; reset it around every test."""
    from swival import model_catalog

    model_catalog.clear_cache()
    yield
    model_catalog.clear_cache()


@pytest.fixture(autouse=True)
def _isolate_global_skills(monkeypatch):
    """Prevent all tests from picking up real ~/.agents/skills/ or ~/.config/swival/skills/.

    Global skill discovery scans Path.home() / ".agents" / "skills" and
    config.global_config_dir() / "skills".  Without isolation, tests that
    create a Session (or call discover_skills) on a machine with real global
    skills become environment-dependent.
    """
    monkeypatch.setattr("swival.skills._global_skill_dirs", lambda: [])


@pytest.fixture(autouse=True)
def _isolate_global_agents_md(monkeypatch):
    """Prevent all tests from picking up real ~/.agents/AGENTS.md."""
    monkeypatch.setattr(
        "swival.agent._global_agents_md_path",
        lambda: Path("/nonexistent/.agents/AGENTS.md"),
    )


def make_message(content=None, tool_calls=None):
    """Assistant message shaped like a provider response object."""
    return types.SimpleNamespace(
        content=content, tool_calls=tool_calls, role="assistant"
    )


def make_tool_call(name, args=None, tc_id="tc1"):
    return types.SimpleNamespace(
        id=tc_id,
        function=types.SimpleNamespace(
            name=name, arguments="{}" if args is None else args
        ),
    )


class ScriptedLLM:
    """call_llm stand-in that plays scripted responses, then a final answer.

    A response may be ``(message, finish_reason)``; a bare message means
    ``"stop"``.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.tail_answer = "all done"

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.responses:
            response = self.responses.pop(0)
            return response if isinstance(response, tuple) else (response, "stop")
        return make_message(content=self.tail_answer), "stop"


def build_loop_kwargs(tmp_path, goal_state=None, *, max_turns=4):
    """Minimal run_agent_loop() kwargs for a mocked provider."""
    return dict(
        api_base="http://x",
        model_id="m",
        max_turns=max_turns,
        max_output_tokens=None,
        temperature=None,
        top_p=None,
        seed=None,
        context_length=128000,
        base_dir=str(tmp_path),
        thinking_state=ThinkingState(),
        todo_state=TodoState(),
        snapshot_state=SnapshotState(),
        goal_state=goal_state,
        resolved_commands={},
        skills_catalog={},
        skill_read_roots=[],
        extra_write_roots=[],
        files_mode="all",
        commands_unrestricted=False,
        shell_allowed=False,
        verbose=False,
        llm_kwargs={"provider": "generic"},
        file_tracker=None,
        continue_here=False,
    )

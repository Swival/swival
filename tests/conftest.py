"""Shared test fixtures."""

import shutil
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

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

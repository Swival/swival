"""Tests for --yolo (unrestricted) mode."""

import os
import shutil
import sys
import types
from pathlib import Path

import pytest

from swival.tools import (
    safe_resolve,
    _is_within_base,
    _read_file,
    _write_file,
    _edit_file,
    _list_files,
    _grep,
    _run_command,
    dispatch,
)
from swival.agent import build_parser


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_dir(tmp_path):
    """A temporary base directory."""
    return tmp_path / "base"


@pytest.fixture
def outside_dir(tmp_path):
    """A directory outside base_dir."""
    d = tmp_path / "outside"
    d.mkdir()
    return d


@pytest.fixture
def setup_dirs(base_dir, outside_dir):
    """Create both dirs and a file in each."""
    base_dir.mkdir()
    (base_dir / "in.txt").write_text("inside base", encoding="utf-8")
    (outside_dir / "out.txt").write_text("outside base", encoding="utf-8")
    return base_dir, outside_dir


# ---------------------------------------------------------------------------
# safe_resolve / _is_within_base
# ---------------------------------------------------------------------------


class TestSafeResolveUnrestricted:
    def test_safe_resolve_unrestricted(self, setup_dirs):
        base, outside = setup_dirs
        outside_file = outside / "out.txt"
        # Without unrestricted, this raises
        with pytest.raises(ValueError):
            safe_resolve(str(outside_file), str(base))
        # With unrestricted, it returns the resolved path
        result = safe_resolve(str(outside_file), str(base), unrestricted=True)
        assert result == outside_file.resolve()

    def test_safe_resolve_unrestricted_blocks_root(self, setup_dirs):
        base, _ = setup_dirs
        with pytest.raises(ValueError, match="filesystem root"):
            safe_resolve("/", str(base), unrestricted=True)

    def test_safe_resolve_unrestricted_allows_subdirs_of_root(self, setup_dirs):
        base, _ = setup_dirs
        result = safe_resolve("/opt", str(base), unrestricted=True)
        assert result == Path("/opt").resolve()

    def test_is_within_base_unrestricted(self, tmp_path):
        some_path = Path("/tmp/nonexistent_abc_xyz")
        base = tmp_path / "base"
        base.mkdir()
        assert not _is_within_base(some_path, base)
        assert _is_within_base(some_path, base, unrestricted=True)


# ---------------------------------------------------------------------------
# File operations outside base_dir
# ---------------------------------------------------------------------------


class TestRootBlocked:
    """Even in unrestricted mode, operating on / is blocked."""

    def test_read_file_root_blocked(self, setup_dirs):
        base, _ = setup_dirs
        result = _read_file("/", str(base), unrestricted=True)
        assert result.startswith("error:")
        assert "filesystem root" in result

    def test_list_files_root_blocked(self, setup_dirs):
        base, _ = setup_dirs
        result = _list_files("*", "/", str(base), unrestricted=True)
        assert result.startswith("error:")
        assert "filesystem root" in result

    def test_grep_root_blocked(self, setup_dirs):
        base, _ = setup_dirs
        result = _grep(".", "/", str(base), unrestricted=True)
        assert result.startswith("error:")
        assert "filesystem root" in result


class TestFileOpsOutsideBase:
    def test_read_outside_base_dir(self, setup_dirs):
        base, outside = setup_dirs
        result = _read_file(str(outside / "out.txt"), str(base), unrestricted=True)
        assert "outside base" in result
        assert not result.startswith("error:")

    def test_read_outside_base_dir_blocked(self, setup_dirs):
        base, outside = setup_dirs
        result = _read_file(str(outside / "out.txt"), str(base))
        assert result.startswith("error:")

    def test_write_outside_base_dir(self, setup_dirs):
        base, outside = setup_dirs
        target = outside / "new.txt"
        result = _write_file(str(target), "hello yolo", str(base), unrestricted=True)
        assert "Wrote" in result
        assert target.read_text(encoding="utf-8") == "hello yolo"

    def test_write_outside_base_dir_blocked(self, setup_dirs):
        base, outside = setup_dirs
        target = outside / "new.txt"
        result = _write_file(str(target), "hello yolo", str(base))
        assert result.startswith("error:")

    def test_edit_outside_base_dir(self, setup_dirs):
        base, outside = setup_dirs
        result = _edit_file(
            str(outside / "out.txt"),
            "outside base",
            "edited content",
            str(base),
            unrestricted=True,
        )
        assert result == f"Edited {outside / 'out.txt'}"
        assert (outside / "out.txt").read_text(encoding="utf-8") == "edited content"

    def test_edit_outside_base_dir_blocked(self, setup_dirs):
        base, outside = setup_dirs
        result = _edit_file(
            str(outside / "out.txt"),
            "outside base",
            "edited",
            str(base),
        )
        assert result.startswith("error:")


# ---------------------------------------------------------------------------
# list_files outside base_dir
# ---------------------------------------------------------------------------


class TestListFilesOutsideBase:
    def test_list_files_outside_base_dir(self, setup_dirs):
        base, outside = setup_dirs
        result = _list_files("*.txt", str(outside), str(base), unrestricted=True)
        assert "out.txt" in result
        assert not result.startswith("error:")
        # Should use absolute path since it's outside base
        assert str(outside) in result

    def test_list_files_outside_base_dir_blocked(self, setup_dirs):
        base, outside = setup_dirs
        result = _list_files("*.txt", str(outside), str(base))
        assert result.startswith("error:")


# ---------------------------------------------------------------------------
# grep outside base_dir
# ---------------------------------------------------------------------------


class TestGrepOutsideBase:
    def test_grep_outside_base_dir(self, setup_dirs):
        base, outside = setup_dirs
        result = _grep("outside", str(outside), str(base), unrestricted=True)
        assert "outside base" in result
        assert not result.startswith("error:")
        assert "No matches" not in result
        # Should use absolute path since it's outside base
        assert str(outside) in result

    def test_grep_outside_base_dir_blocked(self, setup_dirs):
        base, outside = setup_dirs
        result = _grep("outside", str(outside), str(base))
        assert result.startswith("error:")


# ---------------------------------------------------------------------------
# run_command unrestricted
# ---------------------------------------------------------------------------


def _which(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        pytest.skip(f"{name!r} not found on PATH")
    return str(Path(path).resolve())


class TestRunCommandUnrestricted:
    def test_run_any_command(self, tmp_path):
        """Unrestricted mode runs commands not in resolved_commands."""
        result = _run_command(
            ["echo", "yolo"],
            str(tmp_path),
            resolved_commands={},
            unrestricted=True,
        )
        assert "yolo" in result
        assert not result.startswith("error:")

    def test_run_command_with_path(self, tmp_path):
        """Unrestricted mode accepts absolute paths in command[0]."""
        echo_path = _which("echo")
        result = _run_command(
            [echo_path, "path-ok"],
            str(tmp_path),
            resolved_commands={},
            unrestricted=True,
        )
        assert "path-ok" in result
        assert not result.startswith("error:")

    def test_run_command_relative_path_resolves_against_base_dir(self, tmp_path):
        """./tool resolves relative to base_dir, not process CWD."""
        # Create an executable script inside base_dir
        script = tmp_path / "mytool"
        script.write_text("#!/bin/sh\necho relative-ok\n", encoding="utf-8")
        script.chmod(0o755)

        # Run from a different CWD to prove resolution is against base_dir
        original_cwd = os.getcwd()
        try:
            os.chdir("/")
            result = _run_command(
                ["./mytool"],
                str(tmp_path),
                resolved_commands={},
                unrestricted=True,
            )
        finally:
            os.chdir(original_cwd)

        assert "relative-ok" in result
        assert not result.startswith("error:")

    def test_run_command_not_found_unrestricted(self, tmp_path):
        """Unrestricted mode returns clear error for nonexistent commands."""
        result = _run_command(
            ["no_such_cmd_xyz_12345"],
            str(tmp_path),
            resolved_commands={},
            unrestricted=True,
        )
        assert result == "error: command not found on PATH: 'no_such_cmd_xyz_12345'"

    def test_yolo_overrides_allowed_commands(self, tmp_path):
        """When unrestricted, any command runs even if resolved_commands is limited."""
        ls_path = _which("ls")
        # Only "ls" is in resolved_commands, but "echo" should still work
        result = _run_command(
            ["echo", "override"],
            str(tmp_path),
            resolved_commands={"ls": ls_path},
            unrestricted=True,
        )
        assert "override" in result
        assert not result.startswith("error:")


# ---------------------------------------------------------------------------
# dispatch with yolo=True
# ---------------------------------------------------------------------------


class TestDispatchYolo:
    def test_dispatch_read_file_yolo(self, setup_dirs):
        base, outside = setup_dirs
        result = dispatch(
            "read_file",
            {"file_path": str(outside / "out.txt")},
            str(base),
            yolo=True,
        )
        assert "outside base" in result

    def test_dispatch_run_command_yolo(self, tmp_path):
        result = dispatch(
            "run_command",
            {"command": ["echo", "dispatch-yolo"]},
            str(tmp_path),
            yolo=True,
            resolved_commands={},
        )
        assert "dispatch-yolo" in result


# ---------------------------------------------------------------------------
# Agent-level: parser and tool list
# ---------------------------------------------------------------------------


def _make_message(content=None, tool_calls=None):
    msg = types.SimpleNamespace()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.role = "assistant"
    msg.get = lambda key, default=None: getattr(msg, key, default)
    return msg


class TestAgentYolo:
    def test_yolo_flag_parsed(self):
        parser = build_parser()
        args = parser.parse_args(["test question", "--yolo"])
        assert args.yolo is True

    def test_yolo_flag_default_false(self):
        parser = build_parser()
        args = parser.parse_args(["test question"])
        assert args.yolo is False

    def test_yolo_tool_list_includes_run_command(self, tmp_path, monkeypatch):
        """With --yolo and no --allowed-commands, the tools list passed to
        call_llm includes run_command with unrestricted description."""
        from swival import agent

        captured = {}

        def fake_call_llm(*args, **kwargs):
            captured["tools"] = kwargs.get("tools") or args[7]
            captured["messages"] = args[2]
            return _make_message(content="Done."), "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--base-dir",
                str(tmp_path),
                "--yolo",
                "--no-instructions",
            ],
        )

        agent.main()

        tool_names = [t["function"]["name"] for t in captured["tools"]]
        assert "run_command" in tool_names

        rc_tool = next(
            t for t in captured["tools"] if t["function"]["name"] == "run_command"
        )
        assert "any command" in rc_tool["function"]["description"].lower()
        assert "Allowed" not in rc_tool["function"]["description"]

    def test_yolo_system_prompt_text(self, tmp_path, monkeypatch):
        """Yolo mode appends unrestricted command blurb to system prompt."""
        from swival import agent

        captured = {}

        def fake_call_llm(*args, **kwargs):
            captured["messages"] = args[2]
            return _make_message(content="Done."), "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--base-dir",
                str(tmp_path),
                "--yolo",
                "--no-instructions",
            ],
        )

        agent.main()

        system_msg = captured["messages"][0]
        assert system_msg["role"] == "system"
        assert "Run any command" in system_msg["content"]
        assert "Allowed commands" not in system_msg["content"]
        assert "whitelisted" not in system_msg["content"]

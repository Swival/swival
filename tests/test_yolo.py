"""Tests for --yolo (unrestricted) mode."""

import os
import shutil
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from swival.tools import (
    safe_resolve,
    _is_within_base,
    _SHELL_CHARS,
    _read_file,
    _write_file,
    _edit_file,
    _list_files,
    _grep,
    _run_command,
    _run_shell_command,
    _kill_process_tree,
    _split_absolute_glob,
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
# Absolute patterns in unrestricted mode
# ---------------------------------------------------------------------------


class TestSplitAbsoluteGlob:
    def test_deep_path_with_double_star(self):
        root, pattern = _split_absolute_glob("/opt/zig/lib/std/**/*.zig")
        assert root == "/opt/zig/lib/std"
        assert pattern == "**/*.zig"

    def test_single_star(self):
        root, pattern = _split_absolute_glob("/foo/bar/*.txt")
        assert root == "/foo/bar"
        assert pattern == "*.txt"

    def test_glob_in_middle(self):
        root, pattern = _split_absolute_glob("/a/b/*/c.txt")
        assert root == "/a/b"
        assert pattern == "*/c.txt"

    def test_root_glob(self):
        root, pattern = _split_absolute_glob("/*.txt")
        assert root == "/"
        assert pattern == "*.txt"

    def test_windows_drive_letter(self):
        root, pattern = _split_absolute_glob(r"C:\Users\alice\*.py")
        assert root == r"C:\Users\alice"
        assert pattern == "*.py"

    def test_windows_deep_glob(self):
        root, pattern = _split_absolute_glob(r"D:\projects\src\**\*.ts")
        assert root == r"D:\projects\src"
        assert pattern == "**/*.ts"

    def test_windows_unc_path(self):
        root, pattern = _split_absolute_glob(r"\\server\share\docs\*.pdf")
        assert root == r"\\server\share\docs"
        assert pattern == "*.pdf"


class TestAbsolutePatternUnrestricted:
    """In yolo mode, absolute glob patterns should work for list_files and grep."""

    def test_list_files_absolute_pattern(self, setup_dirs):
        base, outside = setup_dirs
        result = _list_files(f"{outside}/*.txt", ".", str(base), unrestricted=True)
        assert "out.txt" in result
        assert not result.startswith("error:")

    def test_list_files_absolute_pattern_blocked_without_yolo(self, setup_dirs):
        base, outside = setup_dirs
        result = _list_files(f"{outside}/*.txt", ".", str(base), unrestricted=False)
        assert result.startswith("error:")
        assert "outside base directory" in result

    def test_list_files_absolute_pattern_deep_glob(self, setup_dirs):
        base, outside = setup_dirs
        sub = outside / "deep"
        sub.mkdir()
        (sub / "nested.py").write_text("x = 1")
        result = _list_files(f"{outside}/**/*.py", ".", str(base), unrestricted=True)
        assert "nested.py" in result

    def test_grep_absolute_include_unrestricted(self, setup_dirs):
        base, outside = setup_dirs
        # grep's include parameter should accept absolute-looking patterns in yolo
        result = _grep(
            "outside",
            str(outside),
            str(base),
            include="/some/abs/*.txt",  # would normally be rejected
            unrestricted=True,
        )
        # The include won't actually match filenames (it's a fnmatch against
        # basenames), but the point is it doesn't error out.
        assert not result.startswith("error:")

    def test_grep_absolute_include_blocked_without_yolo(self, setup_dirs):
        base, outside = setup_dirs
        result = _grep(
            "outside",
            str(base),
            str(base),
            include="/abs/*.txt",
        )
        assert result.startswith("error:")
        assert "must be relative" in result


# ---------------------------------------------------------------------------
# Absolute patterns with --add-dir (non-yolo)
# ---------------------------------------------------------------------------


class TestAbsolutePatternAddDir:
    """Absolute glob patterns should work when the path is within extra roots."""

    def test_list_files_absolute_pattern_via_add_dir(self, setup_dirs):
        base, outside = setup_dirs
        result = _list_files(
            f"{outside}/*.txt",
            ".",
            str(base),
            extra_write_roots=[outside],
        )
        assert "out.txt" in result
        assert not result.startswith("error:")

    def test_list_files_absolute_pattern_unauthorized(self, setup_dirs):
        """Absolute pattern pointing outside all roots should be rejected."""
        base, outside = setup_dirs
        result = _list_files(
            f"{outside}/*.txt",
            ".",
            str(base),
            extra_write_roots=[],  # no extra roots
        )
        assert result.startswith("error:")

    def test_list_files_absolute_pattern_via_read_roots(self, setup_dirs):
        base, outside = setup_dirs
        result = _list_files(
            f"{outside}/*.txt",
            ".",
            str(base),
            extra_read_roots=[outside],
        )
        assert "out.txt" in result
        assert not result.startswith("error:")

    def test_grep_path_via_add_dir(self, setup_dirs):
        base, outside = setup_dirs
        result = _grep(
            "outside",
            str(outside),
            str(base),
            extra_write_roots=[outside],
        )
        assert "outside base" in result
        assert not result.startswith("error:")

    def test_grep_path_via_read_roots(self, setup_dirs):
        base, outside = setup_dirs
        result = _grep(
            "outside",
            str(outside),
            str(base),
            extra_read_roots=[outside],
        )
        assert "outside base" in result
        assert not result.startswith("error:")


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
        from swival.config import _UNSET

        parser = build_parser()
        args = parser.parse_args(["test question"])
        assert args.yolo is _UNSET

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


# ---------------------------------------------------------------------------
# Shell string execution (yolo mode)
# ---------------------------------------------------------------------------

_unix_only = pytest.mark.skipif(sys.platform == "win32", reason="requires /bin/sh")


class TestShellStringExecution:
    @_unix_only
    def test_shell_string_command(self, tmp_path):
        result = _run_command(
            "echo hello && echo world",
            str(tmp_path),
            resolved_commands={},
            unrestricted=True,
        )
        assert "hello" in result
        assert "world" in result
        assert not result.startswith("error:")

    @_unix_only
    def test_shell_pipe(self, tmp_path):
        result = _run_command(
            "echo abc | tr a-z A-Z",
            str(tmp_path),
            resolved_commands={},
            unrestricted=True,
        )
        assert "ABC" in result

    @_unix_only
    def test_shell_redirect(self, tmp_path):
        out = tmp_path / "out"
        result = _run_command(
            f"echo test > {out} && cat {out}",
            str(tmp_path),
            resolved_commands={},
            unrestricted=True,
        )
        assert "test" in result

    @_unix_only
    def test_shell_string_timeout(self, tmp_path):
        result = _run_command(
            "sleep 60",
            str(tmp_path),
            resolved_commands={},
            unrestricted=True,
            timeout=1,
        )
        assert "timed out" in result

    @_unix_only
    def test_shell_string_nonzero_exit(self, tmp_path):
        result = _run_command(
            "exit 42",
            str(tmp_path),
            resolved_commands={},
            unrestricted=True,
        )
        assert "Exit code: 42" in result


class TestShellStringCompat:
    def test_json_array_string_repaired_in_yolo(self, tmp_path):
        """Stringified JSON arrays still take the array path, not sh -c."""
        result = _run_command(
            '["echo", "hello"]',
            str(tmp_path),
            resolved_commands={},
            unrestricted=True,
        )
        assert "hello" in result
        assert "(auto-corrected:" in result

    def test_shell_string_with_metachar_sandboxed_rejected(self, tmp_path):
        """In sandboxed mode, strings with shell chars still error."""
        result = _run_command(
            "echo hello | cat",
            str(tmp_path),
            resolved_commands={},
            unrestricted=False,
        )
        assert result.startswith('error: "command" must be a JSON array')

    @_unix_only
    def test_array_still_works_in_yolo(self, tmp_path):
        """Array form still works in yolo mode."""
        result = _run_command(
            ["echo", "hello"],
            str(tmp_path),
            resolved_commands={},
            unrestricted=True,
        )
        assert "hello" in result
        assert not result.startswith("error:")


class TestYoloSchema:
    def test_yolo_schema_has_oneof(self, tmp_path, monkeypatch):
        """With yolo=True, the tool schema uses oneOf for command."""
        from swival import agent

        captured = {}

        def fake_call_llm(*args, **kwargs):
            captured["tools"] = kwargs.get("tools") or args[7]
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

        rc_tool = next(
            t for t in captured["tools"] if t["function"]["name"] == "run_command"
        )
        cmd_prop = rc_tool["function"]["parameters"]["properties"]["command"]
        assert "oneOf" in cmd_prop
        types = [s.get("type") for s in cmd_prop["oneOf"]]
        assert "string" in types
        assert "array" in types

    def test_yolo_system_prompt_mentions_shell(self, tmp_path, monkeypatch):
        """Yolo system prompt mentions shell strings."""
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
        assert (
            "shell string" in system_msg["content"].lower()
            or "pipes" in system_msg["content"].lower()
        )


# ---------------------------------------------------------------------------
# Windows shell-string path (mock-based, runs on any platform)
# ---------------------------------------------------------------------------


class TestShellStringWindows:
    def test_shell_cmd_uses_cmd_exe_on_windows(self, tmp_path, monkeypatch):
        """On win32, _run_shell_command passes ['cmd.exe', '/c', command]."""
        monkeypatch.setattr(sys, "platform", "win32")
        captured_args = {}

        def fake_popen(cmd, **kwargs):
            captured_args["cmd"] = cmd
            proc = MagicMock()
            proc.stdout.read.return_value = b""
            proc.wait.return_value = 0
            proc.returncode = 0
            proc.pid = 12345
            return proc

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        _run_shell_command("echo hello", str(tmp_path), timeout=30)

        assert captured_args["cmd"] == ["cmd.exe", "/c", "echo hello"]

    def test_shell_cmd_uses_sh_on_unix(self, tmp_path, monkeypatch):
        """On non-win32, _run_shell_command passes ['/bin/sh', '-c', command]."""
        monkeypatch.setattr(sys, "platform", "linux")
        captured_args = {}

        def fake_popen(cmd, **kwargs):
            captured_args["cmd"] = cmd
            captured_args["kwargs"] = kwargs
            proc = MagicMock()
            proc.stdout.read.return_value = b""
            proc.wait.return_value = 0
            proc.returncode = 0
            proc.pid = 12345
            return proc

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        _run_shell_command("echo hello", str(tmp_path), timeout=30)

        assert captured_args["cmd"] == ["/bin/sh", "-c", "echo hello"]
        assert captured_args["kwargs"].get("start_new_session") is True

    def test_no_start_new_session_on_windows(self, tmp_path, monkeypatch):
        """On win32, start_new_session should NOT be set."""
        monkeypatch.setattr(sys, "platform", "win32")
        captured_kwargs = {}

        def fake_popen(cmd, **kwargs):
            captured_kwargs.update(kwargs)
            proc = MagicMock()
            proc.stdout.read.return_value = b""
            proc.wait.return_value = 0
            proc.returncode = 0
            proc.pid = 12345
            return proc

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        _run_shell_command("echo hello", str(tmp_path), timeout=30)

        assert "start_new_session" not in captured_kwargs


# ---------------------------------------------------------------------------
# Auto-split string commands in sandboxed mode
# ---------------------------------------------------------------------------


def _is_safe(s: str) -> bool:
    return not (_SHELL_CHARS & set(s))


class TestIsSafeToSplit:
    def test_plain_command(self):
        assert _is_safe("ls -la src/")

    def test_pipe_rejected(self):
        assert not _is_safe("echo hello | cat")

    def test_redirect_rejected(self):
        assert not _is_safe("echo hello > out.txt")

    def test_semicolon_rejected(self):
        assert not _is_safe("echo a; echo b")

    def test_dollar_rejected(self):
        assert not _is_safe("echo $HOME")

    def test_backtick_rejected(self):
        assert not _is_safe("echo `whoami`")

    def test_backslash_rejected(self):
        assert not _is_safe("echo C:\\Users")

    def test_single_quote_rejected(self):
        assert not _is_safe("echo 'hello world'")

    def test_double_quote_rejected(self):
        assert not _is_safe('echo "hello world"')

    def test_glob_star_rejected(self):
        assert not _is_safe("ls *.py")

    def test_newline_rejected(self):
        assert not _is_safe("echo hello\necho world")

    def test_carriage_return_rejected(self):
        assert not _is_safe("echo hello\recho world")

    def test_crlf_rejected(self):
        assert not _is_safe("echo hello\r\necho world")

    def test_empty_string(self):
        assert _is_safe("")

    def test_tabs_allowed(self):
        assert _is_safe("ls\t-la")


class TestAutoSplitStringCommand:
    @_unix_only
    def test_simple_string_auto_split(self, tmp_path):
        """A plain string without shell chars is auto-split and executed."""
        echo_path = _which("echo")
        result = _run_command(
            "echo hello",
            str(tmp_path),
            resolved_commands={"echo": echo_path},
            unrestricted=False,
        )
        assert "hello" in result
        assert "(auto-corrected:" in result

    def test_shell_chars_rejected(self, tmp_path):
        """Strings with shell metacharacters are rejected in sandboxed mode."""
        result = _run_command(
            "echo hello | cat",
            str(tmp_path),
            resolved_commands={},
            unrestricted=False,
        )
        assert result.startswith('error: "command" must be a JSON array')
        assert "(auto-corrected:" not in result

    def test_whitespace_only_empty(self, tmp_path):
        """Whitespace-only string splits to empty list."""
        result = _run_command(
            "   ",
            str(tmp_path),
            resolved_commands={},
            unrestricted=False,
        )
        assert "error: command list is empty" in result

    def test_json_array_string_still_works(self, tmp_path):
        """JSON-encoded array in a string still takes the JSON parse path."""
        echo_path = _which("echo")
        result = _run_command(
            '["echo", "hello"]',
            str(tmp_path),
            resolved_commands={"echo": echo_path},
            unrestricted=False,
        )
        assert "hello" in result
        assert "(auto-corrected:" in result

    @_unix_only
    def test_tab_separated_string(self, tmp_path):
        """Tab-separated tokens are split correctly."""
        echo_path = _which("echo")
        result = _run_command(
            "echo\thello",
            str(tmp_path),
            resolved_commands={"echo": echo_path},
            unrestricted=False,
        )
        assert "hello" in result
        assert "(auto-corrected:" in result

    def test_backslash_rejected(self, tmp_path):
        """Backslash in command string triggers the error path."""
        result = _run_command(
            "echo C:\\Users",
            str(tmp_path),
            resolved_commands={},
            unrestricted=False,
        )
        assert result.startswith('error: "command" must be a JSON array')

    def test_newline_rejected(self, tmp_path):
        """Newlines in command string are rejected as shell metacharacters."""
        result = _run_command(
            "echo hello\necho world",
            str(tmp_path),
            resolved_commands={},
            unrestricted=False,
        )
        assert result.startswith('error: "command" must be a JSON array')

    def test_carriage_return_rejected(self, tmp_path):
        """CR in command string is rejected as a shell metacharacter."""
        result = _run_command(
            "echo hello\recho world",
            str(tmp_path),
            resolved_commands={},
            unrestricted=False,
        )
        assert result.startswith('error: "command" must be a JSON array')

    @_unix_only
    def test_yolo_string_still_uses_shell(self, tmp_path):
        """In yolo mode, plain strings still go through shell execution."""
        result = _run_command(
            "echo hello && echo world",
            str(tmp_path),
            resolved_commands={},
            unrestricted=True,
        )
        assert "hello" in result
        assert "world" in result
        assert "(auto-corrected:" not in result


class TestKillProcessTreeWindows:
    def test_taskkill_called_on_windows(self, monkeypatch):
        """On win32, _kill_process_tree uses taskkill /T /F."""
        monkeypatch.setattr(sys, "platform", "win32")
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd

        monkeypatch.setattr(subprocess, "run", fake_run)

        proc = MagicMock()
        proc.pid = 42
        proc.kill.return_value = None
        proc.wait.return_value = 0

        _kill_process_tree(proc)

        assert captured["cmd"] == ["taskkill", "/T", "/F", "/PID", "42"]
        proc.kill.assert_called_once()

    def test_no_taskkill_on_unix(self, monkeypatch):
        """On Unix, _kill_process_tree uses killpg, not taskkill."""
        monkeypatch.setattr(sys, "platform", "linux")
        taskkill_called = []
        killpg_called = []

        monkeypatch.setattr(
            subprocess, "run", lambda cmd, **kw: taskkill_called.append(cmd)
        )
        monkeypatch.setattr(
            os, "killpg", lambda pid, sig: killpg_called.append((pid, sig))
        )

        proc = MagicMock()
        proc.pid = 99999
        proc.kill.return_value = None
        proc.wait.return_value = 0

        _kill_process_tree(proc)

        assert not taskkill_called
        assert len(killpg_called) == 1
        assert killpg_called[0][0] == 99999

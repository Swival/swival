"""Tests for the run_command tool."""

import errno
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from swival.tools import _run_command, _read_file, dispatch


@pytest.fixture
def tmp_base(tmp_path):
    """Provide a temporary base directory."""
    return str(tmp_path)


def _which(name: str) -> str:
    """Resolve a command name to its absolute path, skip test if not found."""
    path = shutil.which(name)
    if path is None:
        pytest.skip(f"{name!r} not found on PATH")
    return str(Path(path).resolve())


# ---------- Test 1: Allowed command runs ----------


def test_allowed_command_runs(tmp_base):
    ls_path = _which("ls")
    resolved = {"ls": ls_path}
    result = _run_command(["ls"], tmp_base, resolved)
    # Should succeed without error prefix (empty dir is fine — no output or "(no output)")
    assert not result.startswith("error:")


# ---------- Test 2: Blocked command ----------


def test_blocked_command(tmp_base):
    ls_path = _which("ls")
    resolved = {"ls": ls_path}
    result = _run_command(["rm", "-rf", "/"], tmp_base, resolved)
    assert "error:" in result
    assert "not allowed" in result
    assert "ls" in result  # should list allowed commands


# ---------- Test 3: Path in command[0] rejected ----------


@pytest.mark.parametrize("cmd", ["/usr/bin/ls", "./ls", "../bin/ls"])
def test_path_in_command_rejected(tmp_base, cmd):
    ls_path = _which("ls")
    resolved = {"ls": ls_path}
    result = _run_command([cmd], tmp_base, resolved)
    assert "error:" in result
    assert "bare name" in result


# ---------- Test 4: Path-based bypass blocked ----------


def test_path_bypass_blocked(tmp_base):
    ls_path = _which("ls")
    resolved = {"ls": ls_path}
    result = _run_command(["/tmp/ls"], tmp_base, resolved)
    assert "error:" in result


# ---------- Test 5: Empty command list ----------


def test_empty_command_list(tmp_base):
    result = _run_command([], tmp_base, {"ls": "/bin/ls"})
    assert "error:" in result
    assert "empty" in result


# ---------- Test 5b: String command errors are explicit ----------


def test_string_with_shell_chars_requires_array(tmp_base):
    result = _run_command("grep -n pattern file.py | head", tmp_base, {})
    assert result.startswith(
        'error: "command" must be a JSON array of strings, not a single string.'
    )
    assert 'Right: "command": ["grep", "-n", "pattern", "file.py"]' in result
    assert "Each argument must be a separate element in the array." in result


# ---------- Test 5c: Auto-repair from JSON-encoded array ----------


@pytest.mark.parametrize("unrestricted", [False, True])
def test_auto_repair_json_string_runs(tmp_base, unrestricted):
    echo_path = _which("echo")
    resolved = {"echo": echo_path}
    result = _run_command(
        '["echo", "json-repaired"]',
        tmp_base,
        resolved,
        unrestricted=unrestricted,
    )
    assert "json-repaired" in result
    assert result.rstrip().endswith(
        "(auto-corrected: command was passed as a string, converted to array)"
    )


# ---------- Test 5d: Auto-repair via shlex only in unrestricted mode ----------


@pytest.mark.skipif(sys.platform == "win32", reason="requires /bin/sh")
def test_plain_string_runs_as_shell_in_yolo(tmp_base):
    result = _run_command(
        "echo yolo-repaired",
        tmp_base,
        resolved_commands={},
        unrestricted=True,
    )
    assert "yolo-repaired" in result
    assert not result.startswith("error:")
    assert "(auto-corrected:" not in result


def test_auto_repair_plain_string_sandboxed_auto_splits(tmp_base):
    echo_path = _which("echo")
    resolved = {"echo": echo_path}
    result = _run_command(
        "echo auto-split-ok",
        tmp_base,
        resolved,
        unrestricted=False,
    )
    assert "auto-split-ok" in result
    assert "(auto-corrected:" in result


# ---------- Test 5e: Auto-repair annotation and error-prefix interaction ----------


def test_auto_repair_annotation_appended_on_error(tmp_base):
    result = _run_command(
        '["rm", "-rf", "/"]',
        tmp_base,
        {},
    )
    assert result.startswith("error:")
    assert "not allowed" in result
    assert result.rstrip().endswith(
        "(auto-corrected: command was passed as a string, converted to array)"
    )


def test_auto_repair_annotation_appended_on_timeout(tmp_base):
    sleep_path = _which("sleep")
    resolved = {"sleep": sleep_path}
    result = _run_command(
        '["sleep", "10"]',
        tmp_base,
        resolved,
        timeout=1,
    )
    assert result.startswith("error: command timed out after 1s")
    assert result.rstrip().endswith(
        "(auto-corrected: command was passed as a string, converted to array)"
    )


def test_repaired_error_keeps_error_prefix_for_loop_detection(tmp_base):
    ls_path = _which("ls")
    resolved = {"ls": ls_path}
    result = _run_command(
        '["definitely_not_allowed_cmd_xyz"]',
        tmp_base,
        resolved,
    )
    assert result.startswith("error:")
    assert result.rstrip().endswith(
        "(auto-corrected: command was passed as a string, converted to array)"
    )


# ---------- Test 6: Missing/invalid base_dir ----------


def test_missing_base_dir_reports_error(tmp_path):
    echo_path = _which("echo")
    resolved = {"echo": echo_path}
    missing_dir = tmp_path / "does-not-exist"
    result = _run_command(["echo", "hi"], str(missing_dir), resolved)
    assert "error: base directory does not exist:" in result


def test_base_dir_not_directory_reports_error(tmp_path):
    echo_path = _which("echo")
    resolved = {"echo": echo_path}
    not_a_dir = tmp_path / "not-a-dir"
    not_a_dir.write_text("x")
    result = _run_command(["echo", "hi"], str(not_a_dir), resolved)
    assert "error: base directory is not a directory:" in result


# ---------- Test 7: Timeout enforcement ----------


def test_timeout_enforcement(tmp_base):
    sleep_path = _which("sleep")
    resolved = {"sleep": sleep_path}
    t0 = time.monotonic()
    result = _run_command(["sleep", "10"], tmp_base, resolved, timeout=1)
    elapsed = time.monotonic() - t0
    assert "timed out" in result
    assert elapsed < 5  # should finish well before the 10s sleep


# ---------- Test 8: Timeout clamping ----------


def test_timeout_clamping_high(tmp_base):
    """Timeout of 9999 should be clamped to 120."""
    echo_path = _which("echo")
    resolved = {"echo": echo_path}
    # Just verify it doesn't error — the clamping is internal
    result = _run_command(["echo", "hi"], tmp_base, resolved, timeout=9999)
    assert "hi" in result


def test_timeout_clamping_low(tmp_base):
    """Timeout of -5 should be clamped to 1."""
    echo_path = _which("echo")
    resolved = {"echo": echo_path}
    result = _run_command(["echo", "hi"], tmp_base, resolved, timeout=-5)
    assert "hi" in result


# ---------- Test 9: Non-zero exit code ----------


def test_nonzero_exit_code(tmp_base):
    bash_path = _which("bash")
    resolved = {"bash": bash_path}
    result = _run_command(["bash", "-c", "exit 42"], tmp_base, resolved)
    assert "Exit code: 42" in result


# ---------- Test 10: No shell injection ----------


def test_no_shell_injection(tmp_base):
    ls_path = _which("ls")
    resolved = {"ls": ls_path}
    # The semicolon should be treated as a literal filename argument, not a separator
    result = _run_command(["ls", "; rm -rf /"], tmp_base, resolved)
    # Should NOT have actually executed rm. The result will be an error from ls
    # about a file not found, which is fine.
    assert "error: command timed out" not in result


# ---------- Test 11: Command not in resolved_commands ----------


def test_command_not_in_resolved(tmp_base):
    ls_path = _which("ls")
    resolved = {"ls": ls_path}
    result = _run_command(["nonexistent_command_xyz"], tmp_base, resolved)
    assert "error:" in result
    assert "not allowed" in result
    assert "ls" in result


# ---------- Test 12: stderr capture ----------


def test_stderr_capture(tmp_base):
    bash_path = _which("bash")
    resolved = {"bash": bash_path}
    result = _run_command(["bash", "-c", "echo stderr_msg >&2"], tmp_base, resolved)
    assert "stderr_msg" in result


# ---------- Test 13: Large output truncation without false timeout ----------


def test_large_output_truncation(tmp_base):
    python_path = _which("python3")
    resolved = {"python3": python_path}
    # Produce ~1MB of output — should be saved to .swival/
    result = _run_command(
        ["python3", "-c", "print('x' * 1_000_000)"],
        tmp_base,
        resolved,
        timeout=30,
    )
    assert ".swival/" in result
    assert "read_file" in result
    assert "timed out" not in result  # should NOT have timed out


# ---------- Test 14: Tool absent when no allowed commands ----------


def test_tool_absent_when_no_allowed_commands():
    from swival.tools import TOOLS

    tool_names = [t["function"]["name"] for t in TOOLS]
    assert "run_command" not in tool_names


# ---------- Test 15: Timeout returns partial output ----------


def test_timeout_partial_output(tmp_base):
    bash_path = _which("bash")
    resolved = {"bash": bash_path}
    result = _run_command(
        ["bash", "-c", "echo partial_output_marker; sleep 999"],
        tmp_base,
        resolved,
        timeout=2,
    )
    assert "timed out" in result
    assert "partial_output_marker" in result


# ---------- Test 16: Process tree killed on timeout (Unix-only) ----------


@pytest.mark.skipif(sys.platform == "win32", reason="Unix-only: process group kill")
def test_process_tree_killed(tmp_base):
    bash_path = _which("bash")
    resolved = {"bash": bash_path}
    result = _run_command(
        [
            "bash",
            "-c",
            "echo PGID:$(ps -o pgid= $$ | tr -d ' '); sleep 999 & echo child_started; sleep 999",
        ],
        tmp_base,
        resolved,
        timeout=2,
    )
    assert "child_started" in result
    match = re.search(r"PGID:(\d+)", result)
    assert match, f"missing PGID in output: {result!r}"
    pgid = int(match.group(1))

    # Group should disappear quickly after timeout-driven kill.
    for _ in range(20):
        try:
            os.killpg(pgid, 0)
        except OSError as exc:
            if exc.errno == errno.ESRCH:
                break
            if exc.errno == errno.EPERM:
                pytest.fail(f"permission denied probing process group {pgid}")
            raise
        time.sleep(0.1)
    else:
        pytest.fail(f"process group {pgid} still exists after timeout kill")

    assert "timed out" in result


# ---------- Test 17: PATH shadowing blocked ----------


def test_path_shadowing_blocked(tmp_base, tmp_path):
    """A fake ls on PATH should not run when the real ls is pinned."""
    ls_path = _which("ls")
    resolved = {"ls": ls_path}

    # Create a fake ls script in a temp dir
    fake_dir = tmp_path / "fakepath"
    fake_dir.mkdir()
    fake_ls = fake_dir / "ls"
    fake_ls.write_text("#!/bin/sh\necho FAKE_LS_OUTPUT\n")
    fake_ls.chmod(fake_ls.stat().st_mode | stat.S_IEXEC)

    # Prepend fake dir to PATH — should have no effect since path is pinned
    env_backup = os.environ.get("PATH", "")
    os.environ["PATH"] = str(fake_dir) + os.pathsep + env_backup
    try:
        result = _run_command(["ls"], tmp_base, resolved)
    finally:
        os.environ["PATH"] = env_backup

    assert "FAKE_LS_OUTPUT" not in result


# ---------- Test 18: Startup resolution failure ----------


def test_startup_resolution_failure(tmp_path):
    """If --allowed-commands=nonexistent_xyz, agent.py should exit with error."""
    env = os.environ.copy()
    env["XDG_CONFIG_HOME"] = str(tmp_path)  # isolate from global config
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "swival.agent",
            "test",
            "--model=dummy-model",
            "--allowed-commands=nonexistent_xyz_abc",
        ],
        cwd=str(Path(__file__).parent.parent),
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
    )
    assert result.returncode != 0
    assert "not found on PATH" in result.stderr


# ---------- Test 19: Startup pinning produces absolute paths ----------


def test_startup_pinning_absolute_paths(tmp_path):
    """Startup resolution should produce absolute paths even if PATH has relative entries."""
    rel_bin = tmp_path / "relbin"
    rel_bin.mkdir()
    cmd = rel_bin / "mycmd"
    cmd.write_text("#!/bin/sh\necho hi\n")
    cmd.chmod(cmd.stat().st_mode | stat.S_IEXEC)

    old_cwd = Path.cwd()
    old_path = os.environ.get("PATH", "")
    os.chdir(tmp_path)
    os.environ["PATH"] = f"relbin{os.pathsep}{old_path}"
    try:
        raw = shutil.which("mycmd")
        assert raw is not None
        resolved = Path(raw).resolve()
        assert resolved.is_absolute()
    finally:
        os.chdir(old_cwd)
        os.environ["PATH"] = old_path


# ---------- Test 20: Startup rejects commands inside base_dir ----------


def test_startup_rejects_commands_inside_base_dir(tmp_path):
    """A command whose resolved path is inside base_dir should be rejected."""
    # Create a fake command inside tmp_path (the "base_dir")
    fake_cmd = tmp_path / "mycmd"
    fake_cmd.write_text("#!/bin/sh\necho hello\n")
    fake_cmd.chmod(fake_cmd.stat().st_mode | stat.S_IEXEC)

    # Run agent with base_dir=tmp_path and the fake command on PATH
    env = os.environ.copy()
    env["PATH"] = str(tmp_path) + os.pathsep + env.get("PATH", "")
    env["XDG_CONFIG_HOME"] = str(tmp_path / "xdg")  # isolate from global config

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "swival.agent",
            "test",
            "--model=dummy-model",
            f"--base-dir={tmp_path}",
            "--allowed-commands=mycmd",
        ],
        cwd=str(Path(__file__).parent.parent),
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
    )
    assert result.returncode != 0
    assert "inside base directory" in result.stderr


# ---------- Test 21: CLI parsing ----------


def test_cli_parsing_normal():
    """'ls,git,python3' should parse correctly."""
    raw = "ls,git,python3"
    allowed = {c.strip() for c in raw.split(",") if c.strip()}
    assert allowed == {"ls", "git", "python3"}


def test_cli_parsing_normalization():
    """'ls, git,, ' should parse to {'ls', 'git'}."""
    raw = "ls, git,, "
    allowed = {c.strip() for c in raw.split(",") if c.strip()}
    assert allowed == {"ls", "git"}


# ---------- Test: dispatch wiring ----------


def test_dispatch_run_command(tmp_base):
    echo_path = _which("echo")
    resolved = {"echo": echo_path}
    result = dispatch(
        "run_command",
        {"command": ["echo", "hello_dispatch"]},
        tmp_base,
        resolved_commands=resolved,
    )
    assert "hello_dispatch" in result


# ---------- Test: Small output stays inline ----------


def test_small_output_stays_inline(tmp_base):
    echo_path = _which("echo")
    resolved = {"echo": echo_path}
    result = _run_command(["echo", "hello"], tmp_base, resolved)
    assert "hello" in result
    assert ".swival/" not in result


# ---------- Test: Large output saved to file ----------


def test_large_output_saved_to_file(tmp_base):
    python_path = _which("python3")
    resolved = {"python3": python_path}
    # Produce ~25KB of output (over the 10KB inline limit)
    result = _run_command(
        ["python3", "-c", "print('A' * 25_000)"],
        tmp_base,
        resolved,
        timeout=30,
    )
    assert ".swival/" in result
    assert "too large for context" in result
    assert "read_file" in result

    # The file should actually exist
    swival_dir = Path(tmp_base) / ".swival"
    assert swival_dir.is_dir()
    files = list(swival_dir.glob("cmd_output_*.txt"))
    assert len(files) == 1
    content = files[0].read_text()
    assert "A" * 100 in content


# ---------- Test: Saved file readable via read_file ----------


def test_saved_file_readable_via_read_file(tmp_base):
    python_path = _which("python3")
    resolved = {"python3": python_path}
    result = _run_command(
        ["python3", "-c", "print('B' * 25_000)"],
        tmp_base,
        resolved,
        timeout=30,
    )
    # Extract the .swival/... path from the summary
    match = re.search(r"(\.swival/cmd_output_\w+\.txt)", result)
    assert match, f"no .swival path found in: {result!r}"
    rel_path = match.group(1)

    # Use _read_file to read it back
    read_result = _read_file(rel_path, tmp_base)
    assert "B" * 100 in read_result
    assert not read_result.startswith("error:")


# ---------- Test: .swival dir created on demand ----------


def test_swival_dir_created(tmp_base):
    swival_dir = Path(tmp_base) / ".swival"
    assert not swival_dir.exists()

    python_path = _which("python3")
    resolved = {"python3": python_path}
    _run_command(
        ["python3", "-c", "print('C' * 25_000)"],
        tmp_base,
        resolved,
        timeout=30,
    )
    assert swival_dir.is_dir()


# ---------- Test: .swival file readable via read_file with line numbers ----------


def test_swival_file_readable_with_line_numbers(tmp_base):
    """read_file on .swival output returns line-numbered content like any other file."""
    from swival.tools import _save_large_output

    lines = [f"line_{i}" for i in range(100)]
    payload = "\n".join(lines)
    summary = _save_large_output(payload, tmp_base)
    match = re.search(r"(\.swival/cmd_output_\w+\.txt)", summary)
    assert match, f"no .swival path found in: {summary!r}"
    rel_path = match.group(1)

    read_result = _read_file(rel_path, tmp_base)
    assert "1: " in read_result
    assert "line_0" in read_result


# ---------- Test: Pagination hint shows next offset ----------


def test_pagination_hint_on_truncation(tmp_base):
    """When read_file truncates, the hint tells the model the next offset."""
    # Create a file large enough to exceed 50KB of numbered output
    line = "A" * 80
    text = "\n".join([line] * 1000)
    (Path(tmp_base) / "big.txt").write_text(text)

    result = _read_file("big.txt", tmp_base)
    assert "more lines, use offset=" in result
    # Extract the suggested offset and use it
    match = re.search(r"use offset=(\d+) to continue", result)
    assert match
    next_offset = int(match.group(1))
    assert next_offset > 1

    # Reading from the suggested offset should return more content
    result2 = _read_file("big.txt", tmp_base, offset=next_offset)
    assert "A" * 80 in result2


# ---------- Test: No pagination hint when file fits ----------


def test_no_pagination_hint_when_complete(tmp_base):
    """When the entire file fits, no pagination hint is shown."""
    (Path(tmp_base) / "small.txt").write_text("hello\nworld\n")
    result = _read_file("small.txt", tmp_base)
    assert "more lines" not in result
    assert "hello" in result
    assert "world" in result


# ---------- Test: .swival paginated read covers full content ----------


def test_swival_paginated_read(tmp_base):
    """A large .swival file can be fully read by following pagination hints."""
    from swival.tools import _save_large_output

    # 200 lines — enough to test pagination with offset/limit
    lines = [f"output_line_{i:04d}" for i in range(200)]
    payload = "\n".join(lines)
    summary = _save_large_output(payload, tmp_base)
    match = re.search(r"(\.swival/cmd_output_\w+\.txt)", summary)
    assert match
    rel_path = match.group(1)

    # Read with a small limit to force pagination
    result1 = _read_file(rel_path, tmp_base, limit=50)
    assert "output_line_0000" in result1
    assert "more lines, use offset=" in result1

    offset_match = re.search(r"use offset=(\d+) to continue", result1)
    assert offset_match
    next_offset = int(offset_match.group(1))

    result2 = _read_file(rel_path, tmp_base, offset=next_offset, limit=50)
    assert f"output_line_{next_offset - 1:04d}" in result2


# ---------- Test: Truncated output says "possibly truncated" not "Full" ----------


def test_truncated_output_label(tmp_base):
    """When output exceeds MAX_FILE_OUTPUT, the summary should not say 'Full output'."""
    from swival.tools import _save_large_output

    # Simulate a truncated save
    summary = _save_large_output("x" * 20_000, tmp_base, was_truncated=True)
    assert "possibly truncated" in summary.lower()
    assert "Full output" not in summary

    # Non-truncated save should say "Full output"
    summary2 = _save_large_output("y" * 20_000, tmp_base, was_truncated=False)
    assert "Full output" in summary2

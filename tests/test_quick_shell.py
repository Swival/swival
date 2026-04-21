"""Tests for the !! quick-shell feature: execution, formatting, isolation."""

from __future__ import annotations

from swival.input_dispatch import InputContext, parse_input_line


def _make_ctx(*, messages=None):
    from swival.thinking import ThinkingState
    from swival.todo import TodoState

    msgs = messages if messages is not None else []
    return InputContext(
        messages=msgs,
        tools=[],
        base_dir="/tmp",
        turn_state={"max_turns": 10, "turns_used": 0},
        thinking_state=ThinkingState(),
        todo_state=TodoState(),
        snapshot_state=None,
        file_tracker=None,
        no_history=True,
        continue_here=False,
        verbose=False,
        loop_kwargs={
            "model_id": "test",
            "api_base": "http://test",
            "context_length": 128000,
            "files_mode": "some",
            "compaction_state": None,
            "command_policy": None,
            "top_p": None,
            "seed": None,
            "llm_kwargs": {},
        },
    )


class TestRunQuickShell:
    """Unit tests for the _run_quick_shell helper."""

    def test_echo(self, tmp_path):
        from swival.agent import _run_quick_shell

        rc, out = _run_quick_shell("echo hello", str(tmp_path))
        assert rc == 0
        assert out == "hello"

    def test_nonzero_exit(self, tmp_path):
        from swival.agent import _run_quick_shell

        rc, out = _run_quick_shell("false", str(tmp_path))
        assert rc != 0

    def test_shell_syntax_works(self, tmp_path):
        from swival.agent import _run_quick_shell

        rc, out = _run_quick_shell("echo hello && echo world", str(tmp_path))
        assert rc == 0
        assert "hello" in out
        assert "world" in out

    def test_pipe(self, tmp_path):
        from swival.agent import _run_quick_shell

        rc, out = _run_quick_shell("echo hello | tr h H", str(tmp_path))
        assert rc == 0
        assert out == "Hello"

    def test_cwd_respected(self, tmp_path):
        from swival.agent import _run_quick_shell

        rc, out = _run_quick_shell("pwd", str(tmp_path))
        assert rc == 0
        import os

        assert os.path.realpath(out.strip()) == os.path.realpath(str(tmp_path))

    def test_nonexistent_command(self, tmp_path):
        from swival.agent import _run_quick_shell

        rc, out = _run_quick_shell("nonexistent_cmd_xyz_123", str(tmp_path))
        assert rc != 0


class TestExecuteQuickShell:
    """Integration tests through execute_input."""

    def test_runs_regardless_of_policy(self, tmp_path):
        """!! is user-initiated; command policy does not apply."""
        from swival.agent import execute_input
        from swival.command_policy import CommandPolicy

        ctx = _make_ctx()
        ctx.loop_kwargs["command_policy"] = CommandPolicy("none")
        ctx.base_dir = str(tmp_path)
        parsed = parse_input_line("!! echo ok")
        result = execute_input(parsed, ctx, mode="repl")
        assert result.kind == "state_change"
        assert not result.is_error

    def test_shell_syntax_allowed(self, tmp_path):
        from swival.agent import execute_input

        ctx = _make_ctx()
        ctx.base_dir = str(tmp_path)
        parsed = parse_input_line("!! echo a && echo b")
        result = execute_input(parsed, ctx, mode="repl")
        assert result.kind == "state_change"
        assert not result.is_error


class TestNoLlmContamination:
    """Verify !! does not touch the messages list."""

    def test_messages_unchanged(self, tmp_path):
        from swival.agent import execute_input

        messages = [{"role": "system", "content": "test"}]
        ctx = _make_ctx(messages=messages)
        ctx.base_dir = str(tmp_path)
        original_len = len(messages)
        parsed = parse_input_line("!! echo hello")
        execute_input(parsed, ctx, mode="repl")
        assert len(messages) == original_len
        assert messages[0] == {"role": "system", "content": "test"}


class TestFormatting:
    """Verify fmt.quick_shell output."""

    def test_success_output(self, capsys):
        from swival import fmt

        fmt.quick_shell("echo hello", 0, "hello")
        captured = capsys.readouterr()
        assert "$ echo hello" in captured.err
        assert "hello" in captured.err
        assert "exit" not in captured.err

    def test_failure_shows_exit_code(self, capsys):
        from swival import fmt

        fmt.quick_shell("false", 1, "")
        captured = capsys.readouterr()
        assert "$ false" in captured.err
        assert "exit 1" in captured.err

    def test_empty_output(self, capsys):
        from swival import fmt

        fmt.quick_shell("true", 0, "")
        captured = capsys.readouterr()
        assert "$ true" in captured.err
        lines = [x for x in captured.err.splitlines() if x.strip()]
        assert len(lines) == 1  # just the header

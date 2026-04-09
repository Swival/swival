"""Tests for input_dispatch: parse_input_line, is_command_script, execute_input."""

from __future__ import annotations

import types

from swival.input_dispatch import (
    InputContext,
    is_command_script,
    parse_input_line,
)


class TestParseInputLine:
    def test_empty(self):
        p = parse_input_line("")
        assert p.raw == ""
        assert not p.is_command
        assert not p.is_custom_command

    def test_whitespace_only(self):
        p = parse_input_line("   ")
        assert p.raw == ""

    def test_plain_text(self):
        p = parse_input_line("fix the bug in main.py")
        assert p.raw == "fix the bug in main.py"
        assert not p.is_command
        assert not p.is_custom_command

    def test_slash_command_no_arg(self):
        p = parse_input_line("/help")
        assert p.cmd == "/help"
        assert p.cmd_arg == ""
        assert p.is_command

    def test_slash_command_with_arg(self):
        p = parse_input_line("/simplify swival/agent.py")
        assert p.cmd == "/simplify"
        assert p.cmd_arg == "swival/agent.py"
        assert p.is_command

    def test_slash_command_case_insensitive(self):
        p = parse_input_line("/HELP")
        assert p.cmd == "/help"

    def test_bang_command(self):
        p = parse_input_line("!context")
        assert p.is_custom_command
        assert not p.is_command
        assert p.raw == "!context"

    def test_bang_space_not_command(self):
        """! foo (with space) is plain text, not a custom command."""
        p = parse_input_line("! foo")
        assert not p.is_custom_command
        assert not p.is_command

    def test_unknown_slash(self):
        p = parse_input_line("/nonexistent foo")
        assert p.is_command
        assert p.cmd == "/nonexistent"
        assert p.cmd_arg == "foo"


class TestIsCommandScript:
    def test_plain_text(self):
        assert not is_command_script("fix the bug")

    def test_starts_with_known_command(self):
        assert is_command_script("/simplify swival/agent.py")

    def test_starts_with_bang(self):
        assert is_command_script("!context")

    def test_leading_blank_lines(self):
        assert is_command_script("\n\n/help\nsome text")

    def test_bang_space_not_script(self):
        assert not is_command_script("! not a command")

    def test_empty(self):
        assert not is_command_script("")

    def test_multiline_script(self):
        assert is_command_script("/profile fast\n/simplify agent.py")

    def test_unknown_slash_is_script(self):
        assert is_command_script("/nonexistent\nsome text")

    def test_plain_multiline(self):
        assert not is_command_script("please fix this\n/simplify")


class TestRunInputScript:
    """Tests for run_input_script."""

    def test_state_changes_persist(self):
        from swival.agent import run_input_script

        ctx = self._make_ctx()
        assert ctx.turn_state["max_turns"] == 10
        result = run_input_script("/extend 50\n/extend 100", ctx, mode="oneshot")
        assert ctx.turn_state["max_turns"] == 100
        assert result.text is not None
        assert "100" in result.text

    def test_exit_stops_script(self):
        from swival.agent import run_input_script

        ctx = self._make_ctx()
        run_input_script("/extend 50\n/exit\n/extend 200", ctx, mode="oneshot")
        assert ctx.turn_state["max_turns"] == 50

    def test_repl_only_rejected_in_oneshot(self):
        from swival.agent import run_input_script

        ctx = self._make_ctx()
        result = run_input_script("/continue", ctx, mode="oneshot")
        assert result.text is not None
        assert "not available" in result.text

    def test_last_visible_output_wins(self):
        from swival.agent import run_input_script

        ctx = self._make_ctx()
        result = run_input_script("/help\n/status", ctx, mode="oneshot")
        # Last output should be from /status, not /help
        assert "model:" in result.text

    def test_unknown_slash_returns_error(self):
        from swival.agent import run_input_script

        ctx = self._make_ctx()
        result = run_input_script("/nonexistent arg", ctx, mode="oneshot")
        assert (
            result.text
            == "error: unknown command /nonexistent. Run /help to list commands."
        )

    def test_empty_script(self):
        from swival.agent import run_input_script

        ctx = self._make_ctx()
        result = run_input_script("", ctx, mode="oneshot")
        assert result.text is None

    def _make_ctx(self):
        from swival.thinking import ThinkingState
        from swival.todo import TodoState

        return InputContext(
            messages=[],
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
                "command_policy": types.SimpleNamespace(mode="allowlist"),
                "top_p": 1.0,
                "seed": None,
                "llm_kwargs": {},
            },
        )


class TestExecuteInput:
    """Basic execute_input tests for non-agent-turn commands."""

    def test_exit(self):
        from swival.agent import execute_input

        parsed = parse_input_line("/exit")
        result = execute_input(parsed, self._make_ctx(), mode="repl")
        assert result.stop is True
        assert result.kind == "flow_control"

    def test_quit(self):
        from swival.agent import execute_input

        parsed = parse_input_line("/quit")
        result = execute_input(parsed, self._make_ctx(), mode="repl")
        assert result.stop is True

    def test_help(self):
        from swival.agent import execute_input

        parsed = parse_input_line("/help")
        result = execute_input(parsed, self._make_ctx(), mode="repl")
        assert result.kind == "info"
        assert "/help" in result.text

    def test_extend_double(self):
        from swival.agent import execute_input

        ctx = self._make_ctx()
        parsed = parse_input_line("/extend")
        result = execute_input(parsed, ctx, mode="repl")
        assert result.kind == "state_change"
        assert ctx.turn_state["max_turns"] == 20

    def test_extend_specific(self):
        from swival.agent import execute_input

        ctx = self._make_ctx()
        parsed = parse_input_line("/extend 50")
        execute_input(parsed, ctx, mode="repl")
        assert ctx.turn_state["max_turns"] == 50

    def test_repl_only_in_oneshot(self):
        from swival.agent import execute_input

        parsed = parse_input_line("/continue")
        result = execute_input(parsed, self._make_ctx(), mode="oneshot")
        assert "not available" in result.text

    def test_copy_repl_only_in_oneshot(self):
        from swival.agent import execute_input

        parsed = parse_input_line("/copy")
        result = execute_input(parsed, self._make_ctx(), mode="oneshot")
        assert "not available" in result.text

    def test_unknown_slash_returns_error(self):
        from swival.agent import execute_input

        parsed = parse_input_line("/nonexistent foo")
        result = execute_input(parsed, self._make_ctx(), mode="repl")
        assert result.kind == "info"
        assert result.is_error is True
        assert (
            result.text
            == "error: unknown command /nonexistent. Run /help to list commands."
        )

    def test_empty_line(self):
        from swival.agent import execute_input

        parsed = parse_input_line("")
        result = execute_input(parsed, self._make_ctx(), mode="repl")
        assert result.kind == "flow_control"
        assert result.text is None

    def _make_ctx(self):
        from swival.thinking import ThinkingState
        from swival.todo import TodoState

        return InputContext(
            messages=[],
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
                "command_policy": types.SimpleNamespace(mode="allowlist"),
                "top_p": 1.0,
                "seed": None,
                "llm_kwargs": {},
            },
        )

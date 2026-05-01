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

    def test_multiline_plain_text(self):
        """Multiline plain text is preserved as-is."""
        p = parse_input_line("fix this bug\nit crashes on startup")
        assert p.raw == "fix this bug\nit crashes on startup"
        assert not p.is_command
        assert not p.is_custom_command

    def test_multiline_slash_on_second_line_is_plain_text(self):
        """A / on a non-first line does not trigger command dispatch."""
        p = parse_input_line("please help\n/with this file")
        assert not p.is_command
        assert p.raw == "please help\n/with this file"

    def test_multiline_command_on_first_line(self):
        """A slash command on the first line is still detected."""
        p = parse_input_line("/help\nsome extra text")
        assert p.is_command
        assert p.cmd == "/help"

    def test_multiline_command_arg_includes_continuation(self):
        """Continuation lines after a slash command are included in cmd_arg."""
        p = parse_input_line("/remember this fact\nand this detail\nand more")
        assert p.is_command
        assert p.cmd == "/remember"
        assert p.cmd_arg == "this fact\nand this detail\nand more"

    def test_multiline_command_no_first_line_arg(self):
        """Slash command with no arg on first line gets continuation as cmd_arg."""
        p = parse_input_line("/remember\nthe whole thing")
        assert p.cmd_arg == "the whole thing"

    def test_leading_trailing_blank_lines_stripped(self):
        """Leading/trailing whitespace-only lines are stripped, interior preserved."""
        p = parse_input_line("\n\nhello\nworld\n\n")
        assert p.raw == "hello\nworld"

    def test_whitespace_only_boundary_lines_stripped(self):
        """Lines with only spaces at boundaries are also stripped."""
        p = parse_input_line("\n  \nhello\n  \n")
        assert p.raw == "hello"

    def test_multiline_bang_on_second_line_is_plain_text(self):
        """A ! on a non-first line does not trigger bang command."""
        p = parse_input_line("some text\n!command")
        assert not p.is_custom_command

    def test_quick_shell(self):
        p = parse_input_line("!! git status")
        assert p.is_command
        assert p.cmd == "!!"
        assert p.cmd_arg == "git status"
        assert not p.is_custom_command

    def test_quick_shell_preserves_arg_whitespace(self):
        p = parse_input_line("!!  echo  hello ")
        assert p.cmd == "!!"
        assert p.cmd_arg == " echo  hello"

    def test_double_bang_no_space_is_custom_command(self):
        """!!foo (no space) is a custom command named !foo, not quick shell."""
        p = parse_input_line("!!foo")
        assert p.is_custom_command
        assert not p.is_command

    def test_double_bang_alone_is_custom_command(self):
        """!! alone (no trailing content) is a custom command."""
        p = parse_input_line("!!")
        assert p.is_custom_command
        assert not p.is_command

    def test_single_bang_unchanged(self):
        p = parse_input_line("!deploy")
        assert p.is_custom_command
        assert not p.is_command

    def test_bang_space_unchanged(self):
        p = parse_input_line("! text")
        assert not p.is_custom_command
        assert not p.is_command


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

    def test_quick_shell_not_script(self):
        """!! is REPL-only and must not trigger command-script detection."""
        assert not is_command_script("!! git status")

    def test_quick_shell_then_slash_not_script(self):
        """First line governs; !! on first line means not a script."""
        assert not is_command_script("!! git status\n/help")


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
                "top_p": None,
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

    def test_quick_shell_repl_only_in_oneshot(self):
        from swival.agent import execute_input

        parsed = parse_input_line("!! echo hi")
        result = execute_input(parsed, self._make_ctx(), mode="oneshot")
        assert "not available" in result.text

    def test_quick_shell_empty_arg(self):
        from swival.agent import execute_input

        # "!! " with trailing space: first_line is "!!" after strip, which starts
        # with "!" but isn't "!! " (3+ chars), so it's a custom command.
        # We test the empty-arg guard via a hand-built ParsedInput.
        from swival.input_dispatch import ParsedInput

        p = ParsedInput(raw="!! ", cmd="!!", cmd_arg="", is_command=True)
        result = execute_input(p, self._make_ctx(), mode="repl")
        assert result.is_error
        assert "usage" in result.text

    def test_quick_shell_help_included(self):
        from swival.agent import execute_input

        parsed = parse_input_line("/help")
        result = execute_input(parsed, self._make_ctx(), mode="repl")
        assert "!!" in result.text

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
                "top_p": None,
                "seed": None,
                "llm_kwargs": {},
            },
        )


def _stub_run_agent_step(monkeypatch, *, answer="ok"):
    """Replace _run_agent_step with a recorder. Returns the calls list."""
    from swival import agent
    from swival.input_dispatch import StepResult

    calls: list[dict] = []

    def _fake(
        content, history_label, ctx, *, interrupt_label="question", goal_launch=False
    ):
        calls.append(
            {
                "content": content,
                "history_label": history_label,
                "interrupt_label": interrupt_label,
                "goal_launch": goal_launch,
                "messages_snapshot": list(ctx.messages),
            }
        )
        return StepResult(kind="agent_turn", text=answer)

    monkeypatch.setattr(agent, "_run_agent_step", _fake)
    return calls


def _make_goal_ctx(
    *,
    goal_state=None,
    report=None,
):
    from swival.goal import GoalState
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
            "top_p": None,
            "seed": None,
            "llm_kwargs": {},
            "report": report,
        },
        goal_state=goal_state if goal_state is not None else GoalState(),
    )


class TestGoalCommand:
    """Tests for the /goal slash command."""

    def test_goal_registered(self):
        from swival.input_commands import INPUT_COMMANDS

        assert "/goal" in INPUT_COMMANDS

    def test_bare_goal_no_objective(self, monkeypatch):
        from swival.agent import execute_input

        calls = _stub_run_agent_step(monkeypatch)
        ctx = _make_goal_ctx()
        result = execute_input(parse_input_line("/goal"), ctx, mode="repl")
        assert result.kind == "state_change"
        assert "No goal" in result.text
        assert calls == []

    def test_goal_create_objective_launches_loop(self, monkeypatch):
        from swival.agent import execute_input

        calls = _stub_run_agent_step(monkeypatch)
        ctx = _make_goal_ctx()
        result = execute_input(
            parse_input_line("/goal Ship the auth migration"),
            ctx,
            mode="repl",
        )
        assert result.kind == "agent_turn"
        assert ctx.goal_state.has_active()
        assert len(calls) == 1
        call = calls[0]
        assert call["content"] is None
        assert call["goal_launch"] is True
        assert call["interrupt_label"] == "goal"
        assert "Ship the auth migration" in call["history_label"]
        tool_names = {t["function"]["name"] for t in ctx.tools}
        assert "complete_goal" in tool_names
        assert "get_goal" not in tool_names
        assert "create_goal" not in tool_names
        assert "update_goal" not in tool_names
        # The synthetic start prompt was appended to the transcript before the loop ran.
        synth = ctx.messages[-1]
        assert synth["role"] == "user"
        assert synth["_swival_synthetic"] is True
        assert "[goal start]" in synth["content"]
        assert "Ship the auth migration" in synth["content"]

    def test_goal_create_raises_default_max_turns(self, monkeypatch):
        from swival.agent import execute_input

        _stub_run_agent_step(monkeypatch)
        ctx = _make_goal_ctx()
        ctx.turn_state["max_turns"] = 100
        result = execute_input(parse_input_line("/goal Ship it"), ctx, mode="repl")
        assert result.kind == "agent_turn"
        assert ctx.turn_state["max_turns"] == 500

    def test_goal_create_preserves_non_default_max_turns(self, monkeypatch):
        from swival.agent import execute_input

        _stub_run_agent_step(monkeypatch)
        ctx = _make_goal_ctx()
        ctx.turn_state["max_turns"] = 250
        result = execute_input(parse_input_line("/goal Ship it"), ctx, mode="repl")
        assert result.kind == "agent_turn"
        assert ctx.turn_state["max_turns"] == 250

    def test_goal_summary_after_create(self, monkeypatch):
        from swival.agent import execute_input

        _stub_run_agent_step(monkeypatch)
        ctx = _make_goal_ctx()
        execute_input(parse_input_line("/goal ship it"), ctx, mode="repl")
        result = execute_input(parse_input_line("/goal"), ctx, mode="repl")
        assert "active" in result.text
        assert "ship it" in result.text

    def test_goal_create_blocked_when_existing_does_not_launch(self, monkeypatch):
        from swival.agent import execute_input

        calls = _stub_run_agent_step(monkeypatch)
        ctx = _make_goal_ctx()
        execute_input(parse_input_line("/goal first"), ctx, mode="repl")
        assert len(calls) == 1
        result = execute_input(parse_input_line("/goal second"), ctx, mode="repl")
        assert result.is_error is True
        assert result.kind == "state_change"
        # Second call did NOT launch a new loop turn.
        assert len(calls) == 1
        tool_names = [t["function"]["name"] for t in ctx.tools]
        assert tool_names.count("complete_goal") == 1

    def test_goal_replace_launches_loop(self, monkeypatch):
        from swival.agent import execute_input

        calls = _stub_run_agent_step(monkeypatch)
        ctx = _make_goal_ctx()
        execute_input(parse_input_line("/goal first"), ctx, mode="repl")
        result = execute_input(
            parse_input_line("/goal replace second objective"), ctx, mode="repl"
        )
        assert result.kind == "agent_turn"
        assert ctx.goal_state.get().objective == "second objective"
        assert len(calls) == 2
        assert calls[1]["goal_launch"] is True
        assert "second objective" in calls[1]["history_label"]
        assert "replace" in calls[1]["history_label"]

    def test_goal_pause_and_resume_do_not_launch(self, monkeypatch):
        from swival.agent import execute_input

        calls = _stub_run_agent_step(monkeypatch)
        ctx = _make_goal_ctx()
        execute_input(parse_input_line("/goal ship"), ctx, mode="repl")
        result = execute_input(parse_input_line("/goal pause"), ctx, mode="repl")
        assert result.kind == "state_change"
        assert ctx.goal_state.get().status == "paused"
        result = execute_input(parse_input_line("/goal resume"), ctx, mode="repl")
        assert result.kind == "state_change"
        assert ctx.goal_state.get().status == "active"
        # Only the initial create launched the loop; pause/resume did not.
        assert len(calls) == 1

    def test_goal_pause_no_active(self, monkeypatch):
        from swival.agent import execute_input

        calls = _stub_run_agent_step(monkeypatch)
        ctx = _make_goal_ctx()
        result = execute_input(parse_input_line("/goal pause"), ctx, mode="repl")
        assert result.is_error is True
        assert calls == []

    def test_goal_clear_does_not_launch(self, monkeypatch):
        from swival.agent import execute_input

        calls = _stub_run_agent_step(monkeypatch)
        ctx = _make_goal_ctx()
        execute_input(parse_input_line("/goal ship"), ctx, mode="repl")
        result = execute_input(parse_input_line("/goal clear"), ctx, mode="repl")
        assert result.is_error is False
        assert result.kind == "state_change"
        assert ctx.goal_state.get() is None
        assert len(calls) == 1  # only the create
        tool_names = {t["function"]["name"] for t in ctx.tools}
        assert "complete_goal" not in tool_names

    def test_clear_command_removes_goal_tools(self, monkeypatch):
        from swival.agent import execute_input

        _stub_run_agent_step(monkeypatch)
        ctx = _make_goal_ctx()
        execute_input(parse_input_line("/goal ship"), ctx, mode="repl")
        assert "complete_goal" in {t["function"]["name"] for t in ctx.tools}
        result = execute_input(parse_input_line("/clear"), ctx, mode="repl")
        assert result.kind == "state_change"
        tool_names = {t["function"]["name"] for t in ctx.tools}
        assert "complete_goal" not in tool_names

    def test_completed_goal_removes_goal_tools(self, monkeypatch):
        from swival import agent
        from swival.agent import _ensure_goal_tools_enabled, _run_agent_step
        from swival.goal import GoalStatus

        ctx = _make_goal_ctx()
        ctx.goal_state.create("ship")
        _ensure_goal_tools_enabled(ctx.tools)

        def _fake_invoke(content, ctx_arg, *, goal_launch=False):
            ctx_arg.goal_state.set_status(GoalStatus.COMPLETE)
            return "done", False, False

        monkeypatch.setattr(agent, "_invoke_agent_turn", _fake_invoke)
        result = _run_agent_step(None, "/goal ship", ctx, goal_launch=True)
        assert result.kind == "agent_turn"
        tool_names = {t["function"]["name"] for t in ctx.tools}
        assert "complete_goal" not in tool_names

    def test_goal_clear_no_goal(self, monkeypatch):
        from swival.agent import execute_input

        calls = _stub_run_agent_step(monkeypatch)
        ctx = _make_goal_ctx()
        result = execute_input(parse_input_line("/goal clear"), ctx, mode="repl")
        assert "no goal" in result.text.lower()
        assert calls == []

    def test_goal_oneshot_always_refused_no_launch(self, monkeypatch):
        from swival.agent import execute_input

        calls = _stub_run_agent_step(monkeypatch)
        ctx = _make_goal_ctx()
        result = execute_input(parse_input_line("/goal ship"), ctx, mode="oneshot")
        assert result.is_error is True
        assert result.kind == "state_change"
        assert ctx.goal_state.get() is None
        assert calls == []

    def test_goal_replace_oneshot_refused(self, monkeypatch):
        from swival.agent import execute_input

        calls = _stub_run_agent_step(monkeypatch)
        ctx = _make_goal_ctx()
        result = execute_input(
            parse_input_line("/goal replace ship"), ctx, mode="oneshot"
        )
        assert result.is_error is True
        assert calls == []


class TestGoalCommandReport:
    """Slash transitions record goal events into the report collector."""

    def test_create_records_event(self, monkeypatch):
        from swival.agent import execute_input
        from swival.report import ReportCollector

        _stub_run_agent_step(monkeypatch)
        report = ReportCollector()
        ctx = _make_goal_ctx(report=report)
        execute_input(parse_input_line("/goal ship"), ctx, mode="repl")
        actions = [e["action"] for e in report.goal_events]
        assert actions == ["created"]

    def test_pause_resume_clear_record_events(self, monkeypatch):
        from swival.agent import execute_input
        from swival.report import ReportCollector

        _stub_run_agent_step(monkeypatch)
        report = ReportCollector()
        ctx = _make_goal_ctx(report=report)
        execute_input(parse_input_line("/goal ship"), ctx, mode="repl")
        execute_input(parse_input_line("/goal pause"), ctx, mode="repl")
        execute_input(parse_input_line("/goal resume"), ctx, mode="repl")
        execute_input(parse_input_line("/goal clear"), ctx, mode="repl")
        actions = [e["action"] for e in report.goal_events]
        assert actions == ["created", "paused", "resumed", "cleared"]

    def test_replace_records_event(self, monkeypatch):
        from swival.agent import execute_input
        from swival.report import ReportCollector

        _stub_run_agent_step(monkeypatch)
        report = ReportCollector()
        ctx = _make_goal_ctx(report=report)
        execute_input(parse_input_line("/goal first"), ctx, mode="repl")
        execute_input(parse_input_line("/goal replace second"), ctx, mode="repl")
        actions = [e["action"] for e in report.goal_events]
        assert actions == ["created", "replaced"]

"""Tests for the /reasoning REPL command."""

from unittest.mock import patch

from swival.agent import _repl_reasoning, execute_input
from swival.completer import SwivalCompleter
from swival.input_commands import INPUT_COMMANDS
from swival.input_dispatch import InputContext, parse_input_line
from swival.thinking import ThinkingState
from swival.todo import TodoState


def _make_repl_kwargs(effort="high"):
    llm_kwargs = {"provider": "openrouter", "api_key": "k", "prompt_cache": False}
    if effort is not None:
        llm_kwargs["reasoning_effort"] = effort
    return {
        "model_id": "org/model",
        "api_base": None,
        "context_length": 8192,
        "llm_kwargs": llm_kwargs,
        "max_output_tokens": 4096,
        "temperature": None,
        "top_p": None,
        "seed": None,
    }


def _make_ctx(effort="high"):
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
        interactive=False,
        loop_kwargs=_make_repl_kwargs(effort),
    )


class _FakeSubagentManager:
    def __init__(self, template):
        self._template = template


class TestReplReasoning:
    def test_registered(self):
        info = INPUT_COMMANDS["/reasoning"]
        assert info.kind == "state_change"
        assert info.arg_type == "reasoning"
        assert "repl" in info.modes and "oneshot" in info.modes

    def test_no_arg_shows_current_and_levels(self):
        kwargs = _make_repl_kwargs("medium")
        last, msg, err = _repl_reasoning("", repl_kwargs=kwargs)
        assert not err
        assert last is None
        assert "reasoning effort: medium" in msg
        assert "xhigh" in msg

    def test_no_arg_reports_default_when_unset(self):
        kwargs = _make_repl_kwargs(None)
        _, msg, err = _repl_reasoning("", repl_kwargs=kwargs)
        assert not err
        assert "reasoning effort: default" in msg

    def test_switch_commits_and_records_previous(self):
        kwargs = _make_repl_kwargs("high")
        old_llm_kwargs = kwargs["llm_kwargs"]
        manager = _FakeSubagentManager({"llm_kwargs": old_llm_kwargs})
        last, msg, err = _repl_reasoning(
            "low", repl_kwargs=kwargs, subagent_manager=manager
        )
        assert not err
        assert last == "high"
        assert "low" in msg and "was high" in msg
        assert kwargs["llm_kwargs"]["reasoning_effort"] == "low"
        assert manager._template["llm_kwargs"] is kwargs["llm_kwargs"]
        # The previous dict is left alone so in-flight callers are unaffected.
        assert old_llm_kwargs["reasoning_effort"] == "high"
        assert kwargs["llm_kwargs"]["prompt_cache"] is False

    def test_case_insensitive(self):
        kwargs = _make_repl_kwargs("high")
        _, _, err = _repl_reasoning("XHigh", repl_kwargs=kwargs)
        assert not err
        assert kwargs["llm_kwargs"]["reasoning_effort"] == "xhigh"

    def test_unknown_level_is_error(self):
        kwargs = _make_repl_kwargs("high")
        last, msg, err = _repl_reasoning("turbo", repl_kwargs=kwargs, last_effort="low")
        assert err
        assert last == "low"
        assert "unknown reasoning effort" in msg
        assert kwargs["llm_kwargs"]["reasoning_effort"] == "high"

    def test_same_level_is_noop(self):
        kwargs = _make_repl_kwargs("high")
        last, msg, err = _repl_reasoning("high", repl_kwargs=kwargs, last_effort="low")
        assert not err
        assert last == "low"
        assert "already" in msg

    def test_revert_without_history_is_error(self):
        kwargs = _make_repl_kwargs("high")
        _, msg, err = _repl_reasoning("-", repl_kwargs=kwargs)
        assert err
        assert "no previous" in msg

    def test_revert_restores_previous(self):
        kwargs = _make_repl_kwargs("high")
        last, _, _ = _repl_reasoning("low", repl_kwargs=kwargs)
        last, msg, err = _repl_reasoning("-", repl_kwargs=kwargs, last_effort=last)
        assert not err
        assert last == "low"
        assert kwargs["llm_kwargs"]["reasoning_effort"] == "high"

    def test_unset_effort_reverts_to_default(self):
        kwargs = _make_repl_kwargs(None)
        last, _, _ = _repl_reasoning("high", repl_kwargs=kwargs)
        assert last == "default"
        _, _, err = _repl_reasoning("-", repl_kwargs=kwargs, last_effort=last)
        assert not err
        assert kwargs["llm_kwargs"]["reasoning_effort"] == "default"


class TestReasoningDispatch:
    def test_dispatch_updates_context(self):
        ctx = _make_ctx("high")
        result = execute_input(parse_input_line("/reasoning minimal"), ctx)
        assert result.kind == "state_change"
        assert not result.is_error
        assert ctx.loop_kwargs["llm_kwargs"]["reasoning_effort"] == "minimal"
        assert ctx.last_reasoning_effort == "high"

        result = execute_input(parse_input_line("/reasoning -"), ctx)
        assert not result.is_error
        assert ctx.loop_kwargs["llm_kwargs"]["reasoning_effort"] == "high"
        assert ctx.last_reasoning_effort == "minimal"

    def test_dispatch_error_keeps_state(self):
        ctx = _make_ctx("high")
        result = execute_input(parse_input_line("/reasoning bogus"), ctx)
        assert result.is_error
        assert ctx.loop_kwargs["llm_kwargs"]["reasoning_effort"] == "high"
        assert ctx.last_reasoning_effort is None

    def test_profile_switch_clears_revert_state(self):
        ctx = _make_ctx("high")
        ctx.profiles = {"local": {"provider": "lmstudio"}}
        ctx.raw_llm_baseline = {"provider": "openrouter", "model": "org/model"}
        ctx.pre_profile_baseline = dict(ctx.raw_llm_baseline)
        execute_input(parse_input_line("/reasoning low"), ctx)
        assert ctx.last_reasoning_effort == "high"

        resolved = ("local-model", "http://x", None, 8192, {"provider": "lmstudio"})
        with patch("swival.agent.resolve_provider", return_value=resolved):
            result = execute_input(parse_input_line("/profile local"), ctx)
        assert not result.is_error
        assert "reasoning_effort" not in ctx.loop_kwargs["llm_kwargs"]
        assert ctx.last_reasoning_effort is None

        result = execute_input(parse_input_line("/reasoning -"), ctx)
        assert result.is_error
        assert "no previous" in result.text


class TestReasoningCompletion:
    def test_completes_levels(self):
        from prompt_toolkit.document import Document

        completer = SwivalCompleter({})
        doc = Document("/reasoning m", len("/reasoning m"))
        texts = [c.text for c in completer.get_completions(doc, None)]
        assert texts == ["minimal", "medium"]

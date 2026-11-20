"""Alt+Enter steering: the running loop stops before its next model call, and the correction runs next as a normal prompt."""

import types
from unittest.mock import Mock, patch

import pytest

from swival import agent
from swival.input_dispatch import InputContext
from swival.reviewer import ReviewVerdict
from swival.thinking import ThinkingState
from swival.todo import TodoState
from tests.conftest import ScriptedLLM
from tests.conftest import build_loop_kwargs as _loop_kwargs
from tests.conftest import make_message as _msg
from tests.conftest import make_tool_call as _tool_call

_THINK_TOOL = {
    "type": "function",
    "function": {"name": "think", "parameters": {"type": "object", "properties": {}}},
}


class _SteeredLLM(ScriptedLLM):
    """Checks for a stop the way call_llm does, and presses Alt+Enter while the first call is answered.

    With *during_backoff*, Alt+Enter lands inside the call instead, like during a rate-limit wait.
    """

    def __init__(self, responses, pending, during_backoff=False):
        super().__init__(responses)
        self.pending = pending
        self.during_backoff = during_backoff

    def __call__(self, *args, **kwargs):
        if self.during_backoff:
            self.pending["on"] = True
        kwargs["stop_check"]()
        reply = super().__call__(*args, **kwargs)
        self.pending["on"] = True
        return reply


def _run(monkeypatch, responses, messages, loop_kwargs, *, stops=True, **llm):
    """Runs the loop and returns the turns it counted."""
    pending = {"on": False}
    monkeypatch.setattr(agent, "call_llm", _SteeredLLM(responses, pending, **llm))
    turn_state = {}

    def run():
        return agent.run_agent_loop(
            messages,
            [_THINK_TOOL],
            steering_pending=lambda: pending["on"],
            turn_state=turn_state,
            **loop_kwargs,
        )

    if stops:
        with pytest.raises(agent._SteeringStop):
            run()
    else:
        run()
    return turn_state["turns_used"]


def _input_ctx(tmp_path, **fields):
    return InputContext(
        **{
            "messages": [],
            "tools": [],
            "base_dir": str(tmp_path),
            "turn_state": {"max_turns": 5, "turns_used": 0},
            "thinking_state": ThinkingState(),
            "todo_state": TodoState(),
            "snapshot_state": None,
            "file_tracker": None,
            "no_history": True,
            "continue_here": False,
            "verbose": False,
            "loop_kwargs": {"llm_kwargs": {}},
            **fields,
        }
    )


def _call_llm(messages, **kwargs):
    return agent.call_llm(
        "http://localhost:8080/v1",
        "my-model",
        messages,
        100,
        0.5,
        None,
        None,
        None,
        False,
        provider="openrouter",
        api_key="test",
        **kwargs,
    )


def _response(**message):
    message = types.SimpleNamespace(
        content=message.get("content"),
        tool_calls=message.get("tool_calls"),
        role="assistant",
    )
    choice = types.SimpleNamespace(message=message, finish_reason="stop")
    return types.SimpleNamespace(choices=[choice], usage=None)


def test_back_to_back_user_messages_are_sent_as_one():
    image = [
        {"type": "text", "text": "[image]"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
    ]
    author = {"role": "user", "content": "a", "name": "author"}
    editor = {"role": "user", "content": "b", "name": "editor"}
    messages = [
        {"role": "user", "content": "task"},
        {"role": "user", "content": "more"},
        {"role": "assistant", "content": "ok"},
        author,
        editor,
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": image},
        {"role": "user", "content": "leave the tests alone"},
    ]
    with patch("litellm.completion", return_value=_response(content="fine")) as sent:
        _call_llm(messages)
    assert sent.call_args.kwargs["messages"] == [
        {"role": "user", "content": "task\n\nmore"},
        {"role": "assistant", "content": "ok"},
        # Different names would get attributed to the first one.
        author,
        editor,
        {"role": "assistant", "content": "ok"},
        {
            "role": "user",
            "content": image + [{"type": "text", "text": "leave the tests alone"}],
        },
    ]
    assert len(messages) == 8


def test_call_llm_checks_before_sending_and_before_a_resend(monkeypatch):
    import litellm

    def stop():
        raise agent._SteeringStop

    with patch("litellm.completion") as completion, pytest.raises(agent._SteeringStop):
        _call_llm([{"role": "user", "content": "x"}], stop_check=stop)
    completion.assert_not_called()

    # Alt+Enter during the rate-limit backoff: the resend would have made an edit.
    typed = []
    edit = types.SimpleNamespace(
        id="c1",
        type="function",
        function=types.SimpleNamespace(name="edit_file", arguments="{}"),
    )
    completion = Mock(
        side_effect=[
            litellm.RateLimitError("slow down", "openrouter", "my-model"),
            _response(tool_calls=[edit]),
        ]
    )
    monkeypatch.setattr(agent.time, "sleep", lambda _s: typed.append(True))
    with patch("litellm.completion", completion), pytest.raises(agent._SteeringStop):
        _call_llm(
            [{"role": "user", "content": "x"}],
            stop_check=lambda: typed and stop(),
        )
    assert completion.call_count == 1


def test_the_whole_tool_batch_runs_before_the_stop(tmp_path, monkeypatch):
    batch = _msg(
        tool_calls=[
            _tool_call("think", '{"thought": "one"}', tc_id="t1"),
            _tool_call("think", '{"thought": "two"}', tc_id="t2"),
        ]
    )
    messages = [{"role": "user", "content": "go"}]
    assert _run(monkeypatch, [batch], messages, _loop_kwargs(tmp_path)) == 1
    results = messages[-2:]
    assert [m["tool_call_id"] for m in results] == ["t1", "t2"]
    assert not any(m["content"].startswith("error") for m in results)


@pytest.mark.parametrize(
    "reply, kept",
    [
        # Without the stop, the loop would nudge the model into making the edit.
        (_msg(content="I'll edit the tests."), [("assistant", "I'll edit the tests.")]),
        # Duplicate tool calls get thrown away with a repair prompt.
        (
            _msg(
                tool_calls=[
                    _tool_call("think", '{"thought": "x"}', tc_id="t1"),
                    _tool_call("think", '{"thought": "x"}', tc_id="t2"),
                ]
            ),
            [],
        ),
    ],
)
def test_unanswered_nudges_are_dropped_on_a_stop(tmp_path, monkeypatch, reply, kept):
    messages = [{"role": "user", "content": "go"}]
    responses = [reply, _msg(tool_calls=[_tool_call("edit_file", "{}")])]
    assert _run(monkeypatch, responses, messages, _loop_kwargs(tmp_path)) == 1
    assert [(m["role"], m["content"]) for m in messages] == [("user", "go"), *kept]


def test_a_stop_inside_call_llm_reaches_the_loop(tmp_path, monkeypatch):
    messages = [{"role": "user", "content": "go"}]
    turns = _run(monkeypatch, [], messages, _loop_kwargs(tmp_path), during_backoff=True)
    assert turns == 0
    assert messages == [{"role": "user", "content": "go"}]


def test_turn_limit_with_a_correction_waiting_drops_the_last_nudge(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(agent, "TODO_REMINDER_INTERVAL", 1)
    loop_kwargs = _loop_kwargs(tmp_path, max_turns=1)
    loop_kwargs["todo_state"].process({"action": "add", "tasks": ["write docs"]})
    batch = _msg(tool_calls=[_tool_call("think", '{"thought": "x"}')])
    messages = [{"role": "user", "content": "go"}]
    _run(monkeypatch, [batch], messages, loop_kwargs, stops=False)
    assert messages[-1]["role"] == "tool"


@pytest.mark.parametrize(
    "follow_up, waiting_before, kept",
    [
        (False, False, ["request", None, "skill body"]),
        (True, False, []),
        (True, True, []),
    ],
)
def test_invoke_agent_turn_stops_and_drops_unanswered_follow_ups(
    tmp_path, monkeypatch, follow_up, waiting_before, kept
):
    def loop(messages, *_a, **_k):
        # A $skill in the request gets activated before the first model call.
        messages.append({"role": "assistant", "content": None, "tool_calls": []})
        messages.append({"role": "tool", "content": "skill body"})
        raise agent._SteeringStop(replied=False)

    loop = Mock(side_effect=loop)
    monkeypatch.setattr(agent, "run_agent_loop", loop)
    ctx = _input_ctx(tmp_path, steering_pending=lambda: waiting_before)
    turn = agent._invoke_agent_turn("request", ctx, follow_up=follow_up)
    assert turn.stopped and turn.answer is None
    assert [m["content"] for m in ctx.messages] == kept
    assert loop.called is not waiting_before


@pytest.mark.parametrize("before_review", [True, False])
def test_review_gives_way_to_a_correction(monkeypatch, before_review):
    pending = {"on": before_review}
    reviewer = Mock(max_rounds=3)
    reviewer.review.side_effect = lambda *_a: (
        pending.update(on=True) or ReviewVerdict("retry", text="redo it")
    )
    ctx = types.SimpleNamespace(
        reviewer=reviewer,
        verbose=False,
        interactive=True,
        loop_kwargs={},
        steering_pending=lambda: pending["on"],
    )
    retries = []
    monkeypatch.setattr(
        agent,
        "_invoke_agent_turn",
        lambda *_a, **kwargs: (
            retries.append(kwargs) or agent.TurnOutcome(None, stopped=True)
        ),
    )
    turn = agent._review_and_retry(agent.TurnOutcome("the answer"), "task", ctx, None)
    assert turn.answer == "the answer"
    if before_review:
        reviewer.review.assert_not_called()
        assert retries == [] and not turn.stopped
    else:
        # The retry is a follow-up, so it never starts and the answer stays.
        assert retries == [{"follow_up": True}] and turn.stopped


class _Manager:
    def __init__(self, interrupt=False):
        self.interrupt = interrupt
        self.joins = 0

    def cancel_all(self):
        pass

    def shutdown(self):
        self.joins += 1
        if self.interrupt:
            self.interrupt = False
            raise KeyboardInterrupt

    def fresh_copy(self):
        return _Manager()


def _join_ctx(manager):
    return types.SimpleNamespace(
        subagent_manager=manager,
        subagent_holder=None,
        loop_kwargs={},
        pending_joins=[],
        goal_state=None,
        continue_here=False,
    )


def test_interrupted_join_stays_owed_until_it_completes():
    old = _Manager(interrupt=True)
    ctx = _join_ctx(old)
    with pytest.raises(KeyboardInterrupt):
        agent._reset_subagent(ctx)
    # The old workers may still be editing, so the manager stays reachable.
    assert ctx.subagent_manager is old
    assert ctx.pending_joins == [(old, True)]
    agent._finish_pending_joins(ctx)
    assert old.joins == 2
    assert ctx.pending_joins == []
    assert ctx.subagent_manager is not old


def test_handoff_owes_the_join_even_when_ctrl_c_cuts_it_short(monkeypatch):
    manager = _Manager()
    ctx = _join_ctx(manager)
    ctx.goal_state = Mock()

    def interrupted(ctx):
        raise KeyboardInterrupt

    monkeypatch.setattr(agent, "_save_continue_state", interrupted)
    with pytest.raises(KeyboardInterrupt):
        agent._hand_off_to_correction(ctx)
    ctx.goal_state.pause.assert_called_once()
    assert ctx.pending_joins == [(manager, False)]

    agent._finish_pending_joins(ctx)
    assert manager.joins == 1
    # Kept, so results its workers already finished can still be collected.
    assert ctx.subagent_manager is manager


def test_init_stops_before_its_next_pass_and_still_refreshes(monkeypatch, tmp_path):
    pending = {"on": False}

    def loop(*_a, **_k):
        pending["on"] = True
        return "pass one", False

    loop = Mock(side_effect=loop)
    refreshed = Mock(return_value=types.SimpleNamespace(ok=True))
    monkeypatch.setattr(agent, "run_agent_loop", loop)
    monkeypatch.setattr(agent, "refresh_system_instructions", refreshed)
    ctx = _input_ctx(tmp_path, steering_pending=lambda: pending["on"])
    step = agent._run_init_passes("", ctx)
    assert step.stopped and step.text == "pass one"
    assert loop.call_count == 1
    assert len(ctx.messages) == 1
    refreshed.assert_called_once()


def test_simplify_does_not_start_its_apply_pass(monkeypatch):
    import swival.simplify

    monkeypatch.setattr(agent, "_route_input", lambda *a, **k: None)
    monkeypatch.setattr(swival.simplify, "prepare_simplify_prompt", lambda *a: "apply")
    applied = Mock()
    monkeypatch.setattr(agent, "_run_agent_step", applied)
    step = agent._execute_simplify(
        "", types.SimpleNamespace(steering_pending=lambda: True)
    )
    applied.assert_not_called()
    assert agent._step_outcome(step) == "stopped"
    # Also when an earlier /init pass ran out of turns.
    step.exhausted = True
    assert agent._step_outcome(step) == "stopped"


def test_a_correction_runs_before_the_next_due_loop_firing(monkeypatch):
    pending = {"on": False}
    first, second = types.SimpleNamespace(id="a"), types.SimpleNamespace(id="b")
    registry = Mock()
    registry.__len__ = lambda self: 2
    registry.due.return_value = [first, second]
    fired = []

    def fire(reg, ctx):
        fired.append(reg)
        pending["on"] = True
        return False

    monkeypatch.setattr(agent, "_fire_loop_iteration", fire)
    ctx = types.SimpleNamespace(
        loop_registry=registry, steering_pending=lambda: pending["on"]
    )
    agent._fire_due_loops(ctx)
    assert fired == [first]

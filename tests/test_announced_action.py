"""Recovery for replies that announce an action instead of calling a tool."""

import pytest

from swival import agent
from swival.goal import GoalState
from tests.conftest import ScriptedLLM as _ScriptedLLM
from tests.conftest import build_loop_kwargs as _loop_kwargs
from tests.conftest import make_message as _msg
from tests.conftest import make_tool_call as _tool_call

_THINK_TOOL = {
    "type": "function",
    "function": {"name": "think", "parameters": {"type": "object", "properties": {}}},
}

# Observed in sessions where a local model stopped instead of acting,
# plus one sample per remaining phrase.
_ANNOUNCEMENTS = [
    '\n"Return" is invalid; the valid key list includes "Enter". Let me press "Enter".',
    "\nLet me run the probe now.\n\nBut note the harness's win stub lacks "
    "`document`, so `textures.js` might fail. Let me just run.\n\nAlso: the load "
    "list in harness doesn't include geo.js before world.js? It does. Good.\n\n"
    "Note track.js also uses NS.geo? Let's see. run.",
    "I'll open the file next.",
    "I will check the renderer.",
    "Now I need to check the renderer.",
    "I'm going to reload the page.",
    "I’ll inspect the failing test now.",
    "Let’s run the formatter.",
]

_FINAL_ANSWERS = [
    "Done. All 35 tests pass and the start line now sits on a straight.",
    "The fix is in `src/track.js`. Let me know if you want the probe script kept.",
    "Two options remain. If you want, I'll add the regression test as well.",
    "Should I also update the docs?",
    "I'll leave the deployment step to you since it needs credentials.",
    "Earlier I said I'll fix the stride bug.\n\nThat is done: stride is 32 and "
    "the suite is green.",
    "Done. The helper is now:\n\n```js\n// let me explain\nrun();\n```",
    "",
    None,
]


@pytest.mark.parametrize("text", _ANNOUNCEMENTS)
def test_detects_announced_action(text):
    assert agent._announces_pending_action(text)


@pytest.mark.parametrize("text", _FINAL_ANSWERS)
def test_ignores_final_answers_and_offers(text):
    assert not agent._announces_pending_action(text)


def _nudges(messages):
    return [
        m
        for m in messages
        if m.get("role") == "user" and "announced an action" in m.get("content", "")
    ]


def test_nudge_once_per_stretch_without_tools(tmp_path, monkeypatch):
    llm = _ScriptedLLM(
        [
            _msg(content="Let me look."),
            _msg(tool_calls=[_tool_call("think", '{"thought": "x"}')]),
            _msg(content="Let me run it."),
            _msg(content="Finished."),
        ]
    )
    monkeypatch.setattr(agent, "call_llm", llm)
    messages = [{"role": "user", "content": "go"}]
    answer, exhausted = agent.run_agent_loop(
        messages, [_THINK_TOOL], **_loop_kwargs(tmp_path, max_turns=8)
    )
    assert answer == "Finished."
    assert exhausted is False
    assert llm.calls == 4
    nudges = _nudges(messages)
    assert len(nudges) == 2
    assert all(n["_swival_synthetic"] for n in nudges)


def test_second_announcement_is_accepted(tmp_path, monkeypatch):
    llm = _ScriptedLLM(
        [_msg(content="Let me run the probe."), _msg(content="Let me just run it.")]
    )
    monkeypatch.setattr(agent, "call_llm", llm)
    messages = [{"role": "user", "content": "run it"}]
    answer, _ = agent.run_agent_loop(
        messages, [_THINK_TOOL], **_loop_kwargs(tmp_path, max_turns=8)
    )
    assert answer == "Let me just run it."
    assert llm.calls == 2


def test_no_nudge_on_last_turn(tmp_path, monkeypatch):
    llm = _ScriptedLLM([_msg(content="Let me run it.")])
    monkeypatch.setattr(agent, "call_llm", llm)
    messages = [{"role": "user", "content": "go"}]
    answer, exhausted = agent.run_agent_loop(
        messages, [_THINK_TOOL], **_loop_kwargs(tmp_path, max_turns=1)
    )
    assert answer == "Let me run it."
    assert exhausted is False
    assert not _nudges(messages)


def test_nudge_runs_before_goal_continuation(tmp_path, monkeypatch):
    llm = _ScriptedLLM(
        [
            _msg(content="Let me press Enter."),
            _msg(tool_calls=[_tool_call("think", '{"thought": "x"}')]),
            _msg(content="checkpoint"),
            _msg(content="blocked"),
        ]
    )
    monkeypatch.setattr(agent, "call_llm", llm)
    gs = GoalState()
    gs.create("finish the game")
    messages = [{"role": "user", "content": "go"}]
    answer, _ = agent.run_agent_loop(
        messages, [_THINK_TOOL], **_loop_kwargs(tmp_path, gs, max_turns=8)
    )
    assert answer == "blocked"
    assert llm.calls == 4
    assert len(_nudges(messages)) == 1
    contents = [m["content"] for m in messages if m.get("role") == "user"]
    assert sum("[goal continuation]" in c for c in contents) == 1

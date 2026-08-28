"""Agent-loop integration tests for the Tier 1 repair passes.

Validates that the wiring in ``agent.py`` actually:

* repairs truncated tool-call arguments instead of forcing compaction,
* recovers tool calls that the model leaked into the content channel,
* and suppresses identical repeat calls before they dispatch.
"""

from __future__ import annotations

import json
import sys
import types

import pytest


def _make_message(content=None, tool_calls=None, role="assistant", reasoning=None):
    msg = types.SimpleNamespace()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.role = role
    if reasoning is not None:
        msg.reasoning_content = reasoning
    msg.get = lambda key, default=None: getattr(msg, key, default)
    return msg


def _make_tool_call(name, arguments, call_id="tc1"):
    tc = types.SimpleNamespace()
    tc.id = call_id
    tc.type = "function"
    tc.function = types.SimpleNamespace(name=name, arguments=arguments)
    return tc


def _base_args(tmp_path, question="run", **overrides):
    defaults = dict(
        base_url="http://fake",
        model="test-model",
        max_output_tokens=1024,
        temperature=0.55,
        top_p=None,
        seed=None,
        quiet=False,
        max_turns=8,
        base_dir=str(tmp_path),
        no_system_prompt=True,
        no_instructions=True,
        no_skills=True,
        skills_dir=[],
        system_prompt=None,
        question=question,
        repl=False,
        max_context_tokens=None,
        commands=None,
        add_dir=[],
        add_dir_ro=[],
        provider="lmstudio",
        api_key=None,
        color=False,
        no_color=False,
        files="some",
        yolo=False,
        report=None,
        reviewer=None,
        version=False,
        no_read_guard=False,
        no_history=True,
        init_config=False,
        project=False,
        reviewer_mode=False,
        review_prompt=None,
        objective=None,
        verify=None,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _drive_agent(tmp_path, monkeypatch, args, llm_responses):
    """Drive ``agent.main()`` against a scripted LLM response stream."""
    from swival import agent
    from swival import fmt

    fmt.init(color=False)

    iter_responses = iter(llm_responses)
    captured = []

    def fake_call_llm(*call_args, **kwargs):
        messages = call_args[2]
        captured.append([dict(m) if isinstance(m, dict) else m for m in messages])
        return next(iter_responses)

    monkeypatch.setattr(agent, "call_llm", fake_call_llm)
    monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
    monkeypatch.setattr(sys, "argv", ["agent", args.question])
    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)

    agent.main()
    return captured


def test_truncation_repair_avoids_compaction(tmp_path, monkeypatch):
    """A tool call whose JSON args were truncated mid-string is repaired
    without forcing the compaction ladder to fire."""
    from swival import agent
    from unittest.mock import MagicMock

    compact_spy = MagicMock(wraps=agent.compact_messages)
    monkeypatch.setattr(agent, "compact_messages", compact_spy)

    args = _base_args(tmp_path)
    tc = _make_tool_call("think", '{"thought": "let me check', call_id="t1")
    responses = [
        (_make_message(content="", tool_calls=[tc]), "stop", [], 0, (0, 0)),
        (_make_message(content="all done"), "stop", [], 0, (0, 0)),
    ]

    _drive_agent(tmp_path, monkeypatch, args, responses)
    assert compact_spy.call_count == 0


def test_scavenge_recovers_swival_call(tmp_path, monkeypatch):
    """A model that emits a ``<swival:call>`` envelope in content gets the
    call materialized and executed even though ``tool_calls`` is empty."""
    args = _base_args(tmp_path)
    content = '<swival:call id="s1" name="think">{"thought": "scavenged"}</swival:call>'
    responses = [
        (_make_message(content=content, tool_calls=None), "stop", [], 0, (0, 0)),
        (_make_message(content="answer"), "stop", [], 0, (0, 0)),
    ]

    captured = _drive_agent(tmp_path, monkeypatch, args, responses)
    assert len(captured) >= 2
    tool_messages = [
        m
        for m in captured[1]
        if (m.get("role") if isinstance(m, dict) else m.role) == "tool"
    ]
    assert tool_messages, "scavenged call should have produced a tool result"


def test_scavenge_ignores_prose_mentioning_a_tool_name(tmp_path, monkeypatch):
    """A normal answer that *mentions* a tool name in prose must not be
    misread as a tool-call invitation."""
    args = _base_args(tmp_path)
    prose = (
        "The think tool would be called as "
        '{"name": "think", "arguments": {"thought": "x"}}, '
        "but I'm answering directly here."
    )
    responses = [
        (_make_message(content=prose, tool_calls=None), "stop", [], 0, (0, 0)),
    ]
    captured = _drive_agent(tmp_path, monkeypatch, args, responses)
    tool_messages = [
        m
        for m in captured[-1]
        if (m.get("role") if isinstance(m, dict) else m.role) == "tool"
    ]
    assert not tool_messages


def test_storm_breaker_suppresses_third_identical_call(tmp_path, monkeypatch):
    """Three identical successful tool calls in a row trip the storm
    breaker on the third one and the suppression message flows through
    the consecutive-error guardrail without an infinite loop."""
    args = _base_args(tmp_path)
    tc_args = json.dumps({"file_path": "missing.txt"})
    rf1 = _make_tool_call("read_file", tc_args, call_id="r1")
    rf2 = _make_tool_call("read_file", tc_args, call_id="r2")
    rf3 = _make_tool_call("read_file", tc_args, call_id="r3")
    responses = [
        (_make_message(content="", tool_calls=[rf1]), "tool_calls", [], 0, (0, 0)),
        (_make_message(content="", tool_calls=[rf2]), "tool_calls", [], 0, (0, 0)),
        (_make_message(content="", tool_calls=[rf3]), "tool_calls", [], 0, (0, 0)),
        (_make_message(content="done"), "stop", [], 0, (0, 0)),
    ]

    captured = _drive_agent(tmp_path, monkeypatch, args, responses)
    # On the 4th LLM call (after the storm suppression), the latest tool
    # message must carry the storm-guard error string.
    final_history = captured[-1]
    last_tool = [
        m
        for m in final_history
        if (m.get("role") if isinstance(m, dict) else m.role) == "tool"
    ][-1]
    content = (
        last_tool.get("content") if isinstance(last_tool, dict) else last_tool.content
    )
    assert "repeat-loop guard tripped" in content


def test_storm_breaker_off_dispatches_normally(tmp_path, monkeypatch):
    """With the storm breaker disabled, repeat calls all dispatch."""
    args = _base_args(tmp_path, storm_breaker=False)
    tc_args = json.dumps({"file_path": "missing.txt"})

    def rf(i):
        return _make_tool_call("read_file", tc_args, call_id=f"r{i}")

    responses = [
        (_make_message(content="", tool_calls=[rf(1)]), "tool_calls", [], 0, (0, 0)),
        (_make_message(content="", tool_calls=[rf(2)]), "tool_calls", [], 0, (0, 0)),
        (_make_message(content="", tool_calls=[rf(3)]), "tool_calls", [], 0, (0, 0)),
        (_make_message(content="done"), "stop", [], 0, (0, 0)),
    ]
    captured = _drive_agent(tmp_path, monkeypatch, args, responses)
    final_history = captured[-1]
    tool_msgs = [
        m
        for m in final_history
        if (m.get("role") if isinstance(m, dict) else m.role) == "tool"
    ]
    for m in tool_msgs:
        content = m.get("content") if isinstance(m, dict) else m.content
        assert "repeat-loop guard tripped" not in content


def test_explicit_non_idempotent_mcp_repeat_dispatches_once(tmp_path, monkeypatch):
    from swival import mcp_client

    calls = []

    class FakeMcpManager:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def close(self):
            pass

        def list_tools(self):
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "mcp__soak__record_once",
                        "description": "Record one value once.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "key": {"type": "string"},
                                "value": {},
                            },
                            "required": ["key", "value"],
                        },
                    },
                }
            ]

        def get_tool_info(self):
            return {"soak": [("mcp__soak__record_once", "Record one value once.")]}

        def is_non_idempotent_tool(self, name):
            return name == "mcp__soak__record_once"

        def call_tool(self, name, arguments):
            calls.append((name, arguments))
            return "recorded", False

    monkeypatch.setattr(mcp_client, "McpManager", FakeMcpManager)
    args = _base_args(tmp_path, no_mcp=False, network="full")
    args._resolved_mcp_servers = {"soak": {"command": "unused"}}
    first = _make_tool_call(
        "mcp__soak__record_once",
        '{"key":"only-once","value":{"a":1,"b":2}}',
        call_id="m1",
    )
    repeated = _make_tool_call(
        "mcp__soak__record_once",
        '{"value":{"b":2,"a":1},"key":"only-once"}',
        call_id="m2",
    )
    responses = [
        (_make_message(content="", tool_calls=[first]), "tool_calls", [], 0, (0, 0)),
        (
            _make_message(content="", tool_calls=[repeated]),
            "tool_calls",
            [],
            0,
            (0, 0),
        ),
        (_make_message(content="done"), "stop", [], 0, (0, 0)),
    ]

    captured = _drive_agent(tmp_path, monkeypatch, args, responses)

    assert calls == [
        ("mcp__soak__record_once", {"key": "only-once", "value": {"a": 1, "b": 2}})
    ]
    final_tools = [
        message
        for message in captured[-1]
        if (message.get("role") if isinstance(message, dict) else message.role)
        == "tool"
    ]
    final_content = (
        final_tools[-1].get("content")
        if isinstance(final_tools[-1], dict)
        else final_tools[-1].content
    )
    assert "repeat-loop guard tripped" in final_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

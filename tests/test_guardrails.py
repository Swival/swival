"""Tests for tool-call guardrails in agent.py."""

import json
import sys
import types


def _make_message(content=None, tool_calls=None):
    msg = types.SimpleNamespace()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.role = "assistant"
    msg.get = lambda key, default=None: getattr(msg, key, default)
    return msg


def _make_tool_call(name, arguments, call_id):
    tc = types.SimpleNamespace()
    tc.id = call_id
    tc.function = types.SimpleNamespace()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _base_args(tmp_path, **overrides):
    defaults = dict(
        base_url="http://fake",
        model="test-model",
        max_output_tokens=1024,
        temperature=0.55,
        top_p=1.0,
        seed=None,
        quiet=False,
        max_turns=10,
        base_dir=str(tmp_path),
        no_system_prompt=True,
        no_instructions=True,
        no_skills=True,
        skills_dir=[],
        system_prompt=None,
        question="test guardrails",
        repl=False,
        max_context_tokens=None,
        allowed_commands=None,
        allow_dir=[],
        provider="lmstudio",
        api_key=None,
        color=False,
        no_color=False,
        yolo=False,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _guardrail_user_messages(messages):
    out = []
    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role != "user":
            continue
        content = (
            msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        )
        if content.startswith("IMPORTANT:") or content.startswith("STOP:"):
            out.append(content)
    return out


def test_canonical_error_uses_first_line():
    from swival import agent

    assert (
        agent._canonical_error("error: first line\nextra details")
        == "error: first line"
    )
    assert agent._canonical_error("error: one line only") == "error: one line only"


def test_guardrail_escalates_on_repeated_identical_errors(tmp_path, monkeypatch):
    from swival import agent
    from swival import fmt

    snapshots = []
    guardrail_calls = []
    call_count = 0

    def fake_call_llm(*args, **kwargs):
        nonlocal call_count
        snapshots.append(list(args[2]))
        call_count += 1
        if call_count <= 3:
            tc = _make_tool_call(
                "read_file",
                json.dumps({"file_path": "missing.txt"}),
                call_id=f"call_{call_count}",
            )
            return _make_message(tool_calls=[tc]), "tool_calls"
        return _make_message(content="done"), "stop"

    monkeypatch.setattr(agent, "call_llm", fake_call_llm)
    monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
    monkeypatch.setattr(
        fmt,
        "guardrail",
        lambda tool_name, count, error: guardrail_calls.append(
            (tool_name, count, error)
        ),
    )

    args = _base_args(tmp_path)
    monkeypatch.setattr(sys, "argv", ["agent", "test guardrails"])
    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)

    agent.main()

    assert len(snapshots) == 4

    third_call_guardrails = _guardrail_user_messages(snapshots[2])
    assert any(
        "IMPORTANT: You have called `read_file` 2 times" in m
        for m in third_call_guardrails
    )

    fourth_call_guardrails = _guardrail_user_messages(snapshots[3])
    assert any(
        "STOP: You have failed to use `read_file` correctly 3 times in a row" in m
        for m in fourth_call_guardrails
    )

    assert [count for _tool, count, _err in guardrail_calls] == [2, 3]


def test_guardrail_resets_on_different_error(tmp_path, monkeypatch):
    from swival import agent
    from swival import fmt

    snapshots = []
    guardrail_calls = []
    call_count = 0

    def fake_call_llm(*args, **kwargs):
        nonlocal call_count
        snapshots.append(list(args[2]))
        call_count += 1
        if call_count == 1:
            tc = _make_tool_call(
                "read_file", json.dumps({"file_path": "missing_a.txt"}), "call_1"
            )
            return _make_message(tool_calls=[tc]), "tool_calls"
        if call_count == 2:
            tc = _make_tool_call(
                "read_file", json.dumps({"file_path": "missing_b.txt"}), "call_2"
            )
            return _make_message(tool_calls=[tc]), "tool_calls"
        return _make_message(content="done"), "stop"

    monkeypatch.setattr(agent, "call_llm", fake_call_llm)
    monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
    monkeypatch.setattr(
        fmt,
        "guardrail",
        lambda tool_name, count, error: guardrail_calls.append(
            (tool_name, count, error)
        ),
    )

    args = _base_args(tmp_path)
    monkeypatch.setattr(sys, "argv", ["agent", "test guardrails"])
    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)

    agent.main()

    assert len(snapshots) == 3
    assert _guardrail_user_messages(snapshots[2]) == []
    assert guardrail_calls == []


def test_guardrail_resets_on_success(tmp_path, monkeypatch):
    from swival import agent
    from swival import fmt

    snapshots = []
    guardrail_calls = []
    call_count = 0

    def fake_call_llm(*args, **kwargs):
        nonlocal call_count
        snapshots.append(list(args[2]))
        call_count += 1
        if call_count == 1:
            tc = _make_tool_call(
                "read_file", json.dumps({"file_path": "missing.txt"}), "call_1"
            )
            return _make_message(tool_calls=[tc]), "tool_calls"
        if call_count == 2:
            tc = _make_tool_call("read_file", json.dumps({"file_path": "."}), "call_2")
            return _make_message(tool_calls=[tc]), "tool_calls"
        if call_count == 3:
            tc = _make_tool_call(
                "read_file", json.dumps({"file_path": "missing.txt"}), "call_3"
            )
            return _make_message(tool_calls=[tc]), "tool_calls"
        return _make_message(content="done"), "stop"

    monkeypatch.setattr(agent, "call_llm", fake_call_llm)
    monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
    monkeypatch.setattr(
        fmt,
        "guardrail",
        lambda tool_name, count, error: guardrail_calls.append(
            (tool_name, count, error)
        ),
    )

    args = _base_args(tmp_path)
    monkeypatch.setattr(sys, "argv", ["agent", "test guardrails"])
    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)

    agent.main()

    assert len(snapshots) == 4
    assert _guardrail_user_messages(snapshots[3]) == []
    assert guardrail_calls == []


def test_repaired_run_command_error_still_triggers_guardrail(tmp_path, monkeypatch):
    from swival import agent
    from swival import fmt

    snapshots = []
    guardrail_calls = []
    call_count = 0

    def fake_call_llm(*args, **kwargs):
        nonlocal call_count
        snapshots.append(list(args[2]))
        call_count += 1
        if call_count <= 2:
            tc = _make_tool_call(
                "run_command",
                json.dumps({"command": '["not_allowed_cmd_xyz"]'}),
                call_id=f"call_{call_count}",
            )
            return _make_message(tool_calls=[tc]), "tool_calls"
        return _make_message(content="done"), "stop"

    monkeypatch.setattr(agent, "call_llm", fake_call_llm)
    monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
    monkeypatch.setattr(
        fmt,
        "guardrail",
        lambda tool_name, count, error: guardrail_calls.append(
            (tool_name, count, error)
        ),
    )

    args = _base_args(tmp_path, allowed_commands=None)
    monkeypatch.setattr(sys, "argv", ["agent", "test guardrails"])
    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)

    agent.main()

    assert len(snapshots) == 3
    assert any(
        "IMPORTANT: You have called `run_command` 2 times" in m
        for m in _guardrail_user_messages(snapshots[2])
    )
    assert [count for _tool, count, _err in guardrail_calls] == [2]


def test_multiple_tool_interventions_are_combined_into_one_user_message(
    tmp_path, monkeypatch
):
    from swival import agent
    from swival import fmt

    snapshots = []
    guardrail_calls = []
    call_count = 0

    def fake_call_llm(*args, **kwargs):
        nonlocal call_count
        snapshots.append(list(args[2]))
        call_count += 1
        if call_count <= 2:
            tc1 = _make_tool_call(
                "read_file",
                json.dumps({"file_path": "missing.txt"}),
                call_id=f"read_{call_count}",
            )
            tc2 = _make_tool_call(
                "run_command",
                json.dumps({"command": ["not_allowed_cmd_xyz"]}),
                call_id=f"run_{call_count}",
            )
            return _make_message(tool_calls=[tc1, tc2]), "tool_calls"
        return _make_message(content="done"), "stop"

    monkeypatch.setattr(agent, "call_llm", fake_call_llm)
    monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
    monkeypatch.setattr(
        fmt,
        "guardrail",
        lambda tool_name, count, error: guardrail_calls.append(
            (tool_name, count, error)
        ),
    )

    args = _base_args(tmp_path)
    monkeypatch.setattr(sys, "argv", ["agent", "test guardrails"])
    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)

    agent.main()

    assert len(snapshots) == 3
    guardrail_msgs = _guardrail_user_messages(snapshots[2])
    assert len(guardrail_msgs) == 1
    assert "`read_file` 2 times" in guardrail_msgs[0]
    assert "`run_command` 2 times" in guardrail_msgs[0]

    tool_names = {tool for tool, _count, _err in guardrail_calls}
    counts = sorted(count for _tool, count, _err in guardrail_calls)
    assert tool_names == {"read_file", "run_command"}
    assert counts == [2, 2]

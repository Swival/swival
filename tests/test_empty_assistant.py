"""Tests for empty assistant message recovery."""

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
        question="test empty response",
        repl=False,
        max_context_tokens=None,
        allowed_commands=None,
        add_dir=[],
        add_dir_ro=[],
        provider="lmstudio",
        api_key=None,
        color=False,
        no_color=False,
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


# -- _sanitize_assistant_messages --


def test_sanitize_fixes_empty_dict_messages():
    from swival.agent import _sanitize_assistant_messages

    messages = [
        {"role": "system", "content": "hello"},
        {"role": "assistant", "content": None, "tool_calls": None},
        {"role": "user", "content": "world"},
    ]
    assert _sanitize_assistant_messages(messages) is True
    assert messages[1]["content"] == ""


def test_sanitize_fixes_empty_namespace_messages():
    from swival.agent import _sanitize_assistant_messages

    msg = _make_message(content=None, tool_calls=None)
    messages = [msg]
    assert _sanitize_assistant_messages(messages) is True
    assert msg.content == ""


def test_sanitize_leaves_valid_messages_alone():
    from swival.agent import _sanitize_assistant_messages

    messages = [
        {"role": "assistant", "content": "hello", "tool_calls": None},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
    ]
    assert _sanitize_assistant_messages(messages) is False
    assert messages[0]["content"] == "hello"
    assert messages[1]["content"] is None


def test_sanitize_handles_empty_tool_calls_list():
    """An assistant message with content=None and tool_calls=[] is invalid."""
    from swival.agent import _sanitize_assistant_messages

    messages = [{"role": "assistant", "content": None, "tool_calls": []}]
    assert _sanitize_assistant_messages(messages) is True
    assert messages[0]["content"] == ""


# -- Empty response recovery in agent loop --


def test_empty_response_triggers_continuation(tmp_path, monkeypatch):
    """When the LLM returns an empty message, the loop should ask it to continue."""
    from swival import agent
    from swival import fmt

    fmt.init(color=False)

    call_count = 0

    def fake_call_llm(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: empty response
            return _make_message(content=None, tool_calls=None), "stop"
        # Second call: real answer
        return _make_message(content="the answer"), "stop"

    monkeypatch.setattr(agent, "call_llm", fake_call_llm)
    monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

    args = _base_args(tmp_path)
    monkeypatch.setattr(sys, "argv", ["agent", "test empty response"])
    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)

    agent.main()

    assert call_count == 2, "Should have retried after empty response"


def test_empty_response_message_has_content_in_history(tmp_path, monkeypatch):
    """The empty assistant message should be fixed to have content='' in history."""
    from swival import agent
    from swival import fmt

    fmt.init(color=False)

    captured_messages = []

    def fake_call_llm(*args, **kwargs):
        messages = args[2]
        captured_messages.append(list(messages))
        if len(captured_messages) == 1:
            return _make_message(content=None, tool_calls=None), "stop"
        return _make_message(content="done"), "stop"

    monkeypatch.setattr(agent, "call_llm", fake_call_llm)
    monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

    args = _base_args(tmp_path)
    monkeypatch.setattr(sys, "argv", ["agent", "test"])
    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)

    agent.main()

    # Second call's messages should contain the fixed assistant message
    # and the continuation prompt
    second_call_msgs = captured_messages[1]
    assistant_msgs = [
        m
        for m in second_call_msgs
        if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None))
        == "assistant"
    ]
    # The empty message should have been fixed to content=""
    for am in assistant_msgs:
        content = (
            am.get("content") if isinstance(am, dict) else getattr(am, "content", None)
        )
        assert content is not None, "Assistant message should have content set"

    # Should have a continuation user message
    user_msgs = [
        m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        for m in second_call_msgs
        if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None))
        == "user"
    ]
    assert any("empty" in u.lower() for u in user_msgs), (
        "Should have a continuation prompt mentioning empty response"
    )


# -- call_llm sanitization retry --


def test_call_llm_retries_after_sanitizing_empty_assistant(monkeypatch):
    """call_llm should sanitize and retry when provider rejects empty assistant messages."""
    from swival import agent
    from swival import fmt

    fmt.init(color=False)

    import litellm

    call_count = 0

    def fake_completion(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise litellm.BadRequestError(
                message=(
                    'OpenrouterException - {"error":{"message":"Provider returned error",'
                    '"code":400,"metadata":{"raw":"{\\"message\\":\\"Assistant message '
                    "must have either content or tool_calls, but not "
                    'none.\\"}"}}}'
                ),
                model="test",
                llm_provider="openrouter",
            )
        # Successful retry
        choice = types.SimpleNamespace()
        choice.message = _make_message(content="ok")
        choice.finish_reason = "stop"
        resp = types.SimpleNamespace()
        resp.choices = [choice]
        return resp

    monkeypatch.setattr(litellm, "completion", fake_completion)

    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "assistant", "content": None, "tool_calls": None},
        {"role": "user", "content": "hello"},
    ]

    msg, finish = agent.call_llm(
        "http://fake",
        "test-model",
        messages,
        1024,
        None,
        None,
        None,
        None,
        False,
        provider="openrouter",
        api_key="fake",
    )

    assert call_count == 2
    assert msg.content == "ok"
    # The empty assistant message should have been fixed in place
    assert messages[1]["content"] == ""

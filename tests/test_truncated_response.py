"""Tests for truncated provider responses (issue #22).

When an LLM server returns a partial response the bad message must never
enter the history. Genuine output truncation (finish_reason='length' with
unparseable tool-call arguments) is treated as context overflow so the
compaction ladder runs — trimming the prompt frees output-token budget.

Malformed tool-call arguments under a normal finish_reason are a different
case: the context window is fine, the model just emitted invalid JSON. That
must be recovered with a re-prompt, not by compacting away valid context.
"""

import sys
import types
from unittest.mock import MagicMock


def _make_message(content=None, tool_calls=None, role="assistant"):
    msg = types.SimpleNamespace()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.role = role
    msg.get = lambda key, default=None: getattr(msg, key, default)
    return msg


def _make_tool_call(name="think", arguments='{"thought": "ok"}', call_id="tc1"):
    tc = types.SimpleNamespace()
    tc.id = call_id
    tc.type = "function"
    tc.function = types.SimpleNamespace()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _base_args(tmp_path, question="hi", **overrides):
    defaults = dict(
        base_url="http://fake",
        model="test-model",
        max_output_tokens=1024,
        temperature=0.55,
        top_p=None,
        seed=None,
        quiet=False,
        max_turns=10,
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


# ---------------------------------------------------------------------------
# Unit tests for the validators themselves
# ---------------------------------------------------------------------------


def test_normalize_empty_args():
    from swival.agent import _normalize_tool_call_args

    assert _normalize_tool_call_args("") == "{}"
    assert _normalize_tool_call_args(None) == "{}"
    assert _normalize_tool_call_args('{"x": 1}') == '{"x": 1}'


def test_has_malformed_tool_args_with_object_tc():
    from swival.agent import _has_malformed_tool_args

    msg = _make_message(tool_calls=[_make_tool_call(arguments='{"a": 1}')])
    assert _has_malformed_tool_args(msg) is False

    msg = _make_message(tool_calls=[_make_tool_call(arguments="{")])
    assert _has_malformed_tool_args(msg) is True


def test_has_malformed_tool_args_with_dict_tc():
    from swival.agent import _has_malformed_tool_args

    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc1",
                "type": "function",
                "function": {"name": "think", "arguments": "{"},
            }
        ],
    }
    assert _has_malformed_tool_args(msg) is True


def test_has_malformed_tool_args_empty_string_is_valid():
    """Empty arguments string means "no args" — not malformed."""
    from swival.agent import _has_malformed_tool_args

    msg = _make_message(tool_calls=[_make_tool_call(arguments="")])
    assert _has_malformed_tool_args(msg) is False


def test_has_malformed_tool_args_no_tool_calls():
    from swival.agent import _has_malformed_tool_args

    assert _has_malformed_tool_args(_make_message(content="hi")) is False


def test_duplicate_tool_call_names_canonicalizes_arguments():
    from swival.agent import _duplicate_tool_call_names

    msg = _make_message(
        tool_calls=[
            _make_tool_call(
                name="write_file",
                arguments='{"path":"x","content":"y"}',
                call_id="a",
            ),
            _make_tool_call(
                name="write_file",
                arguments='{"content": "y", "path": "x"}',
                call_id="b",
            ),
        ]
    )
    assert _duplicate_tool_call_names(msg) == ["write_file"]


def test_duplicate_tool_call_names_allows_distinct_arguments():
    from swival.agent import _duplicate_tool_call_names

    msg = _make_message(
        tool_calls=[
            _make_tool_call(arguments='{"thought":"one"}', call_id="a"),
            _make_tool_call(arguments='{"thought":"two"}', call_id="b"),
        ]
    )
    assert _duplicate_tool_call_names(msg) == []


def test_classify_text_only_length_returns_none():
    """Text-only truncation must not be classified — the existing nudge handles it."""
    from swival.agent import _classify_tool_call_truncation

    msg = _make_message(content="partial...", tool_calls=None)
    assert _classify_tool_call_truncation(msg, "length") is None


def test_classify_length_with_parseable_tool_calls():
    """Parseable tool-call args + finish_reason='length' are allowed through.

    The structural-repair pass upstream may have just patched a partial
    JSON tail; rejecting a parseable call would discard recoverable work.
    """
    from swival.agent import _classify_tool_call_truncation

    msg = _make_message(tool_calls=[_make_tool_call(arguments='{"a":1}')])
    assert _classify_tool_call_truncation(msg, "length") is None


def test_classify_length_with_malformed_tool_calls():
    """Malformed args under finish_reason='length' still trip the discard path."""
    from swival.agent import _classify_tool_call_truncation

    msg = _make_message(tool_calls=[_make_tool_call(arguments="{")])
    assert _classify_tool_call_truncation(msg, "length") == "length"


def test_classify_malformed_with_stop():
    from swival.agent import _classify_tool_call_truncation

    msg = _make_message(tool_calls=[_make_tool_call(arguments="{")])
    assert _classify_tool_call_truncation(msg, "stop") == "malformed_args"


# ---------------------------------------------------------------------------
# Agent-loop integration: text-only truncation nudge is preserved
# ---------------------------------------------------------------------------


def test_text_only_length_keeps_nudge_path(tmp_path, monkeypatch):
    """Regression guard for the existing nudge at agent.py:7461.

    finish_reason='length' with no tool_calls must keep the partial
    message in history and inject the continuation user prompt.
    """
    from swival import agent
    from swival import fmt

    fmt.init(color=False)

    captured = []

    def fake_call_llm(*args, **kwargs):
        messages = args[2]
        captured.append([dict(m) if isinstance(m, dict) else m for m in messages])
        if len(captured) == 1:
            return (
                _make_message(content="partial answer cut", tool_calls=None),
                "length",
                [],
                0,
                (0, 0),
            )
        return _make_message(content="final"), "stop", [], 0, (0, 0)

    monkeypatch.setattr(agent, "call_llm", fake_call_llm)
    monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

    args = _base_args(tmp_path)
    monkeypatch.setattr(sys, "argv", ["agent", "q"])
    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)

    agent.main()

    assert len(captured) == 2, "should have called LLM twice"
    second_msgs = captured[1]
    # Partial assistant message must be present
    assistants = [m for m in second_msgs if m.get("role") == "assistant"]
    assert any("partial answer cut" in (m.get("content") or "") for m in assistants)
    # Continuation nudge must be present
    users = [m.get("content", "") for m in second_msgs if m.get("role") == "user"]
    assert any("cut off" in u.lower() for u in users)


# ---------------------------------------------------------------------------
# Agent-loop integration: tool-call truncation is discarded + compacted
# ---------------------------------------------------------------------------


def _run_with_truncation(
    tmp_path, monkeypatch, first_finish_reason, first_args, second_content="done"
):
    """Drive the agent loop with one truncated tool-call response then a clean one.

    Returns (captured_messages_per_call, compact_calls).
    """
    from swival import agent
    from swival import fmt

    fmt.init(color=False)

    captured = []
    compact_spy = MagicMock(wraps=agent.compact_messages)
    monkeypatch.setattr(agent, "compact_messages", compact_spy)

    def fake_call_llm(*args, **kwargs):
        messages = args[2]
        captured.append([dict(m) if isinstance(m, dict) else m for m in messages])
        if len(captured) == 1:
            return (
                _make_message(
                    content="",
                    tool_calls=[_make_tool_call(arguments=first_args)],
                ),
                first_finish_reason,
                [],
                0,
                (0, 0),
            )
        return _make_message(content=second_content), "stop", [], 0, (0, 0)

    monkeypatch.setattr(agent, "call_llm", fake_call_llm)
    monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

    args = _base_args(tmp_path)
    monkeypatch.setattr(sys, "argv", ["agent", "q"])
    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)

    agent.main()

    return captured, compact_spy


def test_length_with_tool_calls_is_discarded(tmp_path, monkeypatch):
    """Truncated tool_call response (finish_reason=length) must not enter history."""
    captured, compact_spy = _run_with_truncation(
        tmp_path, monkeypatch, "length", "{bad"
    )

    assert len(captured) >= 2, "should have called LLM at least twice"
    # The second call's messages must NOT contain an assistant message
    # with the malformed `{bad` arguments.
    for m in captured[1]:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") if isinstance(tc, dict) else tc.function
            raw = fn["arguments"] if isinstance(fn, dict) else fn.arguments
            assert raw != "{bad", "truncated tool_call leaked into history"
    # Compaction must have fired at least once.
    assert compact_spy.call_count >= 1


def test_malformed_args_with_stop_is_reprompted(tmp_path, monkeypatch):
    """Malformed args under a normal finish_reason re-prompt, never compact.

    The context window is irrelevant here, so the bad call is discarded and a
    corrective nudge is injected instead of running the compaction ladder
    (which would throw away valid context on a large conversation).
    """
    captured, compact_spy = _run_with_truncation(tmp_path, monkeypatch, "stop", "{bad")

    # The malformed call must not have leaked into history.
    for m in captured[1]:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") if isinstance(tc, dict) else tc.function
            raw = fn["arguments"] if isinstance(fn, dict) else fn.arguments
            assert raw != "{bad"
    # No compaction: malformed args are not a context-window problem.
    assert compact_spy.call_count == 0
    # A corrective re-prompt must have been injected.
    users = [m.get("content", "") for m in captured[1] if m.get("role") == "user"]
    assert any("malformed JSON arguments" in u for u in users)


def test_malformed_args_repeated_gives_up(tmp_path, monkeypatch):
    """Two consecutive malformed responses give up rather than loop forever."""
    import pytest

    from swival import agent
    from swival import fmt

    fmt.init(color=False)

    calls = []

    def fake_call_llm(*args, **kwargs):
        calls.append(1)
        return (
            _make_message(content="", tool_calls=[_make_tool_call(arguments="{bad")]),
            "stop",
            [],
            0,
            (0, 0),
        )

    monkeypatch.setattr(agent, "call_llm", fake_call_llm)
    monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

    args = _base_args(tmp_path)
    monkeypatch.setattr(sys, "argv", ["agent", "q"])
    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)

    with pytest.raises(SystemExit) as exc:
        agent.main()
    assert exc.value.code == 1
    # First malformed → re-prompt, second malformed → give up. No endless loop.
    assert len(calls) == 2


def test_duplicate_calls_are_discarded_and_reprompted(tmp_path, monkeypatch):
    from swival import agent
    from swival import fmt

    fmt.init(color=False)
    captured = []

    def fake_call_llm(*args, **kwargs):
        captured.append([dict(m) if isinstance(m, dict) else m for m in args[2]])
        if len(captured) == 1:
            return (
                _make_message(
                    content="",
                    tool_calls=[
                        _make_tool_call(call_id="a"),
                        _make_tool_call(call_id="b"),
                    ],
                ),
                "tool_calls",
                [],
                0,
                (0, 0),
            )
        return _make_message(content="done"), "stop", [], 0, (0, 0)

    monkeypatch.setattr(agent, "call_llm", fake_call_llm)
    monkeypatch.setattr(agent, "discover_model", lambda *args: ("test-model", None))
    args = _base_args(tmp_path)
    monkeypatch.setattr(sys, "argv", ["agent", "q"])
    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)

    agent.main()

    assert len(captured) == 2
    assert not any(message.get("role") == "assistant" for message in captured[1])
    users = [
        message.get("content", "")
        for message in captured[1]
        if message.get("role") == "user"
    ]
    assert any("repeated an identical tool call" in content for content in users)


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------


class _FakeCache:
    """Minimal stand-in for LLMCache that records get/put calls."""

    def __init__(self, prime=None):
        self.store = dict(prime or {})
        self.put_calls = []
        self.get_calls = []
        self.delete_calls = []

    def get(self, kwargs):
        self.get_calls.append(kwargs)
        key = self._key(kwargs)
        return self.store.get(key)

    def put(self, kwargs, msg_dict, finish_reason):
        self.put_calls.append((kwargs, msg_dict, finish_reason))
        self.store[self._key(kwargs)] = (msg_dict, finish_reason)

    def delete(self, kwargs):
        self.delete_calls.append(kwargs)
        self.store.pop(self._key(kwargs), None)

    @staticmethod
    def _key(kwargs):
        return repr(sorted(kwargs.items(), key=lambda kv: kv[0]))


def _make_choice(content=None, tool_calls=None, finish_reason="stop"):
    choice = types.SimpleNamespace()
    choice.message = _make_message(content=content, tool_calls=tool_calls)
    choice.finish_reason = finish_reason
    return choice


def _call_llm_with_fake_cache(monkeypatch, fake_cache, choice):
    """Drive call_llm() with a stubbed litellm.completion that returns `choice`."""
    import litellm
    from swival import agent
    from swival import fmt

    fmt.init(color=False)

    response = types.SimpleNamespace(choices=[choice])
    monkeypatch.setattr(litellm, "completion", lambda **kw: response)
    # Bypass the prompt_cache supports() check noise.
    monkeypatch.setattr(
        "litellm.utils.supports_prompt_caching", lambda model: False, raising=False
    )

    return agent.call_llm(
        "http://fake",
        "test-model",
        [{"role": "user", "content": "hi"}],
        1024,
        0.0,
        None,
        None,
        None,
        False,
        provider="lmstudio",
        cache=fake_cache,
        prompt_cache=False,
    )


def test_cache_skips_store_on_length(tmp_path, monkeypatch):
    """A response with finish_reason='length' must not be cached, even if well-formed."""
    fake = _FakeCache()
    choice = _make_choice(content="partial", finish_reason="length")
    _call_llm_with_fake_cache(monkeypatch, fake, choice)
    assert fake.put_calls == [], "length responses must never be cached"


def test_cache_skips_store_on_malformed_args(tmp_path, monkeypatch):
    """finish_reason=stop with malformed tool_call arguments must not be cached."""
    fake = _FakeCache()
    choice = _make_choice(
        tool_calls=[_make_tool_call(arguments="{")], finish_reason="stop"
    )
    _call_llm_with_fake_cache(monkeypatch, fake, choice)
    assert fake.put_calls == []


def test_cache_stores_clean_response(tmp_path, monkeypatch):
    """Sanity: a normal stop response with parseable args is cached."""
    fake = _FakeCache()
    choice = _make_choice(content="hi there", finish_reason="stop")
    _call_llm_with_fake_cache(monkeypatch, fake, choice)
    assert len(fake.put_calls) == 1


def test_cache_hit_poisoned_falls_through(tmp_path, monkeypatch):
    """A preexisting poisoned cache row must be treated as a miss."""
    # Prime the cache with a malformed tool_call entry.
    poisoned_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc1",
                "type": "function",
                "function": {"name": "think", "arguments": "{"},
            }
        ],
    }
    fake = _FakeCache()
    # Manually seed without using put() (so we don't go through the validator).
    fake.store[_FakeCache._key({"x": 1})] = (poisoned_msg, "stop")

    # Make `get` always return the poisoned entry regardless of kwargs.
    def always_poisoned(kwargs):
        fake.get_calls.append(kwargs)
        return (poisoned_msg, "stop")

    fake.get = always_poisoned

    clean_choice = _make_choice(content="fresh answer", finish_reason="stop")
    msg, finish_reason, _, _, _ = _call_llm_with_fake_cache(
        monkeypatch, fake, clean_choice
    )

    # The live call must have run and returned the clean response.
    assert msg.content == "fresh answer"
    assert finish_reason == "stop"
    # Eviction was attempted via delete (we expose it on _FakeCache).
    assert len(fake.delete_calls) == 1

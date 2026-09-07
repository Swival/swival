"""Automatic OpenCode routing headers and conversation lifetimes."""

from unittest.mock import patch
from uuid import UUID

import pytest

from swival import Session, agent


def _response():
    from litellm import ModelResponse

    return ModelResponse(
        choices=[
            {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ]
    )


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://opencode.ai/zen/go/v1", True),
        ("https://opencode.ai/zen/v1", True),
        ("https://OPENCODE.AI:443/zen/go/v1/", True),
        ("https://opencode.ai.example.org/v1", False),
        ("https://example.org/opencode.ai/v1", False),
        ("https://opencode.ai@example.org/v1", False),
        ("http://localhost:8080/v1", False),
    ],
)
def test_headers_only_for_opencode_host(url, expected):
    with patch("litellm.completion", return_value=_response()) as completion:
        agent.call_llm(
            url,
            "model",
            [],
            100,
            None,
            None,
            None,
            None,
            False,
            provider="generic",
            api_key="test",
            user_agent="custom-client",
            session_id="b2bf570f-d242-4e96-84d5-df4b5425a976",
        )
    headers = completion.call_args.kwargs["extra_headers"]
    assert headers["User-Agent"] == "custom-client"
    if expected:
        assert headers["x-opencode-client"] == "swival"
        assert headers["x-opencode-session"] == "b2bf570f-d242-4e96-84d5-df4b5425a976"
    else:
        assert headers == {"User-Agent": "custom-client"}


@pytest.mark.parametrize("reset", ["reset", "/clear", "/new", "run"])
def test_conversation_id_lifetime(tmp_path, monkeypatch, reset):
    monkeypatch.setattr(agent, "discover_generic_context_length", lambda *a: None)
    session = Session(
        provider="generic",
        model="model",
        api_key="test",
        base_url="https://opencode.ai/zen/go/v1",
        base_dir=str(tmp_path),
        history=False,
    )
    with patch("litellm.completion", return_value=_response()) as completion:
        session.ask("first")
        session.ask("second")
        if reset == "reset":
            session.reset()
            session.ask("third")
        elif reset == "run":
            session.run("independent")
        else:
            session.ask(reset, parse_commands=True)
            session.ask("third")
        ids = [
            c.kwargs["extra_headers"]["x-opencode-session"]
            for c in completion.call_args_list
        ]
        assert len(ids) == 3
        assert ids[0] == ids[1]
        assert ids[2] != ids[0]
        for value in ids:
            assert str(UUID(value)) == value
        # A separately constructed session must not reuse the previous ID.
        other = Session(
            provider="generic",
            model="model",
            api_key="test",
            base_url="https://opencode.ai/zen/go/v1",
            base_dir=str(tmp_path),
            history=False,
        )
        other.ask("separate")
        assert (
            completion.call_args.kwargs["extra_headers"]["x-opencode-session"]
            not in ids
        )


def test_summary_forwards_session_id():
    with patch("litellm.completion", return_value=_response()) as completion:
        result = agent._call_summarize_llm(
            "conversation to summarize",
            "summarize",
            agent.call_llm,
            "model",
            "https://opencode.ai/zen/go/v1",
            "test",
            None,
            None,
            "generic",
            provider_kwargs=agent._provider_extra_kwargs(
                {"session_id": "conversation"}
            ),
        )
    assert result == "ok"
    assert (
        completion.call_args.kwargs["extra_headers"]["x-opencode-session"]
        == "conversation"
    )


def test_retry_preserves_generated_id(monkeypatch):
    from litellm import RateLimitError

    monkeypatch.setattr(agent.time, "sleep", lambda _: None)
    with patch(
        "litellm.completion",
        side_effect=[
            RateLimitError("retry", llm_provider="openai", model="model"),
            _response(),
        ],
    ) as completion:
        agent.call_llm(
            "https://opencode.ai/zen/go/v1",
            "model",
            [],
            100,
            None,
            None,
            None,
            None,
            False,
            provider="generic",
            api_key="test",
        )
    headers = [c.kwargs["extra_headers"] for c in completion.call_args_list]
    assert len(headers) == 2
    assert headers[0] == headers[1]
    UUID(headers[0]["x-opencode-session"])


def test_subagents_have_independent_ids(monkeypatch):
    import threading
    from swival.subagent import (
        SubagentHandle,
        _CompositeCancelFlag,
        _subagent_thread_fn,
    )

    ids = []

    def run_loop(messages, tools, **kwargs):
        ids.append(kwargs["llm_kwargs"]["session_id"])
        return "ok", False

    monkeypatch.setattr(agent, "run_agent_loop", run_loop)
    template = {"llm_kwargs": {"provider": "generic", "session_id": "parent"}}
    for index in range(2):
        handle = SubagentHandle(id=str(index), task="task")
        _subagent_thread_fn(
            handle,
            template,
            [],
            "task",
            10,
            None,
            "system",
            _CompositeCancelFlag(None, threading.Event()),
            threading.Semaphore(1),
        )
        assert handle.result == "ok"
    assert len(set(ids)) == 2
    assert "parent" not in ids
    assert template["llm_kwargs"]["session_id"] == "parent"

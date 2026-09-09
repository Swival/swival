"""Tests for model-not-found recovery in call_llm."""

import types
from unittest.mock import MagicMock, patch

import pytest

from swival.agent import (
    call_llm,
    AgentError,
    _MODEL_AUTOFIX,
    _format_model_not_found,
)


def _make_response(content="hello"):
    msg = types.SimpleNamespace(content=content, tool_calls=None, role="assistant")
    choice = types.SimpleNamespace(message=msg, finish_reason="stop")
    return types.SimpleNamespace(choices=[choice])


def _not_found_exc():
    import litellm

    return litellm.NotFoundError(
        message="Model 'sQwopus3.6-27B-Coder-4bit.mlx' not found",
        model="sQwopus3.6-27B-Coder-4bit.mlx",
        llm_provider="openai",
    )


def _call(
    model_id="sQwopus3.6-27B-Coder-4bit.mlx", base_url="http://localhost:8080/v1"
):
    return call_llm(
        base_url,
        model_id,
        [{"role": "user", "content": "hi"}],
        100,
        0.5,
        1.0,
        None,
        None,
        False,
        provider="generic",
        api_key="test",
    )


@pytest.fixture(autouse=True)
def _clear_autofix():
    _MODEL_AUTOFIX.clear()
    yield
    _MODEL_AUTOFIX.clear()


class TestFormatModelNotFound:
    def test_lists_models_and_suggests_closest(self):
        msg = _format_model_not_found(
            "sQwopus3.6-27B-Coder-4bit.mlx",
            ["Qwopus3.6-27B-Coder-4bit.mlx", "Qwopus3.6-27B-Coder-8bit.mlx"],
        )
        assert "Available models:" in msg
        assert "  • Qwopus3.6-27B-Coder-4bit.mlx" in msg
        assert "  • Qwopus3.6-27B-Coder-8bit.mlx" in msg
        assert "Did you mean 'Qwopus3.6-27B-Coder-4bit.mlx'?" in msg

    def test_large_list_shows_closest_matches_only(self):
        available = [f"unrelated-model-{i}" for i in range(30)] + ["target-model"]
        msg = _format_model_not_found("target-modell", available)
        assert "The server reports 31 models" in msg
        assert "target-model" in msg
        assert msg.count("•") <= 20

    def test_no_suggestion_when_nothing_close(self):
        msg = _format_model_not_found("zzz", ["alpha", "beta"])
        assert "Did you mean" not in msg


class TestCallLlmModelNotFound:
    def test_substitution_does_not_change_another_servers_model(self):
        with (
            patch(
                "litellm.completion",
                side_effect=[_not_found_exc(), _make_response(), _make_response()],
            ) as completion,
            patch("swival.agent._list_server_models", return_value=["replacement"]),
        ):
            _call("requested")
            _call("requested", base_url="http://localhost:9090/v1")
        assert [call.kwargs["model"] for call in completion.call_args_list] == [
            "openai/requested",
            "openai/replacement",
            "openai/requested",
        ]

    def test_updated_substitution_is_used_on_next_call(self):
        with (
            patch(
                "litellm.completion",
                side_effect=[
                    _not_found_exc(),
                    _make_response(),
                    _not_found_exc(),
                    _make_response(),
                    _make_response(),
                ],
            ) as completion,
            patch(
                "swival.agent._list_server_models", side_effect=[["first"], ["second"]]
            ),
        ):
            _call("requested")
            _call("requested")
            _call("requested")
        assert [call.kwargs["model"] for call in completion.call_args_list] == [
            "openai/requested",
            "openai/first",
            "openai/first",
            "openai/second",
            "openai/second",
        ]

    def test_multiple_models_raises_formatted_error(self):
        exc = _not_found_exc()
        available = [
            "Qwopus3.6-27B-Coder-4bit.mlx",
            "Qwopus3.6-27B-Coder-8bit.mlx",
        ]
        with (
            patch("litellm.completion", side_effect=exc),
            patch("swival.agent._list_server_models", return_value=available),
        ):
            with pytest.raises(AgentError) as ei:
                _call()
        msg = str(ei.value)
        assert "was not found on the server" in msg
        assert "Qwopus3.6-27B-Coder-8bit.mlx" in msg
        assert "Did you mean 'Qwopus3.6-27B-Coder-4bit.mlx'?" in msg

    def test_single_model_auto_picked(self):
        exc = _not_found_exc()
        resp = _make_response()
        mock = MagicMock(side_effect=[exc, resp])
        with (
            patch("litellm.completion", mock),
            patch(
                "swival.agent._list_server_models",
                return_value=["Qwopus3.6-27B-Coder-4bit.mlx"],
            ),
        ):
            result = _call()
        assert result[0].content == "hello"
        assert mock.call_count == 2
        retry_model = mock.call_args_list[1].kwargs["model"]
        assert retry_model == "openai/Qwopus3.6-27B-Coder-4bit.mlx"

    def test_substitution_remembered_for_next_call(self):
        exc = _not_found_exc()
        mock = MagicMock(side_effect=[exc, _make_response(), _make_response()])
        with (
            patch("litellm.completion", mock),
            patch(
                "swival.agent._list_server_models",
                return_value=["Qwopus3.6-27B-Coder-4bit.mlx"],
            ),
        ):
            _call()
            _call()
        assert mock.call_count == 3
        third_model = mock.call_args_list[2].kwargs["model"]
        assert third_model == "openai/Qwopus3.6-27B-Coder-4bit.mlx"

    def test_model_actually_served_falls_through(self):
        exc = _not_found_exc()
        with (
            patch("litellm.completion", side_effect=exc),
            patch(
                "swival.agent._list_server_models",
                return_value=["sQwopus3.6-27B-Coder-4bit.mlx"],
            ),
        ):
            with pytest.raises(AgentError, match="LLM call failed"):
                _call()

    def test_list_unavailable_falls_through(self):
        exc = _not_found_exc()
        with (
            patch("litellm.completion", side_effect=exc),
            patch("swival.agent._list_server_models", return_value=[]),
        ):
            with pytest.raises(AgentError, match="LLM call failed"):
                _call()

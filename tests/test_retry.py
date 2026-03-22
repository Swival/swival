"""Tests for transient-error retry logic in call_llm / _completion_with_retry."""

import types
from unittest.mock import patch, MagicMock

import pytest

from swival.agent import (
    call_llm,
    _completion_with_retry,
    _is_transient,
    AgentError,
    ContextOverflowError,
)


def _make_response(content="hello"):
    msg = types.SimpleNamespace(content=content, tool_calls=None, role="assistant")
    choice = types.SimpleNamespace(message=msg, finish_reason="stop")
    return types.SimpleNamespace(choices=[choice])


class TestIsTransient:
    def test_api_connection_error(self):
        import litellm

        exc = litellm.APIConnectionError(
            message="Connection reset by peer", llm_provider="openai", model="x"
        )
        assert _is_transient(exc) is True

    def test_timeout(self):
        import litellm

        exc = litellm.Timeout(message="timed out", model="x", llm_provider="openai")
        assert _is_transient(exc) is True

    def test_rate_limit(self):
        import litellm

        exc = litellm.RateLimitError(message="429", llm_provider="openai", model="x")
        assert _is_transient(exc) is True

    def test_internal_server_error(self):
        import litellm

        exc = litellm.InternalServerError(
            message="500", llm_provider="openai", model="x"
        )
        assert _is_transient(exc) is True

    def test_bad_request_not_transient(self):
        import litellm

        exc = litellm.BadRequestError(message="bad", llm_provider="openai", model="x")
        assert _is_transient(exc) is False

    def test_auth_error_not_transient(self):
        import litellm

        exc = litellm.AuthenticationError(
            message="unauthorized", llm_provider="openai", model="x"
        )
        assert _is_transient(exc) is False

    def test_generic_api_error_500(self):
        import litellm

        exc = litellm.APIError(
            status_code=500,
            message="internal server error",
            llm_provider="openai",
            model="x",
        )
        assert _is_transient(exc) is True

    def test_generic_api_error_400(self):
        import litellm

        exc = litellm.APIError(
            status_code=400, message="bad request", llm_provider="openai", model="x"
        )
        assert _is_transient(exc) is False

    def test_string_pattern_connection_reset(self):
        exc = OSError("[Errno 54] Connection reset by peer")
        assert _is_transient(exc) is True

    def test_unrelated_error_not_transient(self):
        exc = ValueError("something unrelated")
        assert _is_transient(exc) is False


class TestCompletionWithRetry:
    def test_succeeds_first_try(self):
        resp = _make_response()
        with patch("litellm.completion", return_value=resp):
            result, retries = _completion_with_retry(
                {"model": "x", "messages": []}, max_retries=5, verbose=False
            )
        assert result is resp
        assert retries == 0

    def test_succeeds_after_transient_errors(self):
        import litellm

        resp = _make_response()
        exc = litellm.APIConnectionError(
            message="Connection reset", llm_provider="openai", model="x"
        )
        mock = MagicMock(side_effect=[exc, exc, resp])
        with patch("litellm.completion", mock), patch("time.sleep"):
            result, retries = _completion_with_retry(
                {"model": "x", "messages": []}, max_retries=5, verbose=False
            )
        assert result is resp
        assert retries == 2
        assert mock.call_count == 3

    def test_exhausts_retries(self):
        import litellm

        exc = litellm.APIConnectionError(
            message="Connection reset", llm_provider="openai", model="x"
        )
        mock = MagicMock(side_effect=[exc] * 3)
        with patch("litellm.completion", mock), patch("time.sleep"):
            with pytest.raises(litellm.APIConnectionError):
                _completion_with_retry(
                    {"model": "x", "messages": []}, max_retries=3, verbose=False
                )
        assert mock.call_count == 3

    def test_non_transient_not_retried(self):
        import litellm

        exc = litellm.BadRequestError(
            message="bad input", llm_provider="openai", model="x"
        )
        mock = MagicMock(side_effect=exc)
        with patch("litellm.completion", mock):
            with pytest.raises(litellm.BadRequestError):
                _completion_with_retry(
                    {"model": "x", "messages": []}, max_retries=5, verbose=False
                )
        assert mock.call_count == 1

    def test_context_overflow_propagates(self):
        import litellm

        mock = MagicMock(
            side_effect=litellm.ContextWindowExceededError(
                message="too long", llm_provider="openai", model="x"
            )
        )
        with patch("litellm.completion", mock):
            with pytest.raises(ContextOverflowError):
                _completion_with_retry(
                    {"model": "x", "messages": []}, max_retries=5, verbose=False
                )


class TestCallLlmRetry:
    def test_returns_4_tuple(self):
        resp = _make_response()
        with patch("litellm.completion", return_value=resp):
            result = call_llm(
                "http://localhost:8080/v1",
                "my-model",
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
        assert len(result) == 4
        msg, finish_reason, cmd_activity, provider_retries = result
        assert msg.content == "hello"
        assert finish_reason == "stop"
        assert cmd_activity == []
        assert provider_retries == 0

    def test_provider_retries_reported(self):
        import litellm

        resp = _make_response()
        exc = litellm.APIConnectionError(
            message="Connection reset", llm_provider="openai", model="x"
        )
        mock = MagicMock(side_effect=[exc, resp])
        with patch("litellm.completion", mock), patch("time.sleep"):
            result = call_llm(
                "http://localhost:8080/v1",
                "my-model",
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
        assert result[3] == 1  # provider_retries

    def test_retries_1_no_retry(self):
        import litellm

        exc = litellm.APIConnectionError(
            message="Connection reset", llm_provider="openai", model="x"
        )
        with patch("litellm.completion", side_effect=exc):
            with pytest.raises(AgentError, match="LLM call failed"):
                call_llm(
                    "http://localhost:8080/v1",
                    "my-model",
                    [{"role": "user", "content": "hi"}],
                    100,
                    0.5,
                    1.0,
                    None,
                    None,
                    False,
                    provider="generic",
                    api_key="test",
                    max_retries=1,
                )

    def test_context_overflow_not_wrapped(self):
        import litellm

        mock = MagicMock(
            side_effect=litellm.ContextWindowExceededError(
                message="too long", llm_provider="openai", model="x"
            )
        )
        with patch("litellm.completion", mock):
            with pytest.raises(ContextOverflowError):
                call_llm(
                    "http://localhost:8080/v1",
                    "my-model",
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

    def test_command_provider_4_tuple(self):
        result = call_llm(
            None,
            "echo hello",
            [{"role": "user", "content": "hi"}],
            100,
            0.5,
            1.0,
            None,
            None,
            False,
            provider="command",
        )
        assert len(result) == 4
        assert result[3] == 0  # provider_retries

    def test_sanitization_retry_also_retries_transient(self):
        """Empty-assistant sanitization triggers a second _completion_with_retry
        call which should also handle transient errors."""
        import litellm

        bad_req = litellm.BadRequestError(
            message="must have either content or tool_calls",
            llm_provider="openai",
            model="x",
        )
        transient = litellm.APIConnectionError(
            message="Connection reset", llm_provider="openai", model="x"
        )
        resp = _make_response()
        # First call: BadRequestError (empty assistant)
        # Second call (after sanitize): transient
        # Third call: success
        mock = MagicMock(side_effect=[bad_req, transient, resp])

        # Need an assistant message with no content to trigger sanitization
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant"},
            {"role": "user", "content": "continue"},
        ]
        with patch("litellm.completion", mock), patch("time.sleep"):
            result = call_llm(
                "http://localhost:8080/v1",
                "my-model",
                messages,
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="generic",
                api_key="test",
                max_retries=5,
            )
        assert result[0].content == "hello"
        assert result[3] == 1  # one retry on the sanitization path

    def test_transient_then_empty_assistant_then_success(self):
        """Transient retry before BadRequestError(empty assistant) counts toward total."""
        import litellm

        transient = litellm.APIConnectionError(
            message="Connection reset", llm_provider="openai", model="x"
        )
        bad_req = litellm.BadRequestError(
            message="must have either content or tool_calls",
            llm_provider="openai",
            model="x",
        )
        resp = _make_response()
        # First call: transient → retry
        # Second call: BadRequestError (empty assistant) → sanitize
        # Third call (after sanitize): success
        mock = MagicMock(side_effect=[transient, bad_req, resp])

        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant"},
            {"role": "user", "content": "continue"},
        ]
        with patch("litellm.completion", mock), patch("time.sleep"):
            result = call_llm(
                "http://localhost:8080/v1",
                "my-model",
                messages,
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="generic",
                api_key="test",
                max_retries=5,
            )
        assert result[0].content == "hello"
        # 1 transient retry in first helper + 0 in second helper = 1 total
        assert result[3] == 1

    def test_empty_assistant_then_context_overflow_raises_coe(self):
        """BadRequestError(empty assistant) followed by BadRequestError(context overflow)
        must raise ContextOverflowError, not AgentError, so the compaction pipeline runs."""
        import litellm

        empty_msg = litellm.BadRequestError(
            message="must have either content or tool_calls",
            llm_provider="openai",
            model="x",
        )
        overflow = litellm.BadRequestError(
            message="maximum context length exceeded",
            llm_provider="openai",
            model="x",
        )
        mock = MagicMock(side_effect=[empty_msg, overflow])

        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant"},
            {"role": "user", "content": "continue"},
        ]
        with patch("litellm.completion", mock):
            with pytest.raises(ContextOverflowError, match="post-sanitization"):
                call_llm(
                    "http://localhost:8080/v1",
                    "my-model",
                    messages,
                    100,
                    0.5,
                    1.0,
                    None,
                    None,
                    False,
                    provider="generic",
                    api_key="test",
                )

    def test_provider_retries_on_failure(self):
        """When call_llm fails after retries, AgentError carries _provider_retries."""
        import litellm

        exc = litellm.APIConnectionError(
            message="Connection reset", llm_provider="openai", model="x"
        )
        mock = MagicMock(side_effect=[exc, exc, exc])
        with patch("litellm.completion", mock), patch("time.sleep"):
            with pytest.raises(AgentError) as exc_info:
                call_llm(
                    "http://localhost:8080/v1",
                    "my-model",
                    [{"role": "user", "content": "hi"}],
                    100,
                    0.5,
                    1.0,
                    None,
                    None,
                    False,
                    provider="generic",
                    api_key="test",
                    max_retries=3,
                )
        assert getattr(exc_info.value, "_provider_retries", None) == 2

    def test_provider_retries_on_context_overflow(self):
        """ContextOverflowError carries _provider_retries."""
        import litellm

        mock = MagicMock(
            side_effect=litellm.ContextWindowExceededError(
                message="too long", llm_provider="openai", model="x"
            )
        )
        with patch("litellm.completion", mock):
            with pytest.raises(ContextOverflowError) as exc_info:
                call_llm(
                    "http://localhost:8080/v1",
                    "my-model",
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
        assert getattr(exc_info.value, "_provider_retries", None) == 0

    def test_max_retries_zero_clamps_to_one(self):
        """max_retries=0 is clamped to 1 (single attempt, no crash)."""
        import litellm

        exc = litellm.APIConnectionError(
            message="Connection reset", llm_provider="openai", model="x"
        )
        with patch("litellm.completion", side_effect=exc):
            with pytest.raises(AgentError, match="LLM call failed"):
                call_llm(
                    "http://localhost:8080/v1",
                    "my-model",
                    [{"role": "user", "content": "hi"}],
                    100,
                    0.5,
                    1.0,
                    None,
                    None,
                    False,
                    provider="generic",
                    api_key="test",
                    max_retries=0,
                )

    def test_session_rejects_retries_zero(self):
        """Session(retries=0) raises ValueError."""
        from swival.session import Session

        with pytest.raises(ValueError, match="retries must be >= 1"):
            Session(retries=0)

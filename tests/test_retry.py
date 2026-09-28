"""Tests for transient-error retry logic in call_llm / _completion_with_retry."""

import copy
import types
from unittest.mock import patch, MagicMock

import pytest
from conftest import PROMPT_REJECTION, policy_violation, responses_rejection

from swival.agent import (
    call_llm,
    _completion_with_retry,
    _is_prompt_rejection,
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

    def test_service_unavailable(self):
        import litellm

        exc = litellm.ServiceUnavailableError(
            message="upstream connect error", llm_provider="openai", model="x"
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

    def test_chatgpt_explicit_retry_api_error_400(self):
        import litellm

        exc = litellm.APIError(
            status_code=400,
            message=(
                "ChatgptException - An error occurred while processing your request. "
                "You can retry your request, or contact us through our help center."
            ),
            llm_provider="openai",
            model="x",
        )
        assert _is_transient(exc) is True

    def test_string_pattern_connection_reset(self):
        exc = OSError("[Errno 54] Connection reset by peer")
        assert _is_transient(exc) is True

    def test_unrelated_error_not_transient(self):
        exc = ValueError("something unrelated")
        assert _is_transient(exc) is False

    def test_sso_token_expired_not_transient(self):
        import litellm

        exc = litellm.APIConnectionError(
            message="Error when retrieving token from sso: "
            "Token has expired and refresh failed",
            llm_provider="bedrock",
            model="x",
        )
        assert _is_transient(exc) is False

    def test_sso_token_missing_not_transient(self):
        import litellm

        exc = litellm.APIConnectionError(
            message="Error loading SSO Token: Token for fastly does not exist",
            llm_provider="bedrock",
            model="x",
        )
        assert _is_transient(exc) is False

    def test_sso_retrieval_network_error_is_transient(self):
        import litellm

        exc = litellm.APIConnectionError(
            message="Error when retrieving token from sso: Connection reset by peer",
            llm_provider="bedrock",
            model="x",
        )
        assert _is_transient(exc) is True


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

    def test_retries_chatgpt_explicit_retry_api_error(self):
        import litellm

        response = _make_response()
        exc = litellm.APIError(
            status_code=400,
            message=(
                "ChatgptException - An error occurred while processing your request. "
                "You can retry your request, or contact us through our help center."
            ),
            llm_provider="openai",
            model="x",
        )
        mock = MagicMock(side_effect=[exc, response])
        with patch("litellm.completion", mock), patch("time.sleep"):
            result, retries = _completion_with_retry(
                {"model": "x", "messages": []}, max_retries=2, verbose=False
            )
        assert result is response
        assert retries == 1
        assert mock.call_count == 2

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

    def test_sso_token_expired_no_retry_no_sleep(self):
        import litellm

        exc = litellm.APIConnectionError(
            message="Error when retrieving token from sso: "
            "Token has expired and refresh failed",
            llm_provider="bedrock",
            model="x",
        )
        with patch("litellm.completion", side_effect=exc):
            with patch("time.sleep") as mock_sleep:
                with pytest.raises(litellm.APIConnectionError) as exc_info:
                    _completion_with_retry(
                        {"model": "x", "messages": []},
                        max_retries=5,
                        verbose=True,
                    )
                assert exc_info.value._provider_retries == 0
                mock_sleep.assert_not_called()


class TestCallLlmRetry:
    def test_returns_5_tuple(self):
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
        assert len(result) == 5
        msg, finish_reason, cmd_activity, provider_retries, cache_stats = result
        assert msg.content == "hello"
        assert finish_reason == "stop"
        assert cmd_activity == []
        assert provider_retries == 0
        assert cache_stats == (0, 0)

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

    def test_command_provider_5_tuple(self):
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
        assert len(result) == 5
        assert result[4] == (0, 0)  # no cache stats for command provider
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

    def test_null_choices_normal_path(self):
        """choices=None in response raises AgentError through normal call_llm path."""
        resp = types.SimpleNamespace(choices=None)
        with patch("litellm.completion", return_value=resp):
            with pytest.raises(AgentError, match="choices=None"):
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

    def test_null_choices_post_sanitization_path(self):
        """choices=None after empty-assistant sanitization raises AgentError."""
        import litellm

        bad_req = litellm.BadRequestError(
            message="must have either content or tool_calls",
            llm_provider="openai",
            model="x",
        )
        null_resp = types.SimpleNamespace(choices=None)
        mock = MagicMock(side_effect=[bad_req, null_resp])

        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant"},
            {"role": "user", "content": "continue"},
        ]
        with patch("litellm.completion", mock):
            with pytest.raises(AgentError, match="choices=None"):
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

    def test_empty_choices_list(self):
        """choices=[] raises AgentError with 'empty choices list' (distinct from None)."""
        resp = types.SimpleNamespace(choices=[])
        with patch("litellm.completion", return_value=resp):
            with pytest.raises(AgentError, match="empty choices list"):
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

    def test_null_choices_retry_exhaustion_message(self):
        """InternalServerError from null choices retries 5 times; final error includes model_id."""
        import litellm

        exc = litellm.InternalServerError(
            message="Invalid response object: choices is None",
            llm_provider="openai",
            model="x",
        )
        mock = MagicMock(side_effect=[exc] * 5)
        with patch("litellm.completion", mock), patch("time.sleep"):
            with pytest.raises(AgentError, match=r"model: my-model") as exc_info:
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
                    max_retries=5,
                )
        assert exc_info.value._provider_retries == 4
        assert mock.call_count == 5

    def test_sso_expiry_error_message_includes_profile(self):
        import litellm

        exc = litellm.APIConnectionError(
            message="Error when retrieving token from sso: "
            "Token has expired and refresh failed",
            llm_provider="bedrock",
            model="x",
        )
        with patch("litellm.completion", side_effect=exc):
            with pytest.raises(
                AgentError, match=r"aws sso login --profile=myprofile"
            ) as exc_info:
                call_llm(
                    None,
                    "anthropic.claude-opus-4-6-v1",
                    [{"role": "user", "content": "hi"}],
                    4096,
                    None,
                    None,
                    None,
                    None,
                    False,
                    provider="bedrock",
                    aws_profile="myprofile",
                )
            assert exc_info.value._provider_retries == 0

    def test_sso_expiry_uses_aws_profile_env(self, monkeypatch):
        import litellm

        monkeypatch.setenv("AWS_PROFILE", "envprofile")
        exc = litellm.APIConnectionError(
            message="Error when retrieving token from sso: "
            "Token has expired and refresh failed",
            llm_provider="bedrock",
            model="x",
        )
        with patch("litellm.completion", side_effect=exc):
            with pytest.raises(
                AgentError, match=r"aws sso login --profile=envprofile"
            ) as exc_info:
                call_llm(
                    None,
                    "anthropic.claude-opus-4-6-v1",
                    [{"role": "user", "content": "hi"}],
                    4096,
                    None,
                    None,
                    None,
                    None,
                    False,
                    provider="bedrock",
                )
            assert exc_info.value._provider_retries == 0

    def test_sso_missing_token_error_message(self):
        import litellm

        exc = litellm.APIConnectionError(
            message="Error loading SSO Token: Token for fastly does not exist",
            llm_provider="bedrock",
            model="x",
        )
        with patch("litellm.completion", side_effect=exc):
            with pytest.raises(
                AgentError, match=r"aws sso login --profile=myprofile"
            ) as exc_info:
                call_llm(
                    None,
                    "anthropic.claude-opus-4-6-v1",
                    [{"role": "user", "content": "hi"}],
                    4096,
                    None,
                    None,
                    None,
                    None,
                    False,
                    provider="bedrock",
                    aws_profile="myprofile",
                )
            assert exc_info.value._provider_retries == 0

    def test_sso_expiry_defaults_to_default_profile(self, monkeypatch):
        import litellm

        monkeypatch.delenv("AWS_PROFILE", raising=False)
        exc = litellm.APIConnectionError(
            message="Error when retrieving token from sso: "
            "Token has expired and refresh failed",
            llm_provider="bedrock",
            model="x",
        )
        with patch("litellm.completion", side_effect=exc):
            with pytest.raises(
                AgentError, match=r"aws sso login --profile=default"
            ) as exc_info:
                call_llm(
                    None,
                    "anthropic.claude-opus-4-6-v1",
                    [{"role": "user", "content": "hi"}],
                    4096,
                    None,
                    None,
                    None,
                    None,
                    False,
                    provider="bedrock",
                )
            assert exc_info.value._provider_retries == 0

    def test_session_rejects_retries_zero(self):
        """Session(retries=0) raises ValueError."""
        from swival.session import Session

        with pytest.raises(ValueError, match="retries must be >= 1"):
            Session(retries=0)


def _delta_chunk(content=None, reasoning_content=None, reasoning=None, tool_calls=None):
    delta = types.SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content,
        reasoning=reasoning,
        tool_calls=tool_calls,
    )
    choice = types.SimpleNamespace(delta=delta)
    return types.SimpleNamespace(choices=[choice])


def _tool_delta(name=None, args=None):
    fn = types.SimpleNamespace(name=name, arguments=args)
    tc = types.SimpleNamespace(function=fn)
    return _delta_chunk(tool_calls=[tc])


class _FakeChannels:
    """Records (reasoning, answer, activity) updates from stream_channels."""

    def __init__(self):
        self.events = []

    def cm(self):
        import contextlib

        @contextlib.contextmanager
        def _fake():
            self.events.append(("enter", None))

            def _update(reasoning="", answer="", activity=""):
                self.events.append(("update", (reasoning, answer, activity)))

            try:
                yield _update
            finally:
                self.events.append(("exit", None))

        return _fake

    @property
    def kinds(self):
        return [k for k, _ in self.events]

    @property
    def updates(self):
        return [v for k, v in self.events if k == "update"]


def _run_splitter(chunks):
    from swival import agent

    s = agent._InlineThinkSplitter()
    answer, reason = [], []
    for c in chunks:
        a, r = s.feed(c)
        answer.append(a)
        reason.append(r)
    a, r = s.flush()
    answer.append(a)
    reason.append(r)
    return "".join(answer), "".join(reason)


class TestInlineThinkSplitter:
    def test_simple_block(self):
        assert _run_splitter(["<think>hi</think>answer"]) == ("answer", "hi")

    def test_open_tag_split_across_chunks(self):
        assert _run_splitter(["<thi", "nk>secret</think>visible"]) == (
            "visible",
            "secret",
        )

    def test_close_tag_split_across_chunks(self):
        assert _run_splitter(["<think>sec", "ret</thi", "nk>done"]) == (
            "done",
            "secret",
        )

    def test_leading_whitespace_still_opens(self):
        assert _run_splitter(["\n  <think>mid</think>post"]) == ("\n  post", "mid")

    def test_no_closing_tag_routes_rest_to_reasoning(self):
        assert _run_splitter(["<think>never closes"]) == ("", "never closes")

    def test_literal_mention_after_answer_not_captured(self):
        # Once real answer text has streamed, a later <think> is literal prose.
        assert _run_splitter(["Use the <think> tag wisely"]) == (
            "Use the <think> tag wisely",
            "",
        )

    def test_partial_tag_at_end_flushed_to_answer(self):
        assert _run_splitter(["hello <thi"]) == ("hello <thi", "")


class TestExtractReasoningDelta:
    def test_reasoning_content_string(self):
        from swival import agent

        d = types.SimpleNamespace(reasoning_content="abc", reasoning=None)
        assert agent._extract_reasoning_delta(d) == "abc"

    def test_reasoning_field_fallback(self):
        from swival import agent

        d = types.SimpleNamespace(reasoning_content=None, reasoning="xyz")
        assert agent._extract_reasoning_delta(d) == "xyz"

    def test_both_present_not_duplicated(self):
        from swival import agent

        d = types.SimpleNamespace(reasoning_content="dup", reasoning="dup")
        assert agent._extract_reasoning_delta(d) == "dup"

    def test_nested_summary_object(self):
        from swival import agent

        nested = types.SimpleNamespace(
            summary="S", thinking=None, text=None, content=None
        )
        d = types.SimpleNamespace(reasoning_content=None, reasoning=nested)
        assert agent._extract_reasoning_delta(d) == "S"

    def test_dict_delta(self):
        from swival import agent

        assert agent._extract_reasoning_delta({"reasoning_content": "d1"}) == "d1"

    def test_nested_list_of_text_parts(self):
        from swival import agent

        d = {"reasoning": {"summary": [{"text": "a"}, {"text": "b"}]}}
        assert agent._extract_reasoning_delta(d) == "ab"

    def test_no_reasoning_returns_empty(self):
        from swival import agent

        d = types.SimpleNamespace(content="hi", reasoning_content=None, reasoning=None)
        assert agent._extract_reasoning_delta(d) == ""


class TestCompletionViaStream:
    def test_stream_entered_only_after_handoff(self, monkeypatch):
        """The live stream display opens only once accumulated text crosses the
        handoff threshold, and on_stream_start fires exactly once right before
        the first update."""
        from io import StringIO

        from rich.console import Console

        from swival import agent, fmt

        old_console = fmt._console
        fmt._console = Console(file=StringIO(), width=40, height=10)
        threshold = max(fmt._console.width, 40)

        rec = _FakeChannels()
        monkeypatch.setattr(fmt, "stream_channels", rec.cm())

        # Three deltas below the threshold individually, crossing it cumulatively
        # on the third.
        step = threshold // 2
        chunks = [_delta_chunk(content="x" * step) for _ in range(3)]

        def fake_completion(**kwargs):
            assert kwargs.get("stream") is True
            return iter(chunks)

        starts = []
        try:
            with (
                patch("litellm.completion", fake_completion),
                patch(
                    "litellm.stream_chunk_builder",
                    lambda chunks, messages=None: "rebuilt",
                ),
            ):
                result = agent._completion_via_stream(
                    {"model": "x", "messages": []},
                    on_stream_start=lambda: starts.append(True),
                )
        finally:
            fmt._console = old_console

        assert result == "rebuilt"
        assert len(starts) == 1
        assert rec.kinds.count("enter") == 1
        # Order: enter the live context, then the first update.
        assert rec.kinds[0] == "enter"
        assert rec.kinds[1] == "update"
        # Nothing was drawn before the threshold was crossed: the first update
        # already holds at least a threshold's worth of answer text.
        first = rec.updates[0]
        assert len(first[1]) >= threshold

    def test_handoff_counts_all_channels(self, monkeypatch):
        """Reasoning text counts toward the marquee handoff threshold too, so a
        long thinking preamble dismisses the marquee just as answer text would."""
        from io import StringIO

        from rich.console import Console

        from swival import agent, fmt

        old_console = fmt._console
        fmt._console = Console(file=StringIO(), width=40, height=10)
        threshold = max(fmt._console.width, 40)

        rec = _FakeChannels()
        monkeypatch.setattr(fmt, "stream_channels", rec.cm())

        step = threshold // 2
        chunks = [_delta_chunk(reasoning_content="r" * step) for _ in range(3)]

        try:
            with (
                patch("litellm.completion", lambda **k: iter(chunks)),
                patch(
                    "litellm.stream_chunk_builder",
                    lambda chunks, messages=None: "rebuilt",
                ),
            ):
                agent._completion_via_stream({"model": "x", "messages": []})
        finally:
            fmt._console = old_console

        # The display opened and the first frame carries reasoning, not answer.
        assert rec.kinds.count("enter") == 1
        reasoning, answer, activity = rec.updates[0]
        assert len(reasoning) >= threshold
        assert answer == ""

    def test_tool_calls_route_to_activity(self, monkeypatch):
        """Streamed tool-call name/args land in the activity channel, never the
        answer or reasoning channels."""
        from io import StringIO

        from rich.console import Console

        from swival import agent, fmt

        old_console = fmt._console
        fmt._console = Console(file=StringIO(), width=40, height=10)

        rec = _FakeChannels()
        monkeypatch.setattr(fmt, "stream_channels", rec.cm())

        pad = "p" * (max(fmt._console.width, 40) + 4)
        chunks = [
            _delta_chunk(content=pad),
            _tool_delta(name="read_file"),
            _tool_delta(args='{"path":"x"}'),
        ]

        try:
            with (
                patch("litellm.completion", lambda **k: iter(chunks)),
                patch(
                    "litellm.stream_chunk_builder",
                    lambda chunks, messages=None: "rebuilt",
                ),
            ):
                agent._completion_via_stream({"model": "x", "messages": []})
        finally:
            fmt._console = old_console

        reasoning, answer, activity = rec.updates[-1]
        assert "read_file" in activity
        assert '{"path":"x"}' in activity
        assert "read_file" not in answer
        assert "read_file" not in reasoning

    def test_display_false_is_noop(self, monkeypatch):
        """With display=False nothing is rendered, but chunks still reassemble."""
        from swival import agent, fmt

        rec = _FakeChannels()
        monkeypatch.setattr(fmt, "stream_channels", rec.cm())

        chunks = [
            _delta_chunk(content="hello"),
            _delta_chunk(reasoning_content="think"),
        ]
        with (
            patch("litellm.completion", lambda **k: iter(chunks)),
            patch(
                "litellm.stream_chunk_builder",
                lambda chunks, messages=None: "rebuilt",
            ),
        ):
            result = agent._completion_via_stream(
                {"model": "x", "messages": []}, display=False
            )

        assert result == "rebuilt"
        assert rec.events == []

    def test_reconstructed_content_unaffected_by_think_routing(self, monkeypatch):
        """Inline <think> routing is display-only: the chunks handed to
        stream_chunk_builder are exactly what streamed, so the rebuilt content
        is identical to the non-streaming path."""
        from io import StringIO

        from rich.console import Console

        from swival import agent, fmt

        old_console = fmt._console
        fmt._console = Console(file=StringIO(), width=40, height=10)

        rec = _FakeChannels()
        monkeypatch.setattr(fmt, "stream_channels", rec.cm())

        pad = "z" * (max(fmt._console.width, 40) + 4)
        chunks = [_delta_chunk(content="<think>secret</think>" + pad)]
        seen = {}

        def fake_builder(chs, messages=None):
            seen["chunks"] = chs
            return "rebuilt"

        try:
            with (
                patch("litellm.completion", lambda **k: iter(chunks)),
                patch("litellm.stream_chunk_builder", fake_builder),
            ):
                agent._completion_via_stream({"model": "x", "messages": []})
        finally:
            fmt._console = old_console

        # The raw chunk objects are passed through untouched (<think> intact).
        assert [id(c) for c in seen["chunks"]] == [id(c) for c in chunks]
        assert "<think>secret</think>" in seen["chunks"][0].choices[0].delta.content
        # The live display split the secret into reasoning, answer kept the pad.
        reasoning, answer, activity = rec.updates[-1]
        assert "secret" in reasoning
        assert "secret" not in answer


class TestStreamedResponseCost:
    def test_cost_computed_from_rebuilt_stream_response(self, monkeypatch):
        """Cost comes from the response stream_chunk_builder returns, which has
        usage but no hidden response_cost, so completion_cost is the source."""
        from conftest import plain_console

        from swival import agent
        from swival.cost import SessionCost

        rebuilt = _make_response()
        rebuilt.usage = types.SimpleNamespace(prompt_tokens=50, completion_tokens=5)

        def fake_stream(
            kwargs, on_stream_start=None, display=True, show_thinking=False
        ):
            return rebuilt

        monkeypatch.setattr(agent, "_completion_via_stream", fake_stream)
        monkeypatch.setattr("sys.stderr", types.SimpleNamespace(isatty=lambda: True))

        seen = {}

        def fake_cost(completion_response=None, model=None):
            seen["response"] = completion_response
            seen["model"] = model
            return 0.007

        monkeypatch.setattr("litellm.completion_cost", fake_cost)

        sc = SessionCost()
        with plain_console(width=80):
            msg, finish, *_ = call_llm(
                None,
                "my-model",
                [{"role": "user", "content": "hi"}],
                100,
                0.5,
                None,
                None,
                None,
                True,  # verbose: required for the streaming path
                provider="openrouter",
                api_key="k",
                session_cost=sc,
            )

        assert msg.content == "hello"
        assert seen["response"] is rebuilt
        assert seen["model"] == "openrouter/my-model"
        snap = sc.snapshot()
        assert snap.priced_calls == 1
        assert snap.known_usd == 0.007


_REJECTION_HINT = (
    "The provider rejected the prompt. Try another model with /model in the "
    "REPL or --model on the next run."
)


def _connection_reset():
    import litellm

    return litellm.APIConnectionError(
        message="Connection reset", llm_provider="openai", model="x"
    )


# Each rejection shape with the provider path that produces it.
_BOTH_SHAPES = pytest.mark.parametrize(
    "rejection, provider",
    [(responses_rejection, "chatgpt"), (policy_violation, "generic")],
    ids=["responses", "chat"],
)


@pytest.fixture
def stub_provider():
    with patch("litellm.completion") as completion, patch("time.sleep") as sleep:
        yield types.SimpleNamespace(completion=completion, sleep=sleep)


def _endpoint(provider):
    if provider == "generic":
        return {"base_url": "http://localhost:8080/v1", "api_key": "test"}
    return {"base_url": None, "api_key": None}


def _call(provider="chatgpt", **kwargs):
    defaults = dict(
        model_id="gpt-6-sol",
        messages=[{"role": "user", "content": "hi"}],
        max_output_tokens=100,
        temperature=None,
        top_p=None,
        seed=None,
        tools=None,
        verbose=False,
        provider=provider,
        **_endpoint(provider),
    )
    return call_llm(**(defaults | kwargs))


def _retry(**kwargs):
    options = {"max_retries": 5, "verbose": False, **kwargs}
    return _completion_with_retry({"model": "x", "messages": []}, **options)


class TestPromptRejectionMatcher:
    @pytest.mark.parametrize(
        "exc",
        [
            responses_rejection(),
            policy_violation(),
            responses_rejection(
                "INVALID PROMPT :\n  Your prompt was flagged as potentially\t"
                "violating our Usage Policy."
            ),
        ],
        ids=["responses", "chat", "loose-spacing"],
    )
    def test_matches(self, exc):
        assert _is_prompt_rejection(exc)

    @pytest.mark.parametrize(
        "exc",
        [
            responses_rejection("Invalid prompt: messages must not be empty"),
            policy_violation(
                "Your prompt was flagged as potentially violating our usage policy."
            ),
            policy_violation("Your request was rejected by our safety system."),
            policy_violation(PROMPT_REJECTION + " The attached image was flagged."),
            RuntimeError(PROMPT_REJECTION),
        ],
        ids=["other-400", "no-prefix", "other-policy", "blames-image", "not-litellm"],
    )
    def test_near_misses(self, exc):
        assert not _is_prompt_rejection(exc)


class TestPromptRejectionRetry:
    @_BOTH_SHAPES
    def test_resent_unchanged_then_succeeds(self, stub_provider, rejection, provider):
        outcomes = iter([rejection(), _make_response("recovered")])
        sent = []

        def completion(**kwargs):
            sent.append(copy.deepcopy(kwargs))
            outcome = next(outcomes)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        stub_provider.completion.side_effect = completion
        tools = [{"type": "function", "function": {"name": "read_file"}}]
        msg, _finish, _activity, retries, _stats = _call(
            provider, tools=tools, reasoning_effort="high"
        )
        assert (msg.content, retries) == ("recovered", 1)
        assert len(sent) == 2
        assert sent[0] == sent[1]

    def test_exhaustion_raises_original_with_count(self, stub_provider):
        from swival import fmt

        stub_provider.completion.side_effect = [responses_rejection()] * 3
        with (
            patch("random.random", return_value=0.5),
            patch.object(fmt, "warning") as warning,
            pytest.raises(type(responses_rejection())) as exc_info,
        ):
            _retry(max_retries=3, verbose=True)
        assert exc_info.value._provider_retries == 2
        assert [c.args[0] for c in warning.call_args_list] == [
            "Prompt rejected; retrying in 2s (attempt 2/3)",
            "Prompt rejected; retrying in 4s (attempt 3/3)",
        ]

    def test_network_errors_and_rejections_share_the_attempt_limit(self, stub_provider):
        stub_provider.completion.side_effect = [
            _connection_reset(),
            responses_rejection(),
            _connection_reset(),
        ]
        with pytest.raises(type(_connection_reset())):
            _retry(max_retries=3)
        assert stub_provider.completion.call_count == 3


class TestPromptRejectionCallLlm:
    @_BOTH_SHAPES
    def test_exhaustion_guidance(self, stub_provider, rejection, provider):
        original = rejection()
        stub_provider.completion.side_effect = [original, rejection(), original]
        with pytest.raises(AgentError) as exc_info:
            _call(provider, max_retries=3)
        err = exc_info.value
        assert "Invalid prompt: your prompt was flagged" in str(err)
        assert str(err).endswith(_REJECTION_HINT)
        assert err.__context__ is original
        assert err._provider_retries == 2
        assert not getattr(err, "_request_shaped", False)

    @pytest.mark.parametrize(
        "broken_history, repair_error, where",
        [
            (
                {"role": "assistant"},
                "must have either content or tool_calls",
                "after message sanitization",
            ),
            (
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ],
                },
                "No tool output found for function call call_1",
                "after orphaned-tool-call fix",
            ),
        ],
        ids=["empty-assistant", "orphaned-tool-call"],
    )
    def test_guidance_after_history_repair(
        self, stub_provider, broken_history, repair_error, where
    ):
        import litellm

        repairable = litellm.BadRequestError(
            message=repair_error, llm_provider="openai", model="x"
        )
        stub_provider.completion.side_effect = [repairable] + [policy_violation()] * 3
        messages = [
            {"role": "user", "content": "hi"},
            broken_history,
            {"role": "user", "content": "continue"},
        ]
        with pytest.raises(AgentError) as exc_info:
            _call("generic", messages=messages, max_retries=3)
        assert where in str(exc_info.value)
        assert str(exc_info.value).endswith(_REJECTION_HINT)
        assert stub_provider.completion.call_count == 4

    def test_mid_stream_rejection_discards_partial_tool_call(
        self, stub_provider, monkeypatch
    ):
        """Nothing streamed before the rejection, including a half-sent tool
        call, ends up in the final response."""
        from conftest import plain_console

        from swival import fmt

        rec = _FakeChannels()
        monkeypatch.setattr(fmt, "stream_channels", rec.cm())
        monkeypatch.setattr("sys.stderr", types.SimpleNamespace(isatty=lambda: True))

        def rejected_stream():
            yield _delta_chunk(content="p" * 100)
            yield _tool_delta(name="write_file", args='{"path": "a.txt", "con')
            raise responses_rejection()

        retry_chunks = [_delta_chunk(content="done")]
        stub_provider.completion.side_effect = [rejected_stream(), iter(retry_chunks)]
        built = []

        def fake_builder(chunks, messages=None):
            built.append(list(chunks))
            return _make_response("done")

        with (
            plain_console(width=80),
            patch("litellm.stream_chunk_builder", fake_builder),
        ):
            msg, _finish, _activity, retries, _stats = _call("generic", verbose=True)

        assert (msg.content, msg.tool_calls, retries) == ("done", None, 1)
        assert built == [retry_chunks]
        assert rec.kinds[0] == "enter" and rec.kinds[-1] == "exit"


class TestPromptRejectionFallbackCalls:
    @_BOTH_SHAPES
    def test_summary_retries_network_errors_only(
        self, stub_provider, rejection, provider
    ):
        from swival.agent import _call_summarize_llm

        stub_provider.completion.side_effect = [
            _connection_reset(),
            rejection(),
            _make_response(),
        ]
        endpoint = _endpoint(provider)
        result = _call_summarize_llm(
            "text",
            "summarize",
            call_llm,
            "gpt-6-sol",
            endpoint["base_url"],
            endpoint["api_key"],
            None,
            None,
            provider,
        )
        assert result is None
        assert stub_provider.completion.call_count == 2

    def test_continue_file_keeps_deterministic_version(self, stub_provider, tmp_path):
        from swival.continue_here import write_continue_file

        stub_provider.completion.side_effect = [
            policy_violation(),
            _make_response("LLM"),
        ]
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "fix the parser"},
        ]
        assert write_continue_file(
            str(tmp_path),
            messages,
            call_llm_fn=call_llm,
            model_id="m",
            provider="generic",
            **_endpoint("generic"),
        )
        assert stub_provider.completion.call_count == 1
        content = (tmp_path / ".swival" / "continue.md").read_text()
        assert "fix the parser" in content

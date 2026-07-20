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

            yield _update

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

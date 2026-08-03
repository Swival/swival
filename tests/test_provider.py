"""Tests for provider routing, model normalization, CLI validation, and path isolation."""

import contextlib
import json
import os
import sys
import time
import types
import urllib.error

import pytest
from unittest.mock import patch, MagicMock

from conftest import forbid_copilot_authenticator

from swival import agent
from swival.agent import (
    call_llm,
    discover_generic_context_length,
    discover_llamacpp_context_length,
    resolve_provider,
    _fix_orphaned_tool_calls,
    _EMPTY_ASSISTANT_PLACEHOLDER,
    _msg_to_dict,
    _pick_best_choice,
    _promote_reasoning_content,
    _resolve_model_str,
    _sanitize_assistant_content,
    _strip_leaked_channel_head,
    _strip_leaked_think_head,
)
from swival.config import ConfigError
from swival.report import AgentError
from swival.tools import sanitize_tools_for_applefm


@pytest.fixture(autouse=True)
def _isolate_credential_caches(tmp_path_factory, monkeypatch):
    """Point the ChatGPT and Copilot token caches at a throwaway directory.

    --logout deletes credential files, and both subscription providers read
    cached tokens through litellm's get_llm_provider(); without isolation
    these tests would depend on (or destroy) the developer's real
    credentials under ~/.config/litellm.

    The ChatGPT cache gets a fake unexpired token: litellm's chatgpt
    adapter starts a device-code login whenever no cached token exists,
    so an empty cache would turn every chatgpt resolution test into a
    15-minute network poll. The Copilot cache stays empty on purpose —
    Swival's preflight must fail fast there, and tests that need a login
    write the token file explicitly.
    """
    base = tmp_path_factory.mktemp("credential-caches")
    chatgpt_dir = base / "chatgpt"
    chatgpt_dir.mkdir()
    (chatgpt_dir / "auth.json").write_text(
        json.dumps(
            {"access_token": "fake-cached-token", "expires_at": time.time() + 86400}
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(chatgpt_dir))
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN_DIR", str(base / "github_copilot"))
    monkeypatch.delenv("CHATGPT_AUTH_FILE", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_ACCESS_TOKEN_FILE", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_API_KEY_FILE", raising=False)


def _patch_urlopen_json(monkeypatch, payload):
    """Make urllib.request.urlopen yield *payload* as a JSON response body."""

    @contextlib.contextmanager
    def _fake_urlopen(req, timeout=10):
        resp = MagicMock()
        resp.read.return_value = json.dumps(payload).encode()
        yield resp

    monkeypatch.setattr(agent.urllib.request, "urlopen", _fake_urlopen)


def _patch_urlopen_error(monkeypatch):
    """Make urllib.request.urlopen raise, simulating an unreachable server."""

    def _boom(req, timeout=10):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(agent.urllib.request, "urlopen", _boom)


class TestImportLitellm:
    """The bundled offline cost map must be forced before litellm is imported."""

    def test_sets_local_cost_map_before_import(self, monkeypatch):
        # Drop the var and any cached litellm so the import path runs fresh.
        monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
        for name in list(sys.modules):
            if name == "litellm" or name.startswith("litellm."):
                monkeypatch.delitem(sys.modules, name, raising=False)

        seen = {}

        import builtins

        real_import = builtins.__import__

        def _spy_import(name, *a, **k):
            if name == "litellm" and "env" not in seen:
                seen["env"] = os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _spy_import)

        agent._import_litellm()

        # The env var was already "True" at the moment litellm got imported.
        assert seen["env"] == "True"
        assert os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"

    def test_does_not_override_explicit_setting(self, monkeypatch):
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "False")
        agent._import_litellm()
        assert os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "False"


# ---------------------------------------------------------------------------
# call_llm routing
# ---------------------------------------------------------------------------


class TestCallLlmRouting:
    """Verify that call_llm passes the right model string, api_key, and api_base."""

    def _mock_response(self):
        choice = MagicMock()
        choice.message = MagicMock(content="ok", tool_calls=None)
        choice.finish_reason = "stop"
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def test_lmstudio_routing(self):
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "http://localhost:1234",
                "my-model",
                [],
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="lmstudio",
                api_key=None,
            )
            mock_comp.assert_called_once()
            kwargs = mock_comp.call_args
            assert kwargs[1]["model"] == "openai/my-model"
            assert kwargs[1]["api_key"] == "lm-studio"
            assert kwargs[1]["api_base"] == "http://localhost:1234/v1"

    def test_huggingface_routing(self):
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                None,
                "zai-org/GLM-5.2",
                [],
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="huggingface",
                api_key="hf_test",
            )
            mock_comp.assert_called_once()
            kwargs = mock_comp.call_args
            assert kwargs[1]["model"] == "huggingface/zai-org/GLM-5.2"
            assert kwargs[1]["api_key"] == "hf_test"
            assert "api_base" not in kwargs[1]

    def test_huggingface_with_base_url(self):
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "https://xyz.endpoints.huggingface.cloud",
                "zai-org/GLM-5.2",
                [],
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="huggingface",
                api_key="hf_test",
            )
            mock_comp.assert_called_once()
            kwargs = mock_comp.call_args
            assert kwargs[1]["api_base"] == "https://xyz.endpoints.huggingface.cloud"

    def test_huggingface_non_chat_model_falls_back_to_text_generation(self):
        import litellm

        error = litellm.BadRequestError(
            message=(
                'HuggingfaceException - {"error":{"message":"The requested model '
                '\'google/gemma-4-E4B-it\' is not a chat model.","type":"invalid_request_error",'
                '"param":"model","code":"model_not_supported"}}'
            ),
            model="huggingface/google/gemma-4-E4B-it",
            llm_provider="huggingface",
        )

        client = MagicMock()
        client.text_generation.return_value = "fallback ok"
        info = MagicMock(
            inference="warm",
            inference_provider_mapping=[],
            pipeline_tag="text-generation",
        )

        with (
            patch("litellm.completion", side_effect=error) as mock_comp,
            patch(
                "huggingface_hub.InferenceClient", return_value=client
            ) as mock_client,
            patch("huggingface_hub.HfApi") as mock_hf_api,
        ):
            mock_hf_api.return_value.model_info.return_value = info
            msg, finish_reason, cmd_activity, retries, cache_stats = call_llm(
                None,
                "google/gemma-4-E4B-it",
                [{"role": "user", "content": "hi"}],
                100,
                0.5,
                1.0,
                7,
                None,
                False,
                provider="huggingface",
                api_key="hf_test",
                max_retries=1,
            )

        mock_comp.assert_called_once()
        mock_client.assert_called_once_with(provider="hf-inference", api_key="hf_test")
        client.text_generation.assert_called_once()
        assert finish_reason == "stop"
        assert msg.content == "fallback ok"
        assert cmd_activity == []
        assert retries == 0
        assert cache_stats == (0, 0)

    def test_huggingface_non_chat_model_not_deployed_gets_clear_error(self):
        import litellm

        error = litellm.BadRequestError(
            message="The requested model 'google/gemma-4-E4B-it' is not a chat model.",
            model="huggingface/google/gemma-4-E4B-it",
            llm_provider="huggingface",
        )
        info = MagicMock(
            inference=None,
            inference_provider_mapping=[],
            pipeline_tag="any-to-any",
        )

        with (
            patch("litellm.completion", side_effect=error),
            patch("huggingface_hub.HfApi") as mock_hf_api,
        ):
            mock_hf_api.return_value.model_info.return_value = info
            with pytest.raises(
                AgentError, match="not deployed by any Hugging Face Inference Provider"
            ):
                call_llm(
                    None,
                    "google/gemma-4-E4B-it",
                    [{"role": "user", "content": "hi"}],
                    100,
                    0.5,
                    1.0,
                    None,
                    None,
                    False,
                    provider="huggingface",
                    api_key="hf_test",
                    max_retries=1,
                )

    def test_huggingface_non_chat_model_with_tools_raises_tools_not_supported(self):
        import litellm
        from swival.report import ToolsNotSupportedError

        error = litellm.BadRequestError(
            message="The requested model 'google/gemma-4-E4B-it' is not a chat model.",
            model="huggingface/google/gemma-4-E4B-it",
            llm_provider="huggingface",
        )

        with patch("litellm.completion", side_effect=error):
            with pytest.raises(ToolsNotSupportedError):
                call_llm(
                    None,
                    "google/gemma-4-E4B-it",
                    [{"role": "user", "content": "hi"}],
                    100,
                    0.5,
                    1.0,
                    None,
                    [{"type": "function", "function": {"name": "dummy"}}],
                    False,
                    provider="huggingface",
                    api_key="hf_test",
                    max_retries=1,
                )

    def test_openrouter_routing(self):
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                None,
                "openrouter/free",
                [],
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="openrouter",
                api_key="sk_or_test_key",
            )
            mock_comp.assert_called_once()
            kwargs = mock_comp.call_args
            assert kwargs[1]["model"] == "openrouter/openrouter/free"
            assert kwargs[1]["api_key"] == "sk_or_test_key"
            assert "api_base" not in kwargs[1]

    def test_openrouter_with_base_url(self):
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "https://custom.openrouter.endpoint",
                "openrouter/free",
                [],
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="openrouter",
                api_key="sk_or_test_key",
            )
            mock_comp.assert_called_once()
            kwargs = mock_comp.call_args
            assert kwargs[1]["api_base"] == "https://custom.openrouter.endpoint"


class TestTopPDefault:
    """Verify that top_p is omitted from provider kwargs when not explicitly set."""

    def _mock_response(self):
        choice = MagicMock()
        choice.message = MagicMock(content="ok", tool_calls=None)
        choice.finish_reason = "stop"
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        resp.usage.prompt_tokens_details = None
        return resp

    def test_top_p_none_omitted_from_kwargs(self):
        """When top_p is None (the default), it must not appear in provider kwargs."""
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "http://localhost:1234/v1",
                "test-model",
                [{"role": "user", "content": "hi"}],
                100,
                None,  # temperature
                None,  # top_p
                None,  # seed
                None,  # tools
                False,
                provider="generic",
                api_key="sk-test",
            )
            mock_comp.assert_called_once()
            kwargs = mock_comp.call_args[1]
            assert "top_p" not in kwargs

    def test_top_p_explicit_included_in_kwargs(self):
        """When top_p is explicitly set, it must appear in provider kwargs."""
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "http://localhost:1234/v1",
                "test-model",
                [{"role": "user", "content": "hi"}],
                100,
                None,  # temperature
                0.9,  # top_p
                None,  # seed
                None,  # tools
                False,
                provider="generic",
                api_key="sk-test",
            )
            mock_comp.assert_called_once()
            kwargs = mock_comp.call_args[1]
            assert kwargs["top_p"] == 0.9

    def test_top_p_none_not_in_verbose_output(self, capsys):
        """Verbose output must not mention top_p when it is unset."""
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "http://localhost:1234/v1",
                "test-model",
                [{"role": "user", "content": "hi"}],
                100,
                None,  # temperature
                None,  # top_p
                None,  # seed
                None,  # tools
                True,  # verbose
                provider="generic",
                api_key="sk-test",
            )
        captured = capsys.readouterr()
        assert "top_p" not in captured.err

    def test_top_p_explicit_in_verbose_output(self, capsys):
        """Verbose output must show top_p when it is explicitly set."""
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "http://localhost:1234/v1",
                "test-model",
                [{"role": "user", "content": "hi"}],
                100,
                None,  # temperature
                0.9,  # top_p
                None,  # seed
                None,  # tools
                True,  # verbose
                provider="generic",
                api_key="sk-test",
            )
        captured = capsys.readouterr()
        assert "top_p=0.9" in captured.err


class TestSessionTopPDefault:
    """Verify Session() exposes the correct top_p default at the API layer."""

    def test_session_default_top_p_is_none(self, tmp_path):
        from swival.session import Session

        s = Session(base_dir=str(tmp_path), history=False)
        assert s.top_p is None

    def test_session_explicit_top_p(self, tmp_path):
        from swival.session import Session

        s = Session(base_dir=str(tmp_path), history=False, top_p=0.95)
        assert s.top_p == 0.95


class TestAssistantContentSanitization:
    """Verify call_llm strips leaked hidden-reasoning markers."""

    def test_sanitize_assistant_content_strips_special_tokens_and_think_prefix(self):
        text = "<|start_header_id|><think>Plan</think>\n\nAnswer"
        assert _sanitize_assistant_content(text) == "Answer"

    def test_sanitize_assistant_content_strips_channel_prefix(self):
        text = "<|channel>thought\n<channel|>Answer"
        assert _sanitize_assistant_content(text) == "Answer"

    def test_sanitize_assistant_content_handles_repeated_think_blocks(self):
        text = "<think>one</think>\n<think>two</think>\nFinal"
        assert _sanitize_assistant_content(text) == "Final"

    def test_sanitize_assistant_content_returns_empty_for_all_think_content(self):
        text = "<think>Plan only</think>"
        assert _sanitize_assistant_content(text) == ""

    def test_sanitize_assistant_content_preserves_empty_string(self):
        assert _sanitize_assistant_content("") == ""

    def test_call_llm_strips_think_prefix_when_opted_in(self):
        message = types.SimpleNamespace(
            role="assistant",
            content="Plan...\n</think>\n\nHello! How can I help?",
            tool_calls=None,
        )
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message, finish_reason="stop")]
        )

        with patch("litellm.completion", return_value=response):
            msg, finish_reason, *_ = call_llm(
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
                api_key="sk-test",
                sanitize_thinking=True,
            )

        assert finish_reason == "stop"
        assert msg.content == "Hello! How can I help?"

    def test_call_llm_clears_think_prefix_on_tool_call_messages(self):
        tool_call = types.SimpleNamespace(
            id="call_1",
            function=types.SimpleNamespace(name="get_time", arguments="{}"),
        )
        message = types.SimpleNamespace(
            role="assistant",
            content="Need the UTC clock.\n</think>",
            tool_calls=[tool_call],
        )
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message, finish_reason="tool_calls")]
        )

        with patch("litellm.completion", return_value=response):
            msg, finish_reason, *_ = call_llm(
                "http://localhost:8080/v1",
                "my-model",
                [{"role": "user", "content": "What time is it in UTC?"}],
                100,
                0.5,
                1.0,
                None,
                [],
                False,
                provider="generic",
                api_key="sk-test",
                sanitize_thinking=True,
            )

        assert finish_reason == "tool_calls"
        assert msg.content == ""
        assert msg.tool_calls == [tool_call]

    def test_call_llm_off_by_default_for_all_providers(self):
        """sanitize_thinking is off by default regardless of provider."""
        for provider in ("generic", "lmstudio", "openrouter", "chatgpt"):
            message = types.SimpleNamespace(
                role="assistant",
                content="<think>Plan</think>\n\nAnswer",
                tool_calls=None,
            )
            response = types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=message, finish_reason="stop")]
            )

            with patch("litellm.completion", return_value=response):
                msg, *_ = call_llm(
                    "http://localhost:8080/v1",
                    "my-model",
                    [{"role": "user", "content": "hi"}],
                    100,
                    0.5,
                    1.0,
                    None,
                    None,
                    False,
                    provider=provider,
                    api_key="sk-test",
                )

            assert msg.content == "<think>Plan</think>\n\nAnswer", (
                f"failed for {provider}"
            )

    def test_call_llm_explicit_sanitize_thinking_true(self):
        message = types.SimpleNamespace(
            role="assistant",
            content="<think>Plan</think>\n\nAnswer",
            tool_calls=None,
        )
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message, finish_reason="stop")]
        )

        with patch("litellm.completion", return_value=response):
            msg, *_ = call_llm(
                "https://openrouter.ai/api/v1",
                "my-model",
                [{"role": "user", "content": "hi"}],
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="openrouter",
                api_key="sk-test",
                sanitize_thinking=True,
            )

        assert msg.content == "Answer"

    def test_strip_leaked_think_head_basic(self):
        text = "</think>\n\nHere is the answer."
        assert _strip_leaked_think_head(text) == "Here is the answer."

    def test_strip_leaked_think_head_with_leading_whitespace(self):
        text = "  \n</think>\n\nAnswer."
        assert _strip_leaked_think_head(text) == "Answer."

    def test_strip_leaked_think_head_preserves_mid_content_tag(self):
        """Narrow always-on strip only touches the head; mid-document tags need --sanitize-thinking."""
        text = "Some preface.\n</think>\nMore content."
        assert _strip_leaked_think_head(text) == text

    def test_strip_leaked_think_head_preserves_inline_tag(self):
        text = "Use `</think>` to close the example tag."
        assert _strip_leaked_think_head(text) == text

    def test_strip_leaked_think_head_preserves_inline_block(self):
        text = "<think>Plan</think>\n\nAnswer"
        assert _strip_leaked_think_head(text) == text

    def test_strip_leaked_think_head_preserves_fenced_code(self):
        """A </think> line inside a fenced code block must survive — it's a literal example."""
        text = (
            "Here's an example of the tag:\n"
            "```xml\n"
            "<think>\n"
            "</think>\n"
            "```\n"
            "That's how it works."
        )
        assert _strip_leaked_think_head(text) == text

    def test_strip_leaked_channel_head_basic(self):
        text = "<|channel>thought\n<channel|>There are 24 words."
        assert _strip_leaked_channel_head(text) == "There are 24 words."

    def test_strip_leaked_channel_head_with_leading_whitespace(self):
        text = "  \n<|channel>final\n<channel|>\nAnswer."
        assert _strip_leaked_channel_head(text) == "Answer."

    def test_strip_leaked_channel_head_preserves_mid_content_marker(self):
        text = "Example:\n<|channel>thought\n<channel|>Answer."
        assert _strip_leaked_channel_head(text) == text

    def test_call_llm_strips_leaked_close_tag_without_sanitize_flag(self):
        """A bare leading </think> is always stripped, even with sanitize_thinking off."""
        message = types.SimpleNamespace(
            role="assistant",
            content="</think>\n\nHere are your services.",
            tool_calls=None,
        )
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message, finish_reason="stop")]
        )

        with patch("litellm.completion", return_value=response):
            msg, *_ = call_llm(
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
                api_key="sk-test",
            )

        assert msg.content == "Here are your services."

    def test_call_llm_strips_leaked_channel_head_without_sanitize_flag(self):
        """A bare leading channel marker is always stripped, even with sanitize_thinking off."""
        message = types.SimpleNamespace(
            role="assistant",
            content="<|channel>thought\n<channel|>There are 24 words.",
            tool_calls=None,
        )
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message, finish_reason="stop")]
        )

        with patch("litellm.completion", return_value=response):
            msg, *_ = call_llm(
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
                api_key="sk-test",
            )

        assert msg.content == "There are 24 words."

    def test_call_llm_preserves_inline_literal_think_tag(self):
        message = types.SimpleNamespace(
            role="assistant",
            content="Use `</think>` to close the example tag.",
            tool_calls=None,
        )
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message, finish_reason="stop")]
        )

        with patch("litellm.completion", return_value=response):
            msg, *_ = call_llm(
                "http://localhost:8080/v1",
                "my-model",
                [{"role": "user", "content": "Show the literal tag"}],
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="generic",
                api_key="sk-test",
                sanitize_thinking=True,
            )

        assert msg.content == "Use `</think>` to close the example tag."


# ---------------------------------------------------------------------------
# Model ID normalization (double-prefix guard)
# ---------------------------------------------------------------------------


class TestModelNormalization:
    def _mock_response(self):
        choice = MagicMock()
        choice.message = MagicMock(content="ok", tool_calls=None)
        choice.finish_reason = "stop"
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def test_bare_model_id(self):
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                None,
                "zai-org/GLM-5.2",
                [],
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="huggingface",
                api_key="hf_test",
            )
            assert mock_comp.call_args[1]["model"] == "huggingface/zai-org/GLM-5.2"

    def test_already_prefixed_no_double(self):
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                None,
                "huggingface/zai-org/GLM-5.2",
                [],
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="huggingface",
                api_key="hf_test",
            )
            assert mock_comp.call_args[1]["model"] == "huggingface/zai-org/GLM-5.2"

    def test_openrouter_already_prefixed_no_double(self):
        # If user passes "openrouter/openrouter/free" (full LiteLLM prefix
        # already included), the result should still be "openrouter/openrouter/free".
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                None,
                "openrouter/openrouter/free",
                [],
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="openrouter",
                api_key="sk_or_test_key",
            )
            assert mock_comp.call_args[1]["model"] == "openrouter/openrouter/free"


# ---------------------------------------------------------------------------
# CLI validation
# ---------------------------------------------------------------------------


class TestCLIValidation:
    def test_huggingface_requires_model(self, monkeypatch):
        from swival import agent, config

        monkeypatch.setattr(config, "load_config", lambda _: {})

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--provider",
                "huggingface",
            ],
        )
        monkeypatch.setenv("HF_TOKEN", "hf_test")
        with pytest.raises(SystemExit) as exc_info:
            agent.main()
        assert exc_info.value.code == 2

    def test_openrouter_requires_model(self, monkeypatch):
        from swival import agent, config

        monkeypatch.setattr(config, "load_config", lambda _: {})
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--provider",
                "openrouter",
            ],
        )
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk_or_test")
        with pytest.raises(SystemExit) as exc_info:
            agent.main()
        assert exc_info.value.code == 2

    def test_huggingface_model_without_slash(self, monkeypatch):
        from swival import agent, config

        monkeypatch.setattr(config, "load_config", lambda _: {})

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--provider",
                "huggingface",
                "--model",
                "badname",
            ],
        )
        monkeypatch.setenv("HF_TOKEN", "hf_test")
        with pytest.raises(SystemExit) as exc_info:
            agent.main()
        assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------


class TestAPIKeyResolution:
    def _make_args_and_run(self, monkeypatch, cli_api_key=None, env_token=None):
        """Helper: run main() with given api-key/env and capture the api_key used."""
        from swival import agent

        argv = [
            "agent",
            "hello",
            "--provider",
            "huggingface",
            "--model",
            "org/model",
        ]
        if cli_api_key:
            argv.extend(["--api-key", cli_api_key])

        monkeypatch.setattr(sys, "argv", argv)
        if env_token:
            monkeypatch.setenv("HF_TOKEN", env_token)
        else:
            monkeypatch.delenv("HF_TOKEN", raising=False)

        captured_key = {}

        def fake_call_llm(*args, **kwargs):
            captured_key["api_key"] = kwargs.get("api_key")
            msg = types.SimpleNamespace(
                content="done", tool_calls=None, role="assistant"
            )
            msg.get = lambda key, default=None: getattr(msg, key, default)
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        agent.main()
        return captured_key.get("api_key")

    def test_cli_api_key_takes_precedence(self, monkeypatch, tmp_path):
        from swival import agent

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--provider",
                "huggingface",
                "--model",
                "org/model",
                "--api-key",
                "hf_cli",
                "--no-system-prompt",
                "--base-dir",
                str(tmp_path),
            ],
        )
        monkeypatch.setenv("HF_TOKEN", "hf_env")

        captured = {}

        def fake_call_llm(*args, **kwargs):
            captured["api_key"] = kwargs.get("api_key")
            msg = types.SimpleNamespace(
                content="done", tool_calls=None, role="assistant"
            )
            msg.get = lambda key, default=None: getattr(msg, key, default)
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        agent.main()
        assert captured["api_key"] == "hf_cli"

    def test_env_var_used_when_no_cli_key(self, monkeypatch, tmp_path):
        from swival import agent, config

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--provider",
                "huggingface",
                "--model",
                "org/model",
                "--no-system-prompt",
                "--base-dir",
                str(tmp_path),
            ],
        )
        monkeypatch.setenv("HF_TOKEN", "hf_env")
        monkeypatch.setattr(config, "load_config", lambda *a, **kw: {})

        captured = {}

        def fake_call_llm(*args, **kwargs):
            captured["api_key"] = kwargs.get("api_key")
            msg = types.SimpleNamespace(
                content="done", tool_calls=None, role="assistant"
            )
            msg.get = lambda key, default=None: getattr(msg, key, default)
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        agent.main()
        assert captured["api_key"] == "hf_env"

    def test_openrouter_cli_key_takes_precedence(self, monkeypatch, tmp_path):
        from swival import agent

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--provider",
                "openrouter",
                "--model",
                "openrouter/free",
                "--api-key",
                "sk_or_cli",
                "--no-system-prompt",
                "--base-dir",
                str(tmp_path),
            ],
        )
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk_or_env")
        monkeypatch.setattr(
            "swival.model_catalog.catalog_context_length", lambda *a, **k: None
        )

        captured = {}

        def fake_call_llm(*args, **kwargs):
            captured["api_key"] = kwargs.get("api_key")
            msg = types.SimpleNamespace(
                content="done", tool_calls=None, role="assistant"
            )
            msg.get = lambda key, default=None: getattr(msg, key, default)
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        agent.main()
        assert captured["api_key"] == "sk_or_cli"

    def test_openrouter_env_var_used(self, monkeypatch, tmp_path):
        from swival import agent, config

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--provider",
                "openrouter",
                "--model",
                "openrouter/free",
                "--no-system-prompt",
                "--base-dir",
                str(tmp_path),
            ],
        )
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk_or_env")
        monkeypatch.setattr(config, "load_config", lambda *a, **kw: {})
        monkeypatch.setattr(
            "swival.model_catalog.catalog_context_length", lambda *a, **k: None
        )

        captured = {}

        def fake_call_llm(*args, **kwargs):
            captured["api_key"] = kwargs.get("api_key")
            msg = types.SimpleNamespace(
                content="done", tool_calls=None, role="assistant"
            )
            msg.get = lambda key, default=None: getattr(msg, key, default)
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        agent.main()
        assert captured["api_key"] == "sk_or_env"

    def test_openrouter_no_key_errors(self, monkeypatch):
        from swival import agent, config

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--provider",
                "openrouter",
                "--model",
                "openrouter/free",
            ],
        )
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setattr(config, "load_config", lambda *a, **kw: {})
        with pytest.raises(SystemExit) as exc_info:
            agent.main()
        assert exc_info.value.code == 2

    def test_no_key_errors(self, monkeypatch):
        from swival import agent, config

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--provider",
                "huggingface",
                "--model",
                "org/model",
            ],
        )
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr(config, "load_config", lambda *a, **kw: {})
        with pytest.raises(SystemExit) as exc_info:
            agent.main()
        assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# Provider path isolation
# ---------------------------------------------------------------------------


class TestProviderPathIsolation:
    def test_huggingface_never_calls_discover_or_configure(self, monkeypatch, tmp_path):
        from swival import agent

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--provider",
                "huggingface",
                "--model",
                "org/model",
                "--api-key",
                "hf_test",
                "--no-system-prompt",
                "--base-dir",
                str(tmp_path),
            ],
        )

        def boom(*args, **kwargs):
            raise AssertionError("Should not be called for huggingface provider")

        monkeypatch.setattr(agent, "discover_model", boom)
        monkeypatch.setattr(agent, "configure_context", boom)

        def fake_call_llm(*args, **kwargs):
            msg = types.SimpleNamespace(
                content="done", tool_calls=None, role="assistant"
            )
            msg.get = lambda key, default=None: getattr(msg, key, default)
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        agent.main()  # Should not raise

    def test_lmstudio_calls_discover_when_no_model(self, monkeypatch, tmp_path):
        from swival import agent, config

        monkeypatch.setattr(config, "load_config", lambda _: {})
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--no-system-prompt",
                "--base-dir",
                str(tmp_path),
            ],
        )

        discover_called = {"value": False}

        def fake_discover(*args, **kwargs):
            discover_called["value"] = True
            return "test-model", 4096

        monkeypatch.setattr(agent, "discover_model", fake_discover)

        def fake_call_llm(*args, **kwargs):
            msg = types.SimpleNamespace(
                content="done", tool_calls=None, role="assistant"
            )
            msg.get = lambda key, default=None: getattr(msg, key, default)
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        agent.main()
        assert discover_called["value"]

    def test_lmstudio_calls_configure_with_max_context(self, monkeypatch, tmp_path):
        from swival import agent, config

        monkeypatch.setattr(config, "load_config", lambda _: {})
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--model",
                "test-model",
                "--max-context-tokens",
                "8192",
                "--max-output-tokens",
                "1024",
                "--no-system-prompt",
                "--base-dir",
                str(tmp_path),
            ],
        )

        configure_called = {"value": False}

        def fake_configure(*args, **kwargs):
            configure_called["value"] = True

        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr(agent, "configure_context", fake_configure)

        def fake_call_llm(*args, **kwargs):
            msg = types.SimpleNamespace(
                content="done", tool_calls=None, role="assistant"
            )
            msg.get = lambda key, default=None: getattr(msg, key, default)
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        agent.main()
        assert configure_called["value"]


# ---------------------------------------------------------------------------
# Generic provider
# ---------------------------------------------------------------------------


class TestDiscoverGenericContextLength:
    """Reading the context window from an OpenAI-compatible /v1/models list."""

    def test_matches_model_by_id(self, monkeypatch):
        _patch_urlopen_json(
            monkeypatch,
            {
                "data": [
                    {"id": "small", "max_model_len": 8192},
                    {"id": "big", "max_model_len": 262144},
                ]
            },
        )
        assert discover_generic_context_length("http://h/v1", "big", False) == 262144

    def test_single_entry_fallback_when_no_id_match(self, monkeypatch):
        _patch_urlopen_json(
            monkeypatch, {"data": [{"id": "only", "context_length": 4096}]}
        )
        assert discover_generic_context_length("http://h/v1", "default", False) == 4096

    def test_no_fallback_with_multiple_entries(self, monkeypatch):
        _patch_urlopen_json(
            monkeypatch,
            {
                "data": [
                    {"id": "a", "max_model_len": 1},
                    {"id": "b", "max_model_len": 2},
                ]
            },
        )
        assert discover_generic_context_length("http://h/v1", "default", False) is None

    def test_alternate_keys(self, monkeypatch):
        _patch_urlopen_json(
            monkeypatch, {"data": [{"id": "m", "context_window": 32768}]}
        )
        assert discover_generic_context_length("http://h/v1", "m", False) == 32768

    def test_missing_field_returns_none(self, monkeypatch):
        _patch_urlopen_json(monkeypatch, {"data": [{"id": "m"}]})
        assert discover_generic_context_length("http://h/v1", "m", False) is None

    def test_unreachable_server_returns_none(self, monkeypatch):
        _patch_urlopen_error(monkeypatch)
        assert discover_generic_context_length("http://h/v1", "m", False) is None


class TestDiscoverLlamacppContextLength:
    """Reading n_ctx from a llama.cpp server's /props endpoint."""

    def test_reads_default_generation_settings(self, monkeypatch):
        _patch_urlopen_json(
            monkeypatch, {"default_generation_settings": {"n_ctx": 16384}}
        )
        assert discover_llamacpp_context_length("http://h:8080", False) == 16384

    def test_reads_top_level_n_ctx(self, monkeypatch):
        _patch_urlopen_json(monkeypatch, {"n_ctx": 8192})
        assert discover_llamacpp_context_length("http://h:8080", False) == 8192

    def test_missing_returns_none(self, monkeypatch):
        _patch_urlopen_json(monkeypatch, {"model_path": "x.gguf"})
        assert discover_llamacpp_context_length("http://h:8080", False) is None

    def test_unreachable_returns_none(self, monkeypatch):
        _patch_urlopen_error(monkeypatch)
        assert discover_llamacpp_context_length("http://h:8080", False) is None


class TestOpenrouterContextFallback:
    """The OpenRouter catalog fills in the context window when litellm can't."""

    def _resolve(self, monkeypatch, litellm_ctx, catalog_ctx, max_context_tokens=None):
        from swival import agent

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk_or_test")
        monkeypatch.setattr(agent, "_litellm_context_length", lambda s: litellm_ctx)
        seen = {}

        def fake_catalog(provider, model_id, base_url=None, api_key=None, **kw):
            seen["args"] = (provider, model_id, base_url, api_key)
            return catalog_ctx

        monkeypatch.setattr("swival.model_catalog.catalog_context_length", fake_catalog)
        _, _, _, ctx, _ = resolve_provider(
            "openrouter", "qwen/new-model", None, None, max_context_tokens, False
        )
        return ctx, seen

    def test_catalog_fallback_when_litellm_unknown(self, monkeypatch):
        ctx, seen = self._resolve(monkeypatch, litellm_ctx=None, catalog_ctx=262144)
        assert ctx == 262144
        assert seen["args"] == ("openrouter", "qwen/new-model", None, "sk_or_test")

    def test_litellm_value_wins_over_catalog(self, monkeypatch):
        ctx, seen = self._resolve(monkeypatch, litellm_ctx=100000, catalog_ctx=262144)
        assert ctx == 100000
        assert "args" not in seen

    def test_explicit_max_context_tokens_wins(self, monkeypatch):
        ctx, seen = self._resolve(
            monkeypatch, litellm_ctx=None, catalog_ctx=262144, max_context_tokens=50000
        )
        assert ctx == 50000
        assert "args" not in seen

    def test_catalog_miss_leaves_context_unknown(self, monkeypatch):
        ctx, seen = self._resolve(monkeypatch, litellm_ctx=None, catalog_ctx=None)
        assert ctx is None
        assert seen["args"] == ("openrouter", "qwen/new-model", None, "sk_or_test")

    @pytest.mark.parametrize(
        "model, expected",
        [
            ("qwen/new-model", 262144),
            ("openrouter/cypher-alpha:free", 32768),  # OpenRouter's own namespace
            ("openrouter/openrouter/cypher-alpha:free", 32768),  # litellm form
        ],
    )
    def test_catalog_matches_every_id_spelling(self, monkeypatch, model, expected):
        from swival import agent

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk_or_test")
        monkeypatch.setattr(agent, "_litellm_context_length", lambda s: None)
        _patch_urlopen_json(
            monkeypatch,
            {
                "data": [
                    {"id": "qwen/new-model", "context_length": 262144},
                    {"id": "openrouter/cypher-alpha:free", "context_length": 32768},
                ]
            },
        )
        _, _, _, ctx, _ = resolve_provider("openrouter", model, None, None, None, False)
        assert ctx == expected

    def test_litellm_probe_uses_call_time_key(self, monkeypatch):
        from swival import agent

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk_or_test")
        calls = []

        def fake_litellm(model_str):
            calls.append(model_str)
            return 40960

        monkeypatch.setattr(agent, "_litellm_context_length", fake_litellm)
        monkeypatch.setattr(
            "swival.model_catalog.catalog_context_length",
            lambda *a, **k: pytest.fail("catalog consulted despite litellm hit"),
        )
        _, _, _, ctx, _ = resolve_provider(
            "openrouter", "openrouter/auto", None, None, None, False
        )
        assert ctx == 40960
        # A single leading "openrouter/" is OpenRouter's namespace, exactly
        # as _resolve_model_str reads it when building the completion call.
        assert calls == ["openrouter/openrouter/auto"]


class TestGenericProviderRouting:
    """Verify call_llm routing for the generic provider."""

    @pytest.fixture(autouse=True)
    def _offline_context_probe(self, monkeypatch):
        """Stub the context-window probe so routing tests never hit a server."""
        monkeypatch.setattr(agent, "discover_generic_context_length", lambda *a: None)

    def _mock_response(self):
        choice = MagicMock()
        choice.message = MagicMock(content="ok", tool_calls=None)
        choice.finish_reason = "stop"
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def test_generic_routing_basic(self):
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "http://localhost:8080/v1",
                "my-model",
                [],
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="generic",
                api_key="sk-test",
            )
            mock_comp.assert_called_once()
            kwargs = mock_comp.call_args[1]
            assert kwargs["model"] == "openai/my-model"
            assert kwargs["api_base"] == "http://localhost:8080/v1"
            assert kwargs["api_key"] == "sk-test"

    def test_generic_passes_base_url_unchanged(self):
        """call_llm passes the base URL as-is; normalization is in resolve_provider."""
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "http://host:9000/v1",
                "m",
                [],
                100,
                None,
                None,
                None,
                None,
                False,
                provider="generic",
                api_key=None,
            )
            assert mock_comp.call_args[1]["api_base"] == "http://host:9000/v1"

    def test_resolve_provider_appends_v1_when_missing(self):
        """resolve_provider normalizes generic URLs by appending /v1."""
        _, api_base, _, _, _ = resolve_provider(
            "generic", "m", None, "http://host:9000", None, False
        )
        assert api_base == "http://host:9000/v1"

    def test_resolve_provider_no_double_v1(self):
        _, api_base, _, _, _ = resolve_provider(
            "generic", "m", None, "http://host:9000/v1", None, False
        )
        assert api_base == "http://host:9000/v1"

    def test_resolve_provider_trailing_slash_stripped(self):
        _, api_base, _, _, _ = resolve_provider(
            "generic", "m", None, "http://host:9000/", None, False
        )
        assert api_base == "http://host:9000/v1"

    def test_resolve_provider_trailing_slash_with_v1(self):
        _, api_base, _, _, _ = resolve_provider(
            "generic", "m", None, "http://host:9000/v1/", None, False
        )
        assert api_base == "http://host:9000/v1"

    def test_resolve_provider_preserves_non_v1_version(self, monkeypatch):
        """A URL ending in a /vN segment (Z.AI's /v4) is left untouched."""
        monkeypatch.setattr(agent, "discover_generic_context_length", lambda *a: None)
        _, api_base, _, _, _ = resolve_provider(
            "generic", "m", None, "https://api.z.ai/api/paas/v4", None, False
        )
        assert api_base == "https://api.z.ai/api/paas/v4"

    def test_resolve_provider_preserves_non_v1_version_trailing_slash(
        self, monkeypatch
    ):
        monkeypatch.setattr(agent, "discover_generic_context_length", lambda *a: None)
        _, api_base, _, _, _ = resolve_provider(
            "generic", "m", None, "https://api.z.ai/api/paas/v4/", None, False
        )
        assert api_base == "https://api.z.ai/api/paas/v4"

    def test_resolve_provider_discovers_context_length(self, monkeypatch):
        """When max_context_tokens is unset, generic reads it from /v1/models."""
        monkeypatch.setattr(
            agent,
            "discover_generic_context_length",
            lambda api_base, model_id, verbose: 262144,
        )
        _, _, _, context_length, _ = resolve_provider(
            "generic", "m", None, "http://host:9000", None, False
        )
        assert context_length == 262144

    def test_resolve_provider_explicit_context_skips_discovery(self, monkeypatch):
        """An explicit max_context_tokens must not trigger discovery."""
        called = False

        def _spy(*a, **k):
            nonlocal called
            called = True
            return 999

        monkeypatch.setattr(agent, "discover_generic_context_length", _spy)
        _, _, _, context_length, _ = resolve_provider(
            "generic", "m", None, "http://host:9000", 4096, False
        )
        assert context_length == 4096
        assert called is False

    def test_generic_no_key_uses_none_placeholder(self):
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "http://host:9000",
                "m",
                [],
                100,
                None,
                None,
                None,
                None,
                False,
                provider="generic",
                api_key=None,
            )
            assert mock_comp.call_args[1]["api_key"] == "none"

    def test_generic_key_passed_through(self):
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "http://host:9000",
                "m",
                [],
                100,
                None,
                None,
                None,
                None,
                False,
                provider="generic",
                api_key="sk-real",
            )
            assert mock_comp.call_args[1]["api_key"] == "sk-real"


class TestAppleFoundationModelsProvider:
    """Apple Foundation Models local server routing and tool-schema sanitizing."""

    def test_model_str_routes_as_openai(self):
        assert _resolve_model_str("applefm", "system") == "openai/system"

    def test_resolve_defaults_base_url(self):
        _, api_base, _, _, _ = resolve_provider(
            "applefm", "system", None, None, None, False
        )
        assert api_base == "http://127.0.0.1:1976/v1"

    def test_resolve_normalizes_custom_base_url(self):
        _, api_base, _, _, _ = resolve_provider(
            "applefm", "system", None, "http://host:1976", None, False
        )
        assert api_base == "http://host:1976/v1"

    def test_defaults_to_pcc_when_model_omitted(self, monkeypatch):
        monkeypatch.setattr(
            agent, "discover_generic_context_length", lambda *a, **k: None
        )
        model_id, _, _, ctx, _ = resolve_provider(
            "applefm", None, None, None, None, False
        )
        assert model_id == "pcc"
        assert ctx == 32768

    def test_on_device_context_default(self, monkeypatch):
        monkeypatch.setattr(
            agent, "discover_generic_context_length", lambda *a, **k: None
        )
        _, _, _, ctx, _ = resolve_provider("applefm", "system", None, None, None, False)
        assert ctx == 4096

    def test_pcc_context_default(self, monkeypatch):
        monkeypatch.setattr(
            agent, "discover_generic_context_length", lambda *a, **k: None
        )
        _, _, _, ctx, _ = resolve_provider("applefm", "pcc", None, None, None, False)
        assert ctx == 32768

    def test_explicit_context_overrides_default(self):
        _, _, _, ctx, _ = resolve_provider("applefm", "system", None, None, 8000, False)
        assert ctx == 8000


class TestAppleFoundationModelsToolSanitizer:
    """sanitize_tools_for_applefm drops what Apple's GenerationSchema can't read."""

    @staticmethod
    def _tool(name, params):
        return {"type": "function", "function": {"name": name, "parameters": params}}

    def test_injects_missing_required(self):
        tool = self._tool("noargs", {"type": "object", "properties": {}})
        kept, dropped = sanitize_tools_for_applefm([tool])
        assert dropped == []
        assert kept[0]["function"]["parameters"]["required"] == []

    def test_does_not_mutate_original(self):
        tool = self._tool("noargs", {"type": "object", "properties": {}})
        sanitize_tools_for_applefm([tool])
        assert "required" not in tool["function"]["parameters"]

    def test_keeps_scalars_and_scalar_arrays(self):
        tool = self._tool(
            "ok",
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "count": {"type": "integer"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name"],
            },
        )
        kept, dropped = sanitize_tools_for_applefm([tool])
        assert dropped == [] and len(kept) == 1

    def test_drops_nested_object(self):
        tool = self._tool(
            "nested",
            {
                "type": "object",
                "properties": {"cfg": {"type": "object", "properties": {}}},
                "required": ["cfg"],
            },
        )
        _, dropped = sanitize_tools_for_applefm([tool])
        assert dropped == ["nested"]

    def test_drops_array_of_objects(self):
        tool = self._tool(
            "batch",
            {
                "type": "object",
                "properties": {"files": {"type": "array", "items": {"type": "object"}}},
                "required": ["files"],
            },
        )
        _, dropped = sanitize_tools_for_applefm([tool])
        assert dropped == ["batch"]

    def test_drops_oneof_union(self):
        tool = self._tool(
            "todo",
            {
                "type": "object",
                "properties": {
                    "tasks": {"oneOf": [{"type": "string"}, {"type": "array"}]}
                },
                "required": ["tasks"],
            },
        )
        _, dropped = sanitize_tools_for_applefm([tool])
        assert dropped == ["todo"]

    def test_missing_parameters_gets_empty_object(self):
        tool = {"type": "function", "function": {"name": "bare"}}
        kept, dropped = sanitize_tools_for_applefm([tool])
        assert dropped == []
        params = kept[0]["function"]["parameters"]
        assert params == {"type": "object", "properties": {}, "required": []}


class TestGenericProviderValidation:
    """CLI-level validation for the generic provider."""

    def test_generic_requires_model(self, monkeypatch):
        from swival import agent, config

        monkeypatch.setattr(config, "load_config", lambda _: {})
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--provider",
                "generic",
                "--base-url",
                "http://localhost:8080",
            ],
        )
        with pytest.raises(SystemExit) as exc_info:
            agent.main()
        assert exc_info.value.code == 2

    def test_generic_works_without_api_key(self, monkeypatch, tmp_path):
        from swival import agent, config

        monkeypatch.setattr(config, "load_config", lambda *a, **kw: {})

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--provider",
                "generic",
                "--model",
                "my-model",
                "--base-url",
                "http://localhost:8080",
                "--no-system-prompt",
                "--base-dir",
                str(tmp_path),
            ],
        )
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        captured = {}

        def fake_call_llm(*args, **kwargs):
            captured["api_key"] = kwargs.get("api_key")
            msg = types.SimpleNamespace(
                content="done", tool_calls=None, role="assistant"
            )
            msg.get = lambda key, default=None: getattr(msg, key, default)
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        agent.main()
        assert captured["api_key"] is None

    def test_generic_picks_up_openai_api_key_env(self, monkeypatch, tmp_path):
        from swival import agent, config

        monkeypatch.setattr(config, "load_config", lambda *a, **kw: {})

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--provider",
                "generic",
                "--model",
                "my-model",
                "--base-url",
                "http://localhost:8080",
                "--no-system-prompt",
                "--base-dir",
                str(tmp_path),
            ],
        )
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

        captured = {}

        def fake_call_llm(*args, **kwargs):
            captured["api_key"] = kwargs.get("api_key")
            msg = types.SimpleNamespace(
                content="done", tool_calls=None, role="assistant"
            )
            msg.get = lambda key, default=None: getattr(msg, key, default)
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        agent.main()
        assert captured["api_key"] == "sk-from-env"

    def test_generic_cli_key_takes_precedence(self, monkeypatch, tmp_path):
        from swival import agent

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--provider",
                "generic",
                "--model",
                "my-model",
                "--base-url",
                "http://localhost:8080",
                "--api-key",
                "sk-cli",
                "--no-system-prompt",
                "--base-dir",
                str(tmp_path),
            ],
        )
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")

        captured = {}

        def fake_call_llm(*args, **kwargs):
            captured["api_key"] = kwargs.get("api_key")
            msg = types.SimpleNamespace(
                content="done", tool_calls=None, role="assistant"
            )
            msg.get = lambda key, default=None: getattr(msg, key, default)
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        agent.main()
        assert captured["api_key"] == "sk-cli"

    def test_generic_never_calls_discover_or_configure(self, monkeypatch, tmp_path):
        from swival import agent

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--provider",
                "generic",
                "--model",
                "my-model",
                "--base-url",
                "http://localhost:8080",
                "--no-system-prompt",
                "--base-dir",
                str(tmp_path),
            ],
        )
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        def boom(*args, **kwargs):
            raise AssertionError("Should not be called for generic provider")

        monkeypatch.setattr(agent, "discover_model", boom)
        monkeypatch.setattr(agent, "configure_context", boom)

        def fake_call_llm(*args, **kwargs):
            msg = types.SimpleNamespace(
                content="done", tool_calls=None, role="assistant"
            )
            msg.get = lambda key, default=None: getattr(msg, key, default)
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        agent.main()  # Should not raise


# ---------------------------------------------------------------------------
# Google provider
# ---------------------------------------------------------------------------


class TestGoogleProviderRouting:
    """Verify google provider routes through the OpenAI-compatible endpoint."""

    _GOOGLE_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"

    def _mock_response(self):
        choice = MagicMock()
        choice.message = MagicMock(content="ok", tool_calls=None)
        choice.finish_reason = "stop"
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def test_google_routes_through_openai_compat(self):
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            _, api_base, api_key, _, llm_kwargs = resolve_provider(
                "google", "gemini-3-flash", "gemini-key", None, None, False
            )
            assert llm_kwargs["provider"] == "generic"
            assert api_base == self._GOOGLE_OPENAI_BASE
            call_llm(
                api_base,
                "gemini-3-flash",
                [],
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider=llm_kwargs["provider"],
                api_key=api_key,
            )
            kwargs = mock_comp.call_args[1]
            assert kwargs["model"] == "openai/gemini-3-flash"
            assert kwargs["api_key"] == "gemini-key"
            assert kwargs["api_base"] == self._GOOGLE_OPENAI_BASE

    def test_google_custom_base_url_overrides_default(self):
        _, api_base, _, _, llm_kwargs = resolve_provider(
            "google", "gemini-3-flash", "k", "https://custom.example.com", None, False
        )
        assert llm_kwargs["provider"] == "generic"
        assert api_base == "https://custom.example.com"


class TestGoogleProviderValidation:
    """CLI-level validation for google provider."""

    def test_google_requires_model(self, monkeypatch):
        from swival import agent, config

        monkeypatch.setattr(config, "load_config", lambda _: {})
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--provider",
                "google",
            ],
        )
        monkeypatch.setenv("OPENAI_API_KEY", "gemini-env")
        with pytest.raises(SystemExit) as exc_info:
            agent.main()
        assert exc_info.value.code == 2

    def test_google_no_base_url_by_default(self, monkeypatch, tmp_path):
        from swival import agent, config

        monkeypatch.setattr(config, "load_config", lambda *a, **kw: {})

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--provider",
                "google",
                "--model",
                "gemini-3-flash",
                "--no-system-prompt",
                "--base-dir",
                str(tmp_path),
            ],
        )
        monkeypatch.setenv("OPENAI_API_KEY", "gemini-env")

        captured = {}

        def fake_call_llm(*args, **kwargs):
            captured["base_url"] = args[0]
            captured["api_key"] = kwargs.get("api_key")
            captured["provider"] = kwargs.get("provider")
            msg = types.SimpleNamespace(
                content="done", tool_calls=None, role="assistant"
            )
            msg.get = lambda key, default=None: getattr(msg, key, default)
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        agent.main()
        assert (
            captured["base_url"]
            == "https://generativelanguage.googleapis.com/v1beta/openai"
        )
        assert captured["api_key"] == "gemini-env"
        assert captured["provider"] == "generic"

    def test_google_uses_gemini_api_key_env(self):
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-env")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        try:
            _, api_base, api_key, _, llm_kwargs = resolve_provider(
                "google", "gemini-3-flash", None, None, None, False
            )
        finally:
            monkeypatch.undo()
        assert api_base == "https://generativelanguage.googleapis.com/v1beta/openai"
        assert api_key == "gemini-env"
        assert llm_kwargs["provider"] == "generic"

    def test_google_never_calls_discover_or_configure(self, monkeypatch, tmp_path):
        from swival import agent

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--provider",
                "google",
                "--model",
                "gemini-3-flash",
                "--api-key",
                "gemini-cli",
                "--no-system-prompt",
                "--base-dir",
                str(tmp_path),
            ],
        )

        def boom(*args, **kwargs):
            raise AssertionError("Should not be called for google provider")

        monkeypatch.setattr(agent, "discover_model", boom)
        monkeypatch.setattr(agent, "configure_context", boom)

        def fake_call_llm(*args, **kwargs):
            msg = types.SimpleNamespace(
                content="done", tool_calls=None, role="assistant"
            )
            msg.get = lambda key, default=None: getattr(msg, key, default)
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        agent.main()


# ---------------------------------------------------------------------------
# ChatGPT provider — call_llm routing
# ---------------------------------------------------------------------------


class TestChatGPTRouting:
    """Verify that call_llm passes the right model string and kwargs for chatgpt."""

    def _mock_response(self):
        choice = MagicMock()
        choice.message = MagicMock(content="ok", tool_calls=None)
        choice.finish_reason = "stop"
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def test_chatgpt_routing_basic(self):
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                None,
                "gpt-5.5",
                [],
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="chatgpt",
                api_key=None,
            )
            mock_comp.assert_called_once()
            kwargs = mock_comp.call_args[1]
            assert kwargs["model"] == "chatgpt/gpt-5.5"
            assert kwargs["extra_headers"]["originator"] == "swival"
            assert kwargs["extra_headers"]["user-agent"].startswith("Swival/")
            assert "api_key" not in kwargs
            assert "api_base" not in kwargs

    def test_chatgpt_identity_hooks_are_scoped_to_call(self, monkeypatch):
        monkeypatch.setenv("CHATGPT_ORIGINATOR", "previous-originator")
        monkeypatch.setenv("CHATGPT_USER_AGENT", "previous-ua")
        monkeypatch.setenv("CHATGPT_DEFAULT_INSTRUCTIONS", "previous-instructions")

        observed = {}

        def fake_completion(**_kwargs):
            from litellm.llms.chatgpt.responses import (
                transformation as responses_transform,
            )

            headers = responses_transform.get_chatgpt_default_headers("tok", None)
            observed["originator"] = headers["originator"]
            observed["user_agent"] = headers["user-agent"]
            observed["instructions"] = (
                responses_transform.get_chatgpt_default_instructions()
            )
            observed["env_user_agent"] = os.environ.get("CHATGPT_USER_AGENT")
            return self._mock_response()

        with patch("litellm.completion", side_effect=fake_completion):
            call_llm(
                None,
                "gpt-5.5",
                [],
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="chatgpt",
                api_key=None,
                user_agent="SwivalTest/1.2.3",
            )

        assert observed == {
            "originator": "swival",
            "user_agent": "SwivalTest/1.2.3",
            "instructions": "You are Swival, a coding agent.",
            "env_user_agent": "previous-ua",
        }
        assert os.environ["CHATGPT_ORIGINATOR"] == "previous-originator"
        assert os.environ["CHATGPT_USER_AGENT"] == "previous-ua"
        assert os.environ["CHATGPT_DEFAULT_INSTRUCTIONS"] == "previous-instructions"

    def test_chatgpt_with_explicit_key(self):
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                None,
                "gpt-5.5",
                [],
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="chatgpt",
                api_key="bearer-token-123",
            )
            mock_comp.assert_called_once()
            kwargs = mock_comp.call_args[1]
            assert kwargs["api_key"] == "bearer-token-123"

    def test_chatgpt_with_base_url(self):
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "https://proxy.example.com",
                "gpt-5.5",
                [],
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="chatgpt",
                api_key=None,
            )
            mock_comp.assert_called_once()
            kwargs = mock_comp.call_args[1]
            assert kwargs["api_base"] == "https://proxy.example.com"

    def test_chatgpt_drops_unsupported_params(self):
        """ChatGPT backend doesn't support top_p, seed, or tool_choice."""
        tools = [{"type": "function", "function": {"name": "test"}}]
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                None,
                "gpt-5.5",
                [],
                100,
                0.7,
                0.9,
                42,
                tools,
                False,
                provider="chatgpt",
                api_key=None,
            )
            mock_comp.assert_called_once()
            kwargs = mock_comp.call_args[1]
            assert kwargs["temperature"] == 0.7
            assert kwargs["tools"] == tools
            assert "top_p" not in kwargs
            assert "seed" not in kwargs
            assert "tool_choice" not in kwargs

    def test_chatgpt_registers_new_gpt5_as_responses_model(self):
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                None,
                "gpt-5.5",
                [],
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="chatgpt",
                api_key=None,
            )
            kwargs = mock_comp.call_args[1]
            assert kwargs["model"] == "chatgpt/gpt-5.5"

        import litellm

        info = litellm.get_model_info("chatgpt/gpt-5.5")
        assert info["mode"] == "responses"
        assert info["litellm_provider"] == "chatgpt"

    def test_chatgpt_registers_gpt5_absent_from_cost_map(self):
        """A brand-new Codex model missing from the bundled cost map must still
        register as a Responses model, or litellm routes it to Chat Completions
        and it hits a Cloudflare challenge.  gpt-5.6-terra reproduces the bug:
        its bare openai key is not in the registry, so the old hardcoded
        chatgpt/gpt-5.5 fallback (also absent) made registration bail out."""
        import litellm
        from swival.agent import _ensure_chatgpt_responses_model_registered

        model_str = "chatgpt/gpt-5.6-terra"
        assert model_str not in litellm.model_cost
        assert "gpt-5.6-terra" not in litellm.model_cost

        _ensure_chatgpt_responses_model_registered(litellm, model_str)

        info = litellm.model_cost[model_str]
        assert info["mode"] == "responses"
        assert info["litellm_provider"] == "chatgpt"
        assert info["max_input_tokens"] > 0


# ---------------------------------------------------------------------------
# ChatGPT model normalization (double-prefix guard)
# ---------------------------------------------------------------------------


class TestChatGPTModelNormalization:
    def _mock_response(self):
        choice = MagicMock()
        choice.message = MagicMock(content="ok", tool_calls=None)
        choice.finish_reason = "stop"
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def test_chatgpt_no_double_prefix(self):
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                None,
                "chatgpt/gpt-5.5",
                [],
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="chatgpt",
                api_key=None,
            )
            assert mock_comp.call_args[1]["model"] == "chatgpt/gpt-5.5"

    def test_chatgpt_bare_model_id(self):
        """Bare model (no prefix) gets chatgpt/ prepended."""
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                None,
                "gpt-5.5",
                [],
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="chatgpt",
                api_key=None,
            )
            assert mock_comp.call_args[1]["model"] == "chatgpt/gpt-5.5"

    def test_chatgpt_double_prefix_stripped(self):
        """chatgpt/chatgpt/model collapses to chatgpt/model."""
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                None,
                "chatgpt/chatgpt/gpt-5.5",
                [],
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="chatgpt",
                api_key=None,
            )
            assert mock_comp.call_args[1]["model"] == "chatgpt/gpt-5.5"


# ---------------------------------------------------------------------------
# ChatGPT CLI validation
# ---------------------------------------------------------------------------


class TestChatGPTCLIValidation:
    def test_chatgpt_requires_model(self, monkeypatch):
        from swival import agent, config

        monkeypatch.setattr(config, "load_config", lambda _: {})
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--provider",
                "chatgpt",
            ],
        )
        with pytest.raises(SystemExit) as exc_info:
            agent.main()
        assert exc_info.value.code == 2

    def test_chatgpt_no_key_ok(self, monkeypatch, tmp_path):
        from swival import agent, config

        monkeypatch.setattr(config, "load_config", lambda *a, **kw: {})

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--provider",
                "chatgpt",
                "--model",
                "gpt-5.5",
                "--no-system-prompt",
                "--base-dir",
                str(tmp_path),
            ],
        )

        captured = {}

        def fake_call_llm(*args, **kwargs):
            captured["api_key"] = kwargs.get("api_key")
            msg = types.SimpleNamespace(
                content="done", tool_calls=None, role="assistant"
            )
            msg.get = lambda key, default=None: getattr(msg, key, default)
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        agent.main()
        assert captured["api_key"] is None

    def test_chatgpt_never_calls_discover_or_configure(self, monkeypatch, tmp_path):
        from swival import agent

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--provider",
                "chatgpt",
                "--model",
                "gpt-5.5",
                "--no-system-prompt",
                "--base-dir",
                str(tmp_path),
            ],
        )

        def boom(*args, **kwargs):
            raise AssertionError("Should not be called for chatgpt provider")

        monkeypatch.setattr(agent, "discover_model", boom)
        monkeypatch.setattr(agent, "configure_context", boom)

        def fake_call_llm(*args, **kwargs):
            msg = types.SimpleNamespace(
                content="done", tool_calls=None, role="assistant"
            )
            msg.get = lambda key, default=None: getattr(msg, key, default)
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        agent.main()  # Should not raise


class TestChatGPTLogoutCLI:
    def test_logout_deletes_token_file(self, monkeypatch, tmp_path, capsys):
        from swival import agent

        token_dir = tmp_path / "tokens"
        token_dir.mkdir()
        auth_path = token_dir / "auth.json"
        auth_path.write_text('{"access_token":"test"}', encoding="utf-8")
        monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(token_dir))
        monkeypatch.delenv("CHATGPT_AUTH_FILE", raising=False)
        monkeypatch.setattr(sys, "argv", ["agent", "--logout"])

        with pytest.raises(SystemExit) as exc_info:
            agent.main()

        assert exc_info.value.code == 0
        assert not auth_path.exists()
        assert f"Deleted ChatGPT OAuth tokens: {auth_path}" in capsys.readouterr().err

    def test_logout_noops_when_token_file_missing(self, monkeypatch, tmp_path, capsys):
        from swival import agent

        monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path / "missing"))
        monkeypatch.delenv("CHATGPT_AUTH_FILE", raising=False)
        monkeypatch.setattr(sys, "argv", ["agent", "--logout"])

        with pytest.raises(SystemExit) as exc_info:
            agent.main()

        assert exc_info.value.code == 0
        assert (
            "No stored ChatGPT or GitHub Copilot credentials found."
            in capsys.readouterr().err
        )

    def test_logout_uses_custom_auth_file_env(self, monkeypatch, tmp_path):
        from swival import agent

        token_dir = tmp_path / "tokens"
        token_dir.mkdir()
        auth_path = token_dir / "custom-auth.json"
        auth_path.write_text('{"access_token":"test"}', encoding="utf-8")
        default_path = token_dir / "auth.json"
        default_path.write_text('{"access_token":"keep"}', encoding="utf-8")
        monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(token_dir))
        monkeypatch.setenv("CHATGPT_AUTH_FILE", "custom-auth.json")
        monkeypatch.setattr(sys, "argv", ["agent", "--logout"])

        with pytest.raises(SystemExit) as exc_info:
            agent.main()

        assert exc_info.value.code == 0
        assert not auth_path.exists()
        assert default_path.exists()

    def test_logout_uses_absolute_auth_file_env(self, monkeypatch, tmp_path):
        from swival import agent

        token_dir = tmp_path / "tokens"
        token_dir.mkdir()
        auth_path = tmp_path / "elsewhere" / "auth.json"
        auth_path.parent.mkdir()
        auth_path.write_text('{"access_token":"test"}', encoding="utf-8")
        monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(token_dir))
        monkeypatch.setenv("CHATGPT_AUTH_FILE", str(auth_path))
        monkeypatch.setattr(sys, "argv", ["agent", "--logout"])

        with pytest.raises(SystemExit) as exc_info:
            agent.main()

        assert exc_info.value.code == 0
        assert not auth_path.exists()

    def test_logout_does_not_load_config(self, monkeypatch, tmp_path):
        from swival import agent, config

        def fail_load_config(*args, **kwargs):
            raise AssertionError("logout should exit before config loading")

        monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
        monkeypatch.delenv("CHATGPT_AUTH_FILE", raising=False)
        monkeypatch.setattr(config, "load_config", fail_load_config)
        monkeypatch.setattr(sys, "argv", ["agent", "--logout"])

        with pytest.raises(SystemExit) as exc_info:
            agent.main()

        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# ChatGPT resolve_provider direct unit tests
# ---------------------------------------------------------------------------


class TestResolveProviderChatGPT:
    """Direct unit tests for resolve_provider() with provider='chatgpt'."""

    def test_returns_model_id(self):
        model_id, _, _, _, _ = resolve_provider(
            "chatgpt", "gpt-5.5", None, None, None, False
        )
        assert model_id == "gpt-5.5"

    def test_no_model_raises(self):
        from swival.config import ConfigError

        with pytest.raises(ConfigError, match="--model is required") as excinfo:
            resolve_provider("chatgpt", None, None, None, None, False)

        assert "Available models:" not in str(excinfo.value)
        assert "docs.litellm.ai/docs/providers/chatgpt" in str(excinfo.value)

    def test_resolved_key_none_when_no_key(self):
        _, _, resolved_key, _, _ = resolve_provider(
            "chatgpt", "gpt-5.5", None, None, None, False
        )
        assert resolved_key is None

    def test_resolved_key_passthrough(self):
        _, _, resolved_key, _, _ = resolve_provider(
            "chatgpt", "gpt-5.5", "bearer-xyz", None, None, False
        )
        assert resolved_key == "bearer-xyz"

    def test_api_base_none_by_default(self):
        _, api_base, _, _, _ = resolve_provider(
            "chatgpt", "gpt-5.5", None, None, None, False
        )
        assert api_base is None

    def test_api_base_passthrough(self):
        _, api_base, _, _, _ = resolve_provider(
            "chatgpt", "gpt-5.5", None, "https://proxy.example.com", None, False
        )
        assert api_base == "https://proxy.example.com"

    def test_context_from_litellm_metadata(self):
        _, _, _, context_length, _ = resolve_provider(
            "chatgpt", "gpt-5.5", None, None, None, False
        )
        # Should resolve from litellm metadata (or None if not in cost map).
        # Don't assert a specific value since it depends on litellm version.
        assert context_length is None or (
            isinstance(context_length, int) and context_length > 0
        )

    def test_context_for_new_gpt5_from_bare_metadata(self):
        _, _, _, context_length, _ = resolve_provider(
            "chatgpt", "gpt-5.5", None, None, None, False
        )
        assert isinstance(context_length, int) and context_length > 0

    def test_context_override(self):
        _, _, _, context_length, _ = resolve_provider(
            "chatgpt", "gpt-5.5", None, None, 65536, False
        )
        assert context_length == 65536

    def test_llm_kwargs_provider(self):
        _, _, _, _, llm_kwargs = resolve_provider(
            "chatgpt", "gpt-5.5", None, None, None, False
        )
        assert llm_kwargs["provider"] == "chatgpt"


# ---------------------------------------------------------------------------
# GitHub Copilot provider
# ---------------------------------------------------------------------------


def _copilot_write_cached_token(contents="gho_test"):
    """Simulate a completed device login in the isolated token cache."""
    from swival.agent import _copilot_access_token_file

    path = _copilot_access_token_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    return path


def _patch_copilot_authenticator(monkeypatch, *, fail=None):
    """Replace LiteLLM's Copilot Authenticator with a device-flow fake.

    The fake prints its instructions to stdout like the real adapter and
    writes the access-token cache on success. Returns a call counter.
    """
    calls = {"n": 0}

    class FakeAuthenticator:
        def get_api_key(self):
            calls["n"] += 1
            if fail is not None:
                raise fail
            print(
                "Please visit https://github.com/login/device and enter code "
                "ABCD-1234 to authenticate.",
                flush=True,
            )
            _copilot_write_cached_token()
            return "copilot-api-key"

    import litellm.llms.github_copilot.authenticator as auth_mod

    monkeypatch.setattr(auth_mod, "Authenticator", FakeAuthenticator)
    return calls


def _force_tty(monkeypatch, value=True):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: value, raising=False)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: value, raising=False)


class TestCopilotModelString:
    def test_bare_model(self):
        assert (
            _resolve_model_str("github_copilot", "gpt-5.1") == "github_copilot/gpt-5.1"
        )

    def test_single_prefix_kept(self):
        assert (
            _resolve_model_str("github_copilot", "github_copilot/gpt-5.1")
            == "github_copilot/gpt-5.1"
        )

    def test_double_prefix_collapses(self):
        assert (
            _resolve_model_str(
                "github_copilot", "github_copilot/github_copilot/gpt-5.1"
            )
            == "github_copilot/gpt-5.1"
        )


class TestCopilotRouting:
    """call_llm request shape for the github_copilot provider."""

    @pytest.fixture(autouse=True)
    def _no_lazy_auth(self, monkeypatch):
        forbid_copilot_authenticator(monkeypatch)
        # call_llm refuses to run without a cached login; simulate one.
        _copilot_write_cached_token()

    def _mock_response(self):
        choice = MagicMock()
        choice.message = MagicMock(content="ok", tool_calls=None)
        choice.finish_reason = "stop"
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def test_routing_tools_and_system_role_preserved(self):
        import litellm

        litellm.disable_copilot_system_to_assistant = False
        messages = [
            {"role": "system", "content": "operating contract"},
            {"role": "user", "content": "hi"},
        ]
        tools = [{"type": "function", "function": {"name": "test"}}]
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                None,
                "gpt-5.1",
                messages,
                100,
                0.7,
                0.9,
                42,
                tools,
                False,
                provider="github_copilot",
                api_key=None,
            )
            mock_comp.assert_called_once()
            kwargs = mock_comp.call_args[1]
            assert kwargs["model"] == "github_copilot/gpt-5.1"
            assert "api_key" not in kwargs
            assert "api_base" not in kwargs
            assert kwargs["tools"] == tools
            assert kwargs["tool_choice"] == "auto"
            assert kwargs["temperature"] == 0.7
            assert kwargs["top_p"] == 0.9
            assert kwargs["seed"] == 42
            sent = kwargs["messages"]
            assert sent[0]["role"] == "system"
            assert sent[0]["content"] == "operating contract"
        assert litellm.disable_copilot_system_to_assistant is True

    def test_base_url_forwarded(self):
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "https://proxy.example.com",
                "gpt-5.1",
                [],
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="github_copilot",
                api_key=None,
            )
            assert mock_comp.call_args[1]["api_base"] == "https://proxy.example.com"

    def test_double_prefix_model_id_collapses(self):
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                None,
                "github_copilot/github_copilot/gpt-5.1",
                [],
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="github_copilot",
                api_key=None,
            )
            assert mock_comp.call_args[1]["model"] == "github_copilot/gpt-5.1"

    def test_cost_not_applicable_and_never_priced(self):
        from swival.agent import (
            _COST_NOT_APPLICABLE_PROVIDERS,
            _response_cost_observation,
        )

        assert "github_copilot" in _COST_NOT_APPLICABLE_PROVIDERS
        # completion_cost() re-enters LiteLLM provider routing, which for
        # Copilot can start a token refresh; it must never be reached.
        with patch(
            "litellm.completion_cost",
            side_effect=AssertionError("pricing must not run for copilot"),
        ):
            obs = _response_cost_observation(
                {}, "github_copilot/gpt-5.1", "github_copilot"
            )
        assert obs.status == "not_applicable"

    def test_no_provider_key_env_entry(self):
        from swival.agent import _PROVIDER_KEY_ENV

        assert "github_copilot" not in _PROVIDER_KEY_ENV
        assert "copilot" not in _PROVIDER_KEY_ENV

    def test_unauthenticated_call_fails_before_completion(self, monkeypatch):
        from swival.agent import _copilot_access_token_file

        _copilot_access_token_file().unlink()
        with patch("litellm.completion") as mock_comp:
            with pytest.raises(AgentError, match="not authenticated"):
                call_llm(
                    None,
                    "gpt-5.1",
                    [],
                    100,
                    0.5,
                    1.0,
                    None,
                    None,
                    False,
                    provider="github_copilot",
                    api_key=None,
                )
            mock_comp.assert_not_called()

    def test_model_not_found_recovery_never_probes_copilot(self, monkeypatch):
        import litellm

        from swival.agent import _list_server_models

        def boom(*args, **kwargs):
            raise AssertionError("get_valid_models must not run for copilot")

        monkeypatch.setattr(litellm, "get_valid_models", boom)
        assert _list_server_models("github_copilot", None, None) == []


class TestResolveProviderCopilot:
    """Direct unit tests for resolve_provider() with provider='github_copilot'."""

    @pytest.fixture(autouse=True)
    def _no_lazy_auth(self, monkeypatch):
        forbid_copilot_authenticator(monkeypatch)

    def test_happy_path_with_cached_auth(self):
        _copilot_write_cached_token()
        model_id, api_base, key, ctx, llm_kwargs = resolve_provider(
            "github_copilot", "gpt-5.1", None, None, None, False
        )
        assert model_id == "gpt-5.1"
        assert api_base is None
        assert key is None
        assert llm_kwargs["provider"] == "github_copilot"
        assert ctx == 128000

    def test_alias_normalizes_to_github_copilot(self):
        _copilot_write_cached_token()
        _, _, _, _, llm_kwargs = resolve_provider(
            "copilot", "gpt-5.1", None, None, None, False
        )
        assert llm_kwargs["provider"] == "github_copilot"

    def test_missing_model_raises(self):
        with pytest.raises(
            ConfigError, match="--model is required when --provider is github_copilot"
        ):
            resolve_provider("github_copilot", None, None, None, None, False)

    def test_api_key_rejected(self):
        with pytest.raises(
            ConfigError, match="--api-key is not supported for github_copilot"
        ):
            resolve_provider("github_copilot", "gpt-5.1", "sk-x", None, None, False)

    def test_api_key_rejected_for_alias(self):
        with pytest.raises(
            ConfigError, match="--api-key is not supported for github_copilot"
        ):
            resolve_provider("copilot", "gpt-5.1", "sk-x", None, None, False)

    def test_base_url_retained(self):
        _copilot_write_cached_token()
        _, api_base, _, _, _ = resolve_provider(
            "github_copilot", "gpt-5.1", None, "https://proxy.example.com", None, False
        )
        assert api_base == "https://proxy.example.com"

    def test_explicit_context_override(self):
        _copilot_write_cached_token()
        _, _, _, ctx, _ = resolve_provider(
            "github_copilot", "gpt-5.1", None, None, 55555, False
        )
        assert ctx == 55555

    def test_prefixed_model_context_lookup(self):
        _copilot_write_cached_token()
        _, _, _, ctx, _ = resolve_provider(
            "github_copilot",
            "github_copilot/github_copilot/gpt-5.1",
            None,
            None,
            None,
            False,
        )
        assert ctx == 128000

    def test_unknown_model_context_is_none(self):
        _copilot_write_cached_token()
        _, _, _, ctx, _ = resolve_provider(
            "github_copilot", "not-a-real-model", None, None, None, False
        )
        assert ctx is None


class TestCopilotAuthPreflight:
    def test_non_interactive_fails_fast_without_polling(self, monkeypatch):
        calls = _patch_copilot_authenticator(monkeypatch)
        with pytest.raises(ConfigError, match="not authenticated") as excinfo:
            agent._ensure_copilot_auth(allow_interactive=False)
        assert "copilot_interactive_auth=True" in str(excinfo.value)
        assert calls["n"] == 0

    def test_interactive_without_tty_fails_fast(self, monkeypatch):
        calls = _patch_copilot_authenticator(monkeypatch)
        _force_tty(monkeypatch, False)
        with pytest.raises(ConfigError, match="interactive terminal"):
            agent._ensure_copilot_auth(allow_interactive=True)
        assert calls["n"] == 0

    def test_interactive_preflight_keeps_stdout_clean(self, monkeypatch, capsys):
        calls = _patch_copilot_authenticator(monkeypatch)
        _force_tty(monkeypatch)
        agent._ensure_copilot_auth(allow_interactive=True)
        out, err = capsys.readouterr()
        assert calls["n"] == 1
        assert out == ""
        assert "ABCD-1234" in err

    def test_cached_auth_bypasses_authenticator_everywhere(self, monkeypatch):
        calls = _patch_copilot_authenticator(monkeypatch)
        _copilot_write_cached_token()
        agent._ensure_copilot_auth(allow_interactive=False)
        agent._ensure_copilot_auth(allow_interactive=True)
        assert calls["n"] == 0

    def test_empty_token_file_counts_as_unauthenticated(self, monkeypatch):
        _patch_copilot_authenticator(monkeypatch)
        _copilot_write_cached_token(contents="  \n")
        with pytest.raises(ConfigError, match="not authenticated"):
            agent._ensure_copilot_auth(allow_interactive=False)

    def test_concurrent_calls_run_single_flow(self, monkeypatch):
        import threading

        calls = _patch_copilot_authenticator(monkeypatch)
        _force_tty(monkeypatch)
        errors = []

        def run():
            try:
                agent._ensure_copilot_auth(allow_interactive=True)
            except Exception as e:  # pragma: no cover - failure detail
                errors.append(e)

        threads = [threading.Thread(target=run) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert calls["n"] == 1

    def test_auth_failure_becomes_config_error(self, monkeypatch):
        _patch_copilot_authenticator(
            monkeypatch, fail=RuntimeError("device flow failed")
        )
        _force_tty(monkeypatch)
        with pytest.raises(ConfigError, match="GitHub Copilot authentication failed"):
            agent._ensure_copilot_auth(allow_interactive=True)


class TestCopilotSession:
    def test_default_session_fails_fast(self, tmp_path, monkeypatch):
        from swival.session import Session

        calls = _patch_copilot_authenticator(monkeypatch)
        _force_tty(monkeypatch)
        session = Session(
            base_dir=str(tmp_path),
            provider="github_copilot",
            model="gpt-5.1",
            history=False,
        )
        with pytest.raises(ConfigError, match="copilot_interactive_auth=True"):
            session._setup()
        assert calls["n"] == 0

    def test_opted_in_session_runs_preflight(self, tmp_path, monkeypatch):
        from swival.session import Session

        calls = _patch_copilot_authenticator(monkeypatch)
        _force_tty(monkeypatch)
        session = Session(
            base_dir=str(tmp_path),
            provider="github_copilot",
            model="gpt-5.1",
            history=False,
            copilot_interactive_auth=True,
        )
        session._setup()
        assert calls["n"] == 1

    def test_cached_auth_default_session_ok(self, tmp_path, monkeypatch):
        from swival.session import Session

        calls = _patch_copilot_authenticator(monkeypatch)
        _copilot_write_cached_token()
        session = Session(
            base_dir=str(tmp_path),
            provider="github_copilot",
            model="gpt-5.1",
            history=False,
        )
        session._setup()
        assert calls["n"] == 0


class TestCopilotCLIValidation:
    def test_requires_model(self, monkeypatch):
        from swival import agent, config

        monkeypatch.setattr(config, "load_config", lambda *a, **kw: {})
        monkeypatch.setattr(
            sys, "argv", ["agent", "hello", "--provider", "github_copilot"]
        )
        with pytest.raises(SystemExit) as exc_info:
            agent.main()
        assert exc_info.value.code == 2

    def test_api_key_rejected_cli(self, monkeypatch, tmp_path, capsys):
        from swival import agent, config

        monkeypatch.setattr(config, "load_config", lambda *a, **kw: {})
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--provider",
                "github_copilot",
                "--model",
                "gpt-5.1",
                "--api-key",
                "sk-x",
                "--base-dir",
                str(tmp_path),
            ],
        )
        with pytest.raises(SystemExit) as exc_info:
            agent.main()
        assert exc_info.value.code == 2
        assert "GitHub device flow" in capsys.readouterr().err

    def test_alias_runs_as_github_copilot(self, monkeypatch, tmp_path):
        from swival import agent, config

        _copilot_write_cached_token()
        monkeypatch.setattr(config, "load_config", lambda *a, **kw: {})
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--provider",
                "copilot",
                "--model",
                "gpt-5.1",
                "--no-system-prompt",
                "--base-dir",
                str(tmp_path),
            ],
        )

        captured = {}

        def fake_call_llm(*args, **kwargs):
            captured["provider"] = kwargs.get("provider")
            captured["api_key"] = kwargs.get("api_key")
            msg = types.SimpleNamespace(
                content="done", tool_calls=None, role="assistant"
            )
            msg.get = lambda key, default=None: getattr(msg, key, default)
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        agent.main()
        assert captured["provider"] == "github_copilot"
        assert captured["api_key"] is None

    def test_unauthenticated_cli_without_tty_fails_fast(
        self, monkeypatch, tmp_path, capsys
    ):
        from swival import agent, config

        calls = _patch_copilot_authenticator(monkeypatch)
        _force_tty(monkeypatch, False)
        monkeypatch.setattr(config, "load_config", lambda *a, **kw: {})
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--provider",
                "github_copilot",
                "--model",
                "gpt-5.1",
                "--base-dir",
                str(tmp_path),
            ],
        )
        with pytest.raises(SystemExit) as exc_info:
            agent.main()
        assert exc_info.value.code == 2
        assert calls["n"] == 0
        assert "device login" in capsys.readouterr().err


class TestCopilotLogoutCLI:
    def test_logout_deletes_only_copilot_credential_files(
        self, monkeypatch, tmp_path, capsys
    ):
        from swival import agent

        token_dir = tmp_path / "copilot-tokens"
        token_dir.mkdir()
        access = token_dir / "access-token"
        access.write_text("gho_x", encoding="utf-8")
        api_key = token_dir / "api-key.json"
        api_key.write_text("{}", encoding="utf-8")
        unrelated = token_dir / "hosts.yml"
        unrelated.write_text("external github state", encoding="utf-8")
        monkeypatch.setenv("GITHUB_COPILOT_TOKEN_DIR", str(token_dir))
        monkeypatch.setattr(sys, "argv", ["agent", "--logout"])

        with pytest.raises(SystemExit) as exc_info:
            agent.main()

        assert exc_info.value.code == 0
        assert not access.exists()
        assert not api_key.exists()
        assert unrelated.exists()
        assert token_dir.is_dir()
        err = capsys.readouterr().err
        assert f"Deleted GitHub Copilot access token: {access}" in err
        assert f"Deleted GitHub Copilot API key: {api_key}" in err

    def test_logout_respects_file_env_overrides(self, monkeypatch, tmp_path):
        from swival import agent

        token_dir = tmp_path / "tokens"
        token_dir.mkdir()
        custom_access = token_dir / "custom-access"
        custom_access.write_text("gho_x", encoding="utf-8")
        custom_api = token_dir / "custom-api.json"
        custom_api.write_text("{}", encoding="utf-8")
        default_access = token_dir / "access-token"
        default_access.write_text("keep", encoding="utf-8")
        monkeypatch.setenv("GITHUB_COPILOT_TOKEN_DIR", str(token_dir))
        monkeypatch.setenv("GITHUB_COPILOT_ACCESS_TOKEN_FILE", "custom-access")
        monkeypatch.setenv("GITHUB_COPILOT_API_KEY_FILE", "custom-api.json")
        monkeypatch.setattr(sys, "argv", ["agent", "--logout"])

        with pytest.raises(SystemExit) as exc_info:
            agent.main()

        assert exc_info.value.code == 0
        assert not custom_access.exists()
        assert not custom_api.exists()
        assert default_access.exists()

    def test_logout_clears_chatgpt_and_copilot_together(self, monkeypatch, tmp_path):
        from swival import agent

        chatgpt_dir = tmp_path / "chatgpt"
        chatgpt_dir.mkdir()
        chatgpt_auth = chatgpt_dir / "auth.json"
        chatgpt_auth.write_text("{}", encoding="utf-8")
        copilot_dir = tmp_path / "copilot"
        copilot_dir.mkdir()
        access = copilot_dir / "access-token"
        access.write_text("gho_x", encoding="utf-8")
        monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(chatgpt_dir))
        monkeypatch.setenv("GITHUB_COPILOT_TOKEN_DIR", str(copilot_dir))
        monkeypatch.setattr(sys, "argv", ["agent", "--logout"])

        with pytest.raises(SystemExit) as exc_info:
            agent.main()

        assert exc_info.value.code == 0
        assert not chatgpt_auth.exists()
        assert not access.exists()


# ---------------------------------------------------------------------------
# Bedrock provider
# ---------------------------------------------------------------------------


class TestBedrockRouting:
    """Verify that call_llm routes bedrock calls correctly."""

    def _mock_response(self):
        choice = MagicMock()
        choice.message = MagicMock(content="ok", tool_calls=None)
        choice.finish_reason = "stop"
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def test_bedrock_routing_no_base_url(self):
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                None,
                "anthropic.claude-3-sonnet-20240229-v1:0",
                [],
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="bedrock",
                api_key=None,
            )
            mock_comp.assert_called_once()
            kwargs = mock_comp.call_args[1]
            assert kwargs["model"] == "bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
            assert "api_key" not in kwargs
            assert "api_base" not in kwargs
            assert "aws_region_name" not in kwargs
            assert "aws_bedrock_runtime_endpoint" not in kwargs

    def test_bedrock_routing_base_url_as_region(self):
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "us-west-2",
                "anthropic.claude-3-sonnet-20240229-v1:0",
                [],
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="bedrock",
                api_key=None,
            )
            mock_comp.assert_called_once()
            kwargs = mock_comp.call_args[1]
            assert kwargs["aws_region_name"] == "us-west-2"
            assert "api_base" not in kwargs
            assert "aws_bedrock_runtime_endpoint" not in kwargs

    def test_bedrock_routing_base_url_as_endpoint(self):
        endpoint = "https://bedrock-runtime.us-west-2.amazonaws.com"
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                endpoint,
                "anthropic.claude-3-sonnet-20240229-v1:0",
                [],
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="bedrock",
                api_key=None,
            )
            mock_comp.assert_called_once()
            kwargs = mock_comp.call_args[1]
            assert kwargs["aws_bedrock_runtime_endpoint"] == endpoint
            assert "aws_region_name" not in kwargs
            assert "api_base" not in kwargs

    def test_bedrock_routing_with_aws_profile(self):
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "us-west-2",
                "anthropic.claude-3-sonnet-20240229-v1:0",
                [],
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="bedrock",
                api_key=None,
                aws_profile="bedrock",
            )
            mock_comp.assert_called_once()
            kwargs = mock_comp.call_args[1]
            assert kwargs["aws_profile_name"] == "bedrock"
            assert kwargs["aws_region_name"] == "us-west-2"


class TestResolveProviderBedrock:
    """Direct unit tests for resolve_provider() with provider='bedrock'."""

    def test_happy_path(self):
        model_id, api_base, resolved_key, context_length, llm_kwargs = resolve_provider(
            "bedrock",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            None,
            None,
            None,
            False,
        )
        assert model_id == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert resolved_key is None
        assert llm_kwargs["provider"] == "bedrock"

    def test_missing_model_raises(self):
        from swival.config import ConfigError

        with pytest.raises(ConfigError, match="--model is required"):
            resolve_provider("bedrock", None, None, None, None, False)

    def test_api_key_rejected(self):
        from swival.config import ConfigError

        with pytest.raises(ConfigError, match="--api-key is not supported for bedrock"):
            resolve_provider(
                "bedrock",
                "anthropic.claude-3-sonnet-20240229-v1:0",
                "some-key",
                None,
                None,
                False,
            )

    def test_base_url_passthrough(self):
        _, api_base, _, _, _ = resolve_provider(
            "bedrock",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            None,
            "us-west-2",
            None,
            False,
        )
        assert api_base == "us-west-2"

    def test_context_override(self):
        _, _, _, context_length, _ = resolve_provider(
            "bedrock",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            None,
            None,
            200000,
            False,
        )
        assert context_length == 200000

    def test_aws_profile_in_llm_kwargs(self):
        _, _, _, _, llm_kwargs = resolve_provider(
            "bedrock",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            None,
            None,
            None,
            False,
            aws_profile="my-profile",
        )
        assert llm_kwargs["aws_profile"] == "my-profile"

    def test_aws_profile_omitted_when_none(self):
        _, _, _, _, llm_kwargs = resolve_provider(
            "bedrock",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            None,
            None,
            None,
            False,
        )
        assert "aws_profile" not in llm_kwargs


class TestBedrockCLIParser:
    """Verify that argparse accepts 'bedrock' as a provider."""

    def test_parser_accepts_bedrock(self):
        from swival.agent import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "--provider",
                "bedrock",
                "--model",
                "anthropic.claude-3-sonnet-20240229-v1:0",
                "task",
            ]
        )
        assert args.provider == "bedrock"


class TestPickBestChoice:
    """Verify that _pick_best_choice prefers tool_calls over text-only choices."""

    def _make_choice(self, *, content=None, tool_calls=None, finish_reason="stop"):
        c = MagicMock()
        c.message = MagicMock(content=content, tool_calls=tool_calls)
        c.finish_reason = finish_reason
        return c

    def test_single_choice(self):
        c = self._make_choice(content="done")
        assert _pick_best_choice([c]) is c

    def test_tool_calls_only(self):
        c = self._make_choice(tool_calls=[{"id": "1"}], finish_reason="tool_calls")
        assert _pick_best_choice([c]) is c

    def test_text_plus_tool_calls_prefers_tools(self):
        text = self._make_choice(content="I will read the file")
        tools = self._make_choice(tool_calls=[{"id": "1"}], finish_reason="tool_calls")
        result = _pick_best_choice([text, tools])
        assert result is tools
        assert result.message.content == "I will read the file"

    def test_multiple_text_merged_into_tool_choice(self):
        t1 = self._make_choice(content="Step 1")
        t2 = self._make_choice(content="Step 2")
        tools = self._make_choice(tool_calls=[{"id": "1"}], finish_reason="tool_calls")
        result = _pick_best_choice([t1, t2, tools])
        assert result is tools
        assert result.message.content == "Step 1\n\nStep 2"

    def test_no_tool_calls_returns_first(self):
        c1 = self._make_choice(content="answer 1")
        c2 = self._make_choice(content="answer 2")
        assert _pick_best_choice([c1, c2]) is c1

    def test_empty_choices_raises(self):
        from swival.report import AgentError

        with pytest.raises(AgentError, match="empty choices"):
            _pick_best_choice([])


# ---------------------------------------------------------------------------
# Profile → provider integration
# ---------------------------------------------------------------------------


class TestProfileProviderIntegration:
    """Verify that profiles resolve into the correct provider/model/args."""

    @staticmethod
    def _make_args(**overrides):
        import argparse
        from swival.config import _UNSET

        defaults = {
            "profile": None,
            "provider": _UNSET,
            "model": _UNSET,
            "api_key": _UNSET,
            "base_url": _UNSET,
            "aws_profile": _UNSET,
            "max_output_tokens": _UNSET,
            "max_context_tokens": _UNSET,
            "temperature": _UNSET,
            "top_p": _UNSET,
            "seed": _UNSET,
            "max_turns": _UNSET,
            "system_prompt": _UNSET,
            "no_system_prompt": _UNSET,
            "commands": _UNSET,
            "yolo": _UNSET,
            "files": _UNSET,
            "add_dir": None,
            "add_dir_ro": None,
            "sandbox": _UNSET,
            "sandbox_session": _UNSET,
            "no_read_guard": _UNSET,
            "no_instructions": _UNSET,
            "no_skills": _UNSET,
            "skills_dir": None,
            "no_history": _UNSET,
            "color": _UNSET,
            "no_color": _UNSET,
            "quiet": _UNSET,
            "reviewer": _UNSET,
            "review_prompt": _UNSET,
            "objective": _UNSET,
            "verify": _UNSET,
            "max_review_rounds": _UNSET,
            "cache": _UNSET,
            "cache_dir": _UNSET,
            "extra_body": _UNSET,
            "reasoning_effort": _UNSET,
            "sanitize_thinking": _UNSET,
            "retries": _UNSET,
            "prompt_cache": _UNSET,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def _resolve_via_profile(self, profile_name, profiles, base_config=None):
        """Helper: resolve a profile through the config pipeline and return args."""
        from swival.config import resolve_profile_config, apply_config_to_args

        config = dict(base_config or {})
        config["profiles"] = profiles
        if profile_name:
            config["active_profile"] = profile_name

        args = self._make_args(profile=profile_name)
        resolve_profile_config(args, config)
        apply_config_to_args(args, config)
        return args

    def test_lmstudio_profile(self):
        profiles = {
            "local": {"provider": "lmstudio", "model": "qwen3-coder"},
        }
        args = self._resolve_via_profile("local", profiles)
        assert args.provider == "lmstudio"
        assert args.model == "qwen3-coder"

    def test_generic_with_base_url(self):
        profiles = {
            "ollama": {
                "provider": "generic",
                "base_url": "http://127.0.0.1:11434",
                "model": "qwen3:32b",
            },
        }
        args = self._resolve_via_profile("ollama", profiles)
        assert args.provider == "generic"
        assert args.base_url == "http://127.0.0.1:11434"
        assert args.model == "qwen3:32b"

    def test_chatgpt_with_reasoning_effort(self):
        profiles = {
            "gpt5": {
                "provider": "chatgpt",
                "model": "gpt-5.4",
                "reasoning_effort": "high",
            },
        }
        args = self._resolve_via_profile("gpt5", profiles)
        assert args.provider == "chatgpt"
        assert args.model == "gpt-5.4"
        assert args.reasoning_effort == "high"

    def test_bedrock_with_aws_profile(self):
        profiles = {
            "bedrock-prod": {
                "provider": "bedrock",
                "model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
                "aws_profile": "prod",
                "base_url": "us-west-2",
            },
        }
        args = self._resolve_via_profile("bedrock-prod", profiles)
        assert args.provider == "bedrock"
        assert args.aws_profile == "prod"
        assert args.base_url == "us-west-2"

    def test_cli_flag_overrides_profile_provider(self):
        """Explicit --provider on CLI overrides the profile's provider."""
        from swival.config import resolve_profile_config, apply_config_to_args

        config = {
            "profiles": {
                "fast": {"provider": "lmstudio", "model": "qwen3"},
            },
        }
        args = self._make_args(profile="fast", provider="generic")
        resolve_profile_config(args, config)
        apply_config_to_args(args, config)
        assert args.provider == "generic"
        assert args.model == "qwen3"


# ---------------------------------------------------------------------------
# Special token escaping for user/system messages
# ---------------------------------------------------------------------------


class TestSpecialTokenEscaping:
    """Verify special tokens in user/system messages are escaped to prevent tokenizer interruption."""

    def test_escape_special_tokens_inserts_zero_width_spaces(self):
        from swival.agent import _escape_special_tokens, _ZWSP

        # Basic case
        result = _escape_special_tokens("<|eot_id|>")
        assert _ZWSP in result
        assert "<|" not in result  # Pattern is broken
        assert (
            result.replace(_ZWSP, "") == "<|eot_id|>"
        )  # Visually same when ZWSP removed

    def test_escape_preserves_text_without_special_tokens(self):
        from swival.agent import _escape_special_tokens

        assert _escape_special_tokens("Hello world") == "Hello world"
        assert _escape_special_tokens("") == ""

    def test_escape_handles_multiple_tokens(self):
        from swival.agent import _escape_special_tokens, _ZWSP

        result = _escape_special_tokens("Use <|eot_id|> and <|start_header_id|> here")
        assert result.count(_ZWSP) > 0
        assert "and" in result
        assert "here" in result

    def test_escape_special_tokens_in_messages_skips_assistant(self):
        from swival.agent import _escape_special_tokens_in_messages, _ZWSP

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is <|eot_id|>?"},
            {"role": "assistant", "content": "<|eot_id|> is a token."},
        ]
        _escape_special_tokens_in_messages(messages)

        assert _ZWSP in messages[1]["content"]
        assert messages[0]["content"] == "You are helpful."
        assert messages[2]["content"] == "<|eot_id|> is a token."

    def test_escape_special_tokens_in_messages_system(self):
        from swival.agent import _escape_special_tokens_in_messages, _ZWSP

        messages = [
            {"role": "system", "content": "Never output <|eot_id|>"},
        ]
        _escape_special_tokens_in_messages(messages)
        assert _ZWSP in messages[0]["content"]

    def test_escape_special_tokens_in_messages_tool(self):
        """Tool messages (e.g., file contents) should also be escaped."""
        from swival.agent import _escape_special_tokens_in_messages, _ZWSP

        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "File content with <|eot_id|> token",
            },
        ]
        _escape_special_tokens_in_messages(messages)
        assert _ZWSP in messages[0]["content"]

    def test_escape_special_tokens_in_multipart_content(self):
        """Multi-part content (text + image) should have text parts escaped."""
        from swival.agent import _escape_special_tokens_in_messages, _ZWSP

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is <|eot_id|>?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/img.png"},
                    },
                ],
            },
        ]
        _escape_special_tokens_in_messages(messages)

        assert _ZWSP in messages[0]["content"][0]["text"]
        assert messages[0]["content"][1] == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/img.png"},
        }

    def test_escape_special_tokens_in_multipart_no_special_tokens(self):
        """Multi-part content without special tokens should be unchanged."""
        from swival.agent import _escape_special_tokens_in_messages

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello world"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/img.png"},
                    },
                ],
            },
        ]
        _escape_special_tokens_in_messages(messages)

        assert messages[0]["content"][0]["text"] == "Hello world"


# ---------------------------------------------------------------------------
# llama.cpp provider
# ---------------------------------------------------------------------------


class TestLlamacppProviderRouting:
    """Verify resolve_provider and call_llm routing for the llamacpp provider."""

    @pytest.fixture(autouse=True)
    def _offline_context_probe(self, monkeypatch):
        """Stub the context-window probes so routing tests never hit a server."""
        monkeypatch.setattr(agent, "discover_llamacpp_context_length", lambda *a: None)
        monkeypatch.setattr(agent, "discover_generic_context_length", lambda *a: None)

    def test_default_base_url(self):
        with patch("swival.agent.discover_llamacpp_model", return_value="test-model"):
            model_id, api_base, key, ctx, kwargs = resolve_provider(
                "llamacpp", None, None, None, None, False
            )
        assert api_base == "http://127.0.0.1:8080/v1"
        assert model_id == "test-model"
        assert key is None

    def test_custom_base_url(self):
        with patch("swival.agent.discover_llamacpp_model", return_value="m"):
            _, api_base, _, _, _ = resolve_provider(
                "llamacpp", None, None, "http://192.168.1.5:9090", None, False
            )
        assert api_base == "http://192.168.1.5:9090/v1"

    def test_v1_not_doubled(self):
        with patch("swival.agent.discover_llamacpp_model", return_value="m"):
            _, api_base, _, _, _ = resolve_provider(
                "llamacpp", None, None, "http://host:8080/v1", None, False
            )
        assert api_base == "http://host:8080/v1"

    def test_explicit_model_skips_discovery(self):
        model_id, api_base, _, _, _ = resolve_provider(
            "llamacpp", "my-model", None, None, None, False
        )
        assert model_id == "my-model"
        assert api_base == "http://127.0.0.1:8080/v1"

    def test_context_length_none_when_undiscoverable(self):
        _, _, _, ctx, _ = resolve_provider("llamacpp", "m", None, None, None, False)
        assert ctx is None

    def test_context_length_from_discovery(self, monkeypatch):
        monkeypatch.setattr(
            agent, "discover_llamacpp_context_length", lambda *a: 262144
        )
        _, _, _, ctx, _ = resolve_provider("llamacpp", "m", None, None, None, False)
        assert ctx == 262144

    def test_context_length_from_flag(self):
        _, _, _, ctx, _ = resolve_provider("llamacpp", "m", None, None, 32768, False)
        assert ctx == 32768

    def test_discovery_failure_raises(self):
        from swival.report import AgentError

        with patch("swival.agent.discover_llamacpp_model", return_value=None):
            with pytest.raises(AgentError, match="no model found"):
                resolve_provider("llamacpp", None, None, None, None, False)

    def test_model_str(self):
        from swival.agent import _resolve_model_str

        assert _resolve_model_str("llamacpp", "my-model") == "openai/my-model"

    def _mock_response(self):
        choice = MagicMock()
        choice.message = MagicMock(content="ok", tool_calls=None)
        choice.finish_reason = "stop"
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def test_call_llm_kwargs(self):
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "http://127.0.0.1:8080/v1",
                "test-model",
                [],
                100,
                None,
                None,
                None,
                None,
                False,
                provider="llamacpp",
                api_key=None,
            )
            kwargs = mock_comp.call_args[1]
            assert kwargs["model"] == "openai/test-model"
            assert kwargs["api_base"] == "http://127.0.0.1:8080/v1"
            assert kwargs["api_key"] == "none"


class TestLlamacppCLIParser:
    def test_parser_accepts_llamacpp(self):
        from swival.agent import build_parser

        parser = build_parser()
        args = parser.parse_args(["--provider", "llamacpp", "task"])
        assert args.provider == "llamacpp"

    def test_parser_accepts_llamacpp_with_model(self):
        from swival.agent import build_parser

        parser = build_parser()
        args = parser.parse_args(
            ["--provider", "llamacpp", "--model", "my-model", "task"]
        )
        assert args.provider == "llamacpp"
        assert args.model == "my-model"


class TestLlamacppDiscoverModel:
    """Test discover_llamacpp_model against mocked server responses."""

    def test_discovers_model_id(self):
        from swival.agent import discover_llamacpp_model

        response_data = (
            b'{"object":"list","data":[{"id":"my-gguf-model","object":"model"}]}'
        )
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_data
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = discover_llamacpp_model("http://127.0.0.1:8080", False)
        assert result == "my-gguf-model"

    def test_returns_none_on_empty_data(self):
        from swival.agent import discover_llamacpp_model

        response_data = b'{"object":"list","data":[]}'
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_data
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = discover_llamacpp_model("http://127.0.0.1:8080", False)
        assert result is None

    def test_connection_error_raises(self):
        import urllib.error
        from swival.report import AgentError
        from swival.agent import discover_llamacpp_model

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("refused"),
        ):
            with pytest.raises(AgentError, match="could not connect"):
                discover_llamacpp_model("http://127.0.0.1:8080", False)


class TestLlamacppMainSmoke:
    """Verify that main() invokes discover_llamacpp_model when --provider llamacpp without --model."""

    def test_calls_discover_when_no_model(self, monkeypatch, tmp_path):
        from swival import agent, config

        monkeypatch.setattr(config, "load_config", lambda _: {})
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--provider",
                "llamacpp",
                "--no-system-prompt",
                "--base-dir",
                str(tmp_path),
            ],
        )

        discover_called = {"value": False}

        def fake_discover(*args, **kwargs):
            discover_called["value"] = True
            return "test-model"

        monkeypatch.setattr(agent, "discover_llamacpp_model", fake_discover)

        def fake_call_llm(*args, **kwargs):
            msg = types.SimpleNamespace(
                content="done", tool_calls=None, role="assistant"
            )
            msg.get = lambda key, default=None: getattr(msg, key, default)
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        agent.main()
        assert discover_called["value"]


class TestLlamacppProfileIntegration:
    """Verify that a profile with provider=llamacpp resolves correctly."""

    def test_profile_sets_llamacpp(self):
        from swival.config import resolve_profile_config, apply_config_to_args

        args = TestProfileProviderIntegration._make_args(profile="local")

        config = {
            "profiles": {
                "local": {"provider": "llamacpp", "model": "test-gguf"},
            }
        }
        resolve_profile_config(args, config)
        apply_config_to_args(args, config)
        assert args.provider == "llamacpp"
        assert args.model == "test-gguf"


# ---------------------------------------------------------------------------
# Kimi reasoning_content injection
# ---------------------------------------------------------------------------


class TestKimiReasoningContent:
    """call_llm injects a reasoning_content placeholder only when the endpoint
    requires it (Moonshot's API, matched by base_url), never from the model
    name alone."""

    def _mock_response(self):
        choice = MagicMock()
        choice.message = MagicMock(content="ok", tool_calls=None)
        choice.finish_reason = "stop"
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def test_kimi_model_injects_reasoning_content(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "file contents"},
        ]
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "https://api.moonshot.cn/v1",
                "kimi-k2.6",
                messages,
                100,
                None,
                None,
                None,
                None,
                False,
                provider="generic",
                api_key="sk-test",
            )
            sent_messages = mock_comp.call_args[1]["messages"]
            assistant_msg = [m for m in sent_messages if m.get("role") == "assistant"][
                0
            ]
            assert assistant_msg["reasoning_content"] == " "

    def test_kimi_model_name_alone_does_not_inject(self):
        """A model named "kimi" served off a non-native endpoint (here a local
        server) is not talking to Moonshot's API, so no placeholder is added.
        The requirement is a property of the endpoint, not the model name."""
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "think", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "thought"},
        ]
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "http://localhost:8080/v1",
                "kimi-k2.6",
                messages,
                100,
                None,
                None,
                None,
                None,
                False,
                provider="generic",
                api_key="sk-test",
            )
            sent_messages = mock_comp.call_args[1]["messages"]
            assistant_msg = [m for m in sent_messages if m.get("role") == "assistant"][
                0
            ]
            assert "reasoning_content" not in assistant_msg

    def test_kimi_model_detected_by_moonshot_base_url(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "checking",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "data"},
        ]
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "https://api.moonshot.cn/v1",
                "some-other-model",
                messages,
                100,
                None,
                None,
                None,
                None,
                False,
                provider="generic",
                api_key="sk-test",
            )
            sent_messages = mock_comp.call_args[1]["messages"]
            assistant_msg = [m for m in sent_messages if m.get("role") == "assistant"][
                0
            ]
            assert assistant_msg["reasoning_content"] == " "

    def test_non_kimi_model_no_injection(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "checking",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "data"},
        ]
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "http://localhost:8080/v1",
                "qwen-2.5",
                messages,
                100,
                None,
                None,
                None,
                None,
                False,
                provider="generic",
                api_key="sk-test",
            )
            sent_messages = mock_comp.call_args[1]["messages"]
            assistant_msg = [m for m in sent_messages if m.get("role") == "assistant"][
                0
            ]
            assert "reasoning_content" not in assistant_msg

    def test_kimi_skips_assistant_without_tool_calls(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "https://api.moonshot.cn/v1",
                "kimi-k2.6",
                messages,
                100,
                None,
                None,
                None,
                None,
                False,
                provider="generic",
                api_key="sk-test",
            )
            sent_messages = mock_comp.call_args[1]["messages"]
            assistant_msg = [m for m in sent_messages if m.get("role") == "assistant"][
                0
            ]
            assert "reasoning_content" not in assistant_msg

    def test_kimi_preserves_existing_reasoning_content(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "ok",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
                "reasoning_content": "I need to call a tool",
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "result"},
        ]
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "https://api.moonshot.cn/v1",
                "kimi-k2.6",
                messages,
                100,
                None,
                None,
                None,
                None,
                False,
                provider="generic",
                api_key="sk-test",
            )
            sent_messages = mock_comp.call_args[1]["messages"]
            assistant_msg = [m for m in sent_messages if m.get("role") == "assistant"][
                0
            ]
            assert assistant_msg["reasoning_content"] == "I need to call a tool"


# ---------------------------------------------------------------------------
# Xiaomi MiMo reasoning_content round-trip
# ---------------------------------------------------------------------------


class TestMimoReasoningContent:
    """Verify reasoning_content survives multi-turn round-trip for Xiaomi MiMo.

    MiMo returns a 400 when a tool-calling assistant turn in the history is
    missing its reasoning_content. See:
    https://platform.xiaomimimo.com/docs/en-US/usage-guide/passing-back-reasoning_content
    """

    def _mock_response(self):
        choice = MagicMock()
        choice.message = MagicMock(content="ok", tool_calls=None)
        choice.finish_reason = "stop"
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def test_mimo_model_name_alone_does_not_inject(self):
        """A model named "mimo" off a non-native endpoint is not the Xiaomi
        platform, so no placeholder is added; detection is by endpoint."""
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "data"},
        ]
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "http://localhost:8080/v1",
                "mimo-v2.5-pro",
                messages,
                100,
                None,
                None,
                None,
                None,
                False,
                provider="generic",
                api_key="sk-test",
            )
            sent = mock_comp.call_args[1]["messages"]
            asst = [m for m in sent if m.get("role") == "assistant"][0]
            assert "reasoning_content" not in asst

    def test_mimo_detected_by_base_url(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "data"},
        ]
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "https://api.xiaomimimo.com/v1",
                "some-other-model",
                messages,
                100,
                None,
                None,
                None,
                None,
                False,
                provider="generic",
                api_key="sk-test",
            )
            sent = mock_comp.call_args[1]["messages"]
            asst = [m for m in sent if m.get("role") == "assistant"][0]
            assert asst["reasoning_content"] == " "

    def test_mimo_preserves_actual_reasoning(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
                "reasoning_content": "I should call f to look up the answer.",
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "data"},
        ]
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "https://api.xiaomimimo.com/v1",
                "mimo-v2.5-pro",
                messages,
                100,
                None,
                None,
                None,
                None,
                False,
                provider="generic",
                api_key="sk-test",
            )
            sent = mock_comp.call_args[1]["messages"]
            asst = [m for m in sent if m.get("role") == "assistant"][0]
            assert asst["reasoning_content"] == "I should call f to look up the answer."

    def test_deepseek_model_name_alone_does_not_inject(self):
        """A model named "deepseek" behind a proxy that is not the DeepSeek
        platform gets no placeholder; the field is replayed only to the native
        endpoint that requires it, matched by base_url."""
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "data"},
        ]
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "http://proxy.local/v1",
                "deepseek-v4-flash",
                messages,
                100,
                None,
                None,
                None,
                None,
                False,
                provider="generic",
                api_key="sk-test",
            )
            sent = mock_comp.call_args[1]["messages"]
            asst = [m for m in sent if m.get("role") == "assistant"][0]
            assert "reasoning_content" not in asst

    def test_deepseek_detected_by_base_url(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "data"},
        ]
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "https://api.deepseek.com/v1",
                "aliased-model",
                messages,
                100,
                None,
                None,
                None,
                None,
                False,
                provider="generic",
                api_key="sk-test",
            )
            sent = mock_comp.call_args[1]["messages"]
            asst = [m for m in sent if m.get("role") == "assistant"][0]
            assert asst["reasoning_content"] == " "


# ---------------------------------------------------------------------------
# reasoning_content round-trip helpers
# ---------------------------------------------------------------------------


class TestMsgToDictReasoningContent:
    def test_keeps_reasoning_when_no_tool_calls(self):
        class FakeMsg:
            def model_dump(self, exclude_none=False):
                return {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "internal scratch",
                }

        d = _msg_to_dict(FakeMsg())
        assert d["reasoning_content"] == "internal scratch"
        assert d["content"] == "answer"

    def test_keeps_reasoning_when_tool_calls_present(self):
        class FakeMsg:
            def model_dump(self, exclude_none=False):
                return {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "I should call the tool.",
                    "tool_calls": [
                        {
                            "id": "tc1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                }

        d = _msg_to_dict(FakeMsg())
        assert d["reasoning_content"] == "I should call the tool."
        assert d["tool_calls"][0]["id"] == "tc1"


class TestPromoteReasoningContent:
    def test_promotes_without_discarding_reasoning(self):
        msg = types.SimpleNamespace(content="", reasoning_content="thinking out loud")
        _promote_reasoning_content(msg)
        assert msg.content == "thinking out loud"
        assert msg.reasoning_content == "thinking out loud"

    def test_skips_when_tool_calls_present(self):
        tc = types.SimpleNamespace(
            id="tc1",
            type="function",
            function=types.SimpleNamespace(name="f", arguments="{}"),
        )
        msg = types.SimpleNamespace(
            content="",
            reasoning_content="I will call f",
            tool_calls=[tc],
        )
        _promote_reasoning_content(msg)
        assert msg.content == ""
        assert msg.reasoning_content == "I will call f"


class TestReasoningContentOutbound:
    """reasoning_content is stripped on outbound for every route except the
    models that require the field back (see _needs_reasoning_content).

    Stripping is the safe default: it round-trips only where a provider
    mandates it and keeps mid-session model/provider switches working, since a
    model never inherits another model's reasoning trace. The field still lives
    in stored history for traces and export; only the outbound copy drops it.
    """

    def _mock_response(self):
        choice = MagicMock()
        choice.message = MagicMock(content="ok", tool_calls=None)
        choice.finish_reason = "stop"
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def test_generic_strips_existing_reasoning_content(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
                "reasoning_content": "leftover thought",
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "data"},
        ]
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "http://localhost:8080/v1",
                "qwen-2.5",
                messages,
                100,
                None,
                None,
                None,
                None,
                False,
                provider="generic",
                api_key="sk-test",
            )
            sent = mock_comp.call_args[1]["messages"]
            asst = [m for m in sent if m.get("role") == "assistant"][0]
            assert "reasoning_content" not in asst
            # Stored history keeps it; only the outbound copy drops it.
            assert messages[1]["reasoning_content"] == "leftover thought"

    def test_generic_strips_reasoning_on_non_tool_call_assistant(self):
        """A replayed/imported transcript may carry reasoning_content on an
        assistant message with no tool_calls. It is another turn's internal
        state, so a non-requiring route drops it rather than replaying it."""
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "earlier answer",
                "reasoning_content": "internal thought from a prior session",
            },
            {"role": "user", "content": "next question"},
        ]
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "http://localhost:8080/v1",
                "qwen-2.5",
                messages,
                100,
                None,
                None,
                None,
                None,
                False,
                provider="generic",
                api_key="sk-test",
            )
            sent = mock_comp.call_args[1]["messages"]
            asst = [m for m in sent if m.get("role") == "assistant"][0]
            assert "reasoning_content" not in asst

    def test_chatgpt_strips_existing_reasoning_content(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
                "reasoning_content": "leftover thought",
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "data"},
        ]
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                None,
                "gpt-5.5",
                messages,
                100,
                None,
                None,
                None,
                None,
                False,
                provider="chatgpt",
                api_key="sk-test",
            )
            sent = mock_comp.call_args[1]["messages"]
            asst = [m for m in sent if m.get("role") == "assistant"][0]
            assert "reasoning_content" not in asst
            assert messages[1]["reasoning_content"] == "leftover thought"

    def test_openai_api_base_strips_existing_reasoning_content(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "prior",
                "reasoning_content": "leftover thought",
            },
            {"role": "user", "content": "next"},
        ]
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "https://api.openai.com/v1",
                "gpt-5.5",
                messages,
                100,
                None,
                None,
                None,
                None,
                False,
                provider="generic",
                api_key="sk-test",
            )
            sent = mock_comp.call_args[1]["messages"]
            asst = [m for m in sent if m.get("role") == "assistant"][0]
            assert "reasoning_content" not in asst
            assert messages[1]["reasoning_content"] == "leftover thought"

    def test_huggingface_strips_existing_reasoning_content(self):
        """The HF router rejects reasoning_content on a replayed tool-call turn,
        so it must be dropped on every huggingface request."""
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
                "reasoning_content": "leftover thought",
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "data"},
        ]
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                None,
                "google/gemma-4-31B-it",
                messages,
                100,
                None,
                None,
                None,
                None,
                False,
                provider="huggingface",
                api_key="hf-test",
            )
            sent = mock_comp.call_args[1]["messages"]
            asst = [m for m in sent if m.get("role") == "assistant"][0]
            assert "reasoning_content" not in asst
            assert messages[1]["reasoning_content"] == "leftover thought"

    def test_huggingface_never_injects_reasoning_placeholder(self):
        """Even a reasoning model routed through HuggingFace must not gain the
        placeholder that direct Kimi/DeepSeek/MiMo endpoints require."""
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "checking",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "data"},
        ]
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                None,
                "deepseek-ai/DeepSeek-V4",
                messages,
                100,
                None,
                None,
                None,
                None,
                False,
                provider="huggingface",
                api_key="hf-test",
            )
            sent = mock_comp.call_args[1]["messages"]
            asst = [m for m in sent if m.get("role") == "assistant"][0]
            assert "reasoning_content" not in asst


class TestOrphanedToolCallsStripsReasoning:
    def test_orphan_cleanup_drops_reasoning_when_emptied(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
                "reasoning_content": "I will call f",
            },
        ]
        fixed = _fix_orphaned_tool_calls(messages)
        assert fixed is True
        asst = messages[1]
        assert "tool_calls" not in asst
        assert "reasoning_content" not in asst
        assert asst["content"] == _EMPTY_ASSISTANT_PLACEHOLDER

    def test_orphan_cleanup_keeps_reasoning_when_tool_calls_partially_kept(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    },
                    {
                        "id": "tc2",
                        "type": "function",
                        "function": {"name": "g", "arguments": "{}"},
                    },
                ],
                "reasoning_content": "I will call f and g",
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "data"},
        ]
        fixed = _fix_orphaned_tool_calls(messages)
        assert fixed is True
        asst = messages[1]
        assert len(asst["tool_calls"]) == 1
        assert asst["tool_calls"][0]["id"] == "tc1"
        assert asst["reasoning_content"] == "I will call f and g"


class TestCompleteOrphanedToolCalls:
    """Backfill placeholder results for interrupted/cancelled tool calls."""

    def test_interrupt_with_no_results_backfills_all(self):
        from swival._msg import _complete_orphaned_tool_calls

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tooluse_a",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    },
                    {
                        "id": "tooluse_b",
                        "type": "function",
                        "function": {"name": "g", "arguments": "{}"},
                    },
                ],
            },
        ]
        inserted = _complete_orphaned_tool_calls(messages, content="error: cut short")
        assert inserted == 2
        results = [m for m in messages if m.get("role") == "tool"]
        assert [r["tool_call_id"] for r in results] == ["tooluse_a", "tooluse_b"]
        assert all(r["content"] == "error: cut short" for r in results)
        # Placeholders sit immediately after the assistant message.
        assert messages[2]["role"] == "tool"
        assert messages[3]["role"] == "tool"

    def test_partial_results_backfills_only_missing(self):
        from swival._msg import _complete_orphaned_tool_calls

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    },
                    {
                        "id": "tc2",
                        "type": "function",
                        "function": {"name": "g", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "real result"},
        ]
        inserted = _complete_orphaned_tool_calls(messages, content="error: cut short")
        assert inserted == 1
        # New placeholder lands right after the existing tc1 result.
        assert messages[3]["tool_call_id"] == "tc2"
        assert messages[3]["content"] == "error: cut short"
        # The real result is untouched.
        assert messages[2]["content"] == "real result"

    def test_fully_satisfied_is_noop(self):
        from swival._msg import _complete_orphaned_tool_calls

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "done"},
            {"role": "assistant", "content": "final answer"},
        ]
        before = [dict(m) for m in messages]
        inserted = _complete_orphaned_tool_calls(messages, content="error: cut short")
        assert inserted == 0
        assert messages == before

    def test_bedrock_error_phrasing_matches_orphan_regex(self):
        from swival.agent import _ORPHANED_TOOL_CALL_RE

        msg = (
            'BedrockException - {"message":"The model returned the following '
            "errors: messages.20: `tool_use` ids were found without `tool_result` "
            "blocks immediately after: tooluse_9NqTHSFFoFTOJmI9s4mbm6. Each "
            "`tool_use` block must have a corresponding `tool_result` block in "
            'the next message."}'
        )
        assert _ORPHANED_TOOL_CALL_RE.search(msg)


# ---------------------------------------------------------------------------
# GEAP (Gemini Enterprise Agent Platform / Vertex AI) provider
# ---------------------------------------------------------------------------


class TestResolveModelStrGeap:
    """Verify _resolve_model_str for geap provider."""

    def test_bare_model(self):
        assert (
            _resolve_model_str("geap", "gemini-3.1-pro") == "vertex_ai/gemini-3.1-pro"
        )

    def test_another_model(self):
        assert (
            _resolve_model_str("geap", "gemini-3-flash") == "vertex_ai/gemini-3-flash"
        )


class TestResolveProviderGeap:
    """Direct unit tests for resolve_provider() with provider='geap'."""

    def test_happy_path(self):
        model_id, api_base, resolved_key, context_length, llm_kwargs = resolve_provider(
            "geap",
            "gemini-3.1-pro",
            None,
            None,
            None,
            False,
            project="my-project",
            location="us-central1",
        )
        assert model_id == "gemini-3.1-pro"
        assert resolved_key is None
        assert llm_kwargs["provider"] == "geap"
        assert llm_kwargs["vertex_project"] == "my-project"
        assert llm_kwargs["vertex_location"] == "us-central1"

    def test_vertexai_alias_normalizes_to_geap(self):
        model_id, _, _, _, llm_kwargs = resolve_provider(
            "vertexai",
            "gemini-3.1-pro",
            None,
            None,
            None,
            False,
            project="my-project",
            location="us-central1",
        )
        assert model_id == "gemini-3.1-pro"
        assert llm_kwargs["provider"] == "geap"
        assert llm_kwargs["vertex_project"] == "my-project"

    def test_missing_model_raises(self):
        with pytest.raises(ConfigError, match="--model is required"):
            resolve_provider(
                "geap",
                None,
                None,
                None,
                None,
                False,
                project="p",
                location="us-central1",
            )

    def test_api_key_rejected(self):
        with pytest.raises(ConfigError, match="--api-key is not supported for geap"):
            resolve_provider(
                "geap",
                "gemini-3.1-pro",
                "some-key",
                None,
                None,
                False,
                project="p",
                location="us-central1",
            )

    def test_prefixed_model_rejected(self):
        with pytest.raises(ConfigError, match="bare model name"):
            resolve_provider(
                "geap",
                "vertex_ai/gemini-3.1-pro",
                None,
                None,
                None,
                False,
                project="p",
                location="us-central1",
            )

    def test_missing_project_raises(self):
        with pytest.raises(ConfigError, match="--gcp-project.*required"):
            resolve_provider(
                "geap",
                "gemini-3.1-pro",
                None,
                None,
                None,
                False,
                location="us-central1",
            )

    def test_project_from_env(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")
        _, _, _, _, llm_kwargs = resolve_provider(
            "geap",
            "gemini-3.1-pro",
            None,
            None,
            None,
            False,
            location="us-central1",
        )
        assert llm_kwargs["vertex_project"] == "env-project"

    def test_explicit_project_overrides_env(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")
        _, _, _, _, llm_kwargs = resolve_provider(
            "geap",
            "gemini-3.1-pro",
            None,
            None,
            None,
            False,
            project="explicit-project",
            location="us-central1",
        )
        assert llm_kwargs["vertex_project"] == "explicit-project"

    def test_missing_location_raises(self):
        with pytest.raises(ConfigError, match="--location is required"):
            resolve_provider(
                "geap",
                "gemini-3.1-pro",
                None,
                None,
                None,
                False,
                project="p",
            )

    def test_context_override(self):
        _, _, _, context_length, _ = resolve_provider(
            "geap",
            "gemini-3.1-pro",
            None,
            None,
            200000,
            False,
            project="p",
            location="us-central1",
        )
        assert context_length == 200000

    def test_credentials_file_missing_raises(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/nonexistent/sa.json")
        with pytest.raises(ConfigError, match="does not exist"):
            resolve_provider(
                "geap",
                "gemini-3.1-pro",
                None,
                None,
                None,
                False,
                project="p",
                location="us-central1",
            )

    def test_base_url_passthrough(self):
        _, api_base, _, _, _ = resolve_provider(
            "geap",
            "gemini-3.1-pro",
            None,
            "https://custom-endpoint.example.com",
            None,
            False,
            project="p",
            location="us-central1",
        )
        assert api_base == "https://custom-endpoint.example.com"


class TestGeapRouting:
    """Verify that call_llm routes geap calls correctly."""

    def _mock_response(self):
        choice = MagicMock()
        choice.message = MagicMock(content="ok", tool_calls=None)
        choice.finish_reason = "stop"
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def test_geap_routing_basic(self):
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                None,
                "gemini-3.1-pro",
                [],
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="geap",
                api_key=None,
                vertex_project="my-project",
                vertex_location="us-central1",
            )
            mock_comp.assert_called_once()
            kwargs = mock_comp.call_args[1]
            assert kwargs["model"] == "vertex_ai/gemini-3.1-pro"
            assert kwargs["vertex_project"] == "my-project"
            assert kwargs["vertex_location"] == "us-central1"
            assert "api_key" not in kwargs
            assert "api_base" not in kwargs

    def test_geap_routing_with_base_url(self):
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "https://custom-endpoint.example.com",
                "gemini-3.1-pro",
                [],
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="geap",
                api_key=None,
                vertex_project="my-project",
                vertex_location="us-central1",
            )
            mock_comp.assert_called_once()
            kwargs = mock_comp.call_args[1]
            assert kwargs["api_base"] == "https://custom-endpoint.example.com"


class TestGeapCLIParser:
    """Verify that argparse accepts 'geap' and 'vertexai' as providers."""

    def test_parser_accepts_geap(self):
        from swival.agent import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "--provider",
                "geap",
                "--model",
                "gemini-3.1-pro",
                "--gcp-project",
                "my-project",
                "--location",
                "us-central1",
                "task",
            ]
        )
        assert args.provider == "geap"
        assert args.gcp_project == "my-project"
        assert args.location == "us-central1"

    def test_parser_accepts_vertexai_alias(self):
        from swival.agent import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "--provider",
                "vertexai",
                "--model",
                "gemini-3.1-pro",
                "--gcp-project",
                "my-project",
                "--location",
                "us-central1",
                "task",
            ]
        )
        assert args.provider == "vertexai"


class TestResponseCostObservation:
    """Pricing one completion response for the session cost subtotal."""

    def _response(self, prompt=100, completion=10, hidden=None):
        usage = types.SimpleNamespace(
            prompt_tokens=prompt, completion_tokens=completion
        )
        resp = types.SimpleNamespace(usage=usage)
        if hidden is not None:
            resp._hidden_params = hidden
        return resp

    def test_hidden_response_cost_wins_over_recalculation(self):
        resp = self._response(hidden={"response_cost": 0.005})
        with patch("litellm.completion_cost") as cc:
            obs = agent._response_cost_observation(resp, "openrouter/m", "openrouter")
        assert obs.status == "known"
        assert obs.usd == 0.005
        cc.assert_not_called()

    def test_dict_shaped_response(self):
        resp = {
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            "_hidden_params": {"response_cost": 0.001},
        }
        obs = agent._response_cost_observation(resp, "openrouter/m", "openrouter")
        assert obs.status == "known"
        assert obs.usd == 0.001

    def test_fallback_to_completion_cost_with_resolved_model(self):
        resp = self._response()
        with patch("litellm.completion_cost", return_value=0.002) as cc:
            obs = agent._response_cost_observation(resp, "openrouter/m", "openrouter")
        assert obs.status == "known"
        assert obs.usd == 0.002
        assert cc.call_args.kwargs["completion_response"] is resp
        assert cc.call_args.kwargs["model"] == "openrouter/m"

    def test_calculation_exception_is_unavailable(self):
        resp = self._response()
        with patch("litellm.completion_cost", side_effect=Exception("unmapped")):
            obs = agent._response_cost_observation(resp, "openrouter/m", "openrouter")
        assert obs.status == "unavailable"

    def test_missing_usage_skips_calculated_zero(self):
        resp = types.SimpleNamespace(usage=None)
        with patch("litellm.completion_cost", return_value=0.0) as cc:
            obs = agent._response_cost_observation(resp, "openrouter/m", "openrouter")
        assert obs.status == "unavailable"
        cc.assert_not_called()

    def test_real_zero_with_usage_stays_known(self):
        resp = self._response()
        with patch("litellm.completion_cost", return_value=0.0):
            obs = agent._response_cost_observation(resp, "openrouter/m", "openrouter")
        assert obs.status == "known"
        assert obs.usd == 0.0

    def test_hidden_zero_without_usage_is_unavailable(self):
        resp = types.SimpleNamespace(usage=None, _hidden_params={"response_cost": 0.0})
        obs = agent._response_cost_observation(resp, "openrouter/m", "openrouter")
        assert obs.status == "unavailable"

    def test_invalid_values_rejected(self):
        for bad in (-0.001, float("inf"), float("nan"), "0.005", True):
            resp = self._response(hidden={"response_cost": bad})
            obs = agent._response_cost_observation(resp, "openrouter/m", "openrouter")
            assert obs.status == "unavailable", bad

    def test_local_and_chatgpt_not_applicable(self):
        resp = self._response(hidden={"response_cost": 0.005})
        for provider in ("command", "lmstudio", "llamacpp", "applefm", "chatgpt"):
            obs = agent._response_cost_observation(resp, "openai/m", provider)
            assert obs.status == "not_applicable", provider

    def test_generic_never_consults_litellm(self):
        # A colliding hosted name yields a map-derived hidden cost; distrust it.
        resp = self._response(hidden={"response_cost": 0.005})
        with patch("litellm.completion_cost") as cc:
            obs = agent._response_cost_observation(resp, "openai/gpt-4o", "generic")
        assert obs.status == "unavailable"
        cc.assert_not_called()

    def _litellm_response(self, model, prompt=1000, completion=100):
        from litellm.types.utils import ModelResponse, Usage

        resp = ModelResponse(
            model=model,
            choices=[{"message": {"role": "assistant", "content": "hi"}}],
        )
        resp.usage = Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )
        resp._hidden_params = {"custom_llm_provider": "openai", "response_cost": 99.0}
        return resp

    def _expected_gemini_cost(self, prompt=1000, completion=100):
        import litellm

        info = litellm.model_cost["gemini/gemini-2.5-flash"]
        return (
            prompt * info["input_cost_per_token"]
            + completion * info["output_cost_per_token"]
        )

    def test_google_prices_real_gemini_response(self):
        # Unmocked regression: neither the "openai" provider stamp nor the
        # routed hidden cost may override the explicit gemini pricing.
        resp = self._litellm_response("gemini-2.5-flash")
        obs = agent._response_cost_observation(
            resp, "openai/gemini-2.5-flash", "generic", "google"
        )
        assert obs.status == "known"
        assert obs.usd == pytest.approx(self._expected_gemini_cost())
        assert resp._hidden_params == {
            "custom_llm_provider": "openai",
            "response_cost": 99.0,
        }

    def test_google_ignores_response_model_name(self):
        # completion_cost() falls back to the response's own model name; a
        # response claiming "gpt-4o" must still price as the gemini model.
        resp = self._litellm_response("gpt-4o")
        obs = agent._response_cost_observation(
            resp, "openai/gemini-2.5-flash", "generic", "google"
        )
        assert obs.status == "known"
        assert obs.usd == pytest.approx(self._expected_gemini_cost())

    def test_google_colliding_model_name_is_unavailable(self):
        # A google-provider model actually named gpt-4o maps to the unmapped
        # gemini/gpt-4o; the openai map must never price it as a fallback.
        resp = self._litellm_response("gpt-4o")
        obs = agent._response_cost_observation(
            resp, "openai/gpt-4o", "generic", "google"
        )
        assert obs.status == "unavailable"

    def test_google_without_usage_is_unavailable(self):
        resp = types.SimpleNamespace(usage=None)
        with patch("litellm.completion_cost", return_value=0.001) as cc:
            obs = agent._response_cost_observation(
                resp, "openai/gemini-2.5-flash", "generic", "google"
            )
        assert obs.status == "unavailable"
        cc.assert_not_called()


class TestCallLlmCostRecording:
    """Every live response reaches the cost observation exactly once."""

    def _call(self, session_cost, **overrides):
        kwargs = dict(provider="openrouter", api_key="k", session_cost=session_cost)
        kwargs.update(overrides)
        return call_llm(
            None,
            "my-model",
            [{"role": "user", "content": "hi"}],
            100,
            0.5,
            None,
            None,
            None,
            False,
            **kwargs,
        )

    def test_ordinary_success_records_once(self, monkeypatch):
        from swival.cost import CostObservation, SessionCost

        msg = types.SimpleNamespace(content="hello", tool_calls=None, role="assistant")
        choice = types.SimpleNamespace(message=msg, finish_reason="stop")
        resp = types.SimpleNamespace(choices=[choice])

        calls = []

        def spy(response, model_str, provider, pricing_provider=None):
            calls.append((response, model_str, provider, pricing_provider))
            return CostObservation("known", 0.003)

        monkeypatch.setattr(agent, "_response_cost_observation", spy)
        sc = SessionCost()
        with patch("litellm.completion", return_value=resp):
            self._call(sc)

        assert len(calls) == 1
        assert calls[0][0] is resp
        assert calls[0][1] == "openrouter/my-model"
        assert calls[0][2] == "openrouter"
        snap = sc.snapshot()
        assert snap.priced_calls == 1
        assert snap.known_usd == 0.003

    def test_pricing_provider_marker_forwarded(self, monkeypatch):
        from swival.cost import CostObservation, SessionCost

        msg = types.SimpleNamespace(content="hello", tool_calls=None, role="assistant")
        choice = types.SimpleNamespace(message=msg, finish_reason="stop")
        resp = types.SimpleNamespace(choices=[choice])

        calls = []

        def spy(response, model_str, provider, pricing_provider=None):
            calls.append(pricing_provider)
            return CostObservation("unavailable")

        monkeypatch.setattr(agent, "_response_cost_observation", spy)
        sc = SessionCost()
        with patch("litellm.completion", return_value=resp):
            self._call(sc, provider="generic", pricing_provider="google")

        assert calls == ["google"]
        assert sc.snapshot().unpriced_calls == 1

    def test_cost_failure_never_breaks_the_call(self):
        from swival.cost import SessionCost

        msg = types.SimpleNamespace(content="hello", tool_calls=None, role="assistant")
        choice = types.SimpleNamespace(message=msg, finish_reason="stop")
        usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=2)
        resp = types.SimpleNamespace(choices=[choice], usage=usage)

        sc = SessionCost()
        with (
            patch("litellm.completion", return_value=resp),
            patch("litellm.completion_cost", side_effect=Exception("unmapped model")),
        ):
            out_msg, finish, *_ = self._call(sc)
        assert out_msg.content == "hello"
        assert finish == "stop"
        assert sc.snapshot().unpriced_calls == 1

    def test_repair_path_records_each_live_response_once(self, monkeypatch):
        """The empty-assistant repair retry produces exactly one live response
        (the first attempt raised before a response existed), so exactly one
        observation is recorded."""
        import litellm

        from swival.cost import CostObservation, SessionCost

        bad_req = litellm.BadRequestError(
            message="must have either content or tool_calls",
            llm_provider="openai",
            model="x",
        )
        msg = types.SimpleNamespace(content="hello", tool_calls=None, role="assistant")
        choice = types.SimpleNamespace(message=msg, finish_reason="stop")
        resp = types.SimpleNamespace(choices=[choice])

        calls = []

        def spy(response, model_str, provider, pricing_provider=None):
            calls.append(response)
            return CostObservation("known", 0.001)

        monkeypatch.setattr(agent, "_response_cost_observation", spy)
        sc = SessionCost()
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant"},
            {"role": "user", "content": "continue"},
        ]
        with patch("litellm.completion", MagicMock(side_effect=[bad_req, resp])):
            out_msg, *_ = call_llm(
                None,
                "my-model",
                messages,
                100,
                0.5,
                None,
                None,
                None,
                False,
                provider="openrouter",
                api_key="k",
                session_cost=sc,
            )

        assert out_msg.content == "hello"
        assert calls == [resp]
        assert sc.snapshot().priced_calls == 1

    def test_no_session_cost_skips_observation(self, monkeypatch):
        msg = types.SimpleNamespace(content="hello", tool_calls=None, role="assistant")
        choice = types.SimpleNamespace(message=msg, finish_reason="stop")
        resp = types.SimpleNamespace(choices=[choice])

        calls = []
        monkeypatch.setattr(
            agent, "_response_cost_observation", lambda *a, **k: calls.append(a)
        )
        with patch("litellm.completion", return_value=resp):
            self._call(None)
        assert calls == []


class TestGooglePricingMarker:
    def test_resolve_provider_sets_marker_for_google(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        monkeypatch.setattr(agent, "_litellm_context_length", lambda s: 1000000)
        *_, llm_kwargs = resolve_provider(
            provider="google",
            model="gemini-2.5-flash",
            api_key=None,
            base_url=None,
            max_context_tokens=None,
            verbose=False,
        )
        assert llm_kwargs["provider"] == "generic"
        assert llm_kwargs["pricing_provider"] == "google"

    def test_other_providers_have_no_marker(self):
        *_, llm_kwargs = resolve_provider(
            provider="openrouter",
            model="openai/gpt-4o",
            api_key="k",
            base_url=None,
            max_context_tokens=None,
            verbose=False,
        )
        assert "pricing_provider" not in llm_kwargs

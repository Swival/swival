"""Tests for provider routing, model normalization, CLI validation, and path isolation."""

import sys
import types

import pytest
from unittest.mock import patch, MagicMock

from swival.agent import (
    call_llm,
    resolve_provider,
    _pick_best_choice,
    _sanitize_assistant_content,
)


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
                "zai-org/GLM-5",
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
            assert kwargs[1]["model"] == "huggingface/zai-org/GLM-5"
            assert kwargs[1]["api_key"] == "hf_test"
            assert "api_base" not in kwargs[1]

    def test_huggingface_with_base_url(self):
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "https://xyz.endpoints.huggingface.cloud",
                "zai-org/GLM-5",
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


class TestAssistantContentSanitization:
    """Verify call_llm strips leaked hidden-reasoning markers."""

    def test_sanitize_assistant_content_strips_special_tokens_and_think_prefix(self):
        text = "<|start_header_id|><think>Plan</think>\n\nAnswer"
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
                "zai-org/GLM-5",
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
            assert mock_comp.call_args[1]["model"] == "huggingface/zai-org/GLM-5"

    def test_already_prefixed_no_double(self):
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                None,
                "huggingface/zai-org/GLM-5",
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
            assert mock_comp.call_args[1]["model"] == "huggingface/zai-org/GLM-5"

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
        from swival import agent

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


class TestGenericProviderRouting:
    """Verify call_llm routing for the generic provider."""

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
                "google", "gemini-2.5-flash", "gemini-key", None, None, False
            )
            assert llm_kwargs["provider"] == "generic"
            assert api_base == self._GOOGLE_OPENAI_BASE
            call_llm(
                api_base,
                "gemini-2.5-flash",
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
            assert kwargs["model"] == "openai/gemini-2.5-flash"
            assert kwargs["api_key"] == "gemini-key"
            assert kwargs["api_base"] == self._GOOGLE_OPENAI_BASE

    def test_google_custom_base_url_overrides_default(self):
        _, api_base, _, _, llm_kwargs = resolve_provider(
            "google", "gemini-2.5-flash", "k", "https://custom.example.com", None, False
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
                "gemini-2.5-flash",
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
                "google", "gemini-2.5-flash", None, None, None, False
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
                "gemini-2.5-flash",
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
                "gpt-5.3-codex",
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
            assert kwargs["model"] == "chatgpt/gpt-5.3-codex"
            assert "api_key" not in kwargs
            assert "api_base" not in kwargs

    def test_chatgpt_with_explicit_key(self):
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                None,
                "gpt-5.3-codex",
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
                "gpt-5.3-codex",
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
                "gpt-5.3-codex",
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
                "chatgpt/gpt-5.3-codex",
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
            assert mock_comp.call_args[1]["model"] == "chatgpt/gpt-5.3-codex"

    def test_chatgpt_bare_model_id(self):
        """Bare model (no prefix) gets chatgpt/ prepended."""
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                None,
                "gpt-5.3-codex",
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
            assert mock_comp.call_args[1]["model"] == "chatgpt/gpt-5.3-codex"

    def test_chatgpt_double_prefix_stripped(self):
        """chatgpt/chatgpt/model collapses to chatgpt/model."""
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                None,
                "chatgpt/chatgpt/gpt-5.3-codex",
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
            assert mock_comp.call_args[1]["model"] == "chatgpt/gpt-5.3-codex"


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
                "gpt-5.3-codex",
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
                "gpt-5.3-codex",
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


# ---------------------------------------------------------------------------
# ChatGPT resolve_provider direct unit tests
# ---------------------------------------------------------------------------


class TestResolveProviderChatGPT:
    """Direct unit tests for resolve_provider() with provider='chatgpt'."""

    def test_returns_model_id(self):
        model_id, _, _, _, _ = resolve_provider(
            "chatgpt", "gpt-5.3-codex", None, None, None, False
        )
        assert model_id == "gpt-5.3-codex"

    def test_no_model_raises(self):
        from swival.config import ConfigError

        with pytest.raises(ConfigError, match="--model is required") as excinfo:
            resolve_provider("chatgpt", None, None, None, None, False)

        assert "Available models:" not in str(excinfo.value)
        assert "docs.litellm.ai/docs/providers/chatgpt" in str(excinfo.value)

    def test_resolved_key_none_when_no_key(self):
        _, _, resolved_key, _, _ = resolve_provider(
            "chatgpt", "gpt-5.3-codex", None, None, None, False
        )
        assert resolved_key is None

    def test_resolved_key_passthrough(self):
        _, _, resolved_key, _, _ = resolve_provider(
            "chatgpt", "gpt-5.3-codex", "bearer-xyz", None, None, False
        )
        assert resolved_key == "bearer-xyz"

    def test_api_base_none_by_default(self):
        _, api_base, _, _, _ = resolve_provider(
            "chatgpt", "gpt-5.3-codex", None, None, None, False
        )
        assert api_base is None

    def test_api_base_passthrough(self):
        _, api_base, _, _, _ = resolve_provider(
            "chatgpt", "gpt-5.3-codex", None, "https://proxy.example.com", None, False
        )
        assert api_base == "https://proxy.example.com"

    def test_context_from_litellm_metadata(self):
        _, _, _, context_length, _ = resolve_provider(
            "chatgpt", "gpt-5.3-codex", None, None, None, False
        )
        # Should resolve from litellm metadata (or None if not in cost map).
        # Don't assert a specific value since it depends on litellm version.
        assert context_length is None or (
            isinstance(context_length, int) and context_length > 0
        )

    def test_context_override(self):
        _, _, _, context_length, _ = resolve_provider(
            "chatgpt", "gpt-5.3-codex", None, None, 65536, False
        )
        assert context_length == 65536

    def test_llm_kwargs_provider(self):
        _, _, _, _, llm_kwargs = resolve_provider(
            "chatgpt", "gpt-5.3-codex", None, None, None, False
        )
        assert llm_kwargs["provider"] == "chatgpt"


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
        assert result.replace(_ZWSP, "") == "<|eot_id|>"  # Visually same when ZWSP removed

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
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ],
            },
        ]
        _escape_special_tokens_in_messages(messages)

        assert _ZWSP in messages[0]["content"][0]["text"]
        assert messages[0]["content"][1] == {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}

    def test_escape_special_tokens_in_multipart_no_special_tokens(self):
        """Multi-part content without special tokens should be unchanged."""
        from swival.agent import _escape_special_tokens_in_messages

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello world"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ],
            },
        ]
        _escape_special_tokens_in_messages(messages)

        assert messages[0]["content"][0]["text"] == "Hello world"

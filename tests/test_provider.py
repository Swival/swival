"""Tests for provider routing, model normalization, CLI validation, and path isolation."""

import sys
import types

import pytest
from unittest.mock import patch, MagicMock

from swival.agent import call_llm, resolve_provider, _pick_best_choice


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
        assert captured["api_key"] == "sk_or_env"

    def test_openrouter_no_key_errors(self, monkeypatch):
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
            ],
        )
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(SystemExit) as exc_info:
            agent.main()
        assert exc_info.value.code == 2

    def test_no_key_errors(self, monkeypatch):
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
            ],
        )
        monkeypatch.delenv("HF_TOKEN", raising=False)
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
                "http://localhost:8080",
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

    def test_generic_appends_v1_when_missing(self):
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
            assert mock_comp.call_args[1]["api_base"] == "http://host:9000/v1"

    def test_generic_no_double_v1(self):
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

    def test_generic_trailing_slash_stripped(self):
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "http://host:9000/",
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

    def test_generic_trailing_slash_with_v1(self):
        with patch("litellm.completion") as mock_comp:
            mock_comp.return_value = self._mock_response()
            call_llm(
                "http://host:9000/v1/",
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

        with pytest.raises(ConfigError, match="--model is required"):
            resolve_provider("chatgpt", None, None, None, None, False)

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

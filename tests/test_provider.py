"""Tests for provider routing, model normalization, CLI validation, and path isolation."""

import sys
import types

import pytest
from unittest.mock import patch, MagicMock

from swival.agent import call_llm


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
        from swival import agent

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
        from swival import agent

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
        from swival import agent

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

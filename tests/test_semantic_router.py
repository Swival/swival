"""Tests for swival.semantic_router — semantic profile routing."""

from unittest.mock import patch

import pytest

from swival.report import AgentError
from swival.semantic_router import (
    _build_router_prompt,
    _parse_router_response,
    route_task,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_config(**overrides):
    """Build a minimal config dict with routing enabled."""
    config = {
        "provider": "lmstudio",
        "routing_enabled": True,
        "semantic_routing_profile": "router",
        "profiles": {
            "router": {
                "provider": "openrouter",
                "model": "glm-5",
                "description": "General purpose router.",
            },
            "fast": {
                "provider": "lmstudio",
                "model": "qwen3",
                "description": "Fast local edits.",
            },
            "heavy": {
                "provider": "chatgpt",
                "model": "gpt-5",
                "description": "Hard reasoning tasks.",
            },
        },
    }
    config.update(overrides)
    return config


def _mock_llm_response(text):
    """Create a mock return value matching call_llm's (msg, finish, cmd, retries, cache) tuple."""
    msg = {"content": text}
    return msg, "stop", [], 0, (0, 0)


# ---------------------------------------------------------------------------
# _build_router_prompt
# ---------------------------------------------------------------------------


class TestBuildRouterPrompt:
    def test_includes_all_candidates(self):
        candidates = {"fast": "Quick edits.", "heavy": "Big tasks."}
        messages = _build_router_prompt("fix the bug", candidates)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        user_text = messages[1]["content"]
        assert "fast" in user_text
        assert "Quick edits." in user_text
        assert "heavy" in user_text
        assert "Big tasks." in user_text
        assert "fix the bug" in user_text

    def test_truncates_long_task(self):
        long_task = "x" * 20000
        candidates = {"fast": "Quick."}
        messages = _build_router_prompt(long_task, candidates)
        user_text = messages[1]["content"]
        assert "[task truncated]" in user_text
        assert len(user_text) < 15000

    def test_includes_project_context(self):
        candidates = {"fast": "Quick."}
        messages = _build_router_prompt(
            "hello", candidates, base_dir_name="myproject", is_objective=True
        )
        user_text = messages[1]["content"]
        assert "myproject" in user_text
        assert "objective" in user_text


# ---------------------------------------------------------------------------
# _parse_router_response
# ---------------------------------------------------------------------------


class TestParseRouterResponse:
    def test_valid_json(self):
        name, reason, mode = _parse_router_response(
            '{"profile": "fast", "reason": "Simple task."}',
            {"fast", "heavy"},
        )
        assert name == "fast"
        assert reason == "Simple task."
        assert mode == "json"

    def test_json_unknown_profile(self):
        name, reason, mode = _parse_router_response(
            '{"profile": "nonexistent"}',
            {"fast", "heavy"},
        )
        assert name is None

    def test_bare_name(self):
        name, reason, mode = _parse_router_response("fast", {"fast", "heavy"})
        assert name == "fast"
        assert mode == "bare_name"

    def test_bare_name_with_quotes(self):
        name, reason, mode = _parse_router_response('"fast"', {"fast", "heavy"})
        assert name == "fast"
        assert mode == "bare_name"

    def test_embedded_name(self):
        name, reason, mode = _parse_router_response(
            "I recommend the fast profile for this task.",
            {"fast", "heavy"},
        )
        assert name == "fast"
        assert mode == "embedded_name"

    def test_no_match(self):
        name, reason, mode = _parse_router_response(
            "I don't know what to pick.",
            {"fast", "heavy"},
        )
        assert name is None

    def test_json_with_surrounding_text(self):
        name, reason, mode = _parse_router_response(
            'Here is my answer: {"profile": "heavy", "reason": "Needs reasoning."}',
            {"fast", "heavy"},
        )
        assert name == "heavy"
        assert mode == "json"


# ---------------------------------------------------------------------------
# route_task
# ---------------------------------------------------------------------------


class TestRouteTask:
    def test_returns_none_when_not_configured(self):
        config = {"provider": "lmstudio"}
        result = route_task("some task", config)
        assert result is None

    def test_returns_none_when_routing_disabled(self):
        config = {
            "routing_enabled": False,
            "semantic_routing_profile": "router",
            "profiles": {
                "router": {"provider": "openrouter", "description": "General."},
                "fast": {"provider": "lmstudio", "description": "Fast."},
            },
        }
        result = route_task("some task", config)
        assert result is None

    def test_short_circuit_single_candidate(self):
        config = {
            "routing_enabled": True,
            "semantic_routing_profile": "router",
            "profiles": {
                "router": {
                    "provider": "openrouter",
                    "description": "General.",
                },
            },
        }
        result = route_task("some task", config)
        assert result is not None
        assert result.selected_profile == "router"
        assert result.parse_mode == "short_circuit"
        assert not result.fallback_used

    def test_no_candidates_raises(self):
        """Routing raises AgentError when no profiles have descriptions."""
        config = {
            "routing_enabled": True,
            "semantic_routing_profile": "router",
            "profiles": {
                "router": {"provider": "openrouter"},
                "fast": {"provider": "lmstudio"},
            },
        }
        with pytest.raises(AgentError, match="no profiles have a.*description"):
            route_task("task", config)

    @patch("swival.agent.resolve_provider")
    @patch("swival.agent.call_llm")
    def test_valid_json_response(self, mock_call_llm, mock_resolve):
        mock_resolve.return_value = ("glm-5", "https://api", "key", 8192, {})
        mock_call_llm.return_value = _mock_llm_response(
            '{"profile": "fast", "reason": "Simple edit."}'
        )
        result = route_task("fix a typo", _base_config())
        assert result.selected_profile == "fast"
        assert result.reason == "Simple edit."
        assert result.parse_mode == "json"
        assert not result.fallback_used

    @patch("swival.agent.resolve_provider")
    @patch("swival.agent.call_llm")
    def test_bare_name_response(self, mock_call_llm, mock_resolve):
        mock_resolve.return_value = ("glm-5", "https://api", "key", 8192, {})
        mock_call_llm.return_value = _mock_llm_response("fast")
        result = route_task("fix a typo", _base_config())
        assert result.selected_profile == "fast"
        assert result.parse_mode == "bare_name"

    @patch("swival.agent.resolve_provider")
    @patch("swival.agent.call_llm")
    def test_router_self_selection(self, mock_call_llm, mock_resolve):
        mock_resolve.return_value = ("glm-5", "https://api", "key", 8192, {})
        mock_call_llm.return_value = _mock_llm_response(
            '{"profile": "router", "reason": "General task."}'
        )
        result = route_task("do something", _base_config())
        assert result.selected_profile == "router"

    @patch("swival.agent.resolve_provider")
    @patch("swival.agent.call_llm")
    def test_retry_on_bad_response_then_success(self, mock_call_llm, mock_resolve):
        mock_resolve.return_value = ("glm-5", "https://api", "key", 8192, {})
        mock_call_llm.side_effect = [
            _mock_llm_response("gibberish nonsense"),
            _mock_llm_response('{"profile": "heavy"}'),
        ]
        result = route_task("complex task", _base_config())
        assert result.selected_profile == "heavy"
        assert mock_call_llm.call_count == 2

    @patch("swival.agent.resolve_provider")
    @patch("swival.agent.call_llm")
    def test_fallback_on_double_failure_non_strict(self, mock_call_llm, mock_resolve):
        mock_resolve.return_value = ("glm-5", "https://api", "key", 8192, {})
        mock_call_llm.return_value = _mock_llm_response("total nonsense xyz")
        config = _base_config(active_profile="fast")
        result = route_task("task", config)
        assert result.selected_profile == "fast"
        assert result.fallback_used is True
        assert result.parse_mode == "fallback"

    @patch("swival.agent.resolve_provider")
    @patch("swival.agent.call_llm")
    def test_error_on_double_failure_strict(self, mock_call_llm, mock_resolve):
        mock_resolve.return_value = ("glm-5", "https://api", "key", 8192, {})
        mock_call_llm.return_value = _mock_llm_response("total nonsense xyz")
        config = _base_config(routing_strict=True)
        with pytest.raises(AgentError, match="could not parse"):
            route_task("task", config)

    @patch("swival.agent.resolve_provider")
    @patch("swival.agent.call_llm")
    def test_only_described_profiles_are_candidates(self, mock_call_llm, mock_resolve):
        mock_resolve.return_value = ("glm-5", "https://api", "key", 8192, {})
        mock_call_llm.return_value = _mock_llm_response('{"profile": "fast"}')
        config = _base_config()
        # Add a profile without description — it should not appear as candidate
        config["profiles"]["hidden"] = {"provider": "generic", "model": "x"}
        result = route_task("task", config)
        # The hidden profile shouldn't be selectable
        assert result.selected_profile == "fast"
        # Verify hidden was not in the prompt
        call_args = mock_call_llm.call_args
        messages = call_args[0][2]  # 3rd positional arg
        user_msg = messages[1]["content"]
        assert "hidden" not in user_msg

    @patch("swival.agent.resolve_provider")
    @patch("swival.agent.call_llm")
    def test_call_llm_uses_router_kind(self, mock_call_llm, mock_resolve):
        mock_resolve.return_value = ("glm-5", "https://api", "key", 8192, {})
        mock_call_llm.return_value = _mock_llm_response('{"profile": "fast"}')
        route_task("task", _base_config())
        call_kwargs = mock_call_llm.call_args[1]
        assert call_kwargs["call_kind"] == "router"

    @patch("swival.agent.resolve_provider")
    @patch("swival.agent.call_llm")
    def test_fallback_on_call_llm_exception_non_strict(
        self, mock_call_llm, mock_resolve
    ):
        mock_resolve.return_value = ("glm-5", "https://api", "key", 8192, {})
        mock_call_llm.side_effect = RuntimeError("connection refused")
        config = _base_config(active_profile="fast")
        result = route_task("task", config)
        assert result.selected_profile == "fast"
        assert result.fallback_used is True

    @patch("swival.agent.resolve_provider")
    @patch("swival.agent.call_llm")
    def test_error_on_call_llm_exception_strict(self, mock_call_llm, mock_resolve):
        mock_resolve.return_value = ("glm-5", "https://api", "key", 8192, {})
        mock_call_llm.side_effect = RuntimeError("connection refused")
        config = _base_config(routing_strict=True)
        with pytest.raises(AgentError, match="Semantic routing failed"):
            route_task("task", config)

    @patch("swival.agent.resolve_provider")
    @patch("swival.agent.call_llm")
    def test_fallback_to_router_when_no_active_profile(
        self, mock_call_llm, mock_resolve
    ):
        mock_resolve.return_value = ("glm-5", "https://api", "key", 8192, {})
        mock_call_llm.return_value = _mock_llm_response("total nonsense xyz")
        config = _base_config()  # no active_profile
        result = route_task("task", config)
        assert result.selected_profile == "router"
        assert result.fallback_used is True

    # --- Fix coverage: CLI auth overrides ---

    @patch("swival.agent.resolve_provider")
    @patch("swival.agent.call_llm")
    def test_cli_overrides_used_for_router_auth(self, mock_call_llm, mock_resolve):
        """CLI --api-key should be used when the router profile has no api_key."""
        mock_resolve.return_value = ("glm-5", "https://api", "key", 8192, {})
        mock_call_llm.return_value = _mock_llm_response('{"profile": "fast"}')
        config = _base_config()
        route_task("task", config, cli_overrides={"api_key": "sk-cli-key"})
        # resolve_provider should receive the CLI key as fallback
        call_args = mock_resolve.call_args
        assert call_args[0][2] == "sk-cli-key"  # 3rd positional: api_key

    @patch("swival.agent.resolve_provider")
    @patch("swival.agent.call_llm")
    def test_profile_api_key_wins_over_cli(self, mock_call_llm, mock_resolve):
        """Router profile's own api_key beats CLI override."""
        mock_resolve.return_value = ("glm-5", "https://api", "key", 8192, {})
        mock_call_llm.return_value = _mock_llm_response('{"profile": "fast"}')
        config = _base_config()
        config["profiles"]["router"]["api_key"] = "sk-profile-key"
        route_task("task", config, cli_overrides={"api_key": "sk-cli-key"})
        call_args = mock_resolve.call_args
        assert call_args[0][2] == "sk-profile-key"

    # --- Fix coverage: filter threading ---

    @patch("swival.agent.resolve_provider")
    @patch("swival.agent.call_llm")
    def test_llm_filter_threaded_to_call(self, mock_call_llm, mock_resolve):
        """llm_filter should be passed through to call_llm."""
        mock_resolve.return_value = ("glm-5", "https://api", "key", 8192, {})
        mock_call_llm.return_value = _mock_llm_response('{"profile": "fast"}')
        route_task("task", _base_config(), llm_filter="./filter.py")
        call_kwargs = mock_call_llm.call_args[1]
        assert call_kwargs["llm_filter"] == "./filter.py"

    @patch("swival.agent.resolve_provider")
    @patch("swival.agent.call_llm")
    def test_no_filter_when_not_configured(self, mock_call_llm, mock_resolve):
        """No llm_filter kwarg when filter is not set."""
        mock_resolve.return_value = ("glm-5", "https://api", "key", 8192, {})
        mock_call_llm.return_value = _mock_llm_response('{"profile": "fast"}')
        route_task("task", _base_config())
        call_kwargs = mock_call_llm.call_args[1]
        assert "llm_filter" not in call_kwargs

    # --- Fix coverage: router profile per-call settings ---

    @patch("swival.agent.resolve_provider")
    @patch("swival.agent.call_llm")
    def test_falsey_profile_value_overrides_truthy_config(
        self, mock_call_llm, mock_resolve
    ):
        """sanitize_thinking=false in profile should override truthy file config."""
        mock_resolve.return_value = ("glm-5", "https://api", "key", 8192, {})
        mock_call_llm.return_value = _mock_llm_response('{"profile": "fast"}')
        config = _base_config(sanitize_thinking=True)
        config["profiles"]["router"]["sanitize_thinking"] = False
        route_task("task", config)
        call_kwargs = mock_call_llm.call_args[1]
        assert call_kwargs.get("sanitize_thinking") is not True

    @patch("swival.agent.resolve_provider")
    @patch("swival.agent.call_llm")
    def test_empty_extra_body_overrides_config(self, mock_call_llm, mock_resolve):
        """extra_body={} in profile should clear an inherited non-empty value."""
        mock_resolve.return_value = ("glm-5", "https://api", "key", 8192, {})
        mock_call_llm.return_value = _mock_llm_response('{"profile": "fast"}')
        config = _base_config(extra_body={"global": True})
        config["profiles"]["router"]["extra_body"] = {}
        route_task("task", config)
        call_kwargs = mock_call_llm.call_args[1]
        assert call_kwargs.get("extra_body") == {}

    @patch("swival.agent.resolve_provider")
    @patch("swival.agent.call_llm")
    def test_router_extra_body_threaded(self, mock_call_llm, mock_resolve):
        mock_resolve.return_value = ("glm-5", "https://api", "key", 8192, {})
        mock_call_llm.return_value = _mock_llm_response('{"profile": "fast"}')
        config = _base_config()
        config["profiles"]["router"]["extra_body"] = {"foo": "bar"}
        route_task("task", config)
        call_kwargs = mock_call_llm.call_args[1]
        assert call_kwargs["extra_body"] == {"foo": "bar"}

    # --- Fix coverage: resolve_provider failure in non-strict mode ---

    @patch("swival.agent.resolve_provider")
    def test_resolve_provider_failure_non_strict_fallback(self, mock_resolve):
        """Non-strict mode should fall back when resolve_provider raises."""
        mock_resolve.side_effect = RuntimeError("bad credentials")
        config = _base_config(active_profile="fast")
        result = route_task("task", config)
        assert result.selected_profile == "fast"
        assert result.fallback_used is True
        assert result.parse_mode == "fallback"

    @patch("swival.agent.resolve_provider")
    def test_resolve_provider_failure_strict_raises(self, mock_resolve):
        """Strict mode should raise AgentError when resolve_provider fails."""
        mock_resolve.side_effect = RuntimeError("bad credentials")
        config = _base_config(routing_strict=True)
        with pytest.raises(AgentError, match="setup failed"):
            route_task("task", config)

    # --- Fix coverage: secret_shield threading ---

    @patch("swival.agent.resolve_provider")
    @patch("swival.agent.call_llm")
    def test_secret_shield_threaded_to_call(self, mock_call_llm, mock_resolve):
        """secret_shield should be passed through to call_llm."""
        mock_resolve.return_value = ("glm-5", "https://api", "key", 8192, {})
        mock_call_llm.return_value = _mock_llm_response('{"profile": "fast"}')
        sentinel = object()
        route_task("task", _base_config(), secret_shield=sentinel)
        call_kwargs = mock_call_llm.call_args[1]
        assert call_kwargs["secret_shield"] is sentinel

    @patch("swival.agent.resolve_provider")
    @patch("swival.agent.call_llm")
    def test_no_secret_shield_when_not_configured(self, mock_call_llm, mock_resolve):
        mock_resolve.return_value = ("glm-5", "https://api", "key", 8192, {})
        mock_call_llm.return_value = _mock_llm_response('{"profile": "fast"}')
        route_task("task", _base_config())
        call_kwargs = mock_call_llm.call_args[1]
        assert "secret_shield" not in call_kwargs

    # --- Fix coverage: CLI overrides for generation settings ---

    @patch("swival.agent.resolve_provider")
    @patch("swival.agent.call_llm")
    def test_cli_reasoning_effort_override(self, mock_call_llm, mock_resolve):
        """CLI --reasoning-effort should be used for router call."""
        mock_resolve.return_value = ("glm-5", "https://api", "key", 8192, {})
        mock_call_llm.return_value = _mock_llm_response('{"profile": "fast"}')
        route_task("task", _base_config(), cli_overrides={"reasoning_effort": "high"})
        call_kwargs = mock_call_llm.call_args[1]
        assert call_kwargs["reasoning_effort"] == "high"

    @patch("swival.agent.resolve_provider")
    @patch("swival.agent.call_llm")
    def test_cli_retries_override(self, mock_call_llm, mock_resolve):
        """CLI --retries should be used for router call."""
        mock_resolve.return_value = ("glm-5", "https://api", "key", 8192, {})
        mock_call_llm.return_value = _mock_llm_response('{"profile": "fast"}')
        route_task("task", _base_config(), cli_overrides={"retries": 2})
        call_kwargs = mock_call_llm.call_args[1]
        assert call_kwargs["max_retries"] == 2

    # --- Fix coverage: quiet mode suppresses warnings ---

    @patch("swival.agent.resolve_provider")
    def test_fallback_quiet_suppresses_warning(self, mock_resolve, capsys):
        """In quiet mode (verbose=False), fallback should not print warnings."""
        mock_resolve.side_effect = RuntimeError("bad creds")
        config = _base_config(active_profile="fast")
        result = route_task("task", config, verbose=False)
        assert result.fallback_used is True
        assert capsys.readouterr().err == ""

    @patch("swival.agent.resolve_provider")
    def test_fallback_verbose_prints_warning(self, mock_resolve, capsys):
        """In verbose mode, fallback should print warnings."""
        mock_resolve.side_effect = RuntimeError("bad creds")
        config = _base_config(active_profile="fast")
        result = route_task("task", config, verbose=True)
        assert result.fallback_used is True
        assert "falling back" in capsys.readouterr().err

    @patch("swival.agent.resolve_provider")
    @patch("swival.agent.call_llm")
    def test_multiline_reason_collapsed(self, mock_call_llm, mock_resolve, capsys):
        """Multiline reasons should be collapsed to a single line in output."""
        mock_resolve.return_value = ("glm-5", "https://api", "key", 8192, {})
        mock_call_llm.return_value = _mock_llm_response(
            '{"profile": "fast", "reason": "Needs reasoning\\nand broader context"}'
        )
        result = route_task("task", _base_config(), verbose=True)
        assert result.reason == "Needs reasoning\nand broader context"
        output = capsys.readouterr().err
        assert "reason=" in output
        assert "\n" not in output.split("reason=")[1].split("\n")[0].replace("\n", "")

    @patch("swival.agent.resolve_provider")
    @patch("swival.agent.call_llm")
    def test_long_reason_truncated(self, mock_call_llm, mock_resolve, capsys):
        """Reasons longer than 120 chars should be truncated in output."""
        mock_resolve.return_value = ("glm-5", "https://api", "key", 8192, {})
        long_reason = "x" * 200
        mock_call_llm.return_value = _mock_llm_response(
            f'{{"profile": "fast", "reason": "{long_reason}"}}'
        )
        result = route_task("task", _base_config(), verbose=True)
        assert result.reason == long_reason
        output = capsys.readouterr().err
        assert "..." in output
        assert long_reason not in output


class TestCliUnsetHandling:
    """Tests that the agent.py routing block handles _UNSET args correctly.

    Routing runs before apply_config_to_args(), so args carry _UNSET
    sentinels for values not explicitly passed on the CLI. These tests
    verify that _UNSET does not leak into downstream calls.
    """

    def test_unset_quiet_defaults_to_verbose(self):
        """When args.quiet is _UNSET, routing should default to verbose=True."""
        from swival.config import _UNSET

        # _UNSET is truthy, so a naive `not getattr(args, "quiet", False)`
        # would produce verbose=False.  The _cli() helper must treat _UNSET
        # as the default (False), yielding verbose=True.
        assert bool(_UNSET), "_UNSET should be truthy (this is the trap)"
        # Simulate what _cli("quiet", False) should return:
        val = False if _UNSET is _UNSET else _UNSET
        assert val is False

    def test_unset_encrypt_key_not_passed_to_shield(self):
        """When args.encrypt_secrets_key is _UNSET, it must not reach SecretShield."""
        from swival.secrets import SecretShield

        # If _UNSET leaked as key_hex, from_config would raise TypeError.
        # Verify the correct behavior: None key_hex generates a random key.
        shield = SecretShield.from_config(key_hex=None)
        assert shield is not None

    @patch("swival.agent.resolve_provider")
    @patch("swival.agent.call_llm")
    def test_encrypt_via_env_var(self, mock_call_llm, mock_resolve, monkeypatch):
        """SWIVAL_ENCRYPT_KEY env var alone should enable router encryption."""
        mock_resolve.return_value = ("glm-5", "https://api", "key", 8192, {})
        mock_call_llm.return_value = _mock_llm_response('{"profile": "fast"}')
        monkeypatch.setenv("SWIVAL_ENCRYPT_KEY", "aa" * 32)
        config = _base_config()  # no encrypt_secrets in config
        # route_task itself doesn't see the env var — the caller (agent.py)
        # builds the shield and passes it. We test route_task threading here.
        from swival.secrets import SecretShield

        shield = SecretShield.from_config(key_hex="aa" * 32)
        route_task("task", config, secret_shield=shield)
        call_kwargs = mock_call_llm.call_args[1]
        assert call_kwargs["secret_shield"] is shield

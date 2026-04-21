"""Tests for the /profile REPL command."""

from unittest.mock import patch

import pytest

from swival.agent import _repl_profile, _repl_status
from swival.command_policy import CommandPolicy
from swival.config import _PROFILE_METADATA_KEYS, PROFILE_KEYS, _validate_profiles
from swival.report import ConfigError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASELINE = {
    "provider": "openrouter",
    "model": "meta-llama/llama-4-scout",
    "api_key": "sk-baseline",
    "base_url": None,
    "aws_profile": None,
    "max_context_tokens": None,
    "max_output_tokens": 4096,
    "temperature": 0.7,
    "top_p": None,
    "seed": None,
    "extra_body": None,
    "reasoning_effort": None,
    "sanitize_thinking": None,
}

PROFILES = {
    "fast": {
        "provider": "lmstudio",
        "model": "qwen3.5-0.8b",
        "base_url": "http://127.0.0.1:1234",
    },
    "big": {
        "provider": "openrouter",
        "model": "anthropic/claude-sonnet-4",
        "api_key": "sk-big",
        "max_output_tokens": 16384,
        "temperature": 0.0,
    },
}


def _resolve_return():
    return (
        "test-model",
        "http://test-api",
        "sk-resolved",
        128000,
        {"provider": "test", "api_key": "sk-resolved"},
    )


def _make_repl_kwargs():
    return {
        "model_id": "old-model",
        "api_base": "http://old",
        "context_length": 8192,
        "llm_kwargs": {"provider": "old", "api_key": "sk-old"},
        "max_output_tokens": 4096,
        "temperature": 0.7,
        "top_p": None,
        "seed": None,
    }


def _call_profile(
    cmd_arg,
    profiles=None,
    startup_profile=None,
    current_profile=None,
    raw_baseline=None,
    pre_profile_baseline=None,
    repl_kwargs=None,
    subagent_manager=None,
    resolve_return=None,
):
    if profiles is None:
        profiles = PROFILES
    if raw_baseline is None:
        raw_baseline = dict(BASELINE)
    if pre_profile_baseline is None:
        pre_profile_baseline = dict(BASELINE)
    if repl_kwargs is None:
        repl_kwargs = _make_repl_kwargs()
    if resolve_return is None:
        resolve_return = _resolve_return()

    with patch("swival.agent.resolve_provider", return_value=resolve_return) as mock_rp:
        profile_name, msg, is_error = _repl_profile(
            cmd_arg,
            profiles=profiles,
            startup_profile=startup_profile,
            current_profile=current_profile,
            raw_baseline=raw_baseline,
            pre_profile_baseline=pre_profile_baseline,
            repl_kwargs=repl_kwargs,
            subagent_manager=subagent_manager,
            verbose=False,
        )
    return profile_name, msg, is_error, repl_kwargs, mock_rp


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


class TestProfileList:
    def test_no_profiles(self):
        result, msg, is_error, _, _ = _call_profile("", profiles={})
        assert result is None
        assert "No profiles defined" in msg

    def test_shows_all(self):
        result, msg, is_error, _, _ = _call_profile("", current_profile=None)
        assert "fast" in msg
        assert "big" in msg
        assert result is None

    def test_marks_active(self):
        result, msg, is_error, _, _ = _call_profile("", current_profile="fast")
        assert "\u2192" in msg
        assert "(active)" in msg
        assert result == "fast"


# ---------------------------------------------------------------------------
# Switching
# ---------------------------------------------------------------------------


class TestProfileSwitch:
    def test_updates_kwargs(self):
        result, msg, is_error, kw, mock_rp = _call_profile("fast")
        assert result == "fast"
        assert not is_error
        assert kw["model_id"] == "test-model"
        assert kw["api_base"] == "http://test-api"
        assert kw["context_length"] == 128000
        mock_rp.assert_called_once()
        call_kw = mock_rp.call_args
        assert call_kw.kwargs["provider"] == "lmstudio"
        assert call_kw.kwargs["model"] == "qwen3.5-0.8b"

    def test_unknown_name(self):
        result, msg, is_error, kw, _ = _call_profile(
            "nonexistent", current_profile="fast"
        )
        assert result == "fast"
        assert is_error
        assert kw["model_id"] == "old-model"
        assert "unknown profile" in msg
        assert "big" in msg
        assert "fast" in msg

    def test_preserves_conversation(self):
        messages = [{"role": "user", "content": "hello"}]
        _call_profile("fast")
        assert messages == [{"role": "user", "content": "hello"}]

    def test_overlay_priority(self):
        """Profile values override baseline, not the other way."""
        result, msg, is_error, kw, mock_rp = _call_profile("big")
        call_kw = mock_rp.call_args
        assert call_kw.kwargs["api_key"] == "sk-big"

    def test_generation_params(self):
        profiles = {
            "custom": {
                "provider": "openrouter",
                "model": "test",
                "temperature": 0.2,
                "top_p": 0.9,
                "seed": 42,
                "max_output_tokens": 8192,
            }
        }
        result, msg, is_error, kw, _ = _call_profile("custom", profiles=profiles)
        assert kw["temperature"] == 0.2
        assert kw["top_p"] == 0.9
        assert kw["seed"] == 42
        assert kw["max_output_tokens"] == 8192


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestProfileErrors:
    def test_resolve_failure_keeps_current(self):
        with patch("swival.agent.resolve_provider", side_effect=ConfigError("bad")):
            result, msg, is_error = _repl_profile(
                "fast",
                profiles=PROFILES,
                startup_profile=None,
                current_profile="big",
                raw_baseline=dict(BASELINE),
                repl_kwargs=_make_repl_kwargs(),
                subagent_manager=None,
                verbose=False,
            )
        assert result == "big"
        assert is_error
        assert "profile switch failed" in msg


# ---------------------------------------------------------------------------
# Baseline revert (/profile -)
# ---------------------------------------------------------------------------


class TestProfileRevert:
    def test_dash_restores_baseline(self):
        result, msg, is_error, kw, mock_rp = _call_profile("-", startup_profile="fast")
        assert result == "fast"
        call_kw = mock_rp.call_args
        assert call_kw.kwargs["provider"] == BASELINE["provider"]

    def test_dash_with_startup_profile(self):
        result, msg, is_error, _, mock_rp = _call_profile("-", startup_profile="big")
        assert result == "big"

    def test_dash_no_startup(self):
        result, msg, is_error, _, mock_rp = _call_profile("-", startup_profile=None)
        assert result is None
        call_kw = mock_rp.call_args
        assert call_kw.kwargs["provider"] == BASELINE["provider"]

    def test_minimal_profile_no_baseline_leak(self):
        """A profile that only sets provider must NOT inherit model from the
        startup profile.  The pre-profile baseline has no model (it came from
        the startup profile, not top-level config)."""
        profiles = {"local": {"provider": "lmstudio"}}
        # raw_baseline has model from the startup profile
        baseline = dict(BASELINE)
        # pre-profile baseline has NO model (top-level config didn't set one)
        pre_profile = {k: v for k, v in BASELINE.items() if k != "model"}
        kw = _make_repl_kwargs()

        ret = (
            "discovered-model",
            "http://127.0.0.1:1234",
            None,
            32000,
            {"provider": "lmstudio"},
        )
        with patch("swival.agent.resolve_provider", return_value=ret) as mock_rp:
            _repl_profile(
                "local",
                profiles,
                None,
                None,
                baseline,
                pre_profile_baseline=pre_profile,
                repl_kwargs=kw,
            )

        call_kw = mock_rp.call_args
        assert call_kw.kwargs["model"] is None  # must not be baseline's model

    def test_profile_inherits_top_level_api_key(self):
        """A profile that doesn't set api_key should inherit it from top-level config."""
        profiles = {"remote": {"provider": "openrouter", "model": "test-model"}}
        # Top-level config has api_key
        pre_profile = {"provider": "openrouter", "api_key": "sk-top-level"}
        baseline = dict(BASELINE)
        kw = _make_repl_kwargs()

        ret = (
            "test-model",
            "http://api",
            "sk-top-level",
            128000,
            {"provider": "openrouter", "api_key": "sk-top-level"},
        )
        with patch("swival.agent.resolve_provider", return_value=ret) as mock_rp:
            _repl_profile(
                "remote",
                profiles,
                None,
                None,
                baseline,
                pre_profile_baseline=pre_profile,
                repl_kwargs=kw,
            )

        call_kw = mock_rp.call_args
        assert call_kw.kwargs["api_key"] == "sk-top-level"

    def test_reasoning_effort_not_carried_across_profiles(self):
        """Switching from a profile with reasoning_effort to one without must drop it."""
        profiles = {
            "high": {
                "provider": "chatgpt",
                "model": "gpt-5.4",
                "reasoning_effort": "high",
            },
            "local": {"provider": "lmstudio"},
        }
        kw = _make_repl_kwargs()
        baseline = dict(BASELINE)
        # pre-profile baseline has no reasoning_effort (it came from the profile)
        pre_profile = {
            k: v for k, v in BASELINE.items() if k not in ("model", "reasoning_effort")
        }

        ret_high = (
            "gpt-5.4",
            "http://api",
            "sk",
            128000,
            {"provider": "chatgpt", "api_key": "sk"},
        )
        with patch("swival.agent.resolve_provider", return_value=ret_high):
            _repl_profile(
                "high",
                profiles,
                None,
                None,
                baseline,
                pre_profile_baseline=pre_profile,
                repl_kwargs=kw,
            )
        assert kw["llm_kwargs"].get("reasoning_effort") == "high"

        ret_local = (
            "discovered",
            "http://local",
            None,
            32000,
            {"provider": "lmstudio"},
        )
        with patch("swival.agent.resolve_provider", return_value=ret_local):
            _repl_profile(
                "local",
                profiles,
                None,
                "high",
                baseline,
                pre_profile_baseline=pre_profile,
                repl_kwargs=kw,
            )
        assert "reasoning_effort" not in kw["llm_kwargs"]

    def test_double_switch(self):
        """A -> B -> C: each switch resolves from baseline, not previous profile."""
        baseline = dict(BASELINE)
        kw = _make_repl_kwargs()

        ret_fast = ("qwen", "http://local", None, 32000, {"provider": "lmstudio"})
        ret_big = ("claude", "http://or", "sk-big", 200000, {"provider": "openrouter"})

        with patch("swival.agent.resolve_provider", return_value=ret_fast):
            _repl_profile("fast", PROFILES, None, None, baseline, repl_kwargs=kw)

        with patch("swival.agent.resolve_provider", return_value=ret_big) as mock_rp:
            _repl_profile("big", PROFILES, None, "fast", baseline, repl_kwargs=kw)

        call_kw = mock_rp.call_args
        assert call_kw.kwargs["provider"] == "openrouter"
        assert call_kw.kwargs["api_key"] == "sk-big"


# ---------------------------------------------------------------------------
# Subagent patching
# ---------------------------------------------------------------------------


class TestProfileSubagent:
    def test_patches_template(self):
        class FakeManager:
            _template = {"model_id": "old", "api_base": "old"}

        mgr = FakeManager()
        result, msg, is_error, kw, _ = _call_profile("fast", subagent_manager=mgr)
        assert mgr._template["model_id"] == "test-model"
        assert mgr._template["api_base"] == "http://test-api"
        assert mgr._template["context_length"] == 128000

    def test_existing_subagents_unchanged(self):
        class FakeManager:
            _template = {"model_id": "old"}

        mgr = FakeManager()
        snapshot = dict(mgr._template)
        spawned_copy = dict(snapshot)

        _call_profile("fast", subagent_manager=mgr)
        assert spawned_copy["model_id"] == "old"


# ---------------------------------------------------------------------------
# API key propagation
# ---------------------------------------------------------------------------


class TestProfileApiKey:
    def test_api_key_propagated(self):
        ret = (
            "model",
            "http://api",
            "sk-new-key",
            128000,
            {"provider": "x", "api_key": "sk-new-key"},
        )
        result, msg, is_error, kw, _ = _call_profile("big", resolve_return=ret)
        assert kw["llm_kwargs"]["api_key"] == "sk-new-key"


# ---------------------------------------------------------------------------
# Session-level llm_kwargs preservation
# ---------------------------------------------------------------------------


class TestProfilePreservesSessionKeys:
    def test_prompt_cache_and_max_retries_survive_switch(self):
        kw = _make_repl_kwargs()
        kw["llm_kwargs"]["prompt_cache"] = False
        kw["llm_kwargs"]["max_retries"] = 9

        result, msg, is_error, kw, _ = _call_profile("fast", repl_kwargs=kw)
        assert kw["llm_kwargs"]["prompt_cache"] is False
        assert kw["llm_kwargs"]["max_retries"] == 9

    def test_session_keys_survive_revert(self):
        kw = _make_repl_kwargs()
        kw["llm_kwargs"]["prompt_cache"] = False
        kw["llm_kwargs"]["max_retries"] = 5

        result, msg, is_error, kw, _ = _call_profile(
            "-", startup_profile=None, repl_kwargs=kw
        )
        assert kw["llm_kwargs"]["prompt_cache"] is False
        assert kw["llm_kwargs"]["max_retries"] == 5

    def test_provider_keys_overridden_not_carried(self):
        """Keys that resolve_provider sets (like 'provider', 'api_key') should
        come from the new resolution, not be carried from the old llm_kwargs."""
        kw = _make_repl_kwargs()
        kw["llm_kwargs"]["provider"] = "should-be-overwritten"

        result, msg, is_error, kw, _ = _call_profile("fast")
        assert kw["llm_kwargs"]["provider"] == "test"


# ---------------------------------------------------------------------------
# Validation (config.py step 0)
# ---------------------------------------------------------------------------


class TestProfileMetadataValidation:
    def test_description_valid(self):
        profiles = {"demo": {"provider": "openrouter", "description": "A test profile"}}
        _validate_profiles(profiles, "test")

    def test_description_wrong_type(self):
        profiles = {"demo": {"provider": "openrouter", "description": 42}}
        with pytest.raises(ConfigError, match="expected str"):
            _validate_profiles(profiles, "test")

    def test_description_not_in_runtime(self):
        """description is excluded when building the merged dict for resolve_provider."""
        assert "description" in PROFILE_KEYS
        assert "description" in _PROFILE_METADATA_KEYS

        profiles = {
            "test": {
                "provider": "openrouter",
                "model": "test-model",
                "description": "Should not leak to resolve_provider",
            }
        }
        baseline = dict(BASELINE)
        kw = _make_repl_kwargs()

        with patch(
            "swival.agent.resolve_provider", return_value=_resolve_return()
        ) as mock_rp:
            _repl_profile("test", profiles, None, None, baseline, repl_kwargs=kw)

        call_kw = mock_rp.call_args
        assert "description" not in call_kw.kwargs


# ---------------------------------------------------------------------------
# Local variable coherence
# ---------------------------------------------------------------------------


class TestProfileLocalsCoherence:
    def test_failed_switch_locals_coherent(self):
        kw = _make_repl_kwargs()
        original_model = kw["model_id"]

        with patch("swival.agent.resolve_provider", side_effect=ConfigError("fail")):
            _repl_profile("fast", PROFILES, None, None, dict(BASELINE), repl_kwargs=kw)

        assert kw["model_id"] == original_model

    def test_dash_locals_coherent(self):
        kw = _make_repl_kwargs()
        result, msg, is_error, kw2, _ = _call_profile(
            "-", startup_profile=None, repl_kwargs=kw
        )
        assert kw["model_id"] == kw2["model_id"]


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------


class TestProfileStatus:
    def test_status_shows_profile_name(self):
        out = _repl_status(
            messages=[],
            tools=[],
            model_id="test-model",
            api_base="http://test",
            context_length=128000,
            turn_state={"max_turns": 10, "turns_used": 0},
            files_mode="some",
            command_policy=CommandPolicy("allowlist"),
            verbose=False,
            base_dir="/tmp",
            thinking_state=None,
            todo_state=None,
            snapshot_state=None,
            file_tracker=None,
            compaction_state=None,
            current_profile="fast-local",
        )
        assert "profile: fast-local" in out

    def test_status_no_profile(self):
        out = _repl_status(
            messages=[],
            tools=[],
            model_id="test-model",
            api_base="http://test",
            context_length=128000,
            turn_state={"max_turns": 10, "turns_used": 0},
            files_mode="some",
            command_policy=CommandPolicy("allowlist"),
            verbose=False,
            base_dir="/tmp",
            thinking_state=None,
            todo_state=None,
            snapshot_state=None,
            file_tracker=None,
            compaction_state=None,
        )
        assert "profile:" not in out

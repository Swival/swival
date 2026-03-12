"""Tests for the --self-review feature."""

import argparse
import shlex
import sys

import pytest
from unittest.mock import patch, MagicMock

from swival import agent
from swival.config import (
    _UNSET,
    apply_config_to_args,
    config_to_session_kwargs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(**overrides):
    """Build a namespace with resolved (post-config-merge) defaults."""
    defaults = dict(
        provider="lmstudio",
        model=None,
        api_key=None,
        base_url=None,
        max_output_tokens=32768,
        max_context_tokens=None,
        yolo=False,
        skills_dir=[],
        report=None,
        cache=False,
        self_review=False,
        reviewer=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_unset_args(**overrides):
    """Build a namespace mimicking build_parser() with _UNSET sentinels."""
    defaults = {
        "provider": _UNSET,
        "model": _UNSET,
        "api_key": _UNSET,
        "base_url": _UNSET,
        "max_output_tokens": _UNSET,
        "max_context_tokens": _UNSET,
        "temperature": _UNSET,
        "top_p": _UNSET,
        "seed": _UNSET,
        "max_turns": _UNSET,
        "system_prompt": _UNSET,
        "no_system_prompt": _UNSET,
        "allowed_commands": _UNSET,
        "yolo": _UNSET,
        "add_dir": None,
        "add_dir_ro": None,
        "sandbox": _UNSET,
        "sandbox_session": _UNSET,
        "sandbox_strict_read": _UNSET,
        "no_sandbox_auto_session": _UNSET,
        "no_read_guard": _UNSET,
        "no_instructions": _UNSET,
        "no_skills": _UNSET,
        "skills_dir": None,
        "no_history": _UNSET,
        "no_memory": _UNSET,
        "no_continue": _UNSET,
        "color": _UNSET,
        "no_color": _UNSET,
        "quiet": _UNSET,
        "reviewer": _UNSET,
        "self_review": _UNSET,
        "review_prompt": _UNSET,
        "objective": _UNSET,
        "verify": _UNSET,
        "max_review_rounds": _UNSET,
        "proactive_summaries": _UNSET,
        "no_mcp": _UNSET,
        "mcp_config": _UNSET,
        "extra_body": _UNSET,
        "reasoning_effort": _UNSET,
        "cache": _UNSET,
        "cache_dir": _UNSET,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ===========================================================================
# _build_self_review_cmd unit tests
# ===========================================================================


class TestBuildSelfReviewCmd:
    def test_basic(self):
        args = _make_args(provider="chatgpt", model="gpt-5.4", yolo=True)
        cmd = agent._build_self_review_cmd(args)
        parts = shlex.split(cmd)
        assert "--reviewer-mode" in parts
        assert "--quiet" in parts
        assert "--yolo" in parts
        assert "--provider" in parts
        idx = parts.index("--provider")
        assert parts[idx + 1] == "chatgpt"
        idx = parts.index("--model")
        assert parts[idx + 1] == "gpt-5.4"

    def test_skills_dir_multiple(self):
        args = _make_args(skills_dir=["/path/a", "/path/b"])
        cmd = agent._build_self_review_cmd(args)
        parts = shlex.split(cmd)
        indices = [i for i, p in enumerate(parts) if p == "--skills-dir"]
        assert len(indices) == 2
        assert parts[indices[0] + 1] == "/path/a"
        assert parts[indices[1] + 1] == "/path/b"

    def test_base_url(self):
        args = _make_args(
            provider="generic", model="m", base_url="https://my.server/v1"
        )
        cmd = agent._build_self_review_cmd(args)
        parts = shlex.split(cmd)
        idx = parts.index("--base-url")
        assert parts[idx + 1] == "https://my.server/v1"

    def test_no_api_key_on_cmdline(self):
        args = _make_args(provider="openrouter", model="m", api_key="sk-secret-key")
        cmd = agent._build_self_review_cmd(args)
        assert "--api-key" not in cmd
        assert "sk-secret-key" not in cmd

    def test_no_report_no_cache(self):
        args = _make_args(report="/tmp/full.json", cache=True)
        cmd = agent._build_self_review_cmd(args)
        assert "--report" not in cmd
        assert "--cache" not in cmd

    def test_correct_token_flags(self):
        args = _make_args(max_context_tokens=65536, max_output_tokens=16384)
        cmd = agent._build_self_review_cmd(args)
        parts = shlex.split(cmd)
        idx = parts.index("--max-context-tokens")
        assert parts[idx + 1] == "65536"
        idx = parts.index("--max-output-tokens")
        assert parts[idx + 1] == "16384"

    def test_default_max_output_tokens_not_mirrored(self):
        args = _make_args(max_output_tokens=32768)
        cmd = agent._build_self_review_cmd(args)
        assert "--max-output-tokens" not in cmd

    def test_lmstudio_provider_not_mirrored(self):
        args = _make_args(provider="lmstudio")
        cmd = agent._build_self_review_cmd(args)
        assert "--provider" not in cmd


# ===========================================================================
# Validation / conflict tests
# ===========================================================================


class TestSelfReviewValidation:
    def test_self_review_and_reviewer_conflict(self, tmp_path, monkeypatch):
        """--self-review and --reviewer together should error in main()."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "swival",
                "--self-review",
                "--reviewer",
                "echo",
                "--base-dir",
                str(tmp_path),
                "question",
            ],
        )
        with pytest.raises(SystemExit):
            agent.main()

    def test_self_review_and_repl_conflict(self, tmp_path, monkeypatch):
        """--self-review and --repl together should error in main()."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        monkeypatch.setattr(
            sys,
            "argv",
            ["swival", "--self-review", "--repl", "--base-dir", str(tmp_path)],
        )
        with pytest.raises(SystemExit):
            agent.main()

    def test_reviewer_mode_and_self_review_cli_conflict(self, tmp_path, monkeypatch):
        """--reviewer-mode --self-review on CLI should error."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        monkeypatch.setattr(
            sys,
            "argv",
            ["swival", "--reviewer-mode", "--self-review", str(tmp_path)],
        )
        with pytest.raises(SystemExit):
            agent.main()


# ===========================================================================
# Config integration
# ===========================================================================


class TestSelfReviewConfig:
    def test_config_self_review_true(self):
        """self_review = true in config should set args.self_review = True."""
        args = _make_unset_args()
        config = {"self_review": True}
        apply_config_to_args(args, config)
        assert args.self_review is True

    def test_config_self_review_cli_overrides(self):
        """CLI --self-review (True) should not be overwritten by config."""
        args = _make_unset_args(self_review=True)
        config = {"self_review": False}
        apply_config_to_args(args, config)
        assert args.self_review is True

    def test_config_self_review_inherited_in_reviewer_mode(self, tmp_path, monkeypatch):
        """Inner reviewer process with self_review = true in config should not fail.

        Simulates the config inheritance hazard: --reviewer-mode is on CLI,
        self_review = true comes from project config. The reviewer-mode branch
        should clear it silently rather than rejecting.
        """
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        # Write a project config with self_review = true
        toml_path = tmp_path / "swival.toml"
        toml_path.write_text("self_review = true\n")

        # The reviewer-mode branch should clear self_review and proceed to
        # run_as_reviewer — we mock that to avoid needing a real LLM.
        monkeypatch.setattr(sys, "argv", ["swival", "--reviewer-mode", str(tmp_path)])
        with patch("swival.reviewer.run_as_reviewer", return_value=0) as mock_rar:
            with pytest.raises(SystemExit) as exc_info:
                agent.main()
            assert exc_info.value.code == 0
            mock_rar.assert_called_once()
            # Verify self_review was cleared on the args passed to run_as_reviewer
            call_args = mock_rar.call_args[0][0]
            assert call_args.self_review is False

    def test_config_to_session_kwargs_drops_self_review(self):
        """self_review should be dropped by config_to_session_kwargs."""
        config = {"self_review": True, "model": "test"}
        kwargs = config_to_session_kwargs(config)
        assert "self_review" not in kwargs
        assert kwargs["model"] == "test"


# ===========================================================================
# Env-based API key
# ===========================================================================


class TestSelfReviewApiKeyEnv:
    @pytest.mark.parametrize(
        "provider,expected_env_var",
        [
            ("huggingface", "HF_TOKEN"),
            ("openrouter", "OPENROUTER_API_KEY"),
            ("generic", "OPENAI_API_KEY"),
            ("chatgpt", "CHATGPT_API_KEY"),
        ],
    )
    def test_api_key_passed_via_provider_env(self, provider, expected_env_var):
        """API key should be injected as provider-specific env var."""
        env_var = agent._PROVIDER_KEY_ENV.get(provider)
        assert env_var == expected_env_var

    def test_api_key_env_not_set_for_lmstudio(self):
        """lmstudio should not have a key env var mapping."""
        assert "lmstudio" not in agent._PROVIDER_KEY_ENV

    def test_chatgpt_resolve_provider_reads_env(self, monkeypatch):
        """resolve_provider() for chatgpt should read CHATGPT_API_KEY from env."""
        monkeypatch.setenv("CHATGPT_API_KEY", "test-chatgpt-key")
        # Mock litellm to avoid import errors
        mock_litellm = MagicMock()
        mock_litellm.get_model_info.return_value = {"max_input_tokens": 128000}
        monkeypatch.setitem(sys.modules, "litellm", mock_litellm)

        _model_id, _api_base, api_key, _ctx, _kwargs = agent.resolve_provider(
            provider="chatgpt",
            model="gpt-5.4",
            api_key=None,
            base_url=None,
            max_context_tokens=None,
            verbose=False,
        )
        assert api_key == "test-chatgpt-key"

    def test_chatgpt_cli_api_key_takes_precedence(self, monkeypatch):
        """Explicit api_key should override CHATGPT_API_KEY env var."""
        monkeypatch.setenv("CHATGPT_API_KEY", "env-key")
        mock_litellm = MagicMock()
        mock_litellm.get_model_info.return_value = {"max_input_tokens": 128000}
        monkeypatch.setitem(sys.modules, "litellm", mock_litellm)

        _model_id, _api_base, api_key, _ctx, _kwargs = agent.resolve_provider(
            provider="chatgpt",
            model="gpt-5.4",
            api_key="cli-key",
            base_url=None,
            max_context_tokens=None,
            verbose=False,
        )
        assert api_key == "cli-key"


# ===========================================================================
# Integration: command + env handoff
# ===========================================================================


class TestSelfReviewHandoff:
    def test_synthesized_cmd_plus_env_key(self):
        """Full handoff: command has no --api-key, env has provider-specific key."""
        args = _make_args(
            provider="openrouter",
            model="meta-llama/llama-4",
            api_key="sk-or-secret",
            yolo=True,
            self_review=True,
            question="do something",
        )

        # Build command — should not contain the key
        cmd = agent._build_self_review_cmd(args)
        assert "--api-key" not in cmd
        assert "sk-or-secret" not in cmd
        assert "--reviewer-mode" in cmd
        assert "--provider" in cmd

        # Simulate the env setup that main() does
        reviewer_env = {"SWIVAL_TASK": args.question}
        if args.self_review and args.api_key:
            env_var = agent._PROVIDER_KEY_ENV.get(args.provider)
            if env_var:
                reviewer_env[env_var] = args.api_key

        assert reviewer_env["OPENROUTER_API_KEY"] == "sk-or-secret"

        # Verify the command would be invoked with those env vars by run_reviewer
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=b"VERDICT: ACCEPT",
                stderr=b"",
            )
            agent.run_reviewer(
                cmd,
                "/tmp/base",
                "answer text",
                verbose=False,
                env_extra=reviewer_env,
            )
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args
            passed_env = call_kwargs.kwargs.get("env") or call_kwargs[1].get("env")
            assert passed_env["OPENROUTER_API_KEY"] == "sk-or-secret"
            passed_argv = (
                call_kwargs.args[0] if call_kwargs.args else call_kwargs[1]["args"]
            )
            argv_str = " ".join(passed_argv)
            assert "--api-key" not in argv_str
            assert "sk-or-secret" not in argv_str

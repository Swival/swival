"""Tests for the --self-review feature."""

import argparse
import json
import sys

import pytest
from unittest.mock import patch, MagicMock

from conftest import make_message
from swival import agent
from swival._msg import _msg_content, _msg_role
from swival.report import AgentError
from swival.config import (
    _UNSET,
    apply_config_to_args,
    config_to_session_kwargs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
        "commands": _UNSET,
        "yolo": _UNSET,
        "files": _UNSET,
        "_files_explicit": False,
        "_resolved_files_mode": "some",
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
# ChatGPT API key resolution
# ===========================================================================


class TestChatgptApiKey:
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


@pytest.fixture
def chat(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
    monkeypatch.setattr(agent, "_show_sandbox_review_hint", lambda args: None)
    monkeypatch.setattr(
        agent,
        "run_reviewer",
        lambda *a, **k: pytest.fail("self-review spawned a reviewer process"),
    )
    return _Chat(monkeypatch)


def _run_main(tmp_path, monkeypatch, *argv):
    monkeypatch.setattr(
        sys,
        "argv",
        ["swival", "--base-dir", str(tmp_path), "-q", "--no-history", *argv],
    )
    with pytest.raises(SystemExit) as exc:
        agent.main()
        raise SystemExit(0)
    return exc.value.code


class TestSelfReviewInProcess:
    """One-shot --self-review runs the review in-process on the agent's runtime."""

    def test_review_and_retry_print_only_the_final_answer(
        self, tmp_path, monkeypatch, capsys, chat
    ):
        chat.answers = ["v1", "v2"]
        chat.verdicts = ["Add a test.\nVERDICT: RETRY", "VERDICT: ACCEPT"]
        assert _run_main(tmp_path, monkeypatch, "--self-review", "fix add") == 0
        assert capsys.readouterr().out.strip() == "v2"
        first = chat.reviews[0]
        assert "<task>\nfix add\n</task>" in first["prompt"]
        assert "<answer>\nv1\n</answer>" in first["prompt"]
        assert first["model"] == chat.agent_calls[0]["model"]
        feedback = chat.agent_calls[1]["last_user"]
        assert feedback.startswith("[REVIEWER FEEDBACK, Round 1]\n")
        assert "Add a test." in feedback

    def test_cli_review_settings_reach_the_reviewer(self, tmp_path, monkeypatch, chat):
        (tmp_path / "criteria.md").write_text("Must keep the public API.")
        chat.answers = ["done"]
        chat.verdicts = ["VERDICT: ACCEPT"]
        code = _run_main(
            tmp_path,
            monkeypatch,
            "--self-review",
            "--review-prompt",
            "Focus on error paths.",
            "--verify",
            "criteria.md",
            "--reasoning-effort",
            "high",
            "fix add",
        )
        assert code == 0
        [review] = chat.reviews
        assert "Focus on error paths." in review["prompt"]
        assert "Must keep the public API." in review["prompt"]
        assert review["reasoning_effort"] == "high"

    def test_configured_objective_does_not_replace_the_task(
        self, tmp_path, monkeypatch, chat
    ):
        (tmp_path / "other.md").write_text("Something else entirely.")
        (tmp_path / "swival.toml").write_text('objective = "other.md"\n')
        chat.answers = ["done"]
        chat.verdicts = ["VERDICT: ACCEPT"]
        assert _run_main(tmp_path, monkeypatch, "--self-review", "fix add") == 0
        [review] = chat.reviews
        assert "<task>\nfix add\n</task>" in review["prompt"]
        assert "Something else entirely." not in review["prompt"]

    def test_report_counts_review_rounds(self, tmp_path, monkeypatch, chat):
        chat.answers = ["v1", "v2"]
        chat.verdicts = ["VERDICT: RETRY", "VERDICT: ACCEPT"]
        report_path = tmp_path / "report.json"
        code = _run_main(
            tmp_path,
            monkeypatch,
            "--self-review",
            "--report",
            str(report_path),
            "fix add",
        )
        assert code == 0
        report = json.loads(report_path.read_text())
        assert report["stats"]["review_rounds"] == 2
        events = [e for e in report["timeline"] if e["type"] == "review"]
        assert [e["exit_code"] for e in events] == [1, 0]


class TestSelfReviewInTheRepl:
    """--self-review and --reviewer work in the REPL, initial question included."""

    def _repl(self, tmp_path, monkeypatch, lines, *argv):
        session = MagicMock()
        session.prompt.side_effect = [*lines, "/exit"]
        monkeypatch.setattr("prompt_toolkit.PromptSession", lambda *a, **k: session)
        return _run_main(tmp_path, monkeypatch, "--repl", *argv), session

    def test_initial_question_and_follow_ups_are_reviewed(
        self, tmp_path, monkeypatch, chat
    ):
        chat.answers = ["Options: 1) retry 2) give up", "Gave up."]
        chat.verdicts = ["VERDICT: ACCEPT", "VERDICT: ACCEPT"]
        code, _ = self._repl(
            tmp_path, monkeypatch, ["use option 2"], "--self-review", "list options"
        )
        assert code == 0
        first, second = chat.reviews
        assert "<task>\nlist options\n</task>" in first["prompt"]
        assert "<conversation>" not in first["prompt"]
        assert "<task>\nuse option 2\n</task>" in second["prompt"]
        assert "Options: 1) retry 2) give up" in second["prompt"]

    def test_external_reviewer_gets_the_conversation(self, tmp_path, monkeypatch, chat):
        script = tmp_path / "review.sh"
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o755)
        envs = []

        def run_reviewer(command, base_dir, answer, verbose, env_extra=None):
            envs.append(env_extra)
            return 0, "", ""

        monkeypatch.setattr(agent, "run_reviewer", run_reviewer)
        chat.answers = ["Options: 1) retry 2) give up", "Gave up."]
        code, _ = self._repl(
            tmp_path,
            monkeypatch,
            ["use option 2"],
            "--reviewer",
            str(script),
            "list options",
        )
        assert code == 0
        assert [e["SWIVAL_TASK"] for e in envs] == ["list options", "use option 2"]
        assert "SWIVAL_CONVERSATION" not in envs[0]
        assert "Options: 1) retry 2) give up" in envs[1]["SWIVAL_CONVERSATION"]

    def test_agent_error_on_the_initial_question_keeps_the_repl_running(
        self, tmp_path, monkeypatch, chat
    ):
        chat.answers = [AgentError("provider down"), "Fixed it."]
        chat.verdicts = ["VERDICT: ACCEPT"]
        code, _ = self._repl(
            tmp_path, monkeypatch, ["try again"], "--self-review", "list options"
        )
        assert code == 0
        # The next line still ran and was reviewed after the error.
        assert [c["last_user"] for c in chat.agent_calls] == [
            "list options",
            "try again",
        ]
        [review] = chat.reviews
        assert "<task>\ntry again\n</task>" in review["prompt"]

    def test_sigterm_on_the_initial_question_reports_the_real_exit(
        self, tmp_path, monkeypatch, chat
    ):
        chat.answers = [SystemExit(143)]
        report_path = tmp_path / "report.json"
        code, _ = self._repl(
            tmp_path,
            monkeypatch,
            [],
            "--self-review",
            "--report",
            str(report_path),
            "list options",
        )
        assert code == 143
        result = json.loads(report_path.read_text())["result"]
        assert (result["outcome"], result["exit_code"]) == ("interrupted", 143)


class _Chat:
    """Stand-in for call_llm that answers agent and review requests apart."""

    def __init__(self, monkeypatch):
        self.answers = []
        self.verdicts = []
        self.agent_calls = []
        self.reviews = []
        monkeypatch.setattr(agent, "call_llm", self)

    def __call__(self, base_url, model_id, messages, *args, **kwargs):
        first = _msg_content(messages[0]) or ""
        if first.startswith("You are reviewing a coding agent's work."):
            self.reviews.append(
                {
                    "model": model_id,
                    "prompt": first,
                    "reasoning_effort": kwargs.get("reasoning_effort"),
                }
            )
            content = self.verdicts.pop(0)
        else:
            last_user = next(
                _msg_content(m) for m in reversed(messages) if _msg_role(m) == "user"
            )
            self.agent_calls.append({"model": model_id, "last_user": last_user})
            content = self.answers.pop(0)
            if isinstance(content, BaseException):
                raise content
        return make_message(content=content), "stop"

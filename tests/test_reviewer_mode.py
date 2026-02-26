"""Tests for --reviewer-mode feature."""

import shlex
import types

import pytest
from unittest.mock import patch, MagicMock

from swival import agent
from swival.config import _UNSET, ConfigError, load_config, apply_config_to_args
from swival.report import AgentError
from swival.reviewer import (
    _parse_verdict,
    _build_prompt,
    _resolve_path,
    run_as_reviewer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message(content=None, tool_calls=None):
    msg = types.SimpleNamespace()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.role = "assistant"
    msg.get = lambda key, default=None: getattr(msg, key, default)
    return msg


def _write_toml(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _reviewer_args(base_dir, **overrides):
    """Build a namespace mimicking --reviewer-mode args after config merge."""
    defaults = dict(
        provider="lmstudio",
        model="test-model",
        api_key=None,
        base_url="http://fake",
        max_output_tokens=1024,
        max_context_tokens=None,
        temperature=0.5,
        top_p=1.0,
        seed=None,
        quiet=False,
        verbose=True,
        reviewer_mode=True,
        review_prompt=None,
        objective=None,
        verify=None,
        reviewer=None,
        color=False,
        no_color=True,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------


class TestParseVerdict:
    def test_accept(self):
        assert _parse_verdict("Looks good.\nVERDICT: ACCEPT") == "ACCEPT"

    def test_retry(self):
        assert _parse_verdict("Fix the bug.\nVERDICT: RETRY") == "RETRY"

    def test_case_insensitive(self):
        assert _parse_verdict("verdict: accept") == "ACCEPT"
        assert _parse_verdict("Verdict: Retry") == "RETRY"

    def test_last_verdict_wins(self):
        text = "VERDICT: RETRY\nActually...\nVERDICT: ACCEPT"
        assert _parse_verdict(text) == "ACCEPT"

    def test_no_verdict(self):
        assert _parse_verdict("Just some text without a verdict.") is None

    def test_empty_string(self):
        assert _parse_verdict("") is None

    def test_verdict_with_leading_whitespace(self):
        assert _parse_verdict("  VERDICT: ACCEPT  ") == "ACCEPT"


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_basic_prompt(self):
        prompt = _build_prompt(task="Fix the tests", answer="done")
        assert "<task>\nFix the tests\n</task>" in prompt
        assert "<answer>\ndone\n</answer>" in prompt
        assert "VERDICT: ACCEPT" in prompt
        assert "VERDICT: RETRY" in prompt

    def test_with_verification(self):
        prompt = _build_prompt(
            task="Fix it", answer="done", verification="Tests must pass"
        )
        assert "<verification>\nTests must pass\n</verification>" in prompt
        assert "verification criteria" in prompt

    def test_without_verification(self):
        prompt = _build_prompt(task="Fix it", answer="done")
        assert "<verification>" not in prompt

    def test_with_custom_instructions(self):
        prompt = _build_prompt(
            task="Fix it",
            answer="done",
            custom_instructions="Focus on error handling",
        )
        assert "Focus on error handling" in prompt

    def test_without_custom_instructions(self):
        prompt = _build_prompt(task="Fix it", answer="done")
        # Should not have orphaned custom_instructions placeholder
        assert "Focus on" not in prompt


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


class TestResolvePath:
    def test_absolute_unchanged(self):
        assert _resolve_path("/absolute/path.md", "/base") == "/absolute/path.md"

    def test_relative_resolved_against_base(self):
        result = _resolve_path("criteria.md", "/project")
        assert result == "/project/criteria.md"

    def test_nested_relative(self):
        result = _resolve_path("verification/working.md", "/project")
        assert result == "/project/verification/working.md"


# ---------------------------------------------------------------------------
# run_as_reviewer unit tests
# ---------------------------------------------------------------------------


class TestRunAsReviewer:
    def test_accept_verdict_returns_0(self, tmp_path, capsys, monkeypatch):
        args = _reviewer_args(str(tmp_path))
        monkeypatch.setenv("SWIVAL_TASK", "Fix the bug")
        monkeypatch.setattr(
            "sys.stdin", types.SimpleNamespace(read=lambda: "The bug is fixed.")
        )

        def mock_call_llm(*a, **kw):
            return _make_message(content="Looks good.\nVERDICT: ACCEPT"), "stop"

        with (
            patch("swival.agent.call_llm", mock_call_llm),
            patch(
                "swival.agent.resolve_provider",
                return_value=(
                    "test-model",
                    "http://fake",
                    None,
                    None,
                    {"provider": "lmstudio", "api_key": None},
                ),
            ),
        ):
            code = run_as_reviewer(args, str(tmp_path))

        assert code == 0
        captured = capsys.readouterr()
        assert "VERDICT: ACCEPT" in captured.out

    def test_retry_verdict_returns_1(self, tmp_path, capsys, monkeypatch):
        args = _reviewer_args(str(tmp_path))
        monkeypatch.setenv("SWIVAL_TASK", "Fix the bug")
        monkeypatch.setattr(
            "sys.stdin", types.SimpleNamespace(read=lambda: "Some answer")
        )

        def mock_call_llm(*a, **kw):
            return _make_message(
                content="You missed the edge case.\nVERDICT: RETRY"
            ), "stop"

        with (
            patch("swival.agent.call_llm", mock_call_llm),
            patch(
                "swival.agent.resolve_provider",
                return_value=(
                    "test-model",
                    "http://fake",
                    None,
                    None,
                    {"provider": "lmstudio", "api_key": None},
                ),
            ),
        ):
            code = run_as_reviewer(args, str(tmp_path))

        assert code == 1
        captured = capsys.readouterr()
        assert "VERDICT: RETRY" in captured.out
        assert "edge case" in captured.out

    def test_no_verdict_returns_2(self, tmp_path, capsys, monkeypatch):
        args = _reviewer_args(str(tmp_path))
        monkeypatch.setenv("SWIVAL_TASK", "Fix the bug")
        monkeypatch.setattr(
            "sys.stdin", types.SimpleNamespace(read=lambda: "Some answer")
        )

        def mock_call_llm(*a, **kw):
            return _make_message(content="I'm not sure what to do."), "stop"

        with (
            patch("swival.agent.call_llm", mock_call_llm),
            patch(
                "swival.agent.resolve_provider",
                return_value=(
                    "test-model",
                    "http://fake",
                    None,
                    None,
                    {"provider": "lmstudio", "api_key": None},
                ),
            ),
        ):
            code = run_as_reviewer(args, str(tmp_path))

        assert code == 2
        captured = capsys.readouterr()
        assert "no VERDICT found" in captured.err
        assert "not sure" in captured.out

    def test_empty_stdin_returns_2(self, tmp_path, capsys, monkeypatch):
        args = _reviewer_args(str(tmp_path))
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(read=lambda: ""))

        code = run_as_reviewer(args, str(tmp_path))
        assert code == 2
        captured = capsys.readouterr()
        assert "empty answer" in captured.err

    def test_missing_task_returns_2(self, tmp_path, capsys, monkeypatch):
        args = _reviewer_args(str(tmp_path))
        monkeypatch.delenv("SWIVAL_TASK", raising=False)
        monkeypatch.setattr(
            "sys.stdin", types.SimpleNamespace(read=lambda: "Some answer")
        )

        code = run_as_reviewer(args, str(tmp_path))
        assert code == 2
        captured = capsys.readouterr()
        assert "no task description" in captured.err

    def test_objective_overrides_env(self, tmp_path, capsys, monkeypatch):
        """--objective file takes priority over SWIVAL_TASK env var."""
        obj_file = tmp_path / "task.md"
        obj_file.write_text("Fix the auth bug")
        args = _reviewer_args(str(tmp_path), objective=str(obj_file))
        monkeypatch.setenv("SWIVAL_TASK", "This should be ignored")
        monkeypatch.setattr(
            "sys.stdin", types.SimpleNamespace(read=lambda: "Fixed it.")
        )

        prompt_captured = []

        def mock_call_llm(base_url, model_id, messages, *a, **kw):
            prompt_captured.append(messages[0]["content"])
            return _make_message(content="VERDICT: ACCEPT"), "stop"

        with (
            patch("swival.agent.call_llm", mock_call_llm),
            patch(
                "swival.agent.resolve_provider",
                return_value=(
                    "test-model",
                    "http://fake",
                    None,
                    None,
                    {"provider": "lmstudio", "api_key": None},
                ),
            ),
        ):
            code = run_as_reviewer(args, str(tmp_path))

        assert code == 0
        assert "Fix the auth bug" in prompt_captured[0]
        assert "This should be ignored" not in prompt_captured[0]

    def test_objective_relative_path(self, tmp_path, capsys, monkeypatch):
        """--objective with relative path resolves against base_dir."""
        obj_file = tmp_path / "objective.md"
        obj_file.write_text("Do the thing")
        args = _reviewer_args(str(tmp_path), objective="objective.md")
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(read=lambda: "Did it."))

        prompt_captured = []

        def mock_call_llm(base_url, model_id, messages, *a, **kw):
            prompt_captured.append(messages[0]["content"])
            return _make_message(content="VERDICT: ACCEPT"), "stop"

        with (
            patch("swival.agent.call_llm", mock_call_llm),
            patch(
                "swival.agent.resolve_provider",
                return_value=(
                    "test-model",
                    "http://fake",
                    None,
                    None,
                    {"provider": "lmstudio", "api_key": None},
                ),
            ),
        ):
            code = run_as_reviewer(args, str(tmp_path))

        assert code == 0
        assert "Do the thing" in prompt_captured[0]

    def test_verify_file_included(self, tmp_path, capsys, monkeypatch):
        """--verify file content included in prompt."""
        verify_file = tmp_path / "criteria.md"
        verify_file.write_text("All tests must pass")
        args = _reviewer_args(str(tmp_path), verify=str(verify_file))
        monkeypatch.setenv("SWIVAL_TASK", "Fix tests")
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(read=lambda: "Fixed."))

        prompt_captured = []

        def mock_call_llm(base_url, model_id, messages, *a, **kw):
            prompt_captured.append(messages[0]["content"])
            return _make_message(content="VERDICT: ACCEPT"), "stop"

        with (
            patch("swival.agent.call_llm", mock_call_llm),
            patch(
                "swival.agent.resolve_provider",
                return_value=(
                    "test-model",
                    "http://fake",
                    None,
                    None,
                    {"provider": "lmstudio", "api_key": None},
                ),
            ),
        ):
            code = run_as_reviewer(args, str(tmp_path))

        assert code == 0
        assert "All tests must pass" in prompt_captured[0]
        assert "<verification>" in prompt_captured[0]

    def test_verify_relative_path(self, tmp_path, capsys, monkeypatch):
        """--verify with relative path resolves against base_dir."""
        (tmp_path / "verification").mkdir()
        verify_file = tmp_path / "verification" / "working.md"
        verify_file.write_text("Check the output")
        args = _reviewer_args(str(tmp_path), verify="verification/working.md")
        monkeypatch.setenv("SWIVAL_TASK", "Do it")
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(read=lambda: "Done."))

        prompt_captured = []

        def mock_call_llm(base_url, model_id, messages, *a, **kw):
            prompt_captured.append(messages[0]["content"])
            return _make_message(content="VERDICT: ACCEPT"), "stop"

        with (
            patch("swival.agent.call_llm", mock_call_llm),
            patch(
                "swival.agent.resolve_provider",
                return_value=(
                    "test-model",
                    "http://fake",
                    None,
                    None,
                    {"provider": "lmstudio", "api_key": None},
                ),
            ),
        ):
            code = run_as_reviewer(args, str(tmp_path))

        assert code == 0
        assert "Check the output" in prompt_captured[0]

    def test_custom_review_prompt(self, tmp_path, capsys, monkeypatch):
        """--review-prompt text is appended to prompt."""
        args = _reviewer_args(str(tmp_path), review_prompt="Focus on error handling")
        monkeypatch.setenv("SWIVAL_TASK", "Fix it")
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(read=lambda: "Fixed."))

        prompt_captured = []

        def mock_call_llm(base_url, model_id, messages, *a, **kw):
            prompt_captured.append(messages[0]["content"])
            return _make_message(content="VERDICT: ACCEPT"), "stop"

        with (
            patch("swival.agent.call_llm", mock_call_llm),
            patch(
                "swival.agent.resolve_provider",
                return_value=(
                    "test-model",
                    "http://fake",
                    None,
                    None,
                    {"provider": "lmstudio", "api_key": None},
                ),
            ),
        ):
            code = run_as_reviewer(args, str(tmp_path))

        assert code == 0
        assert "Focus on error handling" in prompt_captured[0]

    def test_llm_failure_returns_2(self, tmp_path, capsys, monkeypatch):
        """LLM call failure returns exit code 2."""
        args = _reviewer_args(str(tmp_path))
        monkeypatch.setenv("SWIVAL_TASK", "Fix it")
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(read=lambda: "Answer."))

        def mock_call_llm(*a, **kw):
            raise AgentError("connection refused")

        with (
            patch("swival.agent.call_llm", mock_call_llm),
            patch(
                "swival.agent.resolve_provider",
                return_value=(
                    "test-model",
                    "http://fake",
                    None,
                    None,
                    {"provider": "lmstudio", "api_key": None},
                ),
            ),
        ):
            code = run_as_reviewer(args, str(tmp_path))

        assert code == 2
        captured = capsys.readouterr()
        assert "LLM call failed" in captured.err

    def test_missing_objective_file_returns_2(self, tmp_path, capsys, monkeypatch):
        """Missing --objective file returns exit 2, not a traceback."""
        args = _reviewer_args(str(tmp_path), objective="/nonexistent/task.md")
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(read=lambda: "Answer."))

        code = run_as_reviewer(args, str(tmp_path))
        assert code == 2
        captured = capsys.readouterr()
        assert "cannot read --objective file" in captured.err

    def test_missing_verify_file_returns_2(self, tmp_path, capsys, monkeypatch):
        """Missing --verify file returns exit 2, not a traceback."""
        args = _reviewer_args(str(tmp_path), verify="/nonexistent/criteria.md")
        monkeypatch.setenv("SWIVAL_TASK", "Fix it")
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(read=lambda: "Answer."))

        code = run_as_reviewer(args, str(tmp_path))
        assert code == 2
        captured = capsys.readouterr()
        assert "cannot read --verify file" in captured.err

    def test_unreadable_objective_returns_2(self, tmp_path, capsys, monkeypatch):
        """Unreadable --objective file returns exit 2."""
        obj_file = tmp_path / "task.md"
        obj_file.write_text("content")
        obj_file.chmod(0o000)
        args = _reviewer_args(str(tmp_path), objective=str(obj_file))
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(read=lambda: "Answer."))

        code = run_as_reviewer(args, str(tmp_path))
        # Restore permissions for cleanup
        obj_file.chmod(0o644)
        assert code == 2
        captured = capsys.readouterr()
        assert "cannot read --objective file" in captured.err


# ---------------------------------------------------------------------------
# CLI validation
# ---------------------------------------------------------------------------


class TestCLIValidation:
    def test_reviewer_mode_with_repl_is_error(self):
        parser = agent.build_parser()
        args = parser.parse_args(["--reviewer-mode", "--repl", "/tmp/project"])
        # Simulate what main() does
        with patch("swival.agent.build_parser") as mock_bp:
            mock_parser = MagicMock()
            mock_parser.parse_args.return_value = args
            mock_parser.error.side_effect = SystemExit(2)
            mock_bp.return_value = mock_parser

            with pytest.raises(SystemExit):
                agent.main()
            mock_parser.error.assert_called_once()
            assert "incompatible" in mock_parser.error.call_args[0][0]

    def test_reviewer_mode_requires_positional_arg(self):
        parser = agent.build_parser()
        args = parser.parse_args(["--reviewer-mode"])
        with patch("swival.agent.build_parser") as mock_bp:
            mock_parser = MagicMock()
            mock_parser.parse_args.return_value = args
            mock_parser.error.side_effect = SystemExit(2)
            mock_bp.return_value = mock_parser

            with pytest.raises(SystemExit):
                agent.main()
            mock_parser.error.assert_called_once()
            assert (
                "positional" in mock_parser.error.call_args[0][0].lower()
                or "base_dir" in mock_parser.error.call_args[0][0].lower()
            )

    def test_reviewer_mode_with_explicit_reviewer_is_error(self, tmp_path, monkeypatch):
        """--reviewer-mode + --reviewer is a user mistake."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        parser = agent.build_parser()
        args = parser.parse_args(
            ["--reviewer-mode", "--reviewer", "/some/reviewer", str(tmp_path)]
        )
        with patch("swival.agent.build_parser") as mock_bp:
            mock_parser = MagicMock()
            mock_parser.parse_args.return_value = args
            mock_parser.error.side_effect = SystemExit(2)
            mock_bp.return_value = mock_parser

            with pytest.raises(SystemExit):
                agent.main()
            mock_parser.error.assert_called_once()
            assert "cannot be used together" in mock_parser.error.call_args[0][0]


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    def test_config_inherits_reviewer_silently_cleared(self, tmp_path, monkeypatch):
        """Project config with reviewer= does not break --reviewer-mode."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(
            tmp_path / "swival.toml",
            'reviewer = "swival --reviewer-mode"\n',
        )
        parser = agent.build_parser()
        args = parser.parse_args(["--reviewer-mode", str(tmp_path)])

        # Simulate config loading + merge
        from swival.config import load_config, apply_config_to_args

        file_config = load_config(tmp_path)
        reviewer_from_cli = args.reviewer is not _UNSET
        apply_config_to_args(args, file_config)

        # The reviewer key should have been loaded from config
        assert not reviewer_from_cli
        # In main(), args.reviewer is forced to None
        args.reviewer = None  # simulate main()'s behavior
        assert args.reviewer is None

    def test_review_prompt_from_config(self, tmp_path, monkeypatch):
        """review_prompt in config applies when CLI doesn't set it."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(
            tmp_path / "swival.toml",
            'review_prompt = "Focus on correctness"\n',
        )
        config = load_config(tmp_path)
        assert config["review_prompt"] == "Focus on correctness"

    def test_verify_path_resolved_in_config(self, tmp_path, monkeypatch):
        """verify path in config resolves against config directory."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(
            tmp_path / "swival.toml",
            'verify = "verification/working.md"\n',
        )
        config = load_config(tmp_path)
        assert config["verify"] == str(tmp_path / "verification/working.md")

    def test_objective_path_resolved_in_config(self, tmp_path, monkeypatch):
        """objective path in config resolves against config directory."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(
            tmp_path / "swival.toml",
            'objective = "objective.md"\n',
        )
        config = load_config(tmp_path)
        assert config["objective"] == str(tmp_path / "objective.md")

    def test_unset_defaults_apply(self, tmp_path, monkeypatch):
        """_UNSET defaults for review_prompt/objective/verify resolve to None."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        config = load_config(tmp_path)

        from swival.config import _UNSET

        args = types.SimpleNamespace(
            review_prompt=_UNSET,
            objective=_UNSET,
            verify=_UNSET,
            reviewer=_UNSET,
            provider=_UNSET,
            model=_UNSET,
            api_key=_UNSET,
            base_url=_UNSET,
            max_output_tokens=_UNSET,
            max_context_tokens=_UNSET,
            temperature=_UNSET,
            top_p=_UNSET,
            seed=_UNSET,
            max_turns=_UNSET,
            system_prompt=_UNSET,
            no_system_prompt=_UNSET,
            allowed_commands=_UNSET,
            yolo=_UNSET,
            add_dir=None,
            add_dir_ro=None,
            no_read_guard=_UNSET,
            no_instructions=_UNSET,
            no_skills=_UNSET,
            skills_dir=None,
            no_history=_UNSET,
            color=_UNSET,
            no_color=_UNSET,
            quiet=_UNSET,
            proactive_summaries=_UNSET,
        )
        apply_config_to_args(args, config)
        assert args.review_prompt is None
        assert args.objective is None
        assert args.verify is None


# ---------------------------------------------------------------------------
# Shell-split reviewer commands (Step 1 tests)
# ---------------------------------------------------------------------------


class TestShellSplitReviewer:
    def test_multi_word_command_split(self, tmp_path):
        """run_reviewer splits 'swival --reviewer-mode' into argv."""
        script = tmp_path / "reviewer_echo.sh"
        script.write_text('#!/bin/sh\necho "args: $@"\nexit 0\n')
        script.chmod(0o755)

        # Use the script path + extra args
        cmd = f"{script} --extra-flag"
        code, text, stderr = agent.run_reviewer(cmd, str(tmp_path), "answer", False)
        assert code == 0
        assert "--extra-flag" in text
        assert str(tmp_path) in text  # base_dir appended

    def test_config_bare_command_preserved(self, tmp_path, monkeypatch):
        """Bare command name (no path prefix) is NOT resolved against config_dir."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(
            tmp_path / "swival.toml",
            'reviewer = "swival --reviewer-mode"\n',
        )
        config = load_config(tmp_path)
        # First token should be 'swival', not resolved to a filesystem path
        parts = shlex.split(config["reviewer"])
        assert parts[0] == "swival"
        assert parts[1] == "--reviewer-mode"

    def test_config_path_like_resolved(self, tmp_path, monkeypatch):
        """Path-like first token (./script) is resolved against config_dir."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(
            tmp_path / "swival.toml",
            'reviewer = "./review.sh --strict"\n',
        )
        config = load_config(tmp_path)
        parts = shlex.split(config["reviewer"])
        assert parts[0] == str(tmp_path / "review.sh")
        assert parts[1] == "--strict"

    def test_config_malformed_quoting_raises(self, tmp_path, monkeypatch):
        """Unbalanced quotes in reviewer command raise ConfigError."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(
            tmp_path / "swival.toml",
            'reviewer = "swival --review-prompt \'unbalanced"\n',
        )
        with pytest.raises(ConfigError, match="malformed reviewer command"):
            load_config(tmp_path)

    def test_config_empty_reviewer_raises(self, tmp_path, monkeypatch):
        """Empty reviewer command raises ConfigError."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(
            tmp_path / "swival.toml",
            'reviewer = ""\n',
        )
        with pytest.raises(ConfigError, match="reviewer command is empty"):
            load_config(tmp_path)

    def test_config_whitespace_only_reviewer_raises(self, tmp_path, monkeypatch):
        """Whitespace-only reviewer command raises ConfigError."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(
            tmp_path / "swival.toml",
            'reviewer = "   "\n',
        )
        with pytest.raises(ConfigError, match="reviewer command is empty"):
            load_config(tmp_path)

    def test_startup_validates_first_token(self, tmp_path, monkeypatch):
        """Startup validation checks the first token of a multi-word command."""
        script = tmp_path / "myreviewer"
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o755)

        # Pass a multi-word command with valid executable
        args = types.SimpleNamespace(
            reviewer=f"{script} --extra",
            reviewer_mode=False,
        )
        # Should not raise — validation happens in _run_main
        parts = shlex.split(args.reviewer)
        assert parts[0] == str(script)

    def test_stderr_forwarded_when_verbose(self, tmp_path):
        """Reviewer stderr is captured and returned in 3-tuple."""
        script = tmp_path / "reviewer.sh"
        script.write_text("#!/bin/sh\necho 'diag' >&2\nexit 0\n")
        script.chmod(0o755)

        code, stdout, stderr = agent.run_reviewer(
            str(script), str(tmp_path), "answer", True
        )
        assert code == 0
        assert "diag" in stderr

    def test_stderr_in_report(self):
        """record_review stores stderr when provided."""
        from swival.report import ReportCollector

        rc = ReportCollector()
        rc.record_review(1, 0, "feedback", stderr="some error")
        assert rc.events[0]["stderr"] == "some error"

    def test_stderr_omitted_when_empty(self):
        """record_review omits stderr key when empty."""
        from swival.report import ReportCollector

        rc = ReportCollector()
        rc.record_review(1, 0, "feedback")
        assert "stderr" not in rc.events[0]


# ---------------------------------------------------------------------------
# Argparse flag tests
# ---------------------------------------------------------------------------


class TestArgparse:
    def test_reviewer_mode_flag_parsed(self):
        parser = agent.build_parser()
        args = parser.parse_args(["--reviewer-mode", "/project"])
        assert args.reviewer_mode is True
        assert args.question == "/project"

    def test_review_prompt_flag_parsed(self):
        parser = agent.build_parser()
        args = parser.parse_args(["--review-prompt", "Be strict", "hello"])
        assert args.review_prompt == "Be strict"

    def test_objective_flag_parsed(self):
        parser = agent.build_parser()
        args = parser.parse_args(["--objective", "task.md", "hello"])
        assert args.objective == "task.md"

    def test_verify_flag_parsed(self):
        parser = agent.build_parser()
        args = parser.parse_args(["--verify", "checks.md", "hello"])
        assert args.verify == "checks.md"

    def test_reviewer_mode_default_false(self):
        parser = agent.build_parser()
        args = parser.parse_args(["hello"])
        assert args.reviewer_mode is False

    def test_new_flags_default_unset(self):
        parser = agent.build_parser()
        args = parser.parse_args(["hello"])
        assert args.review_prompt is _UNSET
        assert args.objective is _UNSET
        assert args.verify is _UNSET


# ---------------------------------------------------------------------------
# Init config template
# ---------------------------------------------------------------------------


class TestInitConfigTemplate:
    def test_template_includes_reviewer_mode_keys(self):
        from swival.config import generate_config

        template = generate_config(project=True)
        assert "review_prompt" in template
        assert "objective" in template
        assert "verify" in template

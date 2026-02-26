"""Tests for the --reviewer feature."""

import subprocess
import types

import pytest
from unittest.mock import patch, MagicMock

from swival import agent, fmt
from swival.report import AgentError


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


def _base_args(tmp_path, **overrides):
    defaults = dict(
        base_url="http://fake",
        model="test-model",
        max_output_tokens=1024,
        temperature=0.55,
        top_p=1.0,
        seed=None,
        quiet=False,
        max_turns=10,
        base_dir=str(tmp_path),
        no_system_prompt=True,
        no_instructions=True,
        no_skills=True,
        skills_dir=[],
        system_prompt=None,
        question="test task",
        repl=False,
        max_context_tokens=None,
        allowed_commands=None,
        add_dir=[],
        add_dir_ro=[],
        provider="lmstudio",
        api_key=None,
        color=False,
        no_color=False,
        yolo=False,
        report=None,
        reviewer=None,
        version=False,
        no_read_guard=False,
        no_history=True,
        init_config=False,
        project=False,
        reviewer_mode=False,
        review_prompt=None,
        objective=None,
        verify=None,
        max_review_rounds=5,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _simple_llm(*args, **kwargs):
    """LLM that returns a final text answer immediately."""
    return _make_message(content="the answer"), "stop"


# ---------------------------------------------------------------------------
# run_reviewer() unit tests
# ---------------------------------------------------------------------------


class TestRunReviewer:
    def test_returns_exit_code_and_stdout(self, tmp_path):
        script = tmp_path / "reviewer.sh"
        script.write_text("#!/bin/sh\necho 'looks good'\nexit 0\n")
        script.chmod(0o755)

        code, text, _stderr = agent.run_reviewer(
            str(script), str(tmp_path), "answer", False
        )
        assert code == 0
        assert "looks good" in text

    def test_exit_code_1_with_feedback(self, tmp_path):
        script = tmp_path / "reviewer.sh"
        script.write_text("#!/bin/sh\necho 'fix the typo'\nexit 1\n")
        script.chmod(0o755)

        code, text, _stderr = agent.run_reviewer(
            str(script), str(tmp_path), "answer", False
        )
        assert code == 1
        assert "fix the typo" in text

    def test_exit_code_2(self, tmp_path):
        script = tmp_path / "reviewer.sh"
        script.write_text("#!/bin/sh\nexit 2\n")
        script.chmod(0o755)

        code, text, _stderr = agent.run_reviewer(
            str(script), str(tmp_path), "answer", False
        )
        assert code == 2

    def test_receives_answer_on_stdin(self, tmp_path):
        script = tmp_path / "reviewer.sh"
        # Echo back whatever was received on stdin
        script.write_text("#!/bin/sh\ncat\nexit 0\n")
        script.chmod(0o755)

        code, text, _stderr = agent.run_reviewer(
            str(script), str(tmp_path), "my answer", False
        )
        assert code == 0
        assert "my answer" in text

    def test_receives_base_dir_as_arg(self, tmp_path):
        script = tmp_path / "reviewer.sh"
        script.write_text('#!/bin/sh\necho "$1"\nexit 0\n')
        script.chmod(0o755)

        code, text, _stderr = agent.run_reviewer(
            str(script), str(tmp_path), "answer", False
        )
        assert code == 0
        assert str(tmp_path) in text

    def test_timeout_returns_2(self, tmp_path):
        script = tmp_path / "reviewer.sh"
        script.write_text("#!/bin/sh\nsleep 999\n")
        script.chmod(0o755)

        code, text, _stderr = agent.run_reviewer(
            str(script), str(tmp_path), "answer", False, timeout=1
        )
        assert code == 2
        assert text == ""

    def test_file_not_found_returns_2(self):
        code, text, _stderr = agent.run_reviewer(
            "/nonexistent/reviewer", "/tmp", "answer", False
        )
        assert code == 2
        assert text == ""

    def test_permission_error_returns_2(self, tmp_path):
        script = tmp_path / "reviewer.sh"
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o644)  # not executable

        code, text, _stderr = agent.run_reviewer(
            str(script), str(tmp_path), "answer", False
        )
        assert code == 2
        assert text == ""

    def test_unknown_exit_code_passed_through(self, tmp_path):
        script = tmp_path / "reviewer.sh"
        script.write_text("#!/bin/sh\nexit 42\n")
        script.chmod(0o755)

        code, text, _stderr = agent.run_reviewer(
            str(script), str(tmp_path), "answer", False
        )
        assert code == 42

    def test_env_vars_passed_to_subprocess(self, tmp_path):
        """All SWIVAL_* env vars are visible to the reviewer."""
        script = tmp_path / "reviewer.sh"
        script.write_text(
            "#!/bin/sh\n"
            'echo "task=$SWIVAL_TASK"\n'
            'echo "round=$SWIVAL_REVIEW_ROUND"\n'
            'echo "model=$SWIVAL_MODEL"\n'
            "exit 0\n"
        )
        script.chmod(0o755)

        env = {
            "SWIVAL_TASK": "do the thing",
            "SWIVAL_REVIEW_ROUND": "2",
            "SWIVAL_MODEL": "test-model-7b",
        }
        code, text, _stderr = agent.run_reviewer(
            str(script), str(tmp_path), "answer", False, env_extra=env
        )
        assert code == 0
        assert "task=do the thing" in text
        assert "round=2" in text
        assert "model=test-model-7b" in text

    def test_env_vars_inherit_parent_env(self, tmp_path, monkeypatch):
        """Reviewer inherits the parent process environment."""
        monkeypatch.setenv("SWIVAL_TEST_SENTINEL", "hello")
        script = tmp_path / "reviewer.sh"
        script.write_text('#!/bin/sh\necho "$SWIVAL_TEST_SENTINEL"\nexit 0\n')
        script.chmod(0o755)

        code, text, _stderr = agent.run_reviewer(
            str(script),
            str(tmp_path),
            "answer",
            False,
            env_extra={"SWIVAL_TASK": "x"},
        )
        assert code == 0
        assert "hello" in text

    def test_env_extra_none(self, tmp_path):
        """env_extra=None (default) works without crashing."""
        script = tmp_path / "reviewer.sh"
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o755)

        code, text, _stderr = agent.run_reviewer(
            str(script), str(tmp_path), "answer", False
        )
        assert code == 0

    def test_env_vars_override_parent(self, tmp_path, monkeypatch):
        """Swival's env vars override any pre-existing parent values."""
        monkeypatch.setenv("SWIVAL_TASK", "old")
        script = tmp_path / "reviewer.sh"
        script.write_text('#!/bin/sh\necho "$SWIVAL_TASK"\nexit 0\n')
        script.chmod(0o755)

        code, text, _stderr = agent.run_reviewer(
            str(script),
            str(tmp_path),
            "answer",
            False,
            env_extra={"SWIVAL_TASK": "new"},
        )
        assert code == 0
        assert "new" in text
        assert "old" not in text


# ---------------------------------------------------------------------------
# CLI validation
# ---------------------------------------------------------------------------


class TestCLIValidation:
    def test_reviewer_incompatible_with_repl(self):
        """--reviewer + --repl should produce an argparse error."""
        with patch("swival.agent.build_parser") as mock_bp:
            mock_parser = MagicMock()
            mock_args = types.SimpleNamespace(
                question=None,
                repl=True,
                quiet=False,
                verbose=True,
                color=False,
                no_color=False,
                version=False,
                report=None,
                reviewer="/some/reviewer",
                base_dir=".",
                init_config=False,
                project=False,
                reviewer_mode=False,
                review_prompt=None,
                objective=None,
                verify=None,
            )
            mock_parser.parse_args.return_value = mock_args
            mock_parser.error.side_effect = SystemExit(2)
            mock_bp.return_value = mock_parser

            with pytest.raises(SystemExit):
                agent.main()
            mock_parser.error.assert_called_once()
            assert "incompatible" in mock_parser.error.call_args[0][0]

    def test_reviewer_not_found_at_startup(self, tmp_path, monkeypatch):
        """Invalid --reviewer path should raise AgentError."""
        args = _base_args(tmp_path, reviewer="/nonexistent/reviewer")

        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)

        fmt.init(color=False, no_color=True)
        with pytest.raises(SystemExit) as exc_info:
            agent.main()
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Integration: review loop in single-shot path
# ---------------------------------------------------------------------------


class TestReviewLoop:
    def test_accepted_on_first_try(self, tmp_path, capsys, monkeypatch):
        """Reviewer exits 0 → answer printed, loop called once."""
        script = tmp_path / "reviewer.sh"
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o755)

        args = _base_args(tmp_path, reviewer=str(script))
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)
        fmt.init(color=False, no_color=True)

        agent.main()
        captured = capsys.readouterr()
        assert captured.out.strip() == "the answer"

    def test_retry_then_accept(self, tmp_path, capsys, monkeypatch):
        """Reviewer exits 1 with feedback, then 0 on retry."""
        state_file = tmp_path / ".review_state"
        script = tmp_path / "reviewer.sh"
        script.write_text(
            f"#!/bin/sh\n"
            f'if [ ! -f "{state_file}" ]; then\n'
            f'  touch "{state_file}"\n'
            f'  echo "please fix the typo"\n'
            f"  exit 1\n"
            f"else\n"
            f"  exit 0\n"
            f"fi\n"
        )
        script.chmod(0o755)

        call_count = 0

        def counting_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _make_message(content=f"answer v{call_count}"), "stop"

        args = _base_args(tmp_path, reviewer=str(script))
        monkeypatch.setattr(agent, "call_llm", counting_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)
        fmt.init(color=False, no_color=True)

        agent.main()
        captured = capsys.readouterr()
        assert call_count == 2
        assert "answer v2" in captured.out

    def test_reviewer_error_accepts_answer(self, tmp_path, capsys, monkeypatch):
        """Reviewer exits 2 → answer printed as-is."""
        script = tmp_path / "reviewer.sh"
        script.write_text("#!/bin/sh\nexit 2\n")
        script.chmod(0o755)

        args = _base_args(tmp_path, reviewer=str(script))
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)
        fmt.init(color=False, no_color=True)

        agent.main()
        captured = capsys.readouterr()
        assert captured.out.strip() == "the answer"

    def test_unknown_exit_code_accepts_answer(self, tmp_path, capsys, monkeypatch):
        """Reviewer exits 42 → treated like exit 2."""
        script = tmp_path / "reviewer.sh"
        script.write_text("#!/bin/sh\nexit 42\n")
        script.chmod(0o755)

        args = _base_args(tmp_path, reviewer=str(script))
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)
        fmt.init(color=False, no_color=True)

        agent.main()
        captured = capsys.readouterr()
        assert captured.out.strip() == "the answer"

    def test_exhausted_skips_reviewer(self, tmp_path, capsys, monkeypatch):
        """When agent exhausts max_turns, reviewer is NOT called."""
        script = tmp_path / "reviewer.sh"
        # This should never run
        script.write_text("#!/bin/sh\necho 'SHOULD NOT SEE THIS' >&2\nexit 1\n")
        script.chmod(0o755)

        # LLM always returns tool calls, never a final answer
        tc = types.SimpleNamespace()
        tc.id = "tc_1"
        tc.function = types.SimpleNamespace(
            name="think", arguments='{"thought": "hmm"}'
        )

        def looping_llm(*args, **kwargs):
            return _make_message(content=None, tool_calls=[tc]), "tool_calls"

        args = _base_args(tmp_path, reviewer=str(script), max_turns=2)
        monkeypatch.setattr(agent, "call_llm", looping_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)
        fmt.init(color=False, no_color=True)

        with pytest.raises(SystemExit) as exc_info:
            agent.main()
        assert exc_info.value.code == 2

        captured = capsys.readouterr()
        assert "SHOULD NOT SEE THIS" not in captured.err

    def test_max_review_rounds(self, tmp_path, capsys, monkeypatch):
        """After max_review_rounds retries, answer is accepted."""
        script = tmp_path / "reviewer.sh"
        script.write_text("#!/bin/sh\necho 'try again'\nexit 1\n")
        script.chmod(0o755)

        call_count = 0

        def counting_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _make_message(content=f"answer v{call_count}"), "stop"

        args = _base_args(tmp_path, reviewer=str(script))
        monkeypatch.setattr(agent, "call_llm", counting_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)
        fmt.init(color=False, no_color=True)

        agent.main()
        captured = capsys.readouterr()
        # Flow: answer v1 → reviewer retry (round 1) → answer v2 → reviewer retry
        # (round 2) → ... → answer v5 → reviewer retry (round 5, hits cap) → break.
        # The cap check fires when review_round == args.max_review_rounds after the
        # reviewer returns exit 1, so the last agent loop call is for answer v5.
        assert call_count == 5
        assert f"answer v{call_count}" in captured.out

    def test_custom_max_review_rounds(self, tmp_path, capsys, monkeypatch):
        """Custom --max-review-rounds value is respected."""
        script = tmp_path / "reviewer.sh"
        script.write_text("#!/bin/sh\necho 'try again'\nexit 1\n")
        script.chmod(0o755)

        call_count = 0

        def counting_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _make_message(content=f"answer v{call_count}"), "stop"

        args = _base_args(tmp_path, reviewer=str(script), max_review_rounds=3)
        monkeypatch.setattr(agent, "call_llm", counting_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)
        fmt.init(color=False, no_color=True)

        agent.main()
        captured = capsys.readouterr()
        assert call_count == 3
        assert "answer v3" in captured.out

    def test_max_review_rounds_zero_disables_retries(
        self, tmp_path, capsys, monkeypatch
    ):
        """--max-review-rounds 0 accepts the first answer without retries."""
        script = tmp_path / "reviewer.sh"
        # Reviewer always wants to retry — but with 0 rounds, it should never run
        script.write_text("#!/bin/sh\necho 'try again'\nexit 1\n")
        script.chmod(0o755)

        call_count = 0

        def counting_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _make_message(content=f"answer v{call_count}"), "stop"

        args = _base_args(tmp_path, reviewer=str(script), max_review_rounds=0)
        monkeypatch.setattr(agent, "call_llm", counting_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)
        fmt.init(color=False, no_color=True)

        agent.main()
        captured = capsys.readouterr()
        # Agent runs once, reviewer runs once (round 1 >= 0 triggers cap), accepts
        assert call_count == 1
        assert "answer v1" in captured.out

    def test_context_preserved_across_rounds(self, tmp_path, monkeypatch):
        """Messages list accumulates across review rounds."""
        script = tmp_path / "reviewer.sh"
        # Always retry
        script.write_text("#!/bin/sh\necho 'do better'\nexit 1\n")
        script.chmod(0o755)

        messages_at_call = []

        def capturing_llm(*args, **kwargs):
            # args[2] is messages
            messages_at_call.append(len(args[2]))
            return _make_message(content="an answer"), "stop"

        args = _base_args(tmp_path, reviewer=str(script), max_turns=5)
        monkeypatch.setattr(agent, "call_llm", capturing_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)
        fmt.init(color=False, no_color=True)

        agent.main()

        # Each successive call should have more messages (assistant answer + review feedback)
        for i in range(1, len(messages_at_call)):
            assert messages_at_call[i] > messages_at_call[i - 1]

    def test_empty_stdout_on_exit_1(self, tmp_path, capsys, monkeypatch):
        """Reviewer returns exit 1 with empty stdout → empty string appended."""
        state_file = tmp_path / ".empty_state"
        script = tmp_path / "reviewer.sh"
        script.write_text(
            f"#!/bin/sh\n"
            f'if [ ! -f "{state_file}" ]; then\n'
            f'  touch "{state_file}"\n'
            f"  exit 1\n"
            f"else\n"
            f"  exit 0\n"
            f"fi\n"
        )
        script.chmod(0o755)

        call_count = 0

        def counting_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _make_message(content=f"answer v{call_count}"), "stop"

        args = _base_args(tmp_path, reviewer=str(script))
        monkeypatch.setattr(agent, "call_llm", counting_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)
        fmt.init(color=False, no_color=True)

        agent.main()
        assert call_count == 2

    def test_review_round_env_increments(self, tmp_path, monkeypatch):
        """SWIVAL_REVIEW_ROUND increments with each reviewer invocation."""
        log_file = tmp_path / "rounds.log"
        script = tmp_path / "reviewer.sh"
        # Append the round number to a log file, retry twice then accept.
        script.write_text(
            f"#!/bin/sh\n"
            f'echo "$SWIVAL_REVIEW_ROUND" >> "{log_file}"\n'
            f'if [ "$SWIVAL_REVIEW_ROUND" -lt 3 ]; then\n'
            f'  echo "again"\n'
            f"  exit 1\n"
            f"else\n"
            f"  exit 0\n"
            f"fi\n"
        )
        script.chmod(0o755)

        args = _base_args(tmp_path, reviewer=str(script))
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)
        fmt.init(color=False, no_color=True)

        agent.main()

        rounds = log_file.read_text().strip().splitlines()
        assert rounds == ["1", "2", "3"]

    def test_task_env_matches_question(self, tmp_path, monkeypatch):
        """SWIVAL_TASK contains the original question."""
        task_file = tmp_path / "task.txt"
        script = tmp_path / "reviewer.sh"
        script.write_text(f'#!/bin/sh\necho "$SWIVAL_TASK" > "{task_file}"\nexit 0\n')
        script.chmod(0o755)

        args = _base_args(tmp_path, reviewer=str(script), question="hello world")
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)
        fmt.init(color=False, no_color=True)

        agent.main()

        assert task_file.read_text().strip() == "hello world"

    def test_model_env_set(self, tmp_path, monkeypatch):
        """SWIVAL_MODEL is set from args._resolved_model_id."""
        model_file = tmp_path / "model.txt"
        script = tmp_path / "reviewer.sh"
        script.write_text(f'#!/bin/sh\necho "$SWIVAL_MODEL" > "{model_file}"\nexit 0\n')
        script.chmod(0o755)

        args = _base_args(tmp_path, reviewer=str(script), model="my-llm-7b")
        # _run_main() sets _resolved_model_id from args.model for lmstudio
        # provider when a model is specified, so we match args.model here.
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("my-llm-7b", None))
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)
        fmt.init(color=False, no_color=True)

        agent.main()

        assert model_file.read_text().strip() == "my-llm-7b"


# ---------------------------------------------------------------------------
# Failure modes (mocked subprocess)
# ---------------------------------------------------------------------------


class TestReviewerFailures:
    @staticmethod
    def _dummy_reviewer(tmp_path):
        """Create a valid executable so startup validation passes."""
        script = tmp_path / "dummy_reviewer.sh"
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o755)
        return str(script)

    def test_timeout_at_runtime(self, tmp_path, capsys, monkeypatch):
        """Reviewer timeout at runtime → answer accepted, no crash."""
        args = _base_args(tmp_path, reviewer=self._dummy_reviewer(tmp_path))
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)
        fmt.init(color=False, no_color=True)

        def timeout_run(*a, **kw):
            raise subprocess.TimeoutExpired(cmd="reviewer", timeout=120)

        monkeypatch.setattr(subprocess, "run", timeout_run)
        agent.main()
        captured = capsys.readouterr()
        assert captured.out.strip() == "the answer"

    def test_spawn_failure_at_runtime(self, tmp_path, capsys, monkeypatch):
        """FileNotFoundError at runtime → answer accepted, not AgentError."""
        args = _base_args(tmp_path, reviewer=self._dummy_reviewer(tmp_path))
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)
        fmt.init(color=False, no_color=True)

        def fnf_run(*a, **kw):
            raise FileNotFoundError("No such file")

        monkeypatch.setattr(subprocess, "run", fnf_run)
        agent.main()
        captured = capsys.readouterr()
        assert captured.out.strip() == "the answer"

    def test_permission_error_at_runtime(self, tmp_path, capsys, monkeypatch):
        """PermissionError at runtime → answer accepted."""
        args = _base_args(tmp_path, reviewer=self._dummy_reviewer(tmp_path))
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)
        fmt.init(color=False, no_color=True)

        def perm_run(*a, **kw):
            raise PermissionError("Permission denied")

        monkeypatch.setattr(subprocess, "run", perm_run)
        agent.main()
        captured = capsys.readouterr()
        assert captured.out.strip() == "the answer"


# ---------------------------------------------------------------------------
# Interactions: quiet mode
# ---------------------------------------------------------------------------


class TestQuietMode:
    def test_quiet_suppresses_reviewer_diagnostics(self, tmp_path, capsys, monkeypatch):
        """--quiet + --reviewer → no stderr output during review."""
        script = tmp_path / "reviewer.sh"
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o755)

        args = _base_args(tmp_path, reviewer=str(script), quiet=True)
        args.verbose = False
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)
        fmt.init(color=False, no_color=True)

        agent.main()
        captured = capsys.readouterr()
        assert captured.out.strip() == "the answer"
        # No reviewer-related diagnostics
        assert "Review round" not in captured.err
        assert "Reviewer" not in captured.err


# ---------------------------------------------------------------------------
# Interactions: report mode
# ---------------------------------------------------------------------------


class TestReportMode:
    def test_report_includes_review_rounds(self, tmp_path, monkeypatch):
        """--report + --reviewer → report has review_rounds in stats."""
        import json

        script = tmp_path / "reviewer.sh"
        # Exit 1 once, then exit 0
        script.write_text(
            f"#!/bin/sh\n"
            f'if [ ! -f "{tmp_path}/.report_state" ]; then\n'
            f'  touch "{tmp_path}/.report_state"\n'
            f'  echo "fix it"\n'
            f"  exit 1\n"
            f"else\n"
            f"  exit 0\n"
            f"fi\n"
        )
        script.chmod(0o755)

        report_path = tmp_path / "report.json"
        args = _base_args(
            tmp_path,
            reviewer=str(script),
            report=str(report_path),
        )

        call_count = 0

        def counting_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _make_message(content=f"answer v{call_count}"), "stop"

        monkeypatch.setattr(agent, "call_llm", counting_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)
        fmt.init(color=False, no_color=True)

        agent.main()

        data = json.loads(report_path.read_text())
        # Reviewer was invoked twice: once returned 1 (retry), once returned 0 (accept)
        assert data["stats"]["review_rounds"] == 2
        assert data["result"]["outcome"] == "success"
        # Timeline should contain review events
        review_events = [e for e in data["timeline"] if e["type"] == "review"]
        assert len(review_events) == 2
        assert review_events[0]["round"] == 1
        assert review_events[0]["exit_code"] == 1
        assert review_events[0]["feedback"] == "fix it\n"
        assert review_events[1]["round"] == 2
        assert review_events[1]["exit_code"] == 0

    def test_report_review_rounds_zero_without_reviewer(self, tmp_path, monkeypatch):
        """--report without --reviewer → review_rounds is 0."""
        import json

        report_path = tmp_path / "report.json"
        args = _base_args(tmp_path, report=str(report_path))

        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)
        fmt.init(color=False, no_color=True)

        agent.main()

        data = json.loads(report_path.read_text())
        assert data["stats"]["review_rounds"] == 0

    def test_report_review_rounds_on_error(self, tmp_path, monkeypatch):
        """AgentError after reviewer retries → review_rounds reflects actual count."""
        import json

        state_file = tmp_path / ".error_state"
        script = tmp_path / "reviewer.sh"
        script.write_text(
            f'#!/bin/sh\ntouch "{state_file}"\necho "try harder"\nexit 1\n'
        )
        script.chmod(0o755)

        report_path = tmp_path / "report.json"
        args = _base_args(
            tmp_path,
            reviewer=str(script),
            report=str(report_path),
        )

        call_count = 0

        def error_on_second(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_message(content="first answer"), "stop"
            raise AgentError("LLM exploded")

        monkeypatch.setattr(agent, "call_llm", error_on_second)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)
        fmt.init(color=False, no_color=True)

        with pytest.raises(SystemExit) as exc_info:
            agent.main()
        assert exc_info.value.code == 1

        data = json.loads(report_path.read_text())
        assert data["stats"]["review_rounds"] == 1
        assert data["result"]["outcome"] == "error"

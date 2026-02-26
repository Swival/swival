"""Tests for append-only response history (.swival/HISTORY.md)."""

import re
import types
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from swival import agent, fmt
from swival.agent import (
    MAX_HISTORY_SIZE,
    _safe_history_path,
    append_history,
)


@pytest.fixture(autouse=True)
def _init_fmt():
    fmt.init(color=False, no_color=False)


# ---------------------------------------------------------------------------
# Core behavior
# ---------------------------------------------------------------------------


class TestCoreBehavior:
    def test_append_creates_file(self, tmp_path):
        append_history(str(tmp_path), "What is 2+2?", "4")
        history = tmp_path / ".swival" / "HISTORY.md"
        assert history.exists()
        content = history.read_text()
        assert "What is 2+2?" in content
        assert "4" in content

    def test_append_multiple_entries(self, tmp_path):
        base = str(tmp_path)
        append_history(base, "Q1", "A1")
        append_history(base, "Q2", "A2")
        append_history(base, "Q3", "A3")
        content = (tmp_path / ".swival" / "HISTORY.md").read_text()
        assert content.count("---") == 3
        assert content.index("A1") < content.index("A2") < content.index("A3")

    def test_entry_format(self, tmp_path):
        append_history(str(tmp_path), "my question", "my answer")
        content = (tmp_path / ".swival" / "HISTORY.md").read_text()
        # Horizontal rule, bold timestamp, italic question, blank line, answer
        assert content.startswith("---\n\n")
        assert re.search(
            r"\*\*\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\*\* — \*my question\*",
            content,
        )
        assert "\n\nmy answer\n\n" in content

    def test_timestamp_format(self, tmp_path):
        append_history(str(tmp_path), "q", "a")
        content = (tmp_path / ".swival" / "HISTORY.md").read_text()
        match = re.search(r"\*\*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\*\*", content)
        assert match, "no timestamp found"
        # Verify it parses
        from datetime import datetime

        datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Question truncation
# ---------------------------------------------------------------------------


class TestQuestionTruncation:
    def test_long_question_truncated(self, tmp_path):
        long_q = "x" * 250
        append_history(str(tmp_path), long_q, "answer")
        content = (tmp_path / ".swival" / "HISTORY.md").read_text()
        # Header should have 200 chars + ...
        assert "x" * 200 + "..." in content
        assert "x" * 201 not in content

    def test_short_question_not_truncated(self, tmp_path):
        q = "short question"
        append_history(str(tmp_path), q, "answer")
        content = (tmp_path / ".swival" / "HISTORY.md").read_text()
        assert f"*{q}*" in content
        assert "..." not in content


# ---------------------------------------------------------------------------
# Size cap
# ---------------------------------------------------------------------------


class TestSizeCap:
    def test_size_cap_skips_write(self, tmp_path):
        history = tmp_path / ".swival" / "HISTORY.md"
        history.parent.mkdir(parents=True)
        # Pre-populate past the cap
        history.write_text("x" * (MAX_HISTORY_SIZE + 100))
        original_size = history.stat().st_size

        append_history(str(tmp_path), "new q", "new a")
        assert history.stat().st_size == original_size

    def test_size_cap_exact_boundary(self, tmp_path):
        history = tmp_path / ".swival" / "HISTORY.md"
        history.parent.mkdir(parents=True)
        # Exactly at the cap — should be rejected (>=)
        history.write_text("x" * MAX_HISTORY_SIZE)
        original_size = history.stat().st_size

        append_history(str(tmp_path), "new q", "new a")
        assert history.stat().st_size == original_size

    def test_under_cap_writes(self, tmp_path):
        history = tmp_path / ".swival" / "HISTORY.md"
        history.parent.mkdir(parents=True)
        history.write_text("x" * (MAX_HISTORY_SIZE - 1000))

        append_history(str(tmp_path), "new q", "new a")
        assert history.stat().st_size > MAX_HISTORY_SIZE - 1000

    def test_large_single_entry_written(self, tmp_path):
        """File is empty, single entry exceeds cap — written because cap is checked before writing."""
        large_answer = "y" * (MAX_HISTORY_SIZE + 100_000)
        append_history(str(tmp_path), "q", large_answer)
        history = tmp_path / ".swival" / "HISTORY.md"
        assert history.exists()
        assert history.stat().st_size > MAX_HISTORY_SIZE


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


class TestPathSafety:
    def test_symlink_escape_rejected(self, tmp_path, capsys):
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        base = tmp_path / "project"
        base.mkdir()
        # Create symlink: project/.swival -> elsewhere
        (base / ".swival").symlink_to(elsewhere)

        append_history(str(base), "q", "a")
        # Should not have written to the symlink target
        assert not (elsewhere / "HISTORY.md").exists()
        assert "escapes base directory" in capsys.readouterr().err

    def test_safe_path_normal(self, tmp_path):
        result = _safe_history_path(str(tmp_path))
        expected = (tmp_path / ".swival" / "HISTORY.md").resolve()
        assert result == expected


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_write_failure_no_crash(self, tmp_path, monkeypatch, capsys):
        original_open = Path.open

        def failing_open(self, *args, **kwargs):
            if "HISTORY.md" in str(self):
                raise OSError("disk full")
            return original_open(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", failing_open)
        # Should not raise
        append_history(str(tmp_path), "q", "a")
        assert "failed to write history" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# No-op cases
# ---------------------------------------------------------------------------


class TestNoOp:
    def test_empty_answer_not_written(self, tmp_path):
        base = str(tmp_path)
        append_history(base, "q", "")
        append_history(base, "q", "   ")
        append_history(base, "q", "\n\t ")
        assert not (tmp_path / ".swival" / "HISTORY.md").exists()

    def test_creates_swival_dir(self, tmp_path):
        assert not (tmp_path / ".swival").exists()
        append_history(str(tmp_path), "q", "a")
        assert (tmp_path / ".swival").is_dir()
        assert (tmp_path / ".swival" / "HISTORY.md").exists()


# ---------------------------------------------------------------------------
# Persistence across calls
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_file_not_deleted_between_calls(self, tmp_path):
        base = str(tmp_path)
        append_history(base, "Q1", "A1")
        append_history(base, "Q2", "A2")
        content = (tmp_path / ".swival" / "HISTORY.md").read_text()
        assert "A1" in content
        assert "A2" in content


# ---------------------------------------------------------------------------
# /continue question attribution
# ---------------------------------------------------------------------------


class TestContinueLabel:
    def test_continue_label(self, tmp_path):
        append_history(str(tmp_path), "(continued)", "resumed answer")
        content = (tmp_path / ".swival" / "HISTORY.md").read_text()
        assert "*(continued)*" in content


# ---------------------------------------------------------------------------
# Exhausted answers
# ---------------------------------------------------------------------------


class TestExhaustedAnswer:
    def test_exhausted_answer_logged(self, tmp_path):
        """An answer from an exhausted run is still logged normally."""
        append_history(str(tmp_path), "big task", "partial answer from exhausted run")
        content = (tmp_path / ".swival" / "HISTORY.md").read_text()
        assert "partial answer from exhausted run" in content


# ---------------------------------------------------------------------------
# Integration tests (mock LLM, exercise real call sites in agent.py)
# ---------------------------------------------------------------------------


class TestIntegration:
    def _base_args(self, tmp_path, **overrides):
        defaults = dict(
            question="test question",
            repl=False,
            report=None,
            no_history=False,
            provider="lmstudio",
            model="test-model",
            api_key=None,
            base_url="http://localhost:1234",
            max_context_tokens=None,
            max_output_tokens=4096,
            temperature=0.5,
            top_p=1.0,
            seed=None,
            system_prompt=None,
            no_system_prompt=True,
            quiet=False,
            max_turns=5,
            base_dir=str(tmp_path),
            allowed_commands=None,
            no_instructions=True,
            skills_dir=[],
            no_skills=True,
            add_dir=[],
            add_dir_ro=[],
            yolo=False,
            color=False,
            no_color=True,
            verbose=True,
            version=False,
            no_read_guard=False,
            reviewer=None,
            init_config=False,
            project=False,
        )
        defaults.update(overrides)
        return types.SimpleNamespace(**defaults)

    def _mock_call_llm(self, answer_text):
        def mock(*args, **kwargs):
            msg = MagicMock()
            msg.content = answer_text
            msg.tool_calls = None
            return msg, "stop"

        return mock

    def test_single_shot_writes_history(self, tmp_path):
        """Single-shot path in _run_main writes to HISTORY.md."""
        fake_args = self._base_args(tmp_path)

        with (
            patch.object(agent, "build_parser") as mock_parser,
            patch.object(
                agent, "call_llm", side_effect=self._mock_call_llm("the answer")
            ),
            patch.object(agent, "discover_model", return_value=("test-model", None)),
        ):
            mock_parser.return_value.parse_args.return_value = fake_args
            agent.main()

        history = tmp_path / ".swival" / "HISTORY.md"
        assert history.exists()
        content = history.read_text()
        assert "test question" in content
        assert "the answer" in content

    def test_no_history_flag_single_shot(self, tmp_path):
        """--no-history prevents HISTORY.md creation in single-shot mode."""
        fake_args = self._base_args(tmp_path, no_history=True)

        with (
            patch.object(agent, "build_parser") as mock_parser,
            patch.object(
                agent, "call_llm", side_effect=self._mock_call_llm("the answer")
            ),
            patch.object(agent, "discover_model", return_value=("test-model", None)),
        ):
            mock_parser.return_value.parse_args.return_value = fake_args
            agent.main()

        assert not (tmp_path / ".swival" / "HISTORY.md").exists()

    def test_initial_repl_question_logged(self, tmp_path):
        """--repl 'question' logs the initial answer before entering the loop."""
        fake_args = self._base_args(tmp_path, repl=True, question="repl init q")

        with (
            patch.object(agent, "build_parser") as mock_parser,
            patch.object(
                agent, "call_llm", side_effect=self._mock_call_llm("repl init answer")
            ),
            patch.object(agent, "discover_model", return_value=("test-model", None)),
            patch.object(agent, "repl_loop"),  # Don't actually enter interactive loop
        ):
            mock_parser.return_value.parse_args.return_value = fake_args
            agent.main()

        history = tmp_path / ".swival" / "HISTORY.md"
        assert history.exists()
        content = history.read_text()
        assert "repl init q" in content
        assert "repl init answer" in content

    def test_no_history_flag_repl(self, tmp_path):
        """--no-history is threaded through to repl_loop."""
        fake_args = self._base_args(tmp_path, repl=True, question=None, no_history=True)

        with (
            patch.object(agent, "build_parser") as mock_parser,
            patch.object(agent, "discover_model", return_value=("test-model", None)),
            patch.object(agent, "repl_loop") as mock_repl,
        ):
            mock_parser.return_value.parse_args.return_value = fake_args
            agent.main()

        # Verify no_history=True was passed to repl_loop
        mock_repl.assert_called_once()
        _, kwargs = mock_repl.call_args
        assert kwargs["no_history"] is True

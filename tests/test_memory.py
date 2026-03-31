"""Tests for auto-memory (.swival/memory/MEMORY.md)."""

from pathlib import Path
from unittest.mock import patch

import pytest

from swival import fmt
from swival.agent import (
    MAX_MEMORY_CHARS,
    MAX_MEMORY_LINES,
    _safe_memory_path,
    build_system_prompt,
    load_memory,
)


@pytest.fixture(autouse=True)
def _init_fmt():
    fmt.init(color=False, no_color=False)


def _write_memory(tmp_path, content):
    """Helper to write a MEMORY.md file in the expected location."""
    mem_dir = tmp_path / ".swival" / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    mem_file = mem_dir / "MEMORY.md"
    mem_file.write_text(content, encoding="utf-8")
    return mem_file


# ---------------------------------------------------------------------------
# _safe_memory_path
# ---------------------------------------------------------------------------


class TestSafeMemoryPath:
    def test_normal_path(self, tmp_path):
        path = _safe_memory_path(str(tmp_path))
        assert path == (tmp_path / ".swival" / "memory" / "MEMORY.md").resolve()
        assert path.is_relative_to(tmp_path.resolve())

    def test_symlink_escape_file(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "MEMORY.md").write_text("evil", encoding="utf-8")

        mem_dir = tmp_path / "project" / ".swival" / "memory"
        mem_dir.mkdir(parents=True)
        (mem_dir / "MEMORY.md").symlink_to(outside / "MEMORY.md")

        with pytest.raises(ValueError, match="escapes base directory"):
            _safe_memory_path(str(tmp_path / "project"))

    def test_symlink_escape_dir(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "MEMORY.md").write_text("evil", encoding="utf-8")

        swival_dir = tmp_path / "project" / ".swival"
        swival_dir.mkdir(parents=True)
        (swival_dir / "memory").symlink_to(outside)

        with pytest.raises(ValueError, match="escapes base directory"):
            _safe_memory_path(str(tmp_path / "project"))


# ---------------------------------------------------------------------------
# load_memory — budgeted mode (default)
# ---------------------------------------------------------------------------


class TestLoadMemory:
    def test_no_memory_dir(self, tmp_path):
        assert load_memory(str(tmp_path)) == ""

    def test_no_memory_file(self, tmp_path):
        (tmp_path / ".swival" / "memory").mkdir(parents=True)
        assert load_memory(str(tmp_path)) == ""

    def test_basic_load(self, tmp_path):
        _write_memory(tmp_path, "- project uses pytest\n- src/ layout\n")
        result = load_memory(str(tmp_path))
        assert "<memory>" in result
        assert "</memory>" in result
        assert "project uses pytest" in result
        assert "src/ layout" in result

    def test_preamble_contains_not_instructions(self, tmp_path):
        _write_memory(tmp_path, "- some fact\n")
        result = load_memory(str(tmp_path))
        assert "not instructions" in result
        assert "do not override" in result.lower()

    def test_empty_file(self, tmp_path):
        _write_memory(tmp_path, "")
        assert load_memory(str(tmp_path)) == ""

    def test_whitespace_only_file(self, tmp_path):
        _write_memory(tmp_path, "   \n  \n  ")
        assert load_memory(str(tmp_path)) == ""

    def test_non_utf8_bytes(self, tmp_path):
        mem_dir = tmp_path / ".swival" / "memory"
        mem_dir.mkdir(parents=True)
        mem_file = mem_dir / "MEMORY.md"
        mem_file.write_bytes(b"- valid line\n- bad byte \xff here\n")
        result = load_memory(str(tmp_path))
        assert "<memory>" in result
        assert "valid line" in result
        assert "\ufffd" in result  # replacement char

    def test_oserror(self, tmp_path):
        _write_memory(tmp_path, "- some fact\n")
        with patch.object(Path, "open", side_effect=OSError("denied")):
            assert load_memory(str(tmp_path)) == ""

    def test_verbose_logging(self, tmp_path, capsys):
        _write_memory(tmp_path, "- fact one\n- fact two\n")
        load_memory(str(tmp_path), verbose=True)
        stderr = capsys.readouterr().err
        assert "Memory:" in stderr
        assert "entries" in stderr

    def test_bootstrap_entries_always_included(self, tmp_path):
        content = (
            "<!-- bootstrap -->\n"
            "## Provider Info\n"
            "- Always needed\n\n"
            "## Task Notes\n"
            "- Only when relevant\n"
        )
        _write_memory(tmp_path, content)
        result = load_memory(str(tmp_path), user_query="something unrelated")
        assert "Always needed" in result

    def test_retrieval_uses_query(self, tmp_path):
        content = (
            "## Authentication\n"
            "- Token refresh was failing\n\n"
            "## Database\n"
            "- Use PostgreSQL 15\n\n"
            "## CSS Layout\n"
            "- Flexbox for the sidebar\n"
        )
        _write_memory(tmp_path, content)
        result = load_memory(str(tmp_path), user_query="auth token refresh bug")
        assert "Token refresh" in result

    def test_no_query_takes_first_entries(self, tmp_path):
        content = "## First\n- First entry\n\n## Second\n- Second entry\n"
        _write_memory(tmp_path, content)
        result = load_memory(str(tmp_path))
        assert "First entry" in result

    def test_large_file_retrieves_past_8k(self, tmp_path):
        """Budgeted mode reads the entire file, not just the first 8KB."""
        # Build a file larger than MAX_MEMORY_CHARS with a unique entry at the end
        filler_entries = []
        for i in range(100):
            filler_entries.append(f"## Filler {i}\n- {'x' * 80}\n")
        filler = "\n".join(filler_entries)
        assert len(filler) > MAX_MEMORY_CHARS  # confirm it exceeds old cap

        target = "## Secret Target\n- unique_findable_needle_xyz\n"
        _write_memory(tmp_path, filler + "\n" + target)
        result = load_memory(str(tmp_path), user_query="unique_findable_needle_xyz")
        assert "unique_findable_needle_xyz" in result

    def test_telemetry_recorded(self, tmp_path):
        from swival.report import ReportCollector

        _write_memory(tmp_path, "## Topic\n- fact\n")
        report = ReportCollector()
        load_memory(str(tmp_path), user_query="fact", report=report)
        assert report.memory_stats is not None
        assert report.memory_stats["mode"] == "budgeted"
        assert report.memory_stats["total_entries"] == 1
        assert len(report.memory_stats["retrieved_ids"]) >= 1

    def test_telemetry_recorded_when_nothing_injected(self, tmp_path):
        """Telemetry is recorded even when no entries match the query."""
        from swival.report import ReportCollector

        _write_memory(tmp_path, "## Topic\n- some fact\n")
        report = ReportCollector()
        result = load_memory(
            str(tmp_path), user_query="completely unrelated xyzzy", report=report
        )
        assert result == ""
        assert report.memory_stats is not None
        assert report.memory_stats["mode"] == "budgeted"
        assert report.memory_stats["total_entries"] == 1
        assert report.memory_stats["retrieved_ids"] == []
        assert report.memory_stats["retrieval_tokens"] == 0


# ---------------------------------------------------------------------------
# load_memory — full mode (legacy)
# ---------------------------------------------------------------------------


class TestLoadMemoryFull:
    def test_line_limit(self, tmp_path):
        lines = [f"- line {i}\n" for i in range(300)]
        _write_memory(tmp_path, "".join(lines))
        result = load_memory(str(tmp_path), memory_full=True)
        assert f"truncated at {MAX_MEMORY_LINES} lines" in result
        assert "line 0" in result
        assert "line 199" in result
        assert "line 200" not in result

    def test_char_limit(self, tmp_path):
        line = "x" * 199 + "\n"  # 200 chars per line
        content = line * 50  # 10,000 chars total, 50 lines (under line cap)
        _write_memory(tmp_path, content)
        result = load_memory(str(tmp_path), memory_full=True)
        assert f"truncated at {MAX_MEMORY_CHARS} characters" in result

    def test_char_limit_cuts_at_line_boundary(self, tmp_path):
        line = "x" * 199 + "\n"  # 200 chars per line
        content = line * 50  # 10,000 chars total
        _write_memory(tmp_path, content)
        result = load_memory(str(tmp_path), memory_full=True)
        inner = result.split("\n\n", 1)[1].rsplit("\n</memory>", 1)[0]
        for ln in inner.split("\n"):
            if ln.startswith("[..."):
                continue
            if ln.startswith("x"):
                assert len(ln) == 199

    def test_char_limit_single_long_line(self, tmp_path):
        content = "x" * (MAX_MEMORY_CHARS + 1000)
        _write_memory(tmp_path, content)
        result = load_memory(str(tmp_path), memory_full=True)
        assert f"truncated at {MAX_MEMORY_CHARS} characters" in result
        inner = result.split("\n\n", 1)[1].rsplit("\n</memory>", 1)[0]
        if "[... truncated" in inner:
            inner = inner.rsplit("\n[... truncated", 1)[0]
        assert len(inner) <= MAX_MEMORY_CHARS

    def test_line_limit_short_lines(self, tmp_path):
        lines = [f"- line {i}\n" for i in range(300)]
        _write_memory(tmp_path, "".join(lines))
        result = load_memory(str(tmp_path), memory_full=True)
        assert f"truncated at {MAX_MEMORY_LINES} lines" in result
        assert "truncated at 8000 characters" not in result

    def test_line_and_char_limit(self, tmp_path):
        line = "x" * 199 + "\n"
        content = line * 300
        _write_memory(tmp_path, content)
        result = load_memory(str(tmp_path), memory_full=True)
        assert f"truncated at {MAX_MEMORY_CHARS} characters" in result

    def test_verbose_logging(self, tmp_path, capsys):
        _write_memory(tmp_path, "- fact one\n- fact two\n")
        load_memory(str(tmp_path), verbose=True, memory_full=True)
        stderr = capsys.readouterr().err
        assert "Loaded memory" in stderr
        assert "2 lines" in stderr


# ---------------------------------------------------------------------------
# build_system_prompt integration
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def _build(self, tmp_path, **kwargs):
        defaults = dict(
            base_dir=str(tmp_path),
            system_prompt=None,
            no_system_prompt=False,
            no_instructions=True,
            no_memory=False,
            skills_catalog={},
            verbose=False,
        )
        defaults.update(kwargs)
        return build_system_prompt(**defaults)

    def test_memory_in_system_prompt(self, tmp_path):
        _write_memory(tmp_path, "- uses pytest\n")
        content, _ = self._build(tmp_path)
        assert "<memory>" in content
        assert "uses pytest" in content

    def test_no_memory_flag(self, tmp_path):
        _write_memory(tmp_path, "- uses pytest\n")
        content, _ = self._build(tmp_path, no_memory=True)
        assert "<memory>" not in content

    def test_custom_system_prompt_skips_memory(self, tmp_path):
        _write_memory(tmp_path, "- uses pytest\n")
        content, _ = self._build(tmp_path, system_prompt="Custom prompt.")
        assert "<memory>" not in content

    def test_memory_after_instructions(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("Do this.", encoding="utf-8")
        _write_memory(tmp_path, "- fact\n")
        content, _ = self._build(tmp_path, no_instructions=False)
        instr_pos = content.find("</agent-instructions>")
        memory_pos = content.find("<memory>")
        assert instr_pos < memory_pos

    def test_memory_full_flag(self, tmp_path):
        lines = [f"- line {i}\n" for i in range(300)]
        _write_memory(tmp_path, "".join(lines))
        content, _ = self._build(tmp_path, memory_full=True)
        assert f"truncated at {MAX_MEMORY_LINES} lines" in content


# ---------------------------------------------------------------------------
# Session API — memory injection parity
# ---------------------------------------------------------------------------


class TestSessionMemory:
    """Verify Session.run() and Session.ask() inject memory keyed from question."""

    def _make_message(self, content):
        import types

        msg = types.SimpleNamespace()
        msg.content = content
        msg.tool_calls = None
        msg.role = "assistant"
        return msg

    def test_run_injects_query_keyed_memory(self, tmp_path, monkeypatch):
        from swival import Session, agent

        content = (
            "## Authentication\n"
            "- Token refresh was failing\n\n"
            "## Database\n"
            "- Use PostgreSQL 15\n"
        )
        _write_memory(tmp_path, content)

        captured_messages = []

        def capturing_llm(base_url, model_id, messages, *args, **kwargs):
            captured_messages.extend(messages)
            return self._make_message("done"), "stop"

        monkeypatch.setattr(agent, "call_llm", capturing_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("m", None))

        s = Session(base_dir=str(tmp_path), history=False)
        s.run("fix the auth token refresh bug")

        system_msg = captured_messages[0]["content"]
        assert "<memory>" in system_msg
        assert "Token refresh" in system_msg

    def test_run_report_has_memory_stats(self, tmp_path, monkeypatch):
        from swival import Session, agent

        _write_memory(tmp_path, "## Topic\n- some fact\n")

        def simple_llm(base_url, model_id, messages, *args, **kwargs):
            return self._make_message("done"), "stop"

        monkeypatch.setattr(agent, "call_llm", simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("m", None))

        s = Session(base_dir=str(tmp_path), history=False)
        result = s.run("tell me something", report=True)

        assert result.report is not None
        assert "memory" in result.report["stats"]
        assert result.report["stats"]["memory"]["mode"] == "budgeted"

    def test_ask_injects_memory_on_first_call(self, tmp_path, monkeypatch):
        from swival import Session, agent

        _write_memory(tmp_path, "## Auth\n- Token refresh bug\n")

        captured_system = []

        def capturing_llm(base_url, model_id, messages, *args, **kwargs):
            for m in messages:
                if m.get("role") == "system":
                    captured_system.append(m["content"])
            return self._make_message("done"), "stop"

        monkeypatch.setattr(agent, "call_llm", capturing_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("m", None))

        s = Session(base_dir=str(tmp_path), history=False)
        s.ask("fix the auth issue")

        assert len(captured_system) >= 1
        assert "<memory>" in captured_system[0]
        assert "Token refresh" in captured_system[0]

    def test_run_memory_full_report(self, tmp_path, monkeypatch):
        """memory_full=True telemetry flows through Session.run(report=True)."""
        from swival import Session, agent

        _write_memory(tmp_path, "## Topic\n- some fact\n")

        def simple_llm(base_url, model_id, messages, *args, **kwargs):
            return self._make_message("done"), "stop"

        monkeypatch.setattr(agent, "call_llm", simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("m", None))

        s = Session(base_dir=str(tmp_path), history=False, memory_full=True)
        result = s.run("tell me something", report=True)

        assert result.report is not None
        assert "memory" in result.report["stats"]
        assert result.report["stats"]["memory"]["mode"] == "full"

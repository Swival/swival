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
# load_memory
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

    def test_line_limit(self, tmp_path):
        lines = [f"- line {i}\n" for i in range(300)]
        _write_memory(tmp_path, "".join(lines))
        result = load_memory(str(tmp_path))
        assert f"truncated at {MAX_MEMORY_LINES} lines" in result
        assert "line 0" in result
        assert "line 199" in result
        assert "line 200" not in result

    def test_char_limit(self, tmp_path):
        line = "x" * 199 + "\n"  # 200 chars per line
        content = line * 50  # 10,000 chars total, 50 lines (under line cap)
        _write_memory(tmp_path, content)
        result = load_memory(str(tmp_path))
        assert f"truncated at {MAX_MEMORY_CHARS} characters" in result

    def test_char_limit_cuts_at_line_boundary(self, tmp_path):
        line = "x" * 199 + "\n"  # 200 chars per line
        content = line * 50  # 10,000 chars total
        _write_memory(tmp_path, content)
        result = load_memory(str(tmp_path))
        # Extract content between tags
        inner = result.split("\n\n", 1)[1].rsplit("\n</memory>", 1)[0]
        # The actual memory content (after preamble) should end at a line boundary
        memory_lines = inner.split("\n")
        # Last non-truncation-marker line should be a complete line of x's or the marker
        for line in memory_lines:
            if line.startswith("[..."):
                continue
            if line.startswith("x"):
                # Should be full 199 x's, not truncated mid-line
                assert len(line) == 199

    def test_char_limit_single_long_line(self, tmp_path):
        content = "x" * (MAX_MEMORY_CHARS + 1000)
        _write_memory(tmp_path, content)
        result = load_memory(str(tmp_path))
        assert f"truncated at {MAX_MEMORY_CHARS} characters" in result
        # Content portion should be hard-cut at MAX_MEMORY_CHARS
        inner = result.split("\n\n", 1)[1].rsplit("\n</memory>", 1)[0]
        # Remove truncation marker
        if "[... truncated" in inner:
            inner = inner.rsplit("\n[... truncated", 1)[0]
        assert len(inner) <= MAX_MEMORY_CHARS

    def test_line_limit_short_lines(self, tmp_path):
        """300 short lines (~6.2K chars) — only line cap fires, not char cap."""
        lines = [f"- line {i}\n" for i in range(300)]  # ~20 chars each
        _write_memory(tmp_path, "".join(lines))
        result = load_memory(str(tmp_path))
        assert f"truncated at {MAX_MEMORY_LINES} lines" in result
        assert "truncated at 8000 characters" not in result

    def test_line_and_char_limit(self, tmp_path):
        line = "x" * 199 + "\n"  # 200 chars per line
        content = line * 300  # 60,000 chars, 300 lines — both caps applicable
        _write_memory(tmp_path, content)
        result = load_memory(str(tmp_path))
        # char-cap message wins since it's the binding constraint
        assert f"truncated at {MAX_MEMORY_CHARS} characters" in result

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
            yolo=False,
            resolved_commands={},
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

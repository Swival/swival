"""Tests for auto-memory (.swival/memory/MEMORY.md)."""

from pathlib import Path
from unittest.mock import patch

import pytest

from swival import fmt
from swival.agent import build_system_prompt
from swival.memory import (
    MAX_MEMORY_CHARS,
    MAX_MEMORY_LINES,
    safe_memory_path as _safe_memory_path,
    load_memory,
    extract_learned_tags,
    persist_learnings,
    auto_extract_and_persist,
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


# ---------------------------------------------------------------------------
# extract_learned_tags
# ---------------------------------------------------------------------------


class TestExtractLearnedTags:
    def test_no_tags(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        assert extract_learned_tags(messages) == []

    def test_single_tag(self):
        messages = [
            {
                "role": "assistant",
                "content": "I found that <learned>rg is faster than grep</learned> here.",
            },
        ]
        result = extract_learned_tags(messages)
        assert result == ["rg is faster than grep"]

    def test_multiple_tags_same_message(self):
        messages = [
            {
                "role": "assistant",
                "content": (
                    "<learned>fact one</learned> and also <learned>fact two</learned>"
                ),
            },
        ]
        result = extract_learned_tags(messages)
        assert result == ["fact one", "fact two"]

    def test_multiple_messages(self):
        messages = [
            {"role": "assistant", "content": "<learned>first</learned>"},
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "<learned>second</learned>"},
        ]
        result = extract_learned_tags(messages)
        assert result == ["first", "second"]

    def test_deduplication(self):
        messages = [
            {"role": "assistant", "content": "<learned>same fact</learned>"},
            {"role": "assistant", "content": "<learned>same fact</learned>"},
        ]
        result = extract_learned_tags(messages)
        assert result == ["same fact"]

    def test_ignores_user_messages(self):
        messages = [
            {"role": "user", "content": "<learned>not from assistant</learned>"},
        ]
        assert extract_learned_tags(messages) == []

    def test_ignores_tool_messages(self):
        messages = [
            {"role": "tool", "content": "<learned>not from assistant</learned>"},
        ]
        assert extract_learned_tags(messages) == []

    def test_multiline_tag(self):
        messages = [
            {
                "role": "assistant",
                "content": "<learned>line one\nline two</learned>",
            },
        ]
        result = extract_learned_tags(messages)
        assert result == ["line one\nline two"]

    def test_empty_tag_skipped(self):
        messages = [
            {"role": "assistant", "content": "<learned>  </learned>"},
        ]
        assert extract_learned_tags(messages) == []

    def test_empty_content(self):
        messages = [
            {"role": "assistant", "content": ""},
        ]
        assert extract_learned_tags(messages) == []

    def test_none_content(self):
        messages = [
            {"role": "assistant", "content": None},
        ]
        assert extract_learned_tags(messages) == []


# ---------------------------------------------------------------------------
# persist_learnings
# ---------------------------------------------------------------------------


class TestPersistLearnings:
    def test_creates_file_and_dir(self, tmp_path):
        count = persist_learnings(str(tmp_path), ["pytest is great"])
        assert count == 1
        mem = (tmp_path / ".swival" / "memory" / "MEMORY.md").read_text()
        assert "- pytest is great\n" in mem

    def test_appends_to_existing(self, tmp_path):
        _write_memory(tmp_path, "- existing fact\n")
        count = persist_learnings(str(tmp_path), ["new fact"])
        assert count == 1
        mem = (tmp_path / ".swival" / "memory" / "MEMORY.md").read_text()
        assert "- existing fact\n" in mem
        assert "- new fact\n" in mem

    def test_dedup_against_existing(self, tmp_path):
        _write_memory(tmp_path, "- already known\n")
        count = persist_learnings(str(tmp_path), ["already known"])
        assert count == 0

    def test_empty_learnings(self, tmp_path):
        count = persist_learnings(str(tmp_path), [])
        assert count == 0

    def test_multiple_learnings(self, tmp_path):
        count = persist_learnings(str(tmp_path), ["fact A", "fact B", "fact C"])
        assert count == 3
        mem = (tmp_path / ".swival" / "memory" / "MEMORY.md").read_text()
        assert "- fact A\n" in mem
        assert "- fact B\n" in mem
        assert "- fact C\n" in mem

    def test_multiline_learning(self, tmp_path):
        count = persist_learnings(str(tmp_path), ["line one\nline two"])
        assert count == 1
        mem = (tmp_path / ".swival" / "memory" / "MEMORY.md").read_text()
        assert "- line one\n  line two\n" in mem

    def test_size_limit_prevents_write(self, tmp_path):
        _write_memory(tmp_path, "x" * 50_000)
        count = persist_learnings(str(tmp_path), ["new fact"], verbose=True)
        assert count == 0

    def test_partial_dedup(self, tmp_path):
        _write_memory(tmp_path, "- known fact\n")
        count = persist_learnings(str(tmp_path), ["known fact", "new fact"])
        assert count == 1
        mem = (tmp_path / ".swival" / "memory" / "MEMORY.md").read_text()
        assert "- new fact\n" in mem

    def test_multiline_not_suppressed_by_first_line_match(self, tmp_path):
        """A multi-line learning should not be suppressed just because its
        first line appears in an unrelated existing entry."""
        _write_memory(tmp_path, "- rg is faster than grep\n")
        count = persist_learnings(
            str(tmp_path), ["rg is faster than grep\nUse it for content search"]
        )
        assert count == 1
        mem = (tmp_path / ".swival" / "memory" / "MEMORY.md").read_text()
        assert "Use it for content search" in mem

    def test_ensures_newline_before_append(self, tmp_path):
        _write_memory(tmp_path, "- no trailing newline")
        persist_learnings(str(tmp_path), ["new fact"])
        mem = (tmp_path / ".swival" / "memory" / "MEMORY.md").read_text()
        # Should not have "newline" and "- new" jammed together
        assert "\n- new fact\n" in mem


# ---------------------------------------------------------------------------
# auto_extract_and_persist
# ---------------------------------------------------------------------------


class TestAutoExtractAndPersist:
    def test_end_to_end(self, tmp_path):
        messages = [
            {"role": "user", "content": "fix the bug"},
            {
                "role": "assistant",
                "content": "Done. <learned>The config parser needs strip() on keys</learned>",
            },
        ]
        count = auto_extract_and_persist(messages, str(tmp_path))
        assert count == 1
        mem = (tmp_path / ".swival" / "memory" / "MEMORY.md").read_text()
        assert "config parser needs strip()" in mem

    def test_no_learnings(self, tmp_path):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        count = auto_extract_and_persist(messages, str(tmp_path))
        assert count == 0
        assert not (tmp_path / ".swival" / "memory" / "MEMORY.md").exists()

    def test_dedup_across_sessions(self, tmp_path):
        _write_memory(tmp_path, "- old lesson\n")
        messages = [
            {"role": "assistant", "content": "<learned>old lesson</learned>"},
        ]
        count = auto_extract_and_persist(messages, str(tmp_path))
        assert count == 0

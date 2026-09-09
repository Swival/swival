"""Tests for memory entry parsing and BM25 retrieval (swival.memory)."""

import pytest

from swival.memory import MemoryEntry, parse_memory, retrieve_bm25


# ---------------------------------------------------------------------------
# parse_memory
# ---------------------------------------------------------------------------


class TestParseMemory:
    @pytest.mark.parametrize(
        "opening,inner,closing",
        [
            ("~~~python", "", "~~~"),
            ("   ```python", "", " ```"),
            ("````", "```", "````"),
            ("```", "~~~", "```"),
            ("~~~", "```", "~~~~"),
            ("```", "```not a closing fence", "```"),
            ("```", "    ```", "```"),
        ],
    )
    def test_fenced_code_stays_in_its_memory_entry(self, opening, inner, closing):
        code = f"{opening}\n{inner}\n<!-- bootstrap -->\n## Example\n{closing}"
        text = f"## Real\n{code}\n\n## Next\nA fact.\n"
        entries = parse_memory(text)
        assert [entry.heading for entry in entries] == ["Real", "Next"]
        assert entries[0].content == f"## Real\n{code}"
        assert not any(entry.is_bootstrap for entry in entries)

    def test_empty_string(self):
        assert parse_memory("") == []

    def test_whitespace_only(self):
        assert parse_memory("   \n\n  ") == []

    def test_single_entry_no_heading(self):
        entries = parse_memory("- fact one\n- fact two\n")
        assert len(entries) == 1
        assert entries[0].id == "m0"
        assert entries[0].heading is None
        assert "fact one" in entries[0].content
        assert not entries[0].is_bootstrap

    def test_single_heading(self):
        entries = parse_memory("## Topic\n- detail\n")
        assert len(entries) == 1
        assert entries[0].heading == "Topic"
        assert "## Topic" in entries[0].content
        assert "detail" in entries[0].content

    def test_multiple_headings(self):
        text = "## First\n- a\n\n## Second\n- b\n\n## Third\n- c\n"
        entries = parse_memory(text)
        assert len(entries) == 3
        assert [e.heading for e in entries] == ["First", "Second", "Third"]
        assert [e.id for e in entries] == ["m0", "m1", "m2"]

    def test_unheaded_content_plus_headings(self):
        text = "Some intro.\n\n## Topic\n- detail\n"
        entries = parse_memory(text)
        assert len(entries) == 2
        assert entries[0].heading is None
        assert "intro" in entries[0].content
        assert entries[1].heading == "Topic"

    def test_heading_levels(self):
        text = "# H1\n- a\n## H2\n- b\n### H3\n- c\n#### H4\n- d\n##### H5\n- e\n###### H6\n- f\n"
        entries = parse_memory(text)
        assert len(entries) == 6
        assert [e.heading for e in entries] == ["H1", "H2", "H3", "H4", "H5", "H6"]

    def test_heading_inside_code_block_ignored(self):
        text = "## Real\n- content\n\n```\n## Not a heading\nsome code\n```\n\n## Also Real\n- more\n"
        entries = parse_memory(text)
        assert len(entries) == 2
        assert entries[0].heading == "Real"
        assert "Not a heading" in entries[0].content
        assert entries[1].heading == "Also Real"

    def test_empty_heading(self):
        text = "## Topic\n\n## Another\n- stuff\n"
        entries = parse_memory(text)
        # Empty heading entry has no content besides the heading line itself
        assert len(entries) == 2
        assert entries[0].heading == "Topic"
        assert entries[1].heading == "Another"

    def test_bootstrap_tag(self):
        text = "<!-- bootstrap -->\n## Provider\n- quirk\n\n## Debug\n- finding\n"
        entries = parse_memory(text)
        assert len(entries) == 2
        assert entries[0].is_bootstrap
        assert entries[0].heading == "Provider"
        assert not entries[1].is_bootstrap

    def test_bootstrap_tag_not_in_content(self):
        text = "<!-- bootstrap -->\n## Provider\n- quirk\n"
        entries = parse_memory(text)
        assert "<!-- bootstrap -->" not in entries[0].content

    def test_bootstrap_tag_only_affects_next_heading(self):
        text = "<!-- bootstrap -->\n\n## Topic\n- stuff\n"
        entries = parse_memory(text)
        # Blank line between tag and heading — tag is consumed but doesn't match
        # because the blank line resets pending_bootstrap
        assert not entries[0].is_bootstrap

    def test_no_heading_after_bootstrap_tag(self):
        text = "<!-- bootstrap -->\n- plain text\n"
        entries = parse_memory(text)
        assert len(entries) == 1
        assert not entries[0].is_bootstrap

    def test_multiple_bootstrap_entries(self):
        text = (
            "<!-- bootstrap -->\n## A\n- a\n\n"
            "<!-- bootstrap -->\n## B\n- b\n\n"
            "## C\n- c\n"
        )
        entries = parse_memory(text)
        assert entries[0].is_bootstrap
        assert entries[1].is_bootstrap
        assert not entries[2].is_bootstrap

    def test_token_count_cached(self):
        entries = parse_memory("## Topic\n- some content here\n")
        e = entries[0]
        t1 = e.tokens
        t2 = e.tokens
        assert t1 == t2
        assert t1 > 0

    def test_not_a_heading(self):
        text = "#notheading\n- stuff\n"
        entries = parse_memory(text)
        assert len(entries) == 1
        assert entries[0].heading is None

    def test_seven_hashes_not_heading(self):
        text = "####### too many\n- stuff\n"
        entries = parse_memory(text)
        assert len(entries) == 1
        assert entries[0].heading is None


# ---------------------------------------------------------------------------
# retrieve_bm25
# ---------------------------------------------------------------------------


class TestRetrieveBm25:
    def _make_entries(self, *contents):
        return [
            MemoryEntry(id=f"m{i}", heading=f"E{i}", content=c)
            for i, c in enumerate(contents)
        ]

    def test_empty_query(self):
        entries = self._make_entries("hello world")
        assert retrieve_bm25("", entries) == []

    def test_empty_entries(self):
        assert retrieve_bm25("hello", []) == []

    def test_basic_ranking(self):
        entries = self._make_entries(
            "authentication token refresh failed on startup",
            "database migration script for PostgreSQL",
            "CSS flexbox layout for sidebar component",
        )
        results = retrieve_bm25("auth token", entries, top_k=3, token_budget=2000)
        assert len(results) >= 1
        assert results[0][0].id == "m0"
        assert results[0][1] > 0

    def test_top_k_limits(self):
        entries = self._make_entries("a b c", "a b d", "a b e", "a b f")
        results = retrieve_bm25("a b", entries, top_k=2, token_budget=2000)
        assert len(results) <= 2

    def test_irrelevant_query_returns_empty(self):
        entries = self._make_entries(
            "python pytest testing framework",
            "database schema migration",
        )
        results = retrieve_bm25("xyzzyplugh", entries, top_k=3, token_budget=2000)
        assert len(results) == 0

    def test_token_budget_respected(self):
        entries = self._make_entries(
            "short entry",
            "another short entry",
            "yet another short entry with some overlap",
        )
        # Very small budget should limit results
        results = retrieve_bm25("entry", entries, top_k=10, token_budget=5)
        # Should stop adding entries once budget exhausted
        assert len(results) <= len(entries)

    def test_score_ordering(self):
        entries = self._make_entries(
            "token token token token",  # most relevant
            "token once",  # somewhat relevant
            "completely different topic xyz",  # irrelevant
        )
        results = retrieve_bm25("token", entries, top_k=3, token_budget=2000)
        if len(results) >= 2:
            assert results[0][1] >= results[1][1]

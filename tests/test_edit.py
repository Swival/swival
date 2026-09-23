"""Tests for edit.py — the string replacement engine behind edit_file."""

import pytest

from swival.edit import replace, _exact_match_spans, _LINE_BREAK_RE


# =========================================================================
# Exact match
# =========================================================================


class TestExactMatch:
    """Basic exact-match replacements."""

    def test_simple_replacement(self):
        content = "hello world"
        result = replace(content, "hello", "goodbye")
        assert result == "goodbye world"

    def test_multiline_replacement(self):
        content = "aaa\nbbb\nccc\n"
        result = replace(content, "bbb\nccc", "BBB\nCCC")
        assert result == "aaa\nBBB\nCCC\n"

    def test_replacement_at_start(self):
        content = "first\nsecond\nthird\n"
        result = replace(content, "first", "FIRST")
        assert result == "FIRST\nsecond\nthird\n"

    def test_replacement_at_end(self):
        content = "first\nsecond\nthird"
        result = replace(content, "third", "THIRD")
        assert result == "first\nsecond\nTHIRD"

    def test_replacement_preserves_surrounding(self):
        content = "before\ntarget line\nafter\n"
        result = replace(content, "target line", "new line")
        assert result == "before\nnew line\nafter\n"


# =========================================================================
# Replace all
# =========================================================================


class TestReplaceAll:
    """replace_all=True replaces every occurrence."""

    def test_replaces_all_occurrences(self):
        content = "foo bar foo baz foo"
        result = replace(content, "foo", "qux", replace_all=True)
        assert result == "qux bar qux baz qux"

    def test_replaces_all_multiline(self):
        content = "x = 1\ny = 2\nx = 1\ny = 3\n"
        result = replace(content, "x = 1", "x = 0", replace_all=True)
        assert result == "x = 0\ny = 2\nx = 0\ny = 3\n"

    def test_replace_all_single_occurrence(self):
        content = "only once here"
        result = replace(content, "once", "twice", replace_all=True)
        assert result == "only twice here"


# =========================================================================
# Multiple matches error
# =========================================================================


class TestMultipleMatchesError:
    """Ambiguous matches raise ValueError when replace_all=False."""

    def test_duplicate_raises_with_nudge(self):
        content = "aaa\nbbb\naaa\n"
        with pytest.raises(ValueError, match="add line_number"):
            replace(content, "aaa", "ccc")

    def test_adding_context_resolves_ambiguity(self):
        content = "aaa\nbbb\naaa\nccc\n"
        result = replace(content, "aaa\nccc", "AAA\nCCC")
        assert result == "aaa\nbbb\nAAA\nCCC\n"


# =========================================================================
# Not found
# =========================================================================


class TestNotFound:
    """Missing strings raise ValueError."""

    def test_string_not_in_content(self):
        content = "hello world"
        with pytest.raises(ValueError, match="not found"):
            replace(content, "xyz", "abc")

    def test_partial_match_not_found(self):
        content = "abc def ghi"
        with pytest.raises(ValueError, match="not found"):
            replace(content, "abc ghi", "replaced")


# =========================================================================
# No-op (old == new)
# =========================================================================


class TestNoOp:
    """old_string == new_string raises ValueError."""

    def test_identical_strings(self):
        content = "hello world"
        with pytest.raises(ValueError, match="identical"):
            replace(content, "hello", "hello")

    def test_identical_multiline(self):
        content = "a\nb\nc\n"
        with pytest.raises(ValueError, match="identical"):
            replace(content, "a\nb", "a\nb")


# =========================================================================
# Validation
# =========================================================================


class TestValidation:
    """Input validation."""

    def test_empty_old_string_raises(self):
        content = "hello"
        with pytest.raises(ValueError, match="empty"):
            replace(content, "", "something")


# =========================================================================
# Fuzzy matching
# =========================================================================


class TestFuzzyMatching:
    """Whitespace tolerance and Unicode normalization."""

    def test_leading_whitespace_tolerance(self):
        content = "  indented line\nafter\n"
        result = replace(content, "indented line", "new line")
        assert "new line" in result

    def test_trailing_whitespace_tolerance(self):
        content = "line with spaces   \nafter\n"
        result = replace(content, "line with spaces", "clean line")
        assert "clean line" in result

    def test_unicode_smart_quotes_normalized(self):
        content = "print(\u201chello\u201d)\n"
        result = replace(content, 'print("hello")', 'print("world")')
        assert 'print("world")' in result

    def test_unicode_em_dash_normalized(self):
        content = "value \u2014 10\n"
        result = replace(content, "value - 10", "value - 20")
        assert "value - 20" in result

    def test_unicode_ellipsis_normalized(self):
        content = "loading\u2026\n"
        result = replace(content, "loading...", "done")
        assert "done" in result

    def test_non_breaking_space_normalized(self):
        content = "hello\u00a0world\n"
        result = replace(content, "hello world", "hi there")
        assert "hi there" in result


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases:
    """Unusual content shapes."""

    def test_single_line_no_newline(self):
        content = "only line"
        result = replace(content, "only", "first")
        assert result == "first line"

    def test_replacement_with_empty_new_string(self):
        content = "keep\nremove\nkeep\n"
        result = replace(content, "remove\n", "")
        assert result == "keep\nkeep\n"

    def test_multiline_block_replacement(self):
        content = "a\nb\nc\nd\ne\n"
        result = replace(content, "b\nc\nd", "B\nC\nD")
        assert result == "a\nB\nC\nD\ne\n"


class TestLineNumberTargeting:
    """Line-number targeting for disambiguating repeated matches."""

    def test_exact_duplicate_select_second(self):
        content = "aaa\nbbb\naaa\nccc\n"
        result = replace(content, "aaa", "XXX", line_number=3)
        assert result == "aaa\nbbb\nXXX\nccc\n"

    def test_exact_duplicate_select_first(self):
        content = "aaa\nbbb\naaa\nccc\n"
        result = replace(content, "aaa", "XXX", line_number=1)
        assert result == "XXX\nbbb\naaa\nccc\n"

    def test_multiline_match_any_line_in_span(self):
        content = "x\naaa\nbbb\nx\naaa\nbbb\nx\n"
        result = replace(content, "aaa\nbbb", "AAA\nBBB", line_number=5)
        assert result == "x\naaa\nbbb\nx\nAAA\nBBB\nx\n"

    def test_multiline_match_last_line_of_span(self):
        content = "x\naaa\nbbb\nx\naaa\nbbb\nx\n"
        result = replace(content, "aaa\nbbb", "AAA\nBBB", line_number=6)
        assert result == "x\naaa\nbbb\nx\nAAA\nBBB\nx\n"

    def test_fuzzy_trimmed_with_line_number(self):
        content = "  aaa\nbbb\n  aaa\nccc\n"
        result = replace(content, "aaa", "XXX", line_number=3)
        assert "XXX" in result
        assert result.startswith("  aaa\n")

    def test_unicode_normalized_with_line_number(self):
        content = "print(\u201chello\u201d)\nother\nprint(\u201chello\u201d)\n"
        result = replace(content, 'print("hello")', 'print("world")', line_number=3)
        assert 'print("world")' in result
        assert result.startswith("print(\u201chello\u201d)\n")

    def test_no_match_at_line_reports_candidates(self):
        content = "aaa\nbbb\naaa\nccc\n"
        with pytest.raises(ValueError, match=r"no match at line 2.*lines 1, 3"):
            replace(content, "aaa", "XXX", line_number=2)

    def test_not_found_still_raises_not_found(self):
        content = "aaa\nbbb\nccc\n"
        with pytest.raises(ValueError, match="not found"):
            replace(content, "zzz", "XXX", line_number=1)

    def test_invalid_line_number_zero_ignored(self):
        content = "aaa\nbbb\naaa\n"
        with pytest.raises(ValueError, match="multiple matches"):
            replace(content, "aaa", "XXX", line_number=0)

    def test_invalid_line_number_negative_ignored(self):
        content = "aaa\nbbb\naaa\n"
        with pytest.raises(ValueError, match="multiple matches"):
            replace(content, "aaa", "XXX", line_number=-5)

    def test_invalid_line_number_string_ignored(self):
        content = "aaa\nbbb\naaa\n"
        with pytest.raises(ValueError, match="multiple matches"):
            replace(content, "aaa", "XXX", line_number="hello")

    def test_bool_line_number_ignored(self):
        content = "aaa\nbbb\naaa\n"
        with pytest.raises(ValueError, match="multiple matches"):
            replace(content, "aaa", "XXX", line_number=True)

    def test_replace_all_ignores_line_number(self):
        content = "aaa\nbbb\naaa\n"
        result = replace(content, "aaa", "XXX", replace_all=True, line_number=1)
        assert result == "XXX\nbbb\nXXX\n"

    def test_non_first_exact_occurrence_replaced(self):
        content = "x = 1\ny = 2\nx = 1\ny = 3\n"
        result = replace(content, "x = 1", "x = 99", line_number=3)
        assert result == "x = 1\ny = 2\nx = 99\ny = 3\n"

    def test_multiple_matches_on_same_line(self):
        content = "aaa bbb aaa\nccc\n"
        with pytest.raises(ValueError, match="multiple matches on line 1"):
            replace(content, "aaa", "XXX", line_number=1)

    def test_candidate_lines_capped(self):
        content = "\n".join("aaa" for _ in range(10)) + "\n"
        with pytest.raises(ValueError, match=r"\.\.\."):
            replace(content, "aaa", "XXX", line_number=100)


class TestStaleLineAutoApply:
    """A stale line_number is ignored when old_string resolves to exactly one
    location in the whole file."""

    def test_single_match_off_by_a_few_lines_applies(self):
        content = "header\nx = 1\nfooter\n"
        result = replace(content, "x = 1", "x = 99", line_number=10)
        assert result == "header\nx = 99\nfooter\n"

    def test_single_match_off_by_hundreds_still_applies(self):
        content = "pad\npad\ntarget = 1\npad\npad\n"
        result = replace(content, "target = 1", "target = 2", line_number=999)
        assert result == "pad\npad\ntarget = 2\npad\npad\n"

    def test_multiline_single_match_stale_line_past_span_applies(self):
        content = "x\naaa\nbbb\nx\n"
        result = replace(content, "aaa\nbbb", "AAA\nBBB", line_number=99)
        assert result == "x\nAAA\nBBB\nx\n"

    def test_single_fuzzy_match_stale_line_applies(self):
        content = "header\naaa\nfooter\n"
        result = replace(content, "   aaa", "XXX", line_number=99)
        assert result == "header\nXXX\nfooter\n"

    def test_single_unicode_match_stale_line_applies(self):
        content = "header\nprint(“hello”)\nfooter\n"
        result = replace(content, 'print("hello")', 'print("world")', line_number=99)
        assert result == 'header\nprint("world")\nfooter\n'

    def test_later_fuzzy_pass_wins_over_stale_exact(self):
        content = "  aaa\nbbb\naaa\nccc\n"
        result = replace(content, "  aaa", "XXX", line_number=3)
        assert result == "  aaa\nbbb\nXXX\nccc\n"

    def test_later_unicode_pass_wins_over_stale_exact(self):
        content = 'print("hi")\nx\nprint(“hi”)\ny\n'
        result = replace(content, 'print("hi")', "PRINT", line_number=3)
        assert result == 'print("hi")\nx\nPRINT\ny\n'

    def test_two_matches_stale_line_blocks_autoapply(self):
        content = "aaa\nbbb\naaa\nccc\n"
        with pytest.raises(ValueError, match="no match at line"):
            replace(content, "aaa", "XXX", line_number=99)

    def test_two_matches_same_line_stale_line_blocks(self):
        content = "aaa bbb aaa\nccc\n"
        with pytest.raises(ValueError, match="no match at line 2"):
            replace(content, "aaa", "XXX", line_number=2)

    def test_distinct_exact_and_fuzzy_blocks_autoapply(self):
        content = "  aaa\nbbb\naaa\nccc\n"
        with pytest.raises(ValueError, match="no match at line"):
            replace(content, "  aaa", "XXX", line_number=99)

    def test_exact_wins_over_overlapping_fuzzy(self):
        content = "  target\n"
        result = replace(content, "target", "X", line_number=99)
        assert result == "  X\n"

    def test_overlapping_fuzzy_windows_block_autoapply(self):
        content = " a\n a\n a\n"
        with pytest.raises(ValueError, match="no match at line"):
            replace(content, "a\na", "X\nY", line_number=99)

    def test_exact_and_overlapping_fuzzy_window_block_autoapply(self):
        content = "a\na\n a\n"
        with pytest.raises(ValueError, match="no match at line"):
            replace(content, "a\na", "X\nY", line_number=99)

    def test_not_found_with_stale_line_unchanged(self):
        content = "aaa\nbbb\nccc\n"
        with pytest.raises(ValueError, match="not found"):
            replace(content, "zzz", "XXX", line_number=99)


class TestExactMatchSpans:
    """Tests for the _exact_match_spans helper."""

    def test_finds_all_occurrences(self):
        content = "aaa bbb aaa ccc aaa"
        spans = _exact_match_spans(content, "aaa")
        assert len(spans) == 3
        assert all(content[s:e] == "aaa" for s, e in spans)

    def test_returns_empty_for_no_match(self):
        assert _exact_match_spans("hello world", "xyz") == []

    def test_single_match(self):
        spans = _exact_match_spans("hello world", "world")
        assert len(spans) == 1
        assert spans[0] == (6, 11)

    def test_non_overlapping(self):
        spans = _exact_match_spans("aaaa", "aa")
        assert len(spans) == 2
        assert spans == [(0, 2), (2, 4)]


# =========================================================================
# Rich error feedback: closest-match enrichment when old_string is missing
# =========================================================================


class TestClosestMatchFeedback:
    """When old_string isn't found, the error names the most similar line."""

    def test_typo_single_line_includes_closest(self):
        content = "def greet(name):\n    print('hello, ' + name)\n"
        with pytest.raises(ValueError) as excinfo:
            replace(content, "    print('helo, ' + name)", "    print('hi, ' + name)")
        msg = str(excinfo.value)
        assert "old_string not found" in msg
        assert "closest line in file" in msg
        assert "line 2" in msg
        assert "    print('hello, ' + name)" in msg

    def test_whitespace_mismatch_single_line(self):
        content = "if x:\n    return x + y\n"
        with pytest.raises(ValueError) as excinfo:
            replace(content, "    return x+y", "    return x + y + 1")
        msg = str(excinfo.value)
        assert "closest line in file" in msg
        assert "line 2" in msg

    def test_multiline_window_closest(self):
        content = "def f():\n    a = 1\n    b = 2\n    return a + b\n"
        with pytest.raises(ValueError) as excinfo:
            replace(
                content,
                "def f():\n    a = 1\n    b = 3",
                "def f():\n    return 99",
            )
        msg = str(excinfo.value)
        assert "closest window in file" in msg
        assert "lines 1-3" in msg

    def test_no_close_match_omits_block(self):
        content = "hello world\n"
        with pytest.raises(ValueError) as excinfo:
            replace(content, "xyzzy plover", "wow")
        msg = str(excinfo.value)
        assert "old_string not found" in msg
        assert "No close match was found" in msg
        assert "closest" not in msg.lower().split("no close match")[0]

    def test_multiple_matches_lists_snippets(self):
        content = "aaa one\nbbb\naaa two\nccc\naaa three\n"
        with pytest.raises(ValueError) as excinfo:
            replace(content, "aaa", "XXX")
        msg = str(excinfo.value)
        assert "multiple matches (3 found)" in msg
        assert "matches at:" in msg
        assert "line 1:" in msg
        assert "line 3:" in msg
        assert "line 5:" in msg
        assert "'aaa one'" in msg

    def test_multiple_matches_caps_at_five_snippets(self):
        content = "\n".join(f"row {i} aaa" for i in range(10)) + "\n"
        with pytest.raises(ValueError) as excinfo:
            replace(content, "aaa", "X")
        msg = str(excinfo.value)
        assert "multiple matches (10 found)" in msg
        assert "and 5 more" in msg

    def test_line_target_miss_lists_alternatives(self):
        content = "alpha\nbeta\nalpha\n"
        with pytest.raises(ValueError) as excinfo:
            replace(content, "alpha", "X", line_number=2)
        msg = str(excinfo.value)
        assert "no match at line 2" in msg
        assert "1, 3" in msg
        assert "Pass one of those line numbers" in msg


# =========================================================================
# Trailing-newline boundary: old_string excludes a file \n that new_string
# adds back. Without absorption, the splice would double the newline.
# =========================================================================


class TestTrailingNewlineAbsorption:
    """The splice consumes the file's terminating \\n when the model omits
    it from old_string but puts it back in new_string."""

    def test_exact_match_at_eof_does_not_double(self):
        content = "red\norange\ngreen\nblue\npurple\n"
        result = replace(
            content,
            "red\norange\ngreen\nblue\npurple",
            "red\norange\nyellow\ngreen\nblue\npurple\n",
        )
        assert result == "red\norange\nyellow\ngreen\nblue\npurple\n"

    def test_exact_match_mid_file_does_not_double(self):
        content = "aaa\nbbb\nccc\n"
        result = replace(content, "aaa", "xxx\n")
        assert result == "xxx\nbbb\nccc\n"

    def test_replace_all_exact_does_not_double(self):
        content = "aaa\nbbb\naaa\n"
        result = replace(content, "aaa", "xxx\n", replace_all=True)
        assert result == "xxx\nbbb\nxxx\n"

    def test_fuzzy_match_does_not_double(self):
        content = "red  \ngreen  \nblue  \n"
        result = replace(content, "red\ngreen\nblue", "red\nyellow\ngreen\nblue\n")
        assert result == "red\nyellow\ngreen\nblue\n"

    def test_legitimate_mid_line_substring_unaffected(self):
        content = "hello world\nfoo\n"
        result = replace(content, "world", "WORLD")
        assert result == "hello WORLD\nfoo\n"

    def test_no_trailing_newline_in_new_string_unaffected(self):
        content = "aaa\nbbb\n"
        result = replace(content, "aaa", "xxx")
        assert result == "xxx\nbbb\n"

    def test_inserting_extra_line_still_works(self):
        content = "abc\ndef\n"
        result = replace(content, "abc", "abc\nNEW LINE\n")
        assert result == "abc\nNEW LINE\ndef\n"

    def test_multiline_old_no_trailing_newline(self):
        content = "aaa\nbbb\nccc\n"
        result = replace(content, "aaa\nbbb", "XXX\nYYY\n")
        assert result == "XXX\nYYY\nccc\n"

    def test_old_already_has_trailing_newline_unchanged(self):
        content = "aaa\nbbb\n"
        result = replace(content, "aaa\n", "xxx\n")
        assert result == "xxx\nbbb\n"

    def test_match_at_end_with_no_following_newline(self):
        content = "aaa\nbbb"
        result = replace(content, "bbb", "xxx\n")
        assert result == "aaa\nxxx\n"


class TestReplaceAllAdjacency:
    def test_adjacent_matches_keep_both_replacements(self):
        assert replace("abab", "ab", "\nX\n", replace_all=True) == "\nX\n\nX\n"

    def test_absorbed_newline_is_consumed_once(self):
        assert replace("ab\nab\n", "ab", "X\n", replace_all=True) == "X\nX\n"

    def test_fuzzy_matches_sharing_an_empty_line_apply_from_the_end(self):
        # Both line windows contain line 2, so one match starts where the
        # previous one ends. A left-to-right pass would yield "\nX\n\nX\n".
        assert replace("\n\n", " \n ", "\nX\n", replace_all=True) == "\nX\nX\n"


class TestDisplayLineCap:
    """display_line_cap refuses edits that stop at a clipped line's visible prefix."""

    CAP = 10
    ERROR = r"10-character cutoff that read_file and grep apply to line {}"

    def test_guard_is_off_by_default(self):
        assert replace("0123456789HIDDEN\n", "0123456789", "X") == "XHIDDEN\n"

    def test_visible_prefix_of_longer_line_rejected(self):
        content = "head\n0123456789HIDDEN\ntail\n"
        with pytest.raises(ValueError, match=self.ERROR.format(2)):
            replace(content, "0123456789", "X", display_line_cap=self.CAP)

    def test_multiline_span_ending_at_prefix_rejected(self):
        content = "head\n0123456789HIDDEN\n"
        with pytest.raises(ValueError, match=self.ERROR.format(2)):
            replace(content, "head\n0123456789", "X", display_line_cap=self.CAP)

    def test_line_of_exactly_cap_is_editable(self):
        content = "head\n0123456789\ntail"
        result = replace(content, "0123456789", "X", display_line_cap=self.CAP)
        assert result == "head\nX\ntail"

    def test_line_of_exactly_cap_at_eof_is_editable(self):
        result = replace("0123456789", "0123456789", "X", display_line_cap=self.CAP)
        assert result == "X"

    def test_shorter_or_longer_old_string_allowed(self):
        content = "0123456789HIDDEN\n"
        cap = self.CAP
        assert replace(content, "012345678", "X", display_line_cap=cap) == (
            "X9HIDDEN\n"
        )
        assert replace(content, "0123456789H", "X", display_line_cap=cap) == (
            "XIDDEN\n"
        )

    def test_span_not_starting_at_line_start_allowed(self):
        content = "ab0123456789HIDDEN\n"
        result = replace(content, "0123456789", "X", display_line_cap=self.CAP)
        assert result == "abXHIDDEN\n"

    def test_short_substring_in_long_line_allowed(self):
        content = "0123456789HIDDEN\n"
        result = replace(content, "345", "X", display_line_cap=self.CAP)
        assert result == "012X6789HIDDEN\n"

    def test_replace_all_checks_every_selected_span(self):
        content = "0123456789\n0123456789HIDDEN\n"
        with pytest.raises(ValueError, match=self.ERROR.format(2)):
            replace(
                content,
                "0123456789",
                "X",
                replace_all=True,
                display_line_cap=self.CAP,
            )

    def test_replace_all_without_clipped_span_allowed(self):
        content = "0123456789\n0123456789\n"
        result = replace(
            content, "0123456789", "X", replace_all=True, display_line_cap=self.CAP
        )
        assert result == "X\nX\n"

    def test_unselected_clipped_candidate_is_ignored(self):
        content = "0123456789\n0123456789HIDDEN\n"
        result = replace(
            content, "0123456789", "X", line_number=1, display_line_cap=self.CAP
        )
        assert result == "X\n0123456789HIDDEN\n"
        with pytest.raises(ValueError, match=self.ERROR.format(2)):
            replace(
                content, "0123456789", "X", line_number=2, display_line_cap=self.CAP
            )

    def test_stale_line_recovery_is_checked(self):
        content = "a\nb\n0123456789HIDDEN\n"
        # Without the guard, the stale line_number falls back to the lone match.
        result = replace(content, "0123456789", "X", line_number=1)
        assert result == "a\nb\nXHIDDEN\n"
        with pytest.raises(ValueError, match=self.ERROR.format(3)):
            replace(
                content, "0123456789", "X", line_number=1, display_line_cap=self.CAP
            )

    def test_unicode_counts_characters(self):
        content = "ab" + "é" * 8 + "日本\n"
        with pytest.raises(ValueError, match=self.ERROR.format(1)):
            replace(content, "ab" + "é" * 8, "X", display_line_cap=self.CAP)
        result = replace(content, "ab" + "é" * 7, "X", display_line_cap=self.CAP)
        assert result == "Xé日本\n"

    def test_crlf_terminator_ends_the_line(self):
        content = "0123456789\r\nnext\r\n"
        result = replace(content, "0123456789", "X", display_line_cap=self.CAP)
        assert result == "X\r\nnext\r\n"

    def test_splitlines_separators_delimit_lines(self):
        content = "0123456789\x0cnext\n"
        result = replace(content, "0123456789", "X", display_line_cap=self.CAP)
        assert result == "X\x0cnext\n"
        content = "prev\u20280123456789HIDDEN\n"
        with pytest.raises(ValueError, match=self.ERROR.format(2)):
            replace(content, "0123456789", "X", display_line_cap=self.CAP)

    def test_line_trimmed_match_covers_whole_line(self):
        content = "    0123456789HIDDEN\nnext\n"
        result = replace(
            content, "  0123456789HIDDEN  ", "X", display_line_cap=self.CAP
        )
        assert result == "X\nnext\n"


def test_line_break_pattern_matches_splitlines():
    every_char = "".join(
        chr(cp) for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF
    )
    pieces = every_char.splitlines(keepends=True)
    breaks = {piece[-1] for piece in pieces[:-1]}
    assert set(_LINE_BREAK_RE.findall(every_char)) == breaks
    assert len(breaks) == 10

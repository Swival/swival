"""String replacement engine for the edit_file tool.

Provides a single public function `replace()` that finds and replaces text
in file content using multi-pass matching: exact first, then line-trimmed,
then Unicode-normalized.
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Unicode normalization (moved from patch.py)
# ---------------------------------------------------------------------------

_UNICODE_SINGLE_QUOTES = re.compile(r"[\u2018\u2019\u201a\u201b]")
_UNICODE_DOUBLE_QUOTES = re.compile(r"[\u201c\u201d\u201e\u201f]")
_UNICODE_DASHES = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2015]")


def _normalize_unicode(s: str) -> str:
    """Normalize Unicode punctuation to ASCII equivalents."""
    s = _UNICODE_SINGLE_QUOTES.sub("'", s)
    s = _UNICODE_DOUBLE_QUOTES.sub('"', s)
    s = _UNICODE_DASHES.sub("-", s)
    s = s.replace("\u2026", "...")
    s = s.replace("\u00a0", " ")
    return s


# ---------------------------------------------------------------------------
# Line-level fuzzy matching helpers
# ---------------------------------------------------------------------------


def _find_fuzzy(
    content: str, old_string: str, normalize=None
) -> tuple[int, int] | None:
    """Find old_string in content using per-line .strip() comparison.

    When *normalize* is provided, each stripped line is also passed through
    it before comparison (e.g. ``_normalize_unicode``).

    Returns (start_index, end_index) into content, or None.
    """
    content_lines = content.split("\n")
    old_lines = old_string.split("\n")
    old_len = len(old_lines)

    if old_len == 0:
        return None

    prep = normalize or (lambda s: s)
    prepped_old = [prep(line.strip()) for line in old_lines]

    for i in range(len(content_lines) - old_len + 1):
        if all(
            prep(content_lines[i + j].strip()) == prepped_old[j] for j in range(old_len)
        ):
            start = sum(len(content_lines[k]) + 1 for k in range(i))
            end = start + sum(len(content_lines[i + k]) + 1 for k in range(old_len))
            if not old_string.endswith("\n") and end > 0 and end <= len(content) + 1:
                end -= 1
            return (start, end)

    return None


def _count_fuzzy_matches(content: str, old_string: str, normalize=None) -> int:
    """Count how many times old_string fuzzy-matches in content."""
    content_lines = content.split("\n")
    old_lines = old_string.split("\n")
    old_len = len(old_lines)
    count = 0

    prep = normalize or (lambda s: s)
    prepped_old = [prep(line.strip()) for line in old_lines]

    for i in range(len(content_lines) - old_len + 1):
        if all(
            prep(content_lines[i + j].strip()) == prepped_old[j] for j in range(old_len)
        ):
            count += 1

    return count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def replace(
    content: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """Replace old_string with new_string in content.

    Matching strategies (tried in order):
      1. Exact — str.find / str.count
      2. Line-trimmed — sliding window, comparing .strip() per line
      3. Unicode-normalized — strip + smart quotes/dashes/ellipsis → ASCII

    Raises ValueError:
      - "no changes" if old_string == new_string
      - "not found" if no match in any pass
      - "multiple matches" if >1 match and replace_all is False

    When a fuzzy pass matches, the matched text in the original content is
    replaced (preserving the original's surrounding text), and new_string
    is inserted verbatim.
    """
    if old_string == new_string:
        raise ValueError("no changes")

    if not old_string:
        raise ValueError("old_string must not be empty")

    # --- Pass 1: exact ---
    exact_count = content.count(old_string)
    if exact_count == 1 or (exact_count > 0 and replace_all):
        if exact_count > 1 and not replace_all:
            raise ValueError("multiple matches")
        return (
            content.replace(old_string, new_string)
            if replace_all
            else content.replace(old_string, new_string, 1)
        )

    if exact_count > 1 and not replace_all:
        raise ValueError("multiple matches")

    # --- Fuzzy passes: line-trimmed, then Unicode-normalized ---
    for normalize in (None, _normalize_unicode):
        match = _find_fuzzy(content, old_string, normalize=normalize)
        if match is not None:
            if not replace_all:
                count = _count_fuzzy_matches(content, old_string, normalize=normalize)
                if count > 1:
                    raise ValueError("multiple matches")
            result = content[: match[0]] + new_string + content[match[1] :]
            if replace_all:
                while True:
                    next_match = _find_fuzzy(result, old_string, normalize=normalize)
                    if next_match is None:
                        break
                    result = (
                        result[: next_match[0]] + new_string + result[next_match[1] :]
                    )
            return result

    raise ValueError("not found")

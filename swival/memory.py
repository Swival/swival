"""Memory entry parsing and retrieval for budgeted injection."""

import re
from dataclasses import dataclass, field, replace

from .tokens import count_tokens, truncate_to_tokens

# Regex for ATX headings: 1-6 '#' chars followed by a space
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")

_BOOTSTRAP_TAG = "<!-- bootstrap -->"


@dataclass
class MemoryEntry:
    id: str
    heading: str | None  # None for unheaded content at top
    content: str  # full text including heading line
    is_bootstrap: bool = False
    _tokens: int | None = field(default=None, repr=False)

    @property
    def tokens(self) -> int:
        if self._tokens is None:
            self._tokens = count_tokens(self.content)
        return self._tokens


def parse_memory(text: str) -> list[MemoryEntry]:
    """Parse MEMORY.md text into heading-delimited entries.

    Rules:
    - ATX headings (# through ######) start new entries
    - Everything under a heading belongs to that entry until next heading or EOF
    - Unheaded content at top of file is one entry
    - ``<!-- bootstrap -->`` on the line immediately before a heading marks it
    - Headings inside fenced code blocks are ignored
    """
    if not text or not text.strip():
        return []

    lines = text.splitlines(keepends=True)
    entries: list[MemoryEntry] = []
    current_lines: list[str] = []
    current_heading: str | None = None
    current_bootstrap = False
    in_fence = False
    pending_bootstrap = False

    def _flush():
        nonlocal current_lines, current_heading, current_bootstrap
        body = "".join(current_lines)
        if body.strip():
            entries.append(
                MemoryEntry(
                    id=f"m{len(entries)}",
                    heading=current_heading,
                    content=body.strip(),
                    is_bootstrap=current_bootstrap,
                )
            )
        current_lines = []
        current_heading = None
        current_bootstrap = False

    for line in lines:
        stripped = line.rstrip("\n\r")

        # Track fenced code blocks
        if stripped.startswith("```"):
            in_fence = not in_fence

        if in_fence:
            current_lines.append(line)
            pending_bootstrap = False
            continue

        # Check for bootstrap tag
        if stripped.strip() == _BOOTSTRAP_TAG:
            pending_bootstrap = True
            # Don't add the tag line to content — it's metadata
            continue

        # Check for heading
        m = _HEADING_RE.match(stripped)
        if m:
            _flush()
            current_heading = m.group(2).strip()
            current_bootstrap = pending_bootstrap
            pending_bootstrap = False
            current_lines.append(line)
            continue

        pending_bootstrap = False
        current_lines.append(line)

    _flush()
    return entries


def retrieve_bm25(
    query: str,
    entries: list[MemoryEntry],
    *,
    top_k: int = 3,
    token_budget: int = 400,
) -> list[tuple[MemoryEntry, float]]:
    """Rank entries by BM25 relevance to *query*, return top-k within budget.

    Returns list of (entry, score) tuples, highest score first.
    Entries are truncated if they exceed the remaining token budget.
    """
    if not query or not entries:
        return []

    from rank_bm25 import BM25Okapi

    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    corpus = [_tokenize(e.content) for e in entries]
    # BM25Okapi requires non-empty corpus items; replace empties with placeholder
    corpus = [doc if doc else [""] for doc in corpus]
    bm25 = BM25Okapi(corpus)
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    # Filter to entries with actual term overlap before ranking.
    # BM25 IDF can be negative with small corpora, so score sign is unreliable.
    query_set = set(query_tokens)
    scored = [
        (entry, score)
        for entry, doc, score in zip(entries, corpus, scores)
        if query_set & set(doc)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    results: list[tuple[MemoryEntry, float]] = []
    remaining = token_budget
    for entry, score in scored:
        if len(results) >= top_k:
            break
        if remaining <= 0:
            break
        if entry.tokens > remaining:
            # Like the no-query packer: truncate the last entry if enough
            # budget remains, then stop.
            if remaining > 20:
                truncated = replace(
                    entry,
                    content=truncate_to_tokens(entry.content, remaining),
                    _tokens=None,
                )
                results.append((truncated, float(score)))
            break
        results.append((entry, float(score)))
        remaining -= entry.tokens

    return results

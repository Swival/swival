"""Protected regions inside the system message.

Instruction files are mandatory rules. Compaction may shrink everything around
them, but it must never slice through one, and a refreshed ``AGENTS.md`` must
replace the exact text the previous load put there.

Assembly records where each region landed. Every later edit goes through
:func:`splice`, which rebases the recorded offsets. Nothing here searches the
message for a marker, because the text being searched is the user's own file
and a file may contain any marker we could invent.

The metadata rides on the message dict under a key starting with an underscore,
so ``copy`` and ``deepcopy`` carry it through compaction, retries and rollback,
and ``call_llm`` strips it before the provider sees the payload.

Offsets over one flat string are a first implementation, not the end state. The
alternative is to keep the ordered parts on the message and render the content
from them, which holds each region's identity structurally instead of
coordinating it through :func:`find_span` at every editing site.

What says it is time to switch is not how many editors there are but what one
of them needs. The two today, the snapshot suffix in ``_run_agent_loop`` and
the instruction block in ``refresh_system_instructions``, do the same thing:
each replaces its own region and lets :func:`splice` rebase the other's. That
is one protocol applied uniformly, and it holds however many editors share it.

A parts model earns its keep when an editor needs something offsets cannot
express for a region it does not own: reordering, replacing by identity across
a re-render, or changing two regions atomically. The first editor whose
correctness depends on which *other* regions exist is the signal. A third
editor of the same shape as these two is not.
"""

from __future__ import annotations

from ._msg import _msg_get

SPAN_KEY = "_swival_spans"

# Distinguishes "no metadata" from a message that carries an empty list.
_UNTRACKED = object()

KIND_INSTRUCTIONS = "instructions"
KIND_SNAPSHOT = "snapshot"

# Regions compaction may shrink around but never through. Membership follows
# from what a region is, so no record carries a flag that could disagree.
_PROTECTED_KINDS = frozenset({KIND_INSTRUCTIONS})


def is_protected(span: dict) -> bool:
    """Whether compaction must cut around this region rather than through it."""
    return span["kind"] in _PROTECTED_KINDS


def make_span(kind: str, start: int, text: str) -> dict:
    """Build one span record covering ``text`` at ``start``."""
    return {
        "kind": kind,
        "start": start,
        "end": start + len(text),
        "text": text,
    }


def get_spans(msg) -> list[dict]:
    """Return the span records on *msg*, or an empty list."""
    spans = _msg_get(msg, SPAN_KEY)
    return spans if isinstance(spans, list) else []


def set_spans(msg, spans: list[dict]) -> None:
    """Attach *spans* to *msg*.

    The key is written even for an empty list. Its presence is what marks a
    message as one we assembled and can still edit by offset.
    """
    if not isinstance(msg, dict):
        return
    msg[SPAN_KEY] = list(spans)


def is_tracked(msg) -> bool:
    """True when this message carries assembly metadata we own."""
    return _msg_get(msg, SPAN_KEY, _UNTRACKED) is not _UNTRACKED


def find_span(spans: list[dict], kind: str) -> dict | None:
    for span in spans:
        if span["kind"] == kind:
            return span
    return None


def spans_valid(content: str, spans: list[dict]) -> bool:
    """Check that every recorded offset still holds its recorded text.

    A stale record is not repairable by searching. Callers rebuild from the
    assembly data they own, or they report that they could not.
    """
    for span in spans:
        start, end = span["start"], span["end"]
        if start < 0 or end > len(content) or content[start:end] != span["text"]:
            return False
    return True


def segments(content: str, spans: list[dict]) -> list[tuple[dict | None, str]]:
    """Split *content* into ``(span, text)`` pairs in offset order.

    Gaps between spans come back with ``None`` as the span.
    """
    ordered = sorted(spans, key=lambda s: s["start"])
    out: list[tuple[dict | None, str]] = []
    pos = 0
    for span in ordered:
        if span["start"] > pos:
            out.append((None, content[pos : span["start"]]))
        out.append((span, content[span["start"] : span["end"]]))
        pos = span["end"]
    if pos < len(content):
        out.append((None, content[pos:]))
    return out


def _rebuild(parts: list[tuple[dict | None, str]]) -> tuple[str, list[dict]]:
    """Reassemble ``(span, text)`` pairs and recompute every offset."""
    pieces: list[str] = []
    spans: list[dict] = []
    pos = 0
    for span, text in parts:
        if span is not None:
            spans.append(
                {
                    "kind": span["kind"],
                    "start": pos,
                    "end": pos + len(text),
                    "text": text,
                }
            )
        pieces.append(text)
        pos += len(text)
    return "".join(pieces), spans


def splice(
    content: str,
    spans: list[dict],
    start: int,
    end: int,
    replacement: str,
    *,
    target: dict | None = None,
) -> tuple[str, list[dict]]:
    """Replace ``content[start:end]`` and rebase the recorded offsets.

    The span being replaced keeps its identity and takes the new text; an empty
    replacement drops it. Pass *target* to say which span that is. Without it,
    a span whose range matches exactly is assumed to be the one, which is
    ambiguous when two empty spans sit at the same offset.

    A span that merely overlaps the replaced range is dropped. Its content is
    no longer what was recorded, and no honest offset describes it.
    """
    delta = len(replacement) - (end - start)
    out: list[dict] = []
    for span in spans:
        replaced = (
            span is target
            if target is not None
            else (span["start"] == start and span["end"] == end)
        )
        if replaced:
            if replacement:
                out.append(make_span(span["kind"], start, replacement))
            continue
        if span["end"] <= start:
            out.append(dict(span))
        elif span["start"] >= end:
            shifted = dict(span)
            shifted["start"] += delta
            shifted["end"] += delta
            out.append(shifted)
    out.sort(key=lambda s: s["start"])
    return content[:start] + replacement + content[end:], out


def shift(spans: list[dict], delta: int) -> list[dict]:
    """Move every recorded region by *delta*, for text prepended before them."""
    moved = []
    for span in spans:
        copy = dict(span)
        copy["start"] += delta
        copy["end"] += delta
        moved.append(copy)
    return moved


def map_segments(content: str, spans: list[dict], fn) -> tuple[str, list[dict]]:
    """Apply *fn* to every segment and rebase the offsets.

    Used by the outbound special-token escape, which changes the length of the
    text it rewrites. Escaping each segment on its own keeps the recorded
    regions aligned with what the escape produced.
    """
    return _rebuild([(span, fn(text)) for span, text in segments(content, spans)])


def truncate_outside_spans(
    content: str,
    spans: list[dict],
    limit: int,
) -> tuple[str, list[dict]]:
    """Shrink *content* toward *limit* characters without cutting a protected span.

    The free text is trimmed in proportion to its size. When the protected
    spans alone exceed the limit, the result is those spans plus nothing else:
    the caller asked for the smallest honest message, and mandatory rules are
    the floor.
    """
    parts = segments(content, spans)
    protected_len = sum(len(t) for s, t in parts if s is not None and is_protected(s))
    free_len = len(content) - protected_len
    allowance = max(0, limit - protected_len)
    if free_len <= allowance:
        return content, spans

    kept: list[tuple[dict | None, str]] = []
    for span, text in parts:
        if span is not None and is_protected(span):
            kept.append((span, text))
            continue
        keep = (len(text) * allowance) // free_len if free_len else 0
        kept.append((span, text[:keep]))
    return _rebuild(kept)

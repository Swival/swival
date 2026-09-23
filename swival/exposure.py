"""Replay-weighted token exposure for benchmark reports.

Every request resends whatever the transcript still holds, so a tool result
costs its size times the number of requests that carry it.
The meter attributes the content of each sent request to its origin: tool
results by originating tool, replayed tool arguments, compaction and snapshot
recaps, system text, tool schemas, and the rest of the conversation.
Attribution uses tool-call IDs and never touches provider-visible messages.

Exposure ranks where input goes; it is not a bill.
Provider tokenizers, cache discounts, cache writes and generated output all
change what a request costs, so reports carry the provider's own usage and
LiteLLM pricing next to it.
"""

import itertools
import json
import threading
from collections import Counter, OrderedDict

from ._msg import (
    COMPACTION_MARKER,
    RECAP_MARKER,
    _is_synthetic,
    _msg_get,
    _msg_name,
    _msg_role,
    _msg_tool_call_id,
    _msg_tool_calls,
    _tool_call_id,
)
from .snapshot import SNAPSHOT_RECAP_PREFIX
from .tokens import count_tokens, encoder_is_fallback

# Short strings are cheaper to tokenize again than to keep in the cache.
_CACHE_MIN_CHARS = 256
_CACHE_MAX_CHARS = 32 * 1024 * 1024

PROVIDER_RETRY_NOTE = (
    "retries inside the provider SDK are invisible to Swival and not counted"
)

# Internal key naming each tool result, so distinct results are counted once
# however the transcript around them is compacted, dropped or copied.
RESULT_KEY = "_swival_result_id"
_result_ids = itertools.count(1)


def tag_results(messages: list) -> list[str | None]:
    """Name every tool result of a conversation, returning the names by position.

    Tags live on the conversation's own message dicts, so call this before
    anything copies them for sending.
    """
    keys: list[str | None] = []
    for m in messages:
        key = None
        if isinstance(m, dict) and m.get("role") == "tool":
            key = m.get(RESULT_KEY)
            if key is None:
                key = m[RESULT_KEY] = f"r{next(_result_ids)}"
        keys.append(key)
    return keys


def rekey_results(before: list, keys: list, after: list) -> list[str | None]:
    """Carry result names across a rewrite that may reorder messages.

    Results sharing a call ID are matched in order, which a filter that passes
    messages through or redacts them preserves; one that reorders them can
    only swap names among those results, never merge or duplicate them.
    An ID whose number of results changed falls back to :func:`classify`.
    """
    names: dict[str, list[str]] = {}
    for m, key in zip(before, keys):
        tc_id = _msg_tool_call_id(m)
        if key is not None and tc_id:
            names.setdefault(tc_id, []).append(key)
    counts = Counter(_msg_tool_call_id(m) for m in after if _msg_role(m) == "tool")
    taken: Counter = Counter()
    out: list[str | None] = []
    for m in after:
        key = None
        tc_id = _msg_tool_call_id(m) if _msg_role(m) == "tool" else None
        if tc_id and counts[tc_id] == len(names.get(tc_id, ())):
            key = names[tc_id][taken[tc_id]]
            taken[tc_id] += 1
        out.append(key)
    return out


class _TokenCache:
    """LRU of token counts keyed by the string itself.

    A changed message is a different string, so edits and compaction
    invalidate their entries without any bookkeeping.
    """

    def __init__(self, max_chars: int = _CACHE_MAX_CHARS):
        self._entries: OrderedDict[str, int] = OrderedDict()
        self._chars = 0
        self._max_chars = max_chars
        self._lock = threading.Lock()

    def count(self, text: str) -> int:
        if not text:
            return 0
        if len(text) < _CACHE_MIN_CHARS:
            return count_tokens(text)
        with self._lock:
            n = self._entries.get(text)
            if n is not None:
                self._entries.move_to_end(text)
                return n
        n = count_tokens(text)
        with self._lock:
            if text not in self._entries:
                self._entries[text] = n
                self._chars += len(text)
                while self._chars > self._max_chars and self._entries:
                    old, _ = self._entries.popitem(last=False)
                    self._chars -= len(old)
        return n


def _tool_call_fields(tc) -> tuple[str, str]:
    fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
    if fn is None:
        return "", ""
    if isinstance(fn, dict):
        return fn.get("name") or "", fn.get("arguments") or ""
    return getattr(fn, "name", "") or "", getattr(fn, "arguments", "") or ""


def _content_parts(msg) -> tuple[str, int]:
    """Return a message's text and its number of image parts."""
    raw = _msg_get(msg, "content")
    if isinstance(raw, list):
        texts = []
        images = 0
        for part in raw:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                texts.append(part.get("text") or "")
            elif part.get("type") == "image_url":
                images += 1
        return "\n".join(texts), images
    return (raw if isinstance(raw, str) else ""), 0


def _schema_group(schema) -> tuple[str, str]:
    name = ""
    if isinstance(schema, dict):
        name = (schema.get("function") or {}).get("name") or ""
    if name.startswith("mcp__"):
        return "mcp", name
    if name.startswith("a2a__"):
        return "a2a", name
    return "builtin", name


def classify(
    messages: list,
    synthetic: list[bool] | None = None,
    *,
    transcript: bool = False,
    result_keys: list[str | None] | None = None,
) -> list[tuple]:
    """Split a request into ``(bucket, key, text, result_id)`` items.

    *synthetic* and *result_keys* carry the ``_swival_synthetic`` flags and
    result names from before the outbound pipeline stripped internal keys;
    each is ignored when the message count changed in between.

    A tool result belongs to the latest earlier call with its ID. Some local
    servers number calls ``call_0``, ``call_1`` in every response, so without
    a result name the fallback *result_id* adds how many calls with that ID
    came before and the result's rank after them; that is only stable while
    no earlier call is dropped.

    *transcript* measures what the command provider's plain-text transcript
    carries: no structured tool calls or reasoning, and ``<swival:call>``
    blocks in assistant text as the replayed tool arguments.
    """
    if transcript:
        from .agent import _ATTR_RE, _SWIVAL_BLOCK_RE

    if synthetic is not None and len(synthetic) != len(messages):
        synthetic = None
    if result_keys is not None and len(result_keys) != len(messages):
        result_keys = None
    names: dict[str, str] = {}
    issued: Counter = Counter()
    ranks: Counter = Counter()

    items: list[tuple] = []
    for i, m in enumerate(messages):
        role = _msg_role(m)
        text, images = _content_parts(m)
        if images:
            items.append(("images", None, images, None))
        if role == "system":
            items.append(("system", None, text, None))
        elif role == "tool":
            tc_id = _msg_tool_call_id(m)
            name = (names.get(tc_id) if tc_id else None) or _msg_name(m)
            result_id = (
                result_keys[i] if result_keys is not None else _msg_get(m, RESULT_KEY)
            )
            if result_id is None and tc_id:
                result_id = f"{tc_id}#{issued[tc_id]}.{ranks[tc_id]}"
            if tc_id:
                ranks[tc_id] += 1
            if name:
                items.append(("tool_results", name, text, result_id))
            else:
                items.append(("unattributed", None, text, None))
        elif text.startswith((RECAP_MARKER, COMPACTION_MARKER)):
            items.append(("summaries", "compaction", text, None))
        elif role == "assistant" and text.startswith(SNAPSHOT_RECAP_PREFIX):
            items.append(("summaries", "snapshot", text, None))
        elif role == "user":
            flagged = synthetic[i] if synthetic is not None else False
            bucket = "scaffolding" if flagged or _is_synthetic(text) else "user"
            items.append((bucket, None, text, None))
        elif role == "assistant" and transcript:
            rest = text
            for block in _SWIVAL_BLOCK_RE.finditer(text):
                attrs = dict(_ATTR_RE.findall(block.group(1)))
                name, tc_id = attrs.get("name") or "unknown", attrs.get("id")
                if tc_id:
                    names[tc_id] = name
                    issued[tc_id] += 1
                    ranks[tc_id] = 0
                items.append(("tool_arguments", name, block.group(2), None))
                rest = rest.replace(block.group(0), "", 1)
            items.append(("assistant", None, rest, None))
        elif role == "assistant":
            items.append(("assistant", None, text, None))
        else:
            items.append(("unattributed", None, text, None))
        if role == "assistant":
            for tc in _msg_tool_calls(m) or ():
                name, args = _tool_call_fields(tc)
                tc_id = _tool_call_id(tc)
                if tc_id:
                    names[tc_id] = name
                    issued[tc_id] += 1
                    ranks[tc_id] = 0
                if transcript:
                    continue
                if not isinstance(args, str):
                    args = json.dumps(args)
                items.append(("tool_arguments", name or "unknown", args, None))
            reasoning = _msg_get(m, "reasoning_content")
            if reasoning and not transcript:
                items.append(("reasoning", None, str(reasoning), None))
    return items


class _SourceStats:
    def __init__(self):
        self.sent = 0
        self.resent = 0
        self.cache_hits = 0
        self.responses = 0
        self.input: Counter = Counter()
        self.tool_results: Counter = Counter()
        self.tool_arguments: Counter = Counter()
        self.schemas: Counter = Counter()
        self.summaries: Counter = Counter()
        self.images = 0
        self.seen_results: set[str] = set()
        self.result_count: Counter = Counter()
        self.result_tokens: Counter = Counter()
        self.generated_calls: Counter = Counter()
        self.generated_tokens: Counter = Counter()
        self.usage_reported = 0
        self.usage: Counter = Counter()
        self.known_usd = 0.0
        self.priced = 0
        self.unpriced = 0
        self.not_applicable = 0

    def merge(self, other: "_SourceStats") -> None:
        for name in (
            "sent",
            "resent",
            "cache_hits",
            "responses",
            "images",
            "usage_reported",
            "known_usd",
            "priced",
            "unpriced",
            "not_applicable",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for name in (
            "input",
            "tool_results",
            "tool_arguments",
            "schemas",
            "summaries",
            "result_count",
            "result_tokens",
            "generated_calls",
            "generated_tokens",
            "usage",
        ):
            getattr(self, name).update(getattr(other, name))

    def to_dict(self) -> dict:
        input_total = (
            sum(self.input.values())
            + sum(self.tool_results.values())
            + sum(self.tool_arguments.values())
            + sum(self.schemas.values())
            + sum(self.summaries.values())
        )
        usage = None
        if self.usage_reported:
            usage = {
                "input_tokens": self.usage["prompt_tokens"],
                "output_tokens": self.usage["completion_tokens"],
                "cached_tokens": self.usage["cached_tokens"],
                "cache_write_tokens": self.usage["cache_write_tokens"],
            }
        return {
            "requests": {
                "sent": self.sent,
                "resent": self.resent,
                "failed": max(0, self.sent - self.responses),
                "response_cache_hits": self.cache_hits,
            },
            "input": {
                "total": input_total,
                "system": self.input["system"],
                "tool_schemas": dict(self.schemas),
                "tool_results": dict(self.tool_results.most_common()),
                "tool_arguments": dict(self.tool_arguments.most_common()),
                "summaries": dict(self.summaries),
                "user": self.input["user"],
                "assistant": self.input["assistant"],
                "reasoning": self.input["reasoning"],
                "scaffolding": self.input["scaffolding"],
                "unattributed": self.input["unattributed"],
                "image_parts": self.images,
            },
            "results": {
                name: {"count": count, "tokens": self.result_tokens[name]}
                for name, count in self.result_count.most_common()
            },
            "output": {
                "tool_arguments": {
                    name: {"calls": calls, "tokens": self.generated_tokens[name]}
                    for name, calls in self.generated_calls.most_common()
                },
            },
            "provider_usage": {
                "responses": self.responses,
                "reported": self.usage_reported,
                "usage": usage,
            },
            "cost": {
                "known_usd": self.known_usd if self.priced else None,
                "priced_calls": self.priced,
                "unpriced_calls": self.unpriced,
                "not_applicable_calls": self.not_applicable,
            },
        }


class ExposureMeter:
    """Thread-safe accumulator shared by a run, its subagents and summaries."""

    def __init__(self):
        self._lock = threading.Lock()
        self._sources: dict[str, _SourceStats] = {}
        self._tokens = _TokenCache()
        self._provider_requests = False

    def recorder(self, source: str) -> "ExposureRecorder":
        return ExposureRecorder(self, source)

    def _stats(self, source: str) -> _SourceStats:
        stats = self._sources.get(source)
        if stats is None:
            stats = self._sources[source] = _SourceStats()
        return stats

    def count(self, text: str) -> int:
        return self._tokens.count(text)

    def to_dict(self) -> dict | None:
        with self._lock:
            if not self._sources:
                return None
            total = _SourceStats()
            by_source = {}
            for name in sorted(self._sources):
                total.merge(self._sources[name])
                by_source[name] = self._sources[name].to_dict()
            coverage = []
            if self._provider_requests:
                coverage.append(PROVIDER_RETRY_NOTE)
        unit = "bytes" if encoder_is_fallback() else "tokens"
        if unit == "bytes":
            coverage.append(
                "tokenizer data was unavailable, so exposure counts UTF-8 bytes"
            )
        return {
            "unit": unit,
            "coverage": coverage,
            **total.to_dict(),
            "by_source": by_source,
        }


class ExposureRecorder:
    """An :class:`ExposureMeter` view that files everything under one source."""

    def __init__(self, meter: ExposureMeter, source: str):
        self.meter = meter
        self.source = source

    def request(
        self,
        messages: list,
        tools: list | None = None,
        *,
        synthetic: list[bool] | None = None,
        resend: bool = False,
        provider: bool = True,
        transcript: bool = False,
        result_keys: list[str | None] | None = None,
    ) -> None:
        """Record one request as it is sent, retries included."""
        count = self.meter.count
        buckets: Counter = Counter()
        results: Counter = Counter()
        arguments: Counter = Counter()
        summaries: Counter = Counter()
        schemas: Counter = Counter()
        firsts: list[tuple[str, str, int]] = []
        images = 0
        for bucket, key, text, tc_id in classify(
            messages, synthetic, transcript=transcript, result_keys=result_keys
        ):
            if bucket == "images":
                images += text
                continue
            n = count(text)
            if bucket == "tool_results":
                results[key] += n
                if tc_id:
                    firsts.append((tc_id, key, n))
            elif bucket == "tool_arguments":
                arguments[key] += n
            elif bucket == "summaries":
                summaries[key] += n
            else:
                buckets[bucket] += n
        for schema in tools or ():
            group, _name = _schema_group(schema)
            schemas[group] += count(json.dumps(schema))

        with self.meter._lock:
            if provider:
                self.meter._provider_requests = True
            stats = self.meter._stats(self.source)
            stats.sent += 1
            if resend:
                stats.resent += 1
            stats.input.update(buckets)
            stats.tool_results.update(results)
            stats.tool_arguments.update(arguments)
            stats.summaries.update(summaries)
            stats.schemas.update(schemas)
            stats.images += images
            for tc_id, name, n in firsts:
                if tc_id not in stats.seen_results:
                    stats.seen_results.add(tc_id)
                    stats.result_count[name] += 1
                    stats.result_tokens[name] += n

    def cache_hit(self) -> None:
        """Record a response served from the local cache instead of a provider."""
        with self.meter._lock:
            self.meter._stats(self.source).cache_hits += 1

    def response(self, usage=None, cost=None) -> None:
        """Record a response actually served, with its usage and price."""
        reported = None
        # Every real request has input, so an all-zero record is a placeholder
        # from a server that does not count, not a measured zero.
        if usage:
            details = _msg_get(usage, "prompt_tokens_details")
            reported = {
                "prompt_tokens": _msg_get(usage, "prompt_tokens") or 0,
                "completion_tokens": _msg_get(usage, "completion_tokens") or 0,
                "cached_tokens": (_msg_get(details, "cached_tokens") if details else 0)
                or 0,
                "cache_write_tokens": _msg_get(usage, "cache_creation_input_tokens")
                or 0,
            }
            if not (reported["prompt_tokens"] or reported["completion_tokens"]):
                reported = None

        with self.meter._lock:
            stats = self.meter._stats(self.source)
            stats.responses += 1
            if reported is not None:
                stats.usage_reported += 1
                stats.usage.update(reported)
            if cost is not None:
                if cost.status == "known":
                    stats.known_usd += cost.usd
                    stats.priced += 1
                elif cost.status == "unavailable":
                    stats.unpriced += 1
                else:
                    stats.not_applicable += 1

    def generated(self, tool_calls) -> None:
        """Record the tool arguments a response generated, as output."""
        counts: list[tuple[str, int]] = []
        for tc in tool_calls or ():
            name, args = _tool_call_fields(tc)
            if not isinstance(args, str):
                args = json.dumps(args)
            counts.append((name or "unknown", self.meter.count(args)))
        if not counts:
            return
        with self.meter._lock:
            stats = self.meter._stats(self.source)
            for name, n in counts:
                stats.generated_calls[name] += 1
                stats.generated_tokens[name] += n

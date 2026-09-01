"""Structural recovery for tool-call payloads.

Sits in front of :mod:`swival.repair` (which does schema-aware argument
repair).  Each entry point handles one structural failure mode that
shows up in practice with weak or quirky models:

* :func:`repair_truncated_json` — close unbalanced JSON when the model
  hit ``max_tokens`` mid-structure.
* :func:`scavenge_tool_calls` — recover tool calls the model emitted as
  JSON or ``<swival:call>`` blocks in the content channel.
* :class:`StormBreaker` — pre-dispatch suppression of identical
  successful-but-looping repeat calls.
* :func:`analyze_schema`, :func:`flatten_schema`, :func:`nest_arguments`
  — flatten deeply nested schemas to dot-paths so models that drop
  nested args still produce usable calls.

These are kept separate from ``repair.py`` because they operate on the
*envelope* of a tool call, not on already-parsed arguments.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


MAX_SCAVENGE_INPUT = 100 * 1024
ORIGINAL_PREVIEW_BYTES = 200


def _byte_capped_preview(text: str, limit: int = ORIGINAL_PREVIEW_BYTES) -> str:
    """Return *text* truncated so its UTF-8 encoding is at most *limit* bytes."""
    if not text:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    return encoded[:limit].decode("utf-8", errors="ignore")


@dataclass
class TruncationResult:
    repaired: str
    changed: bool
    fallback: bool
    notes: list[str] = field(default_factory=list)
    original_length: int = 0
    repaired_length: int = 0
    original_preview: str | None = None


def repair_truncated_json(text: str) -> TruncationResult:
    """Close an unbalanced JSON object/array so it can be parsed.

    The repair only walks the input once tracking string and escape state
    plus a brace/bracket/quote stack.  It then trims trailing commas,
    fills a dangling ``"key":`` with ``null``, closes an open string, and
    pops the remaining open structures in reverse order.

    Whitespace-only or empty input is treated as ``"{}"`` structurally
    (downstream ``validate_required_args`` still rejects calls that
    actually need arguments).
    """
    if text is None:
        text = ""
    original_length = len(text)

    if not text or not text.strip():
        return TruncationResult(
            repaired="{}",
            changed=text != "{}",
            fallback=False,
            notes=["empty input -> {}"],
            original_length=original_length,
            repaired_length=2,
            original_preview=None,
        )

    try:
        json.loads(text)
        return TruncationResult(
            repaired=text,
            changed=False,
            fallback=False,
            notes=[],
            original_length=original_length,
            repaired_length=original_length,
            original_preview=None,
        )
    except (json.JSONDecodeError, TypeError):
        pass

    stack: list[str] = []
    escaped = False
    in_string = False
    last_significant = -1
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if not c.isspace():
            last_significant = i
        if escaped:
            escaped = False
            i += 1
            continue
        if in_string:
            if c == "\\":
                escaped = True
                i += 1
                continue
            if c == '"':
                in_string = False
                if stack and stack[-1] == '"':
                    stack.pop()
            i += 1
            continue
        if c == '"':
            in_string = True
            stack.append('"')
        elif c in "{[":
            stack.append(c)
        elif c in "}]":
            if stack and stack[-1] in "{[":
                stack.pop()
        i += 1

    notes: list[str] = []
    s = text[: last_significant + 1] if last_significant >= 0 else ""

    if not in_string and s.endswith(","):
        s = s[:-1]
        notes.append("trimmed trailing comma")

    if not in_string and re.search(r'"\s*:\s*$', s):
        s += " null"
        notes.append("filled dangling key with null")

    if in_string:
        # A dangling escape would turn the closing quote into \".
        trailing = len(s) - len(s.rstrip("\\"))
        if trailing % 2 == 1:
            s = s[:-1]
            notes.append("dropped dangling escape")
        s += '"'
        if stack and stack[-1] == '"':
            stack.pop()
        notes.append("closed unterminated string")

    while stack:
        top = stack.pop()
        if top == "{":
            s += "}"
        elif top == "[":
            s += "]"
        elif top == '"':
            s += '"'

    try:
        json.loads(s)
    except (json.JSONDecodeError, TypeError) as err:
        preview = _byte_capped_preview(text)
        notes.append(f"fallback to {{}}: {err}")
        return TruncationResult(
            repaired="{}",
            changed=True,
            fallback=True,
            notes=notes,
            original_length=original_length,
            repaired_length=2,
            original_preview=preview,
        )

    if s == text:
        return TruncationResult(
            repaired=text,
            changed=False,
            fallback=False,
            notes=notes,
            original_length=original_length,
            repaired_length=original_length,
            original_preview=None,
        )

    return TruncationResult(
        repaired=s,
        changed=True,
        fallback=False,
        notes=notes,
        original_length=original_length,
        repaired_length=len(s),
        original_preview=_byte_capped_preview(text),
    )


@dataclass
class ScavengedCall:
    name: str
    arguments: dict[str, Any]
    source: str
    raw: str = ""


@dataclass
class ScavengeResult:
    calls: list[ScavengedCall] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


_SWIVAL_CALL_RE = re.compile(
    r"<swival:call(\s[^>]*?)?>\s*(\{.*?\})\s*</swival:call>",
    re.DOTALL,
)
_SWIVAL_NAME_RE = re.compile(r'(?:^|\s)name\s*=\s*"([^"]+)"')


def scavenge_tool_calls(
    content: str | None,
    reasoning: str | None,
    allowed_names: set[str] | frozenset[str],
    *,
    max_calls: int = 4,
) -> ScavengeResult:
    """Recover tool calls leaked into the message body.

    Walks both the assistant ``content`` and any provider
    ``reasoning_content``, recognizing:

    * ``{"name": ..., "arguments": ...}``
    * ``{"type": "function", "function": {"name": ..., "arguments": ...}}``
    * ``{"tool_name": ..., "tool_args": ...}``
    * ``<swival:call name="...">{...}</swival:call>`` envelopes.

    Candidates are filtered by ``allowed_names``, deduplicated by
    canonical ``(name, args)`` signature, and capped at ``max_calls``.
    Input larger than ``MAX_SCAVENGE_INPUT`` short-circuits to avoid
    polynomial-time scanning on adversarial payloads.
    """
    result = ScavengeResult()
    blobs: list[tuple[str, str]] = []
    if content:
        blobs.append(("content", content))
    if reasoning:
        blobs.append(("reasoning", reasoning))
    if not blobs:
        return result

    for source, text in blobs:
        if len(text) > MAX_SCAVENGE_INPUT:
            result.notes.append(
                f"scavenge skipped: {source} too large ({len(text)} chars)"
            )
            continue
        _scavenge_swival_calls(text, source, allowed_names, result, max_calls)
        if len(result.calls) >= max_calls:
            break
        stripped = _strip_swival_blocks(text)
        _scavenge_json_objects(stripped, source, allowed_names, result, max_calls)
        if len(result.calls) >= max_calls:
            break
    return result


def _strip_swival_blocks(text: str) -> str:
    return _SWIVAL_CALL_RE.sub("", text)


def _scavenge_swival_calls(
    text: str,
    source: str,
    allowed_names: set[str] | frozenset[str],
    result: ScavengeResult,
    max_calls: int,
) -> None:
    for match in _SWIVAL_CALL_RE.finditer(text):
        if len(result.calls) >= max_calls:
            return
        attrs = match.group(1) or ""
        name_match = _SWIVAL_NAME_RE.search(attrs)
        if not name_match:
            result.notes.append("swival:call envelope missing name attribute")
            continue
        name = name_match.group(1)
        if name not in allowed_names:
            result.notes.append(f"swival:call name {name!r} not in allowed_names")
            continue
        body = match.group(2)
        try:
            args = json.loads(body)
        except (json.JSONDecodeError, TypeError) as err:
            result.notes.append(f"swival:call body parse failed: {err}")
            continue
        normalized = _normalize_arguments(args)
        if normalized is None:
            result.notes.append(f"swival:call {name!r}: argument shape rejected")
            continue
        if _is_duplicate(result.calls, name, normalized):
            continue
        result.calls.append(
            ScavengedCall(
                name=name,
                arguments=normalized,
                source=f"{source}/swival",
                raw=match.group(0),
            )
        )
        result.notes.append(f"scavenged swival:call: {name}")


def _scavenge_json_objects(
    text: str,
    source: str,
    allowed_names: set[str] | frozenset[str],
    result: ScavengeResult,
    max_calls: int,
) -> None:
    for blob in _iterate_top_level_json_objects(text):
        if len(result.calls) >= max_calls:
            return
        try:
            parsed = json.loads(blob)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(parsed, dict):
            continue
        candidate = _coerce_candidate(parsed, allowed_names)
        if candidate is None:
            continue
        name, raw_args = candidate
        normalized = _normalize_arguments(raw_args)
        if normalized is None:
            result.notes.append(f"scavenge {name!r}: argument shape rejected")
            continue
        if _is_duplicate(result.calls, name, normalized):
            continue
        result.calls.append(
            ScavengedCall(
                name=name,
                arguments=normalized,
                source=f"{source}/json",
                raw=blob,
            )
        )
        result.notes.append(f"scavenged json call: {name}")


def _iterate_top_level_json_objects(text: str):
    n = len(text)
    i = 0
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_string = False
        escaped = False
        j = i
        matched = False
        while j < n:
            c = text[j]
            if escaped:
                escaped = False
                j += 1
                continue
            if in_string:
                if c == "\\":
                    escaped = True
                    j += 1
                    continue
                if c == '"':
                    in_string = False
                j += 1
                continue
            if c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    yield text[i : j + 1]
                    i = j + 1
                    matched = True
                    break
            j += 1
        if not matched:
            i += 1


def _coerce_candidate(
    parsed: dict[str, Any],
    allowed_names: set[str] | frozenset[str],
) -> tuple[str, Any] | None:
    name = parsed.get("name")
    if isinstance(name, str) and name in allowed_names and "arguments" in parsed:
        return name, parsed.get("arguments")
    if parsed.get("type") == "function" and isinstance(parsed.get("function"), dict):
        fn = parsed["function"]
        fname = fn.get("name")
        if isinstance(fname, str) and fname in allowed_names:
            return fname, fn.get("arguments")
    tn = parsed.get("tool_name")
    if isinstance(tn, str) and tn in allowed_names:
        return tn, parsed.get("tool_args")
    return None


def _normalize_arguments(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            repair = repair_truncated_json(raw)
            if repair.fallback:
                return None
            try:
                parsed = json.loads(repair.repaired)
            except (json.JSONDecodeError, TypeError):
                return None
        if isinstance(parsed, dict):
            return parsed
        return None
    return None


def _canonical_args(args: dict[str, Any]) -> str:
    try:
        return json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return repr(args)


def _is_duplicate(
    existing: list[ScavengedCall], name: str, args: dict[str, Any]
) -> bool:
    sig = _canonical_args(args)
    return any(c.name == name and _canonical_args(c.arguments) == sig for c in existing)


def content_is_pure_tool_call(
    content: str | None,
    allowed_names: set[str] | frozenset[str] | None = None,
) -> bool:
    """True when ``content`` is exactly one or more tool-call envelopes.

    Used by the agent loop to decide whether to fire scavenge: if the
    message body is nothing but tool-call shapes (or ``<swival:call>``
    envelopes), the model meant to issue a tool call rather than to
    answer in prose.

    When ``allowed_names`` is provided every parsed JSON object must
    coerce to one of the recognized call shapes *and* name an allowed
    tool.  An untyped JSON answer like ``{"answer": "..."}`` mixed with
    a recognizable call shape returns ``False`` so that mixed bodies
    don't open the scavenge gate.
    """
    if not content:
        return False
    stripped = content.strip()
    if not stripped:
        return False
    cleaned = _SWIVAL_CALL_RE.sub("", stripped).strip()
    if not cleaned:
        return True
    if not cleaned.startswith("{"):
        return False
    pos = 0
    n = len(cleaned)
    found_any = False
    while pos < n:
        if cleaned[pos].isspace():
            pos += 1
            continue
        if cleaned[pos] != "{":
            return False
        depth = 0
        in_string = False
        escaped = False
        j = pos
        while j < n:
            c = cleaned[j]
            if escaped:
                escaped = False
                j += 1
                continue
            if in_string:
                if c == "\\":
                    escaped = True
                elif c == '"':
                    in_string = False
                j += 1
                continue
            if c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    blob = cleaned[pos : j + 1]
                    try:
                        parsed = json.loads(blob)
                    except (json.JSONDecodeError, TypeError):
                        return False
                    if not isinstance(parsed, dict):
                        return False
                    if allowed_names is not None:
                        candidate = _coerce_candidate(parsed, allowed_names)
                        if candidate is None:
                            return False
                    found_any = True
                    pos = j + 1
                    break
            j += 1
        else:
            return False
    return found_any


@dataclass
class StormVerdict:
    suppress: bool
    reason: str | None = None
    count: int = 0


@dataclass
class _StormEntry:
    name: str
    args: str
    read_only: bool


_FILESYSTEM_MUTATING = frozenset({"write_file", "edit_file", "delete_file"})
_COMMAND_EXECUTING = frozenset({"run_command", "run_shell_command"})
_STORM_EXEMPT = frozenset(
    {"think", "todo", "snapshot", "complete_goal", "check_subagents"}
)
_SIDE_EFFECTING_OTHER = frozenset({"spawn_subagent"})


def is_storm_exempt(name: str) -> bool:
    return name in _STORM_EXEMPT


def is_mutating(name: str) -> bool:
    """Whether a tool call is considered mutating for storm-window purposes.

    Filesystem writes, command execution, and subagent spawns clear the
    prior read-only entries in the storm window (so a verify-read after
    an edit isn't flagged as a repeat) while still counting against
    themselves.
    """
    return (
        name in _FILESYSTEM_MUTATING
        or name in _COMMAND_EXECUTING
        or name in _SIDE_EFFECTING_OTHER
    )


class StormBreaker:
    """Sliding-window suppression for identical, looping tool calls.

    The breaker's lifetime is **per user request**: it persists across
    every assistant/tool iteration triggered by one user input, and is
    reset on a new user turn, ``/clear``, or a new :class:`Session`.
    Inside that lifetime, the third identical ``(name, args)`` tuple in
    the last ``window`` entries is suppressed.
    """

    def __init__(self, window: int = 6, threshold: int = 3) -> None:
        self.window = window
        self.threshold = threshold
        self._recent: list[_StormEntry] = []

    def inspect(
        self,
        name: str,
        args_json: str,
        mutating: bool,
        suppress_on_repeat: bool = False,
    ) -> StormVerdict:
        if is_storm_exempt(name):
            return StormVerdict(suppress=False, count=0)
        canonical = self._canonical(args_json)
        read_only = not mutating
        if mutating:
            self._recent = [e for e in self._recent if not e.read_only]
        count = sum(1 for e in self._recent if e.name == name and e.args == canonical)
        threshold = 2 if suppress_on_repeat else self.threshold
        if count >= threshold - 1:
            return StormVerdict(
                suppress=True,
                reason=(
                    f"{name} called with identical args {count + 1} times — "
                    "repeat-loop guard tripped"
                ),
                count=count + 1,
            )
        self._recent.append(_StormEntry(name=name, args=canonical, read_only=read_only))
        while len(self._recent) > self.window:
            self._recent.pop(0)
        return StormVerdict(suppress=False, count=count + 1)

    def reset(self) -> None:
        self._recent.clear()

    @staticmethod
    def _canonical(args_json: str) -> str:
        if args_json is None:
            return ""
        try:
            parsed = json.loads(args_json)
        except (json.JSONDecodeError, TypeError):
            return args_json
        try:
            return json.dumps(
                parsed, sort_keys=True, separators=(",", ":"), default=str
            )
        except (TypeError, ValueError):
            return args_json


@dataclass
class FlattenDecision:
    should_flatten: bool
    leaf_count: int
    max_depth: int
    unsafe_reason: str | None = None


@dataclass
class FlattenMeta:
    """Side-table for a flattened schema.

    Maps the dot-path keys exposed to the model back to the nested
    layout the underlying tool expects.  Kept off the provider-facing
    schema so providers that reject unknown fields stay happy.
    """

    original_schema: dict[str, Any]
    paths: dict[str, list[str]] = field(default_factory=dict)


_COMPOSITION_KEYS = ("oneOf", "anyOf", "allOf", "not")


def analyze_schema(schema: dict[str, Any] | None) -> FlattenDecision:
    """Decide whether *schema* should be flattened for the model.

    Returns ``should_flatten=True`` when the schema has more than 10
    leaf parameters or a depth greater than 2.  Refuses to flatten
    schemas containing JSON-Schema features that don't round-trip
    cleanly (composition keywords, ``$ref``, open
    ``additionalProperties``, dotted property names, arrays of
    objects, nullable unions) — in v1 we'd rather pass the original
    schema through than risk a wrong nest.
    """
    if not schema or not isinstance(schema, dict):
        return FlattenDecision(False, 0, 0, "no-schema")

    unsafe = _schema_safety_reason(schema)
    leaf_count = 0
    max_depth = 0

    def visit(node: dict[str, Any], depth: int) -> None:
        nonlocal leaf_count, max_depth
        if not isinstance(node, dict):
            leaf_count += 1
            max_depth = max(max_depth, depth)
            return
        if node.get("type") == "object" and isinstance(node.get("properties"), dict):
            for child in node["properties"].values():
                visit(child, depth + 1)
            return
        leaf_count += 1
        max_depth = max(max_depth, depth)

    visit(schema, 0)

    if unsafe is not None:
        return FlattenDecision(False, leaf_count, max_depth, unsafe)

    should = leaf_count > 10 or max_depth > 2
    if should and _flatten_would_lose_required(schema):
        return FlattenDecision(False, leaf_count, max_depth, "required-loss")
    return FlattenDecision(should, leaf_count, max_depth, None)


def _flatten_would_lose_required(schema: dict[str, Any]) -> bool:
    """True when the original schema requires an object whose children are
    all optional — flattening would erase the parent-presence constraint."""
    if schema.get("type") != "object":
        return False
    required = set(schema.get("required") or [])
    if not required:
        return False
    properties = schema.get("properties") or {}
    for key in required:
        child = properties.get(key)
        if isinstance(child, dict) and child.get("type") == "object":
            child_required = set(child.get("required") or [])
            child_props = set((child.get("properties") or {}).keys())
            if not child_required & child_props:
                return True
            if _flatten_would_lose_required(child):
                return True
    return False


def _schema_safety_reason(schema: dict[str, Any]) -> str | None:
    reason: list[str | None] = [None]

    def walk(node: Any) -> None:
        if reason[0] is not None or not isinstance(node, dict):
            return
        if "$ref" in node:
            reason[0] = "ref"
            return
        for key in _COMPOSITION_KEYS:
            if key in node:
                reason[0] = "composition-keyword"
                return
        if "additionalProperties" in node:
            ap = node["additionalProperties"]
            if ap is not False:
                reason[0] = "open-properties"
                return
        node_type = node.get("type")
        if isinstance(node_type, list):
            non_null = [t for t in node_type if t != "null"]
            if "null" in node_type and len(non_null) == 1:
                reason[0] = "nullable-union"
                return
            if len(non_null) > 1:
                reason[0] = "type-union"
                return
        if node_type == "array":
            items = node.get("items")
            if isinstance(items, dict) and items.get("type") == "object":
                reason[0] = "array-of-objects"
                return
            if isinstance(items, list):
                reason[0] = "tuple-items"
                return
        if node_type == "object" and isinstance(node.get("properties"), dict):
            for key in node["properties"]:
                if "." in key:
                    reason[0] = "dotted-property-name"
                    return
            for child in node["properties"].values():
                walk(child)
                if reason[0] is not None:
                    return

    walk(schema)
    return reason[0]


def flatten_schema(
    schema: dict[str, Any],
) -> tuple[dict[str, Any], FlattenMeta]:
    """Flatten a nested object schema into dot-path leaves.

    Returns the provider-facing flat schema and a :class:`FlattenMeta`
    side table that records how to re-nest arguments later.  Required
    fields propagate down the chain — a leaf is required iff every
    ancestor on the path was in its parent's ``required`` list.

    Leaf nodes are deep-copied so the flat schema and the original
    schema share no mutable state — provider-side schema normalization
    cannot bleed back into the side table.
    """
    import copy as _copy

    flat_props: dict[str, Any] = {}
    required: list[str] = []
    paths: dict[str, list[str]] = {}

    def collect(
        prefix: list[str],
        node: dict[str, Any],
        ancestor_required: bool,
    ) -> None:
        if (
            isinstance(node, dict)
            and node.get("type") == "object"
            and isinstance(node.get("properties"), dict)
        ):
            req_set = set(node.get("required") or [])
            for key, child in node["properties"].items():
                child_required = ancestor_required and key in req_set
                collect(prefix + [key], child, child_required)
            return
        path = list(prefix)
        if not path:
            return
        dotted = ".".join(path)
        flat_props[dotted] = _copy.deepcopy(node)
        paths[dotted] = path
        if ancestor_required:
            required.append(dotted)

    collect([], schema, True)

    flat: dict[str, Any] = {
        "type": "object",
        "properties": flat_props,
    }
    if required:
        flat["required"] = required
    if "description" in schema:
        flat["description"] = schema["description"]

    meta = FlattenMeta(original_schema=schema, paths=paths)
    return flat, meta


class NestCollisionError(ValueError):
    """Raised when nest_arguments encounters conflicting flat keys.

    Happens when ``flat_args`` contains both an ancestor key (``outer``)
    and a dotted descendant (``outer.inner``); the two assignments would
    silently overwrite one another depending on dict iteration order, so
    we surface the conflict instead.
    """


def nest_arguments(flat_args: dict[str, Any], meta: FlattenMeta) -> dict[str, Any]:
    """Re-nest a dict of dot-path arguments back to the original shape.

    Raises :class:`NestCollisionError` when two flat keys would write to
    overlapping paths (e.g. ``outer`` and ``outer.inner``).
    """
    out: dict[str, Any] = {}
    assigned_paths: list[tuple[str, ...]] = []
    for key, value in flat_args.items():
        path = meta.paths.get(key)
        if path is None:
            path = key.split(".")
        path_tuple = tuple(path)
        for prior in assigned_paths:
            if _paths_collide(prior, path_tuple):
                raise NestCollisionError(
                    f"flat arguments collide: "
                    f"{'.'.join(prior)!r} vs {'.'.join(path_tuple)!r}"
                )
        assigned_paths.append(path_tuple)
        cursor: Any = out
        for segment in path[:-1]:
            existing = cursor.get(segment) if isinstance(cursor, dict) else None
            if not isinstance(existing, dict):
                existing = {}
                cursor[segment] = existing
            cursor = existing
        cursor[path[-1]] = value
    return out


def _paths_collide(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """True when one path is a strict prefix of the other or they are equal."""
    n = min(len(a), len(b))
    return a[:n] == b[:n]

"""Schema-aware tool-call argument repair.

Recovers common malformed tool calls from weak models before they become
hard failures.  Each repair rule is conservative: it only fires when the
fix is unambiguous.  All repairs are recorded as structured metadata so
the telemetry pipeline can measure which fixes actually help.
"""

from __future__ import annotations

import difflib
import re
from typing import Any


def repair_tool_args(
    args: dict[str, Any],
    schema: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Repair *args* against a JSON-Schema-style *schema*.

    Parameters
    ----------
    args:
        Parsed tool-call arguments (already through ``json.loads``).
    schema:
        The ``"parameters"`` dict from the tool's OpenAI function schema,
        or ``None`` if the schema is unavailable (MCP / dynamic tools).

    Returns
    -------
    tuple of (repaired_args, repairs)
        *repaired_args* is a new dict (never mutates the input).
        *repairs* is a list of repair-action dicts, empty if nothing changed.
        Each action has at least ``{"type": ..., "field": ...}``.
    """
    import json as _json

    repairs: list[dict[str, Any]] = []

    # Handle double-encoded JSON: entire args is a string that parses to
    # a dict (the LLM JSON-stringified its arguments twice).
    if isinstance(args, str):
        try:
            inner = _json.loads(args)
            if isinstance(inner, dict):
                args = inner
                repairs.append({"type": "unwrap_json_string"})
        except (_json.JSONDecodeError, TypeError):
            pass

    if not isinstance(args, dict):
        return args, repairs

    if schema is None:
        return args, repairs

    properties = schema.get("properties", {})
    if not properties:
        return args, repairs

    result = dict(args)

    _repair_unwrap_nested(result, properties, repairs)
    _repair_near_miss_fields(result, properties, repairs)
    _repair_types(result, properties, repairs)
    _repair_path_globs(result, properties, repairs)
    _strip_unknown(result, properties, repairs)

    return result, repairs


_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "path": ("file_path", "image_path"),
    "file": ("file_path",),
    "filename": ("file_path",),
    "filepath": ("file_path",),
    "cmd": ("command",),
}


def _has_schema_affinity(keys: set[str], properties: dict[str, Any]) -> bool:
    """Check if any *keys* match schema properties (direct, alias, or fuzzy)."""
    known = set(properties)
    for key in keys:
        if key in known:
            return True
        if key in _FIELD_ALIASES:
            if any(t in known for t in _FIELD_ALIASES[key]):
                return True
        if difflib.get_close_matches(key, known, n=1, cutoff=0.8):
            return True
    return False


def _repair_unwrap_nested(
    result: dict[str, Any],
    properties: dict[str, Any],
    repairs: list[dict[str, Any]],
) -> None:
    """Unwrap doubly-nested args.

    Models sometimes wrap tool arguments in an extra layer::

        {"command": {"command": "ls -R"}}     → {"command": "ls -R"}
        {"run_command": {"command": ["ls"]}}   → {"command": ["ls"]}
        {"cmd": "{\\"command\\": [\\"ls\\"]}"}  → {"command": ["ls"]}

    The rule fires when the outer dict has exactly one key whose value is
    a dict (or a JSON string that parses to a dict) whose keys have schema
    affinity.  It will *not* fire when the schema expects an ``object``
    type for the outer key (that could be intentional nesting).
    """
    import json as _json  # deferred to avoid top-level import when unused

    if len(result) != 1:
        return

    outer_key = next(iter(result))
    outer_val = result[outer_key]

    inner: dict[str, Any] | None = None
    was_json_string = False

    if isinstance(outer_val, dict):
        inner = outer_val
    elif isinstance(outer_val, str):
        try:
            parsed = _json.loads(outer_val)
            if isinstance(parsed, dict):
                inner = parsed
                was_json_string = True
        except (_json.JSONDecodeError, TypeError):
            pass

    if not inner:
        return

    # If the outer key is a schema property that expects an object, the
    # nesting might be intentional — leave it alone.
    known = set(properties)
    if outer_key in known and properties[outer_key].get("type") == "object":
        return

    # The inner dict must have at least one key with schema affinity
    # (direct match, alias, or fuzzy match).
    if not _has_schema_affinity(set(inner), properties):
        return

    result.clear()
    result.update(inner)
    repairs.append(
        {
            "type": "unwrap_nested",
            "outer_key": outer_key,
            "was_json_string": was_json_string,
        }
    )


def _repair_near_miss_fields(
    result: dict[str, Any],
    properties: dict[str, Any],
    repairs: list[dict[str, Any]],
) -> None:
    """Rename argument keys that are close matches to known property names."""
    known = set(properties)
    renames: list[tuple[str, str]] = []
    for key in list(result):
        if key in known:
            continue
        # Check explicit aliases first (catches pairs too dissimilar for
        # difflib, e.g. "path" → "file_path").
        alias_targets = _FIELD_ALIASES.get(key, ())
        hit = next((t for t in alias_targets if t in known and t not in result), None)
        if hit:
            renames.append((key, hit))
            continue
        matches = difflib.get_close_matches(key, known, n=1, cutoff=0.8)
        if matches:
            correct = matches[0]
            if correct not in result:
                renames.append((key, correct))
    for old, new in renames:
        result[new] = result.pop(old)
        repairs.append({"type": "rename_field", "field": new, "from": old})


def _repair_types(
    result: dict[str, Any],
    properties: dict[str, Any],
    repairs: list[dict[str, Any]],
) -> None:
    """Coerce safe scalar type mismatches."""
    for field, prop in properties.items():
        if field not in result:
            continue
        value = result[field]
        expected = prop.get("type")
        if expected is None:
            continue

        coerced = _coerce_scalar(value, expected)
        if coerced is not _SKIP:
            repairs.append(
                {
                    "type": "coerce_type",
                    "field": field,
                    "from": repr(value),
                    "to": repr(coerced),
                    "expected_type": expected,
                }
            )
            result[field] = coerced


_SKIP = object()

_BOOL_TRUTHY = frozenset({"true", "1", "yes"})
_BOOL_FALSY = frozenset({"false", "0", "no"})


def _coerce_scalar(value: Any, expected: str) -> Any:
    """Try to coerce *value* to *expected* type.  Return ``_SKIP`` if no safe coercion."""
    if expected == "integer" and isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return _SKIP

    if expected == "number" and isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return _SKIP

    if expected == "boolean" and isinstance(value, str):
        low = value.lower().strip()
        if low in _BOOL_TRUTHY:
            return True
        if low in _BOOL_FALSY:
            return False
        return _SKIP

    if expected == "boolean" and isinstance(value, int) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(value)
        return _SKIP

    if expected == "string" and isinstance(value, (int, float, bool)):
        return str(value)

    if expected == "integer" and isinstance(value, float) and value == int(value):
        return int(value)

    return _SKIP


_GLOB_META_RE = re.compile(r"[*?\[\]]")

_PATH_FIELDS = frozenset(
    {
        "path",
        "file_path",
        "image_path",
        "dir",
        "directory",
    }
)


def _repair_path_globs(
    result: dict[str, Any],
    properties: dict[str, Any],
    repairs: list[dict[str, Any]],
) -> None:
    """Strip glob metacharacters from path/directory fields.

    Models frequently pass ``".**"`` or ``"**"`` as a path, mashing together
    ``.`` (current directory) and ``**`` (recursive glob).  The intent is
    "search everything here" — the correct path value is ``"."``.
    """
    for field, prop in properties.items():
        if field not in result:
            continue
        value = result[field]
        if not isinstance(value, str):
            continue
        if not _GLOB_META_RE.search(value):
            continue
        # Only touch fields that are clearly file/directory paths, not
        # pattern or include fields.
        desc = prop.get("description", "").lower()
        if field not in _PATH_FIELDS and "path" not in field:
            continue
        if "pattern" in desc or "regex" in desc or "glob" in desc:
            continue
        cleaned = _GLOB_META_RE.sub("", value).rstrip("/")
        if not cleaned:
            cleaned = "."
        if cleaned != value:
            result[field] = cleaned
            repairs.append(
                {
                    "type": "strip_glob_from_path",
                    "field": field,
                    "from": value,
                    "to": cleaned,
                }
            )


def _strip_unknown(
    result: dict[str, Any],
    properties: dict[str, Any],
    repairs: list[dict[str, Any]],
) -> None:
    """Remove fields not in the schema.  Skips when all fields are unknown
    (the call is wholly malformed — stripping would destroy everything)."""
    known = set(properties)
    if not (known & set(result)):
        return

    for field in sorted(set(result) - known):
        del result[field]
        repairs.append({"type": "strip_unknown", "field": field})


# ── Corrective feedback ─────────────────────────────────────────────

_STRUCTURAL_REPAIRS = frozenset(
    {"unwrap_nested", "unwrap_json_string", "rename_field", "strip_unknown"}
)


def format_repair_feedback(
    name: str,
    raw_args: str,
    repaired_args: dict[str, Any],
    repairs: list[dict[str, Any]],
    schema: dict[str, Any] | None = None,
) -> str:
    """Build a short corrective note that is appended to the tool result.

    Only fires for *structural* repairs (nesting, wrong field names,
    unknown fields) — silent type coercions don't warrant a lecture.

    Parameters
    ----------
    name:
        Tool name (e.g. ``"run_command"``).
    raw_args:
        The raw JSON string the LLM sent as tool-call arguments.
    repaired_args:
        The dict that was actually dispatched after repair.
    repairs:
        The list of repair-action dicts from :func:`repair_tool_args`.
    schema:
        Optional tool parameter schema.  When provided, the "Corrected"
        display upgrades string values to arrays where the schema says so.
    """
    import json as _json

    structural = [r for r in repairs if r["type"] in _STRUCTURAL_REPAIRS]
    if not structural:
        return ""

    # Parse the original args for display.
    try:
        original = _json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    except (_json.JSONDecodeError, TypeError):
        original = raw_args

    # Build an "ideal" version of repaired_args: if the schema expects an
    # array but we still have a bare string, show the split form so the
    # model sees the correct type.
    ideal = dict(repaired_args)
    if schema:
        props = schema.get("properties", {})
        for field, value in list(ideal.items()):
            if (
                field in props
                and props[field].get("type") == "array"
                and isinstance(value, str)
            ):
                ideal[field] = value.split() if value.strip() else []

    lines: list[str] = [f"\n[Syntax correction] Your {name} call was auto-corrected:"]
    lines.append(f"  Received:  {_json.dumps(original, ensure_ascii=False)}")
    lines.append(f"  Corrected: {_json.dumps(ideal, ensure_ascii=False)}")

    for r in structural:
        rtype = r["type"]
        if rtype == "unwrap_nested":
            lines.append(
                "  Arguments must be flat key-value pairs, not wrapped "
                "in an extra object."
            )
        elif rtype == "unwrap_json_string":
            lines.append("  Arguments were double-encoded as a JSON string.")
        elif rtype == "rename_field":
            lines.append(f'  The parameter is "{r["field"]}", not "{r["from"]}".')
        elif rtype == "strip_unknown":
            lines.append(f'  Unknown parameter "{r["field"]}" was removed.')

    # Extra hint when a required field is still the wrong type after repair.
    if schema:
        props = schema.get("properties", {})
        for field, value in repaired_args.items():
            if (
                field in props
                and props[field].get("type") == "array"
                and isinstance(value, str)
            ):
                lines.append(
                    f'  "{field}" must be a JSON array, e.g. ["cmd", "arg1", "arg2"].'
                )

    lines.append("  Use this corrected format for subsequent calls.")
    return "\n".join(lines)

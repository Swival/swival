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
    if not isinstance(args, dict):
        return args, []

    if schema is None:
        return args, []

    properties = schema.get("properties", {})
    if not properties:
        return args, []

    repairs: list[dict[str, Any]] = []
    result = dict(args)

    _repair_wrapped_args(result, properties, repairs)
    _repair_near_miss_fields(result, properties, repairs)
    _repair_types(result, properties, repairs)
    _repair_path_globs(result, properties, repairs)
    _strip_unknown(result, properties, repairs)

    return result, repairs


def _repair_wrapped_args(
    result: dict[str, Any],
    properties: dict[str, Any],
    repairs: list[dict[str, Any]],
) -> None:
    """Unwrap arguments the model accidentally nested inside themselves.

    Some models produce ``{"command": {"command": "ls -la"}}`` instead of
    ``{"command": "ls -la"}``.  When a field's value is a dict whose keys
    are a subset of the top-level schema properties, hoist the inner values
    up to the top level.
    """
    known = set(properties)
    for field in list(result):
        value = result[field]
        if not isinstance(value, dict):
            continue
        # Only unwrap when every key in the inner dict is a known schema field
        if not value or not (set(value) <= known):
            continue
        # Hoist the inner dict's entries to the top level
        for k, v in value.items():
            result[k] = v
        if field not in value:
            del result[field]
        repairs.append({"type": "unwrap_nested", "field": field})


_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "path": ("file_path", "image_path"),
    "file": ("file_path",),
    "filename": ("file_path",),
    "filepath": ("file_path",),
}


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

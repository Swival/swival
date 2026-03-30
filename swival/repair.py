"""Schema-aware tool-call argument repair.

Recovers common malformed tool calls from weak models before they become
hard failures.  Each repair rule is conservative: it only fires when the
fix is unambiguous.  All repairs are recorded as structured metadata so
the telemetry pipeline can measure which fixes actually help.
"""

from __future__ import annotations

import difflib
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

    _repair_near_miss_fields(result, properties, repairs)
    _repair_types(result, properties, repairs)
    _fill_defaults(result, schema, repairs)
    _strip_unknown(result, properties, repairs)

    return result, repairs


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


def _fill_defaults(
    result: dict[str, Any],
    schema: dict[str, Any],
    repairs: list[dict[str, Any]],
) -> None:
    """Fill omitted optional fields that have schema defaults."""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    for field, prop in properties.items():
        if field in result or field in required or "default" not in prop:
            continue
        default = prop["default"]
        result[field] = default
        repairs.append({"type": "fill_default", "field": field, "value": repr(default)})


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

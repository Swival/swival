"""Message accessor helpers for dict-or-namespace messages."""

# Image token budget: worst-case high-detail (85 base + 170 * 16 tiles = 2805)
IMAGE_TOKEN_ESTIMATE = 2805


def _msg_get(msg, key, default=None):
    return (
        msg.get(key, default) if isinstance(msg, dict) else getattr(msg, key, default)
    )


def _msg_role(msg) -> str | None:
    return _msg_get(msg, "role")


def _msg_content(msg) -> str:
    c = _msg_get(msg, "content", "")
    if isinstance(c, list):
        return " ".join(
            part.get("text", "")
            for part in c
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return c or ""


def _msg_tool_calls(msg):
    return _msg_get(msg, "tool_calls")


def _msg_tool_call_id(msg) -> str | None:
    return _msg_get(msg, "tool_call_id")


def _msg_name(msg) -> str:
    return _msg_get(msg, "name") or ""


def _set_msg_content(msg, value: str) -> None:
    if isinstance(msg, dict):
        msg["content"] = value
    else:
        msg.content = value


def _estimate_tokens(text: str) -> int:
    """Rough token estimate without importing tiktoken."""
    return len(text) // 4


_ALWAYS_SYNTHETIC_PREFIXES: tuple[str, ...] = (
    "[REVIEWER FEEDBACK",
    "[image]",
    "[Context for follow-up:",
)


def _is_synthetic(msg) -> bool:
    """Check if a user message is a synthetic intervention, not a real task.

    Accepts a message dict/namespace or a plain content string.

    Uses the ``_swival_synthetic`` marker set by the agent loop when it
    injects nudges, guardrails, and other scaffolding messages.  Falls back
    to bracket-prefixed patterns for content that is always synthetic by
    construction (image, reviewer, command-tool context).
    """
    if not isinstance(msg, str) and _msg_get(msg, "_swival_synthetic"):
        return True
    content = msg if isinstance(msg, str) else _msg_content(msg)
    for prefix in _ALWAYS_SYNTHETIC_PREFIXES:
        if content.startswith(prefix):
            return True
    return False


def _find_current_turn_boundary(messages: list) -> int:
    """Return the index of the most recent real user message, or 0 if none.

    This defines the current-turn boundary: all assistant tool_calls after
    this index are in the current turn and must preserve opaque extras
    (e.g. extra_content.google.thought_signature) for providers that
    require replay metadata.

    Synthetic user messages (nudges, guardrails, recap injections) are
    skipped — they are not real turn boundaries.
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if _msg_role(msg) == "user":
            content = _msg_content(msg)
            if content and not _is_synthetic(msg):
                return i
    return 0


def _canonicalize_tool_calls(messages: list) -> None:
    """Rewrite historical assistant tool_calls to minimal shape.

    Strips provider extras (index, etc.) keeping only id, type,
    function.name, function.arguments.  Skips the most recent assistant
    message with tool_calls so in-flight calls are untouched.

    Current-turn tool calls (after the most recent user message) preserve
    request-shaped extras like extra_content so that providers requiring
    replay metadata (e.g. Gemini thought_signature) are not broken.
    """
    last_tc_idx = None
    current_turn_start = 0
    found_boundary = False
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        role = _msg_role(msg)
        if last_tc_idx is None and role == "assistant" and _msg_tool_calls(msg):
            last_tc_idx = i
        if not found_boundary and role == "user":
            content = _msg_content(msg)
            if content and not _is_synthetic(msg):
                current_turn_start = i
                found_boundary = True
        if last_tc_idx is not None and found_boundary:
            break

    for i, msg in enumerate(messages):
        if i == last_tc_idx:
            continue
        if not isinstance(msg, dict):
            continue
        if _msg_role(msg) != "assistant":
            continue
        tcs = msg.get("tool_calls")
        if not tcs or not isinstance(tcs, list):
            continue

        in_current_turn = i > current_turn_start

        new_tcs = []
        changed = False
        for tc in tcs:
            if isinstance(tc, dict):
                fn = tc.get("function", {})
                canonical = {
                    "id": tc.get("id", ""),
                    "type": tc.get("type", "function"),
                    "function": {
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", ""),
                    },
                }
                if in_current_turn:
                    extra = _msg_get(tc, "extra_content")
                    if extra is not None:
                        canonical["extra_content"] = extra
                if canonical != tc:
                    changed = True
                new_tcs.append(canonical)
            elif hasattr(tc, "function"):
                fn = tc.function
                canonical = {
                    "id": tc.id if hasattr(tc, "id") else "",
                    "type": "function",
                    "function": {
                        "name": fn.name if hasattr(fn, "name") else "",
                        "arguments": (fn.arguments if hasattr(fn, "arguments") else "")
                        or "",
                    },
                }
                if in_current_turn:
                    extra = _msg_get(tc, "extra_content")
                    if extra is not None:
                        canonical["extra_content"] = extra
                changed = True
                new_tcs.append(canonical)
            else:
                new_tcs.append(tc)

        if changed:
            msg["tool_calls"] = new_tcs


def _has_image_content(messages: list) -> bool:
    """Check if any message contains image_url parts."""
    return any(
        isinstance(part, dict) and part.get("type") == "image_url"
        for msg in messages
        if isinstance(msg, dict) and isinstance(msg.get("content"), list)
        for part in msg["content"]
    )

"""Message accessor helpers for dict-or-namespace messages."""

# Image token budget: worst-case high-detail (85 base + 170 * 16 tiles = 2805)
IMAGE_TOKEN_ESTIMATE = 2805

RECAP_MARKER = "[non-instructional context recap"

# Stands in for dropped turns when no summary could be produced.
COMPACTION_MARKER = (
    "[context compacted \u2014 older tool calls and results were "
    "removed to fit context window]"
)


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


def _tool_call_id(tc) -> str | None:
    """Extract the id from a tool_call entry (dict or provider namespace)."""
    return tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)


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
    return content.startswith(_ALWAYS_SYNTHETIC_PREFIXES)


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


def _complete_orphaned_tool_calls(messages: list, *, content: str) -> int:
    """Backfill placeholder tool-result messages for unmatched tool_calls.

    Anthropic-format providers (Bedrock, Vertex) reject a conversation whenever
    a ``tool_use`` block has no ``tool_result`` block in the message that
    follows it. That happens when the agent loop is cut short between emitting
    an assistant tool-call message and producing its results — a user interrupt
    or an external cancellation landing mid-dispatch. Some or all of the call's
    results never get appended, leaving a dangling ``tool_use`` that poisons
    every later request.

    For each assistant message carrying tool_calls, this inserts a placeholder
    result for every tool_call id that lacks one, positioned right after any
    results that did make it in. Keeping the original tool_calls preserves the
    model's reasoning and lets it see that the work was interrupted. Returns the
    number of placeholders inserted.
    """
    inserted = 0
    i = 0
    while i < len(messages):
        msg = messages[i]
        if _msg_role(msg) != "assistant":
            i += 1
            continue
        tool_calls = _msg_tool_calls(msg)
        if not tool_calls:
            i += 1
            continue
        j = i + 1
        satisfied: set[str] = set()
        while j < len(messages) and _msg_role(messages[j]) == "tool":
            tc_id = _msg_tool_call_id(messages[j])
            if tc_id:
                satisfied.add(tc_id)
            j += 1
        backfill = [
            {"role": "tool", "tool_call_id": tc_id, "content": content}
            for tc in tool_calls
            if (tc_id := _tool_call_id(tc)) and tc_id not in satisfied
        ]
        if backfill:
            messages[j:j] = backfill
            inserted += len(backfill)
        i = j + len(backfill)
    return inserted


_MARQUEE_PIECE_BUDGET = 2048
_MARQUEE_SEPARATOR = "   ·   "


def _trim_for_marquee(text: str, budget: int = _MARQUEE_PIECE_BUDGET) -> str:
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text
    if budget == 1:
        return "…"
    return text[: budget - 1] + "…"


def _marquee_text_for_turn(messages: list) -> str | None:
    """Build a marquee string from the tail of messages since the last assistant.

    Walks backward from the end of ``messages`` and collects every contiguous
    non-assistant, non-system entry — i.e. the inputs that will be sent to
    the model on the upcoming LLM call. Tool messages are prefixed with their
    tool name. Each piece is capped via :func:`_trim_for_marquee`. Returns
    ``None`` if no non-blank tail content exists.
    """
    pieces: list[str] = []
    for m in reversed(messages):
        role = _msg_role(m)
        if role in ("assistant", "system"):
            break
        raw = _msg_content(m) or ""
        if not raw.strip():
            continue
        content = _trim_for_marquee(raw)
        if role == "tool":
            name = _msg_name(m) or "tool"
            pieces.append(f"{name}: {content}")
        else:
            pieces.append(content)

    if not pieces:
        return None
    pieces.reverse()
    return _MARQUEE_SEPARATOR.join(pieces)


def _has_image_content(messages: list) -> bool:
    """Check if any message contains image_url parts."""
    return any(
        isinstance(part, dict) and part.get("type") == "image_url"
        for msg in messages
        if isinstance(msg, dict) and isinstance(msg.get("content"), list)
        for part in msg["content"]
    )

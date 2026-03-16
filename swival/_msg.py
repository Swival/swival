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
    return _msg_get(msg, "name", "") or ""


def _set_msg_content(msg, value: str) -> None:
    if isinstance(msg, dict):
        msg["content"] = value
    else:
        msg.content = value


def _has_image_content(messages: list) -> bool:
    """Check if any message contains image_url parts."""
    for m in messages:
        if isinstance(m, dict) and isinstance(m.get("content"), list):
            for part in m["content"]:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False

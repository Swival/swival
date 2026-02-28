"""Message accessor helpers for dict-or-namespace messages."""


def _msg_get(msg, key, default=None):
    return (
        msg.get(key, default) if isinstance(msg, dict) else getattr(msg, key, default)
    )


def _msg_role(msg) -> str | None:
    return _msg_get(msg, "role")


def _msg_content(msg) -> str:
    return _msg_get(msg, "content", "") or ""


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

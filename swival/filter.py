"""Outbound LLM filter: run a user-defined script before sending messages to the provider."""

import json
import shlex
import subprocess
import sys
from ._msg import _msg_get, _msg_role, _msg_tool_calls, _msg_tool_call_id, _msg_name


class FilterError(Exception):
    """Raised when the filter script rejects or fails."""


def _message_to_dict(msg) -> dict:
    """Normalize a message (dict or namespace) to a plain dict for JSON serialization."""
    d: dict = {}
    role = _msg_role(msg)
    if role:
        d["role"] = role
    content = _msg_get(msg, "content")
    if content is not None:
        d["content"] = content
    tool_calls = _msg_tool_calls(msg)
    if tool_calls:
        serialized = []
        for tc in tool_calls:
            entry = {
                "id": tc.id if hasattr(tc, "id") else tc.get("id"),
                "type": "function",
                "function": {
                    "name": (
                        tc.function.name
                        if hasattr(tc, "function") and hasattr(tc.function, "name")
                        else tc.get("function", {}).get("name")
                    ),
                    "arguments": (
                        tc.function.arguments
                        if hasattr(tc, "function") and hasattr(tc.function, "arguments")
                        else tc.get("function", {}).get("arguments", "")
                    ),
                },
            }
            extra = _msg_get(tc, "extra_content")
            if extra is not None:
                entry["extra_content"] = extra
            serialized.append(entry)
        d["tool_calls"] = serialized
    tool_call_id = _msg_tool_call_id(msg)
    if tool_call_id is not None:
        d["tool_call_id"] = tool_call_id
    name = _msg_name(msg)
    if name is not None:
        d["name"] = name
    return d


_VALID_ROLES = {"system", "user", "assistant", "tool"}


def _validate_returned_messages(messages: list) -> None:
    """Validate the structure of messages returned by a filter script."""
    if not isinstance(messages, list):
        raise FilterError("filter returned non-list messages")
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise FilterError(f"filter message [{i}] is not a dict")
        role = msg.get("role")
        if role not in _VALID_ROLES:
            raise FilterError(f"filter message [{i}] has invalid role: {role!r}")
        if role == "tool" and not msg.get("tool_call_id"):
            raise FilterError(f"filter message [{i}] (tool) missing tool_call_id")
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            if not isinstance(tool_calls, list):
                raise FilterError(f"filter message [{i}] tool_calls is not a list")
            for j, tc in enumerate(tool_calls):
                if not isinstance(tc, dict):
                    raise FilterError(
                        f"filter message [{i}] tool_calls[{j}] is not a dict"
                    )
                if not tc.get("id"):
                    raise FilterError(
                        f"filter message [{i}] tool_calls[{j}] missing id"
                    )
                fn = tc.get("function")
                if not isinstance(fn, dict) or not fn.get("name"):
                    raise FilterError(
                        f"filter message [{i}] tool_calls[{j}] missing function.name"
                    )


def run_llm_filter(
    llm_filter: str,
    messages: list,
    model: str,
    provider: str,
    tools: list | None = None,
    call_kind: str = "agent",
    timeout: int = 30,
) -> list:
    """Run the user-defined filter script and return the (possibly modified) messages.

    Raises FilterError if the script blocks, fails, or returns invalid output.
    The original *messages* list is never mutated.
    """
    parts = shlex.split(llm_filter)

    payload = {
        "provider": provider,
        "model": model,
        "call_kind": call_kind,
        "messages": [_message_to_dict(m) for m in messages],
    }
    if tools is not None:
        payload["tools"] = tools

    input_json = json.dumps(payload)

    try:
        result = subprocess.run(
            parts,
            input=input_json,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise FilterError(f"filter command timed out after {timeout}s: {llm_filter}")
    except FileNotFoundError:
        raise FilterError(f"filter command not found: {parts[0]}")
    except OSError as e:
        raise FilterError(f"filter command failed to start: {e}")

    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    if result.returncode != 0:
        raise FilterError(f"filter exited with code {result.returncode}: {llm_filter}")

    try:
        output = json.loads(result.stdout)
    except (json.JSONDecodeError, ValueError) as e:
        raise FilterError(f"filter returned invalid JSON: {e}")

    if not isinstance(output, dict):
        raise FilterError("filter output is not a JSON object")

    if output.get("allow") is False:
        reason = output.get("reason", "blocked by filter")
        raise FilterError(reason)

    if "messages" not in output:
        raise FilterError("filter output missing 'messages' key")

    filtered = output["messages"]
    _validate_returned_messages(filtered)

    return filtered

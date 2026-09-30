"""Call Swival's MCP tools from a run_python snippet.

    import swival_tools

    swival_tools.ALL_TOOLS                  # tools this snippet may call
    swival_tools.help()                     # {name: first line of description}
    swival_tools.help("mcp__srv__tool")     # description and schemas
    swival_tools.call("mcp__srv__tool", {"arg": 1}, timeout=10)

Only what the snippet prints goes back to the model.
"""

import json
import os
import struct
import threading

__all__ = ["ALL_TOOLS", "BridgeError", "ToolError", "call", "help"]

_HEADER = struct.Struct(">I")
# Replies are not tagged, so threads must take turns.
_LOCK = threading.Lock()
# Why Swival stopped answering; later calls fail with the same reason.
_stopped: str | None = None


class ToolError(Exception):
    """The tool failed, or this one call was refused."""


class BridgeError(Exception):
    """Swival stopped answering calls for this run.

    This happens when the timeout expires, a limit is reached, the user
    cancels, or the goal's budget runs out.
    Later calls fail the same way.
    """


def _connect() -> tuple[int, int]:
    spec = os.environ.pop("SWIVAL_PYTHON_BRIDGE", "")
    try:
        write_fd, read_fd = (int(fd) for fd in spec.split(","))
    except ValueError:
        raise ImportError(
            "swival_tools only works inside Swival's run_python tool"
        ) from None
    for fd in (write_fd, read_fd):
        os.set_inheritable(fd, False)
    return write_fd, read_fd


_WRITE_FD, _READ_FD = _connect()


def _read_exact(size: int) -> bytes:
    chunks = []
    while size:
        chunk = os.read(_READ_FD, size)
        if not chunk:
            raise BridgeError("the connection to Swival is closed")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def _request(message: dict):
    global _stopped
    payload = json.dumps(message).encode()
    data = memoryview(_HEADER.pack(len(payload)) + payload)
    with _LOCK:
        if _stopped is not None:
            raise BridgeError(_stopped)
        try:
            while data:
                data = data[os.write(_WRITE_FD, data) :]
            (length,) = _HEADER.unpack(_read_exact(_HEADER.size))
            reply = json.loads(_read_exact(length))
        except OSError:
            raise BridgeError("the connection to Swival is closed") from None
        if reply.get("kind") == "stop":
            _stopped = reply.get("error") or "Swival stopped serving calls"
    if reply.get("ok"):
        return reply.get("value")
    kind = reply.get("kind")
    if kind == "stop":
        raise BridgeError(_stopped)
    if kind == "usage":
        raise ValueError(reply.get("error"))
    raise ToolError(reply.get("error"))


def call(name: str, arguments: dict | None = None, *, timeout: float | None = None):
    """Call a tool and return its result as Python data.

    Structured results come back as sent, JSON text is parsed, and anything
    else comes back as text.
    *timeout* can only shorten the time the call is allowed.
    """
    request = {"op": "call", "name": name, "arguments": arguments}
    if timeout is not None:
        request["timeout"] = timeout
    return _request(request)


def help(name: str | None = None):
    """Describe one tool, or summarize all of them when *name* is omitted."""
    return _request({"op": "help", "name": name})


ALL_TOOLS: list[str] = _request({"op": "list"})

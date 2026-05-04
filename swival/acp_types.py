"""ACP (Agent Client Protocol) wire types and constants.

The wire format is JSON-RPC 2.0 over stdio with newline-delimited JSON: each
message is a single JSON object terminated by '\n'. No Content-Length header.

This module owns the constants and small encode/decode helpers; the protocol
state machine lives in acp_server.py. Spec: https://agentclientprotocol.com.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


PROTOCOL_VERSION = 1


METHOD_INITIALIZE = "initialize"
METHOD_AUTHENTICATE = "authenticate"
METHOD_SESSION_NEW = "session/new"
METHOD_SESSION_LOAD = "session/load"
METHOD_SESSION_PROMPT = "session/prompt"
METHOD_SESSION_CANCEL = "session/cancel"
METHOD_SESSION_UPDATE = "session/update"
METHOD_SESSION_REQUEST_PERMISSION = "session/request_permission"
METHOD_FS_READ_TEXT_FILE = "fs/read_text_file"
METHOD_FS_WRITE_TEXT_FILE = "fs/write_text_file"


STOP_END_TURN = "end_turn"
STOP_MAX_TOKENS = "max_tokens"
STOP_MAX_TURN_REQUESTS = "max_turn_requests"
STOP_REFUSAL = "refusal"
STOP_CANCELLED = "cancelled"


UPDATE_AGENT_MESSAGE_CHUNK = "agent_message_chunk"
UPDATE_AGENT_THOUGHT_CHUNK = "agent_thought_chunk"
UPDATE_TOOL_CALL = "tool_call"
UPDATE_TOOL_CALL_UPDATE = "tool_call_update"
UPDATE_PLAN = "plan"
UPDATE_USER_MESSAGE_CHUNK = "user_message_chunk"


TOOL_STATUS_PENDING = "pending"
TOOL_STATUS_IN_PROGRESS = "in_progress"
TOOL_STATUS_COMPLETED = "completed"
TOOL_STATUS_FAILED = "failed"


TOOL_KIND_READ = "read"
TOOL_KIND_EDIT = "edit"
TOOL_KIND_EXECUTE = "execute"
TOOL_KIND_THINK = "think"
TOOL_KIND_SEARCH = "search"
TOOL_KIND_DELETE = "delete"
TOOL_KIND_FETCH = "fetch"
TOOL_KIND_OTHER = "other"


JSONRPC_VERSION = "2.0"

ERROR_PARSE = -32700
ERROR_INVALID_REQUEST = -32600
ERROR_METHOD_NOT_FOUND = -32601
ERROR_INVALID_PARAMS = -32602
ERROR_INTERNAL = -32603

ERROR_AUTH_REQUIRED = -32000
ERROR_SESSION_NOT_FOUND = -32001
ERROR_UNSUPPORTED_OPERATION = -32002


_TOOL_NAME_TO_KIND: dict[str, str] = {
    "read_file": TOOL_KIND_READ,
    "read_multiple_files": TOOL_KIND_READ,
    "list_files": TOOL_KIND_READ,
    "outline": TOOL_KIND_READ,
    "view_image": TOOL_KIND_READ,
    "grep": TOOL_KIND_SEARCH,
    "write_file": TOOL_KIND_EDIT,
    "edit_file": TOOL_KIND_EDIT,
    "delete_file": TOOL_KIND_DELETE,
    "run_command": TOOL_KIND_EXECUTE,
    "run_shell_command": TOOL_KIND_EXECUTE,
    "fetch_url": TOOL_KIND_FETCH,
    "think": TOOL_KIND_THINK,
}


def tool_kind_for(name: str) -> str:
    """Map a swival tool name to an ACP tool kind."""
    if name in _TOOL_NAME_TO_KIND:
        return _TOOL_NAME_TO_KIND[name]
    if name.startswith("mcp__") or name.startswith("a2a__"):
        return TOOL_KIND_OTHER
    return TOOL_KIND_OTHER


def tool_title_for(name: str, arguments: Any) -> str:
    """Human-readable title for a tool call, used in tool_call updates."""
    if not isinstance(arguments, dict):
        return name
    if name in ("read_file", "outline", "view_image"):
        path = arguments.get("file_path") or arguments.get("path")
        if path:
            return f"{name}: {path}"
    if name == "read_multiple_files":
        paths = arguments.get("file_paths") or []
        if isinstance(paths, list) and paths:
            return f"{name}: {len(paths)} file(s)"
    if name in ("write_file", "edit_file", "delete_file"):
        path = arguments.get("file_path") or arguments.get("path")
        if path:
            return f"{name}: {path}"
    if name in ("run_command", "run_shell_command"):
        cmd = arguments.get("command") or arguments.get("cmd")
        if isinstance(cmd, list):
            cmd = " ".join(str(c) for c in cmd)
        if cmd:
            preview = str(cmd)
            if len(preview) > 80:
                preview = preview[:77] + "..."
            return f"$ {preview}"
    if name == "fetch_url":
        url = arguments.get("url")
        if url:
            return f"fetch: {url}"
    if name == "grep":
        pattern = arguments.get("pattern")
        if pattern:
            return f"grep: {pattern}"
    return name


@dataclass
class JsonRpcRequest:
    """Decoded JSON-RPC 2.0 request or notification."""

    method: str
    params: dict | None = None
    id: int | str | None = None

    @property
    def is_notification(self) -> bool:
        return self.id is None


@dataclass
class JsonRpcResponse:
    """JSON-RPC 2.0 response (success or error)."""

    id: int | str | None
    result: Any = None
    error: dict | None = None

    def to_wire(self) -> dict:
        out: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "id": self.id}
        if self.error is not None:
            out["error"] = self.error
        else:
            out["result"] = self.result if self.result is not None else {}
        return out


@dataclass
class JsonRpcNotification:
    """JSON-RPC 2.0 notification (no id)."""

    method: str
    params: dict = field(default_factory=dict)

    def to_wire(self) -> dict:
        return {
            "jsonrpc": JSONRPC_VERSION,
            "method": self.method,
            "params": self.params,
        }


def encode_message(msg: dict) -> bytes:
    """Encode a JSON-RPC message as a single newline-terminated line."""
    return (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")


def decode_message(line: bytes | str) -> JsonRpcRequest:
    """Decode a single NDJSON line into a JsonRpcRequest.

    Raises json.JSONDecodeError on malformed JSON, ValueError on shape errors.
    """
    if isinstance(line, bytes):
        line = line.decode("utf-8")
    obj = json.loads(line)
    if not isinstance(obj, dict):
        raise ValueError("JSON-RPC message must be an object")
    if obj.get("jsonrpc") != JSONRPC_VERSION:
        raise ValueError(f"Unsupported jsonrpc version: {obj.get('jsonrpc')!r}")
    method = obj.get("method")
    if not isinstance(method, str):
        raise ValueError("missing or invalid 'method'")
    params = obj.get("params")
    if params is not None and not isinstance(params, dict):
        raise ValueError("'params' must be an object if present")
    return JsonRpcRequest(method=method, params=params, id=obj.get("id"))


def make_error(
    request_id: int | str | None,
    code: int,
    message: str,
    data: Any = None,
) -> dict:
    """Build a JSON-RPC error response object."""
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return JsonRpcResponse(id=request_id, error=err).to_wire()


def make_result(request_id: int | str | None, result: Any) -> dict:
    """Build a JSON-RPC success response object."""
    return JsonRpcResponse(id=request_id, result=result).to_wire()


def make_notification(method: str, params: dict) -> dict:
    """Build a JSON-RPC notification object."""
    return JsonRpcNotification(method=method, params=params).to_wire()


def initialize_response(*, protocol_version: int = PROTOCOL_VERSION) -> dict:
    """Build the initialize response payload (the result field of the RPC).

    v1 advertises:
      - loadSession: false (no session resume yet)
      - promptCapabilities: text and resource_link only
      - mcpCapabilities: empty (clients shouldn't push MCP servers)
      - authMethods: [] (provider auth lives in swival config)
    """
    return {
        "protocolVersion": protocol_version,
        "agentCapabilities": {
            "loadSession": False,
            "promptCapabilities": {
                "image": False,
                "audio": False,
                "embeddedContext": False,
            },
            "mcpCapabilities": {
                "http": False,
                "sse": False,
            },
        },
        "authMethods": [],
    }


def session_update_payload(session_id: str, update: dict) -> dict:
    """Wrap an update body in the {sessionId, update} envelope."""
    return {"sessionId": session_id, "update": update}


def text_chunk_update(text: str) -> dict:
    """An agent_message_chunk update body."""
    return {
        "sessionUpdate": UPDATE_AGENT_MESSAGE_CHUNK,
        "content": {"type": "text", "text": text},
    }


def thought_chunk_update(text: str) -> dict:
    """An agent_thought_chunk update body."""
    return {
        "sessionUpdate": UPDATE_AGENT_THOUGHT_CHUNK,
        "content": {"type": "text", "text": text},
    }


def tool_call_update(
    *,
    tool_call_id: str,
    name: str,
    arguments: Any,
    status: str = TOOL_STATUS_IN_PROGRESS,
) -> dict:
    """A tool_call update body (announces a new tool call)."""
    body: dict[str, Any] = {
        "sessionUpdate": UPDATE_TOOL_CALL,
        "toolCallId": tool_call_id,
        "title": tool_title_for(name, arguments),
        "kind": tool_kind_for(name),
        "status": status,
    }
    if isinstance(arguments, dict):
        body["rawInput"] = arguments
    return body


def tool_call_update_update(
    *,
    tool_call_id: str,
    status: str,
    content: str | None = None,
    raw_output: Any = None,
) -> dict:
    """A tool_call_update update body (status transition + result)."""
    body: dict[str, Any] = {
        "sessionUpdate": UPDATE_TOOL_CALL_UPDATE,
        "toolCallId": tool_call_id,
        "status": status,
    }
    if content is not None:
        body["content"] = [
            {"type": "content", "content": {"type": "text", "text": content}}
        ]
    if raw_output is not None:
        body["rawOutput"] = raw_output
    return body


class UnsupportedContentBlockError(ValueError):
    """Raised when a prompt contains a content block we did not advertise.

    Examples: an `image` block when promptCapabilities.image is false, or an
    embedded `resource` block when embeddedContext is false.
    """


def extract_prompt_text(prompt: Any) -> str:
    """Pull text from an ACP session/prompt 'prompt' field.

    Supported in v1 (per ACP baseline + advertised capabilities):
      - text blocks: emitted as-is
      - resource_link blocks: emitted as a "[name](uri)" reference, or just
        the embedded `text` field if the client provided one. Required by
        the ACP baseline regardless of advertised capabilities.

    Rejected (capability not advertised in initialize_response()):
      - image, audio: raise UnsupportedContentBlockError
      - resource (embedded): raise UnsupportedContentBlockError

    Unknown block types are skipped silently for forward compatibility.
    """
    if isinstance(prompt, str):
        return prompt
    if not isinstance(prompt, list):
        return ""
    parts: list[str] = []
    for block in prompt:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text = block.get("text", "")
            if isinstance(text, str):
                parts.append(text)
        elif btype == "resource_link":
            uri = block.get("uri", "")
            inline = block.get("text", "")
            if isinstance(inline, str) and inline:
                parts.append(inline)
                continue
            label = block.get("title") or block.get("name") or ""
            if isinstance(uri, str) and uri:
                if isinstance(label, str) and label:
                    parts.append(f"[{label}]({uri})")
                else:
                    parts.append(uri)
        elif btype in ("image", "audio", "resource"):
            raise UnsupportedContentBlockError(
                f"content block type {btype!r} is not supported by this agent "
                "(capability not advertised in initialize)"
            )
    return "\n\n".join(p for p in parts if p)

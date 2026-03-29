"""A2A (Agent-to-Agent) protocol types and wire constants.

Centralizes all JSON-RPC method names, REST paths, and lightweight
dataclasses for the A2A v1.0 protocol. If the spec changes method
naming or adds HTTP/REST binding support, only this file changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# --- JSON-RPC method names (v1.0 spec, unnamespaced) ---

METHOD_SEND_MESSAGE = "SendMessage"
METHOD_SEND_STREAMING_MESSAGE = "SendStreamingMessage"
METHOD_GET_TASK = "GetTask"
METHOD_LIST_TASKS = "ListTasks"
METHOD_CANCEL_TASK = "CancelTask"
METHOD_SUBSCRIBE_TO_TASK = "SubscribeToTask"
METHOD_CREATE_PUSH_CONFIG = "CreateTaskPushNotificationConfig"
METHOD_GET_PUSH_CONFIG = "GetTaskPushNotificationConfig"
METHOD_LIST_PUSH_CONFIGS = "ListTaskPushNotificationConfigs"
METHOD_DELETE_PUSH_CONFIG = "DeleteTaskPushNotificationConfig"
METHOD_GET_EXTENDED_CARD = "GetExtendedAgentCard"

# --- HTTP/REST paths (v1.0 proto definition) ---

REST_SEND_MESSAGE = "/message:send"
REST_SEND_STREAMING = "/message:stream"
REST_GET_TASK = "/tasks/{id=*}"
REST_LIST_TASKS = "/tasks"
REST_CANCEL_TASK = "/tasks/{id=*}:cancel"
REST_SUBSCRIBE = "/tasks/{id=*}:subscribe"
REST_EXTENDED_CARD = "/extendedAgentCard"

# --- Discovery ---

AGENT_CARD_PATH = "/.well-known/agent-card.json"

# --- Protocol version ---

A2A_VERSION = "1.0"

# --- Task state constants ---

STATE_WORKING = "working"
STATE_COMPLETED = "completed"
STATE_FAILED = "failed"
STATE_CANCELED = "canceled"
STATE_REJECTED = "rejected"
STATE_INPUT_REQUIRED = "input-required"
STATE_AUTH_REQUIRED = "auth-required"

TERMINAL_STATES = frozenset(
    {STATE_COMPLETED, STATE_FAILED, STATE_CANCELED, STATE_REJECTED}
)
INTERRUPTED_STATES = frozenset({STATE_INPUT_REQUIRED, STATE_AUTH_REQUIRED})

# --- Server name validation (same rules as MCP) ---

_SERVER_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_-]")
_DOUBLE_UNDER_RE = re.compile(r"__+")

# Matches A2A metadata header lines: [contextId=...] or [input-required]
A2A_META_PREFIX = re.compile(r"^\[(?:contextId=|input-required\])")


def validate_server_name(name: str) -> None:
    """Validate an A2A server name. Raises ValueError if invalid."""
    from .report import ConfigError

    if not _SERVER_NAME_RE.match(name):
        raise ConfigError(
            f"A2A server name {name!r} is invalid: must match [a-zA-Z0-9_-]+"
        )
    if "__" in name:
        raise ConfigError(
            f"A2A server name {name!r} must not contain double underscores"
        )


def sanitize_skill_id(name: str) -> str:
    """Sanitize a skill ID for use in namespaced tool names."""
    name = _SANITIZE_RE.sub("_", name)
    name = _DOUBLE_UNDER_RE.sub("_", name)
    return name.strip("_-") or "ask"


# --- camelCase serialization ---


def _to_camel(snake: str) -> str:
    """Convert snake_case to camelCase."""
    parts = snake.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def _convert_wire_keys(obj: dict, key_fn) -> dict:
    """Recursively convert dict keys while preserving existing list handling."""
    result = {}
    for k, v in obj.items():
        converted_key = key_fn(k)
        if isinstance(v, dict):
            result[converted_key] = _convert_wire_keys(v, key_fn)
        elif isinstance(v, list):
            result[converted_key] = [
                _convert_wire_keys(i, key_fn) if isinstance(i, dict) else i for i in v
            ]
        else:
            result[converted_key] = v
    return result


def to_wire(obj: dict) -> dict:
    """Recursively convert dict keys from snake_case to camelCase."""
    return _convert_wire_keys(obj, _to_camel)


def from_wire(obj: dict) -> dict:
    """Recursively convert dict keys from camelCase to snake_case."""
    return _convert_wire_keys(obj, _to_snake)


_CAMEL_RE = re.compile(r"(?<=[a-z0-9])([A-Z])")


def _to_snake(camel: str) -> str:
    """Convert camelCase to snake_case."""
    return _CAMEL_RE.sub(r"_\1", camel).lower()


# --- Dataclasses ---


@dataclass
class Part:
    """A2A message part (text only for now)."""

    type: str = "text"
    text: str = ""


@dataclass
class Message:
    """A2A Message object."""

    role: str = "user"
    parts: list[dict] = field(default_factory=list)
    context_id: str | None = None
    task_id: str | None = None

    def to_wire(self) -> dict:
        d: dict[str, Any] = {"role": self.role, "parts": self.parts}
        if self.context_id is not None:
            d["contextId"] = self.context_id
        if self.task_id is not None:
            d["taskId"] = self.task_id
        return d


@dataclass
class SendMessageConfiguration:
    """Configuration for SendMessage request."""

    accepted_output_modes: list[str] = field(
        default_factory=lambda: ["text/plain", "application/json"]
    )
    return_immediately: bool = False

    def to_wire(self) -> dict:
        return {
            "acceptedOutputModes": self.accepted_output_modes,
            "returnImmediately": self.return_immediately,
        }


@dataclass
class TaskStatus:
    """A2A task status."""

    state: str = "working"
    message: Message | None = None

    @classmethod
    def from_wire(cls, data: dict) -> TaskStatus:
        msg = None
        if "message" in data:
            msg_data = data["message"]
            msg = Message(
                role=msg_data.get("role", "agent"),
                parts=msg_data.get("parts", []),
                context_id=msg_data.get("contextId"),
                task_id=msg_data.get("taskId"),
            )
        return cls(state=data.get("state", "working"), message=msg)


@dataclass
class Artifact:
    """A2A artifact (output from a completed task)."""

    parts: list[dict] = field(default_factory=list)
    name: str | None = None
    description: str | None = None

    @classmethod
    def from_wire(cls, data: dict) -> Artifact:
        return cls(
            parts=data.get("parts", []),
            name=data.get("name"),
            description=data.get("description"),
        )


@dataclass
class Task:
    """A2A Task object (response from SendMessage/GetTask)."""

    id: str = ""
    context_id: str = ""
    status: TaskStatus = field(default_factory=TaskStatus)
    artifacts: list[Artifact] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)

    @classmethod
    def from_wire(cls, data: dict) -> Task:
        artifacts = [Artifact.from_wire(a) for a in data.get("artifacts", [])]
        return cls(
            id=data.get("id", ""),
            context_id=data.get("contextId", ""),
            status=TaskStatus.from_wire(data.get("status", {})),
            artifacts=artifacts,
            messages=data.get("messages", []),
        )

    @property
    def state(self) -> str:
        return self.status.state

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def is_interrupted(self) -> bool:
        return self.state in INTERRUPTED_STATES


@dataclass
class AgentSkill:
    """A skill advertised in an Agent Card."""

    id: str = ""
    name: str = ""
    description: str = ""
    input_modes: list[str] = field(default_factory=lambda: ["text/plain"])
    output_modes: list[str] = field(default_factory=lambda: ["text/plain"])
    examples: list[str] = field(default_factory=list)

    @classmethod
    def from_wire(cls, data: dict) -> AgentSkill:
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            input_modes=data.get("inputModes", ["text/plain"]),
            output_modes=data.get("outputModes", ["text/plain"]),
            examples=data.get("examples", []),
        )


@dataclass
class AgentCard:
    """A2A Agent Card (discovery metadata)."""

    name: str = ""
    description: str = ""
    version: str = "0.1.0"
    url: str = ""
    skills: list[AgentSkill] = field(default_factory=list)
    capabilities: dict = field(default_factory=dict)
    default_input_modes: list[str] = field(default_factory=lambda: ["text/plain"])
    default_output_modes: list[str] = field(default_factory=lambda: ["text/plain"])

    @classmethod
    def from_wire(cls, data: dict) -> AgentCard:
        skills = [AgentSkill.from_wire(s) for s in data.get("skills", [])]
        interfaces = data.get("supportedInterfaces", [])
        url = interfaces[0].get("url", "") if interfaces else ""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            version=data.get("version", "0.1.0"),
            url=url,
            skills=skills,
            capabilities=data.get("capabilities", {}),
            default_input_modes=data.get("defaultInputModes", ["text/plain"]),
            default_output_modes=data.get("defaultOutputModes", ["text/plain"]),
        )


# --- JSON-RPC helpers ---


def jsonrpc_request(method: str, params: dict, req_id: int = 1) -> dict:
    """Build a JSON-RPC 2.0 request envelope."""
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": method,
        "params": params,
    }


def parse_jsonrpc_response(data: dict) -> dict:
    """Extract the result from a JSON-RPC 2.0 response.

    Raises ValueError on JSON-RPC error responses.
    """
    if "error" in data:
        err = data["error"]
        code = err.get("code", -1)
        message = err.get("message", "Unknown error")
        raise ValueError(f"JSON-RPC error {code}: {message}")
    return data.get("result", {})


def extract_text_from_parts(parts: list[dict]) -> str:
    """Extract text content from a list of A2A parts."""
    texts = []
    for part in parts:
        if part.get("type") == "text":
            texts.append(part.get("text", ""))
        elif "text" in part and "type" not in part:
            texts.append(part["text"])
    return "\n".join(texts) if texts else ""


def extract_task_text(task: Task) -> str:
    """Extract all text from a task's artifacts and status message."""
    texts = []
    for artifact in task.artifacts:
        text = extract_text_from_parts(artifact.parts)
        if text:
            texts.append(text)
    if task.status.message:
        text = extract_text_from_parts(task.status.message.parts)
        if text:
            texts.append(text)
    return "\n".join(texts) if texts else "(empty response)"


# --- Agent loop event callback types ---

# Event kinds emitted by run_agent_loop() via event_callback
EVENT_TEXT_CHUNK = "text_chunk"
EVENT_TOOL_START = "tool_start"
EVENT_TOOL_FINISH = "tool_finish"
EVENT_TOOL_ERROR = "tool_error"
EVENT_STATUS_UPDATE = "status_update"


@dataclass
class AgentEvent:
    """Event emitted by the agent loop for streaming consumers."""

    kind: str
    data: dict = field(default_factory=dict)


# --- A2A SSE event types (server-side) ---


@dataclass
class TaskStatusUpdateEvent:
    """SSE event for task status transitions and heartbeats."""

    task_id: str
    context_id: str
    state: str
    message: dict | None = None
    metadata: dict | None = None

    def to_wire(self) -> dict:
        result: dict[str, Any] = {
            "taskId": self.task_id,
            "contextId": self.context_id,
            "status": {"state": self.state},
        }
        if self.message is not None:
            result["status"]["message"] = self.message
        if self.metadata is not None:
            result["metadata"] = self.metadata
        return result


@dataclass
class TaskArtifactUpdateEvent:
    """SSE event for incremental artifact delivery (text chunks, final answer)."""

    task_id: str
    context_id: str
    artifact: dict = field(default_factory=dict)

    def to_wire(self) -> dict:
        return {
            "taskId": self.task_id,
            "contextId": self.context_id,
            "artifact": self.artifact,
        }

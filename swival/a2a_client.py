"""A2A (Agent-to-Agent) client integration for swival.

Manages connections to remote A2A agents and exposes their skills
as tools in OpenAI function-calling format, following the same
pattern as McpManager.
"""

import asyncio
import atexit
import threading
import time
from typing import Any

from .a2a_types import (
    AGENT_CARD_PATH,
    METHOD_GET_TASK,
    METHOD_SEND_MESSAGE,
    AgentCard,
    AgentSkill,
    Message,
    SendMessageConfiguration,
    Task,
    extract_task_text,
    extract_text_from_parts,
    jsonrpc_request,
    parse_jsonrpc_response,
    sanitize_skill_id,
)
from .report import ConfigError

# Polling defaults for non-compliant servers
_POLL_INITIAL_DELAY = 0.5
_POLL_MAX_DELAY = 5.0
_POLL_BACKOFF_FACTOR = 1.5
_DEFAULT_TIMEOUT = 300


class A2aShutdownError(Exception):
    """Raised when call_tool() is invoked during or after shutdown."""


class A2aManager:
    """Manages connections to remote A2A agents.

    Runs an asyncio event loop in a background daemon thread.
    All public methods are synchronous -- they submit coroutines via
    run_coroutine_threadsafe() and block on the future.
    """

    def __init__(self, server_configs: dict[str, dict], verbose: bool = False):
        """
        server_configs: {
            "agent-name": {
                "url": "https://agent.example.com",
                "card_url": "https://...",  # optional override
                "auth_type": "bearer",      # optional
                "auth_token": "sk-...",     # optional
                "timeout": 120,             # optional
            }
        }
        """
        self._server_configs = server_configs
        self._verbose = verbose

        # Background event loop
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

        # A2A state (populated by start())
        self._agent_cards: dict[str, AgentCard] = {}
        self._tool_schemas: dict[str, list[dict]] = {}
        self._tool_map: dict[
            str, tuple[str, str]
        ] = {}  # namespaced -> (agent, skill_id)
        self._degraded: set[str] = set()

        # Lifecycle
        self._closing = False
        self._closed = False

    def _server_start_notice(self, name: str, tool_count: int) -> None:
        """Emit a user-facing startup notice for interactive sessions."""
        if not self._verbose:
            return
        from . import fmt

        fmt.a2a_server_start(name, tool_count)

    def _server_error_notice(self, name: str, error: str) -> None:
        """Emit a user-facing error for interactive sessions."""
        if not self._verbose:
            return
        from . import fmt

        fmt.a2a_server_error(name, error)

    def start(self) -> None:
        """Start background event loop, fetch Agent Cards from all configured agents."""
        if self._closed:
            raise A2aShutdownError("manager is already closed")

        loop_ready = threading.Event()
        self._loop = asyncio.new_event_loop()

        def _run_loop():
            self._loop.call_soon(lambda: loop_ready.set())
            self._loop.run_forever()

        self._thread = threading.Thread(
            target=_run_loop,
            name="swival-a2a-loop",
            daemon=True,
        )
        self._thread.start()
        if not loop_ready.wait(timeout=10):
            raise A2aShutdownError("A2A event loop failed to start")

        for name, config in self._server_configs.items():
            try:
                self._fetch_agent_card(name, config)
            except Exception as e:
                self._server_error_notice(name, str(e))

        self._build_tool_map()
        atexit.register(self.close)

    def list_tools(self) -> list[dict]:
        """Return all A2A tools in OpenAI function-calling format."""
        tools = []
        for schemas in self._tool_schemas.values():
            tools.extend(schemas)
        return tools

    def get_tool_info(self) -> dict[str, list[tuple[str, str]]]:
        """Return {agent_name: [(namespaced_name, description), ...]} for prompt building."""
        info: dict[str, list[tuple[str, str]]] = {}
        for namespaced, (agent, _skill) in self._tool_map.items():
            desc = ""
            for schema in self._tool_schemas.get(agent, []):
                if schema["function"]["name"] == namespaced:
                    desc = schema["function"].get("description", "")
                    break
            info.setdefault(agent, []).append((namespaced, desc))
        return info

    def call_tool(self, namespaced_name: str, arguments: dict) -> tuple[str, bool]:
        """Dispatch to the correct agent and return (result_text, is_error)."""
        if self._closing or self._closed:
            raise A2aShutdownError("manager is shutting down")

        if namespaced_name not in self._tool_map:
            return (f"error: unknown A2A tool: {namespaced_name}", True)

        agent_name, skill_id = self._tool_map[namespaced_name]

        if agent_name in self._degraded:
            return (
                f"error: A2A agent {agent_name!r} is unavailable (connection failed)",
                True,
            )

        config = self._server_configs.get(agent_name, {})
        timeout = config.get("timeout", _DEFAULT_TIMEOUT)

        try:
            result = self._run_sync(
                self._send_message(agent_name, arguments, config),
                timeout=timeout,
            )
            return result
        except A2aShutdownError:
            raise
        except TimeoutError:
            return (
                f"error: A2A agent {agent_name!r} timed out after {timeout}s",
                True,
            )
        except Exception as e:
            self._degraded.add(agent_name)
            return (f"error: A2A agent {agent_name!r} failed: {e}", True)

    def close(self) -> None:
        """Idempotent shutdown."""
        if self._closed:
            return
        self._closing = True

        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)

        self._closed = True
        self._closing = False

    # --- Internal helpers ---

    def _run_sync(self, coro, timeout: float = 30):
        """Submit a coroutine to the background loop and wait."""
        if self._loop is None or not self._loop.is_running():
            raise A2aShutdownError("event loop is not running")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except asyncio.CancelledError:
            raise A2aShutdownError("operation cancelled during shutdown")
        except TimeoutError:
            future.cancel()
            raise

    def _fetch_agent_card(self, name: str, config: dict) -> None:
        """Fetch and parse an agent's Agent Card."""
        import httpx

        base_url = config["url"].rstrip("/")
        card_url = config.get("card_url") or (base_url + AGENT_CARD_PATH)
        headers = self._auth_headers(config)

        try:
            with httpx.Client(timeout=15, headers=headers) as client:
                resp = client.get(card_url)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            raise ConfigError(f"Failed to fetch Agent Card from {card_url}: {e}")

        card = AgentCard.from_wire(data)
        if not card.url:
            card.url = base_url
        self._agent_cards[name] = card

        # Build tool schemas from skills
        schemas = []
        if card.skills:
            for skill in card.skills:
                schema = _skill_to_tool(name, skill)
                schemas.append(schema)
        else:
            # No skills declared: expose a single generic "ask" tool
            generic = AgentSkill(
                id="ask",
                name="ask",
                description=card.description or f"Send a message to {card.name}",
            )
            schemas.append(_skill_to_tool(name, generic))

        self._tool_schemas[name] = schemas

        self._server_start_notice(name, len(schemas))

    async def _send_message(
        self, agent_name: str, arguments: dict, config: dict
    ) -> tuple[str, bool]:
        """Send a message to a remote A2A agent and return (text, is_error)."""
        import httpx

        card = self._agent_cards.get(agent_name)
        if card is None:
            return (f"error: no Agent Card for {agent_name!r}", True)

        message_text = arguments.get("message", "")
        context_id = arguments.get("context_id")
        task_id = arguments.get("task_id")

        # Build A2A Message
        msg = Message(
            role="user",
            parts=[{"type": "text", "text": message_text}],
            context_id=context_id,
            task_id=task_id,
        )

        send_config = SendMessageConfiguration()

        params: dict[str, Any] = {
            "message": msg.to_wire(),
            "configuration": send_config.to_wire(),
        }

        request = jsonrpc_request(METHOD_SEND_MESSAGE, params)
        endpoint = card.url.rstrip("/")
        headers = self._auth_headers(config)
        headers["Content-Type"] = "application/json"
        timeout = config.get("timeout", _DEFAULT_TIMEOUT)

        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            resp = await client.post(endpoint, json=request)
            resp.raise_for_status()
            data = resp.json()

        result = parse_jsonrpc_response(data)

        # A2A responses can be either a Task or a direct Message.
        # Tasks have "id" and "status"; Messages have "role" and "parts".
        if _is_message_response(result):
            return self._format_message_result(result, context_id)

        task = Task.from_wire(result)

        # If the task is terminal or interrupted, return immediately
        if task.is_terminal or task.is_interrupted:
            return self._format_task_result(task)

        # If the task has no ID, we can't poll. Terminal/interrupted
        # tasks were already handled above, so a non-terminal task
        # without an ID is a protocol error.
        if not task.id:
            text = extract_task_text(task)
            return (
                f"error: A2A agent {agent_name!r} returned a non-terminal "
                f"task without an ID (state={task.state!r}): {text}",
                True,
            )

        # Fallback: server didn't block -- poll with GetTask
        return await self._poll_task(agent_name, task.id, config, timeout)

    async def _poll_task(
        self,
        agent_name: str,
        task_id: str,
        config: dict,
        timeout: float,
    ) -> tuple[str, bool]:
        """Poll GetTask until the task reaches a terminal or interrupted state."""
        import httpx

        card = self._agent_cards[agent_name]
        endpoint = card.url.rstrip("/")
        headers = self._auth_headers(config)
        headers["Content-Type"] = "application/json"

        delay = _POLL_INITIAL_DELAY
        deadline = time.monotonic() + timeout

        async with httpx.AsyncClient(timeout=30, headers=headers) as client:
            while time.monotonic() < deadline:
                await asyncio.sleep(delay)

                request = jsonrpc_request(METHOD_GET_TASK, {"id": task_id})
                resp = await client.post(endpoint, json=request)
                resp.raise_for_status()
                data = resp.json()

                result = parse_jsonrpc_response(data)
                task = Task.from_wire(result)

                if task.is_terminal or task.is_interrupted:
                    return self._format_task_result(task)

                delay = min(delay * _POLL_BACKOFF_FACTOR, _POLL_MAX_DELAY)

        return (f"error: A2A agent {agent_name!r} timed out after {timeout}s", True)

    @staticmethod
    def _format_message_result(
        msg_data: dict, context_id: str | None
    ) -> tuple[str, bool]:
        """Format a direct Message response as (text, is_error)."""
        msg = Message(
            role=msg_data.get("role", "agent"),
            parts=msg_data.get("parts", []),
            context_id=msg_data.get("contextId"),
            task_id=msg_data.get("taskId"),
        )
        text = extract_text_from_parts(msg.parts)
        if not text:
            text = "(empty response)"
        msg_ctx = msg.context_id or context_id
        if msg_ctx:
            return (f"[contextId={msg_ctx}]\n{text}", False)
        return (text, False)

    def _format_task_result(self, task: Task) -> tuple[str, bool]:
        """Format a completed/interrupted task as (text, is_error)."""
        text = extract_task_text(task)
        state = task.state

        if state in ("failed", "rejected"):
            return (f"error: {text}", True)

        if state == "input-required":
            header = f"[input-required] contextId={task.context_id} taskId={task.id}"
            return (f"{header}\n{text}", False)

        if state == "auth-required":
            return (f"error: [auth-required] {text}", True)

        # completed, canceled, or other terminal
        if task.context_id:
            return (f"[contextId={task.context_id}]\n{text}", False)
        return (text, False)

    @staticmethod
    def _auth_headers(config: dict) -> dict[str, str]:
        """Build auth headers from config."""
        headers: dict[str, str] = {}
        auth_type = config.get("auth_type", "").lower()
        auth_token = config.get("auth_token", "")
        if auth_type == "bearer" and auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        elif auth_type == "api_key" and auth_token:
            headers["X-API-Key"] = auth_token
        return headers

    def _build_tool_map(self) -> None:
        """Build routing table with collision detection.

        Collisions are handled per-agent: the colliding agent's tools
        are all skipped with a warning, matching MCP's behavior.
        """
        tool_map: dict[str, tuple[str, str]] = {}

        for agent_name, schemas in self._tool_schemas.items():
            agent_collisions = []
            for schema in schemas:
                namespaced = schema["function"]["name"]
                skill_id = schema["function"].get("_a2a_skill_id", namespaced)

                if namespaced in tool_map:
                    existing_agent, existing_skill = tool_map[namespaced]
                    agent_collisions.append(
                        f"  {namespaced!r}: {existing_agent}/{existing_skill} "
                        f"vs {agent_name}/{skill_id}"
                    )
                else:
                    tool_map[namespaced] = (agent_name, skill_id)

            if agent_collisions:
                # Remove all this agent's tools from the map
                for schema in schemas:
                    n = schema["function"]["name"]
                    if tool_map.get(n, (None,))[0] == agent_name:
                        del tool_map[n]
                self._tool_schemas[agent_name] = []
                detail = "\n".join(agent_collisions)
                self._server_error_notice(
                    agent_name,
                    f"tool name collision after sanitization, "
                    f"skipping all its tools:\n{detail}",
                )

        self._tool_map = tool_map


def _is_message_response(data: dict) -> bool:
    """Detect whether a JSON-RPC result is a Message rather than a Task.

    Tasks have "id" and "status" fields; Messages have "role" and "parts".
    """
    if "role" in data and "parts" in data and "status" not in data:
        return True
    return False


def _skill_to_tool(agent_name: str, skill: AgentSkill) -> dict:
    """Convert an A2A skill to an OpenAI function-calling tool schema."""
    skill_id = sanitize_skill_id(skill.id or skill.name or "ask")
    namespaced = f"a2a__{agent_name}__{skill_id}"

    description = f"Skill '{skill.name or skill.id}' on remote agent '{agent_name}'"
    if skill.description:
        description += f": {skill.description}"
    if skill.examples:
        examples_text = "; ".join(skill.examples[:3])
        description += f"\nExamples: {examples_text}"

    return {
        "type": "function",
        "function": {
            "name": namespaced,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to send to the remote agent",
                    },
                    "context_id": {
                        "type": "string",
                        "description": (
                            "Optional. Pass a previous contextId to continue "
                            "a conversation"
                        ),
                    },
                    "task_id": {
                        "type": "string",
                        "description": (
                            "Optional. Pass a previous taskId to resume an "
                            "input-required task"
                        ),
                    },
                },
                "required": ["message"],
            },
            "_a2a_skill_id": skill_id,
        },
    }

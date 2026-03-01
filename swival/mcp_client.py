"""MCP (Model Context Protocol) client integration for swival.

Manages connections to multiple MCP servers and exposes their tools
in OpenAI function-calling format alongside swival's built-in tools.
"""

import asyncio
import atexit
import copy
import json
import logging
import re
import threading
from typing import Any

from .report import ConfigError

logger = logging.getLogger(__name__)

_SERVER_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_-]")
_DOUBLE_UNDER_RE = re.compile(r"__+")


class McpShutdownError(Exception):
    """Raised when call_tool() is invoked during or after shutdown."""


class McpManager:
    """Manages connections to multiple MCP servers.

    Runs an asyncio event loop in a background daemon thread.
    All public methods are synchronous — they submit coroutines via
    run_coroutine_threadsafe() and block on the future.

    Each server gets a long-lived asyncio Task that owns its
    AsyncExitStack from connect through shutdown.  This ensures the
    cancel-scopes created by the MCP SDK's anyio transports are always
    entered and exited inside the same Task, avoiding
    "Attempted to exit cancel scope in a different task" errors.
    """

    def __init__(self, server_configs: dict[str, dict], verbose: bool = False):
        """
        server_configs: {
            "server-name": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-brave-search"],
                "env": {"BRAVE_API_KEY": "your-key-here"},
                # OR for HTTP:
                "url": "http://localhost:8080/mcp",
                "headers": {"Authorization": "Bearer ..."},
            }
        }
        """
        self._server_configs = server_configs
        self._verbose = verbose

        # Background event loop
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

        # MCP state (populated by start())
        self._sessions: dict[str, Any] = {}  # server_name -> ClientSession
        self._tool_schemas: dict[
            str, list[dict]
        ] = {}  # server_name -> [openai schemas]
        self._tool_map: dict[
            str, tuple[str, str]
        ] = {}  # namespaced_name -> (server, orig)
        self._degraded: set[str] = set()  # servers that crashed after startup

        # Per-server lifecycle tasks and their shutdown signals
        self._server_tasks: dict[str, asyncio.Task] = {}
        self._shutdown_events: dict[str, asyncio.Event] = {}

        # Lifecycle flags
        self._closing = False
        self._closed = False

    def start(self) -> None:
        """Start background event loop, connect to all servers."""
        if self._closed:
            raise McpShutdownError("manager is already closed")

        # Start background event loop thread with a barrier to ensure
        # the loop is running before we submit coroutines.
        loop_ready = threading.Event()
        self._loop = asyncio.new_event_loop()

        def _run_loop():
            self._loop.call_soon(lambda: loop_ready.set())
            self._loop.run_forever()

        self._thread = threading.Thread(
            target=_run_loop,
            name="swival-mcp-loop",
            daemon=True,
        )
        self._thread.start()
        if not loop_ready.wait(timeout=10):
            raise McpShutdownError("MCP event loop failed to start")

        from . import fmt as _fmt

        # Connect to each server via a long-lived lifecycle task
        for name, config in self._server_configs.items():
            try:
                self._start_server_task(name, config, timeout=30)
            except Exception as e:
                _fmt.mcp_server_error(name, str(e))

        # Build routing table with collision detection
        self._build_tool_map()

        # Register atexit as last-resort cleanup
        atexit.register(self.close)

    def list_tools(self) -> list[dict]:
        """Return all MCP tools in OpenAI function-calling format."""
        tools = []
        for server_name, schemas in self._tool_schemas.items():
            tools.extend(schemas)
        return tools

    def get_tool_info(self) -> dict[str, list[tuple[str, str]]]:
        """Return {server_name: [(namespaced_name, description), ...]} for prompt building."""
        info: dict[str, list[tuple[str, str]]] = {}
        for namespaced, (server, _orig) in self._tool_map.items():
            desc = ""
            for schema in self._tool_schemas.get(server, []):
                if schema["function"]["name"] == namespaced:
                    desc = schema["function"].get("description", "")
                    break
            info.setdefault(server, []).append((namespaced, desc))
        return info

    def call_tool(self, namespaced_name: str, arguments: dict) -> tuple[str, bool]:
        """Dispatch to the correct server and return (result_text, is_error).

        The boolean flag signals whether the result represents an error,
        avoiding fragile ``result.startswith("error:")`` checks by callers.
        """
        if self._closing or self._closed:
            raise McpShutdownError("manager is shutting down")

        if namespaced_name not in self._tool_map:
            return (f"error: unknown MCP tool: {namespaced_name}", True)

        server_name, original_name = self._tool_map[namespaced_name]

        if server_name in self._degraded:
            return (
                f"error: MCP server {server_name!r} is unavailable (crashed or disconnected)",
                True,
            )

        session = self._sessions.get(server_name)
        if session is None:
            return (f"error: MCP server {server_name!r} has no active session", True)

        try:
            result = self._run_sync(
                session.call_tool(original_name, arguments),
                timeout=120,
            )
            return _normalize_result(result)
        except McpShutdownError:
            raise
        except Exception as e:
            # Mark server as degraded
            self._degraded.add(server_name)
            return (f"error: MCP server {server_name!r} failed: {e}", True)

    def close(self) -> None:
        """Idempotent shutdown."""
        if self._closed:
            return
        self._closing = True

        if self._loop is not None and self._loop.is_running():
            try:
                self._run_sync(self._close_all_sessions(), timeout=10)
            except Exception:
                pass

            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                residual_servers = list(self._sessions.keys())
                logger.warning(
                    "MCP event loop thread did not stop cleanly. "
                    f"Residual thread: {self._thread.name}, "
                    f"servers: {residual_servers}"
                )

        self._closed = True
        self._closing = False

    # --- Internal helpers ---

    def _run_sync(self, coro, timeout: float = 30):
        """Submit a coroutine to the background loop and wait for result."""
        if self._loop is None or not self._loop.is_running():
            raise McpShutdownError("event loop is not running")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except asyncio.CancelledError:
            raise McpShutdownError("operation cancelled during shutdown")
        except TimeoutError:
            future.cancel()
            raise

    def _start_server_task(self, name: str, config: dict, timeout: float = 30) -> None:
        """Launch a long-lived task for one server; block until connected.

        The lifecycle task owns the AsyncExitStack so that connect and
        cleanup always happen inside the same asyncio Task — required
        by anyio's cancel-scope tracking.
        """
        ready = threading.Event()
        startup_error: list[BaseException | None] = [None]
        shutdown_event = asyncio.Event()
        self._shutdown_events[name] = shutdown_event

        async def _launch():
            task = asyncio.current_task()
            assert task is not None
            lifecycle_task = asyncio.create_task(
                self._server_lifecycle(
                    name, config, ready, startup_error, shutdown_event
                ),
                name=f"mcp-{name}",
            )
            self._server_tasks[name] = lifecycle_task

        self._run_sync(_launch(), timeout=5)

        if not ready.wait(timeout=timeout):
            task = self._server_tasks.pop(name, None)
            if task:
                self._loop.call_soon_threadsafe(task.cancel)
            self._shutdown_events.pop(name, None)
            raise TimeoutError(f"MCP server {name!r} startup timed out")

        if startup_error[0] is not None:
            self._server_tasks.pop(name, None)
            self._shutdown_events.pop(name, None)
            raise startup_error[0]

    async def _server_lifecycle(
        self,
        name: str,
        config: dict,
        ready: threading.Event,
        startup_error: list[BaseException | None],
        shutdown_event: asyncio.Event,
    ) -> None:
        """Long-lived task owning one server's connection and exit-stack.

        Connect → wait for shutdown signal → clean up, all within one Task.
        """
        from contextlib import AsyncExitStack
        import mcp

        stack = AsyncExitStack()
        await stack.__aenter__()

        try:
            if "url" in config:
                # HTTP/SSE transport
                from mcp.client.sse import sse_client

                read_stream, write_stream = await stack.enter_async_context(
                    sse_client(
                        url=config["url"],
                        headers=config.get("headers"),
                        timeout=10,
                        sse_read_timeout=300,
                    )
                )
            else:
                # Stdio transport
                params = mcp.StdioServerParameters(
                    command=config["command"],
                    args=config.get("args", []),
                    env=config.get("env"),
                )
                read_stream, write_stream = await stack.enter_async_context(
                    mcp.stdio_client(params)
                )

            session = await stack.enter_async_context(
                mcp.ClientSession(read_stream, write_stream)
            )
            await session.initialize()

            # List tools
            tools_result = await session.list_tools()

            self._sessions[name] = session

            # Convert schemas
            self._tool_schemas[name] = [
                _mcp_tool_to_openai(name, tool) for tool in tools_result.tools
            ]

            from . import fmt

            fmt.mcp_server_start(name, len(tools_result.tools))

            ready.set()

            # Keep running until signalled to shut down
            await shutdown_event.wait()
        except Exception as exc:
            startup_error[0] = exc
            ready.set()  # Unblock the caller even on error
        finally:
            try:
                await asyncio.wait_for(stack.aclose(), timeout=5)
            except TimeoutError:
                logger.warning(
                    f"MCP server {name!r}: graceful close timed out "
                    f"(SDK handles SIGTERM→SIGKILL internally)"
                )
            except Exception as e:
                logger.warning(f"Error closing MCP server {name!r}: {e}")
            self._sessions.pop(name, None)

    async def _close_all_sessions(self) -> None:
        """Signal all server lifecycle tasks to shut down and wait.

        Each task cleans up its own AsyncExitStack in the same Task
        that created it, avoiding cancel-scope cross-task errors.
        """
        for event in self._shutdown_events.values():
            event.set()

        tasks = list(self._server_tasks.values())
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception) and not isinstance(
                    r, asyncio.CancelledError
                ):
                    logger.warning(f"MCP server task error during shutdown: {r}")

        self._server_tasks.clear()
        self._shutdown_events.clear()
        self._sessions.clear()

    def _build_tool_map(self) -> None:
        """Build the routing table with collision detection.

        Collisions are handled per-server: the colliding server's tools
        are all skipped with a warning, but other servers continue.
        """
        tool_map: dict[str, tuple[str, str]] = {}

        for server_name, schemas in self._tool_schemas.items():
            server_collisions = []
            for schema in schemas:
                namespaced = schema["function"]["name"]
                original = schema["function"].get("_mcp_original_name", namespaced)

                if namespaced in tool_map:
                    existing_server, existing_orig = tool_map[namespaced]
                    server_collisions.append(
                        f"  {namespaced!r}: {existing_server}/{existing_orig} vs {server_name}/{original}"
                    )
                else:
                    tool_map[namespaced] = (server_name, original)

            if server_collisions:
                from . import fmt

                # Skip this server — remove all its tools from the map
                for schema in schemas:
                    n = schema["function"]["name"]
                    if tool_map.get(n, (None,))[0] == server_name:
                        del tool_map[n]
                self._tool_schemas[server_name] = []
                detail = "\n".join(server_collisions)
                fmt.mcp_server_error(
                    server_name,
                    f"tool name collision after sanitization, "
                    f"skipping all its tools:\n{detail}",
                )

        self._tool_map = tool_map


def _sanitize_tool_name(name: str) -> str:
    """Sanitize an MCP tool name for use in namespaced identifiers."""
    name = _SANITIZE_RE.sub("_", name)
    name = _DOUBLE_UNDER_RE.sub("_", name)
    return name.strip("_-")


def validate_server_name(name: str) -> None:
    """Validate an MCP server name. Raises ConfigError if invalid."""
    if not _SERVER_NAME_RE.match(name):
        raise ConfigError(
            f"MCP server name {name!r} is invalid: must match [a-zA-Z0-9_-]+"
        )
    if "__" in name:
        raise ConfigError(
            f"MCP server name {name!r} must not contain double underscores"
        )


def _mcp_tool_to_openai(server_name: str, tool) -> dict:
    """Convert an MCP Tool object to OpenAI function-calling format."""
    original_name = tool.name
    sanitized_name = _sanitize_tool_name(original_name)
    namespaced = f"mcp__{server_name}__{sanitized_name}"

    # Convert inputSchema
    schema = _convert_schema(tool.inputSchema if tool.inputSchema else {})

    result = {
        "type": "function",
        "function": {
            "name": namespaced,
            "description": tool.description or f"MCP tool from {server_name}",
            "parameters": schema,
            "_mcp_original_name": original_name,
        },
    }
    return result


def _convert_schema(input_schema: dict) -> dict:
    """Convert MCP inputSchema to OpenAI-compatible parameters.

    Whitelist-of-removals approach: keep everything, only strip
    keys known to cause provider rejections.
    """
    schema = copy.deepcopy(input_schema)

    # Ensure top-level type and properties
    if "type" not in schema:
        schema["type"] = "object"
    if "properties" not in schema:
        schema["properties"] = {}

    # Strip keys that OpenAI rejects
    schema.pop("$schema", None)
    schema.pop("$id", None)

    return schema


def _normalize_result(result) -> tuple[str, bool]:
    """Convert MCP CallToolResult to ``(text, is_error)``.

    The boolean flag surfaces ``result.isError`` and envelope ``ok: false``
    structurally so callers don't need to parse the ``"error:"`` prefix.
    """
    # Fast-path for envelope-style tool responses, while preserving existing
    # handling for non-JSON and non-text blocks.
    for block in result.content:
        if getattr(block, "type", None) != "text":
            continue

        raw_text = block.text
        if not isinstance(raw_text, str):
            continue

        try:
            payload = json.loads(raw_text)
        except (TypeError, json.JSONDecodeError):
            continue

        if not isinstance(payload, dict) or "ok" not in payload:
            continue

        if payload.get("ok") is False:
            error_msg = payload.get("error") or payload.get("message")
            if not error_msg:
                stack = payload.get("stack")
                if isinstance(stack, str):
                    error_msg = stack.splitlines()[0] if stack else ""
            if not error_msg:
                error_msg = "MCP tool returned an error"
            return (f"error: {error_msg}", True)

        if payload.get("ok") is True and "result" in payload:
            return (json.dumps(payload["result"], ensure_ascii=False), False)

    parts = []
    for block in result.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            parts.append(block.text)
        elif block_type == "image":
            mime = getattr(block, "mimeType", "unknown")
            data = getattr(block, "data", "")
            parts.append(f"[image: {mime}, {len(data)} bytes]")
        elif block_type == "audio":
            mime = getattr(block, "mimeType", "unknown")
            data = getattr(block, "data", "")
            parts.append(f"[audio: {mime}, {len(data)} bytes]")
        elif block_type == "resource":
            resource = getattr(block, "resource", None)
            if resource and hasattr(resource, "text") and resource.text:
                parts.append(resource.text)
            else:
                uri = getattr(resource, "uri", "unknown") if resource else "unknown"
                parts.append(f"[resource: {uri}]")
        else:
            # Unknown content type — include type info as placeholder
            parts.append(f"[{block_type or 'unknown'}: unsupported content type]")

    text = "\n".join(parts)

    if result.isError:
        err = f"error: {text}" if text else "error: MCP tool returned an error"
        return (err, True)
    return (text if text else "(empty result)", False)

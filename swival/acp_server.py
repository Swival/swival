"""ACP (Agent Client Protocol) server for swival.

Speaks newline-delimited JSON-RPC 2.0 on stdin/stdout. The editor (Zed,
agent-client-protocol.nvim, etc.) spawns swival as a child process; we
expose each open conversation as an ACP session backed by a swival Session.

v1 surface (intentionally narrow):
  - initialize / authenticate (no-op)
  - session/new
  - session/prompt (request-response, returns when the turn ends)
  - session/cancel (notification)
  - session/update (agent->client notifications: text and tool calls)

Out of scope for v1: session/load, session/request_permission round-trip,
fs/* and terminal/* proxies, MCP-via-ACP, multimodal prompts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .acp_types import (
    ERROR_INTERNAL,
    ERROR_INVALID_PARAMS,
    ERROR_INVALID_REQUEST,
    ERROR_METHOD_NOT_FOUND,
    ERROR_PARSE,
    ERROR_SESSION_NOT_FOUND,
    UnsupportedContentBlockError,
    METHOD_AUTHENTICATE,
    METHOD_INITIALIZE,
    METHOD_SESSION_CANCEL,
    METHOD_SESSION_LOAD,
    METHOD_SESSION_NEW,
    METHOD_SESSION_PROMPT,
    METHOD_SESSION_UPDATE,
    PROTOCOL_VERSION,
    STOP_CANCELLED,
    STOP_END_TURN,
    STOP_MAX_TURN_REQUESTS,
    TOOL_STATUS_COMPLETED,
    TOOL_STATUS_FAILED,
    decode_message,
    encode_message,
    extract_prompt_text,
    initialize_response,
    make_error,
    make_notification,
    make_result,
    session_update_payload,
    text_chunk_update,
    tool_call_update,
    tool_call_update_update,
)
from .a2a_types import (
    EVENT_TEXT_CHUNK,
    EVENT_TOOL_ERROR,
    EVENT_TOOL_FINISH,
    EVENT_TOOL_START,
)
from .session import Session


logger = logging.getLogger(__name__)


_LINE_LIMIT = 16 * 1024 * 1024  # 16 MB single-message ceiling
_TOOL_OUTPUT_PREVIEW = 4096


@dataclass
class AcpSession:
    """Server-side ACP session record."""

    session_id: str
    session: Session
    cwd: Path
    cancel_flag: threading.Event
    in_flight: asyncio.Task | None = None
    event_queue: asyncio.Queue | None = None
    announced_tool_calls: set[str] = field(default_factory=set)


class AcpServer:
    """ACP stdio server. One process serves a single editor connection.

    Construct with the same `session_kwargs` shape used by the A2A server,
    then call `serve()` (blocking) to enter the asyncio main loop.
    """

    def __init__(
        self,
        *,
        session_kwargs: dict,
        log_path: str | None = None,
    ) -> None:
        self._session_kwargs = dict(session_kwargs)
        self._log_path = log_path
        self._sessions: dict[str, AcpSession] = {}
        self._initialized = False
        self._client_capabilities: dict = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stdout_lock: asyncio.Lock | None = None
        self._stdout: Any = None  # binary writer

    def serve(self) -> int:
        """Run the asyncio main loop. Returns process exit code."""
        self._configure_logging()
        try:
            asyncio.run(self._run())
            return 0
        except KeyboardInterrupt:
            return 130
        except Exception:
            logger.exception("ACP server crashed")
            return 1

    def _configure_logging(self) -> None:
        if self._log_path:
            handler = logging.FileHandler(self._log_path)
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            )
            root = logging.getLogger()
            root.addHandler(handler)
            root.setLevel(logging.INFO)
            logger.info("ACP server starting; log file: %s", self._log_path)
        else:
            logging.getLogger().addHandler(logging.NullHandler())

    async def _run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stdout_lock = asyncio.Lock()
        self._stdout = sys.stdout.buffer

        reader = asyncio.StreamReader(limit=_LINE_LIMIT)
        protocol = asyncio.StreamReaderProtocol(reader)
        await self._loop.connect_read_pipe(lambda: protocol, sys.stdin)

        try:
            while True:
                try:
                    line = await reader.readuntil(b"\n")
                except asyncio.IncompleteReadError as exc:
                    if exc.partial:
                        await self._handle_line(exc.partial)
                    break
                except asyncio.LimitOverrunError:
                    logger.error("ACP message exceeded line limit; closing")
                    break
                if not line:
                    break
                await self._handle_line(line)
        finally:
            await self._shutdown_all_sessions()

    async def _shutdown_all_sessions(self) -> None:
        for sess in list(self._sessions.values()):
            sess.cancel_flag.set()
            task = sess.in_flight
            if task is not None and not task.done():
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except (asyncio.TimeoutError, Exception):
                    task.cancel()
        self._sessions.clear()

    async def _handle_line(self, line: bytes) -> None:
        text = line.strip()
        if not text:
            return
        if self._log_path:
            try:
                logger.info("recv: %s", text.decode("utf-8", errors="replace"))
            except Exception:
                pass
        try:
            req = decode_message(text)
        except json.JSONDecodeError as exc:
            await self._send(make_error(None, ERROR_PARSE, f"parse error: {exc}"))
            return
        except ValueError as exc:
            await self._send(make_error(None, ERROR_INVALID_REQUEST, str(exc)))
            return
        try:
            await self._dispatch(req)
        except Exception as exc:
            logger.exception("ACP handler crashed for method %s", req.method)
            if not req.is_notification:
                await self._send(
                    make_error(req.id, ERROR_INTERNAL, f"internal error: {exc}")
                )

    async def _dispatch(self, req) -> None:
        method = req.method
        params = req.params or {}
        if method == METHOD_INITIALIZE:
            await self._handle_initialize(req.id, params)
        elif method == METHOD_AUTHENTICATE:
            await self._handle_authenticate(req.id, params)
        elif method == METHOD_SESSION_NEW:
            await self._handle_session_new(req.id, params)
        elif method == METHOD_SESSION_LOAD:
            await self._send(
                make_error(
                    req.id,
                    ERROR_METHOD_NOT_FOUND,
                    "session/load is not supported by this agent",
                )
            )
        elif method == METHOD_SESSION_PROMPT:
            await self._handle_session_prompt(req.id, params)
        elif method == METHOD_SESSION_CANCEL:
            await self._handle_session_cancel(params)
        else:
            if not req.is_notification:
                await self._send(
                    make_error(
                        req.id, ERROR_METHOD_NOT_FOUND, f"unknown method: {method}"
                    )
                )

    async def _handle_initialize(self, request_id, params: dict) -> None:
        client_pv = params.get("protocolVersion")
        if not isinstance(client_pv, int) or client_pv < 1:
            await self._send(
                make_error(
                    request_id,
                    ERROR_INVALID_PARAMS,
                    "protocolVersion must be a positive integer",
                )
            )
            return
        negotiated = min(client_pv, PROTOCOL_VERSION)
        self._client_capabilities = params.get("clientCapabilities") or {}
        self._initialized = True
        await self._send(
            make_result(request_id, initialize_response(protocol_version=negotiated))
        )

    async def _handle_authenticate(self, request_id, params: dict) -> None:
        await self._send(make_result(request_id, {}))

    async def _handle_session_new(self, request_id, params: dict) -> None:
        if not self._initialized:
            await self._send(
                make_error(
                    request_id,
                    ERROR_INVALID_REQUEST,
                    "client must call initialize before session/new",
                )
            )
            return

        cwd_str = params.get("cwd")
        if not isinstance(cwd_str, str) or not cwd_str:
            await self._send(
                make_error(
                    request_id,
                    ERROR_INVALID_PARAMS,
                    "cwd is required and must be a string",
                )
            )
            return
        cwd = Path(cwd_str).expanduser()
        if not cwd.is_absolute():
            await self._send(
                make_error(
                    request_id, ERROR_INVALID_PARAMS, "cwd must be an absolute path"
                )
            )
            return
        if not cwd.is_dir():
            await self._send(
                make_error(
                    request_id, ERROR_INVALID_PARAMS, f"cwd is not a directory: {cwd}"
                )
            )
            return

        if "mcpServers" not in params:
            await self._send(
                make_error(
                    request_id,
                    ERROR_INVALID_PARAMS,
                    "mcpServers is required (use [] for none)",
                )
            )
            return
        mcp_servers = params.get("mcpServers")
        if not isinstance(mcp_servers, list):
            await self._send(
                make_error(
                    request_id,
                    ERROR_INVALID_PARAMS,
                    "mcpServers must be an array",
                )
            )
            return
        if mcp_servers:
            await self._send(
                make_error(
                    request_id,
                    ERROR_INVALID_PARAMS,
                    "ACP-provided MCP servers are not supported by this agent. "
                    "Configure MCP via swival.toml or --mcp-config and pass an "
                    "empty mcpServers array.",
                )
            )
            return

        kwargs = dict(self._session_kwargs)
        kwargs["base_dir"] = str(cwd)
        try:
            session = Session(**kwargs)
        except Exception as exc:
            await self._send(
                make_error(
                    request_id,
                    ERROR_INTERNAL,
                    f"failed to create swival Session: {exc}",
                )
            )
            return

        cancel_flag = threading.Event()
        session.cancel_flag = cancel_flag
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = AcpSession(
            session_id=session_id,
            session=session,
            cwd=cwd,
            cancel_flag=cancel_flag,
        )
        await self._send(make_result(request_id, {"sessionId": session_id}))

    async def _handle_session_prompt(self, request_id, params: dict) -> None:
        session_id = params.get("sessionId")
        if not isinstance(session_id, str) or session_id not in self._sessions:
            await self._send(
                make_error(
                    request_id,
                    ERROR_SESSION_NOT_FOUND,
                    f"unknown sessionId: {session_id!r}",
                )
            )
            return
        sess = self._sessions[session_id]
        if sess.in_flight is not None and not sess.in_flight.done():
            await self._send(
                make_error(
                    request_id,
                    ERROR_INVALID_REQUEST,
                    "another prompt is already in flight for this session",
                )
            )
            return

        try:
            prompt_text = extract_prompt_text(params.get("prompt"))
        except UnsupportedContentBlockError as exc:
            await self._send(make_error(request_id, ERROR_INVALID_PARAMS, str(exc)))
            return
        if not prompt_text:
            await self._send(
                make_error(
                    request_id,
                    ERROR_INVALID_PARAMS,
                    "prompt must contain at least one non-empty text or resource_link block",
                )
            )
            return

        sess.cancel_flag.clear()
        sess.announced_tool_calls.clear()
        sess.event_queue = asyncio.Queue()
        sess.in_flight = asyncio.create_task(
            self._run_prompt(sess, request_id, prompt_text)
        )

    async def _run_prompt(self, sess: AcpSession, request_id, prompt_text: str) -> None:
        loop = asyncio.get_running_loop()
        queue = sess.event_queue
        assert queue is not None

        def event_callback(kind: str, data: dict) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, (kind, data))
            except RuntimeError:
                pass

        sess.session.event_callback = event_callback

        pump = asyncio.create_task(self._pump_events(sess, queue))

        result_holder: dict[str, Any] = {}

        def run_blocking() -> None:
            try:
                result = sess.session.ask(prompt_text)
                result_holder["result"] = result
            except BaseException as exc:
                result_holder["error"] = exc
            finally:
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, None)
                except RuntimeError:
                    pass

        try:
            await asyncio.to_thread(run_blocking)
        finally:
            try:
                await pump
            except Exception:
                logger.exception("event pump crashed")
            sess.session.event_callback = None

        if "error" in result_holder:
            err = result_holder["error"]
            if sess.cancel_flag.is_set():
                await self._send(
                    make_result(request_id, {"stopReason": STOP_CANCELLED})
                )
            else:
                await self._send(
                    make_error(request_id, ERROR_INTERNAL, f"agent loop failed: {err}")
                )
            return

        result = result_holder.get("result")
        if sess.cancel_flag.is_set():
            stop_reason = STOP_CANCELLED
        elif result is not None and result.exhausted:
            stop_reason = STOP_MAX_TURN_REQUESTS
        else:
            stop_reason = STOP_END_TURN
        await self._send(make_result(request_id, {"stopReason": stop_reason}))

    async def _pump_events(self, sess: AcpSession, queue: asyncio.Queue) -> None:
        while True:
            item = await queue.get()
            if item is None:
                return
            kind, data = item
            for body in self._translate_event(sess, kind, data):
                await self._send(
                    make_notification(
                        METHOD_SESSION_UPDATE,
                        session_update_payload(sess.session_id, body),
                    )
                )

    def _translate_event(self, sess: AcpSession, kind: str, data: dict) -> list[dict]:
        if kind == EVENT_TEXT_CHUNK:
            text = data.get("text") or ""
            if not text:
                return []
            return [text_chunk_update(text)]

        if kind == EVENT_TOOL_START:
            tool_id = data.get("id")
            if not isinstance(tool_id, str) or not tool_id:
                return []
            args = self._parse_arguments(data.get("arguments_raw"))
            sess.announced_tool_calls.add(tool_id)
            return [
                tool_call_update(
                    tool_call_id=tool_id,
                    name=data.get("name", ""),
                    arguments=args,
                )
            ]

        if kind in (EVENT_TOOL_FINISH, EVENT_TOOL_ERROR):
            tool_id = data.get("id")
            if not isinstance(tool_id, str) or not tool_id:
                return []
            updates: list[dict] = []
            if tool_id not in sess.announced_tool_calls:
                args = data.get("arguments")
                if not isinstance(args, dict):
                    args = {}
                updates.append(
                    tool_call_update(
                        tool_call_id=tool_id,
                        name=data.get("name", ""),
                        arguments=args,
                    )
                )
                sess.announced_tool_calls.add(tool_id)

            if kind == EVENT_TOOL_FINISH:
                content = data.get("content")
                if isinstance(content, str) and len(content) > _TOOL_OUTPUT_PREVIEW:
                    content = content[:_TOOL_OUTPUT_PREVIEW]
                updates.append(
                    tool_call_update_update(
                        tool_call_id=tool_id,
                        status=TOOL_STATUS_COMPLETED,
                        content=content if isinstance(content, str) else None,
                    )
                )
            else:
                err = data.get("error")
                updates.append(
                    tool_call_update_update(
                        tool_call_id=tool_id,
                        status=TOOL_STATUS_FAILED,
                        content=err if isinstance(err, str) else None,
                    )
                )
            return updates

        return []

    @staticmethod
    def _parse_arguments(raw: Any) -> dict:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw:
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                return {}
            if isinstance(parsed, dict):
                return parsed
        return {}

    async def _handle_session_cancel(self, params: dict) -> None:
        session_id = params.get("sessionId")
        if not isinstance(session_id, str):
            return
        sess = self._sessions.get(session_id)
        if sess is None:
            return
        sess.cancel_flag.set()

    async def _send(self, message: dict) -> None:
        data = encode_message(message)
        if self._log_path:
            try:
                logger.info(
                    "send: %s", data.rstrip(b"\n").decode("utf-8", errors="replace")
                )
            except Exception:
                pass
        assert self._stdout_lock is not None
        async with self._stdout_lock:
            await asyncio.to_thread(self._write_blocking, data)

    def _write_blocking(self, data: bytes) -> None:
        try:
            self._stdout.write(data)
            self._stdout.flush()
        except (BrokenPipeError, OSError) as exc:
            logger.warning("stdout write failed: %s", exc)


def acp_stdout_is_tty() -> bool:
    """Return True if stdout is connected to a TTY (likely a misconfiguration)."""
    try:
        return os.isatty(sys.stdout.fileno())
    except (OSError, ValueError):
        return False

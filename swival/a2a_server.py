"""A2A (Agent-to-Agent) server for swival.

Exposes a swival Session as an A2A endpoint so other agents can call it.
Uses starlette + uvicorn for lightweight async HTTP serving.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import hashlib
import hmac
import json
import logging
import shutil
import threading
import time
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from .a2a_types import (
    A2A_VERSION,
    AGENT_CARD_PATH,
    EVENT_STATUS_UPDATE,
    EVENT_TEXT_CHUNK,
    EVENT_TOOL_ERROR,
    EVENT_TOOL_FINISH,
    EVENT_TOOL_START,
    METHOD_CANCEL_TASK,
    METHOD_GET_TASK,
    METHOD_LIST_TASKS,
    METHOD_SEND_MESSAGE,
    METHOD_SEND_STREAMING_MESSAGE,
    STATE_CANCELED,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_INPUT_REQUIRED,
    STATE_WORKING,
    TERMINAL_STATES,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    extract_text_from_parts,
)
from .session import Session

# Serve mode is the one place where stdlib logging is the primary diagnostics
# channel. A2A server lifecycle/errors should integrate with uvicorn/ASGI host
# logging rather than the interactive Rich CLI formatter.
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TTL = 3600  # 1 hour
DEFAULT_MAX_SESSIONS = 100
CLEANUP_INTERVAL = 300  # 5 minutes
DEFAULT_MAX_REQUEST_SIZE = 1_048_576  # 1 MB
DEFAULT_MAX_REQUESTS_PER_MINUTE = 60
DEFAULT_MAX_CONCURRENT = 10
DEFAULT_HEARTBEAT_INTERVAL = 15.0  # seconds


# ---------------------------------------------------------------------------
# Task dataclass (server-side, not the same as a2a_types.Task which is
# the client-side wire representation)
# ---------------------------------------------------------------------------


@dataclass
class A2aTask:
    """Server-side task record."""

    id: str
    context_id: str
    status: str = STATE_WORKING
    messages: list[dict] = field(default_factory=list)
    artifacts: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)
    cancel_flag: threading.Event = field(default_factory=threading.Event)

    def to_wire(self) -> dict:
        """Serialize to A2A v1.0 wire format (camelCase)."""
        result: dict[str, Any] = {
            "id": self.id,
            "contextId": self.context_id,
            "status": {"state": self.status},
            "artifacts": self.artifacts,
        }
        # Attach the last agent message to the status if present
        for msg in reversed(self.messages):
            if msg.get("role") == "agent":
                result["status"]["message"] = msg
                break
        return result


# ---------------------------------------------------------------------------
# Rate limiter (sliding window per key)
# ---------------------------------------------------------------------------


class _RateLimiter:
    """Simple in-memory sliding-window rate limiter."""

    def __init__(self, max_requests: int, window: float = 60.0):
        self._max = max_requests
        self._window = window
        self._hits: dict[str, collections.deque] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        q = self._hits.setdefault(key, collections.deque())
        # Evict old entries
        while q and q[0] <= now - self._window:
            q.popleft()
        if len(q) >= self._max:
            return False
        q.append(now)
        # Periodically prune empty keys to prevent unbounded dict growth
        if len(self._hits) > self._max * 10:
            empty = [k for k, v in self._hits.items() if not v]
            for k in empty:
                del self._hits[k]
        return True


# ---------------------------------------------------------------------------
# Agent Card generation
# ---------------------------------------------------------------------------


def _build_skills_list(skills: list[dict] | None) -> list[dict]:
    """Convert skill dicts to A2A AgentSkill wire format (camelCase keys)."""
    if not skills:
        return []
    result = []
    for s in skills:
        entry: dict[str, Any] = {"id": s["id"]}
        if "name" in s:
            entry["name"] = s["name"]
        if "description" in s:
            entry["description"] = s["description"]
        if "examples" in s:
            entry["examples"] = s["examples"]
        result.append(entry)
    return result


def build_agent_card(
    session_kwargs: dict,
    host: str,
    port: int,
    *,
    auth_token: str | None = None,
    name: str | None = None,
    description: str | None = None,
    skills: list[dict] | None = None,
) -> dict:
    """Auto-generate an A2A Agent Card from session config.

    Returns a dict ready to be served as JSON at /.well-known/agent-card.json.
    """
    if name is None:
        provider = session_kwargs.get("provider", "lmstudio")
        model = session_kwargs.get("model") or "default"
        name = f"swival ({provider}/{model})"
    if description is None:
        description = (
            "A coding agent powered by swival. Accepts natural-language tasks "
            "and executes them using tool-augmented LLM reasoning."
        )

    url = f"http://{host}:{port}/"

    card: dict[str, Any] = {
        "name": name,
        "description": description,
        "version": "0.1.0",
        "url": url,
        "capabilities": {
            "streaming": True,
            "pushNotifications": False,
            "extendedAgentCard": False,
        },
        "supportedInterfaces": [
            {
                "protocolBinding": "JSONRPC",
                "protocolVersion": A2A_VERSION,
                "url": url,
            }
        ],
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": _build_skills_list(skills),
    }

    if auth_token:
        card["securitySchemes"] = {
            "bearer": {
                "type": "http",
                "scheme": "bearer",
            }
        }
        card["securityRequirements"] = [{"bearer": []}]

    return card


# ---------------------------------------------------------------------------
# JSON-RPC helpers
# ---------------------------------------------------------------------------


def _jsonrpc_error(req_id: Any, code: int, message: str) -> dict:
    """Build a JSON-RPC 2.0 error response."""
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": code, "message": message},
    }


def _jsonrpc_result(req_id: Any, result: Any) -> dict:
    """Build a JSON-RPC 2.0 success response."""
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": result,
    }


# Standard JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
# A2A-specific error codes
TASK_NOT_FOUND = -32001
CONTEXT_NOT_FOUND = -32002
RATE_LIMITED = -32003


# ---------------------------------------------------------------------------
# A2aServer
# ---------------------------------------------------------------------------


class A2aServer:
    """Serves a swival Session over A2A protocol.

    Each unique contextId gets its own Session instance with persistent
    conversation state (via Session.ask()). Tasks are tracked in memory.
    """

    def __init__(
        self,
        session_kwargs: dict,
        host: str = "0.0.0.0",
        port: int = 8080,
        auth_token: str | None = None,
        ttl: int = DEFAULT_TTL,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        name: str | None = None,
        description: str | None = None,
        skills: list[dict] | None = None,
        max_request_size: int = DEFAULT_MAX_REQUEST_SIZE,
        max_requests_per_minute: int = DEFAULT_MAX_REQUESTS_PER_MINUTE,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    ):
        self.session_kwargs = dict(session_kwargs)
        self.host = host
        self.port = port
        self.auth_token = auth_token
        self.ttl = ttl
        self.max_sessions = max_sessions
        self.heartbeat_interval = heartbeat_interval

        # context_id -> Session
        self._sessions: dict[str, Session] = {}
        # context_id -> last-access monotonic time
        self._session_access: dict[str, float] = {}
        # context_id -> asyncio.Lock (serialize per-context calls)
        self._context_locks: dict[str, asyncio.Lock] = {}

        # task_id -> A2aTask
        self._tasks: dict[str, A2aTask] = {}
        # context_id -> [task_id, ...]
        self._context_tasks: dict[str, list[str]] = {}

        # Rate limiting & concurrency
        self._rate_limiter = _RateLimiter(max_requests_per_minute)
        self._max_request_size = max_request_size
        self._concurrency_sem: asyncio.Semaphore | None = None
        self._max_concurrent = max_concurrent

        # Contexts with in-flight ask() calls — protected from eviction/expiry
        self._active_contexts: set[str] = set()

        # Agent card (built once)
        self._agent_card = build_agent_card(
            session_kwargs,
            host,
            port,
            auth_token=auth_token,
            name=name,
            description=description,
            skills=skills,
        )

        self._cleanup_task: asyncio.Task | None = None
        self._app: Starlette | None = None

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _create_session(self, context_id: str) -> Session:
        """Create a new Session for a context, respecting max_sessions via LRU.

        Raises RuntimeError if the hard cap is reached and no idle session
        can be evicted (all existing contexts are actively processing).
        """
        if len(self._sessions) >= self.max_sessions:
            self._evict_lru()
        if len(self._sessions) >= self.max_sessions:
            raise RuntimeError(
                f"Session limit reached ({self.max_sessions}): "
                "all contexts are actively processing"
            )
        # Per-context scratch directory for race-free temp files (todo, cmd_output).
        # Use a hash of the contextId to avoid path traversal from client-supplied
        # values (contextId can be arbitrary strings like "../../tmp/pwn").
        safe_id = hashlib.sha256(context_id.encode()).hexdigest()[:16]
        base_dir = self.session_kwargs.get("base_dir", ".")
        scratch_dir = str(Path(base_dir) / ".swival" / "contexts" / safe_id)
        # Disable continue-here: it exists for humans resuming interrupted CLI
        # sessions; A2A clients retry via the protocol.
        kwargs = {
            **self.session_kwargs,
            "scratch_dir": scratch_dir,
            "continue_here": False,
        }
        session = Session(**kwargs)
        self._sessions[context_id] = session
        self._session_access[context_id] = time.monotonic()
        logger.info("Created session for context %s", context_id)
        return session

    def _get_or_create_session(self, context_id: str) -> Session:
        """Get existing session or create a new one."""
        session = self._sessions.get(context_id)
        if session is None:
            session = self._create_session(context_id)
        self._session_access[context_id] = time.monotonic()
        return session

    def _get_context_lock(self, context_id: str) -> asyncio.Lock:
        """Get or create the per-context asyncio lock."""
        if context_id not in self._context_locks:
            self._context_locks[context_id] = asyncio.Lock()
        return self._context_locks[context_id]

    async def _streaming_cleanup(
        self,
        ask_future: asyncio.Future,
        lock: asyncio.Lock,
        task: A2aTask,
        context_id: str,
    ) -> None:
        """Wait for a background ask_future, finalize the task, then release resources.

        Spawned as an independent task from the SSE generator's finally
        block on client disconnect. Because it's a separate task (not
        shielded inside the cancelled generator), it survives
        cancellation and guarantees the lock and semaphore are held until
        the underlying thread actually finishes.
        """
        try:
            result = await ask_future
            self._finalize_task(task, result, context_id)
        except Exception as exc:
            logger.error(
                "Disconnected session.ask() failed for context %s: %s",
                context_id,
                exc,
                exc_info=True,
            )
            task.status = STATE_FAILED
            task.updated_at = time.monotonic()
            agent_msg = {
                "role": "agent",
                "parts": [{"type": "text", "text": f"Internal error: {exc}"}],
                "contextId": context_id,
                "taskId": task.id,
            }
            task.messages.append(agent_msg)
        finally:
            self._active_contexts.discard(context_id)
            lock.release()
            if self._concurrency_sem is not None:
                self._concurrency_sem.release()

    def _evict_lru(self) -> None:
        """Evict the least-recently-used idle session to make room."""
        # Only consider contexts that are not actively processing
        candidates = {
            ctx: ts
            for ctx, ts in self._session_access.items()
            if ctx not in self._active_contexts
        }
        if not candidates:
            return
        oldest_ctx = min(candidates, key=candidates.get)  # type: ignore[arg-type]
        self._remove_context(oldest_ctx)
        logger.info("Evicted LRU session for context %s", oldest_ctx)

    def _remove_context(self, context_id: str) -> None:
        """Remove a context and all its associated state."""
        session = self._sessions.pop(context_id, None)
        if session is not None:
            # Clean up per-context scratch directory (todo, cmd_output, etc.)
            if session.scratch_dir:
                shutil.rmtree(session.scratch_dir, ignore_errors=True)
            # Best-effort cleanup
            try:
                session.__exit__(None, None, None)
            except Exception:
                logger.debug(
                    "Error closing session for context %s", context_id, exc_info=True
                )
        self._session_access.pop(context_id, None)
        self._context_locks.pop(context_id, None)
        # Remove associated tasks
        task_ids = self._context_tasks.pop(context_id, [])
        for tid in task_ids:
            self._tasks.pop(tid, None)

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def _create_task(self, context_id: str) -> A2aTask:
        """Create a new task for a context."""
        task_id = str(uuid.uuid4())
        now = time.monotonic()
        task = A2aTask(
            id=task_id,
            context_id=context_id,
            status=STATE_WORKING,
            created_at=now,
            updated_at=now,
        )
        self._tasks[task_id] = task
        self._context_tasks.setdefault(context_id, []).append(task_id)
        return task

    # ------------------------------------------------------------------
    # TTL cleanup
    # ------------------------------------------------------------------

    async def _cleanup_loop(self) -> None:
        """Background coroutine that periodically removes expired sessions/tasks."""
        while True:
            await asyncio.sleep(CLEANUP_INTERVAL)
            try:
                self._cleanup_expired()
            except Exception:
                logger.debug("Cleanup error", exc_info=True)

    def _cleanup_expired(self) -> None:
        """Remove sessions and tasks older than TTL, plus orphaned locks."""
        now = time.monotonic()
        expired = [
            ctx_id
            for ctx_id, last_access in self._session_access.items()
            if now - last_access > self.ttl and ctx_id not in self._active_contexts
        ]
        for ctx_id in expired:
            self._remove_context(ctx_id)
            logger.info("Expired session for context %s", ctx_id)
        # Clean up orphaned locks (contexts that were removed but locks linger)
        orphaned = set(self._context_locks) - set(self._sessions)
        for ctx_id in orphaned:
            self._context_locks.pop(ctx_id, None)

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _check_auth(self, request: Request) -> str | None:
        """Check bearer auth if configured. Returns error message or None."""
        if not self.auth_token:
            return None
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer "):
            return "Missing or invalid Authorization header"
        token = auth[7:]
        if not hmac.compare_digest(token, self.auth_token):
            return "Invalid bearer token"
        return None

    def _rate_limit_key(self, request: Request) -> str:
        """Derive the rate-limit key from the request."""
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            return f"token:{auth[7:20]}"
        return f"ip:{request.client.host}" if request.client else "ip:unknown"

    # ------------------------------------------------------------------
    # A2A message extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_message_text(message: dict) -> str:
        """Extract plain text from an A2A message dict."""
        return extract_text_from_parts(message.get("parts", []))

    # ------------------------------------------------------------------
    # Shared message/task setup (used by both blocking and streaming)
    # ------------------------------------------------------------------

    def _validate_message(
        self, params: dict, req_id: Any
    ) -> tuple[dict | None, str, str | None, dict | None]:
        """Validate SendMessage params and resolve context/task IDs.

        Returns (error_or_None, context_id, task_id_or_None, msg_info).
        Does NOT mutate any task state — callers do that under the lock.
        """
        message = params.get("message")
        if not message:
            return (
                _jsonrpc_error(req_id, INVALID_PARAMS, "Missing 'message' in params"),
                "",
                None,
                None,
            )

        message_text = self._extract_message_text(message)
        if not message_text.strip():
            return (
                _jsonrpc_error(req_id, INVALID_PARAMS, "Empty message text"),
                "",
                None,
                None,
            )

        context_id = message.get("contextId") or params.get("contextId")
        task_id = message.get("taskId") or params.get("taskId")

        # Resolve context from existing task if resuming
        if task_id:
            existing_task = self._tasks.get(task_id)
            if existing_task is not None:
                context_id = existing_task.context_id

        if not context_id:
            context_id = str(uuid.uuid4())

        msg_info = {
            "text": message_text,
            "parts": message.get("parts", [{"type": "text", "text": message_text}]),
        }
        return None, context_id, task_id, msg_info

    def _activate_task(
        self, context_id: str, task_id: str | None, msg_info: dict
    ) -> A2aTask:
        """Create or reuse a task and record the user message.

        Must be called under the per-context lock.
        """
        existing_task: A2aTask | None = None
        if task_id:
            existing_task = self._tasks.get(task_id)

        if existing_task and existing_task.status not in TERMINAL_STATES:
            task = existing_task
            task.status = STATE_WORKING
            task.cancel_flag = threading.Event()
            task.updated_at = time.monotonic()
        else:
            task = self._create_task(context_id)

        user_msg = {
            "role": "user",
            "parts": msg_info["parts"],
            "contextId": context_id,
            "taskId": task.id,
        }
        task.messages.append(user_msg)
        return task

    def _finalize_task(self, task: A2aTask, result, context_id: str) -> None:
        """Update task status and record the agent response after session.ask()."""
        answer_text = result.answer or ""

        if task.cancel_flag.is_set():
            task.status = STATE_CANCELED
        elif result.exhausted and result.answer is None:
            task.status = STATE_INPUT_REQUIRED
        elif result.exhausted:
            task.status = STATE_FAILED
        else:
            task.status = STATE_COMPLETED

        now = time.monotonic()
        task.updated_at = now
        # Refresh session access so the context doesn't look stale after
        # a long-running task completes.
        self._session_access[context_id] = now

        agent_msg = {
            "role": "agent",
            "parts": [{"type": "text", "text": answer_text}],
            "contextId": context_id,
            "taskId": task.id,
        }
        task.messages.append(agent_msg)

        if answer_text:
            task.artifacts.append({"parts": [{"type": "text", "text": answer_text}]})

    # ------------------------------------------------------------------
    # JSON-RPC method handlers
    # ------------------------------------------------------------------

    async def _handle_send_message(self, params: dict, req_id: Any) -> dict:
        """Handle SendMessage JSON-RPC method."""
        err, context_id, task_id, msg_info = self._validate_message(params, req_id)
        if err is not None:
            return err

        lock = self._get_context_lock(context_id)
        async with lock:
            try:
                session = self._get_or_create_session(context_id)
            except RuntimeError as exc:
                # Clean up the orphaned lock if no session was created
                if context_id not in self._sessions:
                    self._context_locks.pop(context_id, None)
                return _jsonrpc_error(req_id, RATE_LIMITED, str(exc))

            task = self._activate_task(context_id, task_id, msg_info)

            # Wire cancel flag into session
            session.cancel_flag = task.cancel_flag
            self._active_contexts.add(context_id)

            try:
                result = await asyncio.to_thread(session.ask, msg_info["text"])
            except Exception as exc:
                logger.error(
                    "Session.ask() failed for context %s: %s",
                    context_id,
                    exc,
                    exc_info=True,
                )
                task.status = STATE_FAILED
                task.updated_at = time.monotonic()
                self._session_access[context_id] = task.updated_at
                agent_msg = {
                    "role": "agent",
                    "parts": [{"type": "text", "text": f"Internal error: {exc}"}],
                    "contextId": context_id,
                    "taskId": task.id,
                }
                task.messages.append(agent_msg)
                self._active_contexts.discard(context_id)
                return _jsonrpc_result(req_id, task.to_wire())
            finally:
                session.cancel_flag = None

            # Finalize before clearing active flag so the access timestamp
            # is refreshed while the context is still protected.
            self._finalize_task(task, result, context_id)
            self._active_contexts.discard(context_id)

        return _jsonrpc_result(req_id, task.to_wire())

    async def _handle_send_streaming_message(
        self, params: dict, req_id: Any
    ) -> dict | StreamingResponse:
        """Handle SendStreamingMessage JSON-RPC method via SSE."""
        err, context_id, task_id, msg_info = self._validate_message(params, req_id)
        if err is not None:
            return err

        # Acquire the per-context lock eagerly and admit the session
        # before entering the generator.  On hard-cap rejection we
        # return a normal JSON-RPC error (consistent with the blocking
        # path).  On success, we pass the already-held lock and session
        # into the generator so there is no gap where another request
        # can slip in or evict the session.
        lock = self._get_context_lock(context_id)
        await lock.acquire()
        try:
            session = self._get_or_create_session(context_id)
        except RuntimeError as exc:
            lock.release()
            # Clean up the orphaned lock if no session was created
            if context_id not in self._sessions:
                self._context_locks.pop(context_id, None)
            return _jsonrpc_error(req_id, RATE_LIMITED, str(exc))

        # Mark active immediately so cross-context eviction cannot
        # remove this just-admitted session before the generator starts.
        self._active_contexts.add(context_id)

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def event_callback(kind: str, data: dict) -> None:
            """Thread-safe callback that pushes events to the async queue."""
            loop.call_soon_threadsafe(queue.put_nowait, (kind, data))

        async def sse_generator():
            """Generate SSE events from the agent loop."""
            # The lock is already held by the caller — we own it from
            # here and release it in the finally block (or hand it off
            # to _streaming_cleanup on disconnect).
            task = self._activate_task(context_id, task_id, msg_info)
            session.event_callback = event_callback
            session.cancel_flag = task.cancel_flag

            ask_future = asyncio.ensure_future(
                asyncio.to_thread(session.ask, msg_info["text"])
            )

            status_evt = TaskStatusUpdateEvent(
                task_id=task.id,
                context_id=context_id,
                state=STATE_WORKING,
            )
            yield _sse_frame("TaskStatusUpdateEvent", status_evt.to_wire())

            last_event_time = time.monotonic()
            # Track the last text delivered via text_chunk so we can
            # deduplicate the post-loop final artifact (only suppress if
            # the exact same text was already streamed).
            last_streamed_text: str | None = None

            try:
                while not ask_future.done():
                    try:
                        kind, data = await asyncio.wait_for(
                            queue.get(),
                            timeout=self.heartbeat_interval,
                        )
                        last_event_time = time.monotonic()
                        for frame in _map_event_to_sse(kind, data, task.id, context_id):
                            yield frame
                        if kind == EVENT_TEXT_CHUNK:
                            last_streamed_text = data.get("text", "")

                    except asyncio.TimeoutError:
                        idle_duration = time.monotonic() - last_event_time
                        heartbeat = TaskStatusUpdateEvent(
                            task_id=task.id,
                            context_id=context_id,
                            state=STATE_WORKING,
                            metadata={
                                "type": "heartbeat",
                                "idle": round(idle_duration, 1),
                            },
                        )
                        yield _sse_frame(
                            "TaskStatusUpdateEvent",
                            heartbeat.to_wire(),
                        )
                        last_event_time = time.monotonic()

                # Drain remaining events from the queue
                while not queue.empty():
                    try:
                        kind, data = queue.get_nowait()
                        for frame in _map_event_to_sse(kind, data, task.id, context_id):
                            yield frame
                        if kind == EVENT_TEXT_CHUNK:
                            last_streamed_text = data.get("text", "")
                    except asyncio.QueueEmpty:
                        break

                # Get the result
                try:
                    result = ask_future.result()
                except Exception as exc:
                    logger.error(
                        "Session.ask() failed for context %s: %s",
                        context_id,
                        exc,
                        exc_info=True,
                    )
                    task.status = STATE_FAILED
                    task.updated_at = time.monotonic()
                    agent_msg = {
                        "role": "agent",
                        "parts": [{"type": "text", "text": f"Internal error: {exc}"}],
                        "contextId": context_id,
                        "taskId": task.id,
                    }
                    task.messages.append(agent_msg)
                    final_evt = TaskStatusUpdateEvent(
                        task_id=task.id,
                        context_id=context_id,
                        state=STATE_FAILED,
                        message=agent_msg,
                    )
                    yield _sse_frame(
                        "TaskStatusUpdateEvent",
                        final_evt.to_wire(),
                    )
                    return

                self._finalize_task(task, result, context_id)

                # Emit final artifact unless the exact same text was
                # already streamed via text_chunk.
                answer_text = result.answer or ""
                if answer_text and answer_text != last_streamed_text:
                    final_artifact = TaskArtifactUpdateEvent(
                        task_id=task.id,
                        context_id=context_id,
                        artifact={
                            "parts": [{"type": "text", "text": answer_text}],
                        },
                    )
                    yield _sse_frame(
                        "TaskArtifactUpdateEvent",
                        final_artifact.to_wire(),
                    )

                # Emit final status
                final_status = TaskStatusUpdateEvent(
                    task_id=task.id,
                    context_id=context_id,
                    state=task.status,
                )
                yield _sse_frame(
                    "TaskStatusUpdateEvent",
                    final_status.to_wire(),
                )

            finally:
                session.event_callback = None
                if not ask_future.done():
                    # Client disconnected while ask() is still running.
                    # Signal cancellation and hand off the cleanup to a
                    # background task that will wait for the thread to
                    # finish, finalize the task, then release the lock
                    # and semaphore.  This avoids the "shield +
                    # CancelledError" problem: the cleanup task is
                    # independent and won't be cancelled when the
                    # generator is torn down.  _active_contexts is
                    # cleared inside _streaming_cleanup once done.
                    task.cancel_flag.set()
                    session.cancel_flag = None
                    asyncio.ensure_future(
                        self._streaming_cleanup(ask_future, lock, task, context_id)
                    )
                else:
                    session.cancel_flag = None
                    self._active_contexts.discard(context_id)
                    lock.release()
                    if self._concurrency_sem is not None:
                        self._concurrency_sem.release()

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def _handle_get_task(self, params: dict, req_id: Any) -> dict:
        """Handle GetTask JSON-RPC method."""
        task_id = params.get("id")
        if not task_id:
            return _jsonrpc_error(req_id, INVALID_PARAMS, "Missing 'id' in params")

        task = self._tasks.get(task_id)
        if task is None:
            return _jsonrpc_error(req_id, TASK_NOT_FOUND, f"Task not found: {task_id}")

        return _jsonrpc_result(req_id, task.to_wire())

    async def _handle_list_tasks(self, params: dict, req_id: Any) -> dict:
        """Handle ListTasks JSON-RPC method."""
        context_id = params.get("contextId")
        if context_id:
            task_ids = self._context_tasks.get(context_id, [])
            tasks = [
                self._tasks[tid].to_wire() for tid in task_ids if tid in self._tasks
            ]
        else:
            tasks = [t.to_wire() for t in self._tasks.values()]
        return _jsonrpc_result(req_id, tasks)

    async def _handle_cancel_task(self, params: dict, req_id: Any) -> dict:
        """Handle CancelTask JSON-RPC method."""
        task_id = params.get("id")
        if not task_id:
            return _jsonrpc_error(req_id, INVALID_PARAMS, "Missing 'id' in params")

        task = self._tasks.get(task_id)
        if task is None:
            return _jsonrpc_error(req_id, TASK_NOT_FOUND, f"Task not found: {task_id}")

        if task.status in TERMINAL_STATES:
            return _jsonrpc_error(
                req_id, INVALID_REQUEST, f"Task is already terminal: {task.status}"
            )

        # Signal cancellation to the agent loop
        task.cancel_flag.set()
        task.status = STATE_CANCELED
        task.updated_at = time.monotonic()

        return _jsonrpc_result(req_id, task.to_wire())

    # ------------------------------------------------------------------
    # HTTP handlers
    # ------------------------------------------------------------------

    async def _handle_agent_card(self, request: Request) -> JSONResponse:
        """Serve the Agent Card at /.well-known/agent-card.json."""
        return JSONResponse(self._agent_card)

    async def _handle_jsonrpc(
        self, request: Request
    ) -> JSONResponse | StreamingResponse:
        """Handle JSON-RPC 2.0 requests at POST /."""
        # Auth check
        auth_err = self._check_auth(request)
        if auth_err:
            return JSONResponse(
                _jsonrpc_error(None, -32000, auth_err),
                status_code=401,
            )

        # Rate limiting
        rl_key = self._rate_limit_key(request)
        if not self._rate_limiter.allow(rl_key):
            return JSONResponse(
                _jsonrpc_error(None, RATE_LIMITED, "Rate limit exceeded"),
                status_code=429,
            )

        # Request size check
        content_length = request.headers.get("content-length")
        try:
            cl_int = int(content_length) if content_length else 0
        except (ValueError, TypeError):
            return JSONResponse(
                _jsonrpc_error(None, INVALID_REQUEST, "Invalid Content-Length header"),
                status_code=400,
            )
        if cl_int > self._max_request_size:
            return JSONResponse(
                _jsonrpc_error(
                    None,
                    INVALID_REQUEST,
                    f"Request body too large (max {self._max_request_size} bytes)",
                ),
                status_code=413,
            )

        # Parse body
        try:
            raw = await request.body()
            if len(raw) > self._max_request_size:
                return JSONResponse(
                    _jsonrpc_error(
                        None,
                        INVALID_REQUEST,
                        f"Request body too large (max {self._max_request_size} bytes)",
                    ),
                    status_code=413,
                )
            body = json.loads(raw)
        except Exception:
            return JSONResponse(
                _jsonrpc_error(None, PARSE_ERROR, "Invalid JSON"),
                status_code=400,
            )

        # Validate JSON-RPC envelope
        if not isinstance(body, dict):
            return JSONResponse(
                _jsonrpc_error(None, INVALID_REQUEST, "Request must be a JSON object"),
                status_code=400,
            )

        req_id = body.get("id")
        method = body.get("method")
        params = body.get("params", {})

        if not isinstance(params, dict):
            return JSONResponse(
                _jsonrpc_error(req_id, INVALID_PARAMS, "params must be a JSON object"),
                status_code=200,
            )

        if body.get("jsonrpc") != "2.0":
            return JSONResponse(
                _jsonrpc_error(
                    req_id, INVALID_REQUEST, "Missing or invalid jsonrpc version"
                ),
                status_code=400,
            )

        if not method:
            return JSONResponse(
                _jsonrpc_error(req_id, INVALID_REQUEST, "Missing method"),
                status_code=400,
            )

        # Concurrency limit for message-processing methods
        if method in (METHOD_SEND_MESSAGE, METHOD_SEND_STREAMING_MESSAGE):
            if self._concurrency_sem is None:
                self._concurrency_sem = asyncio.Semaphore(self._max_concurrent)
            if not self._concurrency_sem._value:  # noqa: SLF001
                return JSONResponse(
                    _jsonrpc_error(
                        req_id,
                        RATE_LIMITED,
                        f"Too many concurrent requests (max {self._max_concurrent})",
                    ),
                    status_code=429,
                )
            # For blocking SendMessage, hold sem for the full request
            if method == METHOD_SEND_MESSAGE:
                async with self._concurrency_sem:
                    return await self._dispatch_method(method, params, req_id)
            # For streaming, acquire now; release is inside the generator's
            # finally block so the slot stays held for the stream's lifetime.
            await self._concurrency_sem.acquire()
            resp = await self._dispatch_method(method, params, req_id)
            if not isinstance(resp, StreamingResponse):
                # Error path (validation failure) — release immediately
                self._concurrency_sem.release()
            return resp

        return await self._dispatch_method(method, params, req_id)

    async def _dispatch_method(
        self, method: str, params: dict, req_id: Any
    ) -> JSONResponse | StreamingResponse:
        """Route to the appropriate method handler."""
        if method == METHOD_SEND_MESSAGE:
            result = await self._handle_send_message(params, req_id)
        elif method == METHOD_SEND_STREAMING_MESSAGE:
            result = await self._handle_send_streaming_message(params, req_id)
            if isinstance(result, StreamingResponse):
                return result
        elif method == METHOD_GET_TASK:
            result = await self._handle_get_task(params, req_id)
        elif method == METHOD_LIST_TASKS:
            result = await self._handle_list_tasks(params, req_id)
        elif method == METHOD_CANCEL_TASK:
            result = await self._handle_cancel_task(params, req_id)
        else:
            result = _jsonrpc_error(
                req_id, METHOD_NOT_FOUND, f"Unknown method: {method}"
            )

        return JSONResponse(result)

    # ------------------------------------------------------------------
    # App construction and serve()
    # ------------------------------------------------------------------

    def _build_app(self) -> Starlette:
        """Build the Starlette ASGI application."""
        routes = [
            Route(AGENT_CARD_PATH, self._handle_agent_card, methods=["GET"]),
            Route("/", self._handle_jsonrpc, methods=["POST"]),
        ]

        @contextlib.asynccontextmanager
        async def lifespan(app):
            # Startup
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info(
                "A2A server starting on %s:%d (TTL=%ds, max_sessions=%d)",
                self.host,
                self.port,
                self.ttl,
                self.max_sessions,
            )
            yield
            # Shutdown
            if self._cleanup_task is not None:
                self._cleanup_task.cancel()
                self._cleanup_task = None
            for ctx_id in list(self._sessions):
                self._remove_context(ctx_id)
            logger.info("A2A server shut down, all sessions closed")

        return Starlette(routes=routes, lifespan=lifespan)

    @property
    def app(self) -> Starlette:
        """The ASGI application (for testing or external ASGI servers)."""
        if self._app is None:
            self._app = self._build_app()
        return self._app

    def serve(self) -> None:
        """Start the A2A server (blocking)."""
        import uvicorn

        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
        )


# ---------------------------------------------------------------------------
# SSE formatting
# ---------------------------------------------------------------------------


def _sse_frame(event_type: str, data: dict) -> str:
    """Format an SSE frame with event type and JSON data."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _map_event_to_sse(
    kind: str, data: dict, task_id: str, context_id: str
) -> list[str]:
    """Map an agent loop event to SSE frame(s)."""
    if kind == EVENT_TEXT_CHUNK:
        evt = TaskArtifactUpdateEvent(
            task_id=task_id,
            context_id=context_id,
            artifact={
                "parts": [{"type": "text", "text": data.get("text", "")}],
            },
        )
        return [_sse_frame("TaskArtifactUpdateEvent", evt.to_wire())]

    if kind in (EVENT_TOOL_START, EVENT_TOOL_FINISH, EVENT_TOOL_ERROR):
        meta: dict[str, Any] = {
            "type": kind,
            "tool": data.get("name", ""),
        }
        if "elapsed" in data:
            meta["elapsed"] = data["elapsed"]
        if "error" in data:
            meta["error"] = data["error"]
        evt = TaskStatusUpdateEvent(
            task_id=task_id,
            context_id=context_id,
            state=STATE_WORKING,
            metadata=meta,
        )
        return [_sse_frame("TaskStatusUpdateEvent", evt.to_wire())]

    if kind == EVENT_STATUS_UPDATE:
        # Pass through the full event data as metadata so subtypes
        # (reasoning, cancelled, progress) are preserved for clients.
        meta = dict(data)
        meta.setdefault("type", "progress")
        evt = TaskStatusUpdateEvent(
            task_id=task_id,
            context_id=context_id,
            state=STATE_WORKING,
            metadata=meta,
        )
        return [_sse_frame("TaskStatusUpdateEvent", evt.to_wire())]

    return []

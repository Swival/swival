"""Parallel subagent support: spawn independent agent loops in threads."""

import threading
import time
from dataclasses import dataclass, field

from . import fmt
from .thinking import ThinkingState
from .todo import TodoState
from .snapshot import SnapshotState
from .tracker import FileAccessTracker


_SUBAGENT_TOOLS = {"spawn_subagent", "check_subagents"}
_MAX_CONCURRENT = 4

# Keys from loop_kwargs that represent per-run mutable state or
# non-shareable resources. Excluded when building the subagent template.
SA_TEMPLATE_EXCLUDE = frozenset(
    {
        "thinking_state",
        "todo_state",
        "snapshot_state",
        "file_tracker",
        "compaction_state",
        "cache",
        "cancel_flag",
        "event_callback",
        "report",
    }
)

SPAWN_SUBAGENT_TOOL = {
    "type": "function",
    "function": {
        "name": "spawn_subagent",
        "description": (
            "Launch an independent subagent to work on a task in parallel. "
            "The subagent has access to all file and search tools but cannot spawn "
            "its own subagents. Use for tasks that are independent of your current "
            "work. Returns a subagent ID for tracking."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "Complete task description. Include all necessary context "
                        "— the subagent has no access to your conversation history."
                    ),
                },
                "max_turns": {
                    "type": "integer",
                    "description": "Maximum turns for the subagent (default: 30).",
                    "default": 30,
                },
                "system_hint": {
                    "type": "string",
                    "description": (
                        "Optional extra instructions prepended to the subagent's "
                        "system prompt."
                    ),
                },
            },
            "required": ["task"],
        },
    },
}

CHECK_SUBAGENTS_TOOL = {
    "type": "function",
    "function": {
        "name": "check_subagents",
        "description": (
            "Check status of spawned subagents. Returns status "
            "(running/done/failed/cancelled) and results for completed subagents. "
            "Use 'collect' action to block until a specific subagent finishes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["poll", "collect", "cancel"],
                    "description": (
                        "poll: status of all subagents. "
                        "collect: block until one finishes. "
                        "cancel: cancel a subagent."
                    ),
                    "default": "poll",
                },
                "subagent_id": {
                    "type": "string",
                    "description": "Required for 'collect' and 'cancel' actions.",
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds for 'collect' (default: 300).",
                },
            },
            "required": [],
        },
    },
}


class _CompositeCancelFlag:
    """Event-like object that is 'set' when either constituent flag is set."""

    def __init__(
        self,
        parent_flag: threading.Event | None,
        own_flag: threading.Event,
    ):
        self._parent = parent_flag
        self._own = own_flag

    def is_set(self) -> bool:
        return self._own.is_set() or (
            self._parent is not None and self._parent.is_set()
        )

    def set(self):
        self._own.set()

    def wait(self, timeout=None):
        deadline = time.monotonic() + timeout if timeout else None
        while True:
            if self.is_set():
                return True
            remaining = (deadline - time.monotonic()) if deadline else 0.05
            if remaining <= 0:
                return self.is_set()
            self._own.wait(min(remaining, 0.05))


@dataclass
class SubagentHandle:
    id: str
    task: str
    thread: threading.Thread | None = None
    result: str | None = None
    error: str | None = None
    exhausted: bool = False
    cancelled: bool = False
    cancel_flag: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)

    @property
    def status(self) -> str:
        if not self.done.is_set():
            return "running"
        if self.error:
            return "failed"
        if self.cancelled:
            return "cancelled"
        return "done"


class SubagentManager:
    """Manages parallel subagent threads."""

    def __init__(
        self,
        loop_kwargs_template: dict,
        tools: list,
        resolved_system_content: str | None,
        parent_cancel_flag: threading.Event | None,
        verbose: bool,
    ):
        self._template = loop_kwargs_template
        self._tools = [
            t for t in tools if t.get("function", {}).get("name") not in _SUBAGENT_TOOLS
        ]
        self._system_content = resolved_system_content
        self._parent_cancel_flag = parent_cancel_flag
        self._handles: dict[str, SubagentHandle] = {}
        self._counter = 0
        self._verbose = verbose
        self._lock = threading.Lock()

    def spawn(
        self,
        task: str,
        *,
        max_turns: int | None = None,
        system_hint: str | None = None,
    ) -> str:
        with self._lock:
            running = sum(1 for h in self._handles.values() if not h.done.is_set())
            if len(self._handles) > _MAX_CONCURRENT * 4:
                self._handles = {
                    k: v for k, v in self._handles.items() if not v.done.is_set()
                }
            if running >= _MAX_CONCURRENT:
                return f"error: max {_MAX_CONCURRENT} concurrent subagents"
            self._counter += 1
            sid = f"sub_{self._counter}"

        own_cancel = threading.Event()
        composite_cancel = _CompositeCancelFlag(self._parent_cancel_flag, own_cancel)
        handle = SubagentHandle(id=sid, task=task, cancel_flag=own_cancel)
        t = threading.Thread(
            target=_subagent_thread_fn,
            args=(
                handle,
                self._template,
                self._tools,
                task,
                max_turns or 30,
                system_hint,
                self._system_content,
                composite_cancel,
            ),
            name=f"swival-subagent-{sid}",
            daemon=True,
        )
        handle.thread = t
        with self._lock:
            self._handles[sid] = handle
        t.start()
        return f"Subagent {sid} launched."

    def poll(self) -> str:
        with self._lock:
            handles = list(self._handles.values())
        if not handles:
            return "No subagents."
        lines = []
        for h in handles:
            st = h.status
            line = f"{h.id}: {st}"
            if st == "done" and h.result is not None:
                preview = h.result[:500]
                if len(h.result) > 500:
                    preview += "... [truncated, use collect to get full result]"
                line += f"\n  Result: {preview}"
            elif st == "failed":
                line += f"\n  Error: {h.error}"
            lines.append(line)
        return "\n".join(lines)

    def collect(self, subagent_id: str, *, timeout: float | None = None) -> str:
        with self._lock:
            handle = self._handles.get(subagent_id)
        if handle is None:
            return f"error: unknown subagent {subagent_id!r}"
        t = timeout if timeout is not None else 300
        handle.done.wait(timeout=t)
        if not handle.done.is_set():
            return f"error: subagent {subagent_id} still running after {t}s timeout"
        if handle.error:
            return handle.error
        if handle.cancelled and handle.result is None:
            return f"error: subagent {subagent_id} was cancelled"
        if handle.exhausted and handle.result is None:
            return (
                f"error: subagent {subagent_id} exhausted max turns without an answer"
            )
        return handle.result or "(no result)"

    def cancel(self, subagent_id: str) -> str:
        with self._lock:
            handle = self._handles.get(subagent_id)
        if handle is None:
            return f"error: unknown subagent {subagent_id!r}"
        handle.cancel_flag.set()
        return f"Cancellation signal sent to {subagent_id}."

    def cancel_all(self) -> None:
        with self._lock:
            handles = list(self._handles.values())
        for h in handles:
            h.cancel_flag.set()

    def shutdown(self, timeout: float = 60) -> None:
        with self._lock:
            if not self._handles:
                return
        self.cancel_all()
        deadline = time.monotonic() + timeout
        stragglers = []
        with self._lock:
            handles = list(self._handles.values())
        for handle in handles:
            remaining = max(0, deadline - time.monotonic())
            if handle.thread is not None:
                handle.thread.join(timeout=remaining)
                if handle.thread.is_alive():
                    stragglers.append(handle)
        for handle in stragglers:
            if self._verbose:
                fmt.warning(
                    f"Subagent {handle.id} still running after {timeout}s, "
                    "waiting for it to finish..."
                )
            if handle.thread is not None:
                handle.thread.join()

    def fresh_copy(self) -> "SubagentManager":
        mgr = SubagentManager.__new__(SubagentManager)
        mgr._template = self._template
        mgr._tools = self._tools  # already filtered
        mgr._system_content = self._system_content
        mgr._parent_cancel_flag = threading.Event()
        mgr._handles = {}
        mgr._counter = 0
        mgr._verbose = self._verbose
        mgr._lock = threading.Lock()
        return mgr


def _build_subagent_system(parent_system: str | None, system_hint: str | None) -> str:
    preamble = (
        "You are a subagent working on a specific task within a larger project. "
        "Complete the task and provide your final answer. Be concise and focused. "
        "Do not ask questions — work autonomously with the information provided."
    )
    parts = [preamble]
    if system_hint:
        parts.append(system_hint)
    if parent_system:
        parts.append(parent_system)
    return "\n\n".join(parts)


def _subagent_thread_fn(
    handle: SubagentHandle,
    template: dict,
    tools: list,
    task: str,
    max_turns: int,
    system_hint: str | None,
    system_content: str | None,
    composite_cancel: _CompositeCancelFlag,
):
    try:
        from .agent import run_agent_loop

        thinking_state = ThinkingState(verbose=False)
        todo_state = TodoState(verbose=False)
        snapshot_state = SnapshotState(verbose=False)
        file_tracker = FileAccessTracker()

        full_system = _build_subagent_system(system_content, system_hint)
        messages: list[dict] = [{"role": "system", "content": full_system}]
        messages.append({"role": "user", "content": task})

        kwargs = {**template}
        kwargs.update(
            thinking_state=thinking_state,
            todo_state=todo_state,
            snapshot_state=snapshot_state,
            file_tracker=file_tracker,
            max_turns=max_turns,
            verbose=False,
            continue_here=False,
            cancel_flag=composite_cancel,
            report=None,
            compaction_state=None,
            turn_offset=0,
            cache=None,
            is_subagent=True,
        )

        cancel_before = composite_cancel.is_set()
        answer, exhausted = run_agent_loop(messages, tools, **kwargs)
        # Sample immediately after return to minimize the window for
        # a late cancel_all() to set the flag between return and read.
        cancel_observed = composite_cancel.is_set()
        handle.result = answer
        handle.exhausted = exhausted
        if cancel_before or cancel_observed:
            handle.cancelled = True
    except Exception as e:
        handle.error = f"error: subagent crashed: {e}"
    finally:
        handle.done.set()

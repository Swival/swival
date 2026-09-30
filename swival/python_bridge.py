"""Let run_python snippets call MCP tools.

The snippet sends requests over one pipe and reads replies from another, so
tool results stay in the snippet and only what it prints reaches the model.

Calls go through the same checks as direct tool calls and must finish within
the snippet's own timeout.
They never add a deferred tool's schema to later requests.
"""

import copy
import json
import math
import os
import selectors
import struct
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .mcp_client import _CALL_TIMEOUT, _describe_exception
from .tools import _cap_error_output

CLIENT_DIR = Path(__file__).with_name("python_bridge_client")
ENV_VAR = "SWIVAL_PYTHON_BRIDGE"

MAX_CALLS = 500
MAX_REQUEST_BYTES = 1 << 20
MAX_RESPONSE_BYTES = 10 << 20
MAX_TOTAL_BYTES = 64 << 20
# Keeps a looping snippet from growing the report without end.
MAX_RECORDS = 1000
MAX_RECORDED_NAME_BYTES = 128

# How long to wait for a tool call that is still running when the snippet ends.
FINISH_WAIT = 5.0

# Names listed in the tool description; the rest are in ALL_TOOLS.
_DESCRIBED_NAMES_MAX_CHARS = 1500

_FAILED = frozenset({"refused", "error", "timeout"})

_HEADER = struct.Struct(">I")


class _ProtocolError(Exception):
    pass


@dataclass
class BridgeStats:
    # Providers can reuse tool-call IDs, so each run gets its own.
    execution_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    tool_call_id: str | None = None
    records: list[dict] = field(default_factory=list)
    records_dropped: int = 0
    calls: int = 0
    failures: int = 0
    stop: str | None = None
    # Calls that ended without an answer; the server may still run them.
    uncertain_calls: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    call_time: float = 0.0
    # A call was still running when the snippet ended.
    unfinished: bool = False
    exit_code: int | None = None
    elapsed: float = 0.0

    def to_dict(self) -> dict:
        data = {
            "execution_id": self.execution_id,
            "tool_call_id": self.tool_call_id,
            "exit_code": self.exit_code,
            "elapsed_s": round(self.elapsed, 3),
            "child_calls": self.calls,
            "child_failures": self.failures,
            "bytes_in": self.bytes_in,
            "bytes_out": self.bytes_out,
            "child_time_s": round(self.call_time, 3),
        }
        if self.stop:
            data["stop"] = self.stop
        if self.uncertain_calls:
            data["uncertain_calls"] = self.uncertain_calls
        if self.unfinished:
            data["unfinished_call"] = True
        data["calls"] = [dict(record) for record in self.records]
        if self.records_dropped:
            data["calls_not_listed"] = self.records_dropped
        return data


def bridge_available(net_jail=None) -> bool:
    """Whether snippets can call tools on this system.

    The nono network jail drops the environment and the pipes the bridge
    needs.
    """
    return (
        sys.platform != "win32"
        and not net_jail
        and (CLIENT_DIR / "swival_tools.py").is_file()
    )


def describe(names: list[str]) -> str:
    """Text added to run_python's description when snippets can call tools."""
    listed: list[str] = []
    used = 0
    for name in names:
        if used + len(name) > _DESCRIBED_NAMES_MAX_CHARS:
            break
        listed.append(name)
        used += len(name) + 2
    tools = ", ".join(listed)
    if len(listed) < len(names):
        tools += f", and {len(names) - len(listed)} more in swival_tools.ALL_TOOLS"
    return (
        "\n\nThe snippet can call MCP tools with `import swival_tools`, so "
        "intermediate results stay out of the conversation and only printed "
        "output comes back. "
        "Prefer this over direct tool calls when a task needs many calls or "
        "large results, such as pagination, joins, or filtering. "
        "`swival_tools.call(name, {args})` returns the result as Python data "
        "(structured content, parsed JSON, or text) and raises "
        "`swival_tools.ToolError` on failure. "
        "`swival_tools.help(name)` returns the description and the "
        "input/output schemas; `output_schema` is None when the server "
        "declares none. "
        f"Callable tools: {tools}."
    )


class PythonBridge:
    """Serves the tool calls of one run_python snippet.

    Call ``open()`` before starting the snippet, then ``start()`` once it runs,
    or ``abandon()`` if it failed to start.
    Call ``finish()`` after it exits.
    """

    def __init__(
        self,
        manager,
        allowed: list[str],
        *,
        deadline: float,
        gate,
        goal_state=None,
        cancel_flag=None,
        tool_call_id: str | None = None,
    ):
        self._manager = manager
        self._allowed = list(allowed)
        # Callable tools, with the shortened names used in reports.
        self._record_names = {name: _clip_name(name) for name in self._allowed}
        # Tools whose output reached the snippet.
        self._external: set[str] = set()
        # (tool, start time) of the call waiting for an answer.
        self._in_flight: tuple[str, float] | None = None
        # Once the snippet is told the bridge stopped, nothing more is read.
        self._stop_sent = False
        self._current_record: dict | None = None
        self._deadline = deadline
        self._gate = gate
        self._goal_state = goal_state
        self._cancel_flag = cancel_flag
        self._fds: dict[str, int] = {}
        self._finishing = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.stats = BridgeStats(tool_call_id=tool_call_id)

    def open(self, env: dict[str, str]) -> tuple[int, int]:
        """Create the pipes and point *env* at them; return the child's ends."""
        # One pair at a time, so abandon() can clean up if the second fails.
        req_r, req_w = os.pipe()
        self._fds.update(req_r=req_r, req_w=req_w)
        resp_r, resp_w = os.pipe()
        self._fds.update(resp_r=resp_r, resp_w=resp_w)
        os.set_blocking(req_r, False)
        os.set_blocking(resp_w, False)
        env[ENV_VAR] = f"{req_w},{resp_r}"
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(CLIENT_DIR), env.get("PYTHONPATH")) if p
        )
        return req_w, resp_r

    def start(self) -> None:
        self._close("req_w", "resp_r")
        self._thread = threading.Thread(
            target=self._serve, name="swival-python-bridge", daemon=True
        )
        self._thread.start()

    def abandon(self) -> None:
        self._close(*list(self._fds))

    @property
    def started(self) -> bool:
        return self._thread is not None

    def finish(self) -> BridgeStats:
        """Stop serving, and wait a little for a call that is still running."""
        if self._finishing.is_set():
            return self.stats
        self._finishing.set()
        if self._thread is not None:
            self._thread.join(FINISH_WAIT)
            if self._thread.is_alive():
                # Report the call now as unknown; the thread won't record it
                # again when it returns.
                with self._lock:
                    in_flight, self._in_flight = self._in_flight, None
                    if in_flight is not None:
                        self.stats.unfinished = True
                        name, started = in_flight
                        self._record(
                            name,
                            "unfinished",
                            time.monotonic() - started,
                            remote_unknown=True,
                        )
                return self.stats
        self.abandon()
        return self.stats

    def external_tools(self) -> list[str]:
        with self._lock:
            return sorted(self._record_names[name] for name in self._external)

    def _close(self, *keys: str) -> None:
        with self._lock:
            fds = [self._fds.pop(key, None) for key in keys]
        for fd in fds:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def _serve(self) -> None:
        buf = bytearray()
        sel = selectors.DefaultSelector()
        sel.register(self._fds["req_r"], selectors.EVENT_READ)
        try:
            while not self._finishing.is_set():
                if not sel.select(0.1):
                    continue
                try:
                    chunk = os.read(self._fds["req_r"], 65536)
                except BlockingIOError:
                    continue
                except OSError:
                    return
                if not chunk:
                    return
                buf += chunk
                while (frame := self._take_frame(buf)) is not None:
                    self.stats.bytes_in += len(frame)
                    self._current_record = None
                    # Checked first, so an oversized request never reaches a tool.
                    if self.stats.bytes_in + self.stats.bytes_out > MAX_TOTAL_BYTES:
                        self._stop(_over_total())
                    try:
                        reply = self._handle(frame)
                    except Exception as e:
                        # The snippet waits for a reply, so it must get one.
                        reply = _tool_error(
                            f"error: the bridge failed: {_describe_exception(e)}"
                        )
                    if not self._send(self._charge(reply)):
                        if self._current_record is not None:
                            # The server answered, but the snippet never got it.
                            self._current_record["delivered"] = False
                        return
                    if self._stop_sent:
                        return
        except _ProtocolError as e:
            self._stop(str(e))
            self._send(self._stop_reply())
        finally:
            sel.close()
            self._close("req_r", "resp_w")

    def _take_frame(self, buf: bytearray) -> bytes | None:
        if len(buf) < _HEADER.size:
            return None
        (length,) = _HEADER.unpack_from(buf)
        if length > MAX_REQUEST_BYTES:
            raise _ProtocolError(
                f"a request of {length} bytes exceeds the {MAX_REQUEST_BYTES}-byte limit"
            )
        end = _HEADER.size + length
        if len(buf) < end:
            return None
        frame = bytes(buf[_HEADER.size : end])
        del buf[:end]
        return frame

    def _send(self, payload: bytes) -> bool:
        """Write one reply, counting it only if it was fully delivered."""
        fd = self._fds.get("resp_w")
        if fd is None:
            return False
        view = memoryview(_HEADER.pack(len(payload)) + payload)
        with selectors.DefaultSelector() as sel:
            sel.register(fd, selectors.EVENT_WRITE)
            while view:
                if self._finishing.is_set():
                    return False
                if not sel.select(0.1):
                    continue
                try:
                    written = os.write(fd, view)
                except BlockingIOError:
                    continue
                except OSError:
                    return False
                view = view[written:]
        self.stats.bytes_out += len(payload)
        return True

    def _stop(self, reason: str) -> None:
        if self.stats.stop is None:
            self.stats.stop = reason

    def _stop_reply(self, error: str | None = None) -> bytes:
        self._stop_sent = True
        return _encode({"ok": False, "kind": "stop", "error": error or self.stats.stop})

    def _charge(self, reply: bytes) -> bytes:
        """Return *reply*, or a stop reply if it would pass the data limit."""
        if self.stats.bytes_in + self.stats.bytes_out + len(reply) > MAX_TOTAL_BYTES:
            self._stop(_over_total())
            return self._stop_reply()
        return reply

    def _refuse(self, name: str, reason: str) -> bytes:
        """Stop the bridge and refuse this call without making it."""
        self._stop(reason)
        self._record(name, "stopped")
        return self._stop_reply()

    def _cut_off(self, name: str, duration: float, reason: str) -> bytes:
        """Stop the bridge after a call that ended without an answer."""
        self._stop(reason)
        self._record(name, "stopped", duration, remote_unknown=True)
        return self._stop_reply(
            f"{self.stats.stop}; the call to {self._record_names[name]} was cut "
            "off and may still complete on the server"
        )

    def _record(
        self,
        name: str,
        outcome: str,
        duration: float = 0.0,
        remote_unknown: bool = False,
    ) -> None:
        """Record one call attempt and update the counters."""
        if outcome in _FAILED:
            self.stats.failures += 1
        if remote_unknown:
            self.stats.uncertain_calls += 1
        self.stats.call_time += duration
        if len(self.stats.records) >= MAX_RECORDS:
            self.stats.records_dropped += 1
            return
        record = self._current_record = {
            "id": f"{self.stats.execution_id}-{len(self.stats.records) + 1}",
            "tool": self._record_names.get(name) or _clip_name(name),
            "outcome": outcome,
            "duration_s": round(duration, 3),
        }
        if remote_unknown:
            record["remote_outcome"] = "unknown"
        self.stats.records.append(record)

    def _handle(self, frame: bytes) -> bytes:
        try:
            request = json.loads(frame)
        except (ValueError, RecursionError):
            return _usage("the request is not JSON")
        if not isinstance(request, dict):
            return _usage("the request is not a JSON object")
        op = request.get("op")
        if self.stats.stop is not None:
            if op == "call" and isinstance(request.get("name"), str):
                self._record(request["name"], "stopped")
            return self._stop_reply()
        if op == "list":
            return self._fit({"ok": True, "value": list(self._allowed)})
        if op == "help":
            return self._help(request.get("name"))
        if op == "call":
            return self._call(request)
        return _usage(f"unknown operation {op!r}")

    def _fit(self, reply: dict) -> bytes:
        """Encode a reply to list or help, within the size limit."""
        payload = _encode_bounded(reply, MAX_RESPONSE_BYTES)
        if payload is None:
            return _tool_error(
                f"error: the reply is over the {MAX_RESPONSE_BYTES}-byte limit"
            )
        return payload

    def _help(self, name) -> bytes:
        if name is not None and not isinstance(name, str):
            return _usage("the tool name must be a string")
        if name is None:
            summary = {}
            for tool in self._allowed:
                meta = self._manager.tool_metadata(tool) or {}
                lines = (meta.get("description") or "").strip().splitlines()
                summary[tool] = lines[0] if lines else ""
            described = self._allowed
            reply = self._fit({"ok": True, "value": summary})
        else:
            meta = (
                self._manager.tool_metadata(name)
                if name in self._record_names
                else None
            )
            if meta is None:
                return _unknown_tool(name)
            described = [name]
            reply = self._fit({"ok": True, "value": meta})
        # Descriptions come from the server, so they count as its output.
        with self._lock:
            self._external.update(described)
        return reply

    def _abandoned(self) -> bool:
        """Whether to give up on the call in progress."""
        return self._finishing.is_set() or (
            self._cancel_flag is not None and self._cancel_flag.is_set()
        )

    def _host_stop(self) -> str | None:
        if self._cancel_flag is not None and self._cancel_flag.is_set():
            return "the user cancelled the run"
        if time.monotonic() >= self._deadline:
            return "the run_python timeout expired"
        if self._goal_state is not None and self._goal_state.budget_exhausted():
            return "the goal's budget is exhausted"
        return None

    def _call(self, request: dict) -> bytes:
        name = request.get("name")
        if not isinstance(name, str):
            return _usage("the tool name must be a string")
        arguments = request.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return _usage("arguments must be a dict")
        requested = request.get("timeout")
        if requested is not None and (
            isinstance(requested, bool)
            or not isinstance(requested, (int, float))
            or requested <= 0
        ):
            return _usage("timeout must be a positive number of seconds")
        if name not in self._record_names:
            self._record(name, "refused")
            return _unknown_tool(name)

        reason = self._host_stop()
        if reason is None and self.stats.calls >= MAX_CALLS:
            reason = f"the program made {MAX_CALLS} tool calls, the most allowed"
        if reason is not None:
            return self._refuse(name, reason)
        refusal = self._gate(name, arguments)
        if refusal is not None:
            self._record(name, "refused")
            return _tool_error(refusal)

        remaining = self._deadline - time.monotonic()
        limit = min(_CALL_TIMEOUT, remaining)
        if requested is not None:
            limit = min(limit, requested)
        if limit <= 0:
            return self._refuse(name, "the run_python timeout expired")

        started = time.monotonic()
        # Done under the lock so finish() either sees this call or nothing is
        # sent at all.
        with self._lock:
            ending = self._finishing.is_set()
            if not ending:
                self.stats.calls += 1
                self._in_flight = (name, started)
        if ending:
            return self._refuse(name, "the program ended")
        data = failure = None
        try:
            data = self._manager.call_tool_data(
                name, arguments, limit, abort=self._abandoned
            )
        except Exception as e:
            failure = e
        duration = time.monotonic() - started
        with self._lock:
            if self._in_flight is None:
                # finish() already reported this call.
                return self._stop_reply()
            # Recorded before the slot is cleared, so finish() never sees a
            # half-recorded call.
            reply = self._conclude(name, duration, data, failure, limit, remaining)
            self._in_flight = None
            return reply

    def _conclude(self, name, duration, data, failure, limit, remaining):
        """Record a call that returned and build the snippet's reply."""
        if failure is not None:
            # The message may quote the server.
            self._external.add(name)
            self._record(name, "error", duration)
            return _tool_error(
                _cap_error_output(
                    f"error: the call failed: {_describe_exception(failure)}"
                )
            )
        # Giving up on a call does not stop the server, so these calls may
        # still take effect.
        cause = data.cause
        if cause == "shutdown":
            return self._cut_off(
                name, duration, "Swival is shutting down its MCP connections"
            )
        if cause == "aborted":
            return self._cut_off(
                name,
                duration,
                self._host_stop() or "the program ended during a tool call",
            )
        if data.error is None or data.error.details:
            self._external.add(name)
        if cause == "timeout" and limit >= remaining:
            return self._cut_off(name, duration, "the run_python timeout expired")
        reason = self._host_stop()
        if reason is not None:
            # This call finished, so the snippet still gets its result.
            self._stop(reason)
        if data.error is not None:
            error = data.error.text
            if cause == "timeout":
                error += "; it may still complete on the server"
            elif cause == "connection_lost":
                error += (
                    "; the server may have carried out the call before the "
                    "connection failed"
                )
            self._record(
                name,
                "timeout" if cause == "timeout" else "error",
                duration,
                remote_unknown=cause is not None,
            )
            if data.error.details:
                error += "\n" + _cap_error_output(data.error.details)
            return _tool_error(_cap_error_output(error))
        try:
            payload = _encode_bounded(
                {"ok": True, "value": data.value}, MAX_RESPONSE_BYTES
            )
        except (ValueError, RecursionError) as e:
            self._record(name, "error", duration)
            return _tool_error(f"error: the result cannot be sent as JSON: {e}")
        if payload is None:
            self._record(name, "error", duration)
            return _tool_error(
                f"error: the result is over the {MAX_RESPONSE_BYTES}-byte limit "
                "for one call"
            )
        reply = self._charge(payload)
        self._record(name, "ok" if reply is payload else "stopped", duration)
        return reply


# Longest JSON form of a float, as in -1.7976931348623157e+308.
_FLOAT_MAX_CHARS = 24
_LOG10_2 = math.log10(2)


def _size_bounds(value, cap: int) -> tuple[int, float]:
    """Smallest and largest size *value* can have as JSON, without encoding it.

    Stops early once the smallest size passes *cap*.
    Values that JSON can only write with str() have no upper limit.
    """
    low, high = 0, 0.0
    stack = [value]
    while stack and low <= cap:
        item = stack.pop()
        if isinstance(item, str):
            # Between one and twelve bytes per character, depending on escaping.
            low += len(item) + 2
            high += 12 * len(item) + 2
        elif isinstance(item, bool) or item is None:
            low += 4
            high += 5
        elif isinstance(item, int):
            bits = abs(item).bit_length()
            low += int(max(bits - 1, 0) * _LOG10_2) + 1
            high += int(bits * _LOG10_2) + 2
        elif isinstance(item, float):
            low += 1
            high += _FLOAT_MAX_CHARS
        elif isinstance(item, dict):
            low += 2
            high += 2 + 4 * len(item)
            for key, child in item.items():
                if not isinstance(key, str):
                    high += 2  # quotes around a key that is not a string
                stack.extend((key, child))
        elif isinstance(item, (list, tuple)):
            low += 2
            high += 2 + 2 * len(item)
            stack.extend(item)
        else:
            low += 1
            high = math.inf
    return low, high


def _encode_bounded(reply: dict, limit: int) -> bytes | None:
    """Encode *reply*, or return None if it is over *limit* bytes.

    Values that are surely too big are refused without encoding.
    Values that surely fit are encoded in one go.
    The rest are encoded piece by piece, stopping once the limit is passed.

    A single long string is still encoded whole, so it can cost a few times
    the limit before it is refused.
    """
    low, high = _size_bounds(reply, limit)
    if low > limit:
        return None
    if high <= limit:
        return _encode(reply)
    for ascii_only in (False, True):
        encoder = json.JSONEncoder(ensure_ascii=ascii_only, default=str)
        parts, size = [], 0
        try:
            for chunk in encoder.iterencode(reply):
                data = chunk.encode()
                size += len(data)
                if size > limit:
                    return None
                parts.append(data)
        except UnicodeEncodeError:
            # A lone surrogate from JSON can only be sent escaped.
            continue
        return b"".join(parts)
    return None


def _encode(reply: dict) -> bytes:
    text = json.dumps(reply, ensure_ascii=False, default=str)
    try:
        return text.encode()
    except UnicodeEncodeError:
        # A lone surrogate from JSON can only be sent escaped.
        return json.dumps(reply, ensure_ascii=True, default=str).encode()


def _usage(message: str) -> bytes:
    return _encode({"ok": False, "kind": "usage", "error": message})


def _tool_error(message: str) -> bytes:
    return _encode({"ok": False, "kind": "tool", "error": message})


def _unknown_tool(name: str) -> bytes:
    return _tool_error(
        f"error: {_clip_name(name)!r} cannot be called from run_python; "
        "see swival_tools.ALL_TOOLS"
    )


def _over_total() -> str:
    return f"the program exchanged more than {MAX_TOTAL_BYTES} bytes with its tools"


def _clip_name(name: str) -> str:
    """Shorten a name to MAX_RECORDED_NAME_BYTES of valid UTF-8.

    Lone surrogates become "?" so reports and traces can always be written.
    """
    data = name[: MAX_RECORDED_NAME_BYTES + 1].encode(errors="replace")
    if len(name) <= MAX_RECORDED_NAME_BYTES and len(data) <= MAX_RECORDED_NAME_BYTES:
        return data.decode()
    return data[:MAX_RECORDED_NAME_BYTES].decode(errors="ignore") + "...[truncated]"


def annotate_run_python(tools: list, manager, net_jail=None) -> None:
    """List the callable tools in run_python's description."""
    names = manager.python_tools()
    if not names or not bridge_available(net_jail):
        return
    for i, tool in enumerate(tools):
        if tool.get("function", {}).get("name") == "run_python":
            tool = copy.deepcopy(tool)
            tool["function"]["description"] += describe(names)
            tools[i] = tool

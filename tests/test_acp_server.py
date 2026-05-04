"""Tests for swival.acp_server: protocol handshake, prompts, cancellation."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import types
from typing import Any

import pytest

from swival import acp_server as acp_server_mod
from swival.acp_server import AcpServer
from swival.acp_types import (
    ERROR_INVALID_PARAMS,
    ERROR_INVALID_REQUEST,
    ERROR_METHOD_NOT_FOUND,
    ERROR_SESSION_NOT_FOUND,
    JSONRPC_VERSION,
    METHOD_SESSION_LOAD,
    METHOD_SESSION_UPDATE,
    PROTOCOL_VERSION,
    STOP_CANCELLED,
    STOP_END_TURN,
    TOOL_KIND_READ,
    TOOL_STATUS_COMPLETED,
    UPDATE_AGENT_MESSAGE_CHUNK,
    UPDATE_TOOL_CALL,
    UPDATE_TOOL_CALL_UPDATE,
    encode_message,
    make_notification,
)
from swival.a2a_types import (
    EVENT_TEXT_CHUNK,
    EVENT_TOOL_FINISH,
    EVENT_TOOL_START,
)
from swival.session import Result


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeSession:
    """Stand-in for swival.session.Session that does not call any LLM."""

    def __init__(self, *, base_dir: str, **_: Any):
        self.base_dir = base_dir
        self.event_callback = None
        self.cancel_flag: threading.Event | None = None
        self._script: list[tuple[str, dict]] = []
        self._answer = "ok"
        self._exhausted = False
        self._raise: BaseException | None = None
        self._delay_sec = 0.0
        self._honor_cancel = False

    def script(
        self,
        events: list[tuple[str, dict]],
        *,
        answer: str = "ok",
        exhausted: bool = False,
        raise_exc: BaseException | None = None,
        delay_sec: float = 0.0,
        honor_cancel: bool = False,
    ) -> None:
        self._script = events
        self._answer = answer
        self._exhausted = exhausted
        self._raise = raise_exc
        self._delay_sec = delay_sec
        self._honor_cancel = honor_cancel

    def ask(self, prompt: str) -> Result:
        if self._raise is not None:
            raise self._raise
        cb = self.event_callback
        for kind, data in self._script:
            if cb is not None:
                cb(kind, data)
        if self._delay_sec > 0:
            deadline = time.monotonic() + self._delay_sec
            while time.monotonic() < deadline:
                if (
                    self._honor_cancel
                    and self.cancel_flag is not None
                    and self.cancel_flag.is_set()
                ):
                    return Result(
                        answer=None,
                        exhausted=True,
                        messages=[],
                        report=None,
                    )
                time.sleep(0.01)
        return Result(
            answer=self._answer,
            exhausted=self._exhausted,
            messages=[],
            report=None,
        )


@pytest.fixture
def fake_session_cls(monkeypatch):
    """Replace Session inside acp_server with our scriptable fake."""
    holder: dict[str, _FakeSession] = {}

    def factory(**kwargs):
        s = _FakeSession(**kwargs)
        holder["last"] = s
        return s

    monkeypatch.setattr(acp_server_mod, "Session", factory)
    return holder


@pytest.fixture
def server():
    return AcpServer(session_kwargs={"provider": "lmstudio"})


# ---------------------------------------------------------------------------
# Helper: drive an AcpServer in an asyncio loop, capturing outbound messages
# ---------------------------------------------------------------------------


def _attach_capture(server: AcpServer) -> list[dict]:
    captured: list[dict] = []

    async def fake_send(message: dict) -> None:
        captured.append(message)

    server._send = fake_send  # type: ignore[assignment]
    return captured


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Initialize handshake
# ---------------------------------------------------------------------------


class TestInitialize:
    def test_basic_handshake(self, server):
        captured = _attach_capture(server)
        _run(
            server._handle_initialize(
                1,
                {
                    "protocolVersion": 1,
                    "clientCapabilities": {"fs": {"readTextFile": True}},
                },
            )
        )
        assert len(captured) == 1
        msg = captured[0]
        assert msg["id"] == 1
        result = msg["result"]
        assert result["protocolVersion"] == 1
        assert result["agentCapabilities"]["loadSession"] is False
        assert result["agentCapabilities"]["promptCapabilities"] == {
            "image": False,
            "audio": False,
            "embeddedContext": False,
        }
        assert result["authMethods"] == []
        assert server._initialized is True

    def test_version_negotiation_caps_at_supported(self, server):
        captured = _attach_capture(server)
        _run(server._handle_initialize(1, {"protocolVersion": 99}))
        assert captured[0]["result"]["protocolVersion"] == PROTOCOL_VERSION

    def test_invalid_protocol_version(self, server):
        captured = _attach_capture(server)
        _run(server._handle_initialize(1, {"protocolVersion": "v1"}))
        assert captured[0]["error"]["code"] == ERROR_INVALID_PARAMS
        assert server._initialized is False


# ---------------------------------------------------------------------------
# session/new
# ---------------------------------------------------------------------------


class TestSessionNew:
    def test_requires_initialize(self, server, fake_session_cls, tmp_path):
        captured = _attach_capture(server)
        _run(server._handle_session_new(1, {"cwd": str(tmp_path), "mcpServers": []}))
        assert captured[0]["error"]["code"] == ERROR_INVALID_REQUEST

    def test_creates_session(self, server, fake_session_cls, tmp_path):
        server._initialized = True
        captured = _attach_capture(server)
        _run(server._handle_session_new(1, {"cwd": str(tmp_path), "mcpServers": []}))
        assert "result" in captured[0]
        sid = captured[0]["result"]["sessionId"]
        assert isinstance(sid, str) and sid
        assert sid in server._sessions
        sess = server._sessions[sid]
        assert sess.cwd == tmp_path.resolve() or sess.cwd == tmp_path
        assert isinstance(sess.cancel_flag, threading.Event)
        assert not sess.cancel_flag.is_set()
        assert fake_session_cls["last"].base_dir == str(tmp_path)
        assert fake_session_cls["last"].cancel_flag is sess.cancel_flag

    def test_rejects_relative_cwd(self, server, fake_session_cls):
        server._initialized = True
        captured = _attach_capture(server)
        _run(server._handle_session_new(1, {"cwd": "relative/path", "mcpServers": []}))
        assert captured[0]["error"]["code"] == ERROR_INVALID_PARAMS

    def test_rejects_missing_cwd(self, server, fake_session_cls):
        server._initialized = True
        captured = _attach_capture(server)
        _run(server._handle_session_new(1, {"mcpServers": []}))
        assert captured[0]["error"]["code"] == ERROR_INVALID_PARAMS

    def test_rejects_missing_mcp_servers(self, server, fake_session_cls, tmp_path):
        """ACP requires mcpServers; refusing silently to drop ACP-pushed MCP servers."""
        server._initialized = True
        captured = _attach_capture(server)
        _run(server._handle_session_new(1, {"cwd": str(tmp_path)}))
        assert captured[0]["error"]["code"] == ERROR_INVALID_PARAMS
        assert "mcpServers" in captured[0]["error"]["message"]

    def test_rejects_non_list_mcp_servers(self, server, fake_session_cls, tmp_path):
        server._initialized = True
        captured = _attach_capture(server)
        _run(
            server._handle_session_new(1, {"cwd": str(tmp_path), "mcpServers": "nope"})
        )
        assert captured[0]["error"]["code"] == ERROR_INVALID_PARAMS

    def test_rejects_non_empty_mcp_servers(self, server, fake_session_cls, tmp_path):
        """We don't connect ACP-pushed MCP servers in v1; refuse loudly so the client knows."""
        server._initialized = True
        captured = _attach_capture(server)
        _run(
            server._handle_session_new(
                1,
                {
                    "cwd": str(tmp_path),
                    "mcpServers": [
                        {
                            "name": "fs",
                            "command": "/usr/bin/mcp-fs",
                            "args": [],
                            "env": [],
                        }
                    ],
                },
            )
        )
        assert captured[0]["error"]["code"] == ERROR_INVALID_PARAMS
        assert "MCP" in captured[0]["error"]["message"]
        assert sid_count(server) == 0

    def test_accepts_empty_mcp_servers(self, server, fake_session_cls, tmp_path):
        server._initialized = True
        captured = _attach_capture(server)
        _run(server._handle_session_new(1, {"cwd": str(tmp_path), "mcpServers": []}))
        assert "result" in captured[0]


def sid_count(server) -> int:
    return len(server._sessions)


# ---------------------------------------------------------------------------
# session/load
# ---------------------------------------------------------------------------


class TestSessionLoad:
    def test_reports_unsupported(self, server):
        captured = _attach_capture(server)

        async def go():
            await server._dispatch(
                types.SimpleNamespace(
                    method=METHOD_SESSION_LOAD,
                    params={"sessionId": "x"},
                    id=42,
                    is_notification=False,
                )
            )

        _run(go())
        assert captured[0]["error"]["code"] == ERROR_METHOD_NOT_FOUND


# ---------------------------------------------------------------------------
# session/prompt happy path
# ---------------------------------------------------------------------------


class TestSessionPrompt:
    def _setup_session(self, server, tmp_path) -> str:
        server._initialized = True
        captured = _attach_capture(server)

        async def boot():
            await server._handle_session_new(
                1, {"cwd": str(tmp_path), "mcpServers": []}
            )
            sid = captured[-1]["result"]["sessionId"]
            return sid

        sid = _run(boot())
        captured.clear()
        return sid

    def test_unknown_session(self, server):
        captured = _attach_capture(server)
        _run(
            server._handle_session_prompt(
                7,
                {"sessionId": "nope", "prompt": [{"type": "text", "text": "hi"}]},
            )
        )
        assert captured[0]["error"]["code"] == ERROR_SESSION_NOT_FOUND

    def test_empty_prompt(self, server, fake_session_cls, tmp_path):
        sid = self._setup_session(server, tmp_path)
        captured = _attach_capture(server)
        _run(
            server._handle_session_prompt(
                7, {"sessionId": sid, "prompt": [{"type": "text", "text": ""}]}
            )
        )
        assert captured[0]["error"]["code"] == ERROR_INVALID_PARAMS

    def test_resource_link_only_prompt_is_accepted(
        self, server, fake_session_cls, tmp_path
    ):
        """ACP baseline: a prompt made entirely of resource_link blocks must work."""
        sid = self._setup_session(server, tmp_path)
        fake = fake_session_cls["last"]
        fake.script([], answer="seen")
        captured = _attach_capture(server)

        async def drive():
            await server._handle_session_prompt(
                7,
                {
                    "sessionId": sid,
                    "prompt": [
                        {
                            "type": "resource_link",
                            "uri": "file:///work/notes.md",
                            "name": "notes.md",
                        }
                    ],
                },
            )
            await server._sessions[sid].in_flight

        _run(drive())
        responses = [m for m in captured if "id" in m and "result" in m]
        assert len(responses) == 1
        assert responses[0]["result"] == {"stopReason": STOP_END_TURN}

    def test_image_block_rejected(self, server, fake_session_cls, tmp_path):
        sid = self._setup_session(server, tmp_path)
        captured = _attach_capture(server)
        _run(
            server._handle_session_prompt(
                7,
                {
                    "sessionId": sid,
                    "prompt": [{"type": "image", "data": "x", "mimeType": "image/png"}],
                },
            )
        )
        assert captured[0]["error"]["code"] == ERROR_INVALID_PARAMS
        assert "image" in captured[0]["error"]["message"]

    def test_happy_path_emits_text_and_tool_calls(
        self, server, fake_session_cls, tmp_path
    ):
        sid = self._setup_session(server, tmp_path)
        fake = fake_session_cls["last"]
        fake.script(
            [
                (
                    EVENT_TOOL_START,
                    {
                        "id": "tc1",
                        "name": "read_file",
                        "turn": 1,
                        "arguments_raw": json.dumps({"file_path": "x.txt"}),
                    },
                ),
                (
                    EVENT_TOOL_FINISH,
                    {
                        "id": "tc1",
                        "name": "read_file",
                        "turn": 1,
                        "elapsed": 0.01,
                        "arguments": {"file_path": "x.txt"},
                        "content": "file body",
                    },
                ),
                (EVENT_TEXT_CHUNK, {"text": "Hello there", "turn": 1}),
            ],
            answer="Hello there",
        )

        captured = _attach_capture(server)

        async def drive():
            await server._handle_session_prompt(
                7,
                {
                    "sessionId": sid,
                    "prompt": [{"type": "text", "text": "do it"}],
                },
            )
            sess = server._sessions[sid]
            await sess.in_flight

        _run(drive())

        notifications = [
            m for m in captured if m.get("method") == METHOD_SESSION_UPDATE
        ]
        responses = [m for m in captured if "id" in m and "result" in m]
        assert len(responses) == 1
        assert responses[0]["id"] == 7
        assert responses[0]["result"] == {"stopReason": STOP_END_TURN}

        update_kinds = [n["params"]["update"]["sessionUpdate"] for n in notifications]
        assert UPDATE_TOOL_CALL in update_kinds
        assert UPDATE_TOOL_CALL_UPDATE in update_kinds
        assert UPDATE_AGENT_MESSAGE_CHUNK in update_kinds

        # rawInput was parsed from arguments_raw
        tool_call_msg = next(
            n
            for n in notifications
            if n["params"]["update"]["sessionUpdate"] == UPDATE_TOOL_CALL
        )
        body = tool_call_msg["params"]["update"]
        assert body["toolCallId"] == "tc1"
        assert body["rawInput"] == {"file_path": "x.txt"}
        assert body["kind"] == TOOL_KIND_READ

        # tool_call_update carries content
        finish_msg = next(
            n
            for n in notifications
            if n["params"]["update"]["sessionUpdate"] == UPDATE_TOOL_CALL_UPDATE
        )
        assert finish_msg["params"]["update"]["status"] == TOOL_STATUS_COMPLETED
        assert (
            finish_msg["params"]["update"]["content"][0]["content"]["text"]
            == "file body"
        )

    def test_orphan_finish_synthesises_announce(
        self, server, fake_session_cls, tmp_path
    ):
        sid = self._setup_session(server, tmp_path)
        fake = fake_session_cls["last"]
        fake.script(
            [
                (
                    EVENT_TOOL_FINISH,
                    {
                        "id": "tc-ghost",
                        "name": "edit_file",
                        "turn": 1,
                        "elapsed": 0.01,
                        "arguments": {"file_path": "y.txt"},
                        "content": "patched",
                    },
                ),
            ],
            answer="done",
        )
        captured = _attach_capture(server)

        async def drive():
            await server._handle_session_prompt(
                7,
                {"sessionId": sid, "prompt": [{"type": "text", "text": "go"}]},
            )
            await server._sessions[sid].in_flight

        _run(drive())

        kinds = [
            m["params"]["update"]["sessionUpdate"]
            for m in captured
            if m.get("method") == METHOD_SESSION_UPDATE
        ]
        # Even though TOOL_START never fired, the finish event should produce
        # both an announce (tool_call) and a status transition (tool_call_update)
        assert kinds.count(UPDATE_TOOL_CALL) == 1
        assert kinds.count(UPDATE_TOOL_CALL_UPDATE) == 1


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class TestCancellation:
    def test_cancel_sets_flag_and_returns_cancelled(
        self, server, fake_session_cls, tmp_path
    ):
        server._initialized = True
        captured = _attach_capture(server)

        async def drive():
            await server._handle_session_new(
                1, {"cwd": str(tmp_path), "mcpServers": []}
            )
            sid = captured[-1]["result"]["sessionId"]
            captured.clear()
            fake = fake_session_cls["last"]
            fake.script([], answer="never", delay_sec=2.0, honor_cancel=True)

            await server._handle_session_prompt(
                7,
                {"sessionId": sid, "prompt": [{"type": "text", "text": "stall"}]},
            )
            # Give the worker thread a moment to enter ask()
            await asyncio.sleep(0.05)
            await server._handle_session_cancel({"sessionId": sid})
            await server._sessions[sid].in_flight

        _run(drive())

        responses = [m for m in captured if "id" in m and "result" in m]
        assert any(r["result"] == {"stopReason": STOP_CANCELLED} for r in responses)

    def test_cancel_unknown_session_silently_ignored(self, server):
        captured = _attach_capture(server)
        _run(server._handle_session_cancel({"sessionId": "ghost"}))
        assert captured == []


# ---------------------------------------------------------------------------
# Stdout discipline: every write must be a single newline-terminated JSON line
# ---------------------------------------------------------------------------


class TestStdoutDiscipline:
    def test_encode_message_is_single_line(self):
        for body in [
            make_notification("session/update", {"sessionId": "x", "update": {}}),
            {"jsonrpc": JSONRPC_VERSION, "id": 1, "result": {}},
        ]:
            data = encode_message(body)
            assert data.endswith(b"\n")
            assert data.count(b"\n") == 1
            decoded = json.loads(data.decode("utf-8"))
            assert decoded["jsonrpc"] == JSONRPC_VERSION

    def test_full_session_yields_only_ndjson(
        self, server, fake_session_cls, tmp_path, monkeypatch
    ):
        """Drive a real-ish session through _send and assert every captured
        message survives encode→decode as a single JSON line."""
        captured_bytes: list[bytes] = []

        async def fake_send(message: dict) -> None:
            captured_bytes.append(encode_message(message))

        server._send = fake_send  # type: ignore[assignment]
        server._initialized = True

        async def drive():
            await server._handle_session_new(
                1, {"cwd": str(tmp_path), "mcpServers": []}
            )
            sid = json.loads(captured_bytes[-1].decode("utf-8"))["result"]["sessionId"]
            fake_session_cls["last"].script(
                [(EVENT_TEXT_CHUNK, {"text": "hi", "turn": 1})], answer="hi"
            )
            await server._handle_session_prompt(
                7, {"sessionId": sid, "prompt": [{"type": "text", "text": "go"}]}
            )
            await server._sessions[sid].in_flight

        _run(drive())

        for line in captured_bytes:
            assert line.endswith(b"\n")
            assert line.count(b"\n") == 1
            json.loads(line[:-1])  # must parse as valid JSON


# ---------------------------------------------------------------------------
# authenticate
# ---------------------------------------------------------------------------


class TestAuthenticate:
    def test_authenticate_is_noop_success(self, server):
        captured = _attach_capture(server)
        _run(server._handle_authenticate(1, {"methodId": "anything"}))
        assert captured[0]["result"] == {}


class TestFrameLogging:
    def test_no_log_path_means_no_frame_logging(self, server, caplog):
        captured = _attach_capture(server)
        with caplog.at_level("INFO", logger="swival.acp_server"):
            _run(server._handle_initialize(1, {"protocolVersion": 1}))
        assert all("recv:" not in rec.message for rec in caplog.records)
        assert all("send:" not in rec.message for rec in caplog.records)
        assert captured  # response was still produced

    def test_log_path_captures_inbound_and_outbound(self, tmp_path, caplog):
        server = AcpServer(
            session_kwargs={"provider": "lmstudio"},
            log_path=str(tmp_path / "acp.log"),
        )
        server._stdout_lock = asyncio.Lock()
        written: list[bytes] = []
        server._write_blocking = written.append  # type: ignore[assignment]

        async def drive():
            await server._handle_line(
                b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1}}\n'
            )

        with caplog.at_level("INFO", logger="swival.acp_server"):
            _run(drive())

        msgs = [r.message for r in caplog.records]
        assert any(m.startswith("recv:") for m in msgs)
        assert any(m.startswith("send:") for m in msgs)
        assert written  # the real _send path ran end-to-end

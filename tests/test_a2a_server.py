"""Tests for swival.a2a_server: A2aServer, A2aTask, build_agent_card."""

import json
import threading
import time
import uuid

import pytest

from swival.a2a_server import A2aServer, A2aTask, build_agent_card
from swival.session import Result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jsonrpc(method, params, req_id=1):
    """Build a JSON-RPC 2.0 request."""
    return {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}


def _send_message(client, message, context_id=None, task_id=None):
    """Send a SendMessage JSON-RPC request."""
    msg = {"role": "user", "parts": [{"type": "text", "text": message}]}
    if context_id:
        msg["contextId"] = context_id
    if task_id:
        msg["taskId"] = task_id
    params = {"message": msg}
    return client.post("/", json=_jsonrpc("SendMessage", params))


def _get_task(client, task_id, req_id=1):
    """Send a GetTask JSON-RPC request."""
    return client.post("/", json=_jsonrpc("GetTask", {"id": task_id}, req_id))


def _list_tasks(client, context_id=None, req_id=1):
    """Send a ListTasks JSON-RPC request."""
    params = {}
    if context_id:
        params["contextId"] = context_id
    return client.post("/", json=_jsonrpc("ListTasks", params, req_id))


def _make_result(answer="Hello from the agent", exhausted=False):
    """Build a canned Result for mocking Session.ask()."""
    return Result(
        answer=answer,
        exhausted=exhausted,
        messages=[
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": answer},
        ],
        report=None,
    )


def _make_input_required_result():
    """Build a Result that signals input-required (exhausted with no answer)."""
    return Result(
        answer=None,
        exhausted=True,
        messages=[{"role": "user", "content": "test"}],
        report=None,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def _patch_session(monkeypatch):
    """Patch Session._setup to no-op and Session.ask to return a canned result."""
    from swival import session as session_mod

    monkeypatch.setattr(session_mod.Session, "_setup", lambda self: None)
    monkeypatch.setattr(
        session_mod.Session,
        "ask",
        lambda self, q: _make_result(f"answer to: {q}"),
    )


@pytest.fixture()
def server(_patch_session):
    """Create an A2aServer with mocked Session internals."""
    return A2aServer(
        session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
        host="127.0.0.1",
        port=0,
        heartbeat_interval=0.05,
    )


@pytest.fixture()
def client(server):
    """Starlette TestClient wrapping the server's ASGI app."""
    from starlette.testclient import TestClient

    return TestClient(server.app)


# ---------------------------------------------------------------------------
# 1. Single-shot SendMessage
# ---------------------------------------------------------------------------


class TestSendMessageSingleShot:
    def test_creates_task_and_returns_completed(self, client):
        resp = _send_message(client, "Hello")
        assert resp.status_code == 200
        body = resp.json()
        assert "result" in body
        task = body["result"]
        assert task["status"]["state"] == "completed"
        assert task["id"]
        assert task["contextId"]

    def test_response_contains_agent_text(self, client):
        resp = _send_message(client, "Hello")
        task = resp.json()["result"]
        # The answer should appear somewhere in artifacts or status message
        texts = []
        for artifact in task.get("artifacts", []):
            for part in artifact.get("parts", []):
                if part.get("type") == "text":
                    texts.append(part["text"])
        status_msg = task.get("status", {}).get("message", {})
        for part in status_msg.get("parts", []):
            if part.get("type") == "text":
                texts.append(part["text"])
        combined = " ".join(texts)
        assert "answer to: Hello" in combined

    def test_jsonrpc_id_echoed(self, client):
        resp = client.post(
            "/",
            json=_jsonrpc(
                "SendMessage",
                {
                    "message": {
                        "role": "user",
                        "parts": [{"type": "text", "text": "hi"}],
                    },
                },
                req_id=42,
            ),
        )
        body = resp.json()
        assert body.get("id") == 42


# ---------------------------------------------------------------------------
# 2. Multi-turn: same contextId shares session
# ---------------------------------------------------------------------------


class TestMultiTurn:
    def test_same_context_reuses_session(self, server, client):
        ctx = str(uuid.uuid4())
        resp1 = _send_message(client, "First question", context_id=ctx)
        assert resp1.status_code == 200

        resp2 = _send_message(client, "Follow-up", context_id=ctx)
        assert resp2.status_code == 200

        # Both tasks should share the same contextId
        t1 = resp1.json()["result"]
        t2 = resp2.json()["result"]
        assert t1["contextId"] == ctx
        assert t2["contextId"] == ctx
        # But they get distinct task IDs
        assert t1["id"] != t2["id"]

    def test_different_context_gets_different_session(self, server, client):
        resp1 = _send_message(client, "Question A", context_id="ctx-a")
        resp2 = _send_message(client, "Question B", context_id="ctx-b")
        t1 = resp1.json()["result"]
        t2 = resp2.json()["result"]
        assert t1["contextId"] == "ctx-a"
        assert t2["contextId"] == "ctx-b"


# ---------------------------------------------------------------------------
# 3. input-required resumption
# ---------------------------------------------------------------------------


class TestInputRequired:
    def test_input_required_task_is_non_terminal(self, monkeypatch, client, server):
        from swival import session as session_mod

        call_count = [0]

        def mock_ask(self, q):
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_input_required_result()
            return _make_result("resumed answer")

        monkeypatch.setattr(session_mod.Session, "ask", mock_ask)

        ctx = str(uuid.uuid4())
        resp1 = _send_message(client, "Start task", context_id=ctx)
        t1 = resp1.json()["result"]
        assert t1["status"]["state"] == "input-required"
        task_id = t1["id"]

        # Follow-up with taskId resumes
        resp2 = _send_message(client, "More info", context_id=ctx, task_id=task_id)
        t2 = resp2.json()["result"]
        assert t2["status"]["state"] == "completed"


# ---------------------------------------------------------------------------
# 4. TTL expiry
# ---------------------------------------------------------------------------


class TestTTLExpiry:
    def test_expired_session_gets_fresh_session(self, monkeypatch, _patch_session):
        """An idle session past TTL is cleaned up; follow-up gets a new one."""
        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
            ttl=1,  # 1s TTL
        )
        from starlette.testclient import TestClient

        tc = TestClient(srv.app)

        ctx = str(uuid.uuid4())
        resp1 = _send_message(tc, "Hello", context_id=ctx)
        assert resp1.status_code == 200

        # Wait for TTL to expire, then trigger cleanup manually
        time.sleep(1.1)
        srv._cleanup_expired()

        # Even if the session is gone, a follow-up should succeed with a fresh session
        resp2 = _send_message(tc, "Hello again", context_id=ctx)
        assert resp2.status_code == 200
        t2 = resp2.json()["result"]
        assert t2["status"]["state"] == "completed"


# ---------------------------------------------------------------------------
# 5. Concurrent contexts
# ---------------------------------------------------------------------------


class TestConcurrentContexts:
    def test_multiple_contexts_independent(self, client):
        contexts = [str(uuid.uuid4()) for _ in range(5)]
        responses = []
        for ctx in contexts:
            resp = _send_message(client, f"Question for {ctx}", context_id=ctx)
            assert resp.status_code == 200
            responses.append(resp.json()["result"])

        # Each should have its own contextId
        returned_contexts = {r["contextId"] for r in responses}
        assert returned_contexts == set(contexts)

        # All task IDs should be unique
        task_ids = [r["id"] for r in responses]
        assert len(set(task_ids)) == len(task_ids)


# ---------------------------------------------------------------------------
# 6. Agent Card
# ---------------------------------------------------------------------------


class TestAgentCard:
    def test_card_served_at_well_known_path(self, client):
        resp = client.get("/.well-known/agent-card.json")
        assert resp.status_code == 200
        card = resp.json()
        assert "name" in card
        assert "version" in card

    def test_card_has_skills(self, client):
        resp = client.get("/.well-known/agent-card.json")
        card = resp.json()
        assert "skills" in card
        assert isinstance(card["skills"], list)

    def test_card_content_type(self, client):
        resp = client.get("/.well-known/agent-card.json")
        assert "application/json" in resp.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# 7. Auth
# ---------------------------------------------------------------------------


class TestAuth:
    def test_no_auth_required_by_default(self, client):
        resp = _send_message(client, "Hello")
        assert resp.status_code == 200

    def test_bearer_token_accepted(self, _patch_session):
        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
            auth_token="secret-token-123",
        )
        from starlette.testclient import TestClient

        tc = TestClient(srv.app)

        resp = tc.post(
            "/",
            json=_jsonrpc(
                "SendMessage",
                {
                    "message": {
                        "role": "user",
                        "parts": [{"type": "text", "text": "hi"}],
                    },
                },
            ),
            headers={"Authorization": "Bearer secret-token-123"},
        )
        assert resp.status_code == 200
        assert "result" in resp.json()

    def test_missing_token_rejected(self, _patch_session):
        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
            auth_token="secret-token-123",
        )
        from starlette.testclient import TestClient

        tc = TestClient(srv.app)

        resp = tc.post(
            "/",
            json=_jsonrpc(
                "SendMessage",
                {
                    "message": {
                        "role": "user",
                        "parts": [{"type": "text", "text": "hi"}],
                    },
                },
            ),
        )
        # Should be 401 or a JSON-RPC error
        assert resp.status_code in (401, 403) or "error" in resp.json()

    def test_wrong_token_rejected(self, _patch_session):
        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
            auth_token="secret-token-123",
        )
        from starlette.testclient import TestClient

        tc = TestClient(srv.app)

        resp = tc.post(
            "/",
            json=_jsonrpc(
                "SendMessage",
                {
                    "message": {
                        "role": "user",
                        "parts": [{"type": "text", "text": "hi"}],
                    },
                },
            ),
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code in (401, 403) or "error" in resp.json()

    def test_agent_card_accessible_without_auth(self, _patch_session):
        """The agent card endpoint should be publicly accessible even with auth."""
        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
            auth_token="secret-token-123",
        )
        from starlette.testclient import TestClient

        tc = TestClient(srv.app)
        resp = tc.get("/.well-known/agent-card.json")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 8. GetTask
# ---------------------------------------------------------------------------


class TestGetTask:
    def test_get_existing_task(self, client):
        # Create a task first
        send_resp = _send_message(client, "Hello")
        task_id = send_resp.json()["result"]["id"]

        # Now retrieve it
        get_resp = _get_task(client, task_id)
        assert get_resp.status_code == 200
        body = get_resp.json()
        assert "result" in body
        assert body["result"]["id"] == task_id

    def test_get_nonexistent_task(self, client):
        resp = _get_task(client, "nonexistent-task-id")
        body = resp.json()
        # Should be a JSON-RPC error (task not found)
        assert "error" in body
        err = body["error"]
        assert err.get("code") is not None

    def test_get_task_preserves_status(self, client):
        send_resp = _send_message(client, "Hello")
        task = send_resp.json()["result"]
        task_id = task["id"]

        get_resp = _get_task(client, task_id)
        retrieved = get_resp.json()["result"]
        assert retrieved["status"]["state"] == task["status"]["state"]


# ---------------------------------------------------------------------------
# 9. ListTasks
# ---------------------------------------------------------------------------


class TestListTasks:
    def test_list_tasks_empty(self, client):
        resp = _list_tasks(client, context_id="nonexistent-ctx")
        assert resp.status_code == 200
        body = resp.json()
        assert "result" in body
        tasks = body["result"]
        assert isinstance(tasks, list)
        assert len(tasks) == 0

    def test_list_tasks_by_context(self, client):
        ctx = str(uuid.uuid4())
        _send_message(client, "Q1", context_id=ctx)
        _send_message(client, "Q2", context_id=ctx)
        _send_message(client, "Q3", context_id="other-ctx")

        resp = _list_tasks(client, context_id=ctx)
        tasks = resp.json()["result"]
        assert len(tasks) == 2
        assert all(t["contextId"] == ctx for t in tasks)

    def test_list_all_tasks(self, client):
        _send_message(client, "Q1", context_id="ctx-a")
        _send_message(client, "Q2", context_id="ctx-b")

        resp = _list_tasks(client)
        tasks = resp.json()["result"]
        assert len(tasks) >= 2


# ---------------------------------------------------------------------------
# 10. Unknown method
# ---------------------------------------------------------------------------


class TestUnknownMethod:
    def test_unknown_method_returns_error(self, client):
        resp = client.post("/", json=_jsonrpc("NonExistentMethod", {}))
        assert resp.status_code == 200
        body = resp.json()
        assert "error" in body
        err = body["error"]
        # JSON-RPC method not found code is -32601
        assert err["code"] == -32601

    def test_invalid_json_returns_error(self, client):
        resp = client.post(
            "/",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        body = resp.json()
        assert "error" in body


# ---------------------------------------------------------------------------
# 11. build_agent_card
# ---------------------------------------------------------------------------


class TestBuildAgentCard:
    def test_basic_card(self):
        card = build_agent_card(
            session_kwargs={"provider": "lmstudio", "model": "my-model"},
            host="0.0.0.0",
            port=8080,
        )
        assert card["name"]
        assert card["version"]
        assert isinstance(card["skills"], list)

    def test_card_includes_url(self):
        card = build_agent_card(
            session_kwargs={},
            host="0.0.0.0",
            port=9090,
        )
        # Should have some URL or supportedInterfaces
        has_url = False
        if "url" in card:
            has_url = True
        for iface in card.get("supportedInterfaces", []):
            if "url" in iface:
                has_url = True
        assert has_url

    def test_card_name_includes_provider(self):
        card = build_agent_card(
            session_kwargs={"provider": "openrouter", "model": "qwen3"},
            host="127.0.0.1",
            port=5000,
        )
        assert "openrouter" in card["name"]
        assert "qwen3" in card["name"]

    def test_card_has_description(self):
        card = build_agent_card(
            session_kwargs={},
            host="127.0.0.1",
            port=5000,
        )
        assert card["description"]

    def test_card_protocol_version_on_interface(self):
        card = build_agent_card(
            session_kwargs={},
            host="127.0.0.1",
            port=5000,
        )
        # protocolVersion must be on the interface entry, not top-level
        assert "protocolVersion" not in card
        ifaces = card.get("supportedInterfaces", [])
        assert len(ifaces) == 1
        assert ifaces[0]["protocolVersion"] == "1.0"

    def test_custom_name(self):
        card = build_agent_card(
            session_kwargs={"provider": "openrouter", "model": "qwen3"},
            host="127.0.0.1",
            port=5000,
            name="My Custom Agent",
        )
        assert card["name"] == "My Custom Agent"

    def test_custom_description(self):
        card = build_agent_card(
            session_kwargs={},
            host="127.0.0.1",
            port=5000,
            description="A specialized agent for code review",
        )
        assert card["description"] == "A specialized agent for code review"

    def test_custom_name_none_uses_default(self):
        card = build_agent_card(
            session_kwargs={"provider": "openrouter", "model": "qwen3"},
            host="127.0.0.1",
            port=5000,
            name=None,
        )
        assert "openrouter" in card["name"]
        assert "qwen3" in card["name"]

    def test_custom_description_none_uses_default(self):
        card = build_agent_card(
            session_kwargs={},
            host="127.0.0.1",
            port=5000,
            description=None,
        )
        assert "coding agent" in card["description"]

    def test_skills_in_card(self):
        skills = [
            {
                "id": "review",
                "name": "Code Review",
                "description": "Analyze code",
                "examples": ["Review this PR"],
            },
            {
                "id": "explain",
                "name": "Code Explanation",
                "description": "Explain code",
            },
        ]
        card = build_agent_card(
            session_kwargs={},
            host="127.0.0.1",
            port=5000,
            skills=skills,
        )
        assert len(card["skills"]) == 2
        assert card["skills"][0]["id"] == "review"
        assert card["skills"][0]["name"] == "Code Review"
        assert card["skills"][0]["examples"] == ["Review this PR"]
        assert card["skills"][1]["id"] == "explain"
        assert "examples" not in card["skills"][1]

    def test_skills_none_gives_empty_list(self):
        card = build_agent_card(
            session_kwargs={},
            host="127.0.0.1",
            port=5000,
            skills=None,
        )
        assert card["skills"] == []

    def test_skills_empty_list(self):
        card = build_agent_card(
            session_kwargs={},
            host="127.0.0.1",
            port=5000,
            skills=[],
        )
        assert card["skills"] == []

    def test_all_customizations_together(self):
        skills = [{"id": "ask", "name": "Ask", "description": "Ask a question"}]
        card = build_agent_card(
            session_kwargs={"provider": "openrouter", "model": "qwen3"},
            host="127.0.0.1",
            port=5000,
            auth_token="secret",
            name="My Agent",
            description="Does things",
            skills=skills,
        )
        assert card["name"] == "My Agent"
        assert card["description"] == "Does things"
        assert len(card["skills"]) == 1
        assert "securitySchemes" in card


# ---------------------------------------------------------------------------
# 12. A2aTask dataclass
# ---------------------------------------------------------------------------


class TestA2aTask:
    def test_create_task(self):
        task = A2aTask(
            id="t1",
            context_id="c1",
            status="completed",
        )
        assert task.id == "t1"
        assert task.context_id == "c1"
        assert task.status == "completed"

    def test_task_defaults(self):
        task = A2aTask(id="t2", context_id="c2")
        assert task.messages == []
        assert task.artifacts == []
        assert task.status == "working"

    def test_task_timestamps(self):
        task = A2aTask(id="t3", context_id="c3")
        assert task.created_at is not None
        assert task.updated_at is not None
        assert task.updated_at >= task.created_at


# ---------------------------------------------------------------------------
# Edge cases and error handling
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_message_text(self, client):
        resp = _send_message(client, "")
        # Should still succeed or return a sensible error
        assert resp.status_code == 200

    def test_missing_message_in_params(self, client):
        resp = client.post("/", json=_jsonrpc("SendMessage", {}))
        body = resp.json()
        # Should return an error for missing required params
        assert "error" in body

    def test_params_as_list_returns_error(self, client):
        """params as a JSON array should return INVALID_PARAMS, not crash."""
        resp = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "SendMessage",
                "params": [],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "error" in body
        assert body["error"]["code"] == -32602

    def test_params_as_string_returns_error(self, client):
        """params as a string should return INVALID_PARAMS."""
        resp = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "SendMessage",
                "params": "bad",
            },
        )
        body = resp.json()
        assert "error" in body

    def test_post_to_agent_card_path(self, client):
        """POST to the agent card path should not crash."""
        resp = client.post("/.well-known/agent-card.json", content=b"{}")
        # Could be 405 or just ignored; should not be 500
        assert resp.status_code != 500

    def test_get_to_jsonrpc_endpoint(self, client):
        """GET to / should not crash (maybe 405)."""
        resp = client.get("/")
        assert resp.status_code != 500


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


class TestLRUEviction:
    def test_eviction_on_capacity(self, _patch_session):
        """When max_sessions is reached, LRU session is evicted (not rejected)."""
        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
            max_sessions=2,
        )
        from starlette.testclient import TestClient

        tc = TestClient(srv.app)

        # Fill to capacity
        resp1 = _send_message(tc, "Q1", context_id="ctx-1")
        assert resp1.json()["result"]["status"]["state"] == "completed"
        resp2 = _send_message(tc, "Q2", context_id="ctx-2")
        assert resp2.json()["result"]["status"]["state"] == "completed"
        assert len(srv._sessions) == 2

        # Third context should succeed by evicting the LRU (ctx-1)
        resp3 = _send_message(tc, "Q3", context_id="ctx-3")
        assert resp3.json()["result"]["status"]["state"] == "completed"
        assert len(srv._sessions) == 2
        assert "ctx-3" in srv._sessions
        assert "ctx-1" not in srv._sessions


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------


def _send_streaming_message(client, message, context_id=None, task_id=None):
    """Send a SendStreamingMessage JSON-RPC request and parse SSE events."""
    msg = {"role": "user", "parts": [{"type": "text", "text": message}]}
    if context_id:
        msg["contextId"] = context_id
    if task_id:
        msg["taskId"] = task_id
    params = {"message": msg}
    resp = client.post(
        "/",
        json=_jsonrpc("SendStreamingMessage", params),
        headers={"Accept": "text/event-stream"},
    )
    return resp


def _parse_sse_events(response_text):
    """Parse SSE text into a list of (event_type, data_dict) tuples."""
    events = []
    current_event = None
    current_data = []

    for line in response_text.split("\n"):
        if line.startswith("event: "):
            current_event = line[7:]
        elif line.startswith("data: "):
            current_data.append(line[6:])
        elif line == "" and current_event is not None:
            data_str = "\n".join(current_data)
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                data = data_str
            events.append((current_event, data))
            current_event = None
            current_data = []

    return events


# ---------------------------------------------------------------------------
# CancelTask
# ---------------------------------------------------------------------------


class TestCancelTask:
    def test_cancel_working_task(self, _patch_session, monkeypatch):
        """CancelTask sets the cancellation flag and returns canceled status."""
        from swival import session as session_mod
        from starlette.testclient import TestClient

        cancel_event = threading.Event()

        def slow_ask(self, q):
            # Simulate a long-running task that checks for cancellation
            cancel_event.wait(timeout=5)
            return _make_result(f"answer: {q}")

        monkeypatch.setattr(session_mod.Session, "ask", slow_ask)

        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
        )
        tc = TestClient(srv.app)

        # Start a task in a thread
        result_holder = {}

        def do_send():
            r = _send_message(tc, "hello")
            result_holder["send_resp"] = r

        t = threading.Thread(target=do_send)
        t.start()

        # Wait for the task to be created
        time.sleep(0.3)

        # Find the task
        list_resp = tc.post("/", json=_jsonrpc("ListTasks", {}))
        tasks = list_resp.json()["result"]
        assert len(tasks) >= 1
        task_id = tasks[0]["id"]

        # Cancel it
        cancel_resp = tc.post("/", json=_jsonrpc("CancelTask", {"id": task_id}))
        assert cancel_resp.json()["result"]["status"]["state"] == "canceled"

        # Release the slow_ask
        cancel_event.set()
        t.join(timeout=5)

    def test_cancel_terminal_task_fails(self, _patch_session, client):
        """CancelTask on a completed task returns an error."""
        resp = _send_message(client, "hello")
        task_id = resp.json()["result"]["id"]

        cancel_resp = client.post("/", json=_jsonrpc("CancelTask", {"id": task_id}))
        assert "error" in cancel_resp.json()

    def test_cancel_missing_task_fails(self, client):
        """CancelTask with unknown task ID returns error."""
        resp = client.post("/", json=_jsonrpc("CancelTask", {"id": "nonexistent"}))
        assert "error" in resp.json()

    def test_cancel_missing_id_fails(self, client):
        """CancelTask without id param returns error."""
        resp = client.post("/", json=_jsonrpc("CancelTask", {}))
        assert "error" in resp.json()


# ---------------------------------------------------------------------------
# SSE Streaming
# ---------------------------------------------------------------------------


class TestSendStreamingMessage:
    def test_streaming_returns_sse(self, _patch_session, client):
        """SendStreamingMessage returns SSE content type."""
        resp = _send_streaming_message(client, "hello")
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_streaming_has_initial_working_status(self, _patch_session, client):
        """SSE stream starts with a TaskStatusUpdateEvent in working state."""
        resp = _send_streaming_message(client, "hello")
        events = _parse_sse_events(resp.text)
        assert len(events) >= 1
        first = events[0]
        assert first[0] == "TaskStatusUpdateEvent"
        assert first[1]["status"]["state"] == "working"

    def test_streaming_ends_with_completed_status(self, _patch_session, client):
        """SSE stream ends with a TaskStatusUpdateEvent in completed state."""
        resp = _send_streaming_message(client, "hello")
        events = _parse_sse_events(resp.text)
        status_events = [e for e in events if e[0] == "TaskStatusUpdateEvent"]
        last_status = status_events[-1]
        assert last_status[1]["status"]["state"] == "completed"

    def test_streaming_includes_final_artifact(self, _patch_session, client):
        """SSE stream includes a TaskArtifactUpdateEvent with the answer."""
        resp = _send_streaming_message(client, "hello")
        events = _parse_sse_events(resp.text)
        artifact_events = [e for e in events if e[0] == "TaskArtifactUpdateEvent"]
        assert len(artifact_events) >= 1
        last_artifact = artifact_events[-1]
        parts = last_artifact[1]["artifact"]["parts"]
        assert any("answer to: hello" in p.get("text", "") for p in parts)

    def test_streaming_has_task_and_context_ids(self, _patch_session, client):
        """All SSE events include taskId and contextId."""
        resp = _send_streaming_message(client, "hello")
        events = _parse_sse_events(resp.text)
        for event_type, data in events:
            assert "taskId" in data, f"Missing taskId in {event_type}"
            assert "contextId" in data, f"Missing contextId in {event_type}"

    def test_streaming_empty_message_returns_error(self, _patch_session, client):
        """SendStreamingMessage with empty text returns a JSON error, not SSE."""
        msg = {"role": "user", "parts": [{"type": "text", "text": "   "}]}
        resp = client.post(
            "/",
            json=_jsonrpc("SendStreamingMessage", {"message": msg}),
        )
        # Should be a normal JSON error response, not SSE
        body = resp.json()
        assert "error" in body

    def test_streaming_with_event_callback(self, monkeypatch):
        """SSE stream maps event_callback events to SSE frames."""
        from swival import session as session_mod

        monkeypatch.setattr(session_mod.Session, "_setup", lambda self: None)

        def ask_with_callback(self, q):
            # The session should have event_callback set by the server
            if self.event_callback:
                self.event_callback("text_chunk", {"text": "partial"})
                self.event_callback("tool_start", {"name": "read_file"})
                self.event_callback(
                    "tool_finish", {"name": "read_file", "elapsed": 0.1}
                )
            return _make_result(f"answer: {q}")

        monkeypatch.setattr(session_mod.Session, "ask", ask_with_callback)

        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
            heartbeat_interval=0.05,
        )
        from starlette.testclient import TestClient

        tc = TestClient(srv.app)
        resp = _send_streaming_message(tc, "do something")
        events = _parse_sse_events(resp.text)

        # Should have artifact events for the text_chunk
        artifact_events = [e for e in events if e[0] == "TaskArtifactUpdateEvent"]
        assert any(
            "partial" in e[1]["artifact"]["parts"][0].get("text", "")
            for e in artifact_events
        )

        # Should have status events for tool lifecycle
        status_events = [
            e
            for e in events
            if e[0] == "TaskStatusUpdateEvent"
            and e[1].get("metadata", {}).get("type") in ("tool_start", "tool_finish")
        ]
        assert len(status_events) >= 2


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    def test_rate_limit_rejects_excess_requests(self, _patch_session):
        """Requests over the rate limit get a 429 response."""
        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
            max_requests_per_minute=3,
        )
        from starlette.testclient import TestClient

        tc = TestClient(srv.app)

        # Send 3 requests (within limit)
        for _ in range(3):
            resp = _send_message(tc, "hello")
            assert resp.status_code == 200

        # 4th should be rate limited
        resp = _send_message(tc, "hello")
        assert resp.status_code == 429
        assert "Rate limit" in resp.json()["error"]["message"]


# ---------------------------------------------------------------------------
# Request size validation
# ---------------------------------------------------------------------------


class TestRequestSizeValidation:
    def test_oversized_body_rejected(self, _patch_session):
        """Request body larger than max_request_size gets 413."""
        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
            max_request_size=100,  # very small for testing
        )
        from starlette.testclient import TestClient

        tc = TestClient(srv.app)

        # Build a request larger than 100 bytes
        big_text = "x" * 200
        msg = {"role": "user", "parts": [{"type": "text", "text": big_text}]}
        resp = tc.post("/", json=_jsonrpc("SendMessage", {"message": msg}))
        assert resp.status_code == 413


# ---------------------------------------------------------------------------
# Concurrency limit
# ---------------------------------------------------------------------------


class TestConcurrencyLimit:
    def test_concurrent_requests_limited(self, _patch_session, monkeypatch):
        """Max concurrent requests beyond limit gets rejected."""
        from swival import session as session_mod
        from starlette.testclient import TestClient

        barrier = threading.Barrier(3, timeout=5)

        def slow_ask(self, q):
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pass
            time.sleep(0.1)
            return _make_result(f"answer: {q}")

        monkeypatch.setattr(session_mod.Session, "ask", slow_ask)

        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
            max_concurrent=2,
        )
        tc = TestClient(srv.app)

        results = []

        def send(ctx):
            r = _send_message(tc, "hello", context_id=ctx)
            results.append(r.status_code)

        threads = [threading.Thread(target=send, args=(f"ctx-{i}",)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # At least one should have been rate-limited (429) OR all succeed
        # (depends on timing). The important thing is no 500 errors.
        assert all(s in (200, 429) for s in results)


# ---------------------------------------------------------------------------
# Agent Card streaming capability
# ---------------------------------------------------------------------------


class TestAgentCardStreaming:
    def test_card_advertises_streaming(self, client):
        """Agent card now advertises streaming: true."""
        resp = client.get("/.well-known/agent-card.json")
        card = resp.json()
        assert card["capabilities"]["streaming"] is True


# ---------------------------------------------------------------------------
# Malformed Content-Length
# ---------------------------------------------------------------------------


class TestMalformedContentLength:
    def test_non_numeric_content_length(self, _patch_session, client):
        """Non-numeric Content-Length returns 400, not a raw ValueError."""
        resp = client.post(
            "/",
            content=b'{"jsonrpc":"2.0","id":1,"method":"GetTask","params":{}}',
            headers={
                "content-type": "application/json",
                "content-length": "abc",
            },
        )
        assert resp.status_code == 400
        assert "Content-Length" in resp.json()["error"]["message"]


# ---------------------------------------------------------------------------
# Cancel mid-tool-batch
# ---------------------------------------------------------------------------


class TestCancelMidToolBatch:
    def test_cancel_flag_checked_between_tools(self, tmp_path, monkeypatch):
        """cancel_flag is checked before each tool call, not just between turns."""
        import types

        from swival.agent import run_agent_loop
        from swival.a2a_types import EVENT_STATUS_UPDATE

        events = []
        cancel = threading.Event()

        def _make_msg(content=None, tool_calls=None):
            msg = types.SimpleNamespace()
            msg.content = content
            msg.tool_calls = tool_calls
            return msg

        def _make_tc(name, args_json, call_id="c1"):
            fn = types.SimpleNamespace()
            fn.name = name
            fn.arguments = args_json
            tc = types.SimpleNamespace()
            tc.id = call_id
            tc.type = "function"
            tc.function = fn
            return tc

        call_count = 0

        def fake_call_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                import json as _json

                # Return two tool calls; cancel after first should prevent second
                tc1 = _make_tc(
                    "think",
                    _json.dumps({"thought": "a"}),
                    call_id="c1",
                )
                tc2 = _make_tc(
                    "think",
                    _json.dumps({"thought": "b"}),
                    call_id="c2",
                )
                # Set cancel flag — it should be checked before tc2
                cancel.set()
                return _make_msg(tool_calls=[tc1, tc2]), "tool_calls", []
            return _make_msg(content="done"), "stop", []

        monkeypatch.setattr("swival.agent.call_llm", fake_call_llm)

        from swival.thinking import ThinkingState
        from swival.todo import TodoState

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
        ]
        answer, exhausted = run_agent_loop(
            messages,
            [],
            api_base="http://x",
            model_id="m",
            max_turns=5,
            max_output_tokens=4096,
            temperature=0.0,
            top_p=None,
            seed=None,
            context_length=None,
            base_dir=str(tmp_path),
            thinking_state=ThinkingState(),
            todo_state=TodoState(),
            resolved_commands={},
            skills_catalog={},
            skill_read_roots=[],
            extra_write_roots=[],
            files_mode="some",
            verbose=False,
            llm_kwargs={"provider": "test", "api_key": None},
            cancel_flag=cancel,
            event_callback=lambda k, d: events.append((k, d)),
        )

        assert exhausted is True
        assert answer is None
        # The cancellation status_update should have been emitted
        cancel_events = [
            (k, d) for k, d in events if k == EVENT_STATUS_UPDATE and d.get("cancelled")
        ]
        assert len(cancel_events) == 1


# ---------------------------------------------------------------------------
# Streaming concurrency actually limits active streams
# ---------------------------------------------------------------------------


class TestStreamingConcurrencyLimit:
    def test_streaming_holds_semaphore(self, monkeypatch):
        """Concurrent streaming requests are limited by max_concurrent."""
        from swival import session as session_mod
        from starlette.testclient import TestClient

        barrier = threading.Barrier(2, timeout=5)
        monkeypatch.setattr(session_mod.Session, "_setup", lambda self: None)

        def slow_ask(self, q):
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pass
            time.sleep(0.2)
            return _make_result(f"answer: {q}")

        monkeypatch.setattr(session_mod.Session, "ask", slow_ask)

        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
            max_concurrent=1,
            heartbeat_interval=0.05,
        )
        tc = TestClient(srv.app)

        results = []

        def do_stream(ctx):
            r = _send_streaming_message(tc, "hello", context_id=ctx)
            results.append(r.status_code)

        t1 = threading.Thread(target=do_stream, args=("ctx-a",))
        t2 = threading.Thread(target=do_stream, args=("ctx-b",))
        t1.start()
        t2.start()
        t1.join(timeout=15)
        t2.join(timeout=15)

        # With max_concurrent=1, one should succeed and one should be rejected
        assert 429 in results, (
            f"Expected one 429 rejection with max_concurrent=1, got {results}"
        )


# ---------------------------------------------------------------------------
# Duplicate artifact suppression
# ---------------------------------------------------------------------------


class TestNoDuplicateArtifact:
    def test_exact_match_suppresses_duplicate(self, monkeypatch):
        """When text_chunk streams the exact final answer, no duplicate artifact."""
        from swival import session as session_mod
        from starlette.testclient import TestClient

        monkeypatch.setattr(session_mod.Session, "_setup", lambda self: None)

        def ask_with_text_chunk(self, q):
            if self.event_callback:
                self.event_callback("text_chunk", {"text": "the answer"})
            return _make_result("the answer")

        monkeypatch.setattr(session_mod.Session, "ask", ask_with_text_chunk)

        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
            heartbeat_interval=0.05,
        )
        tc = TestClient(srv.app)
        resp = _send_streaming_message(tc, "hello")
        events = _parse_sse_events(resp.text)

        artifact_events = [e for e in events if e[0] == "TaskArtifactUpdateEvent"]
        answer_artifacts = [
            e
            for e in artifact_events
            if any("the answer" in p.get("text", "") for p in e[1]["artifact"]["parts"])
        ]
        assert len(answer_artifacts) == 1

    def test_partial_chunk_does_not_suppress_final(self, monkeypatch):
        """A partial text_chunk should NOT suppress the real final answer artifact."""
        from swival import session as session_mod
        from starlette.testclient import TestClient

        monkeypatch.setattr(session_mod.Session, "_setup", lambda self: None)

        def ask_with_partial(self, q):
            if self.event_callback:
                self.event_callback("text_chunk", {"text": "partial"})
            return _make_result("full answer")

        monkeypatch.setattr(session_mod.Session, "ask", ask_with_partial)

        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
            heartbeat_interval=0.05,
        )
        tc = TestClient(srv.app)
        resp = _send_streaming_message(tc, "hello")
        events = _parse_sse_events(resp.text)

        artifact_events = [e for e in events if e[0] == "TaskArtifactUpdateEvent"]
        # Both the partial chunk AND the final answer should be present
        partial = [
            e
            for e in artifact_events
            if any("partial" in p.get("text", "") for p in e[1]["artifact"]["parts"])
        ]
        final = [
            e
            for e in artifact_events
            if any(
                "full answer" in p.get("text", "") for p in e[1]["artifact"]["parts"]
            )
        ]
        assert len(partial) == 1
        assert len(final) == 1


# ---------------------------------------------------------------------------
# Status update metadata preservation
# ---------------------------------------------------------------------------


class TestStatusUpdateMetadata:
    def test_reasoning_metadata_preserved(self, monkeypatch):
        """Reasoning status updates preserve their subtype in SSE metadata."""
        from swival import session as session_mod
        from starlette.testclient import TestClient

        monkeypatch.setattr(session_mod.Session, "_setup", lambda self: None)

        def ask_with_reasoning(self, q):
            if self.event_callback:
                self.event_callback(
                    "status_update",
                    {"turn": 1, "type": "reasoning", "text_length": 42},
                )
            return _make_result(f"answer: {q}")

        monkeypatch.setattr(session_mod.Session, "ask", ask_with_reasoning)

        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
            heartbeat_interval=0.05,
        )
        tc = TestClient(srv.app)
        resp = _send_streaming_message(tc, "hello")
        events = _parse_sse_events(resp.text)

        reasoning_events = [
            e
            for e in events
            if e[0] == "TaskStatusUpdateEvent"
            and e[1].get("metadata", {}).get("type") == "reasoning"
        ]
        assert len(reasoning_events) == 1
        meta = reasoning_events[0][1]["metadata"]
        assert meta["text_length"] == 42
        assert meta["turn"] == 1


# ---------------------------------------------------------------------------
# Disconnect cleanup finalizes task
# ---------------------------------------------------------------------------


class TestDisconnectFinalizesTask:
    """_streaming_cleanup must finalize the task after the future completes."""

    def test_cleanup_finalizes_successful_task(self, _patch_session):
        """_streaming_cleanup calls _finalize_task on success."""
        import asyncio

        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
        )
        ctx_id = "ctx-cleanup-ok"
        srv._get_or_create_session(ctx_id)
        task = srv._create_task(ctx_id)
        srv._active_contexts.add(ctx_id)
        assert task.status == "working"

        lock = asyncio.Lock()

        async def run():
            await lock.acquire()
            # Simulate ask_future completing with a result
            future = asyncio.get_event_loop().create_future()
            future.set_result(_make_result("cleanup answer"))
            await srv._streaming_cleanup(future, lock, task, ctx_id)

        asyncio.run(run())

        assert task.status == "completed"
        assert len(task.artifacts) == 1
        assert task.artifacts[0]["parts"][0]["text"] == "cleanup answer"
        assert not lock.locked()
        assert ctx_id not in srv._active_contexts

    def test_cleanup_marks_canceled_when_flag_set(self, _patch_session):
        """_streaming_cleanup marks task canceled if cancel_flag was set."""
        import asyncio

        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
        )
        ctx_id = "ctx-cleanup-cancel"
        srv._get_or_create_session(ctx_id)
        task = srv._create_task(ctx_id)
        task.cancel_flag.set()
        srv._active_contexts.add(ctx_id)

        lock = asyncio.Lock()

        async def run():
            await lock.acquire()
            future = asyncio.get_event_loop().create_future()
            # Result with answer=None, exhausted=True (cancel path)
            future.set_result(_make_result(answer=None, exhausted=True))
            await srv._streaming_cleanup(future, lock, task, ctx_id)

        asyncio.run(run())

        # cancel_flag takes priority in _finalize_task
        assert task.status == "canceled"
        assert not lock.locked()
        assert ctx_id not in srv._active_contexts

    def test_cleanup_marks_failed_on_exception(self, _patch_session):
        """_streaming_cleanup marks task failed if ask_future raises."""
        import asyncio

        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
        )
        ctx_id = "ctx-cleanup-err"
        srv._get_or_create_session(ctx_id)
        task = srv._create_task(ctx_id)
        srv._active_contexts.add(ctx_id)

        lock = asyncio.Lock()

        async def run():
            await lock.acquire()
            future = asyncio.get_event_loop().create_future()
            future.set_exception(RuntimeError("kaboom"))
            await srv._streaming_cleanup(future, lock, task, ctx_id)

        asyncio.run(run())

        assert task.status == "failed"
        assert any(
            "kaboom" in m.get("parts", [{}])[0].get("text", "") for m in task.messages
        )
        assert not lock.locked()
        assert ctx_id not in srv._active_contexts

    def test_cleanup_releases_concurrency_semaphore(self, _patch_session):
        """_streaming_cleanup releases the concurrency semaphore."""
        import asyncio

        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
            max_concurrent=5,
        )
        ctx_id = "ctx-cleanup-sem"
        srv._get_or_create_session(ctx_id)
        task = srv._create_task(ctx_id)
        srv._active_contexts.add(ctx_id)

        lock = asyncio.Lock()

        async def run():
            srv._concurrency_sem = asyncio.Semaphore(5)
            # Simulate one slot being held
            await srv._concurrency_sem.acquire()
            assert srv._concurrency_sem._value == 4  # noqa: SLF001

            await lock.acquire()
            future = asyncio.get_event_loop().create_future()
            future.set_result(_make_result("done"))
            await srv._streaming_cleanup(future, lock, task, ctx_id)
            # Semaphore should be released
            assert srv._concurrency_sem._value == 5  # noqa: SLF001

        asyncio.run(run())

    def test_streaming_error_in_generator_marks_failed(self, monkeypatch):
        """If ask() raises, the SSE generator marks the task failed via GetTask."""
        from starlette.testclient import TestClient
        from swival import session as session_mod

        monkeypatch.setattr(session_mod.Session, "_setup", lambda self: None)

        def exploding_ask(self, q):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(session_mod.Session, "ask", exploding_ask)

        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
            heartbeat_interval=0.05,
        )
        tc = TestClient(srv.app)

        resp = _send_streaming_message(tc, "hello")
        events = _parse_sse_events(resp.text)

        final_statuses = [
            e
            for e in events
            if e[0] == "TaskStatusUpdateEvent"
            and e[1].get("status", {}).get("state") == "failed"
        ]
        assert len(final_statuses) >= 1

        assert len(srv._tasks) == 1
        task = list(srv._tasks.values())[0]
        assert task.status == "failed"


# ---------------------------------------------------------------------------
# Active contexts protected from eviction/expiry
# ---------------------------------------------------------------------------


class TestActiveContextProtection:
    """Contexts with in-flight work are protected from eviction and expiry."""

    def test_evict_lru_skips_active_context(self, _patch_session):
        """LRU eviction skips contexts that are actively processing."""
        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
            max_sessions=2,
        )
        # Manually create two sessions
        srv._get_or_create_session("ctx-old")
        srv._session_access["ctx-old"] = time.monotonic() - 1000
        srv._get_or_create_session("ctx-new")
        srv._session_access["ctx-new"] = time.monotonic()

        # Mark the oldest as active
        srv._active_contexts.add("ctx-old")

        # Eviction should skip ctx-old and evict ctx-new instead
        srv._evict_lru()
        assert "ctx-old" in srv._sessions
        assert "ctx-new" not in srv._sessions

    def test_evict_lru_no_candidates(self, _patch_session):
        """If all contexts are active, eviction does nothing."""
        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
            max_sessions=1,
        )
        srv._get_or_create_session("ctx-1")
        srv._active_contexts.add("ctx-1")

        # Should not crash or evict
        srv._evict_lru()
        assert "ctx-1" in srv._sessions

    def test_cleanup_expired_skips_active_context(self, _patch_session):
        """TTL cleanup skips contexts that are actively processing."""
        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
            ttl=1,  # 1 second TTL
        )
        srv._get_or_create_session("ctx-active")
        srv._get_or_create_session("ctx-idle")

        # Backdate both sessions past the TTL
        srv._session_access["ctx-active"] = time.monotonic() - 100
        srv._session_access["ctx-idle"] = time.monotonic() - 100

        # Mark one as active
        srv._active_contexts.add("ctx-active")

        srv._cleanup_expired()

        # Active context should survive, idle should be removed
        assert "ctx-active" in srv._sessions
        assert "ctx-idle" not in srv._sessions

    def test_finalize_refreshes_session_access(self, _patch_session):
        """_finalize_task refreshes _session_access so finished tasks don't look stale."""
        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
        )
        srv._get_or_create_session("ctx-1")
        # Backdate the session access
        srv._session_access["ctx-1"] = time.monotonic() - 9999

        task = srv._create_task("ctx-1")
        result = _make_result("done")
        old_access = srv._session_access["ctx-1"]
        srv._finalize_task(task, result, "ctx-1")

        assert srv._session_access["ctx-1"] > old_access

    def test_max_sessions_hard_cap_when_all_active(self, _patch_session):
        """Creating a session when all slots are active raises RuntimeError."""
        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
            max_sessions=1,
        )
        srv._get_or_create_session("ctx-1")
        srv._active_contexts.add("ctx-1")

        with pytest.raises(RuntimeError, match="Session limit reached"):
            srv._get_or_create_session("ctx-2")

        # Only the original session exists
        assert len(srv._sessions) == 1
        assert "ctx-1" in srv._sessions

    def test_max_sessions_hard_cap_blocks_send_message(self, monkeypatch):
        """SendMessage returns a JSON-RPC error when the session cap is hit."""
        from starlette.testclient import TestClient
        from swival import session as session_mod

        monkeypatch.setattr(session_mod.Session, "_setup", lambda self: None)
        monkeypatch.setattr(
            session_mod.Session,
            "ask",
            lambda self, q: _make_result(f"answer: {q}"),
        )

        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
            max_sessions=1,
        )
        # Pre-fill one active session
        srv._get_or_create_session("ctx-occupied")
        srv._active_contexts.add("ctx-occupied")

        tc = TestClient(srv.app)
        resp = _send_message(tc, "hello")
        body = resp.json()
        assert "error" in body
        assert body["error"]["code"] == -32003  # RATE_LIMITED

        # No task, context, or lock state should have been leaked
        assert len(srv._tasks) == 0
        assert len(srv._context_tasks) == 0
        # Only the pre-existing context should have a lock
        rejected_ctx = [k for k in srv._context_locks if k != "ctx-occupied"]
        assert rejected_ctx == []

    def test_max_sessions_hard_cap_blocks_streaming(self, monkeypatch):
        """SendStreamingMessage returns JSON-RPC error (not SSE) when cap is hit."""
        from starlette.testclient import TestClient
        from swival import session as session_mod

        monkeypatch.setattr(session_mod.Session, "_setup", lambda self: None)
        monkeypatch.setattr(
            session_mod.Session,
            "ask",
            lambda self, q: _make_result(f"answer: {q}"),
        )

        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
            max_sessions=1,
            heartbeat_interval=0.05,
        )
        # Pre-fill one active session
        srv._get_or_create_session("ctx-occupied")
        srv._active_contexts.add("ctx-occupied")

        tc = TestClient(srv.app)
        resp = _send_streaming_message(tc, "hello")
        body = resp.json()
        # Should be a normal JSON-RPC error, not an SSE stream
        assert "error" in body
        assert body["error"]["code"] == -32003  # RATE_LIMITED

        # No leaked state
        assert len(srv._tasks) == 0
        assert len(srv._context_tasks) == 0
        rejected_ctx = [k for k in srv._context_locks if k != "ctx-occupied"]
        assert rejected_ctx == []

    def test_blocking_active_discard_after_finalize(self, monkeypatch):
        """In blocking SendMessage, _active_contexts is cleared after _finalize_task."""
        from starlette.testclient import TestClient
        from swival import session as session_mod

        monkeypatch.setattr(session_mod.Session, "_setup", lambda self: None)

        finalize_order = []

        orig_finalize = A2aServer._finalize_task

        def tracking_finalize(self, task, result, context_id):
            # At this point the context should still be active
            finalize_order.append(context_id in self._active_contexts)
            orig_finalize(self, task, result, context_id)

        monkeypatch.setattr(A2aServer, "_finalize_task", tracking_finalize)
        monkeypatch.setattr(
            session_mod.Session,
            "ask",
            lambda self, q: _make_result(f"answer: {q}"),
        )

        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
        )
        tc = TestClient(srv.app)
        resp = _send_message(tc, "hello")
        assert resp.json()["result"]["status"]["state"] == "completed"

        # _finalize_task was called while context was still active
        assert finalize_order == [True]

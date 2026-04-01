"""Tests for swival.a2a_client: A2aManager and dispatch integration."""

import pytest

from swival.a2a_client import A2aManager, A2aShutdownError, _skill_to_tool


# --- Helpers ---


def _agent_card(skills=None, name="test-agent"):
    """Build a minimal Agent Card dict."""
    card = {
        "name": name,
        "description": f"Test agent {name}",
        "version": "0.1.0",
        "supportedInterfaces": [
            {
                "url": "https://example.com",
                "protocolBinding": "JSONRPC",
                "protocolVersion": "1.0",
            },
        ],
        "capabilities": {"streaming": False},
    }
    if skills is not None:
        card["skills"] = skills
    return card


def _jsonrpc_response(result, req_id=1):
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _completed_task(task_id="t1", context_id="c1", text="Done!"):
    return {
        "id": task_id,
        "contextId": context_id,
        "status": {"state": "completed"},
        "artifacts": [{"parts": [{"type": "text", "text": text}]}],
    }


def _input_required_task(task_id="t2", context_id="c2", text="Need more info"):
    return {
        "id": task_id,
        "contextId": context_id,
        "status": {
            "state": "input-required",
            "message": {
                "role": "agent",
                "parts": [{"type": "text", "text": text}],
            },
        },
    }


def _working_task(task_id="t3", context_id="c3"):
    return {
        "id": task_id,
        "contextId": context_id,
        "status": {"state": "working"},
    }


def _failed_task(task_id="t4", context_id="c4", text="Something went wrong"):
    return {
        "id": task_id,
        "contextId": context_id,
        "status": {
            "state": "failed",
            "message": {
                "role": "agent",
                "parts": [{"type": "text", "text": text}],
            },
        },
    }


class TestSkillToTool:
    def test_basic_skill(self):
        from swival.a2a_types import AgentSkill

        skill = AgentSkill(id="search", name="Search", description="Search stuff")
        tool = _skill_to_tool("my-agent", skill)
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "a2a__my-agent__search"
        assert "message" in tool["function"]["parameters"]["properties"]
        assert "context_id" in tool["function"]["parameters"]["properties"]
        assert "task_id" in tool["function"]["parameters"]["properties"]
        assert tool["function"]["parameters"]["required"] == ["message"]

    def test_skill_with_examples(self):
        from swival.a2a_types import AgentSkill

        skill = AgentSkill(
            id="search",
            name="Search",
            description="Search stuff",
            examples=["find cats", "look for dogs"],
        )
        tool = _skill_to_tool("agent", skill)
        assert "Examples:" in tool["function"]["description"]

    def test_skill_sanitization(self):
        from swival.a2a_types import AgentSkill

        skill = AgentSkill(id="my.skill/v2", name="test")
        tool = _skill_to_tool("agent", skill)
        assert tool["function"]["name"] == "a2a__agent__my_skill_v2"


class TestA2aManagerStartup:
    """Tests for Agent Card fetching and tool generation."""

    def test_start_fetches_card(self, monkeypatch):
        """A2aManager.start() fetches Agent Card and builds tools."""
        import httpx

        card = _agent_card(
            skills=[
                {"id": "search", "name": "Search", "description": "Search the web"},
            ]
        )

        class FakeResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return card

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def get(self, url):
                return FakeResponse()

        monkeypatch.setattr(httpx, "Client", FakeClient)

        # Suppress fmt output
        import swival.fmt

        monkeypatch.setattr(swival.fmt, "a2a_server_start", lambda *a: None)

        manager = A2aManager(
            {"my-agent": {"url": "https://example.com"}},
            verbose=False,
        )
        manager.start()

        tools = manager.list_tools()
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "a2a__my-agent__search"

        info = manager.get_tool_info()
        assert "my-agent" in info
        assert len(info["my-agent"]) == 1

        manager.close()

    def test_start_no_skills_creates_ask_tool(self, monkeypatch):
        """When no skills are declared, a single 'ask' tool is created."""
        import httpx

        card = _agent_card(skills=[])

        class FakeResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return card

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def get(self, url):
                return FakeResponse()

        monkeypatch.setattr(httpx, "Client", FakeClient)

        import swival.fmt

        monkeypatch.setattr(swival.fmt, "a2a_server_start", lambda *a: None)

        manager = A2aManager(
            {"agent": {"url": "https://example.com"}},
            verbose=False,
        )
        manager.start()

        tools = manager.list_tools()
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "a2a__agent__ask"

        manager.close()

    def test_start_card_fetch_failure(self, monkeypatch):
        """Card fetch failure is shown in verbose mode and skips the agent."""
        import httpx

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def get(self, url):
                raise httpx.ConnectError("fail")

        monkeypatch.setattr(httpx, "Client", FakeClient)

        import swival.fmt

        errors = []
        monkeypatch.setattr(
            swival.fmt, "a2a_server_error", lambda n, e: errors.append(e)
        )

        manager = A2aManager(
            {"bad-agent": {"url": "https://bad.example.com"}},
            verbose=True,
        )
        manager.start()

        assert len(errors) == 1
        assert manager.list_tools() == []

        manager.close()

    def test_start_card_fetch_failure_quiet_suppresses_output(self, monkeypatch):
        """Card fetch failure stays quiet when verbose mode is off."""
        import httpx

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def get(self, url):
                raise httpx.ConnectError("fail")

        monkeypatch.setattr(httpx, "Client", FakeClient)

        import swival.fmt

        errors = []
        monkeypatch.setattr(
            swival.fmt, "a2a_server_error", lambda n, e: errors.append(e)
        )

        manager = A2aManager(
            {"bad-agent": {"url": "https://bad.example.com"}},
            verbose=False,
        )
        manager.start()

        assert errors == []
        assert manager.list_tools() == []

        manager.close()


class TestA2aManagerCallTool:
    """Tests for call_tool() with mock HTTP."""

    def _make_manager_with_card(self, monkeypatch, card, name="test-agent"):
        """Create an A2aManager with a pre-fetched card."""
        import httpx

        class FakeCardResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return card

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def get(self, url):
                return FakeCardResponse()

        monkeypatch.setattr(httpx, "Client", FakeClient)

        import swival.fmt

        monkeypatch.setattr(swival.fmt, "a2a_server_start", lambda *a: None)
        monkeypatch.setattr(swival.fmt, "a2a_server_error", lambda *a: None)

        manager = A2aManager(
            {name: {"url": "https://example.com"}},
            verbose=False,
        )
        manager.start()
        return manager

    def test_happy_path_completed(self, monkeypatch):
        """Blocking SendMessage returns completed task in one round-trip."""
        import httpx

        card = _agent_card(
            skills=[
                {"id": "search", "name": "Search", "description": "Search"},
            ]
        )
        manager = self._make_manager_with_card(monkeypatch, card)

        response_data = _jsonrpc_response(_completed_task())

        class FakeAsyncResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return response_data

        class FakeAsyncClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def post(self, url, **kwargs):
                return FakeAsyncResponse()

        monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

        result, is_error = manager.call_tool(
            "a2a__test-agent__search",
            {"message": "find cats"},
        )

        assert not is_error
        assert "Done!" in result
        assert "contextId=c1" in result

        manager.close()

    def test_input_required(self, monkeypatch):
        """Input-required response includes contextId and taskId."""
        import httpx

        card = _agent_card(
            skills=[
                {"id": "search", "name": "Search", "description": "Search"},
            ]
        )
        manager = self._make_manager_with_card(monkeypatch, card)

        response_data = _jsonrpc_response(_input_required_task())

        class FakeAsyncResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return response_data

        class FakeAsyncClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def post(self, url, **kwargs):
                return FakeAsyncResponse()

        monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

        result, is_error = manager.call_tool(
            "a2a__test-agent__search",
            {"message": "search something"},
        )

        assert not is_error
        assert "[input-required]" in result
        assert "contextId=c2" in result
        assert "taskId=t2" in result

        manager.close()

    def test_failed_task(self, monkeypatch):
        """Failed task returns error."""
        import httpx

        card = _agent_card(
            skills=[
                {"id": "ask", "name": "Ask", "description": "Ask"},
            ]
        )
        manager = self._make_manager_with_card(monkeypatch, card)

        response_data = _jsonrpc_response(_failed_task())

        class FakeAsyncResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return response_data

        class FakeAsyncClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def post(self, url, **kwargs):
                return FakeAsyncResponse()

        monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

        result, is_error = manager.call_tool(
            "a2a__test-agent__ask",
            {"message": "do something"},
        )

        assert is_error
        assert "Something went wrong" in result

        manager.close()

    def test_polling_fallback(self, monkeypatch):
        """Non-compliant server returns working state, verify polling fallback."""
        import httpx

        call_count = {"n": 0}

        class FakeAsyncResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    # First call (SendMessage) returns working
                    return _jsonrpc_response(_working_task())
                else:
                    # Subsequent calls (GetTask) return completed
                    return _jsonrpc_response(
                        _completed_task(task_id="t3", context_id="c3")
                    )

        class FakeAsyncClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def post(self, url, **kwargs):
                return FakeAsyncResponse()

        card = _agent_card(skills=[{"id": "ask", "name": "Ask", "description": "Ask"}])
        manager = self._make_manager_with_card(monkeypatch, card)
        monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

        # Reduce poll delay so test runs fast
        import swival.a2a_client

        monkeypatch.setattr(swival.a2a_client, "_POLL_INITIAL_DELAY", 0.01)
        monkeypatch.setattr(swival.a2a_client, "_POLL_MAX_DELAY", 0.01)

        result, is_error = manager.call_tool(
            "a2a__test-agent__ask",
            {"message": "test"},
        )

        assert not is_error
        assert "Done!" in result
        assert call_count["n"] >= 2  # At least SendMessage + GetTask

        manager.close()

    def test_degradation_on_failure(self, monkeypatch):
        """Repeated failures mark agent as degraded."""
        import httpx

        card = _agent_card(skills=[{"id": "ask", "name": "Ask", "description": "Ask"}])
        manager = self._make_manager_with_card(monkeypatch, card)

        class FakeAsyncClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def post(self, url, **kwargs):
                raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

        # First call fails and marks agent as degraded
        result1, is_error1 = manager.call_tool(
            "a2a__test-agent__ask",
            {"message": "test"},
        )
        assert is_error1
        assert "failed" in result1

        # Second call gets degraded error without trying
        result2, is_error2 = manager.call_tool(
            "a2a__test-agent__ask",
            {"message": "test again"},
        )
        assert is_error2
        assert "unavailable" in result2

        manager.close()

    def test_unknown_tool(self, monkeypatch):
        """Unknown tool name returns error."""
        card = _agent_card(skills=[{"id": "ask", "name": "Ask", "description": "Ask"}])
        manager = self._make_manager_with_card(monkeypatch, card)

        result, is_error = manager.call_tool(
            "a2a__nonexistent__tool",
            {"message": "test"},
        )
        assert is_error
        assert "unknown" in result

        manager.close()

    def test_shutdown_error(self, monkeypatch):
        """call_tool after close raises A2aShutdownError."""
        card = _agent_card(skills=[{"id": "ask", "name": "Ask", "description": "Ask"}])
        manager = self._make_manager_with_card(monkeypatch, card)
        manager.close()

        with pytest.raises(A2aShutdownError):
            manager.call_tool("a2a__test-agent__ask", {"message": "test"})


class TestA2aManagerAuthHeaders:
    def test_bearer_auth(self):
        headers = A2aManager._auth_headers(
            {"auth_type": "bearer", "auth_token": "sk-123"}
        )
        assert headers["Authorization"] == "Bearer sk-123"

    def test_api_key_auth(self):
        headers = A2aManager._auth_headers(
            {"auth_type": "api_key", "auth_token": "key-456"}
        )
        assert headers["X-API-Key"] == "key-456"

    def test_no_auth(self):
        headers = A2aManager._auth_headers({})
        assert headers == {}


class TestCompactToolResult:
    """Test compact_tool_result with a2a__ tools."""

    def test_short_result_unchanged(self):
        from swival.agent import compact_tool_result

        content = "short result"
        result = compact_tool_result("a2a__agent__skill", {}, content)
        assert result == content

    def test_long_result_compacted(self):
        from swival.agent import compact_tool_result

        content = "x" * 2000
        result = compact_tool_result("a2a__agent__skill", {}, content)
        assert "compacted" in result
        assert "a2a__agent__skill" in result
        assert "First 300 chars" in result

    def test_input_required_preserves_ids(self):
        from swival.agent import compact_tool_result

        content = "[input-required] contextId=abc123 taskId=task456\n" + "x" * 2000
        result = compact_tool_result("a2a__agent__skill", {}, content)
        assert "contextId=abc123" in result
        assert "taskId=task456" in result
        assert "compacted" in result

    def test_compaction_loss_no_crash(self):
        """Aggressive compaction dropping input-required entirely doesn't crash."""
        from swival.agent import compact_tool_result

        # This tests that compact_tool_result works on regular content
        # even when the original was input-required (after aggressive drop)
        content = "[contextId=abc123]\n" + "x" * 2000
        result = compact_tool_result("a2a__agent__skill", {}, content)
        assert "compacted" in result


class TestDispatchRouting:
    """Test that dispatch routes a2a__ tools correctly."""

    def test_dispatch_a2a_no_manager(self):
        from swival.tools import dispatch

        result = dispatch("a2a__agent__skill", {"message": "hi"}, "/tmp")
        assert "error" in result
        assert "no A2A manager" in result

    def test_dispatch_a2a_with_manager(self, monkeypatch):
        from swival.tools import dispatch

        class FakeManager:
            def call_tool(self, name, args):
                return (f"result for {args['message']}", False)

        result = dispatch(
            "a2a__agent__skill",
            {"message": "hello"},
            "/tmp",
            a2a_manager=FakeManager(),
        )
        assert "[UNTRUSTED EXTERNAL CONTENT]" in result
        assert "result for hello" in result

    def test_dispatch_a2a_error(self, monkeypatch):
        from swival.tools import dispatch

        class FakeManager:
            def call_tool(self, name, args):
                return ("error: something broke", True)

        result = dispatch(
            "a2a__agent__skill",
            {"message": "hello"},
            "/tmp",
            a2a_manager=FakeManager(),
        )
        assert result.startswith("error:")


class TestA2aConfig:
    """Test A2A config validation and loading."""

    def test_validate_a2a_server_configs_valid(self):
        from swival.config import _validate_a2a_server_configs

        servers = {
            "my-agent": {
                "url": "https://example.com",
                "auth_type": "bearer",
                "auth_token": "sk-123",
                "timeout": 60,
            }
        }
        _validate_a2a_server_configs(servers, "test")

    def test_validate_a2a_server_configs_no_url(self):
        from swival.config import _validate_a2a_server_configs

        with pytest.raises(Exception, match="must have 'url'"):
            _validate_a2a_server_configs(
                {"agent": {"auth_type": "bearer"}},
                "test",
            )

    def test_validate_a2a_server_configs_bad_name(self):
        from swival.config import _validate_a2a_server_configs

        with pytest.raises(Exception, match="invalid"):
            _validate_a2a_server_configs(
                {"bad name": {"url": "https://example.com"}},
                "test",
            )

    def test_validate_a2a_server_configs_bad_type(self):
        from swival.config import _validate_a2a_server_configs

        with pytest.raises(Exception, match="expected str"):
            _validate_a2a_server_configs(
                {"agent": {"url": 123}},
                "test",
            )

    def test_load_a2a_config(self, tmp_path):
        from swival.config import load_a2a_config

        config_file = tmp_path / "a2a.toml"
        config_file.write_text(
            "[a2a_servers.my-agent]\n"
            'url = "https://example.com"\n'
            'auth_type = "bearer"\n'
            'auth_token = "sk-123"\n'
        )
        servers = load_a2a_config(config_file)
        assert "my-agent" in servers
        assert servers["my-agent"]["url"] == "https://example.com"

    def test_load_a2a_config_missing_file(self, tmp_path):
        from swival.config import load_a2a_config

        with pytest.raises(Exception):
            load_a2a_config(tmp_path / "nonexistent.toml")

    def test_load_config_with_a2a(self, tmp_path):
        """load_config picks up a2a_servers from swival.toml."""
        from swival.config import load_config

        toml_file = tmp_path / "swival.toml"
        toml_file.write_text(
            '[a2a_servers.remote]\nurl = "https://remote.example.com"\n'
        )
        config = load_config(tmp_path)
        assert "a2a_servers" in config
        assert "remote" in config["a2a_servers"]


class TestMessageResponse:
    """Test handling of direct Message responses (not Task-shaped)."""

    def _make_manager_with_card(self, monkeypatch, card, name="test-agent"):
        import httpx

        class FakeCardResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return card

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def get(self, url):
                return FakeCardResponse()

        monkeypatch.setattr(httpx, "Client", FakeClient)

        import swival.fmt

        monkeypatch.setattr(swival.fmt, "a2a_server_start", lambda *a: None)
        monkeypatch.setattr(swival.fmt, "a2a_server_error", lambda *a: None)

        manager = A2aManager(
            {name: {"url": "https://example.com"}},
            verbose=False,
        )
        manager.start()
        return manager

    def test_message_response_handled(self, monkeypatch):
        """Server returns a direct Message instead of a Task."""
        import httpx

        card = _agent_card(skills=[{"id": "ask", "name": "Ask", "description": "Ask"}])
        manager = self._make_manager_with_card(monkeypatch, card)

        # A direct message response (no "id", no "status", has "role" and "parts")
        message_result = {
            "role": "agent",
            "parts": [{"type": "text", "text": "Here is your answer"}],
            "contextId": "ctx-msg-1",
        }
        response_data = _jsonrpc_response(message_result)

        class FakeAsyncResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return response_data

        class FakeAsyncClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def post(self, url, **kwargs):
                return FakeAsyncResponse()

        monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

        result, is_error = manager.call_tool(
            "a2a__test-agent__ask",
            {"message": "hello"},
        )

        assert not is_error
        assert "Here is your answer" in result
        assert "contextId=ctx-msg-1" in result

        manager.close()

    def test_message_response_no_context(self, monkeypatch):
        """Message response without contextId still works."""
        import httpx

        card = _agent_card(skills=[{"id": "ask", "name": "Ask", "description": "Ask"}])
        manager = self._make_manager_with_card(monkeypatch, card)

        message_result = {
            "role": "agent",
            "parts": [{"type": "text", "text": "Simple reply"}],
        }
        response_data = _jsonrpc_response(message_result)

        class FakeAsyncResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return response_data

        class FakeAsyncClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def post(self, url, **kwargs):
                return FakeAsyncResponse()

        monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

        result, is_error = manager.call_tool(
            "a2a__test-agent__ask",
            {"message": "hello"},
        )

        assert not is_error
        assert "Simple reply" in result

        manager.close()

    def test_nonterminal_task_with_no_id_is_error(self, monkeypatch):
        """Non-terminal task with empty ID surfaces a protocol error."""
        import httpx

        card = _agent_card(skills=[{"id": "ask", "name": "Ask", "description": "Ask"}])
        manager = self._make_manager_with_card(monkeypatch, card)

        # A task-shaped response with empty id and working status --
        # this is a protocol error since we can't poll without an ID
        task_result = {
            "id": "",
            "contextId": "c-no-id",
            "status": {"state": "working"},
        }
        response_data = _jsonrpc_response(task_result)

        class FakeAsyncResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return response_data

        class FakeAsyncClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def post(self, url, **kwargs):
                return FakeAsyncResponse()

        monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

        result, is_error = manager.call_tool(
            "a2a__test-agent__ask",
            {"message": "hello"},
        )

        assert is_error
        assert "non-terminal" in result
        assert "without an ID" in result

        manager.close()

    def test_terminal_task_with_no_id_succeeds(self, monkeypatch):
        """Terminal task with empty ID still returns result (no polling needed)."""
        import httpx

        card = _agent_card(skills=[{"id": "ask", "name": "Ask", "description": "Ask"}])
        manager = self._make_manager_with_card(monkeypatch, card)

        task_result = {
            "id": "",
            "contextId": "c-no-id",
            "status": {"state": "completed"},
            "artifacts": [{"parts": [{"type": "text", "text": "done"}]}],
        }
        response_data = _jsonrpc_response(task_result)

        class FakeAsyncResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return response_data

        class FakeAsyncClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def post(self, url, **kwargs):
                return FakeAsyncResponse()

        monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

        result, is_error = manager.call_tool(
            "a2a__test-agent__ask",
            {"message": "hello"},
        )

        assert not is_error
        assert "done" in result

        manager.close()


class TestA2aOutputGuard:
    """Test that A2A metadata survives the large-output size guard."""

    def test_small_output_passes_through(self, tmp_path):
        from swival.tools import _guard_a2a_output

        result = "[contextId=abc]\nSmall response"
        assert _guard_a2a_output(result, str(tmp_path), "a2a__x__y") == result

    def test_large_output_preserves_context_id(self, tmp_path):
        from swival.tools import _guard_a2a_output

        body = "x" * 30_000
        result = f"[contextId=abc123]\n{body}"
        guarded = _guard_a2a_output(result, str(tmp_path), "a2a__agent__skill")

        assert guarded.startswith("[contextId=abc123]\n")
        assert "read_file" in guarded or "cmd_output" in guarded

    def test_large_output_preserves_input_required(self, tmp_path):
        from swival.tools import _guard_a2a_output

        body = "x" * 30_000
        result = f"[input-required] contextId=c1 taskId=t1\n{body}"
        guarded = _guard_a2a_output(result, str(tmp_path), "a2a__agent__skill")

        assert "[input-required] contextId=c1 taskId=t1" in guarded
        assert "read_file" in guarded or "cmd_output" in guarded

    def test_large_output_no_metadata(self, tmp_path):
        from swival.tools import _guard_a2a_output

        result = "x" * 30_000
        guarded = _guard_a2a_output(result, str(tmp_path), "a2a__agent__skill")

        assert "[input-required]" not in guarded
        assert "[contextId=" not in guarded
        assert "read_file" in guarded or "cmd_output" in guarded

    def test_large_json_array_not_treated_as_metadata(self, tmp_path):
        """A large JSON array starting with [ must not be treated as A2A metadata."""
        from swival.tools import _guard_a2a_output

        # This is a large JSON array, not A2A metadata
        result = "[" + ",".join(['"item"'] * 5000) + "]"
        assert len(result.encode("utf-8")) > 20 * 1024  # over 20KB

        guarded = _guard_a2a_output(result, str(tmp_path), "a2a__agent__skill")

        # The guard must NOT return the raw content inline
        assert len(guarded.encode("utf-8")) < 1024
        # It should be a file-save notice
        assert "read_file" in guarded or "cmd_output" in guarded
        # And no fake metadata header
        assert not guarded.startswith("[")


class TestToolCollision:
    """Test that tool name collisions are handled like MCP."""

    def test_cross_agent_collision_removes_later_agent(self):
        """When two config keys produce the same namespaced tool name,
        the colliding agent's tools are removed from the advertised set.

        Tool names are a2a__{config_key}__{skill_id}, so to get a
        cross-agent collision we directly populate _tool_schemas with
        conflicting entries and call _build_tool_map().
        """
        from swival.a2a_client import _skill_to_tool
        from swival.a2a_types import AgentSkill

        manager = A2aManager.__new__(A2aManager)
        manager._tool_schemas = {}
        manager._tool_map = {}
        manager._degraded = set()
        manager._verbose = True

        # Both agents use the same config key prefix "shared" with skill "search"
        # Simulate this by giving agent-b a schema whose namespaced name collides
        skill = AgentSkill(id="search", name="Search", description="Search")
        schema_a = _skill_to_tool("shared", skill)
        schema_b = _skill_to_tool("shared", skill)  # same namespaced name

        manager._tool_schemas["agent-a"] = [schema_a]
        manager._tool_schemas["agent-b"] = [schema_b]

        import swival.fmt

        errors = []
        orig_error = swival.fmt.a2a_server_error
        swival.fmt.a2a_server_error = lambda n, e: errors.append((n, e))
        try:
            manager._build_tool_map()
        finally:
            swival.fmt.a2a_server_error = orig_error

        # agent-b should have been removed due to collision
        assert len(errors) == 1
        assert "collision" in errors[0][1]

        # Only agent-a's tool survives
        assert len(manager._tool_map) == 1
        assert manager._tool_map["a2a__shared__search"] == ("agent-a", "search")

        # agent-b's schemas were cleared
        assert manager._tool_schemas["agent-b"] == []
        assert manager._tool_schemas["agent-a"] == [schema_a]

    def test_intra_agent_skill_collision(self, monkeypatch):
        """Skills within one agent that sanitize to the same name."""
        import httpx

        card = _agent_card(
            skills=[
                {"id": "my.search", "name": "Search1", "description": "Search 1"},
                {"id": "my/search", "name": "Search2", "description": "Search 2"},
            ]
        )

        class FakeCardResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return card

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def get(self, url):
                return FakeCardResponse()

        monkeypatch.setattr(httpx, "Client", FakeClient)

        import swival.fmt

        monkeypatch.setattr(swival.fmt, "a2a_server_start", lambda *a: None)
        errors = []
        monkeypatch.setattr(
            swival.fmt, "a2a_server_error", lambda n, e: errors.append((n, e))
        )

        manager = A2aManager(
            {"test-agent": {"url": "https://example.com"}},
            verbose=True,
        )
        manager.start()

        # Both skills sanitize to "my_search", so collision should be detected
        # and agent's tools should be removed
        assert len(errors) == 1
        assert "collision" in errors[0][1]
        assert manager.list_tools() == []

        manager.close()

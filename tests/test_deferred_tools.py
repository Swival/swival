"""Tests for deferred MCP tool schemas and tool_search."""

import json
import threading
import types

from swival.deferred_tools import (
    DEFER_MIN_TOKENS,
    MAX_LOADS,
    TOOL_SEARCH_NAME,
    DeferredTools,
    defer_mcp_tools,
)
from swival.tokens import count_tokens


def _schema(server, tool, description="", params=("x",)):
    return {
        "type": "function",
        "function": {
            "name": f"mcp__{server}__{tool}",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {p: {"type": "string"} for p in params},
            },
        },
    }


def _catalog():
    return [
        _schema("chrome", "click", "Click an element on the page", ("uid",)),
        _schema("chrome", "fill", "Type text into an input", ("uid", "value")),
        _schema("chrome", "navigate_page", "Open a URL", ("url",)),
        _schema("chrome", "take_screenshot", "Capture the page as an image"),
        _schema("chrome", "list_pages", "List open pages"),
        _schema("chrome", "close_page", "Close a page"),
        _schema("chrome", "new_page", "Open a new page", ("url",)),
        _schema("docs", "search", "Search documentation pages", ("query",)),
    ]


def _names(tools):
    return [t["function"]["name"] for t in tools]


class TestDeferral:
    def test_small_catalog_stays_eager(self):
        assert defer_mcp_tools(_catalog()[:1], provider="chatgpt") is None

    def test_large_catalog_is_deferred(self):
        big = [_schema("s", f"t{i}", "word " * 60) for i in range(20)]
        assert count_tokens(json.dumps(big)) > DEFER_MIN_TOKENS
        assert isinstance(defer_mcp_tools(big, provider="chatgpt"), DeferredTools)

    def test_command_provider_and_opt_out_stay_eager(self):
        big = [_schema("s", f"t{i}", "word " * 60) for i in range(20)]
        assert defer_mcp_tools(big, provider="command") is None
        assert defer_mcp_tools(big, provider="chatgpt", enabled=False) is None
        assert defer_mcp_tools([], provider="chatgpt") is None

    def test_search_schema_lists_names_by_server(self):
        schema = DeferredTools(_catalog()).search_schema
        assert schema["function"]["name"] == TOOL_SEARCH_NAME
        description = schema["function"]["description"]
        assert "chrome: click, fill, navigate_page" in description
        assert "\ndocs: search" in description
        assert schema["function"]["parameters"]["required"] == ["query"]


class TestSearch:
    def test_exact_name_wins_and_loads(self):
        d = DeferredTools(_catalog())
        out = d.search("click")
        assert out.splitlines()[1] == "- `mcp__chrome__click`"
        assert _names(d.with_loaded([]))[0] == "mcp__chrome__click"

    def test_full_name_is_exact_too(self):
        d = DeferredTools(_catalog())
        d.search("mcp__docs__search")
        assert _names(d.with_loaded([])) == ["mcp__docs__search"]

    def test_exact_name_loads_only_itself(self):
        d = DeferredTools(_catalog())
        out = d.search("mcp__chrome__click")
        assert _names(d.with_loaded([])) == ["mcp__chrome__click"]
        assert "Also matched" not in out
        d = DeferredTools(_catalog())
        d.search("click")
        assert _names(d.with_loaded([])) == ["mcp__chrome__click"]

    def test_name_hits_rank_above_description_hits(self):
        d = DeferredTools(_catalog())
        out = d.search("page")
        loaded = [line[3:-1] for line in out.splitlines() if line.startswith("- `")]
        assert all("page" in name for name in loaded[:4])

    def test_loads_are_capped_and_overflow_is_named(self):
        d = DeferredTools(_catalog())
        out = d.search("chrome")
        assert len(d.with_loaded([])) == MAX_LOADS
        assert "Also matched but not loaded:" in out
        assert "Search an exact tool name" in out

    def test_equal_queries_load_the_same_tools(self):
        a, b = DeferredTools(_catalog()), DeferredTools(_catalog())
        a.search("page")
        b.search("page")
        assert _names(a.with_loaded([])) == _names(b.with_loaded([]))

    def test_parameters_are_searchable(self):
        d = DeferredTools(_catalog())
        d.search("value")
        assert _names(d.with_loaded([])) == ["mcp__chrome__fill"]

    def test_empty_query_and_no_match(self):
        d = DeferredTools(_catalog())
        assert d.search("  ").startswith("error:")
        assert d.search(None).startswith("error:")
        assert d.search("kubernetes").startswith("No deferred MCP tools matched")
        assert d.with_loaded([]) == []

    def test_repeat_search_reports_already_loaded(self):
        d = DeferredTools(_catalog())
        d.search("click")
        out = d.search("click")
        assert out.startswith("Already loaded:")


class TestState:
    def test_with_loaded_appends_in_load_order_without_duplicates(self):
        d = DeferredTools(_catalog())
        base = [{"type": "function", "function": {"name": "read_file"}}]
        assert d.with_loaded(base) is base
        d.search("fill")
        d.search("click")
        tools = d.with_loaded(base)
        assert _names(tools) == ["read_file", "mcp__chrome__fill", "mcp__chrome__click"]
        assert d.with_loaded(tools) is tools
        assert d.with_loaded(None) is None

    def test_fresh_and_reset(self):
        d = DeferredTools(_catalog())
        d.search("click")
        child = d.fresh()
        assert child.with_loaded([]) == []
        child.search("fill")
        assert _names(d.with_loaded([])) == ["mcp__chrome__click"]
        d.reset()
        assert d.with_loaded([]) == []
        assert d.summary_line() is None

    def test_load_ignores_unknown_names(self):
        d = DeferredTools(_catalog())
        assert d.load("mcp__other__tool") is False
        assert d.load("mcp__chrome__click") is True
        assert d.load("mcp__chrome__click") is False


class _FakeManager:
    def __init__(self, tools=()):
        self.tools = list(tools)
        self.calls = []

    def start(self):
        pass

    def list_tools(self):
        return list(self.tools)

    def get_tool_info(self):
        return {"chrome": [(_names([t])[0], "") for t in self.tools]}

    def call_tool(self, name, args):
        self.calls.append(name)
        return "ok", False

    def is_non_idempotent_tool(self, name):
        return False

    def add_root(self, *a, **k):
        pass

    def close(self):
        pass


class TestDispatch:
    def test_tool_search_routes_to_the_state(self, tmp_path):
        from swival.tools import dispatch

        d = DeferredTools(_catalog())
        out = dispatch(
            TOOL_SEARCH_NAME, {"query": "click"}, str(tmp_path), deferred_tools=d
        )
        assert "mcp__chrome__click" in out

    def test_tool_search_without_state_is_an_error(self, tmp_path):
        from swival.tools import dispatch

        out = dispatch(TOOL_SEARCH_NAME, {"query": "click"}, str(tmp_path))
        assert out.startswith("error:")

    def test_calling_a_deferred_tool_routes_and_loads_it(self, tmp_path):
        from swival.tools import dispatch

        d = DeferredTools(_catalog())
        manager = _FakeManager()
        out = dispatch(
            "mcp__chrome__click",
            {"uid": "1"},
            str(tmp_path),
            mcp_manager=manager,
            deferred_tools=d,
        )
        assert "ok" in out
        assert manager.calls == ["mcp__chrome__click"]
        assert _names(d.with_loaded([])) == ["mcp__chrome__click"]


def _loop_kwargs(tmp_path, **overrides):
    from swival.thinking import ThinkingState
    from swival.todo import TodoState

    kwargs = dict(
        api_base="http://localhost",
        model_id="test-model",
        max_turns=4,
        max_output_tokens=1024,
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
        llm_kwargs={},
        continue_here=False,
    )
    kwargs.update(overrides)
    return kwargs


def _call(call_id, name, **args):
    return types.SimpleNamespace(
        id=call_id,
        function=types.SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


class TestLoop:
    def test_loaded_schema_reaches_the_next_request(self, tmp_path, monkeypatch):
        from swival.agent import run_agent_loop

        d = DeferredTools(_catalog())
        base = [{"type": "function", "function": {"name": "think"}}, d.search_schema]
        seen = []
        replies = [
            [_call("s1", TOOL_SEARCH_NAME, query="click")],
            [_call("c1", "mcp__chrome__click", uid="7")],
        ]

        def fake_call_llm(*args, **kwargs):
            seen.append(_names(args[7]))
            if replies:
                msg = types.SimpleNamespace(
                    content=None, tool_calls=replies.pop(0), role="assistant"
                )
                return msg, "tool_calls"
            return types.SimpleNamespace(
                content="done", tool_calls=None, role="assistant"
            ), "stop"

        monkeypatch.setattr("swival.agent.call_llm", fake_call_llm)
        manager = _FakeManager()
        messages = [{"role": "user", "content": "click it"}]
        answer, _ = run_agent_loop(
            messages,
            base,
            **_loop_kwargs(tmp_path, deferred_tools=d, mcp_manager=manager),
        )
        assert answer == "done"
        assert seen[0] == ["think", TOOL_SEARCH_NAME]
        assert seen[1] == ["think", TOOL_SEARCH_NAME, "mcp__chrome__click"]
        assert seen[2] == seen[1]
        assert manager.calls == ["mcp__chrome__click"]
        assert _names(base) == ["think", TOOL_SEARCH_NAME]

    def test_loaded_schemas_persist_into_the_next_loop(self, tmp_path, monkeypatch):
        from swival.agent import run_agent_loop

        d = DeferredTools(_catalog())
        d.search("fill")
        seen = []

        def fake_call_llm(*args, **kwargs):
            seen.append(_names(args[7]))
            return types.SimpleNamespace(
                content="done", tool_calls=None, role="assistant"
            ), "stop"

        monkeypatch.setattr("swival.agent.call_llm", fake_call_llm)
        run_agent_loop(
            [{"role": "user", "content": "hi"}],
            [d.search_schema],
            **_loop_kwargs(tmp_path, deferred_tools=d),
        )
        assert seen == [[TOOL_SEARCH_NAME, "mcp__chrome__fill"]]

    def test_subagents_get_their_own_loaded_set(self, monkeypatch):
        from swival.subagent import (
            SubagentHandle,
            _CompositeCancelFlag,
            _subagent_thread_fn,
        )

        d = DeferredTools(_catalog())
        d.search("click")
        seen = {}

        def fake_loop(messages, tools, **kwargs):
            seen["deferred"] = kwargs.get("deferred_tools")
            return "done", False

        monkeypatch.setattr("swival.agent.run_agent_loop", fake_loop)
        handle = SubagentHandle(id="sub_1", task="t")
        slot = threading.Semaphore(1)
        slot.acquire()
        _subagent_thread_fn(
            handle,
            {"base_dir": "/tmp/test", "deferred_tools": d},
            [],
            "task",
            5,
            None,
            None,
            _CompositeCancelFlag(None, handle.cancel_flag),
            slot,
        )
        child = seen["deferred"]
        assert child is not d
        assert child.with_loaded([]) == []
        assert "mcp__chrome__fill" in child

    def test_clear_resets_loaded_schemas(self):
        from swival.agent import _repl_clear
        from swival.thinking import ThinkingState

        d = DeferredTools(_catalog())
        d.search("click")
        _repl_clear([], ThinkingState(), deferred_tools=d)
        assert d.with_loaded([]) == []


class TestPromptAndSession:
    def test_prompt_section_mentions_tool_search_when_deferred(self):
        from swival.agent import _format_mcp_tool_info

        info = {"chrome": [("mcp__chrome__click", "Click")]}
        assert TOOL_SEARCH_NAME not in _format_mcp_tool_info(info)
        assert TOOL_SEARCH_NAME in _format_mcp_tool_info(info, deferred=True)

    def test_session_defers_a_large_catalog(self, tmp_path, monkeypatch):
        from swival import agent, mcp_client
        from swival.session import Session

        big = [_schema("chrome", f"t{i}", "word " * 60) for i in range(20)]
        monkeypatch.setattr(mcp_client, "McpManager", lambda *a, **k: _FakeManager(big))
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        seen = []

        def fake_call_llm(*args, **kwargs):
            seen.append(_names(args[7]))
            return types.SimpleNamespace(
                content="done", tool_calls=None, role="assistant"
            ), "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        s = Session(
            base_dir=str(tmp_path),
            history=False,
            mcp_servers={"chrome": {"command": "true"}},
        )
        s.run("hi")
        assert TOOL_SEARCH_NAME in seen[0]
        assert not any(name.startswith("mcp__") for name in seen[0])

    def test_session_opt_out_keeps_schemas(self, tmp_path, monkeypatch):
        from swival import agent, mcp_client
        from swival.session import Session

        big = [_schema("chrome", f"t{i}", "word " * 60) for i in range(20)]
        monkeypatch.setattr(mcp_client, "McpManager", lambda *a, **k: _FakeManager(big))
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        seen = []

        def fake_call_llm(*args, **kwargs):
            seen.append(_names(args[7]))
            return types.SimpleNamespace(
                content="done", tool_calls=None, role="assistant"
            ), "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        s = Session(
            base_dir=str(tmp_path),
            history=False,
            mcp_servers={"chrome": {"command": "true"}},
            defer_mcp_schemas=False,
        )
        s.run("hi")
        assert TOOL_SEARCH_NAME not in seen[0]
        assert sum(name.startswith("mcp__") for name in seen[0]) == 20


class TestForkedConversations:
    def test_copy_keeps_loads_but_stays_independent(self):
        d = DeferredTools(_catalog())
        d.search("click")
        clone = d.copy()
        assert _names(clone.with_loaded([])) == ["mcp__chrome__click"]
        clone.search("fill")
        assert _names(d.with_loaded([])) == ["mcp__chrome__click"]

    def test_side_conversations_do_not_load_into_the_parent(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent))
        from test_btw_command import _make_ctx

        from swival import agent

        d = DeferredTools(_catalog())
        d.search("click")
        for kind in ("btw", "loop"):
            ctx = _make_ctx()
            ctx.loop_kwargs["deferred_tools"] = d
            isolated = agent._build_isolated_ctx(ctx, kind)
            child = isolated.loop_kwargs["deferred_tools"]
            assert child is not d
            child.search("fill")
            assert _names(d.with_loaded([])) == ["mcp__chrome__click"]
            if isolated.subagent_manager is not None:
                isolated.subagent_manager.shutdown()

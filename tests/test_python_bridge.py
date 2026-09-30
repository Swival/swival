"""run_python snippets calling MCP tools through the host bridge."""

import concurrent.futures
import json
import os
import sys
import textwrap
import threading
import time
import types

import pytest
from mcp.types import CallToolResult, TextContent

from swival import mcp_client, python_bridge
from swival.agent import handle_tool_call
from swival.config import ConfigError, _validate_mcp_server_configs
from swival.deferred_tools import DeferredTools
from swival.mcp_client import (
    McpData,
    McpManager,
    _host_error,
    _normalize_result,
    _result_data,
)
from swival.report import ReportCollector
from swival.tools import MCP_INLINE_LIMIT, PYTHON_TOOL, UNTRUSTED_MARKER, dispatch

pytestmark = pytest.mark.skipif(
    not python_bridge.bridge_available(), reason="the bridge needs POSIX pipes"
)

_SERVER = r'''
"""Bare JSON-RPC MCP server with one tool per result shape."""

import json
import os
import sys
import threading
import time

DEEP = {"type": "object", "properties": {"filter": {"type": "object", "properties": {
    "range": {"type": "object", "properties": {
        "lo": {"type": "integer"}, "hi": {"type": "integer"},
    }},
}}}}

TOOLS = {
    "rows": {"inputSchema": {"type": "object", "properties": {"page": {"type": "integer"}}},
             "outputSchema": {"type": "object", "properties": {"rows": {"type": "array"}}},
             "description": "List rows.\nOne page at a time."},
    "echo": {"inputSchema": DEEP},
    "json_text": {}, "plain": {}, "mixed": {}, "empty": {}, "fail": {},
    "slow": {}, "big": {}, "bigfail": {}, "die": {}, "hidden": {},
    "noshape": {"outputSchema": {"type": "object", "properties": {"n": {"type": "integer"}},
                                 "required": ["n"]}},
}

lock = threading.Lock()


def send(payload):
    with lock:
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()


def text(value):
    return {"type": "text", "text": value}


def result(name, args):
    if name == "rows":
        page = args.get("page", 1)
        rows = [{"id": page * 10 + i, "page": page} for i in range(3)]
        return {"content": [text(json.dumps({"rows": rows}))],
                "structuredContent": {"rows": rows}}
    if name == "echo":
        return {"content": [], "structuredContent": {"received": args}}
    if name == "json_text":
        return {"content": [text('{"a": [1, 2]}')]}
    if name == "plain":
        return {"content": [text("hello " + args.get("who", "world"))]}
    if name == "mixed":
        return {"content": [text('{"a": 1}'),
                            {"type": "resource_link", "uri": "file:///r.csv", "name": "r.csv"}]}
    if name == "empty":
        return {"content": []}
    if name == "fail":
        return {"content": [text("quota exceeded")], "isError": True}
    if name == "die":
        os._exit(1)
    if name == "noshape":
        return {"content": [text("no structured content")]}
    if name == "slow":
        # Finishes even after the client gives up, like a real server would.
        time.sleep(args.get("seconds", 5))
        if args.get("marker"):
            open(args["marker"], "w").close()
        return {"content": [text("slow done")]}
    if name == "big":
        return {"content": [text("x" * args.get("size", 1000))]}
    if name == "bigfail":
        return {"content": [text("e" * args.get("size", 1000))], "isError": True}
    return {"content": [text("hidden")]}


def answer(msg):
    params = msg["params"]
    send({"jsonrpc": "2.0", "id": msg["id"],
          "result": result(params["name"], params.get("arguments") or {})})


while True:
    line = sys.stdin.readline()
    if not line:
        break
    msg = json.loads(line)
    method = msg.get("method")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "bridge-fixture", "version": "0"},
        }})
    elif method == "tools/list":
        tools = []
        for name, spec in TOOLS.items():
            tool = {"name": name, "inputSchema": spec.get("inputSchema", {"type": "object"})}
            for key in ("outputSchema", "description"):
                if key in spec:
                    tool[key] = spec[key]
            tools.append(tool)
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {"tools": tools}})
    elif method == "tools/call":
        threading.Thread(target=answer, args=(msg,), daemon=True).start()
'''

_BRIDGED = ["rows", "echo", "json_text", "plain", "mixed", "empty", "fail"]
_BRIDGED += ["slow", "big", "bigfail", "die"]


def _start_manager(tmp_path, name, python_tools):
    script = tmp_path / "server.py"
    script.write_text(_SERVER)
    config = {"command": sys.executable, "args": [str(script)]}
    mgr = McpManager({name: {**config, "python_tools": python_tools}})
    mgr.start()
    return mgr


@pytest.fixture(scope="module")
def manager(tmp_path_factory):
    mgr = _start_manager(tmp_path_factory.mktemp("bridge"), "fx", _BRIDGED)
    yield mgr
    mgr.close()


def _run(manager, tmp_path, code, timeout=20, **kwargs):
    report = kwargs.pop("report", None) or ReportCollector()
    out = dispatch(
        "run_python",
        {"code": textwrap.dedent(code), "timeout": timeout},
        str(tmp_path),
        mcp_manager=manager,
        commands_unrestricted=True,
        report=report,
        **kwargs,
    )
    return out, report


def _body(out: str) -> str:
    """The printed output, without the untrusted header or trailing newline."""
    if out.startswith(UNTRUSTED_MARKER):
        out = out.split("\n\n", 1)[1]
    return out.rstrip("\n")


def _event(report) -> dict:
    (event,) = [e for e in report.events if e["type"] == "python_bridge"]
    return event


def _outcomes(event) -> list[str]:
    return [call["outcome"] for call in event["calls"]]


def _bridge(manager=None, allowed=("t",)):
    return python_bridge.PythonBridge(
        manager, list(allowed), deadline=time.monotonic() + 10, gate=lambda *a: None
    )


def _call_frame(name="t") -> bytes:
    return json.dumps({"op": "call", "name": name}).encode()


class _Answers:
    """A manager whose calls all succeed."""

    def call_tool_data(self, *args, **kwargs):
        return McpData("ok")


def _spy_calls(manager, monkeypatch) -> list:
    calls = []
    original = manager.call_tool_data

    def spy(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(manager, "call_tool_data", spy)
    return calls


def _wait_for(path, seconds=3) -> bool:
    deadline = time.monotonic() + seconds
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    return path.exists()


class TestResults:
    def test_only_printed_output_comes_back(self, manager, tmp_path):
        out, report = _run(
            manager,
            tmp_path,
            """
            import swival_tools as st
            pages = [st.call('mcp__fx__rows', {'page': p}) for p in (1, 2, 3)]
            ids = [row['id'] for page in pages for row in page['rows']]
            print(sum(ids), len(ids))
            """,
        )
        assert _body(out) == "189 9"
        event = _event(report)
        assert event["child_calls"] == 3
        assert event["exit_code"] == 0

    @pytest.mark.parametrize(
        "tool, expected",
        [
            ("json_text", {"a": [1, 2]}),
            ("plain", "hello world"),
            ("mixed", '{"a": 1}\n[resource link: file:///r.csv (r.csv)]'),
            ("empty", None),
        ],
    )
    def test_value_shapes(self, manager, tmp_path, tool, expected):
        out, _ = _run(
            manager,
            tmp_path,
            f"""
            import json, swival_tools as st
            print(json.dumps(st.call('mcp__fx__{tool}')))
            """,
        )
        assert json.loads(_body(out)) == expected

    @pytest.mark.parametrize(
        "code, origin",
        [
            ("print('hi')", None),
            ("print(st.call('mcp__fx__plain'))", "mcp__fx__plain"),
            ("print(st.help('mcp__fx__rows'))", "mcp__fx__rows"),
            ("print(st.help())", "mcp__fx__rows"),
        ],
    )
    def test_output_is_labeled_once_server_text_arrives(
        self, manager, tmp_path, code, origin
    ):
        out, report = _run(manager, tmp_path, f"import swival_tools as st\n{code}\n")
        if origin is None:
            assert out == "hi\n"
            assert report.security_stats["untrusted_inputs"] == 0
        else:
            assert out.startswith(f"{UNTRUSTED_MARKER}\nsource: run_python\n")
            assert origin in out.split("\n")[2]
            assert report.security_stats["untrusted_inputs"] == 1

    def test_timeout_status_stays_first(self, manager, tmp_path):
        out, _ = _run(
            manager,
            tmp_path,
            """
            import time, swival_tools as st
            print(st.call('mcp__fx__plain'), flush=True)
            time.sleep(30)
            """,
            timeout=1,
        )
        assert out.startswith("error: command timed out after 1s\n")
        assert UNTRUSTED_MARKER in out


class TestResultData:
    @pytest.mark.parametrize(
        "text, structured, expected",
        [
            ("summary", {}, {}),
            ("summary", {"n": 0}, {"n": 0}),
            ('{"ok": false, "note": "busy"}', {"records": [1]}, {"records": [1]}),
            (json.dumps({"ok": True, "result": [1, 2]}), None, [1, 2]),
            ("42", None, "42"),
            ("1" * 5000, None, "1" * 5000),
        ],
    )
    def test_values(self, text, structured, expected):
        fields = {"content": [TextContent(type="text", text=text)]}
        if structured is not None:
            fields["structured_content"] = structured
        result = CallToolResult(**fields)
        assert _result_data(result) == McpData(expected)
        if structured is None and not text.startswith("{"):
            assert _normalize_result(result).text == text

    def test_error_keeps_server_details(self):
        result = CallToolResult(
            content=[TextContent(type="text", text="bad key")],
            structured_content={"partial": 1},
            is_error=True,
        )
        data = _result_data(result)
        assert data.value is None
        assert data.error.details == "bad key"


class TestFailures:
    def test_tool_failure_is_catchable(self, manager, tmp_path):
        out, report = _run(
            manager,
            tmp_path,
            """
            import swival_tools as st
            try:
                st.call('mcp__fx__fail')
            except st.ToolError as e:
                print('caught', repr(str(e)))
            print(st.call('mcp__fx__plain', {'who': 'again'}))
            """,
        )
        assert _body(out).splitlines() == [
            "caught 'error: MCP tool returned an error\\nquota exceeded'",
            "hello again",
        ]
        assert _event(report)["child_failures"] == 1

    def test_refused_calls(self, manager, tmp_path):
        class Policy:
            def check(self, name, args):
                if name == "mcp__fx__plain":
                    return "error: tool 'mcp__fx__plain' is not available"
                return None

        out, _ = _run(
            manager,
            tmp_path,
            """
            import swival_tools as st
            for name in ('mcp__fx__hidden', 'mcp__other__x', 'mcp__fx__plain'):
                try:
                    st.call(name)
                except st.ToolError as e:
                    print(e)
            print(st.call('mcp__fx__json_text'))
            """,
            tool_policy=Policy(),
        )
        assert _body(out).splitlines() == [
            "error: 'mcp__fx__hidden' cannot be called from run_python; "
            "see swival_tools.ALL_TOOLS",
            "error: 'mcp__other__x' cannot be called from run_python; "
            "see swival_tools.ALL_TOOLS",
            "error: tool 'mcp__fx__plain' is not available",
            "{'a': [1, 2]}",
        ]

    def test_bad_requests_raise_value_error(self, manager, tmp_path):
        out, _ = _run(
            manager,
            tmp_path,
            """
            import swival_tools as st
            for attempt in (
                lambda: st.call('mcp__fx__plain', [1]),
                lambda: st.call('mcp__fx__plain', timeout=-1),
                lambda: st.call(['mcp__fx__plain']),
                lambda: st.help(['mcp__fx__plain']),
            ):
                try:
                    attempt()
                except ValueError as e:
                    print(e)
            print(st.call('mcp__fx__plain'))
            """,
        )
        assert _body(out).splitlines() == [
            "arguments must be a dict",
            "timeout must be a positive number of seconds",
            "the tool name must be a string",
            "the tool name must be a string",
            "hello world",
        ]

    def test_server_error_text_is_clipped(self, manager, tmp_path):
        out, _ = _run(
            manager,
            tmp_path,
            """
            import swival_tools as st
            try:
                st.call('mcp__fx__bigfail', {'size': 200000})
            except st.ToolError as e:
                print(len(str(e).encode()), str(e).endswith('[error output truncated]'))
            """,
        )
        size, marked = _body(out).split()
        assert int(size) <= MCP_INLINE_LIMIT + 30
        assert marked == "True"

    def test_unexpected_failure_is_recorded(self, manager, tmp_path, monkeypatch):
        def fail(*args, **kwargs):
            time.sleep(0.01)
            raise RuntimeError("parser broke")

        monkeypatch.setattr(manager, "call_tool_data", fail)
        out, report = _run(
            manager,
            tmp_path,
            """
            import swival_tools as st
            try:
                st.call('mcp__fx__plain')
            except st.ToolError as e:
                print(e)
            """,
        )
        assert _body(out) == "error: the call failed: RuntimeError: parser broke"
        event = _event(report)
        assert event["child_calls"] == event["child_failures"] == 1
        (record,) = event["calls"]
        assert record["tool"] == "mcp__fx__plain"
        assert record["outcome"] == "error"
        assert record["duration_s"] >= 0.01

    def test_output_schema_mismatch_keeps_the_server(self, manager):
        data = manager.call_tool_data("mcp__fx__noshape", {}, 5)
        assert data.error.text == "error: MCP server 'fx' returned an invalid result"
        assert "has an output schema" in data.error.details
        assert data.cause is None
        assert manager.call_tool("mcp__fx__noshape", {}).is_error
        assert not manager._degraded
        assert manager.call_tool_data("mcp__fx__plain", {}, 5).value == "hello world"

    def test_lost_connection_is_uncertain(self, tmp_path):
        mgr = _start_manager(tmp_path, "doomed", ["die", "fail"])
        try:
            answered = mgr.call_tool_data("mcp__doomed__fail", {}, 10)
            assert answered.error is not None and answered.cause is None
            out, report = _run(
                mgr,
                tmp_path,
                """
                import swival_tools as st
                try:
                    st.call('mcp__doomed__die')
                except st.ToolError as e:
                    print(str(e).splitlines()[0])
                """,
            )
        finally:
            mgr.close()
        assert "the server may have carried out the call" in out
        event = _event(report)
        assert event["uncertain_calls"] == 1
        assert event["calls"][0]["remote_outcome"] == "unknown"

    def test_connection_loss_belongs_to_its_own_call(self, manager, monkeypatch):
        def rejected_while_another_call_breaks(*args, **kwargs):
            manager._degraded.add("fx")
            return _host_error("MCP server 'fx' rejected the call", "bad params")

        monkeypatch.setattr(manager, "_call_raw", rejected_while_another_call_breaks)
        try:
            data = manager.call_tool_data("mcp__fx__plain", {}, 5)
        finally:
            manager._degraded.discard("fx")
        assert data.error is not None
        assert data.cause is None


class TestDiscovery:
    def test_help(self, manager, tmp_path):
        out, _ = _run(
            manager,
            tmp_path,
            """
            import json, swival_tools as st
            try:
                st.help('mcp__fx__hidden')
            except st.ToolError as e:
                print(e)
            answers = [st.help(), st.help('mcp__fx__rows'), st.help('mcp__fx__plain')]
            print(json.dumps(answers))
            """,
        )
        refused, answers = _body(out).splitlines()
        assert "cannot be called from run_python" in refused
        summary, rows, plain = json.loads(answers)
        assert sorted(summary) == sorted(f"mcp__fx__{n}" for n in _BRIDGED)
        assert summary["mcp__fx__rows"] == "List rows."
        assert rows["description"] == "List rows.\nOne page at a time."
        assert rows["output_schema"]["properties"]["rows"] == {"type": "array"}
        assert plain["output_schema"] is None

    def test_oversized_help_is_not_a_failed_call(self, monkeypatch):
        class Manager:
            def tool_metadata(self, name):
                return {"name": name, "description": "x" * (2 << 20)}

        monkeypatch.setattr(python_bridge, "MAX_RESPONSE_BYTES", 1 << 20)
        bridge = _bridge(Manager())
        reply = json.loads(bridge._handle(b'{"op": "help", "name": "t"}'))
        assert "over the 1048576-byte limit" in reply["error"]
        assert bridge.stats.failures == 0
        assert bridge.stats.records == []

    def test_code_path_keeps_nested_arguments(self, manager, tmp_path):
        nested = {"filter": {"range": {"lo": 1, "hi": 5}}}
        out, _ = _run(
            manager,
            tmp_path,
            f"""
            import json, swival_tools as st
            schema = st.help('mcp__fx__echo')['input_schema']
            print(json.dumps([schema, st.call('mcp__fx__echo', {nested!r})]))
            """,
        )
        schema, echoed = json.loads(_body(out))
        assert "filter" in schema["properties"]
        assert echoed == {"received": nested}

        # The model gets a flattened schema, and a direct call is re-nested.
        echo = next(
            t for t in manager.list_tools() if t["function"]["name"] == "mcp__fx__echo"
        )
        props = echo["function"]["parameters"]["properties"]
        flat = {
            next(p for p in props if p.endswith(leaf)): value
            for leaf, value in (("lo", 1), ("hi", 5))
        }
        direct = dispatch("mcp__fx__echo", flat, str(tmp_path), mcp_manager=manager)
        assert json.loads(_body(direct)) == {"received": nested}

    def test_code_calls_do_not_load_deferred_schemas(self, manager, tmp_path):
        deferred = DeferredTools(manager.list_tools())
        _run(
            manager,
            tmp_path,
            """
            import swival_tools as st
            st.help('mcp__fx__rows')
            st.call('mcp__fx__rows', {'page': 1})
            """,
            deferred_tools=deferred,
        )
        assert deferred.with_loaded([]) == []

        # A direct call does load the schema, which is what makes the
        # check above meaningful.
        dispatch(
            "mcp__fx__rows",
            {"page": 1},
            str(tmp_path),
            mcp_manager=manager,
            deferred_tools=deferred,
        )
        loaded = deferred.with_loaded([])
        assert [t["function"]["name"] for t in loaded] == ["mcp__fx__rows"]


class TestStops:
    def test_run_timeout_cuts_off_a_slow_call(self, manager, tmp_path):
        marker = tmp_path / "done"
        started = time.monotonic()
        out, report = _run(
            manager,
            tmp_path,
            f"""
            import swival_tools as st
            try:
                st.call('mcp__fx__slow', {{'seconds': 1.5, 'marker': {str(marker)!r}}})
            except st.BridgeError as e:
                print(e)
            """,
            timeout=1,
            tool_meta=(tool_meta := {}),
        )
        assert time.monotonic() - started < 4
        assert not marker.exists()
        event = _event(report)
        assert event["stop"] == "the run_python timeout expired"
        assert event["uncertain_calls"] == 1
        assert event["calls"][0]["remote_outcome"] == "unknown"
        assert tool_meta["_swival_child_calls"]["calls"] == event["calls"]
        assert "unfinished_call" not in event
        assert (
            "the run_python timeout expired; the call to mcp__fx__slow was cut off "
            "and may still complete on the server"
        ) in out
        assert out.endswith(
            "note: 1 tool call was cut off before the server answered; "
            "whether it took effect is unknown"
        )
        assert _wait_for(marker)

    def test_requested_timeout_is_a_tool_error(self, manager, tmp_path):
        marker = tmp_path / "done"
        out, report = _run(
            manager,
            tmp_path,
            f"""
            import swival_tools as st
            args = {{'seconds': 0.4, 'marker': {str(marker)!r}}}
            try:
                st.call('mcp__fx__slow', args, timeout=0.05)
            except st.ToolError as e:
                print(e)
            print(st.call('mcp__fx__plain'))
            """,
        )
        assert not marker.exists()
        assert _body(out).splitlines() == [
            "error: MCP tool 'mcp__fx__slow' timed out after 0.05s; "
            "it may still complete on the server",
            "hello world",
            "",
            "note: 1 tool call was cut off before the server answered; "
            "whether it took effect is unknown",
        ]
        event = _event(report)
        assert "stop" not in event
        assert event["uncertain_calls"] == 1
        assert event["calls"][0]["outcome"] == "timeout"
        assert event["calls"][0]["remote_outcome"] == "unknown"
        assert _wait_for(marker)

    def test_call_cap(self, manager, tmp_path, monkeypatch):
        monkeypatch.setattr(python_bridge, "MAX_CALLS", 2)
        out, report = _run(
            manager,
            tmp_path,
            """
            import swival_tools as st
            for i in range(4):
                try:
                    st.call('mcp__fx__plain')
                    print('ok', i)
                except st.BridgeError as e:
                    print('stopped', i, e)
            """,
        )
        assert _body(out).splitlines() == [
            "ok 0",
            "ok 1",
            "stopped 2 the program made 2 tool calls, the most allowed",
            "stopped 3 the program made 2 tool calls, the most allowed",
        ]
        assert _event(report)["child_calls"] == 2

    @pytest.mark.parametrize("tool", ["big", "bigfail"])
    def test_data_total(self, manager, tmp_path, monkeypatch, tool):
        monkeypatch.setattr(python_bridge, "MAX_TOTAL_BYTES", 4000)
        out, report = _run(
            manager,
            tmp_path,
            f"""
            import swival_tools as st
            for i in range(20):
                try:
                    st.call('mcp__fx__{tool}', {{'size': 1000}})
                except st.ToolError:
                    pass
                except st.BridgeError as e:
                    print(e)
                    break
            """,
        )
        assert _body(out) == "the program exchanged more than 4000 bytes with its tools"
        event = _event(report)
        # Only the short stop reply itself may go past the total.
        assert event["bytes_in"] + event["bytes_out"] <= 4000 + 200

    def test_oversized_result_fails_only_its_call(self, manager, tmp_path, monkeypatch):
        monkeypatch.setattr(python_bridge, "MAX_RESPONSE_BYTES", 1000)
        monkeypatch.setattr(python_bridge, "MAX_TOTAL_BYTES", 3000)
        out, _ = _run(
            manager,
            tmp_path,
            """
            import swival_tools as st
            for _ in range(3):
                st.call('mcp__fx__big', {'size': 500})
            try:
                st.call('mcp__fx__big', {'size': 5000})
            except st.ToolError as e:
                print(e)
            print(st.call('mcp__fx__plain'))
            """,
        )
        assert _body(out).splitlines() == [
            "error: the result is over the 1000-byte limit for one call",
            "hello world",
        ]

    def test_over_budget_request_never_reaches_the_tool(
        self, manager, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(python_bridge, "MAX_TOTAL_BYTES", 512)
        calls = _spy_calls(manager, monkeypatch)
        out, report = _run(
            manager,
            tmp_path,
            """
            import swival_tools as st
            for args in ({'who': 'x' * 4096}, {}):
                try:
                    st.call('mcp__fx__plain', args)
                except st.BridgeError as e:
                    print(e)
            """,
        )
        assert calls == []
        assert (
            _body(out).splitlines()
            == ["the program exchanged more than 512 bytes with its tools"] * 2
        )
        event = _event(report)
        assert event["child_calls"] == 0
        # The snippet's module refuses the second call without sending it.
        assert _outcomes(event) == ["stopped"]

    def test_no_traffic_after_a_stop(self, manager, tmp_path, monkeypatch):
        monkeypatch.setattr(python_bridge, "MAX_TOTAL_BYTES", 512)
        out, report = _run(
            manager,
            tmp_path,
            """
            import swival_tools as st
            stops = 0
            for _ in range(2000):
                try:
                    st.help()
                except st.BridgeError:
                    stops += 1
            print(stops)
            """,
        )
        event = _event(report)
        assert event["bytes_in"] + event["bytes_out"] < 2 * 512
        assert int(_body(out)) > 1990

    def test_oversized_request_ends_the_bridge(self, manager, tmp_path, monkeypatch):
        monkeypatch.setattr(python_bridge, "MAX_REQUEST_BYTES", 200)
        out, _ = _run(
            manager,
            tmp_path,
            """
            import swival_tools as st
            for who in ('x' * 500, 'y'):
                try:
                    st.call('mcp__fx__plain', {'who': who})
                except st.BridgeError as e:
                    print(e)
            """,
        )
        assert (
            _body(out).splitlines()
            == ["a request of 566 bytes exceeds the 200-byte limit"] * 2
        )

    @pytest.mark.parametrize("stop", ["cancel", "budget"])
    def test_a_call_finishing_as_the_run_stops_keeps_its_result(
        self, manager, tmp_path, monkeypatch, stop
    ):
        cancel = threading.Event()
        goal = types.SimpleNamespace(exhausted=False)
        goal.budget_exhausted = lambda: goal.exhausted
        original = manager.call_tool_data

        def call_then_stop(*args, **kwargs):
            data = original(*args, **kwargs)
            cancel.set() if stop == "cancel" else setattr(goal, "exhausted", True)
            return data

        monkeypatch.setattr(manager, "call_tool_data", call_then_stop)
        out, report = _run(
            manager,
            tmp_path,
            """
            import swival_tools as st
            print(st.call('mcp__fx__plain'), flush=True)
            try:
                st.call('mcp__fx__plain')
            except st.BridgeError as e:
                print('stopped:', e, flush=True)
            """,
            cancel_flag=cancel if stop == "cancel" else None,
            goal_state=goal,
        )
        assert "hello world" in out
        if stop == "budget":
            assert _body(out).endswith("stopped: the goal's budget is exhausted")
        assert _outcomes(_event(report)) == ["ok", "stopped"]

    def test_cancel_gives_up_on_a_call_in_flight(self, manager, tmp_path):
        cancel = threading.Event()
        threading.Timer(0.3, cancel.set).start()
        threads = threading.active_count()
        started = time.monotonic()
        out, report = _run(
            manager,
            tmp_path,
            "import swival_tools as st\nst.call('mcp__fx__slow', {'seconds': 10})\n",
            cancel_flag=cancel,
        )
        assert time.monotonic() - started < 2
        event = _event(report)
        assert event["stop"] == "the user cancelled the run"
        assert event["uncertain_calls"] == 1
        assert _outcomes(event) == ["stopped"]
        assert "cut off before the server answered" in out
        assert threading.active_count() <= threads

    @pytest.mark.parametrize(
        "bridged, preset", [(True, False), (False, False), (False, True)]
    )
    def test_cancel_kills_the_program(self, manager, tmp_path, bridged, preset):
        cancel = threading.Event()
        if preset:
            cancel.set()
        else:
            threading.Timer(0.3, cancel.set).start()
        started = time.monotonic()
        out = dispatch(
            "run_python",
            {"code": "import time\nprint('start', flush=True)\ntime.sleep(30)\n"},
            str(tmp_path),
            mcp_manager=manager if bridged else None,
            commands_unrestricted=True,
            cancel_flag=cancel,
        )
        assert time.monotonic() - started < 3
        assert out.startswith("error: cancelled before the program finished")

    def test_an_abandoned_call_is_cancelled_before_the_first_wait(self, manager):
        started = time.monotonic()
        data = manager.call_tool_data(
            "mcp__fx__slow", {"seconds": 0.2}, 5, abort=lambda: True
        )
        assert data.cause == "aborted"
        assert time.monotonic() - started < 0.1

    def test_an_answer_arriving_at_the_deadline_is_kept(self, manager, monkeypatch):
        real_wait = concurrent.futures.wait
        lied = []

        def late_wait(futures, timeout=None):
            # The first wait wakes up after the deadline and sees nothing
            # done, although the call has finished by then.
            if lied:
                return real_wait(futures, timeout=timeout)
            lied.append(True)
            real_wait(futures, timeout=5)
            time.sleep(0.1)
            return set(), set(futures)

        monkeypatch.setattr(mcp_client.concurrent.futures, "wait", late_wait)
        data = manager.call_tool_data("mcp__fx__plain", {}, 0.05)
        assert data.cause is None
        assert data.value == "hello world"

    def test_restricted_commands_get_no_bridge(self, manager, tmp_path):
        report = ReportCollector()
        out = dispatch(
            "run_python",
            {"code": "import swival_tools"},
            str(tmp_path),
            mcp_manager=manager,
            report=report,
        )
        assert out.startswith("error: run_python tool is not available")
        assert report.events == []


class TestRecords:
    def test_each_call_is_recorded(self, manager, tmp_path):
        _, report = _run(
            manager,
            tmp_path,
            """
            import swival_tools as st
            st.call('mcp__fx__plain')
            for name in ('mcp__fx__fail', 'mcp__fx__hidden'):
                try:
                    st.call(name)
                except st.ToolError:
                    pass
            """,
            tool_call_id="call_0",
        )
        event = _event(report)
        assert event["tool_call_id"] == "call_0"
        assert [c["id"] for c in event["calls"]] == [
            f"{event['execution_id']}-{n}" for n in (1, 2, 3)
        ]
        assert [(c["tool"], c["outcome"]) for c in event["calls"]] == [
            ("mcp__fx__plain", "ok"),
            ("mcp__fx__fail", "error"),
            ("mcp__fx__hidden", "refused"),
        ]

    def test_repeated_provider_call_ids_stay_distinct(self, manager, tmp_path):
        report = ReportCollector()
        for _ in range(2):
            _run(
                manager,
                tmp_path,
                "import swival_tools as st\nst.call('mcp__fx__plain')\n",
                tool_call_id="call_0",
                report=report,
            )
        first, second = [e for e in report.events if e["type"] == "python_bridge"]
        assert first["execution_id"] != second["execution_id"]
        assert first["calls"][0]["id"] != second["calls"][0]["id"]

    def test_records_and_names_are_bounded(self, manager, tmp_path, monkeypatch):
        # Room for two of these requests before the data total stops the run.
        monkeypatch.setattr(python_bridge, "MAX_TOTAL_BYTES", 10_000)
        monkeypatch.setattr(python_bridge, "MAX_RECORDS", 2)
        calls = _spy_calls(manager, monkeypatch)
        _, report = _run(
            manager,
            tmp_path,
            """
            import swival_tools as st
            for _ in range(6):
                try:
                    st.call('x' * 4096)
                except (st.ToolError, st.BridgeError):
                    pass
            """,
        )
        assert calls == []
        event = _event(report)
        assert _outcomes(event) == ["refused", "refused"]
        assert event["calls_not_listed"] == 1
        assert event["child_failures"] == 2
        limit = python_bridge.MAX_RECORDED_NAME_BYTES + len("...[truncated]")
        assert all(len(c["tool"].encode()) <= limit for c in event["calls"])

    def test_a_long_tool_name_is_clipped_once(self):
        long_name = "mcp__srv__" + "t" * 100_000
        bridge = _bridge(_Answers(), [long_name])
        for _ in range(3):
            bridge._handle(_call_frame(long_name))
        shared = bridge.stats.records[0]["tool"]
        assert all(r["tool"] is shared for r in bridge.stats.records)
        assert shared.endswith("...[truncated]")
        assert bridge.external_tools() == [shared]

    def test_a_lone_surrogate_in_a_name_is_recorded(self, manager, tmp_path):
        out, report = _run(
            manager,
            tmp_path,
            r"""
            import swival_tools as st
            try:
                st.call('bad\ud800name')
            except st.ToolError:
                print('refused')
            """,
        )
        assert _body(out) == "refused"
        event = _event(report)
        assert event["child_failures"] == 1
        assert [(c["tool"], c["outcome"]) for c in event["calls"]] == [
            ("bad?name", "refused")
        ]

    def test_shutdown_cuts_off_a_call(self, manager, tmp_path, monkeypatch):
        monkeypatch.setattr(
            manager,
            "call_tool_data",
            lambda *a, **k: McpData(error=_host_error("closing"), cause="shutdown"),
        )
        out, report = _run(
            manager,
            tmp_path,
            """
            import swival_tools as st
            try:
                st.call('mcp__fx__plain')
            except st.BridgeError as e:
                print(e)
            """,
        )
        event = _event(report)
        assert len(event["calls"]) == event["child_calls"] == 1
        assert event["uncertain_calls"] == 1
        assert (
            "Swival is shutting down its MCP connections; the call to "
            "mcp__fx__plain was cut off and may still complete on the server"
        ) in out

    def test_a_call_running_past_finish_is_reported_once(
        self, manager, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(python_bridge, "FINISH_WAIT", 0.3)
        release = threading.Event()
        bridges = []
        real_bridge = python_bridge.PythonBridge

        def keep(*args, **kwargs):
            bridges.append(real_bridge(*args, **kwargs))
            return bridges[-1]

        def stuck(*args, **kwargs):
            release.wait(5)
            return McpData("late")

        monkeypatch.setattr(python_bridge, "PythonBridge", keep)
        monkeypatch.setattr(manager, "call_tool_data", stuck)
        out, report = _run(
            manager,
            tmp_path,
            "import swival_tools as st\nst.call('mcp__fx__plain')\n",
            timeout=1,
        )
        event = _event(report)
        assert event["child_calls"] == event["uncertain_calls"] == 1
        assert event["unfinished_call"] is True
        assert event["calls"][0]["outcome"] == "unfinished"
        assert "still running when the program ended" in out
        release.set()
        time.sleep(0.3)
        assert len(bridges[0].stats.records) == 1

    def test_records_travel_with_the_tool_message(self, manager, tmp_path):
        code = "import swival_tools as st\nprint(st.call('mcp__fx__plain'))\n"
        tool_call = types.SimpleNamespace(
            id="call_7",
            function=types.SimpleNamespace(
                name="run_python", arguments=json.dumps({"code": code})
            ),
        )
        tool_msg, meta = handle_tool_call(
            tool_call,
            str(tmp_path),
            None,
            False,
            commands_unrestricted=True,
            mcp_manager=manager,
        )
        record = tool_msg["_swival_child_calls"]
        assert record["tool_call_id"] == "call_7"
        assert [c["tool"] for c in record["calls"]] == ["mcp__fx__plain"]
        assert meta["succeeded"]

    def test_an_undelivered_answer_is_marked(self):
        bridge = _bridge(_Answers())
        req_w, _ = bridge.open({})
        frame = _call_frame()
        os.write(req_w, len(frame).to_bytes(4, "big") + frame)
        # With no child holding the other end, the reply cannot be delivered.
        bridge.start()
        bridge._thread.join(5)
        (record,) = bridge.stats.to_dict()["calls"]
        assert record["outcome"] == "ok"
        assert record["delivered"] is False
        bridge.finish()

    @pytest.mark.parametrize("failure", ["closed", "partial"])
    def test_undelivered_bytes_are_not_counted(self, failure):
        bridge = _bridge()
        bridge.open({})
        if failure == "closed":
            bridge._close("resp_r", "req_w")
            assert bridge._send(b"x" * 100) is False
        else:
            sender = threading.Thread(target=bridge._send, args=(b"x" * (1 << 20),))
            sender.start()
            time.sleep(0.2)
            bridge._finishing.set()
            sender.join(5)
        assert bridge.stats.bytes_out == 0
        bridge.abandon()


class TestEncoding:
    def test_encoding_is_bounded(self, monkeypatch):
        class Unreachable:
            def __str__(self):
                raise AssertionError("encoding went past the limit")

        real_encode = python_bridge._encode
        full = []
        monkeypatch.setattr(
            python_bridge, "_encode", lambda r: full.append(r) or real_encode(r)
        )
        encode = python_bridge._encode_bounded

        assert (
            encode({"v": ["x" * (1 << 20), "y" * (1 << 20), Unreachable()]}, 1 << 20)
            is None
        )
        assert encode({"v": [0] * (2 << 20) + [Unreachable()]}, 1 << 20) is None
        # 200 characters, but 800 bytes, before the sentinel.
        assert encode({"v": ["\U0001f600" * 10] * 20 + [Unreachable()]}, 500) is None
        assert encode({"v": [10**4000] * 10}, 1000) is None
        # Might not fit, so it is encoded piece by piece, and it does fit.
        floats = {"v": [0.1] * 150}
        assert encode(floats, 1000) == json.dumps(floats).encode()
        # None of the above went through the one-go encoder.
        assert full == []

        small = {"a": "é"}
        assert encode(small, 100) == real_encode(small)
        assert full == [small]
        assert json.loads(encode({"v": "\ud800"}, 100)) == {"v": "\ud800"}


class TestSetup:
    def test_description_names_the_callable_tools(self, manager):
        tools = [json.loads(json.dumps(PYTHON_TOOL))]
        python_bridge.annotate_run_python(tools, manager)
        description = tools[0]["function"]["description"]
        assert description.startswith(PYTHON_TOOL["function"]["description"])
        assert "mcp__fx__rows" in description
        assert "mcp__fx__hidden" not in description
        assert (
            tools[0]["function"]["parameters"] == PYTHON_TOOL["function"]["parameters"]
        )

    @pytest.mark.parametrize("jailed", [False, True])
    def test_no_description_without_a_bridge(self, manager, jailed):
        tools = [PYTHON_TOOL]
        if jailed:
            python_bridge.annotate_run_python(tools, manager, ["nono", "run", "--"])
        else:
            python_bridge.annotate_run_python(tools, McpManager({}))
        assert tools == [PYTHON_TOOL]

    def test_long_catalogs_are_cut(self):
        names = [f"mcp__srv__tool_{i:04d}" for i in range(500)]
        text = python_bridge.describe(names)
        assert len(text) < python_bridge._DESCRIBED_NAMES_MAX_CHARS + 800
        assert "more in swival_tools.ALL_TOOLS" in text

    def test_no_module_without_python_tools(self, tmp_path):
        out = dispatch(
            "run_python",
            {"code": "import swival_tools"},
            str(tmp_path),
            mcp_manager=McpManager({}),
            commands_unrestricted=True,
        )
        assert "ModuleNotFoundError" in out

    def test_pruned_tools_leave_the_catalog(self, manager):
        route = manager._tool_map.pop("mcp__fx__plain")
        try:
            assert "mcp__fx__plain" not in manager.python_tools()
            assert "mcp__fx__rows" in manager.python_tools()
        finally:
            manager._tool_map["mcp__fx__plain"] = route

    def test_config_validation(self):
        def validate(value):
            _validate_mcp_server_configs(
                {"s": {"command": "x", "python_tools": value}}, "test"
            )

        validate(["a", "b"])
        with pytest.raises(ConfigError, match=r"python_tools\[1\]: expected string"):
            validate(["a", 1])
        with pytest.raises(ConfigError, match="python_tools: expected list"):
            validate("a")


class TestRobustness:
    def test_threads_share_the_bridge(self, manager, tmp_path):
        out, _ = _run(
            manager,
            tmp_path,
            """
            import threading, swival_tools as st
            got = {}
            def work(n):
                for _ in range(5):
                    answer = st.call('mcp__fx__plain', {'who': str(n)})
                    got.setdefault(n, set()).add(answer)
            threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            print(all(got[n] == {'hello ' + str(n)} for n in range(8)))
            """,
        )
        assert _body(out) == "True"

    def test_a_deeply_nested_request_gets_a_reply(self, manager, tmp_path):
        out, _ = _run(
            manager,
            tmp_path,
            """
            import json, os, struct, swival_tools as st
            body = b'[' * 200000 + b']' * 200000
            os.write(st._WRITE_FD, struct.pack('>I', len(body)) + body)
            (n,) = struct.unpack('>I', st._read_exact(4))
            print(json.loads(st._read_exact(n))['error'])
            print(st.call('mcp__fx__plain'))
            """,
        )
        assert _body(out).splitlines() == ["the request is not JSON", "hello world"]

    def test_a_failed_second_pipe_leaks_nothing(self, monkeypatch):
        real_pipe = os.pipe
        made = []

        def pipe():
            if made:
                raise OSError(24, "Too many open files")
            made.extend(real_pipe())
            return tuple(made)

        bridge = _bridge()
        monkeypatch.setattr(os, "pipe", pipe)
        with pytest.raises(OSError):
            bridge.open({})
        bridge.abandon()
        for fd in made:
            with pytest.raises(OSError):
                os.fstat(fd)

    def test_nothing_is_sent_once_finish_has_begun(self):
        sent = []

        class Manager:
            def call_tool_data(self, *args, **kwargs):
                sent.append(args[0])

        bridge = _bridge(Manager())
        bridge._finishing.set()
        reply = json.loads(bridge._handle(_call_frame()))
        assert reply["kind"] == "stop"
        assert sent == []
        assert bridge.stats.calls == 0
        assert _outcomes(bridge.stats.to_dict()) == ["stopped"]

    def test_finish_waits_for_a_call_being_recorded(self, monkeypatch):
        monkeypatch.setattr(python_bridge, "FINISH_WAIT", 0.1)
        bridge = _bridge(_Answers())
        concluding, release = threading.Event(), threading.Event()
        real_conclude = bridge._conclude

        def slow_conclude(*args):
            concluding.set()
            release.wait(5)
            return real_conclude(*args)

        bridge._conclude = slow_conclude
        worker = threading.Thread(target=bridge._handle, args=(_call_frame(),))
        bridge._thread = worker
        worker.start()
        assert concluding.wait(5)
        finisher = threading.Thread(target=bridge.finish)
        finisher.start()
        finisher.join(0.5)
        assert finisher.is_alive()
        release.set()
        finisher.join(5)
        worker.join(5)
        assert _outcomes(bridge.stats.to_dict()) == ["ok"]
        assert not bridge.stats.unfinished
        assert bridge.stats.uncertain_calls == 0

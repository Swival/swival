"""Tests for MCP (Model Context Protocol) server support."""

import json
import textwrap
from unittest.mock import MagicMock

import pytest

from swival.mcp_client import (
    McpManager,
    McpShutdownError,
    _convert_schema,
    _normalize_result,
    _sanitize_tool_name,
    _mcp_tool_to_openai,
    validate_server_name,
)
from swival.report import ConfigError


# ---------------------------------------------------------------------------
# Schema conversion tests
# ---------------------------------------------------------------------------


class TestConvertSchema:
    def test_minimal_schema(self):
        result = _convert_schema({})
        assert result == {"type": "object", "properties": {}}

    def test_preserves_existing_fields(self):
        schema = {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
            },
            "required": ["path"],
        }
        result = _convert_schema(schema)
        assert result["required"] == ["path"]
        assert result["properties"]["path"]["type"] == "string"

    def test_strips_dollar_schema(self):
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "$id": "my-schema",
            "type": "object",
            "properties": {"x": {"type": "integer"}},
        }
        result = _convert_schema(schema)
        assert "$schema" not in result
        assert "$id" not in result
        assert result["properties"]["x"]["type"] == "integer"

    def test_adds_missing_type(self):
        schema = {"properties": {"a": {"type": "string"}}}
        result = _convert_schema(schema)
        assert result["type"] == "object"

    def test_adds_missing_properties(self):
        schema = {"type": "object"}
        result = _convert_schema(schema)
        assert result["properties"] == {}

    def test_preserves_additional_properties(self):
        schema = {
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        }
        result = _convert_schema(schema)
        assert result["additionalProperties"] is True

    def test_preserves_oneof(self):
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "integer"},
                    ]
                }
            },
        }
        result = _convert_schema(schema)
        assert len(result["properties"]["value"]["oneOf"]) == 2

    def test_preserves_nested_objects(self):
        schema = {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {
                        "timeout": {"type": "integer", "default": 30},
                    },
                }
            },
        }
        result = _convert_schema(schema)
        nested = result["properties"]["config"]["properties"]
        assert nested["timeout"]["default"] == 30

    def test_does_not_mutate_input(self):
        original = {
            "type": "object",
            "$schema": "http://json-schema.org/draft-07/schema#",
            "properties": {"x": {"type": "string"}},
        }
        import copy

        before = copy.deepcopy(original)
        _convert_schema(original)
        assert original == before

    def test_preserves_enum(self):
        schema = {
            "type": "object",
            "properties": {
                "level": {"type": "string", "enum": ["low", "medium", "high"]},
            },
        }
        result = _convert_schema(schema)
        assert result["properties"]["level"]["enum"] == ["low", "medium", "high"]

    def test_preserves_empty_required(self):
        schema = {"type": "object", "properties": {}, "required": []}
        result = _convert_schema(schema)
        assert result["required"] == []


# ---------------------------------------------------------------------------
# Result normalization tests
# ---------------------------------------------------------------------------


class _MockBlock:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _MockResult:
    def __init__(self, content, isError=False):
        self.content = content
        self.isError = isError


class TestNormalizeResult:
    def test_single_text(self):
        result = _MockResult([_MockBlock(type="text", text="hello world")])
        assert _normalize_result(result) == ("hello world", False)

    def test_multiple_text_blocks(self):
        result = _MockResult(
            [
                _MockBlock(type="text", text="line 1"),
                _MockBlock(type="text", text="line 2"),
            ]
        )
        assert _normalize_result(result) == ("line 1\nline 2", False)

    def test_image_block(self):
        result = _MockResult(
            [
                _MockBlock(type="image", mimeType="image/png", data="abc123"),
            ]
        )
        assert _normalize_result(result) == ("[image: image/png, 6 bytes]", False)

    def test_audio_block(self):
        result = _MockResult(
            [
                _MockBlock(type="audio", mimeType="audio/mp3", data="xyz"),
            ]
        )
        assert _normalize_result(result) == ("[audio: audio/mp3, 3 bytes]", False)

    def test_resource_with_text(self):
        resource = _MockBlock(text="resource content", uri="file:///tmp/f.txt")
        result = _MockResult([_MockBlock(type="resource", resource=resource)])
        assert _normalize_result(result) == ("resource content", False)

    def test_resource_without_text(self):
        resource = _MockBlock(uri="file:///tmp/f.txt")
        result = _MockResult([_MockBlock(type="resource", resource=resource)])
        assert _normalize_result(result) == ("[resource: file:///tmp/f.txt]", False)

    def test_is_error(self):
        result = _MockResult(
            [_MockBlock(type="text", text="something went wrong")],
            isError=True,
        )
        assert _normalize_result(result) == ("error: something went wrong", True)

    def test_is_error_empty(self):
        result = _MockResult([], isError=True)
        assert _normalize_result(result) == ("error: MCP tool returned an error", True)

    def test_empty_result(self):
        result = _MockResult([])
        assert _normalize_result(result) == ("(empty result)", False)

    def test_unknown_block_type(self):
        result = _MockResult([_MockBlock(type="video")])
        assert _normalize_result(result) == ("[video: unsupported content type]", False)

    def test_mixed_content(self):
        result = _MockResult(
            [
                _MockBlock(type="text", text="Result:"),
                _MockBlock(type="image", mimeType="image/jpeg", data="data" * 100),
            ]
        )
        text, is_err = _normalize_result(result)
        assert not is_err
        assert text.startswith("Result:\n")
        assert "[image: image/jpeg," in text


# ---------------------------------------------------------------------------
# Tool name sanitization tests
# ---------------------------------------------------------------------------


class TestSanitizeToolName:
    def test_simple_name(self):
        assert _sanitize_tool_name("read_file") == "read_file"

    def test_collapses_double_underscores(self):
        assert _sanitize_tool_name("my__tool") == "my_tool"

    def test_replaces_special_chars(self):
        assert _sanitize_tool_name("get.data!now") == "get_data_now"

    def test_strips_leading_trailing(self):
        assert _sanitize_tool_name("-_name_-") == "name"

    def test_preserves_hyphens(self):
        assert _sanitize_tool_name("my-tool") == "my-tool"


# ---------------------------------------------------------------------------
# Server name validation tests
# ---------------------------------------------------------------------------


class TestValidateServerName:
    def test_valid_name(self):
        validate_server_name("my-server")
        validate_server_name("server123")
        validate_server_name("a")

    def test_rejects_double_underscore(self):
        with pytest.raises(ConfigError, match="double underscores"):
            validate_server_name("my__server")

    def test_rejects_special_chars(self):
        with pytest.raises(ConfigError, match="invalid"):
            validate_server_name("my.server")

    def test_rejects_spaces(self):
        with pytest.raises(ConfigError, match="invalid"):
            validate_server_name("my server")

    def test_rejects_empty(self):
        with pytest.raises(ConfigError, match="invalid"):
            validate_server_name("")


# ---------------------------------------------------------------------------
# MCP tool to OpenAI conversion tests
# ---------------------------------------------------------------------------


class TestMcpToolToOpenai:
    def test_basic_conversion(self):
        tool = _MockBlock(
            name="read_file",
            description="Read a file",
            inputSchema={"type": "object", "properties": {"path": {"type": "string"}}},
        )
        result, original_name = _mcp_tool_to_openai("filesystem", tool)
        assert result["type"] == "function"
        assert result["function"]["name"] == "mcp__filesystem__read_file"
        assert result["function"]["description"] == "Read a file"
        assert (
            result["function"]["parameters"]["properties"]["path"]["type"] == "string"
        )
        assert original_name == "read_file"

    def test_no_description(self):
        tool = _MockBlock(name="ping", description=None, inputSchema={})
        result, original_name = _mcp_tool_to_openai("server1", tool)
        assert "MCP tool from server1" in result["function"]["description"]
        assert original_name == "ping"

    def test_no_input_schema(self):
        tool = _MockBlock(name="ping", description="Ping", inputSchema=None)
        result, original_name = _mcp_tool_to_openai("server1", tool)
        assert result["function"]["parameters"]["type"] == "object"
        assert result["function"]["parameters"]["properties"] == {}
        assert original_name == "ping"

    def test_returns_original_name_without_exposing_private_field(self):
        tool = _MockBlock(
            name="get.data",
            description="Get data",
            inputSchema={},
        )
        result, original_name = _mcp_tool_to_openai("srv", tool)
        assert result["function"]["name"] == "mcp__srv__get_data"
        assert original_name == "get.data"
        assert "_mcp_original_name" not in result["function"]


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestMcpConfig:
    def test_load_mcp_json(self, tmp_path):
        from swival.config import load_mcp_json

        mcp_file = tmp_path / "mcp.json"
        mcp_file.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "fs": {"command": "npx", "args": ["-y", "server-fs"]},
                    }
                }
            )
        )
        result = load_mcp_json(mcp_file)
        assert "fs" in result
        assert result["fs"]["command"] == "npx"

    def test_load_mcp_json_missing_file(self, tmp_path):
        from swival.config import load_mcp_json

        with pytest.raises(ConfigError, match="cannot read"):
            load_mcp_json(tmp_path / "nonexistent.json")

    def test_load_mcp_json_invalid_json(self, tmp_path):
        from swival.config import load_mcp_json

        mcp_file = tmp_path / "mcp.json"
        mcp_file.write_text("{invalid json")
        with pytest.raises(ConfigError, match="invalid JSON"):
            load_mcp_json(mcp_file)

    def test_load_mcp_json_invalid_structure(self, tmp_path):
        from swival.config import load_mcp_json

        mcp_file = tmp_path / "mcp.json"
        mcp_file.write_text(json.dumps({"mcpServers": "not a dict"}))
        with pytest.raises(ConfigError, match="must be a JSON object"):
            load_mcp_json(mcp_file)

    def test_merge_mcp_configs_toml_wins(self):
        from swival.config import merge_mcp_configs

        toml = {"server": {"command": "toml-cmd"}}
        json_ = {"server": {"command": "json-cmd"}, "other": {"command": "other"}}
        merged = merge_mcp_configs(toml, json_)
        assert merged["server"]["command"] == "toml-cmd"
        assert merged["other"]["command"] == "other"

    def test_merge_mcp_configs_both_none(self):
        from swival.config import merge_mcp_configs

        assert merge_mcp_configs(None, None) == {}

    def test_mcp_servers_in_toml(self, tmp_path):
        """Test that mcp_servers section is extracted from TOML config."""
        from swival.config import _load_single

        config_file = tmp_path / "swival.toml"
        config_file.write_text(
            textwrap.dedent("""\
            [mcp_servers.myserver]
            command = "my-cmd"
            args = ["--flag"]
        """)
        )
        result = _load_single(config_file, str(config_file))
        assert "mcp_servers" in result
        assert result["mcp_servers"]["myserver"]["command"] == "my-cmd"

    def test_mcp_servers_invalid_name_in_toml(self, tmp_path):
        from swival.config import _load_single

        config_file = tmp_path / "swival.toml"
        config_file.write_text(
            textwrap.dedent("""\
            [mcp_servers."bad name"]
            command = "my-cmd"
        """)
        )
        with pytest.raises(ConfigError, match="invalid"):
            _load_single(config_file, str(config_file))

    def test_mcp_json_missing_command_and_url(self, tmp_path):
        from swival.config import load_mcp_json

        mcp_file = tmp_path / "mcp.json"
        mcp_file.write_text(
            json.dumps({"mcpServers": {"bad": {"env": {"KEY": "val"}}}})
        )
        with pytest.raises(ConfigError, match="must have"):
            load_mcp_json(mcp_file)

    def test_mcp_json_both_command_and_url(self, tmp_path):
        from swival.config import load_mcp_json

        mcp_file = tmp_path / "mcp.json"
        mcp_file.write_text(
            json.dumps({"mcpServers": {"bad": {"command": "cmd", "url": "http://x"}}})
        )
        with pytest.raises(ConfigError, match="cannot have both"):
            load_mcp_json(mcp_file)

    def test_mcp_server_command_wrong_type(self, tmp_path):
        from swival.config import load_mcp_json

        mcp_file = tmp_path / "mcp.json"
        mcp_file.write_text(json.dumps({"mcpServers": {"s": {"command": 42}}}))
        with pytest.raises(ConfigError, match="expected str, got int"):
            load_mcp_json(mcp_file)

    def test_mcp_server_args_wrong_type(self, tmp_path):
        from swival.config import load_mcp_json

        mcp_file = tmp_path / "mcp.json"
        mcp_file.write_text(
            json.dumps({"mcpServers": {"s": {"command": "cmd", "args": "bad"}}})
        )
        with pytest.raises(ConfigError, match="expected list, got str"):
            load_mcp_json(mcp_file)

    def test_mcp_server_args_element_wrong_type(self, tmp_path):
        from swival.config import load_mcp_json

        mcp_file = tmp_path / "mcp.json"
        mcp_file.write_text(
            json.dumps({"mcpServers": {"s": {"command": "cmd", "args": [1]}}})
        )
        with pytest.raises(ConfigError, match="args\\[0\\].*expected string"):
            load_mcp_json(mcp_file)

    def test_mcp_server_env_wrong_type(self, tmp_path):
        from swival.config import load_mcp_json

        mcp_file = tmp_path / "mcp.json"
        mcp_file.write_text(
            json.dumps({"mcpServers": {"s": {"command": "cmd", "env": "bad"}}})
        )
        with pytest.raises(ConfigError, match="expected dict, got str"):
            load_mcp_json(mcp_file)

    def test_mcp_server_env_value_wrong_type(self, tmp_path):
        from swival.config import load_mcp_json

        mcp_file = tmp_path / "mcp.json"
        mcp_file.write_text(
            json.dumps({"mcpServers": {"s": {"command": "cmd", "env": {"K": 1}}}})
        )
        with pytest.raises(ConfigError, match="env\\.K.*expected string"):
            load_mcp_json(mcp_file)

    def test_mcp_server_headers_value_wrong_type(self, tmp_path):
        from swival.config import load_mcp_json

        mcp_file = tmp_path / "mcp.json"
        mcp_file.write_text(
            json.dumps({"mcpServers": {"s": {"url": "http://x", "headers": {"H": 1}}}})
        )
        with pytest.raises(ConfigError, match="headers\\.H.*expected string"):
            load_mcp_json(mcp_file)

    def test_mcp_server_url_wrong_type(self, tmp_path):
        from swival.config import load_mcp_json

        mcp_file = tmp_path / "mcp.json"
        mcp_file.write_text(json.dumps({"mcpServers": {"s": {"url": 123}}}))
        with pytest.raises(ConfigError, match="expected str, got int"):
            load_mcp_json(mcp_file)

    def test_config_to_session_kwargs_passes_mcp_servers(self):
        """mcp_servers should pass through to Session kwargs."""
        from swival.config import config_to_session_kwargs

        config = {
            "provider": "lmstudio",
            "mcp_servers": {"fs": {"command": "cmd"}},
        }
        kwargs = config_to_session_kwargs(config)
        assert "mcp_servers" in kwargs
        assert kwargs["mcp_servers"]["fs"]["command"] == "cmd"

    def test_config_to_session_kwargs_drops_no_mcp(self):
        """no_mcp is a CLI concern, not a Session concern."""
        from swival.config import config_to_session_kwargs

        config = {"no_mcp": True}
        kwargs = config_to_session_kwargs(config)
        assert "no_mcp" not in kwargs


# ---------------------------------------------------------------------------
# Dispatch integration tests
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_mcp_prefix_routes_to_manager(self):
        from swival.tools import dispatch

        manager = MagicMock()
        manager.call_tool.return_value = ("tool result", False)

        result = dispatch(
            "mcp__server__tool",
            {"arg": "val"},
            "/base",
            mcp_manager=manager,
        )
        assert "[UNTRUSTED EXTERNAL CONTENT]" in result
        assert "tool result" in result
        manager.call_tool.assert_called_once_with("mcp__server__tool", {"arg": "val"})

    def test_mcp_prefix_no_manager(self):
        from swival.tools import dispatch

        result = dispatch(
            "mcp__server__tool",
            {"arg": "val"},
            "/base",
        )
        assert result.startswith("error:")

    def test_non_mcp_unaffected(self):
        """Non-MCP tool dispatch should work unchanged."""
        from swival.tools import dispatch
        from swival.thinking import ThinkingState

        ts = ThinkingState(verbose=False)
        result = dispatch(
            "think",
            {
                "thought": "test",
                "nextThoughtNeeded": False,
                "thoughtNumber": 1,
                "totalThoughts": 1,
            },
            "/base",
            thinking_state=ts,
        )
        # think returns JSON
        assert '"thought_number"' in result


# ---------------------------------------------------------------------------
# MCP output guard tests
# ---------------------------------------------------------------------------


class TestMcpOutputGuard:
    """Tests for _guard_mcp_output and the dispatch-level size guard."""

    def test_small_result_passthrough(self, tmp_path):
        from swival.tools import _guard_mcp_output

        result = _guard_mcp_output("small output", str(tmp_path), "mcp__s__t")
        assert result == "small output"

    def test_at_limit_stays_inline(self, tmp_path):
        from swival.tools import _guard_mcp_output, MCP_INLINE_LIMIT

        # Exactly MCP_INLINE_LIMIT bytes should stay inline
        payload = "x" * MCP_INLINE_LIMIT
        assert len(payload.encode("utf-8")) == MCP_INLINE_LIMIT
        result = _guard_mcp_output(payload, str(tmp_path), "mcp__s__t")
        assert result == payload

    def test_over_limit_saves_to_file(self, tmp_path):
        from swival.tools import _guard_mcp_output, MCP_INLINE_LIMIT

        payload = "x" * (MCP_INLINE_LIMIT + 1)
        result = _guard_mcp_output(payload, str(tmp_path), "mcp__s__t")
        assert "read_file" in result
        assert "Tool output from mcp__s__t" in result
        # File should exist in .swival/
        swival_dir = tmp_path / ".swival"
        assert swival_dir.exists()
        files = list(swival_dir.glob("cmd_output_*.txt"))
        assert len(files) == 1
        file_content = files[0].read_text()
        # File should contain the untrusted header followed by the payload
        assert file_content.startswith("[UNTRUSTED EXTERNAL CONTENT]")
        assert payload in file_content

    def test_pointer_message_wording(self, tmp_path):
        from swival.tools import _guard_mcp_output, MCP_INLINE_LIMIT

        payload = "y" * (MCP_INLINE_LIMIT * 5)
        result = _guard_mcp_output(payload, str(tmp_path), "mcp__server__tool")
        assert "Tool output from mcp__server__tool" in result
        assert "Command output" not in result
        assert "Full output saved to" in result

    def test_over_max_file_truncates(self, tmp_path):
        from swival.tools import _guard_mcp_output, MCP_FILE_LIMIT

        payload = "z" * (MCP_FILE_LIMIT + 1000)
        result = _guard_mcp_output(payload, str(tmp_path), "mcp__s__t")
        assert "possibly truncated" in result.lower()
        # File should be capped at MCP_FILE_LIMIT + small untrusted header
        swival_dir = tmp_path / ".swival"
        files = list(swival_dir.glob("cmd_output_*.txt"))
        assert len(files) == 1
        file_content = files[0].read_text()
        # The payload portion must be truncated to MCP_FILE_LIMIT;
        # the file also contains the untrusted-content header (~160 bytes).
        header_end = file_content.index("\n\n") + 2
        payload_bytes = len(file_content[header_end:].encode("utf-8"))
        assert payload_bytes <= MCP_FILE_LIMIT

    def test_error_small_passthrough(self, tmp_path):
        """Small error results pass through unchanged."""
        from swival.tools import dispatch

        manager = MagicMock()
        manager.call_tool.return_value = ("error: something broke", True)

        result = dispatch("mcp__s__t", {}, str(tmp_path), mcp_manager=manager)
        assert result == "error: something broke"

    def test_error_large_truncated_inline(self, tmp_path):
        """Giant error payloads are truncated inline, not saved to file."""
        from swival.tools import dispatch, MCP_INLINE_LIMIT

        giant_error = "error: " + "x" * (MCP_INLINE_LIMIT * 2)
        manager = MagicMock()
        manager.call_tool.return_value = (giant_error, True)

        result = dispatch("mcp__s__t", {}, str(tmp_path), mcp_manager=manager)
        assert result.endswith("[error output truncated]")
        result_bytes = result.encode("utf-8")
        # The truncated content (before suffix) should be at most MCP_INLINE_LIMIT
        assert len(result_bytes) <= MCP_INLINE_LIMIT + len(
            "\n[error output truncated]".encode("utf-8")
        )
        # No file should be created
        swival_dir = tmp_path / ".swival"
        if swival_dir.exists():
            assert list(swival_dir.glob("cmd_output_*.txt")) == []

    def test_disk_failure_fallback(self, tmp_path, monkeypatch):
        """When .swival/ can't be created, falls back to inline truncation."""
        from swival.tools import _guard_mcp_output, MCP_INLINE_LIMIT
        from pathlib import Path

        payload = "a" * (MCP_INLINE_LIMIT + 5000)
        # Make mkdir raise
        original_mkdir = Path.mkdir

        def _fail_mkdir(self, *args, **kwargs):
            if ".swival" in str(self):
                raise OSError("disk full")
            return original_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", _fail_mkdir)
        result = _guard_mcp_output(payload, str(tmp_path), "mcp__s__t")
        assert "failed to create .swival/" in result
        # Result should be truncated to roughly MCP_INLINE_LIMIT
        assert len(result.encode("utf-8")) < MCP_INLINE_LIMIT + 200

    def test_non_ascii_boundary(self, tmp_path):
        """Byte-based threshold handles multibyte UTF-8 correctly."""
        from swival.tools import _guard_mcp_output, MCP_INLINE_LIMIT

        # Each char is 3 bytes in UTF-8
        char = "\u00e9"  # é — 2 bytes in UTF-8
        assert len(char.encode("utf-8")) == 2

        # Create a payload that's under the limit in chars but over in bytes
        num_chars = MCP_INLINE_LIMIT  # each is 2 bytes, so 2x the byte limit
        payload = char * num_chars
        assert len(payload.encode("utf-8")) > MCP_INLINE_LIMIT

        result = _guard_mcp_output(payload, str(tmp_path), "mcp__s__t")
        # Should be saved to file since byte count exceeds limit
        assert "read_file" in result


# ---------------------------------------------------------------------------
# McpManager lifecycle tests (mocked)
# ---------------------------------------------------------------------------


class TestMcpManagerLifecycle:
    def test_close_idempotent(self):
        mgr = McpManager({}, verbose=False)
        mgr._closed = True
        mgr.close()  # should not raise

    def test_call_tool_after_close(self):
        mgr = McpManager({}, verbose=False)
        mgr._closed = True
        with pytest.raises(McpShutdownError):
            mgr.call_tool("mcp__s__t", {})

    def test_call_tool_during_closing(self):
        mgr = McpManager({}, verbose=False)
        mgr._closing = True
        with pytest.raises(McpShutdownError):
            mgr.call_tool("mcp__s__t", {})

    def test_call_tool_unknown_name(self):
        mgr = McpManager({}, verbose=False)
        mgr._tool_map = {}
        result, is_err = mgr.call_tool("mcp__unknown__tool", {})
        assert is_err
        assert "unknown" in result

    def test_call_tool_degraded_server(self):
        mgr = McpManager({}, verbose=False)
        mgr._tool_map = {"mcp__s__t": ("s", "t")}
        mgr._degraded = {"s"}
        result, is_err = mgr.call_tool("mcp__s__t", {})
        assert is_err
        assert "unavailable" in result

    def test_call_tool_success_returns_tuple(self):
        """Successful call_tool() returns (text, False) through _normalize_result."""
        import asyncio
        import threading

        mgr = McpManager({}, verbose=False)
        mgr._tool_map = {"mcp__s__t": ("s", "t")}

        # Mock a session whose call_tool returns a coroutine
        mock_result = _MockResult(
            [_MockBlock(type="text", text="success output")],
            isError=False,
        )

        async def _fake_call_tool(name, args):
            return mock_result

        mock_session = MagicMock()
        mock_session.call_tool = _fake_call_tool
        mgr._sessions = {"s": mock_session}

        # Need a running loop for _run_sync
        mgr._loop = asyncio.new_event_loop()
        loop_ready = threading.Event()
        mgr._loop.call_soon(lambda: loop_ready.set())
        mgr._thread = threading.Thread(target=mgr._loop.run_forever, daemon=True)
        mgr._thread.start()
        loop_ready.wait(timeout=5)

        try:
            result, is_err = mgr.call_tool("mcp__s__t", {"key": "val"})
            assert not is_err
            assert result == "success output"
        finally:
            mgr.close()

    def test_start_after_close_raises(self):
        mgr = McpManager({}, verbose=False)
        mgr._closed = True
        with pytest.raises(McpShutdownError):
            mgr.start()

    def test_start_creates_running_loop(self):
        """start() with no servers should still create a running loop."""
        mgr = McpManager({}, verbose=False)
        mgr.start()
        try:
            assert mgr._loop is not None
            assert mgr._loop.is_running()
            assert mgr._thread is not None
            assert mgr._thread.is_alive()
        finally:
            mgr.close()

    def test_list_tools_empty(self):
        mgr = McpManager({}, verbose=False)
        assert mgr.list_tools() == []

    def test_get_tool_info_empty(self):
        mgr = McpManager({}, verbose=False)
        assert mgr.get_tool_info() == {}


# ---------------------------------------------------------------------------
# Collision detection tests
# ---------------------------------------------------------------------------


class TestCollisionDetection:
    def test_collision_skips_server_and_warns_when_verbose(self, capsys):
        mgr = McpManager({}, verbose=True)
        # Simulate two tools with the same namespaced name from one server
        mgr._tool_schemas = {
            "server1": [
                {
                    "type": "function",
                    "function": {
                        "name": "mcp__server1__tool",
                        "_mcp_original_name": "tool.v1",
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "mcp__server1__tool",
                        "_mcp_original_name": "tool.v2",
                    },
                },
            ],
        }
        mgr._tool_original_names = {
            "server1": {
                "mcp__server1__tool": "tool.v2",
            }
        }
        mgr._build_tool_map()
        # Colliding server's tools should be skipped
        assert mgr._tool_schemas["server1"] == []
        assert "mcp__server1__tool" not in mgr._tool_map
        # Warning printed to stderr
        captured = capsys.readouterr()
        assert "collision" in captured.err

    def test_collision_skips_server_and_stays_quiet_when_not_verbose(self, capsys):
        mgr = McpManager({}, verbose=False)
        mgr._tool_schemas = {
            "server1": [
                {
                    "type": "function",
                    "function": {
                        "name": "mcp__server1__tool",
                        "_mcp_original_name": "tool.v1",
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "mcp__server1__tool",
                        "_mcp_original_name": "tool.v2",
                    },
                },
            ],
        }
        mgr._tool_original_names = {
            "server1": {
                "mcp__server1__tool": "tool.v2",
            }
        }
        mgr._build_tool_map()
        assert mgr._tool_schemas["server1"] == []
        assert "mcp__server1__tool" not in mgr._tool_map
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_collision_does_not_affect_other_servers(self, capsys):
        mgr = McpManager({}, verbose=False)
        mgr._tool_schemas = {
            "good": [
                {
                    "type": "function",
                    "function": {
                        "name": "mcp__good__tool_a",
                        "_mcp_original_name": "tool_a",
                    },
                },
            ],
            "bad": [
                {
                    "type": "function",
                    "function": {
                        "name": "mcp__bad__tool",
                        "_mcp_original_name": "tool.v1",
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "mcp__bad__tool",
                        "_mcp_original_name": "tool.v2",
                    },
                },
            ],
        }
        mgr._tool_original_names = {
            "good": {
                "mcp__good__tool_a": "tool_a",
            },
            "bad": {
                "mcp__bad__tool": "tool.v2",
            },
        }
        mgr._build_tool_map()
        # Good server unaffected
        assert "mcp__good__tool_a" in mgr._tool_map
        assert mgr._tool_map["mcp__good__tool_a"] == ("good", "tool_a")
        # Bad server's tools skipped
        assert mgr._tool_schemas["bad"] == []


# ---------------------------------------------------------------------------
# Token budget tests
# ---------------------------------------------------------------------------


class TestTokenBudget:
    def test_no_context_length_returns_tools(self):
        from swival.agent import enforce_mcp_token_budget

        tools = [{"type": "function", "function": {"name": "test"}}]
        result = enforce_mcp_token_budget(tools, MagicMock(), None)
        assert result == tools

    def test_no_manager_returns_tools(self):
        from swival.agent import enforce_mcp_token_budget

        tools = [{"type": "function", "function": {"name": "test"}}]
        result = enforce_mcp_token_budget(tools, None, 100000)
        assert result == tools

    def test_under_threshold_returns_unchanged(self):
        from swival.agent import enforce_mcp_token_budget

        tools = [
            {"type": "function", "function": {"name": "read_file", "parameters": {}}}
        ]
        mgr = MagicMock()
        mgr.get_tool_info.return_value = {}
        result = enforce_mcp_token_budget(tools, mgr, 1000000)
        assert result == tools


# ---------------------------------------------------------------------------
# System prompt MCP section tests
# ---------------------------------------------------------------------------


class TestMcpSystemPrompt:
    def test_format_mcp_tool_info(self):
        from swival.agent import _format_mcp_tool_info

        info = {
            "filesystem": [
                ("mcp__filesystem__read_file", "Read a file"),
                ("mcp__filesystem__write_file", "Write a file"),
            ],
        }
        text = _format_mcp_tool_info(info)
        assert "## MCP Tools" in text
        assert "**filesystem**" in text
        assert "`mcp__filesystem__read_file`" in text
        assert "Read a file" in text

    def test_format_mcp_tool_info_empty(self):
        from swival.agent import _format_mcp_tool_info

        text = _format_mcp_tool_info({})
        assert "## MCP Tools" in text

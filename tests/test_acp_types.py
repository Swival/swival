"""Tests for swival.acp_types: constants, helpers, encode/decode."""

import json

import pytest

from swival.acp_types import (
    ERROR_INVALID_PARAMS,
    JSONRPC_VERSION,
    METHOD_INITIALIZE,
    PROTOCOL_VERSION,
    TOOL_KIND_EDIT,
    TOOL_KIND_EXECUTE,
    TOOL_KIND_OTHER,
    TOOL_KIND_READ,
    TOOL_KIND_SEARCH,
    TOOL_KIND_THINK,
    TOOL_STATUS_COMPLETED,
    TOOL_STATUS_FAILED,
    TOOL_STATUS_IN_PROGRESS,
    UPDATE_AGENT_MESSAGE_CHUNK,
    UPDATE_TOOL_CALL,
    UPDATE_TOOL_CALL_UPDATE,
    UnsupportedContentBlockError,
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
    tool_kind_for,
    tool_title_for,
)


class TestToolKindMapping:
    def test_known_tools(self):
        assert tool_kind_for("read_file") == TOOL_KIND_READ
        assert tool_kind_for("write_file") == TOOL_KIND_EDIT
        assert tool_kind_for("edit_file") == TOOL_KIND_EDIT
        assert tool_kind_for("run_command") == TOOL_KIND_EXECUTE
        assert tool_kind_for("grep") == TOOL_KIND_SEARCH
        assert tool_kind_for("think") == TOOL_KIND_THINK

    def test_namespaced_tools(self):
        assert tool_kind_for("mcp__server__do_thing") == TOOL_KIND_OTHER
        assert tool_kind_for("a2a__agent__ask") == TOOL_KIND_OTHER

    def test_unknown_tool(self):
        assert tool_kind_for("custom_widget") == TOOL_KIND_OTHER


class TestToolTitle:
    def test_read_file(self):
        assert "foo.txt" in tool_title_for("read_file", {"file_path": "foo.txt"})

    def test_run_command_string(self):
        title = tool_title_for("run_command", {"command": "ls -la"})
        assert title.startswith("$")
        assert "ls -la" in title

    def test_run_command_truncates(self):
        long_cmd = "echo " + "x" * 200
        title = tool_title_for("run_command", {"command": long_cmd})
        assert title.endswith("...")
        assert len(title) < 100

    def test_unknown_tool_uses_name(self):
        assert tool_title_for("mystery", {"foo": "bar"}) == "mystery"

    def test_no_arguments(self):
        assert tool_title_for("read_file", None) == "read_file"


class TestEncodeDecode:
    def test_round_trip_request(self):
        msg = {
            "jsonrpc": JSONRPC_VERSION,
            "id": 1,
            "method": METHOD_INITIALIZE,
            "params": {"protocolVersion": 1},
        }
        line = encode_message(msg)
        assert line.endswith(b"\n")
        # No embedded newlines in the JSON portion
        assert line.count(b"\n") == 1
        decoded = decode_message(line)
        assert decoded.method == METHOD_INITIALIZE
        assert decoded.id == 1
        assert decoded.params == {"protocolVersion": 1}
        assert not decoded.is_notification

    def test_decode_notification(self):
        line = encode_message(
            {
                "jsonrpc": JSONRPC_VERSION,
                "method": "session/cancel",
                "params": {"sessionId": "abc"},
            }
        )
        decoded = decode_message(line)
        assert decoded.is_notification
        assert decoded.method == "session/cancel"
        assert decoded.params == {"sessionId": "abc"}

    def test_decode_rejects_wrong_version(self):
        line = b'{"jsonrpc":"1.0","method":"x"}'
        with pytest.raises(ValueError):
            decode_message(line)

    def test_decode_rejects_missing_method(self):
        line = b'{"jsonrpc":"2.0","id":1}'
        with pytest.raises(ValueError):
            decode_message(line)

    def test_decode_rejects_non_object(self):
        line = b"[1,2,3]"
        with pytest.raises(ValueError):
            decode_message(line)

    def test_decode_rejects_bad_json(self):
        with pytest.raises(json.JSONDecodeError):
            decode_message(b"{not json")


class TestResponseBuilders:
    def test_make_result(self):
        msg = make_result(7, {"ok": True})
        assert msg == {"jsonrpc": JSONRPC_VERSION, "id": 7, "result": {"ok": True}}

    def test_make_result_with_none(self):
        msg = make_result(7, None)
        assert msg["result"] == {}

    def test_make_error(self):
        msg = make_error(7, ERROR_INVALID_PARAMS, "bad", data={"field": "cwd"})
        assert msg["jsonrpc"] == JSONRPC_VERSION
        assert msg["id"] == 7
        assert msg["error"]["code"] == ERROR_INVALID_PARAMS
        assert msg["error"]["message"] == "bad"
        assert msg["error"]["data"] == {"field": "cwd"}
        assert "result" not in msg

    def test_make_notification(self):
        msg = make_notification("session/update", {"sessionId": "x"})
        assert msg == {
            "jsonrpc": JSONRPC_VERSION,
            "method": "session/update",
            "params": {"sessionId": "x"},
        }
        assert "id" not in msg


class TestInitializeResponse:
    def test_default_shape(self):
        body = initialize_response()
        assert body["protocolVersion"] == PROTOCOL_VERSION
        caps = body["agentCapabilities"]
        assert caps["loadSession"] is False
        assert caps["promptCapabilities"] == {
            "image": False,
            "audio": False,
            "embeddedContext": False,
        }
        assert caps["mcpCapabilities"] == {"http": False, "sse": False}
        assert body["authMethods"] == []

    def test_negotiated_version(self):
        body = initialize_response(protocol_version=1)
        assert body["protocolVersion"] == 1


class TestUpdateBuilders:
    def test_text_chunk(self):
        body = text_chunk_update("hello")
        assert body == {
            "sessionUpdate": UPDATE_AGENT_MESSAGE_CHUNK,
            "content": {"type": "text", "text": "hello"},
        }

    def test_tool_call_pending(self):
        body = tool_call_update(
            tool_call_id="tc1",
            name="read_file",
            arguments={"file_path": "x.txt"},
        )
        assert body["sessionUpdate"] == UPDATE_TOOL_CALL
        assert body["toolCallId"] == "tc1"
        assert body["status"] == TOOL_STATUS_IN_PROGRESS
        assert body["kind"] == TOOL_KIND_READ
        assert body["rawInput"] == {"file_path": "x.txt"}
        assert "x.txt" in body["title"]

    def test_tool_call_no_arguments(self):
        body = tool_call_update(tool_call_id="tc1", name="read_file", arguments=None)
        assert "rawInput" not in body
        assert body["title"] == "read_file"

    def test_tool_call_completed_with_content(self):
        body = tool_call_update_update(
            tool_call_id="tc1",
            status=TOOL_STATUS_COMPLETED,
            content="file contents",
        )
        assert body["sessionUpdate"] == UPDATE_TOOL_CALL_UPDATE
        assert body["status"] == TOOL_STATUS_COMPLETED
        assert body["content"] == [
            {"type": "content", "content": {"type": "text", "text": "file contents"}}
        ]

    def test_tool_call_failed(self):
        body = tool_call_update_update(
            tool_call_id="tc1", status=TOOL_STATUS_FAILED, content="error: nope"
        )
        assert body["status"] == TOOL_STATUS_FAILED
        assert body["content"][0]["content"]["text"] == "error: nope"

    def test_session_update_envelope(self):
        env = session_update_payload("sess1", text_chunk_update("hi"))
        assert env["sessionId"] == "sess1"
        assert env["update"]["sessionUpdate"] == UPDATE_AGENT_MESSAGE_CHUNK


class TestExtractPromptText:
    def test_empty(self):
        assert extract_prompt_text(None) == ""
        assert extract_prompt_text([]) == ""

    def test_single_text_block(self):
        assert extract_prompt_text([{"type": "text", "text": "hello"}]) == "hello"

    def test_multiple_blocks(self):
        prompt = [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
        assert extract_prompt_text(prompt) == "first\n\nsecond"

    def test_resource_link_with_uri_and_name(self):
        prompt = [
            {
                "type": "resource_link",
                "uri": "file:///work/notes.md",
                "name": "notes.md",
            }
        ]
        assert extract_prompt_text(prompt) == "[notes.md](file:///work/notes.md)"

    def test_resource_link_uri_only(self):
        prompt = [{"type": "resource_link", "uri": "https://example.com/x"}]
        assert extract_prompt_text(prompt) == "https://example.com/x"

    def test_resource_link_with_inline_text(self):
        prompt = [
            {
                "type": "resource_link",
                "uri": "file:///a.txt",
                "name": "a.txt",
                "text": "inline body",
            }
        ]
        assert extract_prompt_text(prompt) == "inline body"

    def test_resource_link_only_prompt(self):
        """ACP baseline: clients can send a prompt made entirely of resource_link blocks."""
        prompt = [
            {
                "type": "resource_link",
                "uri": "file:///a.txt",
                "name": "a.txt",
            },
            {
                "type": "resource_link",
                "uri": "file:///b.txt",
                "name": "b.txt",
            },
        ]
        result = extract_prompt_text(prompt)
        assert "a.txt" in result and "b.txt" in result
        assert result  # non-empty: server should accept this prompt

    def test_image_block_rejected(self):
        prompt = [{"type": "image", "data": "base64...", "mimeType": "image/png"}]
        with pytest.raises(UnsupportedContentBlockError):
            extract_prompt_text(prompt)

    def test_audio_block_rejected(self):
        prompt = [{"type": "audio", "data": "base64...", "mimeType": "audio/wav"}]
        with pytest.raises(UnsupportedContentBlockError):
            extract_prompt_text(prompt)

    def test_embedded_resource_rejected(self):
        """We advertise embeddedContext: false, so this must not be silently accepted."""
        prompt = [{"type": "resource", "resource": {"uri": "x", "text": "y"}}]
        with pytest.raises(UnsupportedContentBlockError):
            extract_prompt_text(prompt)

    def test_unknown_block_type_skipped(self):
        prompt = [
            {"type": "text", "text": "ok"},
            {"type": "future_thing_we_dont_know", "value": 42},
        ]
        assert extract_prompt_text(prompt) == "ok"

    def test_string_passthrough(self):
        assert extract_prompt_text("plain") == "plain"

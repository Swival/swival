"""Tests for the outbound LLM filter feature."""

import sys
import textwrap
import types

import pytest

from swival.filter import run_llm_filter, FilterError


def _make_script(tmp_path, name, body):
    """Write a Python filter script, return a shell command to run it."""
    script = tmp_path / name
    code = "import json, sys\n" + textwrap.dedent(body).strip() + "\n"
    script.write_text(code)
    import shlex

    return f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"


def _msgs():
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Tell me about corp.example"},
    ]


class TestRunLlmFilter:
    def test_passthrough(self, tmp_path):
        """Filter that returns messages unchanged."""
        script = _make_script(
            tmp_path,
            "pass.py",
            """
payload = json.load(sys.stdin)
json.dump({"messages": payload["messages"]}, sys.stdout)
""",
        )
        result = run_llm_filter(script, _msgs(), "test-model", "generic")
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["content"] == "Tell me about corp.example"

    def test_rewrite(self, tmp_path):
        """Filter that redacts content."""
        script = _make_script(
            tmp_path,
            "redact.py",
            """
payload = json.load(sys.stdin)
for msg in payload["messages"]:
    if isinstance(msg.get("content"), str):
        msg["content"] = msg["content"].replace("corp.example", "[REDACTED]")
json.dump({"messages": payload["messages"]}, sys.stdout)
""",
        )
        result = run_llm_filter(script, _msgs(), "test-model", "generic")
        assert "[REDACTED]" in result[1]["content"]
        assert "corp.example" not in result[1]["content"]

    def test_block(self, tmp_path):
        """Filter that blocks the request."""
        script = _make_script(
            tmp_path,
            "block.py",
            """
json.dump({"allow": False, "reason": "contains internal URL"}, sys.stdout)
""",
        )
        with pytest.raises(FilterError, match="contains internal URL"):
            run_llm_filter(script, _msgs(), "test-model", "generic")

    def test_nonzero_exit(self, tmp_path):
        """Non-zero exit code raises FilterError."""
        script = _make_script(tmp_path, "fail.py", "sys.exit(1)")
        with pytest.raises(FilterError, match="exited with code 1"):
            run_llm_filter(script, _msgs(), "test-model", "generic")

    def test_bad_json(self, tmp_path):
        """Invalid JSON output raises FilterError."""
        script = _make_script(tmp_path, "bad.py", 'print("not json")')
        with pytest.raises(FilterError, match="invalid JSON"):
            run_llm_filter(script, _msgs(), "test-model", "generic")

    def test_missing_messages_key(self, tmp_path):
        """Output without 'messages' key raises FilterError."""
        script = _make_script(tmp_path, "nokey.py", 'json.dump({"foo": 1}, sys.stdout)')
        with pytest.raises(FilterError, match="missing 'messages' key"):
            run_llm_filter(script, _msgs(), "test-model", "generic")

    def test_timeout(self, tmp_path):
        """Script that hangs times out."""
        script = _make_script(
            tmp_path,
            "hang.py",
            """
import time
time.sleep(60)
""",
        )
        with pytest.raises(FilterError, match="timed out"):
            run_llm_filter(script, _msgs(), "test-model", "generic", timeout=1)

    def test_stderr_forwarded(self, tmp_path, capsys):
        """Script stderr is forwarded to our stderr."""
        script = _make_script(
            tmp_path,
            "warn.py",
            """
print("warning: something", file=sys.stderr)
payload = json.load(sys.stdin)
json.dump({"messages": payload["messages"]}, sys.stdout)
""",
        )
        run_llm_filter(script, _msgs(), "test-model", "generic")
        captured = capsys.readouterr()
        assert "warning: something" in captured.err

    def test_original_messages_not_mutated(self, tmp_path):
        """The original messages list must not be mutated."""
        script = _make_script(
            tmp_path,
            "rewrite.py",
            """
payload = json.load(sys.stdin)
payload["messages"][1]["content"] = "CHANGED"
json.dump({"messages": payload["messages"]}, sys.stdout)
""",
        )
        msgs = _msgs()
        original_content = msgs[1]["content"]
        result = run_llm_filter(script, msgs, "test-model", "generic")
        assert result[1]["content"] == "CHANGED"
        assert msgs[1]["content"] == original_content

    def test_invalid_role_rejected(self, tmp_path):
        """Messages with bad roles are rejected."""
        script = _make_script(
            tmp_path,
            "badrole.py",
            """
json.dump({"messages": [{"role": "hacker", "content": "hi"}]}, sys.stdout)
""",
        )
        with pytest.raises(FilterError, match="invalid role"):
            run_llm_filter(script, _msgs(), "test-model", "generic")

    def test_payload_includes_metadata(self, tmp_path):
        """Filter receives provider, model, call_kind, and tools."""
        script = _make_script(
            tmp_path,
            "echo.py",
            """
payload = json.load(sys.stdin)
assert payload["provider"] == "openrouter"
assert payload["model"] == "gpt-5"
assert payload["call_kind"] == "summary"
assert payload["tools"] == [{"type": "function", "function": {"name": "read"}}]
json.dump({"messages": payload["messages"]}, sys.stdout)
""",
        )
        tools = [{"type": "function", "function": {"name": "read"}}]
        run_llm_filter(
            script,
            _msgs(),
            "gpt-5",
            "openrouter",
            tools=tools,
            call_kind="summary",
        )

    def test_namespace_messages(self, tmp_path):
        """Filter handles namespace (non-dict) messages."""
        script = _make_script(
            tmp_path,
            "pass.py",
            """
payload = json.load(sys.stdin)
json.dump({"messages": payload["messages"]}, sys.stdout)
""",
        )
        msgs = [
            types.SimpleNamespace(role="system", content="sys prompt"),
            types.SimpleNamespace(role="user", content="hello"),
        ]
        result = run_llm_filter(script, msgs, "m", "generic")
        assert result[0]["role"] == "system"
        assert result[1]["content"] == "hello"

    def test_command_not_found(self, tmp_path):
        """Non-existent command raises FilterError."""
        with pytest.raises(FilterError, match="not found"):
            run_llm_filter("/nonexistent/filter-xyz-123", _msgs(), "m", "generic")


class TestCallLlmFilterIntegration:
    """Test that the filter is wired correctly into call_llm."""

    def test_filter_blocks_llm_call(self, tmp_path, monkeypatch):
        """When filter blocks, call_llm raises AgentError."""
        from swival import agent
        from swival.report import AgentError

        script = _make_script(
            tmp_path,
            "block.py",
            """
json.dump({"allow": False, "reason": "blocked by policy"}, sys.stdout)
""",
        )

        # call_llm should never reach litellm
        def _boom(*a, **kw):
            raise AssertionError("should not reach provider")

        monkeypatch.setattr("litellm.completion", _boom, raising=False)

        with pytest.raises(AgentError, match="blocked by policy"):
            agent.call_llm(
                base_url="http://localhost",
                model_id="test",
                messages=_msgs(),
                max_output_tokens=100,
                temperature=0,
                top_p=None,
                seed=None,
                tools=None,
                verbose=False,
                provider="generic",
                llm_filter=script,
            )

    def test_filter_runs_before_secret_shield(self, tmp_path, monkeypatch):
        """Filter sees plaintext messages, not encrypted ones."""
        from swival import agent

        script = _make_script(
            tmp_path,
            "check.py",
            """
payload = json.load(sys.stdin)
content = payload["messages"][1]["content"]
assert "corp.example" in content, f"expected plaintext, got: {content}"
json.dump({"messages": payload["messages"]}, sys.stdout)
""",
        )

        class FakeShield:
            def encrypt_messages(self, messages):
                import copy

                enc = copy.deepcopy(messages)
                for m in enc:
                    if isinstance(m.get("content"), str):
                        m["content"] = m["content"].replace("corp.example", "ENC_TOKEN")
                return enc

            def reverse_known(self, text):
                return text.replace("ENC_TOKEN", "corp.example") if text else text

        captured_messages = []

        def _fake_completion(**kwargs):
            captured_messages.extend(kwargs["messages"])
            resp = types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content="done", tool_calls=None, role="assistant"
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )
            return resp

        import litellm

        monkeypatch.setattr(litellm, "completion", _fake_completion)
        monkeypatch.setattr(litellm, "suppress_debug_info", True)

        msg, reason, *_ = agent.call_llm(
            base_url="http://localhost",
            model_id="test",
            messages=_msgs(),
            max_output_tokens=100,
            temperature=0,
            top_p=None,
            seed=None,
            tools=None,
            verbose=False,
            provider="generic",
            llm_filter=script,
            secret_shield=FakeShield(),
        )
        assert reason == "stop"
        # Secret shield ran after filter, so provider got encrypted content
        assert any("ENC_TOKEN" in str(m.get("content", "")) for m in captured_messages)


class TestValidation:
    """Test tightened message validation."""

    def test_tool_message_missing_tool_call_id(self, tmp_path):
        script = _make_script(
            tmp_path,
            "bad_tool.py",
            """
json.dump({"messages": [
    {"role": "tool", "content": "result"}
]}, sys.stdout)
""",
        )
        with pytest.raises(FilterError, match="missing tool_call_id"):
            run_llm_filter(script, _msgs(), "m", "generic")

    def test_tool_calls_missing_id(self, tmp_path):
        script = _make_script(
            tmp_path,
            "bad_tc.py",
            """
json.dump({"messages": [
    {"role": "assistant", "tool_calls": [
        {"function": {"name": "read", "arguments": "{}"}}
    ]}
]}, sys.stdout)
""",
        )
        with pytest.raises(FilterError, match="missing id"):
            run_llm_filter(script, _msgs(), "m", "generic")

    def test_tool_calls_missing_function_name(self, tmp_path):
        script = _make_script(
            tmp_path,
            "bad_fn.py",
            """
json.dump({"messages": [
    {"role": "assistant", "tool_calls": [
        {"id": "tc1", "function": {"arguments": "{}"}}
    ]}
]}, sys.stdout)
""",
        )
        with pytest.raises(FilterError, match="missing function.name"):
            run_llm_filter(script, _msgs(), "m", "generic")

    def test_valid_tool_calls_pass(self, tmp_path):
        script = _make_script(
            tmp_path,
            "ok_tc.py",
            """
json.dump({"messages": [
    {"role": "assistant", "tool_calls": [
        {"id": "tc1", "type": "function", "function": {"name": "read", "arguments": "{}"}}
    ]},
    {"role": "tool", "tool_call_id": "tc1", "content": "ok"}
]}, sys.stdout)
""",
        )
        result = run_llm_filter(script, _msgs(), "m", "generic")
        assert len(result) == 2
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "tool"

    def test_multimodal_content_preserved(self, tmp_path):
        """Structured content arrays survive filter round-trip."""
        script = _make_script(
            tmp_path,
            "pass.py",
            """
payload = json.load(sys.stdin)
json.dump({"messages": payload["messages"]}, sys.stdout)
""",
        )
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc"},
                    },
                ],
            },
        ]
        result = run_llm_filter(script, msgs, "m", "generic")
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][1]["type"] == "image_url"

"""Tests for the command provider."""

import subprocess as sp

import pytest

from swival.agent import (
    _render_transcript,
    _make_synthetic_message,
    _parse_swival_calls,
    _render_swival_tool_catalog,
    _COMMAND_TOOL_CONTEXT_PREFIX,
    build_system_prompt,
    call_llm,
    is_pinned,
    resolve_provider,
    SYNTHETIC_USER_PREFIXES,
)
from swival.config import _resolve_command_model
from swival.report import AgentError, ConfigError


# ---------------------------------------------------------------------------
# resolve_provider
# ---------------------------------------------------------------------------


class TestResolveCommand:
    def test_requires_model(self):
        with pytest.raises(ConfigError):
            resolve_provider("command", None, None, None, None, False)

    def test_empty_model_rejected(self):
        with pytest.raises(ConfigError):
            resolve_provider("command", "  ", None, None, None, False)

    def test_returns_correct_shape(self):
        model_id, base, key, ctx, kwargs = resolve_provider(
            "command", "echo hello", None, None, None, False
        )
        assert "echo" in model_id
        assert base is None
        assert key is None
        assert kwargs["provider"] == "command"

    def test_preserves_max_context_tokens(self):
        _, _, _, ctx, _ = resolve_provider(
            "command", "echo hello", None, None, 8192, False
        )
        assert ctx == 8192

    def test_none_context_when_no_max(self):
        _, _, _, ctx, _ = resolve_provider(
            "command", "echo hello", None, None, None, False
        )
        assert ctx is None

    def test_rejects_missing_command(self):
        with pytest.raises(ConfigError, match="command not found"):
            resolve_provider(
                "command", "nonexistent_binary_xyz", None, None, None, False
            )


# ---------------------------------------------------------------------------
# call_llm (command provider)
# ---------------------------------------------------------------------------


class TestCallCommand:
    def test_simple_echo(self):
        msg, reason, *_ = call_llm(
            None,
            "echo 'hello world'",
            [{"role": "user", "content": "hi"}],
            100,
            0.5,
            1.0,
            None,
            None,
            False,
            provider="command",
        )
        assert msg.content == "hello world"
        assert msg.tool_calls is None
        assert reason == "stop"

    def test_receives_full_transcript_on_stdin(self):
        msg, *_ = call_llm(
            None,
            "cat",
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "test input"},
            ],
            100,
            0.5,
            1.0,
            None,
            None,
            False,
            provider="command",
        )
        assert "[system]" in msg.content
        assert "You are helpful." in msg.content
        assert "[user]" in msg.content
        assert "test input" in msg.content

    def test_model_dump_exclude_none(self):
        msg, *_ = call_llm(
            None,
            "echo ok",
            [{"role": "user", "content": "hi"}],
            100,
            0.5,
            1.0,
            None,
            None,
            False,
            provider="command",
        )
        dumped = msg.model_dump(exclude_none=True)
        assert dumped == {"role": "assistant", "content": "ok"}
        assert "tool_calls" not in dumped

    def test_model_dump_full(self):
        msg, *_ = call_llm(
            None,
            "echo ok",
            [{"role": "user", "content": "hi"}],
            100,
            0.5,
            1.0,
            None,
            None,
            False,
            provider="command",
        )
        dumped = msg.model_dump()
        assert dumped["tool_calls"] is None

    def test_nonzero_exit(self):
        with pytest.raises(AgentError):
            call_llm(
                None,
                "false",
                [{"role": "user", "content": "hi"}],
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="command",
            )

    def test_timeout(self, monkeypatch):
        def fake_run(*a, **kw):
            raise sp.TimeoutExpired(cmd=a[0], timeout=1)

        monkeypatch.setattr(sp, "run", fake_run)
        with pytest.raises(AgentError, match="timed out"):
            call_llm(
                None,
                "sleep 999",
                [{"role": "user", "content": "hi"}],
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="command",
            )

    def test_os_error(self):
        with pytest.raises(AgentError, match="failed to start"):
            call_llm(
                None,
                "/dev/null",
                [{"role": "user", "content": "hi"}],
                100,
                0.5,
                1.0,
                None,
                None,
                False,
                provider="command",
            )

    def test_max_output_tokens_truncates(self):
        # "echo" produces a short output; use a script that generates many tokens
        msg, *_ = call_llm(
            None,
            "echo 'word ' 'word ' 'word ' 'word ' 'word ' 'word ' 'word ' 'word ' 'word ' 'word '",
            [{"role": "user", "content": "hi"}],
            2,  # max_output_tokens = 2 tokens
            0.5,
            1.0,
            None,
            None,
            False,
            provider="command",
        )
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        assert len(enc.encode(msg.content)) <= 2

    def test_stderr_suppressed_in_quiet_mode(self, tmp_path):
        """When verbose=False (quiet mode), stderr is not printed."""
        script = tmp_path / "warn.sh"
        script.write_text("#!/bin/sh\necho 'result'\necho 'warning' >&2")
        script.chmod(0o755)
        msg, *_ = call_llm(
            None,
            str(script),
            [{"role": "user", "content": "hi"}],
            100,
            0.5,
            1.0,
            None,
            None,
            False,  # verbose=False (quiet mode)
            provider="command",
        )
        assert msg.content == "result"


# ---------------------------------------------------------------------------
# _render_transcript
# ---------------------------------------------------------------------------


class TestRenderTranscript:
    def test_system_user_assistant(self):
        transcript = _render_transcript(
            [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
        )
        assert "[system]\nBe helpful." in transcript
        assert "[user]\nHello" in transcript
        assert "[assistant]\nHi there" in transcript

    def test_tool_results_get_function_name(self):
        transcript = _render_transcript(
            [
                {"role": "user", "content": "Read foo.py"},
                {
                    "role": "assistant",
                    "content": "I'll read that file.",
                    "tool_calls": [
                        {
                            "id": "tc_1",
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "tc_1", "content": "print('hello')"},
            ]
        )
        assert "[tool:read_file]" in transcript
        assert "print('hello')" in transcript

    def test_tool_results_without_matching_id_fallback(self):
        transcript = _render_transcript(
            [
                {
                    "role": "tool",
                    "tool_call_id": "unknown_id",
                    "content": "some result",
                },
            ]
        )
        assert "[tool:tool]" in transcript

    def test_image_placeholder(self):
        transcript = _render_transcript(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Look at this"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,abc"},
                        },
                    ],
                },
            ]
        )
        assert "Look at this" in transcript
        assert "[image omitted]" in transcript

    def test_multipart_text_only(self):
        transcript = _render_transcript(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First"},
                        {"type": "text", "text": "Second"},
                    ],
                },
            ]
        )
        assert "First" in transcript
        assert "Second" in transcript

    def test_empty_content_skipped(self):
        transcript = _render_transcript(
            [
                {"role": "assistant", "content": ""},
                {"role": "user", "content": "hello"},
            ]
        )
        assert "[assistant]" not in transcript
        assert "[user]\nhello" in transcript


# ---------------------------------------------------------------------------
# _make_synthetic_message
# ---------------------------------------------------------------------------


class TestSyntheticMessage:
    def test_attributes(self):
        msg = _make_synthetic_message("hello")
        assert msg.role == "assistant"
        assert msg.content == "hello"
        assert msg.tool_calls is None

    def test_getattr_compat(self):
        msg = _make_synthetic_message("hello")
        assert getattr(msg, "content", None) == "hello"
        assert getattr(msg, "tool_calls", None) is None

    def test_model_dump_exclude_none(self):
        msg = _make_synthetic_message("hello")
        d = msg.model_dump(exclude_none=True)
        assert d == {"role": "assistant", "content": "hello"}
        assert "tool_calls" not in d

    def test_model_dump_full(self):
        msg = _make_synthetic_message("hello")
        d = msg.model_dump()
        assert d == {"role": "assistant", "content": "hello", "tool_calls": None}


# ---------------------------------------------------------------------------
# config: _resolve_command_model
# ---------------------------------------------------------------------------


class TestConfigResolution:
    def test_relative_path_resolved_against_config_dir(self, tmp_path):
        script = tmp_path / "script.sh"
        script.write_text("#!/bin/sh\necho hi")
        script.chmod(0o755)
        config = {"provider": "command", "model": "./script.sh"}
        _resolve_command_model(config, tmp_path, "test")
        assert config["model"] == str(script)

    def test_absolute_path_unchanged(self, tmp_path):
        script = tmp_path / "script.sh"
        script.write_text("#!/bin/sh\necho hi")
        script.chmod(0o755)
        config = {"provider": "command", "model": str(script)}
        _resolve_command_model(config, tmp_path, "test")
        assert config["model"] == str(script)

    def test_bare_command_unchanged(self, tmp_path):
        config = {"provider": "command", "model": "codex exec --full-auto"}
        _resolve_command_model(config, tmp_path, "test")
        assert config["model"] == "codex exec --full-auto"

    def test_noop_for_other_providers(self, tmp_path):
        config = {"provider": "lmstudio", "model": "./script.sh"}
        _resolve_command_model(config, tmp_path, "test")
        assert config["model"] == "./script.sh"


# ---------------------------------------------------------------------------
# build_system_prompt (command provider)
# ---------------------------------------------------------------------------


class TestCommandSystemPrompt:
    def test_command_provider_excludes_tool_instructions(self, tmp_path):
        """Command provider should not mention tools like read_file, write_file."""
        content, _ = build_system_prompt(
            base_dir=str(tmp_path),
            system_prompt=None,
            no_system_prompt=False,
            no_instructions=True,
            no_memory=True,
            skills_catalog={},
            verbose=False,
            provider="command",
        )
        assert "read_file" not in content
        assert "write_file" not in content
        assert "think" not in content

    def test_command_provider_custom_prompt_preserved(self, tmp_path):
        """Explicit --system-prompt overrides the command default too."""
        content, _ = build_system_prompt(
            base_dir=str(tmp_path),
            system_prompt="Custom prompt.",
            no_system_prompt=False,
            no_instructions=True,
            no_memory=True,
            skills_catalog={},
            verbose=False,
            provider="command",
        )
        assert "Custom prompt." in content

    def test_non_command_provider_includes_tools(self, tmp_path):
        """Non-command providers get the default tool-oriented prompt."""
        content, _ = build_system_prompt(
            base_dir=str(tmp_path),
            system_prompt=None,
            no_system_prompt=False,
            no_instructions=True,
            no_memory=True,
            skills_catalog={},
            verbose=False,
            provider="lmstudio",
        )
        assert "read_file" in content

    def test_command_provider_excludes_yolo(self, tmp_path):
        """Command provider should not include run_command help even with yolo."""
        content, _ = build_system_prompt(
            base_dir=str(tmp_path),
            system_prompt=None,
            no_system_prompt=False,
            no_instructions=True,
            no_memory=True,
            skills_catalog={},
            verbose=False,
            provider="command",
        )
        assert "run_command" not in content

    def test_command_provider_excludes_whitelisted_commands(self, tmp_path):
        content, _ = build_system_prompt(
            base_dir=str(tmp_path),
            system_prompt=None,
            no_system_prompt=False,
            no_instructions=True,
            no_memory=True,
            skills_catalog={},
            verbose=False,
            provider="command",
        )
        assert "run_command" not in content

    def test_command_provider_excludes_skills(self, tmp_path):
        content, _ = build_system_prompt(
            base_dir=str(tmp_path),
            system_prompt=None,
            no_system_prompt=False,
            no_instructions=True,
            no_memory=True,
            skills_catalog={"my-skill": {"name": "my-skill", "description": "test"}},
            verbose=False,
            provider="command",
        )
        assert "use_skill" not in content
        assert "my-skill" not in content

    def test_command_provider_excludes_mcp(self, tmp_path):
        content, _ = build_system_prompt(
            base_dir=str(tmp_path),
            system_prompt=None,
            no_system_prompt=False,
            no_instructions=True,
            no_memory=True,
            skills_catalog={},
            verbose=False,
            mcp_tool_info={"server1": [{"name": "mcp_tool", "description": "test"}]},
            provider="command",
        )
        assert "mcp_tool" not in content


# ---------------------------------------------------------------------------
# _parse_swival_calls
# ---------------------------------------------------------------------------


class TestParseSwivalCalls:
    def test_basic_call(self):
        text = """I'll query the database.
<swival:call id="c1" name="mcp__db__query">
{"sql": "SELECT 1"}
</swival:call>
"""
        calls = _parse_swival_calls(text)
        assert len(calls) == 1
        assert calls[0] == ("c1", "mcp__db__query", {"sql": "SELECT 1"})

    def test_reversed_attribute_order(self):
        text = '<swival:call name="mcp__db__query" id="c1">{"sql": "SELECT 1"}</swival:call>'
        calls = _parse_swival_calls(text)
        assert len(calls) == 1
        assert calls[0][0] == "c1"
        assert calls[0][1] == "mcp__db__query"

    def test_extra_attributes_ignored(self):
        text = '<swival:call id="c1" name="tool" priority="high">{"a": 1}</swival:call>'
        calls = _parse_swival_calls(text)
        assert len(calls) == 1
        assert calls[0] == ("c1", "tool", {"a": 1})

    def test_multiple_calls(self):
        text = """
<swival:call id="c1" name="t1">{"x": 1}</swival:call>
Some reasoning text.
<swival:call id="c2" name="t2">{"y": 2}</swival:call>
"""
        calls = _parse_swival_calls(text)
        assert len(calls) == 2
        assert calls[0][0] == "c1"
        assert calls[1][0] == "c2"

    def test_missing_id_skipped(self):
        text = '<swival:call name="tool">{"a": 1}</swival:call>'
        calls = _parse_swival_calls(text)
        assert len(calls) == 0

    def test_missing_name_skipped(self):
        text = '<swival:call id="c1">{"a": 1}</swival:call>'
        calls = _parse_swival_calls(text)
        assert len(calls) == 0

    def test_malformed_json_returns_parse_error(self):
        text = '<swival:call id="c1" name="tool">{bad json}</swival:call>'
        calls = _parse_swival_calls(text)
        assert len(calls) == 1
        assert calls[0][0] == "c1"
        assert calls[0][1] == "tool"
        assert "_parse_error" in calls[0][2]

    def test_no_calls_returns_empty(self):
        text = "Just some plain text response with no tool calls."
        calls = _parse_swival_calls(text)
        assert calls == []

    def test_extra_whitespace_in_tag(self):
        text = '<swival:call  id="c1"  name="tool" >{"a": 1}</swival:call>'
        calls = _parse_swival_calls(text)
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# _render_swival_tool_catalog
# ---------------------------------------------------------------------------


class TestRenderSwivalToolCatalog:
    def test_basic_catalog(self):
        schemas = [
            {
                "type": "function",
                "function": {
                    "name": "mcp__db__query",
                    "description": "Run a SQL query.",
                    "parameters": {
                        "type": "object",
                        "properties": {"sql": {"type": "string"}},
                        "required": ["sql"],
                    },
                },
            }
        ]
        result = _render_swival_tool_catalog(schemas)
        assert "mcp__db__query" in result
        assert "Run a SQL query." in result
        assert '"sql": string' in result

    def test_optional_params(self):
        schemas = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "string"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a"],
                    },
                },
            }
        ]
        result = _render_swival_tool_catalog(schemas)
        assert '"a": string' in result
        assert '"b?": integer' in result


# ---------------------------------------------------------------------------
# _render_transcript with swival_result messages
# ---------------------------------------------------------------------------


class TestRenderTranscriptSwivalResult:
    def test_swival_result_rendering(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "let me check"},
            {
                "role": "tool",
                "tool_call_id": "c1",
                "name": "mcp__db__query",
                "content": '[{"count": 42}]',
            },
        ]
        transcript = _render_transcript(messages)
        assert '[swival_result id="c1" name="mcp__db__query"]' in transcript
        assert '[{"count": 42}]' in transcript

    def test_a2a_result_rendering(self):
        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "tool",
                "tool_call_id": "c2",
                "name": "a2a__agent__ask",
                "content": "agent response",
            },
        ]
        transcript = _render_transcript(messages)
        assert '[swival_result id="c2" name="a2a__agent__ask"]' in transcript

    def test_use_skill_result_rendering(self):
        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "tool",
                "tool_call_id": "c3",
                "name": "use_skill",
                "content": "skill output",
            },
        ]
        transcript = _render_transcript(messages)
        assert '[swival_result id="c3" name="use_skill"]' in transcript

    def test_regular_tool_still_uses_old_format(self):
        """Non-swival tools should still render as [tool:name]."""
        import types

        tc = types.SimpleNamespace(
            id="tc1",
            function=types.SimpleNamespace(name="read_file", arguments="{}"),
        )
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tc],
            },
            {
                "role": "tool",
                "tool_call_id": "tc1",
                "content": "file content",
            },
        ]
        transcript = _render_transcript(messages)
        assert "[tool:read_file]" in transcript
        assert "swival_result" not in transcript


# ---------------------------------------------------------------------------
# is_pinned / SYNTHETIC_USER_PREFIXES
# ---------------------------------------------------------------------------


class TestCommandToolContextPrefix:
    def test_prefix_in_synthetic_user_prefixes(self):
        assert _COMMAND_TOOL_CONTEXT_PREFIX in SYNTHETIC_USER_PREFIXES

    def test_is_pinned_returns_false_for_context_message(self):
        turn = [
            {
                "role": "user",
                "content": (
                    _COMMAND_TOOL_CONTEXT_PREFIX + " external tool calls made during "
                    "the previous response:\n  - mcp__db__query: ok\n]"
                ),
            }
        ]
        assert is_pinned(turn) is False

    def test_is_pinned_returns_true_for_real_user(self):
        turn = [{"role": "user", "content": "Fix the bug in main.py"}]
        assert is_pinned(turn) is True


# ---------------------------------------------------------------------------
# build_system_prompt with command_tool_schemas
# ---------------------------------------------------------------------------


class TestCommandProviderToolCatalog:
    def test_catalog_injected_when_schemas_present(self, tmp_path):
        schemas = [
            {
                "type": "function",
                "function": {
                    "name": "mcp__db__query",
                    "description": "Run SQL.",
                    "parameters": {
                        "type": "object",
                        "properties": {"sql": {"type": "string"}},
                        "required": ["sql"],
                    },
                },
            }
        ]
        content, _ = build_system_prompt(
            base_dir=str(tmp_path),
            system_prompt=None,
            no_system_prompt=False,
            no_instructions=True,
            no_memory=True,
            skills_catalog={},
            verbose=False,
            provider="command",
            command_tool_schemas=schemas,
        )
        assert "swival:call" in content
        assert "mcp__db__query" in content
        assert "Run SQL." in content
        assert "UNIQUE_ID" in content

    def test_no_catalog_when_no_schemas(self, tmp_path):
        content, _ = build_system_prompt(
            base_dir=str(tmp_path),
            system_prompt=None,
            no_system_prompt=False,
            no_instructions=True,
            no_memory=True,
            skills_catalog={},
            verbose=False,
            provider="command",
            command_tool_schemas=None,
        )
        assert "swival:call" not in content

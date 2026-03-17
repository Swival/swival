"""Tests for the command provider."""

import subprocess as sp

import pytest

from swival.agent import (
    _render_transcript,
    _make_synthetic_message,
    build_system_prompt,
    call_llm,
    resolve_provider,
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
        msg, reason = call_llm(
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
        msg, _ = call_llm(
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
        msg, _ = call_llm(
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
        msg, _ = call_llm(
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
        msg, _ = call_llm(
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
        msg, _ = call_llm(
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
        config = {"provider": "command", "model": "claude -p"}
        _resolve_command_model(config, tmp_path, "test")
        assert config["model"] == "claude -p"

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
            yolo=False,
            resolved_commands={},
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
            yolo=False,
            resolved_commands={},
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
            yolo=False,
            resolved_commands={},
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
            yolo=True,
            resolved_commands={},
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
            yolo=False,
            resolved_commands={"ls": "/bin/ls"},
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
            yolo=False,
            resolved_commands={},
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
            yolo=False,
            resolved_commands={},
            verbose=False,
            mcp_tool_info={"server1": [{"name": "mcp_tool", "description": "test"}]},
            provider="command",
        )
        assert "mcp_tool" not in content

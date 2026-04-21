"""Tests for custom command execution (!command) in the REPL."""

import stat
import types
from unittest.mock import patch, MagicMock

from swival.agent import (
    _repl_run_custom_command,
    _truncate_for_context,
    repl_loop,
)
from swival.thinking import ThinkingState
from swival.todo import TodoState
from swival.snapshot import SnapshotState


def _make_script(path, content="#!/bin/sh\necho hello", executable=True):
    """Create a script file, optionally marking it executable."""
    path.write_text(content)
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _commands_dir(tmp_path):
    """Return a commands directory inside a fake XDG config home."""
    d = tmp_path / "config" / "swival" / "commands"
    d.mkdir(parents=True)
    return d


# ---------------------------------------------------------------------------
# _repl_run_custom_command
# ---------------------------------------------------------------------------


class TestReplRunCustomCommand:
    def test_happy_path(self, tmp_path, monkeypatch):
        cmd_dir = _commands_dir(tmp_path)
        _make_script(cmd_dir / "greet", '#!/bin/sh\necho "hi there"')
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command("!greet", str(tmp_path))
        assert result is not None
        cmd_name, stdout, inline_path = result
        assert cmd_name == "greet"
        assert stdout == "hi there"
        assert inline_path is None

    def test_with_arguments(self, tmp_path, monkeypatch):
        cmd_dir = _commands_dir(tmp_path)
        # $1 is base_dir, $2 is the raw argument string
        _make_script(cmd_dir / "echo-args", '#!/bin/sh\necho "$2"')
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command(
            '!echo-args "hello world" --flag', str(tmp_path)
        )
        assert result is not None
        _, stdout, _ = result
        assert stdout == '"hello world" --flag'

    def test_swival_model_env_var(self, tmp_path, monkeypatch):
        cmd_dir = _commands_dir(tmp_path)
        _make_script(cmd_dir / "show-model", '#!/bin/sh\necho "$SWIVAL_MODEL"')
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command(
            "!show-model", str(tmp_path), model_id="qwen3.5-9b"
        )
        assert result is not None
        _, stdout, _ = result
        assert stdout == "qwen3.5-9b"

    def test_base_dir_passed_as_first_arg(self, tmp_path, monkeypatch):
        cmd_dir = _commands_dir(tmp_path)
        _make_script(cmd_dir / "show-base", '#!/bin/sh\necho "$1"')
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command("!show-base", str(tmp_path))
        assert result is not None
        _, stdout, _ = result
        assert stdout == str(tmp_path)

    def test_extension_fallback(self, tmp_path, monkeypatch):
        cmd_dir = _commands_dir(tmp_path)
        _make_script(cmd_dir / "greet.sh", '#!/bin/sh\necho "hi from sh"')
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command("!greet", str(tmp_path))
        assert result is not None
        assert result[1] == "hi from sh"

    def test_extension_ambiguous(self, tmp_path, monkeypatch):
        cmd_dir = _commands_dir(tmp_path)
        _make_script(cmd_dir / "dup.sh", "#!/bin/sh\necho a")
        _make_script(cmd_dir / "dup.py", '#!/usr/bin/env python3\nprint("b")')
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command("!dup", str(tmp_path))
        assert result is None

    def test_nonexecutable_sidecar_ignored(self, tmp_path, monkeypatch):
        cmd_dir = _commands_dir(tmp_path)
        _make_script(cmd_dir / "tool.sh", '#!/bin/sh\necho "ok"')
        _make_script(cmd_dir / "tool.txt", "just a note", executable=False)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command("!tool", str(tmp_path))
        assert result is not None
        assert result[1] == "ok"

    def test_exact_name_preferred_over_extension(self, tmp_path, monkeypatch):
        cmd_dir = _commands_dir(tmp_path)
        _make_script(cmd_dir / "tool", '#!/bin/sh\necho "exact"')
        _make_script(cmd_dir / "tool.sh", '#!/bin/sh\necho "extension"')
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command("!tool", str(tmp_path))
        assert result is not None
        assert result[1] == "exact"

    def test_cwd_is_base_dir(self, tmp_path, monkeypatch):
        cmd_dir = _commands_dir(tmp_path)
        _make_script(cmd_dir / "show-cwd", "#!/bin/sh\npwd")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        result = _repl_run_custom_command("!show-cwd", str(project_dir))
        assert result is not None
        _, stdout, _ = result
        assert stdout == str(project_dir)

    def test_command_not_found(self, tmp_path, monkeypatch, capsys):
        _commands_dir(tmp_path)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command("!nosuch", str(tmp_path))
        assert result is None

    def test_text_file_inlined(self, tmp_path, monkeypatch):
        cmd_dir = _commands_dir(tmp_path)
        (cmd_dir / "prompt").write_text("explain $1 in detail")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command("!prompt recursion", str(tmp_path))
        assert result is not None
        cmd_name, content, inline_path = result
        assert cmd_name == "prompt"
        assert content == "explain recursion in detail"
        assert inline_path is not None

    def test_text_file_no_arg_placeholder_preserved(self, tmp_path, monkeypatch):
        cmd_dir = _commands_dir(tmp_path)
        (cmd_dir / "tmpl").write_text("describe $1")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command("!tmpl", str(tmp_path))
        assert result is not None
        _, content, _ = result
        assert content == "describe $1"

    def test_binary_file_not_executable(self, tmp_path, monkeypatch, capsys):
        cmd_dir = _commands_dir(tmp_path)
        (cmd_dir / "binary").write_bytes(b"\x00\x01\x02\x03binary data")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command("!binary", str(tmp_path))
        assert result is None

    def test_nonzero_exit_with_stderr(self, tmp_path, monkeypatch):
        cmd_dir = _commands_dir(tmp_path)
        _make_script(cmd_dir / "fail", "#!/bin/sh\necho oops >&2; exit 1")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command("!fail", str(tmp_path))
        assert result is None

    def test_nonzero_exit_stdout_only(self, tmp_path, monkeypatch):
        cmd_dir = _commands_dir(tmp_path)
        _make_script(cmd_dir / "fail2", "#!/bin/sh\necho 'stdout err'; exit 1")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command("!fail2", str(tmp_path))
        assert result is None

    def test_nonzero_exit_no_output(self, tmp_path, monkeypatch):
        cmd_dir = _commands_dir(tmp_path)
        _make_script(cmd_dir / "fail3", "#!/bin/sh\nexit 42")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command("!fail3", str(tmp_path))
        assert result is None

    def test_empty_output(self, tmp_path, monkeypatch):
        cmd_dir = _commands_dir(tmp_path)
        _make_script(cmd_dir / "empty", "#!/bin/sh\n")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command("!empty", str(tmp_path))
        assert result is None

    def test_path_traversal_rejected(self, tmp_path, monkeypatch):
        _commands_dir(tmp_path)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command("!../evil", str(tmp_path))
        assert result is None

    def test_slash_in_name_rejected(self, tmp_path, monkeypatch):
        _commands_dir(tmp_path)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command("!foo/bar", str(tmp_path))
        assert result is None

    def test_timeout(self, tmp_path, monkeypatch):
        cmd_dir = _commands_dir(tmp_path)
        _make_script(cmd_dir / "slow", "#!/bin/sh\nsleep 999")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        with patch(
            "swival.agent.subprocess.run",
            side_effect=__import__("subprocess").TimeoutExpired("slow", 30),
        ):
            result = _repl_run_custom_command("!slow", str(tmp_path))
        assert result is None

    def test_bad_shebang(self, tmp_path, monkeypatch):
        cmd_dir = _commands_dir(tmp_path)
        _make_script(cmd_dir / "bad", "#!/no/such/interpreter\necho hi")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command("!bad", str(tmp_path))
        assert result is None

    def test_unclosed_quotes_passed_through(self, tmp_path, monkeypatch):
        cmd_dir = _commands_dir(tmp_path)
        _make_script(cmd_dir / "cmd", '#!/bin/sh\necho "$2"')
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command('!cmd "unclosed', str(tmp_path))
        assert result is not None
        _, stdout, _ = result
        assert stdout == '"unclosed'

    def test_xdg_config_home_override(self, tmp_path, monkeypatch):
        custom_xdg = tmp_path / "custom_xdg"
        cmd_dir = custom_xdg / "swival" / "commands"
        cmd_dir.mkdir(parents=True)
        _make_script(cmd_dir / "test-cmd", '#!/bin/sh\necho "from custom xdg"')
        monkeypatch.setenv("XDG_CONFIG_HOME", str(custom_xdg))

        result = _repl_run_custom_command("!test-cmd", str(tmp_path))
        assert result is not None
        assert result[1] == "from custom xdg"

    def test_stderr_from_success(self, tmp_path, monkeypatch, capsys):
        cmd_dir = _commands_dir(tmp_path)
        _make_script(cmd_dir / "warn", '#!/bin/sh\necho "warning" >&2; echo "output"')
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command("!warn", str(tmp_path))
        assert result is not None
        _, stdout, _ = result
        assert stdout == "output"
        captured = capsys.readouterr()
        assert "warning" in captured.err

    def test_no_commands_directory(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        result = _repl_run_custom_command("!foo", str(tmp_path))
        assert result is None

    def test_command_name_validation(self, tmp_path, monkeypatch):
        _commands_dir(tmp_path)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        for bad_name in ["foo;bar", "foo&bar", ""]:
            result = _repl_run_custom_command(f"!{bad_name}", str(tmp_path))
            assert result is None

    def test_args_as_single_string(self, tmp_path, monkeypatch):
        cmd_dir = _commands_dir(tmp_path)
        _make_script(cmd_dir / "mycmd", '#!/bin/sh\necho "$2"')
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command("!mycmd a b c", str(tmp_path))
        assert result is not None
        _, stdout, _ = result
        assert stdout == "a b c"

    def test_no_args_only_base_dir(self, tmp_path, monkeypatch):
        cmd_dir = _commands_dir(tmp_path)
        _make_script(cmd_dir / "mycmd", '#!/bin/sh\necho "$#"')
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command("!mycmd", str(tmp_path))
        assert result is not None
        _, stdout, _ = result
        assert stdout == "1"

    def test_leading_trailing_spaces_stripped(self, tmp_path, monkeypatch):
        cmd_dir = _commands_dir(tmp_path)
        _make_script(cmd_dir / "mycmd", '#!/bin/sh\necho "$2"')
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command("!mycmd   a b   ", str(tmp_path))
        assert result is not None
        _, stdout, _ = result
        assert stdout == "a b"

    def test_whitespace_after_bang(self, tmp_path, monkeypatch):
        cmd_dir = _commands_dir(tmp_path)
        _make_script(cmd_dir / "greet", '#!/bin/sh\necho "hi"')
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command("!  greet", str(tmp_path))
        assert result is not None
        cmd_name, stdout, _ = result
        assert cmd_name == "greet"
        assert stdout == "hi"

    def test_text_file_preserves_whitespace(self, tmp_path, monkeypatch):
        cmd_dir = _commands_dir(tmp_path)
        (cmd_dir / "padded").write_text("\n  preamble\n\ncontent\n\n  postamble\n")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command("!padded", str(tmp_path))
        assert result is not None
        _, content, _ = result
        assert content == "\n  preamble\n\ncontent\n\n  postamble\n"

    def test_text_file_multibyte_at_512_boundary(self, tmp_path, monkeypatch):
        cmd_dir = _commands_dir(tmp_path)
        (cmd_dir / "emoji").write_text("a" * 511 + "\U0001f642 hello")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command("!emoji", str(tmp_path))
        assert result is not None
        _, content, inline_path = result
        assert inline_path is not None
        assert "\U0001f642 hello" in content

    def test_exec_beats_text_same_stem(self, tmp_path, monkeypatch):
        cmd_dir = _commands_dir(tmp_path)
        _make_script(cmd_dir / "tool.sh", '#!/bin/sh\necho "exec"')
        (cmd_dir / "tool").write_text("text template")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command("!tool", str(tmp_path))
        assert result is not None
        _, content, inline_path = result
        assert content == "exec"
        assert inline_path is None

    def test_stem_exec_beats_exact_text(self, tmp_path, monkeypatch):
        cmd_dir = _commands_dir(tmp_path)
        _make_script(cmd_dir / "deploy.sh", '#!/bin/sh\necho "deployed"')
        (cmd_dir / "deploy").write_text("text fallback")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command("!deploy", str(tmp_path))
        assert result is not None
        _, content, inline_path = result
        assert content == "deployed"
        assert inline_path is None

    def test_multiple_text_candidates_ambiguous(self, tmp_path, monkeypatch):
        cmd_dir = _commands_dir(tmp_path)
        (cmd_dir / "ask.txt").write_text("text a")
        (cmd_dir / "ask.md").write_text("text b")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command("!ask", str(tmp_path))
        assert result is None

    def test_dotfile_invisible(self, tmp_path, monkeypatch):
        cmd_dir = _commands_dir(tmp_path)
        (cmd_dir / ".hidden").write_text("should not be found")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command("!.hidden", str(tmp_path))
        assert result is None

    def test_backup_file_not_matched_by_stem(self, tmp_path, monkeypatch):
        cmd_dir = _commands_dir(tmp_path)
        (cmd_dir / "ask~").write_text("backup")
        (cmd_dir / "ask.bak").write_text("also backup")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        result = _repl_run_custom_command("!ask", str(tmp_path))
        assert result is None


class TestDiscoverCustomCommandsTextFiles:
    def test_text_file_included(self, tmp_path):
        from swival.agent import discover_custom_commands

        cmd_dir = tmp_path / "commands"
        cmd_dir.mkdir()
        (cmd_dir / "mytemplate").write_text("a prompt template")

        with patch("swival.config.global_config_dir", return_value=tmp_path):
            result = discover_custom_commands()

        assert "mytemplate" in result

    def test_multibyte_at_512_boundary_included(self, tmp_path):
        from swival.agent import discover_custom_commands

        cmd_dir = tmp_path / "commands"
        cmd_dir.mkdir()
        (cmd_dir / "emoji").write_text("a" * 511 + "\U0001f642 end")

        with patch("swival.config.global_config_dir", return_value=tmp_path):
            result = discover_custom_commands()

        assert "emoji" in result

    def test_binary_file_excluded(self, tmp_path):
        from swival.agent import discover_custom_commands

        cmd_dir = tmp_path / "commands"
        cmd_dir.mkdir()
        (cmd_dir / "binary").write_bytes(b"\x00\x01\x02binary data")

        with patch("swival.config.global_config_dir", return_value=tmp_path):
            result = discover_custom_commands()

        assert "binary" not in result

    def test_dotfile_excluded(self, tmp_path):
        from swival.agent import discover_custom_commands

        cmd_dir = tmp_path / "commands"
        cmd_dir.mkdir()
        (cmd_dir / ".hidden").write_text("hidden")

        with patch("swival.config.global_config_dir", return_value=tmp_path):
            result = discover_custom_commands()

        assert result == []

    def test_backup_excluded(self, tmp_path):
        from swival.agent import discover_custom_commands

        cmd_dir = tmp_path / "commands"
        cmd_dir.mkdir()
        (cmd_dir / "tool~").write_text("backup")
        (cmd_dir / "tool.bak").write_text("backup")

        with patch("swival.config.global_config_dir", return_value=tmp_path):
            result = discover_custom_commands()

        assert result == []


# ---------------------------------------------------------------------------
# _truncate_for_context
# ---------------------------------------------------------------------------


class TestTruncateForContext:
    def test_no_context_length_passes_through(self):
        result = _truncate_for_context("hello world", [], [], None)
        assert result == "hello world"

    def test_no_context_length_hard_cap(self):
        big_text = "x" * 200_000
        result = _truncate_for_context(big_text, [], [], None)
        assert len(result.encode()) <= 100_000

    def test_no_context_length_hard_cap_multibyte(self):
        big_text = "\U0001f642" * 100_001  # 4 bytes each
        result = _truncate_for_context(big_text, [], [], None)
        assert len(result.encode()) <= 100_000

    def test_fits_in_context(self):
        messages = [{"role": "system", "content": "you are helpful"}]
        tools = []
        result = _truncate_for_context("short text", messages, tools, 100_000)
        assert result == "short text"

    def test_truncated_when_over_budget(self):
        big_text = "word " * 50000
        messages = [{"role": "system", "content": "sys"}]
        result = _truncate_for_context(big_text, messages, [], 1000)
        assert result is not None
        assert len(result) < len(big_text)

    def test_zero_headroom_returns_none(self):
        big_sys = "x " * 50000
        messages = [{"role": "system", "content": big_sys}]
        result = _truncate_for_context("text", messages, [], 100)
        assert result is None


# ---------------------------------------------------------------------------
# REPL integration (mocked agent loop)
# ---------------------------------------------------------------------------


def _make_text_response(text):
    msg = types.SimpleNamespace(content=text, tool_calls=None, role="assistant")
    return msg, "stop"


def _repl_kwargs(tmp_path, **overrides):
    defaults = dict(
        api_base="http://127.0.0.1:1234",
        model_id="test-model",
        max_turns=5,
        max_output_tokens=1024,
        temperature=0.5,
        top_p=None,
        seed=None,
        context_length=100_000,
        base_dir=str(tmp_path),
        thinking_state=ThinkingState(verbose=False),
        todo_state=TodoState(verbose=False),
        snapshot_state=SnapshotState(),
        resolved_commands={},
        skills_catalog={},
        skill_read_roots=[],
        extra_write_roots=[],
        files_mode="all",
        verbose=False,
        llm_kwargs={"provider": "lmstudio", "api_key": None},
        no_history=True,
    )
    defaults.update(overrides)
    return defaults


def _patch_session(inputs):
    mock_session = MagicMock()
    side = []
    for v in inputs:
        if v is EOFError:
            side.append(EOFError())
        elif v is KeyboardInterrupt:
            side.append(KeyboardInterrupt())
        else:
            side.append(v)
    mock_session.prompt.side_effect = side
    return patch("prompt_toolkit.PromptSession", return_value=mock_session)


class TestReplCustomCommandIntegration:
    """Test the custom command branch inside repl_loop via mocking."""

    def test_history_vs_prompt(self, tmp_path, monkeypatch):
        """run_agent_loop receives raw stdout; append_history receives [!cmd] form."""
        cmd_dir = _commands_dir(tmp_path)
        _make_script(cmd_dir / "info", '#!/bin/sh\necho "some info"')
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        messages = [{"role": "system", "content": "sys"}]
        agent_loop_calls = []

        def mock_agent_loop(msgs, tools, **kwargs):
            agent_loop_calls.append(list(msgs))
            return "the answer", False

        with (
            _patch_session(["!info", "/exit"]),
            patch("swival.agent.run_agent_loop", side_effect=mock_agent_loop),
            patch("swival.agent.append_history") as mock_history,
        ):
            repl_loop(messages, [], **_repl_kwargs(tmp_path, no_history=False))

        assert len(agent_loop_calls) == 1
        last_user_msg = agent_loop_calls[0][-1]
        assert last_user_msg["content"] == "some info"

        mock_history.assert_called_once()
        history_question = mock_history.call_args[0][1]
        assert history_question == "[!info] !info"

    def test_exhausted_turns(self, tmp_path, monkeypatch, capsys):
        cmd_dir = _commands_dir(tmp_path)
        _make_script(cmd_dir / "info", '#!/bin/sh\necho "data"')
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        messages = [{"role": "system", "content": "sys"}]

        def mock_agent_loop(msgs, tools, **kwargs):
            return "answer", True

        with (
            _patch_session(["!info", "/exit"]),
            patch("swival.agent.run_agent_loop", side_effect=mock_agent_loop),
        ):
            repl_loop(messages, [], **_repl_kwargs(tmp_path, verbose=True))

        captured = capsys.readouterr()
        assert "max turns" in captured.err

    def test_keyboard_interrupt(self, tmp_path, monkeypatch):
        cmd_dir = _commands_dir(tmp_path)
        _make_script(cmd_dir / "info", '#!/bin/sh\necho "data"')
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        messages = [{"role": "system", "content": "sys"}]

        def mock_agent_loop(msgs, tools, **kwargs):
            raise KeyboardInterrupt

        with (
            _patch_session(["!info", "/exit"]),
            patch("swival.agent.run_agent_loop", side_effect=mock_agent_loop),
        ):
            repl_loop(messages, [], **_repl_kwargs(tmp_path, continue_here=False))

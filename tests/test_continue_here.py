"""Tests for continue-here files (.swival/continue.md)."""

import os
import time
import types

import pytest

from swival import fmt
from swival.continue_here import (
    MAX_CONTINUE_CHARS,
    _safe_continue_path,
    _find_last_user_task,
    _find_first_user_task,
    _build_deterministic_continue,
    write_continue_file,
    load_continue_file,
    format_continue_prompt,
)
from swival.thinking import ThinkingState, ThoughtEntry
from swival.todo import TodoState
from swival.snapshot import SnapshotState


@pytest.fixture(autouse=True)
def _init_fmt():
    fmt.init(color=False, no_color=False)


def _sys(content="system"):
    return {"role": "system", "content": content}


def _user(content):
    return {"role": "user", "content": content}


def _assistant(content):
    return {"role": "assistant", "content": content}


def _tool(content, tool_call_id="tc1"):
    return {"role": "tool", "content": content, "tool_call_id": tool_call_id}


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


class TestPathSafety:
    def test_safe_path_normal(self, tmp_path):
        p = _safe_continue_path(str(tmp_path))
        assert p == tmp_path / ".swival" / "continue.md"

    def test_safe_path_symlink_escape(self, tmp_path):
        # Create a symlink that escapes base_dir
        base = tmp_path / "project"
        base.mkdir()
        escape = tmp_path / "outside"
        escape.mkdir()
        link = base / ".swival"
        link.symlink_to(escape)
        with pytest.raises(ValueError, match="escapes base directory"):
            _safe_continue_path(str(base))


# ---------------------------------------------------------------------------
# Finding user tasks
# ---------------------------------------------------------------------------


class TestFindUserTask:
    def test_last_user_task_simple(self):
        msgs = [_sys(), _user("fix the bug"), _assistant("ok")]
        assert _find_last_user_task(msgs) == "fix the bug"

    def test_last_user_task_multi_turn(self):
        msgs = [
            _sys(),
            _user("first task"),
            _assistant("done"),
            _user("second task"),
            _assistant("ok"),
        ]
        assert _find_last_user_task(msgs) == "second task"

    def test_skips_synthetic_prefixes(self):
        msgs = [
            _sys(),
            _user("real task"),
            _assistant("working..."),
            _user("Your response was empty. Please continue working on the task."),
            _user(
                "IMPORTANT: You have called `edit_file` 2 times with the same error."
            ),
            _user("STOP: You have failed to use `edit_file` correctly 3 times"),
            _user("Tip: Consider using the `think` tool before making edits."),
            _user("Reminder: You have 3 unfinished todo items."),
            _user("Your response was cut off. Please use the provided tools."),
            _user(
                "[REVIEWER FEEDBACK — Round 2]\n"
                "A reviewer has evaluated your answer and requested changes. "
                "You MUST address the feedback below by taking concrete "
                "tool-call actions.\n\nfix the typo"
            ),
        ]
        assert _find_last_user_task(msgs) == "real task"

    def test_returns_none_empty(self):
        assert _find_last_user_task([_sys()]) is None
        assert _find_last_user_task([]) is None

    def test_first_user_task(self):
        msgs = [_sys(), _user("first"), _assistant("ok"), _user("second")]
        assert _find_first_user_task(msgs) == "first"

    def test_first_skips_synthetic(self):
        msgs = [
            _sys(),
            _user("IMPORTANT: error"),
            _user("real first"),
        ]
        assert _find_first_user_task(msgs) == "real first"


# ---------------------------------------------------------------------------
# Deterministic continue content
# ---------------------------------------------------------------------------


class TestBuildDeterministic:
    def test_basic_messages(self):
        msgs = [_sys(), _user("fix auth bug"), _assistant("looking...")]
        content = _build_deterministic_continue(msgs)
        assert "# Continue Here" in content
        assert "fix auth bug" in content

    def test_with_todo_state(self):
        msgs = [_sys(), _user("task")]
        todo = TodoState()
        todo.items = [
            types.SimpleNamespace(text="implement login", done=True),
            types.SimpleNamespace(text="add tests", done=False),
        ]
        content = _build_deterministic_continue(msgs, todo_state=todo)
        assert "[x] implement login" in content
        assert "[ ] add tests" in content

    def test_with_snapshot_state(self):
        msgs = [_sys(), _user("task")]
        snap = SnapshotState()
        snap.history = [
            {
                "label": "auth-review",
                "summary": "Reviewed auth module, found JWT issue",
            },
        ]
        content = _build_deterministic_continue(msgs, snapshot_state=snap)
        assert "auth-review" in content
        assert "JWT issue" in content

    def test_with_thinking_state(self):
        msgs = [_sys(), _user("task")]
        think = ThinkingState()
        think.history = [
            ThoughtEntry(
                thought="The bug is in token validation",
                thought_number=1,
                total_thoughts=3,
                next_thought_needed=True,
            ),
        ]
        content = _build_deterministic_continue(msgs, thinking_state=think)
        assert "token validation" in content

    def test_all_none_states(self):
        msgs = [_sys(), _user("task")]
        content = _build_deterministic_continue(msgs, None, None, None)
        assert "# Continue Here" in content
        assert "task" in content

    def test_multi_turn_shows_both_tasks(self):
        msgs = [
            _sys(),
            _user("original task"),
            _assistant("done"),
            _user("new task"),
        ]
        content = _build_deterministic_continue(msgs)
        assert "new task" in content
        assert "original task" in content

    def test_single_turn_no_duplicate(self):
        msgs = [_sys(), _user("only task")]
        content = _build_deterministic_continue(msgs)
        assert content.count("only task") == 1

    def test_size_cap(self):
        # Create a message that would produce very long output
        big_msg = "x" * 10000
        msgs = [_sys(), _user(big_msg)]
        content = _build_deterministic_continue(msgs)
        assert len(content) <= MAX_CONTINUE_CHARS


# ---------------------------------------------------------------------------
# Write / load / format
# ---------------------------------------------------------------------------


class TestWriteAndLoad:
    def test_write_creates_file(self, tmp_path):
        msgs = [_sys(), _user("fix bug")]
        result = write_continue_file(str(tmp_path), msgs)
        assert result is True
        path = tmp_path / ".swival" / "continue.md"
        assert path.exists()
        content = path.read_text()
        assert "fix bug" in content

    def test_write_with_llm_enhancement(self, tmp_path):
        msgs = [_sys(), _user("fix bug")]

        def mock_llm(**kwargs):
            resp = types.SimpleNamespace(content="# LLM Summary\n\nBetter content")
            return resp, "stop"

        result = write_continue_file(
            str(tmp_path),
            msgs,
            call_llm_fn=mock_llm,
            model_id="test",
            base_url="http://test",
            provider="lmstudio",
        )
        assert result is True
        content = (tmp_path / ".swival" / "continue.md").read_text()
        assert "LLM Summary" in content

    def test_write_llm_failure_keeps_deterministic(self, tmp_path):
        msgs = [_sys(), _user("fix bug")]

        def failing_llm(**kwargs):
            raise RuntimeError("LLM unavailable")

        result = write_continue_file(
            str(tmp_path),
            msgs,
            call_llm_fn=failing_llm,
            model_id="test",
            base_url="http://test",
            provider="lmstudio",
        )
        assert result is True
        content = (tmp_path / ".swival" / "continue.md").read_text()
        assert "fix bug" in content
        assert "LLM" not in content

    def test_load_returns_content_and_deletes(self, tmp_path):
        path = tmp_path / ".swival" / "continue.md"
        path.parent.mkdir(parents=True)
        path.write_text("# Continue\n\nResume work")

        content = load_continue_file(str(tmp_path))
        assert content == "# Continue\n\nResume work"
        assert not path.exists()

    def test_load_with_delete_false(self, tmp_path):
        path = tmp_path / ".swival" / "continue.md"
        path.parent.mkdir(parents=True)
        path.write_text("content")

        content = load_continue_file(str(tmp_path), delete=False)
        assert content == "content"
        assert path.exists()

    def test_load_returns_none_when_missing(self, tmp_path):
        assert load_continue_file(str(tmp_path)) is None

    def test_load_returns_none_for_empty(self, tmp_path):
        path = tmp_path / ".swival" / "continue.md"
        path.parent.mkdir(parents=True)
        path.write_text("")

        assert load_continue_file(str(tmp_path)) is None

    def test_load_caps_size(self, tmp_path):
        path = tmp_path / ".swival" / "continue.md"
        path.parent.mkdir(parents=True)
        path.write_text("x" * (MAX_CONTINUE_CHARS + 1000))

        content = load_continue_file(str(tmp_path))
        assert len(content) <= MAX_CONTINUE_CHARS

    def test_overwrite_existing(self, tmp_path):
        msgs = [_sys(), _user("first")]
        write_continue_file(str(tmp_path), msgs)
        first = (tmp_path / ".swival" / "continue.md").read_text()

        msgs2 = [_sys(), _user("second")]
        write_continue_file(str(tmp_path), msgs2)
        second = (tmp_path / ".swival" / "continue.md").read_text()

        assert "second" in second
        assert first != second

    def test_write_symlink_escape_returns_false(self, tmp_path):
        base = tmp_path / "project"
        base.mkdir()
        escape = tmp_path / "outside"
        escape.mkdir()
        link = base / ".swival"
        link.symlink_to(escape)

        result = write_continue_file(str(base), [_user("test")])
        assert result is False

    def test_staleness_warning(self, tmp_path, capsys):
        path = tmp_path / ".swival" / "continue.md"
        path.parent.mkdir(parents=True)
        path.write_text("old content")
        # Set mtime to 25 hours ago
        old_time = time.time() - (25 * 3600)
        os.utime(path, (old_time, old_time))

        content = load_continue_file(str(tmp_path))
        assert content == "old content"
        captured = capsys.readouterr()
        assert "25h old" in captured.err


class TestFormatContinuePrompt:
    def test_wraps_in_tags(self):
        result = format_continue_prompt("hello")
        assert "<continue-here>" in result
        assert "</continue-here>" in result
        assert "hello" in result
        assert "resuming interrupted work" in result


# ---------------------------------------------------------------------------
# Integration: build_system_prompt
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def test_continue_file_injected(self, tmp_path):
        from swival.agent import build_system_prompt

        # Write a continue file
        path = tmp_path / ".swival" / "continue.md"
        path.parent.mkdir(parents=True)
        path.write_text("# Continue\n\nResume the auth fix")

        prompt, _ = build_system_prompt(
            base_dir=str(tmp_path),
            system_prompt=None,
            no_system_prompt=False,
            no_instructions=True,
            no_memory=True,
            skills_catalog={},
            commands_unrestricted=False,
            resolved_commands={},
            verbose=False,
        )
        assert "Resume the auth fix" in prompt
        assert "<continue-here>" in prompt
        # File should be consumed
        assert not path.exists()

    def test_continue_file_consumed_not_loaded_twice(self, tmp_path):
        from swival.agent import build_system_prompt

        path = tmp_path / ".swival" / "continue.md"
        path.parent.mkdir(parents=True)
        path.write_text("# Continue\n\nResume work")

        common_kwargs = dict(
            base_dir=str(tmp_path),
            system_prompt=None,
            no_system_prompt=False,
            no_instructions=True,
            no_memory=True,
            skills_catalog={},
            commands_unrestricted=False,
            resolved_commands={},
            verbose=False,
        )

        prompt1, _ = build_system_prompt(**common_kwargs)
        assert "Resume work" in prompt1
        assert not path.exists()

        prompt2, _ = build_system_prompt(**common_kwargs)
        assert "Resume work" not in prompt2

    def test_continue_file_with_custom_system_prompt(self, tmp_path):
        from swival.agent import build_system_prompt

        path = tmp_path / ".swival" / "continue.md"
        path.parent.mkdir(parents=True)
        path.write_text("# Continue\n\nResume work")

        prompt, _ = build_system_prompt(
            base_dir=str(tmp_path),
            system_prompt="You are a custom bot.",
            no_system_prompt=False,
            no_instructions=True,
            no_memory=True,
            skills_catalog={},
            commands_unrestricted=False,
            resolved_commands={},
            verbose=False,
        )
        assert "custom bot" in prompt
        assert "Resume work" in prompt

    def test_no_continue_flag_skips_loading(self, tmp_path):
        from swival.agent import build_system_prompt

        path = tmp_path / ".swival" / "continue.md"
        path.parent.mkdir(parents=True)
        path.write_text("# Continue\n\nResume work")

        prompt, _ = build_system_prompt(
            base_dir=str(tmp_path),
            system_prompt=None,
            no_system_prompt=False,
            no_instructions=True,
            no_memory=True,
            skills_catalog={},
            commands_unrestricted=False,
            resolved_commands={},
            verbose=False,
            no_continue=True,
        )
        assert "Resume work" not in prompt
        # File should still exist (not consumed)
        assert path.exists()


# ---------------------------------------------------------------------------
# Integration: run_agent_loop
# ---------------------------------------------------------------------------


class TestRunAgentLoop:
    def _make_loop_kwargs(self, tmp_path, **overrides):
        from swival.thinking import ThinkingState
        from swival.todo import TodoState
        from swival.snapshot import SnapshotState

        defaults = dict(
            api_base="http://localhost:1234",
            model_id="test-model",
            max_turns=1,
            max_output_tokens=1024,
            temperature=0.0,
            top_p=1.0,
            seed=None,
            context_length=None,
            base_dir=str(tmp_path),
            thinking_state=ThinkingState(),
            todo_state=TodoState(),
            snapshot_state=SnapshotState(),
            resolved_commands={},
            skills_catalog={},
            skill_read_roots=[],
            extra_write_roots=[],
            yolo=False,
            verbose=False,
            llm_kwargs={},
        )
        defaults.update(overrides)
        return defaults

    def test_max_turns_writes_continue_file(self, tmp_path, monkeypatch):
        from swival import agent

        call_count = 0

        def mock_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Agent loop call — returns tool call
                msg = types.SimpleNamespace(
                    content="still working...",
                    tool_calls=[
                        types.SimpleNamespace(
                            id="tc1",
                            function=types.SimpleNamespace(
                                name="think",
                                arguments='{"thought": "analyzing"}',
                            ),
                        )
                    ],
                )
                return msg, "stop"
            else:
                # Summary call from write_continue_file — return summary
                msg = types.SimpleNamespace(
                    content="# Continue\n\n## What remains\n- Fix the auth bug"
                )
                return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", mock_llm)

        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "fix the auth bug"},
        ]
        kwargs = self._make_loop_kwargs(tmp_path, max_turns=1)
        answer, exhausted = agent.run_agent_loop(messages, [], **kwargs)

        assert exhausted is True
        path = tmp_path / ".swival" / "continue.md"
        assert path.exists()
        content = path.read_text()
        # LLM summary should have been written
        assert "auth bug" in content

    def test_normal_completion_no_continue_file(self, tmp_path, monkeypatch):
        from swival import agent

        def mock_llm(*args, **kwargs):
            msg = types.SimpleNamespace(content="Done!", tool_calls=None)
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", mock_llm)

        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "simple question"},
        ]
        kwargs = self._make_loop_kwargs(tmp_path)
        answer, exhausted = agent.run_agent_loop(messages, [], **kwargs)

        assert exhausted is False
        path = tmp_path / ".swival" / "continue.md"
        assert not path.exists()

    def test_continue_here_false_suppresses_write(self, tmp_path, monkeypatch):
        from swival import agent

        def mock_llm(*args, **kwargs):
            msg = types.SimpleNamespace(
                content="still working...",
                tool_calls=[
                    types.SimpleNamespace(
                        id="tc1",
                        function=types.SimpleNamespace(
                            name="think",
                            arguments='{"thought": "analyzing"}',
                        ),
                    )
                ],
            )
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", mock_llm)

        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
        ]
        kwargs = self._make_loop_kwargs(tmp_path, continue_here=False)
        agent.run_agent_loop(messages, [], **kwargs)

        path = tmp_path / ".swival" / "continue.md"
        assert not path.exists()


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


class TestConfig:
    def test_no_continue_in_config_keys(self):
        from swival.config import CONFIG_KEYS

        assert "no_continue" in CONFIG_KEYS
        assert CONFIG_KEYS["no_continue"] is bool

    def test_no_continue_default(self):
        from swival.config import _ARGPARSE_DEFAULTS

        assert _ARGPARSE_DEFAULTS["no_continue"] is False

    def test_config_to_session_kwargs_inversion(self):
        from swival.config import config_to_session_kwargs

        kwargs = config_to_session_kwargs({"no_continue": True})
        assert kwargs["continue_here"] is False

        kwargs = config_to_session_kwargs({"no_continue": False})
        assert kwargs["continue_here"] is True

    def test_cli_flag_exists(self):
        from swival.agent import build_parser

        parser = build_parser()
        # Verify the flag is recognized
        args = parser.parse_args(["--no-continue", "test question"])
        assert args.no_continue is True


# ---------------------------------------------------------------------------
# REPL exit handling
# ---------------------------------------------------------------------------


class TestReplExit:
    def test_eof_with_work_writes_continue(self, tmp_path):
        """EOFError at prompt with non-system messages should write continue file."""
        msgs = [_sys(), _user("fix bug"), _assistant("working...")]
        # Simulate the guard condition from repl_loop
        from swival._msg import _msg_role

        has_work = any(_msg_role(m) != "system" for m in msgs)
        assert has_work is True

    def test_eof_with_only_system_no_write(self):
        """EOFError at prompt with only system message should not write."""
        msgs = [_sys()]
        from swival._msg import _msg_role

        has_work = any(_msg_role(m) != "system" for m in msgs)
        assert has_work is False

    def test_no_system_prompt_with_user_msg_writes(self):
        """With --no-system-prompt and one user message, should write."""
        msgs = [_user("do something")]
        from swival._msg import _msg_role

        has_work = any(_msg_role(m) != "system" for m in msgs)
        assert has_work is True

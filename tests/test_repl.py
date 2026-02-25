"""Tests for REPL mode: argument parsing, run_agent_loop, and repl_loop."""

from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from swival.agent import (
    build_parser,
    run_agent_loop,
    repl_loop,
    ContextOverflowError,
    _repl_help,
    _repl_clear,
    _repl_add_dir,
    _repl_compact,
)
from swival.thinking import ThinkingState
from swival.tools import dispatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sys(content):
    return {"role": "system", "content": content}


def _user(content):
    return {"role": "user", "content": content}


def _make_text_response(text):
    """Create a (message, finish_reason) tuple for a plain text response."""
    msg = SimpleNamespace(content=text, tool_calls=None, role="assistant")
    return msg, "stop"


def _make_tool_response(tool_calls, content=None):
    """Create a (message, finish_reason) tuple for a tool-call response."""
    tcs = [
        SimpleNamespace(
            id=tc_id,
            function=SimpleNamespace(name=name, arguments=args),
        )
        for tc_id, name, args in tool_calls
    ]
    msg = SimpleNamespace(content=content, tool_calls=tcs, role="assistant")
    return msg, "stop"


def _loop_kwargs(tmp_path, **overrides):
    """Build minimal kwargs for run_agent_loop / repl_loop."""
    defaults = dict(
        api_base="http://127.0.0.1:1234",
        model_id="test-model",
        max_turns=5,
        max_output_tokens=1024,
        temperature=0.5,
        top_p=1.0,
        seed=None,
        context_length=None,
        base_dir=str(tmp_path),
        thinking_state=ThinkingState(verbose=False, notes_dir=str(tmp_path)),
        resolved_commands={},
        skills_catalog={},
        skill_read_roots=[],
        extra_write_roots=[],
        yolo=False,
        verbose=False,
        llm_kwargs={"provider": "lmstudio", "api_key": None},
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestArgumentParsing:
    def test_question_optional_with_repl(self):
        parser = build_parser()
        args = parser.parse_args(["--repl"])
        assert args.repl is True
        assert args.question is None

    def test_question_required_without_repl(self):
        """Without --repl and no question, main() should call parser.error()."""
        # The parser itself accepts question as nargs="?", so parse_args([])
        # succeeds. Validation happens in main().
        with patch("swival.agent.build_parser") as mock_bp:
            mock_parser = MagicMock()
            mock_args = SimpleNamespace(
                question=None,
                repl=False,
                quiet=False,
                verbose=True,
                color=False,
                no_color=False,
            )
            mock_parser.parse_args.return_value = mock_args
            mock_parser.error.side_effect = SystemExit(2)
            mock_bp.return_value = mock_parser

            from swival.agent import main

            with pytest.raises(SystemExit):
                main()
            mock_parser.error.assert_called_once()

    def test_question_with_repl(self):
        parser = build_parser()
        args = parser.parse_args(["--repl", "initial question"])
        assert args.repl is True
        assert args.question == "initial question"


# ---------------------------------------------------------------------------
# run_agent_loop
# ---------------------------------------------------------------------------


class TestRunAgentLoop:
    def test_returns_answer(self, tmp_path):
        """A text-only LLM response returns (answer, False)."""
        messages = [_sys("system"), _user("hello")]
        original_len = len(messages)

        with patch(
            "swival.agent.call_llm", return_value=_make_text_response("the answer")
        ):
            answer, exhausted = run_agent_loop(messages, [], **_loop_kwargs(tmp_path))

        assert answer == "the answer"
        assert exhausted is False
        # Assistant message should be appended
        assert len(messages) > original_len

    def test_max_turns_exhausted(self, tmp_path):
        """When only tool calls come back, returns (last_text, True) after max_turns."""
        messages = [_sys("system"), _user("hello")]

        call_count = 0

        def fake_call_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Return a tool call every time, with some content on the last one
            content = f"thinking step {call_count}" if call_count == 2 else None
            return _make_tool_response(
                [("tc1", "read_file", '{"path": "x.txt"}')],
                content=content,
            )

        with (
            patch("swival.agent.call_llm", side_effect=fake_call_llm),
            patch(
                "swival.agent.handle_tool_call",
                return_value={
                    "role": "tool",
                    "tool_call_id": "tc1",
                    "content": "file contents",
                },
            ),
        ):
            answer, exhausted = run_agent_loop(
                messages, [], **_loop_kwargs(tmp_path, max_turns=2)
            )

        assert exhausted is True
        assert answer == "thinking step 2"

    def test_max_turns_no_text(self, tmp_path):
        """When max_turns exhausted and no assistant text, returns (None, True)."""
        messages = [_sys("system"), _user("hello")]

        with (
            patch(
                "swival.agent.call_llm",
                return_value=_make_tool_response(
                    [("tc1", "read_file", '{"path": "x.txt"}')]
                ),
            ),
            patch(
                "swival.agent.handle_tool_call",
                return_value={"role": "tool", "tool_call_id": "tc1", "content": "ok"},
            ),
        ):
            answer, exhausted = run_agent_loop(
                messages, [], **_loop_kwargs(tmp_path, max_turns=1)
            )

        assert exhausted is True
        assert answer is None


# ---------------------------------------------------------------------------
# repl_loop
# ---------------------------------------------------------------------------


class TestReplLoop:
    def _mock_session(self, inputs):
        """Create a mock PromptSession whose .prompt() returns values from inputs."""
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
        return mock_session

    def _patch_session(self, tmp_path, inputs):
        """Return a patch context that replaces PromptSession with a mock."""
        mock_session = self._mock_session(inputs)
        return patch(
            "prompt_toolkit.PromptSession",
            return_value=mock_session,
        )

    def test_exit_command(self, tmp_path):
        """Feeding /exit exits the loop without error."""
        messages = [_sys("system")]
        with self._patch_session(tmp_path, ["/exit"]):
            repl_loop(messages, [], **_loop_kwargs(tmp_path))
        # Only the system message should remain (no user message added for /exit)
        assert len(messages) == 1

    def test_quit_command(self, tmp_path):
        """Feeding /quit exits the loop without error."""
        messages = [_sys("system")]
        with self._patch_session(tmp_path, ["/quit"]):
            repl_loop(messages, [], **_loop_kwargs(tmp_path))
        assert len(messages) == 1

    def test_eof(self, tmp_path):
        """EOF (Ctrl-D) exits the loop cleanly."""
        messages = [_sys("system")]
        with self._patch_session(tmp_path, [EOFError]):
            repl_loop(messages, [], **_loop_kwargs(tmp_path))
        assert len(messages) == 1

    def test_empty_lines_ignored(self, tmp_path):
        """Empty lines don't trigger run_agent_loop calls."""
        messages = [_sys("system")]
        inputs = ["", "", "", "hello", "/exit"]

        with (
            self._patch_session(tmp_path, inputs),
            patch(
                "swival.agent.run_agent_loop", return_value=("answer", False)
            ) as mock_loop,
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path))

        # Only one call for "hello"
        assert mock_loop.call_count == 1

    def test_message_history_persists(self, tmp_path):
        """Second question sees messages from first question in history."""
        messages = [_sys("system")]

        call_messages = []

        def fake_run(msgs, tools, **kwargs):
            # Record snapshot of messages at call time
            call_messages.append(list(msgs))
            return ("answer", False)

        inputs = ["first question", "second question", "/exit"]
        with (
            self._patch_session(tmp_path, inputs),
            patch("swival.agent.run_agent_loop", side_effect=fake_run),
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path))

        # First call: system + "first question"
        assert len(call_messages[0]) == 2
        assert call_messages[0][1]["content"] == "first question"
        # Second call: system + "first question" + "second question"
        assert len(call_messages[1]) == 3
        assert call_messages[1][2]["content"] == "second question"

    def test_ctrl_c_during_loop(self, tmp_path):
        """KeyboardInterrupt during run_agent_loop doesn't crash the REPL."""
        messages = [_sys("system")]

        call_count = 0

        def fake_run(msgs, tools, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise KeyboardInterrupt
            return ("answer", False)

        inputs = ["interrupted", "ok", "/exit"]
        with (
            self._patch_session(tmp_path, inputs),
            patch("swival.agent.run_agent_loop", side_effect=fake_run),
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path))

        # Both questions were attempted
        assert call_count == 2

    def test_answer_on_stdout_not_stderr(self, tmp_path, capsys):
        """The answer appears on stdout, not stderr."""
        messages = [_sys("system")]

        inputs = ["hello", "/exit"]
        with (
            self._patch_session(tmp_path, inputs),
            patch("swival.agent.run_agent_loop", return_value=("answer", False)),
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path))

        captured = capsys.readouterr()
        assert "answer" in captured.out


# ---------------------------------------------------------------------------
# Compaction preserves list identity
# ---------------------------------------------------------------------------


class TestCompactionListIdentity:
    def test_compaction_preserves_list_identity(self, tmp_path):
        """messages[:] = ... preserves the list object identity through compaction."""
        messages = [_sys("system"), _user("hello")]
        original_id = id(messages)

        call_count = 0

        def fake_call_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ContextOverflowError("overflow")
            return _make_text_response("recovered")

        with patch("swival.agent.call_llm", side_effect=fake_call_llm):
            answer, exhausted = run_agent_loop(messages, [], **_loop_kwargs(tmp_path))

        # The list object should be the same reference
        assert id(messages) == original_id
        assert answer == "recovered"
        assert exhausted is False


# ---------------------------------------------------------------------------
# /help command
# ---------------------------------------------------------------------------


class TestHelpCommand:
    def test_help_prints_commands(self, capsys):
        """'/help' prints the command list via fmt.info()."""
        _repl_help()
        captured = capsys.readouterr()
        assert "/help" in captured.err
        assert "/clear" in captured.err
        assert "/compact" in captured.err
        assert "/add-dir" in captured.err
        assert "/exit" in captured.err

    def test_help_in_repl(self, tmp_path):
        """/help does not append to messages or call the model."""
        messages = [_sys("system")]
        mock_session = MagicMock()
        mock_session.prompt.side_effect = ["/help", "/exit"]

        with (
            patch("prompt_toolkit.PromptSession", return_value=mock_session),
            patch("swival.agent.run_agent_loop") as mock_loop,
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path))

        # /help should not trigger a model call
        assert mock_loop.call_count == 0
        # messages should only have the system message
        assert len(messages) == 1


# ---------------------------------------------------------------------------
# /clear command
# ---------------------------------------------------------------------------


class TestClearCommand:
    def test_clear_resets_messages(self, tmp_path):
        """After /clear, messages are reduced to just system prompt."""
        messages = [
            _sys("system"),
            _user("q1"),
            {"role": "assistant", "content": "a1"},
            _user("q2"),
        ]
        ts = ThinkingState(verbose=False, notes_dir=str(tmp_path))
        _repl_clear(messages, ts)
        assert len(messages) == 1
        assert messages[0]["role"] == "system"

    def test_clear_resets_thinking_state(self, tmp_path):
        """After /clear, ThinkingState history/branches/note_count are reset."""
        ts = ThinkingState(verbose=False, notes_dir=str(tmp_path))
        from swival.thinking import ThoughtEntry

        ts.history.append(ThoughtEntry("t", 1, 1, False))
        ts.branches["b1"] = [ThoughtEntry("t", 1, 1, False)]
        ts.note_count = 5

        messages = [_sys("system"), _user("q1")]
        _repl_clear(messages, ts)

        assert ts.history == []
        assert ts.branches == {}
        assert ts.note_count == 0

    def test_clear_deletes_notes_file(self, tmp_path):
        """After /clear, the persisted notes file is deleted."""
        swival_dir = tmp_path / ".swival"
        swival_dir.mkdir()
        notes_file = swival_dir / "notes.md"
        notes_file.write_text("some notes")

        ts = ThinkingState(verbose=False, notes_dir=str(tmp_path))
        messages = [_sys("system")]
        _repl_clear(messages, ts)

        assert not notes_file.exists()

    def test_clear_skips_symlink_escape(self, tmp_path):
        """If .swival is a symlink outside notes_dir, /clear does NOT delete target."""
        outside = tmp_path / "outside"
        outside.mkdir()
        target_notes = outside / "notes.md"
        target_notes.write_text("precious data")

        # Make .swival a symlink pointing outside tmp_path/inner
        inner = tmp_path / "inner"
        inner.mkdir()
        symlink = inner / ".swival"
        symlink.symlink_to(outside)

        # Create ThinkingState with a safe dir first, then swap notes_dir
        # to the compromised dir (ThinkingState.__init__ validates eagerly)
        ts = ThinkingState(verbose=False, notes_dir=None)
        ts.notes_dir = str(inner)
        messages = [_sys("system")]
        _repl_clear(messages, ts)

        # The target file should NOT have been deleted
        assert target_notes.exists()

    def test_clear_in_repl(self, tmp_path):
        """Full integration: /clear in REPL resets messages between questions."""
        messages = [_sys("system")]

        call_messages = []

        def fake_run(msgs, tools, **kwargs):
            call_messages.append(list(msgs))
            return ("answer", False)

        inputs = ["q1", "/clear", "q2", "/exit"]
        mock_session = MagicMock()
        mock_session.prompt.side_effect = inputs

        with (
            patch("prompt_toolkit.PromptSession", return_value=mock_session),
            patch("swival.agent.run_agent_loop", side_effect=fake_run),
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path))

        # First call: system + q1
        assert len(call_messages[0]) == 2
        # After /clear, second call: system + q2 only
        assert len(call_messages[1]) == 2
        assert call_messages[1][1]["content"] == "q2"


# ---------------------------------------------------------------------------
# /add-dir command
# ---------------------------------------------------------------------------


class TestAddDirCommand:
    def test_add_dir_valid(self, tmp_path):
        """/add-dir with a valid directory appends to extra_write_roots."""
        extra = []
        _repl_add_dir(str(tmp_path), extra)
        assert tmp_path.resolve() in extra

    def test_add_dir_missing_arg(self, capsys):
        """/add-dir with no argument prints a warning."""
        extra = []
        _repl_add_dir("", extra)
        assert extra == []
        captured = capsys.readouterr()
        assert "requires a path" in captured.err

    def test_add_dir_nonexistent(self, capsys):
        """/add-dir with nonexistent path prints a warning."""
        extra = []
        _repl_add_dir("/nonexistent_path_abc123", extra)
        assert extra == []
        captured = capsys.readouterr()
        assert "not a directory" in captured.err

    def test_add_dir_duplicate(self, tmp_path, capsys):
        """Adding same dir twice doesn't duplicate it."""
        extra = [tmp_path.resolve()]
        _repl_add_dir(str(tmp_path), extra)
        assert len(extra) == 1
        captured = capsys.readouterr()
        assert "already in whitelist" in captured.err

    def test_add_dir_root_rejected(self, capsys):
        """/add-dir / is rejected."""
        extra = []
        _repl_add_dir("/", extra)
        assert extra == []
        captured = capsys.readouterr()
        assert "filesystem root" in captured.err

    def test_add_dir_enables_file_access(self, tmp_path):
        """After /add-dir, read_file can access files in the added directory."""
        extra_dir = tmp_path / "extra"
        extra_dir.mkdir()
        test_file = extra_dir / "test.txt"
        test_file.write_text("hello from extra")

        base_dir = tmp_path / "base"
        base_dir.mkdir()

        extra_write_roots = []

        # Before /add-dir, reading should fail (outside base_dir)
        result = dispatch(
            "read_file",
            {"file_path": str(test_file)},
            base_dir=str(base_dir),
            extra_write_roots=extra_write_roots,
            skill_read_roots=[],
        )
        assert result.startswith("error:")

        # After /add-dir, reading should succeed
        _repl_add_dir(str(extra_dir), extra_write_roots)
        result = dispatch(
            "read_file",
            {"file_path": str(test_file)},
            base_dir=str(base_dir),
            extra_write_roots=extra_write_roots,
            skill_read_roots=[],
        )
        assert "hello from extra" in result


# ---------------------------------------------------------------------------
# /compact command
# ---------------------------------------------------------------------------


class TestCompactCommand:
    def test_compact_truncates_old_results(self, tmp_path, capsys):
        """Messages with large tool results get truncated."""
        big_result = "x" * 5000
        messages = [
            _sys("system"),
            _user("q1"),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "tc1", "function": {"name": "read_file", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": big_result},
            # Need more recent turns so the big result is in the "old" zone
            # (compact_messages skips the most recent 2 turns)
            _user("q2"),
            {"role": "assistant", "content": "a2"},
            _user("q3"),
            {"role": "assistant", "content": "a3"},
            _user("q4"),
            {"role": "assistant", "content": "a4"},
        ]

        _repl_compact(messages, [], None, "")

        captured = capsys.readouterr()
        assert "compacted:" in captured.err
        # The big tool result should have been truncated
        tool_msg = next(m for m in messages if m.get("role") == "tool")
        assert len(tool_msg["content"]) < len(big_result)

    def test_compact_drop_flag(self, tmp_path, capsys):
        """/compact --drop additionally drops middle turns."""
        messages = [
            _sys("system"),
            _user("q1"),
            {"role": "assistant", "content": "a1"},
            _user("q2"),
            {"role": "assistant", "content": "a2"},
            _user("q3"),
            {"role": "assistant", "content": "a3"},
            _user("q4"),
            {"role": "assistant", "content": "a4"},
            _user("q5"),
            {"role": "assistant", "content": "a5"},
        ]
        before_count = len(messages)

        _repl_compact(messages, [], None, "--drop")

        captured = capsys.readouterr()
        assert "compacted:" in captured.err
        # Should have fewer messages after dropping middle turns
        assert len(messages) < before_count


# ---------------------------------------------------------------------------
# Unknown /-prefixed input passes through
# ---------------------------------------------------------------------------


class TestUnknownSlashCommand:
    def test_unknown_slash_sent_to_model(self, tmp_path):
        """/foo bar is appended to messages and sent to the model."""
        messages = [_sys("system")]

        call_messages = []

        def fake_run(msgs, tools, **kwargs):
            call_messages.append(list(msgs))
            return ("answer", False)

        inputs = ["/foo bar", "/exit"]
        mock_session = MagicMock()
        mock_session.prompt.side_effect = inputs

        with (
            patch("prompt_toolkit.PromptSession", return_value=mock_session),
            patch("swival.agent.run_agent_loop", side_effect=fake_run),
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path))

        assert len(call_messages) == 1
        assert call_messages[0][1]["content"] == "/foo bar"

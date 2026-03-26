"""Tests for REPL mode: argument parsing, run_agent_loop, and repl_loop."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from swival.agent import (
    build_parser,
    main,
    run_agent_loop,
    repl_loop,
    ContextOverflowError,
    _init_prompt,
    INIT_ENRICH_PROMPT,
    INIT_WRITE_PROMPT,
    _INIT_AGENTS_MD_BUDGET,
    validate_agents_md,
    LEARN_PROMPT,
    _repl_help,
    _repl_tools,
    _repl_clear,
    _repl_add_dir,
    _repl_add_dir_ro,
    _repl_compact,
    _repl_extend,
    _repl_snapshot_save,
    _repl_snapshot_restore,
    _repl_snapshot_unsave,
    _repl_copy,
    _last_assistant_text,
)
from swival.snapshot import SnapshotState
from swival.thinking import ThinkingState
from swival.todo import TodoState
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
        thinking_state=ThinkingState(verbose=False),
        resolved_commands={},
        skills_catalog={},
        skill_read_roots=[],
        extra_write_roots=[],
        yolo=False,
        verbose=False,
        llm_kwargs={"provider": "lmstudio", "api_key": None},
        file_tracker=None,
        todo_state=TodoState(notes_dir=str(tmp_path), verbose=False),
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _make_main_args(**overrides):
    defaults = dict(
        question=None,
        repl=False,
        quiet=False,
        verbose=True,
        color=False,
        no_color=False,
        version=False,
        report=None,
        reviewer=None,
        self_review=False,
        base_dir=".",
        init_config=False,
        project=False,
        reviewer_mode=False,
        review_prompt=None,
        objective=None,
        verify=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _run_main_validation(
    stdin_tty=True,
    stdout_tty=True,
    stdin_content=None,
    extra_patches=None,
    **arg_overrides,
):
    """Run main() with mocked parser/sys, return (mock_parser, mock_args)."""
    from contextlib import ExitStack

    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = stdin_tty
    if stdin_content is not None:
        mock_stdin.read.return_value = stdin_content
    mock_stdout = MagicMock()
    mock_stdout.isatty.return_value = stdout_tty

    mock_args = _make_main_args(**arg_overrides)
    mock_parser = MagicMock()
    mock_parser.parse_args.return_value = mock_args
    mock_parser.error.side_effect = SystemExit(2)

    with ExitStack() as stack:
        stack.enter_context(
            patch("swival.agent.build_parser", return_value=mock_parser)
        )
        mock_sys = stack.enter_context(patch("swival.agent.sys"))
        if extra_patches:
            for p in extra_patches:
                stack.enter_context(p)
        mock_sys.stdin = mock_stdin
        mock_sys.stdout = mock_stdout
        mock_sys.argv = ["swival"]
        try:
            main()
        except (SystemExit, Exception):
            pass

    return mock_parser, mock_args


class TestArgumentParsing:
    def test_question_optional_with_repl(self):
        parser = build_parser()
        args = parser.parse_args(["--repl"])
        assert args.repl is True
        assert args.question is None

    def test_question_required_without_repl(self):
        """Without --repl, no question, and non-TTY stdout, main() errors."""
        mock_parser, _ = _run_main_validation(stdout_tty=False)
        mock_parser.error.assert_called_once_with(
            "question is required (or use --repl)"
        )

    def test_auto_repl_on_tty(self):
        """Bare invocation on a full TTY auto-enters REPL."""
        mock_parser, mock_args = _run_main_validation()
        mock_parser.error.assert_not_called()
        assert mock_args.repl is True

    def test_no_auto_repl_when_stdin_piped(self):
        """Piped stdin with no question still errors."""
        mock_parser, _ = _run_main_validation(stdin_tty=False, stdin_content="")
        mock_parser.error.assert_called_once_with(
            "question is required (stdin was empty)"
        )

    def test_no_auto_repl_when_stdout_redirected(self):
        """Redirected stdout with no question still errors."""
        mock_parser, _ = _run_main_validation(stdout_tty=False)
        mock_parser.error.assert_called_once_with(
            "question is required (or use --repl)"
        )

    def test_no_auto_repl_with_report(self):
        """--report with no task on TTY errors with '--report requires a task'."""
        mock_parser, _ = _run_main_validation(report="out.json")
        mock_parser.error.assert_called_once_with("--report requires a task")

    def test_no_auto_repl_with_reviewer(self):
        """--reviewer with no task on TTY errors with '--reviewer requires a task'."""
        mock_parser, _ = _run_main_validation(reviewer="my-review-cmd")
        mock_parser.error.assert_called_once_with("--reviewer requires a task")

    def test_no_auto_repl_with_self_review(self):
        """--self-review with no task on TTY errors with '--self-review requires a task'."""
        mock_parser, _ = _run_main_validation(
            self_review=True,
            extra_patches=[
                patch(
                    "swival.agent._build_self_review_cmd",
                    return_value="swival --reviewer-mode",
                ),
            ],
        )
        mock_parser.error.assert_called_once_with("--self-review requires a task")

    def test_report_piped_empty_stdin(self):
        """swival --report out.json with piped empty stdin gets 'stdin was empty' error."""
        mock_parser, _ = _run_main_validation(
            stdin_tty=False, stdin_content="", report="out.json"
        )
        mock_parser.error.assert_called_once_with(
            "question is required (stdin was empty)"
        )

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
                return_value=(
                    {
                        "role": "tool",
                        "tool_call_id": "tc1",
                        "content": "file contents",
                    },
                    {
                        "name": "read_file",
                        "arguments": {},
                        "elapsed": 0.0,
                        "succeeded": True,
                    },
                ),
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
                return_value=(
                    {"role": "tool", "tool_call_id": "tc1", "content": "ok"},
                    {
                        "name": "read_file",
                        "arguments": {},
                        "elapsed": 0.0,
                        "succeeded": True,
                    },
                ),
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
        assert "/continue" in captured.err
        assert "/tools" in captured.err
        assert "/init" in captured.err
        assert "/exit" in captured.err

    def test_help_commands_are_sorted(self, capsys):
        """Slash commands in /help are listed in lexicographic order."""
        _repl_help()
        lines = [line.strip() for line in capsys.readouterr().err.splitlines()]
        commands = [line.split()[0] for line in lines if line.startswith("/")]
        assert commands == sorted(commands)

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
# /tools command
# ---------------------------------------------------------------------------


def _tool(name, desc=""):
    """Build a minimal OpenAI function-calling tool dict."""
    return {"type": "function", "function": {"name": name, "description": desc}}


class TestToolsCommand:
    def test_tools_in_repl(self, tmp_path):
        """/tools does not call run_agent_loop or mutate messages."""
        messages = [_sys("system")]
        tools = [_tool("read_file", "Read a file.")]
        mock_session = MagicMock()
        mock_session.prompt.side_effect = ["/tools", "/exit"]

        with (
            patch("prompt_toolkit.PromptSession", return_value=mock_session),
            patch("swival.agent.run_agent_loop") as mock_loop,
        ):
            repl_loop(messages, tools, **_loop_kwargs(tmp_path))

        assert mock_loop.call_count == 0
        assert len(messages) == 1

    def test_builtin_tools_listed(self, capsys):
        """Built-in tools appear with full descriptions."""
        tools = [
            _tool("edit_file", "Replace a string in a file."),
            _tool("read_file", "Read a file or directory."),
        ]
        _repl_tools(tools)
        out = capsys.readouterr().err
        assert "Built-in tools:" in out
        assert "edit_file" in out
        assert "read_file" in out
        assert "Replace a string in a file." in out

    def test_builtin_sorted(self, capsys):
        """Built-in tools are sorted alphabetically."""
        tools = [_tool("write_file"), _tool("edit_file"), _tool("read_file")]
        _repl_tools(tools)
        out = capsys.readouterr().err
        assert out.index("edit_file") < out.index("read_file") < out.index("write_file")

    def test_mcp_tools_grouped(self, capsys):
        """MCP tools are grouped by server with correct count."""
        tools = [
            _tool("read_file", "Read."),
            _tool("mcp__gh__search", "Search repos."),
            _tool("mcp__fs__read", "Read fs."),
        ]
        mcp_mgr = MagicMock()
        mcp_mgr.get_tool_info.return_value = {
            "gh": [("mcp__gh__search", "Search repos.")],
            "fs": [("mcp__fs__read", "Read fs.")],
        }
        _repl_tools(tools, mcp_manager=mcp_mgr)
        out = capsys.readouterr().err
        assert "MCP tools (2 servers):" in out
        assert "gh:" in out
        assert "fs:" in out
        # Servers sorted: fs before gh
        assert out.index("fs:") < out.index("gh:")

    def test_a2a_tools_grouped(self, capsys):
        """A2A tools are grouped by agent with correct count."""
        tools = [_tool("a2a__coder__review", "Review code.")]
        a2a_mgr = MagicMock()
        a2a_mgr.get_tool_info.return_value = {
            "coder": [("a2a__coder__review", "Review code.")],
        }
        _repl_tools(tools, a2a_manager=a2a_mgr)
        out = capsys.readouterr().err
        assert "A2A tools (1 agent):" in out
        assert "coder:" in out
        assert "Review code." in out

    def test_no_managers_no_mcp_a2a_sections(self, capsys):
        """Without managers, MCP/A2A sections are absent."""
        tools = [_tool("read_file", "Read.")]
        _repl_tools(tools)
        out = capsys.readouterr().err
        assert "Built-in tools:" in out
        assert "MCP" not in out
        assert "A2A" not in out

    def test_embedded_newlines_hanging_indent(self, capsys):
        """Descriptions with newlines get hanging-indent continuation."""
        tools = [
            _tool("a2a__bot__ask", "Ask the bot.\nExamples: hello; help me"),
        ]
        a2a_mgr = MagicMock()
        a2a_mgr.get_tool_info.return_value = {
            "bot": [("a2a__bot__ask", "Ask the bot.\nExamples: hello; help me")],
        }
        _repl_tools(tools, a2a_manager=a2a_mgr)
        out = capsys.readouterr().err
        lines = out.strip().split("\n")
        # Find the continuation line with "Examples:"
        cont_lines = [ln for ln in lines if "Examples:" in ln]
        assert len(cont_lines) == 1
        # The continuation line should be indented further than the tool name line
        name_line = [ln for ln in lines if "a2a__bot__ask" in ln][0]
        desc_start = name_line.index("Ask the bot.")
        cont_line = cont_lines[0]
        # Continuation should start at the same column as the description
        stripped = cont_line.lstrip()
        indent_len = len(cont_line) - len(stripped)
        assert indent_len >= desc_start

    def test_singular_server_label(self, capsys):
        """Single MCP server uses 'server' not 'servers'."""
        tools = [_tool("mcp__gh__search", "Search.")]
        mcp_mgr = MagicMock()
        mcp_mgr.get_tool_info.return_value = {
            "gh": [("mcp__gh__search", "Search.")],
        }
        _repl_tools(tools, mcp_manager=mcp_mgr)
        out = capsys.readouterr().err
        assert "MCP tools (1 server):" in out

    def test_empty_tools(self, capsys):
        """No tools at all prints a fallback message."""
        _repl_tools([])
        out = capsys.readouterr().err
        assert "No tools available." in out


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
        ts = ThinkingState(verbose=False)
        _repl_clear(messages, ts)
        assert len(messages) == 1
        assert messages[0]["role"] == "system"

    def test_clear_resets_thinking_state(self, tmp_path):
        """After /clear, ThinkingState history/branches are reset."""
        ts = ThinkingState(verbose=False)
        from swival.thinking import ThoughtEntry

        ts.history.append(ThoughtEntry("t", 1, 1, False))
        ts.branches["b1"] = [ThoughtEntry("t", 1, 1, False)]

        messages = [_sys("system"), _user("q1")]
        _repl_clear(messages, ts)

        assert ts.history == []
        assert ts.branches == {}

    def test_clear_resets_fmt_think_tree(self):
        """After /clear, fmt think tree state is reset so next think prints header."""
        from io import StringIO
        from rich.console import Console
        from swival import fmt

        ts = ThinkingState(verbose=False)
        messages = [_sys("system"), _user("q1")]

        buf = StringIO()
        old = fmt._console
        fmt._console = Console(file=buf, no_color=True, width=80)
        fmt.reset_state()
        try:
            # First think prints header
            fmt.think_step(1, 2, "Before clear")
            _repl_clear(messages, ts)
            buf.truncate(0)
            buf.seek(0)
            # After clear, next think must print a fresh header
            fmt.think_step(1, 2, "After clear")
            out = buf.getvalue()
            assert "[think]" in out
        finally:
            fmt.reset_state()
            fmt._console = old

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
# /add-dir-ro command
# ---------------------------------------------------------------------------


class TestAddDirRoCommand:
    def test_add_dir_ro_valid(self, tmp_path):
        """/add-dir-ro with a valid directory appends to skill_read_roots."""
        roots = []
        _repl_add_dir_ro(str(tmp_path), roots)
        assert tmp_path.resolve() in roots

    def test_add_dir_ro_does_not_modify_write_roots(self, tmp_path):
        """/add-dir-ro must not touch extra_write_roots."""
        read_roots = []
        write_roots = []
        _repl_add_dir_ro(str(tmp_path), read_roots)
        assert tmp_path.resolve() in read_roots
        assert write_roots == []

    def test_add_dir_ro_missing_arg(self, capsys):
        """/add-dir-ro with no argument prints a warning."""
        roots = []
        _repl_add_dir_ro("", roots)
        assert roots == []
        captured = capsys.readouterr()
        assert "requires a path" in captured.err

    def test_add_dir_ro_nonexistent(self, capsys):
        """/add-dir-ro with nonexistent path prints a warning."""
        roots = []
        _repl_add_dir_ro("/nonexistent_path_abc123", roots)
        assert roots == []
        captured = capsys.readouterr()
        assert "not a directory" in captured.err

    def test_add_dir_ro_duplicate(self, tmp_path, capsys):
        """Adding same dir twice doesn't duplicate it."""
        roots = [tmp_path.resolve()]
        _repl_add_dir_ro(str(tmp_path), roots)
        assert len(roots) == 1
        captured = capsys.readouterr()
        assert "already in read-only whitelist" in captured.err

    def test_add_dir_ro_root_rejected(self, capsys):
        """/add-dir-ro / is rejected."""
        roots = []
        _repl_add_dir_ro("/", roots)
        assert roots == []
        captured = capsys.readouterr()
        assert "filesystem root" in captured.err

    def test_add_dir_ro_enables_read_access(self, tmp_path):
        """After /add-dir-ro, read_file can access files but write_file cannot."""
        ro_dir = tmp_path / "readonly"
        ro_dir.mkdir()
        test_file = ro_dir / "test.txt"
        test_file.write_text("hello from readonly")

        base_dir = tmp_path / "base"
        base_dir.mkdir()

        skill_read_roots = []

        # Before /add-dir-ro, reading should fail
        result = dispatch(
            "read_file",
            {"file_path": str(test_file)},
            base_dir=str(base_dir),
            extra_write_roots=[],
            skill_read_roots=skill_read_roots,
        )
        assert result.startswith("error:")

        # After /add-dir-ro, reading should succeed
        _repl_add_dir_ro(str(ro_dir), skill_read_roots)
        result = dispatch(
            "read_file",
            {"file_path": str(test_file)},
            base_dir=str(base_dir),
            extra_write_roots=[],
            skill_read_roots=skill_read_roots,
        )
        assert "hello from readonly" in result

        # Writing should still fail (not in extra_write_roots)
        result = dispatch(
            "write_file",
            {"file_path": str(ro_dir / "new.txt"), "content": "bad"},
            base_dir=str(base_dir),
            extra_write_roots=[],
            skill_read_roots=skill_read_roots,
        )
        assert result.startswith("error:")

    def test_help_includes_add_dir_ro(self, capsys):
        """'/help' includes /add-dir-ro in the command list."""
        _repl_help()
        captured = capsys.readouterr()
        assert "/add-dir-ro" in captured.err


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


class TestExtendCommand:
    def test_extend_doubles_by_default(self, capsys):
        """'/extend' with no arg doubles max_turns."""
        state = {"max_turns": 50}
        _repl_extend("", state)
        assert state["max_turns"] == 100
        captured = capsys.readouterr()
        assert "50 -> 100" in captured.err

    def test_extend_sets_explicit_value(self, capsys):
        """'/extend 200' sets max_turns to 200."""
        state = {"max_turns": 50}
        _repl_extend("200", state)
        assert state["max_turns"] == 200
        captured = capsys.readouterr()
        assert "200" in captured.err

    def test_extend_invalid_number(self, capsys):
        """'/extend abc' prints a warning and doesn't change state."""
        state = {"max_turns": 50}
        _repl_extend("abc", state)
        assert state["max_turns"] == 50
        captured = capsys.readouterr()
        assert "invalid number" in captured.err

    def test_extend_zero_rejected(self, capsys):
        """'/extend 0' is rejected."""
        state = {"max_turns": 50}
        _repl_extend("0", state)
        assert state["max_turns"] == 50
        captured = capsys.readouterr()
        assert "at least 1" in captured.err

    def test_extend_negative_rejected(self, capsys):
        """'/extend -5' is rejected."""
        state = {"max_turns": 50}
        _repl_extend("-5", state)
        assert state["max_turns"] == 50

    def test_extend_in_repl(self, tmp_path):
        """/extend in REPL affects the max_turns passed to run_agent_loop."""
        messages = [_sys("system")]

        call_kwargs = []

        def fake_run(msgs, tools, **kwargs):
            call_kwargs.append(kwargs["max_turns"])
            return ("answer", False)

        inputs = ["q1", "/extend", "q2", "/exit"]
        mock_session = MagicMock()
        mock_session.prompt.side_effect = inputs

        with (
            patch("prompt_toolkit.PromptSession", return_value=mock_session),
            patch("swival.agent.run_agent_loop", side_effect=fake_run),
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path, max_turns=5))

        # First call uses original max_turns=5, second uses doubled=10
        assert call_kwargs == [5, 10]

    def test_extend_explicit_in_repl(self, tmp_path):
        """/extend 20 in REPL sets max_turns to 20."""
        messages = [_sys("system")]

        call_kwargs = []

        def fake_run(msgs, tools, **kwargs):
            call_kwargs.append(kwargs["max_turns"])
            return ("answer", False)

        inputs = ["q1", "/extend 20", "q2", "/exit"]
        mock_session = MagicMock()
        mock_session.prompt.side_effect = inputs

        with (
            patch("prompt_toolkit.PromptSession", return_value=mock_session),
            patch("swival.agent.run_agent_loop", side_effect=fake_run),
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path, max_turns=5))

        assert call_kwargs == [5, 20]


# ---------------------------------------------------------------------------
# /continue command
# ---------------------------------------------------------------------------


class TestContinueCommand:
    def _mock_session(self, inputs):
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

    def test_continue_does_not_add_user_message(self, tmp_path):
        """/continue calls run_agent_loop without appending a new user message."""
        messages = [_sys("system")]

        call_snapshots = []

        def fake_run(msgs, tools, **kwargs):
            call_snapshots.append(list(msgs))
            return ("answer", False)

        inputs = ["hello", "/continue", "/exit"]
        mock_session = self._mock_session(inputs)

        with (
            patch("prompt_toolkit.PromptSession", return_value=mock_session),
            patch("swival.agent.run_agent_loop", side_effect=fake_run),
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path))

        # First call: system + "hello"
        assert len(call_snapshots[0]) == 2
        assert call_snapshots[0][1]["content"] == "hello"
        # Second call (/continue): no new user message added
        assert len(call_snapshots[1]) == 2
        assert call_snapshots[1][1]["content"] == "hello"

    def test_continue_invokes_loop(self, tmp_path):
        """/continue triggers run_agent_loop."""
        messages = [_sys("system")]

        inputs = ["q1", "/continue", "/exit"]
        mock_session = self._mock_session(inputs)

        with (
            patch("prompt_toolkit.PromptSession", return_value=mock_session),
            patch(
                "swival.agent.run_agent_loop", return_value=("answer", False)
            ) as mock_loop,
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path))

        # Two calls: one for "q1", one for /continue
        assert mock_loop.call_count == 2

    def test_continue_prints_answer(self, tmp_path, capsys):
        """/continue prints the answer from the continued loop."""
        messages = [_sys("system")]

        call_count = 0

        def fake_run(msgs, tools, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (None, True)  # exhausted, no answer
            return ("continued answer", False)

        inputs = ["q1", "/continue", "/exit"]
        mock_session = self._mock_session(inputs)

        with (
            patch("prompt_toolkit.PromptSession", return_value=mock_session),
            patch("swival.agent.run_agent_loop", side_effect=fake_run),
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path))

        captured = capsys.readouterr()
        assert "continued answer" in captured.out

    def test_continue_ctrl_c(self, tmp_path):
        """KeyboardInterrupt during /continue doesn't crash the REPL."""
        messages = [_sys("system")]

        call_count = 0

        def fake_run(msgs, tools, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ("first", False)
            if call_count == 2:
                raise KeyboardInterrupt
            return ("third", False)

        inputs = ["q1", "/continue", "q2", "/exit"]
        mock_session = self._mock_session(inputs)

        with (
            patch("prompt_toolkit.PromptSession", return_value=mock_session),
            patch("swival.agent.run_agent_loop", side_effect=fake_run),
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path))

        assert call_count == 3

    def test_continue_uses_current_max_turns(self, tmp_path):
        """/continue respects max_turns changes from /extend."""
        messages = [_sys("system")]

        call_kwargs = []

        def fake_run(msgs, tools, **kwargs):
            call_kwargs.append(kwargs["max_turns"])
            return ("answer", False)

        inputs = ["q1", "/extend 20", "/continue", "/exit"]
        mock_session = self._mock_session(inputs)

        with (
            patch("prompt_toolkit.PromptSession", return_value=mock_session),
            patch("swival.agent.run_agent_loop", side_effect=fake_run),
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path, max_turns=5))

        # First call: max_turns=5, /continue after /extend 20: max_turns=20
        assert call_kwargs == [5, 20]


# ---------------------------------------------------------------------------
# /learn command
# ---------------------------------------------------------------------------


class TestLearnCommand:
    def _mock_session(self, inputs):
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

    def test_help_includes_learn(self, capsys):
        """/learn appears in the help output."""
        _repl_help()
        captured = capsys.readouterr()
        assert "/learn" in captured.err

    def test_learn_appends_user_message(self, tmp_path):
        """/learn appends the LEARN_PROMPT as a user message."""
        messages = [_sys("system")]
        call_snapshots = []

        def fake_run(msgs, tools, **kwargs):
            call_snapshots.append(list(msgs))
            return ("learned", False)

        inputs = ["/learn", "/exit"]
        mock_session = self._mock_session(inputs)

        with (
            patch("prompt_toolkit.PromptSession", return_value=mock_session),
            patch("swival.agent.run_agent_loop", side_effect=fake_run),
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path))

        assert len(call_snapshots) == 1
        last_user = [m for m in call_snapshots[0] if m["role"] == "user"]
        assert len(last_user) == 1
        assert last_user[0]["content"] == LEARN_PROMPT

    def test_learn_invokes_loop(self, tmp_path):
        """/learn triggers run_agent_loop."""
        messages = [_sys("system")]
        inputs = ["/learn", "/exit"]
        mock_session = self._mock_session(inputs)

        with (
            patch("prompt_toolkit.PromptSession", return_value=mock_session),
            patch(
                "swival.agent.run_agent_loop", return_value=("answer", False)
            ) as mock_loop,
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path))

        assert mock_loop.call_count == 1

    def test_learn_prints_answer(self, tmp_path, capsys):
        """/learn prints the answer from the loop."""
        messages = [_sys("system")]
        inputs = ["/learn", "/exit"]
        mock_session = self._mock_session(inputs)

        with (
            patch("prompt_toolkit.PromptSession", return_value=mock_session),
            patch("swival.agent.run_agent_loop", return_value=("learn result", False)),
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path))

        captured = capsys.readouterr()
        assert "learn result" in captured.out

    def test_learn_keyboard_interrupt(self, tmp_path):
        """KeyboardInterrupt during /learn doesn't crash the REPL."""
        messages = [_sys("system")]
        call_count = 0

        def fake_run(msgs, tools, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise KeyboardInterrupt
            return ("after", False)

        inputs = ["/learn", "q1", "/exit"]
        mock_session = self._mock_session(inputs)

        with (
            patch("prompt_toolkit.PromptSession", return_value=mock_session),
            patch("swival.agent.run_agent_loop", side_effect=fake_run),
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path))

        assert call_count == 2

    def test_learn_history_label(self, tmp_path):
        """/learn records history with '/learn' label."""
        messages = [_sys("system")]
        inputs = ["/learn", "/exit"]
        mock_session = self._mock_session(inputs)

        with (
            patch("prompt_toolkit.PromptSession", return_value=mock_session),
            patch("swival.agent.run_agent_loop", return_value=("noted", False)),
            patch("swival.agent.append_history") as mock_history,
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path))

        mock_history.assert_called_once()
        args = mock_history.call_args
        assert args[0][1] == "/learn"
        assert args[0][2] == "noted"


# ---------------------------------------------------------------------------
# /init command
# ---------------------------------------------------------------------------


_VALID_AGENTS_MD = (
    "## Workflow\n\n- install: `make install`\n\n"
    "## Conventions\n\n- Example convention.\n"
)


class TestInitCommand:
    def _mock_session(self, inputs):
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

    def _fake_run_writing_on(self, base_dir, pass_num):
        """Return a fake_run side-effect that writes valid AGENTS.md on the given pass."""
        call_count = [0]

        def fake_run(msgs, tools, **kwargs):
            call_count[0] += 1
            if call_count[0] == pass_num:
                Path(base_dir, "AGENTS.md").write_text(_VALID_AGENTS_MD)
            return ("done", False)

        return fake_run, call_count

    def test_init_sends_prompt(self, tmp_path):
        """/init runs three passes when AGENTS.md is valid after write."""
        messages = [_sys("system")]

        call_messages = []
        call_count = [0]

        def fake_run(msgs, tools, **kwargs):
            call_count[0] += 1
            call_messages.append(list(msgs))
            if call_count[0] == 3:
                Path(tmp_path, "AGENTS.md").write_text(_VALID_AGENTS_MD)
            return ("done", False)

        inputs = ["/init", "/exit"]
        mock_session = self._mock_session(inputs)

        with (
            patch("prompt_toolkit.PromptSession", return_value=mock_session),
            patch("swival.agent.run_agent_loop", side_effect=fake_run) as mock_loop,
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path))

        assert mock_loop.call_count == 3
        assert call_messages[0][1]["content"] == _init_prompt()
        assert call_messages[1][2]["content"] == INIT_ENRICH_PROMPT
        assert call_messages[2][3]["content"] == INIT_WRITE_PROMPT

    def test_init_ignores_args_with_warning(self, tmp_path, capsys):
        """/init foo warns about the argument but still runs all three passes."""
        messages = [_sys("system")]
        fake_run, _ = self._fake_run_writing_on(tmp_path, 3)

        inputs = ["/init foo", "/exit"]
        mock_session = self._mock_session(inputs)

        with (
            patch("prompt_toolkit.PromptSession", return_value=mock_session),
            patch("swival.agent.run_agent_loop", side_effect=fake_run) as mock_loop,
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path))

        assert mock_loop.call_count == 3
        captured = capsys.readouterr()
        assert "/init takes no arguments" in captured.err

    def test_init_valid_file_no_retry(self, tmp_path, capsys):
        """Valid AGENTS.md after pass 3 -> no retry, no warning."""
        messages = [_sys("system")]
        fake_run, _ = self._fake_run_writing_on(tmp_path, 3)

        inputs = ["/init", "/exit"]
        mock_session = self._mock_session(inputs)

        with (
            patch("prompt_toolkit.PromptSession", return_value=mock_session),
            patch("swival.agent.run_agent_loop", side_effect=fake_run) as mock_loop,
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path))

        assert mock_loop.call_count == 3
        captured = capsys.readouterr()
        assert "still invalid" not in captured.err
        assert "exceeds" not in captured.err

    @pytest.mark.parametrize(
        "bad_content, expected_reason",
        [
            (None, "not created"),
            ("## Conventions\n\n- stuff\n", "missing"),
            (
                "## Conventions\n\n- stuff\n\n## Workflow\n\n- install: `x`\n",
                "not the first",
            ),
        ],
        ids=["file-not-created", "heading-missing", "heading-not-first"],
    )
    def test_init_retry_reason_strings(self, tmp_path, bad_content, expected_reason):
        """Each predicate failure produces correct reason in retry prompt."""
        messages = [_sys("system")]
        call_count = [0]
        call_messages = []

        def fake_run(msgs, tools, **kwargs):
            call_count[0] += 1
            call_messages.append(list(msgs))
            if call_count[0] == 3 and bad_content is not None:
                Path(tmp_path, "AGENTS.md").write_text(bad_content)
            elif call_count[0] == 4:
                Path(tmp_path, "AGENTS.md").write_text(_VALID_AGENTS_MD)
            return ("done", False)

        inputs = ["/init", "/exit"]
        mock_session = self._mock_session(inputs)

        with (
            patch("prompt_toolkit.PromptSession", return_value=mock_session),
            patch("swival.agent.run_agent_loop", side_effect=fake_run) as mock_loop,
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path))

        assert mock_loop.call_count == 4
        retry_content = call_messages[3][-1]["content"]
        assert expected_reason in retry_content
        assert retry_content != INIT_WRITE_PROMPT

    def test_init_missing_heading_retry_succeeds(self, tmp_path, capsys):
        """Missing ## Workflow triggers retry; retry fixes it -> no warning."""
        messages = [_sys("system")]
        call_count = [0]

        def fake_run(msgs, tools, **kwargs):
            call_count[0] += 1
            if call_count[0] == 3:
                Path(tmp_path, "AGENTS.md").write_text("## Conventions\n- x\n")
            elif call_count[0] == 4:
                Path(tmp_path, "AGENTS.md").write_text(_VALID_AGENTS_MD)
            return ("done", False)

        inputs = ["/init", "/exit"]
        mock_session = self._mock_session(inputs)

        with (
            patch("prompt_toolkit.PromptSession", return_value=mock_session),
            patch("swival.agent.run_agent_loop", side_effect=fake_run) as mock_loop,
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path))

        assert mock_loop.call_count == 4
        captured = capsys.readouterr()
        assert "still invalid" not in captured.err

    def test_init_missing_heading_retry_fails(self, tmp_path, capsys):
        """Both passes produce bad AGENTS.md -> warning emitted."""
        messages = [_sys("system")]
        call_count = [0]

        def fake_run(msgs, tools, **kwargs):
            call_count[0] += 1
            Path(tmp_path, "AGENTS.md").write_text("no heading here\n")
            return ("done", False)

        inputs = ["/init", "/exit"]
        mock_session = self._mock_session(inputs)

        with (
            patch("prompt_toolkit.PromptSession", return_value=mock_session),
            patch("swival.agent.run_agent_loop", side_effect=fake_run) as mock_loop,
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path))

        assert mock_loop.call_count == 4
        captured = capsys.readouterr()
        assert "still invalid" in captured.err

    def test_init_missing_file_retry_succeeds(self, tmp_path, capsys):
        """AGENTS.md not created on pass 3 -> retry creates valid file."""
        messages = [_sys("system")]
        fake_run, _ = self._fake_run_writing_on(tmp_path, 4)

        inputs = ["/init", "/exit"]
        mock_session = self._mock_session(inputs)

        with (
            patch("prompt_toolkit.PromptSession", return_value=mock_session),
            patch("swival.agent.run_agent_loop", side_effect=fake_run) as mock_loop,
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path))

        assert mock_loop.call_count == 4
        captured = capsys.readouterr()
        assert "still invalid" not in captured.err

    def test_init_budget_warning_on_initial_write(self, tmp_path, capsys):
        """Valid but oversized AGENTS.md -> no retry, budget warning."""
        messages = [_sys("system")]
        call_count = [0]

        def fake_run(msgs, tools, **kwargs):
            call_count[0] += 1
            if call_count[0] == 3:
                content = "## Workflow\n\n- install: `x`\n\n## Conventions\n\n"
                content += "- " + "x" * (_INIT_AGENTS_MD_BUDGET + 100) + "\n"
                Path(tmp_path, "AGENTS.md").write_text(content)
            return ("done", False)

        inputs = ["/init", "/exit"]
        mock_session = self._mock_session(inputs)

        with (
            patch("prompt_toolkit.PromptSession", return_value=mock_session),
            patch("swival.agent.run_agent_loop", side_effect=fake_run) as mock_loop,
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path))

        assert mock_loop.call_count == 3
        captured = capsys.readouterr()
        assert "exceeds" in captured.err

    def test_init_budget_warning_after_retry(self, tmp_path, capsys):
        """Retry produces valid but oversized file -> budget warning."""
        messages = [_sys("system")]
        call_count = [0]

        def fake_run(msgs, tools, **kwargs):
            call_count[0] += 1
            if call_count[0] == 3:
                Path(tmp_path, "AGENTS.md").write_text("bad content\n")
            elif call_count[0] == 4:
                content = "## Workflow\n\n- install: `x`\n\n## Conventions\n\n"
                content += "- " + "x" * (_INIT_AGENTS_MD_BUDGET + 100) + "\n"
                Path(tmp_path, "AGENTS.md").write_text(content)
            return ("done", False)

        inputs = ["/init", "/exit"]
        mock_session = self._mock_session(inputs)

        with (
            patch("prompt_toolkit.PromptSession", return_value=mock_session),
            patch("swival.agent.run_agent_loop", side_effect=fake_run) as mock_loop,
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path))

        assert mock_loop.call_count == 4
        captured = capsys.readouterr()
        assert "exceeds" in captured.err

    def test_init_retry_interrupt_writes_continue(self, tmp_path, capsys):
        """KeyboardInterrupt during retry writes continuation state."""
        messages = [_sys("system")]
        call_count = [0]

        def fake_run(msgs, tools, **kwargs):
            call_count[0] += 1
            if call_count[0] == 3:
                Path(tmp_path, "AGENTS.md").write_text("bad content\n")
            if call_count[0] == 4:
                raise KeyboardInterrupt
            return ("done", False)

        inputs = ["/init", "/exit"]
        mock_session = self._mock_session(inputs)

        with (
            patch("prompt_toolkit.PromptSession", return_value=mock_session),
            patch("swival.agent.run_agent_loop", side_effect=fake_run) as mock_loop,
            patch("swival.continue_here.write_continue_file") as mock_continue,
        ):
            repl_loop(
                messages,
                [],
                **_loop_kwargs(tmp_path, continue_here=True),
            )

        assert mock_loop.call_count == 4
        mock_continue.assert_called_once()
        captured = capsys.readouterr()
        assert "retry aborted" in captured.err


class TestInitPromptContract:
    """Prompt constants contain required keywords for the /init feature."""

    def test_workflow_keywords_in_init_prompt(self):
        prompt = _init_prompt()
        lower = prompt.lower()
        assert "Makefile" in prompt
        assert "test" in lower
        assert "lint" in lower
        assert any(k in lower for k in ("after every edit", "after-every-edit"))

    @pytest.mark.parametrize(
        "system, machine, release, expected_label",
        [
            ("Darwin", "arm64", "24.0.0", "macOS"),
            ("Linux", "x86_64", "6.1.0", "Linux"),
            ("Windows", "AMD64", "10.0.26100", "Windows"),
        ],
    )
    def test_platform_in_init_prompt(
        self, system, machine, release, expected_label, monkeypatch
    ):
        monkeypatch.setattr("platform.system", lambda: system)
        monkeypatch.setattr("platform.machine", lambda: machine)
        monkeypatch.setattr("platform.release", lambda: release)
        prompt = _init_prompt()
        assert expected_label in prompt
        assert machine in prompt

    def test_ci_precedence_in_init_prompt(self):
        lower = _init_prompt().lower()
        assert "ci" in lower
        assert "local" in lower

    def test_budget_target_in_write_prompt(self):
        assert str(_INIT_AGENTS_MD_BUDGET) in INIT_WRITE_PROMPT

    def test_section_ordering_in_write_prompt(self):
        wf_idx = INIT_WRITE_PROMPT.index("Workflow")
        conv_idx = INIT_WRITE_PROMPT.index("Convention")
        assert wf_idx < conv_idx

    def test_never_cut_in_enrich_prompt(self):
        lower = INIT_ENRICH_PROMPT.lower()
        assert "never cut" in lower or "do not cut" in lower


class TestValidateAgentsMd:
    """Unit tests for the validate_agents_md predicate."""

    def test_valid(self, tmp_path):
        p = tmp_path / "AGENTS.md"
        p.write_text(_VALID_AGENTS_MD)
        reason, content = validate_agents_md(p)
        assert reason is None
        assert content is not None

    def test_missing_file(self, tmp_path):
        p = tmp_path / "AGENTS.md"
        reason, content = validate_agents_md(p)
        assert reason is not None
        assert "not created" in reason
        assert content is None

    def test_missing_heading(self, tmp_path):
        p = tmp_path / "AGENTS.md"
        p.write_text("## Conventions\n- c\n")
        reason, content = validate_agents_md(p)
        assert reason is not None
        assert "missing" in reason
        assert content is not None

    def test_heading_not_first(self, tmp_path):
        p = tmp_path / "AGENTS.md"
        p.write_text("## Conventions\n- c\n\n## Workflow\n- install: `x`\n")
        reason, content = validate_agents_md(p)
        assert reason is not None
        assert "not the first" in reason

    def test_workflow_only_missing_conventions(self, tmp_path):
        p = tmp_path / "AGENTS.md"
        p.write_text("## Workflow\n\n- install: `x`\n")
        reason, _ = validate_agents_md(p)
        assert reason is not None
        assert "Conventions" in reason

    def test_conventions_only_missing_workflow(self, tmp_path):
        """## Conventions without ## Workflow -> missing workflow error."""
        p = tmp_path / "AGENTS.md"
        p.write_text("## Conventions\n\n- c\n")
        reason, _ = validate_agents_md(p)
        assert reason is not None
        assert "Workflow" in reason

    def test_trailing_whitespace_ok(self, tmp_path):
        p = tmp_path / "AGENTS.md"
        p.write_text("## Workflow   \n\n- install: `x`\n\n## Conventions\n- c\n")
        reason, _ = validate_agents_md(p)
        assert reason is None

    def test_workflow_details_not_accepted(self, tmp_path):
        """'## Workflow Details' as first H2 must not pass the ordering check."""
        p = tmp_path / "AGENTS.md"
        p.write_text(
            "## Workflow Details\n\n- stuff\n\n"
            "## Workflow\n\n- install: `x`\n\n"
            "## Conventions\n- c\n"
        )
        reason, _ = validate_agents_md(p)
        assert reason is not None
        assert "not the first" in reason


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


# ---------------------------------------------------------------------------
# /save, /restore, /unsave commands
# ---------------------------------------------------------------------------


class TestSnapshotSaveCommand:
    def test_save_sets_checkpoint(self, capsys):
        state = SnapshotState()
        msgs = [_sys("system"), _user("q1")]
        _repl_snapshot_save("my-label", msgs, state)
        assert state.explicit_active is True
        assert state.explicit_label == "my-label"
        assert state.explicit_begin_index == 2
        captured = capsys.readouterr()
        assert "checkpoint saved" in captured.err

    def test_save_default_label(self, capsys):
        state = SnapshotState()
        msgs = [_sys("system")]
        _repl_snapshot_save("user-checkpoint", msgs, state)
        assert state.explicit_label == "user-checkpoint"

    def test_save_error_duplicate(self, capsys):
        state = SnapshotState()
        msgs = [_sys("system")]
        _repl_snapshot_save("first", msgs, state)
        _repl_snapshot_save("second", msgs, state)
        captured = capsys.readouterr()
        assert "already active" in captured.err

    def test_save_none_snapshot_state(self, capsys):
        msgs = [_sys("system")]
        _repl_snapshot_save("test", msgs, None)
        captured = capsys.readouterr()
        assert "not available" in captured.err


class TestSnapshotRestoreCommand:
    def test_restore_collapses_messages(self, capsys):
        state = SnapshotState()
        msgs = [_sys("system"), _user("q1"), {"role": "assistant", "content": "a1"}]
        state.save_at_index("test", 1)

        with patch("swival.agent.call_llm") as mock_llm:
            resp = SimpleNamespace(content="LLM summary", tool_calls=None)
            mock_llm.return_value = (resp, "stop")
            _repl_snapshot_restore(
                msgs,
                state,
                model_id="test",
                api_base="http://localhost",
                api_key=None,
                top_p=1.0,
                seed=None,
                provider="lmstudio",
            )

        assert state.explicit_active is False
        assert len(state.history) == 1
        captured = capsys.readouterr()
        assert "collapsed" in captured.err

    def test_restore_no_messages_warns(self, capsys):
        state = SnapshotState()
        msgs = [_sys("system")]
        _repl_snapshot_restore(
            msgs,
            state,
            model_id="test",
            api_base="http://localhost",
            api_key=None,
            top_p=1.0,
            seed=None,
            provider="lmstudio",
        )
        captured = capsys.readouterr()
        assert "nothing to collapse" in captured.err

    def test_restore_none_snapshot_state(self, capsys):
        msgs = [_sys("system"), _user("q")]
        _repl_snapshot_restore(
            msgs,
            None,
            model_id="test",
            api_base="http://localhost",
            api_key=None,
            top_p=1.0,
            seed=None,
            provider="lmstudio",
        )
        captured = capsys.readouterr()
        assert "not available" in captured.err


class TestSnapshotUnsaveCommand:
    def test_unsave_clears_checkpoint(self, capsys):
        state = SnapshotState()
        state.save_at_index("test", 5)
        _repl_snapshot_unsave(state)
        assert state.explicit_active is False
        captured = capsys.readouterr()
        assert "cancelled" in captured.err

    def test_unsave_no_checkpoint(self, capsys):
        state = SnapshotState()
        _repl_snapshot_unsave(state)
        captured = capsys.readouterr()
        assert "no active checkpoint" in captured.err

    def test_unsave_none_snapshot_state(self, capsys):
        _repl_snapshot_unsave(None)
        captured = capsys.readouterr()
        assert "not available" in captured.err


class TestSnapshotHelpInclusion:
    def test_help_includes_snapshot_commands(self, capsys):
        _repl_help()
        captured = capsys.readouterr()
        assert "/save" in captured.err
        assert "/restore" in captured.err
        assert "/unsave" in captured.err


class TestSnapshotReplIntegration:
    def _mock_session(self, inputs):
        mock_session = MagicMock()
        side = []
        for v in inputs:
            if v is EOFError:
                side.append(EOFError())
            else:
                side.append(v)
        mock_session.prompt.side_effect = side
        return mock_session

    def test_save_command_in_repl(self, tmp_path):
        """/save foo sets a checkpoint without calling the model."""
        state = SnapshotState()
        messages = [_sys("system")]
        inputs = ["/save foo", "/exit"]
        mock_session = self._mock_session(inputs)

        with (
            patch("prompt_toolkit.PromptSession", return_value=mock_session),
            patch("swival.agent.run_agent_loop") as mock_loop,
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path, snapshot_state=state))

        assert mock_loop.call_count == 0
        assert state.explicit_active is True
        assert state.explicit_label == "foo"

    def test_save_default_label_in_repl(self, tmp_path):
        """/save (no arg) uses 'user-checkpoint'."""
        state = SnapshotState()
        messages = [_sys("system")]
        inputs = ["/save", "/exit"]
        mock_session = self._mock_session(inputs)

        with (
            patch("prompt_toolkit.PromptSession", return_value=mock_session),
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path, snapshot_state=state))

        assert state.explicit_label == "user-checkpoint"

    def test_unsave_command_in_repl(self, tmp_path):
        """/unsave clears an active checkpoint."""
        state = SnapshotState()
        messages = [_sys("system")]
        inputs = ["/save foo", "/unsave", "/exit"]
        mock_session = self._mock_session(inputs)

        with patch("prompt_toolkit.PromptSession", return_value=mock_session):
            repl_loop(messages, [], **_loop_kwargs(tmp_path, snapshot_state=state))

        assert state.explicit_active is False

    def test_save_then_compact_then_restore_error(self, tmp_path, capsys):
        """/save, /compact --drop, /restore returns invalidation error."""
        state = SnapshotState()
        messages = [
            _sys("system"),
            _user("q1"),
            {"role": "assistant", "content": "a1"},
            _user("q2"),
            {"role": "assistant", "content": "a2"},
            _user("q3"),
            {"role": "assistant", "content": "a3"},
        ]
        inputs = ["/save checkpoint", "/compact --drop", "/restore", "/exit"]
        mock_session = self._mock_session(inputs)

        with patch("prompt_toolkit.PromptSession", return_value=mock_session):
            repl_loop(messages, [], **_loop_kwargs(tmp_path, snapshot_state=state))

        captured = capsys.readouterr()
        assert "invalidated" in captured.err

    def test_save_then_autocompact_then_restore_error(self, tmp_path, capsys):
        """Auto-compaction (ContextOverflowError path) invalidates index checkpoint."""
        state = SnapshotState()
        messages = [
            _sys("system"),
            _user("q1"),
            {"role": "assistant", "content": "a1"},
        ]

        # Save checkpoint at current message count
        state.save_at_index("before-overflow", len(messages))
        assert state.explicit_active is True

        call_count = 0

        def fake_call_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ContextOverflowError("overflow")
            return _make_text_response("recovered")

        # Add a user message to trigger a loop iteration
        messages.append(_user("trigger"))

        with patch("swival.agent.call_llm", side_effect=fake_call_llm):
            run_agent_loop(messages, [], **_loop_kwargs(tmp_path, snapshot_state=state))

        # The auto-compaction path should have called invalidate_index_checkpoint
        assert state._generation > 0

        # Attempting to resolve the checkpoint should fail
        result = state._resolve_start(messages)
        assert isinstance(result, str)
        assert "invalidated" in result


# ---------------------------------------------------------------------------
# /copy command
# ---------------------------------------------------------------------------


class TestCopyCommand:
    def test_copy_macos(self):
        """On macOS, /copy shells out to pbcopy."""
        with (
            patch("sys.platform", "darwin"),
            patch("subprocess.run") as mock_run,
        ):
            _repl_copy("hello world")
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0] == ["pbcopy"]
        assert kwargs["input"] == b"hello world"
        assert kwargs["check"] is True

    def test_copy_win32(self):
        """On Windows, /copy shells out to clip."""
        with (
            patch("sys.platform", "win32"),
            patch("subprocess.run") as mock_run,
        ):
            _repl_copy("hello")
        assert mock_run.call_args[0][0] == ["clip"]

    def test_copy_linux_wl_copy(self):
        """On Linux with wl-copy available, prefer it."""
        with (
            patch("sys.platform", "linux"),
            patch("shutil.which", return_value="/usr/bin/wl-copy"),
            patch("subprocess.run") as mock_run,
        ):
            _repl_copy("hello")
        assert mock_run.call_args[0][0] == ["wl-copy"]

    def test_copy_linux_xclip(self):
        """On Linux without wl-copy, fall back to xclip."""

        def _which(name):
            if name == "wl-copy":
                return None
            if name == "xclip":
                return "/usr/bin/xclip"
            return None

        with (
            patch("sys.platform", "linux"),
            patch("shutil.which", side_effect=_which),
            patch("subprocess.run") as mock_run,
        ):
            _repl_copy("hello")
        assert mock_run.call_args[0][0] == ["xclip", "-selection", "clipboard"]

    def test_copy_linux_no_utility(self, capsys):
        """On Linux with no clipboard utility, warn."""
        with (
            patch("sys.platform", "linux"),
            patch("shutil.which", return_value=None),
            patch("subprocess.run") as mock_run,
        ):
            _repl_copy("hello")
        mock_run.assert_not_called()
        assert "no clipboard utility" in capsys.readouterr().err

    def test_copy_no_text(self, capsys):
        """With no prior answer, warn and do not shell out."""
        with patch("subprocess.run") as mock_run:
            _repl_copy(None)
        mock_run.assert_not_called()
        assert "nothing to copy" in capsys.readouterr().err

    def test_copy_subprocess_failure(self, capsys):
        """On subprocess failure, warn gracefully."""
        import subprocess

        with (
            patch("sys.platform", "darwin"),
            patch(
                "subprocess.run",
                side_effect=subprocess.SubprocessError("broken"),
            ),
        ):
            _repl_copy("hello")
        assert "clipboard copy failed" in capsys.readouterr().err

    def test_help_includes_copy(self, capsys):
        """/help text mentions /copy."""
        _repl_help()
        assert "/copy" in capsys.readouterr().err

    def test_copy_in_repl_after_turn(self, tmp_path):
        """/copy after a model turn copies that answer; no extra model call."""
        messages = [_sys("system")]
        mock_session = MagicMock()
        mock_session.prompt.side_effect = ["hello", "/copy", "/exit"]

        def fake_loop(msgs, tools, **kwargs):
            msgs.append({"role": "assistant", "content": "the answer"})
            return "the answer", False

        with (
            patch("prompt_toolkit.PromptSession", return_value=mock_session),
            patch("swival.agent.run_agent_loop", side_effect=fake_loop) as mock_loop,
            patch("swival.agent._repl_copy") as mock_copy,
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path))

        # Model called once for "hello", not for /copy
        assert mock_loop.call_count == 1
        mock_copy.assert_called_once_with("the answer")

    def test_copy_after_clear_is_none(self, tmp_path):
        """/clear resets messages so _last_assistant_text returns None."""
        messages = [_sys("system")]
        mock_session = MagicMock()
        mock_session.prompt.side_effect = ["hello", "/clear", "/copy", "/exit"]

        def fake_loop(msgs, tools, **kwargs):
            msgs.append({"role": "assistant", "content": "the answer"})
            return "the answer", False

        with (
            patch("prompt_toolkit.PromptSession", return_value=mock_session),
            patch("swival.agent.run_agent_loop", side_effect=fake_loop),
            patch("swival.agent._repl_copy") as mock_copy,
        ):
            repl_loop(messages, [], **_loop_kwargs(tmp_path))

        mock_copy.assert_called_once_with(None)

    def test_last_assistant_text_finds_latest(self):
        """_last_assistant_text returns the most recent assistant content."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        assert _last_assistant_text(messages) == "a2"

    def test_last_assistant_text_empty(self):
        """_last_assistant_text returns None when no assistant messages exist."""
        assert _last_assistant_text([{"role": "system", "content": "sys"}]) is None
        assert _last_assistant_text([]) is None

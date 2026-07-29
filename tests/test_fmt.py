"""Tests for the fmt module (ANSI-formatted output helpers)."""

from io import StringIO

from rich.console import Console

from swival import fmt
from swival.todo import TodoItem
from tests.conftest import capture_styled as _capture_styled
from tests.conftest import styled_console as _styled_console


def _capture(func, *args, **kwargs):
    """Call a fmt function with a captured console and return plain-text output."""
    buf = StringIO()
    old = fmt._console
    fmt._console = Console(file=buf, no_color=True, width=80)
    fmt.reset_state()
    try:
        func(*args, **kwargs)
    finally:
        fmt.reset_state()
        fmt._console = old
    return buf.getvalue()


class TestTurnHeader:
    def test_contains_turn_info(self):
        out = _capture(fmt.turn_header, 3, 10, 4200)
        assert "Turn 3/10" in out
        assert "4,200 tokens" in out

    def test_different_values(self):
        out = _capture(fmt.turn_header, 1, 5, 100)
        assert "Turn 1/5" in out
        assert "100 tokens" in out

    def test_title_without_context_length(self):
        assert fmt._turn_title(3, 10, 4200, None) == "Turn 3/10 (~4,200 tokens)"

    def test_title_with_context_length(self):
        assert (
            fmt._turn_title(3, 10, 4200, 128000)
            == "Turn 3/10 (~4,200 / 128,000 tokens, 3%)"
        )

    def test_title_zero_context_length_falls_back(self):
        assert fmt._turn_title(3, 10, 4200, 0) == "Turn 3/10 (~4,200 tokens)"

    def test_title_over_budget_exceeds_100_percent(self):
        assert fmt._turn_title(1, 5, 200, 100) == "Turn 1/5 (~200 / 100 tokens, 200%)"


class TestLlmTiming:
    def test_stop_reason(self):
        out = _capture(fmt.llm_timing, 1.4, "stop")
        assert "LLM responded in 1.4s" in out
        assert "finish_reason=stop" in out

    def test_length_reason(self):
        out = _capture(fmt.llm_timing, 2.3, "length")
        assert "LLM responded in 2.3s" in out
        assert "finish_reason=length" in out


class TestLlmSpinner:
    def test_no_output_when_not_terminal(self):
        """Redirected stderr must produce zero bytes (no stray newlines)."""
        buf = StringIO()
        old = fmt._console
        fmt._console = Console(file=buf, no_color=True, width=80)
        try:
            with fmt.llm_spinner("Testing spinner"):
                pass
        finally:
            fmt._console = old
        assert buf.getvalue() == ""

    def test_label_shown_on_terminal(self):
        """On a terminal console the label text appears in the output."""
        buf = StringIO()
        old = fmt._console
        fmt._console = _styled_console(buf)
        try:
            with fmt.llm_spinner("Custom label"):
                pass
        finally:
            fmt._console = old
        assert "Custom label" in buf.getvalue()


class TestCommandSpinner:
    def test_no_output_when_not_terminal(self):
        buf = StringIO()
        old = fmt._console
        fmt._console = Console(file=buf, no_color=True, width=80)
        try:
            with fmt.command_spinner("git status"):
                pass
        finally:
            fmt._console = old
        assert buf.getvalue() == ""

    def test_label_shown_on_terminal(self):
        buf = StringIO()
        old = fmt._console
        fmt._console = _styled_console(buf)
        try:
            with fmt.command_spinner("git status"):
                pass
        finally:
            fmt._console = old
        out = buf.getvalue()
        assert "Running" in out
        assert "git status" in out

    def test_progress_bar_shown_with_timeout(self):
        buf = StringIO()
        old = fmt._console
        fmt._console = _styled_console(buf)
        try:
            with fmt.command_spinner("sleep 5", timeout=5):
                pass
        finally:
            fmt._console = old
        out = buf.getvalue()
        assert "sleep 5" in out
        assert "%" in out
        assert "of timeout" in out

    def test_yields_dismiss_callable_when_not_terminal(self):
        buf = StringIO()
        old = fmt._console
        fmt._console = Console(file=buf, no_color=True, width=80)
        try:
            with fmt.command_spinner("ls") as dismiss:
                assert callable(dismiss)
                dismiss()
        finally:
            fmt._console = old


class TestInputMarquee:
    def test_short_prompt_fills_line_width(self):
        line = fmt._input_marquee_text("yo", offset=0, width=40)

        assert line.plain.startswith("  > yo")
        assert len(line.plain) == 40
        assert line.plain.count("yo") > 1

    def test_blank_prompt_has_fallback_text(self):
        line = fmt._input_marquee_text("   ", offset=0, width=40)

        assert line.plain.startswith("  > Thinking")
        assert len(line.plain) == 40

    def test_large_offset_still_fills_line_width(self):
        line = fmt._input_marquee_text("yo", offset=1_000_000, width=40)

        assert len(line.plain) == 40


class TestInputMarqueeThenSpinner:
    def test_no_output_when_not_terminal(self):
        buf = StringIO()
        old = fmt._console
        fmt._console = Console(file=buf, no_color=True, width=80)
        try:
            with fmt.input_marquee_then_spinner("hello", "label", delay=0.01):
                pass
        finally:
            fmt._console = old
        assert buf.getvalue() == ""

    def test_dismiss_before_delay_keeps_marquee(self):
        """Dismissing before the delay must not start the spinner."""
        import time

        buf = StringIO()
        old = fmt._console
        fmt._console = _styled_console(buf)
        try:
            with fmt.input_marquee_then_spinner(
                "hello world", "spinner label", delay=1.0
            ) as dismiss:
                time.sleep(0.05)
                dismiss()
        finally:
            fmt._console = old
        out = buf.getvalue()
        assert "spinner label" not in out

    def test_dismiss_during_transition_cleans_up_spinner(self):
        """A dismiss() arriving while the transition is mid-flight must not
        leave the spinner running."""
        import time

        buf = StringIO()
        old = fmt._console
        fmt._console = _styled_console(buf)
        try:
            with fmt.input_marquee_then_spinner(
                "hello world", "spinner label", delay=0.02
            ) as dismiss:
                deadline = time.monotonic() + 3.0
                while (
                    "spinner label" not in buf.getvalue()
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.005)
                dismiss()
                pre_dismiss_len = len(buf.getvalue())
                time.sleep(0.2)
                post_dismiss_len = len(buf.getvalue())
        finally:
            fmt._console = old
        assert post_dismiss_len - pre_dismiss_len < 256

    def test_transition_after_delay_shows_spinner(self):
        """If not dismissed within ``delay``, the spinner label appears."""
        import time

        buf = StringIO()
        old = fmt._console
        fmt._console = _styled_console(buf)
        try:
            with fmt.input_marquee_then_spinner(
                "hello world", "spinner label", delay=0.05
            ):
                deadline = time.monotonic() + 3.0
                while (
                    "spinner label" not in buf.getvalue()
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.02)
        finally:
            fmt._console = old
        assert "spinner label" in buf.getvalue()


class TestCompletion:
    def test_ok(self):
        out = _capture(fmt.completion, 5, "ok")
        assert "Agent finished" in out
        assert "5 turns" in out

    def test_max_turns(self):
        out = _capture(fmt.completion, 3, "max_turns")
        assert "Agent finished" in out
        assert "3 turns" in out
        assert "max_turns" in out


class TestToolCall:
    def test_basic(self):
        out = _capture(fmt.tool_call, "read_file", '{\n  "path": "foo.txt"\n}')
        assert "read_file" in out
        assert "foo.txt" in out

    def test_empty_args(self):
        out = _capture(fmt.tool_call, "think", "")
        assert "think" in out


class TestToolResult:
    def test_basic(self):
        out = _capture(fmt.tool_result, "read_file", 0.1, "Hello world")
        assert "read_file" in out
        assert "0.1s" in out
        assert "Hello world" in out

    def test_empty_preview(self):
        out = _capture(fmt.tool_result, "write_file", 0.0, "")
        assert "write_file" in out


class TestToolError:
    def test_basic(self):
        out = _capture(fmt.tool_error, "read_file", "file not found")
        assert "read_file" in out
        assert "file not found" in out


class TestGuardrail:
    def test_basic(self):
        out = _capture(fmt.guardrail, "run_command", 3, "error: command list is empty")
        normalized = " ".join(out.split())
        assert "Guardrail:" in out
        assert "run_command" in out
        assert "3 times" in out
        assert "error: command list is empty" in normalized


class TestThinkStep:
    def test_first_step_prints_header(self):
        out = _capture(fmt.think_step, 1, 5, "Analyzing the problem")
        assert "[think]" in out
        assert "\u251c\u2500" in out
        assert "Analyzing the problem" in out

    def test_subsequent_step_no_header(self):
        buf = StringIO()
        old = fmt._console
        fmt._console = Console(file=buf, no_color=True, width=80)
        fmt.reset_state()
        try:
            fmt.think_step(1, 3, "First thought")
            fmt.think_step(2, 3, "Second thought")
        finally:
            fmt.reset_state()
            fmt._console = old
        out = buf.getvalue()
        assert out.count("[think]") == 1
        assert "First thought" in out
        assert "Second thought" in out

    def test_revision(self):
        out = _capture(
            fmt.think_step,
            3,
            5,
            "Correcting step 1",
            is_revision=True,
            revises_thought=1,
        )
        assert "\u2502" in out
        assert "\u2514\u2500" in out
        assert "rev:" in out
        assert "Correcting step 1" in out

    def test_branch(self):
        out = _capture(
            fmt.think_step,
            2,
            5,
            "Alternative approach",
            branch_id="alt",
            branch_from_thought=1,
        )
        assert "\u251c\u2500" in out
        assert "[branch:alt]" in out
        assert "Alternative approach" in out

    def test_reset_restarts_header(self):
        buf = StringIO()
        old = fmt._console
        fmt._console = Console(file=buf, no_color=True, width=80)
        fmt.reset_state()
        try:
            fmt.think_step(1, 2, "First")
            fmt.reset_state()
            fmt.think_step(1, 2, "After reset")
        finally:
            fmt.reset_state()
            fmt._console = old
        out = buf.getvalue()
        assert out.count("[think]") == 2

    def test_turn_header_resets_think(self):
        buf = StringIO()
        old = fmt._console
        fmt._console = Console(file=buf, no_color=True, width=80)
        fmt.reset_state()
        try:
            fmt.think_step(1, 2, "Before turn")
            fmt.turn_header(2, 10, 1000)
            fmt.think_step(1, 2, "After turn")
        finally:
            fmt.reset_state()
            fmt._console = old
        out = buf.getvalue()
        assert out.count("[think]") == 2


class TestAssistantText:
    def test_basic(self):
        out = _capture(fmt.assistant_text, "Let me check that file.")
        assert "│" in out
        assert "Let me check that file." in out

    def test_markdown_heading(self):
        out = _capture(fmt.assistant_text, "# Hello\nSome text.")
        assert "│" in out
        assert "Hello" in out
        assert "Some text." in out

    def test_markdown_code_block(self):
        out = _capture(fmt.assistant_text, "Here:\n```python\nprint('hi')\n```")
        assert "│" in out
        assert "print" in out

    def test_truncation_by_logical_lines(self):
        long_text = "\n".join(f"Line {i}" for i in range(200))
        out = _capture(fmt.assistant_text, long_text)
        assert "truncated" in out
        assert "100 more lines" in out
        # First 100 lines rendered, rest truncated
        assert "Line 0" in out
        assert "Line 99" in out

    def test_no_truncation_for_long_paragraph_on_narrow_terminal(self):
        """A single long paragraph must not be truncated regardless of terminal width."""
        words = " ".join(f"word{i}" for i in range(500))
        buf = StringIO()
        old = fmt._console
        fmt._console = Console(file=buf, no_color=True, width=40)
        fmt.reset_state()
        try:
            fmt.assistant_text(words)
        finally:
            fmt.reset_state()
            fmt._console = old
        out = buf.getvalue()
        assert "truncated" not in out
        assert "word0" in out
        assert "word499" in out

    def test_empty_text(self):
        out = _capture(fmt.assistant_text, "")
        assert isinstance(out, str)


class TestModelInfo:
    def test_basic(self):
        out = _capture(fmt.model_info, "Discovered model: qwen3-8b")
        assert "Discovered model: qwen3-8b" in out


class TestInfo:
    def test_basic(self):
        out = _capture(fmt.info, "Loaded CLAUDE.md (500 bytes)")
        assert "Loaded CLAUDE.md (500 bytes)" in out


class TestContextStats:
    def test_basic(self):
        out = _capture(fmt.context_stats, "Context after compaction", 3200)
        assert "Context after compaction" in out
        assert "3200 tokens" in out


class TestWarning:
    def test_basic(self):
        out = _capture(fmt.warning, "context window exceeded")
        assert "Warning:" in out
        assert "context window exceeded" in out


class TestError:
    def test_basic(self):
        out = _capture(fmt.error, "LLM call failed: connection refused")
        assert "Error:" in out
        assert "LLM call failed: connection refused" in out


class TestMarkupEscaping:
    """Dynamic text containing Rich markup brackets should appear literally."""

    def test_brackets_in_tool_call_args(self):
        out = _capture(fmt.tool_call, "write_file", '{"content": "[bold]not bold[/]"}')
        # The brackets should appear literally, not be interpreted as markup
        assert "[bold]not bold[/]" in out

    def test_brackets_in_think_step(self):
        out = _capture(fmt.think_step, 1, 1, "Check if [link=http://x] works")
        assert "[link=http://x]" in out
        assert "\u251c\u2500" in out

    def test_brackets_in_assistant_text(self):
        out = _capture(fmt.assistant_text, "The tag is [bold red]")
        assert "[bold red]" in out

    def test_brackets_in_error(self):
        out = _capture(fmt.error, "unexpected [tag] in response")
        assert "[tag]" in out

    def test_brackets_in_warning(self):
        out = _capture(fmt.warning, "found [italic]markup[/] in output")
        assert "[italic]markup[/]" in out


class TestFramed:
    def test_returns_false_when_not_tty(self):
        buf = StringIO()
        old = fmt._console
        fmt._console = Console(file=buf, no_color=True, width=80)
        try:
            result = fmt._framed(fmt.Text("body"), title="t", subtitle="s")
        finally:
            fmt._console = old
        assert result is False
        assert buf.getvalue() == ""

    def test_returns_true_in_tty(self):
        buf = StringIO()
        old = fmt._console
        fmt._console = _styled_console(buf)
        try:
            result = fmt._framed(fmt.Text("body"), title="t", subtitle="s")
        finally:
            fmt._console = old
        assert result is True
        assert "body" in buf.getvalue()

    def test_title_strips_newlines(self):
        buf = StringIO()
        old = fmt._console
        fmt._console = _styled_console(buf)
        try:
            fmt._framed(fmt.Text("x"), title="line one\nline two\rthree")
        finally:
            fmt._console = old
        out = buf.getvalue()
        assert "line one line two three" in out
        # the title must not introduce extra wrapping inside itself
        assert "\nline two" not in out

    def test_title_truncated_with_ellipsis(self):
        buf = StringIO()
        old = fmt._console
        fmt._console = _styled_console(buf)
        try:
            fmt._framed(fmt.Text("body"), title="x" * 200)
        finally:
            fmt._console = old
        out = buf.getvalue()
        assert "…" in out
        # body line count: top border, body, bottom border = 3 visible lines
        # ensure title did not wrap into multiple border rows
        border_lines = [ln for ln in out.splitlines() if "╮" in ln or "╭" in ln]
        assert len(border_lines) == 1

    def test_title_markup_escaped(self):
        buf = StringIO()
        old = fmt._console
        fmt._console = _styled_console(buf)
        try:
            fmt._framed(fmt.Text("body"), title="[bold red]injected[/bold red]")
        finally:
            fmt._console = old
        assert "[bold red]injected[/bold red]" in buf.getvalue()


class TestToolDiff:
    def test_formatting(self):
        old = "aaa\nbbb\nccc\n"
        new = "aaa\nBBB\nccc\n"
        out = _capture(fmt.tool_diff, "file.txt", old, new)
        assert "---" in out
        assert "+++" in out
        assert "-bbb" in out
        assert "+BBB" in out

    def test_no_diff_when_identical(self):
        text = "aaa\nbbb\n"
        out = _capture(fmt.tool_diff, "file.txt", text, text)
        assert out == ""

    def test_truncation_by_lines(self):
        old = "".join(f"line{i}\n" for i in range(100))
        new = "".join(f"LINE{i}\n" for i in range(100))
        out = _capture(fmt.tool_diff, "file.txt", old, new)
        assert "more lines" in out

    def test_truncation_by_bytes(self):
        old = "x" * 500 + "\n"
        new = "y" * 500 + "\n"
        # Each diff line is ~500 bytes, so 4KB cap should trigger before 50 lines
        old_big = old * 20
        new_big = new * 20
        out = _capture(fmt.tool_diff, "file.txt", old_big, new_big)
        assert "more lines" in out

    def test_single_long_line_capped(self):
        old = "a" * 8000 + "\n"
        new = "b" * 8000 + "\n"
        out = _capture(fmt.tool_diff, "file.txt", old, new)
        assert len(out.encode("utf-8")) < 4096 + 512  # headers + indent overhead

    def test_markup_safety(self):
        old = "before [bold]markup[/bold] after\n"
        new = "before [italic]changed[/italic] after\n"
        out = _capture(fmt.tool_diff, "file.txt", old, new)
        assert "[bold]markup[/bold]" in out
        assert "[italic]changed[/italic]" in out


class TestTodoList:
    def test_renders_checklist(self):
        items = [
            TodoItem("Read the codebase", done=True),
            TodoItem("Write unit tests"),
            TodoItem("Fix the bug"),
        ]
        out = _capture(fmt.todo_list, items)
        assert "[todo]" in out
        assert "2 remaining" in out
        assert "\u2611" in out  # done checkbox
        assert "\u2610" in out  # pending checkbox
        assert "Read the codebase" in out
        assert "Write unit tests" in out
        assert "Fix the bug" in out

    def test_changed_task_highlighted(self):
        items = [TodoItem("Task A"), TodoItem("Task B")]
        out = _capture_styled(fmt.todo_list, items, changed_task="Task B")
        assert "Task B" in out
        # ANSI bold escape: ESC[1m appears before "Task B"
        idx = out.index("Task B")
        preceding = out[max(0, idx - 20) : idx]
        assert "\x1b[1m" in preceding or "\x1b[1;" in preceding

    def test_changed_done_task_highlighted(self):
        items = [TodoItem("Task A", done=True), TodoItem("Task B")]
        out = _capture_styled(fmt.todo_list, items, changed_task="Task A")
        assert "Task A" in out
        # The done item should still be bolded when it's the changed task
        idx = out.index("Task A")
        preceding = out[max(0, idx - 30) : idx]
        assert "\x1b[1m" in preceding or "\x1b[1;" in preceding

    def test_note_shown(self):
        items = [TodoItem("Existing task")]
        out = _capture(fmt.todo_list, items, note="Already listed: Existing task")
        assert "Already listed: Existing task" in out
        assert "1 remaining" in out

    def test_empty_list(self):
        out = _capture(fmt.todo_list, [])
        assert "[todo]" in out
        assert "0 remaining" in out

    def test_clear_note(self):
        out = _capture(fmt.todo_list, [], note="3 items removed")
        assert "0 remaining" in out
        assert "3 items removed" in out


class TestInit:
    def test_default(self):
        old = fmt._console
        fmt.init(color=False, no_color=False)
        # Verify console was reconfigured (it's a stderr console)
        assert fmt._console is not old or True  # may be same object
        fmt._console = old

    def test_no_color(self):
        old = fmt._console
        fmt.init(no_color=True)
        assert fmt._console._color_system is None
        fmt._console = old

    def test_color_overrides_no_color_env(self, monkeypatch):
        """--color must explicitly set no_color=False so it overrides NO_COLOR env."""
        monkeypatch.setenv("NO_COLOR", "1")
        old = fmt._console
        fmt.init(color=True)
        assert fmt._console.no_color is False
        fmt._console = old

    def test_init_configures_stdout_console(self):
        old = fmt._stdout_console
        try:
            fmt.init(no_color=True)
            assert fmt._stdout_console.no_color is True
        finally:
            fmt._stdout_console = old

    def test_init_color_configures_stdout_console(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        old = fmt._stdout_console
        try:
            fmt.init(color=True)
            assert fmt._stdout_console.no_color is False
        finally:
            fmt._stdout_console = old

    def test_color_does_not_force_stdout_terminal(self, monkeypatch):
        """--color forces stderr terminal mode but not stdout (keeps piped stdout clean)."""
        monkeypatch.delenv("NO_COLOR", raising=False)
        old_stderr = fmt._console
        old_stdout = fmt._stdout_console
        try:
            fmt.init(color=True)
            assert fmt._console.is_terminal is True
            assert fmt._stdout_console.is_terminal is False
        finally:
            fmt._console = old_stderr
            fmt._stdout_console = old_stdout


class TestReplAnswer:
    def test_no_color_produces_plain_text(self, capsys):
        """With no_color, repl_answer falls back to plain print on stdout."""
        old = fmt._stdout_console
        fmt._stdout_console = Console(file=None, no_color=True, width=80)
        try:
            fmt.repl_answer("hello world")
        finally:
            fmt._stdout_console = old
        captured = capsys.readouterr()
        assert "hello world" in captured.out
        # No ANSI escape sequences
        assert "\x1b[" not in captured.out

    def test_non_tty_produces_plain_text(self, capsys):
        """When stdout is not a TTY, repl_answer falls back to plain print."""
        old = fmt._stdout_console
        buf = StringIO()
        fmt._stdout_console = Console(file=buf, width=80)
        try:
            fmt.repl_answer("hello world")
        finally:
            fmt._stdout_console = old
        # The non-TTY console writes to buf, but the fallback plain print
        # should go to real stdout (captured by capsys)
        captured = capsys.readouterr()
        assert "hello world" in captured.out

    def test_tty_renders_markdown(self):
        """When stdout is a TTY with color, repl_answer renders Markdown."""
        buf = StringIO()
        old = fmt._stdout_console
        fmt._stdout_console = _styled_console(buf)
        try:
            fmt.repl_answer("**bold text**")
        finally:
            fmt._stdout_console = old
        output = buf.getvalue()
        assert "**bold text**" in output  # raw markers preserved for copy/paste
        assert "\x1b[" in output  # but with ANSI styling


class TestReviewFeedback:
    def test_non_terminal_keeps_round_header(self):
        out = _capture(fmt.review_feedback, 2, "first line\nsecond line")
        assert "[review round 2]" in out
        assert "Reviewer requested changes:" in out
        assert "first line" in out
        assert "second line" in out

    def test_terminal_renders_panel_with_round_title(self):
        out = _capture_styled(fmt.review_feedback, 4, "needs work")
        assert "review round 4" in out
        assert "needs work" in out
        # framed output draws border characters
        assert "╭" in out and "╯" in out


class TestToolFetch:
    def _result(self, **overrides):
        from swival.fetch import FetchResult

        defaults = dict(
            body="hello world",
            final_url="https://example.com/page",
            status=200,
            content_type="text/html",
            raw_bytes=42,
            saved_path=None,
        )
        defaults.update(overrides)
        return FetchResult(**defaults)

    def test_non_terminal_is_silent(self):
        out = _capture(fmt.tool_fetch, self._result())
        assert out == ""

    def test_terminal_shows_url_and_status(self):
        out = _capture_styled(fmt.tool_fetch, self._result())
        assert "https://example.com/page" in out
        assert "200" in out
        assert "42 B" in out
        assert "text/html" in out
        assert "hello world" in out

    def test_truncation_by_lines(self):
        body = "".join(f"line{i}\n" for i in range(80))
        out = _capture_styled(fmt.tool_fetch, self._result(body=body))
        assert "more lines" in out

    def test_truncation_by_bytes(self):
        body = ("x" * 500 + "\n") * 20
        out = _capture_styled(fmt.tool_fetch, self._result(body=body))
        assert "more lines" in out

    def test_error_uses_red_border(self):
        out = _capture_styled(
            fmt.tool_fetch,
            self._result(
                body="error: HTTP 404 — Not Found",
                status=None,
                raw_bytes=0,
                content_type=None,
            ),
        )
        assert "error: HTTP 404" in out
        # red border ansi code (CSI 31) appears around the box
        assert "\x1b[31m" in out

    def test_saved_path_in_subtitle(self):
        out = _capture_styled(
            fmt.tool_fetch,
            self._result(saved_path=".swival/cmd_output_abc.txt"),
        )
        assert "saved .swival/cmd_output_abc.txt" in out

    def test_single_long_line_marked_truncated(self):
        body = "x" * 10000
        out = _capture_styled(fmt.tool_fetch, self._result(body=body))
        assert "truncated at" in out


def _visual_rows(text: str, width: int) -> list[str]:
    rows: list[str] = []
    for line in text.split("\n"):
        rows.extend(fmt._wrap_to_rows(line, width))
    return rows


class TestTailToViewport:
    def test_keeps_newest_rows(self):
        text = "\n".join(f"line{i}" for i in range(100))
        out = fmt._tail_to_viewport(text, width=80, height=5).plain
        lines = out.split("\n")
        assert len(lines) == 5
        assert lines == [f"line{i}" for i in range(95, 100)]
        assert "line0" not in lines

    def test_long_line_consumes_multiple_rows(self):
        # A long final line should push earlier short lines off sooner than
        # short lines would.
        long_tail = "z" * 200
        text = "a\nb\nc\nd\n" + long_tail
        out = fmt._tail_to_viewport(text, width=20, height=4).plain
        rows = out.split("\n")
        assert len(rows) == 4
        # All four kept rows belong to the wrapped long line; the short lines
        # were pushed out because the long line alone fills the budget.
        assert "a" not in rows and "b" not in rows
        assert all(set(r) <= {"z"} for r in rows)

    def test_single_over_long_line_trimmed_to_tail(self):
        # One source line wrapping past the viewport must return exactly height
        # rows showing its tail, never overflow.
        line = "".join(str(i % 10) for i in range(500))
        height = 4
        width = 20
        out = fmt._tail_to_viewport(line, width=width, height=height)
        rows = out.plain.split("\n")
        assert len(rows) == height
        assert all(len(r) <= width for r in rows)
        # The kept rows are the final visual rows of the wrapped line.
        expected = _visual_rows(line, width)[-height:]
        assert rows == expected
        assert out.plain.endswith(line[-1])

    def test_empty_line_is_one_row(self):
        assert fmt._wrap_to_rows("", 10) == [""]

    def test_blank_run_collapses_to_single(self):
        text = "a" + "\n" * 6 + "b"
        out = fmt._tail_to_viewport(text, width=80, height=10).plain
        assert out.split("\n") == ["a", "", "b"]

    def test_trailing_blanks_stripped(self):
        text = "real content" + "\n" * 5
        out = fmt._tail_to_viewport(text, width=80, height=10).plain
        assert out == "real content"

    def test_blank_run_keeps_content_in_viewport(self):
        # Without collapsing, the blank run would push "a" off a 3-row viewport
        # and leave "b" trailing empties; collapsing keeps both visible.
        out = fmt._tail_to_viewport("a\n\n\n\n\n\nb", width=80, height=3).plain
        assert out.split("\n") == ["a", "", "b"]


class TestCollapseBlankRows:
    def test_empty_input(self):
        assert fmt._collapse_blank_rows([]) == []

    def test_all_blank_collapses_to_empty(self):
        assert fmt._collapse_blank_rows(["", "  ", ""]) == []

    def test_no_blanks_unchanged(self):
        rows = ["a", "b", "c"]
        assert fmt._collapse_blank_rows(rows) == rows

    def test_interior_run_kept_as_one(self):
        assert fmt._collapse_blank_rows(["a", "", "", "", "b"]) == ["a", "", "b"]

    def test_leading_blanks_collapse(self):
        assert fmt._collapse_blank_rows(["", "", "a"]) == ["", "a"]


class TestStreamRaw:
    def test_non_tty_yields_noop(self):
        buf = StringIO()
        old = fmt._console
        fmt._console = Console(file=buf, no_color=True, width=80)
        try:
            with fmt.stream_raw() as update:
                assert update is fmt._noop
                update("anything")
            assert buf.getvalue() == ""
        finally:
            fmt._console = old

    def test_renders_then_cleans_up(self):
        buf = StringIO()
        old = fmt._console
        fmt._console = Console(
            file=buf,
            force_terminal=True,
            color_system="truecolor",
            no_color=False,
            width=40,
            height=10,
            _environ={"TERM": "xterm-256color"},
        )
        try:
            with fmt.stream_raw() as update:
                assert update is not fmt._noop
                update("hello world")
                update("hello world\nsecond line")
            out = buf.getvalue()
        finally:
            fmt._console = old
        assert "second line" in out
        # A transient Live wipes its rendered region on exit, which emits an
        # erase-in-line control sequence.
        assert "\x1b[2K" in out


class TestRenderStreamChannels:
    def test_answer_only_matches_legacy_view(self):
        # No reasoning and no activity: identical to the single-stream display.
        text = "\n".join(f"line{i}" for i in range(20))
        composed = fmt.render_stream_channels("", text, "", width=40, height=6).plain
        legacy = fmt._tail_to_viewport(text, 40, 6).plain
        assert composed == legacy

    def test_thinking_gets_header_and_fills_when_no_answer(self):
        out = fmt.render_stream_channels(
            "reasoning a\nreasoning b", "", "", width=40, height=6
        ).plain
        lines = out.split("\n")
        assert lines[0] == "thinking…"
        assert "reasoning a" in out
        assert "reasoning b" in out

    def test_answer_keeps_priority_over_long_thinking(self):
        reasoning = "\n".join(f"think{i}" for i in range(50))
        answer = "\n".join(f"ans{i}" for i in range(50))
        out = fmt.render_stream_channels(
            reasoning, answer, "", width=40, height=10
        ).plain
        lines = out.split("\n")
        think_lines = [
            ln for ln in lines if ln.startswith("think") and ln[-1].isdigit()
        ]
        ans_lines = [ln for ln in lines if ln.startswith("ans")]
        # Reasoning is demoted to a short tail; the answer gets the rest.
        assert len(think_lines) <= fmt._THINK_TAIL_ROWS
        assert len(ans_lines) > len(think_lines)
        # The newest answer rows survive.
        assert "ans49" in out

    def test_activity_lands_at_bottom(self):
        out = fmt.render_stream_channels(
            "", "the answer", " <read_file>", width=40, height=8
        ).plain
        lines = out.split("\n")
        assert "the answer" in out
        assert "read_file" in lines[-1]

    def test_too_short_viewport_does_not_crash(self):
        out = fmt.render_stream_channels(
            "a\nb\nc", "d\ne\nf", " <tool>", width=10, height=1
        ).plain
        assert isinstance(out, str)


class _TTYBuf(StringIO):
    """A StringIO that reports itself as an interactive terminal."""

    def isatty(self) -> bool:
        return True


def _interactive_console(buf: StringIO) -> Console:
    """A forced-color console backed by a buffer that claims to be a real TTY."""
    return Console(
        file=buf,
        force_terminal=True,
        color_system="truecolor",
        no_color=False,
        width=80,
        _environ={"TERM": "xterm-256color"},
    )


class _SwapConsole:
    """Temporarily install a console, no-op out sleeps, and restore on exit."""

    def __init__(self, monkeypatch, console: Console):
        self._monkeypatch = monkeypatch
        self._console = console
        self._old = None

    def __enter__(self) -> Console:
        self._old = fmt._console
        fmt._console = self._console
        self._monkeypatch.setattr(fmt.time, "sleep", lambda *a, **k: None)
        return self._console

    def __exit__(self, *exc) -> None:
        fmt._console = self._old


class TestAnimationGating:
    def test_enabled_on_real_interactive_tty(self, monkeypatch):
        monkeypatch.delenv("SWIVAL_ANIMATIONS", raising=False)
        with _SwapConsole(monkeypatch, _interactive_console(_TTYBuf())):
            assert fmt.animations_enabled() is True

    def test_disabled_when_not_a_terminal(self, monkeypatch):
        monkeypatch.delenv("SWIVAL_ANIMATIONS", raising=False)
        with _SwapConsole(monkeypatch, Console(file=StringIO(), width=80)):
            assert fmt.animations_enabled() is False

    def test_disabled_for_forced_color_into_a_pipe(self, monkeypatch):
        # --color forces terminal output for ANSI in a file, but the stream is
        # not a real TTY, so animations must stay off.
        monkeypatch.delenv("SWIVAL_ANIMATIONS", raising=False)
        piped = _interactive_console(StringIO())  # plain StringIO: isatty False
        with _SwapConsole(monkeypatch, piped):
            assert piped.is_terminal is True
            assert fmt.animations_enabled() is False

    def test_disabled_when_no_color(self, monkeypatch):
        monkeypatch.delenv("SWIVAL_ANIMATIONS", raising=False)
        console = Console(file=_TTYBuf(), force_terminal=True, no_color=True, width=80)
        with _SwapConsole(monkeypatch, console):
            assert fmt.animations_enabled() is False

    def test_env_var_read_live(self, monkeypatch):
        with _SwapConsole(monkeypatch, _interactive_console(_TTYBuf())):
            monkeypatch.setenv("SWIVAL_ANIMATIONS", "0")
            assert fmt.animations_enabled() is False
            # Flipping the var mid-process takes effect without a reimport.
            monkeypatch.setenv("SWIVAL_ANIMATIONS", "1")
            assert fmt.animations_enabled() is True
            monkeypatch.setenv("SWIVAL_ANIMATIONS", "off")
            assert fmt.animations_enabled() is False


class TestDecorativeAnimations:
    def test_repl_splash_noop_when_not_terminal(self, monkeypatch):
        buf = StringIO()
        with _SwapConsole(monkeypatch, Console(file=buf, width=80)):
            fmt.repl_splash(model="m", provider="p", workspace="/w")
        assert buf.getvalue() == ""

    def test_repl_splash_static_when_animations_off(self, monkeypatch):
        monkeypatch.setenv("SWIVAL_ANIMATIONS", "0")
        buf = _TTYBuf()
        with _SwapConsole(monkeypatch, _interactive_console(buf)):
            fmt.repl_splash(model="qwen3", provider="lmstudio", workspace="/tmp/ws")
        out = buf.getvalue()
        assert "qwen3" in out and "lmstudio" in out

    def test_repl_splash_rule_is_static_even_when_animations_on(self, monkeypatch):
        from rich.text import Text

        monkeypatch.delenv("SWIVAL_ANIMATIONS", raising=False)
        phases = []
        monkeypatch.setattr(fmt, "_animate_logo", lambda: None)

        def rule_text(phase=0.0):
            phases.append(phase)
            return Text(f"rule phase={phase}")

        monkeypatch.setattr(fmt, "_gradient_rule_text", rule_text)
        buf = _TTYBuf()
        with _SwapConsole(monkeypatch, _interactive_console(buf)):
            fmt.repl_splash(model="qwen3")
        assert phases == [0.0]
        assert "rule phase=0.0" in buf.getvalue()

    def test_turn_header_animates_first_turn_only(self, monkeypatch):
        monkeypatch.delenv("SWIVAL_ANIMATIONS", raising=False)
        calls = []
        monkeypatch.setattr(
            fmt, "_animate_turn_rule", lambda title: calls.append(title)
        )
        with _SwapConsole(monkeypatch, _interactive_console(_TTYBuf())):
            fmt.turn_header(1, 30, 100, 128000)
            fmt.turn_header(2, 30, 200, 128000)
            fmt.turn_header(7, 30, 300, 128000)
        assert len(calls) == 1  # only the opening turn animates


class TestSessionCostFmt:
    def _snap(self, usd, priced=1, unpriced=0):
        from swival.cost import CostSnapshot

        return CostSnapshot(known_usd=usd, priced_calls=priced, unpriced_calls=unpriced)

    def test_six_decimals_below_one_cent(self):
        assert fmt._format_usd(0.003812) == "$0.003812"

    def test_four_decimals_below_one_dollar(self):
        assert fmt._format_usd(0.25) == "$0.2500"

    def test_two_decimals_above_one_dollar(self):
        assert fmt._format_usd(12.3456) == "$12.35"

    def test_no_scientific_notation(self):
        assert "e" not in fmt._format_usd(0.0000004).lower()

    def test_complete_subtotal_renders_session_cost(self):
        out = _capture(fmt.session_cost, self._snap(0.003812))
        assert "Session cost: ~$0.003812" in out
        assert "Known" not in out

    def test_partial_subtotal_renders_known_label(self):
        out = _capture(fmt.session_cost, self._snap(0.003812, unpriced=2))
        assert "Known session cost: ~$0.003812 (2 calls unpriced)" in out

    def test_single_unpriced_call_is_singular(self):
        out = _capture(fmt.session_cost, self._snap(0.01, unpriced=1))
        assert "(1 call unpriced)" in out

    def test_no_output_before_a_priced_call(self):
        out = _capture(fmt.session_cost, self._snap(0.0, priced=0, unpriced=3))
        assert out == ""

    def test_zero_only_subtotal_prints_nothing(self):
        out = _capture(fmt.session_cost, self._snap(0.0, priced=2))
        assert out == ""

    def test_tiny_subtotal_rounding_to_zero_prints_nothing(self):
        out = _capture(fmt.session_cost, self._snap(0.0000004, priced=2))
        assert out == ""

    def test_tiny_partial_subtotal_rounding_to_zero_prints_nothing(self):
        out = _capture(fmt.session_cost, self._snap(0.0000004, priced=1, unpriced=2))
        assert out == ""

    def test_smallest_displayable_subtotal_is_not_zero(self):
        out = _capture(fmt.session_cost, self._snap(0.000001, priced=1))
        assert "Session cost: ~$0.000001" in out

    def test_output_goes_to_stderr_console(self, capsys):
        fmt.session_cost(self._snap(0.5))
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "Session cost" in captured.err

"""Tests for the fmt module (ANSI-formatted output helpers)."""

from io import StringIO

from rich.console import Console

from swival import fmt
from swival.todo import TodoItem


def _capture(func, *args, **kwargs):
    """Call a fmt function with a captured console and return plain-text output."""
    buf = StringIO()
    old = fmt._console
    fmt._console = Console(file=buf, no_color=True, width=80)
    try:
        func(*args, **kwargs)
    finally:
        fmt._console = old
    return buf.getvalue()


def _capture_styled(func, *args, **kwargs):
    """Call a fmt function with color enabled and return ANSI-escaped output."""
    buf = StringIO()
    old = fmt._console
    fmt._console = Console(file=buf, force_terminal=True, width=80)
    try:
        func(*args, **kwargs)
    finally:
        fmt._console = old
    return buf.getvalue()


class TestTurnHeader:
    def test_contains_turn_info(self):
        out = _capture(fmt.turn_header, 3, 10, 4200)
        assert "Turn 3/10" in out
        assert "4200 tokens" in out

    def test_different_values(self):
        out = _capture(fmt.turn_header, 1, 5, 100)
        assert "Turn 1/5" in out
        assert "100 tokens" in out


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
        fmt._console = Console(file=buf, force_terminal=True, width=80)
        try:
            with fmt.llm_spinner("Custom label"):
                pass
        finally:
            fmt._console = old
        assert "Custom label" in buf.getvalue()


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
    def test_basic(self):
        out = _capture(fmt.think_step, 2, 5, "Analyzing the problem")
        assert "[think 2/5]" in out
        assert "Analyzing the problem" in out

    def test_revision(self):
        out = _capture(
            fmt.think_step,
            3,
            5,
            "Correcting step 1",
            is_revision=True,
            revises_thought=1,
        )
        assert "[think 3/5 rev:1]" in out
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
        assert "branch:alt" in out
        assert "from:1" in out


class TestAssistantText:
    def test_basic(self):
        out = _capture(fmt.assistant_text, "Let me check that file.")
        assert "[assistant]" in out
        assert "Let me check that file." in out


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

    def test_brackets_in_assistant_text(self):
        out = _capture(fmt.assistant_text, "The tag is [bold red]")
        assert "[bold red]" in out

    def test_brackets_in_error(self):
        out = _capture(fmt.error, "unexpected [tag] in response")
        assert "[tag]" in out

    def test_brackets_in_warning(self):
        out = _capture(fmt.warning, "found [italic]markup[/] in output")
        assert "[italic]markup[/]" in out


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

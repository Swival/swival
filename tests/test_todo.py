"""Tests for the todo tool: TodoState, dispatch integration, and agent logging."""

import json
import os
import sys
import types


from swival.todo import TodoState, MAX_ITEMS, MAX_ITEM_TEXT
from swival.tools import dispatch


# ---------------------------------------------------------------------------
# TodoState.process() unit tests
# ---------------------------------------------------------------------------


class TestAdd:
    def test_add_creates_item(self):
        state = TodoState()
        result = json.loads(state.process({"action": "add", "task": "Fix the bug"}))
        assert result["action"] == "add"
        assert result["total"] == 1
        assert result["remaining"] == 1
        assert result["items"] == [{"task": "Fix the bug", "done": False}]
        assert "note" not in result

    def test_add_multiple(self):
        state = TodoState()
        state.process({"action": "add", "task": "First"})
        result = json.loads(state.process({"action": "add", "task": "Second"}))
        assert result["total"] == 2
        assert result["remaining"] == 2

    def test_add_deduplicates_exact_match(self):
        state = TodoState()
        state.process({"action": "add", "task": "Same task"})
        result = json.loads(state.process({"action": "add", "task": "Same task"}))
        assert result["total"] == 1
        assert result["remaining"] == 1
        assert state.add_count == 1
        # Duplicate response must include a note so models know to move on
        assert "note" in result
        assert "already in list" in result["note"]

    def test_add_deduplicates_case_insensitive(self):
        state = TodoState()
        state.process({"action": "add", "task": "Fix login bug"})
        result = json.loads(state.process({"action": "add", "task": "fix login bug"}))
        assert result["total"] == 1
        assert result["items"] == [{"task": "Fix login bug", "done": False}]
        assert "note" in result

    def test_add_without_task(self):
        state = TodoState()
        result = state.process({"action": "add"})
        assert result.startswith("error:")

    def test_add_empty_task(self):
        state = TodoState()
        result = state.process({"action": "add", "task": ""})
        assert result.startswith("error:")

    def test_add_whitespace_task(self):
        state = TodoState()
        result = state.process({"action": "add", "task": "   "})
        assert result.startswith("error:")


class TestDone:
    def test_done_marks_item(self):
        state = TodoState()
        state.process({"action": "add", "task": "Fix the bug"})
        result = json.loads(state.process({"action": "done", "task": "Fix the bug"}))
        assert result["items"][0]["done"] is True
        assert result["remaining"] == 0

    def test_done_already_done_is_noop(self):
        state = TodoState()
        state.process({"action": "add", "task": "Fix the bug"})
        state.process({"action": "done", "task": "Fix the bug"})
        # Second done on same item is a no-op, not an error
        result = json.loads(state.process({"action": "done", "task": "Fix the bug"}))
        assert result["items"][0]["done"] is True
        assert result["remaining"] == 0
        # done_count should only have incremented once
        assert state.done_count == 1

    def test_done_without_task(self):
        state = TodoState()
        result = state.process({"action": "done"})
        assert result.startswith("error:")

    def test_done_no_match(self):
        state = TodoState()
        state.process({"action": "add", "task": "Something"})
        result = state.process({"action": "done", "task": "Nonexistent"})
        assert "error:" in result
        assert "no task matching" in result


class TestRemove:
    def test_remove_deletes_item(self):
        state = TodoState()
        state.process({"action": "add", "task": "First"})
        state.process({"action": "add", "task": "Second"})
        result = json.loads(state.process({"action": "remove", "task": "First"}))
        assert result["total"] == 1
        assert result["items"][0]["task"] == "Second"

    def test_remove_no_match(self):
        state = TodoState()
        state.process({"action": "add", "task": "Something"})
        result = state.process({"action": "remove", "task": "Nonexistent"})
        assert result.startswith("error:")


class TestClear:
    def test_clear_removes_all(self):
        state = TodoState()
        state.process({"action": "add", "task": "First"})
        state.process({"action": "add", "task": "Second"})
        result = json.loads(state.process({"action": "clear"}))
        assert result["action"] == "clear"
        assert result["total"] == 0
        assert result["remaining"] == 0
        assert result["items"] == []

    def test_clear_empty_list(self):
        state = TodoState()
        result = json.loads(state.process({"action": "clear"}))
        assert result["total"] == 0


class TestList:
    def test_list_empty(self):
        state = TodoState()
        result = json.loads(state.process({"action": "list"}))
        assert result == {"action": "list", "total": 0, "remaining": 0, "items": []}

    def test_list_with_items(self):
        state = TodoState()
        state.process({"action": "add", "task": "A"})
        state.process({"action": "add", "task": "B"})
        state.process({"action": "done", "task": "A"})
        result = json.loads(state.process({"action": "list"}))
        assert result["total"] == 2
        assert result["remaining"] == 1
        assert result["items"][0] == {"task": "A", "done": True}
        assert result["items"][1] == {"task": "B", "done": False}


# ---------------------------------------------------------------------------
# Matching tests
# ---------------------------------------------------------------------------


class TestMatching:
    def test_exact_match_case_insensitive(self):
        state = TodoState()
        state.process({"action": "add", "task": "Fix the Bug"})
        result = json.loads(state.process({"action": "done", "task": "fix the bug"}))
        assert result["items"][0]["done"] is True

    def test_prefix_match(self):
        state = TodoState()
        state.process({"action": "add", "task": "Fix the bug in login handler"})
        result = json.loads(state.process({"action": "done", "task": "Fix the bug"}))
        assert result["items"][0]["done"] is True

    def test_substring_match(self):
        state = TodoState()
        state.process({"action": "add", "task": "Fix the bug in login handler"})
        result = json.loads(state.process({"action": "done", "task": "login handler"}))
        assert result["items"][0]["done"] is True

    def test_ambiguous_match(self):
        state = TodoState()
        state.process({"action": "add", "task": "Fix bug in login"})
        state.process({"action": "add", "task": "Fix bug in signup"})
        result = state.process({"action": "done", "task": "Fix bug"})
        assert result.startswith("error:")
        assert "matches multiple" in result

    def test_no_match(self):
        state = TodoState()
        state.process({"action": "add", "task": "Something"})
        result = state.process({"action": "done", "task": "Nothing"})
        assert "no task matching" in result

    def test_remove_includes_done_items(self):
        """Remove matches done items too (e.g. wrong item marked done by mistake)."""
        state = TodoState()
        state.process({"action": "add", "task": "Task A"})
        state.process({"action": "done", "task": "Task A"})
        result = json.loads(state.process({"action": "remove", "task": "Task A"}))
        assert result["total"] == 0

    def test_done_matches_already_done_items(self):
        """Done matches even already-completed items (no-op)."""
        state = TodoState()
        state.process({"action": "add", "task": "Task A"})
        state.process({"action": "done", "task": "Task A"})
        # Second done should still find and succeed (no-op)
        result = json.loads(state.process({"action": "done", "task": "Task A"}))
        assert result["items"][0]["done"] is True


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_add_creates_file(self, tmp_path):
        todo_path = tmp_path / ".swival" / "todo.md"
        state = TodoState(notes_dir=str(tmp_path))
        state.process({"action": "add", "task": "Read the test suite"})
        assert todo_path.exists()
        content = todo_path.read_text()
        assert "- [ ] Read the test suite" in content

    def test_done_updates_file(self, tmp_path):
        todo_path = tmp_path / ".swival" / "todo.md"
        state = TodoState(notes_dir=str(tmp_path))
        state.process({"action": "add", "task": "Read the test suite"})
        state.process({"action": "done", "task": "Read the test suite"})
        content = todo_path.read_text()
        assert "- [x] Read the test suite" in content

    def test_clear_empties_file(self, tmp_path):
        todo_path = tmp_path / ".swival" / "todo.md"
        state = TodoState(notes_dir=str(tmp_path))
        state.process({"action": "add", "task": "Task"})
        state.process({"action": "clear"})
        content = todo_path.read_text()
        assert content == ""

    def test_session_isolation(self, tmp_path):
        """New TodoState deletes stale todo file from a prior session."""
        todo_path = tmp_path / ".swival" / "todo.md"
        todo_path.parent.mkdir(parents=True, exist_ok=True)
        todo_path.write_text("- [ ] Old stale task\n")
        # Constructing a new TodoState should delete the old file
        TodoState(notes_dir=str(tmp_path))
        assert not todo_path.exists()

    def test_init_cleanup_ignores_missing_file(self, tmp_path):
        # No .swival/ directory at all — should not raise
        state = TodoState(notes_dir=str(tmp_path))
        assert state.add_count == 0

    def test_no_notes_dir_no_crash(self):
        """Without notes_dir, items work in memory but no file is created."""
        state = TodoState()
        result = json.loads(state.process({"action": "add", "task": "Memory only"}))
        assert result["total"] == 1
        # No file created anywhere

    def test_symlinked_swival_disables_persistence(self, tmp_path):
        """If .swival is a symlink escaping base_dir, persistence is disabled (no crash)."""
        outside = tmp_path / "outside"
        outside.mkdir()
        target_file = outside / "todo.md"
        target_file.write_text("external data")

        scratch = tmp_path / "base" / ".swival"
        scratch.parent.mkdir()
        os.symlink(str(outside), str(scratch))

        # Should not raise — just disables persistence
        state = TodoState(notes_dir=str(tmp_path / "base"))
        assert state.notes_dir is None
        # Items still work in memory
        result = json.loads(state.process({"action": "add", "task": "In-memory only"}))
        assert result["total"] == 1


# ---------------------------------------------------------------------------
# Caps tests
# ---------------------------------------------------------------------------


class TestCaps:
    def test_max_items(self):
        state = TodoState()
        for i in range(MAX_ITEMS):
            result = state.process({"action": "add", "task": f"Task {i}"})
            assert not result.startswith("error:"), f"Task {i} should succeed"
        # 51st item should fail
        result = state.process({"action": "add", "task": "One too many"})
        assert result.startswith("error:")
        assert "full" in result

    def test_text_too_long_returns_error(self):
        state = TodoState()
        result = state.process({"action": "add", "task": "x" * (MAX_ITEM_TEXT + 1)})
        assert result.startswith("error:")
        assert "500" in result
        # Item should not have been added
        assert len(state.items) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_invalid_action(self):
        state = TodoState()
        result = state.process({"action": "invalid"})
        assert result.startswith("error:")

    def test_missing_action(self):
        state = TodoState()
        result = state.process({})
        assert result.startswith("error:")

    def test_remove_without_task(self):
        state = TodoState()
        result = state.process({"action": "remove"})
        assert result.startswith("error:")


# ---------------------------------------------------------------------------
# Usage counters and summary
# ---------------------------------------------------------------------------


class TestUsageCounters:
    def test_add_count(self):
        state = TodoState()
        state.process({"action": "add", "task": "A"})
        state.process({"action": "add", "task": "B"})
        assert state.add_count == 2

    def test_done_count(self):
        state = TodoState()
        state.process({"action": "add", "task": "A"})
        state.process({"action": "done", "task": "A"})
        assert state.done_count == 1

    def test_done_noop_does_not_increment(self):
        state = TodoState()
        state.process({"action": "add", "task": "A"})
        state.process({"action": "done", "task": "A"})
        state.process({"action": "done", "task": "A"})
        assert state.done_count == 1

    def test_summary_line_never_called(self):
        state = TodoState()
        assert state.summary_line() is None

    def test_summary_line(self):
        state = TodoState()
        state.process({"action": "add", "task": "A"})
        state.process({"action": "add", "task": "B"})
        state.process({"action": "add", "task": "C"})
        state.process({"action": "done", "task": "A"})
        assert state.summary_line() == "todo: 3 added, 1 done, 2 remaining"

    def test_reset(self, tmp_path):
        state = TodoState(notes_dir=str(tmp_path))
        state.process({"action": "add", "task": "A"})
        state.process({"action": "done", "task": "A"})
        state.reset()
        assert state.items == []
        assert state.add_count == 0
        assert state.done_count == 0
        assert state._total_actions == 0
        assert state.summary_line() is None
        todo_path = tmp_path / ".swival" / "todo.md"
        assert not todo_path.exists()

    def test_summary_line_all_done(self):
        state = TodoState()
        state.process({"action": "add", "task": "A"})
        state.process({"action": "done", "task": "A"})
        assert state.summary_line() == "todo: 1 added, 1 done, 0 remaining"


# ---------------------------------------------------------------------------
# Dispatch integration
# ---------------------------------------------------------------------------


class TestDispatchIntegration:
    def test_dispatch_todo_add(self, tmp_path):
        state = TodoState()
        result = dispatch(
            "todo",
            {"action": "add", "task": "test"},
            str(tmp_path),
            todo_state=state,
        )
        parsed = json.loads(result)
        assert parsed["total"] == 1

    def test_dispatch_todo_without_state(self, tmp_path):
        result = dispatch(
            "todo",
            {"action": "list"},
            str(tmp_path),
        )
        assert result == "error: todo tool is not available"


# ---------------------------------------------------------------------------
# Verbose logging
# ---------------------------------------------------------------------------


class TestLogging:
    def _reinit_console(self):
        from swival import fmt

        fmt.init(color=False, no_color=False)

    def test_verbose_add_logs(self, capsys):
        self._reinit_console()
        state = TodoState(verbose=True)
        state.process({"action": "add", "task": "Fix the bug"})
        captured = capsys.readouterr()
        assert "[todo +1]" in captured.err
        assert "Fix the bug" in captured.err

    def test_verbose_done_logs(self, capsys):
        self._reinit_console()
        state = TodoState(verbose=True)
        state.process({"action": "add", "task": "Fix the bug"})
        _ = capsys.readouterr()  # discard add output
        state.process({"action": "done", "task": "Fix the bug"})
        captured = capsys.readouterr()
        assert "[todo \u2713]" in captured.err

    def test_verbose_clear_logs(self, capsys):
        self._reinit_console()
        state = TodoState(verbose=True)
        state.process({"action": "add", "task": "A"})
        state.process({"action": "add", "task": "B"})
        _ = capsys.readouterr()
        state.process({"action": "clear"})
        captured = capsys.readouterr()
        assert "[todo cleared]" in captured.err

    def test_quiet_no_stderr(self, capsys):
        self._reinit_console()
        state = TodoState(verbose=False)
        state.process({"action": "add", "task": "Silent task"})
        captured = capsys.readouterr()
        assert captured.err == ""


# ---------------------------------------------------------------------------
# Agent log-skip integration test
# ---------------------------------------------------------------------------


class _FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, name, arguments_json):
        self.id = "call_test"
        self.function = _FakeFunction(name, arguments_json)


class TestAgentLogSkip:
    """Verify that agent.py's handle_tool_call skips generic logging for todo."""

    def test_todo_skips_generic_log(self, tmp_path, monkeypatch):
        from swival import agent
        from swival import fmt
        from swival.thinking import ThinkingState

        calls = []
        monkeypatch.setattr(
            fmt, "tool_call", lambda name, args: calls.append(("tool_call", name))
        )
        monkeypatch.setattr(
            fmt,
            "tool_result",
            lambda name, elapsed, preview: calls.append(("tool_result", name)),
        )
        monkeypatch.setattr(
            fmt, "tool_error", lambda name, msg: calls.append(("tool_error", name))
        )

        thinking_state = ThinkingState(verbose=False)
        todo_state = TodoState(verbose=False)

        tool_call = _FakeToolCall(
            "todo",
            json.dumps({"action": "add", "task": "Test task"}),
        )
        result_msg, _meta = agent.handle_tool_call(
            tool_call,
            str(tmp_path),
            thinking_state,
            verbose=True,
            todo_state=todo_state,
        )
        assert result_msg["role"] == "tool"

        # No fmt.tool_* calls should have been made for todo
        assert not calls, f"unexpected fmt calls for todo: {calls}"


# ---------------------------------------------------------------------------
# Todo reminder tests (agent loop intervention)
# ---------------------------------------------------------------------------


def _make_message(content=None, tool_calls=None):
    msg = types.SimpleNamespace()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.role = "assistant"
    msg.get = lambda key, default=None: getattr(msg, key, default)
    return msg


def _make_tool_call(name, arguments, call_id):
    tc = types.SimpleNamespace()
    tc.id = call_id
    tc.function = types.SimpleNamespace()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _base_args(tmp_path, **overrides):
    defaults = dict(
        base_url="http://fake",
        model="test-model",
        max_output_tokens=1024,
        temperature=0.55,
        top_p=1.0,
        seed=None,
        quiet=False,
        max_turns=10,
        base_dir=str(tmp_path),
        no_system_prompt=True,
        no_instructions=True,
        no_skills=True,
        skills_dir=[],
        system_prompt=None,
        question="test todo reminder",
        repl=False,
        max_context_tokens=None,
        allowed_commands=None,
        add_dir=[],
        add_dir_ro=[],
        provider="lmstudio",
        api_key=None,
        color=False,
        no_color=False,
        yolo=False,
        report=None,
        reviewer=None,
        version=False,
        no_read_guard=False,
        no_history=True,
        init_config=False,
        project=False,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _user_messages(messages):
    """Extract user-role message contents from the messages list."""
    out = []
    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "user":
            content = (
                msg.get("content")
                if isinstance(msg, dict)
                else getattr(msg, "content", "")
            )
            out.append(content)
    return out


class TestTodoReminder:
    """Test that the agent loop injects todo reminders after inactivity."""

    def test_reminder_fires_after_interval(self, tmp_path, monkeypatch):
        """Reminder fires after TODO_REMINDER_INTERVAL turns of non-todo tool use."""
        from swival import agent, fmt

        fmt.init(color=False, no_color=False)

        snapshots = []
        call_count = 0

        # Turn 1: model calls todo add (sets todo_last_used=1)
        # Turn 2-4: model calls read_file (non-todo) — 3 turns of inactivity
        # Turn 4 should trigger a reminder (turns - todo_last_used >= 3)
        # Turn 5: model returns final answer
        def fake_call_llm(*args, **kwargs):
            nonlocal call_count
            snapshots.append(list(args[2]))
            call_count += 1
            if call_count == 1:
                tc = _make_tool_call(
                    "todo",
                    json.dumps({"action": "add", "task": "Implement feature X"}),
                    "call_1",
                )
                return _make_message(tool_calls=[tc]), "tool_calls"
            if call_count <= 4:
                # Create a file so read_file succeeds
                (tmp_path / "test.txt").write_text("hello")
                tc = _make_tool_call(
                    "read_file",
                    json.dumps({"file_path": "test.txt"}),
                    f"call_{call_count}",
                )
                return _make_message(tool_calls=[tc]), "tool_calls"
            return _make_message(content="done"), "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        args = _base_args(tmp_path)
        monkeypatch.setattr(sys, "argv", ["agent", "test todo reminder"])
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)

        agent.main()

        # The reminder should be injected as a user message containing "Reminder:"
        # Check the snapshot seen by turn 5 (the final answer turn)
        all_user_msgs = _user_messages(snapshots[-1])
        reminder_msgs = [
            m for m in all_user_msgs if "Reminder:" in m and "todo" in m.lower()
        ]
        assert len(reminder_msgs) == 1, (
            f"Expected 1 reminder, got {len(reminder_msgs)}: {reminder_msgs}"
        )
        assert "Implement feature X" in reminder_msgs[0]

    def test_no_reminder_when_all_done(self, tmp_path, monkeypatch):
        """No reminder when all todo items are completed."""
        from swival import agent, fmt

        fmt.init(color=False, no_color=False)

        call_count = 0
        snapshots = []

        # Turn 1: add a todo item
        # Turn 2: mark it done
        # Turns 3-5: non-todo tool use — should NOT trigger reminder
        # Turn 6: final answer
        def fake_call_llm(*args, **kwargs):
            nonlocal call_count
            snapshots.append(list(args[2]))
            call_count += 1
            if call_count == 1:
                tc = _make_tool_call(
                    "todo",
                    json.dumps({"action": "add", "task": "Task A"}),
                    "call_1",
                )
                return _make_message(tool_calls=[tc]), "tool_calls"
            if call_count == 2:
                tc = _make_tool_call(
                    "todo",
                    json.dumps({"action": "done", "task": "Task A"}),
                    "call_2",
                )
                return _make_message(tool_calls=[tc]), "tool_calls"
            if call_count <= 5:
                (tmp_path / "test.txt").write_text("hello")
                tc = _make_tool_call(
                    "read_file",
                    json.dumps({"file_path": "test.txt"}),
                    f"call_{call_count}",
                )
                return _make_message(tool_calls=[tc]), "tool_calls"
            return _make_message(content="done"), "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        args = _base_args(tmp_path)
        monkeypatch.setattr(sys, "argv", ["agent", "test"])
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)

        agent.main()

        all_user_msgs = _user_messages(snapshots[-1])
        reminder_msgs = [
            m for m in all_user_msgs if "Reminder:" in m and "todo" in m.lower()
        ]
        assert len(reminder_msgs) == 0, f"Expected no reminders, got: {reminder_msgs}"

    def test_no_reminder_within_interval(self, tmp_path, monkeypatch):
        """No reminder when todo was used recently (within interval)."""
        from swival import agent, fmt

        fmt.init(color=False, no_color=False)

        call_count = 0
        snapshots = []

        # Turn 1: add a todo item
        # Turn 2-3: non-todo tool use (only 2 turns of inactivity, < interval of 3)
        # Turn 4: final answer — no reminder expected
        def fake_call_llm(*args, **kwargs):
            nonlocal call_count
            snapshots.append(list(args[2]))
            call_count += 1
            if call_count == 1:
                tc = _make_tool_call(
                    "todo",
                    json.dumps({"action": "add", "task": "Task B"}),
                    "call_1",
                )
                return _make_message(tool_calls=[tc]), "tool_calls"
            if call_count <= 3:
                (tmp_path / "test.txt").write_text("hello")
                tc = _make_tool_call(
                    "read_file",
                    json.dumps({"file_path": "test.txt"}),
                    f"call_{call_count}",
                )
                return _make_message(tool_calls=[tc]), "tool_calls"
            return _make_message(content="done"), "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        args = _base_args(tmp_path)
        monkeypatch.setattr(sys, "argv", ["agent", "test"])
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)

        agent.main()

        all_user_msgs = _user_messages(snapshots[-1])
        reminder_msgs = [
            m for m in all_user_msgs if "Reminder:" in m and "todo" in m.lower()
        ]
        assert len(reminder_msgs) == 0, f"Expected no reminders, got: {reminder_msgs}"

    def test_reminder_resets_interval(self, tmp_path, monkeypatch):
        """After a reminder fires, the interval resets (no back-to-back reminders)."""
        from swival import agent, fmt

        fmt.init(color=False, no_color=False)

        call_count = 0
        snapshots = []

        # Turn 1: add a todo item (todo_last_used=1)
        # Turns 2-4: non-todo (fires reminder at turn 4, resets todo_last_used=4)
        # Turn 5: non-todo (only 1 turn since reset, no reminder)
        # Turn 6: final answer
        def fake_call_llm(*args, **kwargs):
            nonlocal call_count
            snapshots.append(list(args[2]))
            call_count += 1
            if call_count == 1:
                tc = _make_tool_call(
                    "todo",
                    json.dumps({"action": "add", "task": "Task C"}),
                    "call_1",
                )
                return _make_message(tool_calls=[tc]), "tool_calls"
            if call_count <= 5:
                (tmp_path / "test.txt").write_text("hello")
                tc = _make_tool_call(
                    "read_file",
                    json.dumps({"file_path": "test.txt"}),
                    f"call_{call_count}",
                )
                return _make_message(tool_calls=[tc]), "tool_calls"
            return _make_message(content="done"), "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        args = _base_args(tmp_path)
        monkeypatch.setattr(sys, "argv", ["agent", "test"])
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)

        agent.main()

        all_user_msgs = _user_messages(snapshots[-1])
        reminder_msgs = [
            m for m in all_user_msgs if "Reminder:" in m and "todo" in m.lower()
        ]
        # Should have exactly 1 reminder (at turn 4), not 2
        assert len(reminder_msgs) == 1, (
            f"Expected 1 reminder, got {len(reminder_msgs)}: {reminder_msgs}"
        )

"""Tests for the think tool: ThinkingState, dispatch integration, and agent logging."""

import json
import os
import pathlib
import tempfile

import pytest

from swival.thinking import ThinkingState, MAX_NOTES, MAX_NOTE_LENGTH
from swival.tools import dispatch


# ---------------------------------------------------------------------------
# ThinkingState.process() unit tests
# ---------------------------------------------------------------------------


class TestLinearFlow:
    def test_sequential_thoughts(self):
        state = ThinkingState()
        for i in range(1, 4):
            result = json.loads(
                state.process(
                    {
                        "thought": f"Step {i}",
                        "thought_number": i,
                        "total_thoughts": 3,
                        "next_thought_needed": i < 3,
                    }
                )
            )
            assert result["thought_number"] == i
            assert result["history_length"] == i
            assert result["total_thoughts"] == 3
        assert result["next_thought_needed"] is False


class TestRevision:
    def test_valid_revision(self):
        state = ThinkingState()
        state.process(
            {
                "thought": "First thought",
                "thought_number": 1,
                "total_thoughts": 3,
                "next_thought_needed": True,
            }
        )
        result = json.loads(
            state.process(
                {
                    "thought": "Correcting step 1",
                    "thought_number": 2,
                    "total_thoughts": 3,
                    "next_thought_needed": True,
                    "is_revision": True,
                    "revises_thought": 1,
                }
            )
        )
        assert result["history_length"] == 2

    def test_is_revision_without_revises_thought(self):
        state = ThinkingState()
        state.process(
            {
                "thought": "First",
                "thought_number": 1,
                "total_thoughts": 2,
                "next_thought_needed": True,
            }
        )
        result = state.process(
            {
                "thought": "Bad revision",
                "thought_number": 2,
                "total_thoughts": 2,
                "next_thought_needed": True,
                "is_revision": True,
            }
        )
        assert result == "error: is_revision requires revises_thought"

    def test_revises_thought_without_is_revision(self):
        state = ThinkingState()
        state.process(
            {
                "thought": "First",
                "thought_number": 1,
                "total_thoughts": 2,
                "next_thought_needed": True,
            }
        )
        result = state.process(
            {
                "thought": "Bad",
                "thought_number": 2,
                "total_thoughts": 2,
                "next_thought_needed": True,
                "revises_thought": 1,
            }
        )
        assert result == "error: revises_thought requires is_revision=true"

    def test_revises_nonexistent_thought(self):
        state = ThinkingState()
        state.process(
            {
                "thought": "First",
                "thought_number": 1,
                "total_thoughts": 2,
                "next_thought_needed": True,
            }
        )
        result = state.process(
            {
                "thought": "Bad",
                "thought_number": 2,
                "total_thoughts": 2,
                "next_thought_needed": True,
                "is_revision": True,
                "revises_thought": 5,
            }
        )
        assert result == "error: revises_thought=5 not found in history"

    def test_revision_with_nonsequential_numbers(self):
        """thought_number=5 exists in history; revising it should work."""
        state = ThinkingState()
        state.process(
            {
                "thought": "Jump to 5",
                "thought_number": 5,
                "total_thoughts": 10,
                "next_thought_needed": True,
            }
        )
        # Revising thought 5 should succeed
        result = json.loads(
            state.process(
                {
                    "thought": "Fix thought 5",
                    "thought_number": 6,
                    "total_thoughts": 10,
                    "next_thought_needed": True,
                    "is_revision": True,
                    "revises_thought": 5,
                }
            )
        )
        assert result["history_length"] == 2
        # Revising thought 1 (never recorded) should fail
        result = state.process(
            {
                "thought": "Bad ref",
                "thought_number": 7,
                "total_thoughts": 10,
                "next_thought_needed": True,
                "is_revision": True,
                "revises_thought": 1,
            }
        )
        assert result == "error: revises_thought=1 not found in history"


class TestBranching:
    def test_valid_branch(self):
        state = ThinkingState()
        state.process(
            {
                "thought": "First",
                "thought_number": 1,
                "total_thoughts": 3,
                "next_thought_needed": True,
            }
        )
        result = json.loads(
            state.process(
                {
                    "thought": "Alternative approach",
                    "thought_number": 2,
                    "total_thoughts": 3,
                    "next_thought_needed": True,
                    "branch_from_thought": 1,
                    "branch_id": "approach-b",
                }
            )
        )
        assert "approach-b" in result["branches"]

    def test_branch_from_without_branch_id(self):
        state = ThinkingState()
        state.process(
            {
                "thought": "First",
                "thought_number": 1,
                "total_thoughts": 2,
                "next_thought_needed": True,
            }
        )
        result = state.process(
            {
                "thought": "Bad",
                "thought_number": 2,
                "total_thoughts": 2,
                "next_thought_needed": True,
                "branch_from_thought": 1,
            }
        )
        assert result == "error: branch_from_thought requires branch_id"

    def test_branch_id_without_branch_from(self):
        state = ThinkingState()
        state.process(
            {
                "thought": "First",
                "thought_number": 1,
                "total_thoughts": 2,
                "next_thought_needed": True,
            }
        )
        result = state.process(
            {
                "thought": "Bad",
                "thought_number": 2,
                "total_thoughts": 2,
                "next_thought_needed": True,
                "branch_id": "orphan",
            }
        )
        assert result == "error: branch_id requires branch_from_thought"

    def test_branch_from_nonexistent_thought(self):
        state = ThinkingState()
        state.process(
            {
                "thought": "First",
                "thought_number": 1,
                "total_thoughts": 2,
                "next_thought_needed": True,
            }
        )
        result = state.process(
            {
                "thought": "Bad",
                "thought_number": 2,
                "total_thoughts": 2,
                "next_thought_needed": True,
                "branch_from_thought": 99,
                "branch_id": "bad",
            }
        )
        assert result == "error: branch_from_thought=99 not found in history"

    def test_branch_with_nonsequential_numbers(self):
        """branch_from_thought should match actual thought_number, not index."""
        state = ThinkingState()
        state.process(
            {
                "thought": "Jump to 10",
                "thought_number": 10,
                "total_thoughts": 20,
                "next_thought_needed": True,
            }
        )
        # Branching from thought 10 should work
        result = json.loads(
            state.process(
                {
                    "thought": "Alt from 10",
                    "thought_number": 11,
                    "total_thoughts": 20,
                    "next_thought_needed": True,
                    "branch_from_thought": 10,
                    "branch_id": "alt",
                }
            )
        )
        assert "alt" in result["branches"]
        # Branching from thought 1 (never recorded) should fail
        result = state.process(
            {
                "thought": "Bad branch",
                "thought_number": 12,
                "total_thoughts": 20,
                "next_thought_needed": True,
                "branch_from_thought": 1,
                "branch_id": "bad",
            }
        )
        assert result == "error: branch_from_thought=1 not found in history"

    def test_branch_id_too_long(self):
        state = ThinkingState()
        state.process(
            {
                "thought": "First",
                "thought_number": 1,
                "total_thoughts": 2,
                "next_thought_needed": True,
            }
        )
        result = state.process(
            {
                "thought": "Bad",
                "thought_number": 2,
                "total_thoughts": 2,
                "next_thought_needed": True,
                "branch_from_thought": 1,
                "branch_id": "x" * 51,
            }
        )
        assert result == "error: branch_id exceeds 50 character limit"

    def test_blank_branch_id(self):
        state = ThinkingState()
        state.process(
            {
                "thought": "First",
                "thought_number": 1,
                "total_thoughts": 2,
                "next_thought_needed": True,
            }
        )
        result = state.process(
            {
                "thought": "Bad",
                "thought_number": 2,
                "total_thoughts": 2,
                "next_thought_needed": True,
                "branch_from_thought": 1,
                "branch_id": "   ",
            }
        )
        assert result == "error: branch_id must not be blank"

    def test_too_many_branches(self):
        state = ThinkingState()
        # Create 20 branches
        for i in range(1, 22):
            state.process(
                {
                    "thought": f"Thought {i}",
                    "thought_number": i,
                    "total_thoughts": 25,
                    "next_thought_needed": True,
                }
            )
        # Now branch from existing thoughts
        for i in range(20):
            result = state.process(
                {
                    "thought": f"Branch {i}",
                    "thought_number": 22 + i,
                    "total_thoughts": 50,
                    "next_thought_needed": True,
                    "branch_from_thought": 1,
                    "branch_id": f"branch-{i}",
                }
            )
            assert not result.startswith("error"), (
                f"Branch {i} should succeed: {result}"
            )
        # 21st branch should fail
        result = state.process(
            {
                "thought": "One too many",
                "thought_number": 42,
                "total_thoughts": 50,
                "next_thought_needed": True,
                "branch_from_thought": 1,
                "branch_id": "branch-20",
            }
        )
        assert result == "error: too many branches (20 max)"


class TestAutoAdjust:
    def test_thought_number_exceeds_total(self):
        state = ThinkingState()
        result = json.loads(
            state.process(
                {
                    "thought": "Overshot",
                    "thought_number": 7,
                    "total_thoughts": 3,
                    "next_thought_needed": True,
                }
            )
        )
        assert result["total_thoughts"] == 7


class TestTruncation:
    def test_long_thought_truncated(self):
        state = ThinkingState()
        long_text = "x" * 15000
        result = json.loads(
            state.process(
                {
                    "thought": long_text,
                    "thought_number": 1,
                    "total_thoughts": 1,
                    "next_thought_needed": False,
                }
            )
        )
        assert result["history_length"] == 1
        assert len(state.history[0].thought) == 10000


class TestHistoryCap:
    def test_201st_thought_rejected(self):
        state = ThinkingState()
        for i in range(1, 201):
            result = state.process(
                {
                    "thought": f"Step {i}",
                    "thought_number": i,
                    "total_thoughts": 250,
                    "next_thought_needed": True,
                }
            )
            assert not result.startswith("error"), f"Step {i} should succeed"
        result = state.process(
            {
                "thought": "One too many",
                "thought_number": 201,
                "total_thoughts": 250,
                "next_thought_needed": True,
            }
        )
        assert result == "error: thinking history full (200 steps max)"


class TestLogging:
    def _reinit_console(self):
        """Reinitialize fmt console so it uses the current (capsys-patched) stderr."""
        from swival import fmt

        fmt.init(color=False, no_color=False)

    def test_verbose_logs_to_stderr(self, capsys):
        self._reinit_console()
        state = ThinkingState(verbose=True)
        state.process(
            {
                "thought": "Checking edge cases",
                "thought_number": 1,
                "total_thoughts": 3,
                "next_thought_needed": True,
            }
        )
        captured = capsys.readouterr()
        assert "[think 1/3]" in captured.err
        assert "Checking edge cases" in captured.err

    def test_quiet_no_stderr(self, capsys):
        self._reinit_console()
        state = ThinkingState(verbose=False)
        state.process(
            {
                "thought": "Silent thought",
                "thought_number": 1,
                "total_thoughts": 1,
                "next_thought_needed": False,
            }
        )
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_newline_normalization(self, capsys):
        self._reinit_console()
        state = ThinkingState(verbose=True)
        state.process(
            {
                "thought": "Line one\nLine two\n\nLine four",
                "thought_number": 1,
                "total_thoughts": 1,
                "next_thought_needed": False,
            }
        )
        captured = capsys.readouterr()
        lines = captured.err.strip().split("\n")
        assert len(lines) == 1
        assert "Line one Line two Line four" in lines[0]

    def test_revision_log_format(self, capsys):
        self._reinit_console()
        state = ThinkingState(verbose=True)
        state.process(
            {
                "thought": "First",
                "thought_number": 1,
                "total_thoughts": 3,
                "next_thought_needed": True,
            }
        )
        state.process(
            {
                "thought": "Revised first",
                "thought_number": 2,
                "total_thoughts": 3,
                "next_thought_needed": True,
                "is_revision": True,
                "revises_thought": 1,
            }
        )
        captured = capsys.readouterr()
        assert "rev:1" in captured.err

    def test_branch_log_format(self, capsys):
        self._reinit_console()
        state = ThinkingState(verbose=True)
        state.process(
            {
                "thought": "Main line",
                "thought_number": 1,
                "total_thoughts": 3,
                "next_thought_needed": True,
            }
        )
        state.process(
            {
                "thought": "Alternative",
                "thought_number": 2,
                "total_thoughts": 3,
                "next_thought_needed": True,
                "branch_from_thought": 1,
                "branch_id": "alt",
            }
        )
        captured = capsys.readouterr()
        assert "branch:alt" in captured.err
        assert "from:1" in captured.err


# ---------------------------------------------------------------------------
# Dispatch integration / regression tests
# ---------------------------------------------------------------------------


class TestDispatchRegression:
    """Verify existing tools still work with the new **kwargs signature."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create a test file
        import os

        with open(os.path.join(self.tmpdir, "hello.txt"), "w") as f:
            f.write("Hello world\n")

    def test_read_file_no_kwargs(self):
        result = dispatch("read_file", {"file_path": "hello.txt"}, self.tmpdir)
        assert "Hello world" in result

    def test_write_file_no_kwargs(self):
        result = dispatch(
            "write_file", {"file_path": "new.txt", "content": "test"}, self.tmpdir
        )
        assert "Wrote" in result

    def test_grep_no_kwargs(self):
        result = dispatch("grep", {"pattern": "Hello"}, self.tmpdir)
        assert "Hello" in result

    def test_think_without_state(self):
        result = dispatch(
            "think",
            {
                "thought": "test",
                "thought_number": 1,
                "total_thoughts": 1,
                "next_thought_needed": False,
            },
            self.tmpdir,
        )
        assert result == "error: think tool is not available"

    def test_think_with_state(self):
        state = ThinkingState()
        result = dispatch(
            "think",
            {
                "thought": "test",
                "thought_number": 1,
                "total_thoughts": 1,
                "next_thought_needed": False,
            },
            self.tmpdir,
            thinking_state=state,
        )
        parsed = json.loads(result)
        assert parsed["history_length"] == 1


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
    """Verify that agent.py's handle_tool_call skips generic logging for think.

    Monkeypatches fmt.tool_call / fmt.tool_result / fmt.tool_error to track
    which tool names are logged. Think calls should be skipped.
    """

    def test_think_skips_generic_log(self, monkeypatch):
        from swival import agent
        from swival import fmt

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
        base_dir = tempfile.mkdtemp()

        # Call the real handle_tool_call for a "think" tool call
        tool_call = _FakeToolCall(
            "think",
            json.dumps(
                {
                    "thought": "Planning step",
                    "thought_number": 1,
                    "total_thoughts": 2,
                    "next_thought_needed": True,
                }
            ),
        )
        result_msg, _meta = agent.handle_tool_call(
            tool_call, base_dir, thinking_state, verbose=True
        )
        assert result_msg["role"] == "tool"

        # No fmt.tool_* calls should have been made for think
        assert not calls, f"unexpected fmt calls for think: {calls}"

        # Now call handle_tool_call for "read_file" — generic logging should appear
        calls.clear()
        import os

        with open(os.path.join(base_dir, "test.txt"), "w") as f:
            f.write("hello\n")

        tool_call = _FakeToolCall("read_file", json.dumps({"file_path": "test.txt"}))
        result_msg, _meta = agent.handle_tool_call(
            tool_call, base_dir, thinking_state, verbose=True
        )
        assert result_msg["role"] == "tool"

        tool_call_names = [name for action, name in calls if action == "tool_call"]
        tool_result_names = [name for action, name in calls if action == "tool_result"]
        assert "read_file" in tool_call_names, f"missing tool_call: {calls}"
        assert "read_file" in tool_result_names, f"missing tool_result: {calls}"


# ---------------------------------------------------------------------------
# Persistent notes tests
# ---------------------------------------------------------------------------


def _thought(n, total=5, note=None, branch_id=None, branch_from=None):
    """Helper to build a thought args dict."""
    args = {
        "thought": f"Thought {n}",
        "thought_number": n,
        "total_thoughts": total,
        "next_thought_needed": n < total,
    }
    if note is not None:
        args["note"] = note
    if branch_id is not None:
        args["branch_id"] = branch_id
        args["branch_from_thought"] = branch_from
    return args


class TestNotes:
    """Tests for the persistent notes feature."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.notes_path = pathlib.Path(self.tmpdir) / ".swival" / "notes.md"

    # -- Core behavior --

    def test_note_creates_file(self):
        state = ThinkingState(notes_dir=self.tmpdir)
        state.process(_thought(1, note="Key finding: X is true"))
        assert self.notes_path.exists()
        content = self.notes_path.read_text()
        assert "Key finding: X is true" in content
        assert "## Note 1 (thought 1)" in content

    def test_note_response_fields(self):
        state = ThinkingState(notes_dir=self.tmpdir)
        result = json.loads(state.process(_thought(1, note="save this")))
        assert result["note_saved"] is True
        assert result["notes_file"] == ".swival/notes.md"
        assert "note_error" not in result

    def test_multiple_notes_append(self):
        state = ThinkingState(notes_dir=self.tmpdir)
        state.process(_thought(1, note="First note"))
        state.process(_thought(2, note="Second note"))
        state.process(_thought(3, note="Third note"))
        content = self.notes_path.read_text()
        assert "## Note 1 (thought 1)" in content
        assert "## Note 2 (thought 2)" in content
        assert "## Note 3 (thought 3)" in content
        assert (
            content.index("First note")
            < content.index("Second note")
            < content.index("Third note")
        )

    def test_note_with_branch_context(self):
        state = ThinkingState(notes_dir=self.tmpdir)
        state.process(_thought(1))
        state.process(
            _thought(2, note="Branch note", branch_id="approach-b", branch_from=1)
        )
        content = self.notes_path.read_text()
        assert "## Note 1 (thought 2, branch: approach-b)" in content

    # -- Limits --

    def test_note_truncation(self):
        state = ThinkingState(notes_dir=self.tmpdir)
        long_note = "x" * (MAX_NOTE_LENGTH + 1000)
        state.process(_thought(1, note=long_note))
        content = self.notes_path.read_text()
        # Header + newline + note text + newline + blank line
        note_body = content.split("\n", 1)[1].strip()
        assert len(note_body) == MAX_NOTE_LENGTH

    def test_note_cap(self):
        state = ThinkingState(notes_dir=self.tmpdir)
        for i in range(1, MAX_NOTES + 1):
            result = json.loads(
                state.process(_thought(i, total=MAX_NOTES + 5, note=f"Note {i}"))
            )
            assert result["note_saved"] is True
        # 51st note should fail
        result = json.loads(
            state.process(
                _thought(MAX_NOTES + 1, total=MAX_NOTES + 5, note="One too many")
            )
        )
        assert result["note_saved"] is False
        assert result["note_error"] == "cap_exceeded"
        # But the thought itself still records
        assert result["history_length"] == MAX_NOTES + 1

    # -- No-op cases --

    def test_no_note_no_file(self):
        state = ThinkingState(notes_dir=self.tmpdir)
        result = json.loads(state.process(_thought(1)))
        assert "note_saved" not in result
        assert not self.notes_path.exists()

    def test_empty_note_ignored(self):
        state = ThinkingState(notes_dir=self.tmpdir)
        for empty in ["", "   ", "\n\t "]:
            result = json.loads(state.process(_thought(1, note=empty)))
            assert "note_saved" not in result
        assert not self.notes_path.exists()

    def test_notes_without_notes_dir(self):
        state = ThinkingState()  # no notes_dir
        result = json.loads(state.process(_thought(1, note="Try to save")))
        assert result["note_saved"] is False
        assert result["note_error"] == "no_notes_dir"

    # -- Session isolation --

    def test_preexisting_notes_file_cleared_on_init(self):
        self.notes_path.parent.mkdir(parents=True, exist_ok=True)
        self.notes_path.write_text("## Note 1 (thought 1)\nOld stale content\n\n")
        # Constructing a new ThinkingState should delete the old file
        ThinkingState(notes_dir=self.tmpdir)
        assert not self.notes_path.exists()

    def test_init_cleanup_ignores_missing_file(self):
        # No .swival/ directory at all — should not raise
        state = ThinkingState(notes_dir=self.tmpdir)
        assert state.note_count == 0

    def test_two_notes_both_present(self):
        state = ThinkingState(notes_dir=self.tmpdir)
        state.process(_thought(1, note="Alpha"))
        state.process(_thought(2, note="Beta"))
        content = self.notes_path.read_text()
        assert "Alpha" in content
        assert "Beta" in content

    # -- Failure modes --

    def test_disk_write_failure(self, monkeypatch):
        state = ThinkingState(notes_dir=self.tmpdir)
        resolved_notes = (pathlib.Path(self.tmpdir) / ".swival" / "notes.md").resolve()

        original_open = pathlib.Path.open

        def failing_open(self_path, *args, **kwargs):
            if self_path.resolve() == resolved_notes:
                raise OSError("simulated disk failure")
            return original_open(self_path, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "open", failing_open)

        result = json.loads(state.process(_thought(1, note="Should fail")))
        assert result["note_saved"] is False
        assert result["note_error"] == "write_failed"
        # Thought still recorded
        assert result["history_length"] == 1
        # note_count should not have incremented
        assert state.note_count == 0

    def test_init_unlink_oserror_propagates(self, monkeypatch):
        # Create the notes file so unlink is attempted
        self.notes_path.parent.mkdir(parents=True, exist_ok=True)
        self.notes_path.write_text("stale")
        resolved_notes = self.notes_path.resolve()

        original_unlink = pathlib.Path.unlink

        def failing_unlink(self_path, *args, **kwargs):
            if self_path.resolve() == resolved_notes:
                raise PermissionError("simulated permission denied")
            return original_unlink(self_path, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "unlink", failing_unlink)

        with pytest.raises(PermissionError, match="simulated permission denied"):
            ThinkingState(notes_dir=self.tmpdir)

    # -- Symlink escape --

    def test_symlinked_swival_blocked_on_init(self):
        """If .swival is a symlink pointing outside base_dir, init raises."""
        outside = tempfile.mkdtemp()
        target_file = pathlib.Path(outside) / "notes.md"
        target_file.write_text("external data")

        scratch = pathlib.Path(self.tmpdir) / ".swival"
        os.symlink(outside, scratch)

        # notes.md resolves outside tmpdir — _safe_notes_path should raise
        with pytest.raises(ValueError, match="escapes base directory"):
            ThinkingState(notes_dir=self.tmpdir)

    def test_symlinked_swival_blocked_on_write(self):
        """If .swival becomes a symlink after init, note write returns write_failed."""
        state = ThinkingState(notes_dir=self.tmpdir)

        outside = tempfile.mkdtemp()
        scratch = pathlib.Path(self.tmpdir) / ".swival"
        scratch.mkdir(exist_ok=True)
        # Replace .swival with a symlink to outside
        scratch.rmdir()
        os.symlink(outside, scratch)

        result = json.loads(state.process(_thought(1, note="Escape attempt")))
        assert result["note_saved"] is False
        assert result["note_error"] == "write_failed"
        # Thought still recorded
        assert result["history_length"] == 1

    # -- Unicode encoding --

    def test_note_with_non_ascii_content(self):
        """Notes with non-ASCII chars write correctly with explicit UTF-8."""
        state = ThinkingState(notes_dir=self.tmpdir)
        note = "Ünïcödé: 日本語 emoji 🎉"
        result = json.loads(state.process(_thought(1, note=note)))
        assert result["note_saved"] is True
        content = self.notes_path.read_text(encoding="utf-8")
        assert note in content

    # -- Backward compatibility --

    def test_constructor_defaults(self):
        s1 = ThinkingState()
        assert s1.notes_dir is None
        assert s1.note_count == 0

        s2 = ThinkingState(verbose=True)
        assert s2.notes_dir is None
        assert s2.verbose is True

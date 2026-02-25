"""Tests for the read-before-write guard (FileAccessTracker)."""

import os

import pytest

from swival.tracker import FileAccessTracker
from swival.tools import _read_file, _write_file, _edit_file, dispatch
from swival.agent import _repl_clear, compact_messages, drop_middle_turns
from swival.thinking import ThinkingState


# =========================================================================
# Unit tests for FileAccessTracker
# =========================================================================


class TestFileAccessTracker:
    def test_new_file_allowed(self):
        t = FileAccessTracker()
        assert t.check_write_allowed("/tmp/new.txt", exists=False) is None

    def test_existing_file_blocked_without_read(self):
        t = FileAccessTracker()
        result = t.check_write_allowed("/tmp/old.txt", exists=True)
        assert result is not None
        assert result.startswith("error:")

    def test_existing_file_allowed_after_read(self):
        t = FileAccessTracker()
        t.record_read("/tmp/old.txt")
        assert t.check_write_allowed("/tmp/old.txt", exists=True) is None

    def test_existing_file_allowed_after_write(self):
        t = FileAccessTracker()
        t.record_write("/tmp/old.txt")
        assert t.check_write_allowed("/tmp/old.txt", exists=True) is None

    def test_reset_clears_state(self):
        t = FileAccessTracker()
        t.record_read("/a")
        t.record_write("/b")
        t.reset()
        assert t.check_write_allowed("/a", exists=True) is not None
        assert t.check_write_allowed("/b", exists=True) is not None


# =========================================================================
# Tool-level tests: _write_file
# =========================================================================


class TestWriteFileGuard:
    def test_write_new_file_no_prior_read(self, tmp_path):
        """Creating a new file should always succeed."""
        tracker = FileAccessTracker()
        result = _write_file("new.txt", "hello", str(tmp_path), tracker=tracker)
        assert result.startswith("Wrote")
        assert (tmp_path / "new.txt").read_text() == "hello"

    def test_write_existing_file_blocked(self, tmp_path):
        """Overwriting an existing file without reading it first should fail."""
        (tmp_path / "exist.txt").write_text("original")
        tracker = FileAccessTracker()
        result = _write_file("exist.txt", "overwrite", str(tmp_path), tracker=tracker)
        assert result.startswith("error:")
        assert "hasn't been read" in result
        # File should be unchanged
        assert (tmp_path / "exist.txt").read_text() == "original"

    def test_write_existing_file_after_read(self, tmp_path):
        """Overwriting an existing file after reading it should succeed."""
        (tmp_path / "exist.txt").write_text("original")
        tracker = FileAccessTracker()
        _read_file("exist.txt", str(tmp_path), tracker=tracker)
        result = _write_file("exist.txt", "updated", str(tmp_path), tracker=tracker)
        assert result.startswith("Wrote")
        assert (tmp_path / "exist.txt").read_text() == "updated"

    def test_rewrite_file_created_in_session(self, tmp_path):
        """A file created by write_file should be re-writable without a read."""
        tracker = FileAccessTracker()
        _write_file("new.txt", "v1", str(tmp_path), tracker=tracker)
        result = _write_file("new.txt", "v2", str(tmp_path), tracker=tracker)
        assert result.startswith("Wrote")
        assert (tmp_path / "new.txt").read_text() == "v2"

    def test_tracker_none_allows_all(self, tmp_path):
        """When tracker is None (guard disabled), all writes succeed."""
        (tmp_path / "exist.txt").write_text("original")
        result = _write_file("exist.txt", "overwrite", str(tmp_path), tracker=None)
        assert result.startswith("Wrote")


# =========================================================================
# Tool-level tests: _edit_file
# =========================================================================


class TestEditFileGuard:
    def test_edit_existing_file_blocked(self, tmp_path):
        """Editing an existing file without reading it first should fail."""
        (tmp_path / "exist.txt").write_text("hello world")
        tracker = FileAccessTracker()
        result = _edit_file(
            "exist.txt", "hello", "goodbye", str(tmp_path), tracker=tracker
        )
        assert result.startswith("error:")
        assert "hasn't been read" in result
        # File should be unchanged
        assert (tmp_path / "exist.txt").read_text() == "hello world"

    def test_edit_existing_file_after_read(self, tmp_path):
        """Editing an existing file after reading it should succeed."""
        (tmp_path / "exist.txt").write_text("hello world")
        tracker = FileAccessTracker()
        _read_file("exist.txt", str(tmp_path), tracker=tracker)
        result = _edit_file(
            "exist.txt", "hello", "goodbye", str(tmp_path), tracker=tracker
        )
        assert result == "Edited exist.txt"
        assert (tmp_path / "exist.txt").read_text() == "goodbye world"

    def test_edit_tracker_none_allows_all(self, tmp_path):
        """When tracker is None, edits proceed without read check."""
        (tmp_path / "exist.txt").write_text("hello world")
        result = _edit_file(
            "exist.txt", "hello", "goodbye", str(tmp_path), tracker=None
        )
        assert result == "Edited exist.txt"


# =========================================================================
# Path resolution tests
# =========================================================================


class TestPathResolution:
    def test_read_relative_write_absolute(self, tmp_path):
        """Reading via relative path and writing via absolute should match."""
        (tmp_path / "file.txt").write_text("content")
        tracker = FileAccessTracker()
        # Read via relative path (resolved inside _read_file)
        _read_file("file.txt", str(tmp_path), tracker=tracker)
        # Write via absolute path
        abs_path = str(tmp_path / "file.txt")
        result = _write_file(abs_path, "new", str(tmp_path), tracker=tracker)
        assert result.startswith("Wrote")

    def test_read_absolute_write_relative(self, tmp_path):
        """Reading via absolute path and writing via relative should match."""
        (tmp_path / "file.txt").write_text("content")
        tracker = FileAccessTracker()
        abs_path = str(tmp_path / "file.txt")
        _read_file(abs_path, str(tmp_path), tracker=tracker)
        result = _write_file("file.txt", "new", str(tmp_path), tracker=tracker)
        assert result.startswith("Wrote")


# =========================================================================
# Read registration edge cases
# =========================================================================


class TestReadRegistration:
    def test_directory_read_does_not_register(self, tmp_path):
        """Reading a directory should not register any file as read."""
        (tmp_path / "child.txt").write_text("data")
        tracker = FileAccessTracker()
        _read_file(".", str(tmp_path), tracker=tracker)
        # child.txt was listed but not read — writing it should be blocked
        result = _write_file("child.txt", "new", str(tmp_path), tracker=tracker)
        assert result.startswith("error:")

    def test_offset_past_eof_does_not_register(self, tmp_path):
        """read_file with offset past EOF returns no content — should not register."""
        (tmp_path / "short.txt").write_text("one\ntwo\n")
        tracker = FileAccessTracker()
        result = _read_file("short.txt", str(tmp_path), offset=999, tracker=tracker)
        # No actual lines shown
        assert str(tmp_path / "short.txt") not in tracker.read_files
        # Write should be blocked
        result = _write_file("short.txt", "new", str(tmp_path), tracker=tracker)
        assert result.startswith("error:")

    def test_limit_zero_does_not_register(self, tmp_path):
        """read_file with limit=0 returns no content — should not register."""
        (tmp_path / "data.txt").write_text("content here\n")
        tracker = FileAccessTracker()
        result = _read_file("data.txt", str(tmp_path), limit=0, tracker=tracker)
        assert str(tmp_path / "data.txt") not in tracker.read_files
        # Write should be blocked
        result = _write_file("data.txt", "new", str(tmp_path), tracker=tracker)
        assert result.startswith("error:")

    def test_empty_file_registers(self, tmp_path):
        """read_file on a genuinely empty file should register (model saw the truth)."""
        (tmp_path / "empty.txt").write_text("")
        tracker = FileAccessTracker()
        _read_file("empty.txt", str(tmp_path), tracker=tracker)
        assert str(tmp_path / "empty.txt") in tracker.read_files
        # Write should succeed
        result = _write_file("empty.txt", "now has content", str(tmp_path), tracker=tracker)
        assert result.startswith("Wrote")

    def test_binary_file_read_does_not_register(self, tmp_path):
        """Attempting to read a binary file should not register it."""
        binfile = tmp_path / "data.bin"
        binfile.write_bytes(b"\x00\x01\x02\x03")
        tracker = FileAccessTracker()
        result = _read_file("data.bin", str(tmp_path), tracker=tracker)
        assert "binary" in result
        # File was not successfully read — writing should be blocked
        result = _write_file("data.bin", "text", str(tmp_path), tracker=tracker)
        assert result.startswith("error:")
        assert "hasn't been read" in result


# =========================================================================
# Dispatch integration
# =========================================================================


class TestDispatchIntegration:
    def test_dispatch_passes_tracker_to_read(self, tmp_path):
        """dispatch('read_file') should pass the tracker through."""
        (tmp_path / "f.txt").write_text("hi")
        tracker = FileAccessTracker()
        dispatch("read_file", {"file_path": "f.txt"}, str(tmp_path), file_tracker=tracker)
        assert str(tmp_path / "f.txt") in tracker.read_files

    def test_dispatch_passes_tracker_to_write(self, tmp_path):
        """dispatch('write_file') should enforce read-before-write."""
        (tmp_path / "f.txt").write_text("hi")
        tracker = FileAccessTracker()
        result = dispatch(
            "write_file",
            {"file_path": "f.txt", "content": "new"},
            str(tmp_path),
            file_tracker=tracker,
        )
        assert result.startswith("error:")

    def test_dispatch_passes_tracker_to_edit(self, tmp_path):
        """dispatch('edit_file') should enforce read-before-write."""
        (tmp_path / "f.txt").write_text("hi there")
        tracker = FileAccessTracker()
        result = dispatch(
            "edit_file",
            {"file_path": "f.txt", "old_string": "hi", "new_string": "bye"},
            str(tmp_path),
            file_tracker=tracker,
        )
        assert result.startswith("error:")


# =========================================================================
# REPL /clear resets tracker
# =========================================================================


class TestReplClearResetsTracker:
    def test_clear_resets_file_tracker(self, tmp_path):
        """After /clear, the file tracker should be fully reset."""
        tracker = FileAccessTracker()
        tracker.record_read("/some/file.txt")
        tracker.record_write("/some/new.txt")

        messages = [{"role": "system", "content": "sys"}]
        ts = ThinkingState(verbose=False, notes_dir=str(tmp_path))
        _repl_clear(messages, ts, file_tracker=tracker)

        assert len(tracker.read_files) == 0
        assert len(tracker.written_files) == 0

    def test_clear_with_none_tracker(self, tmp_path):
        """Passing file_tracker=None to _repl_clear should not error."""
        messages = [{"role": "system", "content": "sys"}]
        ts = ThinkingState(verbose=False, notes_dir=str(tmp_path))
        _repl_clear(messages, ts, file_tracker=None)  # should not raise

    def test_clear_then_write_blocked(self, tmp_path):
        """Read, then /clear, then write — should be blocked."""
        (tmp_path / "f.txt").write_text("original")
        tracker = FileAccessTracker()
        _read_file("f.txt", str(tmp_path), tracker=tracker)
        # Verify read succeeded
        assert str(tmp_path / "f.txt") in tracker.read_files

        # /clear
        messages = [{"role": "system", "content": "sys"}]
        ts = ThinkingState(verbose=False, notes_dir=str(tmp_path))
        _repl_clear(messages, ts, file_tracker=tracker)

        # Write should now be blocked
        result = _write_file("f.txt", "new", str(tmp_path), tracker=tracker)
        assert result.startswith("error:")


# =========================================================================
# Compaction does NOT reset the tracker (regression tests)
# =========================================================================


class TestCompactionPreservesTracker:
    def _make_messages_with_read(self):
        """Build a message list that looks like a real conversation with tool calls."""
        return [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Read the file and edit it."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"file_path": "test.txt"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "tc1",
                "content": "1: line one\n2: line two\n" + "x" * 2000,
            },
            {"role": "assistant", "content": "I've read the file."},
            {"role": "user", "content": "Now edit it."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc2",
                        "type": "function",
                        "function": {
                            "name": "edit_file",
                            "arguments": '{"file_path": "test.txt", "old_string": "one", "new_string": "ONE"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc2", "content": "Edited test.txt"},
            {"role": "assistant", "content": "Done editing."},
        ]

    def test_compact_messages_preserves_tracker(self, tmp_path):
        """compact_messages() does not affect the tracker — reads remain valid."""
        (tmp_path / "test.txt").write_text("line one\nline two\n")
        tracker = FileAccessTracker()
        _read_file("test.txt", str(tmp_path), tracker=tracker)
        resolved = str(tmp_path / "test.txt")
        assert resolved in tracker.read_files

        # Run compact_messages on a conversation
        messages = self._make_messages_with_read()
        compacted = compact_messages(messages)
        # Tracker state is completely untouched by compaction
        assert resolved in tracker.read_files

        # Write should still be allowed
        result = _write_file("test.txt", "new content", str(tmp_path), tracker=tracker)
        assert result.startswith("Wrote")

    def test_drop_middle_turns_preserves_tracker(self, tmp_path):
        """drop_middle_turns() does not affect the tracker — reads remain valid."""
        (tmp_path / "test.txt").write_text("line one\nline two\n")
        tracker = FileAccessTracker()
        _read_file("test.txt", str(tmp_path), tracker=tracker)
        resolved = str(tmp_path / "test.txt")
        assert resolved in tracker.read_files

        # Run drop_middle_turns on a conversation
        messages = self._make_messages_with_read()
        dropped = drop_middle_turns(messages)
        # Tracker state is completely untouched by dropping turns
        assert resolved in tracker.read_files

        # Write should still be allowed
        result = _write_file("test.txt", "new content", str(tmp_path), tracker=tracker)
        assert result.startswith("Wrote")


# =========================================================================
# --no-read-guard flag
# =========================================================================


class TestNoReadGuardFlag:
    def test_flag_parsed(self):
        """--no-read-guard should be parsed by argparse."""
        from swival.agent import build_parser

        parser = build_parser()
        args = parser.parse_args(["--no-read-guard", "question"])
        assert args.no_read_guard is True

    def test_flag_default_false(self):
        """Without --no-read-guard, the flag defaults to False."""
        from swival.agent import build_parser

        parser = build_parser()
        args = parser.parse_args(["question"])
        assert args.no_read_guard is False

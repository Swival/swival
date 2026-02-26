"""Tests for tools.py and edit.py modules."""

import os

import pytest

from swival.tools import (
    _read_file,
    _write_file,
    _edit_file,
    dispatch,
    MAX_LINE_LENGTH,
    MAX_OUTPUT_BYTES,
)


# =========================================================================
# read_file -- positive paths
# =========================================================================


class TestReadFilePositive:
    """Positive-path tests for _read_file (and dispatch('read_file', ...))."""

    def test_read_existing_text_file(self, tmp_path):
        """Reading a plain text file returns line-numbered output."""
        f = tmp_path / "hello.txt"
        f.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

        result = _read_file("hello.txt", str(tmp_path))
        assert result == "1: alpha\n2: beta\n3: gamma"

    def test_read_directory_listing(self, tmp_path):
        """Reading a directory lists entries with / suffix for subdirs."""
        (tmp_path / "subdir").mkdir()
        (tmp_path / "file.txt").write_text("hi", encoding="utf-8")

        result = _read_file(".", str(tmp_path))
        lines = result.split("\n")
        # Should contain both entries
        assert "file.txt" in lines
        assert "subdir/" in lines

    def test_read_with_offset_and_limit(self, tmp_path):
        """offset and limit slice the returned lines correctly."""
        f = tmp_path / "nums.txt"
        f.write_text(
            "\n".join(f"line{i}" for i in range(1, 11)) + "\n", encoding="utf-8"
        )

        # offset=3, limit=4 => lines 3..6 (1-based), with hint about remaining
        result = _read_file("nums.txt", str(tmp_path), offset=3, limit=4)
        assert result.startswith("3: line3\n4: line4\n5: line5\n6: line6")
        assert "4 more lines, use offset=7 to continue" in result


# =========================================================================
# read_file -- tail support
# =========================================================================


class TestReadFileTail:
    """Tests for read_file tail parameter."""

    def _make_file(self, tmp_path, n=10):
        """Create a file with n lines: line1, line2, ..., lineN."""
        f = tmp_path / "data.txt"
        f.write_text(
            "\n".join(f"line{i}" for i in range(1, n + 1)) + "\n", encoding="utf-8"
        )
        return f

    def test_tail_returns_last_n_lines(self, tmp_path):
        """tail=3 on a 10-line file returns lines 8-10."""
        self._make_file(tmp_path, 10)
        result = _read_file("data.txt", str(tmp_path), tail=3)
        assert "8: line8" in result
        assert "9: line9" in result
        assert "10: line10" in result
        assert "7: line7" not in result

    def test_tail_exceeds_file_length(self, tmp_path):
        """tail=100 on a 5-line file returns all 5 lines."""
        self._make_file(tmp_path, 5)
        result = _read_file("data.txt", str(tmp_path), tail=100)
        assert "1: line1" in result
        assert "5: line5" in result
        assert "more lines" not in result

    def test_tail_with_limit(self, tmp_path):
        """tail=10, limit=3 returns 3 lines with continuation hint."""
        self._make_file(tmp_path, 20)
        result = _read_file("data.txt", str(tmp_path), tail=10, limit=3)
        # Last 10 lines start at line 11; limit=3 gives lines 11-13
        assert "11: line11" in result
        assert "13: line13" in result
        assert "14: line14" not in result
        assert "more lines, use offset=" in result

    def test_tail_pagination_flow(self, tmp_path):
        """Follow up a tail call with the returned offset to get the next page."""
        self._make_file(tmp_path, 20)
        # First call: tail=10, limit=3 -> lines 11-13
        result1 = _read_file("data.txt", str(tmp_path), tail=10, limit=3)
        assert "11: line11" in result1
        # Extract offset from hint
        import re

        m = re.search(r"offset=(\d+)", result1)
        assert m, f"No offset hint found in: {result1}"
        next_offset = int(m.group(1))
        assert next_offset == 14
        # Second call: use offset (no tail) to continue
        result2 = _read_file("data.txt", str(tmp_path), offset=next_offset, limit=3)
        assert "14: line14" in result2
        assert "15: line15" in result2
        assert "16: line16" in result2

    def test_tail_ignores_offset(self, tmp_path):
        """tail=5, offset=1 still returns the last 5 lines."""
        self._make_file(tmp_path, 10)
        result = _read_file("data.txt", str(tmp_path), tail=5, offset=1)
        assert "6: line6" in result
        assert "10: line10" in result
        assert "5: line5" not in result

    def test_tail_line_numbers_correct(self, tmp_path):
        """Line numbers in output match actual 1-based positions."""
        self._make_file(tmp_path, 10)
        result = _read_file("data.txt", str(tmp_path), tail=3)
        lines = [ln for ln in result.split("\n") if ln and not ln.startswith("[")]
        assert lines[0] == "8: line8"
        assert lines[1] == "9: line9"
        assert lines[2] == "10: line10"

    def test_tail_on_directory_ignored(self, tmp_path):
        """tail has no effect on directory listings."""
        (tmp_path / "a.txt").write_text("x", encoding="utf-8")
        (tmp_path / "b.txt").write_text("y", encoding="utf-8")
        result_no_tail = _read_file(".", str(tmp_path))
        result_with_tail = _read_file(".", str(tmp_path), tail=1)
        assert result_no_tail == result_with_tail

    def test_tail_nonpositive_clamped(self, tmp_path):
        """tail=0 and tail=-3 both behave like tail=1 (return last line)."""
        self._make_file(tmp_path, 5)
        for t in [0, -3]:
            result = _read_file("data.txt", str(tmp_path), tail=t)
            assert "5: line5" in result
            assert "4: line4" not in result

    def test_tail_numeric_string_coerced(self, tmp_path):
        """tail='5' (numeric string) is coerced to int and works."""
        self._make_file(tmp_path, 5)
        result = dispatch(
            "read_file", {"file_path": "data.txt", "tail": "5"}, str(tmp_path)
        )
        assert "5: line5" in result

    def test_tail_non_numeric_string_returns_error(self, tmp_path):
        """tail='abc' via dispatch returns an error string."""
        self._make_file(tmp_path, 5)
        result = dispatch(
            "read_file", {"file_path": "data.txt", "tail": "abc"}, str(tmp_path)
        )
        assert result.startswith("error:")
        assert "integer" in result

    def test_tail_via_dispatch(self, tmp_path):
        """End-to-end through dispatch('read_file', ...)."""
        self._make_file(tmp_path, 10)
        result = dispatch(
            "read_file", {"file_path": "data.txt", "tail": 3}, str(tmp_path)
        )
        assert "8: line8" in result
        assert "10: line10" in result


# =========================================================================
# write_file -- positive paths
# =========================================================================


class TestWriteFilePositive:
    """Positive-path tests for _write_file."""

    def test_write_new_file(self, tmp_path):
        """Writing a new file creates it with the expected content."""
        result = _write_file("out.txt", "hello world", str(tmp_path))
        assert "Wrote" in result
        assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "hello world"

    def test_write_creates_parent_dirs(self, tmp_path):
        """Parent directories are created automatically."""
        result = _write_file("a/b/c/deep.txt", "nested", str(tmp_path))
        assert "Wrote" in result
        assert (tmp_path / "a" / "b" / "c" / "deep.txt").read_text(
            encoding="utf-8"
        ) == "nested"


# =========================================================================
# write_file -- move_from
# =========================================================================


class TestWriteFileMoveFrom:
    """Tests for _write_file move_from (atomic rename) parameter."""

    def test_rename_moves_file(self, tmp_path):
        """move_from atomically renames source to destination."""
        (tmp_path / "old.txt").write_text("keep this", encoding="utf-8")
        result = _write_file("new.txt", None, str(tmp_path), move_from="old.txt")
        assert result.startswith("Moved")
        assert "old.txt" in result and "new.txt" in result
        assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "keep this"
        assert not (tmp_path / "old.txt").exists()

    def test_rename_preserves_content_exactly(self, tmp_path):
        """Atomic rename preserves file content byte-for-byte."""
        original = "line 1\nline 2\n\ttabbed\n"
        (tmp_path / "src.py").write_text(original, encoding="utf-8")
        _write_file("dst.py", None, str(tmp_path), move_from="src.py")
        assert (tmp_path / "dst.py").read_text(encoding="utf-8") == original

    def test_rename_works_for_binary(self, tmp_path):
        """Atomic rename works for binary files (no content read/encode)."""
        data = bytes(range(256))
        (tmp_path / "bin.dat").write_bytes(data)
        result = _write_file("moved.dat", None, str(tmp_path), move_from="bin.dat")
        assert result.startswith("Moved")
        assert (tmp_path / "moved.dat").read_bytes() == data

    def test_move_from_src_not_found(self, tmp_path):
        """Error if move_from source does not exist."""
        result = _write_file("dst.txt", None, str(tmp_path), move_from="missing.txt")
        assert result.startswith("error:")
        assert "missing.txt" in result

    def test_move_from_self_move_rejected(self, tmp_path):
        """Error when source and destination resolve to the same path."""
        (tmp_path / "same.txt").write_text("x", encoding="utf-8")
        result = _write_file("same.txt", None, str(tmp_path), move_from="same.txt")
        assert result.startswith("error:")
        assert "same path" in result

    def test_move_from_directory_rejected(self, tmp_path):
        """Error when move_from is a directory."""
        (tmp_path / "subdir").mkdir()
        result = _write_file("dst.txt", None, str(tmp_path), move_from="subdir")
        assert result.startswith("error:")
        assert "directory" in result

    def test_move_from_outside_sandbox_rejected(self, tmp_path):
        """move_from path outside base_dir is rejected."""
        result = _write_file("dst.txt", None, str(tmp_path), move_from="../outside.txt")
        assert result.startswith("error:")

    def test_move_from_dangling_symlink_allowed(self, tmp_path):
        """move_from accepts a dangling symlink, consistent with delete_file."""
        link = tmp_path / "link.txt"
        link.symlink_to(tmp_path / "nonexistent_target.txt")
        assert link.is_symlink() and not link.exists()

        result = _write_file("dst.txt", None, str(tmp_path), move_from="link.txt")
        assert result.startswith("Moved")
        assert not link.exists()

    def test_content_and_move_from_mutually_exclusive(self, tmp_path):
        """Setting both content and move_from is an error."""
        (tmp_path / "src.txt").write_text("x", encoding="utf-8")
        result = _write_file("dst.txt", "x", str(tmp_path), move_from="src.txt")
        assert result.startswith("error:")
        assert "not both" in result

    def test_neither_content_nor_move_from_is_error(self, tmp_path):
        """Omitting both content and move_from is an error."""
        result = _write_file("new.txt", None, str(tmp_path))
        assert result.startswith("error:")

    def test_empty_move_from_treated_as_absent(self, tmp_path):
        """Empty move_from string is treated as not provided."""
        result = _write_file("new.txt", "hello", str(tmp_path), move_from="")
        assert not result.startswith("error:")
        assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "hello"

    def test_empty_content_with_move_from(self, tmp_path):
        """Empty content string with move_from does the rename."""
        (tmp_path / "src.txt").write_text("data", encoding="utf-8")
        result = _write_file("dst.txt", "", str(tmp_path), move_from="src.txt")
        assert not result.startswith("error:")
        assert (tmp_path / "dst.txt").read_text(encoding="utf-8") == "data"
        assert not (tmp_path / "src.txt").exists()


# =========================================================================
# edit_file -- positive paths
# =========================================================================


class TestEditFilePositive:
    """Positive-path tests for _edit_file (via dispatch or directly)."""

    def test_simple_edit_via_dispatch(self, tmp_path):
        """dispatch('edit_file', ...) replaces text in an existing file."""
        (tmp_path / "data.txt").write_text("aaa\nbbb\nccc\n", encoding="utf-8")

        result = dispatch(
            "edit_file",
            {
                "file_path": "data.txt",
                "old_string": "bbb",
                "new_string": "BBB",
            },
            str(tmp_path),
        )
        assert "Edited" in result
        content = (tmp_path / "data.txt").read_text(encoding="utf-8")
        assert content == "aaa\nBBB\nccc\n"

    def test_multiline_edit(self, tmp_path):
        """Multi-line old_string is replaced correctly."""
        (tmp_path / "data.txt").write_text("aaa\nbbb\nccc\nddd\n", encoding="utf-8")

        result = _edit_file("data.txt", "bbb\nccc", "BBB\nCCC", str(tmp_path))
        assert "Edited" in result
        content = (tmp_path / "data.txt").read_text(encoding="utf-8")
        assert content == "aaa\nBBB\nCCC\nddd\n"

    def test_replace_all_via_dispatch(self, tmp_path):
        """dispatch with replace_all=True replaces all occurrences."""
        (tmp_path / "data.txt").write_text("x\ny\nx\nz\n", encoding="utf-8")

        result = dispatch(
            "edit_file",
            {
                "file_path": "data.txt",
                "old_string": "x",
                "new_string": "X",
                "replace_all": True,
            },
            str(tmp_path),
        )
        assert "Edited" in result
        content = (tmp_path / "data.txt").read_text(encoding="utf-8")
        assert content == "X\ny\nX\nz\n"


# =========================================================================
# Error handling
# =========================================================================


class TestErrorHandling:
    """Error-handling tests for dispatch and _read_file."""

    def test_dispatch_unknown_tool_raises_key_error(self, tmp_path):
        """dispatch() with an unrecognised tool name raises KeyError."""
        with pytest.raises(KeyError, match="Unknown tool"):
            dispatch("no_such_tool", {}, str(tmp_path))

    def test_read_binary_file_returns_error(self, tmp_path):
        """Reading a binary file (containing null bytes) returns an error string."""
        f = tmp_path / "img.bin"
        f.write_bytes(b"\x89PNG\r\n\x00\x00" + b"\x00" * 100)

        result = _read_file("img.bin", str(tmp_path))
        assert result.startswith("error:")
        assert "binary" in result

    def test_read_nonexistent_path_returns_error(self, tmp_path):
        """Reading a path that does not exist returns an error string."""
        result = _read_file("no_such_file.txt", str(tmp_path))
        assert result.startswith("error:")
        assert "does not exist" in result

    def test_read_non_utf8_returns_decode_error(self, tmp_path):
        """A file with non-UTF-8 bytes (but no nulls) triggers a decode error."""
        f = tmp_path / "bad.txt"
        # Latin-1 encoded bytes that are invalid UTF-8 (no null bytes though)
        f.write_bytes(b"caf\xe9 cr\xe8me\n")

        result = _read_file("bad.txt", str(tmp_path))
        assert result.startswith("error:")
        assert "UTF-8" in result or "decode" in result.lower()


# =========================================================================
# Sandbox tests
# =========================================================================


class TestSandbox:
    """Path-sandboxing tests for safe_resolve."""

    def test_dotdot_escape_rejected(self, tmp_path):
        """A path containing .. that escapes base_dir is rejected."""
        result = _read_file("../../../etc/passwd", str(tmp_path))
        assert result.startswith("error:")
        assert "outside" in result.lower() or "escape" in result.lower()

    def test_symlink_escape_rejected(self, tmp_path):
        """A symlink inside base_dir pointing outside is rejected."""
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        secret = outside_dir / "secret.txt"
        secret.write_text("top secret", encoding="utf-8")

        # Create a sandboxed directory and a symlink that points outside it
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        link = sandbox / "escape_link"
        link.symlink_to(secret)

        result = _read_file("escape_link", str(sandbox))
        assert result.startswith("error:")
        assert "outside" in result.lower() or "escape" in result.lower()


# =========================================================================
# edit_file -- error handling
# =========================================================================


class TestEditFileErrors:
    """Error-handling tests for _edit_file dispatch-level formatting."""

    def test_missing_file_returns_error(self, tmp_path):
        result = _edit_file("no_such.txt", "old", "new", str(tmp_path))
        assert result.startswith("error:")
        assert "does not exist" in result

    def test_empty_old_string_returns_error(self, tmp_path):
        (tmp_path / "f.txt").write_text("hello", encoding="utf-8")
        result = _edit_file("f.txt", "", "new", str(tmp_path))
        assert result.startswith("error:")
        assert "empty" in result

    def test_not_found_returns_error(self, tmp_path):
        (tmp_path / "f.txt").write_text("hello", encoding="utf-8")
        result = _edit_file("f.txt", "xyz", "abc", str(tmp_path))
        assert result.startswith("error:")
        assert "not found" in result

    def test_multiple_matches_returns_error(self, tmp_path):
        (tmp_path / "f.txt").write_text("aaa\nbbb\naaa\n", encoding="utf-8")
        result = _edit_file("f.txt", "aaa", "ccc", str(tmp_path))
        assert result.startswith("error:")
        assert "multiple matches" in result

    def test_path_escape_returns_error(self, tmp_path):
        result = _edit_file("../../etc/passwd", "root", "x", str(tmp_path))
        assert result.startswith("error:")
        assert "outside" in result.lower() or "escape" in result.lower()


# =========================================================================
# Other
# =========================================================================


class TestOther:
    """Miscellaneous tests for read_file truncation and output cap."""

    def test_long_lines_truncated_at_2000_chars(self, tmp_path):
        """Lines longer than MAX_LINE_LENGTH are truncated."""
        long_line = "x" * 5000
        (tmp_path / "wide.txt").write_text(long_line + "\n", encoding="utf-8")

        result = _read_file("wide.txt", str(tmp_path))
        # The output line is "1: " + truncated content
        returned_line = result.split("\n")[0]
        content_part = returned_line[len("1: ") :]
        assert len(content_part) == MAX_LINE_LENGTH

    def test_output_capped_at_50kb(self, tmp_path):
        """Output is capped at MAX_OUTPUT_BYTES (50 KB) with a truncation marker."""
        # Generate a file large enough to exceed 50 KB of numbered output.
        # Each line "NNNN: <80 chars>" is ~87 bytes.  We need ~600 lines to
        # be safe.  Use 1000 lines of 80 chars each.
        line = "A" * 80
        text = "\n".join([line] * 1000) + "\n"
        (tmp_path / "big.txt").write_text(text, encoding="utf-8")

        result = _read_file("big.txt", str(tmp_path))
        assert "more lines, use offset=" in result
        # Byte size of the result (before the marker) should be ≤ MAX_OUTPUT_BYTES
        # (the marker itself is appended after the cap check, so total may be
        # slightly over, but the actual line data must be under).
        lines_before_marker = result.rsplit("\n[", 1)[0]
        assert len(lines_before_marker.encode("utf-8")) <= MAX_OUTPUT_BYTES


class TestDirectoryListingCap:
    """Regression: directory listings must respect the 50KB output cap."""

    def test_large_directory_truncated_at_50kb(self, tmp_path):
        """A directory with many files triggers the truncation marker."""
        # Create enough files to exceed 50KB of listing output.
        # Each filename is ~30 chars + newline ≈ 31 bytes. Need ~1700 files.
        for i in range(2000):
            (tmp_path / f"file_{i:06d}_padding_name.txt").write_text(
                "x", encoding="utf-8"
            )

        result = _read_file(".", str(tmp_path))
        assert result.endswith("[truncated at 50KB]")
        lines_before = result.rsplit("\n[truncated at 50KB]", 1)[0]
        assert len(lines_before.encode("utf-8")) <= MAX_OUTPUT_BYTES


class TestAgentLoop:
    """Tests for agent loop behavior (mocked LLM)."""

    def test_no_tool_calls_terminates_immediately(self, monkeypatch, capsys):
        """When the model returns no tool_calls, the loop prints content and exits."""
        from unittest.mock import MagicMock
        from swival import agent

        # Build a fake message with text content and no tool_calls
        fake_msg = MagicMock()
        fake_msg.tool_calls = None
        fake_msg.content = "The answer is 42."
        fake_msg.role = "assistant"

        monkeypatch.setattr(agent, "call_llm", lambda *a, **kw: (fake_msg, "stop"))
        monkeypatch.setattr(
            agent, "discover_model", lambda *a, **kw: ("fake-model", None)
        )

        monkeypatch.setattr(
            "sys.argv", ["agent", "what is the answer?", "--model", "fake"]
        )
        agent.main()

        captured = capsys.readouterr()
        assert "The answer is 42." in captured.out

    def test_max_turns_zero_exits_with_code_2(self, tmp_path):
        """--max-turns 0 means the loop never runs; exit code should be 2."""
        import subprocess

        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "swival.agent",
                "hello",
                "--max-turns",
                "0",
                "--model",
                "fake-model",
                "--base-url",
                "http://127.0.0.1:1",
            ],  # won't connect
            capture_output=True,
            text=True,
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        )
        # The loop condition `while turns < 0` is false immediately,
        # so it hits the else branch and exits with code 2.
        # However the LLM call happens inside the loop, so with max-turns=0
        # it should just hit the else branch without trying to connect.
        assert result.returncode == 2
        assert "max turns" in result.stderr.lower()

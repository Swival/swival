"""Tests for tools.py and edit.py modules."""

import os

import pytest

from swival.tools import (
    _expand_tilde,
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
# read_file -- missing MEMORY.md hint
# =========================================================================


class TestReadFileMemoryHint:
    """When read_file targets a missing .swival/memory/MEMORY.md, return a
    helpful hint instead of a generic error."""

    def test_missing_memory_returns_hint(self, tmp_path):
        result = _read_file(".swival/memory/MEMORY.md", base_dir=str(tmp_path))
        assert not result.startswith("error:")
        assert "does not exist yet" in result

    def test_other_missing_file_still_errors(self, tmp_path):
        result = _read_file("foo.txt", base_dir=str(tmp_path))
        assert result.startswith("error: path does not exist:")

    def test_symlinked_memory_dir_returns_hint(self, tmp_path):
        real_dir = tmp_path / "real_memory"
        real_dir.mkdir()
        swival_dir = tmp_path / ".swival"
        swival_dir.mkdir()
        (swival_dir / "memory").symlink_to(real_dir)
        result = _read_file(".swival/memory/MEMORY.md", base_dir=str(tmp_path))
        assert not result.startswith("error:")
        assert "does not exist yet" in result


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
        """tail=5 returns the last 5 lines even when offset is set."""
        self._make_file(tmp_path, 10)
        # offset=7 would normally start at line 7, but tail takes precedence
        result = _read_file("data.txt", str(tmp_path), tail=5, offset=7)
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
        """tail=0 and tail=-3 clamp to 1, then promote to limit (returns whole file)."""
        self._make_file(tmp_path, 5)
        for t in [0, -3]:
            result = _read_file("data.txt", str(tmp_path), tail=t)
            assert "5: line5" in result
            assert "1: line1" in result

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

    def test_dispatch_tail_ignores_offset(self, tmp_path):
        """dispatch('read_file') with tail set should ignore a stray offset."""
        self._make_file(tmp_path, 10)
        result = dispatch(
            "read_file",
            {"file_path": "data.txt", "tail": 3, "offset": 1000},
            str(tmp_path),
        )
        assert "8: line8" in result
        assert "10: line10" in result

    def test_tail_1_with_large_limit_uses_limit(self, tmp_path):
        """tail=1 with limit>1 is treated as tail=limit (model meant 'from the end')."""
        self._make_file(tmp_path, 20)
        result = _read_file("data.txt", str(tmp_path), tail=1, limit=5)
        # Should return last 5 lines, not last 1 line
        assert "16: line16" in result
        assert "20: line20" in result
        assert "15: line15" not in result


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


class TestWriteFileProtectsConfig:
    """write_file must not overwrite project config files."""

    @pytest.mark.parametrize("name", ["swival.toml", "mcp.json"])
    def test_write_config_blocked(self, tmp_path, name):
        result = _write_file(name, "model = 'hacked'\n", str(tmp_path))
        assert result.startswith("error:")
        assert name in result
        assert not (tmp_path / name).exists()

    @pytest.mark.parametrize("name", ["swival.toml", "mcp.json"])
    def test_move_to_config_blocked(self, tmp_path, name):
        (tmp_path / "payload.txt").write_text("bad", encoding="utf-8")
        result = _write_file(name, None, str(tmp_path), move_from="payload.txt")
        assert result.startswith("error:")
        assert name in result
        # Source file should still exist.
        assert (tmp_path / "payload.txt").exists()


class TestEditFileProtectsConfig:
    """edit_file must not modify project config files."""

    @pytest.mark.parametrize("name", ["swival.toml", "mcp.json"])
    def test_edit_config_blocked(self, tmp_path, name):
        config = tmp_path / name
        config.write_text("model = 'original'\n", encoding="utf-8")
        result = _edit_file(name, "original", "hacked", str(tmp_path))
        assert result.startswith("error:")
        assert name in result
        # Content should be unchanged.
        assert config.read_text(encoding="utf-8") == "model = 'original'\n"


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

    def test_edit_file_no_diff_when_quiet(self, tmp_path, capsys):
        """_edit_file(verbose=False) produces no stderr diff output."""
        (tmp_path / "data.txt").write_text("aaa\nbbb\nccc\n", encoding="utf-8")
        result = _edit_file("data.txt", "bbb", "BBB", str(tmp_path), verbose=False)
        assert "Edited" in result
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_edit_file_return_value_unchanged(self, tmp_path):
        """_edit_file still returns 'Edited {path}' regardless of verbose."""
        (tmp_path / "data.txt").write_text("aaa\nbbb\nccc\n", encoding="utf-8")
        result = _edit_file("data.txt", "bbb", "BBB", str(tmp_path), verbose=True)
        assert result == "Edited data.txt"

    def test_dispatch_edit_quiet_no_diff(self, tmp_path, monkeypatch):
        """dispatch('edit_file', verbose=False) does not call fmt.tool_diff."""
        (tmp_path / "data.txt").write_text("aaa\nbbb\nccc\n", encoding="utf-8")
        from unittest.mock import patch

        with patch("swival.fmt.tool_diff") as mock_diff:
            dispatch(
                "edit_file",
                {"file_path": "data.txt", "old_string": "bbb", "new_string": "BBB"},
                str(tmp_path),
                verbose=False,
            )
            mock_diff.assert_not_called()

    def test_dispatch_edit_verbose_calls_diff(self, tmp_path):
        """dispatch('edit_file', verbose=True) calls fmt.tool_diff with correct args."""
        (tmp_path / "data.txt").write_text("aaa\nbbb\nccc\n", encoding="utf-8")
        from unittest.mock import patch

        with patch("swival.fmt.tool_diff") as mock_diff:
            dispatch(
                "edit_file",
                {"file_path": "data.txt", "old_string": "bbb", "new_string": "BBB"},
                str(tmp_path),
                verbose=True,
            )
            mock_diff.assert_called_once_with(
                "data.txt", "aaa\nbbb\nccc\n", "aaa\nBBB\nccc\n"
            )

    def test_edit_file_diff_exception_swallowed(self, tmp_path):
        """If fmt.tool_diff raises, edit still succeeds."""
        (tmp_path / "data.txt").write_text("aaa\nbbb\nccc\n", encoding="utf-8")
        from unittest.mock import patch

        with patch("swival.fmt.tool_diff", side_effect=RuntimeError("boom")):
            result = _edit_file("data.txt", "bbb", "BBB", str(tmp_path), verbose=True)
        assert result == "Edited data.txt"
        assert (tmp_path / "data.txt").read_text() == "aaa\nBBB\nccc\n"


# =========================================================================
# Error handling
# =========================================================================


class TestErrorHandling:
    """Error-handling tests for dispatch and _read_file."""

    def test_dispatch_unknown_tool_raises_key_error(self, tmp_path):
        """dispatch() with an unrecognised tool name raises KeyError."""
        with pytest.raises(KeyError, match="Unknown tool"):
            dispatch("no_such_tool", {}, str(tmp_path))

    def test_dispatch_unknown_tool_lists_available(self, tmp_path):
        """Unknown tool error lists available tools."""
        with pytest.raises(KeyError, match="Available tools:.*run_command"):
            dispatch("no_such_tool", {}, str(tmp_path))

    def test_dispatch_hallucinated_shell_suggests_run_command_restricted(
        self, tmp_path
    ):
        """In restricted mode, shell-ish aliases downgrade to run_command."""
        with pytest.raises(KeyError, match="Did you mean 'run_command'"):
            dispatch("execute_shell_command", {}, str(tmp_path))

    def test_dispatch_hallucinated_shell_suggests_run_shell_command_unrestricted(
        self, tmp_path
    ):
        """With shell_allowed, shell-ish aliases suggest run_shell_command."""
        with pytest.raises(KeyError, match="Did you mean 'run_shell_command'"):
            dispatch(
                "bash",
                {},
                str(tmp_path),
                commands_unrestricted=True,
                shell_allowed=True,
            )

    def test_dispatch_unknown_tool_lists_run_shell_command_unrestricted(self, tmp_path):
        """With shell_allowed, available-tools list includes run_shell_command."""
        with pytest.raises(KeyError, match="run_shell_command") as exc_info:
            dispatch(
                "no_such_tool_xyz",
                {},
                str(tmp_path),
                commands_unrestricted=True,
                shell_allowed=True,
            )
        assert "Available tools:" in str(exc_info.value)

    def test_dispatch_unknown_tool_omits_run_shell_command_restricted(self, tmp_path):
        """In restricted mode, available-tools list omits run_shell_command."""
        with pytest.raises(KeyError) as exc_info:
            dispatch("no_such_tool_xyz", {}, str(tmp_path))
        assert "run_shell_command" not in str(exc_info.value)

    def test_dispatch_run_shell_command_blocked_without_shell_allowed(self, tmp_path):
        """run_shell_command blocked when shell_allowed=False."""
        result = dispatch(
            "run_shell_command",
            {"command": "echo hello"},
            str(tmp_path),
            commands_unrestricted=True,
            shell_allowed=False,
            resolved_commands={},
        )
        assert result.startswith("error:")
        assert "not available" in result

    def test_dispatch_shell_blocked_in_ask_mode(self, tmp_path):
        """run_shell_command blocked with unrestricted=True but shell_allowed=False."""
        result = dispatch(
            "run_shell_command",
            {"command": "ls -la | grep foo"},
            str(tmp_path),
            commands_unrestricted=True,
            shell_allowed=False,
            resolved_commands={},
        )
        assert result.startswith("error:")
        assert "not available" in result

    def test_dispatch_run_command_no_shell_escalation_ask_mode(self, tmp_path):
        """Shell-char string to run_command with shell_allowed=False gets normalization error."""
        result = dispatch(
            "run_command",
            {"command": "echo hello && echo world"},
            str(tmp_path),
            commands_unrestricted=True,
            shell_allowed=False,
            resolved_commands={},
        )
        assert result.startswith("error:")
        assert "JSON array" in result

    def test_dispatch_run_command_plain_string_split_ask_mode(self, tmp_path):
        """Plain string to run_command with shell_allowed=False splits to argv."""
        result = dispatch(
            "run_command",
            {"command": "echo hello"},
            str(tmp_path),
            commands_unrestricted=True,
            shell_allowed=False,
            resolved_commands={},
        )
        assert "hello" in result
        assert not result.startswith("error:")

    def test_dispatch_unknown_tool_omits_shell_in_ask_mode(self, tmp_path):
        """With unrestricted=True but shell_allowed=False, shell not in available list."""
        with pytest.raises(KeyError) as exc_info:
            dispatch(
                "no_such_tool_xyz",
                {},
                str(tmp_path),
                commands_unrestricted=True,
                shell_allowed=False,
            )
        assert "run_shell_command" not in str(exc_info.value)

    def test_dispatch_alias_downgrades_to_run_command_ask_mode(self, tmp_path):
        """Shell alias with shell_allowed=False suggests run_command."""
        with pytest.raises(KeyError, match="Did you mean 'run_command'"):
            dispatch(
                "bash",
                {},
                str(tmp_path),
                commands_unrestricted=True,
                shell_allowed=False,
            )

    def test_dispatch_hallucinated_search_suggests_grep(self, tmp_path):
        """Hallucinated 'search' suggests 'grep'."""
        with pytest.raises(KeyError, match="Did you mean 'grep'"):
            dispatch("search", {}, str(tmp_path))

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
# Build tools tests
# =========================================================================


class TestBuildTools:
    """Tests for build_tools() shell_allowed gating."""

    def test_ask_mode_no_shell(self):
        from swival.agent import build_tools

        tools = build_tools({}, {}, commands_unrestricted=True, shell_allowed=False)
        names = [t["function"]["name"] for t in tools]
        assert "run_command" in names
        assert "run_shell_command" not in names

    def test_all_mode_has_shell(self):
        from swival.agent import build_tools

        tools = build_tools({}, {}, commands_unrestricted=True, shell_allowed=True)
        names = [t["function"]["name"] for t in tools]
        assert "run_command" in names
        assert "run_shell_command" in names

    def test_ask_mode_run_command_description_no_shell_hint(self):
        from swival.agent import build_tools

        tools = build_tools({}, {}, commands_unrestricted=True, shell_allowed=False)
        rc = [t for t in tools if t["function"]["name"] == "run_command"][0]
        desc = rc["function"]["description"]
        assert "run_shell_command" not in desc
        assert "not supported" in desc

    def test_all_mode_run_command_description_mentions_shell(self):
        from swival.agent import build_tools

        tools = build_tools({}, {}, commands_unrestricted=True, shell_allowed=True)
        rc = [t for t in tools if t["function"]["name"] == "run_command"][0]
        desc = rc["function"]["description"]
        assert "run_shell_command" in desc

    def test_goal_tools_omitted_by_default(self):
        from swival.agent import build_tools

        tools = build_tools({}, {}, commands_unrestricted=True)
        names = {t["function"]["name"] for t in tools}
        assert "complete_goal" not in names
        assert "get_goal" not in names
        assert "create_goal" not in names
        assert "update_goal" not in names

    def test_goal_tools_can_be_enabled(self):
        from swival.agent import build_tools

        tools = build_tools({}, {}, commands_unrestricted=True, goal_tools=True)
        names = {t["function"]["name"] for t in tools}
        assert "complete_goal" in names
        assert "get_goal" not in names
        assert "create_goal" not in names
        assert "update_goal" not in names


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


class TestEditFileLineNumber:
    """Line-number targeting through _edit_file and dispatch."""

    def test_edit_file_with_line_number(self, tmp_path):
        (tmp_path / "f.txt").write_text("aaa\nbbb\naaa\nccc\n", encoding="utf-8")
        result = _edit_file("f.txt", "aaa", "XXX", str(tmp_path), line_number=3)
        assert "Edited" in result
        content = (tmp_path / "f.txt").read_text(encoding="utf-8")
        assert content == "aaa\nbbb\nXXX\nccc\n"

    def test_dispatch_forwards_line_number(self, tmp_path):
        (tmp_path / "f.txt").write_text("aaa\nbbb\naaa\nccc\n", encoding="utf-8")
        result = dispatch(
            "edit_file",
            {
                "file_path": "f.txt",
                "old_string": "aaa",
                "new_string": "XXX",
                "line_number": 1,
            },
            base_dir=str(tmp_path),
        )
        assert "Edited" in result
        content = (tmp_path / "f.txt").read_text(encoding="utf-8")
        assert content == "XXX\nbbb\naaa\nccc\n"

    def test_invalid_line_number_zero_ignored(self, tmp_path):
        (tmp_path / "f.txt").write_text("aaa\nbbb\naaa\n", encoding="utf-8")
        result = _edit_file("f.txt", "aaa", "XXX", str(tmp_path), line_number=0)
        assert result.startswith("error:")
        assert "multiple matches" in result

    def test_invalid_non_numeric_line_number_ignored(self, tmp_path):
        (tmp_path / "f.txt").write_text("aaa\nbbb\naaa\n", encoding="utf-8")
        result = dispatch(
            "edit_file",
            {
                "file_path": "f.txt",
                "old_string": "aaa",
                "new_string": "XXX",
                "line_number": "hello",
            },
            base_dir=str(tmp_path),
        )
        assert result.startswith("error:")
        assert "multiple matches" in result

    def test_ambiguous_duplicate_nudge_error(self, tmp_path):
        (tmp_path / "f.txt").write_text("aaa\nbbb\naaa\n", encoding="utf-8")
        result = _edit_file("f.txt", "aaa", "XXX", str(tmp_path))
        assert result.startswith("error:")
        assert "line_number" in result

    def test_replace_all_ignores_line_number(self, tmp_path):
        (tmp_path / "f.txt").write_text("aaa\nbbb\naaa\n", encoding="utf-8")
        result = _edit_file(
            "f.txt", "aaa", "XXX", str(tmp_path), replace_all=True, line_number=1
        )
        assert "Edited" in result
        content = (tmp_path / "f.txt").read_text(encoding="utf-8")
        assert content == "XXX\nbbb\nXXX\n"


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
        from swival import agent, config

        monkeypatch.setattr(config, "load_config", lambda _: {})

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
                "--provider",
                "lmstudio",
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


class TestExpandTilde:
    def test_home_slash_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        assert _expand_tilde("~/foo.txt") == str(tmp_path / "foo.txt")

    def test_bare_tilde(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        assert _expand_tilde("~") == str(tmp_path)

    def test_tilde_otheruser_raises(self):
        with pytest.raises(ValueError, match="~user syntax"):
            _expand_tilde("~otheruser/foo")

    def test_bare_tilde_otheruser_raises(self):
        with pytest.raises(ValueError, match="~user syntax"):
            _expand_tilde("~otheruser")

    def test_absolute_path_unchanged(self):
        assert _expand_tilde("/absolute/path") == "/absolute/path"

    def test_relative_path_unchanged(self):
        assert _expand_tilde("relative/path") == "relative/path"

    def test_backslash_tilde_rejected(self):
        with pytest.raises(ValueError, match="~user syntax"):
            _expand_tilde("~\\foo")


class TestOutlineDispatch:
    """Dispatch tests for the outline tool."""

    def test_outline_dispatch_single_file(self, tmp_path):
        f = tmp_path / "s.py"
        f.write_text("def hello(): pass\n")
        result = dispatch(
            "outline", {"file_path": str(f)}, str(tmp_path), files_mode="all"
        )
        assert "def hello()" in result

    def test_outline_dispatch_batch(self, tmp_path):
        a = tmp_path / "a.py"
        a.write_text("class A: pass\n")
        b = tmp_path / "b.py"
        b.write_text("def b(): pass\n")
        result = dispatch(
            "outline",
            {"files": [{"file_path": str(a)}, {"file_path": str(b)}]},
            str(tmp_path),
            files_mode="all",
        )
        assert "=== FILE:" in result
        assert "class A" in result
        assert "def b()" in result

    def test_outline_dispatch_both_args_rejected(self, tmp_path):
        result = dispatch(
            "outline",
            {"file_path": "x.py", "files": [{"file_path": "x.py"}]},
            str(tmp_path),
            files_mode="all",
        )
        assert result == "error: set file_path or files, not both"

    def test_outline_dispatch_neither_arg(self, tmp_path):
        result = dispatch("outline", {}, str(tmp_path), files_mode="all")
        assert result == "error: file_path or files is required"

    def test_outline_dispatch_files_not_array(self, tmp_path):
        result = dispatch(
            "outline",
            {"files": {"file_path": "x.py"}},
            str(tmp_path),
            files_mode="all",
        )
        assert result == "error: 'files' must be an array"

    def test_outline_dispatch_files_bare_string(self, tmp_path):
        f = tmp_path / "t.py"
        f.write_text("def t(): pass\n")
        result = dispatch(
            "outline",
            {"files": str(f)},
            str(tmp_path),
            files_mode="all",
        )
        assert "def t()" in result
        assert "files_succeeded: 1" in result

    def test_outline_aliases_suggest_not_route(self, tmp_path):
        with pytest.raises(KeyError, match="Did you mean 'outline'"):
            dispatch("code_outline", {}, str(tmp_path))
        with pytest.raises(KeyError, match="Did you mean 'outline'"):
            dispatch("file_outline", {}, str(tmp_path))


# =========================================================================
# Goal tool (complete_goal)
# =========================================================================


class TestGoalTools:
    """Goal completion tool dispatch and the budget-limited gate."""

    def _state(self):
        from swival.goal import GoalState

        return GoalState()

    def test_complete_goal_schema_exists(self):
        from swival.tools import COMPLETE_GOAL_TOOL

        assert COMPLETE_GOAL_TOOL["type"] == "function"
        assert COMPLETE_GOAL_TOOL["function"]["name"] == "complete_goal"
        assert "description" in COMPLETE_GOAL_TOOL["function"]
        assert COMPLETE_GOAL_TOOL["function"]["parameters"]["properties"] == {}

    def test_only_complete_goal_registered_in_schema_index(self):
        from swival.tools import get_tool_schema

        assert get_tool_schema("complete_goal") is not None
        assert get_tool_schema("get_goal") is None
        assert get_tool_schema("create_goal") is None
        assert get_tool_schema("update_goal") is None

    def test_complete_goal_rejects_args(self, tmp_path):
        state = self._state()
        state.create("x")
        result = dispatch(
            "complete_goal", {"status": "complete"}, str(tmp_path), goal_state=state
        )
        assert result.startswith("error:")
        assert "no arguments" in result

    def test_complete_goal_includes_budget_report(self, tmp_path):
        import json

        state = self._state()
        state.create("ship", token_budget=1000)
        state.account(tokens_delta=250)
        result = dispatch("complete_goal", {}, str(tmp_path), goal_state=state)
        payload = json.loads(result)
        assert payload["goal"]["status"] == "complete"
        assert "completion_budget_report" in payload
        assert "250" in payload["completion_budget_report"]
        assert "1000" in payload["completion_budget_report"]

    def test_complete_goal_no_goal(self, tmp_path):
        result = dispatch("complete_goal", {}, str(tmp_path), goal_state=self._state())
        assert result.startswith("error:")

    def test_removed_goal_tools_are_unknown(self, tmp_path):
        state = self._state()
        state.create("x")
        for name in ("get_goal", "create_goal", "update_goal"):
            with pytest.raises(KeyError, match="Unknown tool"):
                dispatch(name, {}, str(tmp_path), goal_state=state)

    def test_budget_exhausted_blocks_mutating_tools(self, tmp_path):
        state = self._state()
        state.create("x", token_budget=100)
        state.account(tokens_delta=200)
        assert state.budget_exhausted()

        # Mutating tools rejected with the canonical wrap-up message.
        f = tmp_path / "x.txt"
        result = dispatch(
            "write_file",
            {"file_path": str(f), "content": "x"},
            str(tmp_path),
            goal_state=state,
            files_mode="all",
        )
        assert "budget" in result and result.startswith("error:")

    def test_budget_exhausted_allows_read_only_tools(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("hi\n")

        state = self._state()
        state.create("x", token_budget=100)
        state.account(tokens_delta=200)

        result = dispatch(
            "read_file",
            {"file_path": "x.txt"},
            str(tmp_path),
            goal_state=state,
        )
        assert not result.startswith("error:")
        assert "hi" in result

    def test_budget_exhausted_allows_complete_goal(self, tmp_path):
        state = self._state()
        state.create("x", token_budget=100)
        state.account(tokens_delta=200)
        result = dispatch("complete_goal", {}, str(tmp_path), goal_state=state)
        assert not result.startswith("error:")


class TestBudgetGateActionLevel:
    """Action-level gating for stateful tools after budget exhaustion."""

    def _budget_exhausted_state(self):
        from swival.goal import GoalState

        gs = GoalState()
        gs.create("x", token_budget=10)
        gs.account(tokens_delta=999)
        assert gs.budget_exhausted()
        return gs

    def test_todo_list_allowed_after_budget(self, tmp_path):
        from swival.todo import TodoState

        ts = TodoState()
        ts.process({"action": "add", "tasks": ["a"]})
        result = dispatch(
            "todo",
            {"action": "list"},
            str(tmp_path),
            goal_state=self._budget_exhausted_state(),
            todo_state=ts,
        )
        assert not result.startswith("error:")

    def test_todo_add_blocked_after_budget(self, tmp_path):
        from swival.todo import TodoState

        result = dispatch(
            "todo",
            {"action": "add", "tasks": ["sneaky"]},
            str(tmp_path),
            goal_state=self._budget_exhausted_state(),
            todo_state=TodoState(),
        )
        assert "budget" in result and result.startswith("error:")

    def test_todo_done_blocked_after_budget(self, tmp_path):
        from swival.todo import TodoState

        result = dispatch(
            "todo",
            {"action": "done", "tasks": ["a"]},
            str(tmp_path),
            goal_state=self._budget_exhausted_state(),
            todo_state=TodoState(),
        )
        assert result.startswith("error:") and "budget" in result

    def test_todo_clear_blocked_after_budget(self, tmp_path):
        from swival.todo import TodoState

        result = dispatch(
            "todo",
            {"action": "clear"},
            str(tmp_path),
            goal_state=self._budget_exhausted_state(),
            todo_state=TodoState(),
        )
        assert result.startswith("error:") and "budget" in result

    def test_snapshot_status_allowed_after_budget(self, tmp_path):
        from swival.snapshot import SnapshotState

        result = dispatch(
            "snapshot",
            {"action": "status"},
            str(tmp_path),
            goal_state=self._budget_exhausted_state(),
            snapshot_state=SnapshotState(),
            messages=[],
        )
        assert not result.startswith("error:")

    def test_snapshot_save_blocked_after_budget(self, tmp_path):
        from swival.snapshot import SnapshotState

        result = dispatch(
            "snapshot",
            {"action": "save", "label": "wrap"},
            str(tmp_path),
            goal_state=self._budget_exhausted_state(),
            snapshot_state=SnapshotState(),
            messages=[],
        )
        assert result.startswith("error:") and "budget" in result

    def test_snapshot_restore_blocked_after_budget(self, tmp_path):
        from swival.snapshot import SnapshotState

        result = dispatch(
            "snapshot",
            {"action": "restore", "summary": "x"},
            str(tmp_path),
            goal_state=self._budget_exhausted_state(),
            snapshot_state=SnapshotState(),
            messages=[],
        )
        assert result.startswith("error:") and "budget" in result

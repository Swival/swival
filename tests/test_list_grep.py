"""Tests for list_files and grep tools."""

import os
import time

import pytest

from swival.tools import _check_pattern, _grep, _is_within_base, _list_files, dispatch


@pytest.fixture
def sandbox(tmp_path):
    """Create a sandbox directory with test files."""
    # Create some Python files
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("import os\nprint('hello')\n")
    (tmp_path / "src" / "utils.py").write_text("def helper():\n    return 42\n")
    (tmp_path / "src" / "sub").mkdir()
    (tmp_path / "src" / "sub" / "deep.py").write_text("# deep module\nx = 1\n")

    # Create some non-Python files
    (tmp_path / "README.txt").write_text("This is the readme.\n")
    (tmp_path / "config.json").write_text('{"key": "value"}\n')

    # Create a .git directory (should be excluded)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n")
    (tmp_path / ".git" / "objects").mkdir()
    (tmp_path / ".git" / "objects" / "ab").mkdir()
    (tmp_path / ".git" / "objects" / "ab" / "cdef").write_text("blob")

    # Create a binary file
    (tmp_path / "image.bin").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00")

    return tmp_path


# --- _check_pattern tests ---


class TestCheckPattern:
    def test_valid_pattern(self):
        assert _check_pattern("**/*.py") is None
        assert _check_pattern("src/*.ts") is None
        assert _check_pattern("*.txt") is None

    def test_dotdot_rejected(self):
        result = _check_pattern("../*.py")
        assert result is not None
        assert "error" in result
        assert ".." in result

    def test_nested_dotdot_rejected(self):
        result = _check_pattern("foo/../../bar")
        assert result is not None
        assert "error" in result

    def test_absolute_posix_rejected(self):
        result = _check_pattern("/etc/*.py")
        assert result is not None
        assert "error" in result
        assert "absolute" in result

    def test_absolute_windows_rejected(self):
        result = _check_pattern("C:\\Users\\*.py")
        assert result is not None
        assert "error" in result

    def test_windows_backslash_dotdot_rejected(self):
        """Regression: ..\\*.py must be rejected even though POSIX parsing misses it."""
        result = _check_pattern("..\\*.py")
        assert result is not None
        assert "error" in result
        assert ".." in result

    def test_windows_nested_backslash_dotdot_rejected(self):
        result = _check_pattern("foo\\..\\bar")
        assert result is not None
        assert "error" in result


# --- _is_within_base tests ---


class TestIsWithinBase:
    def test_within(self, tmp_path):
        child = tmp_path / "foo.txt"
        child.touch()
        assert _is_within_base(child, tmp_path) is True

    def test_outside(self, tmp_path):
        outside = tmp_path.parent / "outside.txt"
        assert _is_within_base(outside, tmp_path) is False

    def test_nonexistent(self, tmp_path):
        # Path doesn't need to exist for the check
        child = tmp_path / "nonexistent"
        assert _is_within_base(child, tmp_path) is True


# --- list_files tests ---


class TestListFiles:
    def test_basic_glob(self, sandbox):
        result = _list_files("*.txt", ".", str(sandbox))
        assert "README.txt" in result

    def test_nested_glob(self, sandbox):
        result = _list_files("**/*.py", ".", str(sandbox))
        assert "src/main.py" in result
        assert "src/utils.py" in result
        assert "src/sub/deep.py" in result

    def test_subdir_path(self, sandbox):
        result = _list_files("*.py", "src", str(sandbox))
        assert "src/main.py" in result
        assert "src/utils.py" in result
        # deep.py is in src/sub, not directly in src
        assert "deep.py" not in result

    def test_git_excluded(self, sandbox):
        result = _list_files("**/*", ".", str(sandbox))
        assert ".git" not in result
        assert "config" not in result or ".git/config" not in result

    def test_no_matches(self, sandbox):
        result = _list_files("*.rs", ".", str(sandbox))
        assert "No files matched" in result

    def test_dotdot_pattern_rejected(self, sandbox):
        result = _list_files("../*.py", ".", str(sandbox))
        assert "error" in result
        assert ".." in result

    def test_absolute_pattern_rejected(self, sandbox):
        result = _list_files("/etc/*", ".", str(sandbox))
        assert "error" in result
        assert "outside base directory" in result

    def test_path_escape_rejected(self, sandbox):
        result = _list_files("*.py", "../outside", str(sandbox))
        assert "error" in result

    def test_symlink_escape_skipped(self, sandbox):
        """Symlinks pointing outside the sandbox are silently skipped."""
        outside_dir = sandbox.parent / "outside_target"
        outside_dir.mkdir(exist_ok=True)
        (outside_dir / "secret.py").write_text("SECRET = True\n")
        # Create symlink inside sandbox pointing outside
        symlink = sandbox / "escape_link"
        try:
            symlink.symlink_to(outside_dir)
        except OSError:
            pytest.skip("Cannot create symlinks on this platform")

        result = _list_files("**/*.py", ".", str(sandbox))
        assert "secret.py" not in result

    def test_sorted_by_mtime(self, sandbox):
        """Results should be sorted newest first."""
        # Touch files with different mtimes
        old_file = sandbox / "old.py"
        new_file = sandbox / "new.py"
        old_file.write_text("old")
        time.sleep(0.05)
        new_file.write_text("new")

        result = _list_files("*.py", ".", str(sandbox))
        lines = result.strip().split("\n")
        # new.py should appear before old.py
        new_idx = next(i for i, line in enumerate(lines) if "new.py" in line)
        old_idx = next(i for i, line in enumerate(lines) if "old.py" in line)
        assert new_idx < old_idx

    def test_truncation_at_100(self, sandbox):
        """Results should be capped at 100."""
        # Create 110 files
        many_dir = sandbox / "many"
        many_dir.mkdir()
        for i in range(110):
            (many_dir / f"file_{i:04d}.txt").write_text(f"content {i}")

        result = _list_files("**/*.txt", ".", str(sandbox))
        assert "Showing first 100 of 111 matches" in result

    def test_walk_truncation_stops_early(self, sandbox, monkeypatch):
        """Walk should bail out when the visit cap is reached."""
        from swival import tools

        many_dir = sandbox / "many"
        many_dir.mkdir()
        for i in range(50):
            (many_dir / f"file_{i:04d}.txt").write_text(f"content {i}")

        monkeypatch.setattr(tools, "MAX_LIST_WALK_ENTRIES", 10)
        result = _list_files("**/*.txt", ".", str(sandbox))
        assert "Search stopped after visiting 10 entries" in result

    def test_walk_truncation_no_matches(self, sandbox, monkeypatch):
        """When walk halts early with no matches, message should say so."""
        from swival import tools

        many_dir = sandbox / "many"
        many_dir.mkdir()
        for i in range(50):
            (many_dir / f"file_{i:04d}.txt").write_text(f"content {i}")

        monkeypatch.setattr(tools, "MAX_LIST_WALK_ENTRIES", 5)
        result = _list_files("**/*.NOMATCH", ".", str(sandbox))
        assert "No files matched the pattern in the first 5 entries" in result

    def test_nonexistent_path(self, sandbox):
        result = _list_files("*.py", "nonexistent", str(sandbox))
        assert "error" in result

    def test_single_file(self, sandbox):
        """When path is a file, return it directly (ignore pattern)."""
        result = _list_files("*.txt", "src/main.py", str(sandbox))
        assert "src/main.py" in result
        assert "error" not in result

    def test_single_file_with_pattern(self, sandbox):
        """Pattern is irrelevant when path is a single file."""
        result = _list_files("*.NOPE", "README.txt", str(sandbox))
        assert "README.txt" in result
        assert "error" not in result

    def test_broken_symlink_does_not_abort_listing(self, sandbox):
        """Broken symlinks (e.g. npm's node_modules/.bin) are skipped, not fatal."""
        bin_dir = sandbox / "node_modules" / ".bin"
        bin_dir.mkdir(parents=True)
        (sandbox / "real.txt").write_text("hello")
        os.symlink(bin_dir / "missing-target", bin_dir / "is-docker")

        result = _list_files("**/*", ".", str(sandbox))
        assert "error" not in result
        assert "real.txt" in result

    def test_dispatch_list_files(self, sandbox):
        result = dispatch("list_files", {"pattern": "**/*.py"}, str(sandbox))
        assert "src/main.py" in result


# --- grep tests ---


class TestGrep:
    def test_basic_match(self, sandbox):
        result = _grep("import", ".", str(sandbox))
        assert "Found" in result
        assert "src/main.py" in result
        assert "import os" in result

    def test_regex_match(self, sandbox):
        result = _grep(r"def \w+", ".", str(sandbox))
        assert "src/utils.py" in result
        assert "def helper" in result

    def test_include_filter(self, sandbox):
        result = _grep(".", ".", str(sandbox), include="*.txt")
        assert "README.txt" in result
        # Python files should not appear
        assert "main.py" not in result

    def test_binary_skipped(self, sandbox):
        result = _grep("PNG", ".", str(sandbox))
        assert "image.bin" not in result

    def test_git_excluded(self, sandbox):
        result = _grep("core", ".", str(sandbox))
        assert ".git" not in result

    def test_no_matches(self, sandbox):
        result = _grep("zzz_nonexistent_pattern_zzz", ".", str(sandbox))
        assert "No matches found" in result

    def test_invalid_regex(self, sandbox):
        result = _grep("[invalid", ".", str(sandbox))
        assert "error" in result
        assert "invalid regex" in result

    def test_include_dotdot_rejected(self, sandbox):
        result = _grep("import", ".", str(sandbox), include="../*.py")
        assert "error" in result
        assert ".." in result

    def test_include_absolute_rejected(self, sandbox):
        result = _grep("import", ".", str(sandbox), include="/etc/*.py")
        assert "error" in result
        assert "absolute" in result

    def test_path_escape_rejected(self, sandbox):
        result = _grep("import", "../outside", str(sandbox))
        assert "error" in result

    def test_symlink_escape_skipped(self, sandbox):
        """Symlinks pointing outside the sandbox are not searched."""
        outside_dir = sandbox.parent / "outside_grep_target"
        outside_dir.mkdir(exist_ok=True)
        (outside_dir / "secret.py").write_text("SECRET_KEY = 'abc123'\n")
        symlink = sandbox / "linked_dir"
        try:
            symlink.symlink_to(outside_dir)
        except OSError:
            pytest.skip("Cannot create symlinks on this platform")

        result = _grep("SECRET_KEY", ".", str(sandbox))
        assert "secret.py" not in result
        assert "SECRET_KEY" not in result or "No matches" in result

    def test_output_grouped_by_file(self, sandbox):
        """Output should be grouped by file with blank lines between groups."""
        # Write content that will match in multiple files
        (sandbox / "a.py").write_text("MARKER = 1\n")
        (sandbox / "b.py").write_text("MARKER = 2\n")

        result = _grep("MARKER", ".", str(sandbox))
        assert "Found" in result
        # Should have file headers
        assert "a.py:" in result
        assert "b.py:" in result

    def test_line_truncation(self, sandbox):
        """Lines longer than MAX_LINE_LENGTH should be truncated."""
        long_line = "x" * 3000
        (sandbox / "long.py").write_text(f"# {long_line}\n")

        result = _grep("x{10,}", ".", str(sandbox))
        # The matched line should be present but truncated
        lines = result.split("\n")
        for line in lines:
            assert len(line) <= 2100  # 2000 + "  Line N: " prefix

    def test_line_numbers(self, sandbox):
        result = _grep("return", ".", str(sandbox))
        assert "Line 2" in result  # "return 42" is line 2 of utils.py

    def test_dispatch_grep(self, sandbox):
        result = dispatch(
            "grep", {"pattern": "import", "include": "*.py"}, str(sandbox)
        )
        assert "src/main.py" in result

    def test_truncation_at_100_matches(self, sandbox):
        """Results should be capped at 100 matches."""
        many_dir = sandbox / "grep_many"
        many_dir.mkdir()
        # Create files with enough matching lines to exceed 100
        for i in range(20):
            lines = [f"FINDME line {j}" for j in range(10)]
            (many_dir / f"file_{i:04d}.txt").write_text("\n".join(lines))

        result = _grep("FINDME", ".", str(sandbox))
        assert "truncated" in result.lower() or "100" in result

    def test_truncation_prefers_newest_files(self, sandbox):
        """Regression: when >100 matches exist, the top 100 must come from newest files."""
        many_dir = sandbox / "grep_order"
        many_dir.mkdir()
        # Create old files first (lots of matches)
        for i in range(15):
            lines = [f"ORDERMATCH {j}" for j in range(10)]
            (many_dir / f"old_{i:04d}.txt").write_text("\n".join(lines))
            time.sleep(0.01)

        # Create a newest file with one match
        time.sleep(0.05)
        newest = many_dir / "newest.txt"
        newest.write_text("ORDERMATCH from newest\n")

        result = _grep("ORDERMATCH", str(many_dir), str(sandbox))
        # The newest file must appear in results even though there are >100 total matches
        assert "newest.txt" in result

    def test_include_windows_backslash_dotdot_rejected(self, sandbox):
        """Regression: include with backslash .. must be rejected."""
        result = _grep("import", ".", str(sandbox), include="..\\*.py")
        assert "error" in result
        assert ".." in result

    def test_include_double_star_glob(self, sandbox):
        """include='**/*.zig' should match .zig files at every depth."""
        (sandbox / "a.zig").write_text("MATCH\n")
        d1 = sandbox / "dir1"
        d1.mkdir()
        (d1 / "b.zig").write_text("MATCH\n")
        d2 = d1 / "dir2"
        d2.mkdir()
        (d2 / "c.zig").write_text("MATCH\n")
        # A non-.zig file should be excluded
        (d2 / "skip.txt").write_text("MATCH\n")

        result = _grep("MATCH", ".", str(sandbox), include="**/*.zig")
        assert "a.zig" in result
        assert "b.zig" in result
        assert "c.zig" in result
        assert "skip.txt" not in result

    def test_include_nested_double_star_glob(self, sandbox):
        """include='**/**/*.py' should also work."""
        result = _grep("import", ".", str(sandbox), include="**/**/*.py")
        assert "Found" in result
        assert "main.py" in result

    def test_case_insensitive(self, sandbox):
        """case_insensitive=True should match regardless of case."""
        (sandbox / "ci.txt").write_text("Hello World\nhello world\nHELLO WORLD\n")
        result = _grep(
            "hello", ".", str(sandbox), include="ci.txt", case_insensitive=True
        )
        assert "Found 3 matches" in result

    def test_single_file(self, sandbox):
        """When path is a file, grep it directly (no include needed)."""
        result = _grep("import", "src/main.py", str(sandbox))
        assert "Found" in result
        assert "main.py" in result
        assert "import os" in result

    def test_single_file_with_include(self, sandbox):
        """include is ignored when path is a specific file."""
        result = _grep("import", "src/main.py", str(sandbox), include="*.py")
        assert "Found" in result
        assert "import os" in result

    def test_single_file_with_mismatched_include(self, sandbox):
        """include is irrelevant for a single file, even when it doesn't match."""
        result = _grep("import", "src/main.py", str(sandbox), include="*.txt")
        assert "Found" in result
        assert "import os" in result

    def test_single_file_binary_skipped(self, sandbox):
        """Binary file returns 'No matches found.' — silent skip, no error."""
        result = _grep("PNG", "image.bin", str(sandbox))
        assert result == "No matches found."
        assert "error" not in result.lower()

    def test_single_file_no_match(self, sandbox):
        """Single file with no matching lines."""
        result = _grep("zzz_no_such_thing", "src/main.py", str(sandbox))
        assert result == "No matches found."

    def test_case_sensitive_default(self, sandbox):
        """Default search should be case-sensitive."""
        (sandbox / "cs.txt").write_text("Hello World\nhello world\nHELLO WORLD\n")
        result = _grep("hello", ".", str(sandbox), include="cs.txt")
        assert "Found 1 match" in result

    def test_grep_context_lines_zero(self, sandbox):
        """context_lines=0 produces identical output to default (no markers)."""
        result_default = _grep("import", "src/main.py", str(sandbox))
        result_zero = _grep("import", "src/main.py", str(sandbox), context_lines=0)
        assert result_default == result_zero
        assert "<<<" not in result_zero

    def test_grep_context_lines_one(self, sandbox):
        """context_lines=1 shows ±1 lines with <<< marker on matches."""
        (sandbox / "ctx.txt").write_text("aaa\nbbb\nccc\nddd\neee\n")
        result = _grep("ccc", "ctx.txt", str(sandbox), context_lines=1)
        assert "Found 1 match" in result
        assert "Line 2: bbb" in result
        assert "Line 3: ccc  <<<" in result
        assert "Line 4: ddd" in result
        # Lines outside the window should not appear
        assert "Line 1" not in result
        assert "Line 5" not in result

    def test_grep_context_lines_overlapping(self, sandbox):
        """Overlapping context windows are merged, no duplicate lines."""
        (sandbox / "overlap.txt").write_text("1\n2\n3\n4\n5\n6\n7\n")
        # Match lines 3 and 5, context_lines=1 → windows [2-4] and [4-6] overlap at 4
        result = _grep(r"^[35]$", "overlap.txt", str(sandbox), context_lines=1)
        assert "Found 2 match" in result
        lines = result.split("\n")
        line_nums = [
            x.strip().split(":")[0] for x in lines if x.strip().startswith("Line")
        ]
        # No duplicate line numbers
        assert len(line_nums) == len(set(line_nums))
        # No separator between merged blocks
        assert "--" not in result
        # Lines 2-6 should be present
        assert "Line 2: 2" in result
        assert "Line 3: 3  <<<" in result
        assert "Line 4: 4" in result
        assert "Line 5: 5  <<<" in result
        assert "Line 6: 6" in result

    def test_grep_context_lines_at_file_edges(self, sandbox):
        """Context at file start/end is truncated gracefully."""
        (sandbox / "edge.txt").write_text("first\nsecond\nthird\n")
        # Match first line with context_lines=2
        result = _grep("first", "edge.txt", str(sandbox), context_lines=2)
        assert "Line 1: first  <<<" in result
        assert "Line 2: second" in result
        assert "Line 3: third" in result
        # Match last line
        result2 = _grep("third", "edge.txt", str(sandbox), context_lines=2)
        assert "Line 1: first" in result2
        assert "Line 2: second" in result2
        assert "Line 3: third  <<<" in result2

    def test_grep_context_lines_marker(self, sandbox):
        """Only matching lines have <<< marker, context lines don't."""
        (sandbox / "mark.txt").write_text("aaa\nTARGET\nccc\n")
        result = _grep("TARGET", "mark.txt", str(sandbox), context_lines=1)
        for line in result.split("\n"):
            if "TARGET" in line and line.strip().startswith("Line"):
                assert line.endswith("<<<")
            elif line.strip().startswith("Line"):
                assert not line.endswith("<<<")

    def test_grep_context_lines_multi_file(self, sandbox):
        """Context works across multiple files."""
        (sandbox / "m1.txt").write_text("aaa\nMATCH\nccc\n")
        (sandbox / "m2.txt").write_text("xxx\nyyy\nMATCH\nzzz\n")
        result = _grep("MATCH", ".", str(sandbox), include="m*.txt", context_lines=1)
        assert "m1.txt" in result
        assert "m2.txt" in result
        assert "<<<" in result

    def test_grep_context_lines_byte_cap(self, sandbox):
        """Context lines count toward MAX_OUTPUT_BYTES but not MAX_GREP_MATCHES."""
        many_dir = sandbox / "ctx_cap"
        many_dir.mkdir()
        # Create files with matches — context adds lines but shouldn't affect match cap
        for i in range(5):
            content = "\n".join(f"line{j}" for j in range(10))
            (many_dir / f"f{i}.txt").write_text(content)
        result = _grep("line5", str(many_dir), str(sandbox), context_lines=2)
        assert "Found 5 match" in result
        # Context lines should be present
        assert "line3" in result or "line4" in result

    def test_grep_context_lines_negative(self, sandbox):
        """Negative context_lines is treated as 0."""
        result_neg = _grep("import", "src/main.py", str(sandbox), context_lines=-1)
        result_zero = _grep("import", "src/main.py", str(sandbox), context_lines=0)
        assert result_neg == result_zero
        assert "<<<" not in result_neg

    def test_grep_context_lines_separator(self, sandbox):
        """Non-contiguous blocks within a file are separated by --."""
        (sandbox / "sep.txt").write_text("1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n")
        # Match lines 2 and 9, context_lines=1 → blocks [1-3] and [8-10], non-contiguous
        result = _grep(r"^[29]$", "sep.txt", str(sandbox), context_lines=1)
        assert "  --" in result


class TestListFileTilde:
    def test_tilde_pattern_finds_files(self, tmp_path, monkeypatch):
        """~/subdir/**/*.py finds files when HOME is under base_dir."""
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        sub = home / "subdir"
        sub.mkdir()
        (sub / "a.py").write_text("# a")
        (sub / "b.py").write_text("# b")

        result = _list_files("~/subdir/**/*.py", ".", str(tmp_path))
        assert "a.py" in result
        assert "b.py" in result

    def test_tilde_pattern_outside_roots_rejected(self, tmp_path, monkeypatch):
        """~/subdir/**/*.py fails when HOME is outside allowed roots."""
        home = tmp_path / "home"
        home.mkdir()
        base = tmp_path / "project"
        base.mkdir()
        monkeypatch.setenv("HOME", str(home))

        result = _list_files("~/subdir/**/*.py", ".", str(base))
        assert result.startswith("error:")

    def test_tilde_otheruser_pattern_rejected(self, tmp_path):
        """~otheruser/**/*.py returns an error."""
        result = _list_files("~otheruser/**/*.py", ".", str(tmp_path))
        assert result.startswith("error:")
        assert "~user syntax" in result


class TestGrepTilde:
    def test_grep_tilde_path(self, tmp_path, monkeypatch):
        """grep with path=~/subdir works when HOME is under base_dir."""
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        sub = home / "subdir"
        sub.mkdir()
        (sub / "code.py").write_text("def hello():\n    pass\n")

        result = _grep("hello", "~/subdir", str(tmp_path))
        assert "hello" in result
        assert "No matches" not in result

    def test_grep_tilde_path_outside_roots_rejected(self, tmp_path, monkeypatch):
        """grep with path=~/subdir fails when HOME is outside allowed roots."""
        home = tmp_path / "home"
        home.mkdir()
        base = tmp_path / "project"
        base.mkdir()
        monkeypatch.setenv("HOME", str(home))

        result = _grep("pattern", "~/subdir", str(base))
        assert result.startswith("error:")

    def test_grep_tilde_include_no_match(self, tmp_path):
        """grep with include=~weird*.py is not an error — just no matches."""
        (tmp_path / "normal.py").write_text("hello\n")
        result = _grep("hello", ".", str(tmp_path), include="~weird*.py")
        assert "No matches" in result
        assert not result.startswith("error:")


# --- grep memory bounds (issue #34) ---


class TestGrepStreaming:
    """grep must not hold whole files or every match in memory."""

    def test_iter_lines_matches_splitlines_across_chunks(self, tmp_path):
        from swival import tools
        from swival.tools import _iter_lines

        chunk = tools.GREP_CHUNK_BYTES
        # Pieces chosen so that "\r\n" and a multibyte character straddle
        # the first chunk boundary, with every separator splitlines knows.
        text = "a" * (chunk - 1) + "\r\n" + "b" * (chunk - 2) + "é" + "\n"
        text += "x\x0cy\x0bz\x1cw\x1dv\x1eu\x85t s r\rq\r\n\n\rlast"
        fp = tmp_path / "chunky.txt"
        fp.write_bytes(text.encode("utf-8"))

        assert list(_iter_lines(fp)) == text.splitlines()

    def test_iter_lines_trailing_cr_and_empty_file(self, tmp_path):
        from swival.tools import _iter_lines

        fp = tmp_path / "cr.txt"
        fp.write_bytes(b"abc\r")
        assert list(_iter_lines(fp)) == ["abc"]
        fp.write_bytes(b"")
        assert list(_iter_lines(fp)) == []

    def test_iter_lines_binary_yields_nothing(self, tmp_path):
        from swival.tools import _iter_lines

        fp = tmp_path / "bin.dat"
        fp.write_bytes(b"needle\x00needle\n")
        assert list(_iter_lines(fp)) == []

    def test_invalid_utf8_mid_file_skips_whole_file(self, tmp_path):
        """A decode error after some matches drops the file, as read_text did."""
        (tmp_path / "bad.txt").write_bytes(b"needle first\n" * 5 + b"\xff\xfe\n")
        (tmp_path / "good.txt").write_text("needle ok\n")

        result = _grep("needle", ".", str(tmp_path))
        assert "Found 1 match" in result
        assert "bad.txt" not in result
        assert "good.txt" in result

    def test_line_numbers_agree_with_read_file(self, tmp_path):
        """Form feeds are line breaks for read_file, so they must be for grep too."""
        from swival.tools import _read_file

        (tmp_path / "ff.c").write_text("int a;\n\x0cint needle;\nint b;\n")
        result = _grep("needle", "ff.c", str(tmp_path))
        assert "Line 3: int needle;" in result
        assert "int needle;" in _read_file("ff.c", str(tmp_path), offset=3, limit=1)

    def test_peak_memory_stays_flat(self, tmp_path):
        """Regression for the OOM in issue #34.

        The old implementation retained every matching line of every file
        (and, with context, every line of every file) before capping,
        so peak memory grew with the corpus. It now stays near constant.
        """
        import tracemalloc

        line = "needle " + "p" * 40 + "\n"
        for i in range(8):
            (tmp_path / f"big{i}.txt").write_text(line * 20_000)
        corpus_bytes = sum(p.stat().st_size for p in tmp_path.iterdir())
        assert corpus_bytes > 7_000_000

        for ctx in (0, 2):
            tracemalloc.start()
            result = _grep("needle", ".", str(tmp_path), context_lines=ctx)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            assert "Found 160000 matches" in result
            assert peak < 3_000_000, f"peak {peak} bytes with context_lines={ctx}"

    def test_exact_count_and_newest_first_with_context(self, tmp_path):
        """Matches past the cap are still counted; retained ones are the newest."""
        for i in range(5):
            (tmp_path / f"old{i}.txt").write_text("needle\n" * 50)
            os.utime(tmp_path / f"old{i}.txt", (1_000_000 + i, 1_000_000 + i))
        (tmp_path / "new.txt").write_text("above\nneedle here\nbelow\n")
        os.utime(tmp_path / "new.txt", (2_000_000, 2_000_000))

        result = _grep("needle", ".", str(tmp_path), context_lines=1)
        assert result.startswith("Found 251 matches")
        assert "new.txt:" in result
        assert "Line 1: above" in result
        assert "Line 2: needle here  <<<" in result
        assert "Line 3: below" in result
        assert "Results truncated: showing first 100 matches" in result
        # 100 retained: 1 from new.txt, 50 from old4, 49 from old3.
        assert "old4.txt:" in result
        assert "old3.txt:" in result
        assert "old2.txt:" not in result

    def test_context_blocks_merge_and_trail_past_cap(self, tmp_path):
        """Context windows merge when they touch, and the last retained
        match still shows its trailing context even when later matches
        are beyond the cap."""
        from swival import tools

        lines = [
            "m1",
            "gap",
            "gap",
            "m2",
            "tail",
            "tail",
            "far",
            "far",
            "far",
            "m3",
            "end",
        ]
        (tmp_path / "f.txt").write_text("\n".join(lines) + "\n")

        result = _grep(r"^m\d$", "f.txt", str(tmp_path), context_lines=2)
        assert result.count("  --") == 1
        assert "Line 3: gap" in result and "Line 6: tail" in result
        assert "Line 7: far" not in result
        assert "Line 8: far  <<<" not in result and "Line 8: far" in result

        # With the cap at 2, m3 is only counted. Its line still appears
        # as plain trailing context of m2, and nothing after it.
        original = tools.MAX_GREP_MATCHES
        tools.MAX_GREP_MATCHES = 2
        try:
            result = _grep(r"^m\d$", "f.txt", str(tmp_path), context_lines=6)
        finally:
            tools.MAX_GREP_MATCHES = original
        assert result.startswith("Found 3 matches")
        assert result.count("<<<") == 2
        assert "Line 10: m3\n(Results truncated" in result
        assert "Line 11: end" not in result

    def test_file_cap_keeps_newest_files_with_note(self, tmp_path):
        """Under the file cap, the newest files are searched regardless of
        the order in which the walk visits them."""
        from swival import tools

        for i in range(6):
            (tmp_path / f"f{i}.txt").write_text(f"needle f{i}\n")
        # Make the alphabetically first files the oldest ones.
        for i in range(6):
            os.utime(tmp_path / f"f{i}.txt", (1_000_000 + i, 1_000_000 + i))
        original = tools.MAX_GREP_FILES
        tools.MAX_GREP_FILES = 4
        try:
            result = _grep("needle", ".", str(tmp_path))
            missing = _grep("absent", ".", str(tmp_path))
        finally:
            tools.MAX_GREP_FILES = original

        assert result.startswith("Found 4 matches")
        for i in (2, 3, 4, 5):
            assert f"needle f{i}" in result
        for i in (0, 1):
            assert f"needle f{i}" not in result
        note = "Only the 4 most recently modified of 6 files were searched"
        assert note in result
        assert missing.startswith("No matches found.")
        assert note in missing

    def test_symlinked_directory_is_not_descended(self, tmp_path):
        """Parity with os.walk: symlinks to directories are listed, not walked."""
        (tmp_path / "real").mkdir()
        (tmp_path / "real" / "a.txt").write_text("needle\n")
        try:
            (tmp_path / "link").symlink_to(tmp_path / "real")
        except OSError:
            pytest.skip("Cannot create symlinks on this platform")

        result = _grep("needle", ".", str(tmp_path))
        assert result.startswith("Found 1 match")
        assert "real/a.txt" in result
        assert "link/" not in result

    def test_unreadable_directory_is_skipped(self, tmp_path):
        if os.geteuid() == 0:
            pytest.skip("root ignores directory permissions")
        (tmp_path / "ok.txt").write_text("needle\n")
        locked = tmp_path / "locked"
        locked.mkdir()
        (locked / "hidden.txt").write_text("needle\n")
        locked.chmod(0)
        try:
            result = _grep("needle", ".", str(tmp_path))
        finally:
            locked.chmod(0o755)
        assert result.startswith("Found 1 match")
        assert "ok.txt" in result

    def test_line_cap_is_exact_at_the_boundary(self, tmp_path):
        """A line of exactly the cap is accepted even when its CRLF straddles
        a chunk; one character more is rejected however the line ends."""
        from swival import tools
        from swival.tools import _GrepLineTooLong, _iter_lines

        fp = tmp_path / "edge.txt"
        original = (tools.MAX_GREP_LINE_CHARS, tools.GREP_CHUNK_BYTES)
        tools.MAX_GREP_LINE_CHARS, tools.GREP_CHUNK_BYTES = 4, 5
        try:
            fp.write_bytes(b"abcd\r\nxy\n")
            assert list(_iter_lines(fp)) == ["abcd", "xy"]
            fp.write_bytes(b"abcd\r")
            assert list(_iter_lines(fp)) == ["abcd"]
            fp.write_bytes(b"abcde\n")
            with pytest.raises(_GrepLineTooLong):
                list(_iter_lines(fp))
            fp.write_bytes(b"x\nabcde")
            with pytest.raises(_GrepLineTooLong):
                list(_iter_lines(fp))
        finally:
            tools.MAX_GREP_LINE_CHARS, tools.GREP_CHUNK_BYTES = original

    def test_context_lines_are_clamped(self, tmp_path):
        from swival import tools

        (tmp_path / "f.txt").write_text("\n".join(str(i) for i in range(400)) + "\n")
        capped = _grep(
            "^200$", "f.txt", str(tmp_path), context_lines=tools.MAX_GREP_CONTEXT_LINES
        )
        huge = _grep("^200$", "f.txt", str(tmp_path), context_lines=10**9)
        assert huge == capped
        assert "Line 101: 100" in huge and "Line 100: 99" not in huge
        assert "Line 301: 300" in huge and "Line 302: 301" not in huge

    def test_oversized_line_skips_file_with_note(self, tmp_path):
        from swival import tools

        (tmp_path / "minified.js").write_text("needle " + "z" * 5000)
        (tmp_path / "ok.js").write_text("needle\n")
        original = tools.MAX_GREP_LINE_CHARS
        tools.MAX_GREP_LINE_CHARS = 1024
        try:
            result = _grep("needle", ".", str(tmp_path))
        finally:
            tools.MAX_GREP_LINE_CHARS = original

        assert "Found 1 match" in result
        assert "ok.js" in result
        assert "minified.js" not in result
        assert "Skipped 1 file containing a line longer than" in result

    def test_long_retained_lines_are_clipped_with_context(self, tmp_path):
        (tmp_path / "long.txt").write_text(
            "needle " + "q" * 5000 + "\nctx " + "r" * 5000 + "\n"
        )
        result = _grep("needle", "long.txt", str(tmp_path), context_lines=1)
        for line in result.splitlines():
            assert len(line) <= 2100

    def test_equal_mtime_cap_keeps_first_paths(self, tmp_path):
        """With equal mtimes, the files kept under the cap are the ones the
        output order lists first."""
        from swival import tools

        for name in "abcde":
            (tmp_path / f"{name}.txt").write_text(f"needle {name}\n")
            os.utime(tmp_path / f"{name}.txt", (1_000_000, 1_000_000))
        original = tools.MAX_GREP_FILES
        tools.MAX_GREP_FILES = 3
        try:
            result = _grep("needle", ".", str(tmp_path))
        finally:
            tools.MAX_GREP_FILES = original
        assert (
            result.index("needle a")
            < result.index("needle b")
            < result.index("needle c")
        )
        assert "needle d" not in result and "needle e" not in result

    def test_directory_iteration_error_is_skipped(self, tmp_path, monkeypatch):
        """An OSError while listing a directory drops that directory only,
        like os.walk with onerror=None."""
        (tmp_path / "flaky").mkdir()
        (tmp_path / "flaky" / "x.txt").write_text("needle flaky\n")
        (tmp_path / "flaky" / "y.txt").write_text("needle flaky\n")
        (tmp_path / "ok.txt").write_text("needle ok\n")
        real_scandir = os.scandir

        class _FlakyScanner:
            def __init__(self, inner):
                self._inner = inner

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self._inner.close()

            def __iter__(self):
                yield next(iter(self._inner))
                raise OSError("read error")

        def fake_scandir(path):
            scanner = real_scandir(path)
            if os.path.basename(path) == "flaky":
                return _FlakyScanner(scanner)
            return scanner

        monkeypatch.setattr(os, "scandir", fake_scandir)
        result = _grep("needle", ".", str(tmp_path))
        # The entry seen before the error is kept, the rest of the
        # directory is dropped, and the search goes on elsewhere.
        assert result.startswith("Found 2 matches")
        assert "needle ok" in result
        assert result.count("needle flaky") == 1

    def test_binary_probe_does_not_shrink_with_chunk_size(self, tmp_path):
        from swival import tools
        from swival.tools import _iter_lines

        fp = tmp_path / "late_nul.bin"
        fp.write_bytes(b"needle\n" * 100 + b"\x00" + b"needle\n")
        original = tools.GREP_CHUNK_BYTES
        tools.GREP_CHUNK_BYTES = 16
        try:
            assert list(_iter_lines(fp)) == []
        finally:
            tools.GREP_CHUNK_BYTES = original

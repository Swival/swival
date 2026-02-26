"""Tests for list_files and grep tools."""

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
        assert "truncated" in result.lower() or "100" in result

    def test_nonexistent_path(self, sandbox):
        result = _list_files("*.py", "nonexistent", str(sandbox))
        assert "error" in result

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

    def test_case_sensitive_default(self, sandbox):
        """Default search should be case-sensitive."""
        (sandbox / "cs.txt").write_text("Hello World\nhello world\nHELLO WORLD\n")
        result = _grep("hello", ".", str(sandbox), include="cs.txt")
        assert "Found 1 match" in result

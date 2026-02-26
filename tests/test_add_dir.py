"""Tests for --add-dir and --add-dir-ro features."""

import os
import sys

import pytest

from swival.tools import (
    safe_resolve,
    _read_file,
    _write_file,
    _edit_file,
    _delete_file,
    _list_files,
    _grep,
    dispatch,
)


# =========================================================================
# safe_resolve
# =========================================================================


class TestSafeResolveExtraWriteRoots:
    def test_path_inside_extra_write_root(self, tmp_path):
        """A path inside an extra_write_roots entry resolves successfully."""
        base = tmp_path / "project"
        base.mkdir()
        extra = tmp_path / "extra"
        extra.mkdir()
        target = extra / "file.txt"
        target.write_text("hello", encoding="utf-8")

        result = safe_resolve(str(target), str(base), extra_write_roots=[extra])
        assert result == target.resolve()

    def test_path_outside_all_roots_raises(self, tmp_path):
        """A path outside base_dir, extra_read_roots, and extra_write_roots raises ValueError."""
        base = tmp_path / "project"
        base.mkdir()
        extra = tmp_path / "extra"
        extra.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        target = outside / "secret.txt"
        target.write_text("secret", encoding="utf-8")

        with pytest.raises(ValueError, match="outside base directory"):
            safe_resolve(str(target), str(base), extra_write_roots=[extra])

    def test_extra_write_root_checked_after_read_roots(self, tmp_path):
        """extra_write_roots is checked after extra_read_roots; both work."""
        base = tmp_path / "project"
        base.mkdir()
        read_dir = tmp_path / "readonly"
        read_dir.mkdir()
        write_dir = tmp_path / "writable"
        write_dir.mkdir()

        read_file = read_dir / "r.txt"
        read_file.write_text("r", encoding="utf-8")
        write_file = write_dir / "w.txt"
        write_file.write_text("w", encoding="utf-8")

        # Both should resolve
        assert (
            safe_resolve(str(read_file), str(base), extra_read_roots=[read_dir])
            == read_file.resolve()
        )
        assert (
            safe_resolve(str(write_file), str(base), extra_write_roots=[write_dir])
            == write_file.resolve()
        )


# =========================================================================
# File operations with extra_write_roots
# =========================================================================


class TestFileOpsExtraWriteRoots:
    def test_read_file_extra_write_root(self, tmp_path):
        """Reading a file inside an allowed dir succeeds."""
        base = tmp_path / "project"
        base.mkdir()
        extra = tmp_path / "extra"
        extra.mkdir()
        target = extra / "data.txt"
        target.write_text("line1\nline2\n", encoding="utf-8")

        result = _read_file(str(target), str(base), extra_write_roots=[extra])
        assert "1: line1" in result
        assert "2: line2" in result

    def test_write_file_extra_write_root(self, tmp_path):
        """Writing a file inside an allowed dir succeeds."""
        base = tmp_path / "project"
        base.mkdir()
        extra = tmp_path / "extra"
        extra.mkdir()
        target = extra / "output.txt"

        result = _write_file(
            str(target), "hello world", str(base), extra_write_roots=[extra]
        )
        assert "Wrote" in result
        assert target.read_text(encoding="utf-8") == "hello world"

    def test_edit_file_extra_write_root(self, tmp_path):
        """Editing a file inside an allowed dir succeeds."""
        base = tmp_path / "project"
        base.mkdir()
        extra = tmp_path / "extra"
        extra.mkdir()
        target = extra / "config.txt"
        target.write_text("old value", encoding="utf-8")

        result = _edit_file(
            str(target), "old value", "new value", str(base), extra_write_roots=[extra]
        )
        assert "Edited" in result
        assert target.read_text(encoding="utf-8") == "new value"

    def test_write_file_skill_read_root_rejected(self, tmp_path):
        """Writing to a path that's only in extra_read_roots (skill path) is rejected."""
        base = tmp_path / "project"
        base.mkdir()
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        target = skill_dir / "file.txt"

        # _write_file does not accept extra_read_roots, so a path only in
        # extra_read_roots should be rejected (it's not in extra_write_roots)
        result = _write_file(str(target), "data", str(base))
        assert result.startswith("error:")

    def test_read_file_outside_all_roots_rejected(self, tmp_path):
        """Reading a file outside all roots is rejected."""
        base = tmp_path / "project"
        base.mkdir()
        extra = tmp_path / "extra"
        extra.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        target = outside / "secret.txt"
        target.write_text("secret", encoding="utf-8")

        result = _read_file(str(target), str(base), extra_write_roots=[extra])
        assert result.startswith("error:")


# =========================================================================
# list_files / grep with extra_write_roots
# =========================================================================


class TestListGrepExtraWriteRoots:
    def test_list_files_extra_write_root(self, tmp_path):
        """Listing an allowed dir returns results with absolute paths."""
        base = tmp_path / "project"
        base.mkdir()
        extra = tmp_path / "extra"
        extra.mkdir()
        (extra / "a.py").write_text("# a", encoding="utf-8")
        (extra / "b.py").write_text("# b", encoding="utf-8")

        result = _list_files(
            "**/*.py", str(extra), str(base), extra_write_roots=[extra]
        )
        assert "a.py" in result
        assert "b.py" in result
        # Paths should be absolute since extra dir is outside base_dir
        for line in result.strip().split("\n"):
            if line.startswith("("):
                continue  # skip truncation messages
            assert os.path.isabs(line), f"Expected absolute path, got: {line}"

    def test_grep_extra_write_root(self, tmp_path):
        """Grepping in an allowed dir returns matches with absolute paths."""
        base = tmp_path / "project"
        base.mkdir()
        extra = tmp_path / "extra"
        extra.mkdir()
        (extra / "code.py").write_text("def hello():\n    pass\n", encoding="utf-8")

        result = _grep("hello", str(extra), str(base), extra_write_roots=[extra])
        assert "hello" in result
        assert "No matches" not in result

    def test_list_files_outside_extra_write_root_rejected(self, tmp_path):
        """Listing a dir outside all roots is rejected."""
        base = tmp_path / "project"
        base.mkdir()
        extra = tmp_path / "extra"
        extra.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()

        result = _list_files("**/*", str(outside), str(base), extra_write_roots=[extra])
        assert result.startswith("error:")

    def test_list_files_symlink_escape_blocked(self, tmp_path):
        """A symlink inside an allowed dir pointing outside all roots is excluded."""
        base = tmp_path / "project"
        base.mkdir()
        extra = tmp_path / "extra"
        extra.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        secret = outside / "secret.txt"
        secret.write_text("secret data", encoding="utf-8")

        # Create a symlink inside extra pointing to outside
        link = extra / "escape.txt"
        link.symlink_to(secret)
        # Also create a legit file
        (extra / "legit.txt").write_text("ok", encoding="utf-8")

        result = _list_files(
            "**/*.txt", str(extra), str(base), extra_write_roots=[extra]
        )
        assert "legit.txt" in result
        assert "escape.txt" not in result
        assert "secret" not in result


# =========================================================================
# CLI validation
# =========================================================================


class TestAddDirCLIValidation:
    """Integration tests that call agent.main() to verify --add-dir validation."""

    def test_add_dir_nonexistent(self, tmp_path, monkeypatch):
        """Passing a non-existent path exits with sys.exit(1)."""
        from swival import agent

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "--base-dir",
                str(tmp_path),
                "--add-dir",
                str(tmp_path / "nope"),
                "question",
            ],
        )
        with pytest.raises(SystemExit) as exc_info:
            agent.main()
        assert exc_info.value.code == 1

    def test_add_dir_is_file(self, tmp_path, monkeypatch):
        """Passing a file (not directory) exits with sys.exit(1)."""
        f = tmp_path / "afile.txt"
        f.write_text("hi", encoding="utf-8")

        from swival import agent

        monkeypatch.setattr(
            sys,
            "argv",
            ["agent", "--base-dir", str(tmp_path), "--add-dir", str(f), "question"],
        )
        with pytest.raises(SystemExit) as exc_info:
            agent.main()
        assert exc_info.value.code == 1

    def test_add_dir_root_rejected(self, tmp_path, monkeypatch):
        """Passing / (filesystem root) exits with sys.exit(1)."""
        from swival import agent

        monkeypatch.setattr(
            sys,
            "argv",
            ["agent", "--base-dir", str(tmp_path), "--add-dir", "/", "question"],
        )
        with pytest.raises(SystemExit) as exc_info:
            agent.main()
        assert exc_info.value.code == 1

    def test_add_dir_tilde_expansion(self, tmp_path, monkeypatch):
        """~/somedir resolves correctly via expanduser() in main()."""
        target = tmp_path / "homedir"
        target.mkdir()
        monkeypatch.setenv("HOME", str(tmp_path))

        from swival import agent

        # Let main() get past validation, then capture allowed_dirs
        # by patching discover_model to avoid LM Studio connection
        captured = {}

        def fake_run(messages, tools, **kwargs):
            captured["extra_write_roots"] = kwargs.get("extra_write_roots", [])
            return "done", False

        monkeypatch.setattr(agent, "run_agent_loop", fake_run)
        monkeypatch.setattr(
            agent, "discover_model", lambda *a, **kw: ("test-model", 4096)
        )

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "--base-dir",
                str(tmp_path),
                "--add-dir",
                "~/homedir",
                "question",
            ],
        )
        agent.main()

        assert len(captured["extra_write_roots"]) == 1
        assert captured["extra_write_roots"][0] == target.resolve()

    def test_add_dir_with_yolo_valid(self, tmp_path, monkeypatch):
        """--yolo --add-dir <valid> validates at startup but doesn't error."""
        extra = tmp_path / "extra"
        extra.mkdir()

        from swival import agent

        captured = {}

        def fake_run(messages, tools, **kwargs):
            captured["extra_write_roots"] = kwargs.get("extra_write_roots", [])
            captured["yolo"] = kwargs.get("yolo", False)
            return "done", False

        monkeypatch.setattr(agent, "run_agent_loop", fake_run)
        monkeypatch.setattr(
            agent, "discover_model", lambda *a, **kw: ("test-model", 4096)
        )

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "--base-dir",
                str(tmp_path),
                "--yolo",
                "--add-dir",
                str(extra),
                "question",
            ],
        )
        agent.main()

        assert captured["yolo"] is True
        assert len(captured["extra_write_roots"]) == 1

    def test_add_dir_with_yolo_invalid(self, tmp_path, monkeypatch):
        """--yolo --add-dir <nonexistent> still exits with sys.exit(1)."""
        from swival import agent

        monkeypatch.setattr(
            agent, "discover_model", lambda *a, **kw: ("test-model", 4096)
        )

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "--base-dir",
                str(tmp_path),
                "--yolo",
                "--add-dir",
                str(tmp_path / "nope"),
                "question",
            ],
        )
        with pytest.raises(SystemExit) as exc_info:
            agent.main()
        assert exc_info.value.code == 1


# =========================================================================
# dispatch integration
# =========================================================================


class TestDispatchExtraWriteRoots:
    def test_dispatch_read_file_extra_write_roots(self, tmp_path):
        """dispatch() correctly forwards extra_write_roots for read_file."""
        base = tmp_path / "project"
        base.mkdir()
        extra = tmp_path / "extra"
        extra.mkdir()
        target = extra / "file.txt"
        target.write_text("content\n", encoding="utf-8")

        result = dispatch(
            "read_file",
            {"file_path": str(target)},
            str(base),
            extra_write_roots=[extra],
        )
        assert "1: content" in result

    def test_dispatch_write_file_extra_write_roots(self, tmp_path):
        """dispatch() correctly forwards extra_write_roots for write_file."""
        base = tmp_path / "project"
        base.mkdir()
        extra = tmp_path / "extra"
        extra.mkdir()
        target = extra / "new.txt"

        result = dispatch(
            "write_file",
            {"file_path": str(target), "content": "hello"},
            str(base),
            extra_write_roots=[extra],
        )
        assert "Wrote" in result
        assert target.read_text(encoding="utf-8") == "hello"

    def test_dispatch_edit_file_extra_write_roots(self, tmp_path):
        """dispatch() correctly forwards extra_write_roots for edit_file."""
        base = tmp_path / "project"
        base.mkdir()
        extra = tmp_path / "extra"
        extra.mkdir()
        target = extra / "edit.txt"
        target.write_text("old text", encoding="utf-8")

        result = dispatch(
            "edit_file",
            {
                "file_path": str(target),
                "old_string": "old text",
                "new_string": "new text",
            },
            str(base),
            extra_write_roots=[extra],
        )
        assert "Edited" in result
        assert target.read_text(encoding="utf-8") == "new text"

    def test_dispatch_list_files_extra_write_roots(self, tmp_path):
        """dispatch() correctly forwards extra_write_roots for list_files."""
        base = tmp_path / "project"
        base.mkdir()
        extra = tmp_path / "extra"
        extra.mkdir()
        (extra / "test.py").write_text("# test", encoding="utf-8")

        result = dispatch(
            "list_files",
            {"pattern": "**/*.py", "path": str(extra)},
            str(base),
            extra_write_roots=[extra],
        )
        assert "test.py" in result

    def test_dispatch_grep_extra_write_roots(self, tmp_path):
        """dispatch() correctly forwards extra_write_roots for grep."""
        base = tmp_path / "project"
        base.mkdir()
        extra = tmp_path / "extra"
        extra.mkdir()
        (extra / "search.py").write_text("def find_me():\n    pass\n", encoding="utf-8")

        result = dispatch(
            "grep",
            {"pattern": "find_me", "path": str(extra)},
            str(base),
            extra_write_roots=[extra],
        )
        assert "find_me" in result
        assert "No matches" not in result

    def test_dispatch_without_extra_write_roots_rejects(self, tmp_path):
        """dispatch() without extra_write_roots rejects access outside base."""
        base = tmp_path / "project"
        base.mkdir()
        extra = tmp_path / "extra"
        extra.mkdir()
        target = extra / "file.txt"
        target.write_text("data", encoding="utf-8")

        result = dispatch(
            "read_file",
            {"file_path": str(target)},
            str(base),
        )
        assert result.startswith("error:")


# =========================================================================
# safe_resolve with extra_read_roots (--add-dir-ro)
# =========================================================================


class TestSafeResolveExtraReadRoots:
    def test_path_inside_extra_read_root(self, tmp_path):
        """A path inside an extra_read_roots entry resolves successfully."""
        base = tmp_path / "project"
        base.mkdir()
        ro = tmp_path / "readonly"
        ro.mkdir()
        target = ro / "file.txt"
        target.write_text("hello", encoding="utf-8")

        result = safe_resolve(str(target), str(base), extra_read_roots=[ro])
        assert result == target.resolve()

    def test_path_outside_all_roots_raises(self, tmp_path):
        """A path outside base_dir, extra_read_roots, and extra_write_roots raises ValueError."""
        base = tmp_path / "project"
        base.mkdir()
        ro = tmp_path / "readonly"
        ro.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        target = outside / "secret.txt"
        target.write_text("secret", encoding="utf-8")

        with pytest.raises(ValueError, match="outside base directory"):
            safe_resolve(str(target), str(base), extra_read_roots=[ro])

    def test_read_root_does_not_grant_write(self, tmp_path):
        """A path in extra_read_roots resolves for reads but not for write-only calls."""
        base = tmp_path / "project"
        base.mkdir()
        ro = tmp_path / "readonly"
        ro.mkdir()
        target = ro / "file.txt"
        target.write_text("data", encoding="utf-8")

        # Resolves when read roots are provided
        assert (
            safe_resolve(str(target), str(base), extra_read_roots=[ro])
            == target.resolve()
        )

        # Fails when only write roots are checked (simulating write path)
        with pytest.raises(ValueError, match="outside base directory"):
            safe_resolve(str(target), str(base))


# =========================================================================
# File operations with extra_read_roots (--add-dir-ro)
# =========================================================================


class TestFileOpsExtraReadRoots:
    def test_read_file_extra_read_root(self, tmp_path):
        """Reading a file inside a read-only dir succeeds."""
        base = tmp_path / "project"
        base.mkdir()
        ro = tmp_path / "readonly"
        ro.mkdir()
        target = ro / "data.txt"
        target.write_text("line1\nline2\n", encoding="utf-8")

        result = _read_file(str(target), str(base), extra_read_roots=[ro])
        assert "1: line1" in result
        assert "2: line2" in result

    def test_write_file_read_root_rejected(self, tmp_path):
        """Writing to a path only in extra_read_roots is rejected."""
        base = tmp_path / "project"
        base.mkdir()
        ro = tmp_path / "readonly"
        ro.mkdir()
        target = ro / "output.txt"

        result = _write_file(str(target), "hello", str(base))
        assert result.startswith("error:")

    def test_edit_file_read_root_rejected(self, tmp_path):
        """Editing a file only in extra_read_roots is rejected."""
        base = tmp_path / "project"
        base.mkdir()
        ro = tmp_path / "readonly"
        ro.mkdir()
        target = ro / "config.txt"
        target.write_text("old value", encoding="utf-8")

        result = _edit_file(str(target), "old value", "new value", str(base))
        assert result.startswith("error:")
        # File content must be unchanged
        assert target.read_text(encoding="utf-8") == "old value"

    def test_delete_file_read_root_rejected(self, tmp_path):
        """Deleting a file only in extra_read_roots is rejected."""
        base = tmp_path / "project"
        base.mkdir()
        ro = tmp_path / "readonly"
        ro.mkdir()
        target = ro / "keep.txt"
        target.write_text("important", encoding="utf-8")

        result = _delete_file(str(target), str(base))
        assert result.startswith("error:")
        # File must still exist
        assert target.exists()


# =========================================================================
# list_files / grep with extra_read_roots (--add-dir-ro)
# =========================================================================


class TestListGrepExtraReadRoots:
    def test_list_files_extra_read_root(self, tmp_path):
        """Listing a read-only dir returns results."""
        base = tmp_path / "project"
        base.mkdir()
        ro = tmp_path / "readonly"
        ro.mkdir()
        (ro / "a.py").write_text("# a", encoding="utf-8")
        (ro / "b.py").write_text("# b", encoding="utf-8")

        result = _list_files("**/*.py", str(ro), str(base), extra_read_roots=[ro])
        assert "a.py" in result
        assert "b.py" in result

    def test_grep_extra_read_root(self, tmp_path):
        """Grepping in a read-only dir returns matches."""
        base = tmp_path / "project"
        base.mkdir()
        ro = tmp_path / "readonly"
        ro.mkdir()
        (ro / "code.py").write_text("def hello():\n    pass\n", encoding="utf-8")

        result = _grep("hello", str(ro), str(base), extra_read_roots=[ro])
        assert "hello" in result
        assert "No matches" not in result


# =========================================================================
# CLI validation for --add-dir-ro
# =========================================================================


class TestAddDirRoCLIValidation:
    """Integration tests that call agent.main() to verify --add-dir-ro validation."""

    def test_add_dir_ro_nonexistent(self, tmp_path, monkeypatch):
        """Passing a non-existent path exits with sys.exit(1)."""
        from swival import agent

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "--base-dir",
                str(tmp_path),
                "--add-dir-ro",
                str(tmp_path / "nope"),
                "question",
            ],
        )
        with pytest.raises(SystemExit) as exc_info:
            agent.main()
        assert exc_info.value.code == 1

    def test_add_dir_ro_is_file(self, tmp_path, monkeypatch):
        """Passing a file (not directory) exits with sys.exit(1)."""
        f = tmp_path / "afile.txt"
        f.write_text("hi", encoding="utf-8")

        from swival import agent

        monkeypatch.setattr(
            sys,
            "argv",
            ["agent", "--base-dir", str(tmp_path), "--add-dir-ro", str(f), "question"],
        )
        with pytest.raises(SystemExit) as exc_info:
            agent.main()
        assert exc_info.value.code == 1

    def test_add_dir_ro_root_rejected(self, tmp_path, monkeypatch):
        """Passing / (filesystem root) exits with sys.exit(1)."""
        from swival import agent

        monkeypatch.setattr(
            sys,
            "argv",
            ["agent", "--base-dir", str(tmp_path), "--add-dir-ro", "/", "question"],
        )
        with pytest.raises(SystemExit) as exc_info:
            agent.main()
        assert exc_info.value.code == 1

    def test_add_dir_ro_tilde_expansion(self, tmp_path, monkeypatch):
        """~/somedir resolves correctly via expanduser() in main()."""
        target = tmp_path / "homedir"
        target.mkdir()
        monkeypatch.setenv("HOME", str(tmp_path))

        from swival import agent

        captured = {}

        def fake_run(messages, tools, **kwargs):
            captured["skill_read_roots"] = kwargs.get("skill_read_roots", [])
            return "done", False

        monkeypatch.setattr(agent, "run_agent_loop", fake_run)
        monkeypatch.setattr(
            agent, "discover_model", lambda *a, **kw: ("test-model", 4096)
        )

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "--base-dir",
                str(tmp_path),
                "--add-dir-ro",
                "~/homedir",
                "question",
            ],
        )
        agent.main()

        assert target.resolve() in captured["skill_read_roots"]

    def test_add_dir_ro_with_yolo_valid(self, tmp_path, monkeypatch):
        """--yolo --add-dir-ro <valid> validates at startup but doesn't error."""
        ro = tmp_path / "readonly"
        ro.mkdir()

        from swival import agent

        captured = {}

        def fake_run(messages, tools, **kwargs):
            captured["skill_read_roots"] = kwargs.get("skill_read_roots", [])
            captured["yolo"] = kwargs.get("yolo", False)
            return "done", False

        monkeypatch.setattr(agent, "run_agent_loop", fake_run)
        monkeypatch.setattr(
            agent, "discover_model", lambda *a, **kw: ("test-model", 4096)
        )

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "--base-dir",
                str(tmp_path),
                "--yolo",
                "--add-dir-ro",
                str(ro),
                "question",
            ],
        )
        agent.main()

        assert captured["yolo"] is True
        assert ro.resolve() in captured["skill_read_roots"]

    def test_add_dir_ro_with_yolo_invalid(self, tmp_path, monkeypatch):
        """--yolo --add-dir-ro <nonexistent> still exits with sys.exit(1)."""
        from swival import agent

        monkeypatch.setattr(
            agent, "discover_model", lambda *a, **kw: ("test-model", 4096)
        )

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "--base-dir",
                str(tmp_path),
                "--yolo",
                "--add-dir-ro",
                str(tmp_path / "nope"),
                "question",
            ],
        )
        with pytest.raises(SystemExit) as exc_info:
            agent.main()
        assert exc_info.value.code == 1

    def test_add_dir_and_add_dir_ro_coexist(self, tmp_path, monkeypatch):
        """Both --add-dir and --add-dir-ro work together: RW dir writable, RO dir read-only."""
        rw_dir = tmp_path / "writable"
        rw_dir.mkdir()
        ro_dir = tmp_path / "readonly"
        ro_dir.mkdir()

        from swival import agent

        captured = {}

        def fake_run(messages, tools, **kwargs):
            captured["extra_write_roots"] = kwargs.get("extra_write_roots", [])
            captured["skill_read_roots"] = kwargs.get("skill_read_roots", [])
            return "done", False

        monkeypatch.setattr(agent, "run_agent_loop", fake_run)
        monkeypatch.setattr(
            agent, "discover_model", lambda *a, **kw: ("test-model", 4096)
        )

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "--base-dir",
                str(tmp_path),
                "--add-dir",
                str(rw_dir),
                "--add-dir-ro",
                str(ro_dir),
                "question",
            ],
        )
        agent.main()

        assert rw_dir.resolve() in captured["extra_write_roots"]
        assert ro_dir.resolve() in captured["skill_read_roots"]
        # RO dir should NOT be in write roots
        assert ro_dir.resolve() not in captured["extra_write_roots"]
        # RW dir should NOT be in read-only roots
        assert rw_dir.resolve() not in captured["skill_read_roots"]


# =========================================================================
# dispatch integration with extra_read_roots (--add-dir-ro)
# =========================================================================


class TestDispatchExtraReadRoots:
    def test_dispatch_read_file_extra_read_roots(self, tmp_path):
        """dispatch() correctly forwards skill_read_roots for read_file."""
        base = tmp_path / "project"
        base.mkdir()
        ro = tmp_path / "readonly"
        ro.mkdir()
        target = ro / "file.txt"
        target.write_text("content\n", encoding="utf-8")

        result = dispatch(
            "read_file",
            {"file_path": str(target)},
            str(base),
            skill_read_roots=[ro],
        )
        assert "1: content" in result

    def test_dispatch_write_file_read_root_rejected(self, tmp_path):
        """dispatch() rejects writing to a path only in skill_read_roots."""
        base = tmp_path / "project"
        base.mkdir()
        ro = tmp_path / "readonly"
        ro.mkdir()
        target = ro / "new.txt"

        result = dispatch(
            "write_file",
            {"file_path": str(target), "content": "hello"},
            str(base),
            skill_read_roots=[ro],
        )
        assert result.startswith("error:")
        assert not target.exists()

    def test_dispatch_edit_file_read_root_rejected(self, tmp_path):
        """dispatch() rejects editing a file only in skill_read_roots."""
        base = tmp_path / "project"
        base.mkdir()
        ro = tmp_path / "readonly"
        ro.mkdir()
        target = ro / "edit.txt"
        target.write_text("old text", encoding="utf-8")

        result = dispatch(
            "edit_file",
            {
                "file_path": str(target),
                "old_string": "old text",
                "new_string": "new text",
            },
            str(base),
            skill_read_roots=[ro],
        )
        assert result.startswith("error:")
        assert target.read_text(encoding="utf-8") == "old text"

    def test_dispatch_list_files_extra_read_roots(self, tmp_path):
        """dispatch() correctly forwards skill_read_roots for list_files."""
        base = tmp_path / "project"
        base.mkdir()
        ro = tmp_path / "readonly"
        ro.mkdir()
        (ro / "test.py").write_text("# test", encoding="utf-8")

        result = dispatch(
            "list_files",
            {"pattern": "**/*.py", "path": str(ro)},
            str(base),
            skill_read_roots=[ro],
        )
        assert "test.py" in result

    def test_dispatch_grep_extra_read_roots(self, tmp_path):
        """dispatch() correctly forwards skill_read_roots for grep."""
        base = tmp_path / "project"
        base.mkdir()
        ro = tmp_path / "readonly"
        ro.mkdir()
        (ro / "search.py").write_text("def find_me():\n    pass\n", encoding="utf-8")

        result = dispatch(
            "grep",
            {"pattern": "find_me", "path": str(ro)},
            str(base),
            skill_read_roots=[ro],
        )
        assert "find_me" in result
        assert "No matches" not in result

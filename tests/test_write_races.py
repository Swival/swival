"""Tests for .swival/ write-race fixes (issue #7).

Validates per-context scratch directory isolation, shared-state locking,
and graceful handling of vanished temp files.
"""

import os
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from swival.tools import (
    _delete_file,
    _read_file,
    _save_large_output,
    _cleanup_trash,
    SWIVAL_DIR,
)


# ---------------------------------------------------------------------------
# 1. Multi-context cmd_output isolation
# ---------------------------------------------------------------------------


class TestCmdOutputIsolation:
    def test_scratch_dir_used_when_set(self, tmp_path):
        """_save_large_output writes to scratch_dir, not base_dir/.swival/."""
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        output = "x" * 50_000

        result = _save_large_output(output, str(tmp_path), scratch_dir=str(scratch))

        # File should be in scratch, not in .swival/
        assert list(scratch.glob("cmd_output_*.txt"))
        assert not list((tmp_path / SWIVAL_DIR).glob("cmd_output_*.txt"))
        assert "cmd_output_" in result

    def test_returned_path_points_to_scratch_dir(self, tmp_path):
        """The path returned by _save_large_output is resolvable via read_file."""
        scratch = tmp_path / ".swival" / "contexts" / "abc123"
        scratch.mkdir(parents=True)
        output = "payload " * 10_000

        result = _save_large_output(output, str(tmp_path), scratch_dir=str(scratch))

        # Extract the path from the result message
        for line in result.splitlines():
            if "saved to:" in line.lower() or "saved to" in line.lower():
                rel_path = line.split(":")[-1].strip()
                break
        else:
            pytest.fail(f"No 'saved to' path in result: {result}")

        # The path should resolve to the actual file
        full_path = tmp_path / rel_path
        assert full_path.exists(), (
            f"File not found at {full_path} (rel_path={rel_path})"
        )
        assert full_path.read_text().startswith("payload ")

    def test_default_without_scratch_dir(self, tmp_path):
        """Without scratch_dir, falls back to base_dir/.swival/."""
        output = "x" * 50_000
        _save_large_output(output, str(tmp_path))

        assert list((tmp_path / SWIVAL_DIR).glob("cmd_output_*.txt"))

    def test_two_contexts_write_separately(self, tmp_path):
        """Two contexts with different scratch dirs don't see each other's files."""
        scratch_a = tmp_path / "ctx_a"
        scratch_b = tmp_path / "ctx_b"
        scratch_a.mkdir()
        scratch_b.mkdir()
        output = "y" * 50_000

        _save_large_output(output, str(tmp_path), scratch_dir=str(scratch_a))
        _save_large_output(output, str(tmp_path), scratch_dir=str(scratch_b))

        files_a = list(scratch_a.glob("cmd_output_*.txt"))
        files_b = list(scratch_b.glob("cmd_output_*.txt"))
        assert len(files_a) == 1
        assert len(files_b) == 1
        assert files_a[0].name != files_b[0].name


# ---------------------------------------------------------------------------
# 2. Vanished cmd_output read
# ---------------------------------------------------------------------------


class TestVanishedCmdOutput:
    def test_read_deleted_file_returns_error(self, tmp_path):
        """Reading a file that vanishes between existence check and open."""
        target = tmp_path / "cmd_output_abc123.txt"
        target.write_text("hello")

        # Patch open to simulate the file vanishing after the existence check
        original_open = open

        def vanishing_open(path, *args, **kwargs):
            if "cmd_output_abc123" in str(path):
                target.unlink(missing_ok=True)
                raise FileNotFoundError(f"No such file: {path}")
            return original_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=vanishing_open):
            result = _read_file(str(target), str(tmp_path), offset=1, limit=100)

        assert result.startswith("error:")

    def test_read_vanishes_after_binary_check(self, tmp_path):
        """File vanishes between the binary check and read_text()."""
        target = tmp_path / "cmd_output_late.txt"
        target.write_text("hello world")

        original_read_text = Path.read_text

        def vanishing_read_text(self_path, *args, **kwargs):
            if "cmd_output_late" in str(self_path):
                target.unlink(missing_ok=True)
                raise FileNotFoundError(f"No such file: {self_path}")
            return original_read_text(self_path, *args, **kwargs)

        with patch.object(Path, "read_text", vanishing_read_text):
            result = _read_file(str(target), str(tmp_path), offset=1, limit=100)

        assert result.startswith("error:")
        assert "removed after check" in result

    def test_read_nonexistent_returns_error(self, tmp_path):
        """Reading a file that doesn't exist returns a clean error."""
        result = _read_file(
            str(tmp_path / "cmd_output_gone.txt"),
            str(tmp_path),
            offset=1,
            limit=100,
        )
        assert result.startswith("error:")


# ---------------------------------------------------------------------------
# 3. Trash cleanup serialization
# ---------------------------------------------------------------------------


class TestTrashCleanupSerialization:
    def _populate_trash(self, tmp_path, count=10, file_size=1024):
        """Create trash entries for testing."""
        trash_root = tmp_path / SWIVAL_DIR / "trash"
        trash_root.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            entry_dir = trash_root / f"entry_{i:04d}"
            entry_dir.mkdir()
            (entry_dir / "file.txt").write_text("x" * file_size)
        return trash_root

    def test_concurrent_cleanup_no_errors(self, tmp_path):
        """Multiple threads running _cleanup_trash don't raise errors."""
        self._populate_trash(tmp_path, count=20, file_size=512)
        errors = []

        def run_cleanup():
            try:
                _cleanup_trash(str(tmp_path))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_cleanup) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test_concurrent_deletes_no_errors(self, tmp_path):
        """Multiple threads deleting files via _delete_file don't raise."""
        files = []
        for i in range(10):
            f = tmp_path / f"file_{i}.txt"
            f.write_text(f"content {i}")
            files.append(f)

        errors = []

        def delete_one(path):
            try:
                _delete_file(str(path), str(tmp_path))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=delete_one, args=(f,)) for f in files]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # All files should be trashed
        for f in files:
            assert not f.exists()


# ---------------------------------------------------------------------------
# 4. History append under contention
# ---------------------------------------------------------------------------


class TestHistoryContention:
    def test_concurrent_appends_no_corruption(self, tmp_path):
        """Concurrent history appends produce valid entries without corruption."""
        from swival.agent import append_history

        n_threads = 5
        n_writes = 10
        errors = []

        def write_entries(thread_id):
            try:
                for i in range(n_writes):
                    append_history(
                        str(tmp_path),
                        f"question-{thread_id}-{i}",
                        f"answer-{thread_id}-{i}",
                        diagnostics=False,
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=write_entries, args=(tid,))
            for tid in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

        history_path = tmp_path / SWIVAL_DIR / "HISTORY.md"
        assert history_path.exists()
        content = history_path.read_text()

        # Every entry should be present
        for tid in range(n_threads):
            for i in range(n_writes):
                assert f"answer-{tid}-{i}" in content

    def test_capacity_respected(self, tmp_path):
        """History stays bounded by MAX_HISTORY_SIZE: hitting the cap trims older entries."""
        from swival.agent import append_history, MAX_HISTORY_SIZE

        # Write until capacity
        for i in range(2000):
            append_history(
                str(tmp_path),
                f"q{i}",
                "a" * 500,
                diagnostics=False,
            )
            history_path = tmp_path / SWIVAL_DIR / "HISTORY.md"
            if (
                history_path.exists()
                and history_path.stat().st_size >= MAX_HISTORY_SIZE
            ):
                break

        # One more write triggers a trim; the new entry is appended to the trimmed file
        append_history(str(tmp_path), "extra", "fresh entry", diagnostics=False)
        size_after = history_path.stat().st_size
        content = history_path.read_text()

        assert size_after <= MAX_HISTORY_SIZE
        assert "fresh entry" in content


# ---------------------------------------------------------------------------
# 5. Context ID sanitization
# ---------------------------------------------------------------------------


class TestContextIdSanitization:
    def test_traversal_context_id_stays_inside_swival(self, tmp_path):
        """A malicious contextId cannot escape .swival/contexts/."""
        from swival.a2a_server import A2aServer

        session_kwargs = {"base_dir": str(tmp_path), "provider": "generic"}
        server = A2aServer(session_kwargs=session_kwargs, host="127.0.0.1", port=0)

        # Simulate creating a session with a path-traversal contextId
        malicious_id = "../../../../tmp/pwn"
        session = server._create_session(malicious_id)

        scratch = Path(session.scratch_dir)
        # scratch_dir must resolve inside base_dir/.swival/contexts/
        assert scratch.resolve().is_relative_to(
            (tmp_path / ".swival" / "contexts").resolve()
        ), f"scratch_dir escaped: {scratch}"

    def test_different_ids_get_different_dirs(self, tmp_path):
        """Different contextIds produce different scratch directories."""
        from swival.a2a_server import A2aServer

        session_kwargs = {"base_dir": str(tmp_path), "provider": "generic"}
        server = A2aServer(session_kwargs=session_kwargs, host="127.0.0.1", port=0)

        s1 = server._create_session("ctx-1")
        s2 = server._create_session("ctx-2")
        assert s1.scratch_dir != s2.scratch_dir


# ---------------------------------------------------------------------------
# 6. Lock fallback paths
# ---------------------------------------------------------------------------


class TestLockFallbacks:
    def test_trash_lock_mkdir_fails_still_executes_body(self, tmp_path):
        """_trash_lock falls back to no-op when mkdir raises OSError."""
        from swival.tools import _trash_lock

        executed = False
        with patch.object(Path, "mkdir", side_effect=OSError("permission denied")):
            with _trash_lock(str(tmp_path)):
                executed = True
        assert executed

    def test_trash_lock_no_fcntl_still_executes_body(self, tmp_path):
        """_trash_lock falls back to no-op when fcntl is unavailable."""
        import builtins

        from swival.tools import _trash_lock

        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "fcntl":
                raise ImportError("no fcntl on Windows")
            return original_import(name, *args, **kwargs)

        executed = False
        with patch.object(builtins, "__import__", side_effect=fake_import):
            with _trash_lock(str(tmp_path)):
                executed = True
        assert executed

    def test_trash_delete_works_when_lock_file_fails(self, tmp_path):
        """_delete_file succeeds even when the lock file can't be created.

        The lock gracefully degrades; the delete still moves the file to trash.
        """
        target = tmp_path / "doomed.txt"
        target.write_text("bye")

        import swival.tools as tools_mod

        original_os_open = os.open

        def fail_lock_open(path, *args, **kwargs):
            if ".lock" in str(path):
                raise OSError("cannot create lock file")
            return original_os_open(path, *args, **kwargs)

        with patch.object(tools_mod.os, "open", side_effect=fail_lock_open):
            result = _delete_file(str(target), str(tmp_path))

        assert result.startswith("Trashed")
        assert not target.exists()

    def test_append_history_no_fcntl(self, tmp_path):
        """append_history writes correctly when fcntl is unavailable."""
        import builtins

        from swival.agent import append_history

        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "fcntl":
                raise ImportError("no fcntl on Windows")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=fake_import):
            append_history(
                str(tmp_path),
                "test-question",
                "test-answer",
                diagnostics=False,
            )

        history_path = tmp_path / SWIVAL_DIR / "HISTORY.md"
        assert history_path.exists()
        content = history_path.read_text()
        assert "test-answer" in content
        assert "test-question" in content

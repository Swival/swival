"""Tests for .swival/ write-race fixes (issue #7).

Validates per-context scratch directory isolation, shared-state locking,
and graceful handling of vanished temp files.
"""

import os
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from swival.todo import TodoState
from swival.tools import (
    _delete_file,
    _read_file,
    _save_large_output,
    _cleanup_trash,
    SWIVAL_DIR,
)


# ---------------------------------------------------------------------------
# 1. Multi-context todo isolation
# ---------------------------------------------------------------------------


class TestTodoIsolation:
    def test_two_contexts_do_not_interfere(self, tmp_path):
        """Two TodoStates with different todo_dir write independent lists."""
        dir_a = tmp_path / "ctx_a"
        dir_b = tmp_path / "ctx_b"
        dir_a.mkdir()
        dir_b.mkdir()

        todo_a = TodoState(notes_dir=str(tmp_path), todo_dir=str(dir_a))
        todo_b = TodoState(notes_dir=str(tmp_path), todo_dir=str(dir_b))

        todo_a.process({"action": "add", "tasks": "task-A1"})
        todo_a.process({"action": "add", "tasks": "task-A2"})
        todo_b.process({"action": "add", "tasks": "task-B1"})

        assert len(todo_a.items) == 2
        assert len(todo_b.items) == 1

        # Verify files are separate
        file_a = dir_a / "todo.md"
        file_b = dir_b / "todo.md"
        assert file_a.exists()
        assert file_b.exists()
        assert "task-A1" in file_a.read_text()
        assert "task-B1" in file_b.read_text()
        assert "task-B1" not in file_a.read_text()

    def test_concurrent_adds_isolated(self, tmp_path):
        """Concurrent adds to separate todo_dirs don't corrupt either list."""
        n_items = 20
        dir_a = tmp_path / "ctx_a"
        dir_b = tmp_path / "ctx_b"
        dir_a.mkdir()
        dir_b.mkdir()

        todo_a = TodoState(notes_dir=str(tmp_path), todo_dir=str(dir_a))
        todo_b = TodoState(notes_dir=str(tmp_path), todo_dir=str(dir_b))

        errors = []

        def add_items(todo, prefix, count):
            try:
                for i in range(count):
                    todo.process({"action": "add", "tasks": f"{prefix}-{i}"})
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=add_items, args=(todo_a, "A", n_items))
        t2 = threading.Thread(target=add_items, args=(todo_b, "B", n_items))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors
        assert len(todo_a.items) == n_items
        assert len(todo_b.items) == n_items
        assert all(item.text.startswith("A-") for item in todo_a.items)
        assert all(item.text.startswith("B-") for item in todo_b.items)

    def test_default_notes_dir_unchanged(self, tmp_path):
        """When todo_dir is None, TodoState uses notes_dir/.swival/todo.md."""
        todo = TodoState(notes_dir=str(tmp_path))
        todo.process({"action": "add", "tasks": "hello"})
        assert (tmp_path / ".swival" / "todo.md").exists()

    def test_reset_clears_todo_dir_file(self, tmp_path):
        """reset() empties todo.md in todo_dir, not just notes_dir/.swival/."""
        ctx_dir = tmp_path / "ctx"
        ctx_dir.mkdir()
        todo = TodoState(notes_dir=str(tmp_path), todo_dir=str(ctx_dir))
        todo.process({"action": "add", "tasks": "item1"})
        assert (ctx_dir / "todo.md").exists()

        todo.reset()
        assert (ctx_dir / "todo.md").read_text() == ""
        assert len(todo.items) == 0

    def test_concurrent_saves_no_corruption(self, tmp_path):
        """Concurrent _save() calls to the same todo.md don't produce garbled output."""
        n_threads = 5
        n_writes = 20
        todo_dir = tmp_path / "shared"
        todo_dir.mkdir()
        errors = []

        def write_items(thread_id):
            try:
                state = TodoState(notes_dir=str(tmp_path), todo_dir=str(todo_dir))
                for i in range(n_writes):
                    state.items.clear()
                    from swival.todo import TodoItem

                    state.items.append(TodoItem(text=f"t{thread_id}-{i}"))
                    state._save()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=write_items, args=(tid,))
            for tid in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

        content = (todo_dir / "todo.md").read_text()
        lines = [ln for ln in content.splitlines() if ln.strip()]
        for line in lines:
            assert line.startswith("- ["), f"corrupted line: {line!r}"

    def test_concurrent_adds_no_lost_updates(self, tmp_path):
        """Two sessions adding distinct items to the same file preserve both sides."""
        todo_dir = tmp_path / "shared"
        todo_dir.mkdir()

        state_a = TodoState(notes_dir=str(tmp_path), todo_dir=str(todo_dir))
        state_b = TodoState(notes_dir=str(tmp_path), todo_dir=str(todo_dir))

        errors = []

        def add_items(state, prefix, count):
            try:
                for i in range(count):
                    state.process({"action": "add", "tasks": f"{prefix}-{i}"})
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=add_items, args=(state_a, "A", 10))
        t2 = threading.Thread(target=add_items, args=(state_b, "B", 10))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors

        from swival.todo import _parse_todo_file

        on_disk = {item.text for item in _parse_todo_file(todo_dir / "todo.md")}
        for i in range(10):
            assert f"A-{i}" in on_disk, f"A-{i} lost"
            assert f"B-{i}" in on_disk, f"B-{i} lost"

    def test_concurrent_done_not_reverted(self, tmp_path):
        """If one session marks an item done, a concurrent save doesn't revert it."""
        todo_dir = tmp_path / "shared"
        todo_dir.mkdir()

        state_a = TodoState(notes_dir=str(tmp_path), todo_dir=str(todo_dir))
        state_b = TodoState(notes_dir=str(tmp_path), todo_dir=str(todo_dir))

        # Both sessions know about the same item.
        state_a.process({"action": "add", "tasks": "shared-task"})
        # B picks it up via a save-merge cycle.
        state_b.process({"action": "add", "tasks": "shared-task"})

        # A marks it done.
        state_a.process({"action": "done", "tasks": "shared-task"})

        # B adds something else — its save must not revert A's done.
        state_b.process({"action": "add", "tasks": "other-task"})

        from swival.todo import _parse_todo_file

        on_disk = {
            item.text: item.done for item in _parse_todo_file(todo_dir / "todo.md")
        }
        assert on_disk["shared-task"] is True, "done status reverted by concurrent save"

    def test_remove_not_resurrected_by_own_save(self, tmp_path):
        """After session A removes an item, A's own later saves don't resurrect it
        even if another session wrote the item back to disk in the meantime."""
        todo_dir = tmp_path / "shared"
        todo_dir.mkdir()

        state_a = TodoState(notes_dir=str(tmp_path), todo_dir=str(todo_dir))
        state_a.process({"action": "add", "tasks": "doomed"})
        state_a.process({"action": "remove", "tasks": "doomed"})

        # Simulate another session writing "doomed" back to disk.
        (todo_dir / "todo.md").write_text("- [ ] doomed\n")

        # A adds something else — merge must not pull "doomed" back in.
        state_a.process({"action": "add", "tasks": "survivor"})

        from swival.todo import _parse_todo_file

        on_disk = {item.text for item in _parse_todo_file(todo_dir / "todo.md")}
        assert "doomed" not in on_disk, "'doomed' resurrected by own merge"
        assert "survivor" in on_disk

    def test_init_baseline_not_absorbed(self, tmp_path):
        """__init__ preserves the file AND baseline items on disk, but does
        not absorb them into self.items (session isolation)."""
        todo_dir = tmp_path / "ctx"
        todo_dir.mkdir()
        (todo_dir / "todo.md").write_text("- [ ] stale\n")

        state = TodoState(notes_dir=str(tmp_path), todo_dir=str(todo_dir))

        # File is preserved.
        assert (todo_dir / "todo.md").exists()
        assert "stale" in (todo_dir / "todo.md").read_text()

        # Adding a new item keeps stale on disk but doesn't pull it into self.items.
        state.process({"action": "add", "tasks": "fresh"})
        from swival.todo import _parse_todo_file

        on_disk = {item.text for item in _parse_todo_file(todo_dir / "todo.md")}
        assert "fresh" in on_disk
        assert "stale" in on_disk  # preserved for concurrent sessions
        assert not any(i.text == "stale" for i in state.items)  # not absorbed

    def test_clear_removes_baseline_from_disk(self, tmp_path):
        """clear wipes baseline items from disk, not just self.items."""
        todo_dir = tmp_path / "ctx"
        todo_dir.mkdir()
        (todo_dir / "todo.md").write_text("- [ ] stale\n")

        state = TodoState(notes_dir=str(tmp_path), todo_dir=str(todo_dir))
        state.process({"action": "clear"})

        content = (todo_dir / "todo.md").read_text()
        assert content == ""

    def test_init_does_not_nuke_concurrent_session(self, tmp_path):
        """A second session starting AND saving doesn't destroy the first
        session's items from disk."""
        todo_dir = tmp_path / "shared"
        todo_dir.mkdir()

        state_a = TodoState(notes_dir=str(tmp_path), todo_dir=str(todo_dir))
        state_a.process({"action": "add", "tasks": "alive"})

        # Session B starts and immediately saves its own item.
        state_b = TodoState(notes_dir=str(tmp_path), todo_dir=str(todo_dir))
        state_b.process({"action": "add", "tasks": "b-item"})

        from swival.todo import _parse_todo_file

        on_disk = {item.text for item in _parse_todo_file(todo_dir / "todo.md")}
        assert "alive" in on_disk, "session B save destroyed session A's items"
        assert "b-item" in on_disk

        # A can still save and its item survives alongside B's.
        state_a.process({"action": "add", "tasks": "still-here"})
        on_disk = {item.text for item in _parse_todo_file(todo_dir / "todo.md")}
        assert "alive" in on_disk
        assert "still-here" in on_disk
        assert "b-item" in on_disk

    def test_reset_writes_empty_under_lock(self, tmp_path):
        """reset() empties the file under lock; other sessions can restore,
        and the resetting session's future saves don't re-delete them."""
        todo_dir = tmp_path / "shared"
        todo_dir.mkdir()

        state_a = TodoState(notes_dir=str(tmp_path), todo_dir=str(todo_dir))
        state_b = TodoState(notes_dir=str(tmp_path), todo_dir=str(todo_dir))

        state_a.process({"action": "add", "tasks": "a-item"})
        state_b.process({"action": "add", "tasks": "b-item"})

        # A resets — file goes empty.
        state_a.reset()
        content = (todo_dir / "todo.md").read_text()
        assert content == ""
        assert (todo_dir / "todo.md.lock").exists()

        # B saves again — its item is restored from B's in-memory state.
        state_b.process({"action": "add", "tasks": "b-extra"})
        from swival.todo import _parse_todo_file

        on_disk = {item.text for item in _parse_todo_file(todo_dir / "todo.md")}
        assert "b-item" in on_disk
        assert "b-extra" in on_disk

        # A saves again — must NOT re-delete B's restored items.
        state_a.process({"action": "add", "tasks": "a-post-reset"})
        on_disk = {item.text for item in _parse_todo_file(todo_dir / "todo.md")}
        assert "b-item" in on_disk, "reset session re-deleted peer's restored items"
        assert "b-extra" in on_disk, "reset session re-deleted peer's restored items"
        assert "a-post-reset" in on_disk

    def test_merge_preserves_all_successful_adds(self, tmp_path):
        """Both sessions' successful adds survive the merge — no silent discard."""
        import json

        todo_dir = tmp_path / "shared"
        todo_dir.mkdir()

        state_a = TodoState(notes_dir=str(tmp_path), todo_dir=str(todo_dir))
        state_b = TodoState(notes_dir=str(tmp_path), todo_dir=str(todo_dir))

        # Both sessions fill to 49 with the same items (dedup'd).
        for i in range(49):
            state_a.process({"action": "add", "tasks": f"shared-{i}"})
            state_b.process({"action": "add", "tasks": f"shared-{i}"})

        # Each adds a different 50th item — both return success.
        result_a = json.loads(state_a.process({"action": "add", "tasks": "a-50th"}))
        result_b = json.loads(state_b.process({"action": "add", "tasks": "b-50th"}))
        assert result_a["action"] == "add"
        assert result_b["action"] == "add"

        from swival.todo import _parse_todo_file

        on_disk = {item.text for item in _parse_todo_file(todo_dir / "todo.md")}
        assert "a-50th" in on_disk, "a-50th silently discarded by merge"
        assert "b-50th" in on_disk, "b-50th silently discarded by merge"


# ---------------------------------------------------------------------------
# 2. Multi-context cmd_output isolation
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
# 3. Vanished cmd_output read
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
# 4. Trash cleanup serialization
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
# 5. History append under contention
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
        """History stops growing beyond MAX_HISTORY_SIZE."""
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

        # One more write should be skipped
        append_history(str(tmp_path), "extra", "should not appear", diagnostics=False)
        size_after = history_path.stat().st_size

        # Allow tolerance of one entry (the lock makes the check atomic,
        # but the entry that pushed us over the cap is already written)
        assert size_after <= MAX_HISTORY_SIZE + 1024


# ---------------------------------------------------------------------------
# 6. Context ID sanitization
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
# 7. Lock fallback paths
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

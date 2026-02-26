"""Tests for delete_file (soft-delete to .swival/trash/)."""

import json
import os
import shutil
import sys
import time
from pathlib import Path
from unittest.mock import patch


from swival.tools import (
    _delete_file,
    _cleanup_trash,
    _read_file,
    _write_file,
    dispatch,
    SWIVAL_DIR,
    TRASH_MAX_AGE,
    TRASH_MAX_BYTES,
)
from swival.tracker import FileAccessTracker


# =========================================================================
# Helpers
# =========================================================================


def _trash_root(base: Path) -> Path:
    return base / SWIVAL_DIR / "trash"


def _read_index(base: Path) -> list[dict]:
    idx = _trash_root(base) / "index.jsonl"
    if not idx.exists():
        return []
    lines = idx.read_text().strip().splitlines()
    return [json.loads(line) for line in lines]


def _make_old_trash_entry(
    base: Path, trash_id: str, filename: str, content: bytes, age_seconds: float
) -> Path:
    """Create a trash entry with a specific age."""
    d = _trash_root(base) / trash_id
    d.mkdir(parents=True)
    f = d / filename
    f.write_bytes(content)
    old_time = time.time() - age_seconds
    os.utime(str(d), (old_time, old_time))
    return d


# =========================================================================
# Basic delete_file tests
# =========================================================================


class TestDeleteFile:
    def test_delete_moves_to_trash(self, tmp_path):
        (tmp_path / "foo.txt").write_text("hello")
        _delete_file("foo.txt", str(tmp_path))
        assert not (tmp_path / "foo.txt").exists()
        # File should be in trash.
        trash = _trash_root(tmp_path)
        dirs = [d for d in trash.iterdir() if d.is_dir()]
        assert len(dirs) == 1
        trashed = dirs[0] / "foo.txt"
        assert trashed.read_text() == "hello"

    def test_delete_returns_trash_id(self, tmp_path):
        (tmp_path / "foo.txt").write_text("data")
        result = _delete_file("foo.txt", str(tmp_path))
        assert result.startswith("Trashed foo.txt -> .swival/trash/")
        trash_id = result.split("/")[-1]
        assert len(trash_id) == 32  # uuid4 hex

    def test_delete_index_entry(self, tmp_path):
        (tmp_path / "foo.txt").write_text("data")
        result = _delete_file("foo.txt", str(tmp_path), tool_call_id="call_abc")
        entries = _read_index(tmp_path)
        assert len(entries) == 1
        e = entries[0]
        assert e["original_path"] == "foo.txt"
        assert e["tool_call_id"] == "call_abc"
        assert "timestamp" in e
        assert e["trash_id"] in result

    def test_delete_two_same_name(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        (tmp_path / "a" / "f.txt").write_text("aaa")
        (tmp_path / "b" / "f.txt").write_text("bbb")
        r1 = _delete_file("a/f.txt", str(tmp_path))
        r2 = _delete_file("b/f.txt", str(tmp_path))
        assert r1 != r2
        entries = _read_index(tmp_path)
        assert len(entries) == 2
        ids = {e["trash_id"] for e in entries}
        assert len(ids) == 2

    def test_delete_nonexistent(self, tmp_path):
        result = _delete_file("nope.txt", str(tmp_path))
        assert result == "error: file not found: nope.txt"

    def test_delete_directory_rejected(self, tmp_path):
        (tmp_path / "subdir").mkdir()
        result = _delete_file("subdir", str(tmp_path))
        assert "error:" in result
        assert "directory" in result
        assert (tmp_path / "subdir").is_dir()

    def test_delete_outside_sandbox(self, tmp_path):
        result = _delete_file("../escape.txt", str(tmp_path))
        assert result.startswith("error:")


# =========================================================================
# Read guard interaction
# =========================================================================


class TestDeleteReadGuard:
    def test_delete_blocked_without_read(self, tmp_path):
        (tmp_path / "f.txt").write_text("content")
        tracker = FileAccessTracker()
        result = _delete_file("f.txt", str(tmp_path), tracker=tracker)
        assert result.startswith("error:")
        assert "hasn't been read" in result
        assert (tmp_path / "f.txt").exists()

    def test_delete_allowed_after_read(self, tmp_path):
        (tmp_path / "f.txt").write_text("content")
        tracker = FileAccessTracker()
        _read_file("f.txt", str(tmp_path), tracker=tracker)
        result = _delete_file("f.txt", str(tmp_path), tracker=tracker)
        assert result.startswith("Trashed")
        assert not (tmp_path / "f.txt").exists()

    def test_delete_own_creation(self, tmp_path):
        tracker = FileAccessTracker()
        _write_file("new.txt", "hello", str(tmp_path), tracker=tracker)
        result = _delete_file("new.txt", str(tmp_path), tracker=tracker)
        assert result.startswith("Trashed")

    def test_delete_no_tracker(self, tmp_path):
        (tmp_path / "f.txt").write_text("content")
        result = _delete_file("f.txt", str(tmp_path), tracker=None)
        assert result.startswith("Trashed")

    def test_delete_then_recreate(self, tmp_path):
        """After deletion, the path is in written_files so recreating is allowed."""
        (tmp_path / "f.txt").write_text("v1")
        tracker = FileAccessTracker()
        _read_file("f.txt", str(tmp_path), tracker=tracker)
        _delete_file("f.txt", str(tmp_path), tracker=tracker)
        result = _write_file("f.txt", "v2", str(tmp_path), tracker=tracker)
        assert result.startswith("Wrote")


# =========================================================================
# Extra write roots and yolo mode
# =========================================================================


class TestDeleteRoots:
    def test_delete_extra_write_roots(self, tmp_path):
        allowed = tmp_path / "external"
        allowed.mkdir()
        (allowed / "data.csv").write_text("a,b,c")
        result = _delete_file(
            str(allowed / "data.csv"),
            str(tmp_path),
            extra_write_roots=[allowed],
        )
        assert result.startswith("Trashed")

    def test_delete_yolo(self, tmp_path):
        (tmp_path / "f.txt").write_text("data")
        result = _delete_file("f.txt", str(tmp_path), unrestricted=True)
        assert result.startswith("Trashed")

    def test_delete_add_dir_absolute_path(self, tmp_path):
        """File in --add-dir records absolute original_path in index."""
        allowed = tmp_path / "ext"
        allowed.mkdir()
        (allowed / "dump.csv").write_text("x")
        abs_path = str(allowed / "dump.csv")
        _delete_file(
            abs_path,
            str(tmp_path),
            extra_write_roots=[allowed],
        )
        entries = _read_index(tmp_path)
        assert entries[0]["original_path"] == abs_path


# =========================================================================
# Symlink handling
# =========================================================================


class TestDeleteSymlinks:
    def test_delete_symlink_moves_link(self, tmp_path):
        target = tmp_path / "target.txt"
        target.write_text("real content")
        link = tmp_path / "link.txt"
        link.symlink_to(target)
        result = _delete_file("link.txt", str(tmp_path))
        assert result.startswith("Trashed")
        assert not link.exists() and not link.is_symlink()
        assert target.read_text() == "real content"

    def test_delete_symlink_to_dir(self, tmp_path):
        target_dir = tmp_path / "mydir"
        target_dir.mkdir()
        (target_dir / "child.txt").write_text("c")
        link = tmp_path / "dirlink"
        link.symlink_to(target_dir)
        result = _delete_file("dirlink", str(tmp_path))
        assert result.startswith("Trashed")
        assert not link.is_symlink()
        assert target_dir.is_dir()
        assert (target_dir / "child.txt").read_text() == "c"

    def test_delete_symlink_to_dir_skips_guard(self, tmp_path):
        """Symlink-to-dir deletion succeeds with tracker even without prior read."""
        target_dir = tmp_path / "mydir"
        target_dir.mkdir()
        link = tmp_path / "dirlink"
        link.symlink_to(target_dir)
        tracker = FileAccessTracker()
        result = _delete_file("dirlink", str(tmp_path), tracker=tracker)
        assert result.startswith("Trashed")

    def test_delete_dangling_symlink(self, tmp_path):
        target = tmp_path / "gone.txt"
        target.write_text("temp")
        link = tmp_path / "dangle.txt"
        link.symlink_to(target)
        target.unlink()
        assert link.is_symlink()
        assert not link.exists()  # dangling
        result = _delete_file("dangle.txt", str(tmp_path))
        assert result.startswith("Trashed")
        assert not link.is_symlink()

    def test_delete_symlink_outside_sandbox(self, tmp_path):
        """Symlink inside repo pointing outside sandbox is rejected by safe_resolve."""
        link = tmp_path / "escape.txt"
        link.symlink_to("/etc/passwd")
        result = _delete_file("escape.txt", str(tmp_path))
        assert result.startswith("error:")

    def test_delete_symlink_does_not_authorize_target_write(self, tmp_path):
        """Deleting a symlink must not allow writing to the target without a read."""
        target = tmp_path / "target.txt"
        target.write_text("original")
        link = tmp_path / "lnk.txt"
        link.symlink_to(target)
        tracker = FileAccessTracker()
        result = _delete_file("lnk.txt", str(tmp_path), tracker=tracker)
        assert result.startswith("Trashed")
        # Now try to overwrite the target — should be blocked.
        result = _write_file("target.txt", "pwned", str(tmp_path), tracker=tracker)
        assert result.startswith("error:")
        assert "hasn't been read" in result
        assert target.read_text() == "original"


# =========================================================================
# Index append failure
# =========================================================================


class TestDeleteIndexFailure:
    def test_index_append_failure(self, tmp_path):
        """Move succeeds but index write fails -> tool returns success."""
        (tmp_path / "f.txt").write_text("data")
        with patch("swival.tools.os.open", side_effect=OSError("disk full")):
            result = _delete_file("f.txt", str(tmp_path))
        assert result.startswith("Trashed")
        assert not (tmp_path / "f.txt").exists()
        # Index should be empty since the write failed.
        assert _read_index(tmp_path) == []


# =========================================================================
# Dispatch routing
# =========================================================================


class TestDeleteDispatch:
    def test_dispatch_routes_delete(self, tmp_path):
        (tmp_path / "f.txt").write_text("data")
        result = dispatch(
            "delete_file",
            {"file_path": "f.txt"},
            str(tmp_path),
            tool_call_id="call_xyz",
        )
        assert result.startswith("Trashed")
        entries = _read_index(tmp_path)
        assert entries[0]["tool_call_id"] == "call_xyz"

    def test_dispatch_passes_tracker(self, tmp_path):
        (tmp_path / "f.txt").write_text("data")
        tracker = FileAccessTracker()
        result = dispatch(
            "delete_file",
            {"file_path": "f.txt"},
            str(tmp_path),
            file_tracker=tracker,
        )
        assert result.startswith("error:")


# =========================================================================
# Retention / cleanup
# =========================================================================


class TestCleanupTrash:
    def test_cleanup_removes_old_entries(self, tmp_path):
        _make_old_trash_entry(tmp_path, "old1", "a.txt", b"aaa", TRASH_MAX_AGE + 100)
        _make_old_trash_entry(tmp_path, "fresh", "b.txt", b"bbb", 10)
        _cleanup_trash(str(tmp_path))
        trash = _trash_root(tmp_path)
        assert not (trash / "old1").exists()
        assert (trash / "fresh").exists()

    def test_cleanup_respects_size_cap(self, tmp_path):
        chunk = b"x" * (TRASH_MAX_BYTES // 2 + 1)
        _make_old_trash_entry(tmp_path, "older", "a.bin", chunk, 200)
        _make_old_trash_entry(tmp_path, "newer", "b.bin", chunk, 100)
        _cleanup_trash(str(tmp_path))
        trash = _trash_root(tmp_path)
        # older should be evicted to get under cap, newer kept.
        assert not (trash / "older").exists()
        assert (trash / "newer").exists()

    def test_cleanup_excluded_counts_in_budget(self, tmp_path):
        """30MB excluded + 30MB old -> old evicted (60MB > 50MB cap)."""
        big = b"x" * (30 * 1024 * 1024)
        _make_old_trash_entry(tmp_path, "excl", "a.bin", big, 50)
        _make_old_trash_entry(tmp_path, "old1", "b.bin", big, 200)
        _cleanup_trash(str(tmp_path), exclude="excl")
        trash = _trash_root(tmp_path)
        assert (trash / "excl").exists()
        assert not (trash / "old1").exists()

    def test_cleanup_preserves_just_trashed(self, tmp_path):
        chunk = b"x" * (TRASH_MAX_BYTES // 2 + 1)
        _make_old_trash_entry(tmp_path, "older", "a.bin", chunk, 200)
        _make_old_trash_entry(tmp_path, "just_added", "b.bin", chunk, 1)
        _cleanup_trash(str(tmp_path), exclude="just_added")
        trash = _trash_root(tmp_path)
        assert not (trash / "older").exists()
        assert (trash / "just_added").exists()

    def test_cleanup_large_single_file(self, tmp_path):
        """A single file larger than the cap evicts everything else but survives."""
        huge = b"x" * (TRASH_MAX_BYTES + 1024)
        small = b"y" * 100
        _make_old_trash_entry(tmp_path, "huge1", "big.bin", huge, 10)
        _make_old_trash_entry(tmp_path, "small1", "s.txt", small, 200)
        _cleanup_trash(str(tmp_path), exclude="huge1")
        trash = _trash_root(tmp_path)
        assert (trash / "huge1").exists()
        assert not (trash / "small1").exists()

    def test_cleanup_race_entry_already_gone(self, tmp_path):
        """Cleanup handles a concurrently removed entry without raising."""
        _make_old_trash_entry(tmp_path, "vanish", "a.txt", b"data", TRASH_MAX_AGE + 100)
        trash = _trash_root(tmp_path)
        # Remove it before cleanup runs (simulate race).
        shutil.rmtree(trash / "vanish")
        # Should not raise.
        _cleanup_trash(str(tmp_path))

    def test_cleanup_empty_trash(self, tmp_path):
        """Cleanup on non-existent trash dir is a no-op."""
        _cleanup_trash(str(tmp_path))  # should not raise

    def test_dir_size_does_not_follow_symlinks(self, tmp_path):
        """_dir_size should use lstat so symlinks aren't charged as target size."""
        from swival.tools import _dir_size

        # Create a large file outside the measured directory.
        big_file = tmp_path / "big.bin"
        big_file.write_bytes(b"x" * 1_000_000)

        # Create a trash-like directory with just a symlink to it.
        d = tmp_path / "entry"
        d.mkdir()
        (d / "link").symlink_to(big_file)

        size = _dir_size(d)
        # Should be the lstat size of the symlink (small), not the 1MB target.
        assert size < 1000

    def test_cleanup_stat_race(self, tmp_path):
        """Entry vanishes between iterdir and stat — handled gracefully."""
        d = _trash_root(tmp_path) / "vanish2"
        d.mkdir(parents=True)
        (d / "f.txt").write_bytes(b"data")

        original_stat = Path.stat

        def flaky_stat(self, *a, **kw):
            if self.name == "vanish2":
                raise FileNotFoundError("gone")
            return original_stat(self, *a, **kw)

        with patch.object(Path, "stat", flaky_stat):
            _cleanup_trash(str(tmp_path))  # should not raise


# =========================================================================
# Agent integration: think nudge
# =========================================================================


class TestThinkNudge:
    def test_think_nudge_fires_on_delete_without_think(self, tmp_path, monkeypatch):
        """Nudge fires when delete_file is used without prior think call."""
        import types
        from swival import agent
        from swival import fmt

        fmt.init(no_color=True)

        snapshots = []
        call_count = 0

        (tmp_path / "doomed.txt").write_text("bye\n")

        def _make_msg(content=None, tool_calls=None):
            m = types.SimpleNamespace()
            m.content = content
            m.tool_calls = tool_calls
            m.role = "assistant"
            m.get = lambda key, default=None: getattr(m, key, default)
            return m

        def _make_tc(name, arguments, call_id):
            tc = types.SimpleNamespace()
            tc.id = call_id
            tc.function = types.SimpleNamespace()
            tc.function.name = name
            tc.function.arguments = arguments
            return tc

        def fake_call_llm(*args, **kwargs):
            nonlocal call_count
            snapshots.append(list(args[2]))
            call_count += 1
            if call_count == 1:
                tc = _make_tc(
                    "delete_file",
                    json.dumps({"file_path": "doomed.txt"}),
                    call_id="call_del",
                )
                return _make_msg(tool_calls=[tc]), "tool_calls"
            return _make_msg(content="done"), "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

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
            question="test nudge",
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
            no_read_guard=True,
            no_history=True,
            init_config=False,
            project=False,
        )
        args = types.SimpleNamespace(**defaults)
        monkeypatch.setattr(sys, "argv", ["agent", "test nudge"])
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)

        agent.main()

        # The second LLM call should see the nudge.
        assert len(snapshots) == 2
        tips = []
        for msg in snapshots[1]:
            role = (
                msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            )
            if role != "user":
                continue
            content = (
                msg.get("content")
                if isinstance(msg, dict)
                else getattr(msg, "content", "")
            )
            if content and content.startswith("Tip:"):
                tips.append(content)
        assert len(tips) == 1
        assert "think" in tips[0].lower()


# =========================================================================
# Agent integration: tool_call_id forwarding
# =========================================================================


class TestToolCallIdForwarding:
    def test_tool_call_id_reaches_delete(self, tmp_path):
        """tool_call_id kwarg reaches _delete_file via dispatch."""
        (tmp_path / "f.txt").write_text("data")
        result = dispatch(
            "delete_file",
            {"file_path": "f.txt"},
            str(tmp_path),
            tool_call_id="call_123",
        )
        assert result.startswith("Trashed")
        entries = _read_index(tmp_path)
        assert entries[0]["tool_call_id"] == "call_123"

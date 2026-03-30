"""Tests for the --files option (none/some/all)."""

from pathlib import Path

import pytest

from swival.tools import (
    safe_resolve,
    _is_within_base,
    _read_file,
    _write_file,
    _edit_file,
    _delete_file,
    _list_files,
    _grep,
    dispatch,
)


class TestSafeResolve:
    def test_none_blocks_project_file(self, tmp_path):
        (tmp_path / "foo.txt").write_text("hello")
        with pytest.raises(ValueError, match="outside .swival/"):
            safe_resolve("foo.txt", str(tmp_path), files_mode="none")

    def test_none_allows_swival_dir(self, tmp_path):
        swival = tmp_path / ".swival" / "memory"
        swival.mkdir(parents=True)
        (swival / "MEMORY.md").write_text("notes")
        result = safe_resolve(
            ".swival/memory/MEMORY.md", str(tmp_path), files_mode="none"
        )
        assert result == (tmp_path / ".swival" / "memory" / "MEMORY.md").resolve()

    def test_some_allows_base_dir(self, tmp_path):
        (tmp_path / "foo.txt").write_text("hello")
        result = safe_resolve("foo.txt", str(tmp_path), files_mode="some")
        assert result == (tmp_path / "foo.txt").resolve()

    def test_some_blocks_outside(self, tmp_path):
        with pytest.raises(ValueError, match="outside base directory"):
            safe_resolve("/etc/passwd", str(tmp_path), files_mode="some")

    def test_all_allows_anywhere(self, tmp_path):
        result = safe_resolve("/etc/hosts", str(tmp_path), files_mode="all")
        assert result == Path("/etc/hosts").resolve()

    def test_all_blocks_root(self, tmp_path):
        with pytest.raises(ValueError, match="filesystem root"):
            safe_resolve("/", str(tmp_path), files_mode="all")


class TestIsWithinBase:
    def test_none_blocks_project_file(self, tmp_path):
        f = tmp_path / "foo.txt"
        f.write_text("x")
        assert not _is_within_base(f, tmp_path, files_mode="none")

    def test_none_allows_swival_file(self, tmp_path):
        swival = tmp_path / ".swival"
        swival.mkdir()
        f = swival / "test.txt"
        f.write_text("x")
        assert _is_within_base(f, tmp_path, files_mode="none")

    def test_some_allows_base_dir(self, tmp_path):
        f = tmp_path / "foo.txt"
        f.write_text("x")
        assert _is_within_base(f, tmp_path, files_mode="some")

    def test_all_allows_anything(self, tmp_path):
        assert _is_within_base(Path("/etc/hosts"), tmp_path, files_mode="all")


class TestReadFile:
    def test_none_blocks_project_read(self, tmp_path):
        (tmp_path / "foo.txt").write_text("hello")
        result = _read_file("foo.txt", str(tmp_path), files_mode="none")
        assert result.startswith("error:")

    def test_none_allows_swival_read(self, tmp_path):
        swival = tmp_path / ".swival" / "memory"
        swival.mkdir(parents=True)
        (swival / "MEMORY.md").write_text("line1\nline2")
        result = _read_file(
            ".swival/memory/MEMORY.md", str(tmp_path), files_mode="none"
        )
        assert "line1" in result

    def test_all_reads_anywhere(self, tmp_path):
        (tmp_path / "foo.txt").write_text("hello")
        result = _read_file(str(tmp_path / "foo.txt"), str(tmp_path), files_mode="all")
        assert "hello" in result


class TestWriteFile:
    def test_none_blocks_project_write(self, tmp_path):
        result = _write_file("foo.txt", "content", str(tmp_path), files_mode="none")
        assert result.startswith("error:")
        assert not (tmp_path / "foo.txt").exists()

    def test_none_allows_swival_write(self, tmp_path):
        (tmp_path / ".swival").mkdir()
        result = _write_file(
            ".swival/test.txt", "content", str(tmp_path), files_mode="none"
        )
        assert not result.startswith("error:")
        assert (tmp_path / ".swival" / "test.txt").read_text() == "content"

    def test_none_blocks_move_from_outside(self, tmp_path):
        (tmp_path / ".swival").mkdir()
        (tmp_path / "src.txt").write_text("data")
        result = _write_file(
            ".swival/dst.txt",
            None,
            str(tmp_path),
            move_from="src.txt",
            files_mode="none",
        )
        assert result.startswith("error:")

    def test_none_allows_move_within_swival(self, tmp_path):
        swival = tmp_path / ".swival"
        swival.mkdir()
        (swival / "src.txt").write_text("data")
        result = _write_file(
            ".swival/dst.txt",
            None,
            str(tmp_path),
            move_from=".swival/src.txt",
            files_mode="none",
        )
        assert not result.startswith("error:")


class TestEditFile:
    def test_none_blocks_project_edit(self, tmp_path):
        (tmp_path / "foo.txt").write_text("hello world")
        result = _edit_file(
            "foo.txt", "hello", "goodbye", str(tmp_path), files_mode="none"
        )
        assert result.startswith("error:")


class TestDeleteFile:
    def test_none_blocks_project_delete(self, tmp_path):
        (tmp_path / "foo.txt").write_text("hello")
        result = _delete_file("foo.txt", str(tmp_path), files_mode="none")
        assert result.startswith("error:")
        assert (tmp_path / "foo.txt").exists()


class TestListFiles:
    def test_none_blocks_project_listing(self, tmp_path):
        (tmp_path / "foo.txt").write_text("hello")
        result = _list_files("*", ".", str(tmp_path), files_mode="none")
        assert result.startswith("error:")

    def test_none_allows_swival_listing(self, tmp_path):
        swival = tmp_path / ".swival"
        swival.mkdir()
        (swival / "test.txt").write_text("hello")
        result = _list_files("*", ".swival", str(tmp_path), files_mode="none")
        assert "test.txt" in result


class TestGrep:
    def test_none_blocks_project_grep(self, tmp_path):
        (tmp_path / "foo.txt").write_text("hello world")
        result = _grep("hello", ".", str(tmp_path), files_mode="none")
        assert result.startswith("error:")

    def test_none_allows_swival_grep(self, tmp_path):
        swival = tmp_path / ".swival"
        swival.mkdir()
        (swival / "test.txt").write_text("hello world")
        result = _grep("hello", ".swival", str(tmp_path), files_mode="none")
        assert "hello world" in result

    def test_all_skips_include_validation(self, tmp_path):
        (tmp_path / "foo.txt").write_text("hello")
        result = _grep(
            "hello", ".", str(tmp_path), include="/abs/pattern", files_mode="all"
        )
        assert not result.startswith("error: pattern")

    def test_some_enforces_include_validation(self, tmp_path):
        (tmp_path / "foo.txt").write_text("hello")
        result = _grep(
            "hello", ".", str(tmp_path), include="/abs/pattern", files_mode="some"
        )
        assert "error:" in result

    def test_none_enforces_include_validation(self, tmp_path):
        swival = tmp_path / ".swival"
        swival.mkdir()
        (swival / "foo.txt").write_text("hello")
        result = _grep(
            "hello",
            ".swival",
            str(tmp_path),
            include="/abs/pattern",
            files_mode="none",
        )
        assert "error:" in result


class TestDispatch:
    def test_dispatch_read_file_none(self, tmp_path):
        (tmp_path / "foo.txt").write_text("hello")
        result = dispatch(
            "read_file",
            {"file_path": "foo.txt"},
            str(tmp_path),
            files_mode="none",
        )
        assert result.startswith("error:")

    def test_dispatch_read_file_all(self, tmp_path):
        (tmp_path / "foo.txt").write_text("hello")
        result = dispatch(
            "read_file",
            {"file_path": str(tmp_path / "foo.txt")},
            str(tmp_path),
            files_mode="all",
        )
        assert "hello" in result

    def test_dispatch_run_command_independent(self, tmp_path):
        """run_command uses commands_unrestricted, not files_mode."""
        result = dispatch(
            "run_command",
            {"command": ["echo", "hi"]},
            str(tmp_path),
            files_mode="none",
            commands_unrestricted=True,
            resolved_commands={},
        )
        assert "hi" in result


class TestConfig:
    def test_files_validation_rejects_invalid(self):
        from swival.config import _validate_config

        with pytest.raises(Exception, match="'files' must be"):
            _validate_config({"files": "invalid"}, "test")

    def test_files_validation_accepts_valid(self):
        from swival.config import _validate_config

        for val in ("none", "some", "all"):
            _validate_config({"files": val}, "test")

    def test_config_to_session_kwargs_passthrough(self):
        from swival.config import config_to_session_kwargs

        result = config_to_session_kwargs({"files": "none"})
        assert result["files"] == "none"

    def test_args_to_session_kwargs_includes_files(self, tmp_path):
        from types import SimpleNamespace
        from swival.config import args_to_session_kwargs

        args = SimpleNamespace(
            files="none",
            yolo=False,
            commands="all",
            provider="lmstudio",
            model=None,
            api_key=None,
            base_url=None,
            max_turns=100,
            max_output_tokens=32768,
            max_context_tokens=None,
            temperature=None,
            top_p=1.0,
            seed=None,
            system_prompt=None,
            no_system_prompt=False,
            no_instructions=False,
            no_skills=False,
            sandbox="builtin",
            sandbox_session=None,
            sandbox_strict_read=False,
            no_sandbox_auto_session=False,
            no_read_guard=False,
            no_history=False,
            no_memory=False,
            quiet=False,
            no_continue=False,
            no_lifecycle=False,
            add_dir=[],
            add_dir_ro=[],
            skills_dir=None,
            subagents=False,
            no_subagents=False,
            memory_full=False,
            config_dir=None,
            proactive_summaries=False,
            extra_body=None,
            reasoning_effort=None,
            sanitize_thinking=False,
            prompt_cache=True,
            cache=False,
            cache_dir=None,
            retries=5,
            llm_filter=None,
            encrypt_secrets=False,
            no_encrypt_secrets=False,
            encrypt_secrets_key=None,
            encrypt_secrets_tweak=None,
            encrypt_secrets_patterns=None,
            lifecycle_command=None,
            lifecycle_timeout=300,
            lifecycle_fail_closed=False,
            aws_profile=None,
        )
        result = args_to_session_kwargs(args, str(tmp_path))
        assert result["files"] == "none"


class TestSessionFiles:
    def test_session_yolo_upgrades_files(self):
        from swival.session import Session

        s = Session(yolo=True)
        assert s.files == "all"

    def test_session_explicit_files_not_overridden(self):
        from swival.session import Session

        s = Session(yolo=True, files="none")
        assert s.files == "none"

    def test_session_default_files(self):
        from swival.session import Session

        s = Session()
        assert s.files == "some"

    def test_session_explicit_some_not_overridden_by_yolo(self):
        from swival.session import Session

        s = Session(yolo=True, files="some")
        assert s.files == "some"

    def test_session_yolo_explicit_commands_none(self):
        from swival.session import Session

        s = Session(yolo=True, commands="none")
        assert s.commands == "none"
        assert s.files == "all"

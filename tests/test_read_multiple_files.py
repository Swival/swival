"""Tests for the read_multiple_files tool."""

import pytest

from swival.tools import _read_files, dispatch


class TestReadMultipleFilesBasic:
    """Basic positive-path tests."""

    def test_read_two_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("alpha\nbeta\n")
        (tmp_path / "b.txt").write_text("gamma\ndelta\n")

        result = _read_files(
            [{"file_path": "a.txt"}, {"file_path": "b.txt"}],
            str(tmp_path),
        )
        assert result.startswith("files_succeeded: 2")
        assert "files_with_errors: 0" in result
        assert "batch_truncated: false" in result
        assert "=== FILE: a.txt ===" in result
        assert "status: ok" in result
        assert "request: offset=1 limit=2000" in result
        assert "content_truncated: false" in result
        assert "1: alpha" in result
        assert "2: beta" in result
        assert "=== FILE: b.txt ===" in result
        assert "1: gamma" in result
        assert "2: delta" in result

    def test_single_file(self, tmp_path):
        (tmp_path / "one.txt").write_text("hello\n")
        result = _read_files([{"file_path": "one.txt"}], str(tmp_path))
        assert "files_succeeded: 1" in result
        assert "=== FILE: one.txt ===" in result
        assert "1: hello" in result

    def test_sections_separated_by_blank_line(self, tmp_path):
        (tmp_path / "a.txt").write_text("x\n")
        (tmp_path / "b.txt").write_text("y\n")
        result = _read_files(
            [{"file_path": "a.txt"}, {"file_path": "b.txt"}],
            str(tmp_path),
        )
        assert "\n\n=== FILE: a.txt ===" in result
        assert "\n\n=== FILE: b.txt ===" in result


class TestReadMultipleFilesOffsetLimit:
    """Per-file offset/limit/tail support."""

    def test_different_offsets(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("\n".join(f"line{i}" for i in range(1, 11)) + "\n")

        result = _read_files(
            [
                {"file_path": "data.txt", "offset": 3, "limit": 2},
                {"file_path": "data.txt", "offset": 8, "limit": 2},
            ],
            str(tmp_path),
        )
        assert "request: offset=3 limit=2" in result
        assert "request: offset=8 limit=2" in result
        assert "3: line3" in result
        assert "4: line4" in result
        assert "8: line8" in result
        assert "9: line9" in result

    def test_tail(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("\n".join(f"line{i}" for i in range(1, 11)) + "\n")

        result = _read_files(
            [{"file_path": "data.txt", "tail": 3}],
            str(tmp_path),
        )
        assert "request: tail=3" in result
        assert "8: line8" in result
        assert "10: line10" in result
        assert "1: line1" not in result

    def test_next_offset_metadata_present_when_content_truncated(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("\n".join(f"line{i}" for i in range(1, 8)) + "\n")

        result = _read_files(
            [{"file_path": "data.txt", "offset": 2, "limit": 2}],
            str(tmp_path),
        )
        assert "content_truncated: true" in result
        assert "[next_offset=4]" in result


class TestReadMultipleFilesErrors:
    """Per-file error handling."""

    def test_missing_file_inline_error(self, tmp_path):
        (tmp_path / "good.txt").write_text("ok\n")
        result = _read_files(
            [{"file_path": "good.txt"}, {"file_path": "missing.txt"}],
            str(tmp_path),
        )
        assert "files_succeeded: 1" in result
        assert "files_with_errors: 1" in result
        assert "=== FILE: good.txt ===" in result
        assert "1: ok" in result
        assert "=== FILE: missing.txt ===" in result
        assert "status: error" in result
        assert "error: path does not exist: missing.txt" in result

    def test_empty_files_list(self, tmp_path):
        result = _read_files([], str(tmp_path))
        assert result == "error: files list is empty"

    def test_too_many_files(self, tmp_path):
        files = [{"file_path": f"f{i}.txt"} for i in range(25)]
        result = _read_files(files, str(tmp_path))
        assert "error: too many files" in result
        assert "maximum is 20" in result

    def test_missing_file_path_key(self, tmp_path):
        result = _read_files([{"offset": 1}], str(tmp_path))
        assert "=== FILE: file 1 ===" in result
        assert "status: error" in result
        assert "error: missing file_path" in result

    def test_invalid_offset(self, tmp_path):
        (tmp_path / "a.txt").write_text("x\n")
        result = _read_files(
            [{"file_path": "a.txt", "offset": "abc"}],
            str(tmp_path),
        )
        assert "status: error" in result
        assert "error: offset must be an integer" in result

    def test_invalid_limit(self, tmp_path):
        (tmp_path / "a.txt").write_text("x\n")
        result = _read_files(
            [{"file_path": "a.txt", "limit": "abc"}],
            str(tmp_path),
        )
        assert "status: error" in result
        assert "error: limit must be an integer" in result

    def test_invalid_tail(self, tmp_path):
        (tmp_path / "a.txt").write_text("x\n")
        result = _read_files(
            [{"file_path": "a.txt", "tail": "abc"}],
            str(tmp_path),
        )
        assert "status: error" in result
        assert "error: tail must be an integer" in result

    def test_string_elements_coerced_to_dicts(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello\n")
        (tmp_path / "b.txt").write_text("world\n")
        result = _read_files(["a.txt", "b.txt"], str(tmp_path))
        assert "files_succeeded: 2" in result
        assert "=== FILE: a.txt ===" in result
        assert "1: hello" in result
        assert "=== FILE: b.txt ===" in result
        assert "1: world" in result

    def test_non_dict_non_string_element_rejected(self, tmp_path):
        result = _read_files([123], str(tmp_path))
        assert "files_with_errors: 1" in result
        assert "error: expected object or string, got int" in result

    def test_binary_file_inline_error(self, tmp_path):
        (tmp_path / "good.txt").write_text("ok\n")
        (tmp_path / "bin.dat").write_bytes(b"\x00\x01\x02\x03")
        result = _read_files(
            [{"file_path": "good.txt"}, {"file_path": "bin.dat"}],
            str(tmp_path),
        )
        assert "1: ok" in result
        assert "=== FILE: bin.dat ===" in result
        assert "status: error" in result
        assert "binary file detected" in result


class TestReadMultipleFilesPathEscape:
    """Sandbox enforcement."""

    def test_dotdot_escape_rejected(self, tmp_path):
        result = _read_files(
            [{"file_path": "../../../etc/passwd"}],
            str(tmp_path),
        )
        assert "status: error" in result
        assert "error:" in result

    def test_symlink_escape_rejected(self, tmp_path):
        import os

        target = tmp_path.parent / "secret.txt"
        target.write_text("secret\n")
        link = tmp_path / "escape"
        os.symlink(target, link)
        result = _read_files(
            [{"file_path": "escape"}],
            str(tmp_path),
        )
        assert "status: error" in result
        assert "error:" in result


class TestReadMultipleFilesTruncation:
    """Total output truncation."""

    def test_truncation_when_budget_exceeded(self, tmp_path):
        per_file_lines = 500
        big_content = "\n".join(f"{'x' * 200}" for _ in range(per_file_lines)) + "\n"
        num_files = 10
        for i in range(num_files):
            (tmp_path / f"f{i:04d}.txt").write_text(big_content)

        files = [{"file_path": f"f{i:04d}.txt"} for i in range(num_files)]
        result = _read_files(files, str(tmp_path))
        assert "batch_truncated: true" in result
        assert "[batch_truncated:" in result
        assert "skipped due to size limit" in result

    def test_single_oversized_file_still_returns_content(self, tmp_path):
        big_content = "\n".join(f"line{i} {'x' * 200}" for i in range(500)) + "\n"
        (tmp_path / "big.txt").write_text(big_content)

        result = _read_files([{"file_path": "big.txt"}], str(tmp_path))
        assert "=== FILE: big.txt ===" in result
        assert "status: ok" in result
        assert "1: line0" in result
        assert "batch_truncated: false" in result
        assert "[batch_truncated:" not in result

    def test_oversized_first_of_two_includes_first(self, tmp_path):
        big_content = "\n".join(f"line{i} {'x' * 200}" for i in range(500)) + "\n"
        (tmp_path / "big.txt").write_text(big_content)
        (tmp_path / "small.txt").write_text("hello\n")

        result = _read_files(
            [{"file_path": "big.txt"}, {"file_path": "small.txt"}],
            str(tmp_path),
        )
        assert "=== FILE: big.txt ===" in result
        assert "1: line0" in result
        assert "batch_truncated: true" in result
        assert "[batch_truncated: 1 file(s) skipped due to size limit]" in result
        assert "=== FILE: small.txt ===" not in result


class TestReadMultipleFilesDirectories:
    """Directory rejection."""

    def test_directory_rejected_inline(self, tmp_path):
        (tmp_path / "subdir").mkdir()
        (tmp_path / "good.txt").write_text("ok\n")

        result = _read_files(
            [{"file_path": "good.txt"}, {"file_path": "subdir"}],
            str(tmp_path),
        )
        assert "=== FILE: good.txt ===" in result
        assert "1: ok" in result
        assert "=== FILE: subdir ===" in result
        assert "status: error" in result
        assert "is a directory" in result

    def test_directory_only(self, tmp_path):
        (tmp_path / "mydir").mkdir()
        result = _read_files([{"file_path": "mydir"}], str(tmp_path))
        assert "=== FILE: mydir ===" in result
        assert "status: error" in result
        assert "is a directory" in result


class TestReadMultipleFilesTracker:
    """Read-before-write tracking."""

    def test_tracker_records_reads(self, tmp_path):
        (tmp_path / "a.txt").write_text("x\n")
        (tmp_path / "b.txt").write_text("y\n")

        class FakeTracker:
            def __init__(self):
                self.reads = []

            def record_read(self, path):
                self.reads.append(path)

        tracker = FakeTracker()
        _read_files(
            [{"file_path": "a.txt"}, {"file_path": "b.txt"}],
            str(tmp_path),
            tracker=tracker,
        )
        assert len(tracker.reads) == 2
        assert any("a.txt" in r for r in tracker.reads)
        assert any("b.txt" in r for r in tracker.reads)


class TestReadMultipleFilesDispatch:
    """Dispatch integration."""

    def test_dispatch_read_multiple_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello\n")
        (tmp_path / "b.txt").write_text("world\n")

        result = dispatch(
            "read_multiple_files",
            {"files": [{"file_path": "a.txt"}, {"file_path": "b.txt"}]},
            str(tmp_path),
        )
        assert result.startswith("files_succeeded:")
        assert "=== FILE: a.txt ===" in result
        assert "1: hello" in result
        assert "=== FILE: b.txt ===" in result
        assert "1: world" in result

    def test_dispatch_invalid_files_arg(self, tmp_path):
        result = dispatch(
            "read_multiple_files",
            {"files": "not a list"},
            str(tmp_path),
        )
        assert "error:" in result

    def test_alias_read_files_suggests_correct_name(self, tmp_path):
        with pytest.raises(KeyError, match="read_multiple_files"):
            dispatch("read_files", {}, str(tmp_path))

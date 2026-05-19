"""Tests for the checksum stale-guard surfaced by read_file and edit_file."""

import hashlib
import re

import pytest

from swival.tools import (
    CHECKSUM_HEX_LEN,
    _compute_checksum,
    _edit_file,
    _read_file,
    _read_files,
)


CHECKSUM_TRAILER_RE = re.compile(rf"\[checksum=([0-9a-f]{{{CHECKSUM_HEX_LEN}}})\]")


def _extract_checksum(read_output: str) -> str:
    match = CHECKSUM_TRAILER_RE.search(read_output)
    assert match, f"no [checksum=...] trailer in:\n{read_output}"
    return match.group(1)


def _expected_checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:CHECKSUM_HEX_LEN]


class TestReadFileEmitsHash:
    def test_trailer_present(self, tmp_path):
        p = tmp_path / "hello.txt"
        p.write_text("alpha\nbeta\n", encoding="utf-8")
        result = _read_file("hello.txt", str(tmp_path))
        tag = _extract_checksum(result)
        assert tag == _expected_checksum(p.read_bytes())

    def test_trailer_is_last_line(self, tmp_path):
        p = tmp_path / "hello.txt"
        p.write_text("alpha\n", encoding="utf-8")
        result = _read_file("hello.txt", str(tmp_path))
        assert (
            result.splitlines()[-1]
            == f"[checksum={_expected_checksum(p.read_bytes())}]"
        )

    def test_empty_file_still_has_hash(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("", encoding="utf-8")
        result = _read_file("empty.txt", str(tmp_path))
        assert result == f"[checksum={_expected_checksum(b'')}]"

    def test_hash_changes_on_edit(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("one\n", encoding="utf-8")
        first = _extract_checksum(_read_file("f.txt", str(tmp_path)))
        p.write_text("two\n", encoding="utf-8")
        second = _extract_checksum(_read_file("f.txt", str(tmp_path)))
        assert first != second

    def test_hash_stable_across_reads(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("same\n", encoding="utf-8")
        a = _extract_checksum(_read_file("f.txt", str(tmp_path)))
        b = _extract_checksum(_read_file("f.txt", str(tmp_path)))
        assert a == b

    def test_hash_after_pagination_trailer(self, tmp_path):
        p = tmp_path / "many.txt"
        p.write_text(
            "\n".join(f"line{i}" for i in range(1, 11)) + "\n", encoding="utf-8"
        )
        result = _read_file("many.txt", str(tmp_path), offset=1, limit=3)
        lines = result.splitlines()
        assert "more lines, use offset=" in lines[-2]
        assert CHECKSUM_TRAILER_RE.fullmatch(lines[-1])

    def test_directory_listing_omits_hash(self, tmp_path):
        (tmp_path / "a.txt").write_text("x", encoding="utf-8")
        result = _read_file(".", str(tmp_path))
        assert "[checksum=" not in result

    def test_hash_does_not_appear_in_error(self, tmp_path):
        result = _read_file("nope.txt", str(tmp_path))
        assert result.startswith("error:")
        assert "[checksum=" not in result


class TestReadMultipleFilesHash:
    def test_section_carries_checksum(self, tmp_path):
        p = tmp_path / "x.txt"
        p.write_text("abc\n", encoding="utf-8")
        result = _read_files([{"file_path": "x.txt"}], str(tmp_path))
        expected = _expected_checksum(p.read_bytes())
        assert f"checksum: {expected}" in result
        assert "[checksum=" not in result

    def test_per_file_checksums_independent(self, tmp_path):
        (tmp_path / "a.txt").write_text("aaa\n", encoding="utf-8")
        (tmp_path / "b.txt").write_text("bbb\n", encoding="utf-8")
        result = _read_files(
            [{"file_path": "a.txt"}, {"file_path": "b.txt"}], str(tmp_path)
        )
        hashes = re.findall(
            rf"^checksum: ([0-9a-f]{{{CHECKSUM_HEX_LEN}}})$", result, re.MULTILINE
        )
        assert len(hashes) == 2
        assert hashes[0] != hashes[1]


class TestEditFileHashGuard:
    def test_match_accepts_edit(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("hello world\n", encoding="utf-8")
        tag = _extract_checksum(_read_file("f.txt", str(tmp_path)))
        result = _edit_file(
            "f.txt",
            old_string="hello",
            new_string="goodbye",
            base_dir=str(tmp_path),
            checksum=tag,
        )
        assert not result.startswith("error:"), result
        assert p.read_text(encoding="utf-8") == "goodbye world\n"

    def test_mismatch_rejects_edit(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("hello world\n", encoding="utf-8")
        result = _edit_file(
            "f.txt",
            old_string="hello",
            new_string="goodbye",
            base_dir=str(tmp_path),
            checksum="deadbeef",
        )
        assert result.startswith("error:")
        assert "checksum mismatch" in result
        assert "Re-read and retry" in result
        assert p.read_text(encoding="utf-8") == "hello world\n"

    def test_missing_hash_still_works(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("hello world\n", encoding="utf-8")
        result = _edit_file(
            "f.txt",
            old_string="hello",
            new_string="goodbye",
            base_dir=str(tmp_path),
        )
        assert not result.startswith("error:"), result
        assert p.read_text(encoding="utf-8") == "goodbye world\n"

    def test_hash_is_case_insensitive(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("hi\n", encoding="utf-8")
        tag = _extract_checksum(_read_file("f.txt", str(tmp_path)))
        result = _edit_file(
            "f.txt",
            old_string="hi",
            new_string="bye",
            base_dir=str(tmp_path),
            checksum=tag.upper(),
        )
        assert not result.startswith("error:"), result

    def test_stale_hash_caught_after_external_write(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("v1\n", encoding="utf-8")
        tag = _extract_checksum(_read_file("f.txt", str(tmp_path)))
        p.write_text("v2\n", encoding="utf-8")
        result = _edit_file(
            "f.txt",
            old_string="v2",
            new_string="v3",
            base_dir=str(tmp_path),
            checksum=tag,
        )
        assert result.startswith("error:")
        assert "checksum mismatch" in result
        assert p.read_text(encoding="utf-8") == "v2\n"

    def test_non_string_hash_ignored(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("x\n", encoding="utf-8")
        result = _edit_file(
            "f.txt",
            old_string="x",
            new_string="y",
            base_dir=str(tmp_path),
            checksum=123,
        )
        assert not result.startswith("error:"), result
        assert p.read_text(encoding="utf-8") == "y\n"

    @pytest.mark.parametrize(
        "bogus",
        [
            "0",
            "1",
            "abc",
            "nope",
            "0123456789",
            "0123456789abc0",
            "0123456789xy",
            " ",
        ],
    )
    def test_malformed_hash_ignored(self, tmp_path, bogus):
        p = tmp_path / "f.txt"
        p.write_text("x\n", encoding="utf-8")
        result = _edit_file(
            "f.txt",
            old_string="x",
            new_string="y",
            base_dir=str(tmp_path),
            checksum=bogus,
        )
        assert not result.startswith("error:"), (bogus, result)
        assert p.read_text(encoding="utf-8") == "y\n"


class TestComputeFileHash:
    def test_known_value(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_bytes(b"hello")
        assert _compute_checksum(p) == _expected_checksum(b"hello")

    def test_missing_file_returns_none(self, tmp_path):
        assert _compute_checksum(tmp_path / "missing.txt") is None

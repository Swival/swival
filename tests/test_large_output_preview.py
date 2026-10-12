"""Tests for the inline preview added by _save_large_output()."""

import re
import subprocess
import sys

from swival.tools import (
    LARGE_OUTPUT_PREVIEW_BYTES,
    LARGE_OUTPUT_PREVIEW_LINES,
    _extract_preview,
    _guard_a2a_output,
    _guard_mcp_output,
    _save_large_output,
)

_TRUNCATION_LINES_RE = re.compile(
    r"\[\.\.\. preview includes lines 1-(\d+); use read_file offset=(\d+) for more\]"
)
_TRUNCATION_PARTIAL_RE = re.compile(
    r"\[\.\.\. preview truncated within line (\d+); use read_file for full output\]"
)
_ANY_TRUNCATION_RE = re.compile(r"\[\.\.\. preview (?:includes|truncated)[^\]]*\]")


def _between_sentinels(text: str) -> str:
    """Return the text between [preview] and [/preview] sentinels."""
    m = re.search(r"\[preview\]\n(.*)\n\[/preview\]", text, re.DOTALL)
    assert m, f"no [preview]…[/preview] block in:\n{text}"
    return m.group(1)


def _strip_truncation(text: str) -> str:
    """Remove the truncation indicator if it is the final line."""
    lines = text.split("\n")
    if lines and _ANY_TRUNCATION_RE.match(lines[-1]):
        lines = lines[:-1]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# _extract_preview unit tests
# ---------------------------------------------------------------------------


class TestExtractPreview:
    def test_line_limit(self):
        lines = [f"line_{i}" for i in range(200)]
        meta = _extract_preview("\n".join(lines))
        assert meta.last_line <= LARGE_OUTPUT_PREVIEW_LINES
        assert not meta.partial_line

    def test_byte_limit(self):
        line = "x" * 200
        payload = "\n".join([line] * 200)
        meta = _extract_preview(payload)
        assert len(meta.text.encode("utf-8")) <= LARGE_OUTPUT_PREVIEW_BYTES
        assert not meta.partial_line

    def test_short_input_returned_fully(self):
        payload = "hello\nworld"
        meta = _extract_preview(payload)
        assert meta.text == payload
        assert meta.last_line == 2
        assert not meta.partial_line

    def test_byte_limit_cuts_on_line_boundary(self):
        meta = _extract_preview("\n".join(["a" * 100] * 200))
        assert not meta.text.endswith("\n")
        assert "\n" in meta.text
        assert not meta.partial_line

    def test_line_count_matches_content(self):
        lines = [f"line_{i}" for i in range(10)]
        meta = _extract_preview("\n".join(lines))
        assert meta.last_line == 10
        assert meta.text == "\n".join(lines)
        assert not meta.partial_line

    def test_single_long_line_is_partial(self):
        payload = "x" * 5000
        meta = _extract_preview(payload)
        assert meta.partial_line
        assert meta.last_line == 1
        assert len(meta.text.encode("utf-8")) <= LARGE_OUTPUT_PREVIEW_BYTES


# ---------------------------------------------------------------------------
# _save_large_output — preview sentinels and file persistence
# ---------------------------------------------------------------------------


class TestSaveLargeOutputPreview:
    def test_preview_sentinels_present(self, tmp_path):
        payload = "x" * 20_000
        result = _save_large_output(payload, str(tmp_path))

        assert "[preview]" in result
        assert "[/preview]" in result
        assert "too large for context" in result

    def test_file_still_saved(self, tmp_path):
        payload = "line\n" * 5000
        result = _save_large_output(payload, str(tmp_path))

        swival_dir = tmp_path / ".swival"
        files = list(swival_dir.glob("cmd_output_*.txt"))
        assert len(files) == 1
        assert files[0].read_text() == payload

        assert "cmd_output_" in result

    def test_multiline_truncation_shows_offset(self, tmp_path):
        payload = "\n".join(f"line_{i}" for i in range(500))
        result = _save_large_output(payload, str(tmp_path))

        body = _between_sentinels(result)
        m = _TRUNCATION_LINES_RE.search(body)
        assert m, f"no line-range truncation indicator in:\n{body}"
        last_line = int(m.group(1))
        next_offset = int(m.group(2))
        assert 1 <= last_line <= LARGE_OUTPUT_PREVIEW_LINES
        assert next_offset == last_line + 1

    def test_single_long_line_truncation_message(self, tmp_path):
        payload = "A" * 20_000
        result = _save_large_output(payload, str(tmp_path))

        body = _between_sentinels(result)
        m = _TRUNCATION_PARTIAL_RE.search(body)
        assert m, f"expected partial-line truncation in:\n{body}"
        assert int(m.group(1)) == 1
        assert _TRUNCATION_LINES_RE.search(body) is None

    def test_preview_body_within_byte_budget(self, tmp_path):
        payload = "a" * 20_000
        result = _save_large_output(payload, str(tmp_path))

        body = _between_sentinels(result)
        body_content = _strip_truncation(body)
        assert len(body_content.encode("utf-8")) <= LARGE_OUTPUT_PREVIEW_BYTES

    def test_preview_body_within_line_budget(self, tmp_path):
        payload = "\n".join(f"short_{i}" for i in range(500))
        result = _save_large_output(payload, str(tmp_path))

        body = _between_sentinels(result)
        body_lines = body.split("\n")
        if body_lines and _ANY_TRUNCATION_RE.match(body_lines[-1]):
            body_lines = body_lines[:-1]
        assert len(body_lines) <= LARGE_OUTPUT_PREVIEW_LINES


# ---------------------------------------------------------------------------
# Untrusted content handling
# ---------------------------------------------------------------------------


class TestUntrustedPreview:
    def test_untrusted_header_inside_sentinels(self, tmp_path):
        payload = "data\n" * 5000
        result = _save_large_output(
            payload, str(tmp_path), untrusted_source="mcp__srv__tool"
        )

        body = _between_sentinels(result)
        assert "[UNTRUSTED EXTERNAL CONTENT]" in body
        assert "source: mcp__srv__tool" in body
        assert "policy:" in body

    def test_untrusted_header_does_not_eat_body_budget(self, tmp_path):
        line = "y" * 200
        payload = "\n".join([line] * 200)
        result = _save_large_output(
            payload, str(tmp_path), untrusted_source="mcp__s__t"
        )

        body = _between_sentinels(result)
        lines = body.split("\n")
        header_end = 0
        for i, line in enumerate(lines):
            if line.startswith("policy:"):
                header_end = i + 2
                break
        tail = lines[header_end:]
        if tail and _ANY_TRUNCATION_RE.match(tail[-1]):
            tail = tail[:-1]
        content_lines = tail
        content = "\n".join(content_lines)
        assert len(content.encode("utf-8")) > LARGE_OUTPUT_PREVIEW_BYTES // 2

    def test_untrusted_header_in_file(self, tmp_path):
        payload = "z" * 20_000
        _save_large_output(payload, str(tmp_path), untrusted_source="mcp__s__t")

        files = list((tmp_path / ".swival").glob("cmd_output_*.txt"))
        assert len(files) == 1
        content = files[0].read_text()
        assert content.startswith("[UNTRUSTED EXTERNAL CONTENT]")
        assert payload in content


# ---------------------------------------------------------------------------
# _capture_process integration — status before the spill notice
# ---------------------------------------------------------------------------


def _capture_big_failure(tmp_path):
    from swival.tools import _capture_process

    proc = subprocess.Popen(
        [sys.executable, "-c", "import sys; print('A' * 25_000); sys.exit(42)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return _capture_process(proc, timeout=30, base_dir=str(tmp_path))


class TestCaptureProcessPreview:
    def test_exit_code_before_notice(self, tmp_path):
        result = _capture_big_failure(tmp_path)

        assert result.startswith("Exit code: 42\nCommand output too large")
        assert result.endswith("[/preview]")
        saved = next((tmp_path / ".swival").glob("cmd_output_*.txt"))
        assert saved.read_text() == "Exit code: 42\n" + "A" * 25_000 + "\n"

    def test_failed_save_keeps_one_status(self, tmp_path, monkeypatch):
        import swival.tools as tools

        def refuse(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(tools, "open", refuse, raising=False)
        result = _capture_big_failure(tmp_path)

        assert result.startswith("Exit code: 42\nAAA")
        assert result.count("Exit code: 42") == 1


# ---------------------------------------------------------------------------
# _guard_a2a_output integration — A2A metadata before preview
# ---------------------------------------------------------------------------


class TestA2aOutputPreview:
    def test_context_id_before_preview(self, tmp_path):
        body = "x" * 30_000
        payload = f"[contextId=abc123]\n{body}"
        result = _guard_a2a_output(payload, str(tmp_path), "a2a__agent__skill")

        assert result.startswith("[contextId=abc123]\n")
        assert "[preview]" in result
        assert "[/preview]" in result

        preview_body = _between_sentinels(result)
        assert "[contextId=abc123]" not in preview_body

    def test_input_required_before_preview(self, tmp_path):
        body = "y" * 30_000
        payload = f"[input-required] contextId=c1 taskId=t1\n{body}"
        result = _guard_a2a_output(payload, str(tmp_path), "a2a__agent__skill")

        assert "[input-required] contextId=c1 taskId=t1" in result

        meta_pos = result.index("[input-required]")
        preview_start = result.index("[preview]")
        assert meta_pos < preview_start


# ---------------------------------------------------------------------------
# _guard_mcp_output integration — untrusted header inside sentinels
# ---------------------------------------------------------------------------


class TestMcpOutputPreview:
    def test_untrusted_header_in_preview(self, tmp_path):
        payload = "data " * 10_000  # ~50KB
        result = _guard_mcp_output(payload, str(tmp_path), "mcp__srv__tool")

        assert "[preview]" in result
        assert "[/preview]" in result

        preview_body = _between_sentinels(result)
        assert "[UNTRUSTED EXTERNAL CONTENT]" in preview_body
        assert "source: mcp__srv__tool" in preview_body

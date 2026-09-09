"""Tests for the terminal control-sequence sanitizer."""

import pytest

from swival import terminal
from swival.terminal import (
    MAX_COMMITTED_BYTES,
    TerminalSink,
    sanitize_terminal_output,
)


def _feed_chunks(*chunks: bytes) -> str:
    sink = TerminalSink()
    for c in chunks:
        sink.feed(c)
    return sink.finalize()


def test_carriage_return_single_line_bar():
    assert sanitize_terminal_output("10%\r50%\r100%\n") == "100%"


def test_crlf_preserved_as_line_breaks():
    assert sanitize_terminal_output("line1\r\nline2\r\nline3") == "line1\nline2\nline3"


def test_plain_output_passes_through_byte_for_byte():
    text = "first\n\nsecond line   \nthird\n"
    assert sanitize_terminal_output(text) == text


def test_plain_trailing_whitespace_and_blank_lines_preserved():
    text = "a   \n\n\nb"
    assert sanitize_terminal_output(text) == text


def test_tqdm_style_frame_collapses():
    frames = "".join(f"\r{p:3d}%|{'#' * (p // 10):<10}| {p}/100" for p in (0, 37, 100))
    out = sanitize_terminal_output(frames + "\n")
    assert out == "100%|##########| 100/100"


def test_cursor_up_multibar_repaint_collapses():
    # Two progress bars repainted three times via cursor-up.
    out = []
    for step in (0, 50, 100):
        out.append(f"download A: {step}%\n")
        out.append(f"download B: {step}%\n")
        if step != 100:
            out.append("\x1b[2A")  # move up two lines for the next repaint
    result = sanitize_terminal_output("".join(out))
    assert result == "download A: 100%\ndownload B: 100%"
    assert "\x1b" not in result


def test_cursor_up_shorter_replacement_clears_with_erase_line():
    # First frame is long, second is shorter and uses erase-to-end-of-line.
    stream = "Processing file_aaaaaaaa.bin\n\x1b[1A\rDone\x1b[K\n"
    result = sanitize_terminal_output(stream)
    assert result == "Done"
    assert "aaaa" not in result


def test_cursor_up_shorter_without_erase_leaves_tail():
    # Without erase, a real terminal keeps the stale tail; we match that.
    stream = "100%done\n\x1b[1A\r50%\n"
    assert sanitize_terminal_output(stream) == "50%%done"


@pytest.mark.parametrize("command, last_line", [("B", "bottomX"), ("E", "Xottom")])
@pytest.mark.parametrize("count", [1, 2000])
def test_cursor_down_stops_at_bottom(monkeypatch, command, last_line, count):
    monkeypatch.setattr(terminal, "MAX_ROWS", 3)
    stream = f"top\nmiddle\nbottom\x1b[{count}{command}X"
    assert sanitize_terminal_output(stream) == f"top\nmiddle\n{last_line}"


@pytest.mark.parametrize("command", ["B", "E"])
def test_large_cursor_down_never_allocates_past_screen(command):
    class BoundedScreen(list):
        def append(self, row):
            assert len(self) < terminal.MAX_ROWS, "screen allocation exceeded its cap"
            super().append(row)

    sink = TerminalSink()
    sink._screen = BoundedScreen([[]])
    sink.feed(f"top\x1b[1000000000000{command}X".encode())
    lines = sink.finalize().splitlines()
    assert len(lines) == terminal.MAX_ROWS
    assert lines[0] == "top"
    assert lines[-1].strip() == "X"


def test_sgr_color_stripped():
    assert sanitize_terminal_output("\x1b[1;31mERROR\x1b[0m: failed") == "ERROR: failed"


def test_osc_title_and_hyperlink_stripped():
    title = "\x1b]0;my window title\x07hello"
    assert sanitize_terminal_output(title) == "hello"
    link = "\x1b]8;;https://example.com\x07click\x1b]8;;\x07 here"
    assert sanitize_terminal_output(link) == "click here"


def test_osc52_clipboard_stripped():
    stream = "before\x1b]52;c;ZXZpbA==\x07after"
    assert sanitize_terminal_output(stream) == "beforeafter"


def test_save_restore_cursor_no_leak():
    # Save at col 1, write past it, restore, overwrite from the saved column.
    esc = "X\x1b7YYYY\x1b8Z"  # ESC 7 / ESC 8
    assert sanitize_terminal_output(esc) == "XZYYY"
    csi = "X\x1b[sYYYY\x1b[uZ"  # CSI s / CSI u
    assert sanitize_terminal_output(csi) == "XZYYY"


def test_escape_split_across_chunks():
    # The CSI "31m" is split so the sink must hold the partial sequence.
    out = _feed_chunks(b"\x1b[3", b"1mred\x1b[0m done")
    assert out == "red done"


def test_utf8_character_split_across_chunks():
    snowman = "☃".encode("utf-8")  # 3 bytes
    out = _feed_chunks(b"snow" + snowman[:1], snowman[1:] + b"man")
    assert out == "snow☃man"


def test_non_ascii_c1_byte_range_round_trips_on_plain_path():
    # These characters' UTF-8 bytes fall in 0x80-0x9f but they are not C1
    # controls, so they must stay on the plain path and round-trip exactly.
    text = "café déjà Привет 日本語 šž\n"
    assert sanitize_terminal_output(text) == text


def test_large_plain_output_keeps_head_and_marks_truncated():
    line = "x" * 1000 + "\n"
    big = line * 2000  # ~2 MB, no control bytes
    sink = TerminalSink()
    sink.feed(big.encode("utf-8"))
    result = sink.finalize()
    assert sink.output_truncated is True
    assert len(result.encode("utf-8")) <= MAX_COMMITTED_BYTES
    assert result.startswith("x" * 1000)  # head kept, not tail


def test_allowlist_no_raw_controls_remain():
    stream = (
        "\x1b[2J\x1b[H\x1b[1;32mstatus\x1b[0m\r\x07\b"
        "\x1b]0;title\x07line\x1b[Kmore\n\ttabbed\x1b[3D\x00\x07"
    )
    result = sanitize_terminal_output(stream)
    forbidden = ("\x1b", "\r", "\b", "\x07", "\x00")
    for ch in forbidden:
        assert ch not in result
    for o in range(0x80, 0xA0):
        assert chr(o) not in result


def test_pathological_repaint_stays_bounded():
    stream = ("\rspinner frame\x1b[K" * 100000) + "\nfinal\n"
    sink = TerminalSink()
    sink.feed(stream.encode("utf-8"))
    result = sink.finalize()
    assert result == "spinner frame\nfinal"
    assert len(result.encode("utf-8")) < 10_000


def test_committed_ring_drops_oldest_when_scrolling():
    # Many scrolled lines past the cap: oldest drop, newest survive, flag set.
    line = "y" * 200 + "\n"
    big = line * 8000  # ~1.6 MB scrolled through control-mode
    sink = TerminalSink()
    sink.feed(b"\x1b[0m")  # force emulated mode up front
    sink.feed(big.encode("utf-8"))
    result = sink.finalize()
    assert sink.output_truncated is True
    assert len(result.encode("utf-8")) <= MAX_COMMITTED_BYTES + 200_000
    assert result.endswith("y" * 200)  # tail kept


def test_backspace_moves_cursor_back():
    assert sanitize_terminal_output("abc\b\bX") == "aXc"


def test_empty_output():
    assert sanitize_terminal_output("") == ""
    assert _feed_chunks() == ""


def test_tab_kept_literal_on_plain_path():
    # No control trigger, so the tab passes through untouched.
    assert sanitize_terminal_output("a\tb") == "a\tb"


def test_tab_expands_to_stop_in_emulated_mode():
    # A control byte forces emulation; the tab then positions the cursor.
    assert sanitize_terminal_output("\x1b[0ma\tb") == "a" + " " * 7 + "b"


@pytest.mark.parametrize("code", [*range(9), *range(11, 32), 127, *range(128, 160)])
def test_individual_controls_never_escape_plain_mode(code):
    control = chr(code)
    for chunks in (
        (f"before{control}after".encode(),),
        (b"before", control.encode(), b"after"),
    ):
        assert control not in _feed_chunks(*chunks)


@pytest.mark.parametrize(
    "sequence", ["\x1b(0", "\x1b)0", "\x1b(B", "\x1b%G", "\x1b$)B"]
)
def test_charset_designation_consumed_across_chunk_boundaries(sequence):
    data = ("before" + sequence + "after").encode()
    for split in range(len(data) + 1):
        assert _feed_chunks(data[:split], data[split:]) == "beforeafter"

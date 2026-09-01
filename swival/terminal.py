"""Streaming sanitizer for terminal control sequences in command output.

A program that draws a progress bar, spinner, or any other live-updating
display writes the same screen region over and over, separated by carriage
returns, cursor movements, and erase sequences. Concatenating the raw byte
stream keeps every intermediate frame, which both wastes the model's context
and corrupts the user's terminal when the captured string is echoed back.

`TerminalSink` consumes the byte stream as it is read from the pipe and
emulates a small terminal: a bounded virtual screen for the live region plus a
byte-capped ring for lines that have scrolled away. In-place repaints collapse
onto the same cells, so what survives is the final frame, the way a human would
see it after the command finished.

The sink starts in plain mode and stays there for output that never uses a
control character, passing it through byte-for-byte. It switches to emulated
mode the first time a control character appears, which is the only time the
screen model is worth its cost. Memory is bounded by the screen size plus the
committed-output cap, independent of how many frames the program drew.

`sanitize_terminal_output()` is the one-shot wrapper for callers that already
hold the whole string, such as the REPL quick-shell echo.
"""

import codecs
import re

MAX_ROWS = 200
MAX_COLS = 400
MAX_COMMITTED_BYTES = 1 * 1024 * 1024
TAB_WIDTH = 8
_MAX_PENDING = 8192

_CONTROL_RE = re.compile("[\x1b\r\x08-]")
_C1_INTRODUCERS = frozenset({0x90, 0x9B, 0x9D, 0x9E, 0x9F})


def _join_one_newline(a: str, b: str) -> str:
    """Join two regions with exactly one newline, and none if either is empty."""
    if not a:
        return b
    if not b:
        return a
    if a.endswith("\n"):
        return a + b
    return a + "\n" + b


class TerminalSink:
    """Streaming terminal emulator that collapses live-updating output.

    Feed raw byte chunks with :meth:`feed`, then read the cleaned string with
    :meth:`finalize`. ``output_truncated`` becomes true if retained output
    exceeded the cap and content had to be dropped.
    """

    def __init__(self):
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._mode = "plain"
        self._plain: list[str] = []
        self._plain_len = 0
        self._plain_capped = False
        self.output_truncated = False

        self._pending = ""
        self._screen: list[list[str]] = [[]]
        self._row = 0
        self._col = 0
        self._saved_cursor: tuple[int, int] | None = None
        self._committed: list[str] = []
        self._committed_bytes = 0

        self._finalized = False
        self._result: str | None = None

    def feed(self, chunk: bytes) -> None:
        if self._finalized:
            return
        text = self._decoder.decode(chunk)
        if text:
            self._feed_text(text)

    def finalize(self) -> str:
        if not self._finalized:
            tail = self._decoder.decode(b"", final=True)
            if tail:
                self._feed_text(tail)
            self._finalized = True
            self._result = self._build()
        return self._result

    def _build(self) -> str:
        if self._mode == "plain":
            return "".join(self._plain)
        active = ["".join(row) for row in self._screen]
        while active and not active[-1].strip():
            active.pop()
        active_text = "\n".join(active)
        committed_text = "\n".join(self._committed)
        return _join_one_newline(committed_text, active_text)

    def _feed_text(self, text: str) -> None:
        if self._mode == "emulated":
            self._feed_emulated(text)
            return
        if self._plain_capped:
            if text:
                self.output_truncated = True
            return
        if _CONTROL_RE.search(text) is None:
            self._append_plain(text)
            return
        buffered = "".join(self._plain)
        self._plain = []
        self._plain_len = 0
        self._mode = "emulated"
        if buffered:
            self._feed_emulated(buffered)
        self._feed_emulated(text)

    def _append_plain(self, text: str) -> None:
        enc = text.encode("utf-8")
        remaining = MAX_COMMITTED_BYTES - self._plain_len
        if len(enc) <= remaining:
            self._plain.append(text)
            self._plain_len += len(enc)
            if self._plain_len >= MAX_COMMITTED_BYTES:
                # Exactly filling the cap drops nothing; later input marks
                # the truncation.
                self._plain_capped = True
            return
        head_bytes = enc[:remaining]
        head = head_bytes.decode("utf-8", errors="ignore")
        if head:
            self._plain.append(head)
            self._plain_len += len(head_bytes)
        self._plain_capped = True
        self.output_truncated = True

    def _feed_emulated(self, text: str) -> None:
        data = self._pending + text
        self._pending = ""
        i = 0
        n = len(data)
        while i < n:
            ch = data[i]
            o = ord(ch)
            if ch == "\x1b" or o in _C1_INTRODUCERS:
                new_i, incomplete = self._handle_escape(data, i)
                if incomplete:
                    rest = data[i:]
                    if len(rest) <= _MAX_PENDING:
                        self._pending = rest
                    return
                i = new_i
                continue
            if ch == "\n":
                self._line_feed()
            elif ch == "\r":
                self._col = 0
            elif ch == "\x08":
                if self._col > 0:
                    self._col -= 1
            elif ch == "\t":
                self._tab()
            elif o < 0x20 or o == 0x7F or 0x80 <= o <= 0x9F:
                pass
            else:
                self._put(ch)
            i += 1

    def _handle_escape(self, data: str, i: int) -> tuple[int, bool]:
        n = len(data)
        ch = data[i]
        if ch == "\x1b":
            if i + 1 >= n:
                return i, True
            nxt = data[i + 1]
            if nxt == "[":
                return self._consume_csi(data, i + 2)
            if nxt in ("]", "P", "^", "_"):
                return self._consume_string(data, i + 2)
            if nxt == "7":
                self._save_cursor()
                return i + 2, False
            if nxt == "8":
                self._restore_cursor()
                return i + 2, False
            return i + 2, False
        o = ord(ch)
        if o == 0x9B:
            return self._consume_csi(data, i + 1)
        return self._consume_string(data, i + 1)

    def _consume_csi(self, data: str, start: int) -> tuple[int, bool]:
        n = len(data)
        j = start
        while j < n:
            oc = ord(data[j])
            if 0x20 <= oc <= 0x3F:
                j += 1
            elif 0x40 <= oc <= 0x7E:
                self._apply_csi(data[start:j], data[j])
                return j + 1, False
            else:
                return j, False
        return start, True

    def _consume_string(self, data: str, start: int) -> tuple[int, bool]:
        n = len(data)
        j = start
        while j < n:
            c = data[j]
            if c == "\x07":
                return j + 1, False
            if c == "\x1b":
                if j + 1 >= n:
                    return start, True
                if data[j + 1] == "\\":
                    return j + 2, False
                return j, False
            if ord(c) == 0x9C:
                return j + 1, False
            j += 1
        return start, True

    @staticmethod
    def _parse_params(params: str) -> list[int]:
        clean = params.lstrip("?>=!")
        out: list[int] = []
        for part in clean.split(";"):
            part = part.strip()
            if not part:
                out.append(0)
                continue
            try:
                out.append(int(part))
            except ValueError:
                out.append(0)
        return out

    def _apply_csi(self, params: str, final: str) -> None:
        if final == "m":
            return
        if final == "s":
            self._save_cursor()
            return
        if final == "u":
            self._restore_cursor()
            return
        nums = self._parse_params(params)
        first = nums[0] if nums else 0
        count = max(1, first)
        if final == "A":
            self._row = max(0, self._row - count)
        elif final == "B":
            self._advance_row(count)
        elif final == "C":
            self._col = min(MAX_COLS - 1, self._col + count)
        elif final == "D":
            self._col = max(0, self._col - count)
        elif final == "E":
            self._advance_row(count)
            self._col = 0
        elif final == "F":
            self._row = max(0, self._row - count)
            self._col = 0
        elif final == "G":
            self._col = max(0, min(MAX_COLS - 1, count - 1))
        elif final in ("H", "f"):
            r = (nums[0] - 1) if (len(nums) >= 1 and nums[0]) else 0
            c = (nums[1] - 1) if (len(nums) >= 2 and nums[1]) else 0
            self._row = max(0, min(r, MAX_ROWS - 1))
            while len(self._screen) <= self._row:
                self._screen.append([])
            self._col = max(0, min(MAX_COLS - 1, c))
        elif final == "K":
            self._erase_line(first)
        elif final == "J":
            self._erase_display(first)

    def _put(self, ch: str) -> None:
        if self._col >= MAX_COLS:
            self._line_feed()
        row = self._screen[self._row]
        while len(row) <= self._col:
            row.append(" ")
        row[self._col] = ch
        self._col += 1

    def _tab(self) -> None:
        stop = ((self._col // TAB_WIDTH) + 1) * TAB_WIDTH
        self._col = min(stop, MAX_COLS - 1)

    def _line_feed(self) -> None:
        self._col = 0
        self._advance_row(1)

    def _advance_row(self, k: int) -> None:
        self._row += k
        while len(self._screen) <= self._row:
            self._screen.append([])
        excess = len(self._screen) - MAX_ROWS
        if excess > 0:
            for _ in range(excess):
                self._commit_line("".join(self._screen.pop(0)))
            self._row = max(0, self._row - excess)
            if self._saved_cursor is not None:
                sr, sc = self._saved_cursor
                self._saved_cursor = (max(0, sr - excess), sc)

    def _commit_line(self, line: str) -> None:
        self._committed.append(line)
        self._committed_bytes += len(line.encode("utf-8")) + 1
        while self._committed_bytes > MAX_COMMITTED_BYTES and len(self._committed) > 1:
            dropped = self._committed.pop(0)
            self._committed_bytes -= len(dropped.encode("utf-8")) + 1
            self.output_truncated = True

    def _blank_to_col(self, row: list[str]) -> None:
        while len(row) <= self._col:
            row.append(" ")
        for c in range(self._col + 1):
            row[c] = " "

    def _erase_line(self, mode: int) -> None:
        row = self._screen[self._row]
        if mode == 0:
            del row[self._col :]
        elif mode == 1:
            self._blank_to_col(row)
        elif mode == 2:
            row.clear()

    def _erase_display(self, mode: int) -> None:
        if mode == 0:
            del self._screen[self._row][self._col :]
            del self._screen[self._row + 1 :]
        elif mode == 1:
            for r in range(self._row):
                self._screen[r] = []
            self._blank_to_col(self._screen[self._row])
        elif mode == 2:
            for r in range(len(self._screen)):
                self._screen[r] = []

    def _save_cursor(self) -> None:
        self._saved_cursor = (self._row, self._col)

    def _restore_cursor(self) -> None:
        if self._saved_cursor is None:
            return
        r, c = self._saved_cursor
        self._row = max(0, r)
        while len(self._screen) <= self._row:
            self._screen.append([])
        self._col = max(0, c)


def sanitize_terminal_output(text: str) -> str:
    """Collapse terminal control sequences in a complete output string."""
    sink = TerminalSink()
    sink.feed(text.encode("utf-8"))
    return sink.finalize()

"""Let herdr and other terminals know what Swival is doing.

Classic herdr is told through its command line tool. herdr-core, the herdr app and other terminals read status updates and the window title straight from the terminal.
"""

import atexit
from contextlib import contextmanager
import os
import re
import subprocess
import sys
import threading
import time
import unicodedata

from ._env import child_env
from ._msg import _trim_for_marquee

OWNER_ENV = "SWIVAL_HERDR_OWNER"
TERMINAL_OWNER_ENV = "SWIVAL_TERMINAL_OWNER"

# Older VTE terminals print unknown escape sequences instead of ignoring them.
_MIN_VTE_VERSION = 5400

_TITLE_CHARS = 50

# herdr-core wraps messages from other agents in <hc> tags, which don't belong in a title.
_HC_TAGS = re.compile(r"^\s*<hc\b[^<>]*>|</hc>\s*$")

# Save the user's window title before we change it, and put it back when we exit.
_PUSH_TITLE = "\x1b[22;0t"
_POP_TITLE = "\x1b[23;0t"


def _status_terminal():
    """Return stderr's file descriptor and terminal name, or None if stderr isn't a terminal."""
    # The Windows console would print these sequences as plain text.
    if sys.platform == "win32":
        return None
    try:
        fd = sys.stderr.fileno()
        name = os.ttyname(fd)
    except Exception:
        return None
    vte = os.environ.get("VTE_VERSION", "")
    if vte.isdigit() and int(vte) < _MIN_VTE_VERSION:
        return None
    return fd, name


def _status(body):
    # Program Status Protocol: https://www.superlogical.com/rex/docs/build/program-status
    return f"\x1b]7501;{body}\x1b\\"


def _title(text):
    return f"\x1b]0;{text}\x07"


def _title_text(text):
    """Turn the first line of a prompt into a short window title."""
    for line in text.splitlines():
        line = _HC_TAGS.sub("", line)
        # Drop control and invisible characters, which could break out of the escape sequence or hide text.
        words = "".join(
            ch for ch in line if ch.isspace() or unicodedata.category(ch)[0] != "C"
        ).split()
        if words:
            return _trim_for_marquee(" ".join(words), _TITLE_CHARS)
    return ""


def _write(fd, data):
    # Write to the file descriptor directly, so wrappers around sys.stderr can't hold back or reflow the sequence.
    try:
        os.write(fd, data.encode())
    except OSError:
        pass


class HerdrActivity:
    def __init__(self):
        self._lock = threading.RLock()
        self._turns = 0
        self._seq = 0
        self._registered = False
        self._pane_claimed = False
        # Terminals we wrote to, and whether we changed their title.
        self._terminals = {}

    def _send(self, command, state=None):
        binary = os.environ.get("HERDR_BIN_PATH")
        pane = os.environ.get("HERDR_PANE_ID")
        if os.environ.get("HERDR_ENV") != "1" or not binary or not pane:
            return
        self._seq = max(time.time_ns(), self._seq + 1)
        argv = [
            binary,
            "pane",
            command,
            pane,
            "--source",
            "swival",
            "--agent",
            "swival",
            "--seq",
            str(self._seq),
        ]
        if state is not None:
            argv.extend(["--state", state])
        try:
            subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=0.5,
                check=False,
                env=child_env(),
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _owns(self, env, place):
        # Child processes inherit this marker, so a Swival started by another one in the same pane or terminal stays quiet instead of overwriting its status.
        marker = f"{os.getpid()}@{place}"
        current = os.environ.get(env, "")
        if current != marker:
            if current.partition("@")[2] == place:
                return False
            os.environ[env] = marker
        return True

    def _terminal(self):
        found = _status_terminal()
        if found is None:
            return None
        fd, name = found
        return fd if self._owns(TERMINAL_OWNER_ENV, name) else None

    def report(self, state, kind=None):
        """Report ``state`` wherever we can, and return the terminal it went to."""
        pane = os.environ.get("HERDR_PANE_ID")
        with self._lock:
            to_pane = (
                os.environ.get("HERDR_ENV") == "1"
                and bool(pane)
                and self._owns(OWNER_ENV, pane)
            )
            terminal = self._terminal()
            if not to_pane and terminal is None:
                return None
            if not self._registered:
                atexit.register(self.release)
                self._registered = True
            # Update the terminal first, so a slow herdr command can't hold it up.
            if terminal is not None:
                self._terminals.setdefault(terminal, False)
                body = f"state={state}:app=swival"
                if kind is not None:
                    body += f":kind={kind}"
                _write(terminal, _status(body))
            if to_pane:
                self._pane_claimed = True
                self._send("report-agent", state)
            return terminal

    def release(self):
        with self._lock:
            # Clean up every terminal we wrote to, even if stderr points elsewhere now.
            for terminal, titled in self._terminals.items():
                data = _status("state=clear")
                if titled:
                    # Terminals with a title stack get their old title back, and the others get a blank one rather than a stale one.
                    data += _title("") + _POP_TITLE
                _write(terminal, data)
            self._terminals.clear()
            if self._pane_claimed:
                self._pane_claimed = False
                self._send("release-agent")

    def _set_title(self, terminal, text):
        title = _title_text(text)
        if not title:
            return
        # Only save the user's title the first time.
        push = "" if self._terminals[terminal] else _PUSH_TITLE
        self._terminals[terminal] = True
        _write(terminal, push + _title(title))

    @contextmanager
    def turn(self, *, enabled=True, title=None):
        if not enabled:
            yield
            return
        with self._lock:
            self._turns += 1
        try:
            with self._lock:
                if self._turns == 1:
                    terminal = self.report("working")
                    if terminal is not None and title:
                        self._set_title(terminal, title)
            yield
        finally:
            with self._lock:
                self._turns -= 1
                if self._turns == 0:
                    self.report("idle")

    @contextmanager
    def approval(self):
        with self._lock:
            active = self._turns > 0
        try:
            if active:
                self.report("blocked", kind="permission")
            yield
        finally:
            with self._lock:
                if active:
                    self.report("working" if self._turns else "idle")


activity = HerdrActivity()

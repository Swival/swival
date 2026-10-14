"""Report agent activity when running inside a herdr pane."""

import atexit
from contextlib import contextmanager
import os
import subprocess
import threading
import time

from ._env import child_env


class HerdrActivity:
    def __init__(self):
        self._lock = threading.RLock()
        self._turns = 0
        self._seq = 0
        self._registered = False
        self._claimed = False

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

    def report(self, state):
        if os.environ.get("HERDR_ENV") != "1":
            return
        with self._lock:
            if not self._registered:
                atexit.register(self.release)
                self._registered = True
            self._claimed = True
            self._send("report-agent", state)

    def release(self):
        with self._lock:
            if self._claimed:
                self._claimed = False
                self._send("release-agent")

    @contextmanager
    def turn(self, *, enabled=True):
        if not enabled:
            yield
            return
        with self._lock:
            self._turns += 1
        try:
            with self._lock:
                if self._turns == 1:
                    self.report("working")
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
                self.report("blocked")
            yield
        finally:
            with self._lock:
                if active:
                    self.report("working" if self._turns else "idle")


activity = HerdrActivity()

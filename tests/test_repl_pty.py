"""End-to-end REPL test in a pseudo terminal.

Runs the real CLI against a small OpenAI-compatible server that streams a
slow canned reply, drives it through a pty, and checks the rendered screen
with a terminal emulator. This is the only test that exercises the real
prompt_toolkit application, the output routing above the prompt, queueing
while a turn runs, and Ctrl-C delivery to the main thread.

Wall-clock sensitive, so it carries the ``stress`` marker and is excluded from
``make test``. Run it with ``pytest -m stress tests/test_repl_pty.py``.
"""

import fcntl
import json
import os
import pty
import select
import struct
import subprocess
import sys
import termios
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pyte = pytest.importorskip("pyte")

pytestmark = [
    pytest.mark.stress,
    pytest.mark.skipif(sys.platform == "win32", reason="needs a pty"),
]

COLS, ROWS = 100, 32
REPLY = "Here is what I found. " * 12


def _chunk(delta, finish=None):
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "fake-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        return

    def do_GET(self):
        body = json.dumps({"data": [{"id": "fake-model", "object": "model"}]})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        time.sleep(1.0)
        if not payload.get("stream"):
            body = json.dumps(
                {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "fake-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": REPLY},
                            "finish_reason": "stop",
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self._sse(_chunk({"role": "assistant", "content": ""}))
        for word in REPLY.split(" "):
            self._sse(_chunk({"content": word + " "}))
            time.sleep(0.08)
        self._sse(_chunk({}, finish="stop"))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _sse(self, obj):
        self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
        self.wfile.flush()


@pytest.fixture(scope="module")
def fake_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()


class _Repl:
    def __init__(self, workdir, base_url, xdg_home):
        self.master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
        env = dict(os.environ)
        env.update(
            TERM="xterm-256color",
            COLORTERM="truecolor",
            SWIVAL_ANIMATIONS="0",
            XDG_CONFIG_HOME=str(xdg_home),
        )

        def _ctty():
            os.setsid()
            fcntl.ioctl(slave, termios.TIOCSCTTY, 0)

        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from swival.agent import main; main()",
                "--repl",
                "--provider",
                "generic",
                "--base-url",
                base_url,
                "--model",
                "fake-model",
                "--api-key",
                "x",
                "--max-turns",
                "4",
            ],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            cwd=str(workdir),
            env=env,
            preexec_fn=_ctty,
            close_fds=True,
        )
        os.close(slave)
        self.screen = pyte.Screen(COLS, ROWS)
        self.stream = pyte.ByteStream(self.screen)

    def pump(self, seconds):
        deadline = time.monotonic() + seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            ready, _, _ = select.select([self.master], [], [], min(remaining, 0.05))
            if not ready:
                continue
            try:
                data = os.read(self.master, 65536)
            except OSError:
                return
            if not data:
                return
            self.stream.feed(data)
            for _ in range(data.count(b"\x1b[6n")):
                reply = f"\x1b[{self.screen.cursor.y + 1};{self.screen.cursor.x + 1}R"
                os.write(self.master, reply.encode())

    def wait_for(self, needle, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.pump(0.2)
            if needle in self.text():
                return True
        return False

    def send(self, text):
        os.write(self.master, text.encode())

    def text(self):
        return "\n".join(line.rstrip() for line in self.screen.display)

    def close(self):
        try:
            return self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            return None


@pytest.fixture
def repl(tmp_path, fake_server):
    workdir = tmp_path / "work"
    workdir.mkdir()
    (tmp_path / "xdg").mkdir()
    instance = _Repl(workdir, fake_server, tmp_path / "xdg")
    try:
        yield instance
    finally:
        if instance.proc.poll() is None:
            instance.proc.kill()
        instance.proc.wait(timeout=5)


def test_type_while_streaming_then_queue_and_interrupt(repl):
    assert repl.wait_for("Ask anything", 20)
    repl.send("first question\r")
    assert repl.wait_for("Here is what I found", 15)
    screen = repl.text()
    assert "❯ first question" in screen
    assert "Turn 1/4" in screen

    repl.send("typed during the turn")
    repl.pump(0.5)
    assert "typed during the turn" in repl.text()
    repl.send("\r")
    assert repl.wait_for("queued", 5)

    assert repl.wait_for("❯ typed during the turn", 20)
    repl.pump(1.5)
    repl.send("\x03")
    assert repl.wait_for("interrupted", 5)
    assert repl.wait_for("Ask anything", 5)

    repl.send("/exit\r")
    assert repl.close() == 0


def test_ctrl_c_at_idle_prompt(repl):
    assert repl.wait_for("Ask anything", 20)
    repl.send("half typed")
    repl.pump(0.3)
    assert "half typed" in repl.text()
    repl.send("\x03")
    repl.pump(0.5)
    assert "half typed" not in repl.text()
    assert "Ask anything" in repl.text()
    repl.send("\x03")
    assert repl.close() == 0

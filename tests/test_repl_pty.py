"""End-to-end REPL test in a pseudo terminal.

Runs the real CLI against a small OpenAI-compatible server that streams a
slow canned reply, drives it through a pty, and checks the rendered screen
with a terminal emulator. This is the only test that exercises the real
prompt_toolkit application, the output routing above the prompt, queueing
while a turn runs, and Ctrl-C delivery to the main thread.

Wall-clock sensitive, so it carries the ``stress`` marker and is excluded from
``make test``. Run it with ``pytest -m stress tests/test_repl_pty.py``.
"""

import contextlib
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

    def _reply(self, payload):
        time.sleep(1.0)
        return {"role": "assistant", "content": REPLY}

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        message = self._reply(payload)
        finish = "tool_calls" if message.get("tool_calls") else "stop"
        if not payload.get("stream"):
            body = json.dumps(
                {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "fake-model",
                    "choices": [
                        {"index": 0, "message": message, "finish_reason": finish}
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
        if message.get("tool_calls"):
            calls = [{"index": i, **tc} for i, tc in enumerate(message["tool_calls"])]
            self._sse(_chunk({"role": "assistant", "tool_calls": calls}))
        else:
            self._sse(_chunk({"role": "assistant", "content": ""}))
            words = message["content"].split(" ")
            for index, word in enumerate(words):
                self._sse(
                    _chunk({"content": word if index == len(words) - 1 else word + " "})
                )
                time.sleep(0.08)
        self._sse(_chunk({}, finish=finish))
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


@contextlib.contextmanager
def _running_repl(tmp_path, base_url):
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "notes.txt").write_text("notes\n")
    (tmp_path / "xdg").mkdir()
    instance = _Repl(workdir, base_url, tmp_path / "xdg")
    try:
        yield instance
    finally:
        if instance.proc.poll() is None:
            instance.proc.kill()
        instance.proc.wait(timeout=5)


@pytest.fixture
def repl(tmp_path, fake_server):
    with _running_repl(tmp_path, fake_server) as instance:
        yield instance


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


TOOL_CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "read_file", "arguments": '{"file_path": "notes.txt"}'},
}


@pytest.fixture
def scripted_server():
    """Answers each request with the next scripted reply ("tool" means a read_file call) and keeps the messages it got."""

    class _Scripted(_Handler):
        replies = []
        requests = []

        def _reply(self, payload):
            self.requests.append(payload.get("messages", []))
            # Slow only the first time, to leave room for typing the correction.
            if len(self.requests) == 1:
                time.sleep(2.0)
            reply = self.replies.pop(0) if self.replies else "All done."
            if reply == "tool":
                return {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]}
            return {"role": "assistant", "content": reply}

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Scripted)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", _Scripted
    finally:
        server.shutdown()


def _wait_for_requests(repl, handler, count, timeout):
    # Keep reading the terminal, or the REPL blocks on a full pty before sending anything.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and len(handler.requests) < count:
        repl.pump(0.1)
    return len(handler.requests) >= count


@pytest.fixture
def scripted_repl(tmp_path, scripted_server):
    url, handler = scripted_server
    with _running_repl(tmp_path, url) as instance:
        yield instance, handler


def _correct_during_first_request(repl, handler, correction):
    assert repl.wait_for("Ask anything", 20)
    repl.send("edit the tests\r")
    assert _wait_for_requests(repl, handler, 1, 10)
    repl.send(correction)
    repl.pump(0.3)
    repl.send("\x1b\r")


def test_alt_enter_reaches_the_model_after_the_tool_batch(scripted_repl):
    repl, handler = scripted_repl
    handler.replies[:] = ["tool", "Left the tests alone."]
    _correct_during_first_request(repl, handler, "leave the tests alone")
    assert repl.wait_for("Left the tests alone.", 20)

    # With Enter, the correction would only show up after the final answer.
    second = handler.requests[1]
    assert second[-2]["role"] == "tool"
    assert second[-1]["role"] == "user"
    assert "leave the tests alone" in json.dumps(second[-1]["content"])
    assert not any(m["role"] == "assistant" and m.get("content") for m in second)

    repl.send("/exit\r")
    assert repl.close() == 0


def test_alt_enter_during_the_final_answer_runs_next(scripted_repl):
    repl, handler = scripted_repl
    handler.replies[:] = ["Edited the tests.", "Reverted them."]
    _correct_during_first_request(repl, handler, "actually, leave them alone")
    # With Enter it would show as queued.
    assert repl.wait_for("next   actually, leave them alone", 5)
    assert "queued" not in repl.text()
    assert repl.wait_for("Edited the tests.", 20)
    assert repl.wait_for("Reverted them.", 20)

    second = handler.requests[1]
    assert second[-2]["role"] == "assistant"
    assert second[-2]["content"] == "Edited the tests."
    assert second[-1]["role"] == "user"
    assert "actually, leave them alone" in json.dumps(second[-1]["content"])

    repl.send("/exit\r")
    assert repl.close() == 0

"""Chrome DevTools Protocol driver for the browser_open / browser_eval tools.

Drives a headless Chrome (or Chromium) over CDP using only ``websockets`` (the
synchronous client) and ``httpx`` — both already in the dependency tree. No
Puppeteer or Playwright. The point is to load and render a page with a real
browser engine (running its JavaScript) so the model can search the web or
script a page, rather than fetching static HTML over plain HTTP.

A single browser process is shared per Swival process. It launches lazily on the
first tool call and is torn down at exit. Public surface used by ``tools.py``:

    configure(...)   set launch options (called once during session setup)
    browser_open()   navigate and return the rendered page as markdown/text/html
    browser_eval()   run a JavaScript function in the current page
    shutdown()       stop the browser (also registered with atexit)
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from websockets.sync.client import connect as _ws_connect

DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 120
_STARTUP_TIMEOUT = 20.0
_NETWORK_IDLE_QUIET = 0.5
_NETWORK_IDLE_MAX = 4.0
# How long after an evaluate() to watch for a navigation it may have triggered
# (form submit, click, location change) before deciding none happened.
_NAV_DETECT_WINDOW = 0.4

_MAC_CHROME_PATHS = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
)
_PATH_NAMES = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "chrome",
)


def find_chrome(explicit: str | None = None) -> str | None:
    """Locate a Chrome/Chromium executable.

    Order: explicit path, ``SWIVAL_CHROME`` / ``CHROME_PATH`` env vars, common
    macOS app bundles, then PATH lookups. Returns ``None`` if nothing is found.
    """
    candidates: list[str | None] = [
        explicit,
        os.environ.get("SWIVAL_CHROME"),
        os.environ.get("CHROME_PATH"),
    ]
    for cand in candidates:
        if cand and Path(cand).exists():
            return cand
    if sys.platform == "darwin":
        for path in _MAC_CHROME_PATHS:
            if Path(path).exists():
                return path
    for name in _PATH_NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None


class BrowserError(Exception):
    """A browser launch, connection, or CDP error worth surfacing to the model."""


@dataclass
class BrowserConfig:
    enabled: bool = True
    headless: bool = True
    chrome_path: str | None = None
    profile: str | None = None


_config = BrowserConfig()
_session: "BrowserSession | None" = None
_lock = threading.Lock()


def configure(
    *,
    enabled: bool = True,
    headless: bool = True,
    chrome_path: str | None = None,
    profile: str | None = None,
) -> None:
    """Set browser launch options for this process (called during session setup)."""
    global _config
    _config = BrowserConfig(
        enabled=enabled, headless=headless, chrome_path=chrome_path, profile=profile
    )


def is_available(cfg: BrowserConfig | None = None) -> bool:
    """True when the browser is enabled and a Chrome binary can be located."""
    cfg = cfg or _config
    return cfg.enabled and find_chrome(cfg.chrome_path) is not None


class BrowserSession:
    """Owns one Chrome process and a CDP websocket to a single page target."""

    def __init__(self, cfg: BrowserConfig):
        self._cfg = cfg
        self._proc: subprocess.Popen | None = None
        self._ws = None
        self._profile_dir: str | None = None
        self._owns_profile = False
        self._msg_id = 0
        self._port: int | None = None
        self._call_lock = threading.Lock()
        # Per-action state, reset before each navigate()/evaluate().
        self._load_fired = False
        self._nav_started = False
        self._inflight = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        chrome = find_chrome(self._cfg.chrome_path)
        if not chrome:
            raise BrowserError(
                "no Chrome or Chromium executable found; install Chrome or set "
                "--browser-path / SWIVAL_CHROME"
            )
        if self._cfg.profile:
            self._profile_dir = self._cfg.profile
            os.makedirs(self._profile_dir, exist_ok=True)
        else:
            self._profile_dir = tempfile.mkdtemp(prefix="swival-chrome-")
            self._owns_profile = True

        args = [
            chrome,
            "--remote-debugging-port=0",
            "--remote-allow-origins=*",
            f"--user-data-dir={self._profile_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-background-networking",
            "--disable-sync",
            "--disable-default-apps",
            "--disable-popup-blocking",
            "--hide-crash-restore-bubble",
            "--disable-features=Translate,MediaRouter",
        ]
        if self._cfg.headless:
            args += ["--headless=new", "--disable-gpu"]
        if _needs_no_sandbox():
            args.append("--no-sandbox")
        args.append("about:blank")

        self._proc = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        self._port = self._read_devtools_port()
        self._connect_page()

    def _read_devtools_port(self) -> int:
        port_file = Path(self._profile_dir) / "DevToolsActivePort"
        deadline = time.monotonic() + _STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise BrowserError(
                    f"Chrome exited during startup (code {self._proc.returncode})"
                )
            if port_file.exists():
                text = port_file.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    try:
                        return int(text.splitlines()[0])
                    except (ValueError, IndexError):
                        pass
            time.sleep(0.05)
        raise BrowserError("timed out waiting for Chrome DevTools to come up")

    def _connect_page(self) -> None:
        base = f"http://127.0.0.1:{self._port}"
        ws_url: str | None = None
        user_agent: str | None = None
        with httpx.Client(timeout=10.0) as client:
            try:
                user_agent = client.get(f"{base}/json/version").json().get("User-Agent")
            except (httpx.HTTPError, json.JSONDecodeError):
                user_agent = None
            try:
                resp = client.put(f"{base}/json/new?about:blank")
                if resp.status_code == 405:
                    resp = client.get(f"{base}/json/new?about:blank")
                resp.raise_for_status()
                ws_url = resp.json().get("webSocketDebuggerUrl")
            except (httpx.HTTPError, json.JSONDecodeError):
                ws_url = None
            if not ws_url:
                # Fall back to an existing page target.
                targets = client.get(f"{base}/json").json()
                for t in targets:
                    if t.get("type") == "page" and t.get("webSocketDebuggerUrl"):
                        ws_url = t["webSocketDebuggerUrl"]
                        break
        if not ws_url:
            raise BrowserError("could not obtain a CDP page websocket endpoint")
        self._ws = _ws_connect(ws_url, max_size=64 * 1024 * 1024, open_timeout=10)
        self._command("Page.enable")
        self._command("Runtime.enable")
        self._command("Network.enable")
        # Present as a normal desktop Chrome rather than HeadlessChrome, which
        # many sites (search engines especially) refuse to serve.
        if user_agent and "Headless" in user_agent:
            self._command(
                "Network.setUserAgentOverride",
                {"userAgent": user_agent.replace("HeadlessChrome", "Chrome")},
            )

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
        if self._owns_profile and self._profile_dir:
            shutil.rmtree(self._profile_dir, ignore_errors=True)
            self._profile_dir = None

    def is_alive(self) -> bool:
        return (
            self._proc is not None
            and self._proc.poll() is None
            and self._ws is not None
        )

    # -- CDP plumbing ------------------------------------------------------

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    def _send(self, method: str, params: dict | None = None) -> int:
        mid = self._next_id()
        self._ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        return mid

    def _recv_one(self, timeout: float) -> dict:
        """Receive and parse one CDP message, updating navigation state."""
        raw = self._ws.recv(timeout=max(0.0, timeout))
        msg = json.loads(raw)
        method = msg.get("method")
        if method == "Page.loadEventFired":
            self._load_fired = True
        elif method in (
            "Page.frameStartedNavigating",
            "Page.frameScheduledNavigation",
            "Page.frameRequestedNavigation",
        ):
            self._nav_started = True
        elif method == "Network.requestWillBeSent":
            self._inflight += 1
        elif method in ("Network.loadingFinished", "Network.loadingFailed"):
            if self._inflight > 0:
                self._inflight -= 1
        return msg

    def _command(
        self, method: str, params: dict | None = None, timeout: float = 10.0
    ) -> dict:
        """Send a CDP command and return its result, ignoring interleaved events."""
        mid = self._send(method, params)
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BrowserError(f"timed out waiting for {method} result")
            msg = self._recv_one(remaining)
            if msg.get("id") == mid:
                if "error" in msg:
                    err = msg["error"]
                    raise BrowserError(f"{method} failed: {err.get('message', err)}")
                return msg.get("result", {})

    def _pump_until(self, predicate, deadline: float) -> bool:
        """Read events until ``predicate()`` is true or the deadline passes."""
        while not predicate():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                self._recv_one(remaining)
            except TimeoutError:
                return predicate()
        return True

    def _wait_network_idle(self, deadline: float) -> None:
        """Wait for a quiet period with no in-flight requests, bounded by deadline."""
        while time.monotonic() < deadline:
            quiet = min(_NETWORK_IDLE_QUIET, deadline - time.monotonic())
            try:
                self._recv_one(quiet)
            except TimeoutError:
                if self._inflight <= 0:
                    return

    # -- high-level operations --------------------------------------------

    def navigate(self, url: str, timeout: float) -> None:
        with self._call_lock:
            self._load_fired = False
            self._inflight = 0
            deadline = time.monotonic() + timeout
            result = self._command("Page.navigate", {"url": url}, timeout=timeout)
            if result.get("errorText"):
                raise BrowserError(f"navigation to {url} failed: {result['errorText']}")
            self._pump_until(lambda: self._load_fired, deadline)
            # Settle for network idle, but cap it well below the hard navigation
            # timeout so a page with perpetual background activity can't stall us.
            settle_deadline = min(deadline, time.monotonic() + _NETWORK_IDLE_MAX)
            self._wait_network_idle(settle_deadline)

    def evaluate(
        self,
        expression: str,
        timeout: float,
        await_promise: bool = True,
        detect_navigation: bool = False,
    ) -> dict:
        with self._call_lock:
            deadline = time.monotonic() + timeout
            self._load_fired = False
            self._nav_started = False
            self._inflight = 0
            try:
                result = self._command(
                    "Runtime.evaluate",
                    {
                        "expression": expression,
                        "returnByValue": True,
                        "awaitPromise": await_promise,
                        "userGesture": True,
                    },
                    timeout=timeout,
                )
            except BrowserError as exc:
                # A navigation triggered by the script can destroy the execution
                # context out from under the evaluation. Treat that as a
                # successful navigation rather than an error.
                if _is_context_destroyed(exc):
                    self._wait_after_navigation(deadline)
                    return {}
                raise
            if "exceptionDetails" in result:
                raise BrowserError(_format_exception(result["exceptionDetails"]))
            if detect_navigation:
                # The script may have kicked off a navigation (form submit, click,
                # location change). Watch briefly; if one started, wait for the new
                # page so a follow-up evaluate() reads the navigated page.
                grace = min(deadline, time.monotonic() + _NAV_DETECT_WINDOW)
                self._pump_until(lambda: self._nav_started or self._load_fired, grace)
                if self._nav_started or self._load_fired:
                    self._wait_after_navigation(deadline)
            return result.get("result", {})

    def _wait_after_navigation(self, deadline: float) -> None:
        """After a script-triggered navigation, wait for load then network idle."""
        if not self._load_fired:
            self._pump_until(lambda: self._load_fired, deadline)
        settle_deadline = min(deadline, time.monotonic() + _NETWORK_IDLE_MAX)
        self._wait_network_idle(settle_deadline)


def _needs_no_sandbox() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


def _is_context_destroyed(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "context was destroyed" in msg
        or "context with given id" in msg
        or "cannot find context" in msg
        or "execution context" in msg
        and "destroyed" in msg
    )


def _format_exception(details: dict) -> str:
    exc = details.get("exception", {})
    text = (
        exc.get("description") or exc.get("value") or details.get("text") or "exception"
    )
    return f"JavaScript error: {text}".replace("\n", " ")[:500]


def _get_session() -> BrowserSession:
    global _session
    with _lock:
        if _session is not None and not _session.is_alive():
            _session.close()
            _session = None
        if _session is None:
            sess = BrowserSession(_config)
            sess.start()
            _session = sess
            atexit.register(shutdown)
        return _session


def shutdown() -> None:
    global _session
    with _lock:
        if _session is not None:
            _session.close()
            _session = None


def _clamp_timeout(timeout) -> float:
    try:
        t = float(timeout)
    except (TypeError, ValueError):
        return float(DEFAULT_TIMEOUT)
    if t <= 0:
        return float(DEFAULT_TIMEOUT)
    return float(min(t, MAX_TIMEOUT))


_EXTRACT_JS = {
    "text": "document.body ? document.body.innerText : ''",
    "html": "document.documentElement ? document.documentElement.outerHTML : ''",
    "markdown": "document.documentElement ? document.documentElement.outerHTML : ''",
}


def browser_open(
    url: str,
    format: str = "markdown",
    base_dir: str | None = None,
    scratch_dir: str | None = None,
    timeout=DEFAULT_TIMEOUT,
    wait_ms=None,
) -> str:
    """Open ``url`` in Chrome and return the rendered page as markdown/text/html.

    Returns the content string on success, ``"error: ..."`` on failure.
    """
    from . import tools

    if not isinstance(url, str) or not url.strip():
        return "error: url is required"
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return "error: url must start with http:// or https://"
    if format not in ("markdown", "text", "html"):
        return f"error: format must be markdown, text, or html (got {format!r})"
    if not is_available():
        return _unavailable_error()

    t = _clamp_timeout(timeout)
    try:
        session = _get_session()
        session.navigate(url, t)
        if wait_ms:
            try:
                time.sleep(min(float(wait_ms) / 1000.0, MAX_TIMEOUT))
            except (TypeError, ValueError):
                pass
        meta = session.evaluate(
            "JSON.stringify({title: document.title, url: location.href})", t
        )
        info = json.loads(meta.get("value") or "{}")
        raw = session.evaluate(_EXTRACT_JS[format], t).get("value") or ""
    except BrowserError as exc:
        return f"error: {exc}"
    except Exception as exc:  # noqa: BLE001 — surface any driver failure as a tool error
        return f"error: browser failure: {type(exc).__name__}: {exc}"

    body = _render_content(raw, format)
    title = info.get("title") or ""
    final_url = info.get("url") or url
    header = f"# {title}\n" if title and format != "html" else ""
    output = f"{header}<{final_url}>\n\n{body}" if format != "html" else body

    encoded = output.encode("utf-8")
    if base_dir and len(encoded) > tools.MAX_OUTPUT_BYTES:
        output, _saved = tools._save_large_output_with_path(
            output,
            base_dir,
            scratch_dir=scratch_dir,
            untrusted_source="browser_open",
            untrusted_origin=final_url,
        )
    elif len(encoded) > tools.MAX_OUTPUT_BYTES:
        total = len(encoded)
        output = (
            encoded[: tools.MAX_OUTPUT_BYTES].decode("utf-8", errors="ignore")
            + f"\n[content truncated at {tools.MAX_OUTPUT_BYTES} bytes, total was {total} bytes]"
        )
    return output


def _render_content(raw: str, format: str) -> str:
    if format != "markdown":
        return raw
    if not raw.strip():
        return ""
    try:
        from html_to_markdown import convert

        result = convert(raw)
        text = result.content if hasattr(result, "content") else result
    except Exception:
        return raw
    if not isinstance(text, str):
        return raw
    import re

    return re.sub(r"!\[([^\]]*)\]\(data:[^)]+\)", r"![image: \1](data: omitted)", text)


def browser_eval(function: str, timeout=DEFAULT_TIMEOUT) -> str:
    """Run a JavaScript function in the current page; return its JSON result.

    ``function`` is a JS function expression, e.g. ``() => document.title`` or
    ``() => [...document.querySelectorAll('a')].map(a => a.href)``. The function
    may be async; its resolved return value is JSON-serialized. If the script
    triggers a navigation (form submit, click, location change), the call waits
    for the new page to finish loading before returning.
    """
    if not isinstance(function, str) or not function.strip():
        return "error: function is required (e.g. () => document.title)"
    if not is_available():
        return _unavailable_error()

    t = _clamp_timeout(timeout)
    wrapped = (
        "(async () => {"
        f" const __r = await ({function})();"
        " try { const __s = JSON.stringify(__r); return __s === undefined ? 'undefined' : __s; }"
        " catch (e) { return JSON.stringify(String(__r)); }"
        "})()"
    )
    try:
        session = _get_session()
        result = session.evaluate(wrapped, t, detect_navigation=True)
    except BrowserError as exc:
        return f"error: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"error: browser failure: {type(exc).__name__}: {exc}"
    value = result.get("value")
    if value is None:
        return "navigated (the page changed; call browser_eval again to read it)"
    return value if isinstance(value, str) else json.dumps(value)


_AFFORDANCE_JS = """
JSON.stringify((() => {
  const inputs = [...document.querySelectorAll('input, textarea')].filter((e) => {
    const t = (e.type || 'text').toLowerCase();
    return !['hidden', 'submit', 'button', 'image', 'reset', 'checkbox', 'radio', 'file'].includes(t);
  });
  return {
    forms: document.querySelectorAll('form').length,
    text_inputs: inputs.length,
    buttons: document.querySelectorAll('button, input[type=submit], input[type=button], [role=button]').length,
    links: document.querySelectorAll('a[href]').length,
  };
})())
"""


def interaction_summary(timeout=DEFAULT_TIMEOUT) -> dict:
    """Count interactive affordances on the currently open page.

    Returns a dict of integer counts (``forms``, ``text_inputs``, ``buttons``,
    ``links``) so the caller can ground a "use browser_eval next" hint in what
    the page actually offers. Only integers cross back — never page-controlled
    strings — so the counts are safe to embed in a trusted message. Returns an
    empty dict if the browser is unavailable or detection fails.
    """
    if not is_available():
        return {}
    try:
        session = _get_session()
        raw = session.evaluate(_AFFORDANCE_JS, _clamp_timeout(timeout)).get("value")
        data = json.loads(raw) if raw else {}
    except Exception:  # noqa: BLE001 — detection is best-effort; the hint degrades gracefully
        return {}
    return {k: int(v) for k, v in data.items() if isinstance(v, (int, float))}


def _unavailable_error() -> str:
    if not _config.enabled:
        return "error: browser tools are disabled (remove --no-browser to enable)"
    return (
        "error: no Chrome or Chromium executable found; install Google Chrome or "
        "set --browser-path / the SWIVAL_CHROME environment variable"
    )

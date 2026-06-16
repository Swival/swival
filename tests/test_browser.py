"""Tests for the Chrome-backed browser_open / browser_eval tools (browser.py).

The CDP message-handling logic is tested against a scripted fake websocket so no
real Chrome is launched. A real end-to-end test is gated behind the
``SWIVAL_BROWSER_LIVE=1`` environment variable.
"""

import json
import os

import pytest

import swival.browser as browser
from swival import agent
from swival.tools import _browser_open_hint, dispatch


@pytest.fixture(autouse=True)
def _reset_browser_state():
    """Each test starts from a known config and no live session."""
    browser.shutdown()
    browser.configure(enabled=True, headless=True)
    yield
    browser.shutdown()
    browser.configure(enabled=True, headless=True)


class FakeWS:
    """A scripted CDP websocket: queued inbound messages, recorded outbound."""

    def __init__(self, inbound=None):
        self.sent = []
        self._queue = list(inbound or [])
        self.closed = False

    def send(self, data):
        self.sent.append(json.loads(data))

    def recv(self, timeout=None):
        if not self._queue:
            raise TimeoutError("no more scripted messages")
        item = self._queue.pop(0)
        return json.dumps(item) if not isinstance(item, str) else item

    def close(self):
        self.closed = True


def _session_with_ws(ws):
    sess = browser.BrowserSession(browser.BrowserConfig())
    sess._ws = ws
    return sess


# =========================================================================
# find_chrome / is_available
# =========================================================================


class TestDiscovery:
    def test_explicit_path_wins(self, tmp_path):
        fake = tmp_path / "chrome"
        fake.write_text("#!/bin/sh\n")
        assert browser.find_chrome(str(fake)) == str(fake)

    def test_env_override(self, tmp_path, monkeypatch):
        fake = tmp_path / "chromium"
        fake.write_text("#!/bin/sh\n")
        monkeypatch.setenv("SWIVAL_CHROME", str(fake))
        assert browser.find_chrome() == str(fake)

    def test_missing_returns_none(self, monkeypatch):
        monkeypatch.delenv("SWIVAL_CHROME", raising=False)
        monkeypatch.delenv("CHROME_PATH", raising=False)
        monkeypatch.setattr(browser, "_MAC_CHROME_PATHS", ())
        monkeypatch.setattr(browser.shutil, "which", lambda _name: None)
        monkeypatch.setattr(browser.sys, "platform", "linux")
        assert browser.find_chrome() is None

    def test_is_available_reflects_config(self, tmp_path, monkeypatch):
        fake = tmp_path / "chrome"
        fake.write_text("")
        monkeypatch.setattr(browser, "find_chrome", lambda _e=None: str(fake))
        browser.configure(enabled=True)
        assert browser.is_available() is True
        browser.configure(enabled=False)
        assert browser.is_available() is False

    def test_is_available_false_without_chrome(self, monkeypatch):
        monkeypatch.setattr(browser, "find_chrome", lambda _e=None: None)
        browser.configure(enabled=True)
        assert browser.is_available() is False


# =========================================================================
# CDP message handling (fake websocket)
# =========================================================================


class TestCdpPlumbing:
    def test_recv_one_tracks_load_event(self):
        sess = _session_with_ws(FakeWS([{"method": "Page.loadEventFired"}]))
        sess._recv_one(0.1)
        assert sess._load_fired is True

    def test_recv_one_tracks_inflight(self):
        ws = FakeWS(
            [
                {"method": "Network.requestWillBeSent"},
                {"method": "Network.requestWillBeSent"},
                {"method": "Network.loadingFinished"},
                {"method": "Network.loadingFailed"},
            ]
        )
        sess = _session_with_ws(ws)
        sess._recv_one(0.1)
        sess._recv_one(0.1)
        assert sess._inflight == 2
        sess._recv_one(0.1)
        sess._recv_one(0.1)
        assert sess._inflight == 0

    def test_command_skips_events_and_matches_id(self):
        # An interleaved event arrives before the id-1 response.
        ws = FakeWS(
            [
                {"method": "Page.loadEventFired"},
                {"id": 1, "result": {"value": 42}},
            ]
        )
        sess = _session_with_ws(ws)
        result = sess._command("Runtime.evaluate", {"expression": "x"})
        assert result == {"value": 42}
        assert sess._ws.sent[0]["method"] == "Runtime.evaluate"
        assert sess._load_fired is True

    def test_command_raises_on_cdp_error(self):
        ws = FakeWS([{"id": 1, "error": {"message": "bad expression"}}])
        sess = _session_with_ws(ws)
        with pytest.raises(browser.BrowserError, match="bad expression"):
            sess._command("Runtime.evaluate", {"expression": "@@"})

    def test_pump_until_stops_on_predicate(self):
        ws = FakeWS(
            [
                {"method": "Network.requestWillBeSent"},
                {"method": "Page.loadEventFired"},
            ]
        )
        sess = _session_with_ws(ws)
        deadline = browser.time.monotonic() + 5
        assert sess._pump_until(lambda: sess._load_fired, deadline) is True

    def test_evaluate_returns_result_value(self):
        ws = FakeWS(
            [{"id": 1, "result": {"result": {"type": "string", "value": "hi"}}}]
        )
        sess = _session_with_ws(ws)
        out = sess.evaluate("'hi'", timeout=5)
        assert out == {"type": "string", "value": "hi"}

    def test_evaluate_raises_on_js_exception(self):
        ws = FakeWS(
            [
                {
                    "id": 1,
                    "result": {
                        "result": {"type": "object"},
                        "exceptionDetails": {
                            "exception": {"description": "Error: boom"}
                        },
                    },
                }
            ]
        )
        sess = _session_with_ws(ws)
        with pytest.raises(browser.BrowserError, match="boom"):
            sess.evaluate("throw new Error('boom')", timeout=5)

    def test_recv_one_tracks_nav_started(self):
        sess = _session_with_ws(FakeWS([{"method": "Page.frameStartedNavigating"}]))
        sess._recv_one(0.1)
        assert sess._nav_started is True

    def test_evaluate_waits_for_triggered_navigation(self):
        # Script returns, then a navigation it kicked off starts and the new
        # page loads. detect_navigation=True must consume both before returning.
        ws = FakeWS(
            [
                {"id": 1, "result": {"result": {"type": "string", "value": "ok"}}},
                {"method": "Page.frameStartedNavigating"},
                {"method": "Page.loadEventFired"},
            ]
        )
        sess = _session_with_ws(ws)
        out = sess.evaluate("submit()", timeout=5, detect_navigation=True)
        assert out == {"type": "string", "value": "ok"}
        assert sess._nav_started is True
        assert sess._load_fired is True

    def test_evaluate_no_wait_without_detect(self):
        # Same script but detect_navigation defaults False: the nav events are
        # left unconsumed and load_fired stays False.
        ws = FakeWS(
            [
                {"id": 1, "result": {"result": {"value": "ok"}}},
                {"method": "Page.loadEventFired"},
            ]
        )
        sess = _session_with_ws(ws)
        out = sess.evaluate("noop()", timeout=5)
        assert out == {"value": "ok"}
        assert sess._load_fired is False

    def test_evaluate_context_destroyed_is_navigation(self):
        # A navigation that destroys the execution context is a success, not an
        # error: wait for the new page and return an empty result.
        ws = FakeWS(
            [
                {"id": 1, "error": {"message": "Cannot find context with given id"}},
                {"method": "Page.loadEventFired"},
            ]
        )
        sess = _session_with_ws(ws)
        out = sess.evaluate("location.href='/x'", timeout=5, detect_navigation=True)
        assert out == {}
        assert sess._load_fired is True

    def test_is_context_destroyed(self):
        assert browser._is_context_destroyed(Exception("context was destroyed"))
        assert browser._is_context_destroyed(Exception("Cannot find context"))
        assert not browser._is_context_destroyed(Exception("syntax error"))


# =========================================================================
# helpers
# =========================================================================


class TestHelpers:
    def test_clamp_timeout(self):
        assert browser._clamp_timeout(10) == 10.0
        assert browser._clamp_timeout(0) == float(browser.DEFAULT_TIMEOUT)
        assert browser._clamp_timeout(-5) == float(browser.DEFAULT_TIMEOUT)
        assert browser._clamp_timeout("nan-ish") == float(browser.DEFAULT_TIMEOUT)
        assert browser._clamp_timeout(9999) == float(browser.MAX_TIMEOUT)

    def test_render_content_markdown(self):
        out = browser._render_content("<h1>Hi</h1><p>There</p>", "markdown")
        assert "Hi" in out and "There" in out

    def test_render_content_passthrough(self):
        html = "<h1>Hi</h1>"
        assert browser._render_content(html, "html") == html
        assert browser._render_content(html, "text") == html

    def test_render_content_strips_data_images(self):
        html = '<img src="data:image/png;base64,AAAA" alt="x">'
        out = browser._render_content(html, "markdown")
        assert "data:image/png;base64,AAAA" not in out

    def test_format_exception_is_compact(self):
        msg = browser._format_exception(
            {"exception": {"description": "Error: a\nb\nc"}}
        )
        assert "\n" not in msg
        assert "Error: a" in msg

    def test_interaction_summary_coerces_to_ints(self, monkeypatch):
        monkeypatch.setattr(browser, "is_available", lambda *a, **k: True)

        class StubSession:
            def evaluate(self, expression, timeout, **kwargs):
                return {"value": json.dumps({"forms": 2, "text_inputs": 1.0})}

        monkeypatch.setattr(browser, "_get_session", lambda: StubSession())
        out = browser.interaction_summary()
        assert out == {"forms": 2, "text_inputs": 1}
        assert all(isinstance(v, int) for v in out.values())

    def test_interaction_summary_degrades_on_failure(self, monkeypatch):
        monkeypatch.setattr(browser, "is_available", lambda *a, **k: True)

        class StubSession:
            def evaluate(self, expression, timeout, **kwargs):
                raise browser.BrowserError("boom")

        monkeypatch.setattr(browser, "_get_session", lambda: StubSession())
        assert browser.interaction_summary() == {}

    def test_browser_open_hint_branches(self):
        with_form = _browser_open_hint({"forms": 1, "text_inputs": 1})
        assert "form(s)" in with_form and "YOUR QUERY" in with_form
        no_form = _browser_open_hint({"forms": 0, "text_inputs": 0})
        assert "form(s)" not in no_form
        # Both branches must name browser_eval and flag the live page.
        for hint in (with_form, no_form):
            assert "browser_eval" in hint
            assert "OPEN" in hint


# =========================================================================
# tool validation and dispatch (no real browser)
# =========================================================================


class TestToolValidation:
    def test_open_rejects_non_http(self):
        assert browser.browser_open("ftp://x").startswith("error:")
        assert browser.browser_open("").startswith("error:")

    def test_open_rejects_bad_format(self, monkeypatch):
        monkeypatch.setattr(browser, "is_available", lambda *a, **k: True)
        assert browser.browser_open("https://x", format="pdf").startswith("error:")

    def test_eval_rejects_empty(self):
        assert browser.browser_eval("").startswith("error:")
        assert browser.browser_eval("   ").startswith("error:")

    def test_unavailable_when_disabled(self):
        browser.configure(enabled=False)
        out = browser.browser_open("https://example.com")
        assert out.startswith("error:") and "disabled" in out

    def test_unavailable_without_chrome(self, monkeypatch):
        browser.configure(enabled=True)
        monkeypatch.setattr(browser, "find_chrome", lambda _e=None: None)
        out = browser.browser_eval("() => 1")
        assert out.startswith("error:") and "Chrome" in out


class TestDispatchIntegration:
    """browser_open / browser_eval via dispatch using a stub session."""

    def _install_stub(self, monkeypatch, *, nav=None, evals=None):
        monkeypatch.setattr(browser, "is_available", lambda *a, **k: True)

        class StubSession:
            def navigate(self, url, timeout):
                if nav:
                    nav(url)

            def evaluate(
                self, expression, timeout, await_promise=True, detect_navigation=False
            ):
                return {"value": evals(expression)}

        monkeypatch.setattr(browser, "_get_session", lambda: StubSession())

    def test_browser_open_via_dispatch(self, monkeypatch):
        def evals(expr):
            if "location.href" in expr:
                return json.dumps({"title": "T", "url": "https://e/"})
            return "<h1>Body</h1>"

        self._install_stub(monkeypatch, evals=evals)
        out = dispatch("browser_open", {"url": "https://e", "format": "markdown"}, ".")
        assert "UNTRUSTED EXTERNAL CONTENT" in out
        assert "# T" in out
        assert "Body" in out
        # The trusted hint must point the model at browser_eval, before the data.
        assert "browser_eval" in out
        assert out.index("[browser]") < out.index("UNTRUSTED EXTERNAL CONTENT")

    def test_browser_open_hint_grounded_in_forms(self, monkeypatch):
        def evals(expr):
            if "location.href" in expr:
                return json.dumps({"title": "Search", "url": "https://s/"})
            if "text_inputs" in expr:  # the affordance probe
                return json.dumps(
                    {"forms": 1, "text_inputs": 1, "buttons": 2, "links": 5}
                )
            return "<h1>Search</h1>"

        self._install_stub(monkeypatch, evals=evals)
        out = dispatch("browser_open", {"url": "https://s"}, ".")
        # Grounded in the detected form: the model is handed a fill+submit example.
        assert "form(s)" in out
        assert "YOUR QUERY" in out
        assert "requestSubmit" in out

    def test_browser_eval_via_dispatch(self, monkeypatch):
        self._install_stub(monkeypatch, evals=lambda expr: '["a","b"]')
        out = dispatch("browser_eval", {"function": "() => ['a','b']"}, ".")
        assert "UNTRUSTED EXTERNAL CONTENT" in out
        assert '["a","b"]' in out

    def test_browser_open_error_not_wrapped(self):
        # No stub: disabled config yields an error string, returned verbatim.
        browser.configure(enabled=False)
        out = dispatch("browser_open", {"url": "https://e"}, ".")
        assert out.startswith("error:")
        assert "UNTRUSTED" not in out


# =========================================================================
# build_tools registration
# =========================================================================


class TestRegistration:
    def test_registered_when_enabled(self):
        tools = agent.build_tools({}, {}, commands_unrestricted=False, browser=True)
        names = {t["function"]["name"] for t in tools}
        assert {"browser_open", "browser_eval"} <= names

    def test_absent_when_disabled(self):
        tools = agent.build_tools({}, {}, commands_unrestricted=False, browser=False)
        names = {t["function"]["name"] for t in tools}
        assert not ({"browser_open", "browser_eval"} & names)


# =========================================================================
# live end-to-end (opt-in)
# =========================================================================


@pytest.mark.skipif(
    os.environ.get("SWIVAL_BROWSER_LIVE") != "1",
    reason="set SWIVAL_BROWSER_LIVE=1 to run the real-Chrome end-to-end test",
)
class TestLive:
    def test_open_and_eval_example_com(self, tmp_path):
        browser.configure(enabled=True, headless=True)
        out = browser.browser_open(
            "https://example.com", format="markdown", base_dir=str(tmp_path)
        )
        assert "Example Domain" in out
        title = browser.browser_eval("() => document.title")
        assert "Example Domain" in title

    def test_fill_submit_form_then_read_results(self, tmp_path):
        # The core interactive workflow: open a search homepage, fill and submit
        # its form with JS (which navigates), then read the results page — all
        # without a manual delay, because browser_eval waits for the navigation.
        browser.configure(enabled=True, headless=True)
        browser.browser_open(
            "https://lite.duckduckgo.com/lite/", format="text", base_dir=str(tmp_path)
        )
        browser.browser_eval(
            "() => { document.querySelector('input[name=q]').value = 'anthropic claude'; "
            "document.forms[0].submit(); }"
        )
        results = browser.browser_eval(
            "() => [...document.querySelectorAll('a.result-link')].slice(0,3)"
            ".map(a => a.textContent.trim())"
        )
        assert "claude" in results.lower()

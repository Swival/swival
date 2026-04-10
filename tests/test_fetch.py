"""Tests for fetch.py — fetch_url tool."""

import http.client
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from swival.fetch import (
    MAX_OUTPUT_BYTES,
    MAX_RESPONSE_SIZE,
    _RedirectError,
    _check_url_safety,
    _html_to_text,
    fetch_url,
)


# =========================================================================
# _html_to_text
# =========================================================================


class TestHtmlToText:
    def test_basic_text_extraction(self):
        html = "<html><body><p>Hello world</p></body></html>"
        assert "Hello world" in _html_to_text(html)

    def test_strips_script_and_style(self):
        html = (
            "<html><head><style>body { color: red; }</style></head>"
            "<body><script>alert('x')</script><p>visible</p></body></html>"
        )
        text = _html_to_text(html)
        assert "visible" in text
        assert "alert" not in text
        assert "color: red" not in text

    def test_strips_noscript_and_svg(self):
        html = (
            "<body><noscript>Enable JS</noscript>"
            "<svg><text>icon</text></svg>"
            "<p>real content</p></body>"
        )
        text = _html_to_text(html)
        assert "real content" in text
        assert "Enable JS" not in text
        assert "icon" not in text

    def test_block_elements_produce_newlines(self):
        html = "<div>first</div><div>second</div>"
        text = _html_to_text(html)
        assert "first" in text
        assert "second" in text
        # Block elements should cause separation
        assert "first" != text  # not just the word

    def test_html_entities_decoded(self):
        html = "<p>5 &gt; 3 &amp; 2 &lt; 4</p>"
        text = _html_to_text(html)
        assert "5 > 3 & 2 < 4" in text

    def test_collapses_whitespace(self):
        html = "<p>  lots   of   spaces  </p>"
        text = _html_to_text(html)
        assert "lots of spaces" in text


# =========================================================================
# _check_url_safety
# =========================================================================


class TestUrlSafety:
    def test_rejects_ftp_scheme(self):
        result = _check_url_safety("ftp://example.com/file")
        assert result is not None
        assert "not allowed" in result

    def test_rejects_file_scheme(self):
        result = _check_url_safety("file:///etc/passwd")
        assert result is not None
        assert "not allowed" in result

    def test_rejects_javascript_scheme(self):
        result = _check_url_safety("javascript:alert(1)")
        assert result is not None
        assert "not allowed" in result

    def test_rejects_empty_hostname(self):
        result = _check_url_safety("http://")
        assert result is not None
        assert "could not parse hostname" in result

    @patch("swival.fetch.socket.getaddrinfo")
    def test_blocks_loopback(self, mock_dns):
        mock_dns.return_value = [
            (2, 1, 0, "", ("127.0.0.1", 0)),
        ]
        result = _check_url_safety("http://evil.com")
        assert result is not None
        assert "private/internal" in result
        assert "127.0.0.1" in result

    @patch("swival.fetch.socket.getaddrinfo")
    def test_blocks_private_10(self, mock_dns):
        mock_dns.return_value = [
            (2, 1, 0, "", ("10.0.0.1", 0)),
        ]
        result = _check_url_safety("http://internal.corp")
        assert result is not None
        assert "private/internal" in result

    @patch("swival.fetch.socket.getaddrinfo")
    def test_blocks_private_192(self, mock_dns):
        mock_dns.return_value = [
            (2, 1, 0, "", ("192.168.1.1", 0)),
        ]
        result = _check_url_safety("http://router.local")
        assert result is not None
        assert "private/internal" in result

    @patch("swival.fetch.socket.getaddrinfo")
    def test_blocks_private_172(self, mock_dns):
        mock_dns.return_value = [
            (2, 1, 0, "", ("172.16.0.1", 0)),
        ]
        result = _check_url_safety("http://internal.net")
        assert result is not None
        assert "private/internal" in result

    @patch("swival.fetch.socket.getaddrinfo")
    def test_blocks_link_local(self, mock_dns):
        mock_dns.return_value = [
            (2, 1, 0, "", ("169.254.169.254", 0)),
        ]
        result = _check_url_safety("http://metadata.internal")
        assert result is not None
        assert "private/internal" in result

    @patch("swival.fetch.socket.getaddrinfo")
    def test_blocks_ipv6_loopback_rebinding(self, mock_dns):
        mock_dns.return_value = [
            (10, 1, 0, "", ("::1", 0, 0, 0)),
        ]
        result = _check_url_safety("http://evil6.com")
        assert result is not None
        assert "private/internal" in result

    @patch("swival.fetch.socket.getaddrinfo")
    def test_allows_localhost(self, mock_dns):
        mock_dns.return_value = [
            (2, 1, 0, "", ("127.0.0.1", 0)),
        ]
        result = _check_url_safety("http://localhost:8765/")
        assert result is None

    @patch("swival.fetch.socket.getaddrinfo")
    def test_allows_localhost_ipv4(self, mock_dns):
        mock_dns.return_value = [
            (2, 1, 0, "", ("127.0.0.1", 0)),
        ]
        result = _check_url_safety("http://127.0.0.1:8080/")
        assert result is None

    @patch("swival.fetch.socket.getaddrinfo")
    def test_allows_localhost_ipv6(self, mock_dns):
        mock_dns.return_value = [
            (10, 1, 0, "", ("::1", 0, 0, 0)),
        ]
        result = _check_url_safety("http://[::1]:8080/")
        assert result is None

    @patch("swival.fetch.socket.getaddrinfo")
    def test_allows_public_address(self, mock_dns):
        mock_dns.return_value = [
            (2, 1, 0, "", ("93.184.216.34", 0)),
        ]
        result = _check_url_safety("http://example.com")
        assert result is None

    @patch("swival.fetch.socket.getaddrinfo")
    def test_dns_resolution_failure(self, mock_dns):
        import socket

        mock_dns.side_effect = socket.gaierror("Name or service not known")
        result = _check_url_safety("http://nonexistent.invalid")
        assert result is not None
        assert "could not resolve" in result


# =========================================================================
# fetch_url — format handling
# =========================================================================


def _make_response(
    body: bytes, content_type: str = "text/html; charset=utf-8", status: int = 200
):
    """Create a mock HTTP response."""
    resp = MagicMock()
    resp.read.return_value = body
    resp.headers = http.client.HTTPMessage()
    resp.headers["Content-Type"] = content_type
    resp.status = status
    return resp


class TestFetchUrlFormats:
    @patch("swival.fetch.socket.getaddrinfo")
    @patch("swival.fetch.urllib.request.build_opener")
    def test_raw_html(self, mock_opener_factory, mock_dns):
        mock_dns.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]
        resp = _make_response(b"<h1>Hello</h1>")
        opener = MagicMock()
        opener.open.return_value = resp
        mock_opener_factory.return_value = opener

        result = fetch_url("http://example.com", format="html")
        assert "<h1>Hello</h1>" in result

    @patch("swival.fetch.socket.getaddrinfo")
    @patch("swival.fetch.urllib.request.build_opener")
    def test_text_format(self, mock_opener_factory, mock_dns):
        mock_dns.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]
        html_body = b"<html><body><p>Hello</p><script>evil()</script></body></html>"
        resp = _make_response(html_body)
        opener = MagicMock()
        opener.open.return_value = resp
        mock_opener_factory.return_value = opener

        result = fetch_url("http://example.com", format="text")
        assert "Hello" in result
        assert "evil()" not in result

    @patch("swival.fetch.socket.getaddrinfo")
    @patch("swival.fetch.urllib.request.build_opener")
    def test_markdown_format(self, mock_opener_factory, mock_dns):
        mock_dns.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]
        html_body = b"<html><body><h1>Title</h1><p>Paragraph</p></body></html>"
        resp = _make_response(html_body)
        opener = MagicMock()
        opener.open.return_value = resp
        mock_opener_factory.return_value = opener

        result = fetch_url("http://example.com", format="markdown")
        # html-to-markdown should convert h1 to # Title
        assert "Title" in result
        assert "Paragraph" in result

    @patch("swival.fetch.socket.getaddrinfo")
    @patch("swival.fetch.urllib.request.build_opener")
    def test_markdown_strips_data_uri_images(self, mock_opener_factory, mock_dns):
        mock_dns.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]
        html_body = (
            b"<html><body>"
            b'<img src="data:image/png;base64,iVBORw0KGgoAAAANS=" alt="logo">'
            b'<img src="https://example.com/img.png" alt="normal">'
            b"</body></html>"
        )
        resp = _make_response(html_body)
        opener = MagicMock()
        opener.open.return_value = resp
        mock_opener_factory.return_value = opener

        result = fetch_url("http://example.com", format="markdown")
        # data URI should be stripped
        assert "base64" not in result
        assert "iVBORw0" not in result
        # alt text preserved
        assert "logo" in result
        # normal image preserved
        assert "https://example.com/img.png" in result
        assert "normal" in result

    @patch("swival.fetch.socket.getaddrinfo")
    @patch("swival.fetch.urllib.request.build_opener")
    def test_markdown_strips_data_uri_no_alt(self, mock_opener_factory, mock_dns):
        mock_dns.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]
        html_body = (
            b'<html><body><img src="data:image/svg+xml;base64,PHN2Zz4="></body></html>'
        )
        resp = _make_response(html_body)
        opener = MagicMock()
        opener.open.return_value = resp
        mock_opener_factory.return_value = opener

        result = fetch_url("http://example.com", format="markdown")
        assert "base64" not in result
        assert "PHN2Zz4" not in result

    def test_invalid_format(self):
        result = fetch_url("http://example.com", format="pdf")
        assert result.startswith("error:")
        assert "invalid format" in result


# =========================================================================
# fetch_url — URL validation
# =========================================================================


class TestFetchUrlValidation:
    def test_empty_url(self):
        result = fetch_url("")
        assert result.startswith("error:")

    def test_ftp_url(self):
        result = fetch_url("ftp://example.com/file")
        assert result.startswith("error:")
        assert "not allowed" in result

    def test_file_url(self):
        result = fetch_url("file:///etc/passwd")
        assert result.startswith("error:")
        assert "not allowed" in result

    def test_no_scheme(self):
        # Missing scheme — urlparse puts everything in path, hostname is None
        result = fetch_url("example.com")
        assert result.startswith("error:")


# =========================================================================
# fetch_url — timeout clamping
# =========================================================================


class TestTimeoutClamping:
    @patch("swival.fetch.socket.getaddrinfo")
    @patch("swival.fetch.urllib.request.build_opener")
    def test_timeout_clamped_low(self, mock_opener_factory, mock_dns):
        mock_dns.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]
        resp = _make_response(b"ok", "text/plain")
        opener = MagicMock()
        opener.open.return_value = resp
        mock_opener_factory.return_value = opener

        fetch_url("http://example.com", format="text", timeout=-5)
        # Should have been clamped to 1
        call_args = opener.open.call_args
        assert call_args[1]["timeout"] == 1 or call_args.kwargs.get("timeout") == 1

    @patch("swival.fetch.socket.getaddrinfo")
    @patch("swival.fetch.urllib.request.build_opener")
    def test_timeout_clamped_high(self, mock_opener_factory, mock_dns):
        mock_dns.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]
        resp = _make_response(b"ok", "text/plain")
        opener = MagicMock()
        opener.open.return_value = resp
        mock_opener_factory.return_value = opener

        fetch_url("http://example.com", format="text", timeout=999)
        call_args = opener.open.call_args
        assert call_args[1]["timeout"] == 120 or call_args.kwargs.get("timeout") == 120

    def test_timeout_string_returns_error(self):
        result = fetch_url("http://example.com", timeout="30")
        assert result.startswith("error:")
        assert "timeout must be a number" in result

    def test_timeout_none_returns_error(self):
        result = fetch_url("http://example.com", timeout=None)
        assert result.startswith("error:")
        assert "timeout must be a number" in result


# =========================================================================
# fetch_url — response close
# =========================================================================


class TestResponseClose:
    @patch("swival.fetch.socket.getaddrinfo")
    @patch("swival.fetch.urllib.request.build_opener")
    def test_response_closed_on_success(self, mock_opener_factory, mock_dns):
        mock_dns.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]
        resp = _make_response(b"<p>hello</p>")
        opener = MagicMock()
        opener.open.return_value = resp
        mock_opener_factory.return_value = opener

        fetch_url("http://example.com", format="text")
        resp.close.assert_called_once()

    @patch("swival.fetch.socket.getaddrinfo")
    @patch("swival.fetch.urllib.request.build_opener")
    def test_response_closed_on_binary_content_type(
        self, mock_opener_factory, mock_dns
    ):
        mock_dns.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]
        resp = _make_response(b"\x89PNG", "image/png")
        opener = MagicMock()
        opener.open.return_value = resp
        mock_opener_factory.return_value = opener

        fetch_url("http://example.com")
        resp.close.assert_called_once()

    @patch("swival.fetch.socket.getaddrinfo")
    @patch("swival.fetch.urllib.request.build_opener")
    def test_response_closed_on_too_large(self, mock_opener_factory, mock_dns):
        mock_dns.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]
        resp = _make_response(b"x" * (MAX_RESPONSE_SIZE + 1), "text/plain")
        opener = MagicMock()
        opener.open.return_value = resp
        mock_opener_factory.return_value = opener

        fetch_url("http://example.com", format="text")
        resp.close.assert_called_once()

    @patch("swival.fetch.socket.getaddrinfo")
    @patch("swival.fetch.urllib.request.build_opener")
    def test_response_closed_on_null_bytes(self, mock_opener_factory, mock_dns):
        mock_dns.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]
        resp = _make_response(b"text\x00binary", "text/html")
        opener = MagicMock()
        opener.open.return_value = resp
        mock_opener_factory.return_value = opener

        fetch_url("http://example.com")
        resp.close.assert_called_once()


# =========================================================================
# fetch_url — size limits
# =========================================================================


class TestSizeLimits:
    @patch("swival.fetch.socket.getaddrinfo")
    @patch("swival.fetch.urllib.request.build_opener")
    def test_response_too_large(self, mock_opener_factory, mock_dns):
        mock_dns.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]
        # Return data just over the limit
        resp = _make_response(b"x" * (MAX_RESPONSE_SIZE + 1), "text/plain")
        opener = MagicMock()
        opener.open.return_value = resp
        mock_opener_factory.return_value = opener

        result = fetch_url("http://example.com", format="text")
        assert result.startswith("error:")
        assert "too large" in result

    @patch("swival.fetch.socket.getaddrinfo")
    @patch("swival.fetch.urllib.request.build_opener")
    def test_output_truncation_no_base_dir(self, mock_opener_factory, mock_dns):
        """Without base_dir, large output is truncated inline."""
        mock_dns.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]
        big_text = "a" * (MAX_OUTPUT_BYTES + 1000)
        resp = _make_response(big_text.encode("utf-8"), "text/plain")
        opener = MagicMock()
        opener.open.return_value = resp
        mock_opener_factory.return_value = opener

        result = fetch_url("http://example.com", format="html")
        assert "content truncated" in result

    @patch("swival.fetch.socket.getaddrinfo")
    @patch("swival.fetch.urllib.request.build_opener")
    def test_large_output_saved_to_file(self, mock_opener_factory, mock_dns, tmp_path):
        """With base_dir, large output is saved to .swival/ for pagination."""
        mock_dns.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]
        big_text = "a" * (MAX_OUTPUT_BYTES + 1000)
        resp = _make_response(big_text.encode("utf-8"), "text/plain")
        opener = MagicMock()
        opener.open.return_value = resp
        mock_opener_factory.return_value = opener

        result = fetch_url("http://example.com", format="html", base_dir=str(tmp_path))
        assert "too large for context" in result
        assert ".swival/" in result
        assert "read_file" in result

    @patch("swival.fetch.socket.getaddrinfo")
    @patch("swival.fetch.urllib.request.build_opener")
    def test_saved_file_readable(self, mock_opener_factory, mock_dns, tmp_path):
        """Saved file contains full output and is readable via read_file."""
        from swival.tools import _read_file

        mock_dns.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]
        big_text = "line_" + "a" * (MAX_OUTPUT_BYTES + 1000)
        resp = _make_response(big_text.encode("utf-8"), "text/plain")
        opener = MagicMock()
        opener.open.return_value = resp
        mock_opener_factory.return_value = opener

        result = fetch_url("http://example.com", format="html", base_dir=str(tmp_path))
        # Extract the .swival/cmd_output_*.txt path from the result
        for word in result.split():
            if word.startswith(".swival/"):
                rel_path = word
                break
        else:
            pytest.fail(f"No .swival/ path found in result: {result}")

        content = _read_file(rel_path, str(tmp_path))
        assert "line_" in content

    @patch("swival.fetch.socket.getaddrinfo")
    @patch("swival.fetch.urllib.request.build_opener")
    def test_small_output_stays_inline(self, mock_opener_factory, mock_dns, tmp_path):
        """Output under the limit is returned inline even with base_dir."""
        mock_dns.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]
        resp = _make_response(b"<p>small</p>")
        opener = MagicMock()
        opener.open.return_value = resp
        mock_opener_factory.return_value = opener

        result = fetch_url("http://example.com", format="text", base_dir=str(tmp_path))
        assert "small" in result
        assert ".swival/" not in result


# =========================================================================
# fetch_url — binary detection
# =========================================================================


class TestBinaryDetection:
    @patch("swival.fetch.socket.getaddrinfo")
    @patch("swival.fetch.urllib.request.build_opener")
    def test_binary_content_type(self, mock_opener_factory, mock_dns):
        mock_dns.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]
        resp = _make_response(b"\x89PNG\r\n", "image/png")
        opener = MagicMock()
        opener.open.return_value = resp
        mock_opener_factory.return_value = opener

        result = fetch_url("http://example.com/image.png")
        assert result.startswith("error:")
        assert "binary content" in result
        assert "image/png" in result

    @patch("swival.fetch.socket.getaddrinfo")
    @patch("swival.fetch.urllib.request.build_opener")
    def test_binary_null_bytes(self, mock_opener_factory, mock_dns):
        mock_dns.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]
        data = b"some text\x00binary data"
        resp = _make_response(data, "text/html")
        opener = MagicMock()
        opener.open.return_value = resp
        mock_opener_factory.return_value = opener

        result = fetch_url("http://example.com/weird")
        assert result.startswith("error:")
        assert "binary content" in result


# =========================================================================
# fetch_url — HTTP errors
# =========================================================================


class TestHttpErrors:
    @patch("swival.fetch.socket.getaddrinfo")
    @patch("swival.fetch.urllib.request.build_opener")
    def test_404_error(self, mock_opener_factory, mock_dns):
        import urllib.error

        mock_dns.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]
        opener = MagicMock()
        opener.open.side_effect = urllib.error.HTTPError(
            "http://example.com/missing", 404, "Not Found", {}, BytesIO(b"")
        )
        mock_opener_factory.return_value = opener

        result = fetch_url("http://example.com/missing")
        assert result.startswith("error:")
        assert "404" in result

    @patch("swival.fetch.socket.getaddrinfo")
    @patch("swival.fetch.urllib.request.build_opener")
    def test_500_error(self, mock_opener_factory, mock_dns):
        import urllib.error

        mock_dns.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]
        opener = MagicMock()
        opener.open.side_effect = urllib.error.HTTPError(
            "http://example.com/broken", 500, "Internal Server Error", {}, BytesIO(b"")
        )
        mock_opener_factory.return_value = opener

        result = fetch_url("http://example.com/broken")
        assert result.startswith("error:")
        assert "500" in result

    @patch("swival.fetch.socket.getaddrinfo")
    @patch("swival.fetch.urllib.request.build_opener")
    def test_timeout_error(self, mock_opener_factory, mock_dns):
        mock_dns.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]
        opener = MagicMock()
        opener.open.side_effect = TimeoutError()
        mock_opener_factory.return_value = opener

        result = fetch_url("http://example.com/slow", timeout=5)
        assert result.startswith("error:")
        assert "timed out" in result


# =========================================================================
# fetch_url — encoding
# =========================================================================


class TestEncoding:
    @patch("swival.fetch.socket.getaddrinfo")
    @patch("swival.fetch.urllib.request.build_opener")
    def test_utf8_content(self, mock_opener_factory, mock_dns):
        mock_dns.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]
        body = "Ünïcödé tëxt".encode("utf-8")
        resp = _make_response(body, "text/html; charset=utf-8")
        opener = MagicMock()
        opener.open.return_value = resp
        mock_opener_factory.return_value = opener

        result = fetch_url("http://example.com", format="html")
        assert "Ünïcödé" in result

    @patch("swival.fetch.socket.getaddrinfo")
    @patch("swival.fetch.urllib.request.build_opener")
    def test_latin1_charset(self, mock_opener_factory, mock_dns):
        mock_dns.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]
        body = "café".encode("latin-1")
        resp = _make_response(body, "text/html; charset=iso-8859-1")
        opener = MagicMock()
        opener.open.return_value = resp
        mock_opener_factory.return_value = opener

        result = fetch_url("http://example.com", format="html")
        assert "café" in result

    @patch("swival.fetch.socket.getaddrinfo")
    @patch("swival.fetch.urllib.request.build_opener")
    def test_latin1_fallback(self, mock_opener_factory, mock_dns):
        mock_dns.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]
        # Bytes that are valid latin-1 but not valid utf-8
        body = b"\xe9\xe8\xea"  # é è ê in latin-1
        resp = _make_response(body, "text/html")
        opener = MagicMock()
        opener.open.return_value = resp
        mock_opener_factory.return_value = opener

        result = fetch_url("http://example.com", format="html")
        # Should not error out — falls back to latin-1
        assert not result.startswith("error:")


# =========================================================================
# fetch_url — redirect safety
# =========================================================================


class TestRedirectSafety:
    @patch("swival.fetch.socket.getaddrinfo")
    @patch("swival.fetch.urllib.request.build_opener")
    def test_redirect_to_private_blocked(self, mock_opener_factory, mock_dns):
        """Redirect to a private (non-loopback) address should be blocked."""

        def dns_side_effect(hostname, *args, **kwargs):
            if hostname == "public.com":
                return [(2, 1, 0, "", ("93.184.216.34", 0))]
            elif hostname == "10.0.0.1":
                return [(2, 1, 0, "", ("10.0.0.1", 0))]
            return [(2, 1, 0, "", ("93.184.216.34", 0))]

        mock_dns.side_effect = dns_side_effect

        opener = MagicMock()
        opener.open.side_effect = _RedirectError("http://10.0.0.1/secret", 302)
        mock_opener_factory.return_value = opener

        result = fetch_url("http://public.com")
        assert result.startswith("error:")
        assert "private/internal" in result

    @patch("swival.fetch.socket.getaddrinfo")
    @patch("swival.fetch.urllib.request.build_opener")
    def test_redirect_to_metadata_blocked(self, mock_opener_factory, mock_dns):
        """Redirect to cloud metadata endpoint should be blocked."""

        def dns_side_effect(hostname, *args, **kwargs):
            if hostname == "public.com":
                return [(2, 1, 0, "", ("93.184.216.34", 0))]
            elif hostname == "169.254.169.254":
                return [(2, 1, 0, "", ("169.254.169.254", 0))]
            return [(2, 1, 0, "", ("93.184.216.34", 0))]

        mock_dns.side_effect = dns_side_effect

        opener = MagicMock()
        opener.open.side_effect = _RedirectError(
            "http://169.254.169.254/latest/meta-data/", 302
        )
        mock_opener_factory.return_value = opener

        result = fetch_url("http://public.com")
        assert result.startswith("error:")
        assert "private/internal" in result

    @patch("swival.fetch.socket.getaddrinfo")
    @patch("swival.fetch.urllib.request.build_opener")
    def test_redirect_to_ftp_blocked(self, mock_opener_factory, mock_dns):
        """Redirect to non-HTTP scheme should be blocked."""
        mock_dns.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]

        opener = MagicMock()
        opener.open.side_effect = _RedirectError("ftp://evil.com/file", 302)
        mock_opener_factory.return_value = opener

        result = fetch_url("http://public.com")
        assert result.startswith("error:")
        assert "not allowed" in result

    @patch("swival.fetch.socket.getaddrinfo")
    @patch("swival.fetch.urllib.request.build_opener")
    def test_relative_redirect_allowed(self, mock_opener_factory, mock_dns):
        """Relative redirect on the same host should work."""
        mock_dns.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]

        call_count = [0]

        def open_side_effect(req, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise _RedirectError("/login", 302)
            return _make_response(b"<p>Login page</p>")

        opener = MagicMock()
        opener.open.side_effect = open_side_effect
        mock_opener_factory.return_value = opener

        result = fetch_url("http://example.com/dashboard", format="text")
        assert not result.startswith("error:")
        assert "Login page" in result

    @patch("swival.fetch.socket.getaddrinfo")
    @patch("swival.fetch.urllib.request.build_opener")
    def test_too_many_redirects(self, mock_opener_factory, mock_dns):
        """Exceeding redirect limit should return error."""
        mock_dns.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]

        opener = MagicMock()
        # Every request redirects
        opener.open.side_effect = _RedirectError("http://example.com/next", 302)
        mock_opener_factory.return_value = opener

        result = fetch_url("http://example.com/start")
        assert result.startswith("error:")
        assert "too many redirects" in result

    @patch("swival.fetch.socket.getaddrinfo")
    @patch("swival.fetch.urllib.request.build_opener")
    def test_redirect_chain_success(self, mock_opener_factory, mock_dns):
        """A chain of safe redirects should succeed."""
        mock_dns.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]

        call_count = [0]

        def open_side_effect(req, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 3:
                raise _RedirectError(f"http://example.com/step{call_count[0]}", 302)
            return _make_response(b"<p>final</p>")

        opener = MagicMock()
        opener.open.side_effect = open_side_effect
        mock_opener_factory.return_value = opener

        result = fetch_url("http://example.com/start", format="text")
        assert not result.startswith("error:")
        assert "final" in result


# =========================================================================
# dispatch wiring
# =========================================================================


class TestDispatchWiring:
    @patch("swival.fetch.socket.getaddrinfo")
    @patch("swival.fetch.urllib.request.build_opener")
    def test_dispatch_routes_fetch_url(self, mock_opener_factory, mock_dns, tmp_path):
        from swival.tools import dispatch

        mock_dns.return_value = [(2, 1, 0, "", ("93.184.216.34", 0))]
        resp = _make_response(b"<p>dispatched</p>")
        opener = MagicMock()
        opener.open.return_value = resp
        mock_opener_factory.return_value = opener

        result = dispatch(
            "fetch_url",
            {"url": "http://example.com", "format": "text"},
            str(tmp_path),
        )
        assert "dispatched" in result

    def test_fetch_url_in_tools_list(self):
        from swival.tools import TOOLS

        names = [t["function"]["name"] for t in TOOLS]
        assert "fetch_url" in names

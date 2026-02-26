"""Fetch URL tool — retrieves web page content as markdown, text, or raw HTML."""

import html
import html.parser
import ipaddress
import re
import socket
import urllib.parse
import urllib.request

MAX_RESPONSE_SIZE = 5 * 1024 * 1024  # 5 MB raw download cap
MAX_OUTPUT_BYTES = 50 * 1024  # 50 KB converted output cap (same as read_file)
MAX_REDIRECTS = 10

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/markdown,text/html,text/plain,application/json,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Block elements that should produce line breaks in text extraction
_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "br",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "tr",
        "blockquote",
        "pre",
        "hr",
        "dt",
        "dd",
        "section",
        "article",
        "header",
        "footer",
        "nav",
        "main",
        "table",
        "thead",
        "tbody",
        "tfoot",
        "figure",
        "figcaption",
    }
)

# Tags whose content should be skipped entirely
_SKIP_TAGS = frozenset({"script", "style", "noscript", "svg"})


class _RedirectError(Exception):
    def __init__(self, url: str, code: int):
        self.url = url
        self.code = code


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise _RedirectError(newurl, code)


class _TextExtractor(html.parser.HTMLParser):
    """Extract readable text from HTML, skipping script/style/svg/noscript."""

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS and self._skip_depth == 0:
            self._parts.append("\n")

    def handle_endtag(self, tag: str):
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_TAGS and self._skip_depth == 0:
            self._parts.append("\n")

    def handle_data(self, data: str):
        if self._skip_depth == 0:
            self._parts.append(data)

    def handle_entityref(self, name: str):
        if self._skip_depth == 0:
            self._parts.append(html.unescape(f"&{name};"))

    def handle_charref(self, name: str):
        if self._skip_depth == 0:
            self._parts.append(html.unescape(f"&#{name};"))

    def get_text(self) -> str:
        text = "".join(self._parts)
        # Collapse runs of whitespace within lines
        text = re.sub(r"[^\S\n]+", " ", text)
        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _html_to_text(body: str) -> str:
    """Convert HTML to plain text."""
    parser = _TextExtractor()
    parser.feed(body)
    return parser.get_text()


def _check_url_safety(url: str) -> str | None:
    """Return an error string if the URL has a bad scheme or targets a private address, else None."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return (
            f"error: url scheme {parsed.scheme!r} is not allowed, must be http or https"
        )
    hostname = parsed.hostname
    if not hostname:
        return "error: could not parse hostname from url"
    try:
        infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror as e:
        return f"error: could not resolve hostname {hostname!r}: {e}"
    for family, _, _, _, sockaddr in infos:
        addr = ipaddress.ip_address(sockaddr[0])
        if (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_reserved
        ):
            return f"error: url resolves to private/internal address ({addr}), blocked for security"
    return None


def _decode_response(data: bytes, content_type: str | None) -> str:
    """Decode response bytes to string, using charset from Content-Type or falling back."""
    # Try charset from Content-Type header
    charset = None
    if content_type:
        for part in content_type.split(";"):
            part = part.strip()
            if part.lower().startswith("charset="):
                charset = part.split("=", 1)[1].strip().strip("\"'")
                break

    # Try specified charset, then UTF-8, then latin-1 as last resort
    for encoding in [charset, "utf-8", "latin-1"]:
        if encoding is None:
            continue
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue

    # latin-1 never fails, so we should never reach here
    return data.decode("latin-1")


def fetch_url(
    url: str, format: str = "markdown", timeout: int = 30, base_dir: str | None = None
) -> str:
    """Fetch a URL and return its content as markdown, text, or raw HTML.

    Returns content string on success, "error: ..." on failure.
    When base_dir is provided and output exceeds the inline limit,
    saves to .swival/ for pagination via read_file.
    """
    # Validate format
    if format not in ("markdown", "text", "html"):
        return (
            f"error: invalid format {format!r}, must be 'markdown', 'text', or 'html'"
        )

    # Validate URL is a string
    if not url or not isinstance(url, str):
        return "error: url must be a non-empty string"

    # Validate and clamp timeout
    if not isinstance(timeout, (int, float)):
        return f"error: timeout must be a number, got {type(timeout).__name__}"
    timeout = max(1, min(int(timeout), 120))

    current_url = url
    opener = urllib.request.build_opener(_NoRedirectHandler)

    for _ in range(MAX_REDIRECTS + 1):
        # Safety check every URL in the redirect chain
        err = _check_url_safety(current_url)
        if err:
            return err

        req = urllib.request.Request(current_url, headers=HEADERS)
        try:
            resp = opener.open(req, timeout=timeout)
            break  # success, no redirect
        except _RedirectError as r:
            current_url = urllib.parse.urljoin(current_url, r.url)
        except urllib.error.HTTPError as e:
            return f"error: HTTP {e.code} — {e.reason}"
        except urllib.error.URLError as e:
            reason = str(e.reason)
            if "timed out" in reason.lower() or "timeout" in reason.lower():
                return f"error: request timed out after {timeout} seconds"
            return f"error: could not connect to {urllib.parse.urlparse(current_url).hostname}: {reason}"
        except TimeoutError:
            return f"error: request timed out after {timeout} seconds"
        except OSError as e:
            return f"error: could not connect to {urllib.parse.urlparse(current_url).hostname}: {e}"
    else:
        return f"error: too many redirects (limit is {MAX_REDIRECTS})"

    # Read and process response — always close when done
    try:
        # Check content type for binary
        content_type = resp.headers.get("Content-Type", "")
        mime = content_type.split(";")[0].strip().lower()
        if (
            mime
            and not mime.startswith("text/")
            and mime
            not in (
                "application/json",
                "application/xml",
                "application/xhtml+xml",
                "application/javascript",
                "application/ecmascript",
                "application/rss+xml",
                "application/atom+xml",
            )
        ):
            return (
                f"error: binary content (content-type: {mime}), cannot display as text"
            )

        # Read response body with size limit
        try:
            data = resp.read(MAX_RESPONSE_SIZE + 1)
        except TimeoutError:
            return f"error: request timed out after {timeout} seconds"
        except OSError as e:
            return f"error: failed to read response: {e}"

        if len(data) > MAX_RESPONSE_SIZE:
            return f"error: response too large ({len(data)} bytes, limit is 5MB)"

        # Check for binary content via null bytes
        if b"\x00" in data[:8192]:
            return "error: binary content detected (null bytes found), cannot display as text"

        # Decode
        body = _decode_response(data, content_type)
    finally:
        resp.close()

    # Convert to requested format
    if format == "html":
        output = body
    elif format == "text":
        output = _html_to_text(body)
    else:  # markdown
        if mime == "text/markdown":
            output = body
        else:
            try:
                from html_to_markdown import convert

                output = convert(body)
            except Exception as e:
                return f"error: failed to convert HTML to markdown: {e}"

    # Save large output to file for pagination, or truncate inline
    encoded = output.encode("utf-8")
    if len(encoded) > MAX_OUTPUT_BYTES and base_dir:
        from .tools import _save_large_output

        return _save_large_output(output, base_dir)
    elif len(encoded) > MAX_OUTPUT_BYTES:
        total = len(encoded)
        truncated = encoded[:MAX_OUTPUT_BYTES].decode("utf-8", errors="ignore")
        output = (
            truncated
            + f"\n[content truncated at {MAX_OUTPUT_BYTES} bytes, total was {total} bytes]"
        )

    return output

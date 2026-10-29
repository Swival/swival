"""Built-in web search through free search services that need no account or key.

Each search goes to the next service in a ring.
A service that fails or rate limits is skipped for a while, and the answer
is cut down to the passages that best match the query.

Service health, the result cache and the requests in flight are shared by
the whole process, because every session in it uses the same IP address.
What each caller allows travels in a `WebSearchPolicy` and is checked on
every search.
"""

from __future__ import annotations

import asyncio
import collections
import concurrent.futures
import json
import logging
import random
import re
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

from .repair import _FIELD_ALIASES
from .report import ConfigError

logger = logging.getLogger(__name__)

PROVIDERS = ("parallel", "keenable", "exa", "tavily")

ENDPOINTS = {
    "parallel": "https://search.parallel.ai/mcp",
    "keenable": "https://api.keenable.ai/v1/search/public",
    "exa": "https://mcp.exa.ai/mcp",
    "tavily": "https://api.tavily.com/search",
}

DEADLINE = 20.0
REQUEST_TIMEOUT = 10.0
FILTER_TIMEOUT = 30.0
# Lets the engine report its own timeout before the caller gives up on it.
WAIT_GRACE = 0.5
MAX_BODY_BYTES = 1024 * 1024
MAX_QUERY_CHARS = 400
CACHE_TTL = 15 * 60
CACHE_SIZE = 128
MAX_IN_FLIGHT = 2
CALL_WINDOW = 10 * 60
MAX_CALLS_PER_WINDOW = 30
DECOY_THRESHOLD = 0.3
CUTOFF_RATIO = 0.5
VENDOR_RESULTS = 8
DNS_THREADS = 4

MAX_RESULTS = 5
MAX_CHARS = 6000
SMALL_CONTEXT = 16 * 1024
SMALL_MAX_RESULTS = 3
SMALL_MAX_CHARS = 3000
PASSAGES_PER_RESULT = 2
PASSAGE_CHARS = 500
MIN_PASSAGE_CHARS = 80
LEAD_CHARS = 300
MAX_TITLE_CHARS = 200
MAX_URL_CHARS = 2000
# Services send up to about 11 KB per result, so anything longer is junk.
MAX_HIT_CHARS = 20_000

THROTTLE_COOLDOWN = 60.0
MAX_THROTTLE_COOLDOWN = 15 * 60.0
MAX_RETRY_AFTER = 60 * 60.0
COOLDOWNS = {
    "refused": 30 * 60.0,
    "drift": 30 * 60.0,
    "malformed": 5 * 60.0,
    "vendor_error": 60.0,
}

MCP_PROTOCOL = "2025-06-18"
MCP_ACCEPT = "application/json, text/event-stream"

DISABLED = "error: web_search is disabled in this session"
BAD_QUERY = "error: web_search needs a non-empty 'query' string"
CREDENTIAL = (
    "error: the query looks like it contains a credential; remove it and search again."
)
FILTER_BLOCKED = "error: web search blocked by llm_filter"
CANCELLED = "error: web search was cancelled"
UNAVAILABLE = (
    "error: web search is temporarily unavailable (all search providers are "
    "rate limiting this machine). Do not retry or rephrase now. Use fetch_url "
    "on a URL you already know, or continue without web results."
)
FAILING = (
    "error: web search is temporarily unavailable (the search providers are "
    "failing or unreachable). Do not retry or rephrase now. Use fetch_url on a "
    "URL you already know, or continue without web results."
)
TIMED_OUT = (
    "error: web search timed out. Do not retry or rephrase now. Use fetch_url "
    "on a URL you already know, or continue without web results."
)
CEILING = (
    "error: web search limit reached for now. Do not retry or rephrase now; "
    "continue with the results you have."
)
REJECTED = "error: every search provider rejected the request"
FOOTER = "Read a full page with fetch_url."

_QUOTA_RE = re.compile(r"rate[ _-]?limit|quota|too many requests|free[ _-]tier", re.I)
_INVALID_PARAMS_RE = re.compile(r"invalid (?:params|parameters|arguments?)", re.I)
_QUERY_ALIASES = ("query", *(k for k, v in _FIELD_ALIASES.items() if v == ("query",)))


@dataclass(frozen=True)
class WebSearchPolicy:
    """What one caller allows a search to do.

    The engine never keeps it, so sessions with different rules in the same
    process never see each other's settings.
    """

    providers: tuple[str, ...] = PROVIDERS
    llm_filter: str | None = None
    origin_provider: str | None = None
    origin_model: str | None = None
    secret_patterns: tuple = ()
    max_output_bytes: int = 50 * 1024
    context_length: int | None = None

    def for_runtime(self, *, provider, model, context_length) -> "WebSearchPolicy":
        """Follow the model the loop runs on now, since /model can change it."""
        return replace(
            self,
            origin_provider=provider,
            origin_model=model,
            context_length=context_length,
        )


def validate_providers(value, source: str = "web_search_providers") -> tuple:
    """Check a provider list from config or the library and return it as a tuple."""
    if not isinstance(value, (list, tuple)) or not value:
        raise ConfigError(f"{source} must be a non-empty list of provider names")
    for i, name in enumerate(value):
        if not isinstance(name, str):
            raise ConfigError(
                f"{source}[{i}]: expected string, got {type(name).__name__}"
            )
        if name not in PROVIDERS:
            raise ConfigError(
                f"{source}[{i}]: unknown provider {name!r}; "
                f"known providers: {', '.join(PROVIDERS)}"
            )
    return tuple(dict.fromkeys(value))


def make_policy(
    *,
    providers=None,
    llm_filter: str | None = None,
    secret_patterns: list | None = None,
    max_output_kb: int = 50,
) -> WebSearchPolicy:
    """Build a policy, so bad settings fail at startup rather than on a search."""
    patterns = ()
    if secret_patterns:
        from .secrets import build_token_patterns

        patterns = tuple(build_token_patterns(secret_patterns))
    return WebSearchPolicy(
        providers=PROVIDERS if providers is None else validate_providers(providers),
        llm_filter=llm_filter or None,
        secret_patterns=patterns,
        max_output_bytes=max_output_kb * 1024,
    )


@dataclass(frozen=True)
class Hit:
    title: str
    url: str
    text: str
    published: str = ""


class VendorError(Exception):
    """A service answer that isn't a usable list of results.

    ``kind`` is one of throttled, refused, vendor_error, drift, malformed.
    """

    def __init__(self, kind: str, detail: str = "", retry_after: float | None = None):
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind = kind
        self.detail = detail
        self.retry_after = retry_after


@dataclass
class VendorResponse:
    hits: list
    # Set when a successful answer says the quota is used up.
    cooldown: float | None = None


def query_text(value) -> str | None:
    """The query as one string, or None when it isn't text.

    Small models sometimes send a list of words, so those are joined.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
        return " ".join(value)
    return None


def normalize_query(query: str) -> str:
    """Collapse whitespace and cut the query; case is kept on purpose."""
    return " ".join(query.split())[:MAX_QUERY_CHARS].strip()


def query_from_args(args) -> str:
    """The query to show for raw arguments, which may use an alias."""
    if not isinstance(args, dict):
        return ""
    for key in _QUERY_ALIASES:
        text = query_text(args.get(key))
        if text is not None:
            return normalize_query(text)
    return ""


# Every repeat below has an upper bound: service text can't be trusted, and
# an unbounded repeat can make broken markup take quadratic time.
_ANSI_RE = re.compile(
    r"\x1b\[[0-?]{0,32}[ -/]{0,8}[@-~]|\x1b\][^\x07\x1b]{0,256}(?:\x07|\x1b\\)|\x1b."
)
_CONTROL_RE = re.compile(
    r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069\ufeff\ud800-\udfff]"
)
_LINK = (
    r"\[([^\]\n]{0,300})\]\((?:[^()\s]|\([^()\s]{0,100}\)){0,2000}"
    r"(?:\s+\"[^\"\n]{0,200}\")?\)"
)
_IMAGE_RE = re.compile("!" + _LINK)
_LINK_RE = re.compile(_LINK)
# No part can contain a colon, so a match never runs into the next "data:".
_DATA_URI_RE = re.compile(
    r"\bdata:(?:[\w!#$&^.+-]+/[\w!#$&^.+-]+)?(?:;[^,\s():;]*)*,"
    r"(?:[^\s()]|\([^\s()]*\))*",
    re.I,
)
# 1,000 characters of URL or base64 text in a row are a payload or a tracking
# link, never prose.
# Code is full of brackets and commas that break such runs, and a 4096-bit
# RSA public key still fits under the limit.
_RUN_CHARS = r"A-Za-z0-9+/=%._~:?&#@!-"
_LONG_RUN_RE = re.compile(rf"(?<![{_RUN_CHARS}])[{_RUN_CHARS}]{{1000,}}")
_ANGLE_URL_RE = re.compile(r"<(https?://[^>\s]{1,2000})>")
# Services cut excerpts anywhere, even in the middle of a link.
_CUT_LINK_RE = re.compile(r"\[([^\]\n]{0,300})\]\([^)\s]{0,2000}$")
_NAV_LINE_RE = re.compile(
    r"^[\s*+\-|]*(?:" + _LINK.replace("([^", "(?:[^", 1) + r"[\s*+\-|,\u00b7]*)+$"
)
_ELLIPSIS_LINE_RE = re.compile(r"^\s*(?:\.\.\.|\u2026)\s*$")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|(?<=[\u3002\uff01\uff1f])")
_WORD_RE = re.compile(r"\w[\w.\-]*")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_SEPARATOR_RE = re.compile(r"[._\-]+")
_STOPWORDS = frozenset(
    "a an and are as at be by do does for from how in is it of on or the "
    "this that to vs what when why with".split()
)


def clean_line(text: str) -> str:
    """One line of plain text: no escape codes, control characters or line breaks."""
    text = _ANSI_RE.sub("", text)
    text = _CONTROL_RE.sub(" ", text.replace("\n", " ").replace("\t", " "))
    return " ".join(text.split())


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (
        0x3040 <= o <= 0x30FF
        or 0x3400 <= o <= 0x4DBF
        or 0x4E00 <= o <= 0x9FFF
        or 0xAC00 <= o <= 0xD7AF
        or 0xF900 <= o <= 0xFAFF
        or 0x20000 <= o <= 0x2FA1F
    )


def _word_tokens(word: str, out: list) -> None:
    word = word.strip("._-")
    if not word:
        return
    whole = word.casefold()
    if len(whole) >= 2:
        out.append(whole)
    for piece in _SEPARATOR_RE.split(word):
        if not piece:
            continue
        folded = piece.casefold()
        if folded != whole and len(folded) >= 2:
            out.append(folded)
        humps = _CAMEL_RE.findall(piece) if piece.isascii() else []
        if len(humps) > 1:
            out.extend(h.casefold() for h in humps if len(h) >= 2)


def tokenize(text: str) -> list[str]:
    """Search words for BM25 and the decoy check.

    Identifiers are kept whole and also split on ``_``, ``.``, ``-`` and
    camelCase, so ``follow_redirects`` matches "follow redirects".
    Chinese, Japanese and Korean text has no spaces, so it is cut into
    pairs of characters.
    """
    tokens: list[str] = []
    for match in _WORD_RE.finditer(text):
        word = match.group()
        if word.isascii():
            _word_tokens(word, tokens)
            continue
        start = 0
        for i in range(1, len(word) + 1):
            if i < len(word) and _is_cjk(word[i]) == _is_cjk(word[start]):
                continue
            run = word[start:i]
            if _is_cjk(run[0]):
                if len(run) == 1:
                    tokens.append(run)
                else:
                    tokens.extend(run[j : j + 2] for j in range(len(run) - 1))
            else:
                _word_tokens(run, tokens)
            start = i
    return tokens


def coverage(query: str, hits: list) -> float:
    """Share of the query's words found in the top results.

    Throttled or poisoned answers often look like real results about
    something else, and this catches most of them.
    """
    terms = set(tokenize(query))
    terms = {t for t in terms if t not in _STOPWORDS} or terms
    if not terms:
        return 1.0
    seen: set[str] = set()
    for hit in hits[:MAX_RESULTS]:
        seen.update(tokenize(f"{hit.title} {hit.url} {hit.text}"))
        if terms <= seen:
            return 1.0
    return len(terms & seen) / len(terms)


def _normalize_text(text: str) -> list[str]:
    """The paragraphs of a service excerpt, as clean single lines."""
    text = _ANSI_RE.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))
    # Long payloads go first, so the markup around them gets short enough
    # for the patterns below.
    text = _LONG_RUN_RE.sub("[omitted]", text)
    text = _DATA_URI_RE.sub("", text)
    text = _IMAGE_RE.sub("", text)
    paragraphs: list[str] = []
    current: list[str] = []

    def flush():
        if current:
            line = clean_line(" ".join(current))
            if line:
                paragraphs.append(line)
            current.clear()

    for raw in text.split("\n"):
        line = raw.strip()
        if not line or _ELLIPSIS_LINE_RE.match(line) or line.startswith("```"):
            flush()
            continue
        if _NAV_LINE_RE.match(line):
            continue
        line = _LINK_RE.sub(r"\1", line)
        line = _CUT_LINK_RE.sub(r"\1", line)
        line = _ANGLE_URL_RE.sub(r"\1", line)
        line = re.sub(r"^(?:#{1,6}\s+|>\s?)", "", line)
        line = line.replace("**", "")
        if line.strip():
            current.append(line)
    flush()
    return paragraphs


def _split_long(paragraph: str) -> list[str]:
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_RE.split(paragraph):
        sentence = sentence.strip()
        if not sentence:
            continue
        # A sentence too short to keep on its own sticks to the next one,
        # but only if that one has to be cut anyway.
        too_long = len(current) + 1 + len(sentence) > PASSAGE_CHARS
        glue = len(current) < MIN_PASSAGE_CHARS and len(sentence) > PASSAGE_CHARS
        if current and too_long and not glue:
            pieces.append(current)
            current = ""
        current = f"{current} {sentence}" if current else sentence
        while len(current) > PASSAGE_CHARS:
            cut = current.rfind(" ", 0, PASSAGE_CHARS)
            if cut < PASSAGE_CHARS // 2:
                cut = PASSAGE_CHARS
            pieces.append(current[:cut].strip())
            current = current[cut:].strip()
    if current:
        pieces.append(current)
    return pieces


def split_passages(text: str) -> list[str]:
    """Cut an excerpt into passages of about PASSAGE_CHARS characters.

    Short paragraphs are joined with their neighbors, long ones are cut at
    sentence ends, and passages under MIN_PASSAGE_CHARS are dropped.
    """
    passages: list[str] = []
    current = ""
    for paragraph in _normalize_text(text):
        if len(paragraph) > PASSAGE_CHARS:
            if current:
                passages.append(current)
                current = ""
            passages.extend(_split_long(paragraph))
            continue
        if current and len(current) + 1 + len(paragraph) > PASSAGE_CHARS:
            passages.append(current)
            current = ""
        current = f"{current} {paragraph}" if current else paragraph
    if current:
        passages.append(current)
    return [p for p in passages if len(p) >= MIN_PASSAGE_CHARS]


def _lead(passages: list[str], text: str) -> list[str]:
    if passages:
        lead = passages[0]
    else:
        lead = " ".join(_normalize_text(text))
    if not lead:
        return []
    if len(lead) > LEAD_CHARS:
        cut = lead.rfind(" ", 0, LEAD_CHARS)
        lead = lead[: cut if cut > LEAD_CHARS // 2 else LEAD_CHARS].rstrip() + " ..."
    return [lead]


def select_passages(query: str, hits: list) -> list[list[str]]:
    """Pick up to two passages per hit, the ones that best match the query.

    Passages from all hits are scored together with BM25, and only those
    scoring at least half as well as the best one are kept.
    Hits keep the service's order, which knows more than BM25 over excerpts.
    """
    from rank_bm25 import BM25Plus

    per_hit = [split_passages(hit.text) for hit in hits]
    query_terms = list(dict.fromkeys(tokenize(query)))
    corpus = []
    for i, passages in enumerate(per_hit):
        for j, passage in enumerate(passages):
            tokens = tokenize(passage)
            if tokens:
                corpus.append((i, j, tokens))

    chosen: dict[int, list[tuple[float, int]]] = {}
    degenerate = True
    terms = set(query_terms)
    overlapping = [k for k, (_, _, tokens) in enumerate(corpus) if terms & set(tokens)]
    if query_terms and overlapping:
        # With the default delta, every passage gets points for every query
        # word, matched or not, which lets weak matches pass the cutoff.
        scores = BM25Plus([tokens for _, _, tokens in corpus], delta=0).get_scores(
            query_terms
        )
        best = max(scores[k] for k in overlapping)
        if best > 0:
            degenerate = False
            cutoff = best * CUTOFF_RATIO
            for k in overlapping:
                if scores[k] >= cutoff:
                    i, j, _ = corpus[k]
                    chosen.setdefault(i, []).append((float(scores[k]), j))

    selected = []
    for i, hit in enumerate(hits):
        picks = chosen.get(i)
        if degenerate or not picks:
            selected.append(_lead(per_hit[i], hit.text))
            continue
        top = sorted(picks, key=lambda p: (-p[0], p[1]))[:PASSAGES_PER_RESULT]
        selected.append([per_hit[i][j] for _, j in sorted(top, key=lambda p: p[1])])
    return selected


def _untrusted_header(origin: str) -> str:
    from .tools import _untrusted_header as header

    return header("web_search", origin)


def truncate_utf8(text: str, max_bytes: int) -> str:
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    marker = "\n[truncated]"
    room = max(0, max_bytes - len(marker))
    return data[:room].decode("utf-8", errors="ignore") + marker[: max(0, max_bytes)]


def format_results(
    query: str,
    hits: list,
    *,
    origin: str,
    max_bytes: int,
    context_length: int | None = None,
) -> str:
    """Render hits as the numbered, untrusted-labeled text the model sees.

    Every result that fits keeps its title and URL even when its passages
    had to go, since they cost little and give the model something to fetch.
    """
    if context_length is not None and context_length <= SMALL_CONTEXT:
        max_results, max_chars = SMALL_MAX_RESULTS, SMALL_MAX_CHARS
    else:
        max_results, max_chars = MAX_RESULTS, MAX_CHARS
    hits = hits[:max_results]
    passages = select_passages(query, hits)
    header = _untrusted_header(origin)
    footer = "\n\n" + FOOTER
    size = len(header.encode("utf-8")) + len(footer.encode("utf-8"))
    blocks: list[str] = []
    chars = 0
    out_of_chars = False
    for n, (hit, picks) in enumerate(zip(hits, passages), 1):
        head = f"{n}. {hit.title}"
        if hit.published:
            head += f" ({hit.published})"
        head += f"\n   {hit.url}"
        kept = []
        for passage in picks:
            if out_of_chars or chars + len(passage) > max_chars:
                out_of_chars = True
                break
            kept.append(passage)
        while True:
            block = head + "".join(f"\n   {p}" for p in kept)
            cost = len(block.encode("utf-8")) + (2 if blocks else 0)
            if size + cost <= max_bytes or not kept:
                break
            kept.pop()
        if size + cost > max_bytes:
            break
        blocks.append(block)
        size += cost
        chars += sum(len(p) for p in kept)
    if not blocks:
        first = hits[0]
        return truncate_utf8(header + f"1. {first.title}\n   {first.url}", max_bytes)
    return header + "\n\n".join(blocks) + footer


def compact_result(args, content: str) -> str:
    """A web_search result without its passages, for compacted history.

    Titles and URLs come from the services, so they stay under a rebuilt
    untrusted header; only the first line, the model's own query, is outside.
    """
    from .tools import UNTRUSTED_MARKER

    query = json.dumps(query_from_args(args) or "?", ensure_ascii=False)
    if not content.startswith(UNTRUSTED_MARKER):
        return f"[web_search: {query}, {len(content)} chars, compacted]"
    header, _, body = content.partition("\n\n")
    origin = ""
    for line in header.splitlines():
        if line.startswith("origin: "):
            origin = line[len("origin: ") :]
    results = re.findall(r"(?m)^(\d+\. .*)\n   (https?://\S+)$", body)
    lines = "\n\n".join(f"{title}\n   {url}" for title, url in results)
    first = f"[web_search: {query}, {len(results)} results, passages compacted]"
    return f"{first}\n{_untrusted_header(origin)}{lines}"


def _detail(text: str) -> str:
    return clean_line(text)[:300]


def _sample(body: bytes) -> str:
    return _detail(body[:2000].decode("utf-8", errors="replace"))


def _parse_reset(value: str, now: float) -> float | None:
    """Seconds until a Retry-After or x-ratelimit-reset time.

    Services send seconds to wait, epoch seconds, HTTP dates or ISO 8601
    timestamps (Keenable).
    """
    value = value.strip()
    try:
        number = float(value)
    except ValueError:
        number = None
    if number is not None:
        if number > 1e12:
            number /= 1000
        if number > 1e9:
            return max(0.0, number - now)
        return max(0.0, number)
    try:
        when = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, when.timestamp() - now)


def _retry_after(headers, names=("retry-after", "x-ratelimit-reset")) -> float | None:
    now = time.time()
    for name in names:
        value = headers.get(name)
        seconds = _parse_reset(value, now) if value else None
        if seconds is not None:
            return min(seconds, MAX_RETRY_AFTER)
    return None


def _quota_cooldown(headers) -> float | None:
    """How long to skip a service whose successful answer used up its quota."""
    if (headers.get("x-ratelimit-remaining") or "").strip() != "0":
        return None
    seconds = _retry_after(headers, ("x-ratelimit-reset",))
    return THROTTLE_COOLDOWN if seconds is None else seconds


def _check_status(status: int, headers, body: bytes) -> None:
    if 200 <= status < 300:
        return
    detail = _sample(body)
    if status in (429, 432):
        raise VendorError("throttled", detail, _retry_after(headers))
    if status in (401, 402, 403):
        raise VendorError("refused", f"HTTP {status}: {detail}")
    if 300 <= status < 400:
        raise VendorError("vendor_error", f"unexpected redirect (HTTP {status})")
    if status >= 500:
        raise VendorError("vendor_error", f"HTTP {status}")
    if _QUOTA_RE.search(detail):
        raise VendorError("throttled", detail, _retry_after(headers))
    raise VendorError("drift", f"HTTP {status}: {detail}")


def _envelope_error(text: str, *, drift: bool = False) -> VendorError:
    """Classify an error that came back inside a successful HTTP answer.

    Only the wording tells whether it is a rate limit.
    """
    if _QUOTA_RE.search(text):
        return VendorError("throttled", _detail(text))
    if drift or _INVALID_PARAMS_RE.search(text):
        return VendorError("drift", _detail(text))
    return VendorError("vendor_error", _detail(text))


def _rpc_error(error) -> VendorError:
    if not isinstance(error, dict):
        return VendorError("vendor_error", _detail(str(error)))
    message = str(error.get("message") or "")
    data = error.get("data")
    text = f"{message} {data}" if data else message
    return _envelope_error(text, drift=error.get("code") == -32602)


def _split_lines(text: str) -> list[str]:
    # str.splitlines() also splits on characters that can appear in CJK text.
    return re.split(r"\r\n|\r|\n", text)


def _sse_messages(body: bytes, content_type: str) -> list:
    match = re.search(r"charset=([\w-]+)", content_type, re.I)
    try:
        text = body.decode(match.group(1) if match else "utf-8", errors="replace")
    except LookupError:
        text = body.decode("utf-8", errors="replace")
    messages = []
    data: list[str] = []
    for line in [*_split_lines(text), ""]:
        if not line:
            if data:
                try:
                    messages.append(json.loads("\n".join(data)))
                except ValueError:
                    pass
                data = []
            continue
        if line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    return messages


def _rpc_message(headers, body: bytes) -> dict:
    """The JSON-RPC answer in a JSON or event-stream body.

    Servers can send notifications first, so the first message with a result
    or an error wins.
    """
    content_type = headers.get("content-type") or ""
    if "text/event-stream" in content_type:
        messages = _sse_messages(body, content_type)
    else:
        try:
            parsed = json.loads(body)
        except ValueError:
            raise VendorError("malformed", _sample(body)) from None
        messages = parsed if isinstance(parsed, list) else [parsed]
    for message in messages:
        if isinstance(message, dict) and ("result" in message or "error" in message):
            return message
    raise VendorError("malformed", _sample(body))


def _rpc_response(status: int, headers, body: bytes) -> dict:
    _check_status(status, headers, body)
    message = _rpc_message(headers, body)
    if "error" in message:
        raise _rpc_error(message["error"])
    return message


def _mcp_headers(session: str | None = None) -> dict:
    headers = {"Accept": MCP_ACCEPT}
    if session:
        headers["Mcp-Session-Id"] = session
        headers["MCP-Protocol-Version"] = MCP_PROTOCOL
    return headers


def _mcp_text(result: dict) -> str:
    parts = []
    for item in result.get("content") or []:
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            parts.append(item["text"])
    return "\n".join(parts)


def _published(value) -> str:
    if not isinstance(value, str):
        return ""
    match = re.match(r"\s*(\d{4}-\d{2}-\d{2})", value)
    return match.group(1) if match else ""


def make_hit(title, url, text, published="") -> Hit | None:
    """A cleaned hit, or None when the URL isn't plain http(s)."""
    if not isinstance(url, str):
        return None
    url = url.strip()
    if len(url) > MAX_URL_CHARS or any(
        ord(c) <= 0x20 or 0x7F <= ord(c) <= 0x9F or 0xD800 <= ord(c) <= 0xDFFF
        for c in url
    ):
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None
    title = clean_line(title) if isinstance(title, str) else ""
    if len(title) > MAX_TITLE_CHARS:
        title = title[: MAX_TITLE_CHARS - 3].rstrip() + "..."
    return Hit(
        title=title or parts.netloc,
        url=url,
        text=text[:MAX_HIT_CHARS] if isinstance(text, str) else "",
        published=_published(published),
    )


def _rows(results: list, text_of, date_key: str | None = None) -> list:
    hits = []
    for r in results:
        if isinstance(r, dict):
            hit = make_hit(r.get("title"), r.get("url"), text_of(r), r.get(date_key))
            if hit is not None:
                hits.append(hit)
    return hits


def _parse_results(data, text_of, date_key: str | None = None) -> list:
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        raise _shape_error(data)
    return _rows(results, text_of, date_key)


def _excerpts(row: dict) -> str:
    excerpts = row.get("excerpts")
    if not isinstance(excerpts, list):
        return ""
    return "\n\n".join(e for e in excerpts if isinstance(e, str))


def parse_parallel(result: dict) -> list:
    data = result.get("structuredContent")
    if not isinstance(data, dict):
        text = _mcp_text(result)
        try:
            data = json.loads(text)
        except ValueError:
            raise VendorError("malformed", _detail(text)) from None
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        raise VendorError("malformed", "no results list")
    return _rows(results, _excerpts, "publish_date")


_EXA_FIELD_RE = re.compile(
    r"^(Title|URL|Published|Author|Highlights|Text|Summary):\s?(.*)$"
)


def parse_exa(text: str) -> list:
    """Parse Exa's plain-text list of ``Title:`` and ``URL:`` blocks."""
    if "URL:" not in text:
        if re.search(r"no (?:search )?results", text, re.I) or not text.strip():
            return []
        raise VendorError("malformed", _detail(text))
    hits = []
    entry: dict | None = None
    body: list[str] = []
    in_body = False
    after_rule = False

    def flush():
        if entry is not None:
            hits.append(
                make_hit(
                    entry.get("Title"),
                    entry.get("URL"),
                    "\n".join(body).strip(),
                    entry.get("Published"),
                )
            )

    for line in _split_lines(text):
        if line.strip() == "---":
            after_rule = True
            continue
        match = _EXA_FIELD_RE.match(line)
        if match and match.group(1) == "Title" and (entry is None or after_rule):
            flush()
            entry, body, in_body = {"Title": match.group(2)}, [], False
            after_rule = False
            continue
        if line.strip():
            after_rule = False
        if entry is None:
            continue
        if match and not in_body:
            key, value = match.groups()
            if key in ("Highlights", "Text", "Summary"):
                in_body = True
                if value:
                    body.append(value)
            else:
                entry[key] = value
            continue
        body.append(line)
    flush()
    return [hit for hit in hits if hit is not None]


def parse_keenable(data) -> list:
    return _parse_results(data, lambda r: r.get("snippet") or r.get("description"))


def parse_tavily(data) -> list:
    return _parse_results(data, lambda r: r.get("content"), "published_date")


def _shape_error(data) -> VendorError:
    """An answer without results: either an error message or a shape we don't know."""
    if isinstance(data, dict):
        for key in ("error", "detail", "message"):
            value = data.get(key)
            if value:
                text = value if isinstance(value, str) else json.dumps(value)
                return _envelope_error(text)
    return VendorError("malformed", _detail(json.dumps(data)))


def _make_client():
    import http.cookiejar

    import httpx2

    from .agent import _swival_user_agent

    class RejectCookies(http.cookiejar.DefaultCookiePolicy):
        """Never keep or send cookies, so services can't link searches."""

        def set_ok(self, cookie, request):
            return False

        def return_ok(self, cookie, request):
            return False

    return httpx2.AsyncClient(
        follow_redirects=False,
        cookies=http.cookiejar.CookieJar(policy=RejectCookies()),
        timeout=httpx2.Timeout(REQUEST_TIMEOUT),
        headers={"User-Agent": _swival_user_agent()},
        limits=httpx2.Limits(max_connections=8, max_keepalive_connections=4),
    )


class Transport:
    """HTTP for the adapters: one shared client, size-capped answers, MCP sessions."""

    def __init__(self, client):
        self.client = client
        # Random per process, so Parallel can't link searches across processes.
        self.session_id = uuid.uuid4().hex
        self.mcp_sessions: dict[str, str] = {}

    async def post(self, url: str, payload, headers: dict | None = None):
        import httpx2

        request_headers = {"Content-Type": "application/json", **(headers or {})}
        content = json.dumps(payload).encode("utf-8")
        try:
            async with self.client.stream(
                "POST", url, content=content, headers=request_headers
            ) as response:
                chunks = []
                size = 0
                # Read in chunks, so an oversized answer is cut off early.
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > MAX_BODY_BYTES:
                        raise VendorError("malformed", "response body over 1 MB")
                    chunks.append(chunk)
                return response.status_code, response.headers, b"".join(chunks)
        except httpx2.TimeoutException:
            raise VendorError("vendor_error", "timed out") from None
        except httpx2.DecodingError:
            raise VendorError("malformed", "undecodable response body") from None
        except (httpx2.HTTPError, OSError) as exc:
            raise VendorError("vendor_error", type(exc).__name__) from None

    async def post_json(self, url: str, payload, headers: dict | None = None):
        status, response_headers, body = await self.post(url, payload, headers)
        _check_status(status, response_headers, body)
        try:
            return json.loads(body), response_headers
        except ValueError:
            raise VendorError("malformed", _sample(body)) from None

    async def mcp_call(self, vendor: str, url: str, tool: str, arguments: dict):
        """Call an MCP tool directly, without the usual handshake.

        Both MCP services accept that today.
        If one starts asking for a session, the handshake runs once and the
        call is retried with the session it gave.
        """
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        }
        retried = False
        while True:
            session = self.mcp_sessions.get(vendor)
            status, headers, body = await self.post(url, payload, _mcp_headers(session))
            if not retried and _wants_session(status, body, session is not None):
                self.mcp_sessions.pop(vendor, None)
                await self._initialize(vendor, url)
                retried = True
                continue
            result = _rpc_response(status, headers, body).get("result")
            if not isinstance(result, dict):
                raise VendorError("malformed", "JSON-RPC result is not an object")
            if result.get("isError"):
                raise _envelope_error(_mcp_text(result))
            return result, headers

    async def _initialize(self, vendor: str, url: str) -> None:
        from .agent import _swival_version

        payload = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_PROTOCOL,
                "capabilities": {},
                "clientInfo": {"name": "swival", "version": _swival_version()},
            },
        }
        status, headers, body = await self.post(url, payload, _mcp_headers())
        _rpc_response(status, headers, body)
        session = headers.get("mcp-session-id")
        if not session:
            raise VendorError("drift", "initialize returned no session id")
        self.mcp_sessions[vendor] = session
        await self.post(
            url,
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            _mcp_headers(session),
        )


def _wants_session(status: int, body: bytes, had_session: bool) -> bool:
    if status == 404 and had_session:
        return True
    return status in (400, 404) and b"session" in body.lower()


async def search_parallel(transport: Transport, query: str) -> VendorResponse:
    result, headers = await transport.mcp_call(
        "parallel",
        ENDPOINTS["parallel"],
        "web_search",
        {
            "objective": query,
            "search_queries": [query],
            "session_id": transport.session_id,
        },
    )
    return VendorResponse(parse_parallel(result), _quota_cooldown(headers))


async def search_exa(transport: Transport, query: str) -> VendorResponse:
    result, headers = await transport.mcp_call(
        "exa",
        ENDPOINTS["exa"],
        "web_search_exa",
        {"query": query, "numResults": VENDOR_RESULTS},
    )
    return VendorResponse(parse_exa(_mcp_text(result)), _quota_cooldown(headers))


async def search_keenable(transport: Transport, query: str) -> VendorResponse:
    data, headers = await transport.post_json(
        ENDPOINTS["keenable"],
        {"query": query, "max_results": VENDOR_RESULTS},
        {"X-Keenable-Title": "swival"},
    )
    return VendorResponse(parse_keenable(data), _quota_cooldown(headers))


async def search_tavily(transport: Transport, query: str) -> VendorResponse:
    data, headers = await transport.post_json(
        ENDPOINTS["tavily"],
        {
            "query": query,
            "search_depth": "basic",
            "chunks_per_source": 3,
            "max_results": VENDOR_RESULTS,
        },
        {"X-Tavily-Access-Mode": "keyless"},
    )
    return VendorResponse(parse_tavily(data), _quota_cooldown(headers))


ADAPTERS = {
    "parallel": search_parallel,
    "keenable": search_keenable,
    "exa": search_exa,
    "tavily": search_tavily,
}


@dataclass
class _Health:
    until: float = 0.0
    kind: str = ""
    throttles: int = 0
    # Bumped whenever a cooldown is set, so a late answer can't undo it.
    changed_seq: int = 0


@dataclass
class _Flight:
    task: asyncio.Task
    destinations: tuple
    waiters: int = 0


@dataclass(frozen=True)
class Attempt:
    vendor: str
    # ok, empty, decoy, cooling, or a VendorError kind
    outcome: str
    detail: str = ""


@dataclass(frozen=True)
class Outcome:
    hits: list = field(default_factory=list)
    vendor: str = ""
    error: str = ""
    details: tuple = ()
    attempts: tuple = ()
    cache_hit: bool = False
    joined: bool = False

    @property
    def requests(self) -> int:
        return sum(a.outcome != "cooling" for a in self.attempts)

    @property
    def failovers(self) -> int:
        return max(0, self.requests - 1)


class Cancelled(Exception):
    pass


def _failure_error(kinds: list[str]) -> str:
    if kinds and all(k == "drift" for k in kinds):
        return REJECTED
    unavailable = sum(k in ("throttled", "refused") for k in kinds)
    if kinds and unavailable * 2 >= len(kinds):
        return UNAVAILABLE
    return FAILING


def _as_vendor_error(exc: Exception, vendor: str) -> VendorError:
    if isinstance(exc, VendorError):
        return exc
    if isinstance(exc, TimeoutError):
        return VendorError("vendor_error", "timed out")
    logger.debug("web search adapter %s failed", vendor, exc_info=exc)
    return VendorError("vendor_error", type(exc).__name__)


def _resolve_on_daemon_threads(loop) -> None:
    """Make *loop* look up host names on daemon threads.

    A stuck lookup can't be stopped, and Python waits for the default
    executor's threads before exiting, so one stuck lookup would keep the
    process alive.
    Daemon threads don't hold up the exit, and at most DNS_THREADS run at once.
    """
    slots = asyncio.Semaphore(DNS_THREADS)

    def settle(future, result, error):
        slots.release()
        if future.done():
            return
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(result)

    async def getaddrinfo(host, port, *, family=0, type=0, proto=0, flags=0):
        await slots.acquire()
        future = loop.create_future()

        def resolve():
            result = error = None
            try:
                result = socket.getaddrinfo(host, port, family, type, proto, flags)
            except Exception as exc:
                error = exc
            try:
                loop.call_soon_threadsafe(settle, future, result, error)
            except RuntimeError:
                pass

        try:
            threading.Thread(
                target=resolve, name="swival-websearch-dns", daemon=True
            ).start()
        except BaseException:
            slots.release()
            raise
        return await future

    loop.getaddrinfo = getaddrinfo


class Engine:
    """Search state shared by the whole process, run on its own event loop.

    All network work happens on that loop, so a deadline can stop it
    completely: a server that answers slowly or never is cut off instead of
    being left running.
    """

    def __init__(
        self,
        adapters: dict | None = None,
        *,
        transport_factory=None,
        clock=time.monotonic,
        rng: random.Random | None = None,
    ):
        self._adapters = dict(ADAPTERS if adapters is None else adapters)
        self._transport_factory = transport_factory or (
            lambda: Transport(_make_client())
        )
        self._transport: Transport | None = None
        self._clock = clock
        self._lock = threading.Lock()
        self._health = {name: _Health() for name in PROVIDERS}
        self._cursor = (rng or random.Random()).randrange(len(PROVIDERS))
        self._cache: collections.OrderedDict = collections.OrderedDict()
        self._flights: dict[str, list[_Flight]] = {}
        self._calls: collections.deque = collections.deque()
        self._seq = 0
        self._slots: asyncio.Semaphore | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def plan(self, allowed) -> list[str]:
        """Order to try services in for the next search, skipping those cooling down.

        The ring moves one step per search, so the load spreads out.
        """
        now = self._clock()
        with self._lock:
            start = self._cursor
            self._cursor = (start + 1) % len(PROVIDERS)
            ring = PROVIDERS[start:] + PROVIDERS[:start]
            return [n for n in ring if n in allowed and self._health[n].until <= now]

    def search(
        self, query, destinations, allowed, timeout: float, cancel_flag=None
    ) -> Outcome:
        """Run one search on the engine loop, giving up after *timeout* seconds."""
        loop = self._ensure_started()
        future = asyncio.run_coroutine_threadsafe(
            asyncio.wait_for(
                self._search(query, tuple(destinations), tuple(allowed)), timeout
            ),
            loop,
        )
        give_up = time.monotonic() + timeout + WAIT_GRACE
        while True:
            remaining = give_up - time.monotonic()
            if remaining <= 0:
                future.cancel()
                raise TimeoutError
            try:
                return future.result(timeout=min(remaining, 0.1))
            except TimeoutError:
                if future.done():
                    raise
                if cancel_flag is not None and cancel_flag.is_set():
                    future.cancel()
                    raise Cancelled from None
            except concurrent.futures.CancelledError:
                raise Cancelled from None

    def close(self) -> None:
        """Cancel pending work, close the client and stop the loop."""
        with self._lock:
            loop, thread = self._loop, self._thread
            self._loop = self._thread = None
        if loop is None:
            return

        async def shutdown():
            current = asyncio.current_task()
            tasks = [t for t in asyncio.all_tasks() if t is not current]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self._transport is not None:
                client = getattr(self._transport, "client", None)
                self._transport = None
                if client is not None:
                    await client.aclose()

        try:
            asyncio.run_coroutine_threadsafe(shutdown(), loop).result(timeout=5)
        except Exception as exc:
            logger.debug("web search engine shutdown: %r", exc)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        if not thread.is_alive():
            loop.close()

    def pending_tasks(self) -> int:
        """Tasks still running on the engine loop (for tests)."""
        loop = self._loop
        if loop is None:
            return 0

        async def count():
            current = asyncio.current_task()
            return sum(
                1 for t in asyncio.all_tasks() if t is not current and not t.done()
            )

        return asyncio.run_coroutine_threadsafe(count(), loop).result(timeout=5)

    def _ensure_started(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is not None:
                return self._loop
            loop = asyncio.new_event_loop()
            _resolve_on_daemon_threads(loop)
            thread = threading.Thread(
                target=loop.run_forever, name="swival-websearch", daemon=True
            )
            thread.start()
            self._slots = asyncio.Semaphore(MAX_IN_FLIGHT)
            self._loop, self._thread = loop, thread
            return loop

    async def _search(self, query, destinations, allowed) -> Outcome:
        cached = self._cache_get(query, allowed)
        if cached is not None:
            return cached
        if not destinations:
            return Outcome(error=self._cooling_error(allowed))
        # Only share a request that won't reach a service this caller excludes.
        flight = next(
            (
                f
                for f in self._flights.get(query, ())
                if not f.task.done()
                and not f.task.cancelling()
                and set(f.destinations) <= set(destinations)
            ),
            None,
        )
        joined = flight is not None
        if not joined:
            # The engine owns the request, so the callers sharing it keep it
            # when the one who started it times out.
            task = asyncio.ensure_future(self._lead(query, destinations))
            flight = _Flight(task, destinations)
            self._flights.setdefault(query, []).append(flight)
            task.add_done_callback(
                lambda _t, q=query, f=flight: self._forget_flight(q, f)
            )
        flight.waiters += 1
        try:
            outcome = await asyncio.shield(flight.task)
        finally:
            flight.waiters -= 1
            if flight.waiters == 0 and not flight.task.done():
                flight.task.cancel()
                self._forget_flight(query, flight)
        return replace(outcome, joined=True, attempts=()) if joined else outcome

    def _forget_flight(self, query, flight) -> None:
        flights = self._flights.get(query, [])
        if flight in flights:
            flights.remove(flight)
        if not flights:
            self._flights.pop(query, None)

    async def _lead(self, query, destinations) -> Outcome:
        async with asyncio.timeout(DEADLINE):
            async with self._slots:
                outcome = await self._walk(query, destinations)
        if outcome.hits and not outcome.error:
            self._cache_put(query, outcome)
        return outcome

    async def _walk(self, query, destinations) -> Outcome:
        attempts: list[Attempt] = []
        failures: list[str] = []
        best: tuple[float, list, str] | None = None
        soft_misses = 0
        ceiling = False
        for name in destinations:
            if self._cooling(name):
                attempts.append(Attempt(name, "cooling"))
                failures.append(self._health[name].kind or "vendor_error")
                continue
            if not self._take_call():
                ceiling = True
                break
            if self._transport is None:
                self._transport = self._transport_factory()
            started = self._seq
            try:
                async with asyncio.timeout(REQUEST_TIMEOUT):
                    response = await self._adapters[name](self._transport, query)
            except asyncio.CancelledError:
                # Cut off by the caller's deadline: count it like any stall.
                self._failed(
                    name, VendorError("vendor_error", "cut off at the deadline")
                )
                raise
            except Exception as exc:
                error = _as_vendor_error(exc, name)
                self._failed(name, error)
                attempts.append(Attempt(name, error.kind, error.detail))
                failures.append(error.kind)
                continue
            self._succeeded(name, response, started)
            if response.hits:
                score = coverage(query, response.hits)
                if best is None or score > best[0]:
                    best = (score, response.hits, name)
                if score >= DECOY_THRESHOLD:
                    attempts.append(Attempt(name, "ok"))
                    break
                attempts.append(Attempt(name, "decoy"))
            else:
                attempts.append(Attempt(name, "empty"))
            # An empty or off-topic answer can be a hidden rate limit, so one
            # more service gets a try, but not all of them.
            soft_misses += 1
            if soft_misses >= 2:
                break

        if best is not None:
            return Outcome(hits=best[1], vendor=best[2], attempts=tuple(attempts))
        # One empty answer next to failures may be a hidden rate limit too,
        # and "no results" would make the model rephrase.
        if soft_misses >= 2 or (soft_misses and not failures and not ceiling):
            return Outcome(attempts=tuple(attempts))
        # Services we never asked because of our own limit didn't fail.
        if ceiling:
            return Outcome(error=CEILING, attempts=tuple(attempts))
        error = _failure_error(failures)
        details = ()
        if error == REJECTED:
            details = tuple(
                f"{a.vendor}: {a.detail or a.outcome}"
                for a in attempts
                if a.outcome == "drift"
            )
        return Outcome(error=error, details=details, attempts=tuple(attempts))

    def _cooling_error(self, allowed) -> str:
        with self._lock:
            kinds = [self._health[n].kind or "vendor_error" for n in allowed]
        return _failure_error(kinds)

    def _cooling(self, name: str) -> bool:
        with self._lock:
            return self._health[name].until > self._clock()

    def _take_call(self) -> bool:
        now = self._clock()
        while self._calls and self._calls[0] <= now - CALL_WINDOW:
            self._calls.popleft()
        if len(self._calls) >= MAX_CALLS_PER_WINDOW:
            return False
        self._calls.append(now)
        return True

    def _failed(self, name: str, exc: VendorError) -> None:
        now = self._clock()
        with self._lock:
            health = self._health[name]
            if exc.kind == "throttled":
                health.throttles += 1
                delay = (
                    exc.retry_after
                    if exc.retry_after is not None
                    else min(
                        THROTTLE_COOLDOWN * 2 ** (health.throttles - 1),
                        MAX_THROTTLE_COOLDOWN,
                    )
                )
            else:
                delay = COOLDOWNS.get(exc.kind, COOLDOWNS["vendor_error"])
            # Never shorten a cooldown: an older request that fails late
            # must not cut short a newer rate limit.
            if now + delay >= health.until:
                health.until = now + delay
                health.kind = exc.kind
            self._seq += 1
            health.changed_seq = self._seq
        if exc.kind in ("drift", "malformed"):
            logger.debug("web search: %s answered %s: %s", name, exc.kind, exc.detail)

    def _succeeded(self, name: str, response: VendorResponse, started: int) -> None:
        now = self._clock()
        with self._lock:
            health = self._health[name]
            # A request that started later already set a cooldown, and that
            # is newer news than this answer.
            newer = health.changed_seq > started
            if not newer:
                health.throttles = 0
            if response.cooldown is not None:
                if now + response.cooldown >= health.until:
                    health.until = now + response.cooldown
                    health.kind = "throttled"
                self._seq += 1
                health.changed_seq = self._seq
            elif not newer:
                health.until = 0.0
                health.kind = ""

    def _cache_get(self, query, allowed) -> Outcome | None:
        entry = self._cache.get(query)
        if entry is None:
            return None
        expires, outcome = entry
        if expires <= self._clock():
            del self._cache[query]
            return None
        if outcome.vendor not in allowed:
            return None
        self._cache.move_to_end(query)
        return replace(outcome, cache_hit=True, attempts=())

    def _cache_put(self, query, outcome: Outcome) -> None:
        self._cache[query] = (
            self._clock() + CACHE_TTL,
            replace(outcome, hits=outcome.hits[:MAX_RESULTS]),
        )
        self._cache.move_to_end(query)
        while len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)


_ENGINE: Engine | None = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> Engine:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = Engine()
        return _ENGINE


def set_engine(engine: Engine | None) -> None:
    """Replace the process engine and close the old one (for tests)."""
    global _ENGINE
    with _ENGINE_LOCK:
        old, _ENGINE = _ENGINE, engine
    if old is not None and old is not engine:
        old.close()


def _filter_query(policy: WebSearchPolicy, query: str, destinations, deadline):
    """Run the query through llm_filter and return (query, error)."""
    from .filter import FilterError, FilterTimeout, run_llm_filter

    timeout = min(FILTER_TIMEOUT, deadline - time.monotonic())
    if timeout <= 0:
        return None, TIMED_OUT
    try:
        messages = run_llm_filter(
            policy.llm_filter,
            [{"role": "user", "content": query}],
            model="web_search",
            provider="web_search",
            call_kind="web_search",
            timeout=round(timeout, 1),
            extra={
                "destinations": list(destinations),
                "origin": {
                    "provider": policy.origin_provider,
                    "model": policy.origin_model,
                },
            },
        )
    except FilterTimeout:
        # Stopped by the search deadline rather than by the filter's rules.
        return None, TIMED_OUT if timeout < FILTER_TIMEOUT else FILTER_BLOCKED
    except FilterError:
        return None, FILTER_BLOCKED
    if (
        isinstance(messages, list)
        and len(messages) == 1
        and isinstance(messages[0], dict)
        and messages[0].get("role") == "user"
        and isinstance(messages[0].get("content"), str)
    ):
        return messages[0]["content"], None
    return None, FILTER_BLOCKED


def _render(outcome: Outcome, query: str, policy: WebSearchPolicy) -> str:
    if outcome.error == REJECTED and outcome.details:
        from .tools import _wrap_untrusted_error

        text = _wrap_untrusted_error(REJECTED, "\n".join(outcome.details), "web_search")
        return truncate_utf8(text, policy.max_output_bytes)
    if outcome.error:
        return outcome.error
    if not outcome.hits:
        return truncate_utf8(
            f'No results for "{query}". Try fewer or different terms.',
            policy.max_output_bytes,
        )
    return format_results(
        query,
        outcome.hits,
        origin=outcome.vendor,
        max_bytes=policy.max_output_bytes,
        context_length=policy.context_length,
    )


def run_web_search(
    args: dict,
    policy: WebSearchPolicy | None,
    *,
    report=None,
    verbose: bool = False,
    cancel_flag=None,
) -> str:
    """The web_search tool. The result never goes over the policy's byte cap."""
    if policy is None:
        return DISABLED
    from .secrets import contains_secret

    started = time.monotonic()
    deadline = started + DEADLINE
    query = query_text(args.get("query"))
    if query is None or not query.strip():
        return BAD_QUERY
    # Check the whole query first: cutting it could split a token in two.
    if contains_secret(query, policy.secret_patterns):
        return CREDENTIAL

    engine = get_engine()
    destinations = engine.plan(policy.providers)
    if policy.llm_filter:
        query, error = _filter_query(policy, query, destinations, deadline)
        if error is not None:
            return error
        if contains_secret(query, policy.secret_patterns):
            return CREDENTIAL
    query = normalize_query(query)
    if not query:
        return BAD_QUERY
    # Cutting the query can also turn an overlong token into a valid one.
    if contains_secret(query, policy.secret_patterns):
        return CREDENTIAL

    remaining = deadline - time.monotonic()
    try:
        if remaining <= 0:
            raise TimeoutError
        outcome = engine.search(
            query, destinations, policy.providers, remaining, cancel_flag
        )
    except TimeoutError:
        outcome = Outcome(error=TIMED_OUT)
    except Cancelled:
        outcome = Outcome(error=CANCELLED)
    except Exception as exc:
        logger.debug("web search failed", exc_info=True)
        outcome = Outcome(error=f"error: web search failed ({type(exc).__name__})")
    elapsed = time.monotonic() - started

    try:
        result = _render(outcome, query, policy)
    except Exception as exc:
        logger.debug("web search formatting failed", exc_info=True)
        outcome = Outcome(error=f"error: web search failed ({type(exc).__name__})")
        result = outcome.error
    shown = len(re.findall(r"(?m)^\d+\. ", result)) if outcome.hits else 0

    if report is not None:
        if outcome.hits or outcome.details:
            report.record_untrusted_input("web_search", origin=outcome.vendor)
        report.record_web_search(
            vendor=outcome.vendor,
            requests=outcome.requests,
            latency=elapsed,
            cache_hit=outcome.cache_hit,
            results=shown,
            error=result.split("\n", 1)[0] if result.startswith("error:") else None,
        )
    if verbose:
        try:
            from . import fmt

            fmt.tool_web_search(query, outcome, results=shown, elapsed=elapsed)
        except Exception as exc:
            logger.debug("tool_web_search failed: %s", exc)
    return result

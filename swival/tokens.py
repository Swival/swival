"""Shared token counting and truncation using tiktoken."""

import threading

import tiktoken


class _FallbackEncoder:
    """Conservative byte counting when tokenizer data is unavailable offline."""

    def encode(self, text: str, **kwargs) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="ignore")


_encoder = None
_is_fallback = False
_warned = False
_lock = threading.Lock()


def _get_encoder():
    """Return the shared encoder, building it on first use.

    Subagents count tokens from their own threads, and building the encoder can
    reach the network. The lock keeps one failed attempt from marking the shared
    counts as byte estimates while another thread has the real tokenizer.
    """
    global _encoder, _is_fallback
    if _encoder is not None:
        return _encoder
    with _lock:
        if _encoder is None:
            try:
                _encoder = tiktoken.get_encoding("cl100k_base")
            except Exception:
                _encoder = _FallbackEncoder()
                _is_fallback = True
    return _encoder


def encoder_is_fallback() -> bool:
    """True when these counts are UTF-8 bytes rather than tokens.

    The fallback is not a scaled estimate. A French probe counted 1,100 real
    tokens against 4,300 fallback bytes, and the ratio moves with the language
    and the tokenizer, so there is no division that makes one into the other.
    Callers use this to keep such a number advisory: report it, compare two of
    them, but never weigh one against a token budget.
    """
    _get_encoder()
    return _is_fallback


def warn_fallback_once() -> bool:
    """Say once that token counts are estimates. Returns True the first time."""
    global _warned
    if not encoder_is_fallback() or _warned:
        return False
    _warned = True
    from . import fmt

    fmt.warning(
        "tokenizer data is unavailable, so context sizes are byte estimates; "
        "preventive compaction and output clamping are off for this session"
    )
    return True


def count_tokens(text: str) -> int:
    """Count the number of tokens in *text*."""
    return len(_get_encoder().encode(text, disallowed_special=()))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Return *text* truncated to at most *max_tokens* tokens."""
    encoder = _get_encoder()
    tokens = encoder.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    return encoder.decode(tokens[:max_tokens])

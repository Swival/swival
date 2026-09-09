"""Shared token counting and truncation using tiktoken."""

import tiktoken


class _FallbackEncoder:
    """Conservative byte counting when tokenizer data is unavailable offline."""

    def encode(self, text: str, **kwargs) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="ignore")


try:
    _encoder = tiktoken.get_encoding("cl100k_base")
except Exception:
    _encoder = _FallbackEncoder()


def count_tokens(text: str) -> int:
    """Count the number of tokens in *text*."""
    return len(_encoder.encode(text, disallowed_special=()))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Return *text* truncated to at most *max_tokens* tokens."""
    tokens = _encoder.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    return _encoder.decode(tokens[:max_tokens])

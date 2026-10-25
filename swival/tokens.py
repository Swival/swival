"""Shared token counting and truncation using tiktoken."""

import queue
import threading


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
TOKENIZER_INIT_TIMEOUT = 2.0


def _load_encoder():
    import os
    from importlib.metadata import distribution

    import tiktoken

    cache_dir = os.environ.get("CUSTOM_TIKTOKEN_CACHE_DIR")
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    else:
        cache_dir = str(
            distribution("litellm").locate_file("litellm/litellm_core_utils/tokenizers")
        )
    os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir
    return tiktoken.get_encoding("cl100k_base")


def _get_encoder():
    """Use the same token counter all session without waiting indefinitely to load it."""
    global _encoder, _is_fallback
    if _encoder is not None:
        return _encoder
    with _lock:
        if _encoder is None:
            result = queue.SimpleQueue()

            def load():
                try:
                    encoder = _load_encoder()
                except Exception:
                    encoder = _FallbackEncoder()
                result.put(encoder)

            # Let slow downloads finish for later runs without delaying startup or exit.
            threading.Thread(target=load, daemon=True).start()
            try:
                encoder = result.get(timeout=TOKENIZER_INIT_TIMEOUT)
            except queue.Empty:
                encoder = _FallbackEncoder()
            _is_fallback = isinstance(encoder, _FallbackEncoder)
            _encoder = encoder
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

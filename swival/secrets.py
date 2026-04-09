"""Transparent secret encryption/decryption for LLM message boundaries.

Encrypts recognized credential tokens (API keys, PATs, etc.) before messages
leave the machine, and decrypts them when the LLM references them in responses.
The LLM provider sees format-preserving fakes; tool dispatch and local storage
see real values.
"""

import copy
import os
import threading
from contextlib import contextmanager
from typing import Any

from fast_cipher.tokens import TokenEncryptor
from fast_cipher.tokens.alphabets import ALPHANUMERIC
from fast_cipher.tokens.types import SimpleTokenPattern

from .report import ConfigError

ENCRYPT_KEY_ENV = "SWIVAL_ENCRYPT_KEY"


class SecretShield:
    """Encrypt secrets outbound, decrypt inbound.  Provenance-checked."""

    def __init__(
        self,
        *,
        key: bytes | None = None,
        tweak: bytes | None = None,
        extra_patterns: list | None = None,
    ):
        if key is None:
            key = os.urandom(32)

        self._encryptor = TokenEncryptor(key, tweak=tweak)
        self._registry: dict[str, str] = {}  # ciphertext -> plaintext
        self._lock = threading.Lock()
        self._destroyed = False

        if extra_patterns:
            for pat in extra_patterns:
                pat = dict(pat)
                pat.setdefault("body_alphabet", ALPHANUMERIC)
                pat.setdefault("min_body_length", len(pat.get("prefix", "")) + 8)
                self._encryptor.register(SimpleTokenPattern(**pat))

    @classmethod
    def from_config(
        cls,
        *,
        key_hex: str | None = None,
        tweak_str: str | None = None,
        extra_patterns: list | None = None,
    ) -> "SecretShield":
        """Construct from string config values (hex key, UTF-8 tweak)."""
        key = bytes.fromhex(key_hex) if key_hex else None
        tweak = tweak_str.encode("utf-8") if tweak_str else None
        return cls(key=key, tweak=tweak, extra_patterns=extra_patterns)

    # --- Outbound (before LLM) ---

    def encrypt_messages(self, messages: list[dict]) -> list[dict]:
        """Deep-copy messages and encrypt allowlisted fields.

        Populates the internal registry with new ciphertext->plaintext pairs.
        Returns the encrypted copy; the original list is not modified.
        """
        if self._destroyed:
            raise ConfigError("SecretShield has been destroyed")

        encrypted = copy.deepcopy(messages)
        for msg in encrypted:
            role = msg.get("role", "")
            if role in ("system", "user", "tool", "assistant"):
                self._encrypt_content(msg)
            if role == "assistant":
                self._encrypt_tool_calls(msg)
        return encrypted

    def _encrypt_content(self, msg: dict) -> None:
        content = msg.get("content")
        if content is None:
            return

        if isinstance(content, str):
            msg["content"] = self._encrypt_and_record(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str):
                        part["text"] = self._encrypt_and_record(text)

    def _encrypt_tool_calls(self, msg: dict) -> None:
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            return

        for tc in tool_calls:
            func = tc.get("function") if isinstance(tc, dict) else None
            if func is None:
                continue
            args_str = func.get("arguments")
            if isinstance(args_str, str):
                func["arguments"] = self._encrypt_and_record(args_str)

    def _encrypt_and_record(self, text: str) -> str:
        if not text:
            return text
        encrypted, mappings = self._encryptor.encrypt(text)
        with self._lock:
            for m in mappings:
                self._registry[m.ciphertext] = m.plaintext
        return encrypted

    # --- Inbound (after LLM) ---

    def encrypt_text(self, text: str) -> str:
        """Encrypt a single string."""
        if self._destroyed:
            raise ConfigError("SecretShield has been destroyed")
        if not text:
            return text
        return self._encrypt_and_record(text)

    def encrypt_obj(self, obj: Any) -> Any:
        """Recursively encrypt all string values in a JSON-serializable structure."""
        if isinstance(obj, str):
            return self.encrypt_text(obj)
        if isinstance(obj, dict):
            return {k: self.encrypt_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.encrypt_obj(v) for v in obj]
        return obj

    @classmethod
    @contextmanager
    def ensure(cls, shield: "SecretShield | None"):
        """Yield shield if provided, else create and destroy an ephemeral one."""
        if shield is not None:
            yield shield
            return
        s = cls()
        try:
            yield s
        finally:
            s.destroy()

    def reverse_known(self, text: str) -> str:
        """Replace only ciphertext tokens that we actually emitted.

        Operates on raw strings (no JSON parsing). Model-invented
        token-shaped strings pass through untouched.
        """
        if self._destroyed or not self._registry:
            return text

        with self._lock:
            snapshot = dict(self._registry)
        for ciphertext, plaintext in snapshot.items():
            text = text.replace(ciphertext, plaintext)

        return text

    # --- Lifecycle ---

    def destroy(self) -> None:
        """Zeroize key material and clear the registry."""
        if self._destroyed:
            return
        try:
            self._encryptor.destroy()
        except Exception:
            pass
        self._registry.clear()
        self._encryptor = None  # type: ignore[assignment]
        self._destroyed = True

    @property
    def destroyed(self) -> bool:
        return self._destroyed

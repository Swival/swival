"""Local LLM response cache for offline demos.

Caches LLM request/response pairs in a SQLite database so that identical
requests return cached responses without contacting the LLM.
"""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class LLMCache:
    """SQLite-backed LLM response cache.

    Usage::

        cache = LLMCache(Path(".swival/cache.db"))
        cache.open()
        hit = cache.get(completion_kwargs)
        if hit is None:
            msg, finish = call_llm(...)
            cache.put(completion_kwargs, msg, finish)
        cache.close()
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def open(self) -> None:
        """Create tables if needed and open the connection."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key      TEXT PRIMARY KEY,
                request  TEXT NOT NULL,
                response TEXT NOT NULL,
                model    TEXT NOT NULL,
                created  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )

    def close(self) -> None:
        """Close the database connection (idempotent)."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- Cache operations --

    def get(self, completion_kwargs: dict) -> tuple | None:
        """Look up a cached response.

        Returns ``(message_dict, finish_reason)`` on hit, ``None`` on miss.
        The message_dict must be reconstructed by the caller.
        """
        assert self._conn is not None, "cache not open"
        key = self._cache_key(completion_kwargs)
        row = self._conn.execute(
            "SELECT response FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        return data["message"], data["finish_reason"]

    def put(
        self,
        completion_kwargs: dict,
        message_dict: dict,
        finish_reason: str,
    ) -> None:
        """Store a response in the cache.

        ``message_dict`` should already be a plain dict (via model_dump or
        equivalent).
        """
        assert self._conn is not None, "cache not open"
        key = self._cache_key(completion_kwargs)
        request_json = json.dumps(completion_kwargs, sort_keys=True, default=str)
        response_json = json.dumps(
            {"message": message_dict, "finish_reason": finish_reason}
        )
        model = completion_kwargs.get("model", "unknown")
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, request, response, model, created) "
            "VALUES (?, ?, ?, ?, ?)",
            (key, request_json, response_json, model, now),
        )
        self._conn.commit()

    @staticmethod
    def _cache_key(completion_kwargs: dict) -> str:
        """Compute a deterministic SHA-256 hash of the request parameters."""
        # Include only fields that affect the response
        key_fields = {}
        for k in (
            "model",
            "messages",
            "tools",
            "tool_choice",
            "temperature",
            "top_p",
            "seed",
            "max_tokens",
            "extra_body",
            "reasoning_effort",
            # Cache-identity fields (prefixed with _ to avoid litellm collision)
            "_provider",
            "_api_base",
        ):
            if k in completion_kwargs:
                key_fields[k] = completion_kwargs[k]
        # Exclude system messages entirely so the cache key is stable across
        # runs regardless of system prompt changes.
        if "messages" in key_fields:
            key_fields["messages"] = [
                m
                for m in key_fields["messages"]
                if not (isinstance(m, dict) and m.get("role") == "system")
            ]
        canonical = json.dumps(key_fields, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    # -- Meta table --

    def get_meta(self, key: str) -> str | None:
        """Read a value from the meta table."""
        assert self._conn is not None, "cache not open"
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        """Write a value to the meta table."""
        assert self._conn is not None, "cache not open"
        self._conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    # -- Utility --

    def stats(self) -> dict:
        """Return cache statistics."""
        assert self._conn is not None, "cache not open"
        count = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        size = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {"entries": count, "size_bytes": size}

    def clear(self) -> None:
        """Drop all cached entries (keeps meta)."""
        assert self._conn is not None, "cache not open"
        self._conn.execute("DELETE FROM cache")
        self._conn.commit()


def open_cache(base_dir: str, cache_dir: str | None = None) -> LLMCache:
    """Create and open an LLMCache with default directory resolution."""
    import os

    resolved = cache_dir or os.path.join(base_dir, ".swival")
    cache = LLMCache(Path(resolved) / "cache.db")
    cache.open()
    return cache


def _reconstruct_message(d: dict):
    """Rebuild a litellm Message from a plain dict.

    Returns a proper litellm Message with Pydantic model_dump() support
    and correct nested types for tool_calls.
    """
    from litellm.types.utils import ChatCompletionMessageToolCall, Function, Message

    data = dict(d)
    if data.get("tool_calls"):
        rebuilt = []
        for tc in data["tool_calls"]:
            # Preserve all fields (including 'index' and any provider-specific
            # extras) so that model_dump() round-trips identically — otherwise
            # the cache key for the *next* turn changes and every subsequent
            # lookup is a miss.
            tc_kwargs = dict(tc)
            tc_kwargs["function"] = Function(
                name=tc["function"]["name"],
                arguments=tc["function"]["arguments"],
            )
            tc_kwargs.setdefault("type", "function")
            rebuilt.append(ChatCompletionMessageToolCall(**tc_kwargs))
        data["tool_calls"] = rebuilt
    return Message(**data)

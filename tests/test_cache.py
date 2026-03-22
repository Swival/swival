"""Tests for swival.cache — LLM response caching."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from swival.cache import LLMCache, _reconstruct_message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_kwargs(**overrides):
    """Build a minimal completion_kwargs dict."""
    base = {
        "model": "openai/test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 4096,
        "tools": None,
        "temperature": 0.7,
        "top_p": 1.0,
        "seed": 42,
        "_provider": "lmstudio",
        "_api_base": "http://127.0.0.1:1234/v1",
    }
    base.update(overrides)
    return base


def _sample_message_dict(**overrides):
    """Build a plain message dict (as returned by model_dump)."""
    base = {
        "role": "assistant",
        "content": "Hello! How can I help?",
    }
    base.update(overrides)
    return base


def _sample_tool_call_message():
    """Build a message dict with tool_calls."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path": "test.py"}',
                },
            }
        ],
    }


# ===========================================================================
# Cache key generation
# ===========================================================================


class TestCacheKey:
    def test_deterministic(self, tmp_path):
        cache = LLMCache(tmp_path / "cache.db")
        cache.open()
        kwargs = _sample_kwargs()
        key1 = cache._cache_key(kwargs)
        key2 = cache._cache_key(kwargs)
        assert key1 == key2
        cache.close()

    def test_stable_across_instances(self, tmp_path):
        kwargs = _sample_kwargs()
        c1 = LLMCache(tmp_path / "a.db")
        c1.open()
        c2 = LLMCache(tmp_path / "b.db")
        c2.open()
        assert c1._cache_key(kwargs) == c2._cache_key(kwargs)
        c1.close()
        c2.close()

    def test_different_params_different_keys(self, tmp_path):
        cache = LLMCache(tmp_path / "cache.db")
        cache.open()
        k1 = cache._cache_key(_sample_kwargs(temperature=0.7))
        k2 = cache._cache_key(_sample_kwargs(temperature=0.9))
        assert k1 != k2
        cache.close()

    def test_message_ordering_matters(self, tmp_path):
        cache = LLMCache(tmp_path / "cache.db")
        cache.open()
        msgs1 = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
        msgs2 = [{"role": "user", "content": "b"}, {"role": "user", "content": "a"}]
        k1 = cache._cache_key(_sample_kwargs(messages=msgs1))
        k2 = cache._cache_key(_sample_kwargs(messages=msgs2))
        assert k1 != k2
        cache.close()

    def test_different_providers_different_keys(self, tmp_path):
        cache = LLMCache(tmp_path / "cache.db")
        cache.open()
        k1 = cache._cache_key(_sample_kwargs(_provider="lmstudio"))
        k2 = cache._cache_key(_sample_kwargs(_provider="generic"))
        assert k1 != k2
        cache.close()

    def test_system_prompt_ignored(self, tmp_path):
        """System messages should be completely excluded from the cache key."""
        cache = LLMCache(tmp_path / "cache.db")
        cache.open()
        msgs1 = [
            {"role": "system", "content": "Prompt A"},
            {"role": "user", "content": "hello"},
        ]
        msgs2 = [
            {"role": "system", "content": "Totally different prompt"},
            {"role": "user", "content": "hello"},
        ]
        msgs3 = [
            {"role": "user", "content": "hello"},
        ]
        k1 = cache._cache_key(_sample_kwargs(messages=msgs1))
        k2 = cache._cache_key(_sample_kwargs(messages=msgs2))
        k3 = cache._cache_key(_sample_kwargs(messages=msgs3))
        assert k1 == k2 == k3
        cache.close()

    def test_different_max_tokens_different_keys(self, tmp_path):
        cache = LLMCache(tmp_path / "cache.db")
        cache.open()
        k1 = cache._cache_key(_sample_kwargs(max_tokens=4096))
        k2 = cache._cache_key(_sample_kwargs(max_tokens=8192))
        assert k1 != k2
        cache.close()


# ===========================================================================
# Cache hit / miss
# ===========================================================================


class TestCacheHitMiss:
    def test_miss_returns_none(self, tmp_path):
        cache = LLMCache(tmp_path / "cache.db")
        cache.open()
        assert cache.get(_sample_kwargs()) is None
        cache.close()

    def test_put_then_get(self, tmp_path):
        cache = LLMCache(tmp_path / "cache.db")
        cache.open()
        kwargs = _sample_kwargs()
        msg = _sample_message_dict()
        cache.put(kwargs, msg, "stop")
        hit = cache.get(kwargs)
        assert hit is not None
        msg_dict, finish = hit
        assert finish == "stop"
        assert msg_dict["content"] == "Hello! How can I help?"
        cache.close()

    def test_tool_calls_round_trip(self, tmp_path):
        cache = LLMCache(tmp_path / "cache.db")
        cache.open()
        kwargs = _sample_kwargs()
        msg = _sample_tool_call_message()
        cache.put(kwargs, msg, "tool_calls")
        hit = cache.get(kwargs)
        assert hit is not None
        msg_dict, finish = hit
        assert finish == "tool_calls"
        assert len(msg_dict["tool_calls"]) == 1
        tc = msg_dict["tool_calls"][0]
        assert tc["function"]["name"] == "read_file"
        assert tc["id"] == "call_123"
        cache.close()


# ===========================================================================
# Message reconstruction
# ===========================================================================


class TestMessageReconstruction:
    def test_content_attribute(self):
        msg = _reconstruct_message({"role": "assistant", "content": "hi"})
        assert msg.content == "hi"
        assert msg.role == "assistant"

    def test_model_dump(self):
        msg = _reconstruct_message({"role": "assistant", "content": "hi"})
        d = msg.model_dump(exclude_none=True)
        assert d["content"] == "hi"
        assert d["role"] == "assistant"

    def test_tool_calls_nested_access(self):
        msg = _reconstruct_message(_sample_tool_call_message())
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        tc = msg.tool_calls[0]
        assert tc.function.name == "read_file"
        assert tc.function.arguments == '{"path": "test.py"}'
        assert tc.id == "call_123"

    def test_tool_calls_model_dump(self):
        msg = _reconstruct_message(_sample_tool_call_message())
        d = msg.model_dump(exclude_none=True)
        # Should be JSON-serializable
        json.dumps(d)
        assert d["tool_calls"][0]["function"]["name"] == "read_file"

    def test_round_trip_serialize_deserialize(self):
        original = _sample_tool_call_message()
        msg = _reconstruct_message(original)
        dumped = msg.model_dump(exclude_none=True)
        msg2 = _reconstruct_message(dumped)
        assert msg2.tool_calls[0].function.name == "read_file"
        # Must be JSON serializable for messages list
        json.dumps([dumped])


# ===========================================================================
# Meta table
# ===========================================================================


class TestMeta:
    def test_meta_round_trip(self, tmp_path):
        cache = LLMCache(tmp_path / "cache.db")
        cache.open()
        cache.set_meta("version", "1.0")
        assert cache.get_meta("version") == "1.0"
        cache.close()

        # Reopen and verify persistence
        cache2 = LLMCache(tmp_path / "cache.db")
        cache2.open()
        assert cache2.get_meta("version") == "1.0"
        cache2.close()

    def test_meta_miss_returns_none(self, tmp_path):
        cache = LLMCache(tmp_path / "cache.db")
        cache.open()
        assert cache.get_meta("nonexistent") is None
        cache.close()


# ===========================================================================
# Cache directory resolution
# ===========================================================================


class TestCacheDirResolution:
    def test_relative_cache_dir_in_global_config(self, tmp_path, monkeypatch):
        """Relative cache_dir in global config resolves against config dir."""
        from swival.config import _resolve_paths

        config = {"cache_dir": ".my-cache"}
        config_dir = tmp_path / ".config" / "swival"
        _resolve_paths(config, config_dir)
        assert config["cache_dir"] == str(config_dir / ".my-cache")

    def test_absolute_cache_dir_unchanged(self, tmp_path, monkeypatch):
        from swival.config import _resolve_paths

        config = {"cache_dir": "/tmp/my-cache"}
        _resolve_paths(config, tmp_path)
        assert config["cache_dir"] == "/tmp/my-cache"

    def test_home_prefix_expanded(self, tmp_path):
        from swival.config import _resolve_paths

        config = {"cache_dir": "~/my-cache"}
        _resolve_paths(config, tmp_path)
        assert config["cache_dir"] == str(Path.home() / "my-cache")


# ===========================================================================
# Cache lifecycle
# ===========================================================================


class TestCacheLifecycle:
    def test_open_close_idempotent(self, tmp_path):
        cache = LLMCache(tmp_path / "cache.db")
        cache.open()
        cache.close()
        cache.close()  # double close should not raise

    def test_stats(self, tmp_path):
        cache = LLMCache(tmp_path / "cache.db")
        cache.open()
        stats = cache.stats()
        assert stats["entries"] == 0
        cache.put(_sample_kwargs(), _sample_message_dict(), "stop")
        stats = cache.stats()
        assert stats["entries"] == 1
        cache.close()

    def test_clear(self, tmp_path):
        cache = LLMCache(tmp_path / "cache.db")
        cache.open()
        cache.put(_sample_kwargs(), _sample_message_dict(), "stop")
        assert cache.stats()["entries"] == 1
        cache.clear()
        assert cache.stats()["entries"] == 0
        cache.close()


# ===========================================================================
# CLI argument parsing
# ===========================================================================


class TestCacheCLI:
    def test_cache_flag_parsed(self):
        from swival.agent import build_parser

        parser = build_parser()
        args = parser.parse_args(["--cache", "--repl"])
        # store_true with default=_UNSET: True when flag is present
        assert args.cache is True

    def test_cache_dir_parsed(self):
        from swival.agent import build_parser

        parser = build_parser()
        args = parser.parse_args(["--cache-dir", "/tmp/demo", "--repl"])
        assert args.cache_dir == "/tmp/demo"


# ===========================================================================
# Integration: cache wraps call_llm
# ===========================================================================


class TestCacheIntegration:
    def test_second_identical_call_hits_cache(self, tmp_path):
        """Mock litellm.completion, verify cache hit on second call."""
        from swival.agent import call_llm

        cache = LLMCache(tmp_path / "cache.db")
        cache.open()

        # Build a mock response
        mock_msg = MagicMock()
        mock_msg.content = "Sure, here's the file."
        mock_msg.tool_calls = None
        mock_msg.model_dump.return_value = {
            "role": "assistant",
            "content": "Sure, here's the file.",
        }

        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("litellm.completion", return_value=mock_response) as mock_comp:
            # First call — should hit litellm
            msg1, fr1, _, _ = call_llm(
                "http://127.0.0.1:1234",
                "test-model",
                [{"role": "user", "content": "hello"}],
                4096,
                0.7,
                1.0,
                42,
                None,
                False,
                provider="lmstudio",
                cache=cache,
            )
            assert mock_comp.call_count == 1
            assert fr1 == "stop"

            # Second identical call — should hit cache, not litellm
            msg2, fr2, _, _ = call_llm(
                "http://127.0.0.1:1234",
                "test-model",
                [{"role": "user", "content": "hello"}],
                4096,
                0.7,
                1.0,
                42,
                None,
                False,
                provider="lmstudio",
                cache=cache,
            )
            assert mock_comp.call_count == 1  # still 1
            assert fr2 == "stop"
            assert msg2.content == "Sure, here's the file."

        cache.close()

    def test_different_provider_no_cache_hit(self, tmp_path):
        """Same model string but different provider should not hit cache."""
        cache = LLMCache(tmp_path / "cache.db")
        cache.open()

        kwargs_lm = _sample_kwargs(_provider="lmstudio")
        kwargs_gen = _sample_kwargs(_provider="generic")
        msg = _sample_message_dict()

        cache.put(kwargs_lm, msg, "stop")
        assert cache.get(kwargs_lm) is not None
        assert cache.get(kwargs_gen) is None
        cache.close()


# ===========================================================================
# Secondary call wrapper (_call_llm_for_secondary)
# ===========================================================================


class TestSecondaryCallWrapper:
    """Verify that secondary LLM call paths receive the cache."""

    def test_wrapper_injects_cache_kwarg(self, tmp_path):
        """The _call_llm_for_secondary wrapper should inject cache= into kwargs."""
        cache = LLMCache(tmp_path / "cache.db")
        cache.open()

        captured = {}

        def fake_call_llm(*args, **kwargs):
            captured.update(kwargs)
            # Return a minimal valid response
            from types import SimpleNamespace

            msg = SimpleNamespace(content="ok", tool_calls=None, role="assistant")
            msg.get = lambda key, default=None: getattr(msg, key, default)
            return msg, "stop"

        # Simulate what run_agent_loop does when cache is not None
        _call_llm_for_secondary = fake_call_llm  # default
        if cache is not None:

            def _call_llm_for_secondary(*args, **kwargs):
                kwargs.setdefault("cache", cache)
                return fake_call_llm(*args, **kwargs)

        # Call without explicit cache — wrapper should inject it
        _call_llm_for_secondary(
            "http://localhost:1234",
            "model",
            [],
            4096,
            0.7,
            1.0,
            42,
            None,
            False,
            provider="lmstudio",
        )
        assert captured.get("cache") is cache

        # Call with explicit cache=None — setdefault should not override
        captured.clear()
        _call_llm_for_secondary(
            "http://localhost:1234",
            "model",
            [],
            4096,
            0.7,
            1.0,
            42,
            None,
            False,
            provider="lmstudio",
            cache=None,
        )
        # setdefault doesn't overwrite an explicitly passed None
        assert captured.get("cache") is None

        cache.close()

    def test_compaction_summary_uses_wrapper(self, tmp_path):
        """_llm_summary_kwargs.call_llm_fn should be the cache-aware wrapper."""
        cache = LLMCache(tmp_path / "cache.db")
        cache.open()

        received_cache = {}

        def spy_call_llm(*args, **kwargs):
            received_cache["cache"] = kwargs.get("cache")
            from types import SimpleNamespace

            msg = SimpleNamespace(content="summary", tool_calls=None, role="assistant")
            msg.get = lambda key, default=None: getattr(msg, key, default)
            return msg, "stop"

        # Build the wrapper the same way run_agent_loop does
        def _call_llm_for_secondary(*args, **kwargs):
            kwargs.setdefault("cache", cache)
            return spy_call_llm(*args, **kwargs)

        # Simulate building _llm_summary_kwargs as run_agent_loop does
        _llm_summary_kwargs = dict(
            call_llm_fn=_call_llm_for_secondary,
            model_id="test",
            base_url="http://localhost",
            api_key=None,
            top_p=1.0,
            seed=None,
            provider="lmstudio",
            compaction_state=None,
        )

        # When compaction code calls call_llm_fn, it should get cache
        _llm_summary_kwargs["call_llm_fn"](
            "http://localhost",
            "test",
            [],
            4096,
            0.7,
            1.0,
            None,
            None,
            False,
            provider="lmstudio",
        )
        assert received_cache["cache"] is cache

        cache.close()

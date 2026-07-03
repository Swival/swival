"""Tests for swival/model_catalog.py (no real network)."""

import io
import json
import urllib.error

import pytest

from swival import model_catalog as mc


def _fake_get_json(responses):
    """Return a _get_json stand-in serving canned payloads keyed by URL substring."""

    def fake(url, api_key, timeout):
        for fragment, payload in responses.items():
            if fragment in url:
                if isinstance(payload, Exception):
                    raise payload
                return payload
        raise AssertionError(f"unexpected URL fetched: {url}")

    return fake


HF_ROUTER_PAYLOAD = {
    "object": "list",
    "data": [
        {
            "id": "org/served-model",
            "architecture": {"input_modalities": ["text", "image"]},
            "providers": [
                {
                    "provider": "novita",
                    "status": "live",
                    "context_length": 1048576,
                    "pricing": {"input": 1.4, "output": 4.4},
                    "supports_tools": True,
                    "is_free": False,
                },
                {
                    "provider": "cheapo",
                    "status": "live",
                    "context_length": 262144,
                    "pricing": {"input": 0.9, "output": 3.0},
                    "supports_tools": True,
                    "is_free": False,
                },
                {"provider": "dead", "status": "staging"},
            ],
        },
        {
            "id": "org/staging-only",
            "providers": [{"provider": "x", "status": "staging"}],
        },
        {
            "id": "org/no-metadata",
            "providers": [{"provider": "bare", "status": "live"}],
        },
    ],
}


def test_hf_router_catalog(monkeypatch):
    monkeypatch.setattr(
        mc, "_get_json", _fake_get_json({"router.huggingface.co": HF_ROUTER_PAYLOAD})
    )
    catalog = mc.list_models("huggingface")

    ids = [e.id for e in catalog.entries]
    assert "org/served-model" in ids
    assert "org/staging-only" not in ids  # no live provider

    entry = next(e for e in catalog.entries if e.id == "org/served-model")
    assert entry.context_length == 1048576  # max across live providers
    assert entry.price_in == 0.9  # cheapest live provider
    assert entry.price_out == 3.0
    assert entry.supports_tools is True
    assert "2 providers" in entry.tags
    assert "vision" in entry.tags
    assert "novita" in entry.detail and "cheapo" in entry.detail

    bare = next(e for e in catalog.entries if e.id == "org/no-metadata")
    assert bare.supports_tools is None
    assert bare.context_length is None


def test_hf_price_tie_with_missing_output_price(monkeypatch):
    payload = {
        "data": [
            {
                "id": "org/tied",
                "providers": [
                    {
                        "provider": "a",
                        "status": "live",
                        "pricing": {"input": 0.5},
                    },
                    {
                        "provider": "b",
                        "status": "live",
                        "pricing": {"input": 0.5, "output": 1.5},
                    },
                ],
            }
        ]
    }
    monkeypatch.setattr(
        mc, "_get_json", _fake_get_json({"router.huggingface.co": payload})
    )
    entry = mc.list_models("huggingface").entries[0]
    assert entry.price_in == 0.5
    assert entry.price_out == 1.5  # the fully priced provider wins the tie


def test_hf_dedicated_endpoint_uses_generic_listing(monkeypatch):
    payload = {"data": [{"id": "my-model", "max_model_len": 8192}]}
    monkeypatch.setattr(
        mc, "_get_json", _fake_get_json({"my-endpoint.example/v1/models": payload})
    )
    catalog = mc.list_models("huggingface", base_url="http://my-endpoint.example")
    assert catalog.entries[0].id == "my-model"
    assert catalog.entries[0].context_length == 8192


def test_hf_hub_search(monkeypatch):
    payload = [
        {
            "id": "org/tail-model",
            "inferenceProviderMapping": [
                {
                    "provider": "novita",
                    "status": "live",
                    "task": "conversational",
                    "features": {"toolCalling": True},
                    "providerDetails": {
                        "context_length": 131072,
                        "pricing": {"input": 0.2, "output": 0.6},
                    },
                }
            ],
        },
        {
            "id": "org/embedding-model",
            "inferenceProviderMapping": [
                {"provider": "novita", "status": "live", "task": "feature-extraction"}
            ],
        },
    ]
    monkeypatch.setattr(
        mc, "_get_json", _fake_get_json({"huggingface.co/api/models": payload})
    )
    entries = mc.search_hf_models("tail")
    assert [e.id for e in entries] == ["org/tail-model"]
    assert entries[0].supports_tools is True
    assert entries[0].context_length == 131072
    assert entries[0].price_in == 0.2


def test_hf_model_status_dict_mapping(monkeypatch):
    # The model-detail endpoint returns a dict keyed by provider name (unlike
    # the list endpoint, which returns a list of dicts).
    payload = {
        "id": "org/detail-model",
        "inferenceProviderMapping": {
            "novita": {"status": "live", "task": "conversational"},
            "deepinfra": {"status": "error", "task": "conversational"},
            "cheapo": {"status": "live", "task": "feature-extraction"},
        },
    }
    monkeypatch.setattr(
        mc, "_get_json", _fake_get_json({"api/models/org/detail-model": payload})
    )
    exists, entry, others = mc.hf_model_status("org/detail-model")
    assert exists
    assert entry is not None
    assert entry.id == "org/detail-model"
    assert "novita" in entry.detail
    assert sorted(others) == ["cheapo (live)", "deepinfra (error)"]


def test_hf_model_status_list_mapping_and_404(monkeypatch):
    payload = {
        "id": "org/list-model",
        "inferenceProviderMapping": [
            {"provider": "novita", "status": "live", "task": "conversational"}
        ],
    }
    monkeypatch.setattr(
        mc, "_get_json", _fake_get_json({"api/models/org/list-model": payload})
    )
    exists, entry, others = mc.hf_model_status("org/list-model")
    assert exists
    assert entry is not None and entry.id == "org/list-model"
    assert others == []

    def gone(url, api_key, timeout):
        raise mc.CatalogUnavailable("model list request to x failed: HTTP 404")

    monkeypatch.setattr(mc, "_get_json", gone)
    assert mc.hf_model_status("org/missing") == (False, None, [])


def test_lmstudio_catalog(monkeypatch):
    payload = {
        "data": [
            {
                "id": "loaded-model",
                "type": "llm",
                "loaded_instances": [{"config": {"context_length": 32768}}],
            },
            {
                "id": "cold-model",
                "type": "llm",
                "loaded_instances": [],
                "max_context_length": 131072,
                "quantization": "Q4_K_M",
            },
            {"id": "embedder", "type": "embedding"},
            {
                "id": "vision-model",
                "type": "vlm",
                "loaded_instances": [],
            },
        ]
    }
    monkeypatch.setattr(mc, "_get_json", _fake_get_json({"/api/v1/models": payload}))
    catalog = mc.list_models("lmstudio")

    ids = [e.id for e in catalog.entries]
    assert "embedder" not in ids
    assert ids[0] == "loaded-model"  # loaded models sort first

    loaded = catalog.entries[0]
    assert loaded.loaded is True
    assert loaded.context_length == 32768

    cold = next(e for e in catalog.entries if e.id == "cold-model")
    assert cold.loaded is False
    assert cold.context_length == 131072
    assert "Q4_K_M" in cold.tags

    vision = next(e for e in catalog.entries if e.id == "vision-model")
    assert "vision" in vision.tags


def test_lmstudio_empty_raises(monkeypatch):
    monkeypatch.setattr(
        mc, "_get_json", _fake_get_json({"/api/v1/models": {"data": []}})
    )
    with pytest.raises(mc.CatalogUnavailable) as exc:
        mc.list_models("lmstudio")
    assert "no downloaded LLMs" in str(exc.value)


def test_llamacpp_catalog_with_context_probe(monkeypatch):
    responses = {
        "/v1/models": {"data": [{"id": "gguf-model"}]},
        "/props": {"default_generation_settings": {"n_ctx": 16384}},
    }
    monkeypatch.setattr(mc, "_get_json", _fake_get_json(responses))
    catalog = mc.list_models("llamacpp")
    assert catalog.entries[0].id == "gguf-model"
    assert catalog.entries[0].context_length == 16384


def test_llamacpp_props_failure_is_tolerated(monkeypatch):
    responses = {
        "/v1/models": {"data": [{"id": "gguf-model"}]},
        "/props": mc.CatalogUnavailable("nope"),
    }
    monkeypatch.setattr(mc, "_get_json", _fake_get_json(responses))
    catalog = mc.list_models("llamacpp")
    assert catalog.entries[0].context_length is None


def test_openrouter_catalog(monkeypatch):
    payload = {
        "data": [
            {
                "id": "acme/big-model",
                "name": "Big Model",
                "context_length": 262144,
                "pricing": {"prompt": "0.0000014", "completion": "0.0000044"},
                "supported_parameters": ["tools", "temperature"],
            },
            {
                "id": "acme/free-model:free",
                "context_length": 8192,
                "pricing": {"prompt": 0, "completion": 0},
                "supported_parameters": [],
            },
        ]
    }
    monkeypatch.setattr(mc, "_get_json", _fake_get_json({"openrouter.ai": payload}))
    catalog = mc.list_models("openrouter")

    big = catalog.entries[0]
    assert big.display_name == "Big Model"
    assert big.supports_tools is True
    assert big.price_in == pytest.approx(1.4)
    assert big.price_out == pytest.approx(4.4)

    free = catalog.entries[1]
    assert free.supports_tools is False
    assert "free" in free.tags


def test_catalog_context_length(monkeypatch):
    payload = {
        "data": [
            {"id": "acme/big-model", "context_length": 262144},
            {"id": "acme/no-context"},
        ]
    }
    monkeypatch.setattr(mc, "_get_json", _fake_get_json({"openrouter.ai": payload}))
    assert mc.catalog_context_length("openrouter", "acme/big-model") == 262144
    assert mc.catalog_context_length("openrouter", "ACME/Big-Model") == 262144
    assert mc.catalog_context_length("openrouter", "acme/no-context") is None
    assert mc.catalog_context_length("openrouter", "acme/absent") is None


def test_catalog_context_length_unavailable(monkeypatch):
    def boom(url, api_key, timeout):
        raise mc.CatalogUnavailable("network down")

    monkeypatch.setattr(mc, "_get_json", boom)
    assert mc.catalog_context_length("openrouter", "acme/big-model") is None


def test_cached_context_length(monkeypatch):
    payload = {"data": [{"id": "served", "max_model_len": 65536}]}
    monkeypatch.setattr(mc, "_get_json", _fake_get_json({"/models": payload}))
    assert mc.cached_context_length("generic", "served", "http://x") is None
    mc.list_models("generic", base_url="http://x")
    assert mc.cached_context_length("generic", "served", "http://x") == 65536
    assert mc.cached_context_length("generic", "SERVED", "http://x") == 65536
    assert mc.cached_context_length("generic", "absent", "http://x") is None
    assert mc.cached_context_length("generic", "served", "http://other") is None


def test_google_requires_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(mc.CatalogUnavailable) as exc:
        mc.list_models("google")
    assert "API key" in str(exc.value)


def test_google_catalog_strips_models_prefix(monkeypatch):
    payload = {"data": [{"id": "models/gemini-3-flash"}, {"id": "gemini-3-pro"}]}
    monkeypatch.setattr(mc, "_get_json", _fake_get_json({"/models": payload}))
    catalog = mc.list_models("google", api_key="k")
    assert [e.id for e in catalog.entries] == ["gemini-3-flash", "gemini-3-pro"]


def test_generic_requires_base_url():
    with pytest.raises(mc.CatalogUnavailable):
        mc.list_models("generic")


def test_unlistable_provider_raises():
    with pytest.raises(mc.CatalogUnavailable) as exc:
        mc.list_models("bedrock")
    assert exc.value.hint is not None
    assert not mc.supports_listing("bedrock")
    assert not mc.supports_listing("command")
    assert mc.supports_listing("huggingface")
    assert mc.supports_listing("mlx")  # onboarding alias for generic


def test_cache_and_refresh(monkeypatch):
    calls = {"n": 0}

    def counting(url, api_key, timeout):
        calls["n"] += 1
        return {"data": [{"id": f"m{calls['n']}"}]}

    monkeypatch.setattr(mc, "_get_json", counting)
    first = mc.list_models("generic", base_url="http://x")
    again = mc.list_models("generic", base_url="http://x")
    assert first is again
    assert calls["n"] == 1

    refreshed = mc.list_models("generic", base_url="http://x", refresh=True)
    assert calls["n"] == 2
    assert refreshed.entries[0].id == "m2"

    assert mc.cached_entries("generic", "http://x")[0].id == "m2"
    assert mc.cached_entries("generic", "http://other") is None


def test_get_json_wraps_network_errors(monkeypatch):
    def boom(req, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(mc.urllib.request, "urlopen", boom)
    with pytest.raises(mc.CatalogUnavailable) as exc:
        mc._get_json("http://unreachable.example/v1/models", None, 1.0)
    assert "could not fetch" in str(exc.value)


def test_get_json_wraps_http_errors(monkeypatch):
    def gone(req, timeout):
        raise urllib.error.HTTPError(
            "http://x", 404, "not found", None, io.BytesIO(b"")
        )

    monkeypatch.setattr(mc.urllib.request, "urlopen", gone)
    with pytest.raises(mc.CatalogUnavailable) as exc:
        mc._get_json("http://x", None, 1.0)
    assert "HTTP 404" in str(exc.value)


def test_get_json_parses_and_sends_auth(monkeypatch):
    seen = {}

    class FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout):
        seen["auth"] = req.headers.get("Authorization")
        return FakeResp(json.dumps({"ok": True}).encode())

    monkeypatch.setattr(mc.urllib.request, "urlopen", fake_urlopen)
    assert mc._get_json("http://x", "sekrit", 1.0) == {"ok": True}
    assert seen["auth"] == "Bearer sekrit"

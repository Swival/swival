"""Tests for swival/model_catalog.py (no real network)."""

import io
import json
import urllib.error

import pytest

from swival import model_catalog as mc


def _fake_get_json(responses, calls=None):
    """Return a _get_json stand-in serving canned payloads keyed by URL substring.

    Each request is appended to *calls* when a list is given.
    """

    def fake(url, api_key, timeout, headers=None):
        if calls is not None:
            calls.append((url, api_key, headers))
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


def test_chatgpt_catalog_includes_new_codex_models():
    """The bundled litellm registry lags behind the Codex backend, so the
    catalog supplements it with known-current model names."""
    catalog = mc.list_models("chatgpt")
    ids = [e.id for e in catalog.entries]

    for expected in (
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.6-sol",
        "gpt-6-astra",
        "gpt-6-luna",
        "gpt-6-sol",
        "gpt-6.1-sol",
    ):
        assert expected in ids
    # No duplicates, and entries stay sorted by id.
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)


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
        mc.list_models("geap")
    assert exc.value.hint is not None
    assert mc.supports_listing("bedrock")
    assert not mc.supports_listing("geap")
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
        seen.clear()
        seen.update({k.lower(): v for k, v in req.header_items()})
        return FakeResp(json.dumps({"ok": True}).encode())

    monkeypatch.setattr(mc.urllib.request, "urlopen", fake_urlopen)
    assert mc._get_json("http://x", "sekrit", 1.0) == {"ok": True}
    assert seen["authorization"] == "Bearer sekrit"

    mc._get_json("http://x", None, 1.0, headers={"x-api-key": "k"})
    assert seen["x-api-key"] == "k"
    assert "authorization" not in seen


def test_anthropic_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(mc.CatalogUnavailable) as exc:
        mc.list_models("anthropic")
    assert "ANTHROPIC_API_KEY" in exc.value.hint


def test_anthropic_catalog_merges_live_and_litellm_data(monkeypatch):
    calls = []
    listing = {
        "data": [
            {"id": "claude-opus-5-5", "display_name": "Claude Opus 5.5"},
            {
                "id": "claude-brand-new",
                "display_name": "Claude New",
                "max_input_tokens": 500_000,
                "capabilities": {"image_input": {"supported": True}},
            },
        ]
    }
    monkeypatch.setattr(mc, "_get_json", _fake_get_json({"/v1/models": listing}, calls))
    monkeypatch.setattr(
        mc,
        "_litellm_registry",
        lambda: {
            "claude-opus-5-5": {
                "max_input_tokens": 1_000_000,
                "input_cost_per_token": 4e-06,
                "output_cost_per_token": 2e-05,
                "supports_function_calling": True,
                "supports_vision": True,
            }
        },
    )
    catalog = mc.list_models("anthropic", api_key="sk-ant")

    assert calls == [
        (
            "https://api.anthropic.com/v1/models?limit=1000",
            None,
            {"x-api-key": "sk-ant", "anthropic-version": mc.ANTHROPIC_API_VERSION},
        )
    ]
    known, new = catalog.entries
    assert known.id == "claude-opus-5-5"
    assert known.display_name == "Claude Opus 5.5"
    assert known.context_length == 1_000_000
    assert known.price_in == pytest.approx(4.0)
    assert known.price_out == pytest.approx(20.0)
    assert known.supports_tools is True
    assert known.tags == ("vision",)
    assert new.id == "claude-brand-new"
    assert new.context_length == 500_000
    assert new.tags == ("vision",)
    assert new.price_in is None
    assert catalog.source == "Anthropic"


def test_anthropic_catalog_uses_env_key_and_base_url(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    calls = []
    listing = {"data": [{"id": "claude-x"}]}
    monkeypatch.setattr(mc, "_get_json", _fake_get_json({"/v1/models": listing}, calls))
    monkeypatch.setattr(mc, "_litellm_registry", lambda: {})
    mc.list_models("anthropic", base_url="https://gw.example/v1/")
    url, _, headers = calls[0]
    assert url == "https://gw.example/v1/models?limit=1000"
    assert headers["x-api-key"] == "sk-env"


def test_known_models_only_extend_a_partial_catalog():
    partial = mc.Catalog([mc.ModelEntry(id="listed")], "src", exhaustive=False)
    merged = mc.with_known_models(partial, ["LISTED", "extra", "extra"])
    assert [e.id for e in merged.entries] == ["listed", "extra"]
    assert not merged.exhaustive
    assert [e.id for e in partial.entries] == ["listed"]

    complete = mc.Catalog([mc.ModelEntry(id="listed")], "src")
    assert mc.with_known_models(complete, ["extra"]) is complete


def test_known_models_replace_a_missing_or_failed_listing(monkeypatch):
    catalog, warning = mc.list_models_or_known("geap", known=["gemini-a", "gemini-b"])
    assert [e.id for e in catalog.entries] == ["gemini-a", "gemini-b"]
    assert not catalog.exhaustive
    assert warning is None

    def down(base_url, api_key, timeout):
        raise mc.CatalogUnavailable("server down", hint="start it")

    monkeypatch.setitem(mc._FETCHERS, "lmstudio", down)
    catalog, warning = mc.list_models_or_known("lmstudio", known=["local"])
    assert [e.id for e in catalog.entries] == ["local"]
    assert warning == "cannot list models: server down\nstart it"

    with pytest.raises(mc.CatalogUnavailable):
        mc.list_models_or_known("lmstudio")


BEDROCK_OPUS_ARN = (
    "arn:aws:bedrock:us-east-2::foundation-model/anthropic.claude-opus-5-5"
)


def _bedrock_model(model_id, **overrides):
    summary = {
        "modelArn": f"arn:aws:bedrock:us-east-2::foundation-model/{model_id}",
        "modelId": model_id,
        "modelName": model_id.upper(),
        "providerName": "Test",
        "inputModalities": ["TEXT"],
        "outputModalities": ["TEXT"],
        "responseStreamingSupported": True,
        "customizationsSupported": [],
        "inferenceTypesSupported": ["ON_DEMAND"],
        "modelLifecycle": {"status": "ACTIVE"},
    }
    return summary | overrides


def _bedrock_profile(profile_id, arn=BEDROCK_OPUS_ARN, **overrides):
    summary = {
        "inferenceProfileName": profile_id.upper(),
        "inferenceProfileArn": f"arn:aws:bedrock:us-east-2:1:inference-profile/{profile_id}",
        "models": [{"modelArn": arn}],
        "inferenceProfileId": profile_id,
        "status": "ACTIVE",
        "type": "SYSTEM_DEFINED",
    }
    return summary | overrides


class _BedrockAWS:
    """A real botocore client with canned replies.

    botocore checks our calls and the replies against the real AWS API.
    """

    def __init__(self, monkeypatch):
        import boto3
        from botocore.credentials import Credentials
        from botocore.stub import Stubber

        mc._litellm_registry()
        from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

        self.bedrock = boto3.client(
            "bedrock",
            region_name="us-east-2",
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
        )
        self.stub = Stubber(self.bedrock)
        self.stub.activate()
        self.sessions = []
        self.configs = []
        self.credential_requests = []
        self.credentials = Credentials("AKTEST", "secret", "token")
        self.registry = {}
        monkeypatch.setattr(boto3, "Session", self._session)
        monkeypatch.setattr(mc, "_litellm_registry", lambda: self.registry)
        monkeypatch.setattr(BaseAWSLLM, "get_credentials", self._get_credentials)

    def _session(self, **kwargs):
        self.sessions.append(kwargs)
        return self

    def _get_credentials(self, **kwargs):
        self.credential_requests.append(kwargs)
        return self.credentials

    def client(self, service, config=None):
        assert service == "bedrock"
        self.configs.append(config)
        return self.bedrock

    def respond_foundation(self, *models):
        self.stub.add_response(
            "list_foundation_models", {"modelSummaries": list(models)}, {}
        )

    def respond_profiles(self, *profiles):
        self.stub.add_response(
            "list_inference_profiles",
            {"inferenceProfileSummaries": list(profiles)},
            {},
        )

    def deny(self, operation):
        self.stub.add_client_error(
            operation, service_error_code="AccessDeniedException"
        )


@pytest.fixture
def bedrock_aws(monkeypatch):
    for name in (
        "AWS_REGION_NAME",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_BEARER_TOKEN_BEDROCK",
    ):
        monkeypatch.delenv(name, raising=False)
    return _BedrockAWS(monkeypatch)


def test_bedrock_lists_profiles_then_directly_invokable_models(bedrock_aws):
    bedrock_aws.registry = {
        "global.anthropic.claude-opus-5-5": {
            "mode": "chat",
            "max_input_tokens": 1_000_000,
            "supports_function_calling": True,
            "input_cost_per_token": 5e-06,
            "output_cost_per_token": 2.5e-05,
        },
        "amazon.rerank-v2:0": {"mode": "rerank"},
    }
    nova_arn = "arn:aws:bedrock:us-east-2::foundation-model/amazon.nova-pro-v1:0"
    embed_arn = "arn:aws:bedrock:us-east-2::foundation-model/cohere.embed-v4:0"
    team_arn = "arn:aws:bedrock:us-east-2:1:application-inference-profile/a0a0"
    bedrock_aws.respond_foundation(
        _bedrock_model(
            "anthropic.claude-opus-5-5",
            inputModalities=["TEXT", "IMAGE"],
            inferenceTypesSupported=["INFERENCE_PROFILE"],
        ),
        _bedrock_model("amazon.nova-pro-v1:0"),
        _bedrock_model("amazon.rerank-v2:0"),
        _bedrock_model("cohere.embed-v4:0", outputModalities=["EMBEDDING"]),
        _bedrock_model("acme.big:0:28k", inferenceTypesSupported=["PROVISIONED"]),
        _bedrock_model("meta.old", modelLifecycle={"status": "LEGACY"}),
    )
    bedrock_aws.respond_profiles(
        _bedrock_profile("us.amazon.nova-pro-v1:0", arn=nova_arn),
        _bedrock_profile("global.anthropic.claude-opus-5-5"),
        _bedrock_profile("us.cohere.embed-v4:0", arn=embed_arn),
        _bedrock_profile("us.retired", status="INACTIVE"),
        _bedrock_profile(
            "a0a0",
            type="APPLICATION",
            inferenceProfileArn=team_arn,
            inferenceProfileName="team",
        ),
    )

    catalog = mc.list_models("bedrock", "us-east-2", aws_profile="prod", timeout=4.0)

    assert [e.id for e in catalog.entries] == [
        team_arn,
        "global.anthropic.claude-opus-5-5",
        "us.amazon.nova-pro-v1:0",
        "amazon.nova-pro-v1:0",
        "meta.old",
    ]
    assert catalog.source == "AWS Bedrock (us-east-2)"
    assert not catalog.exhaustive
    assert catalog.entries[0].display_name == "team"
    opus = catalog.entries[1]
    assert opus.display_name == "GLOBAL.ANTHROPIC.CLAUDE-OPUS-5-5"
    assert opus.context_length == 1_000_000
    assert opus.supports_tools is True
    assert opus.price_in == pytest.approx(5.0)
    assert opus.price_out == pytest.approx(25.0)
    assert "vision" in opus.tags
    assert "legacy" in catalog.entries[4].tags
    assert bedrock_aws.credential_requests == [
        {"aws_region_name": "us-east-2", "aws_profile_name": "prod"}
    ]
    assert bedrock_aws.sessions == [
        {
            "aws_access_key_id": "AKTEST",
            "aws_secret_access_key": "secret",
            "aws_session_token": "token",
            "region_name": "us-east-2",
        }
    ]
    config = bedrock_aws.configs[0]
    assert (config.connect_timeout, config.read_timeout) == (4.0, 4.0)
    assert config.retries == {"total_max_attempts": 1}
    bedrock_aws.stub.assert_no_pending_responses()


def test_bedrock_follows_inference_profile_pages(bedrock_aws):
    bedrock_aws.respond_foundation()
    bedrock_aws.stub.add_response(
        "list_inference_profiles",
        {
            "inferenceProfileSummaries": [_bedrock_profile("us.a")],
            "nextToken": "more",
        },
        {},
    )
    bedrock_aws.stub.add_response(
        "list_inference_profiles",
        {"inferenceProfileSummaries": [_bedrock_profile("us.b")]},
        {"nextToken": "more"},
    )
    catalog = mc.list_models("bedrock", "us-east-2")
    assert [e.id for e in catalog.entries] == ["us.a", "us.b"]
    bedrock_aws.stub.assert_no_pending_responses()


def test_bedrock_keeps_what_iam_allows(bedrock_aws):
    bedrock_aws.deny("list_foundation_models")
    bedrock_aws.respond_profiles(_bedrock_profile("us.a"))
    catalog = mc.list_models("bedrock", "us-east-2")
    assert [e.id for e in catalog.entries] == ["us.a"]

    bedrock_aws.deny("list_foundation_models")
    bedrock_aws.deny("list_inference_profiles")
    with pytest.raises(mc.CatalogUnavailable) as exc:
        mc.list_models("bedrock", "us-east-2", refresh=True)
    assert "bedrock:ListFoundationModels" in exc.value.hint
    assert "bedrock:ListInferenceProfiles" in exc.value.hint


def test_bedrock_failures_become_catalog_errors(bedrock_aws, monkeypatch):
    def reason():
        with pytest.raises(mc.CatalogUnavailable) as exc:
            mc.list_models("bedrock", "us-east-2")
        assert exc.value.hint is not None
        return exc.value.reason

    bedrock_aws.stub.add_client_error(
        "list_foundation_models",
        service_error_code="ExpiredTokenException",
        service_message="the token expired",
    )
    message = reason()
    assert "the token expired" in message and "us-east-2" in message
    bedrock_aws.stub.assert_no_pending_responses()

    bedrock_aws.respond_foundation(
        _bedrock_model("cohere.embed-v4:0", outputModalities=["EMBEDDING"])
    )
    bedrock_aws.respond_profiles()
    assert "no text models" in reason()

    bedrock_aws.credentials = None
    assert "Unable to locate credentials" in reason()

    def broken():
        raise RuntimeError("registry exploded")

    monkeypatch.setattr(mc, "_litellm_registry", broken)
    assert "registry exploded" in reason()


def test_bedrock_bearer_token_needs_no_aws_credentials(bedrock_aws, monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "token")
    bedrock_aws.credentials = None
    bedrock_aws.respond_foundation(_bedrock_model("amazon.nova-pro-v1:0"))
    bedrock_aws.respond_profiles()
    mc.list_models("bedrock", "us-east-2")
    assert bedrock_aws.sessions == [{"region_name": "us-east-2"}]


@pytest.mark.parametrize(
    "aws_profile, expected_key",
    [(None, "AKFROMENV"), ("explicit", "AKEXPLICIT")],
)
def test_bedrock_authenticates_like_chat_calls(
    monkeypatch, tmp_path, aws_profile, expected_key
):
    """boto3 alone ignores AWS_PROFILE_NAME, so this runs the real LiteLLM code."""
    import boto3

    credentials = tmp_path / "credentials"
    credentials.write_text(
        "[from-env]\naws_access_key_id = AKFROMENV\naws_secret_access_key = s\n"
        "[explicit]\naws_access_key_id = AKEXPLICIT\naws_secret_access_key = s\n"
    )
    for name in (
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_ROLE_NAME",
        "AWS_ROLE_ARN",
        "AWS_WEB_IDENTITY_TOKEN",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_BEARER_TOKEN_BEDROCK",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-config"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_PROFILE_NAME", "from-env")
    seen = {}

    def capture_client(session, service, config=None):
        seen["key"] = session.get_credentials().access_key
        raise RuntimeError("stop before any network call")

    monkeypatch.setattr(boto3.session.Session, "client", capture_client)
    with pytest.raises(mc.CatalogUnavailable):
        mc.list_models("bedrock", "us-east-2", aws_profile=aws_profile)
    assert seen["key"] == expected_key


def test_bedrock_custom_endpoint_is_not_listed(bedrock_aws):
    with pytest.raises(mc.CatalogUnavailable) as exc:
        mc.list_models("bedrock", "https://vpce-1.bedrock-runtime.example")
    assert "custom endpoint" in exc.value.reason
    assert exc.value.hint is not None
    assert bedrock_aws.sessions == []


def test_bedrock_lists_the_region_chat_calls_would_use(bedrock_aws, monkeypatch):
    monkeypatch.setenv("AWS_REGION_NAME", "eu-west-3")
    bedrock_aws.respond_foundation(_bedrock_model("amazon.nova-pro-v1:0"))
    bedrock_aws.respond_profiles()
    catalog = mc.list_models("bedrock")
    assert bedrock_aws.sessions[0]["region_name"] == "eu-west-3"
    assert catalog.source == "AWS Bedrock (eu-west-3)"


def test_bedrock_catalogs_are_kept_per_aws_profile(bedrock_aws):
    bedrock_aws.respond_foundation()
    bedrock_aws.respond_profiles(_bedrock_profile("us.a"))
    bedrock_aws.respond_foundation()
    bedrock_aws.respond_profiles(_bedrock_profile("us.b"))

    first = mc.list_models("bedrock", "us-east-2", aws_profile="a")
    second = mc.list_models("bedrock", "us-east-2", aws_profile="b")

    assert first.entries[0].id == "us.a"
    assert second.entries[0].id == "us.b"
    assert mc.cached_entries("bedrock", "us-east-2", "a")[0].id == "us.a"
    assert mc.cached_entries("bedrock", "us-east-2") is None


def test_aws_profile_does_not_split_the_cache_of_other_providers(monkeypatch):
    catalog = mc.Catalog([mc.ModelEntry(id="local")], "src")
    monkeypatch.setitem(mc._FETCHERS, "lmstudio", lambda *_: catalog)
    mc.list_models("lmstudio", aws_profile="left-over")
    assert mc.cached_entries("lmstudio", None)[0].id == "local"

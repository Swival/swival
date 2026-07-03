"""Provider model catalogs: list the models a provider can serve right now.

Every fetcher normalizes into :class:`ModelEntry` so the picker, the /model
command, and onboarding all consume one shape. Fetch failures raise
:class:`CatalogUnavailable`; callers degrade to manual model entry.

This module must stay import-light and must not import ``swival.agent``.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

HF_ROUTER_MODELS_URL = "https://router.huggingface.co/v1/models"
HF_HUB_MODELS_URL = "https://huggingface.co/api/models"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
GOOGLE_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"

LMSTUDIO_DEFAULT_BASE = "http://127.0.0.1:1234"
LLAMACPP_DEFAULT_BASE = "http://127.0.0.1:8080"

_CACHE_TTL_SECONDS = 300.0


class CatalogUnavailable(Exception):
    """A provider's model list could not be fetched.

    ``reason`` says what happened; ``hint`` tells the user what to do next.
    """

    def __init__(self, reason: str, hint: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.hint = hint


@dataclass(frozen=True)
class ModelEntry:
    """One selectable model, normalized across providers."""

    id: str
    display_name: str | None = None
    context_length: int | None = None
    supports_tools: bool | None = None
    price_in: float | None = None  # USD per million input tokens
    price_out: float | None = None  # USD per million output tokens
    loaded: bool | None = None  # LM Studio only
    tags: tuple[str, ...] = ()
    detail: str | None = None


@dataclass
class Catalog:
    entries: list[ModelEntry]
    source: str


_cache: dict[tuple[str, str | None], tuple[float, Catalog]] = {}


def normalize_provider(provider: str) -> str:
    """Collapse provider aliases (vertexai, mlx) onto their real providers."""
    if provider == "vertexai":
        return "geap"
    if provider == "mlx":
        return "generic"
    return provider


def supports_listing(provider: str) -> bool:
    """True when the provider's current model list can be enumerated.

    bedrock/geap/command have no fetcher: /model still accepts explicit ids
    for them.
    """
    return normalize_provider(provider) in _FETCHERS


def is_hf_router(provider: str, base_url: str | None) -> bool:
    """True when models come from HuggingFace's public inference-provider
    router (as opposed to a dedicated endpoint), which is where the explorer
    features (hub-wide search, exact-id status checks) apply."""
    return normalize_provider(provider) == "huggingface" and not base_url


def cached_entries(provider: str, base_url: str | None) -> list[ModelEntry] | None:
    """Return the cached entries for (provider, base_url) without any network.

    Used by TAB completion, which must never block on a fetch.
    """
    hit = _cache.get((normalize_provider(provider), base_url))
    if hit is None:
        return None
    return hit[1].entries


def clear_cache() -> None:
    _cache.clear()


def list_models(
    provider: str,
    base_url: str | None = None,
    api_key: str | None = None,
    *,
    timeout: float = 6.0,
    refresh: bool = False,
) -> Catalog:
    """Fetch the current model catalog for *provider*.

    Results are cached per (provider, base_url) for a few minutes so
    reopening the picker is instant; ``refresh=True`` bypasses the cache.
    Raises :class:`CatalogUnavailable` when the provider has no listing
    support or the fetch fails.
    """
    provider = normalize_provider(provider)
    key = (provider, base_url)
    if not refresh:
        hit = _cache.get(key)
        if hit is not None and (time.monotonic() - hit[0]) < _CACHE_TTL_SECONDS:
            return hit[1]

    fetcher = _FETCHERS.get(provider)
    if fetcher is None:
        raise CatalogUnavailable(
            f"model listing is not supported for provider {provider!r}",
            hint="pass a model id directly: /model <id>",
        )
    catalog = fetcher(base_url, api_key, timeout)
    _cache[key] = (time.monotonic(), catalog)
    return catalog


def _context_for(entries: list[ModelEntry], model_id: str) -> int | None:
    wanted = model_id.lower()
    for e in entries:
        if e.id.lower() == wanted:
            return e.context_length
    return None


def cached_context_length(
    provider: str, model_id: str, base_url: str | None = None
) -> int | None:
    """Context length for *model_id* from the already-fetched catalog.

    Never touches the network: returns None when nothing is cached for
    (provider, base_url) or the cached listing does not carry the model.
    """
    return _context_for(cached_entries(provider, base_url) or [], model_id)


def catalog_context_length(
    provider: str,
    model_id: str,
    base_url: str | None = None,
    api_key: str | None = None,
    *,
    timeout: float = 6.0,
) -> int | None:
    """Best-effort context length for *model_id* from the provider catalog.

    Fetches the catalog (or reuses the cache) and returns the listed context
    length, or None when the catalog is unavailable or does not carry it.
    """
    try:
        catalog = list_models(provider, base_url, api_key, timeout=timeout)
    except CatalogUnavailable:
        return None
    return _context_for(catalog.entries, model_id)


def search_hf_models(
    query: str,
    api_key: str | None = None,
    *,
    timeout: float = 8.0,
    limit: int = 30,
) -> list[ModelEntry]:
    """Search the whole HuggingFace hub for provider-served chat models.

    This is the long-tail escape hatch behind the router catalog: it asks the
    hub for models matching *query* that at least one inference provider
    serves, and keeps only the ones with a live conversational deployment.
    """
    params = urllib.parse.urlencode(
        {
            "inference_provider": "all",
            "search": query,
            "limit": str(limit),
            "expand[]": "inferenceProviderMapping",
        }
    )
    data = _get_json(f"{HF_HUB_MODELS_URL}?{params}", api_key, timeout)
    if not isinstance(data, list):
        raise CatalogUnavailable("unexpected response shape from the HuggingFace hub")

    entries = []
    for m in data:
        live = _live_conversational(m.get("inferenceProviderMapping"))
        if not live:
            continue
        entries.append(_entry_from_hub_mapping(m.get("id", ""), live))
    return entries


def _live_conversational(raw) -> list:
    """Normalize a hub inferenceProviderMapping and keep live chat deployments.

    The model-detail endpoint returns a dict keyed by provider name; the
    list endpoint returns a list of dicts with a "provider" field.
    """
    if isinstance(raw, dict):
        mapping = [dict(v, provider=k) for k, v in raw.items() if isinstance(v, dict)]
    else:
        mapping = [p for p in raw or [] if isinstance(p, dict)]
    return [
        p
        for p in mapping
        if p.get("status") == "live" and p.get("task") == "conversational"
    ]


def hf_model_status(
    model_id: str,
    api_key: str | None = None,
    *,
    timeout: float = 6.0,
) -> tuple[bool, ModelEntry | None, list[str]]:
    """Check one hub model's inference-provider availability.

    Returns ``(exists, entry, others)``: ``entry`` is a ready-to-use
    :class:`ModelEntry` when at least one live conversational deployment
    exists, and ``others`` describes the remaining deployments (e.g.
    ``["fireworks-ai (staging)"]``) for error messages.
    """
    quoted = urllib.parse.quote(model_id, safe="/")
    url = f"{HF_HUB_MODELS_URL}/{quoted}?expand[]=inferenceProviderMapping"
    try:
        data = _get_json(url, api_key, timeout)
    except CatalogUnavailable as e:
        if "HTTP 404" in e.reason or "HTTP 401" in e.reason:
            return False, None, []
        raise
    raw = data.get("inferenceProviderMapping") or []
    live = _live_conversational(raw)
    live_names = {p.get("provider") for p in live}
    if isinstance(raw, dict):
        rest = [(k, v.get("status")) for k, v in raw.items() if isinstance(v, dict)]
    else:
        rest = [
            (p.get("provider", "?"), p.get("status"))
            for p in raw
            if isinstance(p, dict)
        ]
    others = [
        f"{name} ({status or 'unknown'})"
        for name, status in rest
        if name not in live_names
    ]
    entry = _entry_from_hub_mapping(model_id, live) if live else None
    return True, entry, others


_VERSION_SUFFIX = re.compile(r"/v\d+$")


def _openai_base(url: str) -> str:
    """Ensure an OpenAI-compatible base URL carries an API version segment.

    A URL that already ends in a version segment is left alone: that covers
    the usual /v1 as well as providers that version differently, such as
    Z.AI serving from /api/paas/v4. (Mirrors agent._normalize_openai_base,
    duplicated because this module must not import swival.agent.)
    """
    stripped = url.rstrip("/")
    if _VERSION_SUFFIX.search(stripped):
        return stripped
    return f"{stripped}/v1"


def _get_json(url: str, api_key: str | None, timeout: float):
    headers = {"User-Agent": "swival"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise CatalogUnavailable(f"model list request to {url} failed: HTTP {e.code}")
    except (OSError, ValueError) as e:
        raise CatalogUnavailable(f"could not fetch model list from {url}: {e}")


def _as_price(value) -> float | None:
    """Coerce a provider pricing value to float, tolerating strings."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_ctx(n: int | None) -> str:
    if not n:
        return "?"
    if n >= 1_000_000:
        millions = f"{n / 1_000_000:.1f}".removesuffix(".0")
        return f"{millions}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


def _fmt_price(price_in: float | None, price_out: float | None) -> str:
    if price_in is None:
        return ""
    if price_out is None:
        return f"${price_in:g}"
    return f"${price_in:g}/${price_out:g}"


def _fetch_lmstudio(base_url: str | None, api_key: str | None, timeout: float):
    base = (base_url or LMSTUDIO_DEFAULT_BASE).rstrip("/")
    data = _get_json(f"{base}/api/v1/models", api_key, timeout)
    raw = data.get("data") or data.get("models") or []

    entries = []
    for m in raw:
        mtype = m.get("type")
        if mtype not in ("llm", "vlm"):
            continue
        model_id = m.get("id") or m.get("key")
        if not model_id:
            continue
        instances = m.get("loaded_instances") or []
        loaded = bool(instances) or m.get("state") == "loaded"
        context = None
        if instances:
            context = instances[0].get("config", {}).get("context_length")
        context = context or m.get("max_context_length")
        tags = []
        if mtype == "vlm":
            tags.append("vision")
        quant = m.get("quantization")
        if isinstance(quant, str) and quant:
            tags.append(quant)
        entries.append(
            ModelEntry(
                id=model_id,
                context_length=context,
                loaded=loaded,
                tags=tuple(tags),
            )
        )
    if not entries:
        raise CatalogUnavailable(
            f"LM Studio at {base} reports no downloaded LLMs",
            hint="download a model in LM Studio first",
        )
    entries.sort(key=lambda e: not e.loaded)
    return Catalog(entries, source=f"LM Studio at {base}")


def _fetch_llamacpp(base_url: str | None, api_key: str | None, timeout: float):
    base = (base_url or LLAMACPP_DEFAULT_BASE).rstrip("/").removesuffix("/v1")
    data = _get_json(f"{base}/v1/models", api_key, timeout)
    raw = data.get("data") or []
    context = _probe_llamacpp_context(base, api_key, timeout) if len(raw) == 1 else None
    entries = [
        ModelEntry(id=m["id"], context_length=context)
        for m in raw
        if isinstance(m.get("id"), str)
    ]
    if not entries:
        raise CatalogUnavailable(
            f"llama.cpp server at {base} reports no models",
            hint="check that llama-server is running with a model",
        )
    return Catalog(entries, source=f"llama.cpp at {base}")


def _probe_llamacpp_context(
    base: str, api_key: str | None, timeout: float
) -> int | None:
    try:
        props = _get_json(f"{base}/props", api_key, timeout)
    except CatalogUnavailable:
        return None
    for value in (
        props.get("default_generation_settings", {}).get("n_ctx"),
        props.get("n_ctx"),
    ):
        if isinstance(value, int) and value > 0:
            return value
    return None


_GENERIC_CONTEXT_KEYS = (
    "max_model_len",
    "max_context_length",
    "context_length",
    "context_window",
)


def _fetch_generic(base_url: str | None, api_key: str | None, timeout: float):
    if not base_url:
        raise CatalogUnavailable("the generic provider needs a base_url to list models")
    base = _openai_base(base_url)
    data = _get_json(f"{base}/models", api_key, timeout)
    raw = data.get("data") or []
    entries = []
    for m in raw:
        model_id = m.get("id")
        if not isinstance(model_id, str):
            continue
        context = next(
            (
                m[k]
                for k in _GENERIC_CONTEXT_KEYS
                if isinstance(m.get(k), int) and m[k] > 0
            ),
            None,
        )
        entries.append(ModelEntry(id=model_id, context_length=context))
    if not entries:
        raise CatalogUnavailable(f"the server at {base} reports no models")
    return Catalog(entries, source=f"server at {base}")


def _fetch_applefm(base_url: str | None, api_key: str | None, timeout: float):
    return _fetch_generic(base_url or "http://127.0.0.1:1976/v1", api_key, timeout)


def _hf_aggregate(
    live: list,
) -> tuple[int | None, float | None, float | None, bool | None]:
    """Aggregate context/pricing/tool support across a model's live providers."""
    contexts = [
        p["context_length"]
        for p in live
        if isinstance(p.get("context_length"), int) and p["context_length"] > 0
    ]
    context = max(contexts) if contexts else None

    priced = []
    for p in live:
        pricing = p.get("pricing") or {}
        pin = _as_price(pricing.get("input"))
        pout = _as_price(pricing.get("output"))
        if pin is not None:
            priced.append((pin, pout))
    if priced:
        # Explicit key: a bare min() would compare the output prices to break
        # input-price ties, and those can be None.
        price_in, price_out = min(
            priced, key=lambda t: (t[0], t[1] if t[1] is not None else float("inf"))
        )
    else:
        price_in, price_out = None, None

    tool_votes = [p["supports_tools"] for p in live if "supports_tools" in p]
    if any(tool_votes):
        tools = True
    elif tool_votes:
        tools = False
    else:
        tools = None
    return context, price_in, price_out, tools


def _hf_detail(live: list) -> str:
    parts = []
    for p in live:
        name = p.get("provider", "?")
        bits = [name]
        ctx = p.get("context_length")
        if isinstance(ctx, int) and ctx > 0:
            bits.append(_fmt_ctx(ctx))
        pricing = p.get("pricing") or {}
        pin = _as_price(pricing.get("input"))
        pout = _as_price(pricing.get("output"))
        if pin is not None and pout is not None:
            bits.append(_fmt_price(pin, pout))
        parts.append(" ".join(bits))
    return " · ".join(parts)


def _fetch_huggingface(base_url: str | None, api_key: str | None, timeout: float):
    if base_url:
        # A dedicated endpoint serves whatever it serves; try the standard
        # OpenAI-compatible listing and let the caller fall back to manual
        # entry when the endpoint does not expose one.
        return _fetch_generic(base_url, api_key, timeout)

    data = _get_json(HF_ROUTER_MODELS_URL, api_key, timeout)
    raw = data.get("data") or []
    entries = []
    for m in raw:
        model_id = m.get("id")
        if not isinstance(model_id, str):
            continue
        live = [p for p in m.get("providers") or [] if p.get("status") == "live"]
        if not live:
            continue
        modalities = (m.get("architecture") or {}).get("input_modalities") or []
        entries.append(_hf_entry(model_id, live, vision="image" in modalities))
    if not entries:
        raise CatalogUnavailable(
            "the HuggingFace router reports no models with live inference providers"
        )
    return Catalog(entries, source="HuggingFace inference providers")


def _hf_entry(model_id: str, live: list, *, vision: bool = False) -> ModelEntry:
    context, price_in, price_out, tools = _hf_aggregate(live)
    n = len(live)
    tags = ["1 provider" if n == 1 else f"{n} providers"]
    if any(p.get("is_free") for p in live):
        tags.append("free")
    if vision:
        tags.append("vision")
    return ModelEntry(
        id=model_id,
        context_length=context,
        supports_tools=tools,
        price_in=price_in,
        price_out=price_out,
        tags=tuple(tags),
        detail=_hf_detail(live),
    )


def _entry_from_hub_mapping(model_id: str, live: list) -> ModelEntry:
    """Build an entry from the hub API's inferenceProviderMapping shape."""
    flattened = []
    for p in live:
        details = p.get("providerDetails") or {}
        features = p.get("features") or {}
        flat = {
            "provider": p.get("provider"),
            "status": p.get("status"),
            "context_length": details.get("context_length"),
            "pricing": details.get("pricing"),
        }
        if "toolCalling" in features:
            flat["supports_tools"] = features["toolCalling"]
        flattened.append(flat)
    return _hf_entry(model_id, flattened)


def _fetch_openrouter(base_url: str | None, api_key: str | None, timeout: float):
    data = _get_json(OPENROUTER_MODELS_URL, api_key, timeout)
    raw = data.get("data") or []
    entries = []
    for m in raw:
        model_id = m.get("id")
        if not isinstance(model_id, str):
            continue
        pricing = m.get("pricing") or {}
        pin = _as_price(pricing.get("prompt"))
        pout = _as_price(pricing.get("completion"))
        # OpenRouter prices are per token; normalize to USD per Mtok.
        price_in = pin * 1_000_000 if pin is not None else None
        price_out = pout * 1_000_000 if pout is not None else None
        supported = m.get("supported_parameters") or []
        tags = []
        if model_id.endswith(":free") or (price_in == 0 and price_out == 0):
            tags.append("free")
        modalities = (m.get("architecture") or {}).get("input_modalities") or []
        if "image" in modalities:
            tags.append("vision")
        entries.append(
            ModelEntry(
                id=model_id,
                display_name=m.get("name"),
                context_length=m.get("context_length"),
                supports_tools="tools" in supported,
                price_in=price_in,
                price_out=price_out,
                tags=tuple(tags),
            )
        )
    if not entries:
        raise CatalogUnavailable("OpenRouter reports no models")
    return Catalog(entries, source="OpenRouter")


def _fetch_google(base_url: str | None, api_key: str | None, timeout: float):
    key = (
        api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    )
    if not key:
        raise CatalogUnavailable(
            "listing Gemini models requires an API key",
            hint="set GEMINI_API_KEY or api_key in your config",
        )
    base = (base_url or GOOGLE_OPENAI_BASE).rstrip("/")
    data = _get_json(f"{base}/models", key, timeout)
    raw = data.get("data") or []
    entries = [
        ModelEntry(id=m["id"].removeprefix("models/"))
        for m in raw
        if isinstance(m.get("id"), str)
    ]
    if not entries:
        raise CatalogUnavailable("the Gemini endpoint reports no models")
    return Catalog(entries, source="Google Gemini")


def _fetch_chatgpt(base_url: str | None, api_key: str | None, timeout: float):
    """List ChatGPT-backend models from litellm's local cost registry.

    There is no public listing endpoint for the ChatGPT OAuth backend, but
    litellm ships a registry of the models it can route, which tracks the
    supported set better than a hardcoded list here would.
    """
    try:
        # Use litellm's bundled cost map rather than fetching it remotely,
        # matching agent._import_litellm.
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        import litellm

        registry = litellm.model_cost
    except Exception as e:
        raise CatalogUnavailable(f"could not load the litellm model registry: {e}")

    entries = []
    for key, info in sorted(registry.items()):
        if not key.startswith("chatgpt/"):
            continue
        bare = key.removeprefix("chatgpt/")
        if bare.startswith("responses/"):
            continue
        entries.append(
            ModelEntry(
                id=bare,
                context_length=info.get("max_input_tokens"),
                supports_tools=info.get("supports_function_calling"),
            )
        )
    if not entries:
        raise CatalogUnavailable(
            "no ChatGPT models found in the litellm registry",
            hint="pass a model id directly: /model <id>",
        )
    return Catalog(entries, source="known ChatGPT models")


_FETCHERS = {
    "lmstudio": _fetch_lmstudio,
    "llamacpp": _fetch_llamacpp,
    "generic": _fetch_generic,
    "applefm": _fetch_applefm,
    "huggingface": _fetch_huggingface,
    "openrouter": _fetch_openrouter,
    "google": _fetch_google,
    "chatgpt": _fetch_chatgpt,
}

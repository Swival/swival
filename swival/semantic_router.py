"""Semantic routing: ask a router LLM which profile best fits the current task."""

import json
import re
import time
from dataclasses import dataclass

from . import fmt
from ._msg import _msg_content
from .report import AgentError

_REASON_MAX_LEN = 120

_MISSING = object()

_ROUTER_CHAR_BUDGET = 12_000  # ~3000 tokens at ~4 chars/token


@dataclass
class RoutingResult:
    """Outcome of a semantic routing decision."""

    router_profile: str
    selected_profile: str
    reason: str | None
    fallback_used: bool
    parse_mode: (
        str  # "json", "bare_name", "embedded_name", "fallback", "short_circuit", "none"
    )


# --- Prompt building ---


def _build_router_prompt(
    task_text: str,
    candidates: dict[str, str],
    *,
    base_dir_name: str = "",
    is_objective: bool = False,
) -> list[dict]:
    """Build messages for the router LLM call.

    *candidates* maps profile name -> description.
    """
    profile_list = "\n".join(
        f"- {name}: {desc}" for name, desc in sorted(candidates.items())
    )

    truncated = ""
    task_body = task_text
    if len(task_text) > _ROUTER_CHAR_BUDGET:
        task_body = task_text[:_ROUTER_CHAR_BUDGET]
        truncated = "\n[task truncated]"

    context_parts = []
    if base_dir_name:
        context_parts.append(f"Project: {base_dir_name}")
    if is_objective:
        context_parts.append("Mode: objective (multi-step autonomous run)")
    context_line = ("\n".join(context_parts) + "\n") if context_parts else ""

    system = (
        "You are a routing assistant. Given a task, choose which profile is the best fit.\n"
        'Reply with ONLY a JSON object: {"profile": "<name>", "reason": "<short reason>"}\n'
        "No extra text."
    )

    user = (
        f"{context_line}"
        f"Available profiles:\n{profile_list}\n\n"
        f"Task:\n{task_body}{truncated}\n\n"
        f"Choose the best profile."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# --- Response parsing ---


def _parse_router_response(
    text: str,
    valid_names: set[str],
) -> tuple[str | None, str | None, str]:
    """Try to extract a profile name from the router response.

    Returns (profile_name_or_None, reason_or_None, parse_mode).
    """
    text = text.strip()

    try:
        start = text.index("{")
        depth = 0
        end = start
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        obj = json.loads(text[start:end])
        name = obj.get("profile")
        reason = obj.get("reason")
        if isinstance(name, str) and name in valid_names:
            return name, reason if isinstance(reason, str) else None, "json"
    except (ValueError, json.JSONDecodeError):
        pass

    bare = text.strip().strip('"').strip("'")
    if bare in valid_names:
        return bare, None, "bare_name"

    # Only scan short responses to avoid false positives in long prose
    if len(text) < 200:
        for name in valid_names:
            if re.search(rf"\b{re.escape(name)}\b", text):
                return name, None, "embedded_name"

    return None, None, "none"


# --- Main routing function ---


def route_task(
    task_text: str,
    config: dict,
    *,
    verbose: bool = False,
    base_dir_name: str = "",
    is_objective: bool = False,
    cli_overrides: dict | None = None,
    llm_filter: str | None = None,
    secret_shield=None,
) -> RoutingResult | None:
    """Run semantic routing and return the result, or None if routing is not configured.

    *cli_overrides* supplies CLI-explicit values (api_key, base_url,
    aws_profile, extra_body, reasoning_effort, sanitize_thinking, retries)
    that override file config when the router profile body does not set them.

    *llm_filter* and *secret_shield* thread outbound protections through
    to the router ``call_llm`` call.

    Raises AgentError on unrecoverable failures when strict mode is on.
    """
    if not config.get("routing_enabled"):
        return None

    router_name = config.get("semantic_routing_profile")
    strict = config.get("routing_strict", False)
    profiles = config.get("profiles", {})
    overrides = cli_overrides or {}

    # Build candidate set: profiles with a description are routing targets.
    candidates: dict[str, str] = {}
    for name, body in profiles.items():
        desc = body.get("description")
        if desc:
            candidates[name] = desc

    if not candidates:
        raise AgentError(
            "routing is enabled but no profiles have a 'description' to route to"
        )

    # Short-circuit: only one candidate — select it directly, no LLM call
    if len(candidates) == 1:
        only_name = next(iter(candidates))
        if verbose:
            fmt.info(f"Routing: single target {only_name}, skipping router call")
        return RoutingResult(
            router_profile=router_name,
            selected_profile=only_name,
            reason=None,
            fallback_used=False,
            parse_mode="short_circuit",
        )

    active_profile = config.get("active_profile")

    def _fallback_profile() -> str:
        return active_profile if active_profile else router_name

    def _make_fallback(reason: str) -> RoutingResult:
        fb = _fallback_profile()
        if verbose:
            fmt.info(f"warning: {reason}, falling back to {fb!r}")
        return RoutingResult(
            router_profile=router_name,
            selected_profile=fb,
            reason=None,
            fallback_used=True,
            parse_mode="fallback",
        )

    # Resolve router profile's provider/model.
    # Priority: router profile body > CLI override > file config.
    router_body = profiles[router_name]

    def _resolve_key(key: str, default=None):
        for source in (router_body, overrides, config):
            v = source.get(key, _MISSING)
            if v is not _MISSING:
                return v
        return default

    router_provider = _resolve_key("provider", "lmstudio")
    router_model = _resolve_key("model")
    router_api_key = _resolve_key("api_key")
    router_base_url = _resolve_key("base_url")
    router_aws_profile = _resolve_key("aws_profile")

    from . import agent as _agent

    try:
        model_id, api_base, resolved_key, _ctx_len, llm_kwargs = (
            _agent.resolve_provider(
                router_provider,
                router_model,
                router_api_key,
                router_base_url,
                None,  # max_context_tokens
                verbose,
                aws_profile=router_aws_profile,
            )
        )
    except Exception as e:
        if strict:
            raise AgentError(f"Semantic routing setup failed: {e}") from e
        return _make_fallback(f"semantic routing setup failed: {e}")

    router_extra_body = _resolve_key("extra_body")
    router_reasoning = _resolve_key("reasoning_effort")
    router_sanitize = _resolve_key("sanitize_thinking")
    _retries_raw = _resolve_key("retries")
    router_retries = _retries_raw if _retries_raw is not None else 5

    messages = _build_router_prompt(
        task_text,
        candidates,
        base_dir_name=base_dir_name,
        is_objective=is_objective,
    )

    # Pre-build kwargs that don't change between first attempt and retry
    base_call_kwargs = {
        k: v for k, v in llm_kwargs.items() if k not in ("provider", "api_key")
    }
    if router_extra_body is not None:
        base_call_kwargs["extra_body"] = router_extra_body
    if router_reasoning is not None:
        base_call_kwargs["reasoning_effort"] = router_reasoning
    if router_sanitize is not None:
        base_call_kwargs["sanitize_thinking"] = router_sanitize
    base_call_kwargs["max_retries"] = router_retries
    if llm_filter is not None:
        base_call_kwargs["llm_filter"] = llm_filter
    if secret_shield is not None:
        base_call_kwargs["secret_shield"] = secret_shield

    def _attempt(msgs):
        t0 = time.monotonic()
        resp, _finish, _cmd, _retries, _cache = _agent.call_llm(
            api_base,
            model_id,
            msgs,
            256,  # max_output_tokens
            0,  # temperature
            1.0,  # top_p
            None,  # seed
            None,  # tools
            verbose,
            provider=router_provider,
            api_key=resolved_key,
            call_kind="router",
            **base_call_kwargs,
        )
        elapsed = time.monotonic() - t0
        return _msg_content(resp), elapsed

    valid_names = set(candidates)

    # First attempt
    try:
        response_text, elapsed = _attempt(messages)
    except Exception as e:
        if strict:
            raise AgentError(f"Semantic routing failed: {e}") from e
        return _make_fallback(f"semantic routing call failed: {e}")

    name, reason, parse_mode = _parse_router_response(response_text, valid_names)

    # Retry once if parsing failed
    if name is None:
        retry_msg = messages + [
            {"role": "assistant", "content": response_text},
            {
                "role": "user",
                "content": (
                    "Invalid response. Reply with ONLY a JSON object:\n"
                    '{"profile": "<name>", "reason": "<short reason>"}\n'
                    f"Valid profile names: {', '.join(sorted(candidates))}"
                ),
            },
        ]
        try:
            response_text, elapsed2 = _attempt(retry_msg)
            elapsed += elapsed2
        except Exception as e:
            if strict:
                raise AgentError(f"Semantic routing retry failed: {e}") from e
            return _make_fallback(f"semantic routing retry failed: {e}")

        name, reason, parse_mode = _parse_router_response(response_text, valid_names)

    if name is None:
        if strict:
            raise AgentError(
                f"Semantic routing failed: could not parse a valid profile name "
                f"from router response after retry. Response: {response_text!r}"
            )
        return _make_fallback("semantic routing could not parse response")

    if verbose:
        parts = [
            "mode=semantic",
            f"router={router_name}",
            f"selected={name}",
            f"duration={elapsed:.1f}s",
            f"parse={parse_mode}",
        ]
        if reason:
            # Collapse to single line and cap length for clean log output
            clean = " ".join(reason.split())
            if len(clean) > _REASON_MAX_LEN:
                clean = clean[:_REASON_MAX_LEN] + "..."
            parts.append(f"reason={clean}")
        fmt.info(f"Routing: {' '.join(parts)}")

    return RoutingResult(
        router_profile=router_name,
        selected_profile=name,
        reason=reason,
        fallback_used=False,
        parse_mode=parse_mode,
    )

"""Decision model requests, for the optional ``decide`` tool and model routing.

A decision model answers questions about a short text with probabilities.
Requests use LiteLLM's System One API and go through the same filter,
secret encryption, and cost tracking as chat requests.
"""

import json
import math
from dataclasses import dataclass, field

from .cost import price_observation
from .report import ConfigError

# The server silently cuts the end of an input that is too long.
# Questions and labels take more room than their bytes suggest, so they are
# weighted to keep every request safely under its limit.
MAX_REQUEST_UNITS = 15_000
QUESTION_UNITS = 40
OPTION_UNITS = 10

MAX_TIMEOUT = 30

# The server rounds each probability, so a full distribution can miss 1 a bit.
SUM_TOLERANCE = 0.02

_LOCAL_EXAMPLE = (
    "--decision-model clef-flash-oQ4e --decision-base-url http://127.0.0.1:8000/v1"
)
QUESTION_TYPES = ("noul", "choice", "score")


@dataclass(frozen=True)
class DecisionSettings:
    model: str
    api_base: str | None = None
    api_key: str | None = field(default=None, repr=False)

    @property
    def provider(self) -> str:
        return self.model.partition("/")[0]

    def _provider_config(self):
        from litellm.types.utils import LlmProviders
        from litellm.utils import ProviderConfigManager

        return ProviderConfigManager.get_provider_decisions_config(
            model="", provider=LlmProviders(self.provider)
        )

    def endpoint(self) -> str | None:
        if self.api_base is not None:
            return self.api_base
        return self._provider_config().resolve_api_base(None)

    def default_service(self) -> bool:
        """Whether requests go to the provider's own service.

        LiteLLM's prices only apply there.
        """
        if self.api_base is not None:
            return False
        config = self._provider_config()
        default = config.get_default_api_base()
        return default is not None and config.resolve_api_base(None) == default

    def startup_line(self) -> str:
        return f"decision model={self.model}  endpoint={self.endpoint()}"


def _normalize_base_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        raise ConfigError(
            f"decision_base_url must start with http:// or https://, got {url!r}"
        )
    # LiteLLM adds the /v1/systemone path itself.
    if url.endswith("/v1/systemone"):
        url = url.removesuffix("/systemone")
    return url


def resolve_settings(
    model: str | None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> DecisionSettings | None:
    """Return the decision model settings, or None when the tool is off.

    An empty model turns the tool off.
    A server the user named never receives a key from the environment.
    """
    model = (model or "").strip()
    if not model:
        return None

    from litellm.decisions.call import supported_decisions_providers

    prefix, sep, rest = model.partition("/")
    if not (sep and rest and prefix in supported_decisions_providers()):
        if not base_url:
            raise ConfigError(
                f"decision model {model!r} has no LiteLLM decision provider prefix "
                "(such as typesafe/), so it needs a server URL, for example: "
                f"{_LOCAL_EXAMPLE}"
            )
        model = f"hosted_vllm/{model}"

    api_base = _normalize_base_url(base_url) if base_url else None
    if not api_key and api_base is not None:
        from litellm.constants import PLACEHOLDER_API_KEY

        api_key = PLACEHOLDER_API_KEY
    settings = DecisionSettings(model=model, api_base=api_base, api_key=api_key or None)
    if settings.endpoint() is None:
        reason = settings._provider_config().missing_api_base_message(settings.provider)
        raise ConfigError(
            f"decision model {model!r} has no endpoint ({reason}); "
            "set decision_base_url (--decision-base-url)"
        )
    return settings


def _compact(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def encrypt_envelope(envelope: dict, secret_shield) -> dict:
    """The envelope as request() sends it, with its secrets encrypted."""
    return _parse_envelope(secret_shield.encrypt_text(_compact(envelope)))


def _parse_envelope(text: str) -> dict:
    envelope = json.loads(text)
    if not isinstance(envelope, dict) or set(envelope) != {"state", "questions"}:
        raise ValueError("expected a JSON object with only 'state' and 'questions'")
    return envelope


def _filter_envelope(llm_filter: str, envelope: dict, settings) -> dict:
    from .filter import FilterError, run_llm_filter

    messages = run_llm_filter(
        llm_filter,
        [{"role": "user", "content": _compact(envelope)}],
        model=settings.model,
        provider=settings.provider,
        call_kind="decision",
    )
    if (
        len(messages) != 1
        or messages[0].get("role") != "user"
        or not isinstance(messages[0].get("content"), str)
    ):
        raise FilterError("filter must return one user message holding the request")
    try:
        return _parse_envelope(messages[0]["content"])
    except ValueError as e:
        raise FilterError(f"filter returned an invalid decision request: {e}") from e


def _describe_invalid(error) -> str:
    problems: list[str] = []
    for err in error.errors():
        loc = list(err["loc"])
        if loc[:1] == ["state"]:
            text = "state: must be text, a JSON object, or an array"
        else:
            # Drop the question type that pydantic adds to the error location.
            if len(loc) > 2 and loc[0] == "questions" and loc[2] in QUESTION_TYPES:
                del loc[2]
            text = f"{'.'.join(map(str, loc))}: {err['msg']}"
        if text not in problems:
            problems.append(text)
    return "; ".join(problems[:5])


def _units(payload: str, questions: int, options: int) -> int:
    return (
        len(payload.encode("utf-8", "surrogatepass"))
        + QUESTION_UNITS * questions
        + OPTION_UNITS * options
    )


def request_units(questions, payload: str) -> int:
    """Weighted size of a request, compared with MAX_REQUEST_UNITS."""
    options = sum(
        len(question.criteria)
        for question in questions.values()
        if question.type != "noul"
    )
    return _units(payload, len(questions), options)


def envelope_units(envelope: dict) -> int:
    """request_units() for an envelope that has not been validated yet."""
    questions = envelope["questions"]
    options = sum(
        len(question.get("criteria") or ())
        for question in questions.values()
        if question.get("type") != "noul"
    )
    return _units(_compact(envelope), len(questions), options)


def _number(name: str, field_name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: {field_name} is not a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name}: {field_name} is not finite")
    return value


def _probability(name: str, field_name: str, value) -> float:
    value = _number(name, field_name, value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name}: {field_name} {value} is outside [0, 1]")
    return value


def _distribution(name: str, probabilities, expected: set[str]) -> dict:
    if not isinstance(probabilities, dict) or not probabilities:
        raise ValueError(f"{name}: no probabilities")
    unknown = sorted(set(probabilities) - expected)
    if unknown:
        raise ValueError(f"{name}: probabilities for unrequested {unknown}")
    missing = sorted(expected - set(probabilities))
    if missing:
        raise ValueError(f"{name}: no probability for {missing}")
    result = {
        key: _probability(name, f"probability of {key!r}", value)
        for key, value in probabilities.items()
    }
    # Catches score levels LiteLLM filled with zeros, but only when that leaves
    # the total short of 1.
    total = sum(result.values())
    if abs(total - 1.0) > SUM_TOLERANCE:
        raise ValueError(f"{name}: probabilities add up to {total:.3f}, not 1")
    return result


def validated_answers(questions, answers) -> dict:
    """Check the answers against the questions that were sent.

    These are the answers after LiteLLM's conversion, not what the server sent.
    LiteLLM drops refused or mistyped answers, so they show up here as missing,
    and it fills in score levels and legend entries the server left out,
    so this cannot prove that the server sent every field.
    """
    missing = [name for name in questions if name not in answers]
    if missing:
        raise ValueError(f"no answer for {', '.join(missing)}")
    result = {}
    for name, question in questions.items():
        answer = answers[name].model_dump(mode="json")
        if answer.get("type") != question.type:
            raise ValueError(
                f"{name}: expected a {question.type} answer, got {answer.get('type')}"
            )
        if question.type == "noul":
            result[name] = {
                "type": "noul",
                "noul": _probability(name, "noul", answer.get("noul")),
            }
        elif question.type == "choice":
            labels = set(question.criteria)
            if answer.get("choice") not in labels:
                raise ValueError(
                    f"{name}: choice {answer.get('choice')!r} is not a requested label"
                )
            result[name] = {
                "type": "choice",
                "choice": answer["choice"],
                "confidence": _probability(
                    name, "confidence", answer.get("confidence")
                ),
                "probabilities": _distribution(
                    name, answer.get("probabilities"), labels
                ),
            }
        else:
            criteria = question.model_dump(mode="json")["criteria"]
            legend = {str(i): level for i, level in enumerate(criteria)}
            top = len(criteria) - 1
            score = _number(name, "score", answer.get("score"))
            if not 0.0 <= score <= top:
                raise ValueError(f"{name}: score {score} is outside [0, {top}]")
            if answer.get("legend") != legend:
                raise ValueError(f"{name}: legend does not match the requested levels")
            result[name] = {
                "type": "score",
                "score": score,
                "confidence": _probability(
                    name, "confidence", answer.get("confidence")
                ),
                "legend": legend,
                "probabilities": _distribution(
                    name, answer.get("probabilities"), set(legend)
                ),
            }
    return result


def _cost_observation(response, settings: DecisionSettings, usage):
    from litellm.litellm_core_utils.core_helpers import RESPONSE_COST_HEADER

    hidden = getattr(response, "hidden_params", None) or {}
    headers = hidden.get("additional_headers") or {}
    cost = headers.get(RESPONSE_COST_HEADER)
    if cost is None and settings.default_service():
        cost = hidden.get("response_cost")
    return price_observation(cost, has_usage=bool(usage))


def _usage(response) -> dict | None:
    usage = getattr(response, "usage", None)
    if usage is None or not (usage.input_tokens or usage.output_tokens):
        return None
    return {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}


def _exposure_usage(response) -> dict | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "prompt_tokens_details": {"cached_tokens": usage.cached_tokens},
        "cache_creation_input_tokens": usage.cache_write_tokens,
    }


class DecisionFailure(Exception):
    """A decision request that produced no usable answers."""


class DecisionUnavailable(DecisionFailure):
    """The service could not be reached, timed out, or said it was unavailable."""


class DecisionCancelled(DecisionFailure):
    """The cancel flag was set before the answers were ready."""


class DecisionBlocked(DecisionFailure):
    """The outbound filter refused the request, so nothing was sent."""


class DecisionNotSent(DecisionFailure):
    """Secret encryption broke the request, or it is over the hard size cap."""


class DecisionError(DecisionFailure):
    """The request was invalid, failed, or came back with unusable answers."""


@dataclass(frozen=True)
class DecisionResult:
    model: str
    answers: dict
    usage: dict | None = None


def request(
    settings: DecisionSettings,
    envelope: dict,
    *,
    llm_filter: str | None = None,
    secret_shield=None,
    session_cost=None,
    exposure=None,
    timeout: float = MAX_TIMEOUT,
    cancel_flag=None,
) -> DecisionResult:
    """Send one decision request and return its validated answers.

    *envelope* holds ``state`` and ``questions``.
    Each way of failing raises its own ``DecisionFailure`` subclass.
    Error messages may still hold encrypted secrets; callers that show them
    must pass them through ``secret_shield.reverse_known()``.
    """

    def check_cancelled() -> None:
        if cancel_flag is not None and cancel_flag.is_set():
            raise DecisionCancelled("decide cancelled")

    check_cancelled()

    from pydantic import ValidationError

    from .filter import FilterError
    from litellm.types.decisions import DecisionsRequestBody

    envelope = {"state": envelope.get("state"), "questions": envelope.get("questions")}
    if llm_filter is not None:
        try:
            envelope = _filter_envelope(llm_filter, envelope, settings)
        except FilterError as e:
            raise DecisionBlocked(
                f"LLM filter blocked the decision request: {e}"
            ) from e
    if secret_shield is not None:
        try:
            envelope = encrypt_envelope(envelope, secret_shield)
        except ValueError as e:
            raise DecisionNotSent(
                f"secret encryption broke the decision request: {e}"
            ) from e

    try:
        body = DecisionsRequestBody.model_validate(envelope)
    except ValidationError as e:
        raise DecisionError(f"invalid decide arguments: {_describe_invalid(e)}") from e

    payload = _compact(envelope)
    units = request_units(body.questions, payload)
    if units > MAX_REQUEST_UNITS:
        raise DecisionNotSent(
            f"decide request too large ({units} of {MAX_REQUEST_UNITS} units: "
            f"JSON bytes + {QUESTION_UNITS} per question + {OPTION_UNITS} per label "
            "or level). Shorten the state, or ask fewer questions."
        )

    import litellm

    litellm.suppress_debug_info = True
    # Only count requests that are actually sent.
    check_cancelled()
    if exposure is not None:
        exposure.request([{"role": "user", "content": payload}])
    try:
        response = litellm.systemone(
            model=settings.model,
            api_base=settings.api_base,
            api_key=settings.api_key,
            state=envelope["state"],
            questions=envelope["questions"],
            num_retries=0,
            timeout=timeout,
        )
    except Exception as e:
        check_cancelled()
        if isinstance(
            e,
            (
                litellm.APIConnectionError,
                litellm.Timeout,
                litellm.ServiceUnavailableError,
            ),
        ):
            raise DecisionUnavailable(str(e)) from e
        raise DecisionError(f"decision model request failed: {e}") from e

    # Record the cost before checking the answers, since the service may bill
    # for them anyway.
    usage = _usage(response)
    observation = _cost_observation(response, settings, usage)
    if session_cost is not None:
        session_cost.record(observation)
    if exposure is not None:
        exposure.response(_exposure_usage(response), observation)
    check_cancelled()

    try:
        answers = validated_answers(body.questions, response.answers)
    except ValueError as e:
        raise DecisionError(f"decision model returned an unusable response: {e}") from e

    return DecisionResult(
        model=response.model or settings.model, answers=answers, usage=usage
    )


def decide(
    settings: DecisionSettings,
    args: dict,
    *,
    llm_filter: str | None = None,
    secret_shield=None,
    session_cost=None,
    exposure=None,
    timeout: float = MAX_TIMEOUT,
    cancel_flag=None,
) -> str:
    """Run one ``decide`` call and return the tool result.

    Errors come back as ``error:`` strings,
    so the chat model can explain them and continue.
    """

    def reveal(text: str) -> str:
        return secret_shield.reverse_known(text) if secret_shield else text

    try:
        result = request(
            settings,
            args,
            llm_filter=llm_filter,
            secret_shield=secret_shield,
            session_cost=session_cost,
            exposure=exposure,
            timeout=timeout,
            cancel_flag=cancel_flag,
        )
    except DecisionUnavailable as e:
        return reveal(
            "warning: decision model unavailable. Continue this session without "
            "the decision model; do not retry or invent model probabilities. "
            f"Details: {e}"
        )
    except (DecisionCancelled, DecisionBlocked, DecisionNotSent) as e:
        return f"error: {e}"
    except DecisionError as e:
        return reveal(f"error: {e}")

    out = {"model": result.model, "answers": result.answers}
    if result.usage is not None:
        out["usage"] = result.usage
    return reveal(_compact(out))

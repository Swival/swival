"""Pick the profile best suited for each task.

The decision model reads a short summary of the conversation and the
profile descriptions.
Its probabilities decide whether the task moves to another profile.

Only choose_profile() sends a request.
Resolving and switching to a profile is up to the caller.
"""

import json
import time
from dataclasses import dataclass, field

from . import decisions
from ._msg import (
    COMPACTION_MARKER,
    RECAP_MARKER,
    _estimate_tokens,
    _is_synthetic,
    _msg_content,
    _msg_get,
    _msg_role,
    _msg_tool_calls,
)
from .snapshot import SNAPSHOT_RECAP_PREFIX

# How much more likely another profile must be before switching to it.
# Anything from 0.2 to 0.4 gave the same picks on the test cases.
MARGIN = 0.3
# 0.7 - 0.4 comes out just under 0.3 in floating point.
_MARGIN_TOLERANCE = 1e-9

# Size target for the routing request, to keep it fast.
STATE_BUDGET = 6_000

QUESTION = (
    "Pick the model best suited for the latest request, taking the earlier "
    "conversation into account. Prefer the cheapest model that can do it well."
)

REQUEST_LIMIT, REQUEST_HEAD = 2_000, 800
REQUEST_FLOOR, REQUEST_FLOOR_HEAD = 1_000, 400
PREVIOUS_USER_LIMIT, PREVIOUS_USER_HEAD = 400, 160
ASSISTANT_LIMIT, ASSISTANT_HEAD = 1_200, 400
ASSISTANT_FLOOR, ASSISTANT_FLOOR_HEAD = 600, 200
EARLIER_USER_LIMIT = 200
EARLIER_ASSISTANT_TAIL = 300
EARLIER_TURNS = 2
GOAL_LIMIT = 300
MAX_TOOLS = 8
TOOL_LIMIT = 100

_SUMMARY_PREFIXES = (RECAP_MARKER, COMPACTION_MARKER, SNAPSHOT_RECAP_PREFIX)
_MAIN_ARGS = (
    "file_path",
    "image_path",
    "command",
    "url",
    "query",
    "pattern",
    "path",
    "task",
    "name",
    "action",
)


def _cut(text: str, limit: int, head: int) -> str:
    """Keep the first *head* and the last ``limit - head`` characters."""
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return (
        f"{text[:head]}\n[... {omitted:,} characters omitted ...]\n"
        f"{text[len(text) - (limit - head) :]}"
    )


def _tail(text: str, limit: int) -> str:
    return text if len(text) <= limit else "[...] " + text[-limit:]


def _image_parts(content) -> int:
    if not isinstance(content, list):
        return 0
    return sum(
        1 for p in content if isinstance(p, dict) and p.get("type") == "image_url"
    )


def is_request(msg) -> bool:
    """True for a message the user typed, not a recap or an injected note."""
    if _msg_role(msg) != "user":
        return False
    text = _msg_content(msg)
    return (
        bool(text) and not _is_synthetic(msg) and not text.startswith(_SUMMARY_PREFIXES)
    )


def _tool_summary(tc, protect) -> str | None:
    fn = _msg_get(tc, "function")
    name, raw = _msg_get(fn, "name"), _msg_get(fn, "arguments")
    if not name:
        return None
    # Protect the raw arguments, as the chat request does, and the summary
    # again below: parsing or collapsing whitespace can hide a secret.
    if isinstance(raw, str):
        raw = protect(raw)
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        args = None
    arg = ""
    if isinstance(args, dict):
        for key in _MAIN_ARGS:
            value = args.get(key)
            if isinstance(value, list):
                value = " ".join(v for v in value if isinstance(v, str))
            if isinstance(value, str) and value.strip():
                arg = value
                break
        else:
            files = args.get("files")
            if isinstance(files, list):
                arg = " ".join(
                    f["file_path"]
                    for f in files
                    if isinstance(f, dict) and isinstance(f.get("file_path"), str)
                )
    return " ".join(protect(f"{name} {arg}").split())[:TOOL_LIMIT]


@dataclass
class _Turn:
    user: str
    tools: list = field(default_factory=list)
    failed: int = 0
    assistant: str = ""


def _last_reply(messages: list) -> str:
    for m in reversed(messages):
        if _msg_role(m) == "assistant" and _msg_content(m).strip():
            return _msg_content(m)
    return ""


def _turns(messages: list, protect) -> list[_Turn]:
    """The last turns of the transcript, each starting with something the user typed.

    Assistant text before the first one, such as a recap that replaced the
    request, becomes a turn with no user message.
    Only what the state uses is built: each turn's last reply, and the tools
    of the last turn.
    *protect* runs on each whole text before anything is cut from it.
    """
    starts = [i for i, m in enumerate(messages) if is_request(m)]
    kept = starts[-1 - EARLIER_TURNS :]
    turns = []
    if len(starts) <= EARLIER_TURNS:
        lead = _last_reply(messages[: starts[0] if starts else len(messages)])
        if lead:
            turns.append(_Turn(user="", assistant=protect(lead)))
    for start, end in zip(kept, kept[1:] + [len(messages)]):
        span = messages[start + 1 : end]
        reply = _last_reply(span)
        turn = _Turn(user=protect(_msg_content(messages[start])))
        turn.assistant = protect(reply) if reply else ""
        if end == len(messages):
            for m in span:
                role = _msg_role(m)
                if role == "assistant":
                    for tc in _msg_tool_calls(m) or ():
                        summary = _tool_summary(tc, protect)
                        if summary and summary not in turn.tools:
                            turn.tools.append(summary)
                elif role == "tool":
                    # A failed build or test run only shows as a nonzero exit code.
                    code = _msg_get(m, "_swival_exit_code")
                    if _msg_content(m).startswith("error:") or code not in (None, 0):
                        turn.failed += 1
        turns.append(turn)
    return turns


def routing_questions(criteria: dict) -> dict:
    return {
        "profile": {
            "type": "choice",
            "instructions": QUESTION,
            "criteria": dict(criteria),
        }
    }


def _render(text, size, extras, turns, level) -> dict:
    """Build the state, leaving out more as *level* goes from 0 to 6."""
    if level >= 6:
        limit, head = REQUEST_FLOOR, REQUEST_FLOOR_HEAD
    else:
        limit, head = REQUEST_LIMIT, REQUEST_HEAD
    state = {"request": _cut(text, limit, head)}
    if len(text) > limit:
        state["request_size"] = size
    state.update(extras)
    if not turns:
        return state

    last = turns[-1]
    previous = {}
    if level < 2 and last.user:
        previous["user"] = _cut(last.user, PREVIOUS_USER_LIMIT, PREVIOUS_USER_HEAD)
    if level < 1 and last.tools:
        previous["tools"] = last.tools[-MAX_TOOLS:]
    if last.failed:
        previous["failed_tool_calls"] = last.failed
    if last.assistant:
        limit, head = ASSISTANT_LIMIT, ASSISTANT_HEAD
        if level >= 5:
            limit, head = ASSISTANT_FLOOR, ASSISTANT_FLOOR_HEAD
        previous["assistant"] = _cut(last.assistant, limit, head)
    if previous:
        state["previous_turn"] = previous

    if level < 4:
        earlier = []
        for turn in turns[-1 - EARLIER_TURNS : -1]:
            entry = {}
            if turn.user:
                entry["user"] = turn.user[:EARLIER_USER_LIMIT]
            if level < 3 and turn.assistant:
                entry["assistant"] = _tail(turn.assistant, EARLIER_ASSISTANT_TAIL)
            if entry:
                earlier.append(entry)
        if earlier:
            state["earlier_turns"] = earlier
    return state


def build_routing_state(
    messages: list,
    request,
    criteria: dict,
    *,
    goal: str | None = None,
    budget: int | None = None,
    protect=None,
) -> dict | None:
    """Summarize the conversation and the new request for the decision model.

    *messages* is the history without the new request.
    *protect* encrypts the secrets in a text.
    It runs on whole texts, since a cut could leave a secret's body without
    the prefix that makes it recognizable.
    Returns None when even the shortest summary does not fit *budget* next to
    the descriptions.
    """
    protect = protect or (lambda text: text)
    text = protect(_msg_content({"content": request}))
    size = f"{len(text):,} characters, about {_estimate_tokens(text):,} tokens"
    extras = {}
    attached = _image_parts(request)
    if attached:
        extras["images_attached"] = attached
    earlier = sum(_image_parts(_msg_get(m, "content")) for m in messages)
    if earlier:
        extras["images_earlier"] = earlier
    if goal:
        extras["goal"] = protect(goal)[:GOAL_LIMIT]

    turns = _turns(messages, protect)
    questions = routing_questions(criteria)
    budget = STATE_BUDGET if budget is None else budget
    for level in range(7):
        state = _render(text, size, extras, turns, level)
        units = decisions.envelope_units({"state": state, "questions": questions})
        if units <= budget:
            return state
    return None


def description_units(criteria: dict) -> int:
    """Units the question and the descriptions take on their own."""
    return decisions.envelope_units(
        {"state": {"request": ""}, "questions": routing_questions(criteria)}
    )


@dataclass(frozen=True)
class Facts:
    """What is known about what a runtime can do.

    None means unknown, which counts as capable.
    budget is what the prompt may use once room for the reply is set aside.
    """

    vision: bool | None = None
    tools: bool | None = None
    budget: int | None = None


def unfit(
    facts: Facts | None, *, has_images: bool, needed_tokens: int | None
) -> str | None:
    """Why a runtime cannot take this task (vision, tools, context), or None.

    *needed_tokens* is None when the size is unknown.
    """
    if facts is None:
        return None
    if has_images and facts.vision is False:
        return "vision"
    if facts.tools is False:
        return "tools"
    if (
        facts.budget is not None
        and needed_tokens is not None
        and needed_tokens > facts.budget
    ):
        return "context"
    return None


def described(profiles: dict) -> list[str]:
    return [
        name
        for name, body in profiles.items()
        if isinstance(body.get("description"), str) and body["description"].strip()
    ]


def routing_candidates(
    profiles: dict,
    facts: dict,
    *,
    allowlist=None,
    unavailable=(),
    has_images: bool = False,
    needed_tokens: int = 0,
    require_description: bool = True,
) -> list[str]:
    """Profiles that may run this task, in config order.

    The routing_profiles check also accepts listed profiles without a
    description (require_description=False).
    """
    names = described(profiles) if require_description else list(profiles)
    return [
        name
        for name in names
        if (allowlist is None or name in allowlist)
        and profiles[name].get("provider") != "command"
        and name not in unavailable
        and unfit(facts.get(name), has_images=has_images, needed_tokens=needed_tokens)
        is None
    ]


def classify_current(
    current: str | None,
    current_facts: Facts | None,
    candidates: list[str],
    *,
    allowlist=None,
    explicit: bool = False,
    unavailable=(),
    has_images: bool = False,
    needed_tokens: int = 0,
) -> tuple[str, str | None]:
    """Return ``("candidate" | "valid" | "ineligible", reason)``.

    *current* is the profile the runtime came from, or None when the runtime
    is not a configured profile (no profile, or one changed by /model).
    """
    if current in candidates:
        return "candidate", None
    if allowlist is not None and not explicit and current not in allowlist:
        return "ineligible", "unlisted"
    if current in unavailable:
        return "ineligible", "unavailable"
    problem = unfit(current_facts, has_images=has_images, needed_tokens=needed_tokens)
    if problem is not None:
        return "ineligible", problem
    return "valid", None


@dataclass(frozen=True)
class Verdict:
    # Profile to switch to, or None to stay.
    target: str | None
    reason: str


def _best(candidates: list[str], current: str | None, probabilities: dict) -> str:
    best = max(candidates, key=lambda name: probabilities.get(name, 0.0))
    if current in candidates and probabilities.get(current, 0.0) >= probabilities.get(
        best, 0.0
    ):
        return current
    return best


def apply_rule(
    candidates: list[str],
    current: str | None,
    state: str,
    probabilities: dict | None = None,
    *,
    reason: str | None = None,
    failure: str | None = None,
    margin: float = MARGIN,
) -> Verdict:
    """Decide whether this task switches profiles.

    *state* and *reason* come from classify_current().
    *failure* says why there are no probabilities when they were needed.
    *margin* is how far the best profile must lead the current runtime.
    Probabilities are never rescaled after a candidate is dropped.
    """
    if state == "ineligible":
        if not candidates:
            if reason == "unlisted":
                # The routing_profiles check will stop the task.
                return Verdict(None, "not in routing_profiles")
            if reason == "unavailable":
                return Verdict(None, "retrying the unavailable profile")
            return Verdict(None, f"no candidate fits ({reason})")
        if len(candidates) == 1:
            return Verdict(candidates[0], f"current runtime ineligible ({reason})")
        if probabilities is None:
            text = "first candidate"
            return Verdict(candidates[0], f"{text}, {failure}" if failure else text)
        best = _best(candidates, None, probabilities)
        return Verdict(best, f"current runtime ineligible ({reason})")

    if len(candidates) < 2:
        if state == "candidate":
            return Verdict(None, "only candidate")
        return Verdict(None, "no other candidate" if candidates else "no candidates")
    if probabilities is None:
        return Verdict(None, failure or "no decision")
    best = _best(candidates, current, probabilities)
    p_current = probabilities.get(current, 0.0) if state == "candidate" else 0.0
    lead = probabilities.get(best, 0.0) - p_current
    if best == current:
        return Verdict(None, "best pick")
    # Even with no margin, a profile has to beat the current runtime.
    if lead > 0 and lead >= margin - _MARGIN_TOLERANCE:
        return Verdict(best, f"margin {lead:.2f}")
    return Verdict(None, f"margin {lead:.2f} below {margin}")


class Unusable(Exception):
    """A chosen profile could not be resolved, probed, or admitted."""


@dataclass
class RouteDecision:
    # Profile the task runs on, or None when the runtime is not a profile.
    profile: str | None
    previous: str | None
    switched: bool
    reason: str
    probabilities: dict | None = None
    runtime: object = None
    latency: float | None = None
    usage: dict | None = None

    def report_entry(self) -> dict:
        entry = {
            "type": "routing",
            "profile": self.profile,
            "previous": self.previous,
            "switched": self.switched,
        }
        if not self.switched:
            entry["reason"] = self.reason
        if self.probabilities is not None:
            entry["probabilities"] = dict(self.probabilities)
        if self.latency is not None:
            entry["latency_s"] = round(self.latency, 3)
        if self.usage is not None:
            entry["usage"] = dict(self.usage)
        return entry


@dataclass
class RoutingState:
    """Routing settings, caches and the last decision for a session.

    It only exists with model_routing = true.
    enabled says whether the decision model can be asked at all, and
    off_reason says why not, or why routing turned itself off.
    The allowlist (routing_profiles) applies even when routing cannot run.
    """

    profiles: dict
    pre_profile_baseline: dict = field(default_factory=dict)
    settings: object = None
    allowlist: list | None = None
    enabled: bool = False
    off_reason: str | None = None
    # Started on a command provider, whose prompt a switch would not rebuild.
    command_session: bool = False
    paused: bool = False
    # Profile the main session runs on, also used to route its subagents.
    current: str | None = None
    # The user picked the current runtime, so routing_profiles does not apply.
    explicit: bool = False
    # /model changed the model, so the runtime no longer matches its profile.
    override: bool = False
    unavailable: set = field(default_factory=set)
    # Resolved and checked runtimes, by profile name.
    runtimes: dict = field(default_factory=dict)
    # Profile routing just switched to, until its first request gets an answer.
    watch: str | None = None
    # Profiles a routed subagent could not reach.
    # Worker threads add to it, and the main thread moves them to unavailable.
    failed_spawns: list = field(default_factory=list)
    warned: set = field(default_factory=set)
    last: RouteDecision | None = None
    # Raw provider and base_url of the current runtime, to find its model list.
    catalog_key: tuple | None = None
    # Set by /clear: until the new conversation gets a reply, routing does not
    # favor the current profile.
    cleared: bool = False

    @property
    def active(self) -> bool:
        return self.enabled and not self.paused and self.off_reason is None

    def pin(self, current: str | None, *, override: bool = False) -> bool:
        """Record a choice the user made: routing pauses until /route on.

        Returns whether routing was running before.
        """
        was_active = self.active
        self.current = current
        self.paused = True
        self.explicit = True
        self.override = override
        self.watch = None
        return was_active

    def moved(self) -> None:
        """Routing switched profiles, so the runtime is a plain profile again."""
        self.explicit = False
        self.override = False

    def warn_once(self, key: str, warn, text: str) -> None:
        if key not in self.warned:
            self.warned.add(key)
            warn(text)

    def criteria(self, candidates: list[str]) -> dict:
        return {name: self.profiles[name]["description"] for name in candidates}

    def reset(self) -> None:
        self.last = None
        self.cleared = True

    def status(self) -> str:
        if self.off_reason is not None:
            return f"off, {self.off_reason}"
        if not self.enabled:
            return "off"
        return "paused" if self.paused else "on"

    def summary_line(self) -> str | None:
        line = f"routing: {self.status()}"
        if self.last is not None:
            line += f"  (last: {describe(self.last)})"
        return line


def describe(decision: RouteDecision) -> str:
    target = decision.profile or "(current)"
    if decision.switched:
        text = f"{decision.previous or '(current)'} -> {target}"
    else:
        text = f"stayed on {target}, {decision.reason}"
    if decision.probabilities:
        top = max(decision.probabilities.values())
        text += f" ({top:.2f})"
    return text


def _map_strings(obj, fn):
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, dict):
        return {key: _map_strings(value, fn) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_map_strings(value, fn) for value in obj]
    return obj


def _leaves(obj) -> list:
    found = []
    _map_strings(obj, found.append)
    return found


def _reveal(envelope: dict, shield) -> dict:
    """*envelope* with its whole secrets put back, so the outbound filter reads plaintext.

    A string gets its secrets back only when request() would encrypt it back to
    exactly its protected form, without changing anything else it sends.
    Other strings reach the filter still encrypted.
    """
    leaves = _leaves(envelope)
    restored = [shield.reverse_known(text) for text in leaves]

    def build(chosen):
        position = iter(range(len(leaves)))

        def pick(text):
            n = next(position)
            return restored[n] if n in chosen else text

        return _map_strings(envelope, pick)

    def sent(chosen):
        return _leaves(decisions.encrypt_envelope(build(chosen), shield))

    chosen = {n for n, text in enumerate(leaves) if restored[n] != text}
    if not chosen:
        return envelope
    try:
        unchanged = sent(set())
        while chosen:
            out = sent(chosen)
            if len(out) != len(leaves) or any(
                out[n] != unchanged[n] for n in range(len(leaves)) if n not in chosen
            ):
                return envelope
            missed = {n for n in chosen if out[n] != leaves[n]}
            if not missed:
                break
            chosen -= missed
    except (ValueError, TypeError):
        return envelope
    return build(chosen)


def _probabilities(result, candidates: list[str], shield) -> dict | None:
    """The probabilities for exactly *candidates*, or None.

    Secret encryption sends the labels back encrypted, and a filter may rename
    them, so anything but the exact set is rejected.
    """
    answer = result.answers.get("profile")
    probabilities = answer.get("probabilities") if isinstance(answer, dict) else None
    if not isinstance(probabilities, dict):
        return None
    if shield is not None:
        probabilities = {shield.reverse_known(k): v for k, v in probabilities.items()}
    if set(probabilities) != set(candidates):
        return None
    return probabilities


def choose_profile(
    routing: RoutingState,
    messages: list,
    request,
    *,
    candidates: list[str],
    current: str | None,
    state: str,
    reason: str | None = None,
    resolve,
    warn,
    goal: str | None = None,
    send=None,
    margin: float = MARGIN,
    **request_kwargs,
) -> RouteDecision:
    """Ask the decision model if the rule needs it, then pick a profile.

    *resolve* turns a profile name into a runtime, or raises Unusable.
    A profile that fails is dropped, and the rule runs again on the same
    probabilities without asking again.
    DecisionCancelled is not caught, since it aborts the task.
    """
    send = send or decisions.request
    probabilities = usage = latency = why = None
    if len(candidates) >= 2:
        criteria = routing.criteria(candidates)
        shield = request_kwargs.get("secret_shield")
        protect = None
        if shield is not None:

            def protect(text):
                # Decrypting first keeps a text protected twice from being
                # encrypted twice, so its secrets can still be put back.
                return shield.encrypt_text(shield.reverse_known(text))

            criteria = {name: protect(text) for name, text in criteria.items()}
        body = build_routing_state(
            messages, request, criteria, goal=goal, protect=protect
        )
        if body is None:
            why = "state does not fit the budget"
            routing.warn_once(
                "budget",
                warn,
                "routing skipped: the profile descriptions leave no room for the "
                "conversation in the routing request",
            )
        else:
            envelope = {"state": body, "questions": routing_questions(criteria)}
            if shield is not None:
                envelope = _reveal(envelope, shield)
            started = time.monotonic()
            try:
                result = send(routing.settings, envelope, **request_kwargs)
            except decisions.DecisionUnavailable as e:
                why = "decision model unavailable"
                routing.off_reason = why
                routing.warn_once(
                    "unavailable",
                    warn,
                    f"routing off for this session: decision model unavailable ({e})",
                )
            except decisions.DecisionBlocked as e:
                why = "blocked by the LLM filter"
                routing.warn_once("blocked", warn, f"routing skipped: {e}")
            except decisions.DecisionNotSent as e:
                why = "request not sent"
                routing.warn_once("not_sent", warn, f"routing skipped: {e}")
            except decisions.DecisionError as e:
                why = "decision failed"
                warn(f"routing skipped: {e}")
            else:
                usage = result.usage
                probabilities = _probabilities(result, candidates, shield)
                if probabilities is None:
                    why = "decision failed"
                    warn(
                        "routing skipped: the answer does not cover the profiles "
                        "that were asked about"
                    )
            latency = time.monotonic() - started

    remaining = list(candidates)
    runtime = None
    while True:
        verdict = apply_rule(
            remaining,
            current,
            state,
            probabilities,
            reason=reason,
            failure=why,
            margin=margin,
        )
        if verdict.target is None or verdict.target == current:
            break
        try:
            runtime = resolve(verdict.target)
            break
        except Unusable:
            remaining.remove(verdict.target)
    switched = runtime is not None
    return RouteDecision(
        profile=verdict.target if switched else current,
        previous=current,
        switched=switched,
        reason=verdict.reason,
        probabilities=probabilities,
        runtime=runtime,
        latency=latency,
        usage=usage,
    )

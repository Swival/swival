"""Model routing wired into the REPL, one-shot mode and command scripts.

The decision server is a fake ``litellm.systemone``; the chat provider, the
provider resolution and the model catalog are faked too, so every test can
see which model each chat request went to.
"""

import collections
import contextlib
import sys
import time
import types
from unittest.mock import patch

import litellm
import pytest
from litellm.types.decisions import DecisionsResponse

from swival import agent, decisions, fmt, model_catalog
from swival.config import _UNSET
from swival.input_dispatch import InputContext, parse_input_line
from swival.model_catalog import Catalog, CatalogUnavailable, ModelEntry
from swival.report import AgentError, ReportCollector
from swival.routing import RoutingState
from swival.thinking import ThinkingState
from swival.todo import TodoState
from tests.conftest import build_loop_kwargs, make_message
from tests.test_model_not_found import _make_response, _not_found_exc

REAL_CALL_LLM = agent.call_llm

LOCAL = "http://127.0.0.1:8000/v1"

PROFILES = {
    "local": {
        "provider": "lmstudio",
        "model": "qwen-small",
        "description": "Small local model. Free and fast, weak at reasoning.",
    },
    "gpt": {
        "provider": "chatgpt",
        "model": "gpt-6.1-sol",
        "description": "Strong general coding model for typical feature work.",
    },
    "max": {
        "provider": "chatgpt",
        "model": "gpt-6-astra",
        "description": "Maximum reasoning, slow and expensive, for hard problems.",
    },
    "plain": {"provider": "generic", "model": "plain-model", "base_url": "http://p"},
}

ALL_DOWN = object()


class World:
    """Fake decision server, chat provider, resolution and catalog."""

    def __init__(self, monkeypatch):
        self.sent = []
        self.chat = []
        self.resolved = []
        self.windows_asked = []
        self.listed = []
        self.probs = {"local": 0.05, "gpt": 0.9, "max": 0.05}
        self.windows = {}
        self.unresolvable = set()
        self.listings = {}
        self.chat_errors = {}
        monkeypatch.setattr(litellm, "systemone", self.systemone)
        monkeypatch.setattr(agent, "call_llm", self.call_llm)
        monkeypatch.setattr(agent, "resolve_provider", self.resolve_provider)
        monkeypatch.setattr(model_catalog, "list_models", self.list_models)
        monkeypatch.setattr(fmt, "warning", lambda *a, **k: None)
        monkeypatch.setattr(fmt, "error", lambda *a, **k: None)
        monkeypatch.setattr(fmt, "info", lambda *a, **k: None)

    def systemone(self, **kwargs):
        self.sent.append(kwargs)
        if isinstance(self.probs, Exception):
            raise self.probs
        labels = list(kwargs["questions"]["profile"]["criteria"])
        weights = {label: self.probs.get(label, 0.0) for label in labels}
        total = sum(weights.values()) or 1.0
        probs = {label: weight / total for label, weight in weights.items()}
        choice = max(probs, key=probs.get)
        response = DecisionsResponse.model_validate(
            {
                "model": "clef",
                "answers": {
                    "profile": {
                        "type": "choice",
                        "choice": choice,
                        "confidence": probs[choice],
                        "probabilities": probs,
                    }
                },
                "usage": {"input_tokens": 300, "output_tokens": 0},
            }
        )
        response.set_hidden_params({})
        return response

    def call_llm(self, base_url, model_id, messages, *args, **kwargs):
        self.chat.append(
            types.SimpleNamespace(
                model=model_id,
                provider=kwargs.get("provider"),
                autofix=kwargs.get("model_autofix", True),
                llm_filter=kwargs.get("llm_filter"),
                reasoning=kwargs.get("reasoning_effort"),
                temperature=args[1] if len(args) > 1 else None,
            )
        )
        error = self.chat_errors.get(model_id)
        if error is not None:
            raise error
        agent._count_response(kwargs.get("session_id"))
        return make_message(content=f"answer from {model_id}"), "stop"

    def resolve_provider(
        self,
        provider,
        model,
        api_key,
        base_url,
        max_context_tokens,
        verbose,
        aws_profile=None,
        project=None,
        location=None,
    ):
        self.resolved.append((provider, model))
        self.windows_asked.append(max_context_tokens)
        if model in self.unresolvable:
            raise agent.ConfigError(f"cannot reach {model}")
        key = None if provider in ("lmstudio", "llamacpp") else api_key
        window = max_context_tokens or self.windows.get(model, 128_000)
        llm_kwargs = {"provider": provider, "api_key": key}
        return model, base_url or f"http://{provider}", key, window, llm_kwargs

    def list_models(self, provider, base_url=None, api_key=None, **kwargs):
        self.listed.append((provider, base_url, api_key, kwargs.get("refresh")))
        entries = self.listings.get((provider, base_url), ALL_DOWN)
        if entries is ALL_DOWN:
            raise CatalogUnavailable(f"{provider} at {base_url} is down")
        return Catalog([ModelEntry(id=e) for e in entries], source="fake")

    @property
    def models(self):
        return [c.model for c in self.chat]


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    """Keep /model recents and the global config away from the real home."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))


@pytest.fixture
def world(monkeypatch):
    monkeypatch.setattr(agent, "_RESPONSES", collections.OrderedDict())
    return World(monkeypatch)


class FakeSubagents:
    def __init__(self):
        self._template = {}


def _settings():
    return decisions.resolve_settings("clef", LOCAL)


def make_ctx(
    tmp_path,
    *,
    current="gpt",
    profiles=PROFILES,
    allowlist=None,
    enabled=True,
    interactive=True,
    report=None,
    **loop_extra,
):
    body = profiles.get(current) or {"provider": "generic", "model": "baseline"}
    loop_kwargs = build_loop_kwargs(tmp_path)
    loop_kwargs.pop("max_turns")
    loop_kwargs.update(
        model_id=body.get("model"),
        api_base=f"http://{body['provider']}",
        llm_kwargs={
            "provider": body["provider"],
            "session_id": "s1",
            "max_retries": 0,
            "provider_timeout": 30,
        },
        report=report,
        exposure_meter=report.exposure if report is not None else None,
        **loop_extra,
    )
    routing = RoutingState(
        profiles=profiles,
        settings=_settings(),
        allowlist=allowlist,
        enabled=enabled,
        off_reason=None if enabled else "only one profile has a description",
        catalog_key=(body["provider"], body.get("base_url")),
    )
    return InputContext(
        messages=[{"role": "system", "content": "You are a coding agent."}],
        tools=[],
        base_dir=str(tmp_path),
        turn_state={"max_turns": 4, "turns_used": 0},
        thinking_state=ThinkingState(),
        todo_state=TodoState(),
        snapshot_state=None,
        file_tracker=None,
        no_history=True,
        continue_here=False,
        verbose=False,
        loop_kwargs=loop_kwargs,
        interactive=interactive,
        current_profile=current,
        profiles=profiles,
        startup_profile=current,
        raw_llm_baseline={"provider": body["provider"], "model": body.get("model")},
        pre_profile_baseline={},
        routing=routing,
        subagent_manager=FakeSubagents(),
    )


def run(ctx, line, mode="repl", **kwargs):
    return agent.execute_input(parse_input_line(line), ctx, mode=mode, **kwargs)


def small_local_ctx(tmp_path, world, *, window=8_000, allowlist=("local",)):
    """A context on an unlisted runtime, where local has a small window."""
    profiles = dict(PROFILES, local=dict(PROFILES["local"], max_context_tokens=window))
    world.listings[("lmstudio", None)] = ["qwen-small"]
    return make_ctx(
        tmp_path, current="plain", profiles=profiles, allowlist=list(allowlist)
    )


class TestReplTurns:
    def test_switches_before_the_first_request_and_subagents_follow(
        self, tmp_path, world, monkeypatch
    ):
        shown = []
        monkeypatch.setattr(fmt, "route", shown.append)
        world.probs = {"local": 0.05, "gpt": 0.05, "max": 0.9}
        ctx = make_ctx(tmp_path)
        ctx.loop_kwargs["verbose"] = True
        step = run(ctx, "audit the HLS decryption for IV reuse")
        assert "route: gpt -> max (0.90)" in shown
        assert step.text == "answer from gpt-6-astra"
        assert world.models == ["gpt-6-astra"]
        assert world.chat[0].autofix is False
        assert ctx.current_profile == "max"
        assert ctx.subagent_manager._template["model_id"] == "gpt-6-astra"
        assert ctx.loop_kwargs["llm_kwargs"]["session_id"] == "s1"
        assert len(world.sent) == 1

    def test_stays_without_margin_and_records_the_decision(self, tmp_path, world):
        world.probs = {"local": 0.05, "gpt": 0.5, "max": 0.45}
        report = ReportCollector()
        ctx = make_ctx(tmp_path, report=report)
        run(ctx, "add a --verbose flag")
        assert world.models == ["gpt-6.1-sol"]
        assert world.chat[0].autofix is True
        [entry] = [e for e in report.events if e["type"] == "routing"]
        assert entry["switched"] is False and entry["reason"] == "best pick"
        assert set(entry["probabilities"]) == {"local", "gpt", "max"}
        assert entry["usage"] == {"input_tokens": 300, "output_tokens": 0}
        assert "latency_s" in entry
        assert "routing" in report.exposure.to_dict()["by_source"]

    def test_follow_up_sends_the_previous_turn(self, tmp_path, world):
        ctx = make_ctx(tmp_path)
        run(ctx, "is anything wrong in the README?")
        run(ctx, "yes, go ahead")
        state = world.sent[1]["state"]
        assert state["request"] == "yes, go ahead"
        assert state["previous_turn"]["user"] == "is anything wrong in the README?"
        assert state["previous_turn"]["assistant"] == "answer from gpt-6.1-sol"

    def test_custom_command_is_routed_on_its_full_expansion(
        self, tmp_path, world, monkeypatch
    ):
        expansion = "log line\n" * 10_000 + "Which IPs hammer /login?"
        monkeypatch.setattr(
            agent,
            "_repl_run_custom_command",
            lambda raw, base_dir, model_id: ("logs", expansion, None),
        )
        profiles = dict(PROFILES)
        profiles["local"] = dict(PROFILES["local"], max_context_tokens=8_000)
        world.probs = {"gpt": 0.2, "max": 0.8}
        ctx = make_ctx(tmp_path, profiles=profiles, current="local")
        ctx.loop_kwargs["context_length"] = 8_000
        run(ctx, "!logs")
        [sent] = world.sent
        assert list(sent["questions"]["profile"]["criteria"]) == ["gpt", "max"]
        assert sent["state"]["request"].endswith("Which IPs hammer /login?")
        assert sent["state"]["request_size"].startswith(f"{len(expansion):,} ")
        # The chosen model's context window fits the whole output.
        assert ctx.messages[1]["content"] == expansion
        assert world.models == ["gpt-6-astra"]


class TestWhatDoesNotRoute:
    def test_continue_learn_and_btw(self, tmp_path, world):
        ctx = make_ctx(tmp_path)
        run(ctx, "first task")
        assert len(world.sent) == 1
        run(ctx, "/continue")
        run(ctx, "/learn")
        run(ctx, "/btw what does --onto do?")
        assert len(world.sent) == 1
        assert len(world.chat) == 4

    def test_goal_routes_once_at_launch(self, tmp_path, world, monkeypatch):
        from swival.goal import GOAL_START_PREFIX, GoalState

        world.probs = {"local": 0.05, "gpt": 0.05, "max": 0.9}
        sized = []
        estimate = agent._next_request_tokens

        def spy(history, request, tools, loop_kwargs):
            sized.append((request, {t["function"]["name"] for t in tools}))
            return estimate(history, request, tools, loop_kwargs)

        monkeypatch.setattr(agent, "_next_request_tokens", spy)
        ctx = make_ctx(tmp_path)
        ctx.goal_state = GoalState()
        ctx.loop_kwargs["goal_state"] = ctx.goal_state
        run(ctx, "/goal make the flaky resume test pass")
        assert len(world.sent) == 1
        assert world.sent[0]["state"]["request"] == "make the flaky resume test pass"
        assert "goal" not in world.sent[0]["state"]
        assert set(world.models) == {"gpt-6-astra"}
        # The size check sees what the loop sends: the launch prompt, goal tools.
        request, tool_names = sized[0]
        assert request.startswith(GOAL_START_PREFIX) and "complete_goal" in tool_names

    @pytest.mark.parametrize("interactive", [True, False])
    def test_refused_commands_leave_the_session_alone(
        self, tmp_path, world, monkeypatch, interactive
    ):
        from swival.goal import GoalState, GoalStatus
        from swival.loops import LoopRegistry

        monkeypatch.setattr(agent, "_loop_interruptible_sleep", lambda *a: False)
        ctx = make_ctx(
            tmp_path, current="plain", allowlist=["local"], interactive=interactive
        )
        ctx.goal_state = GoalState()
        ctx.loop_kwargs["goal_state"] = ctx.goal_state
        ctx.loop_registry = LoopRegistry()
        ctx.messages += [
            {"role": "user", "content": "earlier task"},
            {"role": "assistant", "content": "done"},
        ]
        history = list(ctx.messages)

        def refused(line, mode="repl"):
            if interactive:
                assert run(ctx, line, mode=mode).is_error
            else:
                with pytest.raises(agent._RoutingStop):
                    run(ctx, line, mode=mode)

        refused("/goal make the flaky resume test pass")
        assert ctx.goal_state.current.status == GoalStatus.PAUSED
        refused("/continue")
        assert ctx.goal_state.current.status == GoalStatus.PAUSED
        refused("/init")
        refused("/loop 5m check the CI status")
        refused("/loop 5m check the CI status", mode="oneshot")
        assert ctx.messages == history and world.chat == []
        assert len(ctx.loop_registry) == 0

    def test_init_is_sized_by_the_conversation_it_keeps(self, tmp_path, world):
        ctx = small_local_ctx(tmp_path, world)
        ctx.subagent_manager = None
        # Far too long for the local window, but /init clears it first.
        ctx.messages.append({"role": "user", "content": "word " * 40_000})
        run(ctx, "/init")
        assert ctx.current_profile == "local" and world.models[0] == "qwen-small"

    @pytest.mark.parametrize("mode", ["repl", "oneshot"])
    def test_loop_is_checked_on_its_body_once_its_arguments_are_valid(
        self, tmp_path, world, monkeypatch, mode
    ):
        from swival.loops import LoopRegistry

        monkeypatch.setattr(agent, "_loop_interruptible_sleep", lambda *a: False)
        monkeypatch.setattr(agent, "_fire_loop_iteration", lambda *a: None)
        ctx = small_local_ctx(tmp_path, world, allowlist=["local", "gpt"])
        ctx.loop_registry = LoopRegistry()
        assert run(ctx, "/loop", mode=mode).is_error
        assert ctx.current_profile == "plain"
        # Far too long for the local window.
        run(ctx, "/loop 5m " + "word " * 40_000, mode=mode)
        assert ctx.current_profile == "gpt"

    def test_repl_loop_iterations(self, tmp_path, world):
        from swival.loops import LoopRegistry

        ctx = make_ctx(tmp_path)
        ctx.loop_registry = LoopRegistry()
        run(ctx, "/loop 5m check the CI status")
        assert world.sent == []
        assert len(world.chat) == 1
        assert ctx.current_profile == "gpt"
        # A loop whose body is a command is refused.
        assert run(ctx, "/loop 5m !logs").is_error
        assert world.sent == [] and len(world.chat) == 1

    def test_simplify_application_turn(self, tmp_path, world, monkeypatch):
        import swival.simplify

        monkeypatch.setattr(
            swival.simplify, "prepare_simplify_prompt", lambda arg, ctx: "apply these"
        )
        ctx = make_ctx(tmp_path)
        run(ctx, "/simplify")
        assert world.sent == []
        assert world.models == ["gpt-6.1-sol"]


class TestScripts:
    def test_each_prompt_is_routed_once(self, tmp_path, world):
        ctx = make_ctx(tmp_path, interactive=False)
        agent.run_input_script("/help\nfirst prompt\nsecond prompt", ctx)
        assert [s["state"]["request"] for s in world.sent] == [
            "first prompt",
            "second prompt",
        ]
        assert len(world.chat) == 2

    def test_oneshot_loop_iterations_do_not_route(self, tmp_path, world, monkeypatch):
        sleeps = []

        def sleep_once(seconds, stop_event):
            sleeps.append(seconds)
            return len(sleeps) < 2

        monkeypatch.setattr(agent, "_loop_interruptible_sleep", sleep_once)
        monkeypatch.setattr(agent, "_loop_emit_oneshot", lambda text: None)
        ctx = make_ctx(tmp_path, interactive=False)
        agent.run_input_script("/loop 5s ping the server\nthen summarize", ctx)
        assert len(world.chat) == 3
        assert [s["state"]["request"] for s in world.sent] == ["then summarize"]


class TestProbe:
    def test_failed_probe_marks_unavailable_and_reuses_the_decision(
        self, tmp_path, world
    ):
        world.probs = {"local": 0.6, "gpt": 0.05, "max": 0.35}
        ctx = make_ctx(tmp_path, current="plain")
        run(ctx, "rename cnt to count")
        assert "local" in ctx.routing.unavailable
        assert world.resolved[0] == ("lmstudio", "qwen-small")
        # max had 0.35 over the valid runtime's zero, without a new request.
        assert len(world.sent) == 1
        assert world.models == ["gpt-6-astra"]
        run(ctx, "rename it back")
        labels = list(world.sent[1]["questions"]["profile"]["criteria"])
        assert labels == ["gpt", "max"]

    def test_probe_bypasses_the_cache_and_checks_the_model(self, world):
        merged = {"provider": "generic", "model": "m1", "base_url": "http://g"}
        runtime = {"llm_kwargs": {"provider": "generic", "api_key": "k"}}
        world.listings[("generic", "http://g")] = ["M1", "m2"]
        assert agent._probe_profile(runtime, merged) is None
        assert world.listed == [("generic", "http://g", "k", True)]
        world.listings[("generic", "http://g")] = ["m2"]
        assert "does not list 'm1'" in agent._probe_profile(runtime, merged)
        del world.listings[("generic", "http://g")]
        assert "down" in agent._probe_profile(runtime, merged)

    @pytest.mark.parametrize("provider", ["chatgpt", "openrouter", "google", "geap"])
    def test_hosted_providers_are_not_probed(self, world, provider):
        runtime = {"llm_kwargs": {"provider": provider, "api_key": "k"}}
        assert (
            agent._probe_profile(runtime, {"provider": provider, "model": "x"}) is None
        )
        assert world.listed == []

    @pytest.mark.parametrize(
        "provider, configured, expected",
        [
            ("lmstudio", "secret", "lm-studio"),
            ("lmstudio", None, "lm-studio"),
            ("llamacpp", "secret", "none"),
            ("generic", "secret", "secret"),
            ("generic", None, "none"),
        ],
    )
    def test_probe_sends_the_key_chat_sends(
        self, monkeypatch, provider, configured, expected
    ):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("m", 4096))
        merged = {
            "provider": provider,
            "model": "m",
            "api_key": configured,
            "base_url": "http://127.0.0.1:9999",
            "max_context_tokens": 4096,
        }
        runtime = agent._resolve_profile_runtime(merged, verbose=False)
        probed = []

        def list_models(provider, base_url=None, api_key=None, **kwargs):
            probed.append(api_key)
            return Catalog([ModelEntry(id="m")], source="fake")

        monkeypatch.setattr(model_catalog, "list_models", list_models)
        assert agent._probe_profile(runtime, merged) is None

        sent = []

        def completion(completion_kwargs, **kwargs):
            sent.append(completion_kwargs["api_key"])
            raise RuntimeError("stop here")

        monkeypatch.setattr(agent, "_completion_with_retry", completion)
        with pytest.raises(Exception):
            agent.call_llm(
                runtime["api_base"],
                runtime["model_id"],
                [{"role": "user", "content": "hi"}],
                100,
                None,
                None,
                None,
                None,
                False,
                **runtime["llm_kwargs"],
            )
        assert probed == sent == [expected]

    def test_window_learned_by_the_probe_is_checked(self, tmp_path, world, monkeypatch):
        profiles = {
            "gpt": PROFILES["gpt"],
            "small": {
                "provider": "generic",
                "model": "tiny",
                "base_url": "http://tiny",
                "description": "A tiny model behind a key.",
            },
        }
        world.windows["tiny"] = None

        def list_models(provider, base_url=None, api_key=None, **kwargs):
            catalog = Catalog([ModelEntry(id="tiny", context_length=4_096)], "f")
            model_catalog._cache[(provider, base_url)] = (time.monotonic(), catalog)
            return catalog

        monkeypatch.setattr(model_catalog, "list_models", list_models)
        world.probs = {"gpt": 0.05, "small": 0.95}
        ctx = make_ctx(tmp_path, profiles=profiles)
        ctx.messages.append({"role": "user", "content": "context " * 6000})
        ctx.messages.append({"role": "assistant", "content": "ok"})
        run(ctx, "next")
        assert world.models == ["gpt-6.1-sol"]
        assert ctx.routing.runtimes["small"]["context_length"] == 4_096


class TestAvailability:
    def test_unavailable_current_is_retried_and_cleared(self, tmp_path, world):
        profiles = {k: PROFILES[k] for k in ("gpt", "plain")}
        ctx = make_ctx(tmp_path, profiles=profiles)
        ctx.routing.unavailable.add("gpt")
        run(ctx, "next task")
        assert world.models == ["gpt-6.1-sol"]
        assert "gpt" not in ctx.routing.unavailable

    def test_window_found_at_resolution_is_checked_before_commit(self, tmp_path, world):
        world.windows["gpt-6-astra"] = 1_000
        world.probs = {"local": 0.05, "gpt": 0.15, "max": 0.8}
        ctx = make_ctx(tmp_path)
        ctx.messages.append({"role": "user", "content": "context " * 2000})
        ctx.messages.append({"role": "assistant", "content": "ok"})
        run(ctx, "now the hard part")
        assert world.models == ["gpt-6.1-sol"]
        assert "max" in ctx.routing.runtimes
        assert "max" not in ctx.routing.unavailable
        run(ctx, "and again")
        labels = list(world.sent[1]["questions"]["profile"]["criteria"])
        assert "max" not in labels

    def test_first_request_failure_marks_the_profile(self, tmp_path, world):
        world.probs = {"local": 0.05, "gpt": 0.05, "max": 0.9}
        world.chat_errors["gpt-6-astra"] = AgentError("connection refused")
        ctx = make_ctx(tmp_path)
        step = run(ctx, "audit the crypto")
        assert step.text is None and world.models == ["gpt-6-astra"]
        assert ctx.routing.unavailable == {"max"}
        # The task failed the usual way: its message is gone.
        assert [m["role"] for m in ctx.messages] == ["system"]
        run(ctx, "audit the crypto")
        assert world.models[-1] == "gpt-6.1-sol"
        labels = list(world.sent[1]["questions"]["profile"]["criteria"])
        assert "max" not in labels

    def test_vision_rejection_and_missing_tool_support(
        self, tmp_path, world, monkeypatch
    ):
        image = {"type": "image_url", "image_url": {"url": "data:x"}}
        monkeypatch.setattr(
            agent,
            "_VISION_REJECTED_MODELS",
            {("http://elsewhere", "openai/qwen-small")},
        )
        monkeypatch.setitem(
            model_catalog._cache,
            ("chatgpt", None),
            (
                time.monotonic(),
                Catalog([ModelEntry(id="gpt-6-astra", supports_tools=False)], "f"),
            ),
        )
        ctx = make_ctx(tmp_path, current="plain")
        ctx.messages.append(
            {"role": "user", "content": [{"type": "text", "text": "[image] x"}, image]}
        )
        ctx.messages.append({"role": "assistant", "content": "a chart"})
        run(ctx, "what about the legend?")
        # Only gpt is left, and a valid runtime with one candidate stays.
        assert world.sent == []
        assert world.models == ["plain-model"]

    def test_explicit_retry_clears_the_unavailable_mark(self, tmp_path, world):
        ctx = make_ctx(tmp_path)
        ctx.routing.unavailable.add("max")
        run(ctx, "/profile max")
        run(ctx, "it works again")
        assert world.models == ["gpt-6-astra"]
        assert "max" not in ctx.routing.unavailable

    def test_an_answer_clears_the_mark_even_if_a_later_request_fails(self):
        from swival.filter import FilterError

        routing = _active_routing()
        blocked = AgentError("LLM filter blocked request")
        blocked.__cause__ = FilterError("no")
        for error in (AgentError("later failure"), None, blocked):
            routing.unavailable = {"max"}
            routing.watch = "max"
            agent._settle_route(routing, error, replied=True)
            assert routing.unavailable == set() and routing.watch is None
        # Without an answer, only a failure the provider can cause marks it.
        routing.watch = "max"
        agent._settle_route(routing, blocked, replied=False)
        assert routing.unavailable == set() and routing.watch == "max"
        agent._settle_route(routing, AgentError("refused"), replied=False)
        assert routing.unavailable == {"max"} and routing.watch is None

    def test_interrupt_after_an_answer_settles(self, tmp_path, world, monkeypatch):
        calls = []

        def call_llm(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                agent._count_response(kwargs.get("session_id"))
                tc = types.SimpleNamespace(
                    id="t1",
                    function=types.SimpleNamespace(name="nosuchtool", arguments="{}"),
                )
                return make_message(tool_calls=[tc]), "tool_calls"
            raise KeyboardInterrupt

        monkeypatch.setattr(agent, "call_llm", call_llm)
        ctx = make_ctx(tmp_path)
        ctx.subagent_manager = None
        ctx.routing.unavailable.add("gpt")
        ctx.routing.enabled = False
        step = run(ctx, "task")
        assert step.interrupted
        assert "gpt" not in ctx.routing.unavailable
        assert ctx.routing.watch is None

    def test_local_notices_are_not_answers(self, tmp_path, world, monkeypatch):
        def exhausted(messages, tools, **kwargs):
            # Every request overflowed; the loop answers with a local notice.
            messages.append(
                {"role": "assistant", "content": "exhausted", "_swival_synthetic": True}
            )
            return "exhausted", True

        monkeypatch.setattr(agent, "run_agent_loop", exhausted)
        ctx = make_ctx(tmp_path)
        ctx.routing.unavailable.add("gpt")
        ctx.routing.enabled = False
        run(ctx, "task")
        assert "gpt" in ctx.routing.unavailable and ctx.routing.watch == "gpt"

    def test_a_reply_call_llm_rejects_still_counts(self, tmp_path, world, monkeypatch):
        # The real call_llm records the response, then rejects choices=None.
        monkeypatch.setattr(agent, "call_llm", REAL_CALL_LLM)
        world.listings[("generic", "http://b")] = ["mb"]
        world.probs = {"a": 0.05, "b": 0.95}
        profiles = {
            "a": {
                "provider": "generic",
                "model": "ma",
                "base_url": "http://a",
                "description": "General model.",
            },
            "b": {
                "provider": "generic",
                "model": "mb",
                "base_url": "http://b",
                "description": "Hard problems.",
            },
        }
        ctx = make_ctx(tmp_path, profiles=profiles, current="a")
        with patch(
            "litellm.completion", return_value=types.SimpleNamespace(choices=None)
        ):
            run(ctx, "audit the crypto")
        assert ctx.current_profile == "b"
        assert ctx.routing.unavailable == set()
        assert ctx.routing.watch is None

    def test_a_background_workers_reply_is_not_the_parents(
        self, tmp_path, world, monkeypatch
    ):
        world.probs = {"local": 0.05, "gpt": 0.05, "max": 0.9}

        def call_llm(*args, **kwargs):
            # A worker from an earlier task answers while the parent's own
            # first request on the new profile fails.
            agent._RESPONSES["an-older-worker"] = 1
            raise AgentError("connection refused")

        monkeypatch.setattr(agent, "call_llm", call_llm)
        ctx = make_ctx(tmp_path)
        run(ctx, "audit the crypto")
        assert ctx.routing.unavailable == {"max"}

    def test_byte_counts_do_not_rule_out_a_window(self, tmp_path, world, monkeypatch):
        from swival import tokens

        # Without tokenizer data, every size is a UTF-8 byte count.
        monkeypatch.setattr(tokens, "_encoder", tokens._FallbackEncoder())
        monkeypatch.setattr(tokens, "_is_fallback", True)
        ctx = small_local_ctx(tmp_path, world, window=16_000)
        # About 4,000 tokens, but 20,000 bytes.
        run(ctx, "word " * 4_000)
        assert world.models == ["qwen-small"]


class TestAutofix:
    @pytest.fixture(autouse=True)
    def _clear(self):
        agent._MODEL_AUTOFIX.clear()
        yield
        agent._MODEL_AUTOFIX.clear()

    def _call(self, **kwargs):
        return agent.call_llm(
            "http://localhost:8080/v1",
            "requested",
            [{"role": "user", "content": "hi"}],
            100,
            0.5,
            1.0,
            None,
            None,
            False,
            provider="generic",
            api_key="test",
            **kwargs,
        )

    def test_routed_runtime_never_swaps_the_model(self):
        with (
            patch("litellm.completion", side_effect=[_not_found_exc()]) as completion,
            patch("swival.agent._list_server_models", return_value=["replacement"]),
        ):
            with pytest.raises(AgentError, match="was not found"):
                self._call(model_autofix=False)
        assert completion.call_count == 1
        assert agent._MODEL_AUTOFIX == {}

    def test_routed_runtime_ignores_a_recorded_swap(self):
        agent._MODEL_AUTOFIX[("http://localhost:8080/v1", "openai/requested")] = (
            "openai/replacement"
        )
        with patch("litellm.completion", return_value=_make_response()) as completion:
            self._call(model_autofix=False)
        assert completion.call_args.kwargs["model"] == "openai/requested"

    def test_model_command_turns_the_autofix_back_on(self, tmp_path, world):
        world.probs = {"local": 0.05, "gpt": 0.05, "max": 0.9}
        ctx = make_ctx(tmp_path)
        run(ctx, "hard task")
        assert ctx.loop_kwargs["llm_kwargs"]["model_autofix"] is False
        run(ctx, "/model gpt-6.1-sol")
        assert "model_autofix" not in ctx.loop_kwargs["llm_kwargs"]

    def test_secondary_calls_keep_the_autofix_off(self, tmp_path, world, monkeypatch):
        from swival.worktree import call_llm_text

        world.probs = {"local": 0.05, "gpt": 0.05, "max": 0.9}
        ctx = make_ctx(tmp_path)
        run(ctx, "hard task")
        extra = agent._provider_extra_kwargs(ctx.loop_kwargs["llm_kwargs"])
        assert extra["model_autofix"] is False

        autofix = []

        def audit_call(*args, **kwargs):
            autofix.append(kwargs.get("model_autofix", True))
            return make_message(content="ok"), "stop", [], 0, (0, 0)

        monkeypatch.setattr(agent, "call_llm", audit_call)
        call_llm_text(
            ctx.loop_kwargs, [{"role": "user", "content": "x"}], temperature=0
        )
        assert autofix == [False]


class TestRoutingProfiles:
    def test_stops_rather_than_run_unlisted(self, tmp_path, world, monkeypatch):
        errors = []
        monkeypatch.setattr(fmt, "error", errors.append)
        ctx = make_ctx(tmp_path, current="plain", allowlist=["local"])
        step = run(ctx, "do something")
        assert step.text is None and not step.interrupted
        assert world.chat == []
        assert "local" in ctx.routing.unavailable
        assert errors and errors[0].startswith("routing_profiles: no listed profile")
        assert [m["role"] for m in ctx.messages] == ["system"]

    def test_non_interactive_stop_raises(self, tmp_path, world):
        ctx = make_ctx(
            tmp_path, current="plain", allowlist=["local"], interactive=False
        )
        with pytest.raises(agent._RoutingStop):
            run(ctx, "do something", mode="oneshot")
        assert world.chat == []

    def test_profile_command_can_leave_the_list(self, tmp_path, world):
        ctx = make_ctx(tmp_path, current="gpt", allowlist=["gpt", "max"])
        step = run(ctx, "/profile plain")
        assert "routing: paused" in step.text
        run(ctx, "do something")
        assert world.models == ["plain-model"]
        assert world.sent == []

    def test_moves_off_unlisted_when_the_service_is_down(self, tmp_path, world):
        world.probs = litellm.APIConnectionError(
            message="refused", llm_provider="hosted_vllm", model="clef"
        )
        ctx = make_ctx(tmp_path, current="plain", allowlist=["gpt", "max"])
        run(ctx, "first")
        assert ctx.routing.off_reason == "decision model unavailable"
        assert world.models == ["gpt-6.1-sol"]
        assert ctx.current_profile == "gpt"

    def test_moves_off_unlisted_with_too_few_descriptions(self, tmp_path, world):
        ctx = make_ctx(tmp_path, current="local", allowlist=["plain"], enabled=False)
        world.listings[("generic", "http://p")] = ["plain-model"]
        run(ctx, "first")
        assert world.sent == []
        assert world.models == ["plain-model"]

    def test_admission_before_btw(self, tmp_path, world):
        ctx = make_ctx(tmp_path, current="plain", allowlist=["gpt"])
        run(ctx, "/btw quick question")
        assert world.models == ["gpt-6.1-sol"]
        assert ctx.current_profile == "gpt"
        assert ctx.routing.watch is None

    def test_admission_before_the_simplify_reviews(self, tmp_path, world, monkeypatch):
        import swival.simplify

        reviewed_on = []

        def prepare(arg, ctx):
            reviewed_on.append(ctx.loop_kwargs["model_id"])
            return "apply these"

        monkeypatch.setattr(swival.simplify, "prepare_simplify_prompt", prepare)
        ctx = make_ctx(tmp_path, current="plain", allowlist=["gpt"])
        run(ctx, "/simplify")
        assert reviewed_on == ["gpt-6.1-sol"]
        assert world.models == ["gpt-6.1-sol"]

    def test_admission_before_audit(self, tmp_path, world, monkeypatch):
        import swival.audit

        audited_on = []

        def audit(arg, ctx):
            audited_on.append(ctx.loop_kwargs["model_id"])
            return "no findings"

        monkeypatch.setattr(swival.audit, "run_audit_command", audit)
        ctx = make_ctx(tmp_path, current="plain", allowlist=["gpt"])
        assert run(ctx, "/audit").text == "no findings"
        assert audited_on == ["gpt-6.1-sol"]

    def test_restore_is_admitted_first(self, tmp_path, world):
        ctx = make_ctx(tmp_path, current="plain", allowlist=["gpt"])
        run(ctx, "/restore")
        assert ctx.current_profile == "gpt"
        ctx = make_ctx(tmp_path, current="plain", allowlist=["local"])
        step = run(ctx, "/restore")
        assert step.is_error and "routing_profiles" in step.text
        assert world.chat == []

    def test_admission_after_a_decision_keeps_its_report_fields(self, tmp_path, world):
        world.unresolvable.add("gpt-6-astra")
        world.listings[("generic", "http://p")] = ["plain-model"]
        report = ReportCollector()
        ctx = make_ctx(tmp_path, allowlist=["local", "max", "plain"], report=report)
        run(ctx, "task")
        # Both described candidates failed, so the routing_profiles check took plain.
        assert world.models == ["plain-model"] and len(world.sent) == 1
        [entry] = [e for e in report.events if e["type"] == "routing"]
        assert entry["profile"] == "plain" and entry["switched"] is True
        assert "latency_s" in entry and entry["usage"] == {
            "input_tokens": 300,
            "output_tokens": 0,
        }


def _startup_args(**kwargs):
    defaults = dict(
        model_routing=True,
        _all_profiles=PROFILES,
        _pre_profile_baseline={},
        _decisions=_settings(),
        routing_profiles=None,
    )
    return types.SimpleNamespace(**{**defaults, **kwargs})


class TestCommandSession:
    def test_never_routes(self, tmp_path, world):
        routing = agent._routing_startup(
            _startup_args(), provider="command", verbose=False
        )
        assert routing.off_reason == "the session started on a command provider"
        assert not routing.active
        loop_kwargs = build_loop_kwargs(tmp_path)
        current, switched = agent._route_task(
            routing,
            "task",
            current=None,
            history=[],
            tools=[],
            loop_kwargs=loop_kwargs,
        )
        assert (current, switched) == (None, False) and world.sent == []
        routing.allowlist = ["gpt"]
        with pytest.raises(agent._RoutingStop):
            agent._route_task(
                routing,
                "task",
                current=None,
                history=[],
                tools=[],
                loop_kwargs=loop_kwargs,
            )


class TestPinning:
    def test_cli_flags_pause(self):
        args = _startup_args(_llm_from_cli=True, _model_from_cli=True)
        routing = agent._routing_startup(args, provider="chatgpt", verbose=False)
        assert routing.enabled and routing.paused
        assert routing.explicit and routing.override

    def test_profile_and_model_pin_and_route_on_resumes(self, tmp_path, world):
        ctx = make_ctx(tmp_path)
        run(ctx, "/profile max")
        assert ctx.routing.paused and not ctx.routing.override
        run(ctx, "task")
        assert world.sent == [] and world.models == ["gpt-6-astra"]
        step = run(ctx, "/route on")
        assert step.text == "routing: on" and ctx.routing.active
        step = run(ctx, "/model other-model")
        assert "routing: paused" in step.text
        assert ctx.routing.override and ctx.current_profile == "max"
        run(ctx, "/route on")
        # After /model the runtime is valid, never the profile it came from:
        # gpt wins over a zero probability for the override.
        world.probs = {"local": 0.1, "gpt": 0.5, "max": 0.4}
        run(ctx, "task")
        assert world.models[-1] == "gpt-6.1-sol"
        assert not ctx.routing.override and not ctx.routing.explicit

    def test_route_command(self, tmp_path, world):
        ctx = make_ctx(tmp_path)
        run(ctx, "first")
        text = run(ctx, "/route").text
        assert text.startswith("routing: on")
        assert "candidates: local, gpt, max" in text
        assert "last: stayed on gpt, best pick" in text
        assert run(ctx, "/route off").text == "routing: paused"
        assert run(ctx, "/route sideways").is_error
        ctx.routing = None
        assert "not configured" in run(ctx, "/route").text

    def test_model_command_on_the_current_model_pins(self, tmp_path, world):
        world.probs = {"local": 0.05, "gpt": 0.05, "max": 0.9}
        ctx = make_ctx(tmp_path)
        run(ctx, "hard task")
        step = run(ctx, "/model gpt-6-astra")
        assert step.text.startswith("already using")
        assert "routing: paused" in step.text
        assert ctx.routing.paused and not ctx.routing.override
        assert "model_autofix" not in ctx.loop_kwargs["llm_kwargs"]
        assert ctx.routing.explicit

    def test_model_listing_is_not_a_choice(self, tmp_path, world):
        world.listings[("chatgpt", None)] = ["gpt-6.1-sol", "gpt-6-astra"]
        ctx = make_ctx(tmp_path, interactive=False)
        step = run(ctx, "/model", mode="oneshot")
        assert not step.is_error and "gpt-6-astra" in step.text
        assert ctx.routing.active and not ctx.routing.explicit


class TestFailureTable:
    def test_unavailable_turns_routing_off_and_runs_the_task(self, tmp_path, world):
        world.probs = litellm.APIConnectionError(
            message="refused", llm_provider="hosted_vllm", model="clef"
        )
        ctx = make_ctx(tmp_path)
        assert run(ctx, "first").text == "answer from gpt-6.1-sol"
        run(ctx, "second")
        assert len(world.sent) == 1
        assert not ctx.routing.active

    def test_cancel_aborts_the_task(self, tmp_path, world, monkeypatch):
        import threading

        flag = threading.Event()
        world.probs = {"local": 0.05, "gpt": 0.05, "max": 0.9}

        def cancel_then_answer(**kwargs):
            flag.set()
            return World.systemone(world, **kwargs)

        monkeypatch.setattr(litellm, "systemone", cancel_then_answer)
        ctx = make_ctx(tmp_path, cancel_flag=flag)
        run(ctx, "task")
        assert world.chat == []
        assert ctx.current_profile == "gpt"

    def test_filter_block_skips_routing_but_not_the_chat_filter(self, tmp_path, world):
        script = tmp_path / "filter.py"
        script.write_text(
            "import json, sys\n"
            "payload = json.load(sys.stdin)\n"
            "if payload['call_kind'] == 'decision':\n"
            "    print(json.dumps({'allow': False, 'reason': 'no excerpts'}))\n"
            "else:\n"
            "    print(json.dumps({'messages': payload['messages']}))\n"
        )
        llm_filter = f"{sys.executable} {script}"
        ctx = make_ctx(tmp_path, llm_filter=llm_filter)
        run(ctx, "first")
        run(ctx, "second")
        assert world.sent == []
        assert "blocked" in ctx.routing.warned
        assert [c.llm_filter for c in world.chat] == [llm_filter, llm_filter]

    def test_not_sent_skips_routing(self, tmp_path, world, monkeypatch):
        monkeypatch.setattr(decisions, "MAX_REQUEST_UNITS", 10)
        ctx = make_ctx(tmp_path)
        run(ctx, "first")
        assert world.sent == [] and "not_sent" in ctx.routing.warned
        assert world.models == ["gpt-6.1-sol"]

    def test_error_skips_routing_and_moves_ineligible_to_first(self, tmp_path, world):
        world.probs = litellm.BadRequestError(
            message="bad", model="clef", llm_provider="hosted_vllm"
        )
        world.listings[("lmstudio", None)] = ["qwen-small"]
        ctx = make_ctx(tmp_path, current="plain", allowlist=["local", "gpt", "max"])
        run(ctx, "first")
        assert world.models == ["qwen-small"]
        assert ctx.routing.active


class TestStartup:
    def test_off_without_model_routing(self):
        args = _startup_args(model_routing=False)
        assert agent._routing_startup(args, provider="chatgpt", verbose=True) is None

    @pytest.mark.parametrize(
        "kwargs, reason",
        [
            (dict(_decisions=None), "no decision model is configured"),
            (
                dict(
                    _all_profiles={"gpt": PROFILES["gpt"], "plain": PROFILES["plain"]}
                ),
                "only one profile has a description",
            ),
            (dict(routing_profiles=["plain"]), "no listed profile has a description"),
        ],
    )
    def test_explains_why_it_cannot_run(self, monkeypatch, kwargs, reason):
        lines = []
        monkeypatch.setattr(fmt, "info", lines.append)
        routing = agent._routing_startup(_startup_args(**kwargs), "chatgpt", True)
        assert not routing.enabled
        assert lines == [f"routing: off, {reason}"]

    def test_lists_the_candidates(self, monkeypatch):
        lines = []
        monkeypatch.setattr(fmt, "info", lines.append)
        routing = agent._routing_startup(_startup_args(), "chatgpt", True)
        assert routing.active
        assert lines == ["routing: local, gpt, max"]

    def test_warns_about_long_descriptions(self, monkeypatch):
        warnings = []
        monkeypatch.setattr(fmt, "warning", warnings.append)
        monkeypatch.setattr(fmt, "info", lambda *a: None)
        profiles = dict(PROFILES)
        profiles["max"] = dict(PROFILES["max"], description="x" * 4000)
        agent._routing_startup(_startup_args(_all_profiles=profiles), "chatgpt", True)
        assert len(warnings) == 1 and "the longest are max" in warnings[0]


def _write_config(tmp_path, extra=""):
    (tmp_path / "swival.toml").write_text(
        'decision_model = "clef"\n'
        f'decision_base_url = "{LOCAL}"\n'
        'active_profile = "gpt"\n'
        "no_history = true\n"
        "no_continue = true\n"
        "no_instructions = true\n"
        "no_skills = true\n"
        "no_mcp = true\n"
        "no_a2a = true\n"
        f"{extra}\n"
        "[profiles.local]\n"
        'provider = "lmstudio"\n'
        'model = "qwen-small"\n'
        'description = "Small local model, for trivial edits."\n'
        "[profiles.gpt]\n"
        'provider = "chatgpt"\n'
        'model = "gpt-6.1-sol"\n'
        'description = "General coding model."\n'
        "[profiles.max]\n"
        'provider = "chatgpt"\n'
        'model = "gpt-6-astra"\n'
        'description = "Maximum reasoning for hard problems."\n'
    )


class TestMain:
    def _main(self, tmp_path, monkeypatch, argv):
        monkeypatch.setattr(
            sys, "argv", ["swival", "--base-dir", str(tmp_path), "-q", *argv]
        )
        monkeypatch.setattr(agent, "_show_sandbox_review_hint", lambda args: None)
        with pytest.raises(SystemExit) as exc:
            agent.main()
            raise SystemExit(0)
        return exc.value.code

    def test_oneshot_routes_once_and_reviewer_rounds_do_not(
        self, tmp_path, world, monkeypatch, capsys
    ):
        _write_config(tmp_path, "model_routing = true")
        state = tmp_path / "reviewed"
        reviewer = tmp_path / "reviewer.sh"
        reviewer.write_text(
            f'#!/bin/sh\nif [ -f "{state}" ]; then exit 0; fi\n'
            f'touch "{state}"\necho "try again"\nexit 1\n'
        )
        reviewer.chmod(0o755)
        world.probs = {"local": 0.05, "gpt": 0.05, "max": 0.9}
        code = self._main(
            tmp_path, monkeypatch, ["--reviewer", str(reviewer), "fix the race"]
        )
        assert code == 0
        assert len(world.sent) == 1
        assert world.models == ["gpt-6-astra", "gpt-6-astra"]
        assert "answer from gpt-6-astra" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "argv", [["rename"], ["--oneshot-commands", "/help\nrename cnt"]]
    )
    def test_report_and_exit_hook_name_the_routed_provider(
        self, tmp_path, world, monkeypatch, argv
    ):
        import json

        from swival import lifecycle

        hooks = []
        monkeypatch.setattr(
            lifecycle,
            "run_lifecycle_hook",
            lambda command, event, base_dir, **kw: hooks.append(
                (event, kw["provider"], kw["model"])
            ),
        )
        _write_config(tmp_path, 'model_routing = true\nlifecycle_command = "true"')
        world.probs = {"local": 0.9, "gpt": 0.05, "max": 0.05}
        world.listings[("lmstudio", None)] = ["qwen-small"]
        path = tmp_path / "report.json"
        assert self._main(tmp_path, monkeypatch, ["--report", str(path), *argv]) == 0
        report = json.loads(path.read_text())
        assert (report["model"], report["provider"]) == ("qwen-small", "lmstudio")
        assert hooks[-1] == ("exit", "lmstudio", "qwen-small")

    def test_spawns_are_routed_with_the_workers_tools(
        self, tmp_path, world, monkeypatch
    ):
        _write_config(tmp_path, "model_routing = true")
        managers = []

        def loop(messages, tools, **kwargs):
            managers.append(kwargs["subagent_manager"])
            return "done", False

        monkeypatch.setattr(agent, "run_agent_loop", loop)
        assert self._main(tmp_path, monkeypatch, ["rename cnt"]) == 0
        [manager] = managers
        assert manager.router.keywords["tools"] is manager._tools

    def test_off_without_model_routing(self, tmp_path, world, monkeypatch):
        _write_config(tmp_path)
        world.probs = {"local": 0.05, "gpt": 0.05, "max": 0.9}
        assert self._main(tmp_path, monkeypatch, ["fix the race"]) == 0
        assert world.sent == [] and world.models == ["gpt-6.1-sol"]

    @pytest.mark.parametrize(
        "flags",
        [["--profile", "gpt"], ["--model", "gpt-6.1-sol"], ["--no-model-routing"]],
    )
    def test_cli_choices_turn_it_off(self, tmp_path, world, monkeypatch, flags):
        _write_config(tmp_path, "model_routing = true")
        world.probs = {"local": 0.05, "gpt": 0.05, "max": 0.9}
        assert self._main(tmp_path, monkeypatch, [*flags, "fix the race"]) == 0
        assert world.sent == [] and world.models == ["gpt-6.1-sol"]

    def test_model_routing_flag(self, tmp_path, world, monkeypatch):
        _write_config(tmp_path)
        world.probs = {"local": 0.05, "gpt": 0.05, "max": 0.9}
        assert self._main(tmp_path, monkeypatch, ["--model-routing", "x"]) == 0
        assert len(world.sent) == 1 and world.models == ["gpt-6-astra"]

    def test_initial_repl_question_is_routed_once(self, tmp_path, world, monkeypatch):
        _write_config(tmp_path, "model_routing = true")
        world.probs = {"local": 0.05, "gpt": 0.05, "max": 0.9}
        seen = {}

        def fake_repl_loop(messages, tools, **kwargs):
            seen.update(kwargs)

        monkeypatch.setattr(agent, "repl_loop", fake_repl_loop)
        self._main(tmp_path, monkeypatch, ["--repl", "fix the race"])
        assert len(world.sent) == 1 and world.models == ["gpt-6-astra"]
        assert seen["current_profile"] == "max"
        assert seen["model_id"] == "gpt-6-astra"
        assert seen["routing"].watch is None
        assert seen["on_startup_runtime"] is False

    def test_list_profiles_shows_descriptions_and_candidates(
        self, tmp_path, monkeypatch, capsys
    ):
        _write_config(tmp_path, "model_routing = true")
        monkeypatch.setattr(
            sys, "argv", ["swival", "--base-dir", str(tmp_path), "--list-profiles"]
        )
        with pytest.raises(SystemExit):
            agent.main()
        out = capsys.readouterr().out.splitlines()
        assert any(line.startswith("  local") and "[routing]" in line for line in out)
        assert "    Small local model, for trivial edits." in out

    def test_self_review_follows_the_routed_profile(self, tmp_path, world, monkeypatch):
        _write_config(tmp_path, "model_routing = true")
        world.probs = {"local": 0.05, "gpt": 0.05, "max": 0.9}
        seen = []

        def fake_reviewer(cmd, base_dir, answer, verbose, env_extra=None):
            seen.append((cmd, env_extra))
            return 0, "", ""

        monkeypatch.setattr(agent, "run_reviewer", fake_reviewer)
        code = self._main(
            tmp_path,
            monkeypatch,
            ["--self-review", "--api-key", "k", "fix the race"],
        )
        assert code == 0 and len(world.sent) == 1
        cmd, env = seen[0]
        assert cmd.endswith("--profile max")
        assert "gpt-6.1-sol" not in cmd and "--provider" not in cmd
        # The startup provider's key is not handed to another profile.
        assert "k" not in env.values()

    def test_routed_self_review_cmd_drops_provider_flags(self):
        args = types.SimpleNamespace(
            commands="none",
            provider="generic",
            model="m",
            base_url="http://x",
            skills_dir=[],
            max_context_tokens=4096,
            max_output_tokens=1024,
            location="eu",
        )
        cmd = agent._build_self_review_cmd(args, files_mode="all", profile="max")
        assert cmd.endswith(
            "--reviewer-mode --quiet --files all --commands none --profile max"
        )
        assert args.model == "m"


class TestConfig:
    def test_routing_profiles_must_name_profiles(self, tmp_path):
        from swival.config import ConfigError, load_config

        _write_config(tmp_path, 'routing_profiles = ["gpt", "nope"]')
        with pytest.raises(ConfigError, match="unknown profile"):
            load_config(tmp_path)

    def test_keys_reach_args(self, tmp_path):
        from swival.config import apply_config_to_args, load_config

        _write_config(tmp_path, 'model_routing = true\nrouting_profiles = ["gpt"]')
        config = load_config(tmp_path)
        args = types.SimpleNamespace(model_routing=_UNSET, routing_profiles=_UNSET)
        apply_config_to_args(args, config)
        assert args.model_routing is True and args.routing_profiles == ["gpt"]


def _active_routing(**kwargs):
    routing = RoutingState(
        profiles=PROFILES, settings=_settings(), enabled=True, **kwargs
    )
    routing.current = "gpt"
    return routing


def _join_workers(manager):
    """Let every worker finish, so none outlives the fakes it runs on."""
    for handle in list(manager._handles.values()):
        handle.thread.join(10)
    alive = [h.id for h in manager._handles.values() if h.thread.is_alive()]
    if alive:
        manager.shutdown()
    assert not alive


def _manager(tmp_path, *, template=None, system=None, router=None):
    from swival.subagent import SubagentManager

    return SubagentManager(
        loop_kwargs_template=template or _parent_kwargs(tmp_path),
        tools=[],
        resolved_system_content=system,
        parent_cancel_flag=None,
        verbose=False,
        router=router,
    )


def _parent_kwargs(tmp_path):
    loop_kwargs = build_loop_kwargs(tmp_path)
    loop_kwargs.update(
        model_id="gpt-6.1-sol",
        api_base="http://chatgpt",
        llm_kwargs={"provider": "chatgpt", "session_id": "parent", "max_retries": 0},
    )
    return loop_kwargs


def _spawn_on_capped_local(world, tmp_path, task, system="sys", **loop_extra):
    """Route a spawn whose favorite, the local model, has a 4,096-token window."""
    profiles = dict(PROFILES)
    profiles["local"] = dict(PROFILES["local"], max_context_tokens=4_096)
    routing = RoutingState(profiles=profiles, settings=_settings(), enabled=True)
    routing.current = "gpt"
    world.probs = {"local": 0.9, "gpt": 0.05, "max": 0.05}
    world.listings[("lmstudio", None)] = ["qwen-small"]
    route = agent._route_spawn(
        routing,
        task,
        system,
        messages=TestSubagents.PARENT,
        loop_kwargs={**_parent_kwargs(tmp_path), **loop_extra},
        tools=[],
    )
    return route, list(world.sent[0]["questions"]["profile"]["criteria"])


class TestSubagents:
    PARENT = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "the downloads hang under load"},
        {"role": "assistant", "content": "I will split the work."},
    ]

    def test_spawn_is_routed_without_moving_the_parent(self, tmp_path, world):
        world.probs = {"local": 0.05, "gpt": 0.05, "max": 0.9}
        routing = _active_routing()
        loop_kwargs = _parent_kwargs(tmp_path)
        report = ReportCollector()
        route = agent._route_spawn(
            routing,
            "find the deadlock in the worker pool",
            "You are a subagent.",
            messages=self.PARENT,
            loop_kwargs=loop_kwargs,
            tools=[],
            report=report,
        )
        runtime = route.runtime
        assert runtime["model_id"] == "gpt-6-astra"
        assert runtime["llm_kwargs"]["model_autofix"] is False
        assert runtime["llm_kwargs"]["max_retries"] == 0
        assert loop_kwargs["model_id"] == "gpt-6.1-sol"
        assert routing.current == "gpt" and routing.last is None
        state = world.sent[0]["state"]
        assert state["request"] == "find the deadlock in the worker pool"
        assert state["previous_turn"]["user"] == "the downloads hang under load"
        [entry] = [e for e in report.events if e["type"] == "routing"]
        assert entry["subagent"] is True and entry["switched"] is True

    def test_spawn_inherits_when_paused_or_staying(self, tmp_path, world):
        routing = _active_routing()
        loop_kwargs = _parent_kwargs(tmp_path)
        args = dict(messages=self.PARENT, loop_kwargs=loop_kwargs, tools=[])
        assert agent._route_spawn(routing, "add a test", "sys", **args) is None
        routing.pin("gpt")
        sent = len(world.sent)
        assert agent._route_spawn(routing, "add a test", "sys", **args) is None
        assert len(world.sent) == sent

    def test_worker_runs_on_the_override_and_the_template_is_untouched(
        self, tmp_path, monkeypatch
    ):
        from swival.subagent import SpawnRoute

        seen = []

        def fake_loop(messages, tools, **kwargs):
            seen.append(kwargs)
            return "done", False

        monkeypatch.setattr(agent, "run_agent_loop", fake_loop)
        template = _parent_kwargs(tmp_path)
        override = {
            "model_id": "gpt-6-astra",
            "llm_kwargs": {"provider": "chatgpt", "model_autofix": False},
        }
        manager = _manager(
            tmp_path,
            template=template,
            system="sys",
            router=lambda task, system: (
                SpawnRoute(runtime=override) if "hard" in task else None
            ),
        )
        manager.spawn("a hard task")
        manager.spawn("an easy task")
        _join_workers(manager)
        models = sorted(k["model_id"] for k in seen)
        assert models == ["gpt-6-astra", "gpt-6.1-sol"]
        hard = next(k for k in seen if k["model_id"] == "gpt-6-astra")
        assert hard["llm_kwargs"]["model_autofix"] is False
        assert hard["llm_kwargs"]["session_id"] != "parent"
        assert template["model_id"] == "gpt-6.1-sol"
        assert manager.fresh_copy().router is manager.router

    def test_interrupted_routing_frees_the_slot(self, tmp_path):
        from swival.subagent import _MAX_CONCURRENT

        def router(task, system):
            raise KeyboardInterrupt

        manager = _manager(tmp_path, router=router)
        with pytest.raises(KeyboardInterrupt):
            manager.spawn("task")
        assert all(
            manager._slots.acquire(blocking=False) for _ in range(_MAX_CONCURRENT)
        )

    def test_repl_loop_routes_and_rebinds_the_router(
        self, tmp_path, world, monkeypatch
    ):
        from tests.test_report_repl import _REPL_LOOP_DEFAULTS, _patch_repl_loop

        _patch_repl_loop(monkeypatch, ["audit the HLS decryption for IV reuse"])
        monkeypatch.setattr(fmt, "repl_answer", lambda *a, **k: None)
        monkeypatch.setattr(fmt, "reset_state", lambda: None)
        world.probs = {"local": 0.05, "gpt": 0.05, "max": 0.9}
        routing = _active_routing()
        parent = _parent_kwargs(tmp_path)
        manager = _manager(tmp_path, template=parent)
        defaults = dict(_REPL_LOOP_DEFAULTS)
        defaults.update(
            model_id=parent["model_id"],
            api_base=parent["api_base"],
            llm_kwargs=dict(parent["llm_kwargs"]),
        )
        agent.repl_loop(
            [{"role": "system", "content": "You are a coding agent."}],
            [],
            **defaults,
            base_dir=str(tmp_path),
            no_history=True,
            continue_here=False,
            subagent_manager=manager,
            profiles=PROFILES,
            startup_profile="gpt",
            raw_llm_baseline={"provider": "chatgpt", "model": "gpt-6.1-sol"},
            routing=routing,
            current_profile="gpt",
        )
        assert world.models == ["gpt-6-astra"]
        assert manager.router.func is agent._route_spawn
        assert manager.router.args == (routing,)
        assert manager.router.keywords["tools"] is manager._tools

    def test_isolated_contexts_drop_the_router(self, tmp_path, world, monkeypatch):
        manager = _manager(tmp_path, router=lambda task, system: None)
        ctx = make_ctx(tmp_path)
        ctx.subagent_manager = manager
        isolated = agent._build_isolated_ctx(ctx, "btw")
        assert isolated.routing is None
        assert isolated.subagent_manager.router is None
        assert manager.router is not None

    def test_spawn_check_counts_the_system_hint(self, tmp_path, world):
        system = "You are a subagent.\n\n" + "hint " * 20_000
        _, labels = _spawn_on_capped_local(world, tmp_path, "rename cnt", system)
        assert "local" not in labels

    def test_spawn_check_uses_the_workers_system_prompt(self, tmp_path, world):
        prompts = []
        manager = _manager(
            tmp_path,
            system="parent prompt",
            router=lambda task, system: prompts.append(system),
        )
        manager.spawn("task", system_hint="a long hint")
        _join_workers(manager)
        assert "a long hint" in prompts[0] and "parent prompt" in prompts[0]

    def test_unanswered_routed_worker_marks_its_profile(
        self, tmp_path, world, monkeypatch
    ):
        world.probs = {"local": 0.05, "gpt": 0.05, "max": 0.9}
        routing = _active_routing()
        loop_kwargs = _parent_kwargs(tmp_path)

        def failing_loop(messages, tools, **kwargs):
            raise AgentError("the provider rejected the credentials")

        monkeypatch.setattr(agent, "run_agent_loop", failing_loop)
        manager = _manager(
            tmp_path,
            template=loop_kwargs,
            system="sys",
            router=lambda task, system: agent._route_spawn(
                routing,
                task,
                system,
                messages=TestSubagents.PARENT,
                loop_kwargs=loop_kwargs,
                tools=[],
            ),
        )
        manager.spawn("find the deadlock")
        _join_workers(manager)
        assert routing.failed_spawns == ["max"]
        manager.spawn("find the deadlock again")
        _join_workers(manager)
        assert "max" in routing.unavailable
        labels = list(world.sent[-1]["questions"]["profile"]["criteria"])
        assert "max" not in labels

    def test_compaction_does_not_make_a_worker_look_unanswered(self, tmp_path, world):
        world.probs = {"local": 0.05, "gpt": 0.05, "max": 0.9}
        routing = _active_routing()
        route = agent._route_spawn(
            routing,
            "find the deadlock",
            "sys",
            messages=TestSubagents.PARENT,
            loop_kwargs=_parent_kwargs(tmp_path),
            tools=[],
        )
        # The evidence is the response count, so compaction cannot hide a reply.
        error = AgentError("connection refused")
        for session_id in ("worker-that-answered", "worker-without-replies"):
            agent._responses(session_id)
        agent._count_response("worker-that-answered")
        route.on_failure(error, "worker-that-answered")
        assert routing.failed_spawns == []
        route.on_failure(error, "worker-without-replies")
        assert routing.failed_spawns == ["max"]

    def test_spawn_check_counts_activated_skills(self, tmp_path, world):
        from swival.skills import discover_skills

        skill = tmp_path / "skills" / "big"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            "---\nname: big\ndescription: A big skill.\n---\n" + "step " * 8000
        )
        catalog = discover_skills(str(tmp_path), [str(tmp_path / "skills")], False)
        assert "big" in catalog
        _, labels = _spawn_on_capped_local(
            world, tmp_path, "use $big to rename cnt", skills_catalog=catalog
        )
        assert "local" not in labels

    def test_spawn_check_ignores_the_parents_loaded_schemas(self, tmp_path, world):
        big = [
            {
                "type": "function",
                "function": {"name": "big", "description": "x " * 20_000},
            }
        ]

        class Deferred:
            def __init__(self, loaded):
                self.loaded = loaded

            def with_loaded(self, tools):
                return list(tools) + self.loaded

            def fresh(self):
                return Deferred([])

        route, _ = _spawn_on_capped_local(
            world, tmp_path, "rename cnt", deferred_tools=Deferred(big)
        )
        assert route.runtime["model_id"] == "qwen-small"


def _session(tmp_path, **kwargs):
    from swival.session import Session

    defaults = dict(
        base_dir=str(tmp_path),
        history=False,
        no_instructions=True,
        no_skills=True,
        profiles=PROFILES,
        active_profile="gpt",
        model_routing=True,
        decision_model="clef",
        decision_base_url=LOCAL,
    )
    defaults.update(kwargs)
    return Session(**defaults)


class TestStartupRuntime:
    def test_routed_return_to_the_startup_profile_uses_its_settings(self):
        profiles = {"gpt": {"provider": "generic", "base_url": "http://a"}}
        raw = {"provider": "llamacpp", "base_url": "http://b"}
        assert agent._current_llm_settings(profiles, "gpt", "gpt", raw, {}) == raw
        settings = agent._current_llm_settings(
            profiles, "gpt", "gpt", raw, {}, on_startup_runtime=False
        )
        assert settings["provider"] == "generic"

    def test_model_completion_follows_a_routed_return(self, tmp_path, monkeypatch):
        from swival import model_prefs

        looked_up = []
        monkeypatch.setattr(
            model_catalog,
            "cached_entries",
            lambda provider, base_url: looked_up.append(provider) or [],
        )
        monkeypatch.setattr(
            model_prefs,
            "load_prefs",
            lambda: types.SimpleNamespace(preferred_for=lambda provider: []),
        )
        ctx = make_ctx(tmp_path)
        ctx.raw_llm_baseline = {"provider": "llamacpp", "base_url": "http://b"}
        agent._model_completion_candidates(ctx)
        ctx.on_startup_runtime = False
        agent._model_completion_candidates(ctx)
        assert looked_up == ["llamacpp", "chatgpt"]

    def test_routed_switch_leaves_the_startup_runtime(self, tmp_path, world):
        world.probs = {"local": 0.05, "gpt": 0.05, "max": 0.9}
        ctx = make_ctx(tmp_path)
        assert ctx.on_startup_runtime
        run(ctx, "hard task")
        assert not ctx.on_startup_runtime
        run(ctx, "/profile -")
        assert ctx.on_startup_runtime

    def test_profile_listing_keeps_provenance(self, tmp_path, world):
        ctx = make_ctx(tmp_path)
        run(ctx, "/profile")
        assert ctx.on_startup_runtime


class TestSession:
    def test_active_profile_applies_like_the_profile_flag(self, tmp_path):
        session = _session(tmp_path)
        assert (session.provider, session.model) == ("chatgpt", "gpt-6.1-sol")
        # Arguments the caller set win, the way CLI flags win over --profile.
        session = _session(tmp_path, model="other", temperature=0.9)
        assert (session.provider, session.model) == ("chatgpt", "other")
        assert session.temperature == 0.9
        assert session._profile_baseline["model"] == "other"
        # Even one equal to its default.
        session = _session(tmp_path, provider="lmstudio")
        assert (session.provider, session.model) == ("lmstudio", "gpt-6.1-sol")

    def test_bad_profiles_are_rejected(self, tmp_path):
        from swival.config import ConfigError

        with pytest.raises(ConfigError, match="not found"):
            _session(tmp_path, active_profile="nope")
        with pytest.raises(ConfigError, match="unknown profile"):
            _session(tmp_path, routing_profiles=["nope"])
        with pytest.raises(ConfigError, match="not allowed in a profile"):
            _session(tmp_path, profiles={"x": {"provider": "generic", "files": "all"}})

    def test_run_routes_each_call_from_the_starting_profile(self, tmp_path, world):
        world.probs = {"local": 0.05, "gpt": 0.05, "max": 0.9}
        session = _session(tmp_path)
        result = session.run("find the race", report=True)
        assert result.answer == "answer from gpt-6-astra"
        assert any(e["type"] == "routing" for e in result.report["timeline"])
        world.probs = {"local": 0.05, "gpt": 0.9, "max": 0.05}
        session.run("add a flag")
        assert world.models == ["gpt-6-astra", "gpt-6.1-sol"]
        assert len(world.sent) == 2

    def test_report_trace_and_exit_hook_name_the_routed_runtime(
        self, tmp_path, world, monkeypatch
    ):
        from swival import lifecycle, traces

        hooks, traced = [], []
        monkeypatch.setattr(
            lifecycle,
            "run_lifecycle_hook",
            lambda command, event, base_dir, **kw: hooks.append(
                (event, kw["provider"], kw["model"])
            ),
        )
        monkeypatch.setattr(
            traces, "write_trace_to_dir", lambda messages, **kw: traced.append(kw)
        )
        world.probs = {"local": 0.9, "gpt": 0.05, "max": 0.05}
        world.listings[("lmstudio", None)] = ["qwen-small"]
        session = _session(
            tmp_path, lifecycle_command="true", trace_dir=str(tmp_path / "traces")
        )
        report = session.run("rename cnt", report=True).report
        assert (report["model"], report["provider"]) == ("qwen-small", "lmstudio")
        assert hooks[-1] == ("exit", "lmstudio", "qwen-small")
        assert traced[-1]["model"] == "qwen-small"

    def test_ask_keeps_the_routed_runtime(self, tmp_path, world):
        world.probs = {"local": 0.05, "gpt": 0.05, "max": 0.9}
        session = _session(tmp_path)
        session.ask("find the race")
        world.probs = {"local": 0.3, "gpt": 0.3, "max": 0.4}
        session.ask("yes, go ahead")
        assert world.models == ["gpt-6-astra", "gpt-6-astra"]
        assert world.sent[1]["state"]["previous_turn"]["user"] == "find the race"
        session.reset()
        session.ask("rename cnt")
        assert world.models[-1] == "gpt-6.1-sol"

    def test_clear_gives_the_current_profile_no_head_start(self, tmp_path, world):
        world.probs = {"local": 0.05, "gpt": 0.05, "max": 0.9}
        session = _session(tmp_path)
        session.ask("find the race")
        session.ask("/clear", parse_commands=True)
        # A notice Swival wrote itself is not a reply.
        session._conv_state["messages"].append(
            {"role": "assistant", "content": "context full", "_swival_synthetic": True}
        )
        world.probs = {"local": 0.05, "gpt": 0.5, "max": 0.45}
        session.ask("rename cnt")
        # The margin is back for run(), and once the new conversation has a reply.
        world.probs = {"local": 0.05, "gpt": 0.45, "max": 0.5}
        session.run("a one-shot question")
        session.ask("and the tests")
        assert world.models == ["gpt-6-astra"] + ["gpt-6.1-sol"] * 3

    # The twin runs the startup model at the same endpoint, through another
    # provider.
    @pytest.mark.parametrize(
        "routed",
        [
            PROFILES["max"],
            dict(PROFILES["max"], provider="openrouter", model="gpt-6.1-sol"),
        ],
    )
    def test_a_window_learned_after_a_switch_stays_off_the_session(
        self, tmp_path, world, routed
    ):
        overflow = agent.ContextOverflowError("maximum context length is 8192 tokens")
        world.chat_errors = {"gpt-6-astra": overflow, "gpt-6.1-sol": overflow}
        world.windows = {"gpt-6-astra": None, "gpt-6.1-sol": None}
        profiles = dict(PROFILES, max=dict(routed, base_url="http://chatgpt"))
        session = _session(tmp_path, profiles=profiles)
        world.probs = {"local": 0.05, "gpt": 0.05, "max": 0.9}
        with contextlib.suppress(AgentError):
            session.run("find the race")
        with contextlib.suppress(AgentError):
            session.ask("find the race")
        with contextlib.suppress(AgentError):
            session.ask("/continue", parse_commands=True)
        assert session._context_length is None
        # On the startup runtime, the session does learn it.
        world.probs = {"local": 0.05, "gpt": 0.9, "max": 0.05}
        with contextlib.suppress(AgentError):
            session.run("rename cnt")
        assert session._context_length == 8192

    def test_commands_pin_and_persist(self, tmp_path, world):
        session = _session(tmp_path)
        result = session.ask("/profile max", parse_commands=True)
        assert "routing: paused" in result.answer
        sent = len(world.sent)
        session.ask("anything")
        assert world.models == ["gpt-6-astra"] and len(world.sent) == sent
        assert session.ask("/route on", parse_commands=True).answer == "routing: on"
        world.probs = {"local": 0.9, "gpt": 0.05, "max": 0.05}
        world.listings[("lmstudio", None)] = ["qwen-small"]
        session.ask("rename cnt")
        assert world.models[-1] == "qwen-small"

    def test_no_routing_without_model_routing(self, tmp_path, world):
        world.probs = {"local": 0.05, "gpt": 0.05, "max": 0.9}
        session = _session(tmp_path, model_routing=False)
        session.run("find the race")
        assert world.sent == [] and world.models == ["gpt-6.1-sol"]

    def test_subagent_spawns_are_routed(self, tmp_path, world, monkeypatch):
        real_loop = agent.run_agent_loop
        routes = []

        def loop(messages, tools, **kwargs):
            world.probs = {"local": 0.05, "gpt": 0.05, "max": 0.9}
            manager = kwargs["subagent_manager"]
            assert manager.router.keywords["tools"] is manager._tools
            routes.append(manager.router("find the deadlock", "You are a subagent."))
            return real_loop(messages, tools, **kwargs)

        monkeypatch.setattr(agent, "run_agent_loop", loop)
        session = _session(tmp_path, subagents=True)
        session.run("split the work")
        [route] = routes
        assert route.runtime["model_id"] == "gpt-6-astra"
        assert world.models == ["gpt-6.1-sol"]

    def test_args_to_session_kwargs(self):
        from swival.config import args_to_session_kwargs

        base = dict(
            _all_profiles=PROFILES,
            _pre_profile_baseline={"api_key": "k"},
            _active_profile="gpt",
            model_routing=True,
            routing_profiles=["gpt", "max"],
            _llm_from_cli=False,
            _model_from_cli=False,
        )
        kwargs = args_to_session_kwargs(types.SimpleNamespace(**base), "/tmp")
        assert kwargs["profiles"] is PROFILES
        assert kwargs["profile_baseline"] == {"api_key": "k"}
        assert kwargs["active_profile"] == "gpt"
        assert kwargs["model_routing"] is True
        assert kwargs["routing_profiles"] == ["gpt", "max"]

        cli = dict(base, _llm_from_cli=True, _model_from_cli=True)
        kwargs = args_to_session_kwargs(types.SimpleNamespace(**cli), "/tmp")
        assert "active_profile" not in kwargs and "model_routing" not in kwargs

    def test_config_with_profiles_builds_a_session(self, tmp_path):
        from swival.config import config_to_session_kwargs, load_config
        from swival.session import Session

        # The active profile wins over top-level settings, as on the command line.
        top = 'model_routing = true\nprovider = "generic"\nmodel = "baseline-model"'
        _write_config(tmp_path, top)
        kwargs = config_to_session_kwargs(load_config(tmp_path))
        session = Session(**{**kwargs, "base_dir": str(tmp_path)})
        assert (session.provider, session.model) == ("chatgpt", "gpt-6.1-sol")
        assert session.model_routing is True
        assert session._profile_baseline["model"] == "baseline-model"

    def test_reset_and_run_do_not_inherit_an_explicit_choice(self, tmp_path, world):
        session = _session(tmp_path, routing_profiles=["max"])
        session.ask("/profile plain", parse_commands=True)
        session.ask("on the explicit profile")
        assert world.models == ["plain-model"]
        session.run("a one-shot question")
        assert world.models[-1] == "gpt-6-astra"
        session.ask("back in the conversation")
        assert world.models[-1] == "plain-model"
        session.reset()
        session.ask("a private question")
        assert world.models[-1] == "gpt-6-astra"

    def test_bridge_keeps_cli_overrides(self, tmp_path):
        from swival.config import args_to_session_kwargs
        from swival.session import Session

        profiles = {
            "gpt": {
                "provider": "chatgpt",
                "model": "gpt-6.1-sol",
                "temperature": 0.1,
                "reasoning_effort": "high",
                "base_url": "http://profile",
                "max_output_tokens": 512,
            }
        }
        args = types.SimpleNamespace(
            provider="chatgpt",
            model="gpt-6.1-sol",
            temperature=0.9,
            reasoning_effort="low",
            base_url="http://cli",
            # Equal to the default, but still set on the command line.
            max_output_tokens=32768,
            _all_profiles=profiles,
            _pre_profile_baseline={},
            _active_profile="gpt",
            _llm_from_cli=False,
            _model_from_cli=False,
        )
        kwargs = args_to_session_kwargs(args, str(tmp_path))
        session = Session(**{**kwargs, "history": False})
        assert session.temperature == 0.9
        assert session.reasoning_effort == "low"
        assert session.base_url == "http://cli"
        assert session.max_output_tokens == 32768

    def test_live_checks_use_the_runtime_not_its_profile(
        self, tmp_path, world, monkeypatch
    ):
        monkeypatch.setitem(
            model_catalog._cache,
            ("openrouter", "http://or"),
            (
                time.monotonic(),
                Catalog([ModelEntry(id="readonly", supports_tools=False)], "f"),
            ),
        )
        world.probs = litellm.APIConnectionError(
            message="refused", llm_provider="hosted_vllm", model="clef"
        )
        # Started on gpt, but the arguments pick a model without tool support.
        session = _session(
            tmp_path, provider="openrouter", model="readonly", base_url="http://or"
        )
        session.run("rename cnt")
        assert "readonly" not in world.models
        assert session._routing.last.reason == "current runtime ineligible (tools)"


class TestSessionCommands:
    def test_model_and_reasoning_last_for_the_conversation(self, tmp_path, world):
        session = _session(tmp_path, model_routing=False)
        session.ask("/reasoning high", parse_commands=True)
        session.ask("/model gpt-6-astra", parse_commands=True)
        session.ask("task")
        assert world.models == ["gpt-6-astra"]
        assert world.chat[0].reasoning == "high"

    def test_session_commands_keep_runtime_provenance(self, tmp_path, world):
        session = _session(tmp_path, model_routing=False)
        session.ask("/profile plain", parse_commands=True)
        session.ask("/model other-model", parse_commands=True)
        assert world.resolved[-1] == ("generic", "other-model")
        session.ask("task")
        assert world.models == ["other-model"]

    def test_failed_routed_command_keeps_its_runtime(
        self, tmp_path, world, monkeypatch
    ):
        monkeypatch.setattr(
            agent,
            "_repl_run_custom_command",
            lambda raw, base_dir, model_id: ("fix", "fix the race", None),
        )
        world.chat_errors["gpt-6-astra"] = AgentError("connection refused")
        session = _session(tmp_path, routing_profiles=["max"])
        with pytest.raises(AgentError):
            session.ask("!fix", parse_commands=True)
        assert session._routing.unavailable == {"max"}
        del world.chat_errors["gpt-6-astra"]
        assert session.ask("try again").answer == "answer from gpt-6-astra"
        assert session._routing.unavailable == set()

    @pytest.mark.parametrize("commands", [["/help", "/clear"], ["/reasoning high"]])
    def test_commands_do_not_freeze_the_settings(self, tmp_path, world, commands):
        session = _session(
            tmp_path, profiles=None, active_profile=None, model_routing=False
        )
        for line in commands:
            session.ask(line, parse_commands=True)
        session.temperature = 0.2
        session.ask("task")
        assert world.chat[-1].temperature == 0.2

    def test_switch_between_equal_runtimes_keeps_identity(self, tmp_path, world):
        profiles = {
            "a": {"provider": "generic", "model": "small", "base_url": "http://g"},
            "capped": {
                "provider": "generic",
                "model": "small",
                "base_url": "http://g",
                "max_context_tokens": 4_096,
            },
        }
        world.windows["small"] = 4_096
        world.windows["large"] = 32_768
        session = _session(
            tmp_path, profiles=profiles, active_profile="a", model_routing=False
        )
        session.ask("/profile capped", parse_commands=True)
        state = session._conv_state
        assert state["current_profile"] == "capped"
        assert state["on_startup_runtime"] is False
        session.ask("/model large", parse_commands=True)
        # The capped profile's own window setting still applies to /model.
        assert world.windows_asked[-1] == 4_096

    @pytest.mark.parametrize("fails_in", [None, "command", "cleanup"])
    @pytest.mark.parametrize("activates", [True, False])
    def test_a_failed_command_keeps_a_pending_instruction_update(
        self, tmp_path, world, monkeypatch, fails_in, activates
    ):
        import swival.session
        from swival.input_dispatch import StepResult

        reason = "AGENTS.md too large"
        session = _session(tmp_path, model_routing=False)
        session.ask("hello")
        session._conv_state["pending_instruction_failure"] = (
            reason if activates else None
        )

        def command(parsed, ctx, mode="repl"):
            # Activate the pending update, or write one too large to activate.
            ctx.pending_instruction_failure = None if activates else reason
            if fails_in == "command":
                raise AgentError("refused")
            return StepResult(kind="info", text="ok")

        def interrupted(*args):
            raise KeyboardInterrupt

        monkeypatch.setattr(agent, "execute_input", command)
        if fails_in == "cleanup":
            monkeypatch.setattr(swival.session, "_shutdown_and_reconcile", interrupted)
        if fails_in is None:
            session.ask("/audit", parse_commands=True)
        else:
            with pytest.raises((AgentError, KeyboardInterrupt)):
                session.ask("/audit", parse_commands=True)
        # A failure rolls the transcript back, so an activated update is
        # pending again.
        active = activates and fails_in is None
        expected = None if active else reason
        assert session._conv_state["pending_instruction_failure"] == expected


class TestResponseCounter:
    def test_response_counts_stay_bounded(self, monkeypatch):
        monkeypatch.setattr(agent, "_RESPONSES", collections.OrderedDict())
        monkeypatch.setattr(agent, "_RESPONSES_LIMIT", 3)
        agent._count_response("not routed")
        for sid in ("a", "b", "c", "d"):
            agent._responses(sid)
        agent._count_response("b")
        agent._responses("e")
        assert list(agent._RESPONSES) == ["d", "b", "e"]
        # A loop starting a task moves to the back, so it is not the one dropped.
        agent._responses("d")
        agent._responses("f")
        assert list(agent._RESPONSES) == ["e", "d", "f"]

    def test_an_evicted_id_that_answers_again_reads_as_newer(self, monkeypatch):
        monkeypatch.setattr(agent, "_RESPONSES", collections.OrderedDict())
        monkeypatch.setattr(agent, "_RESPONSES_LIMIT", 2)
        agent._responses("parent")
        agent._count_response("parent")
        before = agent._responses("parent")
        agent._responses("worker-1")
        agent._responses("worker-2")
        assert "parent" not in agent._RESPONSES
        assert not agent._responses("parent") > before
        agent._count_response("parent")
        assert agent._responses("parent") > before

    def test_short_lived_loops_forget_their_ids(self, tmp_path, world, monkeypatch):
        seen = []

        def counting_llm(*args, **kwargs):
            seen.append(kwargs.get("session_id"))
            agent._count_response(kwargs.get("session_id"))
            return make_message(content="done"), "stop"

        monkeypatch.setattr(agent, "call_llm", counting_llm)
        session = _session(tmp_path, model_routing=False)
        session.run("one task")
        assert seen and seen[-1] not in agent._RESPONSES

        def fake_loop(messages, tools, **kwargs):
            sid = kwargs["llm_kwargs"]["session_id"]
            seen.append(sid)
            agent._count_response(sid)
            return "done", False

        monkeypatch.setattr(agent, "run_agent_loop", fake_loop)
        manager = _manager(
            tmp_path,
        )
        manager.spawn("task")
        _join_workers(manager)
        assert seen[-1] not in agent._RESPONSES

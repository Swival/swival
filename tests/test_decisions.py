"""Tests for the optional decide tool and its decision-model wiring."""

import copy
import json
import sys
import threading
import types
from unittest.mock import MagicMock, patch

import litellm
import pytest
from litellm.constants import PLACEHOLDER_API_KEY
from litellm.litellm_core_utils.core_helpers import RESPONSE_COST_HEADER
from litellm.llms.hosted_vllm.decisions.transformation import HostedVLLMDecisionsConfig
from litellm.types.decisions import DecisionsRequestBody, DecisionsResponse

from swival import agent
from swival.config import (
    PROFILE_KEYS,
    ConfigError,
    apply_config_to_args,
    args_to_session_kwargs,
    config_to_session_kwargs,
    load_config,
)
from swival.cost import SessionCost
from swival.decisions import MAX_REQUEST_UNITS, decide, request_units, resolve_settings
from swival.exposure import ExposureMeter
from swival.secrets import SecretShield
from swival.session import Session
from swival.thinking import ThinkingState
from swival.todo import TodoState
from swival.tools import DECIDE_TOOL, UNTRUSTED_MARKER, dispatch, get_tool_schema
from tests.conftest import ScriptedLLM, build_loop_kwargs, make_message, make_tool_call

LOCAL = "http://127.0.0.1:8000/v1"

QUESTIONS = {
    "mentions_uv": {"type": "noul", "instructions": "Does the README mention uv?"},
    "install_tool": {
        "type": "choice",
        "instructions": "Which installation tool is documented?",
        "criteria": {"uv": "uv sync", "pip": "pip install"},
    },
    "clarity": {
        "type": "score",
        "instructions": "How explicit is the instruction?",
        "criteria": ["missing", "ambiguous", "explicit"],
    },
}

ANSWERS = {
    "mentions_uv": {"type": "noul", "noul": 0.93},
    "install_tool": {
        "type": "choice",
        "choice": "uv",
        "confidence": 0.97,
        "probabilities": {"uv": 0.97, "pip": 0.03},
    },
    "clarity": {
        "type": "score",
        "score": 1.77,
        "confidence": 0.86,
        "legend": {"0": "missing", "1": "ambiguous", "2": "explicit"},
        "probabilities": {"0": 0.08, "1": 0.06, "2": 0.86},
    },
}


def _response(answers=None, *, usage=None, hidden=None):
    response = DecisionsResponse.model_validate(
        {
            "model": "clef",
            "answers": ANSWERS if answers is None else answers,
            "usage": usage or {"input_tokens": 318, "output_tokens": 0},
        }
    )
    response.set_hidden_params(hidden or {})
    return response


class _FakeSystemOne:
    """Stands in for litellm.systemone and records every request."""

    def __init__(self):
        self.calls = []
        self.response = _response()
        self.error = None
        self.on_call = None

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.on_call is not None:
            self.on_call(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


@pytest.fixture
def systemone(monkeypatch):
    fake = _FakeSystemOne()
    monkeypatch.setattr(litellm, "systemone", fake)
    return fake


def _local():
    return resolve_settings("clef-flash-oQ4e", LOCAL)


def _args(state="The README says install with uv sync.", questions=None):
    return {"state": state, "questions": QUESTIONS if questions is None else questions}


def _changed(name, **changes):
    answers = copy.deepcopy(ANSWERS)
    answers[name].update(changes)
    return answers


def _filter(tmp_path, body):
    """A filter command whose Python body sees the payload as `p`."""
    path = tmp_path / "filter.py"
    path.write_text("import json, sys\np = json.load(sys.stdin)\n" + body)
    return f"{sys.executable} {path}"


class TestResolveSettings:
    def test_disabled_when_empty(self):
        assert resolve_settings(None) is None
        assert resolve_settings("  ", LOCAL, "key") is None

    @pytest.mark.parametrize(
        "model, url, expected",
        [
            ("typesafe/jev-latest", None, "typesafe/jev-latest"),
            ("clef", LOCAL, "hosted_vllm/clef"),
            ("team/clef", LOCAL, "hosted_vllm/team/clef"),
        ],
    )
    def test_model_names(self, model, url, expected):
        assert resolve_settings(model, url).model == expected

    @pytest.mark.parametrize(
        "model, url, message",
        [
            ("clef", None, "--decision-base-url http://127"),
            ("hosted_vllm/clef", None, "no endpoint"),
            ("clef", "127.0.0.1:8000", "http:// or https://"),
        ],
    )
    def test_configuration_errors(self, model, url, message, monkeypatch):
        monkeypatch.delenv("HOSTED_VLLM_API_BASE", raising=False)
        with pytest.raises(ConfigError, match=message):
            resolve_settings(model, url)

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:8000/",
            "http://127.0.0.1:8000/v1",
            "http://127.0.0.1:8000/v1/systemone/",
        ],
    )
    def test_url_forms(self, url):
        base = resolve_settings("clef", url).api_base
        complete = HostedVLLMDecisionsConfig().get_complete_url(base, "clef")
        assert complete == "http://127.0.0.1:8000/v1/systemone"

    def test_keys(self, monkeypatch):
        assert resolve_settings("typesafe/jev-latest").api_key is None
        assert resolve_settings("clef", LOCAL, "secret").api_key == "secret"
        assert "secret" not in repr(resolve_settings("clef", LOCAL, "secret"))

        # A server the user named must not get the key from the environment.
        monkeypatch.setenv("HOSTED_VLLM_API_KEY", "leaked")
        assert _local().api_key == PLACEHOLDER_API_KEY
        assert HostedVLLMDecisionsConfig().resolve_api_key(PLACEHOLDER_API_KEY) is None

    def test_endpoint_and_default_service(self, monkeypatch):
        monkeypatch.delenv("TYPESAFE_API_BASE", raising=False)
        hosted = resolve_settings("typesafe/jev-latest")
        assert hosted.startup_line().endswith("endpoint=https://api.typesafe.ai")
        assert hosted.default_service()
        assert not resolve_settings("typesafe/jev-latest", LOCAL).default_service()

        monkeypatch.setenv("TYPESAFE_API_BASE", "http://proxy.example")
        assert not resolve_settings("typesafe/jev-latest").default_service()


class TestDecide:
    def test_request_and_result(self, systemone):
        # Extra fields from the server are left out of the result.
        systemone.response = _response(_changed("mentions_uv", note="extra"))
        out = json.loads(decide(_local(), _args(), timeout=12))
        assert out == {
            "model": "clef",
            "answers": ANSWERS,
            "usage": {"input_tokens": 318, "output_tokens": 0},
        }
        assert systemone.calls == [
            {
                "model": "hosted_vllm/clef-flash-oQ4e",
                "api_base": LOCAL,
                "api_key": PLACEHOLDER_API_KEY,
                "state": "The README says install with uv sync.",
                "questions": QUESTIONS,
                "num_retries": 0,
                "timeout": 12,
            }
        ]

    @pytest.mark.parametrize("state", [{"ticket": "500s"}, ["a", {"b": 1}]])
    def test_structured_state(self, systemone, state):
        decide(_local(), _args(state=state))
        assert systemone.calls[0]["state"] == state

    def test_invalid_arguments_are_not_sent(self, systemone):
        out = decide(
            _local(),
            {"state": 5, "questions": {"a": {"type": "choice", "instructions": "x"}}},
        )
        assert out == (
            "error: invalid decide arguments: "
            "state: must be text, a JSON object, or an array; "
            "questions.a.criteria: Field required"
        )
        assert decide(_local(), {"state": "x"}).startswith("error: invalid decide")
        assert systemone.calls == []

    @pytest.mark.parametrize(
        "error",
        [
            litellm.APIConnectionError(
                message="Connection refused", llm_provider="hosted_vllm", model="m"
            ),
            litellm.Timeout(message="timed out", model="m", llm_provider="hosted_vllm"),
        ],
    )
    def test_service_failures(self, systemone, error):
        systemone.error = error
        meter = ExposureMeter()
        out = decide(_local(), _args(), exposure=meter.recorder("decision"))
        assert out.startswith("error: decision model request failed: ")
        assert error.message in out
        assert meter.to_dict()["by_source"]["decision"]["requests"]["failed"] == 1

    def test_suppresses_litellm_banners(self, systemone, monkeypatch):
        # LiteLLM prints help banners on errors, but stdout is for the answer.
        monkeypatch.setattr(litellm, "suppress_debug_info", False)
        decide(_local(), _args())
        assert litellm.suppress_debug_info is True


class TestCancellation:
    def test_before_the_call(self, systemone):
        flag = threading.Event()
        flag.set()
        assert decide(_local(), _args(), cancel_flag=flag) == "error: decide cancelled"
        assert systemone.calls == []

    def test_after_the_request_is_recorded(self, systemone):
        """A recorded request is always sent, and its cost always counted."""
        flag = threading.Event()
        meter = ExposureMeter()
        recorder = meter.recorder("decision")
        record = recorder.request

        def record_then_cancel(*args, **kwargs):
            record(*args, **kwargs)
            flag.set()

        recorder.request = record_then_cancel
        cost = SessionCost()
        out = decide(
            _local(), _args(), cancel_flag=flag, exposure=recorder, session_cost=cost
        )
        assert out == "error: decide cancelled"
        assert len(systemone.calls) == 1
        requests = meter.to_dict()["by_source"]["decision"]["requests"]
        assert (requests["sent"], requests["failed"]) == (1, 0)
        assert cost.snapshot().unpriced_calls == 1

    def test_while_the_call_fails(self, systemone):
        flag = threading.Event()
        systemone.on_call = lambda _kwargs: flag.set()
        systemone.error = litellm.APIConnectionError(
            message="reset", llm_provider="hosted_vllm", model="m"
        )
        assert decide(_local(), _args(), cancel_flag=flag) == "error: decide cancelled"


def _state_for(units):
    """A state that brings a request with QUESTIONS to exactly *units*."""
    empty = _args(state="")
    questions = DecisionsRequestBody.model_validate(empty).questions
    room = units - request_units(questions, json.dumps(empty, separators=(",", ":")))
    return "é" * (room // 2) + "x" * (room % 2)


class TestSizeLimit:
    def test_boundary(self, systemone):
        state = _state_for(MAX_REQUEST_UNITS)
        assert not decide(_local(), _args(state=state)).startswith("error:")
        out = decide(_local(), _args(state=state + "x"))
        assert out.startswith("error: decide request too large (15001 of 15000 units")
        assert len(systemone.calls) == 1

    def test_questions_and_labels_count(self, systemone):
        many = {
            f"q{i}": {
                "type": "choice",
                "instructions": "w?",
                "criteria": {f"l{j}": None for j in range(30)},
            }
            for i in range(25)
        }
        payload = json.dumps({"state": "x", "questions": many}, separators=(",", ":"))
        assert len(payload.encode()) < MAX_REQUEST_UNITS
        out = decide(_local(), _args(state="x", questions=many))
        assert out.startswith("error: decide request too large")

    def test_checked_after_filter_and_encryption(self, systemone, tmp_path):
        class Expanding:
            def encrypt_text(self, text):
                return text.replace("SECRET", "SECRET!")

            def reverse_known(self, text):
                return text

        grow = _filter(
            tmp_path,
            "env = json.loads(p['messages'][0]['content'])\n"
            "env['state'] += '!'\n"
            "p['messages'][0]['content'] = json.dumps(env)\n"
            "print(json.dumps({'messages': p['messages']}))\n",
        )
        state = _state_for(MAX_REQUEST_UNITS)
        out = decide(_local(), _args(state=state), llm_filter=grow)
        assert out.startswith("error: decide request too large (15001 of 15000")

        state = _state_for(MAX_REQUEST_UNITS - len("SECRET")) + "SECRET"
        out = decide(_local(), _args(state=state), secret_shield=Expanding())
        assert out.startswith("error: decide request too large (15001 of 15000")
        assert systemone.calls == []


class TestFilter:
    def test_filter_sees_and_edits_the_request(self, systemone, tmp_path):
        log = tmp_path / "seen.json"
        redact = _filter(
            tmp_path,
            f"open({str(log)!r}, 'w').write(json.dumps(p))\n"
            "m = p['messages'][0]\n"
            "m['content'] = m['content'].replace('uv sync', '[redacted]')\n"
            "print(json.dumps({'messages': p['messages']}))\n",
        )
        decide(_local(), _args(), llm_filter=redact)
        seen = json.loads(log.read_text())
        assert (seen["call_kind"], seen["provider"], seen["model"]) == (
            "decision",
            "hosted_vllm",
            "hosted_vllm/clef-flash-oQ4e",
        )
        [message] = seen["messages"]
        assert message["role"] == "user"
        assert json.loads(message["content"]) == _args()
        assert "[redacted]" in systemone.calls[0]["state"]

    @pytest.mark.parametrize(
        "output",
        [
            "{'allow': False, 'reason': 'not today'}",
            "{'messages': p['messages'] * 2}",
            "{'messages': [{'role': 'assistant', 'content': 'x'}]}",
            "{'messages': [{'role': 'user', 'content': 'not json'}]}",
            "{'messages': [{'role': 'user', 'content': json.dumps("
            "{'state': 'x', 'questions': {}, 'more': 1})}]}",
        ],
    )
    def test_blocked_or_malformed_output_stops_the_request(
        self, systemone, tmp_path, output
    ):
        bad = _filter(tmp_path, f"print(json.dumps({output}))\n")
        out = decide(_local(), _args(), llm_filter=bad)
        assert out.startswith("error: LLM filter blocked the decision request: ")
        assert systemone.calls == []


class TestEncryption:
    TOKEN = "ghp_abcdefghijklmnopqrstuvwxyz0123456789"
    LABEL = "ghp_ZYXWVUTSRQPONMLKJIHGFEDCBA9876543210"

    def _args(self):
        return {
            "state": f"key {self.TOKEN} found",
            "questions": {
                self.TOKEN: {
                    "type": "choice",
                    "instructions": f"Is {self.LABEL} used?",
                    "criteria": {self.LABEL: "the other", "none": "neither"},
                }
            },
        }

    def test_names_labels_and_text_are_encrypted_then_restored(self, systemone):
        def answer(kwargs):
            [(name, question)] = kwargs["questions"].items()
            [label, _none] = question["criteria"]
            choice = {
                "type": "choice",
                "choice": label,
                "confidence": 0.7,
                "probabilities": {label: 0.7, "none": 0.3},
            }
            systemone.response = _response({name: choice})

        systemone.on_call = answer
        out = decide(_local(), self._args(), secret_shield=SecretShield())
        sent = json.dumps(systemone.calls[0])
        assert self.TOKEN not in sent and self.LABEL not in sent
        assert json.loads(out)["answers"][self.TOKEN]["choice"] == self.LABEL

    def test_errors_are_decrypted(self, systemone):
        def fail(kwargs):
            [name] = kwargs["questions"]
            systemone.error = litellm.BadRequestError(
                message=f"bad question {name}", model="m", llm_provider="hosted_vllm"
            )

        systemone.on_call = fail
        out = decide(_local(), self._args(), secret_shield=SecretShield())
        assert out.startswith("error: decision model request failed: ")
        assert self.TOKEN in out


def _through_litellm(answer):
    """Run a raw server answer through the real LiteLLM conversion."""
    import httpx

    name = "clarity" if answer["type"] == "score" else "install_tool"
    payload = {"model": "clef", "answers": {name: answer}}

    class Client:
        def post(self, url, **kwargs):
            request = httpx.Request("POST", url)
            return httpx.Response(200, json=payload, request=request)

    with patch(
        "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
        return_value=Client(),
    ):
        return decide(_local(), _args(questions={name: QUESTIONS[name]}))


class TestAnswerValidation:
    @pytest.mark.parametrize(
        "answers, expected",
        [
            (
                {k: v for k, v in ANSWERS.items() if k != "clarity"},
                "no answer for clarity",
            ),
            (
                {**ANSWERS, "clarity": ANSWERS["mentions_uv"]},
                "expected a score answer, got noul",
            ),
            (_changed("mentions_uv", noul=1.5), "outside [0, 1]"),
            (_changed("mentions_uv", noul=float("nan")), "not finite"),
            (_changed("install_tool", choice="conda"), "not a requested label"),
            (_changed("install_tool", confidence=-0.1), "outside [0, 1]"),
            (
                _changed("install_tool", probabilities={"uv": 0.9, "x": 0.1}),
                "unrequested",
            ),
            (
                _changed("install_tool", probabilities={"uv": 1.0}),
                "no probability for",
            ),
            (_changed("clarity", score=2.5), "outside [0, 2]"),
            (_changed("clarity", legend={"0": "x", "1": "y", "2": "z"}), "legend"),
            (
                _changed("install_tool", probabilities={"uv": 0.5, "pip": 0.1}),
                "add up to 0.600",
            ),
        ],
    )
    def test_malformed(self, systemone, answers, expected):
        systemone.response = _response(answers)
        out = decide(_local(), _args())
        assert out.startswith("error: decision model returned an unusable response: ")
        assert expected in out

    @pytest.mark.parametrize(
        "answer, expected",
        [
            (
                {
                    "type": "score",
                    "score": 1.5,
                    "confidence": 0.9,
                    "legend": {},
                    "probabilities": {},
                },
                "add up to 0.000",
            ),
            (
                {
                    "type": "choice",
                    "choice": "uv",
                    "confidence": 0.9,
                    "probabilities": {"uv": 0.9},
                },
                "no probability for ['pip']",
            ),
        ],
    )
    def test_raw_server_answers(self, answer, expected):
        out = _through_litellm(answer)
        assert out.startswith("error: decision model returned an unusable response: ")
        assert expected in out

    def test_litellm_fills_in_missing_score_levels(self):
        """Known limit: answers are checked after LiteLLM's conversion.

        The server skipped levels 0 and 2 and the whole legend, but LiteLLM
        fills them in and the total is still 1, so the answer passes.
        """
        out = _through_litellm(
            {
                "type": "score",
                "score": 1.0,
                "confidence": 1.0,
                "legend": {},
                "probabilities": {"1": 1.0},
            }
        )
        clarity = json.loads(out)["answers"]["clarity"]
        assert clarity["probabilities"] == {"0": 0.0, "1": 1.0, "2": 0.0}
        assert clarity["legend"] == ANSWERS["clarity"]["legend"]

    def test_structured_levels_in_the_legend(self, systemone):
        levels = [{"level": "low"}, ["mid", 2], "high"]
        systemone.response = _response(
            {
                "sev": {
                    "type": "score",
                    "score": 1.7,
                    "confidence": 0.7,
                    "legend": {str(i): level for i, level in enumerate(levels)},
                    "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
                }
            }
        )
        questions = {"sev": {"type": "score", "instructions": "?", "criteria": levels}}
        out = json.loads(decide(_local(), _args(questions=questions)))
        assert out["answers"]["sev"]["legend"]["0"] == {"level": "low"}


class TestAccounting:
    @pytest.mark.parametrize(
        "model, url, hidden, expected",
        [
            ("clef", LOCAL, {"response_cost": 0.5}, (0.0, 0, 1)),
            (
                "clef",
                LOCAL,
                {"additional_headers": {RESPONSE_COST_HEADER: 0.002}},
                (0.002, 1, 0),
            ),
            ("typesafe/jev-latest", None, {"response_cost": 1.3e-05}, (1.3e-05, 1, 0)),
        ],
    )
    def test_prices(self, systemone, monkeypatch, model, url, hidden, expected):
        """LiteLLM's own prices only count for the provider's own service."""
        monkeypatch.delenv("TYPESAFE_API_BASE", raising=False)
        systemone.response = _response(hidden=hidden)
        cost = SessionCost()
        decide(resolve_settings(model, url), _args(), session_cost=cost)
        snap = cost.snapshot()
        assert (snap.known_usd, snap.priced_calls, snap.unpriced_calls) == expected

    def test_zero_usage(self, systemone, monkeypatch):
        monkeypatch.delenv("TYPESAFE_API_BASE", raising=False)
        systemone.response = _response(
            usage={"input_tokens": 0, "output_tokens": 0}, hidden={"response_cost": 0.0}
        )
        cost = SessionCost()
        out = decide(
            resolve_settings("typesafe/jev-latest"), _args(), session_cost=cost
        )
        assert "usage" not in json.loads(out)
        assert cost.snapshot().unpriced_calls == 1

    def test_cost_counted_even_for_unusable_answers(self, systemone):
        systemone.response = _response({})
        cost = SessionCost()
        out = decide(_local(), _args(), session_cost=cost)
        assert out.startswith("error: decision model returned an unusable response")
        assert cost.snapshot().unpriced_calls == 1

    def test_exposure(self, systemone):
        meter = ExposureMeter()
        decide(_local(), _args(), exposure=meter.recorder("decision"))
        decision = meter.to_dict()["by_source"]["decision"]
        assert decision["requests"]["sent"] == 1
        assert decision["input"]["user"] > 0
        assert decision["provider_usage"]["usage"]["input_tokens"] == 318


class TestToolWiring:
    def test_schema_and_tool_lists(self):
        from swival.repair import repair_tool_args
        from swival.tools import NETWORK_TOOLS

        def names(tools):
            return [t["function"]["name"] for t in tools]

        assert "decide" not in names(agent.build_tools({}, {}, False))
        tools = agent.build_tools({}, {}, False, network="provider-only", decide=True)
        assert "decide" in names(tools)
        assert "decide" not in NETWORK_TOOLS
        assert names(agent._filter_command_tool_schemas(tools)) == ["decide"]

        # Vertex drops nested schemas, so the description must carry the format.
        from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
            VertexGeminiConfig,
        )

        [mapped] = VertexGeminiConfig()._map_function([copy.deepcopy(DECIDE_TOOL)], {})
        [declaration] = mapped["function_declarations"]
        questions = declaration["parameters"]["properties"]["questions"]
        assert "additionalProperties" not in questions
        for word in ("noul", "choice", "score", "instructions", "criteria"):
            assert word in questions["description"]

        # Argument repair must leave question names and criteria alone.
        schema = get_tool_schema("decide")
        assert schema["required"] == ["state", "questions"]
        args = {
            "state": "x",
            "questions": {"quesion": {"type": "choice", "criteria": {"stat": "y"}}},
        }
        assert repair_tool_args(args, schema) == (args, [])

    def test_dispatch(self, tmp_path):
        recorded = []
        report = types.SimpleNamespace(
            record_untrusted_input=lambda name, origin="": recorded.append(name)
        )

        def run(decide_call):
            return dispatch(
                "decide", _args(), str(tmp_path), decide_call=decide_call, report=report
            )

        assert run(None) == "error: decide tool is not available in this session"
        assert run(lambda args: "error: decide cancelled") == "error: decide cancelled"
        out = run(lambda args: '{"answers":{}}')
        assert out.startswith(UNTRUSTED_MARKER) and out.endswith('{"answers":{}}')
        assert recorded == ["decide"]


class _RecordingLLM(ScriptedLLM):
    """ScriptedLLM that also records the tool names offered on each call."""

    def __init__(self, responses=()):
        super().__init__(responses)
        self.offered = []

    def __call__(self, *args, **kwargs):
        self.offered.append([t["function"]["name"] for t in args[7] or []])
        return super().__call__(*args, **kwargs)


def _decide_then_answer():
    call = make_tool_call("decide", json.dumps(_args()), "d1")
    return _RecordingLLM([make_message(tool_calls=[call])])


def _loop_kwargs(tmp_path, **overrides):
    kwargs = build_loop_kwargs(tmp_path)
    kwargs.update(overrides)
    return kwargs


def _tool_result(messages):
    return next(m["content"] for m in messages if m.get("role") == "tool")


class TestLoop:
    def test_decide_runs_through_the_loop(self, tmp_path, monkeypatch, systemone):
        llm = _decide_then_answer()
        monkeypatch.setattr(agent, "call_llm", llm)
        cost = SessionCost()
        meter = ExposureMeter()
        messages = [{"role": "user", "content": "classify"}]
        agent.run_agent_loop(
            messages,
            agent.build_tools({}, {}, False, decide=True),
            **_loop_kwargs(
                tmp_path,
                decisions=_local(),
                llm_kwargs={"provider": "generic", "provider_timeout": 10},
                session_cost=cost,
                exposure_meter=meter,
            ),
        )
        assert "decide" in llm.offered[0]
        assert systemone.calls[0]["timeout"] == 10
        result = _tool_result(messages)
        assert result.startswith(UNTRUSTED_MARKER) and '"noul":0.93' in result
        assert cost.snapshot().unpriced_calls == 1
        assert meter.to_dict()["by_source"]["decision"]["requests"]["sent"] == 1

    def test_hidden_without_settings(self, tmp_path, monkeypatch, systemone):
        """A worker that shares the parent's tool list must not see decide."""
        llm = _decide_then_answer()
        monkeypatch.setattr(agent, "call_llm", llm)
        parent_tools = agent.build_tools({}, {}, False, decide=True)
        before = list(parent_tools)
        messages = [{"role": "user", "content": "x"}]
        agent.run_agent_loop(messages, parent_tools, **_loop_kwargs(tmp_path))
        assert "decide" not in llm.offered[0]
        assert parent_tools == before
        assert _tool_result(messages) == (
            "error: decide tool is not available in this session"
        )
        assert systemone.calls == []

    def test_loop_filter_applies(self, tmp_path, monkeypatch, systemone):
        monkeypatch.setattr(agent, "call_llm", _decide_then_answer())
        block = _filter(tmp_path, "print(json.dumps({'allow': False}))\n")
        messages = [{"role": "user", "content": "x"}]
        agent.run_agent_loop(
            messages,
            agent.build_tools({}, {}, False, decide=True),
            **_loop_kwargs(tmp_path, decisions=_local(), llm_filter=block),
        )
        assert _tool_result(messages).startswith("error: LLM filter blocked")
        assert systemone.calls == []

    def test_nested_and_command_provider_kwargs(self, tmp_path, monkeypatch):
        captured = {}
        real = agent.handle_tool_call

        def spy(tool_call, *args, **kwargs):
            captured.update(kwargs)
            return real(tool_call, *args, **kwargs)

        monkeypatch.setattr(agent, "handle_tool_call", spy)
        think = make_tool_call("think", '{"thought": "x"}', "t1")
        monkeypatch.setattr(
            agent, "call_llm", ScriptedLLM([make_message(tool_calls=[think])])
        )
        settings = _local()
        flag = threading.Event()
        agent.run_agent_loop(
            [{"role": "user", "content": "x"}],
            agent.build_tools({}, {}, False, decide=True),
            **_loop_kwargs(tmp_path, decisions=settings, cancel_flag=flag),
        )
        assert captured["metaskill_loop_kwargs"]["decisions"] is settings
        call = captured["decide_call"]
        assert call.args == (settings,)
        assert call.keywords["timeout"] == 30
        assert call.keywords["cancel_flag"] is flag

    def test_subagents_inherit_and_isolated_workers_drop(self):
        from swival.subagent import SA_TEMPLATE_EXCLUDE
        from swival.worktree import make_isolated_loop_kwargs

        assert "decisions" not in SA_TEMPLATE_EXCLUDE
        ctx = types.SimpleNamespace(
            loop_kwargs={"decisions": _local(), "max_turns": 5, "llm_kwargs": {}}
        )
        assert "decisions" not in make_isolated_loop_kwargs(ctx, "/tmp/work")

    def test_profile_switch_keeps_decisions(self, monkeypatch):
        decision_keys = {"decision_model", "decision_base_url", "decision_api_key"}
        assert not decision_keys & PROFILE_KEYS
        settings = _local()
        repl_kwargs = {"decisions": settings, "llm_kwargs": {}}
        monkeypatch.setattr(
            agent,
            "resolve_provider",
            lambda **kw: ("other-model", "http://other", None, None, {"provider": "x"}),
        )
        _name, _msg, is_error = agent._repl_profile(
            "fast",
            {"fast": {"provider": "generic", "model": "other-model"}},
            None,
            None,
            {},
            repl_kwargs=repl_kwargs,
        )
        assert not is_error
        assert repl_kwargs["model_id"] == "other-model"
        assert repl_kwargs["decisions"] is settings

    def test_repl_turns_receive_the_setting(self, tmp_path):
        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["classify this", "/exit"]
        prompt_session.default_buffer.text = ""
        settings = _local()
        seen = []

        def fake_loop(messages, tools, **kwargs):
            seen.append(kwargs.get("decisions"))
            return "answer", False

        with (
            patch("prompt_toolkit.PromptSession", return_value=prompt_session),
            patch("swival.agent.run_agent_loop", side_effect=fake_loop),
        ):
            agent.repl_loop(
                [{"role": "system", "content": "system"}],
                agent.build_tools({}, {}, False, decide=True),
                api_base="http://127.0.0.1:1234",
                model_id="test-model",
                max_turns=5,
                max_output_tokens=1024,
                temperature=0.5,
                top_p=None,
                seed=None,
                context_length=None,
                base_dir=str(tmp_path),
                thinking_state=ThinkingState(verbose=False),
                todo_state=TodoState(verbose=False),
                resolved_commands={},
                skills_catalog={},
                skill_read_roots=[],
                extra_write_roots=[],
                verbose=False,
                llm_kwargs={"provider": "lmstudio", "api_key": None},
                no_history=True,
                decisions=settings,
            )
        assert seen == [settings]


class TestEndpointMismatchHint:
    MESSAGE = (
        "Model 'clef-flash-oQ4e' is not an LLM / chat model. "
        "Use /v1/systemone for decision models."
    )

    def _call(self, error):
        with patch("litellm.completion", side_effect=error):
            with pytest.raises(agent.AgentError) as ei:
                agent.call_llm(
                    "http://localhost",
                    "clef-flash-oQ4e",
                    [{"role": "user", "content": "hi"}],
                    100,
                    0.1,
                    1.0,
                    None,
                    None,
                    False,
                    provider="generic",
                    unknown_context_window=True,
                    max_retries=1,
                )
        return ei.value

    @pytest.mark.parametrize(
        "error",
        [
            litellm.BadRequestError(message=MESSAGE, model="m", llm_provider="openai"),
            litellm.APIError(
                message=MESSAGE, status_code=422, model="m", llm_provider="openai"
            ),
        ],
    )
    def test_hint_without_context_reduction(self, error):
        err = self._call(error)
        assert not isinstance(err, agent.ContextOverflowError)
        assert str(err).endswith(
            "Use --decision-model for this model alongside your regular chat --model."
        )
        assert getattr(err, "_request_shaped", False) is False

    @pytest.mark.parametrize(
        "message",
        ["model foo is not a chat model", "Model 'clef-flash-oQ4e' does not exist"],
    )
    def test_no_hint_for_other_errors(self, message):
        error = litellm.BadRequestError(
            message=message, model="m", llm_provider="openai"
        )
        assert "--decision-model" not in str(self._call(error))


def _parse(argv, config=None):
    args = agent.build_parser().parse_args([*argv, "task"])
    apply_config_to_args(args, dict(config or {}))
    return args


class TestConfiguration:
    @pytest.mark.parametrize(
        "argv, config, expected",
        [
            ([], {}, None),
            ([], {"decision_model": "clef"}, "clef"),
            (["--decision-model", "x"], {"decision_model": "clef"}, "x"),
            (["--no-decisions"], {"decision_model": "clef"}, ""),
        ],
    )
    def test_precedence(self, argv, config, expected):
        assert _parse(argv, config).decision_model == expected

    def test_no_decisions_conflicts_with_decision_model(self, capsys):
        with pytest.raises(SystemExit):
            agent.build_parser().parse_args(
                ["--decision-model", "x", "--no-decisions", "task"]
            )
        assert "not allowed with argument" in capsys.readouterr().err

    def test_project_overrides_global(self, tmp_path, monkeypatch):
        xdg = tmp_path / "xdg"
        (xdg / "swival").mkdir(parents=True)
        (xdg / "swival" / "config.toml").write_text(
            f'decision_model = "clef"\ndecision_base_url = "{LOCAL}"\n'
        )
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
        project = tmp_path / "proj"
        project.mkdir()
        (project / "swival.toml").write_text('decision_model = ""\n')
        merged = load_config(project)
        assert (merged["decision_model"], merged["decision_base_url"]) == ("", LOCAL)

    @pytest.mark.parametrize(
        "config, expected",
        [
            ('decision_api_key = "k"\n', "'decision_api_key' in"),
            (
                'api_key = "a"\ndecision_api_key = "k"\n',
                "'api_key' and 'decision_api_key'",
            ),
        ],
    )
    def test_checked_in_key_warning(
        self, tmp_path, monkeypatch, capsys, config, expected
    ):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        (tmp_path / ".git").mkdir()
        (tmp_path / "swival.toml").write_text(config)
        load_config(tmp_path)
        assert expected in capsys.readouterr().err

    def test_session_forwarding(self):
        """Serve and ACP build their sessions from args_to_session_kwargs."""
        args = _parse(
            ["--decision-model", "clef", "--decision-base-url", LOCAL]
            + ["--decision-api-key", "k"]
        )
        kwargs = args_to_session_kwargs(args, "/tmp")
        assert (
            kwargs["decision_model"],
            kwargs["decision_base_url"],
            kwargs["decision_api_key"],
        ) == ("clef", LOCAL, "k")
        assert "decision_model" not in args_to_session_kwargs(_parse([]), "/tmp")
        config = {"decision_model": "", "decision_api_key": "k"}
        Session(**config_to_session_kwargs(config))

    @pytest.mark.parametrize(
        "extra, rejected", [([], True), (["--no-decisions"], False)]
    )
    def test_network_none(self, tmp_path, extra, rejected):
        args = _parse(
            ["--network", "none", "--provider", "command", "--model", "x", *extra],
            {"decision_model": "clef", "decision_base_url": LOCAL},
        )
        args._mcp_servers_toml = args._a2a_servers_toml = None
        error, _ = agent._resolve_network_policy(
            args, tmp_path, network_cli=True, sandbox_cli=False, sandbox_config=False
        )
        assert ("cannot be used with a decision model" in (error or "")) is rejected


class TestSession:
    def _session(self, tmp_path, monkeypatch, **kwargs):
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        return Session(base_dir=str(tmp_path), history=False, **kwargs)

    def test_run_uses_decide(self, tmp_path, monkeypatch, systemone):
        llm = _decide_then_answer()
        monkeypatch.setattr(agent, "call_llm", llm)
        lines = []
        monkeypatch.setattr("swival.fmt.info", lines.append)
        s = self._session(
            tmp_path,
            monkeypatch,
            verbose=True,
            decision_model="clef",
            decision_base_url=LOCAL,
        )
        result = s.run("classify", report=True)
        assert f"decision model=hosted_vllm/clef  endpoint={LOCAL}" in lines
        assert "decide" in llm.offered[0]
        assert systemone.calls[0]["model"] == "hosted_vllm/clef"
        exposure = result.report["stats"]["exposure"]["by_source"]["decision"]
        assert exposure["requests"]["sent"] == 1

    @pytest.mark.parametrize("model", [None, ""])
    def test_disabled(self, tmp_path, monkeypatch, model):
        s = self._session(tmp_path, monkeypatch, decision_model=model)
        s._setup()
        assert "decide" not in [t["function"]["name"] for t in s._tools]
        assert s._build_loop_kwargs(s._make_per_run_state())["decisions"] is None

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"decision_model": "clef"}, "needs a server URL"),
            (
                {
                    "provider": "command",
                    "model": "true",
                    "network": "none",
                    "decision_model": "clef",
                    "decision_base_url": LOCAL,
                },
                "cannot be used with a decision model",
            ),
        ],
    )
    def test_setup_errors(self, tmp_path, monkeypatch, kwargs, message):
        s = self._session(tmp_path, monkeypatch, **kwargs)
        with pytest.raises(ConfigError, match=message):
            s._setup()


class TestCli:
    def _main(self, tmp_path, monkeypatch, argv):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        args = agent.build_parser().parse_args(
            ["--base-dir", str(tmp_path), "--no-history", "--quiet", *argv, "task"]
        )
        monkeypatch.setattr(sys, "argv", ["swival"])
        monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: args)
        try:
            agent.main()
        except SystemExit as e:
            if e.code not in (0, None):
                raise
        return args

    def test_run_uses_decide(self, tmp_path, monkeypatch, systemone):
        llm = _decide_then_answer()
        monkeypatch.setattr(agent, "call_llm", llm)
        argv = ["--decision-model", "clef", "--decision-base-url", LOCAL]
        self._main(tmp_path, monkeypatch, argv)
        assert "decide" in llm.offered[0]
        assert len(systemone.calls) == 1

    def test_no_decisions_overrides_project_config(self, tmp_path, monkeypatch):
        (tmp_path / "swival.toml").write_text(
            f'decision_model = "clef"\ndecision_base_url = "{LOCAL}"\n'
        )
        llm = _RecordingLLM()
        monkeypatch.setattr(agent, "call_llm", llm)
        args = self._main(tmp_path, monkeypatch, ["--no-decisions"])
        assert args._decisions is None
        assert "decide" not in llm.offered[0]

    def test_bare_model_without_url(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(agent, "call_llm", ScriptedLLM([]))
        with pytest.raises(SystemExit) as ei:
            self._main(tmp_path, monkeypatch, ["--decision-model", "clef"])
        assert ei.value.code == 2
        assert "needs a server URL" in capsys.readouterr().err

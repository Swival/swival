"""Tests for replay-weighted token exposure accounting."""

import json
import threading
import types
from unittest.mock import MagicMock, patch

import pytest

from swival import tokens
from swival._msg import COMPACTION_MARKER, RECAP_MARKER
from swival.cost import CostObservation
from swival.exposure import PROVIDER_RETRY_NOTE, ExposureMeter, _TokenCache
from swival.report import ReportCollector
from swival.tokens import count_tokens


def _call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _transcript(result="alpha beta gamma " * 50):
    return [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Fix the bug."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_call("c1", "read_file", '{"file_path": "a.py"}')],
        },
        {"role": "tool", "tool_call_id": "c1", "content": result},
    ]


def _input(meter, source="agent"):
    return meter.to_dict()["by_source"][source]["input"]


class TestAttribution:
    def test_repeated_inclusion_counts_every_request(self):
        meter = ExposureMeter()
        rec = meter.recorder("agent")
        messages = _transcript()
        size = count_tokens(messages[3]["content"])
        for _ in range(3):
            rec.request(messages)
        data = meter.to_dict()
        assert data["input"]["tool_results"] == {"read_file": 3 * size}
        assert data["results"] == {"read_file": {"count": 1, "tokens": size}}
        assert data["requests"]["sent"] == 3

    def test_replacement_counts_only_what_is_sent(self):
        meter = ExposureMeter()
        rec = meter.recorder("agent")
        messages = _transcript()
        original = count_tokens(messages[3]["content"])
        rec.request(messages)
        messages[3]["content"] = "[read_file a.py: 50 lines]"
        replacement = count_tokens(messages[3]["content"])
        rec.request(messages)
        data = meter.to_dict()
        assert data["input"]["tool_results"]["read_file"] == original + replacement
        assert data["results"]["read_file"] == {"count": 1, "tokens": original}

    def test_dropped_messages_stop_counting(self):
        meter = ExposureMeter()
        rec = meter.recorder("agent")
        messages = _transcript()
        size = count_tokens(messages[3]["content"])
        rec.request(messages)
        rec.request(messages[:2])
        assert _input(meter)["tool_results"] == {"read_file": size}
        assert _input(meter)["tool_arguments"] == {
            "read_file": count_tokens('{"file_path": "a.py"}')
        }

    def test_mixed_recaps_are_summaries_without_a_tool_split(self):
        meter = ExposureMeter()
        rec = meter.recorder("agent")
        recap = RECAP_MARKER + " — factual summary]\n\nread a.py, ran grep"
        snapshot = "[snapshot: investigation]\nfound the bug in a.py"
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": recap},
            {"role": "user", "content": COMPACTION_MARKER},
            {"role": "assistant", "content": snapshot},
        ]
        rec.request(messages)
        data = _input(meter)
        assert data["summaries"] == {
            "compaction": count_tokens(recap) + count_tokens(COMPACTION_MARKER),
            "snapshot": count_tokens(snapshot),
        }
        assert data["tool_results"] == {}
        assert data["assistant"] == 0
        assert data["user"] == 0

    def test_unknown_tool_result_is_unattributed(self):
        meter = ExposureMeter()
        meter.recorder("agent").request(
            [{"role": "tool", "tool_call_id": "gone", "content": "orphan output"}]
        )
        data = _input(meter)
        assert data["tool_results"] == {}
        assert data["unattributed"] == count_tokens("orphan output")

    def test_tool_name_on_message_attributes_without_the_call(self):
        meter = ExposureMeter()
        meter.recorder("agent").request(
            [{"role": "tool", "tool_call_id": "x", "name": "grep", "content": "hit"}]
        )
        assert _input(meter)["tool_results"] == {"grep": count_tokens("hit")}

    def test_system_schemas_and_conversation_are_separate(self):
        meter = ExposureMeter()
        schemas = [
            {"type": "function", "function": {"name": "read_file"}},
            {"type": "function", "function": {"name": "mcp__docs__search"}},
            {"type": "function", "function": {"name": "a2a__peer__ask"}},
        ]
        messages = _transcript()
        messages.append(
            {"role": "assistant", "content": "Done.", "reasoning_content": "hmm"}
        )
        meter.recorder("agent").request(messages, schemas)
        data = _input(meter)
        assert data["system"] == count_tokens("You are a coding agent.")
        assert data["user"] == count_tokens("Fix the bug.")
        assert data["assistant"] == count_tokens("Done.")
        assert data["reasoning"] == count_tokens("hmm")
        assert data["tool_schemas"] == {
            "builtin": count_tokens(json.dumps(schemas[0])),
            "mcp": count_tokens(json.dumps(schemas[1])),
            "a2a": count_tokens(json.dumps(schemas[2])),
        }
        parts = (
            data["system"]
            + data["user"]
            + data["assistant"]
            + data["reasoning"]
            + sum(data["tool_schemas"].values())
            + sum(data["tool_results"].values())
            + sum(data["tool_arguments"].values())
        )
        assert data["total"] == parts

    def test_synthetic_flags_mark_scaffolding(self):
        meter = ExposureMeter()
        rec = meter.recorder("agent")
        messages = [
            {"role": "user", "content": "real task"},
            {"role": "user", "content": "nudge text"},
        ]
        rec.request(messages, synthetic=[False, True])
        data = _input(meter)
        assert data["user"] == count_tokens("real task")
        assert data["scaffolding"] == count_tokens("nudge text")

    def test_misaligned_flags_are_ignored(self):
        meter = ExposureMeter()
        meter.recorder("agent").request(
            [{"role": "user", "content": "real task"}], synthetic=[True, True]
        )
        assert _input(meter)["scaffolding"] == 0

    def test_always_synthetic_prefix_needs_no_flag(self):
        meter = ExposureMeter()
        text = "[REVIEWER FEEDBACK] try again"
        meter.recorder("agent").request([{"role": "user", "content": text}])
        assert _input(meter)["scaffolding"] == count_tokens(text)

    def test_image_parts_are_counted_not_tokenized(self):
        meter = ExposureMeter()
        content = [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]
        meter.recorder("agent").request([{"role": "user", "content": content}])
        data = _input(meter)
        assert data["image_parts"] == 1
        assert data["user"] == count_tokens("look")

    def test_unicode_uses_the_shared_tokenizer(self):
        meter = ExposureMeter()
        text = "résumé 東京   ✓ " * 100
        meter.recorder("agent").request(_transcript(result=text))
        assert _input(meter)["tool_results"] == {"read_file": count_tokens(text)}


class TestOutputAndUsage:
    def test_generated_arguments_are_output(self):
        meter = ExposureMeter()
        msg = types.SimpleNamespace(
            tool_calls=[
                types.SimpleNamespace(
                    id="c1",
                    function=types.SimpleNamespace(
                        name="edit_file", arguments='{"old_string": "a"}'
                    ),
                )
            ]
        )
        meter.recorder("agent").generated(msg.tool_calls)
        data = meter.to_dict()
        assert data["output"]["tool_arguments"] == {
            "edit_file": {"calls": 1, "tokens": count_tokens('{"old_string": "a"}')}
        }
        assert data["input"]["tool_arguments"] == {}

    def test_missing_usage_is_unavailable_not_zero(self):
        meter = ExposureMeter()
        rec = meter.recorder("agent")
        rec.request(_transcript())
        rec.response(None, CostObservation("unavailable"))
        data = meter.to_dict()
        assert data["provider_usage"] == {"responses": 1, "reported": 0, "usage": None}
        assert data["cost"]["known_usd"] is None
        assert data["cost"]["unpriced_calls"] == 1

    def test_usage_and_known_cost_accumulate(self):
        meter = ExposureMeter()
        rec = meter.recorder("agent")
        usage = types.SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            prompt_tokens_details=types.SimpleNamespace(cached_tokens=60),
            cache_creation_input_tokens=5,
        )
        rec.response(usage, CostObservation("known", 0.25))
        rec.response(
            {"prompt_tokens": 10, "completion_tokens": 1}, CostObservation("known", 0.0)
        )
        rec.response(None, CostObservation("not_applicable"))
        data = meter.to_dict()
        assert data["provider_usage"]["usage"] == {
            "input_tokens": 110,
            "output_tokens": 21,
            "cached_tokens": 60,
            "cache_write_tokens": 5,
        }
        assert data["provider_usage"]["reported"] == 2
        assert data["cost"] == {
            "known_usd": 0.25,
            "priced_calls": 2,
            "unpriced_calls": 0,
            "not_applicable_calls": 1,
        }

    def test_known_zero_cost_stays_zero(self):
        meter = ExposureMeter()
        meter.recorder("agent").response(None, CostObservation("known", 0.0))
        assert meter.to_dict()["cost"]["known_usd"] == 0.0


class TestMeter:
    def test_empty_meter_reports_nothing(self):
        assert ExposureMeter().to_dict() is None

    def test_sources_are_split_and_totalled(self):
        meter = ExposureMeter()
        meter.recorder("agent").request(_transcript())
        meter.recorder("subagent").request(_transcript())
        meter.recorder("summary").cache_hit()
        data = meter.to_dict()
        assert set(data["by_source"]) == {"agent", "subagent", "summary"}
        assert data["requests"]["sent"] == 2
        assert data["requests"]["response_cache_hits"] == 1
        per_source = data["by_source"]["agent"]["input"]["total"]
        assert data["input"]["total"] == 2 * per_source

    def test_same_call_id_in_two_sources_is_a_result_in_each(self):
        meter = ExposureMeter()
        meter.recorder("agent").request(_transcript())
        meter.recorder("subagent").request(_transcript())
        assert meter.to_dict()["results"]["read_file"]["count"] == 2

    def test_concurrent_recording_is_not_lost(self):
        meter = ExposureMeter()
        messages = _transcript()
        size = count_tokens(messages[3]["content"])

        def worker():
            rec = meter.recorder("subagent")
            for _ in range(50):
                rec.request(messages)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        data = meter.to_dict()
        assert data["requests"]["sent"] == 400
        assert data["input"]["tool_results"]["read_file"] == 400 * size

    def test_provider_note_only_for_provider_requests(self):
        meter = ExposureMeter()
        meter.recorder("agent").request(_transcript(), provider=False)
        assert PROVIDER_RETRY_NOTE not in meter.to_dict()["coverage"]
        meter.recorder("agent").request(_transcript())
        assert PROVIDER_RETRY_NOTE in meter.to_dict()["coverage"]

    def test_byte_fallback_is_labelled(self, monkeypatch):
        monkeypatch.setattr(tokens, "_encoder", tokens._FallbackEncoder())
        monkeypatch.setattr(tokens, "_is_fallback", True)
        meter = ExposureMeter()
        text = "é" * 300
        meter.recorder("agent").request(_transcript(result=text))
        data = meter.to_dict()
        assert data["unit"] == "bytes"
        assert any("UTF-8 bytes" in note for note in data["coverage"])
        assert data["input"]["tool_results"] == {"read_file": len(text.encode())}


class TestTokenCache:
    def test_hits_skip_the_tokenizer(self, monkeypatch):
        cache = _TokenCache()
        text = "x" * 1000
        calls = []

        def counting(s):
            calls.append(s)
            return 7

        monkeypatch.setattr("swival.exposure.count_tokens", counting)
        assert cache.count(text) == 7
        assert cache.count(text) == 7
        assert len(calls) == 1

    def test_changed_content_is_recounted(self):
        cache = _TokenCache()
        a = "word " * 100
        b = a + "more"
        assert cache.count(a) == count_tokens(a)
        assert cache.count(b) == count_tokens(b)

    def test_eviction_keeps_the_bound(self):
        cache = _TokenCache(max_chars=1000)
        for i in range(10):
            cache.count(str(i) * 300)
        assert cache._chars <= 1000
        assert len(cache._entries) == 3


def _response(content="hello", tool_calls=None, usage=None, cost=None):
    msg = types.SimpleNamespace(
        content=content, tool_calls=tool_calls, role="assistant"
    )
    choice = types.SimpleNamespace(message=msg, finish_reason="stop")
    resp = types.SimpleNamespace(choices=[choice], usage=usage)
    if cost is not None:
        resp._hidden_params = {"response_cost": cost}
    return resp


def _call_llm(messages, meter, **kwargs):
    from swival.agent import call_llm

    return call_llm(
        "http://localhost:8080/v1",
        "my-model",
        messages,
        100,
        0.5,
        None,
        None,
        kwargs.pop("tools", None),
        False,
        provider=kwargs.pop("provider", "openrouter"),
        api_key="test",
        exposure=meter.recorder("agent"),
        **kwargs,
    )


class TestCallLlm:
    def test_request_response_usage_and_cost(self):
        meter = ExposureMeter()
        tc = types.SimpleNamespace(
            id="c9",
            function=types.SimpleNamespace(name="grep", arguments='{"pattern": "x"}'),
        )
        usage = types.SimpleNamespace(
            prompt_tokens=42, completion_tokens=7, prompt_tokens_details=None
        )
        resp = _response(tool_calls=[tc], usage=usage, cost=0.5)
        with patch("litellm.completion", return_value=resp):
            _call_llm(_transcript(), meter)
        data = meter.to_dict()
        assert data["requests"] == {
            "sent": 1,
            "resent": 0,
            "failed": 0,
            "response_cache_hits": 0,
        }
        assert data["input"]["tool_results"]["read_file"] > 0
        assert data["output"]["tool_arguments"]["grep"]["calls"] == 1
        assert data["provider_usage"]["usage"]["input_tokens"] == 42
        assert data["cost"]["known_usd"] == 0.5

    def test_transient_retry_counts_the_resend(self):
        import litellm

        meter = ExposureMeter()
        exc = litellm.APIConnectionError(
            message="Connection reset", llm_provider="openai", model="x"
        )
        mock = MagicMock(side_effect=[exc, _response()])
        with patch("litellm.completion", mock), patch("time.sleep"):
            _call_llm(_transcript(), meter)
        data = meter.to_dict()
        assert data["requests"]["sent"] == 2
        assert data["requests"]["resent"] == 1
        assert data["requests"]["failed"] == 1
        size = count_tokens(_transcript()[3]["content"])
        assert data["input"]["tool_results"]["read_file"] == 2 * size
        assert data["results"]["read_file"]["count"] == 1

    def test_failed_request_is_still_exposure(self):
        from swival.agent import AgentError

        meter = ExposureMeter()
        with patch("litellm.completion", side_effect=ValueError("boom")):
            with pytest.raises(AgentError):
                _call_llm(_transcript(), meter)
        data = meter.to_dict()
        assert data["requests"]["sent"] == 1
        assert data["requests"]["failed"] == 1
        assert data["provider_usage"]["responses"] == 0

    def test_response_cache_hit_is_not_a_request(self, tmp_path):
        from swival.cache import LLMCache

        meter = ExposureMeter()
        cache = LLMCache(tmp_path / "cache.db")
        cache.open()
        mock = MagicMock(return_value=_response())
        try:
            with patch("litellm.completion", mock):
                _call_llm(_transcript(), meter, cache=cache)
                _call_llm(_transcript(), meter, cache=cache)
        finally:
            cache.close()
        assert mock.call_count == 1
        data = meter.to_dict()
        assert data["requests"]["sent"] == 1
        assert data["requests"]["response_cache_hits"] == 1
        size = count_tokens(_transcript()[3]["content"])
        assert data["input"]["tool_results"]["read_file"] == size

    def test_synthetic_flags_survive_internal_key_stripping(self):
        meter = ExposureMeter()
        messages = _transcript()
        messages.append(
            {"role": "user", "content": "keep going", "_swival_synthetic": True}
        )
        sent = {}

        def fake(**kwargs):
            sent["messages"] = kwargs["messages"]
            return _response()

        with patch("litellm.completion", side_effect=fake):
            _call_llm(messages, meter)
        assert not any(k.startswith("_") for m in sent["messages"] for k in m)
        assert _input(meter)["scaffolding"] == count_tokens("keep going")

    def test_tool_schemas_are_the_ones_sent(self):
        meter = ExposureMeter()
        schemas = [{"type": "function", "function": {"name": "read_file"}}]
        with patch("litellm.completion", return_value=_response()):
            _call_llm(_transcript(), meter, tools=schemas)
        assert _input(meter)["tool_schemas"] == {
            "builtin": count_tokens(json.dumps(schemas[0]))
        }

    def test_command_provider_counts_each_run(self):
        from swival.agent import call_llm

        meter = ExposureMeter()
        call_llm(
            None,
            "printf done",
            _transcript(),
            100,
            None,
            None,
            None,
            None,
            False,
            provider="command",
            exposure=meter.recorder("agent"),
        )
        data = meter.to_dict()
        assert data["requests"]["sent"] == 1
        assert data["requests"]["failed"] == 0
        assert data["cost"]["not_applicable_calls"] == 1
        assert PROVIDER_RETRY_NOTE not in data["coverage"]


def _loop_kwargs(tmp_path, **overrides):
    from swival.thinking import ThinkingState
    from swival.todo import TodoState

    defaults = dict(
        api_base="http://127.0.0.1:1234",
        model_id="test-model",
        max_turns=4,
        max_output_tokens=1024,
        temperature=0.5,
        top_p=None,
        seed=None,
        context_length=None,
        base_dir=str(tmp_path),
        thinking_state=ThinkingState(verbose=False),
        resolved_commands={},
        skills_catalog={},
        skill_read_roots=[],
        extra_write_roots=[],
        files_mode="some",
        verbose=False,
        llm_kwargs={"provider": "lmstudio", "api_key": None},
        file_tracker=None,
        todo_state=TodoState(verbose=False),
        continue_here=False,
    )
    defaults.update(overrides)
    return defaults


class TestAgentLoop:
    def test_agent_and_summary_requests_are_attributed(self, tmp_path):
        from swival.agent import CompactionState, run_agent_loop
        from swival.tools import TOOLS

        (tmp_path / "a.txt").write_text("hello exposure\n" * 20)
        read_call = types.SimpleNamespace(
            id="r1",
            function=types.SimpleNamespace(
                name="read_file", arguments='{"file_path": "a.txt"}'
            ),
        )
        replies = [_response(content=None, tool_calls=[read_call]), _response("done")]

        def fake(**kwargs):
            if kwargs.get("max_tokens") == 512 and "tools" not in kwargs:
                return _response("summary of the work")
            return replies.pop(0)

        meter = ExposureMeter()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "read a.txt"},
        ]
        tools = [t for t in TOOLS if t["function"]["name"] == "read_file"]
        with patch("litellm.completion", side_effect=fake):
            answer, _ = run_agent_loop(
                messages,
                tools,
                **_loop_kwargs(
                    tmp_path,
                    exposure_meter=meter,
                    compaction_state=CompactionState(checkpoint_interval=1),
                ),
            )
        assert answer == "done"
        data = meter.to_dict()["by_source"]
        assert data["agent"]["requests"]["sent"] == 2
        assert data["summary"]["requests"]["sent"] == 1
        result = next(m for m in messages if m.get("role") == "tool")
        agent_input = data["agent"]["input"]
        assert agent_input["tool_results"] == {
            "read_file": count_tokens(result["content"])
        }
        assert agent_input["tool_arguments"] == {
            "read_file": count_tokens('{"file_path": "a.txt"}')
        }
        assert data["agent"]["output"]["tool_arguments"]["read_file"]["calls"] == 1
        assert data["summary"]["input"]["tool_results"] == {}

    def test_nested_loop_files_under_subagent(self, tmp_path):
        from swival.agent import run_agent_loop

        meter = ExposureMeter()
        with patch("litellm.completion", return_value=_response("ok")):
            run_agent_loop(
                [{"role": "user", "content": "hi"}],
                [],
                **_loop_kwargs(tmp_path, exposure_meter=meter, is_subagent=True),
            )
        assert set(meter.to_dict()["by_source"]) == {"subagent"}

    def test_subagent_template_carries_the_meter(self, monkeypatch):
        from swival.subagent import (
            SA_TEMPLATE_EXCLUDE,
            SubagentHandle,
            _CompositeCancelFlag,
            _subagent_thread_fn,
        )

        assert "exposure_meter" not in SA_TEMPLATE_EXCLUDE
        meter = ExposureMeter()
        seen = {}

        def fake_loop(messages, tools, **kwargs):
            seen.update(kwargs)
            return "done", False

        monkeypatch.setattr("swival.agent.run_agent_loop", fake_loop)
        handle = SubagentHandle(id="sub_1", task="t")
        slot = threading.Semaphore(1)
        slot.acquire()
        _subagent_thread_fn(
            handle,
            {"base_dir": "/tmp/test", "exposure_meter": meter},
            [],
            "task",
            5,
            None,
            None,
            _CompositeCancelFlag(None, handle.cancel_flag),
            slot,
        )
        assert seen["exposure_meter"] is meter
        assert seen["is_subagent"] is True


class TestReport:
    def test_report_omits_exposure_without_requests(self):
        report = ReportCollector()
        built = report.build_report(
            task="t",
            model="m",
            provider="p",
            settings={},
            outcome="success",
            answer="a",
            exit_code=0,
            turns=1,
        )
        assert "exposure" not in built["stats"]

    def test_session_report_includes_exposure(self, tmp_path, monkeypatch):
        from swival import agent
        from swival.session import Session

        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        s = Session(base_dir=str(tmp_path), history=False)
        with patch("litellm.completion", return_value=_response("the answer")):
            result = s.run("question", report=True)
        exposure = result.report["stats"]["exposure"]
        assert exposure["by_source"]["agent"]["requests"]["sent"] == 1
        assert exposure["input"]["system"] > 0
        assert exposure["cost"]["not_applicable_calls"] == 1


class TestReviewFixes:
    def test_reused_call_ids_attribute_to_the_latest_earlier_call(self):
        meter = ExposureMeter()
        messages = [
            {"role": "assistant", "tool_calls": [_call("call_0", "read_file", "{}")]},
            {"role": "tool", "tool_call_id": "call_0", "content": "first result"},
            {"role": "assistant", "tool_calls": [_call("call_0", "grep", "{}")]},
            {
                "role": "tool",
                "tool_call_id": "call_0",
                "content": "second distinct result",
            },
        ]
        meter.recorder("agent").request(messages)
        data = meter.to_dict()
        assert data["input"]["tool_results"] == {
            "read_file": count_tokens("first result"),
            "grep": count_tokens("second distinct result"),
        }
        assert data["results"]["read_file"]["count"] == 1
        assert data["results"]["grep"]["count"] == 1

    def test_reused_ids_count_as_new_results_across_requests(self):
        meter = ExposureMeter()
        rec = meter.recorder("agent")
        first = [
            {"role": "assistant", "tool_calls": [_call("call_0", "read_file", "{}")]},
            {"role": "tool", "tool_call_id": "call_0", "content": "one"},
        ]
        second = first + [
            {"role": "assistant", "tool_calls": [_call("call_0", "read_file", "{}")]},
            {"role": "tool", "tool_call_id": "call_0", "content": "two"},
        ]
        rec.request(first)
        rec.request(second)
        assert meter.to_dict()["results"]["read_file"]["count"] == 2

    def test_outbound_filter_drops_positional_flags(self, tmp_path):
        script = tmp_path / "passthrough.py"
        script.write_text(
            "import json, sys\n"
            "payload = json.load(sys.stdin)\n"
            "print(json.dumps({'messages': list(reversed(payload['messages']))}))\n"
        )
        meter = ExposureMeter()
        messages = [
            {"role": "user", "content": "nudge text", "_swival_synthetic": True},
            {"role": "user", "content": "real task"},
        ]
        with patch("litellm.completion", return_value=_response()):
            _call_llm(messages, meter, llm_filter=f"python3 {script}")
        data = _input(meter)
        assert data["scaffolding"] == 0
        assert data["user"] == count_tokens("nudge text") + count_tokens("real task")

    def test_command_provider_records_generated_calls(self, monkeypatch):
        from swival import agent

        outputs = [
            '<swival:call id="c1" name="read_file">{"file_path": "a.py"}</swival:call>',
            "all done",
        ]
        monkeypatch.setattr(agent, "_run_command_once", lambda *a, **k: outputs.pop(0))
        monkeypatch.setattr(
            agent,
            "handle_tool_call",
            lambda tc, **k: (
                {"role": "tool", "tool_call_id": tc.id, "content": "file body"},
                {
                    "name": tc.function.name,
                    "arguments": {},
                    "elapsed": 0.0,
                    "succeeded": True,
                },
            ),
        )
        meter = ExposureMeter()
        agent.call_llm(
            None,
            "fake-agent",
            [{"role": "user", "content": "go"}],
            100,
            None,
            None,
            None,
            None,
            False,
            provider="command",
            exposure=meter.recorder("agent"),
            command_tool_kwargs={
                "handle_tool_call_kwargs": {},
                "outer_turn": 1,
                "outer_turn_offset": 0,
                "report": None,
                "snapshot_state": None,
                "_emit": lambda *a: None,
            },
        )
        data = meter.to_dict()
        assert data["requests"]["sent"] == 2
        assert data["output"]["tool_arguments"] == {
            "read_file": {"calls": 1, "tokens": count_tokens('{"file_path": "a.py"}')}
        }
        assert data["input"]["tool_results"] == {"read_file": count_tokens("file body")}


class TestCommandTranscriptAndScavenging:
    def test_transcript_mode_skips_structured_calls_and_reasoning(self):
        meter = ExposureMeter()
        messages = [
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "private thoughts",
                "tool_calls": [_call("c1", "read_file", '{"file_path": "big.py"}')],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "file body"},
        ]
        meter.recorder("agent").request(messages, transcript=True)
        data = _input(meter)
        assert data["tool_arguments"] == {}
        assert data["reasoning"] == 0
        assert data["tool_results"] == {"read_file": count_tokens("file body")}

    def test_transcript_mode_counts_swival_call_blocks_as_arguments(self):
        meter = ExposureMeter()
        block = '<swival:call id="a" name="grep">{"pattern": "x"}</swival:call>'
        messages = [
            {"role": "assistant", "content": "Searching.\n" + block},
            {"role": "tool", "tool_call_id": "a", "name": "grep", "content": "hit"},
        ]
        meter.recorder("agent").request(messages, transcript=True)
        data = _input(meter)
        assert data["tool_arguments"] == {"grep": count_tokens('{"pattern": "x"}')}
        assert data["assistant"] == count_tokens("Searching.\n")

    def test_transcript_mode_counts_reused_ids_as_new_results(self):
        meter = ExposureMeter()
        rec = meter.recorder("agent")

        def round_(n):
            block = f'<swival:call id="c1" name="read_file">{{"n": {n}}}</swival:call>'
            return [
                {"role": "assistant", "content": block},
                {"role": "tool", "tool_call_id": "c1", "content": f"body {n}"},
            ]

        rec.request(round_(1), transcript=True)
        rec.request(round_(1) + round_(2), transcript=True)
        assert meter.to_dict()["results"]["read_file"]["count"] == 2

    def test_scavenged_calls_count_as_generated_output(self):
        from swival.agent import _maybe_scavenge_tool_calls

        meter = ExposureMeter()
        content = '<swival:call id="s1" name="think">{"thought": "hmm"}</swival:call>'
        msg = types.SimpleNamespace(content=content, tool_calls=None, role="assistant")
        tools = [{"type": "function", "function": {"name": "think"}}]
        added = _maybe_scavenge_tool_calls(
            msg,
            "stop",
            tools,
            turn=1,
            report=None,
            verbose=False,
            exposure=meter.recorder("agent"),
        )
        assert added == 1
        output = meter.to_dict()["output"]["tool_arguments"]
        assert output["think"]["calls"] == 1


class TestResultIdentity:
    def _turn(self, call_id, content):
        return [
            {"role": "assistant", "tool_calls": [_call(call_id, "read_file", "{}")]},
            {"role": "tool", "tool_call_id": call_id, "content": content},
        ]

    def test_subagents_reusing_call_ids_are_distinct_results(self):
        from swival.exposure import tag_results

        meter = ExposureMeter()
        for content in ("alpha", "beta"):
            messages = self._turn("call_0", content)
            meter.recorder("subagent").request(
                messages, result_keys=tag_results(messages)
            )
        assert meter.to_dict()["results"]["read_file"]["count"] == 2

    def test_dropping_older_turns_keeps_result_identities(self):
        from swival.exposure import tag_results

        meter = ExposureMeter()
        rec = meter.recorder("agent")
        first, second, third = (
            self._turn("call_0", c) for c in ("one", "two", "three")
        )
        history = first + second
        rec.request(history, result_keys=tag_results(history))
        history = second + third
        rec.request(history, result_keys=tag_results(history))
        results = meter.to_dict()["results"]["read_file"]
        assert results["count"] == 3
        assert results["tokens"] == sum(
            count_tokens(c) for c in ("one", "two", "three")
        )

    def test_call_llm_names_results_without_sending_the_name(self):
        from swival.exposure import RESULT_KEY

        meter = ExposureMeter()
        first, second, third = (
            self._turn("call_0", c) for c in ("one", "two", "three")
        )
        sent = []

        def fake(**kwargs):
            sent.append(kwargs["messages"])
            return _response()

        with patch("litellm.completion", side_effect=fake):
            _call_llm(first + second, meter)
            _call_llm(second + third, meter)
        assert not any(RESULT_KEY in m for batch in sent for m in batch)
        assert RESULT_KEY in second[1]
        assert meter.to_dict()["results"]["read_file"]["count"] == 3

    def test_filter_keeps_names_for_call_ids_unique_in_a_request(self, tmp_path):
        script = tmp_path / "rotate.py"
        script.write_text(
            "import json, sys\n"
            "m = json.load(sys.stdin)['messages']\n"
            "print(json.dumps({'messages': m[-2:] + m[:-2]}))\n"
        )
        meter = ExposureMeter()
        for content in ("one", "two"):
            conversation = [{"role": "user", "content": "go"}] + self._turn(
                "call_0", content
            )
            with patch("litellm.completion", return_value=_response()):
                _call_llm(conversation, meter, llm_filter=f"python3 {script}")
        assert meter.to_dict()["results"]["read_file"]["count"] == 2

    def test_rekey_matches_repeated_call_ids_in_order(self):
        from swival.exposure import rekey_results, tag_results

        before = self._turn("call_0", "one") + self._turn("call_0", "two")
        keys = tag_results(before)
        after = [dict(m) for m in before]
        assert rekey_results(before, keys, after) == keys

    def test_rekey_falls_back_when_a_filter_drops_a_result(self):
        from swival.exposure import rekey_results, tag_results

        before = self._turn("call_0", "one") + self._turn("call_0", "two")
        keys = tag_results(before)
        assert rekey_results(before, keys, before[:3]) == [None, None, None]

    def test_filter_keeps_results_sharing_an_id_in_one_message(self, tmp_path):
        script = tmp_path / "passthrough.py"
        script.write_text(
            "import json, sys\n"
            "print(json.dumps({'messages': json.load(sys.stdin)['messages']}))\n"
        )
        meter = ExposureMeter()
        conversation = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "tool_calls": [
                    _call("call_0", "read_file", "{}"),
                    _call("call_0", "grep", "{}"),
                ],
            },
            {"role": "tool", "tool_call_id": "call_0", "content": "file body"},
            {"role": "tool", "tool_call_id": "call_0", "content": "grep hits"},
        ]
        with patch("litellm.completion", return_value=_response()):
            _call_llm(conversation, meter, llm_filter=f"python3 {script}")
            _call_llm(conversation, meter, llm_filter=f"python3 {script}")
        results = meter.to_dict()["results"]
        assert sum(r["count"] for r in results.values()) == 2

    def test_fallback_separates_results_of_one_message(self):
        meter = ExposureMeter()
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    _call("call_0", "read_file", "{}"),
                    _call("call_0", "read_file", "{}"),
                ],
            },
            {"role": "tool", "tool_call_id": "call_0", "content": "one"},
            {"role": "tool", "tool_call_id": "call_0", "content": "two"},
        ]
        meter.recorder("agent").request(messages)
        assert meter.to_dict()["results"]["read_file"]["count"] == 2

    def test_command_rounds_name_their_own_results(self, monkeypatch):
        from swival import agent

        block = '<swival:call id="c1" name="read_file">{"n": 1}</swival:call>'
        outputs = [block, block, "done"]
        monkeypatch.setattr(agent, "_run_command_once", lambda *a, **k: outputs.pop(0))
        bodies = iter(["first body", "second body"])
        monkeypatch.setattr(
            agent,
            "handle_tool_call",
            lambda tc, **k: (
                {"role": "tool", "tool_call_id": tc.id, "content": next(bodies)},
                {
                    "name": tc.function.name,
                    "arguments": {},
                    "elapsed": 0.0,
                    "succeeded": True,
                },
            ),
        )
        meter = ExposureMeter()
        agent.call_llm(
            None,
            "fake-agent",
            [{"role": "user", "content": "go"}],
            100,
            None,
            None,
            None,
            None,
            False,
            provider="command",
            exposure=meter.recorder("agent"),
            command_tool_kwargs={
                "handle_tool_call_kwargs": {},
                "outer_turn": 1,
                "outer_turn_offset": 0,
                "report": None,
                "snapshot_state": None,
                "_emit": lambda *a: None,
            },
        )
        results = meter.to_dict()["results"]["read_file"]
        assert results["count"] == 2
        assert results["tokens"] == count_tokens("first body") + count_tokens(
            "second body"
        )

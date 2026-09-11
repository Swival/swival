"""Tests for the shared output reservation and the budget it leaves the prompt."""

import types

import pytest

from swival import agent
from swival.agent import (
    MIN_OUTPUT_TOKENS,
    OUTPUT_RESERVE_CEILING,
    _prompt_budget,
    clamp_output_tokens,
    output_reserve,
)
from swival.report import ContextOverflowError, ReportCollector
from swival.thinking import ThinkingState
from swival.todo import TodoState

WINDOWS = [8192, 16_384, 32_768, 131_072]


def _msg(content=None, tool_calls=None):
    return types.SimpleNamespace(
        content=content, tool_calls=tool_calls, role="assistant"
    )


def _tool_schemas(count):
    return [
        {
            "type": "function",
            "function": {
                "name": f"tool_{i}",
                "description": "A tool with a long description. " * 20,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "x " * 40}
                    },
                },
            },
        }
        for i in range(count)
    ]


def _loop_kwargs(tmp_path, **overrides):
    defaults = dict(
        api_base="http://127.0.0.1:1234",
        model_id="test-model",
        max_turns=1,
        max_output_tokens=32_768,
        temperature=0.5,
        top_p=None,
        seed=None,
        context_length=16_384,
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
    )
    defaults.update(overrides)
    return defaults


class TestReservePolicy:
    @pytest.mark.parametrize("window", WINDOWS)
    def test_reserve_never_exceeds_the_ceiling(self, window):
        assert output_reserve(window, 32_768) <= OUTPUT_RESERVE_CEILING

    @pytest.mark.parametrize("window", WINDOWS)
    def test_reserve_never_exceeds_an_eighth_of_the_window(self, window):
        assert output_reserve(window, 32_768) <= max(MIN_OUTPUT_TOKENS, window // 8)

    @pytest.mark.parametrize("window", WINDOWS)
    def test_a_small_request_binds(self, window):
        assert output_reserve(window, 200) == 200

    @pytest.mark.parametrize("window", WINDOWS)
    def test_an_unset_request_reserves_the_minimum(self, window):
        assert output_reserve(window, None) == MIN_OUTPUT_TOKENS

    @pytest.mark.parametrize("window", WINDOWS)
    def test_the_prompt_keeps_most_of_the_window(self, window):
        budget = _prompt_budget(window, 32_768)
        assert budget >= int(window * 0.65)
        assert budget <= int(window * 0.90)

    def test_unknown_window_has_no_budget(self):
        assert _prompt_budget(None, 32_768) is None

    def test_the_128k_case_from_the_plan(self):
        assert output_reserve(131_072, 32_768) == 4096
        assert _prompt_budget(131_072, 32_768) == 113_868

    def test_the_16k_case_from_the_plan(self):
        assert _prompt_budget(16_384, 32_768) == 12_697


class TestClampingStaysConsistent:
    @pytest.mark.parametrize("window", WINDOWS)
    def test_output_may_exceed_the_reservation_when_there_is_room(self, window):
        messages = [{"role": "user", "content": "hi"}]
        allowed = clamp_output_tokens(messages, None, window, 32_768)
        assert allowed > output_reserve(window, 32_768)

    def test_output_never_exceeds_the_request(self):
        messages = [{"role": "user", "content": "hi"}]
        assert clamp_output_tokens(messages, None, 131_072, 512) == 512

    def test_a_prompt_at_the_new_threshold_still_leaves_room(self):
        # 128K window, prompt sized to the new budget: the plan expects about
        # 17,000 output tokens to remain.
        messages = [{"role": "user", "content": "word " * 113_000}]
        allowed = clamp_output_tokens(messages, None, 131_072, 32_768)
        assert 10_000 < allowed < 25_000

    def test_tool_heavy_prompt_is_counted_against_the_window(self):
        messages = [{"role": "user", "content": "hi"}]
        tools = _tool_schemas(60)
        with_tools = clamp_output_tokens(messages, tools, 32_768, 32_768)
        without = clamp_output_tokens(messages, None, 32_768, 32_768)
        assert with_tools < without

    def test_no_room_left_raises_for_the_caller_to_compact(self):
        messages = [{"role": "user", "content": "word " * 5000}]
        with pytest.raises(ContextOverflowError):
            clamp_output_tokens(messages, None, 1000, 4096)


class TestAdmissionAndRuntimeAgree:
    @pytest.mark.parametrize("window", WINDOWS)
    def test_an_admitted_prompt_is_under_the_runtime_budget(self, tmp_path, window):
        (tmp_path / "CLAUDE.md").write_text(
            "\n".join(f"- Rule {i}" for i in range(50)), encoding="utf-8"
        )
        result = agent.assemble_system_prompt(
            base_dir=str(tmp_path),
            system_prompt=None,
            no_system_prompt=False,
            no_instructions=False,
            no_memory=True,
            skills_catalog={},
            verbose=False,
            policy="autonomous",
            context_length=window,
            max_output_tokens=32_768,
            tools=_tool_schemas(20),
        )
        if not result.admission.admitted:
            pytest.skip("this window cannot hold the fixture")
        messages = [{"role": "system", "content": result.content}]
        cost = agent.estimate_tokens(messages, _tool_schemas(20))
        assert cost <= _prompt_budget(window, 32_768)


class TestProactiveCompactionFrequency:
    def test_a_16k_session_has_room_before_compaction_repeats(
        self, tmp_path, monkeypatch
    ):
        """An admitted 16K setup must do real work before compaction fires.

        The proactive pass aims below its budget by a hysteresis factor, so a
        single early compaction is normal. Firing on every turn is not.
        """
        (tmp_path / "CLAUDE.md").write_text(
            "\n".join(f"- Rule {i}: run `make check`." for i in range(30)),
            encoding="utf-8",
        )
        prompt = agent.assemble_system_prompt(
            base_dir=str(tmp_path),
            system_prompt=None,
            no_system_prompt=False,
            no_instructions=False,
            no_memory=True,
            skills_catalog={},
            verbose=False,
            policy="autonomous",
            context_length=16_384,
            max_output_tokens=32_768,
            tools=_tool_schemas(10),
        )
        assert prompt.admission.admitted

        turns = {"n": 0}

        def fake_call_llm(*args, **_kw):
            turns["n"] += 1
            if turns["n"] >= 6:
                return _msg(content="done"), "stop", [], 0, (0, 0)
            call = types.SimpleNamespace(
                id=f"tc{turns['n']}",
                function=types.SimpleNamespace(
                    name="read_file", arguments='{"file_path": "notes.txt"}'
                ),
            )
            return _msg(content=None, tool_calls=[call]), "stop", [], 0, (0, 0)

        (tmp_path / "notes.txt").write_text("note line\n" * 60, encoding="utf-8")
        monkeypatch.setattr(agent, "call_llm", fake_call_llm)

        report = ReportCollector()
        messages = [{"role": "system", "content": prompt.content}]
        messages.append({"role": "user", "content": "read notes.txt and summarise"})
        agent.run_agent_loop(
            messages,
            _tool_schemas(10),
            **_loop_kwargs(tmp_path, max_turns=8, report=report),
        )
        compactions = [e for e in report.events if e["type"] == "compaction"]
        assert turns["n"] >= 6
        assert len(compactions) <= 2

    def test_a_session_without_instructions_is_unaffected(self, tmp_path, monkeypatch):
        def fake_call_llm(*args, **_kw):
            return _msg(content="done"), "stop", [], 0, (0, 0)

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        report = ReportCollector()
        messages = [
            {"role": "system", "content": "short system prompt"},
            {"role": "user", "content": "hello"},
        ]
        answer, _ = agent.run_agent_loop(
            messages, None, **_loop_kwargs(tmp_path, report=report)
        )
        assert answer == "done"
        assert [e for e in report.events if e["type"] == "compaction"] == []


class TestLengthRecoveryOnALargeWindow:
    """The bigger prompt allowance makes output recovery load-bearing."""

    def test_text_only_length_truncation_continues(self, tmp_path, monkeypatch):
        calls = {"n": 0}

        def fake_call_llm(*args, **_kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return _msg(content="half an ans"), "length", [], 0, (0, 0)
            return _msg(content="the complete answer"), "stop", [], 0, (0, 0)

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        report = ReportCollector()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "word " * 84_000},
        ]
        answer, _ = agent.run_agent_loop(
            messages,
            None,
            **_loop_kwargs(
                tmp_path, max_turns=3, context_length=131_072, report=report
            ),
        )
        assert answer == "the complete answer"
        assert report.recovered_responses >= 1

    def _watch_dispatch(self, monkeypatch):
        seen = []
        real_dispatch = agent.dispatch

        def watching(name, parsed_args, *args, **kwargs):
            seen.append((name, parsed_args))
            return real_dispatch(name, parsed_args, *args, **kwargs)

        monkeypatch.setattr(agent, "dispatch", watching)
        return seen

    def test_a_repairable_tool_call_is_repaired_then_dispatched(
        self, tmp_path, monkeypatch
    ):
        (tmp_path / "notes.txt").write_text("hello\n", encoding="utf-8")
        calls = {"n": 0}

        def fake_call_llm(*args, **_kw):
            calls["n"] += 1
            if calls["n"] == 1:
                call = types.SimpleNamespace(
                    id="tc1",
                    function=types.SimpleNamespace(
                        name="read_file", arguments='{"file_path": "notes.txt'
                    ),
                )
                return _msg(content=None, tool_calls=[call]), "length", [], 0, (0, 0)
            return _msg(content="recovered"), "stop", [], 0, (0, 0)

        seen = self._watch_dispatch(monkeypatch)
        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        report = ReportCollector()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "read the notes"},
        ]
        answer, _ = agent.run_agent_loop(
            messages,
            _tool_schemas(2),
            **_loop_kwargs(
                tmp_path, max_turns=4, context_length=131_072, report=report
            ),
        )
        assert seen == [("read_file", {"file_path": "notes.txt"})]
        assert report.truncation_repairs >= 1
        assert answer == "recovered"

    def test_an_unrepairable_tool_call_is_never_dispatched(self, tmp_path, monkeypatch):
        calls = {"n": 0}

        def fake_call_llm(*args, **_kw):
            calls["n"] += 1
            if calls["n"] == 1:
                call = types.SimpleNamespace(
                    id="tc1",
                    function=types.SimpleNamespace(
                        name="read_file", arguments='{"file_pa'
                    ),
                )
                return _msg(content=None, tool_calls=[call]), "length", [], 0, (0, 0)
            return _msg(content="recovered"), "stop", [], 0, (0, 0)

        seen = self._watch_dispatch(monkeypatch)
        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "read the notes"},
        ]
        answer, _ = agent.run_agent_loop(
            messages,
            _tool_schemas(2),
            **_loop_kwargs(tmp_path, max_turns=4, context_length=131_072),
        )
        assert seen == []
        assert answer == "recovered"

    def test_repeated_length_truncation_ends(self, tmp_path, monkeypatch):
        calls = {"n": 0}

        def fake_call_llm(*args, **_kw):
            calls["n"] += 1
            return _msg(content="cut off again"), "length", [], 0, (0, 0)

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        answer, exhausted = agent.run_agent_loop(
            messages,
            None,
            **_loop_kwargs(tmp_path, max_turns=3, context_length=131_072),
        )
        assert calls["n"] <= 4
        assert exhausted is True
        assert answer is None or isinstance(answer, str)


class TestReactiveRetriesAreFinite:
    def test_a_provider_that_always_rejects_stops(self, tmp_path, monkeypatch):
        calls = {"n": 0}

        def fake_call_llm(*args, **_kw):
            calls["n"] += 1
            if calls["n"] > 40:
                raise AssertionError("retries did not stop")
            raise ContextOverflowError("too long")

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "word " * 20_000},
        ]
        try:
            agent.run_agent_loop(
                messages,
                None,
                **_loop_kwargs(tmp_path, max_turns=2, context_length=16_384),
            )
        except ContextOverflowError:
            pass
        assert calls["n"] < 40

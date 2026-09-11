"""Tests for the byte-counting fallback, where every size is advisory."""

import subprocess
import sys
import types

import pytest

from swival import agent, instructions as instr, tokens
from swival.report import ContextOverflowError
from swival.thinking import ThinkingState
from swival.todo import TodoState


@pytest.fixture
def offline(monkeypatch):
    """Force the byte-counting encoder, as an offline first run would get."""
    monkeypatch.setattr(tokens, "_encoder", tokens._FallbackEncoder())
    monkeypatch.setattr(tokens, "_is_fallback", True)
    monkeypatch.setattr(tokens, "_warned", False)
    return tokens


def _msg(content=None, tool_calls=None):
    return types.SimpleNamespace(
        content=content, tool_calls=tool_calls, role="assistant"
    )


def _tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                },
            },
        }
    ]


def _loop_kwargs(tmp_path, **overrides):
    defaults = dict(
        api_base="http://127.0.0.1:1234",
        model_id="test-model",
        max_turns=1,
        max_output_tokens=4096,
        temperature=0.5,
        top_p=None,
        seed=None,
        context_length=8192,
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


class TestEncoderReporting:
    def test_real_tokenizer_is_reported_as_exact(self):
        assert tokens.encoder_is_fallback() is False

    def test_fallback_is_reported(self, offline):
        assert tokens.encoder_is_fallback() is True

    def test_warning_fires_once(self, offline, capsys):
        assert tokens.warn_fallback_once() is True
        assert tokens.warn_fallback_once() is False
        assert "byte estimates" in capsys.readouterr().err

    def test_a_fresh_process_without_tokenizer_data_falls_back(self):
        code = (
            "import tiktoken\n"
            "def boom(*a, **k):\n"
            "    raise RuntimeError('no data')\n"
            "tiktoken.get_encoding = boom\n"
            "from swival.tokens import count_tokens, encoder_is_fallback\n"
            "assert count_tokens('hello') == 5\n"
            "assert encoder_is_fallback() is True\n"
            "print('ok')\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        )
        assert out.returncode == 0, out.stderr
        assert "ok" in out.stdout


class TestAwkwardText:
    @pytest.mark.parametrize(
        "text",
        [
            "Il était une fois un café très élégant à Genève.",
            "日本語のテキストをここに置きます。",
            "🚀🙂👨‍👩‍👧‍👦 emoji run",
            "<|endoftext|> <|im_start|> literal special tokens",
        ],
    )
    def test_counting_never_raises(self, text):
        assert tokens.count_tokens(text) > 0

    @pytest.mark.parametrize(
        "text",
        [
            "Il était une fois un café très élégant à Genève.",
            "日本語のテキストをここに置きます。",
            "🚀🙂👨‍👩‍👧‍👦 emoji run",
            "<|endoftext|> <|im_start|> literal special tokens",
        ],
    )
    def test_fallback_counting_never_raises(self, offline, text):
        assert tokens.count_tokens(text) == len(text.encode("utf-8"))

    def test_french_costs_far_more_in_bytes_than_in_tokens(self, monkeypatch):
        text = "Les caractères accentués coûtent cher en octets. " * 40
        real = tokens.count_tokens(text)
        monkeypatch.setattr(tokens, "_encoder", tokens._FallbackEncoder())
        monkeypatch.setattr(tokens, "_is_fallback", True)
        assert tokens.count_tokens(text) > real * 2


class TestAdmissionIsAdvisoryOffline:
    def test_complete_instructions_are_admitted(self, tmp_path, offline):
        body = "\n".join(f"- Règle {i} : lancer `make check`." for i in range(2000))
        (tmp_path / "CLAUDE.md").write_text(body, encoding="utf-8")
        result = agent.assemble_system_prompt(
            base_dir=str(tmp_path),
            system_prompt=None,
            no_system_prompt=False,
            no_instructions=False,
            no_memory=True,
            skills_catalog={},
            verbose=False,
            policy="autonomous",
            context_length=32_768,
            max_output_tokens=4096,
        )
        assert result.admission.admitted
        assert result.admission.advisory
        assert not result.admission.estimate_is_exact
        assert "Règle 1999" in result.content

    def test_the_report_says_the_estimate_is_a_byte_count(self, offline):
        admission = instr.admit(
            "x" * 100,
            context_length=8192,
            prompt_budget=7000,
            fixed_cost=0,
            sources=["/tmp/AGENTS.md"],
        )
        assert not admission.estimate_is_exact
        assert "byte estimates" in "\n".join(instr.breakdown_lines(admission))


class TestOutputClampingOffline:
    def test_clamping_is_off_for_byte_estimates(self, offline):
        messages = [{"role": "user", "content": "x" * 40_000}]
        assert agent.clamp_output_tokens(messages, None, 8192, 4096) == 4096

    def test_clamping_never_raises_locally(self, offline):
        messages = [{"role": "user", "content": "x" * 400_000}]
        assert agent.clamp_output_tokens(messages, None, 1024, 512) == 512

    def test_clamping_still_applies_with_a_real_tokenizer(self):
        messages = [{"role": "user", "content": "word " * 2000}]
        assert agent.clamp_output_tokens(messages, None, 4096, 4096) < 4096


class TestProactiveCompactionOffline:
    def _run(self, tmp_path, monkeypatch, **kwargs):
        seen = []

        def fake_call_llm(*args, **_kw):
            seen.append(len(args[2]))
            return _msg(content="done"), "stop", [], 0, (0, 0)

        compactions = []
        real = agent.compact_to_budget

        def counting(*args, **kw):
            compactions.append(kw.get("budget"))
            return real(*args, **kw)

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        monkeypatch.setattr(agent, "compact_to_budget", counting)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "é" * 30_000},
        ]
        agent.run_agent_loop(messages, None, **_loop_kwargs(tmp_path, **kwargs))
        return compactions

    def test_no_preventive_compaction_offline(self, tmp_path, monkeypatch, offline):
        assert self._run(tmp_path, monkeypatch) == []

    def test_preventive_compaction_still_runs_with_a_real_tokenizer(
        self, tmp_path, monkeypatch
    ):
        assert self._run(tmp_path, monkeypatch) != []

    def test_learned_window_does_not_re_enable_it(self, tmp_path, monkeypatch, offline):
        assert self._run(tmp_path, monkeypatch, context_length=None) == []


class TestReactiveRecoveryOffline:
    def _run(self, tmp_path, monkeypatch, failures, offline_mode, max_calls=40):
        # Each attempt records the prompt it sent and the answer room it asked
        # for, so a test can say which of the two recovery traded.
        attempts = []
        calls = {"n": 0}

        def fake_call_llm(*args, **_kw):
            calls["n"] += 1
            if calls["n"] > max_calls:
                raise AssertionError("recovery did not stop")
            attempts.append(
                {
                    "prompt": sum(len(m.get("content") or "") for m in args[2]),
                    "output": args[3],
                }
            )
            if calls["n"] <= failures:
                raise ContextOverflowError("provider says the prompt is too long")
            return _msg(content="done"), "stop", [], 0, (0, 0)

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "é" * 20_000},
            {"role": "assistant", "content": "ok " * 2000},
            {"role": "user", "content": "and now?"},
        ]
        kwargs = _loop_kwargs(tmp_path, max_turns=2, context_length=None)
        try:
            answer, _exhausted = agent.run_agent_loop(messages, None, **kwargs)
        except ContextOverflowError:
            answer = None
        return answer, attempts, calls["n"]

    def test_first_call_acceptance_needs_no_recovery(
        self, tmp_path, monkeypatch, offline
    ):
        answer, _attempts, calls = self._run(tmp_path, monkeypatch, 0, True)
        assert answer == "done"
        assert calls == 1

    def test_the_first_rejection_trades_answer_room_not_context(
        self, tmp_path, monkeypatch, offline
    ):
        answer, attempts, _calls = self._run(tmp_path, monkeypatch, 1, True)
        assert answer == "done"
        assert attempts[1]["prompt"] == attempts[0]["prompt"]
        assert attempts[1]["output"] < attempts[0]["output"]

    def test_the_prompt_shrinks_once_answer_room_runs_out(
        self, tmp_path, monkeypatch, offline
    ):
        # Refuse the opening call and every free round after it.
        failures = 1 + agent._MAX_OUTPUT_ONLY_ATTEMPTS
        answer, attempts, _calls = self._run(tmp_path, monkeypatch, failures, True)
        assert answer == "done"
        assert attempts[-1]["prompt"] < attempts[0]["prompt"]

    def test_repeated_rejection_stops_with_an_explicit_failure(
        self, tmp_path, monkeypatch, offline
    ):
        answer, _attempts, calls = self._run(tmp_path, monkeypatch, 999, True)
        assert calls < 40
        assert answer is None or "context" in answer.lower()


class TestOversizedOutputRequestOffline:
    """A window a prompt fits in, and an answer request that does not.

    The provider refuses this in the same words it uses for an oversized
    prompt. Shrinking the prompt cannot fix it, so recovery must try the
    answer allowance first, and must not destroy context on the way.
    """

    def _run(self, tmp_path, monkeypatch, *, window=8192, requested=32_768):
        attempts = []

        def provider(*args, **_kw):
            # The fixture text is one token per word, so counting words is what
            # this model's tokenizer would charge for it.
            prompt = sum(len((m.get("content") or "").split()) for m in args[2])
            output = args[3]
            attempts.append({"prompt": prompt, "output": output})
            # A real window: the prompt and the answer share it.
            if prompt + output > window:
                raise ContextOverflowError(
                    f"This model's maximum context length is {window} tokens"
                )
            return _msg(content="done"), "stop", [], 0, (0, 0)

        monkeypatch.setattr(agent, "call_llm", provider)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "word " * 1600},
            {"role": "user", "content": "Finally: cite the file you changed."},
        ]
        answer, _exhausted = agent.run_agent_loop(
            messages,
            None,
            **_loop_kwargs(
                tmp_path,
                max_turns=2,
                context_length=window,
                max_output_tokens=requested,
            ),
        )
        return answer, attempts, messages

    def test_recovery_keeps_the_prompt_and_shortens_the_answer(
        self, tmp_path, monkeypatch, offline
    ):
        answer, attempts, messages = self._run(tmp_path, monkeypatch)
        assert answer == "done"
        assert len(attempts) == 2
        assert attempts[1]["prompt"] == attempts[0]["prompt"]
        assert attempts[1]["output"] < attempts[0]["output"]
        assert "Finally: cite the file you changed." in messages[-2]["content"]

    def test_the_answer_still_gets_useful_room(self, tmp_path, monkeypatch, offline):
        _answer, attempts, _messages = self._run(tmp_path, monkeypatch)
        assert attempts[-1]["output"] > agent.MIN_OUTPUT_TOKENS * 8

    def test_a_real_tokenizer_never_needs_recovery(self, tmp_path, monkeypatch):
        answer, attempts, _messages = self._run(tmp_path, monkeypatch)
        assert answer == "done"
        assert len(attempts) == 1

    def test_a_request_that_already_fits_is_left_alone(
        self, tmp_path, monkeypatch, offline
    ):
        answer, attempts, _messages = self._run(tmp_path, monkeypatch, requested=4096)
        assert answer == "done"
        assert len(attempts) == 1
        assert attempts[0]["output"] == 4096


class TestOutputAllowanceSearchStep:
    """The bracket search, given what recovery knows at each point."""

    def test_the_opening_step_uses_the_reserve_policy(self):
        step = agent.OutputAllowanceSearch._step(None, 32_768, 32_768, 8192)
        assert step == agent.output_reserve(8192, 32_768)

    def test_an_unknown_window_halves_the_request(self):
        assert agent.OutputAllowanceSearch._step(None, 32_768, 32_768, None) == 16_384

    def test_a_second_refusal_halves_again(self):
        assert agent.OutputAllowanceSearch._step(None, 1024, 32_768, 8192) == 512

    def test_it_climbs_from_an_allowance_the_answer_outgrew(self):
        assert agent.OutputAllowanceSearch._step(1024, 32_768, 32_768, 8192) == 2048

    def test_it_bisects_once_climbing_would_overshoot(self):
        step = agent.OutputAllowanceSearch._step(1024, 2048, 32_768, 8192)
        assert 1024 < step < 2048

    def test_it_never_offers_an_allowance_the_window_refused(self):
        for ceiling in (2048, 1200, 1025):
            step = agent.OutputAllowanceSearch._step(1024, ceiling, 32_768, 8192)
            assert step is None or step < ceiling

    def test_a_closed_bracket_has_nothing_left(self):
        assert agent.OutputAllowanceSearch._step(1024, 1025, 32_768, 8192) is None

    def test_a_single_value_between_the_ends_is_still_offered(self):
        # 1024 was outgrown and 1026 was refused, so 1025 is all that is left.
        assert agent.OutputAllowanceSearch._step(1024, 1026, 32_768, 8192) == 1025

    def test_the_middle_never_lands_on_the_excluded_floor(self):
        for ceiling in range(1026, 1040):
            step = agent.OutputAllowanceSearch._step(1024, ceiling, 32_768, 8192)
            assert step is not None
            assert 1024 < step < ceiling

    def test_it_never_exceeds_what_the_user_asked_for(self):
        assert agent.OutputAllowanceSearch._step(2048, None, 4096, 8192) == 4096
        assert agent.OutputAllowanceSearch._step(4096, None, 4096, 8192) is None

    def test_no_request_means_no_step(self):
        assert agent.OutputAllowanceSearch._step(None, 32_768, None, 8192) is None

    def test_it_never_falls_below_the_minimum(self):
        step = agent.OutputAllowanceSearch._step(None, 64, 32_768, 8192)
        assert step is None or step >= agent.MIN_OUTPUT_TOKENS


class TestOutputBackoffKnowsWhenNotToApply:
    """Two overflows read the same on the wire and need opposite remedies."""

    def _run(self, tmp_path, monkeypatch, first_response, *, snapshot_state=None):
        attempts = []
        calls = {"n": 0}

        def provider(*args, **_kw):
            calls["n"] += 1
            attempts.append({"output": args[3]})
            if calls["n"] == 1:
                return first_response
            return _msg(content="done"), "stop", [], 0, (0, 0)

        monkeypatch.setattr(agent, "call_llm", provider)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "word " * 400},
        ]
        kwargs = _loop_kwargs(tmp_path, max_turns=3, context_length=None)
        if snapshot_state is not None:
            kwargs["snapshot_state"] = snapshot_state
        agent.run_agent_loop(messages, _tools(), **kwargs)
        return attempts

    def test_a_length_truncated_tool_call_keeps_its_answer_room(
        self, tmp_path, monkeypatch, offline
    ):
        """The answer was cut off, so taking room away would make it worse."""
        call = types.SimpleNamespace(
            id="tc1",
            function=types.SimpleNamespace(name="read_file", arguments="{bad"),
        )
        truncated = (_msg(content=None, tool_calls=[call]), "length", [], 0, (0, 0))
        attempts = self._run(tmp_path, monkeypatch, truncated)
        assert len(attempts) >= 2
        assert attempts[1]["output"] == attempts[0]["output"]

    def test_a_provider_rejection_does_trade_answer_room(
        self, tmp_path, monkeypatch, offline
    ):
        def _reject(*_a, **_k):
            raise ContextOverflowError("too long")

        attempts = []
        calls = {"n": 0}

        def provider(*args, **_kw):
            calls["n"] += 1
            attempts.append({"output": args[3]})
            if calls["n"] == 1:
                _reject()
            return _msg(content="done"), "stop", [], 0, (0, 0)

        monkeypatch.setattr(agent, "call_llm", provider)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "word " * 400},
        ]
        agent.run_agent_loop(
            messages, None, **_loop_kwargs(tmp_path, max_turns=3, context_length=None)
        )
        assert attempts[1]["output"] < attempts[0]["output"]

    def test_shortening_the_answer_leaves_a_checkpoint_valid(
        self, tmp_path, monkeypatch, offline
    ):
        """Nothing was removed, so an index checkpoint still describes history."""
        from swival.snapshot import SnapshotState

        state = SnapshotState()
        attempts = []
        calls = {"n": 0}

        def provider(*args, **_kw):
            calls["n"] += 1
            attempts.append(args[3])
            if calls["n"] == 1:
                raise ContextOverflowError("too long")
            return _msg(content="done"), "stop", [], 0, (0, 0)

        monkeypatch.setattr(agent, "call_llm", provider)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]
        state.save_at_index("before-overflow", len(messages))
        messages.append({"role": "user", "content": "trigger"})

        agent.run_agent_loop(
            messages,
            None,
            **_loop_kwargs(
                tmp_path, max_turns=3, context_length=None, snapshot_state=state
            ),
        )
        assert attempts[1] < attempts[0]
        assert state._generation == 0


class TestOverflowKindChangesMidRecovery:
    """What failed can change between retries, and the remedy changes with it."""

    def _truncated_tool_call(self):
        call = types.SimpleNamespace(
            id="tc1",
            function=types.SimpleNamespace(name="read_file", arguments="{bad"),
        )
        return _msg(content=None, tool_calls=[call]), "length", [], 0, (0, 0)

    def _run(self, tmp_path, monkeypatch, script, max_calls=20):
        """*script* maps a 1-based call number to "reject", "truncate" or "ok"."""
        attempts = []
        calls = {"n": 0}

        def provider(*args, **_kw):
            calls["n"] += 1
            if calls["n"] > max_calls:
                raise AssertionError("recovery did not stop")
            attempts.append(
                {
                    "output": args[3],
                    "prompt": sum(len(m.get("content") or "") for m in args[2]),
                }
            )
            action = script.get(calls["n"], "ok")
            if action == "reject":
                raise ContextOverflowError("maximum context length is 8192 tokens")
            if action == "truncate":
                return self._truncated_tool_call()
            return _msg(content="done"), "stop", [], 0, (0, 0)

        monkeypatch.setattr(agent, "call_llm", provider)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "word " * 800},
            {"role": "assistant", "content": "prior " * 800},
            {"role": "user", "content": "Finally: cite the file you changed."},
        ]
        agent.run_agent_loop(
            messages,
            _tools(),
            **_loop_kwargs(tmp_path, max_turns=2, context_length=None),
        )
        return attempts, messages

    def test_a_truncated_shortened_answer_stops_the_backoff(
        self, tmp_path, monkeypatch, offline
    ):
        # Reject the full request, then cut off the shortened answer.
        attempts, _messages = self._run(
            tmp_path, monkeypatch, {1: "reject", 2: "truncate"}
        )
        assert len(attempts) >= 3
        assert attempts[1]["output"] < attempts[0]["output"]
        # The third attempt must not ask for even less than the second.
        assert attempts[2]["output"] >= attempts[1]["output"]

    def test_it_asks_for_more_room_with_the_prompt_intact(
        self, tmp_path, monkeypatch, offline
    ):
        """The bracket has closed: too big, then too small. Search between."""
        attempts, _messages = self._run(
            tmp_path, monkeypatch, {1: "reject", 2: "truncate"}
        )
        assert attempts[2]["output"] > attempts[1]["output"]
        assert attempts[2]["prompt"] == attempts[1]["prompt"]

    def test_the_prompt_only_shrinks_once_the_search_is_done(
        self, tmp_path, monkeypatch, offline
    ):
        # Keep cutting the answer off through every free round.
        script = {1: "reject"}
        script.update(
            {n: "truncate" for n in range(2, 2 + agent._MAX_OUTPUT_ONLY_ATTEMPTS)}
        )
        attempts, _messages = self._run(tmp_path, monkeypatch, script)
        assert attempts[-1]["prompt"] < attempts[0]["prompt"]

    def test_the_allowance_never_goes_back_down(self, tmp_path, monkeypatch, offline):
        attempts, _messages = self._run(
            tmp_path, monkeypatch, {1: "reject", 2: "truncate", 3: "truncate"}
        )
        outputs = [a["output"] for a in attempts[1:]]
        assert outputs == sorted(outputs)

    def test_it_never_asks_for_more_than_the_user_did(
        self, tmp_path, monkeypatch, offline
    ):
        script = {1: "reject"}
        script.update(
            {n: "truncate" for n in range(2, 2 + agent._MAX_OUTPUT_ONLY_ATTEMPTS)}
        )
        attempts, _messages = self._run(tmp_path, monkeypatch, script)
        assert max(a["output"] for a in attempts) <= attempts[0]["output"]

    def test_it_never_asks_for_an_allowance_the_window_refused(
        self, tmp_path, monkeypatch, offline
    ):
        # Every allowance after the first rejection stays under it.
        attempts, _messages = self._run(
            tmp_path, monkeypatch, {1: "reject", 2: "truncate", 3: "truncate"}
        )
        rejected = attempts[0]["output"]
        assert all(a["output"] < rejected for a in attempts[1:])

    def test_the_other_order_still_trades_answer_room(
        self, tmp_path, monkeypatch, offline
    ):
        # A truncation first arms no backoff; a later plain rejection is a
        # different failure, but recovery is already committed to the prompt.
        attempts, _messages = self._run(
            tmp_path, monkeypatch, {1: "truncate", 2: "reject"}
        )
        assert attempts[1]["output"] == attempts[0]["output"]
        assert attempts[-1]["prompt"] < attempts[0]["prompt"]


class TestAnswerThatNeedsRoomInASmallWindow:
    """The reported case: the window fits the prompt, but not the request.

    A model that needs 2,048 tokens to finish its tool call, an 8K window, and
    the default 32,768-token request. Recovery has to find the room between the
    allowance the window refuses and the one the answer outgrows, without
    spending the prompt to get there.
    """

    def _run(
        self,
        tmp_path,
        monkeypatch,
        *,
        window=8192,
        needs=2048,
        requested=32_768,
        words=1500,
    ):
        attempts = []

        def provider(*args, **_kw):
            # The fixture text is one token per word.
            prompt = sum(len((m.get("content") or "").split()) for m in args[2])
            output = args[3]
            attempts.append({"prompt": prompt, "output": output})
            if prompt + output > window:
                raise ContextOverflowError(
                    f"This model's maximum context length is {window} tokens"
                )
            if output < needs:
                call = types.SimpleNamespace(
                    id="tc1",
                    function=types.SimpleNamespace(name="read_file", arguments="{bad"),
                )
                return _msg(content=None, tool_calls=[call]), "length", [], 0, (0, 0)
            return _msg(content="done"), "stop", [], 0, (0, 0)

        monkeypatch.setattr(agent, "call_llm", provider)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "word " * words},
            {"role": "user", "content": "Finally: cite the file you changed."},
        ]
        answer, exhausted = agent.run_agent_loop(
            messages,
            _tools(),
            **_loop_kwargs(
                tmp_path,
                max_turns=3,
                context_length=window,
                max_output_tokens=requested,
            ),
        )
        return answer, exhausted, attempts, messages

    def test_it_finds_the_room_and_completes(self, tmp_path, monkeypatch, offline):
        answer, exhausted, attempts, _messages = self._run(tmp_path, monkeypatch)
        assert answer == "done"
        assert exhausted is False
        assert attempts[-1]["output"] >= 2048

    def test_it_never_falls_back_to_the_minimal_prompt(
        self, tmp_path, monkeypatch, offline
    ):
        _answer, _exhausted, attempts, _messages = self._run(tmp_path, monkeypatch)
        assert all(a["output"] > agent.MIN_OUTPUT_TOKENS for a in attempts)

    def test_the_prompt_survives_intact(self, tmp_path, monkeypatch, offline):
        _answer, _exhausted, attempts, messages = self._run(tmp_path, monkeypatch)
        assert {a["prompt"] for a in attempts} == {attempts[0]["prompt"]}
        assert "Finally: cite the file you changed." in messages[-2]["content"]

    def test_it_gets_there_in_a_handful_of_calls(self, tmp_path, monkeypatch, offline):
        _answer, _exhausted, attempts, _messages = self._run(tmp_path, monkeypatch)
        assert len(attempts) <= 4

    def test_a_request_sized_for_the_window_needs_no_recovery(
        self, tmp_path, monkeypatch, offline
    ):
        answer, _exhausted, attempts, _messages = self._run(
            tmp_path, monkeypatch, requested=4096
        )
        assert answer == "done"
        assert len(attempts) == 1

    def test_a_long_prompt_leaves_a_narrow_bracket(
        self, tmp_path, monkeypatch, offline
    ):
        """Little room to spare: the search has to land inside a tight range.

        A 6,400-token prompt in an 8K window leaves under 1,800 tokens for an
        answer that needs 1,500. Climbing overshoots on the first try, so the
        search has to come back down between the two ends rather than carry on
        upward or give up and spend the prompt.
        """
        answer, exhausted, attempts, messages = self._run(
            tmp_path, monkeypatch, words=6400, needs=1500
        )
        assert answer == "done"
        assert exhausted is False
        assert attempts[-1]["output"] >= 1500
        assert {a["prompt"] for a in attempts} == {attempts[0]["prompt"]}
        assert "Finally: cite the file you changed." in messages[-2]["content"]

    def test_a_large_answer_in_a_small_window_still_completes(
        self, tmp_path, monkeypatch, offline
    ):
        """The search has to climb a long way before it overshoots.

        An answer needing 6,000 tokens in an 8K window: every early allowance
        is too small, and the first one that is big enough is too big for the
        window. The workable value is between, and getting there must not cost
        the prompt.
        """
        answer, exhausted, attempts, messages = self._run(
            tmp_path, monkeypatch, needs=6000
        )
        assert answer == "done"
        assert exhausted is False
        assert attempts[-1]["output"] >= 6000
        assert {a["prompt"] for a in attempts} == {attempts[0]["prompt"]}
        assert "Finally: cite the file you changed." in messages[-2]["content"]

    def test_an_exhausted_search_never_carries_a_refused_allowance(
        self, tmp_path, monkeypatch, offline
    ):
        """Free rounds run out; the allowance still improves with the evidence."""
        answer, _exhausted, attempts, _messages = self._run(
            tmp_path, monkeypatch, needs=6000
        )
        refused = [a["output"] for a in attempts if a["prompt"] + a["output"] > 8192]
        assert refused, "the scenario must include a refusal to be meaningful"
        assert attempts[-1]["output"] < min(refused)
        assert answer == "done"

    def test_a_refusal_does_not_outlive_the_prompt_that_caused_it(
        self, tmp_path, monkeypatch, offline
    ):
        """A prompt filling most of the window, and an answer that needs room.

        Nothing large enough fits beside the original prompt, so the search
        exhausts its rounds and compaction starts. What the window refused then
        was that allowance next to that prompt. Once the prompt is smaller the
        refusal describes nothing, and recovery has to be willing to ask for
        more again.
        """
        answer, exhausted, attempts, _messages = self._run(
            tmp_path, monkeypatch, words=6400, needs=2000
        )
        assert answer == "done"
        assert exhausted is False
        assert attempts[-1]["output"] >= 2000

    def test_it_climbs_back_above_the_stale_ceiling(
        self, tmp_path, monkeypatch, offline
    ):
        _answer, _exhausted, attempts, _messages = self._run(
            tmp_path, monkeypatch, words=6400, needs=2000
        )
        refused = [a["output"] for a in attempts if a["prompt"] + a["output"] > 8192]
        assert refused, "the scenario must include a refusal to be meaningful"
        # The winning allowance is one an earlier, larger prompt could not take.
        assert attempts[-1]["output"] >= min(refused)

    def test_the_floor_survives_a_smaller_prompt(self, tmp_path, monkeypatch, offline):
        """An answer that outgrew an allowance still outgrows it."""
        _answer, _exhausted, attempts, _messages = self._run(
            tmp_path, monkeypatch, words=6400, needs=2000
        )
        outgrew = [a["output"] for a in attempts if a["output"] < 2000]
        assert outgrew
        assert attempts[-1]["output"] > max(outgrew)

    def test_an_answer_needing_most_of_the_window_still_completes(
        self, tmp_path, monkeypatch, offline
    ):
        """Nothing fits beside the original prompt, so both halves must work.

        A 6,400-token prompt and an answer needing 7,000 leave no allowance
        that works until the prompt is much smaller. Recovery has to compact
        and keep raising the allowance as it does, without ever asking for more
        than the window holds.
        """
        answer, exhausted, attempts, _messages = self._run(
            tmp_path, monkeypatch, words=6400, needs=7000
        )
        assert answer == "done"
        assert exhausted is False
        assert attempts[-1]["output"] >= 7000

    def test_it_never_asks_for_more_than_the_window_holds(
        self, tmp_path, monkeypatch, offline
    ):
        _answer, _exhausted, attempts, _messages = self._run(
            tmp_path, monkeypatch, words=6400, needs=7000
        )
        assert all(a["output"] <= 8192 for a in attempts)

    def test_no_refused_pair_is_ever_tried_twice(self, tmp_path, monkeypatch, offline):
        _answer, _exhausted, attempts, _messages = self._run(
            tmp_path, monkeypatch, words=6400, needs=7000
        )
        seen = {}
        for a in attempts:
            refused = a["prompt"] + a["output"] > 8192
            key = (a["prompt"], a["output"])
            assert not (refused and seen.get(key)), f"retried a refused pair: {key}"
            seen[key] = refused

    def test_a_clamped_opening_request_is_what_the_ceiling_records(
        self, tmp_path, monkeypatch, offline
    ):
        """The provider judged the number it received, not the one we wanted.

        The opening request is cut to the window before it goes out. Recording
        the original instead would leave the search believing everything below
        32,768 was untried, and offer the window size again.
        """
        _answer, _exhausted, attempts, _messages = self._run(
            tmp_path, monkeypatch, words=1500, needs=2048
        )
        assert attempts[0]["output"] == 8192
        assert [a["output"] for a in attempts[1:]].count(8192) == 0

    def test_that_search_never_retries_a_refused_allowance(
        self, tmp_path, monkeypatch, offline
    ):
        _answer, _exhausted, attempts, _messages = self._run(
            tmp_path, monkeypatch, words=6400, needs=1500
        )
        refused = [a["output"] for a in attempts if a["prompt"] + a["output"] > 8192]
        for ceiling in refused:
            later = attempts[[a["output"] for a in attempts].index(ceiling) + 1 :]
            assert all(a["output"] < ceiling for a in later)


class TestOfflineClampKeepsTheWindowBound:
    """The one clamp that needs no estimate of the prompt."""

    def test_a_request_larger_than_the_window_is_cut_to_it(self, offline):
        messages = [{"role": "user", "content": "x" * 40_000}]
        assert agent.clamp_output_tokens(messages, None, 8192, 32_768) == 8192

    def test_a_request_inside_the_window_is_left_alone(self, offline):
        messages = [{"role": "user", "content": "x" * 40_000}]
        assert agent.clamp_output_tokens(messages, None, 8192, 4096) == 4096

    def test_it_still_never_subtracts_the_prompt(self, offline):
        short = [{"role": "user", "content": "hi"}]
        long = [{"role": "user", "content": "x" * 400_000}]
        assert agent.clamp_output_tokens(
            short, None, 8192, 4096
        ) == agent.clamp_output_tokens(long, None, 8192, 4096)

    def test_an_unknown_window_leaves_the_request_alone(self, offline):
        messages = [{"role": "user", "content": "x" * 40_000}]
        assert agent.clamp_output_tokens(messages, None, None, 32_768) == 32_768


class TestOutputAllowanceSearchTransitions:
    """The bound moves, which is where every bug in this search lived.

    These were only reachable through an end-to-end fake provider before the
    search became an object.
    """

    def _search(self, *, requested=32_768, window=8192, enabled=True, attempts=6):
        return agent.OutputAllowanceSearch(
            requested=requested, window=window, enabled=enabled, attempts=attempts
        )

    def test_a_disabled_search_never_offers_anything(self):
        search = self._search(enabled=False)
        search.opened_with(32_768, truncated=False)
        assert search.current is None
        assert search.take_free_round() is False
        assert search.request == 32_768

    def test_opening_on_a_refusal_sets_the_ceiling(self):
        search = self._search()
        search.opened_with(8192, truncated=False)
        assert search.ceiling == 8192
        assert search.floor is None
        # The clamp had already cut the request to the window, so the opening
        # step halves what was refused rather than starting from the reserve.
        assert search.current == 4096

    def test_an_unclamped_refusal_opens_at_the_reserve(self):
        """Nothing stepped down yet, so the reservation policy sets the pace."""
        search = self._search(requested=4096)
        search.opened_with(4096, truncated=False)
        assert search.current == agent.output_reserve(8192, 4096)

    def test_opening_on_a_truncation_sets_the_floor(self):
        search = self._search()
        search.opened_with(32_768, truncated=True)
        assert search.floor == 32_768
        # Nothing above what the user asked for, so there is nowhere to go.
        assert search.current is None

    def test_a_truncation_raises_the_floor_and_climbs(self):
        search = self._search()
        search.opened_with(8192, truncated=False)
        search.record_truncation(1024)
        assert search.floor == 1024
        assert search.current == 2048

    def test_a_refusal_lowers_the_ceiling_and_bisects(self):
        search = self._search()
        search.opened_with(8192, truncated=False)
        search.record_truncation(1024)
        search.record_refusal(2048)
        assert search.ceiling == 2048
        assert 1024 < search.current < 2048

    def test_a_refusal_never_raises_the_ceiling(self):
        search = self._search()
        search.opened_with(8192, truncated=False)
        search.record_refusal(2048)
        search.record_refusal(4096)
        assert search.ceiling == 2048

    def test_a_truncation_never_lowers_the_floor(self):
        search = self._search()
        search.opened_with(8192, truncated=False)
        search.record_truncation(2048)
        search.record_truncation(1024)
        assert search.floor == 2048

    def test_a_smaller_payload_drops_the_ceiling_but_keeps_the_floor(self):
        search = self._search()
        search.opened_with(8192, truncated=False)
        search.record_truncation(1024)
        search.record_refusal(2048)
        search.payload_changed()
        assert search.ceiling is None
        assert search.floor == 1024
        assert search.current == 2048

    def test_a_smaller_payload_spends_the_room_on_this_round(self):
        search = self._search()
        search.opened_with(8192, truncated=False)
        search.record_truncation(1024)
        search.record_refusal(2048)
        search.take_free_round()
        before = search.request
        search.payload_changed()
        assert search.request > before

    def test_the_window_survives_a_payload_change(self):
        search = self._search()
        search.opened_with(8192, truncated=False)
        search.record_truncation(6144)
        search.payload_changed()
        assert search.current is not None
        assert search.current <= 8192

    def test_free_rounds_run_out_but_the_search_does_not(self):
        search = self._search(attempts=1)
        search.opened_with(8192, truncated=False)
        assert search.take_free_round() is True
        assert search.take_free_round() is False
        # Still applied, so a compacting round carries the best known value.
        search.record_truncation(search.request)
        assert search.current is not None
        search.take_free_round()
        assert search.request == search.current

    def test_a_closed_bracket_stops_the_search(self):
        search = self._search()
        search.opened_with(8192, truncated=False)
        search.record_truncation(1024)
        search.record_refusal(1025)
        assert search.current is None
        assert search.take_free_round() is False

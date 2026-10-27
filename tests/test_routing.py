"""Tests for the pure parts of model routing and choose_profile()."""

import json

import pytest

from swival import decisions, routing
from swival._msg import COMPACTION_MARKER, RECAP_MARKER
from swival.routing import (
    Facts,
    RoutingState,
    Unusable,
    apply_rule,
    build_routing_state,
    choose_profile,
    classify_current,
    routing_candidates,
)
from swival.secrets import SecretShield
from tests.test_decisions import _filter, _response

CRITERIA = {
    "localfast": "Small local model. Trivial edits and short questions.",
    "gpt": "Strong general coding model for typical feature work.",
    "max": "Maximum reasoning, slow and expensive, for very hard problems.",
}

PROFILES = {
    "localfast": {"provider": "lmstudio", "description": CRITERIA["localfast"]},
    "gpt": {"provider": "chatgpt", "model": "gpt", "description": CRITERIA["gpt"]},
    "max": {"provider": "chatgpt", "model": "max", "description": CRITERIA["max"]},
    "plain": {"provider": "generic", "model": "x", "base_url": "http://h"},
}


def _user(text, **extra):
    return {"role": "user", "content": text, **extra}


def _assistant(text=None, tool_calls=None):
    msg = {"role": "assistant", "content": text}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _call(name, args, cid="c1"):
    return {
        "id": cid,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _tool(content, cid="c1", **extra):
    return {"role": "tool", "tool_call_id": cid, "content": content, **extra}


def _state(messages, request="go", **kwargs):
    return build_routing_state(messages, request, CRITERIA, **kwargs)


class TestCutting:
    def test_short_request_is_kept_whole(self):
        state = _state([{"role": "system", "content": "s"}], "fix the typo")
        assert state == {"request": "fix the typo"}

    def test_long_request_keeps_head_and_tail_with_size(self):
        text = "H" * 1000 + "M" * 5000 + "T" * 1500
        state = _state([], text)
        cut = state["request"]
        assert cut.startswith("H" * 800 + "\n[... ")
        assert cut.endswith("]\n" + "T" * 1200)
        assert "5,500 characters omitted" in cut
        assert state["request_size"] == "7,500 characters, about 1,875 tokens"

    def test_previous_turn_keeps_more_of_the_reply_end(self):
        user = "u" * 160 + "x" * 1000 + "END" + "v" * 237
        reply = "a" * 400 + "b" * 2000 + "PROPOSAL" + "c" * 792
        previous = _state([_user(user), _assistant(reply)])["previous_turn"]
        assert previous["user"].startswith("u" * 160 + "\n[... ")
        assert previous["user"].endswith("END" + "v" * 237)
        assert previous["assistant"].startswith("a" * 400 + "\n[... ")
        assert previous["assistant"].endswith("PROPOSAL" + "c" * 792)


class TestPreviousTurn:
    def test_skips_synthetic_messages_and_recaps(self):
        messages = [
            {"role": "system", "content": "sys"},
            _user("audit the HLS decryption"),
            _assistant("Found an IV reuse bug. Want me to fix it?"),
            _user("nudge", _swival_synthetic=True),
            _user("[image] Describe and analyze the attached image(s)."),
            _user(COMPACTION_MARKER),
            _user(RECAP_MARKER + " summary]"),
            _user("[REVIEWER FEEDBACK, Round 1]\nfix it"),
        ]
        state = _state(messages, "yes")
        assert state["previous_turn"]["user"] == "audit the HLS decryption"
        assert state["previous_turn"]["assistant"].startswith("Found an IV")

    def test_goal_is_cut(self):
        state = _state([], "still failing", goal="g" * 500)
        assert state["goal"] == "g" * 300

    def test_recap_without_request_still_gives_context(self):
        # /save before the first task, then /restore: only the recap is left.
        messages = [
            {"role": "system", "content": "s"},
            _assistant("[snapshot: audit]\nfound a timing leak, proposed hmac.Equal"),
        ]
        state = _state(messages, "yes, fix it")
        assert state["previous_turn"] == {
            "assistant": "[snapshot: audit]\nfound a timing leak, proposed hmac.Equal"
        }


class TestEarlierTurns:
    def test_at_most_two_oldest_first_with_user_head_and_assistant_tail(self):
        messages = []
        for i in range(4):
            messages += [_user(f"q{i} " + "u" * 300), _assistant("r" * 400 + f"A{i}")]
        state = _state(messages)
        earlier = state["earlier_turns"]
        assert len(earlier) == 2
        assert earlier[0]["user"].startswith("q1 ")
        assert earlier[1]["user"].startswith("q2 ")
        assert len(earlier[0]["user"]) == 200
        assert earlier[0]["assistant"].startswith("[...] ")
        assert earlier[0]["assistant"].endswith("A1")
        assert len(earlier[0]["assistant"]) == len("[...] ") + 300
        assert state["previous_turn"]["user"].startswith("q3 ")

    def test_short_reply_has_no_marker(self):
        messages = [_user("a"), _assistant("short"), _user("b"), _assistant("x")]
        assert _state(messages)["earlier_turns"] == [
            {"user": "a", "assistant": "short"}
        ]

    def test_recap_before_requests_becomes_an_earlier_turn(self):
        messages = [
            _assistant(RECAP_MARKER + ", summary]\nearlier work on HLS"),
            _user("now the tests"),
            _assistant("they pass"),
        ]
        state = _state(messages)
        assert state["earlier_turns"] == [
            {"assistant": RECAP_MARKER + ", summary]\nearlier work on HLS"}
        ]
        assert state["previous_turn"]["user"] == "now the tests"


class TestTools:
    def test_main_argument_dedup_limit_and_failures(self):
        calls = [_call("read_file", {"file_path": "a.go"}, "c0")]
        calls += [_call("read_file", {"file_path": "a.go"}, "c1")]
        calls += [
            _call("run_command", {"command": ["go", "test", "./..."]}, "c2"),
            _call("grep", {"pattern": "refreshToken", "path": "internal"}, "c3"),
            _call("fetch_url", {"url": "https://example.com/" + "x" * 200}, "c4"),
        ]
        calls += [
            _call("read_file", {"file_path": f"f{i}.go"}, f"d{i}") for i in range(8)
        ]
        messages = [
            _user("why are the tests failing?"),
            _assistant(tool_calls=calls),
            _tool("error: no such file", "c0"),
            _tool("Exit code: 1\nFAIL", "c2", _swival_exit_code=1),
            _tool("ok", "c3", _swival_exit_code=0),
            _assistant("Three tests fail."),
        ]
        prev = _state(messages)["previous_turn"]
        assert prev["failed_tool_calls"] == 2
        tools = prev["tools"]
        assert len(tools) == 8
        assert tools[-1] == "read_file f7.go"
        assert all(len(t) <= 100 for t in tools)

        early = routing._turns(messages, lambda text: text)[0].tools
        assert early[:3] == [
            "read_file a.go",
            "run_command go test ./...",
            "grep refreshToken",
        ]
        assert early[3].startswith("fetch_url https://example.com/x")
        assert len(early[3]) == 100

    def test_files_list_and_namespace_calls(self):
        import types

        ns_call = types.SimpleNamespace(
            id="n1",
            function=types.SimpleNamespace(
                name="read_multiple_files",
                arguments=json.dumps(
                    {"files": [{"file_path": "a"}, {"file_path": "b"}]}
                ),
            ),
        )
        messages = [
            _user("read both"),
            types.SimpleNamespace(role="assistant", content="", tool_calls=[ns_call]),
        ]
        assert _state(messages)["previous_turn"]["tools"] == ["read_multiple_files a b"]


class TestImages:
    def test_counts_new_and_earlier_images(self):
        image = {"type": "image_url", "image_url": {"url": "data:x"}}
        messages = [
            _user("look"),
            {
                "role": "user",
                "content": [{"type": "text", "text": "[image] x"}, image, image],
            },
            _assistant("I see a chart."),
        ]
        request = [{"type": "text", "text": "and this one?"}, image]
        state = build_routing_state(messages, request, CRITERIA)
        assert state["request"] == "and this one?"
        assert state["images_attached"] == 1
        assert state["images_earlier"] == 2
        assert state["previous_turn"]["user"] == "look"


class TestBudget:
    def _big_history(self):
        messages = []
        for i in range(3):
            calls = [
                _call("read_file", {"file_path": f"p{i}{j}" * 10}, f"c{i}{j}")
                for j in range(8)
            ]
            messages += [
                _user(f"question {i} " + "é" * 600),
                _assistant(tool_calls=calls),
                _assistant("ü" * 1500),
            ]
        return messages

    def test_counts_bytes_not_characters(self):
        state = _state([], "é" * 1500)
        units = decisions.envelope_units(
            {"state": state, "questions": routing.routing_questions(CRITERIA)}
        )
        assert units > len(json.dumps(state, ensure_ascii=False)) + 1500

    def test_shrinks_in_order(self, monkeypatch):
        messages = self._big_history()
        full = _state(messages, "r" * 3000, budget=10**9)
        assert "tools" in full["previous_turn"]
        sizes = []
        for level in range(7):
            state = routing._render(
                "r" * 3000,
                full["request_size"],
                {},
                routing._turns(messages, lambda text: text),
                level,
            )
            sizes.append(
                decisions.envelope_units(
                    {"state": state, "questions": routing.routing_questions(CRITERIA)}
                )
            )
        assert sizes == sorted(sizes, reverse=True)

        def at(level):
            budget = sizes[level]
            return _state(messages, "r" * 3000, budget=budget)

        assert "tools" not in at(1)["previous_turn"]
        assert "user" in at(1)["previous_turn"]
        assert "user" not in at(2)["previous_turn"]
        assert all("assistant" not in e for e in at(3)["earlier_turns"])
        assert "earlier_turns" not in at(4)
        assert len(at(5)["previous_turn"]["assistant"]) < 700
        assert len(at(4)["request"]) > 2000
        assert len(at(6)["request"]) < 1100

    def test_does_not_fit(self):
        assert _state([], "x", budget=10) is None
        huge = {"a": "d" * 7000, "b": "e"}
        assert build_routing_state([], "x", huge) is None

    def test_request_cut_by_the_last_shrink_step_has_its_size(self):
        criteria = {"a": "x" * 2200, "b": "y" * 2200}
        state = build_routing_state([], "r" * 1500, criteria)
        assert "omitted" in state["request"]
        assert state["request_size"] == "1,500 characters, about 375 tokens"


def _facts(**kwargs):
    return {name: Facts(**value) for name, value in kwargs.items()}


class TestCandidates:
    def test_admission_ignores_descriptions(self):
        assert routing_candidates(
            PROFILES, {}, allowlist=["plain", "gpt"], require_description=False
        ) == ["gpt", "plain"]

    def test_allowlist_and_command_and_unavailable(self):
        profiles = dict(PROFILES, cmd={"provider": "command", "description": "x"})
        assert routing_candidates(profiles, {}) == ["localfast", "gpt", "max"]
        assert routing_candidates(profiles, {}, allowlist=["max", "localfast"]) == [
            "localfast",
            "max",
        ]
        assert routing_candidates(profiles, {}, unavailable={"gpt"}) == [
            "localfast",
            "max",
        ]

    def test_vision_only_matters_with_images(self):
        facts = _facts(localfast={"vision": False})
        assert "localfast" in routing_candidates(PROFILES, facts)
        assert "localfast" not in routing_candidates(PROFILES, facts, has_images=True)

    def test_tools_and_context(self):
        facts = _facts(localfast={"tools": False}, gpt={"budget": 1000})
        assert routing_candidates(PROFILES, facts, needed_tokens=999) == ["gpt", "max"]
        assert routing_candidates(PROFILES, facts, needed_tokens=1001) == ["max"]

    def test_unknown_counts_as_capable(self):
        facts = _facts(max={})
        assert routing_candidates(
            PROFILES, facts, has_images=True, needed_tokens=10**9
        ) == ["localfast", "gpt", "max"]


class TestClassify:
    def test_candidate(self):
        assert classify_current("gpt", Facts(), ["gpt", "max"]) == ("candidate", None)

    def test_valid_without_description_or_after_model_override(self):
        assert classify_current("plain", Facts(), ["gpt"]) == ("valid", None)
        # /model then /route on: the runtime is no longer the profile.
        assert classify_current(None, Facts(), ["gpt"]) == ("valid", None)

    def test_ineligible_reasons(self):
        assert classify_current("plain", Facts(), [], allowlist=["gpt"]) == (
            "ineligible",
            "unlisted",
        )
        assert classify_current(None, Facts(), [], allowlist=["gpt"]) == (
            "ineligible",
            "unlisted",
        )
        assert classify_current(
            "plain", Facts(), [], allowlist=["gpt"], explicit=True
        ) == ("valid", None)
        assert classify_current("gpt", Facts(), [], unavailable={"gpt"}) == (
            "ineligible",
            "unavailable",
        )
        assert classify_current("gpt", Facts(vision=False), [], has_images=True) == (
            "ineligible",
            "vision",
        )
        assert classify_current("gpt", Facts(budget=10), [], needed_tokens=11) == (
            "ineligible",
            "context",
        )


class TestRule:
    P = {"localfast": 0.1, "gpt": 0.3, "max": 0.6}

    def test_candidate_alone_stays(self):
        assert apply_rule(["gpt"], "gpt", "candidate") == routing.Verdict(
            None, "only candidate"
        )

    def test_valid_with_none_or_one_stays(self):
        assert apply_rule([], None, "valid").target is None
        assert apply_rule(["gpt"], "plain", "valid").target is None

    def test_ineligible_none_depends_on_reason(self):
        assert apply_rule([], "x", "ineligible", reason="vision").target is None
        retry = apply_rule([], "x", "ineligible", reason="unavailable")
        assert retry == routing.Verdict(None, "retrying the unavailable profile")
        unlisted = apply_rule([], "x", "ineligible", reason="unlisted")
        assert unlisted == routing.Verdict(None, "not in routing_profiles")

    def test_ineligible_one_moves(self):
        assert apply_rule(["gpt"], "x", "ineligible", reason="unlisted").target == "gpt"

    def test_rounded_margin_on_the_boundary_switches(self):
        names = ["gpt", "max"]
        assert (
            apply_rule(names, "gpt", "candidate", {"gpt": 0.35, "max": 0.65}).target
            == "max"
        )
        assert (
            apply_rule(names, "gpt", "candidate", {"gpt": 0.36, "max": 0.64}).target
            is None
        )

    def test_valid_compares_with_zero(self):
        names = ["localfast", "gpt", "max"]
        assert apply_rule(names, "plain", "valid", self.P).target == "max"
        low = {"localfast": 0.25, "gpt": 0.25, "max": 0.29}
        assert apply_rule(names, "plain", "valid", low).target is None

    def test_ineligible_moves_whatever_the_margin(self):
        names = ["localfast", "gpt", "max"]
        p = {"localfast": 0.3, "gpt": 0.31, "max": 0.39}
        assert apply_rule(names, "x", "ineligible", p, reason="context").target == "max"

    def test_zero_margin_still_needs_a_better_profile(self):
        names = ["localfast", "gpt", "max"]
        p = {"localfast": 0.3, "gpt": 0.31, "max": 0.39}
        assert apply_rule(names, "gpt", "candidate", p, margin=0).target == "max"
        left = {"localfast": 0.0, "gpt": 0.0, "max": 1.0}
        assert apply_rule(names[:2], "plain", "valid", left, margin=0).target is None

    def test_failed_request(self):
        names = ["localfast", "gpt", "max"]
        assert apply_rule(names, "gpt", "candidate", failure="boom").reason == "boom"
        assert apply_rule(names, "gpt", "candidate").target is None
        assert apply_rule(names, None, "valid").target is None
        assert (
            apply_rule(names, "x", "ineligible", reason="unlisted").target
            == "localfast"
        )

    def test_ties_go_to_current_then_config_order(self):
        names = ["localfast", "gpt", "max"]
        p = {"localfast": 0.45, "gpt": 0.1, "max": 0.45}
        assert apply_rule(names, "max", "candidate", p).target is None
        assert apply_rule(names, "plain", "valid", p).target == "localfast"

    def test_no_renormalization_after_removal(self):
        p = {"localfast": 0.05, "gpt": 0.25, "max": 0.7}
        # Without max, gpt has 0.25: not enough margin over 0 for a valid runtime.
        assert apply_rule(["localfast", "gpt"], "plain", "valid", p).target is None


def _routing(profiles=PROFILES):
    settings = decisions.resolve_settings("clef", "http://127.0.0.1:8000/v1")
    return RoutingState(profiles=profiles, settings=settings, enabled=True)


def _fake_systemone(monkeypatch):
    """Patch litellm.systemone with a server that records what it receives."""
    import litellm

    sent = []

    def systemone(**kwargs):
        sent.append(kwargs)
        first, second = kwargs["questions"]["profile"]["criteria"]
        answer = {
            "type": "choice",
            "choice": second,
            "confidence": 0.9,
            "probabilities": {first: 0.1, second: 0.9},
        }
        return _response({"profile": answer})

    monkeypatch.setattr(litellm, "systemone", systemone)
    return sent


def _answer(probabilities, choice=None):
    choice = choice or max(probabilities, key=probabilities.get)
    return decisions.DecisionResult(
        model="clef",
        answers={
            "profile": {
                "type": "choice",
                "choice": choice,
                "confidence": probabilities[choice],
                "probabilities": probabilities,
            }
        },
        usage={"input_tokens": 400, "output_tokens": 0},
    )


class TestChooseProfile:
    def _choose(
        self,
        state,
        send=None,
        *,
        messages=None,
        request="go",
        current="gpt",
        cstate="candidate",
        resolve=None,
        **kw,
    ):
        warnings = []
        decision = choose_profile(
            state,
            [_user("q"), _assistant("a")] if messages is None else messages,
            request,
            candidates=kw.pop("candidates", ["localfast", "gpt", "max"]),
            current=current,
            state=cstate,
            resolve=resolve or (lambda name: {"profile": name}),
            warn=warnings.append,
            send=send,
            **kw,
        )
        return decision, warnings

    def test_probabilities_are_authoritative(self):
        sent = []

        def send(settings, envelope, **kwargs):
            sent.append(envelope)
            # The choice says gpt, but the probabilities say max.
            return _answer({"localfast": 0.05, "gpt": 0.05, "max": 0.9}, choice="gpt")

        decision, _ = self._choose(_routing(), send)
        assert decision.switched and decision.profile == "max"
        assert decision.runtime == {"profile": "max"}
        assert len(sent) == 1 and decision.usage == {
            "input_tokens": 400,
            "output_tokens": 0,
        }
        question = sent[0]["questions"]["profile"]
        assert list(question["criteria"]) == ["localfast", "gpt", "max"]
        assert question["instructions"] == routing.QUESTION
        assert sent[0]["state"]["request"] == "go"

    def test_no_request_with_one_candidate(self):
        def send(*args, **kwargs):
            raise AssertionError("must not ask")

        decision, _ = self._choose(_routing(), send, candidates=["gpt"])
        assert not decision.switched

    def test_failed_resolution_reuses_probabilities(self):
        calls = []

        def send(settings, envelope, **kwargs):
            calls.append(envelope)
            return _answer({"localfast": 0.05, "gpt": 0.15, "max": 0.8})

        tried = []

        def resolve(name):
            tried.append(name)
            if name == "max":
                raise Unusable("down")
            return name

        decision, _ = self._choose(
            _routing(),
            send,
            current="x",
            cstate="ineligible",
            reason="context",
            resolve=resolve,
        )
        assert tried == ["max", "gpt"]
        assert decision.profile == "gpt" and len(calls) == 1

    @pytest.mark.parametrize(
        "error, key, disables",
        [
            (decisions.DecisionUnavailable("refused"), "unavailable", True),
            (decisions.DecisionBlocked("blocked"), "blocked", False),
            (decisions.DecisionNotSent("too big"), "not_sent", False),
        ],
    )
    def test_failures_warn_once(self, error, key, disables):
        def send(*args, **kwargs):
            raise error

        state = _routing()
        first, warnings = self._choose(state, send)
        assert not first.switched and not first.probabilities
        assert len(warnings) == 1
        assert key in state.warned
        assert (state.off_reason is not None) == disables
        _, again = self._choose(state, send)
        assert again == []

    def test_decision_error_warns_every_time(self):
        def send(*args, **kwargs):
            raise decisions.DecisionError("unusable")

        state = _routing()
        assert len(self._choose(state, send)[1]) == 1
        assert len(self._choose(state, send)[1]) == 1
        assert state.off_reason is None

    def test_failure_moves_ineligible_to_first_candidate(self):
        def send(*args, **kwargs):
            raise decisions.DecisionError("unusable")

        decision, _ = self._choose(
            _routing(), send, current="x", cstate="ineligible", reason="unlisted"
        )
        assert decision.profile == "localfast"

    def test_cancel_propagates(self):
        def send(*args, **kwargs):
            raise decisions.DecisionCancelled("decide cancelled")

        with pytest.raises(decisions.DecisionCancelled):
            self._choose(_routing(), send)

    def test_state_over_budget_skips_the_request(self, monkeypatch):
        monkeypatch.setattr(routing, "STATE_BUDGET", 10)

        def send(*args, **kwargs):
            raise AssertionError("must not ask")

        state = _routing()
        decision, warnings = self._choose(state, send)
        assert decision.reason == "state does not fit the budget"
        assert "budget" in state.warned and len(warnings) == 1

    def test_answer_for_another_question_is_a_failure(self):
        def send(settings, envelope, **kwargs):
            return decisions.DecisionResult(
                model="m", answers={"redacted": {"probabilities": {"gpt": 1.0}}}
            )

        decision, warnings = self._choose(
            _routing(),
            send,
            current="x",
            cstate="ineligible",
            reason="unlisted",
            candidates=["gpt", "max"],
        )
        assert decision.profile == "gpt"
        assert decision.reason.startswith("first candidate")
        assert len(warnings) == 1

    def test_encrypted_labels_are_restored(self, monkeypatch):
        label = "ghp_ZYXWVUTSRQPONMLKJIHGFEDCBA9876543210"
        profiles = {
            "gpt": {"provider": "chatgpt", "description": "default model"},
            label: {"provider": "chatgpt", "description": "the other model"},
        }
        sent = _fake_systemone(monkeypatch)
        decision, _ = self._choose(
            _routing(profiles=profiles),
            candidates=["gpt", label],
            secret_shield=SecretShield(),
        )
        assert label not in json.dumps(sent[0])
        assert decision.switched and decision.profile == label

    def test_secrets_are_encrypted_before_anything_is_cut(self, monkeypatch):
        body = "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
        bearer = "Z9y8X7w6V5u4T3s2R1q0P9o8N7m6L5k4J3i2"
        quoted = "Q1w2E3r4T5y6U7i8O9p0A1s2D3f4G5h6J7k8"
        # Its body hides an AWS key, the only part the envelope would catch.
        header = "A" * 16 + "AKIA" + "B" * 16
        # Both cuts keep the token's body but drop the ghp_ prefix.
        request = "w " * 500 + "ghp_" + body + " " + "z" * (1199 - len(body))
        reply = "a" * 500 + "ghp_" + body + " " + "c" * (799 - len(body))
        commands = [
            # The tool summary limit cuts through the token, and collapsing
            # the tab would break the custom pattern's prefix.
            f"curl --retry 3 --max-time 30 -H 'Authorization:\tBearer {bearer}' x",
            # Only the raw arguments, before parsing, match this prefix.
            f"Token {quoted}",
        ]
        calls = [
            _call("run_command", {"command": c}, f"t{n}")
            for n, c in enumerate(commands)
        ]
        shield = SecretShield(
            extra_patterns=[
                {
                    "name": "bearer",
                    "prefix": "Authorization:\tBearer ",
                    "body_regex": "[A-Za-z0-9]{36}",
                },
                {
                    "name": "quoted",
                    "prefix": '"command": "Token ',
                    "body_regex": "[A-Za-z0-9]{36}",
                },
            ]
        )
        sent = _fake_systemone(monkeypatch)
        self._choose(
            _routing(),
            # Kept whole, but JSON escapes its tab, so it stays encrypted even
            # for the filter.
            messages=[
                _user(f"use Authorization:\tBearer {header}"),
                _assistant(tool_calls=calls),
                _assistant(reply),
            ],
            request=request,
            candidates=["gpt", "max"],
            secret_shield=shield,
        )
        state = sent[0]["state"]
        assert "request_size" in state
        assert "omitted" in state["previous_turn"]["assistant"]
        assert len(state["previous_turn"]["tools"]) == 2
        payload = json.dumps(sent[0])
        leaked = [s for s in (body, bearer, quoted, header[:16]) if s[:20] in payload]
        assert leaked == []

    def test_the_filter_reads_secrets_in_plaintext(self, monkeypatch, tmp_path):
        token = "ghp_A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
        # JSON escapes the newline this heuristic needs, so it cannot be put
        # back, but that must not hide the token from the filter.
        aws = "odJFCrnl2edlBDdz1C5Jau2RJtBRnlWmTSHf6pWk"
        seen = tmp_path / "seen.txt"
        record = _filter(
            tmp_path,
            f"open({str(seen)!r}, 'w').write(p['messages'][0]['content'])\n"
            "print(json.dumps({'messages': p['messages']}))\n",
        )
        # Tool calls are protected twice, and must still come back as plaintext.
        push = _call("run_command", {"command": f"git push https://{token}@x"})
        described = "ghp_Z9y8X7w6V5u4T3s2R1q0P9o8N7m6L5k4J3i2"
        profiles = dict(
            PROFILES,
            gpt=dict(PROFILES["gpt"], description=f"Deploys with {described}"),
            max=dict(PROFILES["max"], description=f"Remote, key:\n{aws}"),
        )
        sent = _fake_systemone(monkeypatch)
        self._choose(
            _routing(profiles),
            messages=[_user("publish it"), _assistant(tool_calls=[push])],
            request=f"tag the release, key:\n{aws}",
            candidates=["gpt", "max"],
            secret_shield=SecretShield(),
            llm_filter=record,
        )
        assert token in seen.read_text() and described in seen.read_text()
        payload = json.dumps(sent[0])
        assert [s for s in (token, described, aws[:20]) if s in payload] == []


class TestState:
    def test_pin_and_move(self):
        state = _routing()
        assert state.active
        state.pin("gpt", override=True)
        assert not state.active and state.explicit and state.override
        state.paused = False
        state.moved()
        assert state.active and not state.explicit and not state.override

    def test_summary(self):
        state = _routing()
        assert state.summary_line() == "routing: on"
        state.last = routing.RouteDecision(
            profile="max",
            previous="gpt",
            switched=True,
            reason="margin 0.8",
            probabilities={"gpt": 0.1, "max": 0.9},
        )
        assert state.summary_line() == "routing: on  (last: gpt -> max (0.90))"
        state.reset()
        state.off_reason = "decision model unavailable"
        assert state.summary_line() == "routing: off, decision model unavailable"

    def test_report_entry(self):
        decision = routing.RouteDecision(
            profile="gpt",
            previous="gpt",
            switched=False,
            reason="best pick",
            probabilities={"gpt": 0.9, "max": 0.1},
            latency=0.61234,
            usage={"input_tokens": 1},
        )
        assert decision.report_entry() == {
            "type": "routing",
            "profile": "gpt",
            "previous": "gpt",
            "switched": False,
            "reason": "best pick",
            "probabilities": {"gpt": 0.9, "max": 0.1},
            "latency_s": 0.612,
            "usage": {"input_tokens": 1},
        }

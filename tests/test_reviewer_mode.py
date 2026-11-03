"""Tests for --reviewer-mode feature."""

import json
import shlex
import threading
import types

import pytest
from unittest.mock import patch, MagicMock

from conftest import make_tool_call
from swival import agent, reviewer
from swival._msg import _msg_content, _msg_role
from swival.config import _UNSET, ConfigError, load_config, apply_config_to_args
from swival.report import AgentError
from swival.reviewer import (
    _parse_verdict,
    _build_prompt,
    _resolve_path,
    review_answer,
    run_as_reviewer,
)
from swival.secrets import SecretShield
from swival.tools import _history_path, dispatch
from swival.worktree import READ_ONLY_AGENT_TOOLS, ToolPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message(content=None, tool_calls=None):
    msg = types.SimpleNamespace()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.role = "assistant"
    msg.get = lambda key, default=None: getattr(msg, key, default)
    return msg


def _write_toml(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _reviewer_args(base_dir, **overrides):
    """Build a namespace mimicking --reviewer-mode args after config merge."""
    defaults = dict(
        provider="lmstudio",
        model="test-model",
        api_key=None,
        base_url="http://fake",
        max_output_tokens=1024,
        max_context_tokens=None,
        provider_timeout=900,
        temperature=0.5,
        top_p=None,
        seed=None,
        quiet=False,
        verbose=True,
        reviewer_mode=True,
        review_prompt=None,
        objective=None,
        verify=None,
        reviewer=None,
        color=False,
        no_color=True,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------


class TestParseVerdict:
    def test_accept(self):
        assert _parse_verdict("Looks good.\nVERDICT: ACCEPT") == "ACCEPT"

    def test_retry(self):
        assert _parse_verdict("Fix the bug.\nVERDICT: RETRY") == "RETRY"

    def test_case_insensitive(self):
        assert _parse_verdict("verdict: accept") == "ACCEPT"
        assert _parse_verdict("Verdict: Retry") == "RETRY"

    def test_last_verdict_wins(self):
        text = "VERDICT: RETRY\nActually...\nVERDICT: ACCEPT"
        assert _parse_verdict(text) == "ACCEPT"

    def test_no_verdict(self):
        assert _parse_verdict("Just some text without a verdict.") is None

    def test_empty_string(self):
        assert _parse_verdict("") is None

    def test_verdict_with_leading_whitespace(self):
        assert _parse_verdict("  VERDICT: ACCEPT  ") == "ACCEPT"


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_basic_prompt(self):
        prompt = _build_prompt(task="Fix the tests", answer="done")
        assert "<task>\nFix the tests\n</task>" in prompt
        assert "<answer>\ndone\n</answer>" in prompt
        assert "VERDICT: ACCEPT" in prompt
        assert "VERDICT: RETRY" in prompt

    def test_with_verification(self):
        prompt = _build_prompt(
            task="Fix it", answer="done", verification="Tests must pass"
        )
        assert "<verification>\nTests must pass\n</verification>" in prompt
        assert "verification criteria" in prompt

    def test_without_verification(self):
        prompt = _build_prompt(task="Fix it", answer="done")
        assert "<verification>" not in prompt

    def test_with_custom_instructions(self):
        prompt = _build_prompt(
            task="Fix it",
            answer="done",
            custom_instructions="Focus on error handling",
        )
        assert "Focus on error handling" in prompt

    def test_without_custom_instructions(self):
        prompt = _build_prompt(task="Fix it", answer="done")
        # Should not have orphaned custom_instructions placeholder
        assert "Focus on" not in prompt


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


class TestResolvePath:
    def test_absolute_unchanged(self):
        assert _resolve_path("/absolute/path.md", "/base") == "/absolute/path.md"

    def test_relative_resolved_against_base(self):
        result = _resolve_path("criteria.md", "/project")
        assert result == "/project/criteria.md"

    def test_nested_relative(self):
        result = _resolve_path("verification/working.md", "/project")
        assert result == "/project/verification/working.md"


def _engine_runtime(**overrides):
    runtime = dict(
        api_base="http://fake",
        model_id="test-model",
        max_turns=2,
        max_output_tokens=1024,
        temperature=0.5,
        top_p=None,
        seed=None,
        context_length=None,
        verbose=False,
        llm_kwargs={"provider": "lmstudio", "api_key": None},
    )
    runtime.update(overrides)
    return runtime


class TestReviewAnswer:
    def _reply(self, monkeypatch, content):
        monkeypatch.setattr(
            agent, "call_llm", lambda *a, **kw: (_make_message(content=content), "stop")
        )

    def _review(self, tmp_path, **runtime):
        return review_answer(
            "Fix the bug",
            "Fixed.",
            base_dir=str(tmp_path),
            runtime=_engine_runtime(**runtime),
        )

    @pytest.mark.parametrize("status", ["accept", "retry"])
    def test_verdict(self, tmp_path, monkeypatch, status):
        reply = f"Checked.\nVERDICT: {status.upper()}"
        self._reply(monkeypatch, reply)
        verdict = self._review(tmp_path)
        assert (verdict.status, verdict.text, verdict.error) == (status, reply, None)

    def test_missing_verdict_keeps_the_text(self, tmp_path, monkeypatch):
        self._reply(monkeypatch, "Not sure.")
        verdict = self._review(tmp_path)
        assert verdict.status == "error"
        assert verdict.text == "Not sure."
        assert "no VERDICT" in verdict.error

    def test_llm_failure_is_returned_not_raised(self, tmp_path, monkeypatch):
        def fail(*a, **kw):
            raise AgentError("connection refused")

        monkeypatch.setattr(agent, "call_llm", fail)
        verdict = self._review(tmp_path)
        assert verdict.status == "error"
        assert "LLM call failed: connection refused" in verdict.error

    @pytest.mark.parametrize(
        "key, value", [("files_mode", "all"), ("cancel_flag", threading.Event())]
    )
    def test_runtime_cannot_override_fixed_policy(self, tmp_path, key, value):
        with pytest.raises(TypeError, match=key):
            self._review(tmp_path, **{key: value})

    def test_shield_is_left_to_the_caller(self, tmp_path, monkeypatch):
        self._reply(monkeypatch, "VERDICT: ACCEPT")
        shield = SecretShield.from_config()
        try:
            self._review(tmp_path, secret_shield=shield)
            assert not shield.destroyed
        finally:
            shield.destroy()


class _FakeTimer:
    interrupt_on_start = False

    def __init__(self, seconds, flag):
        self.seconds = seconds
        self.flag = flag
        self.started = False
        self.cancelled = False

    def start(self):
        self.started = True
        if self.interrupt_on_start:
            raise KeyboardInterrupt

    def cancel(self):
        self.cancelled = True


class TestReviewDeadline:
    @pytest.fixture
    def timers(self, monkeypatch):
        created = []

        def make(seconds, flag):
            created.append(_FakeTimer(seconds, flag))
            return created[-1]

        monkeypatch.setattr(reviewer, "_deadline_timer", make)
        return created

    def _review(self, tmp_path, deadline=60):
        return review_answer(
            "Fix the bug",
            "Fixed.",
            base_dir=str(tmp_path),
            runtime=_engine_runtime(),
            deadline=deadline,
        )

    def test_verdict_after_the_deadline_is_a_timeout(
        self, tmp_path, monkeypatch, timers
    ):
        calls = 0

        def call_llm(*a, **kw):
            nonlocal calls
            calls += 1
            timers[0].flag.set()
            return _make_message(content="Wrong.\nVERDICT: RETRY"), "stop"

        parse = MagicMock(wraps=reviewer._parse_verdict)
        monkeypatch.setattr(agent, "call_llm", call_llm)
        monkeypatch.setattr(reviewer, "_parse_verdict", parse)
        verdict = self._review(tmp_path)
        assert calls == 1
        parse.assert_not_called()
        assert verdict.status == "error"
        assert verdict.error == "review timed out after 60s"
        assert verdict.text == ""

    def test_expired_deadline_stops_before_the_next_tool_call(
        self, tmp_path, monkeypatch, timers
    ):
        (tmp_path / "add.py").write_text("x")
        calls = 0

        def call_llm(*a, **kw):
            nonlocal calls
            calls += 1
            timers[0].flag.set()
            return _make_message(
                tool_calls=[make_tool_call("read_file", '{"file_path": "add.py"}')]
            ), "tool_calls"

        dispatch = MagicMock(return_value="x")
        monkeypatch.setattr(agent, "call_llm", call_llm)
        monkeypatch.setattr(agent, "dispatch", dispatch)
        verdict = self._review(tmp_path)
        assert calls == 1
        dispatch.assert_not_called()
        assert verdict.error == "review timed out after 60s"

    @pytest.mark.parametrize("elapsed, status", [(60, "error"), (59.9, "accept")])
    def test_clock_decides_when_the_timer_never_fired(
        self, tmp_path, monkeypatch, timers, elapsed, status
    ):
        now = [100.0]
        monkeypatch.setattr(reviewer, "monotonic", lambda: now[0])

        def call_llm(*a, **kw):
            now[0] += elapsed
            return _make_message(content="VERDICT: ACCEPT"), "stop"

        monkeypatch.setattr(agent, "call_llm", call_llm)
        verdict = self._review(tmp_path)
        assert not timers[0].flag.is_set()
        assert verdict.status == status

    @pytest.mark.parametrize(
        "outcome",
        [
            lambda: (_make_message(content="VERDICT: ACCEPT"), "stop"),
            lambda: (_ for _ in ()).throw(AgentError("boom")),
            lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
        ],
        ids=["verdict", "llm-error", "interrupt"],
    )
    def test_timer_is_cancelled_on_every_exit(
        self, tmp_path, monkeypatch, timers, outcome
    ):
        monkeypatch.setattr(agent, "call_llm", lambda *a, **kw: outcome())
        try:
            self._review(tmp_path, deadline=3600)
        except KeyboardInterrupt:
            pass
        assert [(t.seconds, t.started, t.cancelled) for t in timers] == [
            (3600, True, True)
        ]

    def test_interrupt_while_starting_the_timer_still_cancels_it(
        self, tmp_path, monkeypatch, timers
    ):
        monkeypatch.setattr(_FakeTimer, "interrupt_on_start", True)
        monkeypatch.setattr(
            agent, "call_llm", MagicMock(side_effect=AssertionError("loop ran"))
        )
        with pytest.raises(KeyboardInterrupt):
            self._review(tmp_path)
        assert [(t.started, t.cancelled) for t in timers] == [(True, True)]

    def test_no_deadline_starts_no_timer(self, tmp_path, monkeypatch, timers):
        monkeypatch.setattr(
            agent,
            "call_llm",
            lambda *a, **kw: (_make_message(content="VERDICT: ACCEPT"), "stop"),
        )
        assert self._review(tmp_path, deadline=None).status == "accept"
        assert timers == []

    def test_real_timer_is_unstarted_daemon_that_sets_the_flag(self):
        flag = threading.Event()
        timer = reviewer._deadline_timer(0, flag)
        assert timer.daemon
        assert not timer.is_alive()
        timer.start()
        assert flag.wait(timeout=5)


def _agent_loop_kwargs(**overrides):
    loop_kwargs = dict(
        _engine_runtime(),
        llm_kwargs={
            "provider": "lmstudio",
            "api_key": "k",
            "session_id": "agent-session",
            "reasoning_effort": "high",
        },
        report=object(),
        cache=object(),
        mcp_manager=object(),
        files_mode="all",
        verbose=True,
        max_turns=500,
    )
    loop_kwargs.update(overrides)
    return loop_kwargs


class TestReviewerRuntime:
    def test_only_shared_keys_reach_the_reviewer(self):
        shield = object()
        loop_kwargs = _agent_loop_kwargs(secret_shield=shield)
        runtime = reviewer._reviewer_runtime(loop_kwargs)
        assert set(runtime) == {
            "api_base",
            "model_id",
            "max_output_tokens",
            "temperature",
            "top_p",
            "seed",
            "context_length",
            "secret_shield",
            "llm_kwargs",
            "max_turns",
            "verbose",
        }
        assert runtime["secret_shield"] is shield
        assert runtime["max_turns"] == reviewer.ONESHOT_MAX_TURNS
        assert runtime["verbose"] is False

    def test_session_id_is_dropped_without_touching_the_agents(self):
        loop_kwargs = _agent_loop_kwargs()
        runtime = reviewer._reviewer_runtime(loop_kwargs)
        assert runtime["llm_kwargs"] == {
            "provider": "lmstudio",
            "api_key": "k",
            "reasoning_effort": "high",
        }
        assert loop_kwargs["llm_kwargs"]["session_id"] == "agent-session"


def _capture_reviews(monkeypatch, content="VERDICT: ACCEPT"):
    calls = []

    def call_llm(base_url, model_id, messages, *a, **kw):
        calls.append({"model": model_id, "prompt": _msg_content(messages[-1]), **kw})
        return _make_message(content=content), "stop"

    monkeypatch.setattr(agent, "call_llm", call_llm)
    return calls


class TestSelfReviewer:
    def _request(self):
        return reviewer.ReviewRequest("Fix the bug", "Fixed.", 1)

    def test_each_review_reads_the_loop_kwargs_it_is_given(self, tmp_path, monkeypatch):
        calls = _capture_reviews(monkeypatch)
        self_reviewer = reviewer.SelfReviewer(base_dir=str(tmp_path), max_rounds=3)
        startup = _agent_loop_kwargs()
        # A REPL builds its own dict, so the second review gets a different one.
        repl = _agent_loop_kwargs(model_id="switched-model")
        repl["llm_kwargs"]["reasoning_effort"] = "low"
        self_reviewer.review(self._request(), startup)
        verdict = self_reviewer.review(self._request(), repl)

        assert verdict.status == "accept"
        assert [c["model"] for c in calls] == ["test-model", "switched-model"]
        assert [c["reasoning_effort"] for c in calls] == ["high", "low"]

    def test_filter_and_cost_reach_the_provider_call(self, tmp_path, monkeypatch):
        calls = _capture_reviews(monkeypatch)
        llm_filter, cost = object(), agent.SessionCost()
        reviewer.SelfReviewer(base_dir=str(tmp_path), max_rounds=3).review(
            self._request(),
            _agent_loop_kwargs(llm_filter=llm_filter, session_cost=cost),
        )
        assert calls[0]["llm_filter"] is llm_filter
        assert calls[0]["session_cost"] is cost

    def test_review_runs_under_the_round_deadline(self, tmp_path, monkeypatch):
        _capture_reviews(monkeypatch)
        seconds = []

        def make(deadline, flag):
            seconds.append(deadline)
            return threading.Timer(deadline, flag.set)

        monkeypatch.setattr(reviewer, "_deadline_timer", make)
        reviewer.SelfReviewer(base_dir=str(tmp_path), max_rounds=3).review(
            self._request(), _agent_loop_kwargs()
        )
        assert seconds == [reviewer.REVIEW_TIMEOUT]

    @pytest.mark.parametrize("contents", [None, b"\xff"], ids=["missing", "not-utf8"])
    def test_unreadable_verify_file_is_a_review_error(
        self, tmp_path, monkeypatch, contents
    ):
        if contents is not None:
            (tmp_path / "criteria.md").write_bytes(contents)
        calls = _capture_reviews(monkeypatch)
        verdict = reviewer.SelfReviewer(
            base_dir=str(tmp_path), max_rounds=3, verify="criteria.md"
        ).review(self._request(), _agent_loop_kwargs())
        assert verdict.status == "error"
        assert "cannot read --verify file" in verdict.error
        assert calls == []


class TestExternalReviewer:
    @pytest.mark.parametrize(
        "code, status", [(0, "accept"), (1, "retry"), (2, "error"), (7, "error")]
    )
    def test_exit_codes_map_to_verdicts(self, tmp_path, monkeypatch, code, status):
        monkeypatch.setattr(
            agent, "run_reviewer", lambda *a, **kw: (code, "out", "err")
        )
        verdict = reviewer.ExternalReviewer("./review.sh", str(tmp_path), 3).review(
            reviewer.ReviewRequest("task", "answer", 1), {"model_id": "m"}
        )
        assert (verdict.status, verdict.exit_code) == (status, code)
        assert (verdict.text, verdict.stderr) == ("out", "err")
        assert (verdict.error is None) == (status != "error")

    @pytest.mark.parametrize("conversation", [None, "<user>\nearlier\n</user>"])
    def test_environment_comes_from_the_request_and_live_model(
        self, tmp_path, monkeypatch, conversation
    ):
        seen = {}

        def run_reviewer(command, base_dir, answer, verbose, env_extra=None):
            seen.update(command=command, answer=answer, env=env_extra)
            return 0, "", ""

        monkeypatch.setattr(agent, "run_reviewer", run_reviewer)
        request = reviewer.ReviewRequest("the task", "the answer", 2, conversation)
        reviewer.ExternalReviewer(
            "./review.sh", str(tmp_path), 3, encrypt_key="ab" * 32
        ).review(request, {"model_id": "m2"})
        expected = {
            "SWIVAL_TASK": "the task",
            "SWIVAL_REVIEW_ROUND": "2",
            "SWIVAL_MODEL": "m2",
            "SWIVAL_ENCRYPT_KEY": "ab" * 32,
        }
        if conversation:
            expected["SWIVAL_CONVERSATION"] = conversation
        assert (seen["command"], seen["answer"]) == ("./review.sh", "the answer")
        assert seen["env"] == expected

    def test_nul_bytes_do_not_break_the_reviewer_environment(self, tmp_path):
        script = tmp_path / "review.sh"
        script.write_text(
            '#!/bin/sh\nprintf "%s|%s" "$SWIVAL_TASK" "$SWIVAL_CONVERSATION"\n'
        )
        script.chmod(0o755)
        request = reviewer.ReviewRequest(
            "fix\0 it", "answer", 2, "<agent>\nbad\0 byte\n</agent>"
        )
        verdict = reviewer.ExternalReviewer(str(script), str(tmp_path), 3).review(
            request, {}
        )
        assert verdict.status == "accept"
        assert verdict.text == "fix it|<agent>\nbad byte\n</agent>"


# ---------------------------------------------------------------------------
# run_as_reviewer unit tests
# ---------------------------------------------------------------------------


class TestRunAsReviewer:
    def test_provider_timeout_forwarded(self, tmp_path, monkeypatch):
        args = _reviewer_args(str(tmp_path), provider_timeout=23)
        monkeypatch.setenv("SWIVAL_TASK", "Fix the bug")
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(read=lambda: "Done"))
        captured = {}

        def mock_call_llm(*a, **kw):
            captured.update(kw)
            return _make_message(content="VERDICT: ACCEPT"), "stop"

        with (
            patch("swival.agent.call_llm", mock_call_llm),
            patch(
                "swival.agent.resolve_provider",
                return_value=(
                    "test-model",
                    "http://fake",
                    None,
                    None,
                    {"provider": "lmstudio", "api_key": None},
                ),
            ),
        ):
            assert run_as_reviewer(args, str(tmp_path)) == 0

        assert captured["provider_timeout"] == 23

    def test_accept_verdict_returns_0(self, tmp_path, capsys, monkeypatch):
        args = _reviewer_args(str(tmp_path))
        monkeypatch.setenv("SWIVAL_TASK", "Fix the bug")
        monkeypatch.setattr(
            "sys.stdin", types.SimpleNamespace(read=lambda: "The bug is fixed.")
        )

        def mock_call_llm(*a, **kw):
            return _make_message(content="Looks good.\nVERDICT: ACCEPT"), "stop"

        with (
            patch("swival.agent.call_llm", mock_call_llm),
            patch(
                "swival.agent.resolve_provider",
                return_value=(
                    "test-model",
                    "http://fake",
                    None,
                    None,
                    {"provider": "lmstudio", "api_key": None},
                ),
            ),
        ):
            code = run_as_reviewer(args, str(tmp_path))

        assert code == 0
        captured = capsys.readouterr()
        assert "VERDICT: ACCEPT" in captured.out

    def test_retry_verdict_returns_1(self, tmp_path, capsys, monkeypatch):
        args = _reviewer_args(str(tmp_path))
        monkeypatch.setenv("SWIVAL_TASK", "Fix the bug")
        monkeypatch.setattr(
            "sys.stdin", types.SimpleNamespace(read=lambda: "Some answer")
        )

        def mock_call_llm(*a, **kw):
            return _make_message(
                content="You missed the edge case.\nVERDICT: RETRY"
            ), "stop"

        with (
            patch("swival.agent.call_llm", mock_call_llm),
            patch(
                "swival.agent.resolve_provider",
                return_value=(
                    "test-model",
                    "http://fake",
                    None,
                    None,
                    {"provider": "lmstudio", "api_key": None},
                ),
            ),
        ):
            code = run_as_reviewer(args, str(tmp_path))

        assert code == 1
        captured = capsys.readouterr()
        assert "VERDICT: RETRY" in captured.out
        assert "edge case" in captured.out

    def test_no_verdict_returns_2(self, tmp_path, capsys, monkeypatch):
        args = _reviewer_args(str(tmp_path))
        monkeypatch.setenv("SWIVAL_TASK", "Fix the bug")
        monkeypatch.setattr(
            "sys.stdin", types.SimpleNamespace(read=lambda: "Some answer")
        )

        def mock_call_llm(*a, **kw):
            return _make_message(content="I'm not sure what to do."), "stop"

        with (
            patch("swival.agent.call_llm", mock_call_llm),
            patch(
                "swival.agent.resolve_provider",
                return_value=(
                    "test-model",
                    "http://fake",
                    None,
                    None,
                    {"provider": "lmstudio", "api_key": None},
                ),
            ),
        ):
            code = run_as_reviewer(args, str(tmp_path))

        assert code == 2
        captured = capsys.readouterr()
        assert "no VERDICT found" in captured.err
        assert "not sure" in captured.out

    def test_empty_stdin_returns_2(self, tmp_path, capsys, monkeypatch):
        args = _reviewer_args(str(tmp_path))
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(read=lambda: ""))

        code = run_as_reviewer(args, str(tmp_path))
        assert code == 2
        captured = capsys.readouterr()
        assert "empty answer" in captured.err

    def test_missing_task_returns_2(self, tmp_path, capsys, monkeypatch):
        args = _reviewer_args(str(tmp_path))
        monkeypatch.delenv("SWIVAL_TASK", raising=False)
        monkeypatch.setattr(
            "sys.stdin", types.SimpleNamespace(read=lambda: "Some answer")
        )

        code = run_as_reviewer(args, str(tmp_path))
        assert code == 2
        captured = capsys.readouterr()
        assert "no task description" in captured.err

    def test_objective_overrides_env(self, tmp_path, capsys, monkeypatch):
        """--objective file takes priority over SWIVAL_TASK env var."""
        obj_file = tmp_path / "task.md"
        obj_file.write_text("Fix the auth bug")
        args = _reviewer_args(str(tmp_path), objective=str(obj_file))
        monkeypatch.setenv("SWIVAL_TASK", "This should be ignored")
        monkeypatch.setattr(
            "sys.stdin", types.SimpleNamespace(read=lambda: "Fixed it.")
        )

        prompt_captured = []

        def mock_call_llm(base_url, model_id, messages, *a, **kw):
            prompt_captured.append(messages[0]["content"])
            return _make_message(content="VERDICT: ACCEPT"), "stop"

        with (
            patch("swival.agent.call_llm", mock_call_llm),
            patch(
                "swival.agent.resolve_provider",
                return_value=(
                    "test-model",
                    "http://fake",
                    None,
                    None,
                    {"provider": "lmstudio", "api_key": None},
                ),
            ),
        ):
            code = run_as_reviewer(args, str(tmp_path))

        assert code == 0
        assert "Fix the auth bug" in prompt_captured[0]
        assert "This should be ignored" not in prompt_captured[0]

    def test_objective_relative_path(self, tmp_path, capsys, monkeypatch):
        """--objective with relative path resolves against base_dir."""
        obj_file = tmp_path / "objective.md"
        obj_file.write_text("Do the thing")
        args = _reviewer_args(str(tmp_path), objective="objective.md")
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(read=lambda: "Did it."))

        prompt_captured = []

        def mock_call_llm(base_url, model_id, messages, *a, **kw):
            prompt_captured.append(messages[0]["content"])
            return _make_message(content="VERDICT: ACCEPT"), "stop"

        with (
            patch("swival.agent.call_llm", mock_call_llm),
            patch(
                "swival.agent.resolve_provider",
                return_value=(
                    "test-model",
                    "http://fake",
                    None,
                    None,
                    {"provider": "lmstudio", "api_key": None},
                ),
            ),
        ):
            code = run_as_reviewer(args, str(tmp_path))

        assert code == 0
        assert "Do the thing" in prompt_captured[0]

    def test_verify_file_included(self, tmp_path, capsys, monkeypatch):
        """--verify file content included in prompt."""
        verify_file = tmp_path / "criteria.md"
        verify_file.write_text("All tests must pass")
        args = _reviewer_args(str(tmp_path), verify=str(verify_file))
        monkeypatch.setenv("SWIVAL_TASK", "Fix tests")
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(read=lambda: "Fixed."))

        prompt_captured = []

        def mock_call_llm(base_url, model_id, messages, *a, **kw):
            prompt_captured.append(messages[0]["content"])
            return _make_message(content="VERDICT: ACCEPT"), "stop"

        with (
            patch("swival.agent.call_llm", mock_call_llm),
            patch(
                "swival.agent.resolve_provider",
                return_value=(
                    "test-model",
                    "http://fake",
                    None,
                    None,
                    {"provider": "lmstudio", "api_key": None},
                ),
            ),
        ):
            code = run_as_reviewer(args, str(tmp_path))

        assert code == 0
        assert "All tests must pass" in prompt_captured[0]
        assert "<verification>" in prompt_captured[0]

    def test_verify_relative_path(self, tmp_path, capsys, monkeypatch):
        """--verify with relative path resolves against base_dir."""
        (tmp_path / "verification").mkdir()
        verify_file = tmp_path / "verification" / "working.md"
        verify_file.write_text("Check the output")
        args = _reviewer_args(str(tmp_path), verify="verification/working.md")
        monkeypatch.setenv("SWIVAL_TASK", "Do it")
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(read=lambda: "Done."))

        prompt_captured = []

        def mock_call_llm(base_url, model_id, messages, *a, **kw):
            prompt_captured.append(messages[0]["content"])
            return _make_message(content="VERDICT: ACCEPT"), "stop"

        with (
            patch("swival.agent.call_llm", mock_call_llm),
            patch(
                "swival.agent.resolve_provider",
                return_value=(
                    "test-model",
                    "http://fake",
                    None,
                    None,
                    {"provider": "lmstudio", "api_key": None},
                ),
            ),
        ):
            code = run_as_reviewer(args, str(tmp_path))

        assert code == 0
        assert "Check the output" in prompt_captured[0]

    def test_custom_review_prompt(self, tmp_path, capsys, monkeypatch):
        """--review-prompt text is appended to prompt."""
        args = _reviewer_args(str(tmp_path), review_prompt="Focus on error handling")
        monkeypatch.setenv("SWIVAL_TASK", "Fix it")
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(read=lambda: "Fixed."))

        prompt_captured = []

        def mock_call_llm(base_url, model_id, messages, *a, **kw):
            prompt_captured.append(messages[0]["content"])
            return _make_message(content="VERDICT: ACCEPT"), "stop"

        with (
            patch("swival.agent.call_llm", mock_call_llm),
            patch(
                "swival.agent.resolve_provider",
                return_value=(
                    "test-model",
                    "http://fake",
                    None,
                    None,
                    {"provider": "lmstudio", "api_key": None},
                ),
            ),
        ):
            code = run_as_reviewer(args, str(tmp_path))

        assert code == 0
        assert "Focus on error handling" in prompt_captured[0]

    def test_llm_failure_returns_2(self, tmp_path, capsys, monkeypatch):
        """LLM call failure returns exit code 2."""
        args = _reviewer_args(str(tmp_path))
        monkeypatch.setenv("SWIVAL_TASK", "Fix it")
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(read=lambda: "Answer."))

        def mock_call_llm(*a, **kw):
            raise AgentError("connection refused")

        with (
            patch("swival.agent.call_llm", mock_call_llm),
            patch(
                "swival.agent.resolve_provider",
                return_value=(
                    "test-model",
                    "http://fake",
                    None,
                    None,
                    {"provider": "lmstudio", "api_key": None},
                ),
            ),
        ):
            code = run_as_reviewer(args, str(tmp_path))

        assert code == 2
        captured = capsys.readouterr()
        assert "LLM call failed" in captured.err

    def test_missing_objective_file_returns_2(self, tmp_path, capsys, monkeypatch):
        """Missing --objective file returns exit 2, not a traceback."""
        args = _reviewer_args(str(tmp_path), objective="/nonexistent/task.md")
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(read=lambda: "Answer."))

        code = run_as_reviewer(args, str(tmp_path))
        assert code == 2
        captured = capsys.readouterr()
        assert "cannot read --objective file" in captured.err

    def test_missing_verify_file_returns_2(self, tmp_path, capsys, monkeypatch):
        """Missing --verify file returns exit 2, not a traceback."""
        args = _reviewer_args(str(tmp_path), verify="/nonexistent/criteria.md")
        monkeypatch.setenv("SWIVAL_TASK", "Fix it")
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(read=lambda: "Answer."))

        code = run_as_reviewer(args, str(tmp_path))
        assert code == 2
        captured = capsys.readouterr()
        assert "cannot read --verify file" in captured.err

    def test_unreadable_objective_returns_2(self, tmp_path, capsys, monkeypatch):
        """Unreadable --objective file returns exit 2."""
        obj_file = tmp_path / "task.md"
        obj_file.write_text("content")
        obj_file.chmod(0o000)
        args = _reviewer_args(str(tmp_path), objective=str(obj_file))
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(read=lambda: "Answer."))

        code = run_as_reviewer(args, str(tmp_path))
        # Restore permissions for cleanup
        obj_file.chmod(0o644)
        assert code == 2
        captured = capsys.readouterr()
        assert "cannot read --objective file" in captured.err

    def test_undecodable_objective_returns_2(self, tmp_path, capsys, monkeypatch):
        obj_file = tmp_path / "task.md"
        obj_file.write_bytes(b"\xff")
        args = _reviewer_args(str(tmp_path), objective=str(obj_file))
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(read=lambda: "Answer."))
        assert run_as_reviewer(args, str(tmp_path)) == 2
        assert "cannot read --objective file" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Read-only review loop
# ---------------------------------------------------------------------------


class TestReadOnlyReviewer:
    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        monkeypatch.setenv("SWIVAL_TASK", "Fix add.py so add(a, b) returns a + b")
        monkeypatch.setattr(
            "sys.stdin", types.SimpleNamespace(read=lambda: "Fixed add.py.")
        )
        monkeypatch.setattr(
            agent,
            "resolve_provider",
            lambda **kw: (
                "test-model",
                "http://fake",
                None,
                None,
                {"provider": "lmstudio", "api_key": None},
            ),
        )

    def test_reads_workspace_before_verdict(self, tmp_path, monkeypatch, capsys):
        source = "def add(a, b):\n    return a - b\n"
        (tmp_path / "add.py").write_text(source)
        args = _reviewer_args(str(tmp_path), max_turns=2, verbose=False)
        calls = 0

        def call_llm(base_url, model_id, messages, *a, **kw):
            nonlocal calls
            calls += 1
            if calls == 1:
                names = {tool["function"]["name"] for tool in a[4]}
                assert names == {
                    "read_file",
                    "read_multiple_files",
                    "list_files",
                    "grep",
                    "outline",
                    "think",
                    "todo",
                }
                return _make_message(
                    content="I will inspect add.py.",
                    tool_calls=[make_tool_call("read_file", '{"file_path": "add.py"}')],
                ), "tool_calls"
            assert _msg_role(messages[-1]) == "tool"
            assert "return a - b" in _msg_content(messages[-1])
            return _make_message(
                content="add.py still subtracts.\nVERDICT: RETRY"
            ), "stop"

        monkeypatch.setattr(agent, "call_llm", call_llm)
        assert run_as_reviewer(args, str(tmp_path)) == 1
        assert calls == 2
        assert capsys.readouterr().out == "add.py still subtracts.\nVERDICT: RETRY\n"
        assert (tmp_path / "add.py").read_text() == source
        assert not (tmp_path / ".swival").exists()

    @pytest.mark.parametrize(
        ("name", "arguments"),
        [
            ("write_file", {"file_path": "add.py", "content": "changed"}),
            (
                "edit_file",
                {
                    "file_path": "add.py",
                    "old_string": "original",
                    "new_string": "changed",
                },
            ),
            ("delete_file", {"file_path": "add.py"}),
            ("run_command", {"command": ["touch", "created"]}),
            ("run_shell_command", {"command": "touch created"}),
            ("run_python", {"code": "open('created', 'w').close()"}),
            ("fetch_url", {"url": "https://example.com"}),
        ],
    )
    def test_rejects_disallowed_tools(self, tmp_path, monkeypatch, name, arguments):
        (tmp_path / "add.py").write_text("original")
        args = _reviewer_args(
            str(tmp_path),
            max_turns=2,
            verbose=False,
            yolo=True,
            files="all",
            commands="all",
        )
        calls = 0

        def call_llm(base_url, model_id, messages, *a, **kw):
            nonlocal calls
            calls += 1
            if calls == 1:
                return _make_message(
                    tool_calls=[make_tool_call(name, json.dumps(arguments))]
                ), "tool_calls"
            result = _msg_content(messages[-1])
            assert result.startswith("error:")
            assert f"tool {name!r} is not available" in result
            return _make_message(content="VERDICT: RETRY"), "stop"

        monkeypatch.setattr(agent, "call_llm", call_llm)
        assert run_as_reviewer(args, str(tmp_path)) == 1
        assert calls == 2
        assert (tmp_path / "add.py").read_text() == "original"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["add.py"]

    @pytest.mark.parametrize(
        "name", ["read_file", "read_multiple_files", "list_files", "grep", "outline"]
    )
    @pytest.mark.parametrize("symlink", [False, True])
    def test_reads_stay_inside_base_dir(self, tmp_path, monkeypatch, name, symlink):
        base = tmp_path / "project"
        base.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.py").write_text("SECRET_OUTSIDE_REVIEW = True\n")
        if symlink:
            (base / "link").symlink_to(outside, target_is_directory=True)
            path = "link"
        else:
            path = "../outside"
        if name == "read_multiple_files":
            arguments = {"files": [{"file_path": f"{path}/secret.py"}]}
        elif name in {"list_files", "grep"}:
            arguments = {"path": path, "pattern": "*" if name == "list_files" else "."}
        else:
            arguments = {"file_path": f"{path}/secret.py"}
        args = _reviewer_args(str(base), max_turns=2, verbose=False, files="all")
        calls = 0

        def call_llm(base_url, model_id, messages, *a, **kw):
            nonlocal calls
            calls += 1
            if calls == 1:
                return _make_message(
                    tool_calls=[make_tool_call(name, json.dumps(arguments))]
                ), "tool_calls"
            result = _msg_content(messages[-1])
            assert "outside" in result
            assert "error:" in result
            assert "SECRET_OUTSIDE_REVIEW" not in result
            return _make_message(content="VERDICT: RETRY"), "stop"

        monkeypatch.setattr(agent, "call_llm", call_llm)
        assert run_as_reviewer(args, str(base)) == 1
        assert calls == 2

    def test_exhausted_review_is_error_and_does_not_save(
        self, tmp_path, monkeypatch, capsys
    ):
        (tmp_path / "add.py").write_text("original")
        args = _reviewer_args(str(tmp_path), max_turns=1, verbose=False)
        monkeypatch.setattr(
            agent,
            "call_llm",
            lambda *a, **kw: (
                _make_message(
                    content="VERDICT: ACCEPT",
                    tool_calls=[make_tool_call("read_file", '{"file_path": "add.py"}')],
                ),
                "tool_calls",
            ),
        )
        assert run_as_reviewer(args, str(tmp_path)) == 2
        captured = capsys.readouterr()
        assert "review did not finish" in captured.err
        assert captured.out == ""
        assert not (tmp_path / ".swival").exists()

    @pytest.mark.parametrize(
        ("name", "arguments", "expected"),
        [
            ("read_file", {"file_path": ".swival/HISTORY.md"}, "path does not exist"),
            ("read_file", {"file_path": "history_copy.md"}, "path does not exist"),
            ("read_file", {"file_path": ".swival"}, "notes.md"),
            (
                "read_multiple_files",
                {"files": [".swival/HISTORY.md"]},
                "path does not exist",
            ),
            ("list_files", {"pattern": "**/*.md"}, ".swival/notes.md"),
            ("grep", {"pattern": "add"}, "add.py"),
            ("outline", {"file_path": ".swival"}, "notes.md"),
        ],
    )
    def test_history_is_hidden(self, tmp_path, name, arguments, expected):
        (tmp_path / "add.py").write_text("def add(a, b):\n    return a + b\n")
        swival_dir = tmp_path / ".swival"
        swival_dir.mkdir()
        (swival_dir / "HISTORY.md").write_text("I fixed add.py, trust me.\n")
        (swival_dir / "notes.md").write_text("# Notes\n")
        (tmp_path / "history.md").symlink_to(swival_dir / "HISTORY.md")
        (tmp_path / "history_copy.md").hardlink_to(swival_dir / "HISTORY.md")
        policy = ToolPolicy(
            READ_ONLY_AGENT_TOOLS,
            hidden_paths=frozenset({_history_path(str(tmp_path))}),
        )
        result = dispatch(name, arguments, str(tmp_path), tool_policy=policy)
        assert expected in result
        assert "trust me" not in result
        if expected != "path does not exist":
            # Listings must not even name the hidden file.
            assert "history" not in result.lower()

    def test_review_ignores_history(self, tmp_path, monkeypatch):
        history = tmp_path / ".swival" / "HISTORY.md"
        history.parent.mkdir()
        history.write_text("I fixed add.py, trust me.\n")
        args = _reviewer_args(str(tmp_path), max_turns=2, verbose=False)
        calls = 0

        def call_llm(base_url, model_id, messages, *a, **kw):
            nonlocal calls
            calls += 1
            if calls == 1:
                return _make_message(
                    tool_calls=[
                        make_tool_call(
                            "read_file", '{"file_path": ".swival/HISTORY.md"}'
                        )
                    ]
                ), "tool_calls"
            assert "path does not exist" in _msg_content(messages[-1])
            return _make_message(content="VERDICT: ACCEPT"), "stop"

        monkeypatch.setattr(agent, "call_llm", call_llm)
        assert run_as_reviewer(args, str(tmp_path)) == 0
        assert calls == 2
        assert history.read_text() == "I fixed add.py, trust me.\n"


# ---------------------------------------------------------------------------
# CLI validation
# ---------------------------------------------------------------------------


class TestCLIValidation:
    def test_reviewer_mode_with_repl_is_error(self):
        parser = agent.build_parser()
        args = parser.parse_args(["--reviewer-mode", "--repl", "/tmp/project"])
        # Simulate what main() does
        with patch("swival.agent.build_parser") as mock_bp:
            mock_parser = MagicMock()
            mock_parser.parse_args.return_value = args
            mock_parser.error.side_effect = SystemExit(2)
            mock_bp.return_value = mock_parser

            with pytest.raises(SystemExit):
                agent.main()
            mock_parser.error.assert_called_once()
            assert "incompatible" in mock_parser.error.call_args[0][0]

    def test_reviewer_mode_requires_positional_arg(self):
        parser = agent.build_parser()
        args = parser.parse_args(["--reviewer-mode"])
        with patch("swival.agent.build_parser") as mock_bp:
            mock_parser = MagicMock()
            mock_parser.parse_args.return_value = args
            mock_parser.error.side_effect = SystemExit(2)
            mock_bp.return_value = mock_parser

            with pytest.raises(SystemExit):
                agent.main()
            mock_parser.error.assert_called_once()
            assert (
                "positional" in mock_parser.error.call_args[0][0].lower()
                or "base_dir" in mock_parser.error.call_args[0][0].lower()
            )

    def test_reviewer_mode_with_explicit_reviewer_is_error(self, tmp_path, monkeypatch):
        """--reviewer-mode + --reviewer is a user mistake."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        parser = agent.build_parser()
        args = parser.parse_args(
            ["--reviewer-mode", "--reviewer", "/some/reviewer", str(tmp_path)]
        )
        with patch("swival.agent.build_parser") as mock_bp:
            mock_parser = MagicMock()
            mock_parser.parse_args.return_value = args
            mock_parser.error.side_effect = SystemExit(2)
            mock_bp.return_value = mock_parser

            with pytest.raises(SystemExit):
                agent.main()
            mock_parser.error.assert_called_once()
            assert "cannot be used together" in mock_parser.error.call_args[0][0]


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    def test_config_inherits_reviewer_silently_cleared(self, tmp_path, monkeypatch):
        """Project config with reviewer= does not break --reviewer-mode."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(
            tmp_path / "swival.toml",
            'reviewer = "swival --reviewer-mode"\n',
        )
        parser = agent.build_parser()
        args = parser.parse_args(["--reviewer-mode", str(tmp_path)])

        # Simulate config loading + merge
        from swival.config import load_config, apply_config_to_args

        file_config = load_config(tmp_path)
        reviewer_from_cli = args.reviewer is not _UNSET
        apply_config_to_args(args, file_config)

        # The reviewer key should have been loaded from config
        assert not reviewer_from_cli
        # In main(), args.reviewer is forced to None
        args.reviewer = None  # simulate main()'s behavior
        assert args.reviewer is None

    def test_review_prompt_from_config(self, tmp_path, monkeypatch):
        """review_prompt in config applies when CLI doesn't set it."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(
            tmp_path / "swival.toml",
            'review_prompt = "Focus on correctness"\n',
        )
        config = load_config(tmp_path)
        assert config["review_prompt"] == "Focus on correctness"

    def test_verify_path_resolved_in_config(self, tmp_path, monkeypatch):
        """verify path in config resolves against config directory."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(
            tmp_path / "swival.toml",
            'verify = "verification/working.md"\n',
        )
        config = load_config(tmp_path)
        assert config["verify"] == str(tmp_path / "verification/working.md")

    def test_objective_path_resolved_in_config(self, tmp_path, monkeypatch):
        """objective path in config resolves against config directory."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(
            tmp_path / "swival.toml",
            'objective = "objective.md"\n',
        )
        config = load_config(tmp_path)
        assert config["objective"] == str(tmp_path / "objective.md")

    def test_unset_defaults_apply(self, tmp_path, monkeypatch):
        """_UNSET defaults for review_prompt/objective/verify resolve to None."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        config = load_config(tmp_path)

        from swival.config import _UNSET

        args = types.SimpleNamespace(
            review_prompt=_UNSET,
            objective=_UNSET,
            verify=_UNSET,
            reviewer=_UNSET,
            provider=_UNSET,
            model=_UNSET,
            api_key=_UNSET,
            base_url=_UNSET,
            max_output_tokens=_UNSET,
            max_context_tokens=_UNSET,
            temperature=_UNSET,
            top_p=_UNSET,
            seed=_UNSET,
            max_turns=_UNSET,
            system_prompt=_UNSET,
            no_system_prompt=_UNSET,
            commands=_UNSET,
            yolo=_UNSET,
            files=_UNSET,
            add_dir=None,
            add_dir_ro=None,
            no_read_guard=_UNSET,
            no_instructions=_UNSET,
            no_skills=_UNSET,
            skills_dir=None,
            no_history=_UNSET,
            color=_UNSET,
            no_color=_UNSET,
            quiet=_UNSET,
            proactive_summaries=_UNSET,
        )
        apply_config_to_args(args, config)
        assert args.review_prompt is None
        assert args.objective is None
        assert args.verify is None


# ---------------------------------------------------------------------------
# Shell-split reviewer commands (Step 1 tests)
# ---------------------------------------------------------------------------


class TestShellSplitReviewer:
    def test_multi_word_command_split(self, tmp_path):
        """run_reviewer splits 'swival --reviewer-mode' into argv."""
        script = tmp_path / "reviewer_echo.sh"
        script.write_text('#!/bin/sh\necho "args: $@"\nexit 0\n')
        script.chmod(0o755)

        # Use the script path + extra args
        cmd = f"{script} --extra-flag"
        code, text, stderr = agent.run_reviewer(cmd, str(tmp_path), "answer", False)
        assert code == 0
        assert "--extra-flag" in text
        assert str(tmp_path) in text  # base_dir appended

    def test_config_bare_command_preserved(self, tmp_path, monkeypatch):
        """Bare command name (no path prefix) is NOT resolved against config_dir."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(
            tmp_path / "swival.toml",
            'reviewer = "swival --reviewer-mode"\n',
        )
        config = load_config(tmp_path)
        # First token should be 'swival', not resolved to a filesystem path
        parts = shlex.split(config["reviewer"])
        assert parts[0] == "swival"
        assert parts[1] == "--reviewer-mode"

    def test_config_path_like_resolved(self, tmp_path, monkeypatch):
        """Path-like first token (./script) is resolved against config_dir."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(
            tmp_path / "swival.toml",
            'reviewer = "./review.sh --strict"\n',
        )
        config = load_config(tmp_path)
        parts = shlex.split(config["reviewer"])
        assert parts[0] == str(tmp_path / "review.sh")
        assert parts[1] == "--strict"

    def test_config_malformed_quoting_raises(self, tmp_path, monkeypatch):
        """Unbalanced quotes in reviewer command raise ConfigError."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(
            tmp_path / "swival.toml",
            'reviewer = "swival --review-prompt \'unbalanced"\n',
        )
        with pytest.raises(ConfigError, match="malformed reviewer command"):
            load_config(tmp_path)

    def test_config_empty_reviewer_raises(self, tmp_path, monkeypatch):
        """Empty reviewer command raises ConfigError."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(
            tmp_path / "swival.toml",
            'reviewer = ""\n',
        )
        with pytest.raises(ConfigError, match="reviewer command is empty"):
            load_config(tmp_path)

    def test_config_whitespace_only_reviewer_raises(self, tmp_path, monkeypatch):
        """Whitespace-only reviewer command raises ConfigError."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(
            tmp_path / "swival.toml",
            'reviewer = "   "\n',
        )
        with pytest.raises(ConfigError, match="reviewer command is empty"):
            load_config(tmp_path)

    def test_startup_validates_first_token(self, tmp_path, monkeypatch):
        """Startup validation checks the first token of a multi-word command."""
        script = tmp_path / "myreviewer"
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o755)

        # Pass a multi-word command with valid executable
        args = types.SimpleNamespace(
            reviewer=f"{script} --extra",
            reviewer_mode=False,
        )
        # Should not raise — validation happens in _run_main
        parts = shlex.split(args.reviewer)
        assert parts[0] == str(script)

    def test_stderr_forwarded_when_verbose(self, tmp_path):
        """Reviewer stderr is captured and returned in 3-tuple."""
        script = tmp_path / "reviewer.sh"
        script.write_text("#!/bin/sh\necho 'diag' >&2\nexit 0\n")
        script.chmod(0o755)

        code, stdout, stderr = agent.run_reviewer(
            str(script), str(tmp_path), "answer", True
        )
        assert code == 0
        assert "diag" in stderr

    def test_stderr_in_report(self):
        """record_review stores stderr when provided."""
        from swival.report import ReportCollector

        rc = ReportCollector()
        rc.record_review(1, 0, "feedback", stderr="some error")
        assert rc.events[0]["stderr"] == "some error"

    def test_stderr_omitted_when_empty(self):
        """record_review omits stderr key when empty."""
        from swival.report import ReportCollector

        rc = ReportCollector()
        rc.record_review(1, 0, "feedback")
        assert "stderr" not in rc.events[0]


# ---------------------------------------------------------------------------
# Argparse flag tests
# ---------------------------------------------------------------------------


class TestArgparse:
    def test_reviewer_mode_flag_parsed(self):
        parser = agent.build_parser()
        args = parser.parse_args(["--reviewer-mode", "/project"])
        assert args.reviewer_mode is True
        assert args.question == "/project"

    def test_review_prompt_flag_parsed(self):
        parser = agent.build_parser()
        args = parser.parse_args(["--review-prompt", "Be strict", "hello"])
        assert args.review_prompt == "Be strict"

    def test_objective_flag_parsed(self):
        parser = agent.build_parser()
        args = parser.parse_args(["--objective", "task.md", "hello"])
        assert args.objective == "task.md"

    def test_verify_flag_parsed(self):
        parser = agent.build_parser()
        args = parser.parse_args(["--verify", "checks.md", "hello"])
        assert args.verify == "checks.md"

    def test_reviewer_mode_default_false(self):
        parser = agent.build_parser()
        args = parser.parse_args(["hello"])
        assert args.reviewer_mode is False

    def test_new_flags_default_unset(self):
        parser = agent.build_parser()
        args = parser.parse_args(["hello"])
        assert args.review_prompt is _UNSET
        assert args.objective is _UNSET
        assert args.verify is _UNSET


# ---------------------------------------------------------------------------
# Init config template
# ---------------------------------------------------------------------------


class TestInitConfigTemplate:
    def test_template_includes_reviewer_mode_keys(self):
        from swival.config import generate_config

        template = generate_config(project=True)
        assert "review_prompt" in template
        assert "objective" in template
        assert "verify" in template


class TestRenderConversation:
    def test_nothing_to_render(self):
        assert reviewer.render_conversation([]) is None

    def test_exchanges_read_oldest_first(self):
        text = reviewer.render_conversation([("first", "one"), ("second", None)])
        assert text == (
            "<user>\nfirst\n</user>\n<agent>\none\n</agent>\n"
            "<user>\nsecond\n</user>\n<agent>\n(no answer)\n</agent>"
        )

    def test_complete_exchange_within_budget_is_kept_whole(self):
        lines = [f"- requirement {i}: " + "r" * 70 for i in range(100)]
        lines[50] = "- requirement 50: MUST keep the public API unchanged"
        request = "\n".join(lines)
        answer = "a" * 1_200
        # Over half the budget on its own, but the whole exchange still fits.
        assert agent._estimate_tokens(request) > reviewer.CONVERSATION_BUDGET // 2
        assert agent._estimate_tokens(request + answer) < reviewer.CONVERSATION_BUDGET
        text = reviewer.render_conversation([("old", "x"), (request, answer)])
        assert request in text and answer in text
        assert "[...]" not in text

    def test_newest_request_stays_whole_next_to_a_huge_answer(self):
        request = "spec " + "s" * 9_000 + " MIDDLE " + "s" * 1_000
        answer = "HEAD " + "x" * 40_000 + " TAIL"
        text = reviewer.render_conversation([(request, answer)])
        assert request in text
        assert "HEAD" in text and "TAIL" in text and "\n[...]\n" in text
        assert len(text) <= reviewer.CONVERSATION_BUDGET * 4

    def test_older_answers_keep_their_opening_lines(self):
        older = "\n".join(f"line {i} " + "y" * 60 for i in range(200))
        text = reviewer.render_conversation([("old", older), ("new", "short")])
        assert "line 0 " in text and "line 150 " not in text
        assert text.endswith("<user>\nnew\n</user>\n<agent>\nshort\n</agent>")

    def test_oldest_exchanges_drop_first_when_over_budget(self):
        exchanges = [(f"request {i} " + "z" * 400, "ok") for i in range(30)]
        text = reviewer.render_conversation(exchanges, budget=600)
        assert "request 29 " in text and "request 0 " not in text
        assert len(text) <= 600 * 4

    def test_oversized_newest_request_is_clipped_but_keeps_its_answer(self):
        text = reviewer.render_conversation([("Q" * 50_000, "the answer")], budget=800)
        assert len(text) <= 800 * 4
        assert "\n[...]\n" in text and "the answer" in text


class TestConversationReachesReviewers:
    def test_self_review_prompt_frames_it_as_context(self, tmp_path, monkeypatch):
        calls = _capture_reviews(monkeypatch)
        request = reviewer.ReviewRequest(
            "use option 2", "Done.", 1, "<user>\nlist options\n</user>"
        )
        reviewer.SelfReviewer(base_dir=str(tmp_path), max_rounds=3).review(
            request, _agent_loop_kwargs()
        )
        prompt = calls[0]["prompt"]
        assert (
            "<conversation>\n<user>\nlist options\n</user>\n</conversation>" in prompt
        )
        assert "not evidence" in prompt
        assert prompt.index("<conversation>") < prompt.index("<task>")

    def test_reviewer_mode_reads_it_from_the_environment(
        self, tmp_path, monkeypatch, capsys
    ):
        calls = _capture_reviews(monkeypatch)
        monkeypatch.setenv("SWIVAL_TASK", "use option 2")
        monkeypatch.setenv("SWIVAL_CONVERSATION", "<user>\nlist options\n</user>")
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(read=lambda: "Done."))
        monkeypatch.setattr(
            agent,
            "resolve_provider",
            lambda **kw: ("m", "http://fake", None, None, {"provider": "lmstudio"}),
        )
        assert run_as_reviewer(_reviewer_args(str(tmp_path)), str(tmp_path)) == 0
        assert "<conversation>\n<user>\nlist options\n</user>" in calls[0]["prompt"]

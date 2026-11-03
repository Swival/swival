"""Tests for answer review inside the shared agent step."""

import copy
import functools
import types

import litellm
import pytest

from conftest import build_loop_kwargs
from swival import agent
from swival._msg import _msg_content, _msg_role
from swival.continue_here import CONTINUE_PATH
from swival.input_dispatch import InputContext, parse_input_line
from swival.loops import LoopRegistry
from swival.report import AgentError, ReportCollector
from swival.reviewer import ReviewVerdict, SelfReviewer
from swival.thinking import ThinkingState
from swival.todo import TodoState

ACCEPT = ReviewVerdict("accept", text="VERDICT: ACCEPT")


def _retry(feedback="add a test"):
    return ReviewVerdict("retry", text=f"{feedback}\nVERDICT: RETRY")


class _Agent:
    """Scripted stand-in for run_agent_loop.

    Each entry is ``(answer, exhausted)``, an exception to raise, or a callable
    that gets the messages and returns ``(answer, exhausted)``.
    With a *report*, each pass bumps its turn count, like real LLM calls do.
    """

    def __init__(self, *script, report=None):
        self.script = list(script)
        self.report = report
        self.requests = []
        self.offsets = []

    def __call__(self, messages, tools, **kwargs):
        self.requests.append(_msg_content(messages[-1]))
        self.offsets.append(kwargs.get("turn_offset"))
        if self.report is not None:
            self.report.max_turn_seen += 5
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        if callable(step):
            return step(messages)
        answer, exhausted = step
        if answer is not None:
            messages.append({"role": "assistant", "content": answer})
        return answer, exhausted


class _Reviewer:
    def __init__(self, *verdicts, max_rounds=15):
        self.verdicts = list(verdicts)
        self.max_rounds = max_rounds
        self.requests = []
        self.loop_kwargs = []

    def review(self, request, loop_kwargs):
        self.requests.append(request)
        self.loop_kwargs.append(loop_kwargs)
        verdict = self.verdicts.pop(0)
        if isinstance(verdict, BaseException):
            raise verdict
        return verdict


def _ctx(tmp_path, reviewer=None, **overrides):
    kwargs = dict(
        messages=[{"role": "system", "content": "sys"}],
        tools=[],
        base_dir=str(tmp_path),
        turn_state={"max_turns": 10, "turns_used": 0},
        thinking_state=ThinkingState(),
        todo_state=TodoState(),
        snapshot_state=None,
        file_tracker=None,
        no_history=False,
        continue_here=True,
        verbose=False,
        loop_kwargs={"model_id": "m1", "llm_kwargs": {}},
        reviewer=reviewer,
    )
    kwargs.update(overrides)
    return InputContext(**kwargs)


def _real_ctx(tmp_path, reviewer=None):
    """A context that runs the real agent loop and call_llm()."""
    loop_kwargs = build_loop_kwargs(tmp_path)
    loop_kwargs.pop("max_turns")
    loop_kwargs["llm_kwargs"] = {
        "provider": "generic",
        "api_key": "x",
        "max_retries": 0,
    }
    return _ctx(
        tmp_path,
        reviewer,
        thinking_state=loop_kwargs["thinking_state"],
        todo_state=loop_kwargs["todo_state"],
        loop_kwargs=loop_kwargs,
        no_history=True,
    )


_BAD_REQUEST = litellm.BadRequestError(
    message="bad input", llm_provider="openai", model="m"
)


def _stub_provider(monkeypatch, *replies):
    """Stub litellm.completion and return the last message text of each request."""
    sent = []

    def completion(**kwargs):
        sent.append(_msg_content(kwargs["messages"][-1]))
        reply = replies[len(sent) - 1]
        if isinstance(reply, BaseException):
            raise reply
        msg = types.SimpleNamespace(content=reply, tool_calls=None, role="assistant")
        choice = types.SimpleNamespace(message=msg, finish_reason="stop")
        return types.SimpleNamespace(choices=[choice])

    monkeypatch.setattr(litellm, "completion", completion)
    return sent


def _step(ctx, content="fix the bug", review_task="fix the bug"):
    return agent._run_agent_step(content, content, ctx, review_task=review_task)


def _run(ctx, line):
    return agent.execute_input(parse_input_line(line), ctx, mode="repl")


def _history(tmp_path):
    path = tmp_path / ".swival" / "HISTORY.md"
    return path.read_text() if path.exists() else ""


def _continue_file(tmp_path):
    return tmp_path / CONTINUE_PATH


def _progress_then_fail(messages, nudge=False):
    messages.append({"role": "assistant", "content": "working"})
    if nudge:
        messages.append(
            {"role": "user", "content": "Reminder: x", "_swival_synthetic": True}
        )
    raise AgentError("down")


@pytest.fixture
def warnings(monkeypatch):
    seen = []
    monkeypatch.setattr(agent.fmt, "warning", seen.append)
    monkeypatch.setattr(agent.fmt, "error", lambda text: None)
    return seen


class TestReviewLoop:
    def test_without_a_reviewer_nothing_is_reviewed_or_kept(
        self, tmp_path, monkeypatch
    ):
        fake = _Agent(("done", False))
        monkeypatch.setattr(agent, "run_agent_loop", fake)
        ctx = _ctx(tmp_path)
        assert _step(ctx).text == "done"
        assert fake.requests == ["fix the bug"]
        assert (ctx.review.task, ctx.review.exchanges) == (None, [])

    def test_accepted_answer_is_reviewed_once_on_the_live_kwargs(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(agent, "run_agent_loop", _Agent(("done", False)))
        reviewer = _Reviewer(ACCEPT)
        ctx = _ctx(tmp_path, reviewer)
        assert _step(ctx).text == "done"
        [request] = reviewer.requests
        assert (request.task, request.answer, request.review_round) == (
            "fix the bug",
            "done",
            1,
        )
        assert reviewer.loop_kwargs[0] is ctx.loop_kwargs
        assert _history(tmp_path).count("done") == 1

    def test_retry_runs_another_pass_on_the_feedback(self, tmp_path, monkeypatch):
        report = ReportCollector()
        fake = _Agent(("first", False), ("second", False), report=report)
        monkeypatch.setattr(agent, "run_agent_loop", fake)
        reviewer = _Reviewer(_retry("cover the empty case"), ACCEPT)
        ctx = _ctx(tmp_path, reviewer)
        ctx.loop_kwargs["report"] = report

        assert _step(ctx).text == "second"
        feedback = fake.requests[1]
        assert feedback.startswith("[REVIEWER FEEDBACK, Round 1]\n")
        assert "cover the empty case" in feedback
        assert [(r.task, r.answer) for r in reviewer.requests] == [
            ("fix the bug", "first"),
            ("fix the bug", "second"),
        ]
        assert ctx.review.task == "fix the bug"
        history = _history(tmp_path)
        assert "second" in history and "first" not in history
        events = [e for e in report.events if e["type"] == "review"]
        assert [(e["round"], e["exit_code"]) for e in events] == [(1, 1), (2, 0)]
        # The retry numbers its turns after the first pass.
        assert fake.offsets == [None, 5]

    @pytest.mark.parametrize("max_rounds, reviews", [(0, 1), (1, 1), (2, 2)])
    def test_max_rounds_caps_retries(self, tmp_path, monkeypatch, max_rounds, reviews):
        fake = _Agent(*[(f"answer {i}", False) for i in range(5)])
        reviewer = _Reviewer(*[_retry()] * 5, max_rounds=max_rounds)
        monkeypatch.setattr(agent, "run_agent_loop", fake)
        result = _step(_ctx(tmp_path, reviewer))
        assert len(reviewer.requests) == len(fake.requests) == reviews
        assert result.text == f"answer {reviews - 1}"

    @pytest.mark.parametrize("contents", [None, b"\xff"], ids=["missing", "not-utf8"])
    def test_failed_review_keeps_the_answer_and_its_reason(
        self, tmp_path, monkeypatch, contents
    ):
        if contents is not None:
            (tmp_path / "criteria.md").write_bytes(contents)
        report = ReportCollector()
        fake = _Agent(("done", False))
        monkeypatch.setattr(agent, "run_agent_loop", fake)
        ctx = _ctx(
            tmp_path,
            SelfReviewer(base_dir=str(tmp_path), max_rounds=3, verify="criteria.md"),
        )
        ctx.loop_kwargs["report"] = report
        assert _step(ctx).text == "done"
        assert len(fake.requests) == 1
        [event] = [e for e in report.events if e["type"] == "review"]
        assert event["exit_code"] == 2
        assert event["error"].startswith("cannot read --verify file")

    def test_exhausted_answer_is_not_reviewed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(agent, "run_agent_loop", _Agent(("partial", True)))
        reviewer = _Reviewer()
        ctx = _ctx(tmp_path, reviewer)
        assert _step(ctx).exhausted
        assert reviewer.requests == []
        assert ctx.review.task == "fix the bug"

    @pytest.mark.parametrize(
        "retry, flag",
        [(("partial", True), "exhausted"), (KeyboardInterrupt(), "interrupted")],
    )
    def test_unfinished_retry_keeps_the_original_task(
        self, tmp_path, monkeypatch, retry, flag
    ):
        monkeypatch.setattr(agent, "run_agent_loop", _Agent(("first", False), retry))
        ctx = _ctx(tmp_path, _Reviewer(_retry()))
        assert getattr(_step(ctx), flag)
        assert ctx.review.task == "fix the bug"

    def test_unreviewed_turn_clears_the_task(self, tmp_path, monkeypatch):
        fake = _Agent(("done", False), ("learned", False))
        monkeypatch.setattr(agent, "run_agent_loop", fake)
        ctx = _ctx(tmp_path, _Reviewer(ACCEPT))
        _step(ctx)
        _step(ctx, content="/learn prompt", review_task=None)
        assert ctx.review.task is None


class TestInterruptsAndTermination:
    def test_ctrl_c_during_review_keeps_the_answer_in_the_repl(
        self, tmp_path, monkeypatch, warnings
    ):
        fake = _Agent(("done", False))
        monkeypatch.setattr(agent, "run_agent_loop", fake)
        result = _step(_ctx(tmp_path, _Reviewer(KeyboardInterrupt())))
        assert (result.text, result.interrupted) == ("done", False)
        assert len(fake.requests) == 1
        assert any("not reviewed" in w for w in warnings)

    def test_ctrl_c_during_review_interrupts_a_non_interactive_run(
        self, tmp_path, monkeypatch, warnings
    ):
        monkeypatch.setattr(agent, "run_agent_loop", _Agent(("done", False)))
        ctx = _ctx(tmp_path, _Reviewer(KeyboardInterrupt()), interactive=False)
        result = _step(ctx)
        assert (result.text, result.interrupted) == (None, True)
        assert _continue_file(tmp_path).exists()

    def test_sigterm_during_review_saves_and_propagates(self, tmp_path, monkeypatch):
        monkeypatch.setattr(agent, "run_agent_loop", _Agent(("done", False)))
        with pytest.raises(SystemExit):
            _step(_ctx(tmp_path, _Reviewer(SystemExit(143))))
        assert _continue_file(tmp_path).exists()

    @pytest.mark.parametrize(
        "interactive, continue_here", [(True, True), (False, True), (True, False)]
    )
    def test_sigterm_during_an_agent_pass_saves_and_propagates(
        self, tmp_path, monkeypatch, interactive, continue_here
    ):
        monkeypatch.setattr(agent, "run_agent_loop", _Agent(SystemExit(143)))
        ctx = _ctx(tmp_path, interactive=interactive, continue_here=continue_here)
        with pytest.raises(SystemExit):
            _step(ctx, review_task=None)
        assert _continue_file(tmp_path).exists() == continue_here

    def test_ctrl_c_during_an_instruction_refresh_keeps_the_task(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            agent, "run_agent_loop", _Agent(("answer A", False), ("A done", False))
        )
        reviewer = _Reviewer(ACCEPT, ACCEPT)
        ctx = _ctx(tmp_path, reviewer)
        _run(ctx, "request A")

        def interrupted_refresh(ctx):
            raise KeyboardInterrupt

        ctx.pending_instruction_failure = "AGENTS.md is too large"
        monkeypatch.setattr(agent, "refresh_system_instructions", interrupted_refresh)
        with pytest.raises(KeyboardInterrupt):
            _run(ctx, "request B")
        assert ctx.review.task == "request A"
        assert not any(_msg_content(m) == "request B" for m in ctx.messages)

        ctx.pending_instruction_failure = None
        _run(ctx, "/continue")
        assert reviewer.requests[-1].task == "request A"

    def test_second_ctrl_c_during_subagent_shutdown_keeps_the_continue_file(
        self, tmp_path, monkeypatch, warnings
    ):
        class StuckSubagents:
            def shutdown(self):
                raise KeyboardInterrupt

            def fresh_copy(self):
                return StuckSubagents()

        monkeypatch.setattr(agent, "run_agent_loop", _Agent(KeyboardInterrupt()))
        ctx = _ctx(tmp_path, subagent_manager=StuckSubagents())
        with pytest.raises(KeyboardInterrupt):
            _step(ctx)
        assert _continue_file(tmp_path).exists()


class TestAgentErrorRollback:
    def test_escaped_request_rolls_back_through_the_real_turn_path(
        self, tmp_path, monkeypatch, warnings
    ):
        sent = _stub_provider(monkeypatch, _BAD_REQUEST)
        ctx = _real_ctx(tmp_path, _Reviewer())
        ctx.review.task = "request A"
        _step(ctx, "Explain <|eot_id|>", "Explain <|eot_id|>")
        assert sent[0] != "Explain <|eot_id|>"
        assert ctx.review.task == "request A"
        assert [_msg_role(m) for m in ctx.messages] == ["system"]

    def test_compaction_copies_keep_the_request_identifiable(
        self, tmp_path, monkeypatch, warnings
    ):
        def compact_then_fail(messages):
            messages[:] = copy.deepcopy(messages)
            raise AgentError("down")

        monkeypatch.setattr(agent, "run_agent_loop", _Agent(compact_then_fail))
        ctx = _ctx(tmp_path, _Reviewer())
        ctx.review.task = "request A"
        _step(ctx, "B", "B")
        assert ctx.review.task == "request A"

    @pytest.mark.parametrize("nudge", [False, True])
    def test_error_after_progress_keeps_the_task(
        self, tmp_path, monkeypatch, warnings, nudge
    ):
        fail = functools.partial(_progress_then_fail, nudge=nudge)
        monkeypatch.setattr(agent, "run_agent_loop", _Agent(fail))
        ctx = _ctx(tmp_path, _Reviewer())
        ctx.review.task = "request A"
        _step(ctx, "request B", "request B")
        assert ctx.review.task == "request B"
        assert any(_msg_content(m) == "request B" for m in ctx.messages)
        # Only the trailing nudge was removed, not the progress.
        assert _msg_role(ctx.messages[-1]) == "assistant"

    def test_failed_retry_keeps_the_rejected_answer(
        self, tmp_path, monkeypatch, warnings
    ):
        sent = _stub_provider(monkeypatch, "first", _BAD_REQUEST)
        ctx = _real_ctx(tmp_path, _Reviewer(_retry("drop the <|eot_id|> token")))
        assert _step(ctx).text == "first"
        assert "<|eot_id|>" not in sent[1]
        assert _msg_content(ctx.messages[-1]) == "first"
        assert ctx.review.task == "fix the bug"
        assert any("reviewer rejected" in w for w in warnings)

    @pytest.mark.parametrize("outcome", ["answer", "progress-then-error"])
    def test_request_marker_is_gone_after_the_turn(
        self, tmp_path, monkeypatch, warnings, outcome
    ):
        step = ("done", False) if outcome == "answer" else _progress_then_fail
        monkeypatch.setattr(agent, "run_agent_loop", _Agent(step))
        ctx = _ctx(tmp_path)
        agent._invoke_agent_turn("B", ctx)
        assert {"role": "user", "content": "B"} in ctx.messages
        assert not any("_swival_request" in m for m in ctx.messages)

    def test_non_interactive_error_still_propagates(self, tmp_path, monkeypatch):
        monkeypatch.setattr(agent, "run_agent_loop", _Agent(AgentError("down")))
        with pytest.raises(AgentError):
            _step(_ctx(tmp_path, interactive=False))


class TestConversationContext:
    def test_follow_up_review_sees_the_earlier_exchange(self, tmp_path, monkeypatch):
        fake = _Agent(("Options: 1) retry 2) give up", False), ("Gave up.", False))
        monkeypatch.setattr(agent, "run_agent_loop", fake)
        reviewer = _Reviewer(ACCEPT, ACCEPT)
        ctx = _ctx(tmp_path, reviewer)
        _run(ctx, "how should I handle the timeout?")
        _run(ctx, "use option 2")

        first, second = reviewer.requests
        assert first.conversation is None
        assert second.task == "use option 2"
        assert "how should I handle the timeout?" in second.conversation
        assert "Options: 1) retry 2) give up" in second.conversation
        assert "use option 2" not in second.conversation

    def test_continue_replaces_the_unfinished_exchange(self, tmp_path, monkeypatch):
        fake = _Agent(("one", False), ("partial", True), ("finished", False))
        monkeypatch.setattr(agent, "run_agent_loop", fake)
        reviewer = _Reviewer(ACCEPT, ACCEPT)
        ctx = _ctx(tmp_path, reviewer)
        _run(ctx, "first task")
        _run(ctx, "second task")
        assert ctx.review.exchanges[-1] == ("second task", "partial")

        _run(ctx, "/continue")
        assert ctx.review.exchanges == [
            ("first task", "one"),
            ("second task", "finished"),
        ]
        continued = reviewer.requests[-1]
        assert continued.task == "second task"
        assert "first task" in continued.conversation
        assert "second task" not in continued.conversation

    def test_failed_continue_keeps_the_answer_for_the_next_follow_up(
        self, tmp_path, monkeypatch, warnings
    ):
        fake = _Agent(
            ("Options: 1) retry 2) give up", False),
            AgentError("down"),
            ("Gave up.", False),
        )
        monkeypatch.setattr(agent, "run_agent_loop", fake)
        reviewer = _Reviewer(ACCEPT, ACCEPT)
        ctx = _ctx(tmp_path, reviewer)
        _run(ctx, "how should I handle the timeout?")
        _run(ctx, "/continue")
        _run(ctx, "use option 2")

        follow_up = reviewer.requests[-1]
        assert follow_up.task == "use option 2"
        assert "Options: 1) retry 2) give up" in follow_up.conversation
        assert "(no answer)" not in follow_up.conversation

    def test_rolled_back_request_is_not_recorded(self, tmp_path, monkeypatch, warnings):
        fake = _Agent(("one", False), AgentError("down"))
        monkeypatch.setattr(agent, "run_agent_loop", fake)
        ctx = _ctx(tmp_path, _Reviewer(ACCEPT))
        _run(ctx, "first task")
        _run(ctx, "second task")
        assert ctx.review.exchanges == [("first task", "one")]
        assert ctx.review.task == "first task"

    def test_bang_command_is_reviewed_against_its_expanded_prompt(
        self, tmp_path, monkeypatch
    ):
        expanded = "Fix the flaky test in tests/test_io.py."
        monkeypatch.setattr(
            agent,
            "_repl_run_custom_command",
            lambda *a, **k: ("fixit", expanded, None),
        )
        monkeypatch.setattr(agent.fmt, "info", lambda text: None)
        monkeypatch.setattr(agent, "run_agent_loop", _Agent(("fixed", False)))
        reviewer = _Reviewer(ACCEPT)
        ctx = _ctx(tmp_path, reviewer)
        _run(ctx, "!fixit tests/test_io.py")
        assert reviewer.requests[0].task == expanded
        assert ctx.review.exchanges == [(expanded, "fixed")]

    @pytest.mark.parametrize("line", ["/clear", "/init"])
    def test_clear_and_init_reset_review_state(self, tmp_path, monkeypatch, line):
        monkeypatch.setattr(
            agent, "run_agent_loop", _Agent(*[("init pass", False)] * 5)
        )
        ctx = _ctx(tmp_path, _Reviewer(ACCEPT))
        ctx.review.record("a task", "done", continued=False)
        ctx.review.task = "a task"
        _run(ctx, line)
        assert (ctx.review.task, ctx.review.exchanges) == (None, [])


class TestIsolation:
    @pytest.mark.parametrize("kind", ["loop", "btw"])
    def test_isolated_context_has_no_reviewer_and_its_own_state(self, tmp_path, kind):
        ctx = _ctx(tmp_path, _Reviewer(ACCEPT))
        iso = agent._build_isolated_ctx(ctx, kind)
        assert iso.reviewer is None
        assert iso.review is not ctx.review

    def test_background_loop_leaves_parent_review_state_alone(
        self, tmp_path, monkeypatch
    ):
        fake = _Agent(("tick", False))
        monkeypatch.setattr(agent, "run_agent_loop", fake)
        monkeypatch.setattr(agent.fmt, "repl_answer", lambda text: None)
        reviewer = _Reviewer()
        registry = LoopRegistry()
        ctx = _ctx(tmp_path, reviewer, loop_registry=registry)
        ctx.review.record("parent task", "parent answer", continued=False)
        ctx.review.task = "parent task"
        review = ctx.review

        agent._execute_loop("5m check the build", ctx, mode="repl")

        # The iteration actually ran.
        assert fake.requests == ["check the build"]
        [registration] = list(registry)
        assert (registration.consecutive_failures, registration.last_error) == (0, None)
        assert ctx.review is review
        assert ctx.review.task == "parent task"
        assert ctx.review.exchanges == [("parent task", "parent answer")]
        assert reviewer.requests == []

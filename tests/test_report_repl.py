"""Tests for REPL report support (--report combined with --repl)."""

import pytest

from swival import agent
from swival.input_dispatch import StepResult
from swival.report import AgentError, ReportCollector
from swival.thinking import ThinkingState
from swival.todo import TodoState


# ---------------------------------------------------------------------------
# ReportCollector unit tests
# ---------------------------------------------------------------------------


class TestRecordReplTurn:
    def test_basic(self):
        rc = ReportCollector()
        rc.record_repl_turn("hello world")
        assert len(rc.events) == 1
        ev = rc.events[0]
        assert ev["type"] == "repl_turn"
        assert ev["turn_offset"] == 0
        assert ev["input"] == "hello world"

    def test_truncates_long_input(self):
        rc = ReportCollector()
        rc.record_repl_turn("x" * 600)
        assert len(rc.events[0]["input"]) == 500

    def test_turn_offset_reflects_max_turn_seen(self):
        rc = ReportCollector()
        rc.record_llm_call(3, 0.1, 100, "stop")
        rc.record_repl_turn("second turn")
        assert rc.events[1]["turn_offset"] == 3


class TestRecordSessionClear:
    def test_basic(self):
        rc = ReportCollector()
        rc.record_session_clear()
        assert len(rc.events) == 1
        assert rc.events[0] == {"type": "session_clear"}


class TestModeField:
    def _build(self, mode):
        rc = ReportCollector()
        return rc.build_report(
            task="t",
            model="m",
            provider="p",
            settings={},
            outcome="success",
            answer=None,
            exit_code=0,
            turns=0,
            mode=mode,
        )

    def test_default_is_oneshot(self):
        rc = ReportCollector()
        r = rc.build_report(
            task="t",
            model="m",
            provider="p",
            settings={},
            outcome="success",
            answer=None,
            exit_code=0,
            turns=0,
        )
        assert r["mode"] == "oneshot"

    def test_repl_mode(self):
        r = self._build("repl")
        assert r["mode"] == "repl"

    def test_finalize_passes_mode(self):
        rc = ReportCollector()
        r = rc.finalize(
            task="t",
            model="m",
            provider="p",
            settings={},
            outcome="success",
            answer=None,
            exit_code=0,
            turns=0,
            mode="repl",
        )
        assert r["mode"] == "repl"
        assert rc._last_report["mode"] == "repl"


class TestMultiTurnAccumulation:
    def test_no_overlapping_turns_with_offset(self):
        rc = ReportCollector()
        rc.record_llm_call(1, 0.1, 100, "stop")
        rc.record_llm_call(2, 0.1, 100, "stop")
        assert rc.max_turn_seen == 2

        rc.record_repl_turn("second question")
        rc.record_llm_call(rc.max_turn_seen + 1, 0.1, 100, "stop")
        rc.record_llm_call(rc.max_turn_seen + 1, 0.1, 100, "stop")
        assert rc.max_turn_seen == 4

        turn_numbers = [e["turn"] for e in rc.events if e["type"] == "llm_call"]
        assert turn_numbers == [1, 2, 3, 4]
        assert len(turn_numbers) == len(set(turn_numbers))

    def test_session_clear_does_not_reset_counters(self):
        rc = ReportCollector()
        rc.record_llm_call(1, 0.5, 100, "stop")
        rc.record_tool_call(1, "read_file", {}, True, 0.1, 50)
        rc.record_session_clear()
        rc.record_llm_call(2, 0.3, 100, "stop")

        assert rc.llm_calls == 2
        assert rc.tool_stats["read_file"]["succeeded"] == 1
        r = rc.build_report(
            task="repl session (2 turns)",
            model="m",
            provider="p",
            settings={},
            outcome="success",
            answer=None,
            exit_code=0,
            turns=2,
            mode="repl",
        )
        assert r["stats"]["llm_calls"] == 2
        assert r["stats"]["tool_calls_total"] == 1
        clear_events = [e for e in r["timeline"] if e["type"] == "session_clear"]
        assert len(clear_events) == 1


# ---------------------------------------------------------------------------
# repl_loop() integration tests
# ---------------------------------------------------------------------------

_REPL_LOOP_DEFAULTS = dict(
    api_base="http://localhost",
    model_id="test-model",
    max_turns=10,
    max_output_tokens=None,
    temperature=0.0,
    top_p=None,
    seed=None,
    context_length=4096,
    thinking_state=ThinkingState(),
    todo_state=TodoState(),
    snapshot_state=None,
    resolved_commands={},
    skills_catalog={},
    skill_read_roots=[],
    extra_write_roots=[],
    files_mode="some",
    verbose=False,
    llm_kwargs={},
)


def _patch_repl_loop(monkeypatch, inputs):
    """Patch prompt_toolkit so repl_loop() reads from a list of inputs.

    Each entry in *inputs* is returned by session.prompt(). After the list is
    exhausted, EOFError is raised (simulating Ctrl-D).

    Also patches the SwivalCompleter import away to avoid pulling in the
    full completer.
    """
    it = iter(inputs)

    class FakeSession:
        def __init__(self, **kw):
            pass

        def prompt(self, *a, **kw):
            try:
                return next(it)
            except StopIteration:
                raise EOFError

    class FakeStyle:
        @staticmethod
        def from_dict(d):
            return None

    class FakeFormattedText:
        def __init__(self, *a, **kw):
            pass

    class FakeFileHistory:
        def __init__(self, *a, **kw):
            pass

        def store_string(self, s):
            pass

    class FakeCompleter:
        def __init__(self, **kw):
            pass

    monkeypatch.setattr(
        "prompt_toolkit.PromptSession",
        FakeSession,
    )
    monkeypatch.setattr(
        "prompt_toolkit.formatted_text.FormattedText",
        FakeFormattedText,
    )
    monkeypatch.setattr(
        "prompt_toolkit.history.FileHistory",
        FakeFileHistory,
    )
    monkeypatch.setattr(
        "prompt_toolkit.styles.Style",
        FakeStyle,
    )
    monkeypatch.setattr(
        "swival.completer.SwivalCompleter",
        FakeCompleter,
    )


class TestReplLoopOnExit:
    """Exercise on_exit through the real repl_loop() code path."""

    def test_clean_exit_eof(self, monkeypatch, tmp_path):
        """EOF after one agent turn calls on_exit with success."""
        report = ReportCollector()
        exit_args = {}

        def on_exit(outcome, exit_code):
            exit_args["outcome"] = outcome
            exit_args["exit_code"] = exit_code

        _patch_repl_loop(monkeypatch, ["hello"])

        monkeypatch.setattr(
            agent,
            "execute_input",
            lambda parsed, ctx, mode="repl": StepResult(
                kind="agent_turn", text="answer"
            ),
        )

        monkeypatch.setattr(agent.fmt, "repl_answer", lambda *a, **kw: None)
        monkeypatch.setattr(agent.fmt, "reset_state", lambda: None)

        agent.repl_loop(
            messages=[{"role": "system", "content": "sys"}],
            tools=[],
            **_REPL_LOOP_DEFAULTS,
            base_dir=str(tmp_path),
            report=report,
            on_exit=on_exit,
        )

        assert exit_args == {"outcome": "success", "exit_code": 0}
        repl_turns = [e for e in report.events if e["type"] == "repl_turn"]
        assert len(repl_turns) == 1
        assert repl_turns[0]["input"] == "hello"

    def test_clean_exit_quit(self, monkeypatch, tmp_path):
        """/quit calls on_exit with success."""
        report = ReportCollector()
        exit_args = {}

        def on_exit(outcome, exit_code):
            exit_args["outcome"] = outcome
            exit_args["exit_code"] = exit_code

        _patch_repl_loop(monkeypatch, ["/quit"])

        monkeypatch.setattr(
            agent,
            "execute_input",
            lambda parsed, ctx, mode="repl": StepResult(kind="flow_control", stop=True),
        )

        monkeypatch.setattr(agent.fmt, "reset_state", lambda: None)

        agent.repl_loop(
            messages=[{"role": "system", "content": "sys"}],
            tools=[],
            **_REPL_LOOP_DEFAULTS,
            base_dir=str(tmp_path),
            report=report,
            on_exit=on_exit,
        )

        assert exit_args == {"outcome": "success", "exit_code": 0}

    def test_error_exit(self, monkeypatch, tmp_path):
        """Exception from execute_input calls on_exit with error."""
        report = ReportCollector()
        exit_args = {}

        def on_exit(outcome, exit_code):
            exit_args["outcome"] = outcome
            exit_args["exit_code"] = exit_code

        _patch_repl_loop(monkeypatch, ["boom"])

        def _exploding_execute(parsed, ctx, mode="repl"):
            raise AgentError("boom")

        monkeypatch.setattr(agent, "execute_input", _exploding_execute)
        monkeypatch.setattr(agent.fmt, "reset_state", lambda: None)

        with pytest.raises(AgentError):
            agent.repl_loop(
                messages=[{"role": "system", "content": "sys"}],
                tools=[],
                **_REPL_LOOP_DEFAULTS,
                base_dir=str(tmp_path),
                report=report,
                on_exit=on_exit,
            )

        assert exit_args == {"outcome": "error", "exit_code": 1}

    def test_turn_offset_advances(self, monkeypatch, tmp_path):
        """turn_offset in loop_kwargs advances after each agent turn."""
        report = ReportCollector()
        report.record_llm_call(1, 0.1, 100, "stop")
        offsets_seen = []

        _patch_repl_loop(monkeypatch, ["q1", "q2"])

        def _tracking_execute(parsed, ctx, mode="repl"):
            offsets_seen.append(ctx.loop_kwargs.get("turn_offset", 0))
            report.record_llm_call(
                ctx.loop_kwargs.get("turn_offset", 0) + 1,
                0.1,
                100,
                "stop",
            )
            return StepResult(kind="agent_turn", text="ok")

        monkeypatch.setattr(agent, "execute_input", _tracking_execute)
        monkeypatch.setattr(agent.fmt, "repl_answer", lambda *a, **kw: None)
        monkeypatch.setattr(agent.fmt, "reset_state", lambda: None)

        agent.repl_loop(
            messages=[{"role": "system", "content": "sys"}],
            tools=[],
            **_REPL_LOOP_DEFAULTS,
            base_dir=str(tmp_path),
            report=report,
            turn_offset=0,
            on_exit=lambda *a: None,
        )

        assert offsets_seen[0] == 0
        assert offsets_seen[1] > offsets_seen[0]

    def test_report_passed_in_loop_kwargs(self, monkeypatch, tmp_path):
        """report is available in ctx.loop_kwargs for run_agent_loop."""
        report = ReportCollector()
        captured_report = []

        _patch_repl_loop(monkeypatch, ["test"])

        def _capturing_execute(parsed, ctx, mode="repl"):
            captured_report.append(ctx.loop_kwargs.get("report"))
            return StepResult(kind="agent_turn", text="ok")

        monkeypatch.setattr(agent, "execute_input", _capturing_execute)
        monkeypatch.setattr(agent.fmt, "repl_answer", lambda *a, **kw: None)
        monkeypatch.setattr(agent.fmt, "reset_state", lambda: None)

        agent.repl_loop(
            messages=[{"role": "system", "content": "sys"}],
            tools=[],
            **_REPL_LOOP_DEFAULTS,
            base_dir=str(tmp_path),
            report=report,
            on_exit=lambda *a: None,
        )

        assert captured_report[0] is report

    def test_double_write_guard(self):
        """main() outer handler skips _write_report when report already finalized."""
        rc = ReportCollector()
        assert not rc.is_finalized
        rc.finalize(
            task="repl session (1 turns)",
            model="m",
            provider="p",
            settings={},
            outcome="error",
            answer=None,
            exit_code=1,
            turns=1,
            mode="repl",
        )
        assert rc.is_finalized
        wrote_fallback = not rc or not rc.is_finalized
        assert not wrote_fallback


class TestArgparseReplReport:
    """Verify --report + --repl no longer raises an error."""

    def test_report_repl_accepted(self, tmp_path):
        from swival.agent import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "--repl",
                "--report",
                str(tmp_path / "report.json"),
                "--provider",
                "generic",
                "--model",
                "test",
            ]
        )
        assert args.repl is True
        assert args.report == str(tmp_path / "report.json")

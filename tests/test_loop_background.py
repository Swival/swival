"""Tests for the background, snapshot-isolated /loop REPL behaviour."""

from __future__ import annotations

import types

from swival import agent, loops as loops_mod
from swival.input_dispatch import InputContext, StepResult
from swival.loops import (
    CANCEL_FAILURES,
    LoopRegistry,
    MAX_ACTIVE_LOOPS,
    WARN_FAILURES,
)
from swival.thinking import ThinkingState
from swival.todo import TodoState


def _make_ctx(*, loop_registry: LoopRegistry | None = None) -> InputContext:
    return InputContext(
        messages=[{"role": "system", "content": "sys"}],
        tools=[],
        base_dir="/tmp",
        turn_state={"max_turns": 10, "turns_used": 0},
        thinking_state=ThinkingState(),
        todo_state=TodoState(),
        snapshot_state=None,
        file_tracker=None,
        no_history=False,
        continue_here=True,
        verbose=False,
        loop_kwargs={
            "model_id": "test",
            "api_base": "http://test",
            "context_length": 128000,
            "files_mode": "some",
            "compaction_state": None,
            "command_policy": types.SimpleNamespace(mode="allowlist"),
            "top_p": None,
            "seed": None,
            "llm_kwargs": {},
        },
        loop_registry=loop_registry,
    )


class _CapturedClock:
    def __init__(self, start: float = 0.0):
        self.t = start

    def advance(self, seconds: float) -> None:
        self.t += seconds

    def __call__(self) -> float:
        return self.t


def _patch_clock(monkeypatch, clock: _CapturedClock) -> None:
    monkeypatch.setattr(loops_mod, "monotonic", clock)


def _ok_step(text: str = "ok") -> StepResult:
    return StepResult(kind="agent_turn", text=text)


def _err_step(text: str = "error: kaboom") -> StepResult:
    return StepResult(kind="agent_turn", text=text, is_error=True)


class TestRegistration:
    def test_first_iteration_runs_immediately(self, monkeypatch):
        registry = LoopRegistry()
        ctx = _make_ctx(loop_registry=registry)

        calls: list[str] = []

        def fake_execute_input(parsed, ctx_arg, *, mode="repl"):
            calls.append(parsed.raw)
            return _ok_step("answer")

        monkeypatch.setattr(agent, "execute_input", fake_execute_input)
        monkeypatch.setattr(agent.fmt, "repl_answer", lambda t: None)

        result = agent._execute_loop("5m foo", ctx, mode="repl")
        assert result.kind == "state_change"
        assert len(registry) == 1
        assert calls == ["foo"]
        reg = list(registry)[0]
        assert reg.interval_seconds == 300
        assert reg.consecutive_failures == 0

    def test_capacity_cap(self, monkeypatch):
        registry = LoopRegistry()
        ctx = _make_ctx(loop_registry=registry)
        monkeypatch.setattr(agent, "execute_input", lambda *a, **k: _ok_step())
        monkeypatch.setattr(agent.fmt, "repl_answer", lambda t: None)

        for i in range(MAX_ACTIVE_LOOPS):
            r = agent._execute_loop(f"1m foo{i}", ctx, mode="repl")
            assert not r.is_error
        result = agent._execute_loop("1m overflow", ctx, mode="repl")
        assert result.is_error
        assert "active loops" in (result.text or "")
        assert len(registry) == MAX_ACTIVE_LOOPS


class TestSnapshotIsolation:
    def test_live_messages_unchanged(self, monkeypatch):
        registry = LoopRegistry()
        ctx = _make_ctx(loop_registry=registry)
        ctx.messages.append({"role": "user", "content": "live marker"})

        def fake_execute_input(parsed, ctx_arg, *, mode="repl"):
            ctx_arg.messages.append({"role": "user", "content": "iter marker"})
            return _ok_step()

        monkeypatch.setattr(agent, "execute_input", fake_execute_input)
        monkeypatch.setattr(agent.fmt, "repl_answer", lambda t: None)

        agent._execute_loop("5m foo", ctx, mode="repl")

        contents = [m.get("content") for m in ctx.messages]
        assert "live marker" in contents
        assert "iter marker" not in contents

    def test_live_todo_thinking_unchanged(self, monkeypatch):
        registry = LoopRegistry()
        ctx = _make_ctx(loop_registry=registry)

        def fake_execute_input(parsed, ctx_arg, *, mode="repl"):
            ctx_arg.thinking_state.think_calls += 1
            return _ok_step()

        monkeypatch.setattr(agent, "execute_input", fake_execute_input)
        monkeypatch.setattr(agent.fmt, "repl_answer", lambda t: None)

        agent._execute_loop("5m foo", ctx, mode="repl")
        assert ctx.thinking_state.think_calls == 0

    def test_live_tools_unchanged(self, monkeypatch):
        """_run_agent_step calls _ensure_goal_tools_disabled(ctx.tools) on
        every plain-prompt turn when goal_state has no active goal. The
        iteration's fresh goal_state has none, so the live tools list
        would lose complete_goal if shared. Verify the fork prevents this."""
        registry = LoopRegistry()
        ctx = _make_ctx(loop_registry=registry)
        ctx.tools.append({"function": {"name": "complete_goal"}})
        ctx.tools.append({"function": {"name": "read_file"}})

        def fake_execute_input(parsed, ctx_arg, *, mode="repl"):
            # Simulate _run_agent_step's post-turn goal-tool disable.
            ctx_arg.tools[:] = [
                t for t in ctx_arg.tools if t["function"]["name"] != "complete_goal"
            ]
            return _ok_step()

        monkeypatch.setattr(agent, "execute_input", fake_execute_input)
        monkeypatch.setattr(agent.fmt, "repl_answer", lambda t: None)

        agent._execute_loop("5m foo", ctx, mode="repl")
        names = [t["function"]["name"] for t in ctx.tools]
        assert "complete_goal" in names
        assert "read_file" in names

    def test_iteration_loop_registry_is_none(self, monkeypatch):
        registry = LoopRegistry()
        ctx = _make_ctx(loop_registry=registry)

        seen: dict = {}

        def fake_execute_input(parsed, ctx_arg, *, mode="repl"):
            seen["loop_registry"] = ctx_arg.loop_registry
            seen["no_history"] = ctx_arg.no_history
            seen["continue_here"] = ctx_arg.continue_here
            return _ok_step()

        monkeypatch.setattr(agent, "execute_input", fake_execute_input)
        monkeypatch.setattr(agent.fmt, "repl_answer", lambda t: None)

        agent._execute_loop("5m foo", ctx, mode="repl")

        assert seen["loop_registry"] is None
        assert seen["no_history"] is True
        assert seen["continue_here"] is False


class TestCadence:
    def test_due_advances_with_clock(self, monkeypatch):
        clock = _CapturedClock(start=100.0)
        _patch_clock(monkeypatch, clock)
        registry = LoopRegistry()
        ctx = _make_ctx(loop_registry=registry)

        fires: list[int] = []

        def fake_execute_input(parsed, ctx_arg, *, mode="repl"):
            fires.append(int(clock.t))
            return _ok_step()

        monkeypatch.setattr(agent, "execute_input", fake_execute_input)
        monkeypatch.setattr(agent.fmt, "repl_answer", lambda t: None)

        agent._execute_loop("30s probe1", ctx, mode="repl")
        agent._execute_loop("1m probe2", ctx, mode="repl")
        assert len(fires) == 2

        # Advance 10s — nothing due yet.
        clock.advance(10)
        agent._fire_due_loops(ctx)
        assert len(fires) == 2

        # Advance to 30s past last fire — only probe1 due.
        clock.advance(25)  # total 35s for probe1, well past 30s
        agent._fire_due_loops(ctx)
        assert len(fires) == 3

        # Advance well past 1m for probe2 — both due now.
        clock.advance(60)
        agent._fire_due_loops(ctx)
        assert len(fires) == 5


class TestUnloop:
    def test_unloop_by_id(self, monkeypatch):
        registry = LoopRegistry()
        ctx = _make_ctx(loop_registry=registry)
        monkeypatch.setattr(agent, "execute_input", lambda *a, **k: _ok_step())
        monkeypatch.setattr(agent.fmt, "repl_answer", lambda t: None)

        agent._execute_loop("5m a", ctx, mode="repl")
        agent._execute_loop("5m b", ctx, mode="repl")
        assert len(registry) == 2

        result = agent._execute_unloop("1", ctx)
        assert not result.is_error
        assert len(registry) == 1
        assert list(registry)[0].id == 2

    def test_unloop_all(self, monkeypatch):
        registry = LoopRegistry()
        ctx = _make_ctx(loop_registry=registry)
        monkeypatch.setattr(agent, "execute_input", lambda *a, **k: _ok_step())
        monkeypatch.setattr(agent.fmt, "repl_answer", lambda t: None)

        agent._execute_loop("5m a", ctx, mode="repl")
        agent._execute_loop("5m b", ctx, mode="repl")
        result = agent._execute_unloop("all", ctx)
        assert not result.is_error
        assert len(registry) == 0

    def test_unloop_unknown_id(self):
        registry = LoopRegistry()
        ctx = _make_ctx(loop_registry=registry)
        result = agent._execute_unloop("99", ctx)
        assert result.is_error


class TestResetCancellation:
    def _populate(self, monkeypatch) -> tuple[InputContext, LoopRegistry]:
        registry = LoopRegistry()
        ctx = _make_ctx(loop_registry=registry)
        monkeypatch.setattr(agent, "execute_input", lambda *a, **k: _ok_step())
        monkeypatch.setattr(agent.fmt, "repl_answer", lambda t: None)
        agent._execute_loop("5m foo", ctx, mode="repl")
        assert len(registry) == 1
        return ctx, registry

    def test_clear_cancels_loops(self, monkeypatch):
        ctx, registry = self._populate(monkeypatch)
        agent._repl_clear(
            ctx.messages,
            ctx.thinking_state,
            todo_state=ctx.todo_state,
            loop_registry=registry,
        )
        assert len(registry) == 0


class TestFailureScheduling:
    def test_failure_advances_last_fire(self, monkeypatch):
        clock = _CapturedClock(start=1000.0)
        _patch_clock(monkeypatch, clock)
        registry = LoopRegistry()
        ctx = _make_ctx(loop_registry=registry)

        def fake_execute_input(parsed, ctx_arg, *, mode="repl"):
            return _err_step()

        monkeypatch.setattr(agent, "execute_input", fake_execute_input)
        monkeypatch.setattr(agent.fmt, "repl_answer", lambda t: None)

        agent._execute_loop("30s foo", ctx, mode="repl")
        reg = list(registry)[0]
        assert reg.consecutive_failures == 1
        first_last_fire = reg.last_fire

        # Without advancing the clock, _fire_due_loops should not re-fire.
        agent._fire_due_loops(ctx)
        assert reg.consecutive_failures == 1
        assert reg.last_fire == first_last_fire

    def test_warn_at_threshold(self, monkeypatch):
        clock = _CapturedClock(start=0.0)
        _patch_clock(monkeypatch, clock)
        registry = LoopRegistry()
        ctx = _make_ctx(loop_registry=registry)

        warnings: list[str] = []
        monkeypatch.setattr(agent, "execute_input", lambda *a, **k: _err_step())
        monkeypatch.setattr(agent.fmt, "warning", lambda t: warnings.append(t))
        monkeypatch.setattr(agent.fmt, "info", lambda t: None)
        monkeypatch.setattr(agent.fmt, "repl_answer", lambda t: None)

        agent._execute_loop("30s foo", ctx, mode="repl")
        reg = list(registry)[0]
        for _ in range(WARN_FAILURES - 1):
            clock.advance(31)
            agent._fire_due_loops(ctx)
        assert reg.consecutive_failures == WARN_FAILURES
        assert any("consecutive failures" in w for w in warnings)

    def test_auto_cancel_at_threshold(self, monkeypatch):
        clock = _CapturedClock(start=0.0)
        _patch_clock(monkeypatch, clock)
        registry = LoopRegistry()
        ctx = _make_ctx(loop_registry=registry)

        warnings: list[str] = []
        monkeypatch.setattr(
            agent, "execute_input", lambda *a, **k: _err_step("error: provider died")
        )
        monkeypatch.setattr(agent.fmt, "warning", lambda t: warnings.append(t))
        monkeypatch.setattr(agent.fmt, "info", lambda t: None)
        monkeypatch.setattr(agent.fmt, "repl_answer", lambda t: None)

        agent._execute_loop("30s foo", ctx, mode="repl")
        for _ in range(CANCEL_FAILURES - 1):
            clock.advance(31)
            agent._fire_due_loops(ctx)
        assert len(registry) == 0
        assert any("auto-cancelled" in w for w in warnings)
        assert any("provider died" in w for w in warnings)

    def test_failure_counter_reset(self, monkeypatch):
        clock = _CapturedClock(start=0.0)
        _patch_clock(monkeypatch, clock)
        registry = LoopRegistry()
        ctx = _make_ctx(loop_registry=registry)

        result_seq = iter(
            [
                _err_step(),
                _err_step(),
                _ok_step("good"),
                _err_step(),
                _err_step(),
                _err_step(),
                _err_step(),
                _err_step(),
            ]
        )
        monkeypatch.setattr(agent, "execute_input", lambda *a, **k: next(result_seq))
        monkeypatch.setattr(agent.fmt, "repl_answer", lambda t: None)

        agent._execute_loop("30s foo", ctx, mode="repl")  # 1st fail
        reg = list(registry)[0]
        assert reg.consecutive_failures == 1

        clock.advance(31)
        agent._fire_due_loops(ctx)  # 2nd fail
        assert reg.consecutive_failures == 2

        clock.advance(31)
        agent._fire_due_loops(ctx)  # success — must reset
        assert reg.consecutive_failures == 0

        for _ in range(5):
            clock.advance(31)
            agent._fire_due_loops(ctx)

        # Five fresh failures after reset — NOT cancelled.
        assert len(registry) == 1
        assert reg.consecutive_failures == 5


class TestIterationCrashIsolation:
    def test_runtime_error_keeps_loop_alive(self, monkeypatch):
        clock = _CapturedClock(start=0.0)
        _patch_clock(monkeypatch, clock)
        registry = LoopRegistry()
        ctx = _make_ctx(loop_registry=registry)

        def fake_execute_input(parsed, ctx_arg, *, mode="repl"):
            raise RuntimeError("boom")

        monkeypatch.setattr(agent, "execute_input", fake_execute_input)
        warnings: list[str] = []
        monkeypatch.setattr(agent.fmt, "warning", lambda t: warnings.append(t))
        monkeypatch.setattr(agent.fmt, "info", lambda t: None)
        monkeypatch.setattr(agent.fmt, "repl_answer", lambda t: None)

        agent._execute_loop("30s foo", ctx, mode="repl")
        assert len(registry) == 1
        reg = list(registry)[0]
        assert reg.consecutive_failures == 1
        assert any("RuntimeError" in w for w in warnings)

    def test_keyboard_interrupt_counts_as_failure(self, monkeypatch):
        clock = _CapturedClock(start=0.0)
        _patch_clock(monkeypatch, clock)
        registry = LoopRegistry()
        ctx = _make_ctx(loop_registry=registry)

        def fake_execute_input(parsed, ctx_arg, *, mode="repl"):
            raise KeyboardInterrupt()

        monkeypatch.setattr(agent, "execute_input", fake_execute_input)
        monkeypatch.setattr(agent.fmt, "warning", lambda t: None)
        monkeypatch.setattr(agent.fmt, "info", lambda t: None)
        monkeypatch.setattr(agent.fmt, "repl_answer", lambda t: None)

        agent._execute_loop("30s foo", ctx, mode="repl")
        assert len(registry) == 1
        assert list(registry)[0].consecutive_failures == 1


class TestReportTurnSync:
    def test_turn_offset_updated_after_iteration(self, monkeypatch):
        registry = LoopRegistry()
        ctx = _make_ctx(loop_registry=registry)

        report = types.SimpleNamespace(max_turn_seen=7)
        ctx.loop_kwargs["report"] = report

        def fake_execute_input(parsed, ctx_arg, *, mode="repl"):
            report.max_turn_seen = 12
            return _ok_step()

        monkeypatch.setattr(agent, "execute_input", fake_execute_input)
        monkeypatch.setattr(agent.fmt, "repl_answer", lambda t: None)

        agent._execute_loop("5m foo", ctx, mode="repl")
        assert ctx.loop_kwargs["turn_offset"] == 12


class TestLoopsCommand:
    def test_loops_table_lists_active(self, monkeypatch):
        registry = LoopRegistry()
        ctx = _make_ctx(loop_registry=registry)
        monkeypatch.setattr(agent, "execute_input", lambda *a, **k: _ok_step())
        monkeypatch.setattr(agent.fmt, "repl_answer", lambda t: None)

        agent._execute_loop("5m alpha", ctx, mode="repl")
        agent._execute_loop("2m beta", ctx, mode="repl")
        table = agent._format_loops_table(registry)
        assert "alpha" in table
        assert "beta" in table
        assert "5m" in table
        assert "2m" in table

    def test_loops_empty(self):
        assert agent._format_loops_table(LoopRegistry()) == "no active loops"


class TestSlashBodyRejection:
    def test_repl_rejects_slash_body(self, monkeypatch):
        registry = LoopRegistry()
        ctx = _make_ctx(loop_registry=registry)
        monkeypatch.setattr(agent, "execute_input", lambda *a, **k: _ok_step())
        monkeypatch.setattr(agent.fmt, "repl_answer", lambda t: None)
        monkeypatch.setattr(agent.fmt, "info", lambda t: None)

        result = agent._execute_loop("5m /audit", ctx, mode="repl")
        assert result.is_error
        assert "must be a plain prompt" in (result.text or "")
        assert len(registry) == 0

    def test_oneshot_rejects_slash_body(self, monkeypatch):
        ctx = _make_ctx()
        monkeypatch.setattr(agent, "execute_input", lambda *a, **k: _ok_step())

        result = agent._execute_loop("5m /audit", ctx, mode="oneshot")
        assert result.is_error
        assert "must be a plain prompt" in (result.text or "")

    def test_rejects_custom_bang_command(self, monkeypatch):
        registry = LoopRegistry()
        ctx = _make_ctx(loop_registry=registry)
        monkeypatch.setattr(agent, "execute_input", lambda *a, **k: _ok_step())
        monkeypatch.setattr(agent.fmt, "repl_answer", lambda t: None)
        monkeypatch.setattr(agent.fmt, "info", lambda t: None)

        result = agent._execute_loop("5m !checks", ctx, mode="repl")
        assert result.is_error
        assert "must be a plain prompt" in (result.text or "")
        assert len(registry) == 0

    def test_double_bang_shell_still_rejected(self, monkeypatch):
        registry = LoopRegistry()
        ctx = _make_ctx(loop_registry=registry)
        monkeypatch.setattr(agent, "execute_input", lambda *a, **k: _ok_step())
        monkeypatch.setattr(agent.fmt, "repl_answer", lambda t: None)
        monkeypatch.setattr(agent.fmt, "info", lambda t: None)

        result = agent._execute_loop("5m !! ls", ctx, mode="repl")
        assert result.is_error

    def test_plain_prompt_accepted(self, monkeypatch):
        registry = LoopRegistry()
        ctx = _make_ctx(loop_registry=registry)
        monkeypatch.setattr(agent, "execute_input", lambda *a, **k: _ok_step())
        monkeypatch.setattr(agent.fmt, "repl_answer", lambda t: None)
        monkeypatch.setattr(agent.fmt, "info", lambda t: None)

        result = agent._execute_loop(
            "5m check PR status and summarize", ctx, mode="repl"
        )
        assert not result.is_error
        assert len(registry) == 1


class TestSkillReadRootsIsolation:
    """The use_skill tool can append to skill_read_roots; the fork must
    protect the live list. /add-dir-ro is rejected by the slash-body guard,
    so the tool path is the only remaining mutation source."""

    def test_skill_root_append_inside_iteration_does_not_touch_live(
        self, monkeypatch, tmp_path
    ):
        registry = LoopRegistry()
        ctx = _make_ctx(loop_registry=registry)
        live_reads: list = []
        ctx.skill_read_roots = live_reads
        ctx.loop_kwargs["skill_read_roots"] = live_reads

        target = tmp_path / "skill"
        target.mkdir()

        def fake_execute_input(parsed, ctx_arg, *, mode="repl"):
            ctx_arg.skill_read_roots.append(target.resolve())
            ctx_arg.loop_kwargs["skill_read_roots"].append("via_kwargs_r")
            return _ok_step()

        monkeypatch.setattr(agent, "execute_input", fake_execute_input)
        monkeypatch.setattr(agent.fmt, "repl_answer", lambda t: None)

        agent._execute_loop("5m foo", ctx, mode="repl")

        assert live_reads == []
        assert ctx.loop_kwargs["skill_read_roots"] is live_reads


class TestNoContinueHereWrites:
    """Iterations must not write .swival/continue files when they exhaust."""

    def test_iter_loop_kwargs_continue_here_false(self):
        registry = LoopRegistry()
        ctx = _make_ctx(loop_registry=registry)
        ctx.loop_kwargs["continue_here"] = True

        reg = registry.register(
            interval_seconds=300,
            prompt="foo",
            parsed_prompt=agent.parse_input_line("foo"),
        )
        iter_ctx, iter_subagent = agent._build_iteration_ctx(reg, ctx)
        try:
            assert iter_ctx.continue_here is False
            assert iter_ctx.no_history is True
            assert iter_ctx.loop_kwargs["continue_here"] is False
            # Live loop_kwargs must remain unchanged.
            assert ctx.loop_kwargs["continue_here"] is True
        finally:
            if iter_subagent is not None and hasattr(iter_subagent, "shutdown"):
                iter_subagent.shutdown()

    def test_continue_file_not_written_on_exhaustion(self, monkeypatch, tmp_path):
        """Simulate run_agent_loop's continue-here check inside the iteration."""
        registry = LoopRegistry()
        ctx = _make_ctx(loop_registry=registry)
        ctx.loop_kwargs["continue_here"] = True
        ctx.base_dir = str(tmp_path)

        writes: list[str] = []

        def fake_execute_input(parsed, ctx_arg, *, mode="repl"):
            # Mirror what run_agent_loop does on exhaustion at agent.py:7768.
            if ctx_arg.loop_kwargs.get("continue_here"):
                writes.append(ctx_arg.base_dir)
            return StepResult(kind="agent_turn", text="ran out", exhausted=True)

        monkeypatch.setattr(agent, "execute_input", fake_execute_input)
        monkeypatch.setattr(agent.fmt, "repl_answer", lambda t: None)
        monkeypatch.setattr(agent.fmt, "warning", lambda t: None)
        monkeypatch.setattr(agent.fmt, "info", lambda t: None)

        agent._execute_loop("5m foo", ctx, mode="repl")
        assert writes == []


class TestClockSeam:
    def test_only_loops_monotonic_patch_suffices(self, monkeypatch):
        """All loop scheduling/status reads must flow through loops.monotonic."""
        clock = _CapturedClock(start=500.0)
        # Patch only the seam — NOT agent.time.monotonic.
        monkeypatch.setattr(loops_mod, "monotonic", clock)

        registry = LoopRegistry()
        ctx = _make_ctx(loop_registry=registry)
        monkeypatch.setattr(agent, "execute_input", lambda *a, **k: _ok_step())
        monkeypatch.setattr(agent.fmt, "repl_answer", lambda t: None)
        monkeypatch.setattr(agent.fmt, "info", lambda t: None)
        monkeypatch.setattr(agent.fmt, "warning", lambda t: None)

        agent._execute_loop("30s foo", ctx, mode="repl")
        reg = list(registry)[0]
        assert reg.last_fire == 500.0

        # Not due yet at 510.
        clock.advance(10)
        agent._fire_due_loops(ctx)
        assert reg.last_fire == 500.0

        # Due after 31 elapsed seconds.
        clock.advance(21)
        agent._fire_due_loops(ctx)
        assert reg.last_fire == 531.0

        # /loops table next-fire arithmetic uses the seam too.
        table = agent._format_loops_table(registry)
        assert "next >= 30s" in table


class TestDriverErrorThresholds:
    def test_repeated_deepcopy_failure_auto_cancels(self, monkeypatch):
        clock = _CapturedClock(start=0.0)
        _patch_clock(monkeypatch, clock)

        registry = LoopRegistry()
        ctx = _make_ctx(loop_registry=registry)

        warnings: list[str] = []
        monkeypatch.setattr(agent.fmt, "warning", lambda t: warnings.append(t))
        monkeypatch.setattr(agent.fmt, "info", lambda t: None)
        monkeypatch.setattr(agent.fmt, "repl_answer", lambda t: None)

        def raising_fork(_messages):
            raise RuntimeError("fork boom")

        monkeypatch.setattr(agent, "_fork_messages", raising_fork)

        # First call registers the loop and the inaugural iteration fails
        # via the deepcopy-driver path.
        result = agent._execute_loop("30s foo", ctx, mode="repl")
        assert result.kind == "state_change"
        assert len(registry) == 1
        reg = list(registry)[0]
        assert reg.consecutive_failures == 1

        for _ in range(CANCEL_FAILURES - 1):
            clock.advance(31)
            agent._fire_due_loops(ctx)

        assert len(registry) == 0
        assert any("auto-cancelled" in w for w in warnings)
        assert any("fork boom" in w for w in warnings)


class TestAutoCancelFooter:
    def test_no_next_fire_footer_after_auto_cancel(self, monkeypatch):
        clock = _CapturedClock(start=0.0)
        _patch_clock(monkeypatch, clock)
        registry = LoopRegistry()
        ctx = _make_ctx(loop_registry=registry)

        infos: list[str] = []
        monkeypatch.setattr(agent.fmt, "info", lambda t: infos.append(t))
        monkeypatch.setattr(agent.fmt, "warning", lambda t: None)
        monkeypatch.setattr(agent, "execute_input", lambda *a, **k: _err_step())
        monkeypatch.setattr(agent.fmt, "repl_answer", lambda t: None)

        agent._execute_loop("30s foo", ctx, mode="repl")
        for _ in range(CANCEL_FAILURES - 1):
            clock.advance(31)
            agent._fire_due_loops(ctx)

        assert len(registry) == 0
        # Find the last info message — it must NOT be the next-fire footer.
        final_done = [m for m in infos if "next fire" in m]
        # There should be WARN_FAILURES+ footers prior to the cancellation,
        # but none on the cancellation iteration itself.
        assert len(final_done) == CANCEL_FAILURES - 1


class TestSessionCostReconcile:
    def test_no_subagent_iteration_renders_cost(self, monkeypatch):
        # A direct-LLM scheduled command records cost without ever creating
        # a subagent manager; the iteration finalizer must still reconcile.
        from conftest import plain_console

        from swival.cost import CostObservation, SessionCost

        registry = LoopRegistry()
        ctx = _make_ctx(loop_registry=registry)
        session_cost = SessionCost()
        ctx.loop_kwargs["session_cost"] = session_cost
        ctx.loop_kwargs["verbose"] = True

        def fake_execute_input(parsed, ctx_arg, *, mode="repl"):
            ctx_arg.loop_kwargs["session_cost"].record(CostObservation("known", 0.002))
            return _ok_step("done")

        monkeypatch.setattr(agent, "execute_input", fake_execute_input)
        monkeypatch.setattr(agent.fmt, "repl_answer", lambda t: None)

        with plain_console() as buf:
            agent._execute_loop("5m foo", ctx, mode="repl")

        lines = [ln for ln in buf.getvalue().splitlines() if "Session cost" in ln]
        assert lines == ["  Session cost: ~$0.002000"]

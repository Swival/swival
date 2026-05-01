"""Integration tests for goal-driven agent loop continuation."""

import types

from swival import agent
from swival.goal import GoalState, GoalStatus
from swival.session import Session
from swival.snapshot import SnapshotState
from swival.thinking import ThinkingState
from swival.todo import TodoState


def _msg(content=None, tool_calls=None):
    m = types.SimpleNamespace()
    m.content = content
    m.tool_calls = tool_calls
    m.role = "assistant"
    return m


def _tool_call(name, args=None, tc_id="tc1"):
    return types.SimpleNamespace(
        id=tc_id,
        function=types.SimpleNamespace(
            name=name, arguments="{}" if args is None else args
        ),
    )


class _ScriptedLLM:
    """Returns scripted responses, then defaults to a final answer."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.tail_answer = "all done"

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.responses:
            return self.responses.pop(0), "stop"
        return _msg(content=self.tail_answer), "stop"


def _build_loop_kwargs(tmp_path, goal_state, *, max_turns=4):
    return dict(
        api_base="http://x",
        model_id="m",
        max_turns=max_turns,
        max_output_tokens=None,
        temperature=None,
        top_p=None,
        seed=None,
        context_length=128000,
        base_dir=str(tmp_path),
        thinking_state=ThinkingState(),
        todo_state=TodoState(),
        snapshot_state=SnapshotState(),
        goal_state=goal_state,
        resolved_commands={},
        skills_catalog={},
        skill_read_roots=[],
        extra_write_roots=[],
        files_mode="all",
        commands_unrestricted=False,
        shell_allowed=False,
        verbose=False,
        llm_kwargs={"provider": "generic"},
        file_tracker=None,
        continue_here=False,
    )


def test_loop_returns_immediately_without_goal(tmp_path, monkeypatch):
    """No active goal → loop exits as today after a final-text turn."""
    llm = _ScriptedLLM([])
    llm.tail_answer = "first answer"
    monkeypatch.setattr(agent, "call_llm", llm)

    gs = GoalState()
    messages = [{"role": "user", "content": "hello"}]
    answer, exhausted = agent.run_agent_loop(
        messages, [], **_build_loop_kwargs(tmp_path, gs)
    )
    assert answer == "first answer"
    assert exhausted is False
    assert llm.calls == 1


def test_active_goal_injects_continuation_then_stops_on_final_text(
    tmp_path, monkeypatch
):
    """Goal active + final text → loop injects a continuation, second turn returns."""
    # First turn: final text. Continuation injected. Second turn: final text again,
    # now a no-tool continuation → suppression kicks in.
    llm = _ScriptedLLM(
        [
            _msg(content="first pass done"),
            _msg(content="still need user input"),
        ]
    )
    monkeypatch.setattr(agent, "call_llm", llm)

    gs = GoalState()
    gs.create("Finish the migration")

    messages = [{"role": "user", "content": "kick off"}]
    answer, exhausted = agent.run_agent_loop(
        messages, [], **_build_loop_kwargs(tmp_path, gs, max_turns=8)
    )
    assert answer == "still need user input"
    assert exhausted is False
    # Two LLM calls: original + one continuation. After the continuation produced
    # no tool calls, suppression kicks in.
    assert llm.calls == 2
    assert gs.continuation_suppressed is True

    # Continuation prompt should be present in the transcript.
    contents = [m["content"] for m in messages if m.get("role") == "user"]
    assert any("[goal continuation]" in c for c in contents)


def test_continuation_consumes_max_turns(tmp_path, monkeypatch):
    """Continuation turns count against --max-turns (no infinite loop)."""
    # Always returns final text without tool calls — but on the *first* turn
    # only, so suppression fires after one continuation. Test bound: ensure
    # we never exceed max_turns even if model would gladly loop.
    llm = _ScriptedLLM([])
    llm.tail_answer = "(final text)"
    monkeypatch.setattr(agent, "call_llm", llm)

    gs = GoalState()
    gs.create("infinite-task")

    messages = [{"role": "user", "content": "go"}]
    answer, exhausted = agent.run_agent_loop(
        messages, [], **_build_loop_kwargs(tmp_path, gs, max_turns=5)
    )
    assert answer == "(final text)"
    # Suppression after the no-tool continuation; should stop in 2 calls.
    assert llm.calls == 2


def test_active_goal_final_turn_gets_final_attempt_prompt(tmp_path, monkeypatch):
    """On the last allowed turn, active goals get an explicit final attempt."""
    llm = _ScriptedLLM(
        [
            _msg(content=None, tool_calls=[_tool_call("think")]),
            _msg(content="final push"),
        ]
    )
    monkeypatch.setattr(agent, "call_llm", llm)

    gs = GoalState()
    gs.create("finish hard task")

    messages = [{"role": "user", "content": "go"}]
    answer, exhausted = agent.run_agent_loop(
        messages, [], **_build_loop_kwargs(tmp_path, gs, max_turns=2)
    )
    assert answer == "final push"
    assert exhausted is False
    assert llm.calls == 2
    contents = [m["content"] for m in messages if m.get("role") == "user"]
    assert sum(1 for c in contents if "[goal final attempt]" in c) == 1
    assert any("final allowed turn" in c for c in contents)


def test_final_attempt_does_not_apply_without_goal(tmp_path, monkeypatch):
    """Normal non-goal max_turn behavior stays unchanged."""
    llm = _ScriptedLLM(
        [
            _msg(content=None, tool_calls=[_tool_call("think")]),
            _msg(content="ordinary final"),
        ]
    )
    monkeypatch.setattr(agent, "call_llm", llm)

    gs = GoalState()
    messages = [{"role": "user", "content": "go"}]
    answer, exhausted = agent.run_agent_loop(
        messages, [], **_build_loop_kwargs(tmp_path, gs, max_turns=2)
    )
    assert answer == "ordinary final"
    assert exhausted is False
    contents = [m["content"] for m in messages if m.get("role") == "user"]
    assert not any("[goal final attempt]" in c for c in contents)


def test_budget_exhaustion_injects_wrap_up_once(tmp_path, monkeypatch):
    """Budget hit during accounting → wrap-up prompt injected, then second turn returns."""
    llm = _ScriptedLLM(
        [
            _msg(content="working on it"),
            _msg(content="wrap-up summary"),
        ]
    )
    monkeypatch.setattr(agent, "call_llm", llm)

    gs = GoalState()
    gs.create("ship", token_budget=1)  # tiny so any call exceeds

    messages = [{"role": "user", "content": "go"}]
    answer, exhausted = agent.run_agent_loop(
        messages, [], **_build_loop_kwargs(tmp_path, gs, max_turns=8)
    )
    assert answer == "wrap-up summary"
    assert gs.current.status == GoalStatus.BUDGET_LIMITED
    contents = [m["content"] for m in messages if m.get("role") == "user"]
    assert any("[goal budget limit]" in c for c in contents)
    # Wrap-up prompt should only be injected once.
    assert sum(1 for c in contents if "[goal budget limit]" in c) == 1


def test_budget_limited_state_blocks_further_continuations(tmp_path, monkeypatch):
    """Once budget_limited, no more continuation prompts (only wrap-up)."""
    # Exhaust budget on call 1, return final text on call 2 → return.
    # Verify no new continuation_prompt was injected after wrap-up.
    llm = _ScriptedLLM(
        [
            _msg(content="first answer"),
            _msg(content="wrap-up done"),
        ]
    )
    monkeypatch.setattr(agent, "call_llm", llm)

    gs = GoalState()
    gs.create("ship", token_budget=1)

    messages = [{"role": "user", "content": "go"}]
    agent.run_agent_loop(messages, [], **_build_loop_kwargs(tmp_path, gs, max_turns=8))
    contents = [m["content"] for m in messages if m.get("role") == "user"]
    # No regular continuation prompt after the budget-limit prompt is in flight.
    cont_count = sum(1 for c in contents if "[goal continuation]" in c)
    bl_count = sum(1 for c in contents if "[goal budget limit]" in c)
    assert bl_count == 1
    assert cont_count == 0  # never inject a regular continuation alongside


def test_no_continuation_when_paused(tmp_path, monkeypatch):
    """Paused goal → no continuation injection."""
    llm = _ScriptedLLM([])
    llm.tail_answer = "ok"
    monkeypatch.setattr(agent, "call_llm", llm)

    gs = GoalState()
    gs.create("ship")
    gs.pause()

    messages = [{"role": "user", "content": "go"}]
    answer, _ = agent.run_agent_loop(
        messages, [], **_build_loop_kwargs(tmp_path, gs, max_turns=4)
    )
    assert answer == "ok"
    assert llm.calls == 1


def test_no_continuation_when_complete(tmp_path, monkeypatch):
    """Completed goal → no continuation injection."""
    llm = _ScriptedLLM([])
    llm.tail_answer = "ok"
    monkeypatch.setattr(agent, "call_llm", llm)

    gs = GoalState()
    gs.create("ship")
    gs.set_status(GoalStatus.COMPLETE)

    messages = [{"role": "user", "content": "go"}]
    answer, _ = agent.run_agent_loop(
        messages, [], **_build_loop_kwargs(tmp_path, gs, max_turns=4)
    )
    assert answer == "ok"
    assert llm.calls == 1


def test_session_clear_resets_goal(tmp_path, monkeypatch):
    """`Session._make_per_run_state` builds a fresh goal_state per call."""
    # New per-run state for each .run() call.
    s = Session(
        base_dir=str(tmp_path),
        provider="generic",
        base_url="http://x",
        model="m",
        no_system_prompt=True,
        history=False,
    )
    s._setup_done = True
    s._model_id = "m"
    s._api_base = "http://x"
    s._resolved_key = None
    s._llm_kwargs = {"provider": "generic"}
    s._tools = []
    s._allowed_dir_paths = []
    s._allowed_dir_ro_paths = []
    s._skills_catalog = {}
    s._resolved_commands = {}
    s._commands_unrestricted = False
    s._shell_allowed = False
    s._secret_shield = None
    s._mcp_manager = None
    s._a2a_manager = None
    s._llm_cache = None
    from swival.command_policy import CommandPolicy

    s._command_policy = CommandPolicy("full")

    state1 = s._make_per_run_state()
    state2 = s._make_per_run_state()
    assert state1["goal_state"] is not state2["goal_state"]
    assert state1["goal_state"].get() is None
    assert state2["goal_state"].get() is None


# ---------------------------------------------------------------------------
# Goal accounting on context-overflow recovery paths
# ---------------------------------------------------------------------------


class _OverflowThenSucceedLLM:
    """First call raises ContextOverflowError, subsequent calls return text."""

    def __init__(self, answer="recovered"):
        self.answer = answer
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            from swival.report import ContextOverflowError

            raise ContextOverflowError("simulated overflow")
        return _msg(content=self.answer), "stop"


def test_compaction_retry_accounts_goal_usage(tmp_path, monkeypatch):
    """Goal token accounting must run on the post-compaction success path."""
    llm = _OverflowThenSucceedLLM()
    monkeypatch.setattr(agent, "call_llm", llm)

    gs = GoalState()
    gs.create("ship", token_budget=1)  # any positive use trips the budget

    messages = [{"role": "user", "content": "go"}]
    agent.run_agent_loop(messages, [], **_build_loop_kwargs(tmp_path, gs, max_turns=4))
    # The compaction-retry success path must have called the accounting helper.
    assert gs.current.tokens_used > 0
    assert gs.current.status == GoalStatus.BUDGET_LIMITED


# ---------------------------------------------------------------------------
# Goal recap is spliced into compaction output
# ---------------------------------------------------------------------------


def test_drop_middle_turns_inserts_goal_recap():
    gs = GoalState()
    gs.create("ship the migration", token_budget=2000)
    gs.account(tokens_delta=750)
    gs.record_next_step("verify staging tests")

    # Build a wide enough conversation to exercise the middle-drop branch.
    messages = [{"role": "system", "content": "sys"}]
    for i in range(8):
        messages.append({"role": "user", "content": f"u{i}"})
        messages.append({"role": "assistant", "content": f"a{i}"})

    out = agent.drop_middle_turns(messages, goal_state=gs)
    contents = [m.get("content", "") for m in out]
    assert any(isinstance(c, str) and c.startswith("[goal state]") for c in contents)
    # Latest next-step text is preserved verbatim.
    assert any(isinstance(c, str) and "verify staging tests" in c for c in contents)


def test_drop_middle_turns_omits_recap_when_no_goal():
    messages = [{"role": "system", "content": "sys"}]
    for i in range(8):
        messages.append({"role": "user", "content": f"u{i}"})
        messages.append({"role": "assistant", "content": f"a{i}"})

    out = agent.drop_middle_turns(messages, goal_state=None)
    contents = [m.get("content", "") for m in out]
    assert not any(
        isinstance(c, str) and c.startswith("[goal state]") for c in contents
    )


def test_aggressive_drop_turns_inserts_goal_recap():
    gs = GoalState()
    gs.create("audit security boundaries")

    messages = [{"role": "system", "content": "sys"}]
    for i in range(6):
        messages.append({"role": "user", "content": f"u{i}"})
        messages.append({"role": "assistant", "content": f"a{i}"})

    out = agent.aggressive_drop_turns(messages, goal_state=gs)
    contents = [m.get("content", "") for m in out]
    assert any(isinstance(c, str) and c.startswith("[goal state]") for c in contents)


def _make_setup_session(tmp_path):
    s = Session(
        base_dir=str(tmp_path),
        provider="generic",
        base_url="http://x",
        model="m",
        no_system_prompt=True,
        history=False,
    )
    s._setup_done = True
    s._model_id = "m"
    s._api_base = "http://x"
    s._resolved_key = None
    s._llm_kwargs = {"provider": "generic"}
    s._tools = []
    s._allowed_dir_paths = []
    s._allowed_dir_ro_paths = []
    s._skills_catalog = {}
    s._resolved_commands = {}
    s._commands_unrestricted = False
    s._shell_allowed = False
    s._secret_shield = None
    s._mcp_manager = None
    s._a2a_manager = None
    s._llm_cache = None
    from swival.command_policy import CommandPolicy

    s._command_policy = CommandPolicy("full")
    return s


def test_repl_compact_drop_inserts_goal_recap():
    from swival.agent import _repl_compact

    gs = GoalState()
    gs.create("audit boundaries", token_budget=2000)
    gs.account(tokens_delta=300)
    gs.record_next_step("write integration tests")

    messages = [{"role": "system", "content": "sys"}]
    for i in range(8):
        messages.append({"role": "user", "content": f"u{i}"})
        messages.append({"role": "assistant", "content": f"a{i}"})

    _repl_compact(messages, [], None, "--drop", None, gs)
    contents = [m.get("content", "") for m in messages]
    assert any(isinstance(c, str) and c.startswith("[goal state]") for c in contents)
    assert any(isinstance(c, str) and "write integration tests" in c for c in contents)


# ---------------------------------------------------------------------------
# Goal-launch turn (synthetic start_prompt, goal_launch_turn=True)
# ---------------------------------------------------------------------------


def test_goal_launch_no_tool_response_suppresses_in_same_turn(tmp_path, monkeypatch):
    """A no-tool first goal-launch response stops in one LLM call.

    Without the launch flag, the loop would inject a continuation prompt and
    burn a second LLM call before suppression. With it, suppression fires
    immediately on the no-tool first turn.
    """
    llm = _ScriptedLLM([_msg(content="blocked: need credentials")])
    monkeypatch.setattr(agent, "call_llm", llm)

    gs = GoalState()
    gs.create("ship the migration")

    messages = [
        {"role": "user", "content": gs.start_prompt(), "_swival_synthetic": True}
    ]
    answer, exhausted = agent.run_agent_loop(
        messages,
        [],
        **_build_loop_kwargs(tmp_path, gs, max_turns=8),
        goal_launch_turn=True,
    )
    assert answer == "blocked: need credentials"
    assert exhausted is False
    assert llm.calls == 1
    assert gs.continuation_suppressed is True
    contents = [m["content"] for m in messages if m.get("role") == "user"]
    # No automatic continuation was injected on top of the start prompt.
    assert not any("[goal continuation]" in c for c in contents)


def test_goal_launch_with_tools_then_continuation(tmp_path, monkeypatch):
    """Goal-launch + tool use → next no-tool turn injects a continuation normally."""
    llm = _ScriptedLLM(
        [
            _msg(tool_calls=[_tool_call("think", '{"thought": "plan"}')]),
            _msg(content="report"),
            _msg(content="final"),
        ]
    )
    monkeypatch.setattr(agent, "call_llm", llm)

    gs = GoalState()
    gs.create("port the loop")

    messages = [
        {"role": "user", "content": gs.start_prompt(), "_swival_synthetic": True}
    ]
    answer, _ = agent.run_agent_loop(
        messages,
        [],
        **_build_loop_kwargs(tmp_path, gs, max_turns=8),
        goal_launch_turn=True,
    )
    assert answer == "final"
    contents = [m["content"] for m in messages if m.get("role") == "user"]
    # The launch flag is consumed by turn 1's tool use, so the no-tool turn 2
    # gets a normal continuation injection (not suppressed prematurely).
    assert any("[goal continuation]" in c for c in contents)


def test_goal_launch_budget_exhaustion_injects_wrap_up_once(tmp_path, monkeypatch):
    """Budget hit on the goal-launch turn injects the wrap-up prompt exactly once."""
    llm = _ScriptedLLM(
        [
            _msg(content="started but ran out"),
            _msg(content="wrap-up summary"),
        ]
    )
    monkeypatch.setattr(agent, "call_llm", llm)

    gs = GoalState()
    gs.create("ship it", token_budget=1)

    messages = [
        {"role": "user", "content": gs.start_prompt(), "_swival_synthetic": True}
    ]
    answer, _ = agent.run_agent_loop(
        messages,
        [],
        **_build_loop_kwargs(tmp_path, gs, max_turns=8),
        goal_launch_turn=True,
    )
    assert answer == "wrap-up summary"
    assert gs.current.status == GoalStatus.BUDGET_LIMITED
    contents = [m["content"] for m in messages if m.get("role") == "user"]
    assert sum(1 for c in contents if "[goal budget limit]" in c) == 1
    assert not any("[goal continuation]" in c for c in contents)


def test_goal_launch_complete_via_complete_goal_exits_cleanly(tmp_path, monkeypatch):
    """Model calls complete_goal on the launch turn -> no continuation injected."""
    # The tool path normally goes through dispatch; for this loop-level test we
    # simulate the resulting state transition by switching the goal to COMPLETE
    # mid-call. The model returns final text on the second turn.
    gs = GoalState()
    gs.create("ship it")

    class _CompletingLLM:
        def __init__(self):
            self.calls = 0

        def __call__(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                # Simulate: model used complete_goal during turn 1.
                gs.set_status(GoalStatus.COMPLETE)
                return _msg(content="all done"), "stop"
            return _msg(content="(should not be reached)"), "stop"

    llm = _CompletingLLM()
    monkeypatch.setattr(agent, "call_llm", llm)

    messages = [
        {"role": "user", "content": gs.start_prompt(), "_swival_synthetic": True}
    ]
    answer, _ = agent.run_agent_loop(
        messages,
        [],
        **_build_loop_kwargs(tmp_path, gs, max_turns=8),
        goal_launch_turn=True,
    )
    assert answer == "all done"
    assert llm.calls == 1
    assert gs.current.status == GoalStatus.COMPLETE
    contents = [m["content"] for m in messages if m.get("role") == "user"]
    assert not any("[goal continuation]" in c for c in contents)
    assert not any("[goal budget limit]" in c for c in contents)


# ---------------------------------------------------------------------------
# GoalState.start_prompt() shape
# ---------------------------------------------------------------------------


def test_start_prompt_includes_objective_and_audit():
    gs = GoalState()
    gs.create("Migrate the auth layer", token_budget=2000)
    body = gs.start_prompt()
    assert "[goal start]" in body
    assert "Migrate the auth layer" in body
    assert "completion audit" in body
    assert "Token budget: 2000" in body


def test_start_prompt_empty_without_goal():
    gs = GoalState()
    assert gs.start_prompt() == ""

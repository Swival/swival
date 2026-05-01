"""Unit tests for swival.goal."""

import time

import pytest

from swival.goal import (
    GOAL_RECAP_PREFIX,
    GoalState,
    GoalStatus,
)


def test_create_and_get():
    state = GoalState()
    assert state.get() is None
    rec = state.create("Ship the auth migration")
    assert rec.objective == "Ship the auth migration"
    assert rec.status == GoalStatus.ACTIVE
    assert rec.token_budget is None
    assert rec.goal_id.startswith("g_")
    assert state.get() is rec
    assert state.has_active() is True


def test_create_rejects_empty_objective():
    state = GoalState()
    with pytest.raises(ValueError):
        state.create("   ")


def test_create_rejects_existing_non_complete():
    state = GoalState()
    state.create("first")
    with pytest.raises(ValueError):
        state.create("second")


def test_create_after_complete_is_allowed():
    state = GoalState()
    state.create("first")
    state.set_status(GoalStatus.COMPLETE)
    rec = state.create("second")
    assert rec.objective == "second"
    assert rec.status == GoalStatus.ACTIVE


def test_replace_resets_counters():
    state = GoalState()
    state.create("first", token_budget=1000)
    state.account(tokens_delta=400)
    rec = state.create("second", token_budget=500, replace=True)
    assert rec.objective == "second"
    assert rec.tokens_used == 0
    assert rec.token_budget == 500
    assert state.continuation_suppressed is False


def test_pause_and_resume():
    state = GoalState()
    state.create("ship it")
    assert state.pause() is True
    assert state.current.status == GoalStatus.PAUSED
    assert state.has_active() is False
    assert state.resume() is True
    assert state.current.status == GoalStatus.ACTIVE
    assert state.has_active() is True


def test_pause_only_when_active():
    state = GoalState()
    assert state.pause() is False
    state.create("ship")
    state.pause()
    assert state.pause() is False  # already paused


def test_resume_only_when_paused():
    state = GoalState()
    state.create("x")
    assert state.resume() is False  # active
    state.set_status(GoalStatus.COMPLETE)
    assert state.resume() is False


def test_clear_wipes_state():
    state = GoalState()
    state.create("x", token_budget=100)
    state.account(tokens_delta=50)
    assert state.clear() is True
    assert state.get() is None
    assert state.continuation_suppressed is False
    assert state.budget_limit_reported_goal_id is None
    assert state.clear() is False


def test_account_tokens_and_budget_transition():
    state = GoalState()
    state.create("x", token_budget=100)
    state.turn_started()
    hit = state.account(tokens_delta=40)
    assert hit is False
    assert state.current.status == GoalStatus.ACTIVE
    hit = state.account(tokens_delta=80)
    assert hit is True
    assert state.current.status == GoalStatus.BUDGET_LIMITED
    assert state.budget_exhausted() is True


def test_account_ignored_when_not_active():
    state = GoalState()
    state.create("x", token_budget=100)
    state.pause()
    state.account(tokens_delta=200)
    assert state.current.tokens_used == 0
    assert state.current.status == GoalStatus.PAUSED


def test_account_records_estimated_flag():
    state = GoalState()
    state.create("x")
    state.account(tokens_delta=10, estimated=True)
    assert state.current.usage_estimated is True


def test_remaining_budget():
    state = GoalState()
    state.create("x", token_budget=200)
    state.account(tokens_delta=70)
    assert state.remaining_budget() == 130
    state.account(tokens_delta=300)
    assert state.remaining_budget() == 0


def test_remaining_budget_none_without_budget():
    state = GoalState()
    state.create("x")
    assert state.remaining_budget() is None


def test_continuation_prompt_contains_objective_verbatim():
    state = GoalState()
    objective = "do {important_thing} and then {other}"
    state.create(objective, token_budget=1000)
    state.account(tokens_delta=100)
    prompt = state.continuation_prompt()
    # Objective rendered verbatim — no template expansion.
    assert objective in prompt
    assert "OBJECTIVE>>>" in prompt
    assert "1000" in prompt
    assert "900" in prompt  # remaining

    # Continuation prompt is recognizable for pruning.
    assert "[goal continuation]" in prompt


def test_continuation_prompt_warns_on_estimated_usage():
    state = GoalState()
    state.create("x", token_budget=1000)
    state.account(tokens_delta=100, estimated=True)
    prompt = state.continuation_prompt()
    assert "estimate" in prompt.lower()


def test_budget_limit_prompt():
    state = GoalState()
    state.create("ship", token_budget=100)
    state.account(tokens_delta=200)
    text = state.budget_limit_prompt()
    assert "[goal budget limit]" in text
    assert "ship" in text
    assert "wrap" in text.lower()


def test_recap_text_contains_facts_verbatim():
    state = GoalState()
    state.create("ship", token_budget=1000)
    state.account(tokens_delta=250)
    state.record_next_step("Run migration on staging at 14:00")
    state.record_blocker("waiting for DBA approval")
    recap = state.recap_text()
    assert recap.startswith(GOAL_RECAP_PREFIX)
    assert "ship" in recap
    assert "tokens_used: 250" in recap
    assert "token_budget: 1000" in recap
    assert "Run migration on staging at 14:00" in recap
    assert "waiting for DBA approval" in recap


def test_recap_text_none_without_goal():
    state = GoalState()
    assert state.recap_text() is None


def test_summary_line():
    state = GoalState()
    assert state.summary_line() is None
    state.create("ship", token_budget=1000)
    state.account(tokens_delta=100)
    line = state.summary_line()
    assert "goal: active" in line
    assert "ship" in line


def test_continuation_suppression_is_session_state():
    state = GoalState()
    state.create("x")
    state.continuation_suppressed = True
    state.set_status(GoalStatus.PAUSED)
    state.set_status(GoalStatus.ACTIVE)
    # Resume clears continuation suppression.
    assert state.continuation_suppressed is False


def test_complete_goal_only_complete_via_set_status():
    state = GoalState()
    state.create("x")
    state.set_status(GoalStatus.COMPLETE)
    assert state.current.status == GoalStatus.COMPLETE


def test_set_status_invalid():
    state = GoalState()
    state.create("x")
    with pytest.raises(ValueError):
        state.set_status("frob")


def test_set_status_no_goal():
    state = GoalState()
    with pytest.raises(ValueError):
        state.set_status(GoalStatus.COMPLETE)


def test_wall_clock_accumulates_across_pauses():
    state = GoalState()
    state.create("x")
    # Simulate active period.
    state.active_started_at = time.monotonic() - 0.5
    state.pause()
    assert state.current.time_used_seconds >= 0.5
    state.resume()
    state.active_started_at = time.monotonic() - 0.2
    state.set_status(GoalStatus.COMPLETE)
    assert state.current.time_used_seconds >= 0.7


def test_active_goal_id_this_turn_protects_replaced_accounting():
    state = GoalState()
    state.create("first", token_budget=1000)
    state.turn_started()
    state.create("second", token_budget=1000, replace=True)
    # Prior turn's accounting must not affect the freshly-created goal.
    state.account(tokens_delta=500)
    # active_goal_id_this_turn was reset by create(replace=True) so
    # accounting should still apply to the new goal.
    assert state.current.tokens_used == 500


def test_status_block_no_goal():
    state = GoalState()
    assert "No goal" in state.status_block()


def test_status_block_with_goal():
    state = GoalState()
    state.create("ship", token_budget=1000)
    state.account(tokens_delta=100)
    state.record_next_step("write tests")
    block = state.status_block()
    assert "ship" in block
    assert "100" in block
    assert "1000" in block
    assert "write tests" in block


def test_to_report_dict():
    state = GoalState()
    assert state.to_report_dict() is None
    state.create("x", token_budget=500)
    state.account(tokens_delta=42)
    d = state.to_report_dict()
    assert d["objective"] == "x"
    assert d["status"] == "active"
    assert d["tokens_used"] == 42
    assert d["token_budget"] == 500

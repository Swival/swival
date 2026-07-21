"""Tests for the text-first multi-agent ``/simplify`` workflow."""

from __future__ import annotations

import contextlib
import threading
import types

from swival.simplify import (
    _REVIEW_ROLES,
    _REVIEWER_SYSTEM,
    _reviewer_prompt,
    build_simplify_prompt,
    prepare_simplify_prompt,
)


def _tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name}}


def _ctx(tmp_path):
    return types.SimpleNamespace(
        base_dir=str(tmp_path),
        tools=[
            *(_tool(name) for name in _REVIEW_ROLES),
            _tool("read_file"),
            _tool("edit_file"),
        ],
        loop_kwargs={"files_mode": "all", "network_mode": "full"},
    )


def _role_from_messages(messages: list[dict]) -> str:
    prompt = messages[-1]["content"]
    return next(role for role in _REVIEW_ROLES if f"Review lens: {role}" in prompt)


def test_main_prompt_uses_prose_reports_and_free_form_focus():
    reports = {role: f"Natural-language {role} findings." for role in _REVIEW_ROLES}
    prompt = build_simplify_prompt("src plus the command parser", reports)

    assert "src plus the command parser" in prompt
    assert "Interpret them; do not parse them as a protocol" in prompt
    assert "Apply only small, high-confidence" in prompt
    assert "normally reject an idea that needs more than two files" in prompt
    assert "Do not introduce a new abstraction" in prompt
    assert "A successful run may make no changes" in prompt
    assert "Run the relevant formatter, tests, lint, build" in prompt
    assert "Do not output JSON" in prompt
    for role, report in reports.items():
        assert f"--- {role} reviewer ---" in prompt
        assert report in prompt


def test_default_scope_searches_existing_code_broadly():
    prompt = build_simplify_prompt("", {})

    assert "Review the current workspace broadly" in prompt
    assert "existing implementation code" in prompt
    assert "programming language" not in prompt.lower()


def test_roles_are_complementary_and_prompts_are_small_model_friendly():
    assert list(_REVIEW_ROLES) == ["local", "reuse", "contracts"]

    for role in _REVIEW_ROLES:
        prompt = _reviewer_prompt(role, "the parser")
        assert "at most three findings" in _REVIEWER_SYSTEM
        assert "Location, Change, Why simpler, Preserve, Risk, Validate" in (
            _REVIEWER_SYSTEM
        )
        assert '"No safe simplification"' in _REVIEWER_SYSTEM
        assert "Stay within this lens" in prompt

    assert "Do not propose a new shared abstraction" in _REVIEW_ROLES["reuse"]
    assert "callers and tests" in _REVIEW_ROLES["contracts"]


def test_complementary_reviewers_run_in_parallel_and_return_plain_text(
    tmp_path, monkeypatch
):
    barrier = threading.Barrier(len(_REVIEW_ROLES), timeout=3)
    seen = []

    def fake_loop(messages, tools, **kwargs):
        role = _role_from_messages(messages)
        seen.append((role, tools, kwargs))
        barrier.wait()
        return (f"{role}: found a concrete simplification in ordinary prose.", False)

    monkeypatch.setattr("swival.agent.run_agent_loop", fake_loop)
    prompt = prepare_simplify_prompt("any language and layout", _ctx(tmp_path))

    assert {role for role, _tools, _kwargs in seen} == set(_REVIEW_ROLES)
    assert all(
        f"{role}: found a concrete simplification" in prompt for role in _REVIEW_ROLES
    )
    for _role, tools, kwargs in seen:
        names = {tool["function"]["name"] for tool in tools}
        assert "read_file" in names
        assert "edit_file" not in names
        assert kwargs["base_dir"] == str(tmp_path)
        assert kwargs["files_mode"] == "some"
        assert kwargs["network_mode"] == "none"
        assert kwargs["max_turns"] == 30
        denied = kwargs["tool_policy"].check("edit_file", {"file_path": "anything"})
        assert denied is not None and denied.startswith("error:")


def test_reviewer_failures_become_advice_gaps_instead_of_aborting(
    tmp_path, monkeypatch
):
    def fake_loop(messages, tools, **kwargs):
        role = _role_from_messages(messages)
        if role == "reuse":
            raise RuntimeError("provider unavailable")
        if role == "local":
            return ("", True)
        return (f"Useful {role} prose", False)

    monkeypatch.setattr("swival.agent.run_agent_loop", fake_loop)
    prompt = prepare_simplify_prompt("", _ctx(tmp_path))

    assert "reuse reviewer was unavailable: provider unavailable" in prompt
    assert "reviewer returned no usable report" in prompt
    assert "reviewer reached its turn limit" in prompt
    assert "Useful contracts prose" in prompt


def test_reviewer_progress_is_shown_live_and_summarized(tmp_path, monkeypatch):
    labels = []
    infos = []
    warnings = []

    @contextlib.contextmanager
    def fake_spinner(label):
        labels.append(label)
        yield labels.append

    monkeypatch.setattr("swival.fmt.step_spinner", fake_spinner)
    monkeypatch.setattr("swival.fmt.info", infos.append)
    monkeypatch.setattr("swival.fmt.warning", warnings.append)

    def fake_loop(messages, tools, **kwargs):
        role = _role_from_messages(messages)
        emit = kwargs["event_callback"]
        emit("status_update", {"turn": 2, "max_turns": 30, "elapsed": 0.1})
        emit("tool_start", {"id": "tc1", "name": "grep", "turn": 2})
        if role == "contracts":
            raise RuntimeError("provider unavailable")
        return (f"{role} report", False)

    monkeypatch.setattr("swival.agent.run_agent_loop", fake_loop)
    prepare_simplify_prompt("", _ctx(tmp_path))

    assert any("reuse 2/30 grep" in label for label in labels)
    assert any("reuse done" in label for label in labels)
    assert any("contracts failed" in label for label in labels)
    assert any("reuse reviewer finished after 2 turns" in line for line in infos)
    assert any("contracts reviewer failed: provider unavailable" in w for w in warnings)


def test_reports_are_bounded_without_requiring_a_schema():
    report = "free-form prose " * 2_000
    prompt = build_simplify_prompt("", {"reuse": report})

    assert "[report truncated]" in prompt
    assert "candidates" not in prompt
    assert "candidate_id" not in prompt

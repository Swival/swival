"""Tests for /remember editing the live prompt."""

from pathlib import Path

from swival import agent, prompt_spans as ps
from swival.agent import _repl_remember, refresh_system_instructions
from swival.input_dispatch import InputContext
from swival.thinking import ThinkingState
from swival.todo import TodoState


def _make_agents_md(tmp_path: Path, conventions: str = "- old fact\n") -> Path:
    p = tmp_path / "AGENTS.md"
    p.write_text(
        f"## Workflow\n\nstuff\n\n## Conventions\n\n{conventions}",
        encoding="utf-8",
    )
    return p


def _prompt(tmp_path, **overrides):
    kwargs = dict(
        base_dir=str(tmp_path),
        system_prompt=None,
        no_system_prompt=False,
        no_instructions=False,
        no_memory=True,
        skills_catalog={},
        verbose=False,
        policy="interactive",
        context_length=128_000,
        max_output_tokens=32_768,
    )
    kwargs.update(overrides)
    return agent.assemble_system_prompt(**kwargs)


def _ctx(tmp_path, result, **overrides):
    message = {"role": "system", "content": result.content}
    ps.set_spans(message, result.spans)
    kwargs = dict(
        messages=[message],
        tools=[],
        base_dir=str(tmp_path),
        turn_state={"max_turns": 10, "turns_used": 0},
        thinking_state=ThinkingState(verbose=False),
        todo_state=TodoState(verbose=False),
        snapshot_state=None,
        file_tracker=None,
        no_history=True,
        continue_here=False,
        verbose=False,
        loop_kwargs={"context_length": 128_000, "max_output_tokens": 32_768},
        instructions_enabled=result.admission is not None,
    )
    kwargs.update(overrides)
    return InputContext(**kwargs)


def _no_user_config(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "swival.config.global_config_dir",
        lambda: tmp_path / "_no_config",
    )


# -- Positive: the recorded block is replaced --


def test_remember_updates_the_recorded_block(tmp_path, monkeypatch):
    _no_user_config(monkeypatch, tmp_path)
    _make_agents_md(tmp_path, "- old fact\n")
    result = _prompt(tmp_path)
    ctx = _ctx(tmp_path, result)
    head = result.content[: ps.find_span(result.spans, ps.KIND_INSTRUCTIONS)["start"]]

    _repl_remember("new fact", ctx)

    content = ctx.messages[0]["content"]
    assert "new fact" in content
    assert "old fact" in content
    assert content.startswith(head)
    assert content.count("<agent-instructions>") == 1
    assert ps.spans_valid(content, ps.get_spans(ctx.messages[0]))


# -- Negative: a prompt we did not assemble is never edited --


def test_remember_leaves_a_custom_prompt_alone(tmp_path, monkeypatch):
    _no_user_config(monkeypatch, tmp_path)
    _make_agents_md(tmp_path)
    result = _prompt(tmp_path, system_prompt="Custom prompt with no instructions.")
    ctx = _ctx(tmp_path, result)
    original = ctx.messages[0]["content"]

    msg, is_error = _repl_remember("new fact", ctx)

    assert ctx.messages[0]["content"] == original
    assert "Saved for future sessions" in msg
    assert is_error is False
    assert ctx.pending_instruction_failure is None
    assert "new fact" in (tmp_path / "AGENTS.md").read_text()


def test_remember_leaves_a_disabled_session_alone(tmp_path, monkeypatch):
    _no_user_config(monkeypatch, tmp_path)
    _make_agents_md(tmp_path)
    result = _prompt(tmp_path, no_instructions=True)
    ctx = _ctx(tmp_path, result)
    original = ctx.messages[0]["content"]

    _repl_remember("new fact", ctx)

    assert ctx.messages[0]["content"] == original
    assert "new fact" in (tmp_path / "AGENTS.md").read_text()


# -- Negative: no system message at all --


def test_refresh_without_a_system_message(tmp_path):
    result = _prompt(tmp_path)
    ctx = _ctx(tmp_path, result, messages=[])
    refresh = refresh_system_instructions(ctx)
    assert refresh.reason == "this session has no system prompt"
    assert refresh.blocking is False
    assert ctx.messages == []

    ctx = _ctx(tmp_path, result, messages=[{"role": "user", "content": "hi"}])
    assert not refresh_system_instructions(ctx).ok
    assert ctx.messages[0]["content"] == "hi"


def test_refresh_without_recorded_metadata(tmp_path):
    result = _prompt(tmp_path)
    plain = {"role": "system", "content": result.content}
    ctx = _ctx(tmp_path, result, messages=[plain])
    refresh = refresh_system_instructions(ctx)
    assert refresh.reason == "this session did not assemble its own system prompt"
    assert refresh.blocking is False
    assert ctx.messages[0]["content"] == result.content


# -- Negative: command provider session --


def test_remember_leaves_the_command_provider_alone(tmp_path, monkeypatch):
    _no_user_config(monkeypatch, tmp_path)
    _make_agents_md(tmp_path)
    result = _prompt(tmp_path, provider="command")
    ctx = _ctx(tmp_path, result)
    original = ctx.messages[0]["content"]

    _repl_remember("new fact", ctx)

    assert ctx.messages[0]["content"] == original


def test_refresh_preserves_intermediate_files(tmp_path, monkeypatch):
    """A refresh must keep the AGENTS.md files between base_dir and start_dir."""
    _no_user_config(monkeypatch, tmp_path)
    sub = tmp_path / "sub"
    sub.mkdir()
    root_md = tmp_path / "AGENTS.md"
    sub_md = sub / "AGENTS.md"
    root_md.write_text("## Workflow\nstuff\n\n## Conventions\n\n- root fact\n")
    sub_md.write_text("sub instructions")

    result = _prompt(tmp_path, start_dir=sub)
    ctx = _ctx(tmp_path, result, start_dir=sub)

    root_md.write_text(
        "## Workflow\nstuff\n\n## Conventions\n\n- root fact\n- new fact\n"
    )
    assert refresh_system_instructions(ctx).ok

    content = ctx.messages[0]["content"]
    assert "new fact" in content
    assert "sub instructions" in content

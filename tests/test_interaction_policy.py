"""Tests for the interaction-policy system prompt substitution."""

import types

import pytest

from swival.agent import (
    _apply_interaction_policy,
    _COMMAND_PROVIDER_SYSTEM_PROMPT,
    build_system_prompt,
)
from swival import Session, agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build(tmp_path, **kwargs):
    """Call build_system_prompt with sensible defaults (placeholders unsubstituted)."""
    defaults = dict(
        base_dir=str(tmp_path),
        system_prompt=None,
        no_system_prompt=False,
        no_instructions=True,
        no_memory=True,
        skills_catalog={},
        commands_unrestricted=False,
        resolved_commands={},
        verbose=False,
    )
    defaults.update(kwargs)
    return build_system_prompt(**defaults)


def _make_message(content=None, tool_calls=None):
    msg = types.SimpleNamespace()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.role = "assistant"
    return msg


def _simple_llm(*args, **kwargs):
    return _make_message(content="the answer"), "stop"


# ---------------------------------------------------------------------------
# Tests 1-3: _apply_interaction_policy on the default prompt
# ---------------------------------------------------------------------------


class TestAutonomousMode:
    def test_renders_autonomous_directives(self, tmp_path):
        content, _ = _build(tmp_path)
        result = _apply_interaction_policy(content, "autonomous")
        assert "do not stop to ask for confirmation" in result
        assert "pick the most likely intent" in result

    def test_no_interactive_phrases(self, tmp_path):
        content, _ = _build(tmp_path)
        result = _apply_interaction_policy(content, "autonomous")
        assert "ask the user to clarify" not in result
        assert "ask the user a brief clarifying question" not in result


class TestInteractiveMode:
    def test_renders_interactive_directives(self, tmp_path):
        content, _ = _build(tmp_path)
        result = _apply_interaction_policy(content, "interactive")
        assert "ask the user to clarify" in result
        assert "ask the user a brief clarifying question" in result

    def test_no_autonomous_phrases(self, tmp_path):
        content, _ = _build(tmp_path)
        result = _apply_interaction_policy(content, "interactive")
        assert "do not stop to ask for confirmation" not in result


class TestNoPlaceholdersRemain:
    @pytest.mark.parametrize("policy", ["autonomous", "interactive"])
    def test_no_placeholders(self, tmp_path, policy):
        content, _ = _build(tmp_path)
        result = _apply_interaction_policy(content, policy)
        assert "{{AUTONOMY_DIRECTIVE}}" not in result
        assert "{{AMBIGUITY_DIRECTIVE}}" not in result


# ---------------------------------------------------------------------------
# Test 4: Custom system_prompt is untouched
# ---------------------------------------------------------------------------


class TestCustomSystemPrompt:
    @pytest.mark.parametrize("policy", ["autonomous", "interactive"])
    def test_custom_prompt_unchanged(self, tmp_path, policy):
        content, _ = _build(tmp_path, system_prompt="My custom prompt.")
        result = _apply_interaction_policy(content, policy)
        assert result.startswith("My custom prompt.")
        assert "do not stop to ask for confirmation" not in result
        assert "ask the user to clarify" not in result


# ---------------------------------------------------------------------------
# Test 5: Command provider prompt is untouched
# ---------------------------------------------------------------------------


class TestCommandProvider:
    @pytest.mark.parametrize("policy", ["autonomous", "interactive"])
    def test_command_provider_unchanged(self, tmp_path, policy):
        content, _ = _build(tmp_path, provider="command")
        result = _apply_interaction_policy(content, policy)
        assert result.startswith(_COMMAND_PROVIDER_SYSTEM_PROMPT)
        assert "{{AUTONOMY_DIRECTIVE}}" not in result
        assert "{{AMBIGUITY_DIRECTIVE}}" not in result
        assert "do not stop to ask for confirmation" not in result
        assert "ask the user to clarify" not in result


# ---------------------------------------------------------------------------
# Tests 6-7: Session.run() and Session.ask() produce the right policy
# ---------------------------------------------------------------------------


class TestSessionRunPolicy:
    def test_run_produces_autonomous_prompt(self, tmp_path, monkeypatch):
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        captured = {}

        def capturing_run_agent_loop(messages, *args, **kwargs):
            captured["system"] = messages[0]["content"]
            return _simple_llm()

        monkeypatch.setattr(agent, "run_agent_loop", capturing_run_agent_loop)

        s = Session(base_dir=str(tmp_path), history=False, no_instructions=True)
        s.run("test question")

        system = captured["system"]
        assert "do not stop to ask for confirmation" in system
        assert "ask the user to clarify" not in system
        assert "{{AUTONOMY_DIRECTIVE}}" not in system


class TestSessionAskPolicy:
    def test_ask_produces_interactive_prompt(self, tmp_path, monkeypatch):
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        captured = {}

        def capturing_run_agent_loop(messages, *args, **kwargs):
            captured["system"] = messages[0]["content"]
            return _simple_llm()

        monkeypatch.setattr(agent, "run_agent_loop", capturing_run_agent_loop)

        s = Session(base_dir=str(tmp_path), history=False, no_instructions=True)
        s.ask("test question")

        system = captured["system"]
        assert "ask the user to clarify" in system
        assert "do not stop to ask for confirmation" not in system
        assert "{{AUTONOMY_DIRECTIVE}}" not in system

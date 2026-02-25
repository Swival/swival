"""Tests for CLAUDE.md / AGENT.md instruction loading."""

import sys
import types

import pytest

from swival.agent import load_instructions, MAX_INSTRUCTIONS_CHARS


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


class TestFileDiscovery:
    def test_no_instructions_files(self, tmp_path):
        result, loaded = load_instructions(str(tmp_path), verbose=False)
        assert result == ""
        assert loaded == []

    def test_claude_md_only(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("Use tabs not spaces.", encoding="utf-8")
        result, loaded = load_instructions(str(tmp_path), verbose=False)
        assert "<project-instructions>" in result
        assert "Use tabs not spaces." in result
        assert "</project-instructions>" in result
        assert "agent-instructions" not in result
        assert loaded == ["CLAUDE.md"]

    def test_agent_md_only(self, tmp_path):
        (tmp_path / "AGENT.md").write_text("Be concise.", encoding="utf-8")
        result, loaded = load_instructions(str(tmp_path), verbose=False)
        assert "<agent-instructions>" in result
        assert "Be concise." in result
        assert "</agent-instructions>" in result
        assert "project-instructions" not in result
        assert loaded == ["AGENT.md"]

    def test_both_files(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("Project rules.", encoding="utf-8")
        (tmp_path / "AGENT.md").write_text("Agent rules.", encoding="utf-8")
        result, loaded = load_instructions(str(tmp_path), verbose=False)
        assert "<project-instructions>" in result
        assert "<agent-instructions>" in result
        # project-instructions comes first
        pi = result.index("<project-instructions>")
        ai = result.index("<agent-instructions>")
        assert pi < ai
        assert loaded == ["CLAUDE.md", "AGENT.md"]


# ---------------------------------------------------------------------------
# Content handling
# ---------------------------------------------------------------------------


class TestContentHandling:
    def test_truncation(self, tmp_path):
        content = "x" * 15_000
        (tmp_path / "CLAUDE.md").write_text(content, encoding="utf-8")
        result, loaded = load_instructions(str(tmp_path), verbose=False)
        # Should be truncated
        assert f"exceeds {MAX_INSTRUCTIONS_CHARS} character limit" in result
        # Extract content between tags
        inner = result.split("<project-instructions>\n", 1)[1].rsplit(
            "\n</project-instructions>", 1
        )[0]
        # Inner should be exactly MAX_INSTRUCTIONS_CHARS x's + truncation notice
        x_part, notice = inner.rsplit("\n", 1)
        assert len(x_part) == MAX_INSTRUCTIONS_CHARS
        assert x_part == "x" * MAX_INSTRUCTIONS_CHARS
        assert "truncated" in notice

    def test_unreadable_file_skipped(self, tmp_path, monkeypatch):
        (tmp_path / "CLAUDE.md").write_text("readable", encoding="utf-8")

        original_open = open

        def bad_open(self, *args, **kwargs):
            if self.name.endswith("CLAUDE.md"):
                raise OSError("Permission denied")
            return original_open(self, *args, **kwargs)

        monkeypatch.setattr("pathlib.Path.open", bad_open)
        result, loaded = load_instructions(str(tmp_path), verbose=False)
        assert result == ""
        assert loaded == []

    def test_empty_file_included(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("", encoding="utf-8")
        result, loaded = load_instructions(str(tmp_path), verbose=False)
        assert "<project-instructions>" in result
        assert "</project-instructions>" in result

    def test_non_utf8_replaced(self, tmp_path):
        # Write raw bytes with invalid UTF-8
        (tmp_path / "CLAUDE.md").write_bytes(b"hello \xff\xfe world")
        result, loaded = load_instructions(str(tmp_path), verbose=False)
        assert "<project-instructions>" in result
        assert "hello" in result
        assert "world" in result
        assert "\ufffd" in result  # replacement character


# ---------------------------------------------------------------------------
# Integration (mock-based)
# ---------------------------------------------------------------------------


def _make_message(content=None, tool_calls=None):
    msg = types.SimpleNamespace()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.role = "assistant"
    msg.get = lambda key, default=None: getattr(msg, key, default)
    return msg


class TestIntegration:
    def test_instructions_in_system_message(self, tmp_path, monkeypatch):
        from swival import agent

        (tmp_path / "CLAUDE.md").write_text("Always use snake_case.", encoding="utf-8")

        captured_messages = {}

        def fake_call_llm(*args, **kwargs):
            captured_messages["messages"] = args[2]  # messages arg
            return _make_message(content="Done."), "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--base-dir",
                str(tmp_path),
            ],
        )

        agent.main()

        msgs = captured_messages["messages"]
        system_msg = msgs[0]
        assert system_msg["role"] == "system"
        assert "<project-instructions>" in system_msg["content"]
        assert "Always use snake_case." in system_msg["content"]

    def test_no_system_prompt_flag_skips_instructions(self, tmp_path, monkeypatch):
        from swival import agent

        (tmp_path / "CLAUDE.md").write_text("Should not appear.", encoding="utf-8")

        captured_messages = {}

        def fake_call_llm(*args, **kwargs):
            captured_messages["messages"] = args[2]
            return _make_message(content="Done."), "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--base-dir",
                str(tmp_path),
                "--no-system-prompt",
            ],
        )

        agent.main()

        msgs = captured_messages["messages"]
        # No system message at all
        assert all(
            (m["role"] if isinstance(m, dict) else m.role) != "system" for m in msgs
        )

    def test_custom_system_prompt_skips_instructions(self, tmp_path, monkeypatch):
        from swival import agent

        (tmp_path / "CLAUDE.md").write_text("Should not appear.", encoding="utf-8")

        captured_messages = {}

        def fake_call_llm(*args, **kwargs):
            captured_messages["messages"] = args[2]
            return _make_message(content="Done."), "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--base-dir",
                str(tmp_path),
                "--system-prompt",
                "Custom prompt only.",
            ],
        )

        agent.main()

        msgs = captured_messages["messages"]
        system_msg = msgs[0]
        assert system_msg["role"] == "system"
        assert system_msg["content"].startswith("Custom prompt only.")
        assert "project-instructions" not in system_msg["content"]
        assert "Should not appear" not in system_msg["content"]

    def test_no_instructions_flag(self, tmp_path, monkeypatch):
        from swival import agent

        (tmp_path / "CLAUDE.md").write_text("Should not appear.", encoding="utf-8")

        captured_messages = {}

        def fake_call_llm(*args, **kwargs):
            captured_messages["messages"] = args[2]
            return _make_message(content="Done."), "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--base-dir",
                str(tmp_path),
                "--no-instructions",
            ],
        )

        agent.main()

        msgs = captured_messages["messages"]
        system_msg = msgs[0]
        assert system_msg["role"] == "system"
        assert "project-instructions" not in system_msg["content"]
        assert "Should not appear" not in system_msg["content"]

    def test_custom_system_prompt_with_commands_appends_run_command(
        self, tmp_path, monkeypatch
    ):
        from swival import agent
        import shutil

        (tmp_path / "CLAUDE.md").write_text("Should not appear.", encoding="utf-8")

        captured_messages = {}

        def fake_call_llm(*args, **kwargs):
            captured_messages["messages"] = args[2]
            return _make_message(content="Done."), "stop"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        # Find a command that exists and is outside tmp_path
        ls_path = shutil.which("ls")
        if ls_path is None:
            pytest.skip("ls not found on PATH")

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "agent",
                "hello",
                "--base-dir",
                str(tmp_path),
                "--system-prompt",
                "Custom prompt.",
                "--allowed-commands",
                "ls",
            ],
        )

        agent.main()

        msgs = captured_messages["messages"]
        system_msg = msgs[0]
        assert system_msg["role"] == "system"
        # run_command section is appended even with custom prompt
        assert "run_command" in system_msg["content"]
        # But CLAUDE.md instructions are NOT appended
        assert "project-instructions" not in system_msg["content"]
        assert "Should not appear" not in system_msg["content"]

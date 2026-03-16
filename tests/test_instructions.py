"""Tests for CLAUDE.md / AGENTS.md instruction loading."""

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
        assert loaded == [str(tmp_path / "CLAUDE.md")]

    def test_agent_md_only(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("Be concise.", encoding="utf-8")
        result, loaded = load_instructions(str(tmp_path), verbose=False)
        assert "<agent-instructions>" in result
        assert "Be concise." in result
        assert "</agent-instructions>" in result
        assert "project-instructions" not in result
        assert loaded == [str(tmp_path / "AGENTS.md")]

    def test_both_files(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("Project rules.", encoding="utf-8")
        (tmp_path / "AGENTS.md").write_text("Agent rules.", encoding="utf-8")
        result, loaded = load_instructions(str(tmp_path), verbose=False)
        assert "<project-instructions>" in result
        assert "<agent-instructions>" in result
        # project-instructions comes first
        pi = result.index("<project-instructions>")
        ai = result.index("<agent-instructions>")
        assert pi < ai
        assert loaded == [str(tmp_path / "CLAUDE.md"), str(tmp_path / "AGENTS.md")]


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


# ---------------------------------------------------------------------------
# User-level AGENTS.md
# ---------------------------------------------------------------------------


class TestUserLevelAgentsMd:
    """Tests for loading user-level AGENTS.md from the config directory."""

    def test_user_level_only(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "AGENTS.md").write_text("User rules.", encoding="utf-8")

        project_dir = tmp_path / "project"
        project_dir.mkdir()

        result, loaded = load_instructions(str(project_dir), config_dir, verbose=False)
        assert "<agent-instructions>" in result
        assert "User rules." in result
        assert f"<!-- user: {config_dir / 'AGENTS.md'} -->" in result
        assert "<!-- project:" not in result
        assert loaded == [str(config_dir / "AGENTS.md")]

    def test_project_level_only_no_config_dir(self, tmp_path):
        """No config_dir passed — behaves like before."""
        (tmp_path / "AGENTS.md").write_text("Project rules.", encoding="utf-8")
        result, loaded = load_instructions(str(tmp_path), verbose=False)
        assert "<agent-instructions>" in result
        assert "Project rules." in result
        assert f"<!-- project: {tmp_path / 'AGENTS.md'} -->" in result
        assert loaded == [str(tmp_path / "AGENTS.md")]

    def test_both_levels(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "AGENTS.md").write_text("User conventions.", encoding="utf-8")

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "AGENTS.md").write_text("Project conventions.", encoding="utf-8")

        result, loaded = load_instructions(str(project_dir), config_dir, verbose=False)
        assert "<agent-instructions>" in result
        assert "User conventions." in result
        assert "Project conventions." in result
        # User-level comes first
        assert result.index("User conventions.") < result.index("Project conventions.")
        assert loaded == [
            str(config_dir / "AGENTS.md"),
            str(project_dir / "AGENTS.md"),
        ]

    def test_neither_level(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        result, loaded = load_instructions(str(project_dir), config_dir, verbose=False)
        assert result == ""
        assert loaded == []

    def test_user_level_exhausts_budget(self, tmp_path):
        """A large user-level file starves project-level content."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "AGENTS.md").write_text(
            "U" * (MAX_INSTRUCTIONS_CHARS + 100), encoding="utf-8"
        )

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "AGENTS.md").write_text("Project content.", encoding="utf-8")

        result, loaded = load_instructions(str(project_dir), config_dir, verbose=False)
        assert "truncated" in result
        # Project content is not present (budget exhausted)
        assert "Project content." not in result
        assert loaded == [str(config_dir / "AGENTS.md")]

    def test_combined_exceeds_budget(self, tmp_path):
        """User-level fits, project-level gets truncated."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        user_content = "U" * (MAX_INSTRUCTIONS_CHARS - 100)
        (config_dir / "AGENTS.md").write_text(user_content, encoding="utf-8")

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "AGENTS.md").write_text("P" * 500, encoding="utf-8")

        result, loaded = load_instructions(str(project_dir), config_dir, verbose=False)
        # Both loaded, but project is truncated
        assert len(loaded) == 2
        assert "truncated" in result
        # User content intact
        assert user_content in result

    def test_no_instructions_skips_both(self, tmp_path):
        """build_system_prompt with no_instructions=True skips everything."""
        from swival.agent import build_system_prompt

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "AGENTS.md").write_text("User rules.", encoding="utf-8")
        (tmp_path / "AGENTS.md").write_text("Project rules.", encoding="utf-8")

        content, loaded = build_system_prompt(
            base_dir=str(tmp_path),
            system_prompt=None,
            no_system_prompt=False,
            no_instructions=True,
            no_memory=True,
            skills_catalog={},
            yolo=False,
            resolved_commands={},
            verbose=False,
            config_dir=config_dir,
        )
        assert "agent-instructions" not in content
        assert loaded == []

    def test_unreadable_user_agents_skipped(self, tmp_path, monkeypatch):
        """Unreadable user-level AGENTS.md is skipped; project-level still loads."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "AGENTS.md").write_text("unreachable", encoding="utf-8")

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "AGENTS.md").write_text("Project ok.", encoding="utf-8")

        original_open = open

        def bad_open(self, *args, **kwargs):
            if str(self).startswith(str(config_dir)) and self.name == "AGENTS.md":
                raise PermissionError("Permission denied")
            return original_open(self, *args, **kwargs)

        monkeypatch.setattr("pathlib.Path.open", bad_open)
        result, loaded = load_instructions(str(project_dir), config_dir, verbose=False)
        assert "Project ok." in result
        assert loaded == [str(project_dir / "AGENTS.md")]

    def test_empty_user_agents_no_empty_block(self, tmp_path):
        """An empty user-level file doesn't produce an empty comment block."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "AGENTS.md").write_text("", encoding="utf-8")

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "AGENTS.md").write_text("Project only.", encoding="utf-8")

        result, loaded = load_instructions(str(project_dir), config_dir, verbose=False)
        assert "Project only." in result
        # Both files loaded (empty is still loaded, consistent with CLAUDE.md behavior)
        assert len(loaded) == 2

    def test_boundary_formatting(self, tmp_path):
        """Exact boundary text: provenance comments, separator, and wrapping."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "AGENTS.md").write_text("User stuff.", encoding="utf-8")

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "AGENTS.md").write_text("Project stuff.", encoding="utf-8")

        result, loaded = load_instructions(str(project_dir), config_dir, verbose=False)

        user_path = config_dir / "AGENTS.md"
        proj_path = project_dir / "AGENTS.md"

        # Exact provenance comments
        assert f"<!-- user: {user_path} -->" in result
        assert f"<!-- project: {proj_path} -->" in result

        # Separated by exactly one blank line (\n\n)
        user_block = f"<!-- user: {user_path} -->\nUser stuff."
        proj_block = f"<!-- project: {proj_path} -->\nProject stuff."
        assert f"{user_block}\n\n{proj_block}" in result

        # Wrapped in a single agent-instructions tag
        assert result.count("<agent-instructions>") == 1
        assert result.count("</agent-instructions>") == 1


# ---------------------------------------------------------------------------
# Global cross-agent AGENTS.md (~/.agents/AGENTS.md)
# ---------------------------------------------------------------------------


class TestGlobalAgentsMd:
    """Tests for loading ~/.agents/AGENTS.md."""

    def _set_global_path(self, monkeypatch, path):
        monkeypatch.setattr("swival.agent._global_agents_md_path", lambda: path)

    def test_global_only(self, tmp_path, monkeypatch):
        global_dir = tmp_path / ".agents"
        global_dir.mkdir()
        (global_dir / "AGENTS.md").write_text("Global rules.", encoding="utf-8")
        self._set_global_path(monkeypatch, global_dir / "AGENTS.md")

        project_dir = tmp_path / "project"
        project_dir.mkdir()

        result, loaded = load_instructions(str(project_dir), verbose=False)
        assert "<agent-instructions>" in result
        assert "Global rules." in result
        assert f"<!-- global: {global_dir / 'AGENTS.md'} -->" in result
        assert loaded == [str(global_dir / "AGENTS.md")]

    def test_all_three_levels(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "AGENTS.md").write_text("User rules.", encoding="utf-8")

        global_dir = tmp_path / ".agents"
        global_dir.mkdir()
        (global_dir / "AGENTS.md").write_text("Global rules.", encoding="utf-8")
        self._set_global_path(monkeypatch, global_dir / "AGENTS.md")

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "AGENTS.md").write_text("Project rules.", encoding="utf-8")

        result, loaded = load_instructions(str(project_dir), config_dir, verbose=False)
        assert "User rules." in result
        assert "Global rules." in result
        assert "Project rules." in result
        # Order: user → global → project
        assert result.index("User rules.") < result.index("Global rules.")
        assert result.index("Global rules.") < result.index("Project rules.")
        assert loaded == [
            str(config_dir / "AGENTS.md"),
            str(global_dir / "AGENTS.md"),
            str(project_dir / "AGENTS.md"),
        ]

    def test_global_and_project_no_user(self, tmp_path, monkeypatch):
        global_dir = tmp_path / ".agents"
        global_dir.mkdir()
        (global_dir / "AGENTS.md").write_text("Global stuff.", encoding="utf-8")
        self._set_global_path(monkeypatch, global_dir / "AGENTS.md")

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "AGENTS.md").write_text("Project stuff.", encoding="utf-8")

        result, loaded = load_instructions(str(project_dir), verbose=False)
        assert "Global stuff." in result
        assert "Project stuff." in result
        assert result.index("Global stuff.") < result.index("Project stuff.")
        assert "<!-- user:" not in result

    def test_user_exhausts_budget_starves_global(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "AGENTS.md").write_text(
            "U" * (MAX_INSTRUCTIONS_CHARS + 100), encoding="utf-8"
        )

        global_dir = tmp_path / ".agents"
        global_dir.mkdir()
        (global_dir / "AGENTS.md").write_text("Global content.", encoding="utf-8")
        self._set_global_path(monkeypatch, global_dir / "AGENTS.md")

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "AGENTS.md").write_text("Project content.", encoding="utf-8")

        result, loaded = load_instructions(str(project_dir), config_dir, verbose=False)
        assert "truncated" in result
        assert "Global content." not in result
        assert "Project content." not in result
        assert loaded == [str(config_dir / "AGENTS.md")]

    def test_user_and_global_exhaust_budget(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "AGENTS.md").write_text(
            "U" * (MAX_INSTRUCTIONS_CHARS - 100), encoding="utf-8"
        )

        global_dir = tmp_path / ".agents"
        global_dir.mkdir()
        (global_dir / "AGENTS.md").write_text("G" * 500, encoding="utf-8")
        self._set_global_path(monkeypatch, global_dir / "AGENTS.md")

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "AGENTS.md").write_text("Project content.", encoding="utf-8")

        result, loaded = load_instructions(str(project_dir), config_dir, verbose=False)
        # User and global loaded, project starved
        assert len(loaded) == 2
        assert "Project content." not in result

    def test_global_unreadable_skipped(self, tmp_path, monkeypatch):
        global_dir = tmp_path / ".agents"
        global_dir.mkdir()
        (global_dir / "AGENTS.md").write_text("unreachable", encoding="utf-8")
        self._set_global_path(monkeypatch, global_dir / "AGENTS.md")

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "AGENTS.md").write_text("Project ok.", encoding="utf-8")

        original_open = open

        def bad_open(self, *args, **kwargs):
            if str(self) == str(global_dir / "AGENTS.md"):
                raise PermissionError("Permission denied")
            return original_open(self, *args, **kwargs)

        monkeypatch.setattr("pathlib.Path.open", bad_open)
        result, loaded = load_instructions(str(project_dir), verbose=False)
        assert "Project ok." in result
        assert loaded == [str(project_dir / "AGENTS.md")]

    def test_global_empty(self, tmp_path, monkeypatch):
        global_dir = tmp_path / ".agents"
        global_dir.mkdir()
        (global_dir / "AGENTS.md").write_text("", encoding="utf-8")
        self._set_global_path(monkeypatch, global_dir / "AGENTS.md")

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "AGENTS.md").write_text("Project only.", encoding="utf-8")

        result, loaded = load_instructions(str(project_dir), verbose=False)
        assert "Project only." in result
        assert len(loaded) == 2

    def test_no_instructions_skips_global(self, tmp_path, monkeypatch):
        from swival.agent import build_system_prompt

        global_dir = tmp_path / ".agents"
        global_dir.mkdir()
        (global_dir / "AGENTS.md").write_text("Global rules.", encoding="utf-8")
        self._set_global_path(monkeypatch, global_dir / "AGENTS.md")

        content, loaded = build_system_prompt(
            base_dir=str(tmp_path),
            system_prompt=None,
            no_system_prompt=False,
            no_instructions=True,
            no_memory=True,
            skills_catalog={},
            yolo=False,
            resolved_commands={},
            verbose=False,
            config_dir=None,
        )
        assert "agent-instructions" not in content
        assert loaded == []

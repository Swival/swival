"""Tests for the REPL command registry and discover_custom_commands()."""

from __future__ import annotations

import inspect
import re
import stat
import sys
from unittest.mock import patch

import pytest

from swival.input_commands import INPUT_COMMANDS


# -- registry / dispatcher guard -------------------------------------------


class TestRegistryMatchesDispatcher:
    """Ensure INPUT_COMMANDS stays in sync with the actual dispatcher."""

    def test_all_commands_accounted_for(self):
        from swival.agent import execute_input

        source = inspect.getsource(execute_input)

        dispatched: set[str] = set()

        # Pattern: cmd in ("/exit", "/quit") or cmd in ("/clear", "/new")
        for group in re.findall(r"cmd\s+in\s+\(([^)]+)\)", source):
            for cmd in re.findall(r'["\'](/[a-z-]+)["\']', group):
                dispatched.add(cmd)

        # Pattern: cmd == "/foo" or cmd == "!!"
        for cmd in re.findall(r'cmd\s*==\s*["\']([/!][a-z!-]+)["\']', source):
            dispatched.add(cmd)

        registry = set(INPUT_COMMANDS.keys())

        missing_from_registry = dispatched - registry
        missing_from_dispatcher = registry - dispatched
        assert not missing_from_registry, (
            f"Commands in dispatcher but not in INPUT_COMMANDS: {missing_from_registry}"
        )
        assert not missing_from_dispatcher, (
            f"Commands in INPUT_COMMANDS but not in dispatcher: {missing_from_dispatcher}"
        )


# -- help output ------------------------------------------------------------


class TestHelpOutput:
    def test_sorted_order(self):
        from swival.agent import _repl_help

        text = _repl_help()
        lines = [line.strip() for line in text.splitlines()]
        commands = [line.split()[0] for line in lines if line.startswith("/")]
        assert commands == sorted(commands)

    def test_contains_all_commands(self):
        from swival.agent import _repl_help

        text = _repl_help()
        for cmd in INPUT_COMMANDS:
            assert cmd in text


# -- discover_custom_commands -----------------------------------------------


class TestDiscoverCustomCommands:
    def test_executable_files(self, tmp_path):
        from swival.agent import discover_custom_commands

        cmd_dir = tmp_path / "commands"
        cmd_dir.mkdir()
        target = cmd_dir / "deploy"
        target.write_text("#!/bin/sh\necho hi")
        target.chmod(target.stat().st_mode | stat.S_IEXEC)

        with patch("swival.config.global_config_dir", return_value=tmp_path):
            result = discover_custom_commands()

        assert "deploy" in result

    def test_non_executable_text_included(self, tmp_path):
        from swival.agent import discover_custom_commands

        cmd_dir = tmp_path / "commands"
        cmd_dir.mkdir()
        (cmd_dir / "readme").write_text("a prompt template")

        with patch("swival.config.global_config_dir", return_value=tmp_path):
            result = discover_custom_commands()

        assert "readme" in result

    def test_non_executable_binary_excluded(self, tmp_path):
        from swival.agent import discover_custom_commands

        cmd_dir = tmp_path / "commands"
        cmd_dir.mkdir()
        (cmd_dir / "readme").write_bytes(b"\x00\x01binary data")

        with patch("swival.config.global_config_dir", return_value=tmp_path):
            result = discover_custom_commands()

        assert result == []

    def test_missing_commands_dir(self, tmp_path):
        from swival.agent import discover_custom_commands

        with patch("swival.config.global_config_dir", return_value=tmp_path):
            assert discover_custom_commands() == []

    def test_invalid_name_excluded(self, tmp_path):
        from swival.agent import discover_custom_commands

        cmd_dir = tmp_path / "commands"
        cmd_dir.mkdir()
        bad = cmd_dir / "bad name"
        bad.write_text("#!/bin/sh\necho hi")
        bad.chmod(bad.stat().st_mode | stat.S_IEXEC)

        with patch("swival.config.global_config_dir", return_value=tmp_path):
            assert discover_custom_commands() == []

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific")
    def test_stem_offered_for_extension_files(self, tmp_path):
        """A file like deploy.sh should offer 'deploy' as a command name."""
        from swival.agent import discover_custom_commands

        cmd_dir = tmp_path / "commands"
        cmd_dir.mkdir()
        cmd = cmd_dir / "build.sh"
        cmd.write_text("#!/bin/sh\necho build")
        cmd.chmod(cmd.stat().st_mode | stat.S_IEXEC)

        with patch("swival.config.global_config_dir", return_value=tmp_path):
            result = discover_custom_commands()

        # Stem "build" should be offered; "build.sh" has a dot so it fails regex
        assert "build" in result
        assert "build.sh" not in result

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific")
    def test_ambiguous_stem_excluded(self, tmp_path):
        """Two executables sharing a stem must not offer that stem."""
        from swival.agent import discover_custom_commands

        cmd_dir = tmp_path / "commands"
        cmd_dir.mkdir()
        for ext in (".sh", ".py"):
            cmd = cmd_dir / f"deploy{ext}"
            cmd.write_text("#!/bin/sh\necho hi")
            cmd.chmod(cmd.stat().st_mode | stat.S_IEXEC)

        with patch("swival.config.global_config_dir", return_value=tmp_path):
            result = discover_custom_commands()

        assert "deploy" not in result

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific")
    def test_exact_name_still_offered_despite_ambiguous_stem(self, tmp_path):
        """An extensionless executable is offered even if other files share its stem."""
        from swival.agent import discover_custom_commands

        cmd_dir = tmp_path / "commands"
        cmd_dir.mkdir()
        # Extensionless file: exact name match in runtime
        exact = cmd_dir / "deploy"
        exact.write_text("#!/bin/sh\necho exact")
        exact.chmod(exact.stat().st_mode | stat.S_IEXEC)
        # Another file sharing the stem
        other = cmd_dir / "deploy.sh"
        other.write_text("#!/bin/sh\necho other")
        other.chmod(other.stat().st_mode | stat.S_IEXEC)

        with patch("swival.config.global_config_dir", return_value=tmp_path):
            result = discover_custom_commands()

        # "deploy" is offered via exact name; stem ambiguity doesn't block it
        assert "deploy" in result

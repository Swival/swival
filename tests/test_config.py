"""Tests for swival.config — TOML config file loading, merging, and CLI integration."""

import argparse
import os
import tomllib
import types

import pytest

from swival.config import (
    _UNSET,
    ConfigError,
    global_config_dir,
    apply_config_to_args,
    config_to_session_kwargs,
    generate_config,
    load_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_toml(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_args(**overrides):
    """Build a namespace mimicking build_parser() with _UNSET sentinels."""
    defaults = {
        "provider": _UNSET,
        "model": _UNSET,
        "api_key": _UNSET,
        "base_url": _UNSET,
        "max_output_tokens": _UNSET,
        "max_context_tokens": _UNSET,
        "temperature": _UNSET,
        "top_p": _UNSET,
        "seed": _UNSET,
        "max_turns": _UNSET,
        "system_prompt": _UNSET,
        "no_system_prompt": _UNSET,
        "commands": _UNSET,
        "yolo": _UNSET,
        "files": _UNSET,
        "add_dir": None,  # append actions use None sentinel
        "add_dir_ro": None,  # append actions use None sentinel
        "sandbox": _UNSET,
        "sandbox_session": _UNSET,
        "no_read_guard": _UNSET,
        "no_instructions": _UNSET,
        "no_skills": _UNSET,
        "skills_dir": None,  # append action
        "no_history": _UNSET,
        "color": _UNSET,
        "no_color": _UNSET,
        "quiet": _UNSET,
        "reviewer": _UNSET,
        "review_prompt": _UNSET,
        "objective": _UNSET,
        "verify": _UNSET,
        "max_review_rounds": _UNSET,
        "cache": _UNSET,
        "cache_dir": _UNSET,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ===========================================================================
# Config loading
# ===========================================================================


class TestLoadConfig:
    def test_missing_files_returns_config_dir_only(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty_xdg"))
        result = load_config(tmp_path)
        assert "config_dir" in result
        # No user-set keys beyond config_dir
        assert set(result.keys()) == {"config_dir"}

    def test_global_only(self, tmp_path, monkeypatch):
        global_dir = tmp_path / "global_cfg"
        monkeypatch.setenv("XDG_CONFIG_HOME", str(global_dir))
        _write_toml(global_dir / "swival" / "config.toml", 'provider = "openrouter"\n')
        result = load_config(tmp_path / "project")
        assert result["provider"] == "openrouter"

    def test_project_only(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(tmp_path / "swival.toml", "max_turns = 42\n")
        result = load_config(tmp_path)
        assert result["max_turns"] == 42

    def test_project_overrides_global(self, tmp_path, monkeypatch):
        global_dir = tmp_path / "global"
        monkeypatch.setenv("XDG_CONFIG_HOME", str(global_dir))
        _write_toml(global_dir / "swival" / "config.toml", "max_turns = 10\n")
        _write_toml(tmp_path / "swival.toml", "max_turns = 50\n")
        result = load_config(tmp_path)
        assert result["max_turns"] == 50

    def test_unknown_keys_warn(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(tmp_path / "swival.toml", 'unknown_key = "hi"\n')
        result = load_config(tmp_path)
        assert "unknown_key" not in result
        assert "unknown config key" in capsys.readouterr().err

    def test_wrong_type_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(tmp_path / "swival.toml", 'max_turns = "not a number"\n')
        with pytest.raises(ConfigError, match="max_turns.*expected int.*got str"):
            load_config(tmp_path)

    def test_invalid_toml_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(tmp_path / "swival.toml", "invalid = [\n")
        with pytest.raises(ConfigError, match="invalid TOML"):
            load_config(tmp_path)

    def test_generate_config_is_valid_toml(self):
        content = generate_config()
        # Extract only the commented-out key=value lines (skip header comments)
        lines = []
        for line in content.splitlines():
            stripped = line.lstrip("# ").strip()
            if "=" in stripped and not stripped.startswith("--"):
                lines.append(stripped)
        # Should parse without error
        tomllib.loads("\n".join(lines))

    def test_generate_config_project_flag(self):
        content = generate_config(project=True)
        assert "Project config" in content
        content_global = generate_config(project=False)
        assert "Global config" in content_global


# ===========================================================================
# Type validation
# ===========================================================================


class TestTypeValidation:
    def test_string_where_int_expected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(tmp_path / "swival.toml", 'max_output_tokens = "big"\n')
        with pytest.raises(
            ConfigError, match="max_output_tokens.*expected int.*got str"
        ):
            load_config(tmp_path)

    def test_mixed_type_list(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(tmp_path / "swival.toml", 'commands = ["ls", 42]\n')
        with pytest.raises(
            ConfigError, match=r"commands\[1\].*expected string.*got int"
        ):
            load_config(tmp_path)

    def test_empty_list_is_valid(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(tmp_path / "swival.toml", "commands = []\n")
        result = load_config(tmp_path)
        assert result["commands"] == []

    def test_toml_int_for_float_field(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(tmp_path / "swival.toml", "temperature = 1\n")
        result = load_config(tmp_path)
        assert result["temperature"] == 1

    def test_bool_for_string_field_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(tmp_path / "swival.toml", "provider = true\n")
        with pytest.raises(ConfigError, match="provider.*expected str.*got bool"):
            load_config(tmp_path)

    def test_bool_for_int_field_raises(self, tmp_path, monkeypatch):
        """bool is subclass of int in Python — config must reject it explicitly."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(tmp_path / "swival.toml", "max_turns = true\n")
        with pytest.raises(ConfigError, match="max_turns.*expected int.*got bool"):
            load_config(tmp_path)

    def test_bool_for_float_field_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(tmp_path / "swival.toml", "temperature = false\n")
        with pytest.raises(ConfigError, match="temperature.*got bool"):
            load_config(tmp_path)


# ===========================================================================
# Mutual exclusion
# ===========================================================================


class TestMutualExclusion:
    def test_both_system_prompt_and_no_system_prompt(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(
            tmp_path / "swival.toml",
            'system_prompt = "hello"\nno_system_prompt = true\n',
        )
        with pytest.raises(ConfigError, match="mutually exclusive"):
            load_config(tmp_path)

    def test_system_prompt_alone(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(tmp_path / "swival.toml", 'system_prompt = "hello"\n')
        result = load_config(tmp_path)
        assert result["system_prompt"] == "hello"

    def test_no_system_prompt_alone(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(tmp_path / "swival.toml", "no_system_prompt = true\n")
        result = load_config(tmp_path)
        assert result["no_system_prompt"] is True

    def test_cross_file_conflict(self, tmp_path, monkeypatch):
        global_dir = tmp_path / "global"
        monkeypatch.setenv("XDG_CONFIG_HOME", str(global_dir))
        _write_toml(global_dir / "swival" / "config.toml", 'system_prompt = "hello"\n')
        _write_toml(tmp_path / "swival.toml", "no_system_prompt = true\n")
        with pytest.raises(ConfigError, match="mutually exclusive"):
            load_config(tmp_path)


# ===========================================================================
# Path resolution
# ===========================================================================


class TestPathResolution:
    def test_relative_allowed_dirs_resolves_to_config_parent(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(tmp_path / "swival.toml", 'allowed_dirs = ["../sibling"]\n')
        result = load_config(tmp_path)
        expected = str(tmp_path / "../sibling")
        assert result["allowed_dirs"] == [expected]

    def test_absolute_path_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(tmp_path / "swival.toml", 'allowed_dirs = ["/absolute/path"]\n')
        result = load_config(tmp_path)
        assert result["allowed_dirs"] == ["/absolute/path"]

    def test_home_expansion(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(tmp_path / "swival.toml", 'allowed_dirs = ["~/projects"]\n')
        result = load_config(tmp_path)
        home = os.path.expanduser("~")
        assert result["allowed_dirs"] == [f"{home}/projects"]

    def test_global_paths_resolve_against_global_dir(self, tmp_path, monkeypatch):
        global_dir = tmp_path / "global"
        monkeypatch.setenv("XDG_CONFIG_HOME", str(global_dir))
        _write_toml(
            global_dir / "swival" / "config.toml", 'skills_dir = ["../../extra"]\n'
        )
        result = load_config(tmp_path / "project")
        expected = str(global_dir / "swival" / "../../extra")
        assert result["skills_dir"] == [expected]

    def test_reviewer_relative_resolves(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(tmp_path / "swival.toml", 'reviewer = "./review.sh"\n')
        result = load_config(tmp_path)
        assert result["reviewer"] == str(tmp_path / "./review.sh")

    def test_reviewer_home_expansion(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(tmp_path / "swival.toml", 'reviewer = "~/bin/review.sh"\n')
        result = load_config(tmp_path)
        home = os.path.expanduser("~")
        assert result["reviewer"] == f"{home}/bin/review.sh"

    def test_allowed_dirs_ro_relative_resolves(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(tmp_path / "swival.toml", 'allowed_dirs_ro = ["../sibling-ro"]\n')
        result = load_config(tmp_path)
        expected = str(tmp_path / "../sibling-ro")
        assert result["allowed_dirs_ro"] == [expected]

    def test_allowed_dirs_ro_absolute_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(
            tmp_path / "swival.toml", 'allowed_dirs_ro = ["/absolute/readonly"]\n'
        )
        result = load_config(tmp_path)
        assert result["allowed_dirs_ro"] == ["/absolute/readonly"]

    def test_allowed_dirs_ro_home_expansion(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(tmp_path / "swival.toml", 'allowed_dirs_ro = ["~/datasets"]\n')
        result = load_config(tmp_path)
        home = os.path.expanduser("~")
        assert result["allowed_dirs_ro"] == [f"{home}/datasets"]

    def test_path_resolution_after_type_validation(self, tmp_path, monkeypatch):
        """Type validation runs before path resolution — bad types don't crash Path()."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(tmp_path / "swival.toml", "allowed_dirs = [42]\n")
        with pytest.raises(ConfigError, match=r"allowed_dirs\[0\].*expected string"):
            load_config(tmp_path)


# ===========================================================================
# apply_config_to_args
# ===========================================================================


class TestApplyConfigToArgs:
    def test_config_fills_unset(self):
        args = _make_args()
        apply_config_to_args(args, {"max_turns": 42, "provider": "openrouter"})
        assert args.max_turns == 42
        assert args.provider == "openrouter"

    def test_cli_beats_config(self):
        args = _make_args(max_turns=200)
        apply_config_to_args(args, {"max_turns": 42})
        assert args.max_turns == 200

    def test_sentinel_resolves_to_default(self):
        args = _make_args()
        apply_config_to_args(args, {})
        assert args.max_turns == 100
        assert args.provider == "lmstudio"
        assert args.yolo is False
        assert args.quiet is False

    def test_store_true_absent_plus_config_true(self):
        args = _make_args()
        apply_config_to_args(args, {"yolo": True})
        assert args.yolo is True

    def test_store_true_flag_present_beats_config(self):
        args = _make_args(yolo=True)
        apply_config_to_args(args, {"yolo": False})
        assert args.yolo is True

    def test_color_config_true(self):
        args = _make_args()
        apply_config_to_args(args, {"color": True})
        assert args.color is True
        assert args.no_color is False

    def test_color_config_false(self):
        args = _make_args()
        apply_config_to_args(args, {"color": False})
        assert args.color is False
        assert args.no_color is True

    def test_color_cli_overrides_config(self):
        args = _make_args(color=True)
        apply_config_to_args(args, {"color": False})
        assert args.color is True  # CLI wins

    def test_no_color_cli_overrides_config(self):
        args = _make_args(no_color=True)
        apply_config_to_args(args, {"color": True})
        assert args.no_color is True  # CLI wins

    def test_allowed_dirs_maps_to_add_dir(self):
        args = _make_args()
        apply_config_to_args(args, {"allowed_dirs": ["/foo", "/bar"]})
        assert args.add_dir == ["/foo", "/bar"]

    def test_allowed_dirs_cli_overrides(self):
        args = _make_args(add_dir=["/cli-dir"])
        apply_config_to_args(args, {"allowed_dirs": ["/config-dir"]})
        assert args.add_dir == ["/cli-dir"]

    def test_skills_dir_from_config(self):
        args = _make_args()
        apply_config_to_args(args, {"skills_dir": ["/extra"]})
        assert args.skills_dir == ["/extra"]

    def test_skills_dir_cli_overrides(self):
        args = _make_args(skills_dir=["/from-cli"])
        apply_config_to_args(args, {"skills_dir": ["/from-config"]})
        assert args.skills_dir == ["/from-cli"]

    def test_allowed_dirs_ro_maps_to_add_dir_ro(self):
        args = _make_args()
        apply_config_to_args(args, {"allowed_dirs_ro": ["/ro1", "/ro2"]})
        assert args.add_dir_ro == ["/ro1", "/ro2"]

    def test_allowed_dirs_ro_cli_overrides(self):
        args = _make_args(add_dir_ro=["/cli-ro"])
        apply_config_to_args(args, {"allowed_dirs_ro": ["/config-ro"]})
        assert args.add_dir_ro == ["/cli-ro"]

    def test_none_sentinel_defaults_to_empty_list(self):
        """append-action dests (add_dir, add_dir_ro, skills_dir) default to [] when unset."""
        args = _make_args()
        apply_config_to_args(args, {})
        assert args.add_dir == []
        assert args.add_dir_ro == []
        assert args.skills_dir == []


# ===========================================================================
# config_to_session_kwargs
# ===========================================================================


class TestConfigToSessionKwargs:
    def test_identity_keys(self):
        kwargs = config_to_session_kwargs({"provider": "openrouter", "max_turns": 50})
        assert kwargs == {"provider": "openrouter", "max_turns": 50}

    def test_inverted_keys(self):
        kwargs = config_to_session_kwargs(
            {
                "no_read_guard": True,
                "no_history": False,
                "no_memory": True,
                "quiet": True,
            }
        )
        assert kwargs["read_guard"] is False
        assert kwargs["history"] is True
        assert kwargs["memory"] is False
        assert kwargs["verbose"] is False

    def test_dropped_keys(self):
        kwargs = config_to_session_kwargs(
            {"color": True, "reviewer": "./review.sh", "provider": "lmstudio"}
        )
        assert "color" not in kwargs
        assert "reviewer" not in kwargs
        assert kwargs["provider"] == "lmstudio"

    def test_accepted_by_session(self):
        from swival.session import Session

        config = {
            "provider": "lmstudio",
            "model": "test",
            "max_turns": 10,
            "no_read_guard": True,
            "no_history": True,
            "quiet": False,
        }
        kwargs = config_to_session_kwargs(config)
        session = Session(**kwargs)
        assert session.provider == "lmstudio"
        assert session.read_guard is False
        assert session.history is False


# ===========================================================================
# resolve_commands accepts both types
# ===========================================================================


class TestResolveCommandsTypes:
    def test_list_input(self):
        from swival.agent import resolve_commands

        result = resolve_commands(["ls"], "/tmp")
        assert "ls" in result


# ===========================================================================
# _report_settings handles both types
# ===========================================================================


class TestReportSettingsTypes:
    def test_string_commands(self):
        args = types.SimpleNamespace(
            temperature=0.5,
            top_p=1.0,
            seed=None,
            max_turns=10,
            max_output_tokens=1024,
            max_context_tokens=None,
            files_mode="some",
            commands="ls,git",
        )
        # Reproduce the logic from _report_settings
        cmds = args.commands
        if isinstance(cmds, list):
            cmd_list = sorted(cmds)
        elif cmds:
            cmd_list = sorted(c.strip() for c in cmds.split(",") if c.strip())
        else:
            cmd_list = []
        assert cmd_list == ["git", "ls"]

    def test_list_commands(self):
        cmds = ["git", "ls"]
        if isinstance(cmds, list):
            cmd_list = sorted(cmds)
        else:
            cmd_list = []
        assert cmd_list == ["git", "ls"]


# ===========================================================================
# --init-config
# ===========================================================================


class TestInitConfig:
    def test_writes_global_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

        dest = global_config_dir() / "config.toml"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(generate_config(project=False), encoding="utf-8")
        assert dest.exists()
        content = dest.read_text()
        assert "provider" in content

    def test_writes_project_config(self, tmp_path):
        from swival.config import generate_config

        dest = tmp_path / "swival.toml"
        dest.write_text(generate_config(project=True), encoding="utf-8")
        assert dest.exists()
        assert "Project config" in dest.read_text()

    def test_refuses_overwrite(self, tmp_path, monkeypatch):
        """_handle_init_config exits non-zero if config already exists."""
        from swival.agent import _handle_init_config

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        cfg = tmp_path / "xdg" / "swival" / "config.toml"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text("existing\n")

        args = types.SimpleNamespace(project=False, base_dir=".")
        with pytest.raises(SystemExit):
            _handle_init_config(args)


# ===========================================================================
# Security: api_key warning
# ===========================================================================


class TestApiKeyWarning:
    def test_api_key_in_git_repo_warns(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        # Create a fake git repo
        (tmp_path / ".git").mkdir()
        _write_toml(tmp_path / "swival.toml", 'api_key = "sk-secret"\n')
        load_config(tmp_path)
        assert "api_key" in capsys.readouterr().err

    def test_api_key_without_git_no_warning(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(tmp_path / "swival.toml", 'api_key = "sk-secret"\n')
        load_config(tmp_path)
        assert "api_key" not in capsys.readouterr().err


# ===========================================================================
# XDG_CONFIG_HOME
# ===========================================================================


class TestGlobalConfigDir:
    def test_respects_xdg(self, monkeypatch):
        from pathlib import Path

        monkeypatch.setenv("XDG_CONFIG_HOME", "/custom/xdg")
        assert global_config_dir() == Path("/custom/xdg/swival")

    def test_default_home(self, monkeypatch):
        from pathlib import Path

        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        expected = Path.home() / ".config" / "swival"
        assert global_config_dir() == expected


# ===========================================================================
# Integration: full CLI → config → resolution
# ===========================================================================


class TestCLIIntegration:
    def test_parse_load_apply(self, tmp_path, monkeypatch):
        """Full flow: parse args → load config → apply → check resolved values."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(tmp_path / "swival.toml", "max_turns = 42\nyolo = true\n")

        from swival.agent import build_parser

        parser = build_parser()
        args = parser.parse_args(["--base-dir", str(tmp_path), "question"])

        config = load_config(tmp_path)
        apply_config_to_args(args, config)

        assert args.max_turns == 42
        assert args.yolo is True
        assert args.provider == "lmstudio"  # default

    def test_cli_flag_overrides_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(tmp_path / "swival.toml", "max_turns = 42\n")

        from swival.agent import build_parser

        parser = build_parser()
        args = parser.parse_args(["--max-turns", "200", "question"])

        config = load_config(tmp_path)
        apply_config_to_args(args, config)

        assert args.max_turns == 200  # CLI wins

    def test_help_lists_all_cli_flags(self):
        from swival.agent import build_parser

        parser = build_parser()
        help_text = parser.format_help()

        option_strings = [
            option
            for action in parser._actions
            for option in action.option_strings
            if option.startswith("-")
        ]

        missing = [option for option in option_strings if option not in help_text]
        assert missing == []

    def test_help_uses_grouped_sections(self):
        from swival.agent import build_parser

        parser = build_parser()
        help_text = parser.format_help()

        for heading in (
            "Task input:",
            "Modes:",
            "Provider and model:",
            "Filesystem and command access:",
            "Prompt, instructions, memory, and skills:",
            "Review and reporting:",
            "Output and setup:",
        ):
            assert heading in help_text

    def test_help_includes_examples(self):
        from swival.agent import build_parser

        parser = build_parser()
        help_text = parser.format_help()

        assert "Examples:" in help_text
        assert "swival -q < task.md" in help_text
        assert "--provider huggingface --model zai-org/GLM-5" in help_text
        assert "swival --yolo --repl" in help_text
        assert "--self-review" in help_text

    def test_commands_list_flows_through(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(tmp_path / "swival.toml", 'commands = ["ls"]\n')

        config = load_config(tmp_path)
        args = _make_args()
        apply_config_to_args(args, config)
        assert args.commands == ["ls"]

        from swival.agent import resolve_commands

        result = resolve_commands(args.commands, str(tmp_path))
        assert "ls" in result

    def test_malformed_toml_clear_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(tmp_path / "swival.toml", "bad syntax {{{")
        with pytest.raises(ConfigError, match="invalid TOML"):
            load_config(tmp_path)

    def test_config_error_surfaces_as_parser_error(self, tmp_path, monkeypatch):
        """Invalid config produces a clean argparse-style error, not a traceback."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(tmp_path / "swival.toml", 'max_turns = "oops"\n')

        from unittest.mock import MagicMock, patch

        from swival import agent

        mock_parser = MagicMock()
        mock_args = types.SimpleNamespace(
            version=False,
            base_dir=str(tmp_path),
            init_config=False,
            project=False,
            reviewer_mode=False,
        )
        mock_parser.parse_args.return_value = mock_args
        mock_parser.error.side_effect = SystemExit(2)

        with patch.object(agent, "build_parser", return_value=mock_parser):
            with pytest.raises(SystemExit):
                agent.main()

        mock_parser.error.assert_called_once()
        assert "max_turns" in mock_parser.error.call_args[0][0]


# ===========================================================================
# max_review_rounds config integration
# ===========================================================================


class TestMaxReviewRoundsConfig:
    def test_default_value(self):
        args = _make_args()
        apply_config_to_args(args, {})
        assert args.max_review_rounds == 15

    def test_config_fills_unset(self):
        args = _make_args()
        apply_config_to_args(args, {"max_review_rounds": 10})
        assert args.max_review_rounds == 10

    def test_cli_beats_config(self):
        args = _make_args(max_review_rounds=10)
        apply_config_to_args(args, {"max_review_rounds": 3})
        assert args.max_review_rounds == 10

    def test_project_overrides_global(self, tmp_path, monkeypatch):
        xdg = tmp_path / "xdg"
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
        _write_toml(xdg / "swival" / "config.toml", "max_review_rounds = 3\n")
        _write_toml(tmp_path / "project" / "swival.toml", "max_review_rounds = 7\n")

        config = load_config(tmp_path / "project")
        assert config["max_review_rounds"] == 7

    def test_global_used_when_no_project(self, tmp_path, monkeypatch):
        xdg = tmp_path / "xdg"
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
        _write_toml(xdg / "swival" / "config.toml", "max_review_rounds = 3\n")

        config = load_config(tmp_path / "project")
        assert config["max_review_rounds"] == 3

    def test_cli_overrides_project_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(tmp_path / "swival.toml", "max_review_rounds = 7\n")

        from swival.agent import build_parser

        parser = build_parser()
        args = parser.parse_args(["--max-review-rounds", "10", "question"])

        config = load_config(tmp_path)
        apply_config_to_args(args, config)

        assert args.max_review_rounds == 10

    def test_dropped_from_session_kwargs(self):
        kwargs = config_to_session_kwargs(
            {"max_review_rounds": 5, "provider": "lmstudio"}
        )
        assert "max_review_rounds" not in kwargs
        assert kwargs["provider"] == "lmstudio"

    def test_negative_value_rejected_post_merge(self, tmp_path, monkeypatch):
        """Negative max_review_rounds in toml is rejected after config merge."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(tmp_path / "swival.toml", "max_review_rounds = -1\n")

        from swival.agent import build_parser

        parser = build_parser()
        args = parser.parse_args(["--base-dir", str(tmp_path), "question"])

        config = load_config(tmp_path)
        apply_config_to_args(args, config)

        assert args.max_review_rounds == -1  # config merged fine

        # But main() should reject it via parser.error
        from unittest.mock import MagicMock, patch
        from swival import agent

        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = args
        mock_parser.error.side_effect = SystemExit(2)

        with patch.object(agent, "build_parser", return_value=mock_parser):
            with pytest.raises(SystemExit):
                agent.main()

        mock_parser.error.assert_called_once()
        assert "max-review-rounds" in mock_parser.error.call_args[0][0]

    def test_in_generate_config(self):
        content = generate_config()
        assert "max_review_rounds" in content


class TestExtraBody:
    """Tests for extra_body config, CLI, and Session pass-through."""

    def test_config_loads_extra_body_dict(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        project = tmp_path / "proj"
        project.mkdir()
        _write_toml(
            project / "swival.toml",
            "extra_body = { chat_template_kwargs = { enable_thinking = false } }\n",
        )
        result = load_config(project)
        assert result["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }

    def test_config_rejects_non_dict_extra_body(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        project = tmp_path / "proj"
        project.mkdir()
        _write_toml(project / "swival.toml", "extra_body = 42\n")
        with pytest.raises(ConfigError, match="extra_body.*expected dict"):
            load_config(project)

    def test_extra_body_does_not_capture_later_keys(self, tmp_path, monkeypatch):
        """Inline extra_body must not swallow keys that follow it."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        project = tmp_path / "proj"
        project.mkdir()
        _write_toml(
            project / "swival.toml",
            "extra_body = { top_k = 20 }\nmax_turns = 5\n",
        )
        result = load_config(project)
        assert result["max_turns"] == 5
        assert result["extra_body"] == {"top_k": 20}

    def test_apply_config_to_args_extra_body(self):
        args = _make_args(extra_body=_UNSET, proactive_summaries=_UNSET, no_mcp=_UNSET)
        config = {"extra_body": {"top_k": 20}}
        apply_config_to_args(args, config)
        assert args.extra_body == {"top_k": 20}

    def test_apply_config_to_args_extra_body_default_none(self):
        args = _make_args(extra_body=_UNSET, proactive_summaries=_UNSET, no_mcp=_UNSET)
        apply_config_to_args(args, {})
        assert args.extra_body is None

    def test_config_to_session_kwargs_passes_extra_body(self):
        kwargs = config_to_session_kwargs({"extra_body": {"top_k": 20}})
        assert kwargs["extra_body"] == {"top_k": 20}

    def test_cli_rejects_non_object_json(self):
        from swival.agent import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--extra-body", "42"])

    def test_cli_rejects_json_array(self):
        from swival.agent import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--extra-body", "[1, 2]"])

    def test_cli_accepts_json_object(self):
        from swival.agent import build_parser

        parser = build_parser()
        ns = parser.parse_args(["--extra-body", '{"top_k": 20}', "hello"])
        assert ns.extra_body == {"top_k": 20}

    def test_session_extra_body_into_llm_kwargs(self):
        """Session should inject extra_body into _llm_kwargs during _setup."""
        from unittest.mock import patch, MagicMock

        from swival.session import Session

        sess = Session(
            provider="generic",
            model="test-model",
            base_url="http://localhost:8000",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        mock_provider = MagicMock(
            return_value=("test-model", "http://localhost:8000", None, None, {})
        )
        with (
            patch("swival.agent.resolve_provider", mock_provider),
            patch("swival.agent.resolve_commands", return_value={}),
            patch("swival.agent.build_tools", return_value=[]),
            patch("swival.agent.build_system_prompt", return_value=(None, [])),
            patch("swival.skills.discover_skills", return_value={}),
            patch("swival.agent.cleanup_old_cmd_outputs"),
        ):
            sess._setup()

        assert sess._llm_kwargs["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }

    def test_session_empty_dict_extra_body_forwarded(self):
        """An explicit empty dict should still be set in _llm_kwargs."""
        from unittest.mock import patch, MagicMock

        from swival.session import Session

        sess = Session(
            provider="generic",
            model="test-model",
            base_url="http://localhost:8000",
            extra_body={},
        )
        mock_provider = MagicMock(
            return_value=("test-model", "http://localhost:8000", None, None, {})
        )
        with (
            patch("swival.agent.resolve_provider", mock_provider),
            patch("swival.agent.resolve_commands", return_value={}),
            patch("swival.agent.build_tools", return_value=[]),
            patch("swival.agent.build_system_prompt", return_value=(None, [])),
            patch("swival.skills.discover_skills", return_value={}),
            patch("swival.agent.cleanup_old_cmd_outputs"),
        ):
            sess._setup()

        assert sess._llm_kwargs["extra_body"] == {}

    def test_call_llm_forwards_extra_body(self):
        """call_llm should include extra_body in litellm.completion kwargs."""
        from unittest.mock import patch, MagicMock

        from swival.agent import call_llm

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message="hi", finish_reason="stop")]

        with patch("litellm.completion", return_value=mock_response) as mock_comp:
            call_llm(
                "http://localhost:8000",
                "test-model",
                [{"role": "user", "content": "hi"}],
                1024,
                None,
                None,
                None,
                None,
                False,
                provider="generic",
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )

        _, kwargs = mock_comp.call_args
        assert kwargs["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }

    def test_call_llm_omits_extra_body_when_none(self):
        """call_llm should not include extra_body key when it is None."""
        from unittest.mock import patch, MagicMock

        from swival.agent import call_llm

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message="hi", finish_reason="stop")]

        with patch("litellm.completion", return_value=mock_response) as mock_comp:
            call_llm(
                "http://localhost:8000",
                "test-model",
                [{"role": "user", "content": "hi"}],
                1024,
                None,
                None,
                None,
                None,
                False,
                provider="generic",
            )

        _, kwargs = mock_comp.call_args
        assert "extra_body" not in kwargs

    def test_generate_config_extra_body_inline(self):
        """Template must use inline syntax, not a [extra_body] table header."""
        content = generate_config()
        assert "[extra_body]" not in content
        assert "extra_body" in content


class TestReasoningEffort:
    """Tests for reasoning_effort config, CLI, and pass-through."""

    def test_config_loads_reasoning_effort(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        project = tmp_path / "proj"
        project.mkdir()
        _write_toml(project / "swival.toml", 'reasoning_effort = "high"\n')
        result = load_config(project)
        assert result["reasoning_effort"] == "high"

    def test_config_rejects_invalid_reasoning_effort(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        project = tmp_path / "proj"
        project.mkdir()
        _write_toml(project / "swival.toml", 'reasoning_effort = "turbo"\n')
        with pytest.raises(ConfigError, match="reasoning_effort.*must be one of"):
            load_config(project)

    def test_config_rejects_non_str_reasoning_effort(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        project = tmp_path / "proj"
        project.mkdir()
        _write_toml(project / "swival.toml", "reasoning_effort = 42\n")
        with pytest.raises(ConfigError, match="reasoning_effort.*expected str"):
            load_config(project)

    def test_apply_config_to_args_reasoning_effort(self):
        args = _make_args(reasoning_effort=_UNSET)
        config = {"reasoning_effort": "medium"}
        apply_config_to_args(args, config)
        assert args.reasoning_effort == "medium"

    def test_apply_config_to_args_reasoning_effort_default_none(self):
        args = _make_args(reasoning_effort=_UNSET)
        apply_config_to_args(args, {})
        assert args.reasoning_effort is None

    def test_config_to_session_kwargs_passes_reasoning_effort(self):
        kwargs = config_to_session_kwargs({"reasoning_effort": "high"})
        assert kwargs["reasoning_effort"] == "high"

    def test_cli_accepts_valid_reasoning_effort(self):
        from swival.agent import build_parser

        parser = build_parser()
        ns = parser.parse_args(["--reasoning-effort", "low", "hello"])
        assert ns.reasoning_effort == "low"

    def test_cli_rejects_invalid_reasoning_effort(self):
        from swival.agent import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--reasoning-effort", "turbo", "hello"])

    def test_call_llm_forwards_reasoning_effort(self):
        from unittest.mock import patch, MagicMock

        from swival.agent import call_llm

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message="hi", finish_reason="stop")]

        with patch("litellm.completion", return_value=mock_response) as mock_comp:
            call_llm(
                "http://localhost:8000",
                "test-model",
                [{"role": "user", "content": "hi"}],
                1024,
                None,
                None,
                None,
                None,
                False,
                provider="generic",
                reasoning_effort="high",
            )

        _, kwargs = mock_comp.call_args
        assert kwargs["reasoning_effort"] == "high"

    def test_call_llm_omits_reasoning_effort_when_none(self):
        from unittest.mock import patch, MagicMock

        from swival.agent import call_llm

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message="hi", finish_reason="stop")]

        with patch("litellm.completion", return_value=mock_response) as mock_comp:
            call_llm(
                "http://localhost:8000",
                "test-model",
                [{"role": "user", "content": "hi"}],
                1024,
                None,
                None,
                None,
                None,
                False,
                provider="generic",
            )

        _, kwargs = mock_comp.call_args
        assert "reasoning_effort" not in kwargs

    def test_in_generate_config(self):
        content = generate_config()
        assert "reasoning_effort" in content


# ===========================================================================
# Serve skills validation
# ===========================================================================


class TestServeSkills:
    """Tests for serve_skills config loading, validation, and merge."""

    def test_validate_valid_skills(self):
        from swival.config import _validate_serve_skills

        skills = [
            {
                "id": "review",
                "name": "Review",
                "description": "Review code",
                "examples": ["Review this"],
            },
            {"id": "explain"},
        ]
        # Should not raise
        _validate_serve_skills(skills, "test")

    def test_validate_missing_id(self):
        from swival.config import _validate_serve_skills

        with pytest.raises(ConfigError, match="missing required key 'id'"):
            _validate_serve_skills([{"name": "Review"}], "test")

    def test_validate_duplicate_id(self):
        from swival.config import _validate_serve_skills

        skills = [{"id": "review"}, {"id": "review"}]
        with pytest.raises(ConfigError, match="duplicate skill ID"):
            _validate_serve_skills(skills, "test")

    def test_validate_id_not_string(self):
        from swival.config import _validate_serve_skills

        with pytest.raises(ConfigError, match="expected string"):
            _validate_serve_skills([{"id": 42}], "test")

    def test_validate_id_mutates_under_sanitization(self):
        from swival.config import _validate_serve_skills

        with pytest.raises(ConfigError, match="not a valid skill ID"):
            _validate_serve_skills([{"id": "-review-"}], "test")

    def test_validate_id_with_spaces_rejected(self):
        from swival.config import _validate_serve_skills

        with pytest.raises(ConfigError, match="not a valid skill ID"):
            _validate_serve_skills([{"id": "my skill"}], "test")

    def test_validate_not_a_dict(self):
        from swival.config import _validate_serve_skills

        with pytest.raises(ConfigError, match="expected a table"):
            _validate_serve_skills(["not a dict"], "test")

    def test_validate_examples_not_list(self):
        from swival.config import _validate_serve_skills

        with pytest.raises(ConfigError, match="expected list"):
            _validate_serve_skills([{"id": "x", "examples": "not a list"}], "test")

    def test_validate_examples_element_not_string(self):
        from swival.config import _validate_serve_skills

        with pytest.raises(ConfigError, match="expected string"):
            _validate_serve_skills([{"id": "x", "examples": [42]}], "test")

    def test_validate_name_not_string(self):
        from swival.config import _validate_serve_skills

        with pytest.raises(ConfigError, match="expected string"):
            _validate_serve_skills([{"id": "x", "name": 42}], "test")

    def test_validate_unknown_keys_warn(self, capsys):
        from swival.config import _validate_serve_skills

        _validate_serve_skills([{"id": "x", "future_field": True}], "test")
        assert "unknown keys" in capsys.readouterr().err

    def test_config_loading_serve_skills(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        toml_content = (
            'serve_name = "Bot"\n'
            'serve_description = "A bot"\n'
            '[[serve_skills]]\nid = "ask"\nname = "Ask"\n'
        )
        _write_toml(tmp_path / "swival.toml", toml_content)
        result = load_config(tmp_path)
        assert result["serve_name"] == "Bot"
        assert result["serve_description"] == "A bot"
        assert len(result["serve_skills"]) == 1
        assert result["serve_skills"][0]["id"] == "ask"

    def test_config_merge_project_replaces_global_skills(self, tmp_path, monkeypatch):
        global_dir = tmp_path / "global"
        global_dir.mkdir()
        monkeypatch.setenv("XDG_CONFIG_HOME", str(global_dir))
        _write_toml(
            global_dir / "swival" / "config.toml",
            '[[serve_skills]]\nid = "global-skill"\n',
        )
        _write_toml(
            tmp_path / "swival.toml",
            '[[serve_skills]]\nid = "project-skill"\n',
        )
        result = load_config(tmp_path)
        assert len(result["serve_skills"]) == 1
        assert result["serve_skills"][0]["id"] == "project-skill"

    def test_config_merge_global_skills_used_when_no_project(
        self, tmp_path, monkeypatch
    ):
        global_dir = tmp_path / "global"
        global_dir.mkdir()
        monkeypatch.setenv("XDG_CONFIG_HOME", str(global_dir))
        _write_toml(
            global_dir / "swival" / "config.toml",
            '[[serve_skills]]\nid = "global-skill"\n',
        )
        # No project config
        result = load_config(tmp_path)
        assert len(result["serve_skills"]) == 1
        assert result["serve_skills"][0]["id"] == "global-skill"

    def test_config_to_session_kwargs_drops_serve_keys(self):
        config = {
            "serve_name": "Bot",
            "serve_description": "A bot",
            "serve_skills": [{"id": "ask"}],
            "provider": "lmstudio",
        }
        kwargs = config_to_session_kwargs(config)
        assert "serve_name" not in kwargs
        assert "serve_description" not in kwargs
        assert "serve_skills" not in kwargs
        assert kwargs["provider"] == "lmstudio"

    def test_serve_skills_not_a_list_in_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        _write_toml(tmp_path / "swival.toml", 'serve_skills = "nope"\n')
        with pytest.raises(ConfigError, match="must be an array"):
            load_config(tmp_path)

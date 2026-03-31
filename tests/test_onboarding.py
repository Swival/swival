"""Tests for swival.onboarding -- first-run interactive setup wizard."""

import argparse
import io
import types
from swival.onboarding import (
    run_onboarding,
    render_minimal_config,
    _mask_secret,
    _prompt_text_required,
    _ask_api_key,
    _ask_huggingface,
    _ask_command,
    _ask_lmstudio,
    _ask_openrouter,
    _ask_google,
    _ask_generic,
    _ask_bedrock,
)
from swival.agent import _should_try_onboarding
from swival.config import _UNSET, _toml_escape


def _make_args(**overrides):
    """Build a minimal argparse namespace for onboarding trigger tests."""
    defaults = {
        "provider": _UNSET,
        "profile": None,
        "reviewer_mode": False,
        "serve": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _capture_stderr(monkeypatch):
    """Redirect Rich console output to a StringIO for assertions."""
    from rich.console import Console

    buf = io.StringIO()
    console = Console(file=buf, no_color=True, width=120)
    monkeypatch.setattr("swival.onboarding._console", console)
    return buf


class TestShouldTryOnboarding:
    def test_true_on_fresh_interactive_start(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("sys.stderr", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("swival.config.global_config_dir", lambda: tmp_path / "cfg")
        args = _make_args()
        assert _should_try_onboarding(args, tmp_path) is True

    def test_false_when_global_config_exists(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "cfg"
        cfg_dir.mkdir()
        (cfg_dir / "config.toml").write_text('provider = "lmstudio"\n')
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("sys.stderr", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("swival.config.global_config_dir", lambda: cfg_dir)
        args = _make_args()
        assert _should_try_onboarding(args, tmp_path) is False

    def test_false_when_project_config_exists(self, tmp_path, monkeypatch):
        (tmp_path / "swival.toml").write_text('provider = "lmstudio"\n')
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("sys.stderr", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("swival.config.global_config_dir", lambda: tmp_path / "cfg")
        args = _make_args()
        assert _should_try_onboarding(args, tmp_path) is False

    def test_false_when_provider_set(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("sys.stderr", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("swival.config.global_config_dir", lambda: tmp_path / "cfg")
        args = _make_args(provider="lmstudio")
        assert _should_try_onboarding(args, tmp_path) is False

    def test_false_when_profile_set(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("sys.stderr", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("swival.config.global_config_dir", lambda: tmp_path / "cfg")
        args = _make_args(profile="fast")
        assert _should_try_onboarding(args, tmp_path) is False

    def test_false_when_stdin_piped(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: False))
        monkeypatch.setattr("sys.stderr", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("swival.config.global_config_dir", lambda: tmp_path / "cfg")
        args = _make_args()
        assert _should_try_onboarding(args, tmp_path) is False

    def test_false_when_stderr_not_tty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("sys.stderr", types.SimpleNamespace(isatty=lambda: False))
        monkeypatch.setattr("swival.config.global_config_dir", lambda: tmp_path / "cfg")
        args = _make_args()
        assert _should_try_onboarding(args, tmp_path) is False

    def test_false_when_reviewer_mode(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("sys.stderr", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("swival.config.global_config_dir", lambda: tmp_path / "cfg")
        args = _make_args(reviewer_mode=True)
        assert _should_try_onboarding(args, tmp_path) is False

    def test_false_when_serve_mode(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("sys.stderr", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("swival.config.global_config_dir", lambda: tmp_path / "cfg")
        args = _make_args(serve=True)
        assert _should_try_onboarding(args, tmp_path) is False

    def test_false_when_skip_marker_exists(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "cfg"
        cfg_dir.mkdir()
        (cfg_dir / ".onboarding-skipped").write_text("")
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("sys.stderr", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("swival.config.global_config_dir", lambda: cfg_dir)
        args = _make_args()
        assert _should_try_onboarding(args, tmp_path) is False


class TestRenderMinimalConfig:
    def test_basic_provider_only(self):
        result = render_minimal_config({"provider": "lmstudio"})
        assert 'provider = "lmstudio"' in result
        assert result.startswith("# Swival config")

    def test_header_comment(self):
        result = render_minimal_config({"provider": "lmstudio"})
        assert "# Run `swival --init-config` to see all available options." in result

    def test_multiple_keys_ordered(self):
        settings = {
            "provider": "openrouter",
            "model": "openai/gpt-4.1",
            "api_key": "sk-123",
            "max_context_tokens": 131072,
        }
        result = render_minimal_config(settings)
        config_lines = [
            ln for ln in result.split("\n") if ln and not ln.startswith("#")
        ]
        assert config_lines[0] == 'provider = "openrouter"'
        assert config_lines[1] == 'model = "openai/gpt-4.1"'
        assert config_lines[2] == 'api_key = "sk-123"'
        assert config_lines[3] == "max_context_tokens = 131072"

    def test_integer_values(self):
        result = render_minimal_config(
            {"provider": "generic", "max_context_tokens": 65536}
        )
        assert "max_context_tokens = 65536" in result

    def test_omits_unknown_keys(self):
        result = render_minimal_config({"provider": "lmstudio", "unknown_key": "val"})
        assert "unknown_key" not in result

    def test_toml_escaping(self):
        result = render_minimal_config({"provider": "generic", "model": 'a"b\\c'})
        assert r'model = "a\"b\\c"' in result

    def test_trailing_newline(self):
        result = render_minimal_config({"provider": "lmstudio"})
        assert result.endswith("\n")


class TestHelpers:
    def test_mask_secret_long(self):
        assert _mask_secret("sk-abcdef1234") == "*********1234"

    def test_mask_secret_short(self):
        assert _mask_secret("abc") == "****"

    def test_toml_escape_backslash(self):
        assert _toml_escape("a\\b") == "a\\\\b"

    def test_toml_escape_quote(self):
        assert _toml_escape('a"b') == 'a\\"b'

    def test_toml_escape_newline(self):
        assert _toml_escape("a\nb") == "a\\nb"


def _patch_prompt_sequence(monkeypatch, responses):
    """Patch _session.prompt to return values from a list in order."""
    call_idx = {"i": 0}

    def fake_prompt(*args, **kwargs):
        if call_idx["i"] < len(responses):
            val = responses[call_idx["i"]]
            call_idx["i"] += 1
            if val is KeyboardInterrupt:
                raise KeyboardInterrupt
            return val
        return ""

    monkeypatch.setattr("swival.onboarding._session.prompt", fake_prompt)


class TestRunOnboarding:
    def test_not_right_now_no_skip_marker(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "cfg"
        monkeypatch.setattr("swival.onboarding.global_config_dir", lambda: cfg_dir)
        _patch_prompt_sequence(monkeypatch, ["3"])
        result = run_onboarding()
        assert result is None
        assert not (cfg_dir / "config.toml").exists()
        assert not (cfg_dir / ".onboarding-skipped").exists()

    def test_dont_show_again_writes_skip_marker(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "cfg"
        monkeypatch.setattr("swival.onboarding.global_config_dir", lambda: cfg_dir)
        _patch_prompt_sequence(monkeypatch, ["4"])
        result = run_onboarding()
        assert result is None
        assert not (cfg_dir / "config.toml").exists()
        assert (cfg_dir / ".onboarding-skipped").exists()

    def test_ctrl_c_exits_cleanly_no_skip_marker(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "cfg"
        monkeypatch.setattr("swival.onboarding.global_config_dir", lambda: cfg_dir)
        _patch_prompt_sequence(monkeypatch, ["2", KeyboardInterrupt])
        result = run_onboarding()
        assert result is None
        assert not (cfg_dir / "config.toml").exists()
        assert not (cfg_dir / ".onboarding-skipped").exists()

    def test_quick_setup_lmstudio(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "cfg"
        monkeypatch.setattr("swival.onboarding.global_config_dir", lambda: cfg_dir)
        buf = _capture_stderr(monkeypatch)
        _patch_prompt_sequence(
            monkeypatch,
            [
                "2",  # Quick setup
                "1",  # LM Studio provider
                "y",  # Use default server
                "",  # Model blank (auto-discovery)
                "1",  # Yes, write config
            ],
        )
        result = run_onboarding()
        assert result is not None
        assert result.exists()
        content = result.read_text()
        assert 'provider = "lmstudio"' in content
        assert "model" not in content
        output = buf.getvalue()
        assert "You're all set" in output

    def test_guided_path_lmstudio(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "cfg"
        monkeypatch.setattr("swival.onboarding.global_config_dir", lambda: cfg_dir)
        buf = _capture_stderr(monkeypatch)
        _patch_prompt_sequence(
            monkeypatch,
            [
                "1",  # Guided tour + setup
                "",  # Press Enter to continue past intro screen
                "1",  # LM Studio provider
                "y",  # Use default server
                "",  # Model blank
                "1",  # Yes, write config
            ],
        )
        result = run_onboarding()
        assert result is not None
        assert result.exists()
        content = result.read_text()
        assert 'provider = "lmstudio"' in content
        output = buf.getvalue()
        assert "Why Swival feels different" in output
        assert "correctness" in output.lower()

    def test_guided_path_shows_differentiators(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "cfg"
        monkeypatch.setattr("swival.onboarding.global_config_dir", lambda: cfg_dir)
        buf = _capture_stderr(monkeypatch)
        _patch_prompt_sequence(
            monkeypatch,
            [
                "1",  # Guided tour
                "",  # Continue past intro
                "1",  # LM Studio
                "y",  # Default server
                "",  # No model
                "1",  # Write config
            ],
        )
        run_onboarding()
        output = buf.getvalue()
        assert "--self-review" in output
        assert "--reviewer" in output
        assert "llm_filter" in output
        assert "--encrypt-secrets" in output
        assert "/learn" in output

    def test_success_screen_has_next_steps(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "cfg"
        monkeypatch.setattr("swival.onboarding.global_config_dir", lambda: cfg_dir)
        buf = _capture_stderr(monkeypatch)
        _patch_prompt_sequence(
            monkeypatch,
            [
                "2",  # Quick setup
                "1",  # LM Studio
                "y",  # Default
                "",  # No model
                "1",  # Write config
            ],
        )
        run_onboarding()
        output = buf.getvalue()
        assert "Start here" in output
        assert "Want stronger review?" in output
        assert "swival --self-review" in output
        assert "swival --reviewer" in output
        assert "Want privacy controls?" in output
        assert "swival --encrypt-secrets" in output
        assert "Want the REPL superpowers?" in output
        assert "/init" in output
        assert "/learn" in output
        assert "/remember" in output
        assert "AGENTS.md" in output
        assert "/simplify" in output
        assert "/copy" in output
        assert "/save" in output
        assert "/restore" in output
        assert "checkpoint" in output
        assert "Want to switch model stacks quickly?" in output
        assert "swival --profile" in output
        assert "swival --list-profiles" in output
        assert "swival --init-config --project" in output
        assert "Want agent-to-agent collaboration?" in output
        assert "A2A" in output
        assert "Want the docs?" in output
        assert "swival.dev" in output
        assert "alongside" in output

    def test_successful_openrouter_with_env_var(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "cfg"
        monkeypatch.setattr("swival.onboarding.global_config_dir", lambda: cfg_dir)
        _patch_prompt_sequence(
            monkeypatch,
            [
                "2",  # Quick setup
                "3",  # OpenRouter
                "openai/gpt-4.1",  # model
                "1",  # I'll set OPENROUTER_API_KEY myself
                "",  # skip context window
                "1",  # Yes, write config
            ],
        )
        result = run_onboarding()
        assert result is not None
        content = result.read_text()
        assert 'provider = "openrouter"' in content
        assert 'model = "openai/gpt-4.1"' in content
        assert "api_key" not in content

    def test_successful_generic_with_api_key(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "cfg"
        monkeypatch.setattr("swival.onboarding.global_config_dir", lambda: cfg_dir)
        _patch_prompt_sequence(
            monkeypatch,
            [
                "2",  # Quick setup
                "5",  # Generic OpenAI-compatible
                "http://localhost:11434",  # base URL
                "qwen3:32b",  # model
                "2",  # Enter API key now
                "sk-test-key",  # the key
                "",  # skip context window
                "1",  # Yes, write config
            ],
        )
        result = run_onboarding()
        assert result is not None
        content = result.read_text()
        assert 'provider = "generic"' in content
        assert 'base_url = "http://localhost:11434"' in content
        assert 'model = "qwen3:32b"' in content
        assert 'api_key = "sk-test-key"' in content

    def test_cancel_at_confirmation_no_skip_marker(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "cfg"
        monkeypatch.setattr("swival.onboarding.global_config_dir", lambda: cfg_dir)
        _patch_prompt_sequence(
            monkeypatch,
            [
                "2",  # Quick setup
                "1",  # LM Studio
                "y",  # Default server
                "",  # No model
                "3",  # Cancel at confirmation
            ],
        )
        result = run_onboarding()
        assert result is None
        assert not (cfg_dir / "config.toml").exists()
        assert not (cfg_dir / ".onboarding-skipped").exists()

    def test_start_over_loops_back(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "cfg"
        monkeypatch.setattr("swival.onboarding.global_config_dir", lambda: cfg_dir)
        _patch_prompt_sequence(
            monkeypatch,
            [
                "2",  # Quick setup
                "1",  # LM Studio (first attempt)
                "y",  # Default server
                "",  # No model
                "2",  # Start over
                "2",  # ChatGPT (second attempt)
                "gpt-4.1",  # model
                "",  # skip reasoning effort
                "1",  # Yes, write config
            ],
        )
        result = run_onboarding()
        assert result is not None
        content = result.read_text()
        assert 'provider = "chatgpt"' in content
        assert 'model = "gpt-4.1"' in content

    def test_no_overwrite_existing_config(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "cfg"
        cfg_dir.mkdir(parents=True)
        existing = cfg_dir / "config.toml"
        existing.write_text('provider = "lmstudio"\n')
        monkeypatch.setattr("swival.onboarding.global_config_dir", lambda: cfg_dir)
        _patch_prompt_sequence(
            monkeypatch,
            [
                "2",  # Quick setup
                "2",  # ChatGPT
                "gpt-4.1",
                "",
                "1",  # Yes, write config
            ],
        )
        result = run_onboarding()
        assert result is None
        assert existing.read_text() == 'provider = "lmstudio"\n'

    def test_provider_list_shows_best_for(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "cfg"
        monkeypatch.setattr("swival.onboarding.global_config_dir", lambda: cfg_dir)
        buf = _capture_stderr(monkeypatch)
        _patch_prompt_sequence(
            monkeypatch,
            [
                "2",  # Quick setup
                "1",  # LM Studio
                "y",
                "",
                "1",  # Write config
            ],
        )
        run_onboarding()
        output = buf.getvalue()
        assert "Best for:" in output

    def test_profiles_microcopy_shown(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "cfg"
        monkeypatch.setattr("swival.onboarding.global_config_dir", lambda: cfg_dir)
        buf = _capture_stderr(monkeypatch)
        _patch_prompt_sequence(
            monkeypatch,
            [
                "2",  # Quick setup
                "1",  # LM Studio
                "y",
                "",
                "1",  # Write config
            ],
        )
        run_onboarding()
        output = buf.getvalue()
        assert "switch later with profiles" in output

    def test_no_stdout_output(self, tmp_path, monkeypatch, capsys):
        cfg_dir = tmp_path / "cfg"
        monkeypatch.setattr("swival.onboarding.global_config_dir", lambda: cfg_dir)
        _capture_stderr(monkeypatch)
        _patch_prompt_sequence(
            monkeypatch,
            [
                "1",  # Guided tour
                "",  # Continue
                "1",  # LM Studio
                "y",
                "",
                "1",
            ],
        )
        run_onboarding()
        captured = capsys.readouterr()
        assert captured.out == ""


class TestRequiredFieldValidation:
    def test_prompt_text_required_reprompts_on_blank(self, monkeypatch):
        buf = _capture_stderr(monkeypatch)
        _patch_prompt_sequence(monkeypatch, ["", "", "real-value"])
        result = _prompt_text_required("Model")
        assert result == "real-value"
        assert "is required" in buf.getvalue()

    def test_prompt_text_required_accepts_first_nonempty(self, monkeypatch):
        _capture_stderr(monkeypatch)
        _patch_prompt_sequence(monkeypatch, ["good-value"])
        result = _prompt_text_required("Model")
        assert result == "good-value"

    def test_ask_api_key_reprompts_on_blank(self, monkeypatch):
        buf = _capture_stderr(monkeypatch)
        _patch_prompt_sequence(
            monkeypatch,
            [
                "2",  # "Enter it now"
                "",  # blank first attempt
                "sk-abc",  # valid second attempt
            ],
        )
        s = {}
        _ask_api_key(s, env_var="TEST_KEY")
        assert s["api_key"] == "sk-abc"
        assert "is required" in buf.getvalue()

    def test_ask_api_key_env_var_skips_prompt(self, monkeypatch):
        _capture_stderr(monkeypatch)
        _patch_prompt_sequence(monkeypatch, ["1"])  # "I'll set ENV myself"
        s = {}
        _ask_api_key(s, env_var="TEST_KEY")
        assert "api_key" not in s

    def test_ask_huggingface_rejects_bare_model(self, monkeypatch):
        buf = _capture_stderr(monkeypatch)
        _patch_prompt_sequence(
            monkeypatch,
            [
                "bare-model",  # no slash
                "org/model",  # valid
                "1",  # "I'll set HF_TOKEN myself"
                "",  # skip endpoint
            ],
        )
        s = {}
        _ask_huggingface(s)
        assert s["model"] == "org/model"
        assert "org/model format" in buf.getvalue()

    def test_ask_command_rejects_nonexistent(self, monkeypatch):
        buf = _capture_stderr(monkeypatch)
        _patch_prompt_sequence(
            monkeypatch,
            [
                "definitely_not_a_real_command_xyz",
                "echo",  # exists on PATH
            ],
        )
        s = {}
        _ask_command(s)
        assert s["model"] == "echo"
        assert "Command not found" in buf.getvalue()

    def test_ask_command_rejects_bad_quoting(self, monkeypatch):
        buf = _capture_stderr(monkeypatch)
        _patch_prompt_sequence(
            monkeypatch,
            [
                "echo 'unclosed",  # bad quoting
                "echo",  # valid
            ],
        )
        s = {}
        _ask_command(s)
        assert s["model"] == "echo"
        assert "Invalid command syntax" in buf.getvalue()

    def test_ask_lmstudio_allows_blank_model(self, monkeypatch):
        _capture_stderr(monkeypatch)
        _patch_prompt_sequence(
            monkeypatch,
            [
                "y",  # use default server
                "",  # blank model
            ],
        )
        s = {}
        _ask_lmstudio(s)
        assert "model" not in s

    def test_ask_openrouter_requires_model(self, monkeypatch):
        buf = _capture_stderr(monkeypatch)
        _patch_prompt_sequence(
            monkeypatch,
            [
                "",  # blank model
                "openai/gpt-5.4",  # valid model
                "1",  # "I'll set OPENROUTER_API_KEY myself"
                "",  # skip context tokens
            ],
        )
        s = {}
        _ask_openrouter(s)
        assert s["model"] == "openai/gpt-5.4"
        assert "is required" in buf.getvalue()

    def test_ask_google_requires_model(self, monkeypatch):
        buf = _capture_stderr(monkeypatch)
        _patch_prompt_sequence(
            monkeypatch,
            [
                "",  # blank model
                "gemini-2.5-flash",  # valid
                "1",  # "I'll set env var myself"
            ],
        )
        s = {}
        _ask_google(s)
        assert s["model"] == "gemini-2.5-flash"
        assert "is required" in buf.getvalue()

    def test_ask_generic_requires_base_url_and_model(self, monkeypatch):
        buf = _capture_stderr(monkeypatch)
        _patch_prompt_sequence(
            monkeypatch,
            [
                "",  # blank base_url
                "http://localhost:11434",  # valid base_url
                "",  # blank model
                "llama3",  # valid model
                "1",  # "I'll set OPENAI_API_KEY myself"
                "",  # skip context tokens
            ],
        )
        s = {}
        _ask_generic(s)
        assert s["base_url"] == "http://localhost:11434"
        assert s["model"] == "llama3"
        assert buf.getvalue().count("is required") == 2

    def test_ask_bedrock_requires_model(self, monkeypatch):
        buf = _capture_stderr(monkeypatch)
        _patch_prompt_sequence(
            monkeypatch,
            [
                "",  # blank
                "global.anthropic.claude-opus-4-6-v1",  # valid
                "",  # skip region
                "",  # skip profile
            ],
        )
        s = {}
        _ask_bedrock(s)
        assert s["model"] == "global.anthropic.claude-opus-4-6-v1"
        assert "is required" in buf.getvalue()

    def test_google_onboarding_mentions_both_env_vars(self, monkeypatch):
        buf = _capture_stderr(monkeypatch)
        _patch_prompt_sequence(
            monkeypatch,
            [
                "gemini-2.5-flash",
                "1",  # "I'll set env var myself"
            ],
        )
        s = {}
        _ask_google(s)
        output = buf.getvalue()
        assert "GEMINI_API_KEY" in output
        assert "OPENAI_API_KEY" in output

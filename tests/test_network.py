"""Tests for the --network policy: resolution, jail wiring, and tool filtering."""

import json
import os
import shutil
import socket
import subprocess
import sys

import pytest

from swival.agent import (
    _resolve_network_policy,
    build_parser,
    build_tools,
)
from swival.config import _UNSET, apply_config_to_args, args_to_session_kwargs
from swival.report import ConfigError, ReportCollector
from swival.mcp_client import _jail_stdio_command
from swival.sandbox_nono import (
    _ENV_MARKER,
    _NONO_ENV,
    build_block_net_wrapper,
    build_nono_argv,
    is_inside_nono,
    is_net_blocked,
)
from swival.session import Session
from swival import tools


NONO_AVAILABLE = shutil.which("nono") is not None

# Captured at import time, before the autouse fixture scrubs the markers:
# a host that runs the suite inside nono cannot nest another sandbox, so
# the live tests must be skipped there.
HOST_INSIDE_NONO = is_inside_nono()


def _loopback_bindable() -> bool:
    try:
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        probe.close()
        return True
    except OSError:
        return False


LOOPBACK_OK = _loopback_bindable()

JAIL = ["nono", "run", "--silent", "--allow", "/base", "--block-net", "--"]


@pytest.fixture(autouse=True)
def _outside_nono(monkeypatch):
    """Neutralize an inherited nono environment.

    These tests exercise Swival's own policy logic; a host that happens to
    run the suite inside nono must not leak its sandbox markers into them.
    Tests that need the markers set them explicitly.
    """
    monkeypatch.delenv(_NONO_ENV, raising=False)
    monkeypatch.delenv(_ENV_MARKER, raising=False)


def _set_cap(tmp_path, monkeypatch, content):
    cap = tmp_path / "cap.json"
    cap.write_text(content, encoding="utf-8")
    monkeypatch.setenv(_NONO_ENV, str(cap))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve(tmp_path, argv=(), config=None, monkeypatch=None):
    """Mimic main(): parse CLI, apply config, run network policy resolution."""
    args = build_parser().parse_args([*argv, "task"])
    network_cli = args.network is not _UNSET
    sandbox_cli = args.sandbox is not _UNSET
    config = dict(config or {})
    args._mcp_servers_toml = config.pop("mcp_servers", None)
    args._a2a_servers_toml = config.pop("a2a_servers", None)
    apply_config_to_args(args, config)
    error, diagnostic = _resolve_network_policy(
        args,
        tmp_path,
        network_cli=network_cli,
        sandbox_cli=sandbox_cli,
        sandbox_config="sandbox" in config,
    )
    return args, error, diagnostic


def _tool_names(tool_list):
    return {t["function"]["name"] for t in tool_list}


needs_nono = pytest.mark.skipif(not NONO_AVAILABLE, reason="nono not installed")


# ===========================================================================
# CLI flag parsing
# ===========================================================================


class TestNetworkFlag:
    def test_default_is_unset(self):
        args = build_parser().parse_args(["task"])
        assert args.network is _UNSET

    def test_accepts_all_modes(self):
        for mode in ("full", "provider-only", "none"):
            args = build_parser().parse_args(["--network", mode, "task"])
            assert args.network == mode

    def test_rejects_unknown_mode(self, capsys):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--network", "offline", "task"])


# ===========================================================================
# Policy resolution: mode x sandbox matrix
# ===========================================================================


class TestNoneResolution:
    def test_full_default_is_noop(self, tmp_path):
        args, error, diag = _resolve(tmp_path)
        assert error is None and diag is None
        assert args.sandbox == "builtin"
        assert args.nono_block_net is False

    def test_none_selects_nono_and_block_net(self, tmp_path):
        args, error, diag = _resolve(
            tmp_path, ["--network", "none", "--provider", "command"]
        )
        assert error is None
        assert args.sandbox == "nono"
        assert args.nono_block_net is True
        assert diag is None  # no configured sandbox was replaced

    def test_config_none_behaves_the_same(self, tmp_path):
        args, error, diag = _resolve(
            tmp_path, config={"network": "none", "provider": "command"}
        )
        assert error is None
        assert args.sandbox == "nono"
        assert args.nono_block_net is True

    def test_none_replaces_config_sandbox_with_diagnostic(self, tmp_path):
        args, error, diag = _resolve(
            tmp_path,
            ["--network", "none", "--provider", "command"],
            config={"sandbox": "agentfs"},
        )
        assert error is None
        assert args.sandbox == "nono"
        assert "replaced the configured sandbox" in diag

    def test_none_with_cli_nono_accepted(self, tmp_path):
        args, error, diag = _resolve(
            tmp_path,
            ["--network", "none", "--sandbox", "nono", "--provider", "command"],
        )
        assert error is None and diag is None
        assert args.nono_block_net is True

    def test_cli_none_with_cli_builtin_rejected(self, tmp_path):
        _, error, _ = _resolve(tmp_path, ["--network", "none", "--sandbox", "builtin"])
        assert "--network none requires the nono sandbox" in error
        assert "--network full" in error

    def test_config_none_with_cli_agentfs_fails_closed(self, tmp_path):
        _, error, _ = _resolve(
            tmp_path, ["--sandbox", "agentfs"], config={"network": "none"}
        )
        assert 'config sets network = "none"' in error
        assert "--network full" in error

    def test_config_none_with_cli_nono_keeps_policy(self, tmp_path):
        args, error, _ = _resolve(
            tmp_path,
            ["--sandbox", "nono", "--provider", "command"],
            config={"network": "none"},
        )
        assert error is None
        assert args.nono_block_net is True

    def test_cli_full_overrides_config_none(self, tmp_path):
        args, error, _ = _resolve(
            tmp_path, ["--network", "full"], config={"network": "none"}
        )
        assert error is None
        assert args.sandbox == "builtin"
        assert args.nono_block_net is False

    def test_none_requires_command_provider(self, tmp_path):
        _, error, _ = _resolve(tmp_path, ["--network", "none"])
        assert "requires the command provider" in error
        assert "lmstudio" in error
        assert "provider-only" in error  # points at the practical alternative

    def test_none_rejects_serve(self, tmp_path):
        _, error, _ = _resolve(
            tmp_path, ["--network", "none", "--provider", "command", "--serve"]
        )
        assert "--serve" in error

    def test_none_rejects_allow_domain(self, tmp_path):
        _, error, _ = _resolve(
            tmp_path,
            [
                "--network",
                "none",
                "--provider",
                "command",
                "--sandbox",
                "nono",
                "--nono-allow-domain",
                "api.example.com",
            ],
        )
        assert "--nono-allow-domain" in error

    def test_none_rejects_network_profile(self, tmp_path):
        _, error, _ = _resolve(
            tmp_path,
            [
                "--network",
                "none",
                "--provider",
                "command",
                "--sandbox",
                "nono",
                "--nono-network-profile",
                "developer",
            ],
        )
        assert "--nono-network-profile" in error

    def test_none_rejects_credential(self, tmp_path):
        _, error, _ = _resolve(
            tmp_path,
            [
                "--network",
                "none",
                "--provider",
                "command",
                "--sandbox",
                "nono",
                "--nono-credential",
                "anthropic",
            ],
        )
        assert "--nono-credential" in error

    def test_default_empty_proxy_values_do_not_conflict(self, tmp_path):
        # CLI defaults are None; config resolution normalizes to [].
        args, error, _ = _resolve(
            tmp_path,
            ["--network", "none", "--provider", "command"],
            config={"nono_allow_domain": [], "nono_credential": []},
        )
        assert error is None

    def test_explicit_block_net_is_redundant_but_harmless(self, tmp_path):
        args, error, _ = _resolve(
            tmp_path,
            [
                "--network",
                "none",
                "--provider",
                "command",
                "--sandbox",
                "nono",
                "--nono-block-net",
            ],
        )
        assert error is None
        assert args.nono_block_net is True


@needs_nono
class TestProviderOnlyResolution:
    def test_any_provider_accepted(self, tmp_path):
        for provider in ("lmstudio", "openrouter", "command"):
            args, error, _ = _resolve(
                tmp_path, ["--network", "provider-only", "--provider", provider]
            )
            assert error is None, error

    def test_sandbox_stays_untouched(self, tmp_path):
        args, error, _ = _resolve(tmp_path, ["--network", "provider-only"])
        assert error is None
        assert args.sandbox == "builtin"
        assert args.nono_block_net is False

    def test_agentfs_sandbox_compatible(self, tmp_path):
        args, error, _ = _resolve(
            tmp_path, ["--network", "provider-only", "--sandbox", "agentfs"]
        )
        assert error is None
        assert args.sandbox == "agentfs"

    def test_nono_sandbox_rejected(self, tmp_path):
        _, error, _ = _resolve(
            tmp_path, ["--network", "provider-only", "--sandbox", "nono"]
        )
        assert "nest nono inside nono" in error

    def test_serve_allowed(self, tmp_path):
        _, error, _ = _resolve(tmp_path, ["--network", "provider-only", "--serve"])
        assert error is None

    def test_rejected_inside_existing_nono(self, tmp_path, monkeypatch):
        monkeypatch.setenv(_NONO_ENV, "/tmp/.nono-test.json")
        _, error, _ = _resolve(tmp_path, ["--network", "provider-only"])
        assert "existing nono sandbox" in error


class TestProviderOnlyWithoutNono:
    def test_missing_nono_binary_is_actionable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: None)
        _, error, _ = _resolve(tmp_path, ["--network", "provider-only"])
        assert "requires the nono binary" in error
        assert "--network full" in error


# ===========================================================================
# Remote integrations (both restricted modes)
# ===========================================================================


class TestRemoteIntegrationRejection:
    NONE_ARGV = ["--network", "none", "--provider", "command"]

    def test_url_mcp_from_toml_rejected(self, tmp_path):
        _, error, _ = _resolve(
            tmp_path,
            self.NONE_ARGV,
            config={"mcp_servers": {"remote-api": {"url": "https://x.example/mcp"}}},
        )
        assert "URL-backed MCP server 'remote-api'" in error
        assert "--no-mcp" in error

    def test_stdio_mcp_accepted(self, tmp_path):
        _, error, _ = _resolve(
            tmp_path,
            self.NONE_ARGV,
            config={"mcp_servers": {"local": {"command": "echo"}}},
        )
        assert error is None

    def test_url_mcp_from_swival_dir_rejected(self, tmp_path):
        mcp_json = tmp_path / ".swival" / "mcp.json"
        mcp_json.parent.mkdir(parents=True)
        mcp_json.write_text(
            json.dumps({"mcpServers": {"remote": {"url": "https://x.example/mcp"}}}),
            encoding="utf-8",
        )
        _, error, _ = _resolve(tmp_path, self.NONE_ARGV)
        assert "URL-backed MCP server 'remote'" in error

    def test_url_mcp_from_mcp_config_flag_rejected(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        cfg.write_text(
            json.dumps({"mcpServers": {"remote": {"url": "https://x.example/mcp"}}}),
            encoding="utf-8",
        )
        _, error, _ = _resolve(tmp_path, [*self.NONE_ARGV, "--mcp-config", str(cfg)])
        assert "URL-backed MCP server 'remote'" in error

    def test_no_mcp_bypasses_url_check(self, tmp_path):
        _, error, _ = _resolve(
            tmp_path,
            [*self.NONE_ARGV, "--no-mcp"],
            config={"mcp_servers": {"remote": {"url": "https://x.example/mcp"}}},
        )
        assert error is None

    def test_a2a_rejected(self, tmp_path):
        _, error, _ = _resolve(
            tmp_path,
            self.NONE_ARGV,
            config={"a2a_servers": {"helper": {"url": "http://127.0.0.1:9/a2a"}}},
        )
        assert "A2A server 'helper'" in error
        assert "--no-a2a" in error

    def test_no_a2a_bypasses_check(self, tmp_path):
        _, error, _ = _resolve(
            tmp_path,
            [*self.NONE_ARGV, "--no-a2a"],
            config={"a2a_servers": {"helper": {"url": "http://127.0.0.1:9/a2a"}}},
        )
        assert error is None

    def test_resolution_is_repeatable_and_cached(self, tmp_path):
        args, error, _ = _resolve(
            tmp_path,
            self.NONE_ARGV,
            config={"mcp_servers": {"local": {"command": "echo"}}},
        )
        assert error is None
        assert args._resolved_mcp_servers == {"local": {"command": "echo"}}
        # A second pass must be a pure function of args (main() runs twice
        # around the nono re-exec) and must reuse the cache.
        args._mcp_servers_toml = {"mutated": {"url": "https://x.example/mcp"}}
        error2, _ = _resolve_network_policy(
            args, tmp_path, network_cli=True, sandbox_cli=False, sandbox_config=False
        )
        assert error2 is None


# ===========================================================================
# Tool schema filtering and dispatch guard
# ===========================================================================


class TestFetchUrlRemoval:
    def test_present_by_default(self):
        assert "fetch_url" in _tool_names(build_tools({}, {}, False))

    def test_absent_in_restricted_modes(self):
        for mode in ("provider-only", "none"):
            assert "fetch_url" not in _tool_names(
                build_tools({}, {}, False, network=mode)
            )

    def test_module_schema_list_unchanged(self):
        build_tools({}, {}, False, network="none")
        assert "fetch_url" in {t["function"]["name"] for t in tools.TOOLS}

    def test_dispatch_rejects_fetch_url_when_restricted(self, tmp_path):
        result = tools.dispatch(
            "fetch_url",
            {"url": "https://example.com"},
            str(tmp_path),
            network_mode="provider-only",
        )
        assert result.startswith("error:")
        assert "provider-only" in result

    def test_dispatch_allows_fetch_url_by_default(self, tmp_path, monkeypatch):
        import swival.fetch as fetch_mod
        import types

        monkeypatch.setattr(
            fetch_mod,
            "_fetch",
            lambda **kw: types.SimpleNamespace(body="ok", url=kw["url"]),
        )
        result = tools.dispatch(
            "fetch_url", {"url": "https://example.com"}, str(tmp_path)
        )
        assert "ok" in result


# ===========================================================================
# The jail prefix and its wrap points
# ===========================================================================


class TestJailPrefix:
    def test_jail_noop_when_none(self):
        assert tools._jail(["ls", "-la"], None) == ["ls", "-la"]

    def test_jail_prepends_prefix(self):
        assert tools._jail(["ls"], JAIL) == [*JAIL, "ls"]

    def test_stdio_jail_rewrites_command(self):
        command, args = _jail_stdio_command("npx", ["-y", "server"], JAIL)
        assert command == "nono"
        assert args == [*JAIL[1:], "npx", "-y", "server"]

    def test_stdio_jail_noop_without_prefix(self):
        command, args = _jail_stdio_command("npx", ["-y", "server"], None)
        assert command == "npx"
        assert args == ["-y", "server"]


class TestWrapPoints:
    """The four Popen sites must apply the jail prefix under provider-only."""

    @pytest.fixture
    def captured_popen(self, monkeypatch):
        captured = {}

        def fake_popen(argv, **kwargs):
            captured["argv"] = argv
            raise OSError("stop after capture")

        monkeypatch.setattr(tools.subprocess, "Popen", fake_popen)
        return captured

    def test_shell_command_wrapped(self, tmp_path, captured_popen):
        tools._run_shell_command("echo hi", str(tmp_path), 5, net_jail=JAIL)
        assert captured_popen["argv"][: len(JAIL)] == JAIL
        assert captured_popen["argv"][len(JAIL) :] == ["/bin/sh", "-c", "echo hi"]

    def test_shell_command_unwrapped_by_default(self, tmp_path, captured_popen):
        tools._run_shell_command("echo hi", str(tmp_path), 5)
        assert captured_popen["argv"] == ["/bin/sh", "-c", "echo hi"]

    def test_argv_command_wrapped(self, tmp_path, captured_popen):
        tools._run_argv_command(
            ["ls"], str(tmp_path), {}, unrestricted=True, net_jail=JAIL
        )
        assert captured_popen["argv"][: len(JAIL)] == JAIL
        assert captured_popen["argv"][len(JAIL)].endswith("ls")

    def test_python_tool_wrapped(self, tmp_path, captured_popen):
        tools._run_python("print(1)", str(tmp_path), 5, net_jail=JAIL)
        assert captured_popen["argv"][: len(JAIL)] == JAIL
        assert captured_popen["argv"][len(JAIL) + 1 :] == ["-c", "print(1)"]

    def test_background_command_wrapped(self, tmp_path, captured_popen):
        tools._run_argv_command(
            ["ls"],
            str(tmp_path),
            {},
            unrestricted=True,
            background=True,
            net_jail=JAIL,
        )
        assert captured_popen["argv"][: len(JAIL)] == JAIL

    def test_dispatch_threads_jail_to_command(self, tmp_path, captured_popen):
        tools.dispatch(
            "run_command",
            {"command": ["ls"]},
            str(tmp_path),
            commands_unrestricted=True,
            net_jail=JAIL,
        )
        assert captured_popen["argv"][: len(JAIL)] == JAIL

    def test_dispatch_threads_jail_to_python(self, tmp_path, captured_popen):
        tools.dispatch(
            "run_python",
            {"code": "print(1)"},
            str(tmp_path),
            commands_unrestricted=True,
            net_jail=JAIL,
        )
        assert captured_popen["argv"][: len(JAIL)] == JAIL


# ===========================================================================
# Wrapper argv construction
# ===========================================================================


class TestBlockNetWrapper:
    def test_wrapper_shape(self, tmp_path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: "/opt/bin/nono")
        argv = build_block_net_wrapper(base_dir=str(tmp_path))
        assert argv[0] == "/opt/bin/nono"
        assert argv[1] == "run"
        assert argv[2] == "--silent"
        assert argv.count("--block-net") == 1
        assert argv[-1] == "--"
        assert str(tmp_path.resolve()) in argv

    def test_wrapper_grants_add_dirs_and_read_dirs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: "/opt/bin/nono")
        extra = tmp_path / "extra"
        ro = tmp_path / "ro"
        extra.mkdir()
        ro.mkdir()
        argv = build_block_net_wrapper(
            base_dir=str(tmp_path), add_dirs=[str(extra)], read_dirs=[str(ro)]
        )
        allow_values = [argv[i + 1] for i, a in enumerate(argv) if a == "--allow"]
        read_values = [argv[i + 1] for i, a in enumerate(argv) if a == "--read"]
        assert str(extra.resolve()) in allow_values
        assert str(ro.resolve()) in read_values

    def test_wrapper_missing_nono_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: None)
        with pytest.raises(ConfigError, match="requires the nono binary"):
            build_block_net_wrapper(base_dir=str(tmp_path))

    def test_build_nono_argv_silent_flag(self, tmp_path):
        argv = build_nono_argv(
            nono_bin="nono",
            base_dir=str(tmp_path),
            add_dirs=[],
            silent=True,
            swival_argv=["swival"],
        )
        assert argv[:3] == ["nono", "run", "--silent"]
        argv_default = build_nono_argv(
            nono_bin="nono",
            base_dir=str(tmp_path),
            add_dirs=[],
            swival_argv=["swival"],
        )
        assert "--silent" not in argv_default


class TestNetBlockedVerification:
    """is_net_blocked() must return True only on positive proof."""

    def test_true_when_cap_records_block(self, tmp_path, monkeypatch):
        _set_cap(tmp_path, monkeypatch, json.dumps({"net_blocked": True}))
        assert is_net_blocked() is True

    def test_false_when_cap_not_blocked(self, tmp_path, monkeypatch):
        _set_cap(tmp_path, monkeypatch, json.dumps({"net_blocked": False}))
        assert is_net_blocked() is False

    def test_false_when_key_missing(self, tmp_path, monkeypatch):
        _set_cap(tmp_path, monkeypatch, json.dumps({"fs": []}))
        assert is_net_blocked() is False

    def test_false_on_malformed_json(self, tmp_path, monkeypatch):
        _set_cap(tmp_path, monkeypatch, "{not json")
        assert is_net_blocked() is False

    def test_false_on_non_object_json(self, tmp_path, monkeypatch):
        _set_cap(tmp_path, monkeypatch, json.dumps(["net_blocked"]))
        assert is_net_blocked() is False

    def test_false_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv(_NONO_ENV, str(tmp_path / "gone.json"))
        assert is_net_blocked() is False

    def test_false_outside_nono(self):
        assert is_net_blocked() is False


# ===========================================================================
# Session API
# ===========================================================================


class TestSessionNetwork:
    def test_invalid_mode_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="'network' must be"):
            Session(base_dir=str(tmp_path), network="offline")

    def test_none_outside_nono_raises(self, tmp_path):
        s = Session(
            base_dir=str(tmp_path), network="none", provider="command", history=False
        )
        with pytest.raises(ConfigError, match="cannot bootstrap"):
            s._setup()

    def test_none_requires_command_provider(self, tmp_path):
        s = Session(base_dir=str(tmp_path), network="none", history=False)
        with pytest.raises(ConfigError, match="requires the command provider"):
            s._setup()

    def test_none_inside_non_blocking_nono_fails_closed(self, tmp_path, monkeypatch):
        _set_cap(tmp_path, monkeypatch, json.dumps({"net_blocked": False}))
        s = Session(
            base_dir=str(tmp_path), network="none", provider="command", history=False
        )
        with pytest.raises(ConfigError, match="capability file does not"):
            s._setup()

    def test_sandbox_nono_block_net_claim_fails_closed(self, tmp_path, monkeypatch):
        _set_cap(tmp_path, monkeypatch, json.dumps({"net_blocked": False}))
        s = Session(
            base_dir=str(tmp_path),
            sandbox="nono",
            nono_block_net=True,
            history=False,
        )
        with pytest.raises(ConfigError, match="does not record --block-net"):
            s._setup()

    def test_provider_only_with_nono_sandbox_raises(self, tmp_path):
        s = Session(
            base_dir=str(tmp_path),
            network="provider-only",
            sandbox="nono",
            history=False,
        )
        with pytest.raises(ConfigError, match="nest nono inside nono"):
            s._setup()

    def test_restricted_mode_rejects_url_mcp(self, tmp_path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: "/opt/bin/nono")
        s = Session(
            base_dir=str(tmp_path),
            network="provider-only",
            mcp_servers={"remote": {"url": "https://x.example/mcp"}},
            history=False,
        )
        with pytest.raises(ConfigError, match="URL-backed MCP server 'remote'"):
            s._setup()

    def test_restricted_mode_rejects_a2a(self, tmp_path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: "/opt/bin/nono")
        s = Session(
            base_dir=str(tmp_path),
            network="provider-only",
            a2a_servers={"helper": {"url": "http://127.0.0.1:9/a2a"}},
            history=False,
        )
        with pytest.raises(ConfigError, match="A2A server 'helper'"):
            s._setup()

    def test_args_to_session_kwargs_passes_network(self, tmp_path):
        args = build_parser().parse_args(["--network", "provider-only", "task"])
        apply_config_to_args(args, {})
        kwargs = args_to_session_kwargs(args, str(tmp_path))
        assert kwargs["network"] == "provider-only"

    def test_provider_only_jail_grants_external_skill_dirs(self, tmp_path, monkeypatch):
        from swival import agent

        monkeypatch.setattr(shutil, "which", lambda name: "/opt/bin/nono")
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        ext = tmp_path / "ext" / "helper"
        ext.mkdir(parents=True)
        (ext / "SKILL.md").write_text(
            "---\nname: helper\ndescription: A helper.\n---\n\nBody.",
            encoding="utf-8",
        )
        base = tmp_path / "ws"
        base.mkdir()

        s = Session(
            base_dir=str(base),
            network="provider-only",
            skills_dir=[str(tmp_path / "ext")],
            history=False,
        )
        s._setup()
        jail = s._net_jail
        read_values = [jail[i + 1] for i, a in enumerate(jail) if a == "--read"]
        assert str(ext.resolve()) in read_values

    def test_session_report_carries_network(self, tmp_path, monkeypatch):
        import types

        from swival import agent

        monkeypatch.setattr(shutil, "which", lambda name: "/opt/bin/nono")
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        def _llm(*args, **kwargs):
            msg = types.SimpleNamespace(content="ok", tool_calls=None, role="assistant")
            return msg, "stop"

        monkeypatch.setattr(agent, "call_llm", _llm)
        s = Session(base_dir=str(tmp_path), network="provider-only", history=False)
        result = s.run("q", report=True)
        assert result.report["network"] == "provider-only"


# ===========================================================================
# Report plumbing
# ===========================================================================


class TestNetworkReport:
    def _build(self, **overrides):
        rc = ReportCollector()
        kwargs = dict(
            task="t",
            model="m",
            provider="command",
            settings={},
            outcome="success",
            answer="a",
            exit_code=0,
            turns=1,
        )
        kwargs.update(overrides)
        return rc, rc.build_report(**kwargs)

    def test_full_mode_has_no_network_field(self):
        _, report = self._build()
        assert "network" not in report

    def test_provider_only_recorded(self):
        _, report = self._build(network_mode="provider-only")
        assert report["network"] == "provider-only"

    def test_none_records_sandbox_offline(self):
        _, report = self._build(
            network_mode="none", sandbox_mode="nono", nono_block_net=True
        )
        assert report["network"] == "none"
        assert report["sandbox"]["network"] == "offline"

    def test_legacy_block_net_records_effective_policy(self):
        # --sandbox nono --nono-block-net without --network none: same
        # restriction, same report field.
        _, report = self._build(sandbox_mode="nono", nono_block_net=True)
        assert report["sandbox"]["network"] == "offline"
        assert "network" not in report

    def test_write_path_carries_network(self, tmp_path):
        rc = ReportCollector()
        rc.finalize(
            task="t",
            model="m",
            provider="command",
            settings={},
            outcome="success",
            answer="a",
            exit_code=0,
            turns=1,
            network_mode="none",
            sandbox_mode="nono",
            nono_block_net=True,
        )
        out = tmp_path / "report.json"
        rc.write(str(out))
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["network"] == "none"
        assert data["sandbox"]["network"] == "offline"


# ===========================================================================
# Guarded integration tests (need a real nono binary)
# ===========================================================================


@needs_nono
@pytest.mark.skipif(
    HOST_INSIDE_NONO, reason="host already runs inside nono; cannot nest sandboxes"
)
@pytest.mark.skipif(not LOOPBACK_OK, reason="loopback sockets unavailable")
class TestLiveEnforcement:
    def _run_swival(self, argv, cwd, timeout=180):
        # The nono re-exec replays sys.argv, so these tests need the real
        # console script (a `python -c` shim would re-exec "-c").
        from pathlib import Path

        swival_bin = Path(sys.executable).parent / "swival"
        if not swival_bin.is_file():
            pytest.skip("swival console script not installed in this environment")
        env = dict(os.environ)
        env.pop(_NONO_ENV, None)
        env.pop(_ENV_MARKER, None)
        return subprocess.run(
            [str(swival_bin), *argv],
            capture_output=True,
            text=True,
            cwd=cwd,
            env=env,
            timeout=timeout,
        )

    def test_none_mode_blocks_dns_public_and_loopback(self, tmp_path):
        # macOS resolves DNS via mDNSResponder outside the sandbox, so probe
        # DNS at the socket layer, not via getaddrinfo.
        import threading

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        threading.Thread(target=listener.accept, daemon=True).start()

        probe = tmp_path / "probe.sh"
        probe.write_text(
            "#!/bin/sh\n"
            "cat > /dev/null\n"
            'python3 -c "\n'
            "import socket\n"
            "def t(label, fn):\n"
            "    try:\n"
            "        fn(); print(label + ':OPEN')\n"
            "    except Exception:\n"
            "        print(label + ':BLOCKED')\n"
            "t('dns', lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM)"
            ".sendto(b'x', ('8.8.8.8', 53)))\n"
            "t('public', lambda: socket.create_connection(('1.1.1.1', 80), timeout=3))\n"
            f"t('loopback', lambda: socket.create_connection(('127.0.0.1', {port}), timeout=3))\n"
            '"\n',
            encoding="utf-8",
        )
        probe.chmod(0o755)

        result = self._run_swival(
            [
                "--network",
                "none",
                "--provider",
                "command",
                "--model",
                str(probe),
                "--no-skills",
                "--no-history",
                "--no-memory",
                "probe",
            ],
            cwd=str(tmp_path),
        )
        listener.close()
        assert result.returncode == 0, result.stderr[-2000:]
        assert "dns:BLOCKED" in result.stdout
        assert "public:BLOCKED" in result.stdout
        assert "loopback:BLOCKED" in result.stdout

    def test_provider_only_parent_reaches_provider_child_blocked(self, tmp_path):
        """The entire feature in one assertion pair: the model transport works
        while an agent command's network probe fails."""
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        probe_cmd = (
            "if curl -s --max-time 3 http://1.1.1.1 >/dev/null 2>&1; "
            "then echo PUBLIC_OPEN; else echo PUBLIC_BLOCKED; fi"
        )
        calls = {"n": 0}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                calls["n"] += 1
                if calls["n"] == 1:
                    msg = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "tc1",
                                "type": "function",
                                "function": {
                                    "name": "run_shell_command",
                                    "arguments": json.dumps({"command": probe_cmd}),
                                },
                            }
                        ],
                    }
                    finish = "tool_calls"
                else:
                    data = json.loads(body)
                    tool_result = ""
                    for m in data.get("messages", []):
                        if m.get("role") == "tool":
                            tool_result = m.get("content", "")
                    msg = {
                        "role": "assistant",
                        "content": "RESULT: " + tool_result.strip(),
                    }
                    finish = "stop"
                resp = {
                    "id": "mock",
                    "object": "chat.completion",
                    "model": "mock",
                    "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
                payload = json.dumps(resp).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            result = self._run_swival(
                [
                    "--network",
                    "provider-only",
                    "--provider",
                    "generic",
                    "--base-url",
                    f"http://127.0.0.1:{port}",
                    "--model",
                    "mock",
                    "--api-key",
                    "dummy",
                    "--no-skills",
                    "--no-history",
                    "--no-memory",
                    "probe the network",
                ],
                cwd=str(tmp_path),
            )
        finally:
            server.shutdown()
        assert result.returncode == 0, result.stderr[-2000:]
        assert "PUBLIC_BLOCKED" in result.stdout
        assert calls["n"] == 2  # the trusted parent reached the provider twice

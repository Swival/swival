"""Tests for swival.sandbox_nono — re-exec bootstrap, argv construction, detection."""

import os
import stat
import sys
from pathlib import Path

import pytest

from swival.report import ConfigError
from swival.sandbox_nono import (
    DEFAULT_PROFILE,
    _ENV_MARKER,
    _NONO_ENV,
    _VERSION_ENV,
    _find_nono,
    build_nono_argv,
    check_sandbox_available,
    cost_map_host_reachable,
    effective_profile,
    get_nono_version,
    is_inside_nono,
    is_sandboxed,
    maybe_reexec,
    probe_nono,
    provider_credential_read_dirs,
    provider_state_dirs,
    rollback_hint,
    writable_temp_dirs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_nono_script(tmp_path, *, version_output="nono 0.59.0"):
    """Write a dummy nono script that prints its args or version."""
    script = tmp_path / "nono"
    script.write_text(
        f'#!/bin/sh\nif [ "$1" = "--version" ]; then echo "{version_output}"; exit 0; fi\necho $@\n',
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def _set_sandboxed(monkeypatch):
    """Set both markers to simulate Swival-initiated re-exec inside nono."""
    monkeypatch.setenv(_ENV_MARKER, "1")
    monkeypatch.setenv(_NONO_ENV, "/tmp/.nono-deadbeef.json")


def _set_external_nono(monkeypatch):
    """Set only NONO_CAP_FILE to simulate external `nono run` wrapping."""
    monkeypatch.delenv(_ENV_MARKER, raising=False)
    monkeypatch.setenv(_NONO_ENV, "/tmp/.nono-deadbeef.json")


def _clear_sandboxed(monkeypatch):
    monkeypatch.delenv(_ENV_MARKER, raising=False)
    monkeypatch.delenv(_NONO_ENV, raising=False)


# ===========================================================================
# is_sandboxed() — requires both markers (Swival re-exec path)
# ===========================================================================


class TestIsSandboxed:
    def test_not_sandboxed_by_default(self, monkeypatch):
        _clear_sandboxed(monkeypatch)
        assert is_sandboxed() is False

    def test_sandboxed_when_both_markers_set(self, monkeypatch):
        _set_sandboxed(monkeypatch)
        assert is_sandboxed() is True

    def test_not_sandboxed_with_only_swival_marker(self, monkeypatch):
        """Setting SWIVAL_NONO_ACTIVE alone must not bypass the check."""
        monkeypatch.setenv(_ENV_MARKER, "1")
        monkeypatch.delenv(_NONO_ENV, raising=False)
        assert is_sandboxed() is False

    def test_not_sandboxed_with_only_nono_marker(self, monkeypatch):
        _set_external_nono(monkeypatch)
        assert is_sandboxed() is False

    def test_not_sandboxed_for_empty_cap_file(self, monkeypatch):
        monkeypatch.setenv(_ENV_MARKER, "1")
        monkeypatch.setenv(_NONO_ENV, "")
        assert is_sandboxed() is False


# ===========================================================================
# is_inside_nono() — accepts external wrapping (nono's own marker only)
# ===========================================================================


class TestIsInsideNono:
    def test_false_by_default(self, monkeypatch):
        _clear_sandboxed(monkeypatch)
        assert is_inside_nono() is False

    def test_true_with_both_markers(self, monkeypatch):
        _set_sandboxed(monkeypatch)
        assert is_inside_nono() is True

    def test_true_with_only_nono_marker(self, monkeypatch):
        _set_external_nono(monkeypatch)
        assert is_inside_nono() is True

    def test_false_with_only_swival_marker(self, monkeypatch):
        monkeypatch.setenv(_ENV_MARKER, "1")
        monkeypatch.delenv(_NONO_ENV, raising=False)
        assert is_inside_nono() is False


# ===========================================================================
# _find_nono()
# ===========================================================================


class TestFindNono:
    def test_found_on_path(self, tmp_path, monkeypatch):
        script = _mock_nono_script(tmp_path)
        monkeypatch.setenv("PATH", str(tmp_path))
        assert _find_nono() == script

    def test_not_found_raises(self, monkeypatch):
        monkeypatch.setenv("PATH", "/nonexistent")
        with pytest.raises(ConfigError):
            _find_nono()

    def test_error_message_mentions_install(self, monkeypatch):
        monkeypatch.setenv("PATH", "/nonexistent")
        with pytest.raises(ConfigError, match="nono.sh"):
            _find_nono()


# ===========================================================================
# probe_nono()
# ===========================================================================


class TestProbeNono:
    def test_parses_version(self, tmp_path, monkeypatch):
        script = _mock_nono_script(tmp_path, version_output="nono 0.59.0")
        assert probe_nono(script)["version"] == "0.59.0"

    def test_parses_version_with_extra_text(self, tmp_path):
        script = _mock_nono_script(tmp_path, version_output="nono 1.2.3-beta build x")
        assert probe_nono(script)["version"] == "1.2.3"

    def test_unparsable_returns_unknown(self, tmp_path):
        script = _mock_nono_script(tmp_path, version_output="no digits here")
        assert probe_nono(script)["version"] == "unknown"

    def test_missing_binary_returns_unknown(self):
        assert probe_nono("/nonexistent/nono")["version"] == "unknown"


# ===========================================================================
# build_nono_argv()
# ===========================================================================


class TestBuildArgv:
    def test_basic_argv(self, tmp_path):
        argv = build_nono_argv(
            nono_bin="/usr/bin/nono",
            base_dir=str(tmp_path),
            add_dirs=[],
            swival_argv=["swival", "task"],
        )
        assert argv[0] == "/usr/bin/nono"
        assert argv[1] == "run"
        assert "--allow" in argv
        allow_idx = argv.index("--allow")
        assert argv[allow_idx + 1] == str(tmp_path.resolve())
        dash_idx = argv.index("--")
        assert argv[dash_idx + 1 :] == ["swival", "task"]

    def test_add_dirs_become_allow(self, tmp_path):
        extra = tmp_path / "extra"
        extra.mkdir()
        argv = build_nono_argv(
            nono_bin="/usr/bin/nono",
            base_dir=str(tmp_path),
            add_dirs=[str(extra)],
            swival_argv=["swival"],
        )
        allow_paths = [argv[i + 1] for i, v in enumerate(argv) if v == "--allow"]
        assert str(tmp_path.resolve()) in allow_paths
        assert str(extra.resolve()) in allow_paths

    def test_profile_flag(self, tmp_path):
        argv = build_nono_argv(
            nono_bin="nono",
            base_dir=str(tmp_path),
            add_dirs=[],
            profile="claude-code",
            swival_argv=["swival"],
        )
        idx = argv.index("--profile")
        assert argv[idx + 1] == "claude-code"

    def test_default_profile_is_swival(self, tmp_path):
        argv = build_nono_argv(
            nono_bin="nono",
            base_dir=str(tmp_path),
            add_dirs=[],
            swival_argv=["swival"],
        )
        idx = argv.index("--profile")
        assert argv[idx + 1] == DEFAULT_PROFILE == "swival"

    def test_read_dirs_become_read(self, tmp_path):
        d = tmp_path / "runtime"
        d.mkdir()
        argv = build_nono_argv(
            nono_bin="nono",
            base_dir=str(tmp_path),
            add_dirs=[],
            read_dirs=[str(d)],
            swival_argv=["swival"],
        )
        reads = [argv[i + 1] for i, v in enumerate(argv) if v == "--read"]
        assert str(d.resolve()) in reads

    def test_extra_allow_dirs_become_allow(self, tmp_path):
        d = tmp_path / "creds"
        d.mkdir()
        argv = build_nono_argv(
            nono_bin="nono",
            base_dir=str(tmp_path),
            add_dirs=[],
            extra_allow_dirs=[str(d)],
            swival_argv=["swival"],
        )
        allow_paths = [argv[i + 1] for i, v in enumerate(argv) if v == "--allow"]
        assert str(d.resolve()) in allow_paths

    def test_temp_dir_becomes_allow(self, tmp_path):
        argv = build_nono_argv(
            nono_bin="nono",
            base_dir=str(tmp_path),
            add_dirs=[],
            swival_argv=["swival"],
        )
        allow_paths = [argv[i + 1] for i, v in enumerate(argv) if v == "--allow"]
        for d in writable_temp_dirs():
            assert d in allow_paths

    def test_rollback_and_block_net(self, tmp_path):
        argv = build_nono_argv(
            nono_bin="nono",
            base_dir=str(tmp_path),
            add_dirs=[],
            rollback=True,
            block_net=True,
            audit_integrity=True,
            swival_argv=["swival"],
        )
        assert "--rollback" in argv
        assert "--block-net" in argv
        assert "--audit-integrity" in argv

    def test_repeatable_domain_and_credential(self, tmp_path):
        argv = build_nono_argv(
            nono_bin="nono",
            base_dir=str(tmp_path),
            add_dirs=[],
            allow_domain=["a.com", "b.com"],
            credential=["anthropic", "openai"],
            network_profile="developer",
            swival_argv=["swival"],
        )
        domains = [argv[i + 1] for i, v in enumerate(argv) if v == "--allow-domain"]
        creds = [argv[i + 1] for i, v in enumerate(argv) if v == "--credential"]
        assert domains == ["a.com", "b.com"]
        assert creds == ["anthropic", "openai"]
        np_idx = argv.index("--network-profile")
        assert argv[np_idx + 1] == "developer"

    def test_no_optional_flags_by_default(self, tmp_path):
        argv = build_nono_argv(
            nono_bin="nono",
            base_dir=str(tmp_path),
            add_dirs=[],
            swival_argv=["swival"],
        )
        for flag in (
            "--rollback",
            "--block-net",
            "--allow-domain",
            "--network-profile",
            "--credential",
            "--audit-integrity",
        ):
            assert flag not in argv

    def test_swival_argv_preserved_exactly(self, tmp_path):
        child = ["swival", "--repl", "--provider", "command"]
        argv = build_nono_argv(
            nono_bin="nono",
            base_dir=str(tmp_path),
            add_dirs=[],
            swival_argv=child,
        )
        dash_idx = argv.index("--")
        assert argv[dash_idx + 1 :] == child


# ===========================================================================
# maybe_reexec()
# ===========================================================================


class TestMaybeReexec:
    def test_noop_for_builtin_mode(self, tmp_path, monkeypatch):
        called = []
        monkeypatch.setattr(os, "execvpe", lambda *a: called.append(a))
        maybe_reexec(sandbox="builtin", base_dir=str(tmp_path), add_dirs=[])
        assert called == []

    def test_noop_when_already_inside_nono(self, tmp_path, monkeypatch):
        _set_sandboxed(monkeypatch)
        called = []
        monkeypatch.setattr(os, "execvpe", lambda *a: called.append(a))
        maybe_reexec(sandbox="nono", base_dir=str(tmp_path), add_dirs=[])
        assert called == []

    def test_raises_when_nono_missing(self, tmp_path, monkeypatch):
        _clear_sandboxed(monkeypatch)
        monkeypatch.setenv("PATH", "/nonexistent")
        with pytest.raises(ConfigError):
            maybe_reexec(sandbox="nono", base_dir=str(tmp_path), add_dirs=[])

    def test_calls_execvpe(self, tmp_path, monkeypatch):
        _clear_sandboxed(monkeypatch)
        _mock_nono_script(tmp_path)
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.setattr(sys, "argv", ["swival", "--repl"])

        captured = {}

        def fake_execvpe(file, args, env):
            captured["file"] = file
            captured["args"] = args
            captured["env"] = env

        monkeypatch.setattr(os, "execvpe", fake_execvpe)

        maybe_reexec(
            sandbox="nono",
            base_dir=str(tmp_path),
            add_dirs=[],
            profile="myprofile",
            rollback=True,
        )

        assert captured["file"].endswith("nono")
        assert captured["args"][1] == "run"
        assert "--rollback" in captured["args"]
        prof_idx = captured["args"].index("--profile")
        assert captured["args"][prof_idx + 1] == "myprofile"
        assert captured["env"][_ENV_MARKER] == "1"
        assert captured["env"][_VERSION_ENV] == "0.59.0"
        dash_idx = captured["args"].index("--")
        assert captured["args"][dash_idx + 1 :] == ["swival", "--repl"]

    def test_open_network_lets_cost_map_download(self, tmp_path, monkeypatch):
        """With no network restriction the child must not force the offline map."""
        _clear_sandboxed(monkeypatch)
        _mock_nono_script(tmp_path)
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.setattr(sys, "argv", ["swival", "task"])
        monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)

        captured = {}
        monkeypatch.setattr(os, "execvpe", lambda f, a, e: captured.update(env=e))
        maybe_reexec(sandbox="nono", base_dir=str(tmp_path), add_dirs=[])
        assert "LITELLM_LOCAL_MODEL_COST_MAP" not in captured["env"]

    def test_block_net_forces_offline_cost_map(self, tmp_path, monkeypatch):
        """A blocked network would doom the download, so force the offline map."""
        _clear_sandboxed(monkeypatch)
        _mock_nono_script(tmp_path)
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.setattr(sys, "argv", ["swival", "task"])
        monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)

        captured = {}
        monkeypatch.setattr(os, "execvpe", lambda f, a, e: captured.update(env=e))
        maybe_reexec(
            sandbox="nono", base_dir=str(tmp_path), add_dirs=[], block_net=True
        )
        assert captured["env"]["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"

    def test_allowlist_with_cost_map_host_downloads(self, tmp_path, monkeypatch):
        """If the cost-map host is in the allowlist the download can proceed."""
        _clear_sandboxed(monkeypatch)
        _mock_nono_script(tmp_path)
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.setattr(sys, "argv", ["swival", "task"])
        monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)

        captured = {}
        monkeypatch.setattr(os, "execvpe", lambda f, a, e: captured.update(env=e))
        maybe_reexec(
            sandbox="nono",
            base_dir=str(tmp_path),
            add_dirs=[],
            allow_domain=["raw.githubusercontent.com"],
        )
        assert "LITELLM_LOCAL_MODEL_COST_MAP" not in captured["env"]

    def test_does_not_chdir(self, tmp_path, monkeypatch):
        """nono enforces by path, so re-exec must not change the CWD."""
        _clear_sandboxed(monkeypatch)
        _mock_nono_script(tmp_path)
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.setattr(sys, "argv", ["swival", "task"])

        chdir_calls = []
        monkeypatch.setattr(os, "chdir", lambda p: chdir_calls.append(p))
        monkeypatch.setattr(os, "execvpe", lambda f, a, e: None)

        maybe_reexec(sandbox="nono", base_dir=str(tmp_path), add_dirs=[])
        assert chdir_calls == []

    def test_swival_marker_alone_does_not_skip_reexec(self, tmp_path, monkeypatch):
        """Only NONO_CAP_FILE proves we are inside nono; the Swival marker alone must not."""
        monkeypatch.setenv(_ENV_MARKER, "1")
        monkeypatch.delenv(_NONO_ENV, raising=False)
        _mock_nono_script(tmp_path)
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.setattr(sys, "argv", ["swival", "task"])

        called = []
        monkeypatch.setattr(os, "execvpe", lambda f, a, e: called.append(a))
        maybe_reexec(sandbox="nono", base_dir=str(tmp_path), add_dirs=[])
        assert len(called) == 1

    def test_reexec_grants_geap_gcloud_credentials(self, tmp_path, monkeypatch):
        """geap re-exec must pass the gcloud credential dir through as a --read grant."""
        _clear_sandboxed(monkeypatch)
        _mock_nono_script(tmp_path)
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.setattr(sys, "argv", ["swival", "task"])
        gcloud = tmp_path / "gcloud"
        gcloud.mkdir()
        monkeypatch.setenv("CLOUDSDK_CONFIG", str(gcloud))
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

        captured = {}
        monkeypatch.setattr(os, "execvpe", lambda f, a, e: captured.update(args=a))
        maybe_reexec(
            sandbox="nono", base_dir=str(tmp_path), add_dirs=[], provider="geap"
        )
        argv = captured["args"]
        reads = {argv[i + 1] for i, a in enumerate(argv) if a == "--read"}
        assert str(gcloud.resolve()) in reads


# ===========================================================================
# External wrapping: nono marker present, Swival marker absent
# ===========================================================================


class TestExternalWrapping:
    def test_is_inside_nono_true(self, monkeypatch):
        _set_external_nono(monkeypatch)
        assert is_inside_nono() is True

    def test_is_sandboxed_false(self, monkeypatch):
        _set_external_nono(monkeypatch)
        assert is_sandboxed() is False

    def test_maybe_reexec_skips(self, tmp_path, monkeypatch):
        _set_external_nono(monkeypatch)
        called = []
        monkeypatch.setattr(os, "execvpe", lambda *a: called.append(a))
        maybe_reexec(sandbox="nono", base_dir=str(tmp_path), add_dirs=[])
        assert called == []

    def test_check_sandbox_available_does_not_raise(self, monkeypatch):
        _set_external_nono(monkeypatch)
        check_sandbox_available()  # should not raise


# ===========================================================================
# check_sandbox_available()
# ===========================================================================


class TestCheckSandboxAvailable:
    def test_raises_when_not_inside(self, monkeypatch):
        _clear_sandboxed(monkeypatch)
        with pytest.raises(ConfigError, match="nono"):
            check_sandbox_available()

    def test_passes_when_swival_reexec(self, monkeypatch):
        _set_sandboxed(monkeypatch)
        check_sandbox_available()


# ===========================================================================
# Misc helpers
# ===========================================================================


class TestMisc:
    def test_get_nono_version_none_by_default(self, monkeypatch):
        monkeypatch.delenv(_VERSION_ENV, raising=False)
        assert get_nono_version() is None

    def test_get_nono_version_from_env(self, monkeypatch):
        monkeypatch.setenv(_VERSION_ENV, "0.59.0")
        assert get_nono_version() == "0.59.0"

    def test_rollback_hint(self):
        assert rollback_hint() == "nono rollback"

    def test_effective_profile_default(self):
        assert effective_profile(None) == "swival"
        assert effective_profile("") == "swival"

    def test_effective_profile_explicit(self):
        assert effective_profile("python-dev") == "python-dev"


class TestCostMapHostReachable:
    def test_open_network_is_reachable(self):
        assert cost_map_host_reachable(False, None) is True
        assert cost_map_host_reachable(False, []) is True

    def test_block_net_is_unreachable(self):
        assert cost_map_host_reachable(True, None) is False
        assert cost_map_host_reachable(True, ["raw.githubusercontent.com"]) is False

    def test_allowlist_without_host_is_unreachable(self):
        assert cost_map_host_reachable(False, ["api.openai.com"]) is False

    def test_allowlist_with_exact_host_is_reachable(self):
        assert cost_map_host_reachable(False, ["raw.githubusercontent.com"]) is True

    def test_allowlist_with_parent_domain_is_reachable(self):
        assert cost_map_host_reachable(False, ["githubusercontent.com"]) is True

    def test_honors_cost_map_url_override(self, monkeypatch):
        monkeypatch.setenv(
            "LITELLM_MODEL_COST_MAP_URL", "https://mirror.internal/costs.json"
        )
        assert cost_map_host_reachable(False, ["mirror.internal"]) is True
        assert cost_map_host_reachable(False, ["raw.githubusercontent.com"]) is False


class TestProviderStateDirs:
    def test_non_chatgpt_provider_has_none(self):
        assert provider_state_dirs("lmstudio") == []
        assert provider_state_dirs(None) == []

    def test_chatgpt_grants_litellm_config_root(self, monkeypatch):
        monkeypatch.delenv("CHATGPT_TOKEN_DIR", raising=False)
        dirs = provider_state_dirs("chatgpt")
        assert len(dirs) == 1
        assert dirs[0].endswith("/.config/litellm")

    def test_chatgpt_respects_token_dir_env(self, monkeypatch, tmp_path):
        token_dir = tmp_path / "custom" / "chatgpt"
        monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(token_dir))
        dirs = provider_state_dirs("chatgpt")
        assert dirs == [str(token_dir.parent)]


class TestProviderCredentialReadDirs:
    def test_plain_provider_has_none(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        assert provider_credential_read_dirs("lmstudio") == []
        assert provider_credential_read_dirs("chatgpt") == []
        assert provider_credential_read_dirs(None) == []

    def test_geap_grants_gcloud_config(self, monkeypatch):
        monkeypatch.delenv("CLOUDSDK_CONFIG", raising=False)
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        dirs = provider_credential_read_dirs("geap")
        assert dirs == [str(Path("~/.config/gcloud").expanduser().resolve())]

    def test_vertexai_alias_grants_gcloud_config(self, monkeypatch):
        monkeypatch.delenv("CLOUDSDK_CONFIG", raising=False)
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        dirs = provider_credential_read_dirs("vertexai")
        assert dirs == [str(Path("~/.config/gcloud").expanduser().resolve())]

    def test_geap_honours_cloudsdk_config(self, monkeypatch, tmp_path):
        custom = tmp_path / "gcloud"
        custom.mkdir()
        monkeypatch.setenv("CLOUDSDK_CONFIG", str(custom))
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        dirs = provider_credential_read_dirs("geap")
        assert dirs == [str(custom.resolve())]

    def test_geap_adds_service_account_key_parent(self, monkeypatch, tmp_path):
        monkeypatch.delenv("CLOUDSDK_CONFIG", raising=False)
        key = tmp_path / "creds" / "sa.json"
        key.parent.mkdir()
        key.write_text("{}")
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(key))
        dirs = provider_credential_read_dirs("geap")
        assert str(key.parent.resolve()) in dirs

    def test_bedrock_grants_aws_dir(self, monkeypatch):
        monkeypatch.delenv("AWS_SHARED_CREDENTIALS_FILE", raising=False)
        monkeypatch.delenv("AWS_CONFIG_FILE", raising=False)
        dirs = provider_credential_read_dirs("bedrock")
        assert dirs == [str(Path("~/.aws").expanduser().resolve())]

    def test_bedrock_honours_credential_file_envs(self, monkeypatch, tmp_path):
        creds = tmp_path / "awsdir" / "credentials"
        config = tmp_path / "awscfg" / "config"
        creds.parent.mkdir()
        config.parent.mkdir()
        monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(creds))
        monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
        dirs = provider_credential_read_dirs("bedrock")
        assert str(creds.parent.resolve()) in dirs
        assert str(config.parent.resolve()) in dirs

    def test_no_duplicates(self, monkeypatch):
        monkeypatch.delenv("CLOUDSDK_CONFIG", raising=False)
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        dirs = provider_credential_read_dirs("geap")
        assert len(dirs) == len(set(dirs))


class TestWritableTempDirs:
    def test_includes_system_temp_dir(self):
        import tempfile

        dirs = writable_temp_dirs()
        assert str(Path(tempfile.gettempdir()).resolve()) in dirs

    def test_honours_gettempdir(self, monkeypatch, tmp_path):
        import swival.sandbox_nono as mod

        custom = tmp_path / "scratch"
        custom.mkdir()
        monkeypatch.setattr(mod.tempfile, "gettempdir", lambda: str(custom))
        dirs = writable_temp_dirs()
        assert str(custom.resolve()) in dirs

    def test_no_duplicates(self):
        dirs = writable_temp_dirs()
        assert len(dirs) == len(set(dirs))

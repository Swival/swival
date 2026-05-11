"""Tests for swival._env.

Covers the PATH-stripping helpers that keep swival's bundled
``sys.prefix/bin`` (the Homebrew launcher prepends it, and it holds
shims for every transitive dependency: mcp, openai, litellm, python,
...) from leaking into child processes and shadowing the user's tools.
"""

import os
import sys

import pytest

from swival import _env
from swival._env import _strip_own_bin, _user_activated_own_venv, child_env


@pytest.fixture
def own_bin(tmp_path, monkeypatch):
    """Pretend sys.prefix is a temp dir so its bin/ is a real directory
    we can put on PATH for the stripping tests."""
    prefix = tmp_path / "prefix"
    (prefix / "bin").mkdir(parents=True)
    monkeypatch.setattr(_env.sys, "prefix", str(prefix))
    return str(prefix / "bin")


def _join(*entries):
    return os.pathsep.join(entries)


class TestStripOwnBin:
    def test_empty_path_returns_empty(self, own_bin):
        assert _strip_own_bin("") == ""

    def test_path_without_own_bin_unchanged(self, own_bin, tmp_path):
        other = str(tmp_path / "other")
        os.makedirs(other)
        path = _join(other, "/usr/bin", "/bin")
        assert _strip_own_bin(path) == path

    def test_own_bin_at_position_zero_removed(self, own_bin):
        path = _join(own_bin, "/usr/bin", "/bin")
        assert _strip_own_bin(path) == _join("/usr/bin", "/bin")

    def test_own_bin_in_middle_removed(self, own_bin):
        path = _join("/usr/bin", own_bin, "/bin")
        assert _strip_own_bin(path) == _join("/usr/bin", "/bin")

    def test_own_bin_via_symlink_removed(self, own_bin, tmp_path):
        link = tmp_path / "link-to-bin"
        os.symlink(own_bin, link)
        path = _join(str(link), "/usr/bin")
        assert _strip_own_bin(path) == "/usr/bin"

    def test_own_bin_appearing_twice_both_removed(self, own_bin):
        path = _join(own_bin, "/usr/bin", own_bin, "/bin")
        assert _strip_own_bin(path) == _join("/usr/bin", "/bin")

    def test_empty_entries_preserved(self, own_bin):
        # POSIX treats empty == cwd; do not silently change semantics.
        path = _join("", "/usr/bin", "")
        assert _strip_own_bin(path) == _join("", "/usr/bin", "")

    def test_realpath_oserror_keeps_entry_verbatim(self, own_bin, monkeypatch):
        # realpath does not raise for merely-missing paths on POSIX, so
        # to actually exercise the defensive branch we patch it.
        real_realpath = os.path.realpath
        weird = "/some/weird/entry"

        def fake_realpath(p, *args, **kwargs):
            if p == weird:
                raise OSError("simulated")
            return real_realpath(p, *args, **kwargs)

        monkeypatch.setattr(_env.os.path, "realpath", fake_realpath)

        path = _join(weird, own_bin, "/usr/bin")
        assert _strip_own_bin(path) == _join(weird, "/usr/bin")


class TestUserActivatedOwnVenv:
    def test_no_virtual_env_returns_false(self):
        env = {}
        assert _user_activated_own_venv(env) is False

    def test_empty_virtual_env_returns_false(self):
        assert _user_activated_own_venv({"VIRTUAL_ENV": ""}) is False

    def test_virtual_env_matches_sys_prefix(self, own_bin):
        # own_bin fixture set sys.prefix to <tmp>/prefix
        env = {"VIRTUAL_ENV": _env.sys.prefix}
        assert _user_activated_own_venv(env) is True

    def test_virtual_env_resolves_via_symlink(self, tmp_path, monkeypatch):
        prefix = tmp_path / "real-prefix"
        (prefix / "bin").mkdir(parents=True)
        monkeypatch.setattr(_env.sys, "prefix", str(prefix))
        link = tmp_path / "link-prefix"
        os.symlink(prefix, link)
        assert _user_activated_own_venv({"VIRTUAL_ENV": str(link)}) is True

    def test_virtual_env_elsewhere_returns_false(self, own_bin, tmp_path):
        other = tmp_path / "different-venv"
        other.mkdir()
        assert _user_activated_own_venv({"VIRTUAL_ENV": str(other)}) is False


class TestChildEnv:
    def test_strips_own_bin_when_not_activated(self, own_bin, monkeypatch):
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.setenv("PATH", _join(own_bin, "/usr/bin", "/bin"))
        result = child_env()
        assert result["PATH"] == _join("/usr/bin", "/bin")

    def test_extra_path_is_sanitized(self, own_bin, monkeypatch):
        # Caller passes a PATH that itself reintroduces own_bin. Strip
        # happens after merge, so the leak is still closed.
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.setenv("PATH", "/usr/bin")
        result = child_env({"PATH": _join(own_bin, "/opt/x")})
        assert result["PATH"] == "/opt/x"

    def test_extra_path_replaces_environ_path(self, own_bin, monkeypatch):
        # Merge order: extra wins over os.environ. Verifies we didn't
        # accidentally union or prepend.
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        result = child_env({"PATH": "/opt/other"})
        assert result["PATH"] == "/opt/other"

    def test_activated_venv_skips_stripping(self, own_bin, monkeypatch):
        # User activated this very venv (uv run, source activate). The
        # active env *is* sys.prefix/bin, and stripping it would
        # surprise them.
        monkeypatch.setenv("VIRTUAL_ENV", _env.sys.prefix)
        monkeypatch.setenv("PATH", _join(own_bin, "/usr/bin"))
        result = child_env()
        assert result["PATH"] == _join(own_bin, "/usr/bin")

    def test_virtual_env_elsewhere_still_strips(self, own_bin, monkeypatch, tmp_path):
        other = tmp_path / "different-venv"
        other.mkdir()
        monkeypatch.setenv("VIRTUAL_ENV", str(other))
        monkeypatch.setenv("PATH", _join(own_bin, "/usr/bin"))
        result = child_env()
        assert result["PATH"] == "/usr/bin"

    def test_extra_cannot_disable_stripping_via_virtual_env(
        self, own_bin, monkeypatch
    ):
        # The threat: parent has no VIRTUAL_ENV, but extra slips
        # VIRTUAL_ENV=<sys.prefix> in alongside a polluted PATH.
        # Activation is decided from the parent env only, so stripping
        # still happens.
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.setenv("PATH", "/usr/bin")
        result = child_env(
            {"VIRTUAL_ENV": _env.sys.prefix, "PATH": _join(own_bin, "/opt/x")}
        )
        assert result["PATH"] == "/opt/x"
        # VIRTUAL_ENV in extra is still forwarded — we just don't let
        # it disable stripping.
        assert result["VIRTUAL_ENV"] == _env.sys.prefix

    def test_path_absent_in_environ_and_extra_stays_absent(self, own_bin, monkeypatch):
        monkeypatch.delenv("PATH", raising=False)
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        result = child_env()
        assert "PATH" not in result

    def test_extra_none_treated_as_empty(self, own_bin, monkeypatch):
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.setenv("PATH", _join(own_bin, "/usr/bin"))
        result = child_env(None)
        assert result["PATH"] == "/usr/bin"

    def test_non_path_extra_passed_through(self, own_bin, monkeypatch):
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.setenv("PATH", "/usr/bin")
        result = child_env({"FASTLY_API_TOKEN": "secret"})
        assert result["FASTLY_API_TOKEN"] == "secret"
        assert result["PATH"] == "/usr/bin"

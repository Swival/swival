"""AgentFS sandbox bootstrap: re-exec Swival inside an AgentFS overlay."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .report import ConfigError

_ENV_MARKER = "SWIVAL_AGENTFS_ACTIVE"
_AGENTFS_ENV = "AGENTFS"
_VERSION_ENV = "SWIVAL_AGENTFS_VERSION"

# Minimum agentfs version that supports --strict-read.
# None means no released version supports it yet.
_STRICT_READ_MIN_VERSION: str | None = None

_PATH_FLAGS = frozenset(
    {
        "--base-dir",
        "--add-dir",
        "--add-dir-ro",
        "--skills-dir",
        "--mcp-config",
        "--objective",
        "--verify",
        "--report",
    }
)


def is_sandboxed() -> bool:
    """Return True if running inside an AgentFS sandbox.

    Checks both our own marker (set during re-exec) and the AGENTFS=1 variable
    that agentfs itself sets in the child environment.  Requiring both prevents
    a simple ``export SWIVAL_AGENTFS_ACTIVE=1`` from bypassing the sandbox.

    For external wrapping (``agentfs run -- swival ...``), only AGENTFS=1 is
    present.  That is also accepted — see ``is_inside_agentfs()``.
    """
    return _has_swival_marker() and _has_agentfs_env()


def is_inside_agentfs() -> bool:
    """Return True if the process is inside agentfs (any entry path).

    True when either:
    - Swival re-exec'd itself (both markers set), or
    - The user wrapped Swival externally with ``agentfs run`` (only AGENTFS=1).
    """
    return _has_agentfs_env()


def _has_swival_marker() -> bool:
    return os.environ.get(_ENV_MARKER) == "1"


def _has_agentfs_env() -> bool:
    return os.environ.get(_AGENTFS_ENV) == "1"


def _find_agentfs() -> str:
    """Locate the agentfs binary. Raises ConfigError if not found."""
    path = shutil.which("agentfs")
    if path is None:
        raise ConfigError(
            "agentfs binary not found on PATH. "
            "Install AgentFS (https://github.com/tursodatabase/agentfs) "
            "or use --sandbox builtin."
        )
    return path


def probe_agentfs(agentfs_bin: str) -> dict:
    """Probe the agentfs binary for version and capability information.

    Runs ``agentfs --version``, parses the output, and returns a dict with:
    - ``version``: version string (e.g. ``"0.6.2"``) or ``"unknown"``
    - ``supports_strict_read``: whether the installed version supports strict read mode

    On any failure (crash, timeout, unparsable output), returns a safe fallback.
    """
    try:
        proc = subprocess.run(
            [agentfs_bin, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = proc.stdout.strip()
        # Parse patterns like "agentfs v0.6.2" or "agentfs 0.6.2-3-gabcdef-dirty"
        m = re.search(r"v?(\d+\.\d+\.\d+)", output)
        version = m.group(1) if m else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        version = "unknown"

    supports_strict_read = False
    if _STRICT_READ_MIN_VERSION is not None and version != "unknown":
        supports_strict_read = _version_gte(version, _STRICT_READ_MIN_VERSION)

    return {"version": version, "supports_strict_read": supports_strict_read}


def _version_gte(version: str, minimum: str) -> bool:
    """Return True if *version* >= *minimum* using simple numeric comparison."""

    def _parts(v: str) -> tuple[int, ...]:
        return tuple(int(x) for x in v.split("."))

    return _parts(version) >= _parts(minimum)


def get_agentfs_version() -> str | None:
    """Return the agentfs version string if running inside an AgentFS sandbox.

    The version is propagated via the ``SWIVAL_AGENTFS_VERSION`` env var
    during re-exec.  Returns ``None`` when not sandboxed or when the env
    var is absent.
    """
    return os.environ.get(_VERSION_ENV)


def _absolutize_argv(argv: list[str]) -> list[str]:
    """Return a copy of argv with path-bearing flag values resolved to absolute paths.

    Handles both ``--flag value`` (two tokens) and ``--flag=value`` (one token)
    forms.  This prevents relative paths from breaking when CWD changes before
    re-exec.
    """
    result = list(argv)
    i = 0
    while i < len(result):
        token = result[i]

        # Stop at argument terminator — everything after is positional.
        if token == "--":
            break

        # --flag=value form
        eq = token.find("=")
        if eq != -1 and token[:eq] in _PATH_FLAGS:
            flag = token[:eq]
            value = token[eq + 1 :]
            result[i] = flag + "=" + str(Path(value).expanduser().resolve())
            i += 1
            continue

        # --flag value form
        if token in _PATH_FLAGS and i + 1 < len(result):
            result[i + 1] = str(Path(result[i + 1]).expanduser().resolve())
            i += 2
            continue

        i += 1
    return result


def build_agentfs_argv(
    *,
    agentfs_bin: str,
    base_dir: str,
    add_dirs: list[str],
    session: str | None,
    swival_argv: list[str],
) -> list[str]:
    """Build the full argv for re-execing Swival inside agentfs run.

    Returns a list like:
        ["agentfs", "run", "--no-default-allows", "--allow", "/path", ...,
         "--session", "id", "--", "swival", ...]
    """
    argv = [agentfs_bin, "run", "--no-default-allows"]

    resolved_base = str(Path(base_dir).resolve())
    argv.extend(["--allow", resolved_base])

    for d in add_dirs:
        resolved = str(Path(d).expanduser().resolve())
        argv.extend(["--allow", resolved])

    if session:
        argv.extend(["--session", session])

    argv.append("--")
    argv.extend(swival_argv)
    return argv


def maybe_reexec(
    *,
    sandbox: str,
    sandbox_session: str | None,
    base_dir: str,
    add_dirs: list[str],
    sandbox_strict_read: bool = False,
) -> None:
    """Re-exec Swival inside AgentFS if sandbox mode requires it.

    Called early in startup, before the agent loop. Does nothing if:
    - sandbox != "agentfs"
    - Already running inside AgentFS (both env markers set)

    When *sandbox_strict_read* is True, probes the agentfs binary for
    strict-read support and raises ``ConfigError`` if the installed
    version does not support it.

    On success, this function does not return (os.execvpe replaces the process).
    On failure, raises ConfigError.
    """
    if sandbox != "agentfs":
        return

    if is_sandboxed():
        return

    agentfs_bin = _find_agentfs()

    probe = probe_agentfs(agentfs_bin)

    if sandbox_strict_read and not probe["supports_strict_read"]:
        raise ConfigError(
            f"--sandbox-strict-read requires AgentFS with strict read support "
            f"(installed: {probe['version']}). "
            f"No current version supports this feature yet."
        )

    resolved_base = str(Path(base_dir).resolve())

    # Resolve all path-bearing flags to absolute before re-exec, because
    # we chdir to base_dir below and relative paths would break.
    child_argv = _absolutize_argv(sys.argv)

    argv = build_agentfs_argv(
        agentfs_bin=agentfs_bin,
        base_dir=resolved_base,
        add_dirs=add_dirs,
        session=sandbox_session,
        swival_argv=child_argv,
    )

    env = os.environ.copy()
    env[_ENV_MARKER] = "1"
    env[_VERSION_ENV] = probe["version"]

    # AgentFS overlays the process CWD. Ensure it matches base_dir so the
    # overlay workspace aligns with the directory Swival considers writable.
    os.chdir(resolved_base)

    os.execvpe(argv[0], argv, env)


def check_sandbox_available() -> None:
    """Raise ConfigError if sandbox="agentfs" is requested but we are not inside agentfs.

    Called by Session to fail fast for library users — the re-exec path only
    works for the CLI entry point, not for programmatic API usage.

    Accepts both Swival-initiated re-exec (both markers) and external wrapping
    (``agentfs run -- python script.py``, which only sets AGENTFS=1).
    """
    if not is_inside_agentfs():
        raise ConfigError(
            'sandbox="agentfs" requires running inside an AgentFS sandbox. '
            "Use the CLI (swival --sandbox agentfs) for automatic re-exec, "
            "or wrap your process with `agentfs run` externally."
        )

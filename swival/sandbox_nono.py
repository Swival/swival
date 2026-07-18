"""Nono sandbox bootstrap: re-exec Swival inside a nono sandbox.

nono is a capability-based sandbox runtime for AI agents.  It uses Landlock
(Linux) and Seatbelt (macOS) to enforce filesystem and network boundaries.
We run it in supervised mode (``nono run -- swival ...``): the parent stays
unsandboxed and provides audit, network proxy, and rollback services while the
child inherits the enforced sandbox.

This module mirrors ``sandbox_agentfs.py``: a ``maybe_reexec()`` that replaces
the process early in startup, plus detection helpers.  It does not route
through the unused ``SandboxBackend`` interface.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from .report import ConfigError
from .sandbox_agentfs import _absolutize_argv

_ENV_MARKER = "SWIVAL_NONO_ACTIVE"
_NONO_ENV = "NONO_CAP_FILE"
_VERSION_ENV = "SWIVAL_NONO_VERSION"

# litellm fetches its model cost map from this host at import time. Knowing it
# lets us decide whether the sandbox's network policy will let the download
# through. The URL is overridable via LITELLM_MODEL_COST_MAP_URL.
_COST_MAP_DEFAULT_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)


def cost_map_host_reachable(block_net: bool, allow_domain: list[str] | None) -> bool:
    """Return True if the nono network policy can reach the cost-map host.

    litellm tries to download a fresh model cost map when it is imported, and
    that import happens inside the sandbox. A blocked or allowlisted network
    turns the fetch into a doomed request that wastes litellm's timeout on
    every startup and trips ``--audit-integrity``. When we can tell the host is
    unreachable we force litellm's bundled offline map instead.

    Conservative on purpose: only ``--block-net`` and a non-empty
    ``--allow-domain`` allowlist are treated as restrictions we can reason
    about. A bare ``--network-profile`` is opaque, so we let the download try
    and rely on litellm's own offline fallback.
    """
    if block_net:
        return False
    if allow_domain:
        url = os.environ.get("LITELLM_MODEL_COST_MAP_URL", _COST_MAP_DEFAULT_URL)
        host = urlparse(url).hostname or ""
        return any(host == d or host.endswith("." + d) for d in allow_domain)
    return True


# nono ships a built-in "swival" profile that grants the Python runtime,
# user tools, and Swival's own config/state directories.  Without it the
# re-exec'd interpreter is not reachable inside the sandbox and exec fails
# with code 127.  Used as the default when no profile is requested.
DEFAULT_PROFILE = "swival"


def effective_profile(profile: str | None) -> str:
    """Return the nono profile to use, defaulting to the built-in swival profile."""
    return profile or DEFAULT_PROFILE


def is_sandboxed() -> bool:
    """Return True if Swival re-exec'd itself inside a nono sandbox.

    Requires both our own marker (set during re-exec) and the ``NONO_CAP_FILE``
    variable that nono always sets in the child environment.  Requiring both
    prevents a bare ``export SWIVAL_NONO_ACTIVE=1`` from masquerading as a
    sandbox.

    For external wrapping (``nono run -- swival ...``), only ``NONO_CAP_FILE``
    is present.  That case is accepted by ``is_inside_nono()``.
    """
    return _has_swival_marker() and _has_nono_env()


def is_inside_nono() -> bool:
    """Return True if the process is inside nono (any entry path).

    True when either Swival re-exec'd itself (both markers set) or the user
    wrapped Swival externally with ``nono run`` (only ``NONO_CAP_FILE`` set).
    """
    return _has_nono_env()


def is_net_blocked() -> bool:
    """Return True if the surrounding nono sandbox provably blocks the network.

    nono writes its effective policy to the capability file that
    ``NONO_CAP_FILE`` points at; ``net_blocked`` is true only when the
    sandbox was started with ``--block-net``.  The marker alone proves
    nothing about network policy, so an externally wrapped ``nono run``
    without ``--block-net`` must not pass for an air gap.  Anything short
    of positive proof — no marker, unreadable file, missing key — returns
    False so offline claims fail closed.
    """
    cap_path = os.environ.get(_NONO_ENV)
    if not cap_path:
        return False
    try:
        with open(cap_path, encoding="utf-8") as fh:
            cap = json.load(fh)
    except (OSError, ValueError):
        return False
    return isinstance(cap, dict) and cap.get("net_blocked") is True


def verify_net_blocked(claim: str) -> None:
    """Raise ConfigError unless the surrounding sandbox provably blocks the net.

    *claim* names whatever promised the air gap (a CLI flag, a Session
    kwarg) so the message points at the caller's own configuration.
    """
    if not is_net_blocked():
        raise ConfigError(
            f"{claim} requires the surrounding nono sandbox to block the "
            "network, but its capability file does not record --block-net. "
            "Re-run the outer `nono run` with --block-net."
        )


def _has_swival_marker() -> bool:
    return os.environ.get(_ENV_MARKER) == "1"


def _has_nono_env() -> bool:
    return bool(os.environ.get(_NONO_ENV))


def _append_resolved_unique(
    paths: list[str], seen: set[str], p: "str | Path | None"
) -> None:
    if not p:
        return
    resolved = str(Path(p).resolve())
    if resolved not in seen:
        seen.add(resolved)
        paths.append(resolved)


def _find_nono(missing_error: str | None = None) -> str:
    """Locate the nono binary. Raises ConfigError with *missing_error*
    (or a --sandbox-flavored default) if not found."""
    path = shutil.which("nono")
    if path is None:
        raise ConfigError(
            missing_error
            or "nono binary not found on PATH. "
            "Install nono (https://nono.sh) or use --sandbox builtin."
        )
    return path


_WRAPPER_MISSING_NONO = (
    'network = "provider-only" requires the nono binary to sandbox '
    "agent commands. Install nono (https://nono.sh) or use --network full."
)


def probe_nono(nono_bin: str) -> dict:
    """Probe the nono binary for version information.

    Runs ``nono --version`` and parses output like ``nono 0.59.0``.  Returns
    ``{"version": "x.y.z"}`` with ``"unknown"`` on any failure.
    """
    try:
        proc = subprocess.run(
            [nono_bin, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = proc.stdout.strip()
        m = re.search(r"v?(\d+\.\d+\.\d+)", output)
        version = m.group(1) if m else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        version = "unknown"
    return {"version": version}


def _runtime_read_paths() -> list[str]:
    """Return directories that must be readable for the re-exec'd Swival to import.

    The built-in nono ``swival`` profile grants common Python runtime locations,
    but not every install layout is covered: a ``uv tool`` venv, a virtualenv, or
    an editable checkout can live anywhere.  We add read-only grants for the
    Swival import root and the interpreter prefixes so ``import swival`` and the
    standard library resolve regardless of how Swival was installed.
    """
    paths: list[str] = []
    seen: set[str] = set()

    # The directory on sys.path that contains the `swival` package: repo root
    # for an editable install, site-packages for a regular install.
    _append_resolved_unique(paths, seen, Path(__file__).parent.parent)
    _append_resolved_unique(paths, seen, sys.prefix)
    _append_resolved_unique(paths, seen, sys.base_prefix)
    return paths


def writable_temp_dirs() -> list[str]:
    """Return the platform temporary directories to grant read+write inside nono.

    Tools routinely scratch in the system temp directory: compilers, package
    managers, and Swival's own large-output spool all expect it to be writable.
    The built-in nono ``swival`` profile does not cover it, so without an
    explicit grant those tools fail the moment they touch ``/tmp``.

    ``tempfile.gettempdir()`` honours ``$TMPDIR`` and is the canonical answer
    per platform.  On macOS that resolves to a per-user ``/var/folders/...``
    path, but plenty of tools hardcode ``/tmp`` regardless, so we add it too
    whenever it exists and differs from the resolved temp dir.
    """
    paths: list[str] = []
    seen: set[str] = set()

    _append_resolved_unique(paths, seen, tempfile.gettempdir())
    if os.path.isdir("/tmp"):
        _append_resolved_unique(paths, seen, "/tmp")
    return paths


def provider_state_dirs(provider: str | None) -> list[str]:
    """Return writable directories a provider needs for its credentials/state.

    The built-in nono ``swival`` profile deliberately denies credential and
    keychain locations.  A provider that stores its own auth tokens on disk
    therefore needs an explicit grant, or it fails when it tries to read or
    create its token directory inside the sandbox.

    Currently only the ``chatgpt`` provider needs this: it caches OAuth tokens
    under ``~/.config/litellm/chatgpt`` (overridable via ``CHATGPT_TOKEN_DIR``).
    We grant the litellm config root so token creation and reads both work.
    """
    if provider == "chatgpt":
        token_dir = os.environ.get(
            "CHATGPT_TOKEN_DIR",
            os.path.expanduser("~/.config/litellm/chatgpt"),
        )
        # Grant the parent so first-run directory creation succeeds too.
        return [str(Path(token_dir).expanduser().parent)]
    return []


def provider_credential_read_dirs(provider: str | None) -> list[str]:
    """Return read-only directories holding a provider's login credentials.

    Some providers authenticate against credential files that the built-in nono
    ``swival`` profile denies.  Unlike ``provider_state_dirs``, these are granted
    read-only: the provider reads a long-lived credential from disk and exchanges
    it for a short-lived access token over the network, so it never writes the
    file back from inside the sandbox.

    - ``geap``/``vertexai`` (Vertex AI): litellm's vertex path loads Application
      Default Credentials from ``~/.config/gcloud`` (``CLOUDSDK_CONFIG`` relocates
      that directory).  ``GOOGLE_APPLICATION_CREDENTIALS`` may point a service
      account key anywhere, so its parent is granted too when set.
    - ``bedrock`` (AWS): boto3 reads credentials and config from ``~/.aws``.
      ``AWS_SHARED_CREDENTIALS_FILE`` and ``AWS_CONFIG_FILE`` relocate those
      files, so their parents are granted when set.
    """
    dirs: list[str] = []
    seen: set[str] = set()

    if provider in ("geap", "vertexai"):
        _append_resolved_unique(
            dirs,
            seen,
            os.path.expanduser(os.environ.get("CLOUDSDK_CONFIG") or "~/.config/gcloud"),
        )
        adc = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if adc:
            _append_resolved_unique(dirs, seen, Path(adc).expanduser().parent)
    elif provider == "bedrock":
        _append_resolved_unique(dirs, seen, os.path.expanduser("~/.aws"))
        for env in ("AWS_SHARED_CREDENTIALS_FILE", "AWS_CONFIG_FILE"):
            val = os.environ.get(env)
            if val:
                _append_resolved_unique(dirs, seen, Path(val).expanduser().parent)

    return dirs


def get_nono_version() -> str | None:
    """Return the nono version if running inside a nono sandbox.

    Propagated via the ``SWIVAL_NONO_VERSION`` env var during re-exec.  Returns
    ``None`` when not sandboxed or when the variable is absent.
    """
    return os.environ.get(_VERSION_ENV)


def rollback_hint() -> str:
    """Return the nono command for reviewing rollback snapshots."""
    return "nono rollback"


def build_nono_argv(
    *,
    nono_bin: str,
    base_dir: str,
    add_dirs: list[str],
    profile: str | None = None,
    rollback: bool = False,
    block_net: bool = False,
    allow_domain: list[str] | None = None,
    network_profile: str | None = None,
    credential: list[str] | None = None,
    audit_integrity: bool = False,
    read_dirs: list[str] | None = None,
    extra_allow_dirs: list[str] | None = None,
    silent: bool = False,
    swival_argv: list[str],
) -> list[str]:
    """Build the full argv for re-execing Swival inside ``nono run``.

    The writable base directory is granted with ``--allow`` (read+write,
    recursive).  Extra ``--add-dir`` entries map to additional ``--allow``
    grants.  The platform temporary directory is always granted so tools that
    scratch in ``/tmp`` keep working.  When no profile is requested, the
    built-in ``swival`` profile is applied so the Python runtime and Swival's
    config/state directories are reachable inside the sandbox.
    """
    argv = [nono_bin, "run"]
    if silent:
        argv.append("--silent")

    resolved_base = str(Path(base_dir).resolve())
    argv.extend(["--allow", resolved_base])

    allow_dirs = list(add_dirs) + list(extra_allow_dirs or []) + writable_temp_dirs()
    for d in allow_dirs:
        resolved = str(Path(d).expanduser().resolve())
        argv.extend(["--allow", resolved])

    for d in read_dirs or []:
        argv.extend(["--read", str(Path(d).expanduser().resolve())])

    argv.extend(["--profile", effective_profile(profile)])
    if rollback:
        argv.append("--rollback")
    if block_net:
        argv.append("--block-net")
    for domain in allow_domain or []:
        argv.extend(["--allow-domain", domain])
    if network_profile:
        argv.extend(["--network-profile", network_profile])
    for service in credential or []:
        argv.extend(["--credential", service])
    if audit_integrity:
        argv.append("--audit-integrity")

    argv.append("--")
    argv.extend(swival_argv)
    return argv


def maybe_reexec(
    *,
    sandbox: str,
    base_dir: str,
    add_dirs: list[str],
    provider: str | None = None,
    profile: str | None = None,
    rollback: bool = False,
    block_net: bool = False,
    allow_domain: list[str] | None = None,
    network_profile: str | None = None,
    credential: list[str] | None = None,
    audit_integrity: bool = False,
) -> None:
    """Re-exec Swival inside nono if sandbox mode requires it.

    Called early in startup, before the agent loop.  Does nothing if:
    - sandbox != "nono"
    - Already running inside nono (``NONO_CAP_FILE`` set), which also covers
      the case where the user wrapped Swival externally with ``nono run``.

    Unlike agentfs, nono enforces by path rather than by overlaying the CWD,
    so we do not ``os.chdir()``.  Path-bearing argv flags are still resolved to
    absolute as a safety measure.

    On success, this function does not return (os.execvpe replaces the
    process).  On failure, raises ConfigError.
    """
    if sandbox != "nono":
        return

    if is_inside_nono():
        return

    nono_bin = _find_nono()
    probe = probe_nono(nono_bin)

    child_argv = _absolutize_argv(sys.argv)

    argv = build_nono_argv(
        nono_bin=nono_bin,
        base_dir=base_dir,
        add_dirs=add_dirs,
        profile=profile,
        rollback=rollback,
        block_net=block_net,
        allow_domain=allow_domain,
        network_profile=network_profile,
        credential=credential,
        audit_integrity=audit_integrity,
        read_dirs=_runtime_read_paths() + provider_credential_read_dirs(provider),
        extra_allow_dirs=provider_state_dirs(provider),
        swival_argv=child_argv,
    )

    # Do NOT strip sys.prefix/bin from PATH here. The exec target is another
    # swival process, which needs its bundled bin/ reachable on entry. User
    # tool spawns inside the re-exec'd swival are protected separately.
    env = os.environ.copy()
    env[_ENV_MARKER] = "1"
    env[_VERSION_ENV] = probe["version"]

    # The re-exec'd Swival imports litellm inside the sandbox, which would try
    # to download the model cost map. If the network policy cannot reach the
    # host, skip the doomed fetch and use litellm's bundled offline map.
    if not cost_map_host_reachable(block_net, allow_domain):
        env.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    os.execvpe(argv[0], argv, env)


def build_block_net_wrapper(
    *,
    base_dir: str,
    add_dirs: list[str] | None = None,
    read_dirs: list[str] | None = None,
) -> list[str]:
    """Build the ``nono run`` prefix that network-jails agent subprocesses.

    Used by ``network = "provider-only"``: the Swival process keeps its
    normal network access for provider calls, while every agent-initiated
    subprocess (commands, shell, python tool, stdio MCP servers) is launched
    through this prefix so its entire subtree runs under ``--block-net``.

    The returned argv ends with ``--``; callers append the payload command.
    Filesystem grants mirror the full nono re-exec (base dir, extra dirs,
    temp dirs, Python runtime, and the built-in ``swival`` profile) because
    this layer is only meant to restrict the network. ``--silent`` keeps
    nono's banner and summary out of captured tool output, and denials fail
    closed because wrapped children never have a TTY to prompt on.
    """
    return build_nono_argv(
        nono_bin=_find_nono(_WRAPPER_MISSING_NONO),
        base_dir=base_dir,
        add_dirs=list(add_dirs or []),
        block_net=True,
        read_dirs=_runtime_read_paths() + list(read_dirs or []),
        silent=True,
        swival_argv=[],
    )


def check_wrapper_available() -> None:
    """Raise ConfigError unless the block-net child wrapper can work here.

    ``network = "provider-only"`` needs a nono binary to jail agent
    subprocesses, and it cannot run inside an existing nono sandbox because
    the wrapper would have to nest nono inside nono.
    """
    if is_inside_nono():
        raise ConfigError(
            'network = "provider-only" cannot run inside an existing nono '
            "sandbox; the outer sandbox already owns network policy. "
            'Use --network none for an air-gapped run, or drop the outer "nono run".'
        )
    _find_nono(_WRAPPER_MISSING_NONO)


def check_sandbox_available() -> None:
    """Raise ConfigError if sandbox="nono" is requested but we are not inside nono.

    Called by Session to fail fast for library users — the re-exec path only
    works for the CLI entry point, not for programmatic API usage.

    Accepts both Swival-initiated re-exec (both markers) and external wrapping
    (``nono run -- python script.py``, which sets only ``NONO_CAP_FILE``).
    """
    if not is_inside_nono():
        raise ConfigError(
            'sandbox="nono" requires running inside a nono sandbox. '
            "Use the CLI (swival --sandbox nono) for automatic re-exec, "
            "or wrap your process with `nono run` externally."
        )

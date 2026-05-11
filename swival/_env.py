"""Subprocess environment helpers.

The Homebrew launcher (and some other venv-based installers) prepend
``<sys.prefix>/bin`` to ``PATH`` so swival's bundled entry points
resolve. That directory also holds console scripts for every transitive
dependency (mcp, openai, litellm, python, ...), any of which can shadow
a tool a child process tries to run by name.

``child_env`` returns an environment with that directory removed from
``PATH``, unless the user has deliberately activated this venv (in
which case ``sys.prefix/bin`` *is* their active environment and we
leave it alone).

See ``PLAN-subprocess-path-leak.md`` in the repo root.
"""

import os
import sys
from collections.abc import Mapping


def child_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build an environment for a child process with swival's bundled
    venv ``bin/`` removed from ``PATH``.

    Order matters:

    * The activation check is read from the **parent** environment
      before merging ``extra``. ``extra`` is child configuration, not
      evidence about how swival itself was launched. A caller that
      slips ``VIRTUAL_ENV`` into ``extra`` must not be able to disable
      stripping for itself.
    * ``PATH`` is stripped **after** merging ``extra`` so a
      caller-supplied ``PATH`` (e.g., a per-server ``config["env"]``
      block in ``mcp.json``) is also sanitized.

    ``PATH`` is only set in the returned dict if it was present in the
    merged environment; an absent ``PATH`` is preserved as absent.
    """
    user_activated = _user_activated_own_venv(os.environ)

    env = os.environ.copy()
    if extra:
        env.update(extra)

    if user_activated:
        return env

    if "PATH" in env:
        env["PATH"] = _strip_own_bin(env["PATH"])
    return env


def _user_activated_own_venv(env: Mapping[str, str]) -> bool:
    """True if the parent env declares an active venv that *is* this
    swival's ``sys.prefix``.

    ``uv run`` and ``source .venv/bin/activate`` both set
    ``VIRTUAL_ENV`` to the activated venv root. The Homebrew launcher
    does not. So when ``VIRTUAL_ENV`` resolves to ``sys.prefix`` the
    user has deliberately put this environment on their ``PATH`` and
    expects its tools to be reachable from children.
    """
    venv = env.get("VIRTUAL_ENV")
    if not venv:
        return False
    try:
        return os.path.realpath(venv) == os.path.realpath(sys.prefix)
    except OSError:
        return False


def _strip_own_bin(path: str) -> str:
    """Return ``path`` with any entry that resolves to
    ``<sys.prefix>/bin`` removed.

    Empty entries are preserved (POSIX treats them as the current
    directory). Entries whose ``realpath`` raises are kept verbatim.
    """
    if not path:
        return path
    try:
        own = os.path.realpath(os.path.join(sys.prefix, "bin"))
    except OSError:
        return path
    sep = os.pathsep
    kept = []
    for entry in path.split(sep):
        if not entry:
            kept.append(entry)
            continue
        try:
            resolved = os.path.realpath(entry)
        except OSError:
            kept.append(entry)
            continue
        if resolved != own:
            kept.append(entry)
    return sep.join(kept)

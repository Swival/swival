"""Generic lifecycle hooks for Swival.

Runs a user-configured command at startup and exit with Git/project metadata
passed via SWIVAL_* environment variables. The hook can hydrate .swival/ from
remote storage on startup and push it back on exit.
"""

import hashlib
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


class LifecycleError(Exception):
    """Raised when a lifecycle hook fails in fail-closed mode."""


_GIT_ENV_MAP = {
    "git_present": "SWIVAL_GIT_PRESENT",
    "repo_root": "SWIVAL_REPO_ROOT",
    "project_rel": "SWIVAL_PROJECT_REL",
    "git_head": "SWIVAL_GIT_HEAD",
    "git_dirty": "SWIVAL_GIT_DIRTY",
    "git_remote": "SWIVAL_GIT_REMOTE",
    "repo_hash": "SWIVAL_REPO_HASH",
    "project_hash": "SWIVAL_PROJECT_HASH",
}


def _git_metadata(base_dir: str) -> dict[str, str]:
    """Discover Git metadata for the project rooted at *base_dir*.

    Returns a dict of SWIVAL_* keys (without the prefix) to string values.
    Non-Git directories get ``git_present=0`` and empty Git fields.
    """
    result: dict[str, str] = {}

    try:
        repo_root = (
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=base_dir,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        result["git_present"] = "0"
        return result

    result["git_present"] = "1"
    result["repo_root"] = repo_root

    # HEAD sha
    try:
        head_sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=base_dir,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        result["git_head"] = head_sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Dirty? (includes staged changes, unstaged changes, and untracked files)
    try:
        status_out = (
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=base_dir,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        result["git_dirty"] = "1" if status_out else "0"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Remote origin URL
    try:
        origin_url = (
            subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"],
                cwd=base_dir,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        if origin_url:
            result["git_remote"] = origin_url
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # project_rel: relative path from repo root to base_dir
    try:
        base_resolved = Path(base_dir).resolve()
        root_resolved = Path(repo_root).resolve()
        rel = base_resolved.relative_to(root_resolved)
        result["project_rel"] = str(rel) if str(rel) != "." else ""
    except ValueError:
        result["project_rel"] = ""

    # Normalized remote (collapse SSH/HTTPS to same form)
    raw_remote = result.get("git_remote", "")
    normalized = _normalize_remote(raw_remote) if raw_remote else ""

    # repo_hash and project_hash
    if normalized:
        result["repo_hash"] = hashlib.sha256(normalized.encode()).hexdigest()[:48]
        project_key = normalized + ":" + result.get("project_rel", "")
        result["project_hash"] = hashlib.sha256(project_key.encode()).hexdigest()[:48]
    elif repo_root:
        result["repo_hash"] = hashlib.sha256(repo_root.encode()).hexdigest()[:48]
        project_key = repo_root + ":" + result.get("project_rel", "")
        result["project_hash"] = hashlib.sha256(project_key.encode()).hexdigest()[:48]

    return result


def _normalize_remote(url: str) -> str:
    """Normalize a Git remote URL so SSH and HTTPS forms collapse."""
    url = url.strip()
    if url.endswith(".git"):
        url = url[:-4]

    # SSH: git@github.com:org/repo -> github.com/org/repo
    if url.startswith("git@"):
        url = url[4:]
        url = url.replace(":", "/", 1)
        return url

    # HTTPS: https://github.com/org/repo -> github.com/org/repo
    for prefix in ("https://", "http://", "ssh://"):
        if url.startswith(prefix):
            url = url[len(prefix) :]
            # Strip user@ if present (ssh://git@...)
            if "@" in url.split("/")[0]:
                url = url.split("@", 1)[1]
            return url

    return url


def build_hook_env(
    *,
    event: str,
    resolved_base: str,
    provider: str | None = None,
    model: str | None = None,
    git_meta: dict[str, str] | None = None,
    report_path: str | None = None,
    outcome: str | None = None,
    exit_code: int | None = None,
) -> dict[str, str]:
    """Build the SWIVAL_* environment dict for a hook invocation.

    *resolved_base* must be an already-resolved absolute path.
    """
    env = dict(os.environ)

    env["SWIVAL_HOOK_EVENT"] = event
    env["SWIVAL_BASE_DIR"] = resolved_base
    env["SWIVAL_SWIVAL_DIR"] = str(Path(resolved_base) / ".swival")

    if provider:
        env["SWIVAL_PROVIDER"] = provider
    if model:
        env["SWIVAL_MODEL"] = model

    if git_meta is None:
        git_meta = _git_metadata(resolved_base)

    for key, env_key in _GIT_ENV_MAP.items():
        val = git_meta.get(key)
        if val is not None:
            env[env_key] = val

    # Exit-only variables
    if event == "exit":
        if report_path:
            env["SWIVAL_REPORT"] = report_path
        if outcome:
            env["SWIVAL_OUTCOME"] = outcome
        if exit_code is not None:
            env["SWIVAL_EXIT_CODE"] = str(exit_code)

    return env


def _warn_hook(result: dict, verbose: bool) -> None:
    """Log a lifecycle hook warning if verbose."""
    if not verbose:
        return
    print(f"warning: {result['error']}", file=sys.stderr)
    stderr = result.get("stderr", "")
    if stderr:
        print(f"  stderr: {stderr[:500]}", file=sys.stderr)


def run_lifecycle_hook(
    command: str,
    event: str,
    base_dir: str,
    *,
    timeout: int = 300,
    fail_closed: bool = False,
    provider: str | None = None,
    model: str | None = None,
    git_meta: dict[str, str] | None = None,
    report_path: str | None = None,
    outcome: str | None = None,
    exit_code: int | None = None,
    verbose: bool = False,
) -> dict:
    """Run a lifecycle hook command.

    Returns a dict with keys: event, exit_code, stdout, stderr, duration, error.
    """
    resolved = str(Path(base_dir).resolve())
    argv = shlex.split(command) + [event, resolved]

    env = build_hook_env(
        event=event,
        resolved_base=resolved,
        provider=provider,
        model=model,
        git_meta=git_meta,
        report_path=report_path,
        outcome=outcome,
        exit_code=exit_code,
    )

    result = {
        "event": event,
        "exit_code": None,
        "stdout": "",
        "stderr": "",
        "duration": 0.0,
        "error": None,
    }

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            cwd=resolved,
            env=env,
            capture_output=True,
            timeout=timeout,
        )
        result["exit_code"] = proc.returncode
        result["stdout"] = proc.stdout.decode(errors="replace")
        result["stderr"] = proc.stderr.decode(errors="replace")
        result["duration"] = time.monotonic() - t0

        if proc.returncode != 0:
            result["error"] = (
                f"lifecycle {event} hook exited with code {proc.returncode}"
            )
            if fail_closed:
                raise LifecycleError(result["error"])
            _warn_hook(result, verbose)

    except subprocess.TimeoutExpired:
        result["duration"] = time.monotonic() - t0
        result["error"] = f"lifecycle {event} hook timed out after {timeout}s"
        if fail_closed:
            raise LifecycleError(result["error"])
        _warn_hook(result, verbose)

    except FileNotFoundError as e:
        result["duration"] = time.monotonic() - t0
        result["error"] = f"lifecycle {event} hook failed to start: {e}"
        if fail_closed:
            raise LifecycleError(result["error"])
        _warn_hook(result, verbose)

    return result

"""Shared helpers for isolated agent loops and temporary Git worktrees."""

from __future__ import annotations

import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import threading

    from .input_dispatch import InputContext

_GIT_TIMEOUT = 30
_GIT_CONFIG_ENV_PREFIXES = ("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")
_GIT_REPOSITORY_ENV = {
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_ATTR_SOURCE",
    "GIT_CEILING_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_CONFIG",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_PARAMETERS",
    "GIT_DIR",
    "GIT_GRAFT_FILE",
    "GIT_IMPLICIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_NO_REPLACE_OBJECTS",
    "GIT_OBJECT_DIRECTORY",
    "GIT_PREFIX",
    "GIT_REPLACE_REF_BASE",
    "GIT_SHALLOW_FILE",
    "GIT_WORK_TREE",
}
_GIT_SAFE_CONFIG = (
    "safe.bareRepository=explicit",
    f"core.hooksPath={os.devnull}",
    "core.fsmonitor=false",
    "attr.tree=",
    "core.attributesFile=",
)


def fd_anchored_cwd(root_fd: int) -> "str | None":
    """A fork-safe, fd-anchored working directory for a git subprocess, or
    None where the platform cannot provide one.

    On Linux ``/proc/self/fd/<n>`` is a per-process magic link that resolves
    through the descriptor to the pinned inode, so a repository root renamed
    and replaced by a symlink cannot redirect the command -- there is no
    pathname left to swap. ``subprocess`` applies this working directory with
    an async-signal-safe ``chdir`` in the child, without any ``preexec_fn``,
    so there is no fork-before-exec hazard in a threaded process.

    The link is trusted only after confirming it stats to the very inode
    ``root_fd`` refers to, so a descriptor that collides with a standard
    stream number (0/1/2) or a stale ``/proc`` entry can never yield a wrong
    anchor. Where ``/proc`` is unavailable (macOS/BSD, where ``/dev/fd/<n>``
    cannot be a directory ``cwd``) there is no fork-safe fd-anchored ``cwd``
    and this returns None."""
    proc_path = f"/proc/self/fd/{root_fd}"
    try:
        if os.path.isdir(proc_path) and os.path.samestat(
            os.stat(proc_path), os.fstat(root_fd)
        ):
            return proc_path
    except OSError:
        pass
    return None


def _fd_cwd(cwd: str, root_fd: "int | None") -> str:
    """The working directory a git subprocess should use: the fd-anchored
    path when one is available (see :func:`fd_anchored_cwd`), otherwise the
    plain pathname. ``root_fd`` of None means no anchoring was requested."""
    if root_fd is None:
        return cwd
    return fd_anchored_cwd(root_fd) or cwd


def _git_env() -> dict[str, str]:
    env = dict(os.environ)
    for name in tuple(env):
        if name in _GIT_REPOSITORY_ENV or name.startswith(_GIT_CONFIG_ENV_PREFIXES):
            env.pop(name)
    env.update(
        {
            "GIT_LFS_SKIP_SMUDGE": "1",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return env


def _git_argv(args: list[str]) -> list[str]:
    command = ["git", "--no-pager"]
    for value in _GIT_SAFE_CONFIG:
        command.extend(("-c", value))
    if args and args[0] == "diff":
        command.extend(
            (
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--ignore-submodules=dirty",
                *args[1:],
            )
        )
    elif args and args[0] == "status":
        command.extend(("status", "--ignore-submodules=dirty", *args[1:]))
    else:
        command.extend(args)
    return command


def _configured_filters(cwd: str, *, timeout: int, root_fd: "int | None") -> set[str]:
    args = [
        "config",
        "--includes",
        "--null",
        "--name-only",
        "--get-regexp",
        r"^filter\.",
    ]
    result = subprocess.run(
        _git_argv(args),
        capture_output=True,
        cwd=_fd_cwd(cwd, root_fd),
        env=_git_env(),
        timeout=timeout,
    )
    if result.returncode == 1:
        return set()
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {stderr}")

    drivers = set()
    for raw_name in result.stdout.split(b"\0"):
        name = raw_name.decode("utf-8", "surrogateescape")
        parsed = name.removeprefix("filter.").rsplit(".", 1)
        if (
            len(parsed) == 2
            and parsed[0]
            and parsed[1].lower()
            in {
                "clean",
                "process",
                "required",
                "smudge",
            }
        ):
            drivers.add(parsed[0])
    return drivers


def git_command(
    args: list[str],
    cwd: str,
    *,
    timeout: int = _GIT_TIMEOUT,
    root_fd: "int | None" = None,
    working_tree: bool = True,
) -> tuple[list[str], str, dict[str, str]]:
    """Build a hardened internal Git subprocess."""
    env = _git_env()
    if working_tree:
        overrides = []
        for driver in sorted(
            _configured_filters(cwd, timeout=timeout, root_fd=root_fd)
        ):
            overrides.extend(
                (
                    (f"filter.{driver}.process", ""),
                    (f"filter.{driver}.clean", ""),
                    (f"filter.{driver}.smudge", ""),
                    (f"filter.{driver}.required", "false"),
                )
            )
        if overrides:
            env["GIT_CONFIG_COUNT"] = str(len(overrides))
            for index, (key, value) in enumerate(overrides):
                env[f"GIT_CONFIG_KEY_{index}"] = key
                env[f"GIT_CONFIG_VALUE_{index}"] = value
    return _git_argv(args), _fd_cwd(cwd, root_fd), env


def git(
    args: list[str],
    cwd: str,
    *,
    timeout: int = _GIT_TIMEOUT,
    root_fd: "int | None" = None,
    working_tree: bool = True,
) -> str:
    """Run a git command and return stripped stdout, raising on failure."""
    return git_raw(
        args,
        cwd,
        timeout=timeout,
        root_fd=root_fd,
        working_tree=working_tree,
    ).strip()


def git_bytes(
    args: list[str],
    cwd: str,
    *,
    timeout: int = _GIT_TIMEOUT,
    root_fd: "int | None" = None,
    working_tree: bool = True,
) -> bytes:
    """Like :func:`git` but returns stdout byte-for-byte, unstripped.

    Patch bytes must keep their trailing newline and any non-UTF-8
    content, otherwise ``git apply`` rejects them and hashes stop being
    reproducible.
    """
    command, git_cwd, env = git_command(
        args,
        cwd,
        timeout=timeout,
        root_fd=root_fd,
        working_tree=working_tree,
    )
    result = subprocess.run(
        command,
        capture_output=True,
        cwd=git_cwd,
        env=env,
        timeout=timeout,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {stderr}")
    return result.stdout


def git_raw(
    args: list[str],
    cwd: str,
    *,
    timeout: int = _GIT_TIMEOUT,
    root_fd: "int | None" = None,
    working_tree: bool = True,
) -> str:
    """Unstripped git stdout decoded with surrogateescape.

    Path listings decoded this way stay comparable with the paths Python
    itself produces via ``os.fsdecode``, so non-UTF-8 filenames survive
    the round trip instead of failing or being silently replaced.
    """
    return git_bytes(
        args,
        cwd,
        timeout=timeout,
        root_fd=root_fd,
        working_tree=working_tree,
    ).decode("utf-8", "surrogateescape")


class Worktree:
    """Context manager for a temporary detached git worktree.

    The worktree is created at an explicit commit so every consumer works
    against pinned content instead of whatever HEAD drifts to mid-run.
    Cleanup is best-effort but layered: prune, forced removal, then a
    plain rmtree for anything git leaves behind.
    """

    def __init__(
        self,
        base_dir: str,
        work_dir: Path,
        commit: str = "HEAD",
        *,
        reclaim_root: Path | None = None,
    ):
        self.base_dir = base_dir
        self.work_dir = work_dir
        self.commit = commit
        # Reclaiming a leftover directory that git no longer recognizes as
        # a worktree is destructive, so it is limited to paths inside the
        # owning command's private state directory, given here as a root.
        self.reclaim_root = reclaim_root

    def _cleanup(self) -> None:
        try:
            git(["worktree", "prune"], self.base_dir)
        except RuntimeError:
            pass
        if self.work_dir.exists():
            try:
                git(
                    ["worktree", "remove", "--force", str(self.work_dir)],
                    self.base_dir,
                )
            except RuntimeError:
                pass
            if self.work_dir.exists():
                shutil.rmtree(self.work_dir, ignore_errors=True)

    _OWNER_MARKER = "swival-owner"

    def _is_registered_worktree(self) -> bool:
        try:
            out = git(["worktree", "list", "--porcelain"], self.base_dir)
            target = self.work_dir.resolve()
        except (RuntimeError, OSError):
            return False
        for line in out.splitlines():
            if line.startswith("worktree "):
                try:
                    if Path(line[len("worktree ") :]).resolve() == target:
                        return True
                except OSError:
                    continue
        return False

    def _admin_dir(self) -> Path | None:
        """The worktree's private ``.git/worktrees/<id>`` admin directory."""
        try:
            out = git(["rev-parse", "--git-dir"], str(self.work_dir))
        except RuntimeError:
            return None
        p = Path(out)
        if not p.is_absolute():
            p = self.work_dir / p
        return p

    def _mark_owned(self) -> None:
        admin = self._admin_dir()
        if admin is not None and admin.is_dir():
            try:
                (admin / self._OWNER_MARKER).write_text(
                    "created by swival; safe to reclaim after a crash\n",
                    encoding="utf-8",
                )
            except OSError:
                pass

    def _is_owned(self) -> bool:
        admin = self._admin_dir()
        return admin is not None and (admin / self._OWNER_MARKER).exists()

    def _reclaimable_unregistered(self) -> bool:
        """Unregistered debris is deletable only when it sits under the
        caller's reclaim root with no symlinked component in between --
        otherwise a symlinked state directory could redirect the cleanup
        at someone else's data."""
        root = self.reclaim_root
        if root is None:
            return False
        try:
            if root.is_symlink():
                return False
            rel = Path(os.path.normpath(self.work_dir)).relative_to(
                os.path.normpath(root)
            )
        except (OSError, ValueError):
            return False
        cur = root
        for part in rel.parts:
            cur = cur / part
            try:
                if cur.is_symlink():
                    return False
            except OSError:
                return False
        return True

    def __enter__(self) -> Path:
        self.work_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            git(["worktree", "prune"], self.base_dir)
        except RuntimeError:
            pass
        if self.work_dir.exists():
            # Registration alone does not make a leftover safe to delete:
            # a registered worktree at this path may be someone's live
            # checkout. Reclaim only worktrees carrying the ownership
            # marker a previous swival run wrote, or unregistered debris
            # sitting symlink-free under the caller's own state directory.
            # Anything else is refused, never deleted.
            if self._is_registered_worktree():
                if not self._is_owned():
                    raise RuntimeError(
                        f"refusing to reuse {self.work_dir}: it is an "
                        "active worktree of this repository that swival "
                        "did not create"
                    )
                self._cleanup()
            elif self._reclaimable_unregistered():
                self._cleanup()
            elif any(self.work_dir.iterdir()):
                raise RuntimeError(
                    f"refusing to reuse {self.work_dir}: it exists and is "
                    "not a worktree of this repository"
                )
        git(
            [
                "worktree",
                "add",
                "--no-checkout",
                "--detach",
                str(self.work_dir),
                self.commit,
            ],
            self.base_dir,
            working_tree=False,
        )
        try:
            git(["reset", "--hard", self.commit], str(self.work_dir))
        except BaseException:
            self._cleanup()
            raise
        self._mark_owned()
        return self.work_dir

    def __exit__(self, *exc):
        self._cleanup()
        return False


def match_path_glob(file: str, glob: str) -> bool:
    """True when ``file`` (a repo-relative path) is selected by ``glob``.

    Selection rules, in order:

    - exact match: ``src/foo.rs`` selects only ``src/foo.rs``.
    - prefix match (no wildcard): ``src`` and ``src/`` both select
      anything under ``src/``.
    - pathlib wildcard match via ``PurePosixPath.full_match``: ``*``
      does *not* cross ``/``, ``?`` matches a single non-separator
      character, ``**`` matches any number of intermediate directories,
      and ``[abc]`` is a character class. A wildcard pattern with no
      ``/`` is implicitly recursive: ``*.rs`` is rewritten to
      ``**/*.rs`` so it keeps selecting every ``.rs`` file at any depth.
      This means ``src/*.rs`` matches only direct ``.rs`` children of a
      top-level ``src/``, and ``src/**/*.rs`` is the recursive form.
    """
    if file == glob:
        return True

    has_wildcard = any(c in glob for c in "*?[")
    if not has_wildcard:
        prefix = glob if glob.endswith("/") else glob + "/"
        return file.startswith(prefix)

    pattern = glob.rstrip("/") or glob
    if "/" not in pattern:
        pattern = f"**/{pattern}"
    return PurePosixPath(file).full_match(pattern)


@dataclass(frozen=True)
class ToolPolicy:
    """Host-enforced tool allowlist for an isolated agent loop."""

    allowed_tools: frozenset[str]

    def check(self, name: str, args: dict) -> str | None:
        if name not in self.allowed_tools:
            return (
                f"error: tool {name!r} is not available in this isolated run; "
                f"available tools: {', '.join(sorted(self.allowed_tools))}"
            )
        return None


READ_ONLY_AGENT_TOOLS = frozenset(
    {
        "read_file",
        "read_multiple_files",
        "list_files",
        "grep",
        "outline",
        "think",
        "todo",
    }
)


def filter_tool_schemas(tools: list, allowed: frozenset[str]) -> list:
    """Keep only the tool schemas whose function name is in ``allowed``."""
    kept = []
    for t in tools or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict) and fn.get("name") in allowed:
            kept.append(t)
    return kept


def make_isolated_loop_kwargs(
    ctx: "InputContext",
    work_dir: Path,
    max_turns: int | None = None,
    *,
    tool_policy: ToolPolicy | None = None,
    files_mode: "str | None" = "some",
    network_mode: "str | None" = "none",
    cancel_flag=None,
    event_callback=None,
) -> dict:
    """Build loop kwargs for an isolated agent loop in a worktree.

    The isolated loop gets fresh scaffolding state, no external managers,
    and no report/event plumbing, so nothing from the parent session leaks
    in and nothing the worker does leaks out. By default the loop is also
    confined to the worktree: reads are rooted at ``work_dir`` and the
    network is off, so a parent session running ``--files all`` cannot
    leak the real checkout into an isolated worker. Pass ``None`` for
    ``files_mode``/``network_mode`` to inherit the parent session's modes.

    ``cancel_flag`` and ``event_callback`` are the only pieces of parent
    plumbing an isolated loop may keep; they must be passed explicitly
    rather than inherited so a caller cannot leak them in by accident.
    """
    from .thinking import ThinkingState
    from .todo import TodoState
    from .tracker import FileAccessTracker

    kw = dict(ctx.loop_kwargs)
    kw["base_dir"] = str(work_dir)
    kw["max_turns"] = max_turns if max_turns is not None else kw.get("max_turns", 100)
    kw["thinking_state"] = ThinkingState(verbose=False)
    kw["todo_state"] = TodoState(verbose=False)
    kw["snapshot_state"] = None
    kw["goal_state"] = None
    kw["file_tracker"] = FileAccessTracker()
    kw["extra_write_roots"] = []
    kw["skill_read_roots"] = []
    kw["skills_catalog"] = {}
    kw["verbose"] = False
    if tool_policy is not None:
        kw["tool_policy"] = tool_policy
    if files_mode is not None:
        kw["files_mode"] = files_mode
    if network_mode is not None:
        kw["network_mode"] = network_mode
    for k in (
        "compaction_state",
        "mcp_manager",
        "a2a_manager",
        "subagent_manager",
        "report",
        "event_callback",
        "cancel_flag",
        "turn_state",
    ):
        kw.pop(k, None)
    if cancel_flag is not None:
        kw["cancel_flag"] = cancel_flag
    if event_callback is not None:
        kw["event_callback"] = event_callback
    return kw


def call_llm_text(
    loop_kwargs: dict,
    messages: list[dict],
    *,
    temperature: float | None = None,
) -> "tuple[str, tuple | None]":
    """One plain no-tool LLM call marshaled from a session's loop kwargs.

    Returns ``(response_text, cache_stats)``. Every pipeline that needs a
    bare reasoning call shares this marshaling, so ``call_llm``'s long
    positional signature has exactly one non-loop call site to keep in
    sync.
    """
    from ._msg import _msg_content
    from .agent import call_llm

    llm_kwargs = loop_kwargs.get("llm_kwargs", {})
    msg, _finish, _activity, _retries, cache_stats = call_llm(
        loop_kwargs["api_base"],
        loop_kwargs["model_id"],
        messages,
        loop_kwargs.get("max_output_tokens"),
        temperature,
        loop_kwargs.get("top_p"),
        loop_kwargs.get("seed"),
        None,  # tools
        False,  # verbose
        provider=llm_kwargs.get("provider", "lmstudio"),
        api_key=llm_kwargs.get("api_key"),
        user_agent=llm_kwargs.get("user_agent"),
        prompt_cache=True,
        aws_profile=llm_kwargs.get("aws_profile"),
        vertex_project=llm_kwargs.get("vertex_project"),
        vertex_location=llm_kwargs.get("vertex_location"),
        pricing_provider=llm_kwargs.get("pricing_provider"),
        session_cost=loop_kwargs.get("session_cost"),
    )
    return _msg_content(msg) or "", cache_stats


@dataclass(frozen=True)
class BatchItemResult:
    """Typed outcome for one item of a bounded batch."""

    ok: bool
    value: object = None
    error: str = ""


def run_bounded_batch(
    fn: Callable,
    items: list,
    *,
    max_workers: int = 4,
    cancel_flag: "threading.Event | None" = None,
) -> list[BatchItemResult]:
    """Run ``fn(item)`` in parallel and return typed results in item order.

    A raised exception becomes a failed :class:`BatchItemResult` instead of
    propagating, and cancellation marks not-yet-started items as failed
    rather than running them.
    """

    def _one(item) -> BatchItemResult:
        if cancel_flag is not None and cancel_flag.is_set():
            return BatchItemResult(ok=False, error="cancelled")
        try:
            return BatchItemResult(ok=True, value=fn(item))
        except Exception as e:
            return BatchItemResult(ok=False, error=str(e))

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = [pool.submit(_one, item) for item in items]
        return [future.result() for future in futures]

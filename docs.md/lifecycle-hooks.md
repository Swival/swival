# Lifecycle Hooks

Swival can run a user-configured command at two points in every session: once at startup (before memory and continue state are loaded) and once at exit (after history, reports, and continue files are written). This lets you sync `.swival/` state to and from remote storage without any provider-specific code in Swival itself.

The feature is generic. Swival does not know or care what the hook does. The canonical use case is syncing `.swival/` to Hugging Face Buckets so that memory, continue files, and reports survive across machines and commits — but any storage backend works.

## Enabling Hooks

On the command line:

```sh
swival --lifecycle-command "./scripts/sync.sh" "task"
```

In a config file:

```toml
lifecycle_command = "./scripts/sync.sh"
lifecycle_timeout = 300
lifecycle_fail_closed = false
```

In the library API:

```python
from swival import Session

with Session(
    lifecycle_command="./scripts/sync.sh",
    lifecycle_timeout=300,
    lifecycle_fail_closed=False,
    lifecycle_enabled=True,
) as s:
    result = s.run("task")
```

The value is a shell command string. It is split with `shlex.split`, and path-like first tokens (`./`, `../`, `~`) resolve against the config file's parent directory, consistent with `reviewer`, `llm_filter`, and other command-valued config keys.

## Command Invocation

Swival calls the configured command as:

```text
<command> startup <base_dir>
<command> exit <base_dir>
```

The first positional argument is the event name (`startup` or `exit`). The second is the absolute path to the project base directory. The working directory is set to `base_dir`.

The command is validated at startup using the same logic as `reviewer` and `llm_filter`: `shlex.split` parses the string, and the first token must be an executable on PATH or a path to an executable file. No `shell=True` — the command runs as a direct subprocess.

## Execution Ordering

### Startup

The startup hook runs after:

- Config file loading and CLI flag merging
- Provider and model resolution
- Git metadata discovery
- `.swival/` directory creation

It runs before:

- Memory loading into the system prompt
- Continue-file resume checks
- REPL startup that depends on `.swival/` content

This ordering is the whole point: a startup hook can download files into `.swival/memory/` or `.swival/continue.md` and Swival will pick them up as if they were already there.

### Exit

The exit hook runs after:

- The final answer is produced (or the run outcome is known)
- `.swival/HISTORY.md` is appended
- The report file is written (if `--report` is set)
- The continue file is written (on interruption paths)

In the CLI, the exit hook runs in the outer `finally` block, so it fires on success, exhaustion, error, and interruption.

### REPL

Hooks run once on process start and once on process exit. They do not run per prompt, on `/clear`, or on `/continue`.

### Reviewer and Serve Modes

Hooks are not run in `--reviewer-mode` or `--serve`. Reviewer mode is a subprocess of the main Swival invocation — running hooks there would cause nested syncs. Serve mode manages per-request sessions with different lifecycle semantics that will be addressed separately.

## Environment Variables

The hook receives all of the parent environment plus these `SWIVAL_*` variables:

| Variable              | Description                                                    | Events    |
| --------------------- | -------------------------------------------------------------- | --------- |
| `SWIVAL_HOOK_EVENT`   | `startup` or `exit`                                            | both      |
| `SWIVAL_BASE_DIR`     | Absolute path to the project base directory                    | both      |
| `SWIVAL_SWIVAL_DIR`   | Path to the `.swival/` directory                               | both      |
| `SWIVAL_PROVIDER`     | Provider name (e.g. `lmstudio`, `openrouter`)                  | both      |
| `SWIVAL_MODEL`        | Resolved model ID                                              | both      |
| `SWIVAL_GIT_PRESENT`  | `1` if inside a Git repo, `0` otherwise                        | both      |
| `SWIVAL_REPO_ROOT`    | Git repo root (absolute path)                                  | both      |
| `SWIVAL_PROJECT_REL`  | Relative path from repo root to base_dir (empty at repo root)  | both      |
| `SWIVAL_GIT_HEAD`     | HEAD commit SHA                                                | both      |
| `SWIVAL_GIT_DIRTY`    | `1` if working tree has staged, unstaged, or untracked changes | both      |
| `SWIVAL_GIT_REMOTE`   | `remote.origin.url` value                                      | both      |
| `SWIVAL_REPO_HASH`    | 48-character hash of normalized repo identity                  | both      |
| `SWIVAL_PROJECT_HASH` | 48-character hash of repo identity + project_rel               | both      |
| `SWIVAL_REPORT`       | Path to the report file                                        | exit only |
| `SWIVAL_OUTCOME`      | Run outcome: `success`, `exhausted`, `error`, `interrupted`    | exit only |
| `SWIVAL_EXIT_CODE`    | Process exit code as a string                                  | exit only |

Git-specific variables are omitted when `SWIVAL_GIT_PRESENT=0`.

`SWIVAL_REPO_HASH` and `SWIVAL_PROJECT_HASH` are derived from the normalized remote URL (SSH and HTTPS forms collapse to the same value). When there is no remote, the local repo root path is used instead. Two repos cloned from the same remote get the same repo hash regardless of where they live on disk.

`SWIVAL_PROJECT_HASH` further incorporates `SWIVAL_PROJECT_REL`, so different subprojects in a monorepo get distinct hashes.

## Failure Semantics

By default, hooks fail open. A nonzero exit code, a timeout, or a spawn failure is silently ignored and the run continues normally. In verbose mode (`--quiet` not set), a warning is printed to stderr. This means a broken sync script will not prevent you from using Swival.

### Fail-Closed Mode

Pass `--lifecycle-fail-closed` (or `lifecycle_fail_closed = true` in config) to make hook failures fatal:

- A startup failure raises an error before the agent loop starts.
- An exit failure forces a nonzero process exit after cleanup completes (CLI) or raises `LifecycleError` (library API).

In the library API, `LifecycleError` propagates from `Session.run()`, `Session.close()`, and `Session.__exit__()` (when no other exception is active).

### Kill Switch

Pass `--no-lifecycle` to disable hooks entirely. This is useful for nested invocations, CI jobs, or debugging.

## Library API

`Session` supports the same lifecycle semantics as the CLI:

```python
from swival import Session

# Single-shot: exit hook runs automatically after run()
result = Session(lifecycle_command="./sync.sh").run("task")

# Multi-turn: exit hook runs on close() or __exit__
with Session(lifecycle_command="./sync.sh") as s:
    s.ask("first question")
    s.ask("follow-up")
# exit hook fires here

# Explicit close without context manager
s = Session(lifecycle_command="./sync.sh")
s.ask("question")
s.close(outcome="success", exit_code=0)
```

For single-shot `run()`, the exit hook fires after local artifacts are written, even if the agent loop raises an exception. The `outcome` and `exit_code` passed to the hook reflect what actually happened.

For multi-turn `ask()`, the exit hook fires when the session is closed — either via `close()` or `__exit__`. It does not fire after each `ask()` call. This matches REPL semantics: one startup, one exit.

## Hugging Face Buckets

HF Buckets provide S3-like object storage on the Hugging Face Hub. Combined with lifecycle hooks, they give Swival commit-scoped remote state with no HF-specific code in Swival itself.

### Bucket Layout

The recommended layout uses one stable bucket per repo identity and commit-specific prefixes inside it:

```text
hf://buckets/<namespace>/swival-<repo_hash>/
    <project_rel>/heads/<head_sha>/
        memory/
            MEMORY.md
            ...
        continue.md
        reports/
            ...
```

This keeps bucket count bounded (one per repo, not one per commit), isolates monorepo subprojects via `project_rel`, and still gives commit-scoped remote state through the `heads/<sha>` prefix.

### Example Sync Script

```sh
#!/bin/sh
# scripts/swival-hf-sync.sh
set -e

EVENT="$1"
BASE_DIR="$2"
SWIVAL_DIR="$BASE_DIR/.swival"

# Skip if not in a Git repo
[ "$SWIVAL_GIT_PRESENT" = "0" ] && exit 0

# Resolve HF namespace (cache in env to avoid repeated calls)
HF_NS="${HF_NS:-$(hf auth whoami --format json 2>/dev/null | grep '"name"' | head -1 | sed 's/.*"name": *"//;s/".*//')}"
[ -z "$HF_NS" ] && exit 0

BUCKET="swival-$SWIVAL_REPO_HASH"

# Build prefix: project_rel/heads/<sha> or just heads/<sha> at repo root
if [ -n "$SWIVAL_PROJECT_REL" ]; then
    PREFIX="$SWIVAL_PROJECT_REL/heads/$SWIVAL_GIT_HEAD"
else
    PREFIX="heads/$SWIVAL_GIT_HEAD"
fi

REMOTE="hf://buckets/$HF_NS/$BUCKET/$PREFIX"

if [ "$EVENT" = "startup" ]; then
    mkdir -p "$SWIVAL_DIR"
    hf buckets sync "$REMOTE" "$SWIVAL_DIR" \
        --include "memory/**" \
        --include "continue.md" \
        --include "reports/**" \
        2>/dev/null || true
elif [ "$EVENT" = "exit" ]; then
    hf buckets sync "$SWIVAL_DIR/" "$REMOTE" \
        --include "memory/**" \
        --include "continue.md" \
        --include "reports/**" \
        --exclude "cache.db" \
        --exclude "trash/**" \
        --exclude "cmd_output_*" \
        2>/dev/null || true
fi
```

Make the script executable and configure it:

```toml
lifecycle_command = "./scripts/swival-hf-sync.sh"
```

### What Gets Synced

The example script syncs:

- `memory/**` — auto-memory entries that accumulate across sessions
- `continue.md` — interrupted session resume state
- `reports/**` — evaluation reports

It excludes:

- `cache.db` — LLM response cache (machine-local, large)
- `trash/**` — soft-deleted files (local safety net only)
- `cmd_output_*` — temporary large command outputs (auto-deleted after 600 seconds)

### Why Repo Hash Instead of Repo Name

Git remotes come in multiple forms:

- `git@github.com:org/repo.git`
- `https://github.com/org/repo.git`
- `ssh://git@github.com/org/repo.git`

Swival normalizes all of these to `github.com/org/repo` before hashing, so the same repo gets the same bucket regardless of how it was cloned. The 48-character (192-bit) SHA-256 prefix is collision-resistant.

### Commit Scoping

Each `heads/<sha>` prefix represents the state of `.swival/` at a particular commit. When you switch branches or rebase, the HEAD changes and Swival naturally starts syncing to a different prefix. Previous prefixes remain in the bucket as historical snapshots.

This is intentional: if you switch back to a commit you worked on before, the hook pulls the memory and continue state from that exact point.

### Purging State

Because the layout uses one stable bucket per repo, cleanup is straightforward.

Delete all Swival state for a repo (replace `<namespace>` with your HF username):

```sh
hf buckets delete "<namespace>/swival-$(echo -n 'github.com/org/repo' | shasum -a 256 | cut -c1-48)"
```

To derive the full bucket path for a local repo:

```sh
cd /path/to/repo
HF_NS=$(hf auth whoami --format json 2>/dev/null | grep '"name"' | head -1 | sed 's/.*"name": *"//;s/".*//')
REMOTE=$(git config --get remote.origin.url)
# Normalize: strip .git suffix, convert SSH to path form
NORM=$(echo "$REMOTE" | sed 's/\.git$//' | sed 's|^git@|ssh://git@|' | sed 's|ssh://[^@]*@||' | sed 's|https\?://||' | sed 's|:|/|')
HASH=$(echo -n "$NORM" | shasum -a 256 | cut -c1-48)
echo "$HF_NS/swival-$HASH"
```

For a single subproject in a monorepo, remove only its prefix path from the bucket rather than the entire bucket.

### Dirty Worktrees

`SWIVAL_GIT_DIRTY` reports `1` when the working tree has staged changes, unstaged modifications, or untracked files. A sync script could use this to branch its behavior — for example, syncing to a `dirty/` prefix instead of `heads/<sha>/` when the tree is dirty, so that uncommitted work does not pollute the clean commit prefix.

The example script above does not do this. It always syncs to `heads/<sha>`, which means dirty-tree sessions overwrite the same prefix as clean-tree sessions for that commit. For most workflows this is fine. Add the dirty prefix if you need strict isolation.

## Interaction with Other Features

**Reviewer and self-review:** Hooks do not run in `--reviewer-mode`. Self-review spawns a reviewer subprocess, but that subprocess gets `--reviewer-mode`, so it also skips hooks. No nested sync.

**A2A serve:** Hooks are excluded from `--serve` in this release. Per-session hooks for long-lived A2A servers need different lifecycle semantics and will be addressed separately.

**AgentFS sandbox:** Hooks run in the same process context as Swival. If you are running inside an AgentFS sandbox, the hook script needs network access to reach the remote storage backend.

**Reports:** Lifecycle events (startup and exit) are recorded in the report timeline when `--report` is set. Each event includes its exit code, duration, and any error message. A fail-closed exit hook failure rewrites the report with `outcome: "error"`.

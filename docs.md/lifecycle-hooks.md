# Lifecycle Hooks

Swival can run a user-configured command at two points in every session: once at startup (before memory and continue state are loaded) and once at exit (after history, reports, and continue files are written). This lets you sync `.swival/` state to and from remote storage without any provider-specific code in Swival itself — and without committing `.swival/` to git. Memory, continue files, and reports stay out of version control but still follow you across machines.

In the interactive REPL, Swival also runs the command each time a task finishes. You can use this to get a notification when a long `/goal` is done. See [Task Notifications](#task-notifications).

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

The value is a shell command string. It is split with `shlex.split`, and path-like first tokens — anything starting with `/`, `~`, or containing a `/` (e.g. `./`, `../`, `.rtk/`, `scripts/`) — resolve against the config file's parent directory, consistent with `reviewer`, `llm_filter`, and other command-valued config keys.

## Command Invocation

Swival calls the configured command as:

```text
<command> startup <base_dir>
<command> exit <base_dir>
<command> task_done <base_dir>
```

The first positional argument is the event name (`startup`, `exit` or `task_done`). The second is the absolute path to the project base directory. The working directory is set to `base_dir`. Only the REPL sends `task_done`.

The command is validated at startup using the same logic as `reviewer` and `llm_filter`: `shlex.split` parses the string, and the first token must be an executable on PATH or a path to an executable file. No `shell=True` — the command runs as a direct subprocess.

## Execution Ordering

### Startup

The startup hook runs after:

- Config file loading and CLI flag merging
- Provider and model resolution
- Git metadata discovery

The `.swival/` directory is not guaranteed to exist yet; a hook that writes into it should `mkdir -p` first (the example script below does).

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

The `startup` and `exit` events run once each, on process start and on process exit. The `task_done` event runs each time a task finishes. See [Task Notifications](#task-notifications).

### Reviewer and Serve Modes

Hooks are not run in `--reviewer-mode`. Reviewer mode is a subprocess of the main Swival invocation; running hooks there would cause nested syncs.

In `--serve`, the server process itself runs no hooks, but each per-context session it creates inherits the lifecycle settings: the startup hook runs when a context's session is first used, and the exit hook runs when that session is evicted or closed. Pass `--no-lifecycle` if you do not want per-session hooks in a long-lived server.

## Environment Variables

The hook receives all of the parent environment plus these `SWIVAL_*` variables:

| Variable              | Description                                                    | Events          |
| --------------------- | -------------------------------------------------------------- | --------------- |
| `SWIVAL_HOOK_EVENT`   | `startup`, `exit` or `task_done`                               | all             |
| `SWIVAL_BASE_DIR`     | Absolute path to the project base directory                    | all             |
| `SWIVAL_SWIVAL_DIR`   | Path to the `.swival/` directory                               | all             |
| `SWIVAL_PROVIDER`     | Provider name (e.g. `lmstudio`, `openrouter`)                  | all             |
| `SWIVAL_MODEL`        | Resolved model ID                                              | all             |
| `SWIVAL_GIT_PRESENT`  | `1` if inside a Git repo, `0` otherwise                        | all             |
| `SWIVAL_REPO_ROOT`    | Git repo root (absolute path)                                  | all             |
| `SWIVAL_PROJECT_REL`  | Relative path from repo root to base_dir (empty at repo root)  | all             |
| `SWIVAL_GIT_HEAD`     | HEAD commit SHA                                                | all             |
| `SWIVAL_GIT_DIRTY`    | `1` if working tree has staged, unstaged, or untracked changes | all             |
| `SWIVAL_GIT_REMOTE`   | `remote.origin.url` value                                      | all             |
| `SWIVAL_REPO_HASH`    | 48-character hash of normalized repo identity                  | all             |
| `SWIVAL_PROJECT_HASH` | 48-character hash of repo identity + project_rel               | all             |
| `SWIVAL_REPORT`       | Path to the report file                                        | exit only       |
| `SWIVAL_OUTCOME`      | Outcome: `success`, `exhausted`, `interrupted`, `error`        | exit, task_done |
| `SWIVAL_EXIT_CODE`    | Process exit code as a string                                  | exit only       |
| `SWIVAL_COMMAND`      | Command of the task, such as `/goal`. Empty for a prompt       | task_done only  |
| `SWIVAL_GOAL_STATUS`  | New goal status, only when the task changed the goal           | task_done only  |

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

The `task_done` event always fails open, even in fail-closed mode. A broken notification script must not end your session.

### Kill Switch

Pass `--no-lifecycle` to disable hooks entirely. This is useful for nested invocations, CI jobs, or debugging.

## Task Notifications

In the REPL, Swival runs the hook with the `task_done` event each time a task finishes. A task is a plain prompt, a custom `!command`, or one of these commands: `/goal`, `/continue`, `/simplify`, `/learn`, `/audit`, `/init` and `/btw`. Each `/loop` iteration is a task too. All other commands, such as `/help` or `/goal pause`, are not tasks.

The hook runs after Swival prints the answer. `SWIVAL_COMMAND` holds the command that started the task, such as `/goal` or `!deploy`. It is empty for a plain prompt, and it is `/loop` for a loop iteration. `SWIVAL_OUTCOME` tells you how the task ended: `success`, `exhausted`, `interrupted` or `error`.

This script speaks when any task ends:

```sh
#!/bin/sh
[ "$1" = "task_done" ] || exit 0
spd-say "Swival task: $SWIVAL_OUTCOME" >/dev/null 2>&1 &
```

`spd-say` is the Linux command. On macOS, use `say` in its place: `say "Swival task: $SWIVAL_OUTCOME"`.

Check the event name first, as this script does. If you already have a sync script, make sure it ignores `task_done`, or it will sync after every task.

### Goal Notifications

`SWIVAL_OUTCOME` describes the task, not the goal. A `/goal` task can end with `success` while the goal still needs more work. A goal can also complete later, through `/continue` or a plain prompt.

This is what `SWIVAL_GOAL_STATUS` is for. Swival sets it only when the task changed the goal. A change is a new goal, or a new status for the current goal. The value is `active`, `paused`, `budget_limited` or `complete`. In all other cases the variable is not set. A completed goal therefore reports `complete` once, not after every later prompt.

This script speaks only when a goal completes:

```sh
#!/bin/sh
[ "$1" = "task_done" ] || exit 0
[ "$SWIVAL_GOAL_STATUS" = "complete" ] || exit 0
spd-say "Swival goal complete" >/dev/null 2>&1 &
```

It does not look at `SWIVAL_COMMAND`, because `/continue` or a plain prompt can also complete the goal.

### Keep The Hook Fast

Swival waits for the hook before it starts the next task. It also waits until the hook closes its output. A slow program must go to the background with its output redirected, as in the last line of both scripts. You can still type while the hook runs, and Swival queues your lines. For this reason a `task_done` hook gets an empty standard input: the terminal belongs to the prompt.

`lifecycle_timeout` applies to this event too. Ctrl-C stops the hook script, but not the programs that the script started.

### Where It Does Not Run

One-shot runs do not send `task_done`. They do not need it, because the `exit` hook already runs when the task ends, with `SWIVAL_OUTCOME` set. The library API and `--serve` do not send it either.

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

For multi-turn `ask()`, the exit hook fires when the session is closed — either via `close()` or `__exit__`. It does not fire after each `ask()` call. This matches REPL semantics: one startup, one exit. `Session` never sends `task_done`.

## Hugging Face Buckets

HF Buckets provide S3-like object storage on the Hugging Face Hub. Combined with lifecycle hooks, they give Swival commit-scoped remote state with no HF-specific code in Swival itself.

Note that HF Buckets are not end-to-end encrypted. Files are encrypted in transit (TLS) and at rest on HF's servers, but Hugging Face itself can access the stored data. If your `.swival/` state contains sensitive information, encrypt it before syncing.

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

# Sync at startup and exit only, not after each REPL task
[ "$EVENT" = "startup" ] || [ "$EVENT" = "exit" ] || exit 0

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

**Reviewer and self-review:** Hooks do not run in `--reviewer-mode`. Self-review doesn't run hooks either. No nested sync.

**A2A serve:** The `--serve` process itself does not run hooks, but the per-context sessions it creates inherit the lifecycle settings and run startup and exit hooks per session (exit fires when a session is evicted or closed). Use `--no-lifecycle` to disable hooks in a long-lived server.

**AgentFS sandbox:** Hooks run in the same process context as Swival. If you are running inside an AgentFS sandbox, the hook script needs network access to reach the remote storage backend.

**Reports:** Lifecycle events (startup, exit and task_done) are recorded in the report timeline when `--report` is set. Each event includes its exit code, duration, and any error message. A fail-closed exit hook failure rewrites the report with `outcome: "error"`.

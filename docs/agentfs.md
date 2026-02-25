# Using Swival with AgentFS

[AgentFS](https://www.agentfs.ai/) gives your agent a copy-on-write filesystem.
Everything the agent writes goes into a SQLite-backed overlay while your real
project files stay untouched. You review what the agent did, test it, and only
then apply the changes to your actual working tree.

This is the workflow: **build in a sandbox, validate, commit to the real
filesystem.**

This also provides better sandboxing than Swival.

## Prerequisites

Install AgentFS:

```sh
curl -fsSL https://agentfs.ai/install | bash
```

You also need a running LLM provider (LM Studio, HuggingFace, etc.). See
[Getting Started](getting-started.md) for that.

## Step 1: Let the agent work in a sandbox

`agentfs run` wraps any command in a copy-on-write sandbox. Add `--session` to
give it a name you can come back to:

```sh
cd ~/my-project

agentfs run --session add-config -- \
    uvx swival "Add a config module that reads from env vars, and update main.py to use it" --yolo --max-turns 20
```

The `--yolo` flag is the natural pairing here. AgentFS handles isolation, so you
can give Swival unrestricted file and command access without risking your real
files.

When Swival finishes, your working tree is exactly as it was before. The
agent's changes live in the session delta at
`~/.agentfs/run/add-config/delta.db`.

## Step 2: Review what changed

See which files the agent touched:

```sh
agentfs diff ~/.agentfs/run/add-config/delta.db
```

```
A f /src/config.py
M f /src/main.py
```

`A` = added, `M` = modified.

## Step 3: Test inside the sandbox

Re-enter the session with a shell. You'll see the full project with the agent's
changes applied:

```sh
agentfs run --session add-config -- bash
```

From inside that shell, run your test suite:

```sh
python -m pytest tests/ -v
```

Or start the app and try it manually:

```sh
python src/main.py
```

If something is off, exit the shell and ask the agent for another pass (see
"Iterating" below). Nothing has touched your real project yet.

## Step 4: Apply the changes

Re-enter the session and copy the changed files back to your working tree. The
diff from step 2 tells you exactly which files to grab:

```sh
agentfs run --session add-config -- \
    sh -c 'cp src/config.py ~/my-project/src/config.py && cp src/main.py ~/my-project/src/main.py'
```

Or for many files, use rsync from inside the session:

```sh
agentfs run --session add-config -- \
    rsync -av src/ ~/my-project/src/
```

Then commit normally:

```sh
cd ~/my-project
git add src/config.py src/main.py
git commit -m "Add config module"
```

If you want to discard the agent's work instead, just delete the session data
at `~/.agentfs/run/add-config/`. Nothing else was changed.

## Iterating on the feature

If the tests fail or you don't like what the agent produced, you can keep going
without starting over. Each invocation with the same `--session` sees the
accumulated changes from previous runs:

```sh
agentfs run --session add-config -- \
    uvx swival "The tests are failing because config.py doesn't handle missing env vars. Fix it." --yolo
```

The agent sees its own prior changes and builds on top of them. Re-enter with
a shell, re-run tests, iterate until it works. Then apply as above.

## Alternative: `agentfs init -c`

If you want the overlay stored alongside your project as a `.db` file:

```sh
cd ~/my-project

agentfs init --base . -c \
    'uvx swival "Add a config module" --yolo --max-turns 20' \
    add-config
```

This creates `.agentfs/add-config.db`, runs Swival inside the overlay, then
saves the delta. You can diff it by name:

```sh
agentfs diff add-config
```

And read files from the overlay without mounting:

```sh
agentfs fs add-config cat /src/config.py
```

To test or apply, mount the overlay and work from the mount point:

```sh
mkdir -p /tmp/sandbox
agentfs mount -f --auto-unmount add-config /tmp/sandbox &

cd /tmp/sandbox
python -m pytest tests/ -v          # validate
cp src/config.py ~/my-project/src/  # apply

kill %1                              # unmount
```

To discard, delete `.agentfs/add-config.db`.

## Alternative: manual mount for REPL mode

When you want an interactive back-and-forth with the agent, mount the overlay
yourself:

```sh
cd ~/my-project
agentfs init --base . sandbox
mkdir -p /tmp/sandbox
agentfs mount -f --auto-unmount sandbox /tmp/sandbox
```

In another terminal:

```sh
uvx swival --repl --base-dir /tmp/sandbox --yolo
```

You can chat with the agent, ask it to make changes, then switch to a third
terminal to run tests against `/tmp/sandbox` in real time. When you're
satisfied, copy files back and unmount.

## Tips

- **`--yolo` + AgentFS is the sweet spot.** Swival's built-in sandbox is
  redundant when AgentFS is handling isolation. `--yolo` gives the agent full
  access to `run_command` and the entire filesystem, with AgentFS as the safety
  net.

- **Session names make things resumable.** `agentfs run --session foo` lets you
  come back to the same overlay state across multiple Swival invocations.

- **AgentFS is invisible to Swival.** There's no special integration. Swival
  just sees a normal filesystem. You either let `agentfs run` handle the
  sandboxing, or point `--base-dir` at a mount.

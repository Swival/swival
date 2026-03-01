# Using Swival With AgentFS

[AgentFS](https://www.agentfs.ai/) gives Swival a copy-on-write filesystem overlay. The agent can edit freely, but your real project files remain unchanged until you explicitly copy changes back. This is a practical workflow for high-autonomy runs because you can inspect and test everything before applying it.

## Integrated Sandbox Mode

The simplest way to use AgentFS with Swival is the built-in sandbox mode:

```sh
swival --sandbox agentfs "Refactor the auth module" --yolo
```

This re-executes Swival inside `agentfs run` automatically. Your `--base-dir` and `--add-dir` paths are mapped to writable overlay directories; everything else is read-only to subprocesses. To persist sandbox state across runs, add `--sandbox-session <id>`.

See [Safety and Sandboxing](safety-and-sandboxing.md) for details on what the integrated mode enforces.

The rest of this page covers manual workflows for cases where you want more control over the overlay lifecycle.

## Prerequisites

Install AgentFS first.

```sh
curl -fsSL https://agentfs.ai/install | bash
```

You also need a working model provider for Swival itself, such as LM Studio or HuggingFace.

## Run Swival Inside A Session Overlay

The integrated mode handles this automatically, but you can also invoke `agentfs run` directly when you want full control over session naming and overlay lifecycle.

```sh
cd ~/my-project

agentfs run --session add-config -- \
    swival "Add a config module that reads from env vars, and update main.py to use it" --yolo --max-turns 20
```

After the run, your working copy is unchanged. The overlay delta lives at `~/.agentfs/run/add-config/delta.db`.

## Review The Delta

You can inspect which paths were added or modified with `agentfs diff`.

```sh
agentfs diff ~/.agentfs/run/add-config/delta.db
```

A typical short output looks like this:

```
A f /src/config.py
M f /src/main.py
```

## Validate Inside The Overlay

Re-enter the same session as a shell, then run tests or manual checks against the overlay view.

```sh
agentfs run --session add-config -- bash
```

Inside that shell, run your usual checks.

```sh
python -m pytest tests/ -v
python src/main.py
```

If validation fails, exit and ask Swival for another pass using the same session name.

## Apply Changes To The Real Project

Once you are satisfied, copy only the files you want back into your actual working tree.

```sh
agentfs run --session add-config -- \
    sh -c 'cp src/config.py ~/my-project/src/config.py && cp src/main.py ~/my-project/src/main.py'
```

For larger updates, `rsync` is often easier.

```sh
agentfs run --session add-config -- \
    rsync -av src/ ~/my-project/src/
```

Then commit normally in your project directory.

```sh
cd ~/my-project
git add src/config.py src/main.py
git commit -m "Add config module"
```

If you decide not to keep the work, delete the session directory at `~/.agentfs/run/add-config/` and your real project remains untouched.

## Iterate Without Starting Over

With `--sandbox agentfs`, Swival automatically reuses the same overlay when you run it again from the same project directory. Each run sees prior overlay changes, giving you a natural loop of generate, validate, and refine before applying files.

If you used a manual session above, reuse the same session name:

```sh
agentfs run --session add-config -- \
    swival "The tests are failing because config.py doesn't handle missing env vars. Fix it." --yolo
```

## Alternative Workflow With `agentfs init -c`

If you prefer a project-local overlay database, initialize AgentFS in the repo and run Swival through `-c`.

```sh
cd ~/my-project

agentfs init --base . -c \
    'swival "Add a config module" --yolo --max-turns 20' \
    add-config
```

This writes `.agentfs/add-config.db`. You can diff by session name.

```sh
agentfs diff add-config
```

You can also inspect files in that overlay directly without mounting.

```sh
agentfs fs add-config cat /src/config.py
```

For testing and selective apply, mount the overlay.

```sh
mkdir -p /tmp/sandbox
agentfs mount -f --auto-unmount add-config /tmp/sandbox &

cd /tmp/sandbox
python -m pytest tests/ -v
cp src/config.py ~/my-project/src/

kill %1
```

If you do not want the result, remove `.agentfs/add-config.db`.

## REPL Sessions

The integrated sandbox mode works with REPL mode directly:

```sh
swival --sandbox agentfs --repl --yolo
```

This gives you an interactive agent session inside the overlay. You can test changes from a separate terminal by re-entering the same session with `agentfs run --session <id> -- bash`.

## Practical Guidance

In day-to-day use, `--sandbox agentfs --yolo` is often the most productive combination because it gives the model full capability while still protecting your real workspace. Deterministic session IDs make iteration resumable across runs in the same project directory — or pass `--sandbox-session <id>` for explicit control.

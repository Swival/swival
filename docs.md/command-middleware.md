# Command Middleware

Swival can run a user-defined command before each `run_command` or `run_shell_command` tool call. The middleware receives a JSON description of the pending command and can pass it through unchanged, rewrite it, or block it entirely.

The most common use is pairing Swival with [RTK](https://github.com/rtk-ai/rtk), which rewrites commands like `git status` to `rtk git status` so the output is token-optimized before it reaches the model's context window.

## Enabling Middleware

On the command line:

```sh
swival --command-middleware "./scripts/rtk-adapter.py" "task"
```

In a config file:

```toml
command_middleware = "./scripts/rtk-adapter.py"
```

In the library API:

```python
from swival import Session

session = Session(command_middleware="./scripts/rtk-adapter.py")
result = session.run("task")
```

The value is a shell command string. It is split with `shlex.split`, and path-like first tokens — anything starting with `/`, `~`, or containing a `/` (e.g. `./`, `../`, `.rtk/`, `scripts/`) — resolve against the config file's parent directory, consistent with `llm_filter` and other command-valued config keys.

## Contract

Swival sends a JSON object to the middleware's stdin:

```json
{
  "phase": "before",
  "tool": "run_shell_command",
  "cwd": "/path/to/project",
  "mode": "shell",
  "command": "git status",
  "timeout": 30,
  "is_subagent": false
}
```

For `run_command` (array argv), `mode` is `"argv"` and `command` is a list:

```json
{
  "phase": "before",
  "tool": "run_command",
  "cwd": "/path/to/project",
  "mode": "argv",
  "command": ["git", "log", "--oneline", "-5"],
  "timeout": 30,
  "is_subagent": false
}
```

The middleware writes a JSON object to stdout. Three response shapes are supported.

**Pass through unchanged:**

```json
{"action": "allow"}
```

**Rewrite the command:**

```json
{"action": "allow", "mode": "shell", "command": "rtk git status"}
```

The rewritten command can switch between `"shell"` and `"argv"` modes. For `"argv"`, `"command"` must be a list of strings.

**Block the command:**

```json
{"action": "deny", "reason": "command not permitted by policy"}
```

A blocked command returns an `error:` result to the model with the denial reason.

## Behavior Rules

| Condition                           | Result                                   |
| ----------------------------------- | ---------------------------------------- |
| `{"action": "allow"}`               | Execute original command                 |
| `{"action": "allow", "command": …}` | Execute rewritten command                |
| `{"action": "deny", "reason": …}`   | Return error to model, skip execution    |
| Non-zero exit                       | Warn (verbose), execute original command |
| Malformed JSON on stdout            | Warn (verbose), execute original command |
| Executable not found                | Warn (verbose), execute original command |
| Timeout (10 seconds)                | Warn (verbose), execute original command |

Middleware fails open. If the middleware process errors, the original command runs unchanged. Swival never silently drops a command due to middleware failure. Warnings appear on stderr when diagnostics are enabled (the default unless `--quiet` is set).

## Policy Re-check After Rewrite

After a successful rewrite, the rewritten command is re-evaluated against Swival's built-in command policy before execution. This means middleware cannot bypass `--commands none`, allowlists, or interactive approval (`--commands ask`). The user's safety settings remain authoritative.

## RTK Integration

[RTK](https://github.com/RTaskforce/rtk) is a CLI proxy that rewrites common commands to token-optimized equivalents. For example, `git status` becomes `rtk git status`, which produces a compact, structured summary instead of raw Git output.

RTK's `rewrite` subcommand takes a raw command string and prints the RTK equivalent if one exists, or produces no output if the command has no RTK counterpart.

### Adapter Script

Save this to `scripts/rtk-adapter.py` (or anywhere you prefer):

```python
#!/usr/bin/env python3

import json
import shlex
import subprocess
import sys


def main():
    payload = json.load(sys.stdin)
    if payload.get("phase") != "before":
        json.dump({"action": "allow"}, sys.stdout)
        return

    mode = payload.get("mode")
    command = payload.get("command")

    if mode == "shell":
        raw = command
    elif mode == "argv":
        raw = shlex.join(command)
    else:
        json.dump({"action": "allow"}, sys.stdout)
        return

    proc = subprocess.run(
        ["rtk", "rewrite", raw],
        capture_output=True,
        text=True,
    )

    rewritten = proc.stdout.strip()
    if rewritten:
        json.dump({"action": "allow", "mode": "shell", "command": rewritten}, sys.stdout)
    else:
        json.dump({"action": "allow"}, sys.stdout)


if __name__ == "__main__":
    main()
```

Make it executable:

```sh
chmod +x scripts/rtk-adapter.py
```

### Testing the Adapter

You can test the adapter directly without running Swival:

```sh
# git status → rtk git status
echo '{"phase":"before","tool":"run_shell_command","cwd":".","mode":"shell","command":"git status","timeout":30,"is_subagent":false}' \
  | python3 scripts/rtk-adapter.py
# → {"action": "allow", "mode": "shell", "command": "rtk git status"}

# ls -la → rtk ls -la
echo '{"phase":"before","tool":"run_command","cwd":".","mode":"argv","command":["ls","-la"],"timeout":30,"is_subagent":false}' \
  | python3 scripts/rtk-adapter.py
# → {"action": "allow", "mode": "shell", "command": "rtk ls -la"}

# echo hello → pass through (no RTK equivalent)
echo '{"phase":"before","tool":"run_command","cwd":".","mode":"argv","command":["echo","hello"],"timeout":30,"is_subagent":false}' \
  | python3 scripts/rtk-adapter.py
# → {"action": "allow"}
```

### Configuration

Enable it for a single run:

```sh
swival --command-middleware "./scripts/rtk-adapter.py" "Investigate the failing tests"
```

Or set it permanently in `swival.toml`:

```toml
command_middleware = "./scripts/rtk-adapter.py"
```

When RTK is active, commands the model issues against the codebase are automatically rewritten. For example, if the model calls `git log --oneline -10`, Swival passes `rtk git log --oneline -10` to the shell instead, and the compact RTK output goes back to the model.

## Limitations

- Only one middleware command is supported. Chain multiple behaviors inside a single adapter script.
- Middleware runs only for `run_command` and `run_shell_command`. Other tools (file reads, web fetches, etc.) are not affected.
- Middleware is stateless across calls. Each invocation is a fresh subprocess.
- The `timeout` field in the payload is the command's execution timeout, not the middleware's own deadline. Middleware itself has a 10-second limit.

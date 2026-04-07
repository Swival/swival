# Custom Commands

Custom commands let you run external scripts and feed their output into the conversation. Type `!name` and Swival looks up an executable called `name` in your commands directory, runs it, and injects its stdout as the next user message.

In interactive mode, custom commands are always available. In one-shot mode, they require `--oneshot-commands` because the input may come from an untrusted source:

```sh
swival --oneshot-commands "!context"
```

## Setup

Create the commands directory and add executable scripts:

```sh
mkdir -p ~/.config/swival/commands
```

If you set `XDG_CONFIG_HOME`, the directory is `$XDG_CONFIG_HOME/swival/commands` instead.

Each file in this directory is a command. The file must be executable and its name (or stem, ignoring extension) must contain only letters, digits, hyphens, and underscores.

On Unix, extensionless files work fine. On Windows, name the file with an extension (`greet.bat`, `greet.cmd`, `greet.exe`). Swival first tries an exact match on the bare name; if that fails, it scans for executable files whose stem matches. The stem comparison is case-insensitive on Windows. If multiple executable files share the same stem (e.g. `greet.sh` and `greet.bat`), Swival reports an ambiguity error. Non-executable files with the same stem are ignored.

```sh
cat > ~/.config/swival/commands/context <<'EOF'
#!/bin/sh
base_dir="$1"
cd "$base_dir"
echo "## Git status"
git status --short
echo "## Recent commits"
git log --oneline -5
EOF
chmod +x ~/.config/swival/commands/context
```

## Usage

Prefix the command name with `!`:

```text
swival> !context
```

This runs the `context` script and sends its stdout to the model as if you had typed it. Everything after the command name is passed as a single argument (`$2`):

```text
swival> !deploy staging environment --dry-run
```

The command receives `$2` = `staging environment --dry-run`. Leading and trailing whitespace is stripped, but the text is otherwise passed through verbatim — no shell parsing or quote handling.

## How Commands Are Called

The command receives `base_dir` (the project root) as `$1`. If arguments are given, the raw text after the command name is stripped and passed as `$2`. If no arguments are given, only `$1` is passed. The working directory is also set to `base_dir`.

```text
$COMMANDS_DIR/name $base_dir "$args"
```

## Environment Variables

Commands inherit the parent environment. Swival also sets:

| Variable       | Description                                                            |
| -------------- | ---------------------------------------------------------------------- |
| `SWIVAL_MODEL` | The resolved model identifier for the current session (when available) |

## Output Handling

The command's stdout is stripped, printed to the terminal for review, and then injected as a user message. If the output is too large for the remaining context window, it is truncated to fit. When the context window size is unknown, a hard cap of 100KB applies.

The command's stderr is printed to the terminal on success. On failure (non-zero exit), Swival prints a single error message using stderr, stdout, or the exit code (in that priority order) and does not inject anything into the conversation.

## Timeouts

Commands have a 30-second timeout. If the command does not finish in time, Swival prints a timeout error and skips the injection.

## History

Custom command output is logged to `.swival/HISTORY.md` with the label `[!name] !name [args...]` so you can distinguish command-driven turns from typed input.

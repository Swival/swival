# Custom Commands

Custom commands let you run external scripts from the REPL and feed their output into the conversation. Type `!name` and Swival looks up an executable called `name` in your commands directory, runs it, and injects its stdout as the next user message.

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

In the REPL, prefix the command name with `!`:

```text
swival> !context
```

This runs the `context` script and sends its stdout to the model as if you had typed it. Additional arguments are passed through:

```text
swival> !context --all
```

Quoted arguments are handled correctly:

```text
swival> !deploy "staging environment" --dry-run
```

## How Commands Are Called

The command receives `base_dir` (the project root) as its first positional argument, followed by any arguments from the REPL line. The working directory is also set to `base_dir`. This matches the convention used by reviewer scripts.

```text
$COMMANDS_DIR/name $base_dir [args...]
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

## Edge Cases

- `!` alone or `! foo` (space after `!`) are treated as regular input, not commands
- Names containing `/`, `..`, spaces, or special characters are rejected
- If the commands directory does not exist, Swival prints a hint and skips
- If the command is not found or not executable, Swival prints an error and skips
- Empty output (stdout is blank after stripping) is skipped with a note
- If a bad shebang or missing interpreter prevents the command from starting, Swival catches the error and continues

## History

Custom command output is logged to `.swival/HISTORY.md` with the label `[!name] !name [args...]` so you can distinguish command-driven turns from typed input.

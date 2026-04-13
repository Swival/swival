# Custom Commands

Custom commands extend Swival with personal scripts and prompt templates stored in your commands directory. Type `!name` and Swival resolves `name` to a file in `~/.config/swival/commands/`, then either runs it and injects its stdout, or inlines its text content directly into the conversation.

Not to be confused with `!! <command>` (quick shell), which runs an arbitrary shell command and prints the output without any LLM involvement. See the [Input Commands](usage.md#input-commands) section for details.

In interactive mode, custom commands are always available. In one-shot mode, they require `--oneshot-commands` because the input may come from an untrusted source:

```sh
swival --oneshot-commands "!context"
```

## Setup

Create the commands directory:

```sh
mkdir -p ~/.config/swival/commands
```

If you set `XDG_CONFIG_HOME`, the directory is `$XDG_CONFIG_HOME/swival/commands` instead.

A file in this directory is a valid command if its name (or stem, ignoring extension) contains only letters, digits, hyphens, and underscores, and the file is not a dot-file or backup artefact (names starting with `.`, ending with `~`, or with extensions `.bak`, `.orig`, `.swp`, `.swo`, `.tmp`, `.pyc` are silently skipped).

There are two kinds of commands:

### Executable scripts

An executable script is run as a subprocess, and its stdout is injected as the next user message. Any language with a shebang line works.

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

### Text templates

A plain text file without the execute bit is treated as a prompt template. Its content is read and inlined directly into the conversation — no subprocess is started. The file must be valid UTF-8 and contain no null bytes (binary files are rejected with "command not executable").

```sh
cat > ~/.config/swival/commands/review <<'EOF'
Review the changes in $1 for correctness, style, and test coverage. Flag any security issues first.
EOF
```

No `chmod +x` is needed. The file is inlined as-is, with leading and trailing whitespace preserved.

## Usage

Prefix the command name with `!`:

```text
swival> !context
swival> !review main branch
```

Everything after the command name is the argument string. How it is used depends on the command type (see below).

## Resolution and Precedence

Swival scans the commands directory for all eligible files whose name or stem matches the command name (case-insensitive on Windows), then picks the best match using this priority order:

1. Exact-name executable (`commands/name` with execute bit)
2. Stem-match executable (`commands/name.sh`, `commands/name.py`, …)
3. Exact-name text template (`commands/name` without execute bit)
4. Stem-match text template (`commands/name.md`, `commands/name.txt`, …)

Executables always win over text templates. If more than one file ties within a tier, Swival reports an ambiguity error. On Unix, extensionless files take the exact-name slots; on Windows, name scripts with an extension (`greet.bat`, `greet.cmd`) and Swival resolves them via the stem.

## Argument Passing

### Executable scripts

The script receives the project root as `$1`. If arguments are provided after the command name, the raw argument string is passed as `$2`. If no arguments are given, only `$1` is passed. The working directory is also set to the project root.

```text
$COMMANDS_DIR/name $base_dir "$args"
```

So `!deploy staging --dry-run` sets `$1` to the project root and `$2` to `staging --dry-run`.

### Text templates

For text templates there is no subprocess, so `$1` is not reserved for the project root. Instead, `$1` and `$@` in the file content are both replaced with the argument string verbatim before injection. If no arguments are given, `$1` and `$@` are left as-is in the content.

```sh
# commands/review (no execute bit)
Review the changes in $1 for correctness and test coverage.

# !review src/auth.py  →  "Review the changes in src/auth.py for correctness and test coverage."
```

Note the difference from executable commands: if you convert a text template into a shell script, rename `$1` to `$2` and add the `base_dir` parameter.

## Environment Variables

Executable scripts inherit the parent environment. Swival also sets:

| Variable       | Description                                                            |
| -------------- | ---------------------------------------------------------------------- |
| `SWIVAL_MODEL` | The resolved model identifier for the current session (when available) |

Environment variables are not relevant for text templates since no process is spawned.

## Output Handling

For executable scripts, the command's stdout is stripped, printed to the terminal for review, and injected as a user message. For text templates, the content is inlined silently with a brief status line (`[!name] inline: ~/path`). Both paths respect the context window: if the content is too large for the remaining window, it is truncated to fit. When the context window size is unknown, a hard cap of 100KB applies.

A whitespace-only result (after stripping) is treated as no output and skipped.

For executable scripts, stderr is printed to the terminal on success. On failure (non-zero exit), Swival prints a single error message using stderr, stdout, or the exit code (in that priority order) and injects nothing.

## Timeouts

Executable scripts have a 30-second timeout. Text templates have no timeout since no process is spawned.

## Community Commands

The [swival-commands](https://github.com/Swival/swival-commands) repository has a growing collection of ready-to-use commands contributed by the community, including a full-repo security audit, a C-specific vulnerability scanner, and a pull request reviewer. Installation is just copying files into your commands directory.

If you have built a command that others might find useful, contributions are welcome there.

## History

Custom command output is logged to `.swival/HISTORY.md` with the label `[!name] !name [args...]` so you can distinguish command-driven turns from typed input.

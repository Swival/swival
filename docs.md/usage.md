# Usage

Swival has two operating modes. The default mode is one-shot, where you give the agent a task and let it run to completion. The second mode is an interactive REPL, where conversation history stays live across follow-up prompts.

## One-Shot Mode

In one-shot mode, you pass one task on the command line and Swival keeps looping through tool calls until it reaches a final answer or hits the turn limit.

```sh
swival "Review this codebase for security issues"
```

The final answer is written to standard output. Diagnostics such as turn logs, timing, and tool traces are written to standard error.

If you want a clean output stream for scripting, use `--quiet` or `-q`.

```sh
swival -q "Summarize the API surface of src/" > api-summary.txt
```

You can grant command execution when needed by explicitly whitelisting command names.

```sh
swival --allowed-commands ls,git,python3 \
    "Create a tool that returns a random number between 0 and 42"
```

That whitelist changes what the agent can do. If `python3` is available, it can use Python for implementation and verification. If no commands are whitelisted, `run_command` is unavailable unless you enable YOLO mode.

A successful run exits with code `0`. A runtime or configuration failure exits with code `1`. A run that reaches the turn limit before finishing exits with code `2`.

## Interactive Mode

REPL mode keeps a shared conversation state, so each new question can build on earlier turns.

```sh
swival --repl
```

You can also launch the REPL and send an initial question immediately.

```sh
swival --repl "Look at the project structure and tell me what this does"
```

The REPL is built on `prompt-toolkit`, so it supports input history, history search, and normal terminal line editing.

## REPL Commands

`/help` prints the command reference in the terminal.

`/clear` drops conversation history back to the initial system state and also resets internal thinking and file-tracking state.

`/compact` compacts older tool output in memory. `/compact --drop` is more aggressive and also drops middle turns.

`/add-dir <path>` grants read and write access to an additional directory for the current session.

`/add-dir-ro <path>` grants read-only access to an additional directory. The agent can read, list, and grep files there but cannot write, edit, or delete them.

`/extend` doubles the current turn budget. `/extend <N>` sets the turn budget to an exact value.

`/continue` restarts the agent loop for the existing conversation without adding a new user message.

`/init` runs a three-pass workflow that scans your project and generates an `AGENTS.md` file.

`/exit` and `/quit` leave the REPL. Pressing `Ctrl-D` exits as well.

## CLI Flags

### Model And Provider Flags

`--provider` chooses the backend provider and defaults to `lmstudio`. Valid values are `lmstudio`, `huggingface`, and `openrouter`.

`--model` overrides auto-discovery with a fixed model identifier.

`--base-url` sets a custom API base URL. For LM Studio, the default base URL is `http://127.0.0.1:1234` when `--base-url` is not set.

`--api-key` provides a key directly on the command line and takes precedence over provider environment variables such as `HF_TOKEN` or `OPENROUTER_API_KEY`.

### Behavior Tuning Flags

`--max-turns` sets the maximum number of loop iterations and defaults to `100`.

`--max-output-tokens` sets the model output budget per call and defaults to `32768`.

`--max-context-tokens` requests a context window size. With LM Studio, this may trigger a model reload.

`--temperature` controls sampling temperature and defaults to the provider default when omitted.

`--top-p` controls nucleus sampling and defaults to `1.0`.

`--seed` passes a random seed for providers that support reproducible sampling.

`--proactive-summaries` enables periodic checkpoint summarization of the conversation. Every ten turns, recent turns are summarized and stored internally. These summaries survive context compaction and give the model a condensed record of earlier work that would otherwise be lost. Useful for long-running sessions.

### Sandboxing Flags

`--base-dir` defines the base directory for file tools and defaults to the current directory.

`--allowed-commands` enables command execution through a comma-separated whitelist.

`--add-dir` grants read and write access to additional directories and can be repeated.

`--add-dir-ro` grants read-only access to additional directories and can be repeated. The agent can read, list, and grep files in these directories but cannot write, edit, or delete them.

`--yolo` disables both filesystem sandbox checks and command whitelisting, except that filesystem root access is still blocked.

`--no-read-guard` disables the read-before-write guard that normally prevents editing existing files before reading them.

### System Prompt And Instruction Flags

`--system-prompt` replaces the built-in prompt with your own prompt text.

`--no-system-prompt` omits the system message entirely.

`--no-instructions` prevents loading `CLAUDE.md` and `AGENTS.md` from both the base directory and the user config directory.

`--system-prompt` and `--no-system-prompt` are mutually exclusive.

### Skills Flags

`--skills-dir` adds external skill directories and can be passed more than once.

`--no-skills` disables skill discovery and removes the `use_skill` tool path.

### Output And Reporting Flags

`--quiet` and `-q` suppress diagnostics and keep terminal output focused on final answers.

`--report FILE` writes a JSON run report to `FILE`. This flag is incompatible with `--repl`.

`--reviewer COMMAND` runs an external reviewer after each answer. The command string is shell-split, so you can pass arguments inline (e.g. `--reviewer "swival --reviewer-mode"`). This flag is incompatible with `--repl`.

`--max-review-rounds N` limits how many times the reviewer can request retries and defaults to `5`. Set to `0` to accept the first answer without retries.

`--reviewer-mode` runs Swival as a reviewer process that speaks the reviewer protocol. It reads `base_dir` from the positional argument, the answer from standard input, evaluates with the LLM, and exits 0 (accept), 1 (retry), or 2 (error). Incompatible with `--repl` and `--reviewer`.

`--review-prompt TEXT` appends custom instructions to the built-in review prompt when running in reviewer mode.

`--objective FILE` reads the task description from a file instead of the `SWIVAL_TASK` environment variable when running in reviewer mode.

`--verify FILE` reads verification criteria from a file and includes them in the review prompt when running in reviewer mode.

`--no-history` disables writes to `.swival/HISTORY.md`.

`--color` forces ANSI color on standard error, and `--no-color` disables ANSI color even on TTY output.

### Configuration Flags

`--init-config` generates a global config file at `~/.config/swival/config.toml` and exits. The file is a fully commented template showing all available settings.

`--init-config --project` generates a project-local config file at `swival.toml` in the current base directory instead.

Neither flag requires a question argument. Both refuse to overwrite an existing config file.

For the full report schema and analysis workflow, see [Reports](reports.md). For the reviewer protocol and examples, see [Reviews](reviews.md).

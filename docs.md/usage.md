# Usage

Swival has two operating modes. The default mode is one-shot, where you give the agent a task and let it run to completion. The second mode is an interactive REPL, where conversation history stays live across follow-up prompts.

## One-Shot Mode

In one-shot mode, you pass one task on the command line and Swival keeps looping through tool calls until it reaches a final answer or hits the turn limit.

```sh
swival "Review this codebase for security issues"
```

If you omit the positional task and pipe stdin, Swival reads the task from stdin instead.

```sh
swival -q < objective.md

cat prompts/review.md | swival --provider huggingface --model zai-org/GLM-5
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

A successful run exits with code `0`. A runtime or configuration failure exits with code `1`. A run that reaches the turn limit before finishing exits with code `2`. A run interrupted with Ctrl+C exits with code `130`.

## Interactive Mode

REPL mode keeps a shared conversation state, so each new question can build on earlier turns.

```sh
swival --repl
```

The REPL is built on `prompt-toolkit`, so it supports input history, history search, and normal terminal line editing.

## REPL Commands

`/help` prints the command reference in the terminal.

`/clear` drops conversation history back to the initial system state and also resets internal thinking and file-tracking state.

`/compact` compacts older tool output in memory. `/compact --drop` is more aggressive and also drops middle turns.

`/save [label]` sets a context checkpoint at the current position. If you omit the label, it defaults to `user-checkpoint`. Only one checkpoint can be active at a time.

`/restore` summarizes everything since the last checkpoint and collapses it into a single recap message. The summary is generated automatically by calling the LLM. If no explicit checkpoint was set with `/save`, it collapses from the last implicit checkpoint (typically the most recent user message). If context was compacted between `/save` and `/restore`, the checkpoint is invalidated and you'll need to `/unsave` and start over.

`/unsave` cancels the active checkpoint without collapsing anything.

`/add-dir <path>` grants read and write access to an additional directory for the current session.

`/add-dir-ro <path>` grants read-only access to an additional directory. The agent can read, list, and grep files there but cannot write, edit, or delete them.

`/extend` doubles the current turn budget. `/extend <N>` sets the turn budget to an exact value.

`/continue` restarts the agent loop for the existing conversation without adding a new user message.

`/continue-status` shows whether a continue file exists from a prior interrupted session and previews its contents.

`/learn` reviews the current session for mistakes and confusions, then persists notes to `.swival/memory/MEMORY.md` for future sessions to learn from. On subsequent runs, memory entries are parsed by heading and selectively injected into the prompt using BM25 retrieval keyed from the user's question, keeping memory token cost bounded.

`/tools` lists all tools available in the current session — built-in, MCP, and A2A — grouped by source with full descriptions.

`/init` runs a three-pass workflow that scans your project and generates an `AGENTS.md` file.

`/exit` and `/quit` leave the REPL. Pressing `Ctrl-D` exits as well.

## CLI Flags

`swival --help` uses the same grouping below and ends with copy-paste examples.

### Model And Provider Flags

`--provider` chooses the backend provider and defaults to `lmstudio`. Valid values are `lmstudio`, `huggingface`, `openrouter`, `generic`, `google`, and `chatgpt` (for ChatGPT Plus/Pro subscriptions).

`--model` overrides auto-discovery with a fixed model identifier.

`--base-url` sets a custom API base URL. For LM Studio, the default base URL is `http://127.0.0.1:1234` when `--base-url` is not set.

`--api-key` provides a key directly on the command line and takes precedence over provider environment variables (`HF_TOKEN` for huggingface, `OPENROUTER_API_KEY` for openrouter, `OPENAI_API_KEY` for generic, `GEMINI_API_KEY` for google, or `CHATGPT_API_KEY` for chatgpt).

### Behavior Tuning Flags

`--max-turns` sets the maximum number of loop iterations and defaults to `100`.

`--max-output-tokens` sets the model output budget per call and defaults to `32768`.

`--max-context-tokens` requests a context window size. With LM Studio, this may trigger a model reload.

`--temperature` controls sampling temperature and defaults to the provider default when omitted.

`--top-p` controls nucleus sampling and defaults to `1.0`.

`--seed` passes a random seed for providers that support reproducible sampling.

`--extra-body JSON` passes extra parameters to the LLM API call. The value must be a JSON object. This is useful for provider-specific or model-specific options that Swival does not expose as dedicated flags.

```sh
swival --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}' "task"
```

`--reasoning-effort LEVEL` sets the reasoning effort for models that support tunable reasoning (e.g. gpt-5.4). Valid levels are `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, and `default`.

```sh
swival --provider chatgpt --model gpt-5.4 --reasoning-effort high "task"
```

`--proactive-summaries` enables periodic checkpoint summarization of the conversation. Every ten turns, recent turns are summarized and stored internally. These summaries survive context compaction and give the model a condensed record of earlier work that would otherwise be lost.

Useful for long-running sessions.

### Sandboxing Flags

`--base-dir` defines the base directory for file tools and defaults to the current directory.

`--allowed-commands` enables command execution through a comma-separated whitelist.

`--add-dir` grants read and write access to additional directories and can be repeated.

`--add-dir-ro` grants read-only access to additional directories and can be repeated. The agent can read, list, and grep files in these directories but cannot write, edit, or delete them.

`--yolo` disables both filesystem sandbox checks and command whitelisting, except that filesystem root access is still blocked.

`--sandbox` selects the sandbox backend. The default is `builtin`, which uses application-layer path guards. Set it to `agentfs` to re-exec Swival inside an [AgentFS](agentfs.md) overlay for OS-enforced write isolation. Requires the `agentfs` binary on PATH.

`--sandbox-session` sets an AgentFS session ID so sandbox state persists across runs. Only valid with `--sandbox agentfs`.

`--sandbox-strict-read` enables strict read isolation inside the AgentFS sandbox. When set, the agent process can only read files that have been explicitly allowed, rather than having unrestricted read access to the host filesystem. This requires an AgentFS version with strict read support. If the installed version does not support it, Swival exits with an error. Only valid with `--sandbox agentfs`.

`--no-sandbox-auto-session` disables the automatic session ID that Swival generates when `--sandbox agentfs` is used without an explicit `--sandbox-session`. By default, Swival derives a deterministic session ID from the project directory so that re-running in the same directory reuses the overlay automatically. Pass this flag to get a fresh, ephemeral overlay each time.

`--no-read-guard` disables the read-before-write guard that normally prevents editing existing files before reading them.

### System Prompt And Instruction Flags

`--system-prompt` replaces the built-in prompt with your own prompt text.

`--no-system-prompt` omits the system message entirely.

`--no-instructions` prevents loading `CLAUDE.md` and `AGENTS.md` from both the base directory and the user config directory.

`--no-memory` prevents loading auto-memory from `.swival/memory/`.

`--memory-full` injects the entire `MEMORY.md` into the prompt instead of the default budgeted retrieval. Useful as a fallback if retrieval misses entries.

`--system-prompt` and `--no-system-prompt` are mutually exclusive.

### Skills Flags

`--skills-dir` adds external skill directories and can be passed more than once.

`--no-skills` disables skill discovery and removes the `use_skill` tool path.

### MCP Flags

`--no-mcp` disables MCP server connections entirely, even if servers are configured in `swival.toml` or `.mcp.json`.

`--mcp-config FILE` provides an explicit path to an MCP JSON config file. When set, this replaces the default `.mcp.json` lookup in the project root.

See [MCP](mcp.md) for full configuration details.

### A2A Flags

`--no-a2a` disables A2A agent connections entirely, even if agents are configured in `swival.toml`.

`--a2a-config FILE` provides an explicit path to an A2A TOML config file with `[a2a_servers.*]` tables.

See [A2A](a2a.md) for full configuration details.

### A2A Server Flags

`--serve` starts Swival as an A2A server instead of running a one-shot task or REPL. Incoming tasks are handled by Session instances keyed by contextId. Incompatible with `--repl`.

`--serve-host HOST` sets the bind address for the A2A server. Default is `0.0.0.0`.

`--serve-port PORT` sets the port for the A2A server. Default is `8080`.

`--serve-auth-token TOKEN` enables bearer token authentication on the A2A server. When set, all requests must include a valid `Authorization: Bearer <token>` header.

`--serve-name NAME` sets a custom agent name in the A2A agent card. Defaults to `swival (provider/model)`.

`--serve-description TEXT` sets a custom agent description in the A2A agent card.

See [A2A](a2a.md) for full server documentation.

### Output And Reporting Flags

`--quiet` and `-q` suppress diagnostics and keep terminal output focused on final answers.

`--report FILE` writes a JSON run report to `FILE`. This flag is incompatible with `--repl`.

`--reviewer COMMAND` runs an external reviewer after each answer. The command string is shell-split, so you can pass arguments inline (e.g. `--reviewer "swival --reviewer-mode"`). This flag is incompatible with `--repl`.

`--max-review-rounds N` limits how many times the reviewer can request retries and defaults to `15`. Set to `0` to accept the first answer without retries.

`--self-review` uses a second Swival instance as reviewer, inheriting provider, model, skills-dir, and yolo settings from the current invocation. Incompatible with `--repl` and `--reviewer`. See [Reviews](reviews.md) for details.

`--reviewer-mode` runs Swival as a reviewer process that speaks the reviewer protocol. It reads `base_dir` from the positional argument, the answer from standard input, evaluates with the LLM, and exits 0 (accept), 1 (retry), or 2 (error). Incompatible with `--repl` and `--reviewer`.

`--review-prompt TEXT` appends custom instructions to the built-in review prompt when running in reviewer mode.

`--objective FILE` reads the task description from a file instead of the `SWIVAL_TASK` environment variable when running in reviewer mode.

`--verify FILE` reads verification criteria from a file and includes them in the review prompt when running in reviewer mode.

`--no-history` disables writes to `.swival/HISTORY.md`.

`--no-continue` disables continue-here files. When set, Swival will not write `.swival/continue.md` on interruption and will not load it on startup.

`--color` forces ANSI color on standard error, and `--no-color` disables ANSI color even on TTY output.

### Caching Flags

`--cache` enables LLM response caching. When an identical request is seen again, the cached response is returned without contacting the LLM. The cache is stored in a SQLite database at `.swival/cache.db` by default. Off by default.

`--cache-dir PATH` overrides the default cache database directory. Useful for sharing a cache across projects.

### Configuration Flags

`--init-config` generates a global config file at `~/.config/swival/config.toml` and exits. The file is a commented starter template covering the main settings and integrations.

`--init-config --project` generates a project-local config file at `swival.toml` in the current base directory instead.

Neither flag requires a question argument. Both refuse to overwrite an existing config file.

### Other Flags

`--version` prints the version and exits.

For the full report schema and analysis workflow, see [Reports](reports.md). For the reviewer protocol and examples, see [Reviews](reviews.md).

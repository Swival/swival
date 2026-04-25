# Usage

Swival has two operating modes. In one-shot mode you give the agent a task and let it run to completion. In interactive mode conversation history stays live across follow-up prompts. When you run `swival` on a terminal without a task, it enters interactive mode automatically.

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

Command execution is unrestricted by default. You can restrict it to a whitelist of command names:

```sh
swival --commands ls,git,python3 \
    "Create a tool that returns a random number between 0 and 42"
```

`--commands` accepts `"all"` (unrestricted, the default), `"none"` (disabled), `"ask"` (interactive per-bucket approval), or a comma-separated whitelist. With `--commands all` or `--yolo`, Swival exposes both `run_command` (array argv) and `run_shell_command` (shell string with pipes, redirects, etc.). In ask and whitelist modes, only `run_command` is available. Pass `--commands none` to remove both tools entirely. Pass `--commands ask` to approve each command category interactively.

A successful run exits with code `0`. A runtime or configuration failure exits with code `1`. A run that reaches the turn limit before finishing exits with code `2`. A run interrupted with Ctrl+C exits with code `130`. A run terminated by SIGTERM exits with code `143`.

## Interactive Mode

REPL mode keeps a shared conversation state, so each new question can build on earlier turns.

```sh
swival
```

The REPL is built on `prompt-toolkit`, so it supports input history, history search, and normal terminal line editing.

## Input Commands

Slash commands work in interactive mode (typed at the REPL prompt) and optionally in one-shot mode. In one-shot mode, command dispatch is disabled by default because the input may come from an untrusted source (a pipe, a file, or an upstream program). Pass `--oneshot-commands` to enable it:

```sh
swival --oneshot-commands "/status"
swival --oneshot-commands $'/profile fast\n/simplify swival/agent.py'
```

Without `--oneshot-commands`, input that looks like a command script is treated as a plain natural-language prompt. A few commands (`/continue`, `/copy`) are REPL-only and are rejected even with `--oneshot-commands`.

`/help` prints the command reference in the terminal.

`/clear` (or `/new`) drops conversation history back to the initial system state and also resets internal thinking and file-tracking state.

`/compact` compacts older tool output in memory. `/compact --drop` is more aggressive and also drops middle turns.

`/save [label]` sets a context checkpoint at the current position. If you omit the label, it defaults to `user-checkpoint`. Only one checkpoint can be active at a time.

`/restore` summarizes everything since the last checkpoint and collapses it into a single recap message. The summary is generated automatically by calling the LLM. If no explicit checkpoint was set with `/save`, it collapses from the last implicit checkpoint (typically the most recent user message). If context was compacted between `/save` and `/restore`, the checkpoint is invalidated and you'll need to `/unsave` and start over.

`/unsave` cancels the active checkpoint without collapsing anything.

`/add-dir <path>` grants read and write access to an additional directory for the current session.

`/add-dir-ro <path>` grants read-only access to an additional directory. The agent can read, list, and grep files there but cannot write, edit, or delete them.

`/extend` doubles the current turn budget. `/extend <N>` sets the turn budget to an exact value.

`/continue` restarts the agent loop for the existing conversation without adding a new user message.

`/status` shows a compact session overview: model, endpoint, context usage, message/turn counts, file access, mode flags, and state summaries (thinking, todo, snapshot, checkpoints, continue file).

`/audit [path|glob]` runs a staged security audit over committed Git-tracked code. It triages files by attack surface, deep-reviews escalated files, verifies each finding with an isolated proof-of-concept agent, and writes patches and reports to `audit-findings/`. Pass `--resume` to continue a previous run, `--regen` to regenerate reports and patches for a completed run, `--workers N` to control parallelism. REPL-only. See [Security Audit](audit.md) for the full walkthrough.

`/learn` reviews the current session for mistakes and confusions, then persists notes to `.swival/memory/MEMORY.md` for future sessions to learn from. On subsequent runs, memory entries are parsed by heading and selectively injected into the prompt using BM25 retrieval keyed from the user's question, keeping memory token cost bounded.

`/simplify [focus]` inspects the codebase for low-risk simplification opportunities and applies them, preserving all observable behavior. Optionally scope it to a file or area (e.g. `/simplify swival/edit.py`). Prefers recently changed code but expands outward as needed.

`/tools` lists all tools available in the current session — built-in, MCP, and A2A — grouped by source with full descriptions.

`/copy` copies the most recent assistant output to the system clipboard. Uses `pbcopy` on macOS, `clip` on Windows, and `wl-copy` or `xclip` on Linux (one of these must be installed). Warns if no output exists yet or if no clipboard utility is found.

`/remember <text>` adds a convention to the project-level `AGENTS.md` file under `## Conventions`. Duplicates are skipped.

`/profile` lists available profiles with the active one marked. `/profile NAME` switches to a different profile mid-session — the LLM settings change but conversation history, tools, and all other state are preserved. `/profile -` reverts to the profile that was active at session start. Each switch re-resolves the provider from scratch, so provider-specific behavior (Google endpoint rewrite, LM Studio model discovery, etc.) works correctly.

`/init` scans your project for build/test/lint/format commands and cross-cutting conventions, then generates a structured `AGENTS.md` file with a `## Workflow` section (exact commands including an after-every-edit reflex) followed by a `## Conventions` section. Validates the output and retries once if the structure is wrong.

`/exit` and `/quit` leave the REPL. Pressing `Ctrl-D` exits as well.

`!! <command>` runs a shell command directly and prints the output. The LLM is not involved — nothing is added to the conversation and no agent turn runs. This is a quick way to check `git status`, run `make test`, or inspect files without leaving the REPL. Pipes, redirects, and `&&` all work. REPL-only; not available in one-shot mode.

`!command [args]` resolves a file from your commands directory (`$XDG_CONFIG_HOME/swival/commands/` or `~/.config/swival/commands/`). Executable files are run as scripts and their stdout is injected; plain text files are inlined directly as a prompt template. In one-shot mode, bang commands require `--oneshot-commands`. See [Custom Commands](custom-commands.md) for setup and details.

## CLI Flags

`swival --help` uses the same grouping below and ends with copy-paste examples.

### Model And Provider Flags

`--profile NAME` selects a named LLM profile from config. Profiles bundle provider, model, and related settings under a short name so you can switch setups without retyping flags. See [Profiles](customization.md#profiles) for how to define them.

```sh
swival --profile gpt5 "review this patch"
```

`--list-profiles` prints available profiles and exits. The active profile is marked with an arrow and shows which layer selected it (CLI flag, project config, or global config).

`--provider` chooses the backend provider and defaults to `lmstudio`. Valid values are `lmstudio`, `llamacpp`, `huggingface`, `openrouter`, `generic`, `google`, `chatgpt` (for ChatGPT Plus/Pro subscriptions), `bedrock` (AWS Bedrock), and `command` (shells out to an external program).

`--model` overrides auto-discovery with a fixed model identifier.

`--base-url` sets a custom API base URL. For LM Studio, the default is `http://127.0.0.1:1234`; for llamacpp, the default is `http://127.0.0.1:8080`.

`--api-key` provides a key directly on the command line and takes precedence over provider environment variables (`HF_TOKEN` for huggingface, `OPENROUTER_API_KEY` for openrouter, `OPENAI_API_KEY` for generic, `GEMINI_API_KEY` for google, `CHATGPT_API_KEY` for chatgpt). Bedrock uses the AWS credential chain instead (`AWS_PROFILE`, env vars, or IAM roles). Local providers (`lmstudio`, `llamacpp`) need no key.

`--user-agent` sets the `User-Agent` header sent with LLM API requests. Defaults to `Swival/<version>`. Useful when a provider requires a specific agent string (e.g. `--user-agent 'KimiCLI/Swival'` for Kimi's API).

`--aws-profile` selects a named AWS profile from `~/.aws/config` for the `bedrock` provider. Overrides the `AWS_PROFILE` environment variable.

When `--profile` is combined with explicit flags like `--provider` or `--model`, the explicit flags win on a per-key basis.

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

`--retries N` sets the maximum number of provider retries on transient network errors and defaults to `5`. Set to `1` to disable retries entirely.

`--reasoning-effort LEVEL` sets the reasoning effort for models that support tunable reasoning (e.g. gpt-5.4). Valid levels are `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, and `default`.

```sh
swival --provider chatgpt --model gpt-5.4 --reasoning-effort high "task"
```

`--proactive-summaries` enables periodic checkpoint summarization of the conversation. Every ten turns, recent turns are summarized and stored internally. These summaries survive context compaction and give the model a condensed record of earlier work that would otherwise be lost.

Useful for long-running sessions.

### Sandboxing Flags

`--base-dir` defines the base directory for file tools and defaults to the current directory.

`--files` controls filesystem tool access. Accepts `"some"` (the default, workspace only), `"all"` (unrestricted), or `"none"` (`.swival/` directory only). With `--files none`, the agent can still run commands and fetch URLs but cannot read or write project files.

`--commands` controls command execution. Accepts `"all"` (unrestricted, the default), `"none"` (disabled), `"ask"` (interactive approval per command bucket), or a comma-separated whitelist. With `--commands all` or `--yolo`, both `run_command` and `run_shell_command` are exposed. In ask and whitelist modes, only `run_command` is available.

`--add-dir` grants read and write access to additional directories and can be repeated.

`--add-dir-ro` grants read-only access to additional directories and can be repeated. The agent can read, list, and grep files in these directories but cannot write, edit, or delete them.

`--oneshot-commands` enables `/` and `!` command dispatch in one-shot mode. By default, command-like input in one-shot mode is treated as plain text because the input may come from an untrusted source. This flag is a no-op in interactive mode, where commands are always available. Can also be set in config as `oneshot_commands = true`.

`--yolo` is shorthand for `--files all --commands all`. It does not imply `--oneshot-commands`. Explicit `--files` or `--commands` flags take precedence over `--yolo`. Filesystem root access is always blocked.

`--sandbox` selects the sandbox backend. The default is `builtin`, which uses application-layer path guards. Set it to `agentfs` to re-exec Swival inside an [AgentFS](agentfs.md) overlay for OS-enforced write isolation. Requires the `agentfs` binary on PATH.

`--sandbox-session` sets an AgentFS session ID so sandbox state persists across runs. Only valid with `--sandbox agentfs`.

`--sandbox-strict-read` enables strict read isolation inside the AgentFS sandbox. When set, the agent process can only read files that have been explicitly allowed, rather than having unrestricted read access to the host filesystem. This requires an AgentFS version with strict read support. If the installed version does not support it, Swival exits with an error. Only valid with `--sandbox agentfs`.

`--no-sandbox-auto-session` disables the automatic session ID that Swival generates when `--sandbox agentfs` is used without an explicit `--sandbox-session`. By default, Swival derives a deterministic session ID from the project directory so that re-running in the same directory reuses the overlay automatically. Pass this flag to get a fresh, ephemeral overlay each time.

`--no-read-guard` disables the read-before-write guard that normally prevents editing existing files before reading them.

### System Prompt And Instruction Flags

`--system-prompt` replaces the built-in prompt with your own prompt text.

`--no-system-prompt` omits the system message entirely.

`--no-instructions` prevents loading `CLAUDE.md` and `AGENTS.md` from the base directory, user config directory, and `~/.agents/`.

`--no-memory` prevents loading auto-memory from `.swival/memory/`.

`--memory-full` injects the entire `MEMORY.md` into the prompt instead of the default budgeted retrieval. Useful as a fallback if retrieval misses entries.

`--system-prompt` and `--no-system-prompt` are mutually exclusive.

### Skills Flags

`--skills-dir` adds external skill directories and can be passed more than once.

`--no-skills` disables skill discovery and removes the `use_skill` tool path.

### MCP Flags

`--no-mcp` disables MCP server connections entirely, even if servers are configured in `swival.toml` or `.swival/mcp.json`.

`--mcp-config FILE` provides an explicit path to an MCP JSON config file. When set, this replaces the default `.swival/mcp.json` lookup.

See [MCP](mcp.md) for full configuration details.

### A2A Flags

`--no-a2a` disables A2A agent connections entirely, even if agents are configured in `swival.toml`.

`--a2a-config FILE` provides an explicit path to an A2A TOML config file with `[a2a_servers.*]` tables.

See [A2A](a2a.md) for full configuration details.

### Subagent Flags

`--subagents` enables parallel subagent support. When enabled, the model gets access to `spawn_subagent` and `check_subagents` tools that let it fork independent tasks into background threads. Each subagent runs its own agent loop with isolated state but shared LLM config and MCP/A2A connections. Off by default.

`--no-subagents` explicitly disables subagent support (the default).

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

`--report FILE` writes a JSON run report to `FILE`. In one-shot mode, requires a task. In REPL mode, the report covers the entire session and is written when the session ends.

`--trace-dir DIR` writes a HuggingFace-compatible JSONL session trace to `DIR`. Each session produces a `<session_id>.jsonl` file that HuggingFace auto-detects as `format:agent-traces` with harness `swival`. Works in both one-shot and REPL modes. See [Reports](reports.md) for details.

`--reviewer COMMAND` runs an external reviewer after each answer. The command string is shell-split, so you can pass arguments inline (e.g. `--reviewer "swival --reviewer-mode"`). Requires a task; incompatible with `--repl`.

`--max-review-rounds N` limits how many times the reviewer can request retries and defaults to `15`. Set to `0` to accept the first answer without retries.

`--self-review` uses a second Swival instance as reviewer, inheriting provider, model, and skills-dir settings from the current invocation. Requires a task; incompatible with `--repl` and `--reviewer`. See [Reviews](reviews.md) for details.

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

`--no-prompt-cache` disables the explicit `cache_control` annotations that Swival injects on the system message for Anthropic, Gemini, and Bedrock providers. Providers that cache automatically (OpenAI, Deepseek) are unaffected by this flag. Prompt caching is on by default.

### Configuration Flags

`--init-config` generates a global config file at `~/.config/swival/config.toml` and exits. The file is a commented starter template covering the main settings and integrations.

`--init-config --project` generates a project-local config file at `swival.toml` in the current base directory instead.

`--logout` deletes locally cached ChatGPT OAuth credentials and exits. The next `--provider chatgpt` run starts the device-code login flow again.

These setup flags do not require a question argument. Config generation refuses to overwrite an existing config file.

### Outbound Filter Flag

`--llm-filter COMMAND` runs a user-defined script before each outbound LLM request. The script receives the message list as JSON on stdin and can modify messages, redact content, or block the request. Fails closed: if the script errors or rejects, nothing is sent to the provider. See [Outbound LLM Filter](llm-filter.md) for the full script contract and examples.

### Encryption And Sanitization Flags

`--encrypt-secrets` enables transparent format-preserving encryption of recognized credential tokens before sending messages to the LLM provider. The LLM sees realistic-looking fakes, and Swival decrypts them back before tool dispatch or final output. Off by default. See [Secret Encryption](secrets.md) for details.

`--no-encrypt-secrets` explicitly disables secret encryption (the default).

`--encrypt-secrets-key HEX` provides a hex-encoded 32-byte key for secret encryption instead of the default random per-session key. Useful for stable ciphertext across sessions.

`--sanitize-thinking` strips leaked `<think>` tags from assistant responses. Some open-weight models served through vLLM emit these markers even when thinking mode is disabled. See [Thinking Tag Sanitization](customization.md#thinking-tag-sanitization) for details.

### Lifecycle Hook Flags

`--lifecycle-command COMMAND` runs a user-defined command at startup and exit as `<command> startup|exit <base_dir>`. The hook receives `SWIVAL_*` environment variables with Git and project metadata, enabling remote sync of `.swival/` state. See [Lifecycle Hooks](lifecycle-hooks.md) for the environment variable contract, execution ordering, and examples.

`--lifecycle-timeout SECONDS` sets the timeout for lifecycle hook execution (default: 300).

`--lifecycle-fail-closed` makes hook failures abort the run instead of logging a warning.

`--no-lifecycle` disables lifecycle hooks entirely, useful for nested or automated invocations.

### Command Middleware Flag

`--command-middleware COMMAND` runs a user-defined script before each `run_command` or `run_shell_command` call. The script can rewrite, block, or pass through commands. The primary use case is integrating with RTK to produce token-optimized command output. See [Command Middleware](command-middleware.md) for the full documentation.

### Other Flags

`--version` prints the version and exits.

For the full report schema and analysis workflow, see [Reports](reports.md). For the reviewer protocol and examples, see [Reviews](reviews.md).

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

cat prompts/review.md | swival --provider huggingface --model zai-org/GLM-5.2
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

`--commands` accepts `"all"` (unrestricted, the default), `"none"` (disabled), `"ask"` (interactive per-bucket approval), or a comma-separated whitelist. With `--commands all` or `--yolo`, Swival exposes both `run_command` (array argv) and `run_shell_command` (shell string with pipes, redirects, etc.). In ask and whitelist modes, only `run_command` is available.

Pass `--commands none` to remove both tools entirely. Pass `--commands ask` to approve each command category interactively.

A successful run exits with code `0`. A runtime or configuration failure exits with code `1`. A run that reaches the turn limit before finishing exits with code `2`. A run interrupted with Ctrl+C exits with code `130`. A run terminated by SIGTERM exits with code `143`.

## Interactive Mode

REPL mode keeps a shared conversation state, so each new question can build on earlier turns.

```sh
swival
```

It starts automatically when no task is given on a terminal. Pass `--repl` to force REPL mode even when a task argument is present or when stdin is not a terminal (useful for `expect`-style scripting). `--repl` is incompatible with `--reviewer`, `--self-review`, `--reviewer-mode`, and `--acp`. (`--serve` and `--acp` are separate run modes; `--serve` simply takes precedence and starts the server instead of a REPL.)

The REPL is built on `prompt-toolkit`, so it supports input history, history search, and normal terminal line editing.

The REPL greets you with an animated splash, and the rule that marks the start of each request flows into place before it settles. These only play on a genuine interactive terminal with color, never when output is piped or redirected. To turn them off entirely, for a slow terminal, a screen reader, a remote shell, or just a preference for reduced motion, set `SWIVAL_ANIMATIONS=0`:

```sh
SWIVAL_ANIMATIONS=0 swival
```

## Input Commands

Slash commands work in interactive mode (typed at the REPL prompt) and optionally in one-shot mode. In one-shot mode, command dispatch is disabled by default because the input may come from an untrusted source (a pipe, a file, or an upstream program). Pass `--oneshot-commands` to enable it:

```sh
swival --oneshot-commands "/status"
swival --oneshot-commands $'/profile fast\n/simplify swival/agent.py'
```

Without `--oneshot-commands`, input that looks like a command script is treated as a plain natural-language prompt. A few commands (`/continue`, `/copy`, `/loops`, `/unloop`, `!!`) are REPL-only and are rejected even with `--oneshot-commands`. Most other commands, including `/loop`, work in both modes.

`/help` prints the command reference in the terminal.

`/clear` (or `/new`) drops conversation history back to the initial system state and also resets internal thinking and file-tracking state.

`/compact` compacts older tool output in memory. `/compact --drop` is more aggressive and also drops middle turns.

`/save [label]` sets a context checkpoint at the current position. If you omit the label, it defaults to `user-checkpoint`. Only one checkpoint can be active at a time.

`/restore` summarizes everything since the last checkpoint and collapses it into a single recap message. The summary is generated automatically by calling the LLM. If no explicit checkpoint was set with `/save`, it collapses from the last implicit checkpoint (typically the most recent user message). If context was compacted between `/save` and `/restore`, the checkpoint is invalidated and you'll need to `/unsave` and start over.

`/unsave` cancels the active checkpoint without collapsing anything.

`/add-dir <path>` grants read and write access to an additional directory for the current session.

`/add-dir-ro <path>` grants read-only access to an additional directory. The agent can read, list, and grep files there but cannot write, edit, or delete them.

`/extend` doubles the current turn budget. `/extend <N>` sets the turn budget to an exact value.

`/goal <objective>` puts the REPL into a persistent goal mode and keeps driving toward that objective across turns until the model calls `complete_goal`, declares a blocker, or `--max-turns` is hit. Use a regular prompt for "answer or do this now"; use `/goal` for "keep working on this until it is actually done."

`/goal` with no argument prints the current status; `/goal replace`, `/goal pause`, `/goal resume`, and `/goal clear` manage the active goal. See [Goals](goal.md) for the full walkthrough, examples, and budget behavior.

`/continue` restarts the agent loop for the existing conversation without adding a new user message.

`/loop [interval] <prompt>` schedules a plain prompt to run on a recurring interval. The body must be plain natural-language text; slash commands and `!custom-command` bodies are rejected at registration time. Run those manually instead.

The interval grammar is forgiving: the compact form (`5m`, `30s`, `1h30m`) still works, and so do natural-language shapes like `1 min`, `5 minutes`, `30 sec`, `every hour`, `every 5 minutes`, `a minute`, `half an hour`, and components joined by `and` such as `1 minute and 30 seconds`. The 5-second floor and 24-hour ceiling still apply.

If the input never looks like an interval, the whole argument becomes the prompt and the interval defaults to 10 minutes. If it begins as an interval but is malformed, the command errors rather than silently scheduling the wrong prompt.

In REPL mode `/loop` is a background scheduler. Registration runs the first iteration immediately, then returns the prompt to you. Subsequent iterations fire between your commands when their interval has elapsed, each in a snapshot of the live session: the iteration sees the current conversation but its own messages, todos, thinking notes, snapshot state, file tracker, and goal state are discarded at the end.

The live transcript is never mutated, and iterations do not write to `.swival/HISTORY.md` or the continue file. Filesystem effects (files written, MCP calls, etc.) are real and not rolled back.

Up to four loops can be registered at once. Loops are in-memory only and are cancelled by `/clear`, `/new`, `/init`, and process exit; they are not persisted across restarts. After three consecutive failures a loop emits a warning, and after six it is auto-cancelled with a footer that includes the last error.

`/loops` lists active schedules with their next fire time and any consecutive-failure count. `/unloop <id>` cancels one schedule by its integer id; `/unloop all` clears them all. `/status` shows a `loops: N active` line when any are registered, plus a per-loop breakdown in verbose mode.

In one-shot mode `/loop` keeps the existing foreground-blocking behaviour, which is the recommended way to run Swival as a long-lived poller under `systemd`, `tmux`, or `nohup`. Each iteration's answer is streamed to stdout with a blank-line separator, diagnostics go to stderr, and `SIGTERM` shuts the loop down cleanly between iterations (a second `SIGTERM` exits immediately with code 143).

`Ctrl-C` once skips the current iteration; press it twice within two seconds to exit the loop. The same slash-body restriction applies.

```sh
swival --oneshot-commands '/loop 5m check open PRs and summarize blockers'
swival --oneshot-commands '/loop every hour run the build and report failures'
swival --oneshot-commands '/loop 30 sec check git status and report unusual activity'
```

`/status` shows a compact session overview: model, endpoint, context usage, message/turn counts, file access, mode flags, and state summaries (thinking, todo, snapshot, active loops, checkpoints, continue file).

`/audit [path|glob ...]` runs a staged security audit over committed Git-tracked code. It triages files by attack surface, deep-reviews escalated files, verifies each finding with an isolated proof-of-concept agent, and writes patches and reports to `audit-findings/`.

Pass `--resume` to continue a previous run (including retrying failed Phase 5 artifacts), `--regen` to regenerate reports and patches for a completed run, `--regen --finding N[,M-R]` to regenerate only selected findings by their 1-based Phase 5 number, `--all` to skip the triage selection and deep-review every file in scope, `--measure-triage` to calibrate triage recall by deep-reviewing every file and tagging findings with their triage decision, `--workers N` to control parallelism, `--patch-max-turns N` to set the Phase 5 patch-generation turn budget (default 50), and `--debug` to write a real-time JSONL debug log to `.swival/audit/debug.jsonl`.

Multiple paths or globs can be passed in a single invocation; they are unioned into one run. Works in interactive (REPL) mode and in one-shot mode when `--oneshot-commands` is set. See [Security Audit](audit.md) for the full walkthrough.

`/learn` reviews the current session for mistakes and confusions, then persists notes to `.swival/memory/MEMORY.md` for future sessions to learn from. On subsequent runs, memory entries are parsed by heading and selectively injected into the prompt using BM25 retrieval keyed from the user's question, keeping memory token cost bounded.

`/simplify [focus]` inspects the codebase for low-risk simplification opportunities and applies them, preserving all observable behavior. Optionally scope it to a file or area (e.g. `/simplify swival/edit.py`). Prefers recently changed code but expands outward as needed.

`/tools` lists all tools available in the current session — built-in, MCP, and A2A — grouped by source with full descriptions.

`/copy` copies the most recent assistant output to the system clipboard. Uses `pbcopy` on macOS, `clip` on Windows, and `wl-copy` or `xclip` on Linux (one of these must be installed). Warns if no output exists yet or if no clipboard utility is found.

`/remember <text>` adds a convention to the project-level `AGENTS.md` file under `## Conventions`. Duplicates are skipped.

`/profile` lists available profiles with the active one marked. `/profile NAME` switches to a different profile mid-session — the LLM settings change but conversation history, tools, and all other state are preserved. `/profile -` reverts to the profile that was active at session start. Each switch re-resolves the provider from scratch, so provider-specific behavior (Google endpoint rewrite, LM Studio model discovery, etc.) works correctly.

`/model` switches models within the current provider, keeping everything else (provider, credentials, reasoning effort, conversation history) as it is. Where `/profile` jumps between configured stacks, `/model` explores what the active provider can serve right now.

With no argument it opens an interactive picker listing the provider's current catalog: type to filter, move with the arrow keys, press Enter to select, and Esc to cancel. Each row shows the context window, price per million tokens, and whether the model reports tool-calling support. Press `*` to tag the highlighted model as a favorite; favorites are pinned to the top of the list, marked with a star, and persist across sessions, which makes hopping between two or three models during a session a two-keystroke affair. `^t` toggles a filter that hides models known not to support tool calling (models that don't report either way stay visible), showing how many it is hiding while active, and `^r` refetches the catalog.

For HuggingFace on the public endpoint the picker becomes a small model explorer: it lists the models currently served by inference providers (fetched from the router, no token needed), aggregates context and pricing across providers, starts with the tools filter enabled, and shows the per-provider breakdown for the highlighted model. `^s` searches the HuggingFace hub for provider-served chat models beyond the router's list.

`/model NAME` switches directly. Exact ids win; fuzzy matches prefer your favorites and recents over catalog entries, and TAB completes from favorites, recents, and the catalog entries already fetched this session (completion never hits the network). An ambiguous name reopens the picker pre-filtered (or lists the candidates in one-shot mode). If a HuggingFace model exists on the hub but no inference provider currently serves it, the error says which deployments exist and suggests close matches that are actually available. `/model -` reverts to the model you were using before the last switch. `/model --fav [ID]` toggles favorite status for the current (or given) model from the command line.

Switches are transactional: the new model is fully resolved (including context-window discovery) before anything changes, so a failed switch leaves the session exactly as it was. A model that reports no tool-calling support switches with a warning, since the agent loop already falls back to plain chat when a model rejects tools.

Catalogs are available for LM Studio (all downloaded models, loaded ones first), llama.cpp, OpenAI-compatible servers, Apple Foundation Models, HuggingFace, OpenRouter, Google Gemini, and ChatGPT. For bedrock, GEAP, and the command provider there is no listing, but `/model ID` still works. Favorites and recents live in `$XDG_CONFIG_HOME/swival/models.toml` (usually `~/.config/swival/models.toml`), a plain TOML file that is safe to edit by hand.

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

`--provider` chooses the backend provider and defaults to `lmstudio`. Valid values are `lmstudio`, `llamacpp`, `huggingface`, `openrouter`, `generic`, `applefm` (experimental Apple Foundation Models local server; defaults to `--model pcc`), `google`, `geap` (Gemini Enterprise / Vertex AI; `vertexai` is an accepted alias), `chatgpt` (for ChatGPT Plus/Pro subscriptions), `bedrock` (AWS Bedrock), and `command` (shells out to an external program).

`--model` overrides auto-discovery with a fixed model identifier.

`--base-url` sets a custom API base URL. For LM Studio, the default is `http://127.0.0.1:1234`; for llamacpp, the default is `http://127.0.0.1:8080`.

`--api-key` provides a key directly on the command line and takes precedence over provider environment variables (`HF_TOKEN` for huggingface, `OPENROUTER_API_KEY` for openrouter, `OPENAI_API_KEY` for generic, `GEMINI_API_KEY` for google, `CHATGPT_API_KEY` for chatgpt). Bedrock uses the AWS credential chain instead (`AWS_PROFILE`, env vars, or IAM roles). Local providers (`lmstudio`, `llamacpp`) need no key.

`--user-agent` sets the `User-Agent` header sent with LLM API requests. Defaults to `Swival/<version>`. Useful when a provider requires a specific agent string (e.g. `--user-agent 'KimiCLI/Swival'` for Kimi's API).

`--aws-profile` selects a named AWS profile from `~/.aws/config` for the `bedrock` provider. Overrides the `AWS_PROFILE` environment variable.

When `--profile` is combined with explicit flags like `--provider` or `--model`, the explicit flags win on a per-key basis.

### Behavior Tuning Flags

`--max-turns` sets the maximum number of loop iterations and defaults to `100`.

`--max-output-tokens` sets the model output budget per call and defaults to `32768`.

`--max-output-lines` sets the default number of lines a file read returns and defaults to `2000`. Reads can still paginate past it with explicit offset and limit.

`--max-output-kb` caps the size of tool output sent to the model — file reads, grep, listings, outline, and fetched URLs — and defaults to `50`. Larger results are truncated to fit.

`--max-context-tokens` requests a context window size. When omitted, Swival auto-detects it from the model server (or LiteLLM's registry for hosted providers); with LM Studio, passing it may trigger a model reload to the requested size. When the window is known, each turn header shows usage against it, e.g. `Turn 3/100 (~4,200 / 131,072 tokens, 3%)`.

`--temperature` controls sampling temperature and defaults to the provider default when omitted.

`--top-p` controls nucleus sampling and defaults to the provider default when omitted.

`--seed` passes a random seed for providers that support reproducible sampling.

`--extra-body JSON` passes extra parameters to the LLM API call. The value must be a JSON object. This is useful for provider-specific or model-specific options that Swival does not expose as dedicated flags.

```sh
swival --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}' "task"
```

`--retries N` sets the maximum number of provider retries on transient network errors and defaults to `5`. Set to `1` to disable retries entirely.

`--reasoning-effort LEVEL` sets the reasoning effort for models that support tunable reasoning (e.g. gpt-5.5). Valid levels are `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, and `default`.

```sh
swival --provider chatgpt --model gpt-5.5 --reasoning-effort high "task"
```

`--show-thinking` keeps the model's streamed thinking in your terminal scrollback after the answer is printed, instead of wiping it when the live stream ends. It only affects the display (the thinking goes to stderr, never into stdout, history, traces, or reports) and it needs a verbose, interactive terminal with a provider that streams reasoning. Without it, a collapsed one-line note (`thinking: N lines / ~M tokens, hidden`) is shown instead. The config-file equivalent is `show_thinking = true`.

### Service Tiers

**Priority processing** gives significantly lower and more consistent latency compared to standard processing, while keeping the same model quality and reasoning effort. It costs more (about 2× standard per-token pricing). Use it for interactive, latency-sensitive workflows where you want faster responses without sacrificing output quality.

**Flex processing** is the opposite — cheaper but with variable latency, as requests may be deferred when capacity is low. Use it for batch jobs, evals, or other asynchronous workloads.

Service tier is not exposed as a dedicated CLI flag. Pass it via `--extra-body`:

```sh
# Priority ("fast" mode — like Codex's fast setting)
swival --provider chatgpt --model gpt-5.5 \
    --extra-body '{"service_tier": "priority"}' "task"
```

In config:

```toml
[chatgpt]
model = "gpt-5.5"
extra_body = { service_tier = "priority" }
```

`reasoning_effort` and `service_tier` are independent knobs — you can combine them to tune both how deeply the model thinks and how fast the response arrives:

```sh
# Fast response + moderate reasoning
swival --provider chatgpt --model gpt-5.5 \
    --reasoning-effort medium \
    --extra-body '{"service_tier": "priority"}' "task"
```

Not all models support service tiers. Check the [OpenAI pricing page](https://platform.openai.com/api/pricing) for the current list of supported models. The available tier values are `"default"` (standard), `"priority"`, and `"flex"`.

`--proactive-summaries` enables periodic checkpoint summarization of the conversation. Every ten turns, recent turns are summarized and stored internally. These summaries survive context compaction and give the model a condensed record of earlier work that would otherwise be lost.

Useful for long-running sessions.

### Sandboxing Flags

`--base-dir` defines the base directory for file tools. When omitted, Swival walks up from the current directory looking for a `.git` directory or a `swival.toml` file, and uses the first match it finds; if neither is found, it falls back to the current directory.

`--files` controls filesystem tool access. Accepts `"some"` (the default, workspace only), `"all"` (unrestricted), or `"none"` (`.swival/` directory only). With `--files none`, the agent can still run commands and fetch URLs but cannot read or write project files.

`--commands` controls command execution. Accepts `"all"` (unrestricted, the default), `"none"` (disabled), `"ask"` (interactive approval per command bucket), or a comma-separated whitelist. With `--commands all` or `--yolo`, both `run_command` and `run_shell_command` are exposed. In ask and whitelist modes, only `run_command` is available.

`--add-dir` grants read and write access to additional directories and can be repeated.

`--add-dir-ro` grants read-only access to additional directories and can be repeated. The agent can read, list, and grep files in these directories but cannot write, edit, or delete them.

`--oneshot-commands` enables `/` and `!` command dispatch in one-shot mode. By default, command-like input in one-shot mode is treated as plain text because the input may come from an untrusted source. This flag is a no-op in interactive mode, where commands are always available. Can also be set in config as `oneshot_commands = true`.

`--yolo` is shorthand for `--files all --commands all`. It does not imply `--oneshot-commands`. Explicit `--files` or `--commands` flags take precedence over `--yolo`. Filesystem root access is always blocked.

`--network` sets the network policy for the run. The default, `full`, changes nothing. `provider-only` keeps the model provider reachable (hosted APIs and local HTTP servers both work) while every agent-initiated subprocess — shell commands, the `run_python` tool, stdio MCP servers, and everything they spawn — runs inside a `nono run --block-net` wrapper with no network access; `fetch_url` is removed as well. `none` goes further: the whole Swival process tree is re-executed under `nono --block-net`, nothing can make an outbound connection (loopback included), and only the `command` provider works. Both restricted modes require the `nono` binary and reject URL-backed MCP servers and A2A clients. Can also be set in config as `network = "provider-only"`; an explicit `--network full` on the command line overrides an inherited value. See [Safety and Sandboxing](safety-and-sandboxing.md) for the exact guarantees.

`--sandbox` selects the sandbox backend. The default is `builtin`, which uses application-layer path guards. Set it to `agentfs` to re-exec Swival inside an [AgentFS](agentfs.md) overlay for OS-enforced write isolation (requires the `agentfs` binary on PATH), or `nono` for capability-based OS sandboxing with Landlock or Seatbelt, network filtering, and optional rollback. The `nono` backend has its own family of flags (`--nono-profile`, `--nono-block-net`, `--nono-allow-domain`, `--nono-rollback`, and more); see [nono](nono.md) for the full list.

`--sandbox-session` sets an AgentFS session ID so sandbox state persists across runs. Only valid with `--sandbox agentfs`.

`--sandbox-strict-read` enables strict read isolation inside the AgentFS sandbox. When set, the agent process can only read files that have been explicitly allowed, rather than having unrestricted read access to the host filesystem.

This requires an AgentFS version with strict read support. If the installed version does not support it, Swival exits with an error. Only valid with `--sandbox agentfs`.

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

`--metaskills POLICY` sets the metaskill execution policy. Valid policies are `local` (default; only metaskills shipped inside the workspace can run), `all` (any discovered metaskill can run, including those in external skill directories), and `off` (metaskill execution is disabled, though static skills remain available). See [Metaskills](metaskills.md) for what metaskills are.

`--no-metaskills` is shorthand for `--metaskills off`: skills remain discoverable as static prompts, but no `SKILL.star` program is executed.

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

### ACP Flags

`--acp` speaks the Agent Client Protocol on stdio, for editor integration (Zed, agent-client-protocol.nvim, and others). Incompatible with `--serve`, `--repl`, and a positional task. See [ACP](acp.md).

`--acp-log PATH` logs the JSON-RPC traffic and diagnostics to `PATH`. Only used with `--acp`.

### Output And Reporting Flags

`--quiet` and `-q` suppress diagnostics and keep terminal output focused on final answers.

`--report FILE` writes a JSON run report to `FILE`. In one-shot mode, requires a task. In REPL mode, the report covers the entire session and is written when the session ends.

`--trace-dir DIR` writes a HuggingFace-compatible JSONL session trace to `DIR`. Each session produces a `<session_id>.jsonl` file that HuggingFace auto-detects as `format:agent-traces` with harness `swival`. Works in both one-shot and REPL modes. See [Reports](reports.md) for details.

`--reviewer COMMAND` runs an external reviewer after each answer. The command string is shell-split, so you can pass arguments inline (e.g. `--reviewer "swival --reviewer-mode"`). Requires a task; incompatible with `--repl`.

`--max-review-rounds N` limits how many times the reviewer can request retries and defaults to `15`. Set to `0` to accept the first answer without retries.

`--self-review` uses a second Swival instance as reviewer, inheriting provider, model, skills-dir, and files settings from the current invocation. Requires a task; incompatible with `--repl` and `--reviewer`. See [Reviews](reviews.md) for details.

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

These setup flags do not require a question argument. Config generation never overwrites an existing config file: if one exists, the template is written to a `.new` file next to it, with your current settings preserved, so you can review it before moving it into place.

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

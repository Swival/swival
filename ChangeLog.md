# Changelog

All notable user-facing changes to Swival.

## 1.0.13

- A goal-driven mode has been added: a structured spin on the Ralph-style "keep prompting until it's done" loop. Set an objective with `/goal <objective>` in the REPL and the agent doesn't get to declare victory and walk away after one turn. The original objective is fed back to the model after every answer, and the loop only ends when the agent itself signals the goal is complete after a real evidence-based audit, declares a blocker, or hits the optional token budget. This makes it practical to point Swival at ambitious, long-running tasks like refactors, audits, or end-to-end fixes, and let it grind for hours without giving up halfway. `/goal pause`, `/goal resume`, `/goal replace`, and `/goal clear` give you full control.
- First-run setup now writes a `[profiles.default]` block to the generated config, so the freshly created file lines up with the profile structure used everywhere else.
- The history file is automatically trimmed when it grows past its maximum capacity.

## 1.0.12

- The system prompt has been optimized for efficiency, and small models may enjoy a significant reduction in token usage.

## 1.0.11

- `/audit` got more constraints to focus more on security issues.
- `/audit` compatibility with models such as Xiaomi MiMo was also improved.

## 1.0.10

- `/audit` now accepts multiple focus paths in a single invocation (`/audit src/auth/ src/api/`).
- Other minor improvements to `/audit` to reduce false positives while exploring more bug classes.

## 1.0.9

- `--logout` has been added to delete locally cached ChatGPT OAuth tokens and exit, so users can sign out without hand-deleting files under `~/.config/litellm/chatgpt/`.
- `/audit` no longer asks the LLM for JSON. Intermediate phase responses now use a simple structured-text format (`@@ name @@` blocks with `key: value` lines), which models emit far more reliably across long prompts than nested JSON.
- `/audit` phase 1 (file profiling) is now dramatically faster on large repositories. File contents are read through a single `git cat-file --batch` process instead of one subprocess per file, cutting the per-file overhead by an order of magnitude on multi-thousand-file scans.
- A `--debug` option has been added to `/audit`. When enabled, a real-time JSONL log is written to `.swival/audit/debug.jsonl` capturing every LLM request and response, parse outcomes, repair attempts, and per-phase metrics, which makes it tractable to diagnose model misbehavior on large audits.
- Another `/audit` improvement: it is now considerably more verbose during phase 3, surfacing per-file progress instead of presenting one long silent batch.
- Phase 5 audit reports no longer occasionally contain raw tool-call JSON (`{"cmd": "ls"}`) or conversational preamble like "I'll inspect the patch...".
- `/audit` prompt cache hit rates have been improved: the bug-class taxonomy and finding metadata interpolated into phase 3 system prompts have been moved into user messages so the system prefix stays static across calls, and per-phase cache statistics are now logged when `--debug` is on.

## 1.0.8

- GPT-5.5 is now recognized by the ChatGPT provider. Older LiteLLM releases that don't yet know about the model are patched at runtime so context-length queries and Responses API routing work out of the box.

## 1.0.7

- Emergency truncation has been added as a last-resort compaction stage.
- Prompt caching now works for tool-less LLM calls such as `/audit`. Previously, cache control breakpoints were only injected when tool schemas were present.
- `/audit` Phase 2 triage now places the repository profile in the system prompt instead of repeating it in every user message, improving prompt cache hit rates and reducing costs.
- `/audit` Phase 3b finding expansions now run sequentially with per-item error handling instead of in parallel, so a single failed expansion no longer kills the entire batch.
- D language (`.d`) files are now recognized as source code by `/audit`.
- LiteLLM has been updated to add support for the Mythos provider.

## 1.0.6

- `top_p` is no longer sent to the provider by default, letting each provider use its own default. The `--top-p` flag is still available to override it explicitly.
- A `--user-agent` option has been added to set a custom `User-Agent` header on LLM API requests. The generic and llama.cpp providers now send `Swival/<version>` by default, and OpenRouter forwards the header when set. This can also be configured via `user_agent` in config files.
- `/audit` path scoping no longer silently skips the target directory when the argument is missing a trailing slash.
- Provider-specific workarounds have been added for Kimi K2.6.

## 1.0.5

- When a file is too large for the LLM's context window during an audit, the audit now progressively truncates it and retries instead of failing outright.
- Audit LLM calls no longer force a fixed temperature and top_p, letting providers that reject custom sampling parameters (such as Anthropic) work without errors.

## 1.0.4

- `/audit` can now be used in one-shot mode, not just the REPL.
- Security audit LLM calls now retry automatically on transient failures (rate limits, timeouts, server errors) with exponential backoff.
- Audit patch generation no longer crashes on files containing non-UTF-8 bytes.

## 1.0.3

- A built-in `/audit` command has been added for deep security audits over committed Git-tracked code. It scans source and config files for vulnerabilities using the session's LLM, produces a structured report with severity ratings, and can optionally generate a patch. Supports Python, JavaScript, TypeScript, Go, Rust, C/C++, Zig, and many other languages.

## 1.0.2

- Subagents now inherit the parent session's proactive context compaction setting, so long-running subagent tasks get the same graduated summarization as the main loop.
- When a subagent hits a context overflow, it now recovers partial results from the last real assistant message instead of failing outright. Recap-only messages are skipped so the recovered text reflects actual work.

## 1.0.1

- Proactive context compaction is now enabled in subagents, giving them the same graduated summarization as the main loop.
- `"context size exceeded"` errors from llama.cpp are now recognized as context overflow, triggering compaction instead of failing the turn.

## 1.0.0

- Hugging Face models that don't support chat completions now fall back to plain text generation in non-tool turns, and models that exist on the Hub but have no live Inference Provider deployment now fail with a clearer error explaining how to run them instead.

## 0.11.3

- All the underscore-prefixed internal keys are now stripped from outbound messages.

## 0.11.2

- Added a quick shell command (`!!`) to the REPL, allowing users to run shell commands without LLM involvement.
- Added an inline `@` trigger for tab-completing file paths mid-prompt in the REPL.
- Fixed Gemini 3 multi-turn tool calling failures by preserving `thought_signature` in current-turn tool calls.
- Custom commands (`!`) now support inlining the content of non-executable text files directly into the prompt.
- JSONL traces now use relative workspace paths instead of absolute paths to reduce sensitive leakage.

## 0.11.1

- `fetch_url` now allows connecting to `localhost`, `127.0.0.1`, and `::1`. Agents frequently run a local server and then need to test or inspect it, and the previous blanket loopback block made that workflow awkward. Other private, link-local, and reserved addresses are still blocked.
- MCP tool names are now stored separately from the tool schema rather than as an internal `_mcp_original_name` field. This fixes Gemini rejecting MCP tool schemas that contained an unrecognized property.

## 0.11.0

- `--command-middleware` adds a hook point before every `run_command` and `run_shell_command` call. The middleware receives a JSON payload on stdin and can pass the command through unchanged, rewrite it, or block it with a reason. Rewritten commands are still validated against Swival's own command policy so the middleware cannot bypass allowlists or `--commands none`.
- When command, MCP, or A2A output exceeds the inline limit and spills to a temp file, the first 50 lines (up to 2 KB) are now included directly in the tool result. The model can usually continue without a follow-up `read_file` call.
- `--report` now works in REPL mode and produces a full-session report on exit.
- HuggingFace-compatible agent trace export (`format:agent-traces`) has been implemented.
- `AGENTS.md` files are loaded from all ancestor directories up to the project root, not just the project root itself.
- Custom commands whose name contains a slash are now resolved relative to the config directory, making it easier to organize commands in subdirectories.

## 0.10.14

- Slash commands (`/`) and custom commands (`!`) can now be used in one-shot (non-REPL) mode. Because one-shot input may come from untrusted sources, command dispatch is disabled by default; pass `--oneshot-commands` to opt in.
- Skill directory scanning depth has been reduced from 5 to 3 to avoid descending into vendored or generated trees.

## 0.10.13

- Swival now auto-detects the project root by walking up to the nearest `.git` directory or `swival.toml`, so launching from a subdirectory keeps file tools and project-scoped behavior anchored to the repository root.
- `edit_file` now accepts an optional `line_number` parameter so targeted replacements can disambiguate repeated matches using the line numbers returned by `read_file`.
- ChatGPT provider handling now tolerates empty Responses API payloads instead of failing the turn.
- `LITELLM_LOCAL_MODEL_COST_MAP` is now enabled unconditionally to avoid unnecessary remote pricing lookups for local-model providers.

## 0.10.12

- Add native support for llama.cpp

## 0.10.11

- Shell-command execution is now only exposed in unrestricted command modes:
  `run_shell_command` is hidden outside `--commands all` / `--yolo`, while
  `run_command` remains available for argv-style execution in `--commands ask`
  and allowlist modes.
- Profiles that omit `max_output_tokens` no longer crash or override provider
  defaults. Swival now preserves an unset output cap instead of substituting a
  large context-derived value.

## 0.10.10

- Swival now automatically falls back to plain chat when a provider or model
  does not support function calling, including OpenRouter's tool-unsupported
  responses.
- Command execution has been split into two tools: `run_command` now takes an
  argv array, while `run_shell_command` takes a shell string and is only
  exposed in unrestricted command modes. This avoids the old union-type schema
  that weaker models often mangled.
- Tool-call repair has been tightened for small models, making malformed
  arguments more likely to be repaired into valid tool calls.

## 0.10.9

- REPL `/profile` switching now correctly inherits top-level config values:
  profiles that omit keys like `api_key` pick them up from the config file
  rather than from the previously active profile.
- Malformed tool-call repair now handles file path parameters: glob
  metacharacters (`*`, `?`, `[]`) are stripped from path and directory fields
  whose schema description does not indicate a glob or pattern value, and
  common field-name aliases (`path`, `file`, `filename`) are mapped to the
  correct schema name. That helps small models.

## 0.10.8

- A new `/profile` REPL command can list available profiles, switch to a different
  LLM profile mid-session, and revert to the startup profile (or baseline config)
  with `/profile -`. `/status` now shows the active profile.
- TAB completion has been added to the REPL for slash commands, custom
  `!commands`, directory-path arguments for `/add-dir` and `/add-dir-ro`, and
  `$skill` mentions.
- `/init` now includes commit and pull request style guidance in generated
  `AGENTS.md` files, derived from recent git history and any PR template.

## 0.10.7

- Interactive command approval mode has been added: `--commands ask` prompts the
  user before every shell command execution. Approvals can be scoped per command
  bucket and persisted to `.swival/approved_buckets`, denied, or allowed once.
  High-risk commands and inline code execution (`bash -c`, `python -c`,
  `node -e`, etc.) are flagged with extra warnings.
- Untrusted external content labeling has been added: output from `fetch_url`,
  MCP servers, and A2A agents is now wrapped with a deterministic
  `[UNTRUSTED EXTERNAL CONTENT]` header before the model sees it, instructing
  the model to treat it as data only. The label is baked into spill files so it
  survives later `read_file` access.
- JSON reports now include a `security` section that tracks command policy
  blocks, approvals, and untrusted input ingestion events.
- Bedrock provider now forwards the AWS profile to the reviewer session.

## 0.10.6

- Special tokens in user, system, and tool messages are now escaped by inserting
  zero-width spaces at token boundaries, preventing the tokenizer from
  misinterpreting literal text as control tokens.
- Tool descriptions have been removed from the system prompt, freeing up context
  space (models already receive tool schemas via the function-calling API).
- Internal litellm fields and `reasoning_content` are now stripped from assistant
  messages before they are sent back to the provider, fixing compatibility
  issues.

## 0.10.5

- `/status` REPL command has been added to show the current session state
  (provider, model, profile, token usage, active tools, and configuration).
- Bedrock provider now suggests the `aws sso login` command when authentication
  fails.
- LM Studio provider now sets `LITELLM_LOCAL_MODEL_COST_MAP` to avoid
  unnecessary remote lookups for model pricing.

## 0.10.4

- Onboarding has been improved.
- Subagents are now auto-enabled when the context window is 100K tokens or
  larger.

## 0.10.3

- An interactive onboarding wizard has been added: on first run with no config
  file, Swival guides the user through provider selection, API key entry, and
  config file creation. Re-running onboarding merges new provider settings into
  an existing config file instead of overwriting it.
- Common malformed tool calls from weaker models are now automatically repaired
  before reaching dispatch: orphaned tool-call references, missing required
  fields, and broken JSON are patched up so the agent loop can continue.

## 0.10.2

- Named LLM profiles have been added: `[profiles.NAME]` tables can be defined
  in config files to bundle provider, model, API key, and other LLM settings
  under a short name. Use `--profile NAME` to select one at runtime, or set
  `active_profile` in config for a default. `--list-profiles` prints all
  available profiles.
- Provider error messages now include the model ID for easier debugging.
- Minimax-specific transient errors are now caught and retried.

## 0.10.1

- Filesystem access controls have been decoupled from `--yolo`:
  `--files` (`all`, `some`, `none`) controls file access independently, and
  `--commands` (`all`, `none`, or a comma-separated whitelist) controls which
  shell commands the agent may run. `--yolo` is now shorthand for
  `--files all --commands all`.
- AWS Bedrock has been added as a provider.
- `/simplify` REPL command has been added: runs a review pass over recently
  changed code, checking for reuse opportunities, quality issues, and
  inefficiencies, then fixes any problems found.
- REPL answers are now rendered as Markdown on TTYs.
- Project-level MCP configuration has been moved from `.mcp.json` to
  `.swival/mcp.json`.

## 0.9.7

- Parallel subagents have been added: `spawn_subagent` launches an independent
  agent loop in a background thread to work on a task concurrently, and
  `check_subagents` polls, collects results, or cancels running subagents.
  Up to 4 subagents can run in parallel. Each gets its own thinking, todo,
  snapshot, and file-tracker state. Subagents have access to all file and
  search tools but cannot spawn their own subagents.
- The todo list is now session-scoped and purely in-memory. It no longer
  persists to `.swival/todo.md` or uses file locking. Concurrent sessions
  get fully independent todo lists with no cross-session interference.
- `/remember <text>` REPL command has been added to persist a project fact
  to `AGENTS.md` under `## Conventions`. The live system prompt is updated
  immediately so the agent sees the new fact without restarting.
- `read_file` on a missing `MEMORY.md` now returns a helpful hint explaining
  its purpose instead of a generic "file not found" error.

## 0.9.6

- Prompt caching has been added. When a provider supports it, the system
  prompt is cached on the first request and reused for subsequent calls,
  reducing costs and latency. Can be disabled with `--no-cache-prompts`.

## 0.9.5

- `outline` tool has been added: shows the structural skeleton of one or more
  files (classes, functions, top-level declarations) with line numbers, without
  bodies. Useful for navigating unfamiliar code.

## 0.9.4

- `/copy` REPL command has been added to copy the last assistant response to
  the clipboard.
- When using LM Studio, the max context length is now always queried from the
  server instead of relying on a hardcoded default.

## 0.9.3

- When Swival is launched on a TTY with no task, it now enters REPL mode
  directly.
- Filesystem built-in tools now expand `~` in paths, so home-directory paths work
  consistently across file reads, writes, edits, deletes, and searches.
- The fetch_url tool has a higher probability to get used consistently
  by small models.

## 0.9.2

- Homebrew installation support has been added.
- `Session.ask()` now rolls back conversation history on failure, so a failed turn
  doesn't corrupt a long-lived Python session.
- Public Python API exceptions have been formalized: `ContextOverflowError` and
  `LifecycleError` are now exported
- The persistent todo list is now safer across concurrent sessions and processes:
  writes use file locking and merge on-disk changes instead of clobbering them.
- SIGTERM now shuts Swival down cleanly with exit code 143, preserves
  continue-here state, and closes MCP/A2A managers during teardown.

## 0.9.1

- Generic lifecycle hooks have been added: user-configured commands run at
  startup and exit, with Git and project metadata passed via `SWIVAL_*`
  environment variables. Startup hooks run before memory and continue-here
  loading so they can hydrate `.swival/` from remote storage; exit hooks run
  after all artifacts are written. Configurable via `swival.toml` or
  `~/.config/swival/config.toml`.
- Custom command arguments are now passed as a single string: `!command a b c`
  calls the script with `$2="a b c"` instead of spreading each word as a
  separate argv entry.

## 0.9.0

- Outbound LLM filter: a new `--llm-filter` flag (and `llm_filter` config key)
  runs a user-defined script before every provider call. The script receives
  messages as JSON on stdin and can redact content or block the request entirely.
  Fails closed — script errors or rejections prevent the request from being sent.
  Runs before secret encryption so filters see human-readable text. Configurable
  from CLI, `swival.toml`, or `~/.config/swival/config.toml`.

## 0.1.36

- Custom commands have been added: executable scripts placed in
  `~/.config/swival/commands/` can be invoked from the REPL with `!name`,
  and their output is injected into the conversation as the next user message.

## 0.1.35

- `/init` workflow discovery is now platform-aware: it detects the current OS
  and architecture and only extracts commands that apply to the host platform.

## 0.1.34

- `/init` now discovers workflow files and validates the generated instructions
  by writing them out and checking the result.
- Transient LLM errors (rate limits, timeouts, server errors) are now retried
  automatically with exponential backoff.
- An interaction-policy system prompt has been added to distinguish REPL and
  autonomous modes, giving the model clearer behavioral guidance for each.

## 0.1.33

- Updated ChangeLog

## 0.1.32

- Last-resort compaction has been added: when the context window is too small for
  tool schemas, all tool definitions are dropped and the system prompt is truncated
  so the conversation can continue as plain chat.
- Command provider now supports tool calling via a `<swival:call>` XML convention,
  allowing external command-based backends to invoke tools.
- Data-URI inlined images are now stripped after HTML-to-markdown conversion to
  avoid bloating context with base64 blobs.
- Markdown comments (`<!-- ... -->`) are now trimmed from skill and agent
  instruction files.
- OpenRouter requests now include `referer` and `title` headers.

## 0.1.31

- The `grep` tool now supports a `context_lines` parameter to show surrounding
  lines before and after each match.
- `/new` has been added as a synonym for `/clear` in the REPL.
- `reasoning_effort` set to `"default"` is now skipped instead of being sent to
  the provider.

## 0.1.30

- Secrets encryption has been added: credential tokens in LLM messages
  can be transparently encrypted before being sent to the provider and decrypted
  on return, preventing accidental leakage through hosted APIs.
- The `--sanitize-thinking` CLI flag has been fixed (it was accepted but ignored
  in 0.1.29).
- `read_multiple_files` now accepts a plain string in addition to an array,
  for resilience with models that pass a single filename as a string.

## 0.1.29

- Command provider has been added for shelling out to external programs as the
  LLM backend: the conversation is passed as a plain-text transcript on stdin,
  and the response is read from stdout.
- Leaked reasoning tags (`<think>`, `</think>`) from models with bogus
  templates can now be stripped. This can be controlled with `sanitize_thinking`
  in config or `--sanitize-thinking`.
- Race conditions when multiple A2A contexts run concurrently have been fixed by
  isolating per-context temporary files (cmd_output) and adding file locks.
- SQLite cross-thread error when `--serve` and `--cache` are combined has been
  fixed.

## 0.1.28

- Support for vision has been added: a new `view_image` tool allows the
agent use vision-enabled models to examine images.
- Skill scanning now skips dot directories.

## 0.1.27

- Skills can now be loaded from `.agents/skills/` and `~/.agents/skills/` directories.
- Global agent instructions via `~/.agents/AGENTS.md` have been added.
- Documentation has been improved with web browsing options, lightpanda MCP server
  usage, and chrome-devtools-mcp examples.

## 0.1.26

- Google Gemini provider has been switched to use the OpenAI-compatible endpoint.
- Built-in help output has been grouped by purpose.
- Documentation and examples have been improved.

## 0.1.25

- Native Google Gemini API support has been added.
- A2A streaming (`SendStreamingMessage`) has been added: real-time SSE delivery of
  status updates, tool lifecycle events, and incremental text.
- `CancelTask` support has been added: per-task cancel flags are checked between
  tool calls and at each turn boundary.
- A2A server hardening has been added: sliding-window rate limiting, request size
  validation, concurrency semaphore, and active-context protection against
  LRU eviction.
- Read access to external skill directories has been auto-granted and supporting
  files are now listed on skill activation.

## 0.1.24

- A2A server mode (`--serve`) has been added: a swival Session can be exposed as
  an A2A endpoint, with context-keyed multi-turn sessions, bearer auth, and
  TTL-based cleanup.
- Customizable A2A server agent card has been added: `--serve-name`,
  `--serve-description`, and `[[serve_skills]]` in `swival.toml` control how the
  agent advertises itself.
- `/tools` REPL command has been added to list available tools.

## 0.1.23

- A2A (Agent-to-Agent) support has been added: remote agents can be connected via
  `[a2a_servers.*]` in `swival.toml` or `--a2a-config`, with tools exposed as
  `a2a__<agent>__<skill>`.
- Budgeted memory injection has been added. `--memory-full` can be used for legacy
  full injection.
- Support for reading questions from stdin when piped has been added.

## 0.1.22

- `--self-review` option has been added: the agent reviews its own work before
  finishing.
- Reviewer feedback visibility has been improved and expected actions have been
  made more explicit.
- Informational stderr from the reviewer is now shown as warnings instead of being
  silently discarded.
- The default number of review rounds has been bumped up to 15.
- A cache miss cascade caused by dropped `tool_call` fields in cached responses
  has been fixed.

## 0.1.21

- Optional SQLite LLM response cache (`--cache`) has been added for faster
  repeated queries, with system-prompt-independent cache keys.
- A deadlock when a shell command backgrounds a child process has been fixed.
- The `todo` tool accepting JSON-encoded array strings instead of proper lists
  has been fixed.

## 0.1.20

- The project-local skills directory has been moved from `skills/` to
  `.swival/skills/`.
- Spurious "shadowed by itself" warnings when `--skills-dir` pointed to the same
  directory as the project-local skills location have been fixed.
- `$skill-name` mention syntax has been added: `$deploy` can be typed in a message
  to automatically activate a skill without the model needing to call `use_skill`.
- The skill catalog in the system prompt has been reworked with file paths, trigger
  rules, and progressive disclosure guidance.
- Auto-injected skills now use assistant+tool message pairs so compaction can
  shrink or drop them under context pressure.
- Auto-activated skills are now recorded in JSON reports.

## 0.1.19

- `/learn` command has been added for interactive skill discovery.

## 0.1.18

- `read_multiple_files` tool has been added for reading several files in a single
  call.
- Continue-here feature has been added: session state is saved on interruption
  (Ctrl+C, max turns, compaction failure) and resumed on next start.
- The `todo` tool has been made to accept multiple tasks in one call.
- The `grep` tool has been extended with additional options.
- Context overflow detection for non-standard exception types has been fixed.

## 0.1.17

- `--reasoning-effort` option has been added.
- Session memories that persist across runs have been added.
- GPT-5.4 has been added to the built-in model list.
- Markdown formatting for agent responses has been added.
- Spinner and progress display have been improved.
- Todo list UI has been improved.
- All CLI options have been listed in `--help` and sorted alphabetically.

## 0.1.16

- Colored diff output has been added to the `edit_file` tool.

## 0.1.15

- `write_file` has been made to coerce JSON content into a string instead of
  erroring.

## 0.1.14

- ChatGPT has been added as a provider (direct OpenAI API).

## 0.1.13

- AgentFS sandbox support has been integrated with auto-session IDs, diff hints,
  and strict read mode.
- "Did you mean?" suggestions for mistyped tool command names have been added.
- MCP servers have been made to inherit the parent process environment variables.

## 0.1.12

- Generic OpenAI-compatible provider has been added for any server that speaks the
  OpenAI API.
- Snapshot tool has been added for proactive context collapse, with `/snapshot` and
  `/restore` REPL commands.
- `--extra-body` option has been added to pass arbitrary JSON to the LLM request
  (useful for disabling thinking, etc.).
- OpenRouter documentation and setup instructions have been added.

## 0.1.11

- MCP (Model Context Protocol) server support has been added. Servers are
  configured in `swival.toml` or `.mcp.json`; tools are exposed as
  `mcp__<server>__<tool>`.
- Configurable size limits for MCP tool output (`MCP_INLINE_LIMIT`,
  `MCP_FILE_LIMIT`) have been added.

## 0.1.10

- Reviewer mode (`--reviewer-mode`) has been added: an LLM-as-judge loop that
  automatically evaluates agent output, with `--objective`, `--verify`,
  and `--review-prompt` options.
- `--max-review-rounds` has been added to cap review iterations.

## 0.1.9

- Graduated context compaction has been introduced: `compact_messages` ->
  `drop_middle_turns` -> `aggressive_drop_turns`, replacing the previous
  all-or-nothing approach.
- `/continue` is now suggested when the agent hits the max turn limit.
- Clamping and retry messages have been improved.

## 0.1.8

- `grep` and `list_files` tools have been made to accept file paths in addition to
  directories.
- `grep` tool output has been improved.
- Whether the model supports vision is now reported.
- Global instructions via `~/.config/swival/AGENTS.md` have been added.
- `--no-instructions` behavior has been clarified.

## 0.1.7

- Configuration file support (`swival.toml` and `~/.config/swival/config.toml`)
  has been added.
- `--add-dir-ro` has been added for read-only additional directories (renamed from
  `--allow-dir`).
- Common command syntax mistakes in yolo mode are now auto-corrected.
- Instructions file has been switched from `ZOK.md` to `AGENT.md`.

## 0.1.6

- `think` tool has been redesigned with numbered thoughts, revisions, and branches.
- CI pipeline has been added.
- `Makefile` with common development commands has been added.
- Trash/undo handling has been fixed.
- Error when the model sends a file size with units has been improved.

## 0.1.5

- `todo` tool has been added: a persistent checklist in `.swival/todo.md` that
  survives context compaction, with periodic reminders and duplicate detection.
- `/init` command has been added for bootstrapping `AGENT.md`.
- A public Python API (`swival.Session`, `swival.run()`) has been exposed.
- A loading spinner during LLM calls has been added.
- The unused `notes` tool has been removed.

## 0.1.4

- OpenRouter has been added as a provider.
- `delete_file` tool has been added.
- `move_file` / `rename_file` tools have been added.
- External reviewer support for automated evaluation has been added.
- Read-before-write is now required: the agent must read a file before editing or
  overwriting it (can be disabled with `--no-read-guard`).
- Final output is now printed even when `--report` is enabled.
- Default values for `temperature` and `top_p` have been removed (the provider
  decides).

## 0.1.3

- Package has been renamed from `swival-agent` to `swival`.
- `--version` flag has been added.
- Recursive skill discovery has been deepened.
- Skill activation events have been included in reports.

## 0.1.2

- `--report` has been added for JSON session reports.
- `--history` has been added to replay previous sessions.
- Thinking tool has been revamped.
- Absolute paths in yolo mode have been allowed.
- Full shell expansion in yolo mode has been added.
- Default max turn limit has been increased.

## 0.1.1

- `--seed` option has been added for deterministic output.

## 0.1.0

Initial release. Core agent loop with tool-use, LM Studio and HuggingFace
providers, file read/write/edit, grep, list_files, run_command, thinking tool,
skills system, and REPL mode.

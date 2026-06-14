# Context Management

Most AI coding agents assume large context windows — 128K tokens or more. Swival is designed to work well with models that have less tokens to work with. Every token matters, so Swival manages context aggressively at every stage: preventing bloat before it happens, giving the agent tools to compress its own history, and recovering gracefully when the window fills up.

You don't need to configure any of this. It works out of the box. But understanding how it works helps you get the most out of small models and explains what's happening when you see compaction messages in the logs.

## The Four Layers

Swival's context management operates as four concentric layers of defense.

**Layer 1 — Prevention.** Tool output is capped before it enters the conversation. A file read that returns 200KB of content is truncated to 50KB. A grep that matches thousands of lines returns the first 100. Command output over 10KB is saved to a file and replaced with a pointer. MCP tool schemas that would consume too much of the window are dropped at startup. These limits keep junk out of the context so compaction rarely needs to fire.

**Layer 2 — Proactive collapse.** The agent can actively manage its own context using the `snapshot` tool. After reading several files to understand a problem, the agent calls `snapshot restore` with a summary of what it learned. The file reads — often 10K+ tokens of dead weight — are replaced with a ~200 token summary. The agent keeps the knowledge; the context gets the space back.

**Layer 3 — Automatic compaction.** Swival keeps the transcript under a safe token budget for you, through one budget-targeted entrypoint that both of its modes share. The common case is *proactive*: before each LLM call, if the prompt is over budget, Swival runs only the cheap, near-lossless rungs of its compaction ladder — garbage-collecting spent scaffolding, shrinking old tool results, and stripping replayed reasoning — so the request rarely reaches the provider's wall at all. A *reactive* net catches the rare overflow that still slips through when a provider's tokenizer counts more than Swival's estimate: it re-compacts below the current size with the full escalating ladder — dropping low-importance turns (the last few protected), dropping tools for a single retry, and finally a deterministic truncation floor — then retries. Temporary tool removal is request-local, so later turns get their tools back.

**Layer 4 — Knowledge survival.** Thinking notes, todo lists, and snapshot summaries live outside the message history in independent channels that compaction cannot touch. Even after the most aggressive compaction wipes nearly everything, the agent still knows its reasoning, its task list, and what it learned during investigation.

## Prevention: Output Size Guards

Every tool that returns content has a hard cap on how much it can put into the conversation.

- **File reads** are capped at 50KB, with individual lines truncated at 2,000 characters. Large files can be paginated with offset and limit parameters. The 50KB cap and the default 2000-line read limit are tunable with `max_output_kb` and `max_output_lines` (config keys or the matching CLI flags); the byte cap also applies to listings, grep, outline, and fetched URLs.
- **Directory listings** return at most 100 entries.
- **Grep results** return at most 100 matches.
- **Command output** is returned inline up to 10KB. Anything larger is written to a temporary file and the conversation receives a short pointer with instructions to paginate.
- **MCP tool results** follow the same pattern: inline up to 20KB, saved to file above that.
- **URL fetches** are capped at 5MB for the raw download and 50KB for the converted output.
- **Instruction files** (`CLAUDE.md`, `AGENTS.md`) are each capped at 10,000 characters.
- **Auto-memory** (`.swival/memory/MEMORY.md`) is injected through a budgeted two-part pipeline. Entries tagged with `<!-- bootstrap -->` are always included (up to 400 tokens). Remaining entries are ranked by BM25 relevance against the user's question and the top results are injected (up to 400 tokens). Total memory cost stays within 800 tokens even for large memory files. Use `--memory-full` to inject everything (legacy behavior).

These limits are deliberately conservative. They prevent a single tool call from consuming a significant fraction of a small context window.

### MCP Schema Budget

MCP servers expose external tools, but their schemas can be large. On a 16K context window, tool schemas alone could eat half the budget before the agent does anything.

At startup, Swival estimates the total token cost of all tool schemas. If they exceed 30% of the context window, it warns. If they exceed 50%, it starts dropping the most expensive MCP server's tools until the budget fits. This happens automatically — you don't need to manually curate which MCP tools are available.

## Proactive Collapse: The Snapshot Tool

The snapshot tool is the centerpiece of Swival's context management. It lets the agent compress its own investigation history on demand.

A typical workflow looks like this: the agent reads 8 files and greps through logs to debug an authentication failure. All those reads add up to maybe 12K tokens sitting in the conversation, content the agent has already processed and drawn conclusions from.

The agent calls `snapshot restore` with a summary like "Root cause: missing null check in auth/parser.py:142. Fix: guard with `if token is None: return default_token`." The 12K tokens of reads collapse to about 200 tokens. The agent continues with a clean context window.

### Save and Restore

The snapshot tool has four actions:

- **save** sets a named checkpoint before the agent starts a block of work.
- **restore** collapses everything since the checkpoint (or the last natural boundary) into a summary.
- **cancel** clears an active checkpoint without collapsing.
- **status** reports the current snapshot state.

If the agent calls `restore` without a prior `save`, it automatically finds the right starting point — usually the most recent user message or the boundary of a previous restore.

### Dirty Scope Protection

If the agent made changes (wrote files, ran commands) between save and restore, the scope is considered "dirty" and the agent must explicitly acknowledge this with `force=true`. Read-only operations like reading files, grepping, viewing images, and thinking don't dirty the scope. This prevents the agent from accidentally summarizing away records of mutations it made.

### History Injection

Completed snapshot summaries are injected into the system prompt at the start of every turn. Even if compaction later drops the summary message from the middle of the conversation, the knowledge survives in the system prompt. This is the bridge between proactive collapse and knowledge survival — snapshots persist through all compaction levels.

### Nudges

After 5 consecutive turns of read-only work (reading files, grepping, thinking), Swival nudges the agent to consider using `snapshot restore` to compress its investigation. The nudge fires once per read streak and doesn't repeat until the agent breaks the streak with a non-read operation.

## The Compaction Ladder

Compaction is a single ladder of strategies, ordered cheapest and most nearly lossless first. One budget-targeted entrypoint, `compact_to_budget`, drives it: given a target token budget, it applies rungs one at a time until the transcript fits — working on a copy of the history so a rung that fails or no-ops never leaves the real conversation half-mutated. If the enabled rungs run out before the budget is met, whatever reduction was achieved is still committed; leaner is better than not.

Two callers share this ladder. The **proactive** pass is the common case. Before every LLM call, if the estimated prompt is over a safe budget, Swival runs only the top three rungs — all near-lossless — to bring it back under, so the request rarely reaches the provider's wall. The **reactive** net handles the overflow that still slips through when a provider's tokenizer counts more tokens than Swival's estimate (detected when the provider rejects the call): it re-compacts below the current size with the full ladder enabled and retries, tightening the budget each round until the request fits.

The rungs, from cheapest to most aggressive:

### Level 1: Garbage-Collect Spent Scaffolding

The cheapest, most nearly lossless rung. As the loop runs, Swival injects synthetic nudges — tool-error guardrails, think/todo/snapshot reminders, empty-response retries — as throwaway messages. Once the model has produced a later response, those nudges have served their purpose and are dead weight. This rung drops them and nothing else. Durable synthetic context (goal recaps, image and command-output placeholders) is explicitly preserved and left for the lossy rungs. Because it only removes scaffolding the model already acted on, the proactive pass runs it freely.

### Level 2: Shrink Tool Results

The gentlest content rung. Old tool results (everything except the two most recent turns) are replaced with structured summaries.

A file read becomes `[read_file: path, N lines — content compacted]`. A batched read becomes `[read_multiple_files: path1, path2, …, N chars — compacted]`. A grep becomes `[grep: 'pattern' in path, ~N matches — compacted]`. Command output keeps its first and last 200 characters.

The agent retains all its turns and the structure of what it did. It loses the detailed content but keeps the metadata.

### Level 3: Strip Reasoning Payloads

Some providers expose hidden or semi-hidden model reasoning in a `reasoning_content` field. That field can be useful for provider replay, but it is often not useful as prompt context and can be large. Swival now counts it in token estimates and strips it during compaction. Providers that require the field on historical tool-call assistant messages keep the minimal placeholder they need.

Swival also compacts old visible assistant text that only led into a tool call, replacing long "reasoning before tool use" prose with a short marker.

### Level 4: Drop Low-Importance Turns

If shrinking results wasn't enough, Swival starts dropping entire turns from the middle of the conversation. Not all turns are equal — each one gets an importance score:

- Turns containing errors or failures score higher (the agent needs to remember what went wrong).
- Turns where files were written or edited score highest (the agent needs to remember what it changed).
- Thinking turns score moderately (reasoning context is valuable).
- Snapshot recap turns score as high as writes (they contain compressed knowledge).

The top half by score is kept. The bottom half is dropped and replaced with a summary generated by the LLM. That summary is prefixed with a marker that tells the model it's a factual recap, not a set of new instructions.

User messages are never silently dropped at this level. Only agent and tool turns are candidates for removal.

### Level 5: Aggressive Drop

Aggressive message compaction. Everything in the middle is dropped — including user messages. Only the system prompt, a summary of what was lost, and the last two turns survive.

If the LLM summary fails, Swival falls back to checkpoint summaries (if proactive summaries are enabled) or a static splice marker.

After any compaction level, the agent retries the LLM call.

### Level 6: Drop Tools For One Retry

If message compaction fails and tool schemas are still attached, Swival drops all tool schemas from that retry request. This is deliberately request-local: the durable tool list is not mutated, so a later turn can restore tools immediately. Permanent no-tools mode is reserved for providers that actually raise `ToolsNotSupportedError`.

### Level 7: Emergency Truncation

If the provider still rejects the request, Swival progressively emergency-truncates the remaining prompt at bounded ratios. It preserves the system prompt when possible and only truncates it in the final "make anything fit" stage. If even the smallest bounded request fails, the run writes a continue-here file and raises a context overflow error.

## Knowledge Survival

The most important design principle: critical state must survive compaction. Three mechanisms ensure this.

### Thinking State

The `think` tool maintains a history of numbered reasoning steps in memory. These steps support revision (correcting earlier conclusions) and branching (exploring alternatives). The thinking history lives entirely outside the message list — compaction doesn't touch it. Thinking turns also get a score bonus when Swival drops low-importance turns (Level 4), making them more likely to be retained than ordinary file reads.

### Todo State

The `todo` tool tracks work items in memory for the duration of the session. When aggressive compaction drops the turns where the agent planned its work, the todo list still exists because the state object lives outside the message history.

Swival also injects periodic reminders — if the agent hasn't checked its todo list for 3 turns and there are unfinished items, a reminder surfaces the list back into the conversation.

### Snapshot History

As described above, completed snapshot summaries are injected directly into the system prompt. This is the most durable persistence channel — the system prompt survives every compaction level, so investigation conclusions are never lost.

### Continue-Here Files

When a session ends abnormally — Ctrl+C, max turns exhausted, compaction failure, or REPL exit — Swival writes a structured `.swival/continue.md` file capturing the current task, todo state, recent tool activity, and key reasoning. On the next session start, this file is loaded into the system prompt and deleted, so the agent picks up where it left off without re-explanation.

The file is always written deterministically first (no network call). On the max-turns path only, Swival optionally enhances it with an LLM-generated summary. If the LLM call fails, the deterministic version is already on disk.

Continue-here files are capped at 4,000 characters. Files older than 24 hours trigger a staleness warning but are still loaded. Use `--no-continue` to disable both writing and reading. The `/status` command includes continue file presence in its session overview.

Together, these four channels mean that even after nuclear compaction wipes the conversation to nearly nothing, the agent still has its reasoning chain, its task list, a record of what it learned during investigation, and — if the session was interrupted — a structured resume plan for the next run.

## Proactive Checkpoint Summaries

Enabled with `--proactive-summaries`, this feature periodically summarizes recent turns after agent turns. Every 10 turns, the last batch is summarized via an LLM call and stored internally.

These summaries serve as a safety net. When Swival drops turns (Level 4 or Level 5) and can't get an LLM summary of the dropped span, it falls back to these pre-computed checkpoint summaries instead of losing the context entirely.

To prevent the checkpoint store from growing without bound, older summaries are periodically consolidated: the oldest half is merged into a single summary, creating a hierarchical map/reduce structure. The total is capped at roughly 2,000 tokens.

## Commands

Swival exposes manual controls for context management as input commands. These work in interactive mode and in one-shot mode when `--oneshot-commands` is set.

`/compact` shrinks old tool results (the Level 2 rung). `/compact --drop` also drops low-importance turns. Both report how many tokens were saved.

`/save [label]` sets a snapshot checkpoint at the current position. `/restore` generates a summary via LLM and collapses everything since the checkpoint. `/unsave` cancels the checkpoint. These are the manual equivalents of the agent's `snapshot` tool.

`/clear` drops everything and resets all internal state — conversation, thinking, todos, snapshots, and file tracking.

For the full command reference, see [Usage](usage.md).

## Configuration

Most context management works automatically. A few settings let you tune it.

`--max-context-tokens` tells Swival how large the context window is.

`--proactive-summaries` enables the periodic checkpoint summarization described above. Recommended for long-running sessions where the agent will go through many investigation and implementation cycles.

`--max-output-tokens` controls the maximum generation budget per LLM call. Before every call, Swival dynamically shrinks this to fit the remaining context space, so you don't need to tune it carefully — but setting it to a reasonable value (the default is 32,768) helps Swival estimate budgets accurately.

These can be set via CLI flags, project config (`swival.toml`), or global config (`~/.config/swival/config.toml`). See [Usage](usage.md) for the full flag reference.

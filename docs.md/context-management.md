# Context Management

Most AI coding agents assume large context windows — 128K tokens or more. Swival is designed to work well with models that have less tokens to work with. Every token matters, so Swival manages context aggressively at every stage: preventing bloat before it happens, giving the agent tools to compress its own history, and recovering gracefully when the window fills up.

You don't need to configure any of this. It works out of the box. But understanding how it works helps you get the most out of small models and explains what's happening when you see compaction messages in the logs.

## The Five Layers

Swival's context management operates as five concentric layers of defense.

**Layer 1 — Prevention.** Tool output is capped before it enters the conversation. A file read that returns 200KB of content is truncated to 50KB. A grep that matches thousands of lines returns the first 100. Command output over 10KB is saved to a file and replaced with a pointer. MCP tool schemas that would consume too much of the window are dropped at startup. These limits keep junk out of the context so compaction rarely needs to fire.

**Layer 2 — Preflight pruning.** Every turn, before the LLM call, Swival runs a pruning pass over the conversation history. It removes expired scaffolding messages (nudges, guardrails, error corrections), folds old state-tool turns (think, todo, snapshot) into compact recaps, and strips provider metadata from historical tool calls. This is the cheapest token reduction — it removes content that is either expired agent scaffolding or exactly reconstructible from live state, without any LLM calls or lossy summarization.

**Layer 3 — Proactive collapse.** The agent can actively manage its own context using the `snapshot` tool. After reading several files to understand a problem, the agent calls `snapshot restore` with a summary of what it learned. The file reads — often 10K+ tokens of dead weight — are replaced with a ~200 token summary. The agent keeps the knowledge; the context gets the space back.

**Layer 4 — Reactive compaction.** When the context fills up despite prevention, pruning, and proactive collapse, Swival runs a graduated compaction pipeline. It starts gentle (shrinking old tool results), escalates if needed (dropping low-importance turns with the last 3 turns protected), and has a last-resort nuclear option (keeping only the system prompt, a summary, and the last 2 turns). Each level is tried in sequence, and the agent only moves to the next if the previous wasn't enough.

**Layer 5 — Knowledge survival.** Thinking notes, todo lists, and snapshot summaries live outside the message history in independent channels that compaction cannot touch. Even after the most aggressive compaction wipes nearly everything, the agent still knows its reasoning, its task list, and what it learned during investigation.

## Prevention: Output Size Guards

Every tool that returns content has a hard cap on how much it can put into the conversation.

- **File reads** are capped at 50KB, with individual lines truncated at 2,000 characters. Large files can be paginated with offset and limit parameters.
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

## Preflight Pruning

Every turn, after snapshot history is injected into the system message and before token estimation, Swival runs a pruning pass over the live conversation. The pass mutates the message list in place so later turns benefit from the same reductions. It never touches the system message or tool schemas.

Four rules run in order.

### Synthetic Message Cleanup

The agent loop injects scaffolding messages during normal operation: nudges when the model makes repeated errors, tips to encourage thinking before editing, reminders about unfinished todo items, and continuation prompts when the model returns an empty response. These are useful for one turn but become dead weight after the model has responded.

The pruning pass removes any synthetic message that has already been followed by an assistant response. Only messages explicitly marked by the agent loop are eligible — real user messages are never touched, even if they happen to start with the same words as a nudge.

### State-Tool Folding

Tools like `think`, `todo`, and `snapshot` generate verbose tool-call and tool-result messages that mostly duplicate state already tracked in memory. Once the model has had a chance to react to a state-tool turn (at least one later assistant response exists), the pruning pass replaces it with a compact recap built from the current live state.

Recaps are placed near the tail of the conversation so the model attends to them naturally. They are formatted as short summaries: the todo recap lists remaining items, the think recap shows recent reasoning steps, and the snapshot recap shows the active checkpoint status.

Mixed turns — an assistant message that calls both a state tool and a non-state tool like `edit_file` — are left untouched. The snapshot save turn that anchors an active explicit checkpoint is also preserved.

### Repair Feedback Expiry

When the model sends malformed tool-call arguments, Swival auto-corrects them and appends a corrective note to the tool result explaining what was wrong. This teaching feedback is valuable for the immediate next attempt but becomes noise in later turns. The pruning pass strips the feedback suffix from tool results older than the two most recent.

### Tool-Call Canonicalization

Different LLM providers attach extra metadata to tool calls — fields like `index` that are useful for streaming but meaningless in replay. The pruning pass rewrites historical tool calls to a minimal shape with just the ID, type, function name, and arguments. The most recent tool-call turn is left untouched.

## Proactive Collapse: The Snapshot Tool

The snapshot tool is the centerpiece of Swival's context management. It lets the agent compress its own investigation history on demand.

A typical workflow looks like this: the agent reads 8 files and greps through logs to debug an authentication failure. All those reads add up to maybe 12K tokens sitting in the conversation — content the agent has already processed and drawn conclusions from. The agent calls `snapshot restore` with a summary like "Root cause: missing null check in auth/parser.py:142. Fix: guard with `if token is None: return default_token`." The 12K tokens of reads collapse to about 200 tokens. The agent continues with a clean context window.

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

## Reactive Compaction

When the context overflows — either detected before the LLM call or reported by the provider afterward — Swival runs up to four compaction levels in sequence.

### Level 1: Shrink Tool Results

The gentlest approach. Old tool results (everything except the two most recent turns) are replaced with structured summaries. A file read becomes `[read_file: path, N lines — content compacted]`. A batched read becomes `[read_multiple_files: path1, path2, …, N chars — compacted]`. A grep becomes `[grep: 'pattern' in path, ~N matches — compacted]`. Command output keeps its first and last 200 characters.

The agent retains all its turns and the structure of what it did. It loses the detailed content but keeps the metadata.

### Level 2: Drop Low-Importance Turns

If shrinking results wasn't enough, Swival starts dropping entire turns from the middle of the conversation. Not all turns are equal — each one gets an importance score:

- Turns containing errors or failures score higher (the agent needs to remember what went wrong).
- Turns where files were written or edited score highest (the agent needs to remember what it changed).
- Thinking turns score moderately (reasoning context is valuable).
- Snapshot recap turns score as high as writes (they contain compressed knowledge).

The top half by score is kept. The bottom half is dropped and replaced with a summary generated by the LLM. That summary is prefixed with a marker that tells the model it's a factual recap, not a set of new instructions.

User messages are never silently dropped at this level. Only agent and tool turns are candidates for removal.

### Level 3: Nuclear Drop

Last resort. Everything in the middle is dropped — including user messages. Only the system prompt, a summary of what was lost, and the last two turns survive.

If the LLM summary fails, Swival falls back to checkpoint summaries (if proactive summaries are enabled) or a static splice marker.

After any compaction level, the agent retries the LLM call.

### Level 4: Drop Tools

If all three message compaction levels fail — typically on models with very small context windows (4K or less) — Swival drops all tool schemas from the request and truncates the system prompt to fit the remaining budget. The model loses its ability to call tools and becomes a plain chatbot, but it can at least produce a text answer instead of aborting.

If even this fails, the run is aborted with an error.

## Knowledge Survival

The most important design principle: critical state must survive compaction. Three mechanisms ensure this.

### Thinking State

The `think` tool maintains a history of numbered reasoning steps in memory. These steps support revision (correcting earlier conclusions) and branching (exploring alternatives). The thinking history lives entirely outside the message list — compaction doesn't touch it. Thinking turns also get a score bonus during Level 2 compaction, making them more likely to be retained than ordinary file reads.

### Todo State

The `todo` tool tracks work items in memory for the duration of the session. When aggressive compaction drops the turns where the agent planned its work, the todo list still exists because the state object lives outside the message history.

Swival also injects periodic reminders — if the agent hasn't checked its todo list for 3 turns and there are unfinished items, a reminder surfaces the list back into the conversation.

### Snapshot History

As described above, completed snapshot summaries are injected directly into the system prompt. This is the most durable persistence channel — the system prompt survives every compaction level, so investigation conclusions are never lost.

### Continue-Here Files

When a session ends abnormally — Ctrl+C, max turns exhausted, compaction failure, or REPL exit — Swival writes a structured `.swival/continue.md` file capturing the current task, todo state, recent tool activity, and key reasoning. On the next session start, this file is loaded into the system prompt and deleted, so the agent picks up where it left off without re-explanation.

The file is always written deterministically first (no network call). On the max-turns path only, Swival optionally enhances it with an LLM-generated summary. If the LLM call fails, the deterministic version is already on disk.

Continue-here files are capped at 4,000 characters. Files older than 24 hours trigger a staleness warning but are still loaded. Use `--no-continue` to disable both writing and reading. The `/status` REPL command includes continue file presence in its session overview.

Together, these channels mean that even after nuclear compaction wipes the conversation to nearly nothing, the agent still has its reasoning chain, its task list, a record of what it learned during investigation, and — if the session was interrupted — a structured resume plan for the next run.

## Proactive Checkpoint Summaries

Enabled with `--proactive-summaries`, this feature periodically summarizes recent turns in the background. Every 10 turns, the last batch is summarized via an LLM call and stored internally.

These summaries serve as a safety net. When Level 2 or Level 3 compaction fires and can't get an LLM summary of the dropped turns, it falls back to these pre-computed checkpoint summaries instead of losing the context entirely.

To prevent the checkpoint store from growing without bound, older summaries are periodically consolidated: the oldest half is merged into a single summary, creating a hierarchical map/reduce structure. The total is capped at roughly 2,000 tokens.

## REPL Commands

The interactive REPL exposes manual controls for context management.

`/compact` triggers Level 1 compaction (shrink tool results). `/compact --drop` also triggers Level 2 (drop low-importance turns). Both report how many tokens were saved.

`/save [label]` sets a snapshot checkpoint at the current position. `/restore` generates a summary via LLM and collapses everything since the checkpoint. `/unsave` cancels the checkpoint. These are the manual equivalents of the agent's `snapshot` tool.

`/clear` drops everything and resets all internal state — conversation, thinking, todos, snapshots, and file tracking.

For the full REPL command reference, see [Usage](usage.md).

## Configuration

Most context management works automatically. A few settings let you tune it.

`--max-context-tokens` tells Swival how large the context window is. With LM Studio this is auto-detected, but for other providers you should set it to match your model. Without it, Swival may not trigger compaction at the right time.

`--proactive-summaries` enables the periodic checkpoint summarization described above. Recommended for long-running sessions where the agent will go through many investigation and implementation cycles.

`--max-output-tokens` controls the maximum generation budget per LLM call. Before every call, Swival dynamically shrinks this to fit the remaining context space, so you don't need to tune it carefully — but setting it to a reasonable value (the default is 32,768) helps Swival estimate budgets accurately.

These can be set via CLI flags, project config (`swival.toml`), or global config (`~/.config/swival/config.toml`). See [Usage](usage.md) for the full flag reference.

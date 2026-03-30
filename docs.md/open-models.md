# Not Just for Frontier Models

Most coding agents are built and tested against one or two frontier models.

They assume clean tool calls, reliable instruction following, and context windows measured in hundreds of thousands of tokens. When those assumptions hold, things work fine. When they don't, the agent falls apart in ways that look like the model's fault but are really the agent's.

Swival takes the opposite approach. It assumes the model will be working with tighter limits and rougher edges. Then it tries to keep the task moving anyway.

## What goes wrong with small models

Smaller and open models hit a few recurring problems when used inside an agent loop. A 16K or 32K context window fills up fast once you add a system prompt, tool schemas, a few file reads, and some back-and-forth. Once the window is full, the agent either crashes or silently drops context and starts hallucinating.

The model might omit required fields in tool calls, nest arguments wrong, or produce JSON that doesn't parse. Formatting rules in the system prompt get ignored. And longer tasks require the model to remember what it already tried, what files it changed, and what it planned to do next. Smaller models lose this thread more easily, especially after compaction or interruption.

None of this means small models are useless. It means the agent needs to do more of the work.

## How Swival compensates

These are engineering choices, not marketing claims. Each one exists because a specific failure mode showed up during testing with real models.

**Graduated context compaction.** When the window fills up, Swival runs a four-level compaction pipeline: shrink old tool results, drop low-value turns (scored by importance), nuclear-drop to just the last two turns, and finally shed tool schemas entirely. Each level fires only if the previous one wasn't enough. See [Context Management](context-management.md) for the full breakdown.

**Knowledge that survives compaction.** Thinking notes, todo lists, and snapshot summaries live outside the message history. Even after the most aggressive compaction, the agent still has its reasoning chain, its task list, and what it learned during investigation. This is the single most important thing for small-model reliability: the agent can lose old messages and keep working.

**Bounded tool output.** File reads are capped at 50KB. Grep returns at most 100 matches. Command output over 10KB is saved to a temp file and replaced with a pointer. MCP tool schemas that would eat more than half the context window are dropped at startup. These limits exist specifically so that one unlucky tool call can't blow the budget on a small window.

**Forgiving parsers.** Tool-call parsing uses a multi-pass approach. If the model's JSON is slightly broken, Swival tries to recover before giving up. The edit tool uses three-pass matching (exact, line-trimmed, unicode-normalized) so edits still land even when the model's whitespace is off. These are small things individually, but they add up to fewer stalled loops.

**Durable state tools.** The `think` tool gives the model a structured scratchpad that persists across compaction. The `todo` tool tracks work items that survive context drops. The `snapshot` tool lets the agent compress its own investigation into a summary and reclaim context space on demand. These tools exist because smaller models need external scaffolding to stay on track through multi-step tasks.

**Error guardrails.** If the model repeats the same error twice, Swival warns it. Three times, it tells the model to stop and try something different. This keeps small models from burning their entire context budget on a loop.

### A real example

Running [qwen3-coder-next](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF) locally in LM Studio with a 32K context window, Swival can complete multi-file refactoring tasks that involve reading several files, planning changes, and editing across modules.

The agent typically fires context compaction two or three times during a task like this. Without compaction and durable state, the same model stalls or hallucinates partway through because the context fills up and earlier reasoning is lost.

This is not a benchmark claim. It is a description of what happens in practice when the agent is designed around the model's real constraints instead of assuming they don't exist. You can reproduce it yourself with `swival --report run.json` to capture a full telemetry report of the run, including compaction events, tool usage, and timing. See [Reports](reports.md) for the details.

If Swival struggles with a model you care about, tell us. Before blaming the model, we'll look for ways to make the agent more forgiving and effective with it. That is what it means to build for small and open models, not just frontier ones: the agent takes responsibility for compensating, not just connecting.

## Related docs

- [Context Management](context-management.md) covers the full compaction pipeline, snapshot tool, and configuration options.
- [Tools](tools.md) documents every built-in tool, including `think`, `todo`, and `snapshot`.
- [Providers](providers.md) explains how to connect Swival to LM Studio, HuggingFace, OpenRouter, and other backends.
- [Reports](reports.md) explains the `--report` flag and the JSON telemetry format.

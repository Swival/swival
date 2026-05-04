# Goals

Set an objective with `/goal <objective>` in the REPL and Swival keeps the agent on task across turns.

It is a structured spin on the Ralph-style "keep prompting until it's done" loop.

The agent does not get to declare victory and walk away after one turn: the original objective is fed back to the model after every answer, and the loop only ends when the agent itself signals the goal is complete after a real evidence-based audit, declares a blocker, or hits the optional token budget.

This makes it practical to point Swival at ambitious, long-running tasks like refactors, audits, or end-to-end fixes, and let it grind for hours without giving up halfway. Pause, resume, replace, or clear the goal at any time.

## When to Use a Goal

Use a regular prompt for "answer or do this now". Use `/goal` for "keep working on this until it is actually done."

```text
/goal make the permute() function faster
```

On a goal like that, Swival will write benchmarks, try variant after variant of `permute()`, measure, discard regressions, keep the wins, and continue iterating until it has produced a measurably faster implementation, without you needing to babysit each round.

Goals are best given as outcomes that can be recognized as done ("get the test suite green," "reduce p99 latency of /search below 50ms," "port module X to async without breaking any caller"), not open-ended directions. The clearer the success condition, the less likely the loop is to declare victory early.

## How the Loop Works

After each turn that produces a final text answer with an active goal, the runtime injects a synthetic user message containing a continuation prompt. The continuation includes the objective verbatim as inert data, current usage, and remaining budget. Goal continuation turns count against `--max-turns` exactly like any other turn, so the loop never bypasses the user's hard ceiling.

If a continuation produces a final text answer with no tool calls, further continuations are suppressed to avoid a final-text loop, and the model's text is returned as a blocker or progress note.

While the goal loop is running, hit `Ctrl+C` at any time to interject. Swival pauses at the next safe point and prompts for a new line of input, which is folded in as an extra user message before the loop resumes. This is the way to redirect the agent mid-flight, supply a missing piece of context, or correct course without clearing the goal.

## The `complete_goal` Tool

Goal state is started and controlled by the operator through `/goal`. The model cannot create, replace, pause, resume, or inspect goals through tools. Swival exposes exactly one goal tool during active goal work: `complete_goal`.

`complete_goal` takes no arguments and marks the active goal complete. Before calling it, the model is expected to run an evidence-based audit that maps every requirement in the objective to real files, command output, or tests. If the model is blocked or needs user input, it should return final text describing the blocker instead of calling `complete_goal`.

## Token Budget Wrap-Up

When the optional `token_budget` is reached, the goal transitions to `budget_limited`. The runtime injects a wrap-up steering prompt, and the dispatcher rejects mutating or work-starting tool calls (write, edit, command execution, subagents, MCP, A2A) with a fixed error string. Read-only tools (`read_file`, `grep`, `list_files`, `fetch_url`, `view_image`, `think`, `todo`, `snapshot`, `outline`) and `complete_goal` remain available for a coherent wrap-up.

## Slash Command Reference

| Command                     | Effect                                                   |
| --------------------------- | -------------------------------------------------------- |
| `/goal`                     | Show current goal status, or "No goal is currently set." |
| `/goal <objective>`         | Create a goal (refused if one already exists).           |
| `/goal replace <objective>` | Replace the existing goal and reset counters.            |
| `/goal pause`               | Pause the active goal.                                   |
| `/goal resume`              | Resume a paused goal.                                    |
| `/goal clear`               | Remove the current goal.                                 |

The set and replace forms are REPL-only.

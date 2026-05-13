# Agent metaSKILLs

Agent MetaSKILLs are small workflow programs for AI agents. Like regular agent
SKILLs, they are meant to be automatically loaded on-demand when the user agent
determines they are needed.

They are a dynamic extension to a regular agent
skill: they sit next to a normal `SKILL.md` file and help an agent do the parts
of a task that should be repeatable, bounded, and easy to inspect.

## 1. Motivation

Static skills are great when the useful thing is instruction:

```text
When deploying, run the deploy command, check the health endpoint, and report
the result.
```

But many useful agent workflows are not just instructions. They are loops:

- Generate an answer, check whether it contains required pieces, and retry if
  it does not.
- Run tests, send the failures back to the agent, and stop when the tests pass.
- Ask for a patch, review it, ask for a fix, and keep a trace of each round.
- Call a verifier after each attempt and return a compact summary of the final
  result.

You can describe those loops in Markdown, but then the model has to remember and
execute the loop correctly every time. That is fragile. A metaskill makes the
loop explicit:

- The workflow has a real attempt limit.
- Commands go through the host's normal policy.
- Nested model calls are counted.
- The final result has a predictable JSON shape.
- A trace shows which steps ran.

In short: use a static skill for guidance; use a metaskill when the skill needs
to run a small, repeatable procedure.

## 2. A tiny metaSKILL

A metaskill is a skill directory with two files:

```text
skills/
  require-heading/
    SKILL.md
    SKILL.star
```

`SKILL.md` describes when the skill should be used and documents the expected
input keys:

```markdown
---
name: require-heading
description: Draft text and retry until it includes a required heading.
metaskill: SKILL.star
metaskill_language: starlark
---

Use this when the user wants a short draft that must include one exact heading.

Input keys:
- `task`: The writing task description.
- `heading`: The exact heading text that must appear in the draft.
```

`SKILL.star` contains the workflow:

```python
def run(input):
    task = input["task"]
    heading = input["heading"]

    for i in range(3):
        result = ask(task, {
            "purpose": "draft",
            "max_turns": 4,
        })

        if heading in result["answer"]:
            return {
                "status": "accepted",
                "answer": result["answer"],
                "attempts": i + 1,
            }

        trace("missing-heading", {
            "attempt": i + 1,
            "heading": heading,
        })

        task = (
            "Revise the draft so it includes this exact heading:\n" +
            heading +
            "\n\nPrevious draft:\n" +
            result["answer"]
        )

    return {
        "status": "exhausted",
        "answer": result["answer"],
        "warning": "The required heading was still missing.",
    }
```

The author writes a normal, small program. The host supplies `ask`, `command`,
and `trace`. The program does not get raw filesystem, process, network, or
environment access.

## 3. Package format

A metaskill is still a skill. It MUST live in a directory containing `SKILL.md`.
The dynamic workflow is a program file in the same directory. The conventional
program filename is `SKILL.star`.

```text
skills/
  test-fix/
    SKILL.md
    SKILL.star
```

`SKILL.md` is the single source of metadata. It MUST start with YAML
frontmatter.

Required fields:

| Field         | Type   | Meaning                                                     |
| ------------- | ------ | ----------------------------------------------------------- |
| `name`        | string | Stable skill identifier.                                    |
| `description` | string | Short catalog description used by users and routing models. |

Optional fields:

| Field                | Type   | Meaning                                                                                                                                     |
| -------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `metaskill`          | string | Relative path from the skill directory to the program file. Defaults to `SKILL.star` when that file exists.                                 |
| `metaskill_language` | string | Program language identifier. This version of the specification defines `starlark`. Defaults to `starlark` when a metaskill file is present. |

Name rules:

- `name` MUST be lowercase ASCII alphanumeric with hyphens.
- `name` MUST NOT contain leading hyphens, trailing hyphens, or consecutive
  hyphens.
- `name` MUST match the skill directory name.
- `name` SHOULD NOT exceed 64 characters.

Description rules:

- `description` MUST be non-empty after Markdown comments are removed.
- `description` SHOULD NOT exceed 1,024 characters.
- `description` SHOULD explain when to use the skill.
- The Markdown body SHOULD NOT exceed 20,000 characters; hosts MAY truncate
  longer bodies during activation.

Input documentation:

- The `SKILL.md` body SHOULD document the expected `input` keys so that models
  and users know what to pass to `run_metaskill`.
- A simple "Input keys:" list with key names and descriptions is sufficient.
- If a host receives an empty `input`, it SHOULD return the skill instructions
  in the error message so the caller can learn the expected keys.

Program file rules:

- `metaskill` MUST be a relative path.
- The resolved program path MUST stay inside the skill directory.
- The program file MUST be UTF-8 text.
- The program file SHOULD NOT exceed 64 KiB.
- The program file MUST NOT contain its own frontmatter.

Hosts MUST ignore unknown metadata fields unless they intentionally implement an
extension. Extension fields SHOULD be prefixed, for example
`x_myhost_feature`.

## 4. Discovery

Skill search locations are host-defined. A portable metaskill package MUST NOT
depend on a specific installation directory.

A conforming host discovers metaskills by:

1. Finding skill directories that contain `SKILL.md`.
2. Parsing and validating `name` and `description`.
3. Resolving `metaskill`, or defaulting to `SKILL.star` when it exists.
4. Rejecting metaskill paths that escape the skill directory.
5. Rejecting unsupported `metaskill_language` values for execution.
6. Marking executable metaskills in the skill catalog.

If multiple discovered skills have the same `name`, the host MUST choose a
deterministic winner according to documented precedence rules.

Catalog entries SHOULD stay compact:

```text
- test-fix: Run tests and ask the agent to fix failures. (metaskill: starlark)
```

## 5. Activation and execution

Activation and execution are separate.

Activation loads `SKILL.md` instructions into context. It should not execute
code.

Execution runs the metaskill program. It can spend nested model calls, run
allowed commands, and cause side effects through the host. A host SHOULD expose
execution through an operation named `run_metaskill`.

If a host supports skill mentions such as `$test-fix`, those mentions SHOULD
activate the skill instructions. They SHOULD NOT automatically execute the
metaskill unless the user clearly asked for that behavior or the host has a
separate documented policy.

## 6. Runtime language

This specification defines [Starlark](https://starlark-lang.org), a safe subset of Python, as the portable metaskill language.

A metaskill program MUST define:

```python
def run(input):
    return "done"
```

The host calls `run(input)` exactly once per execution. `input` is a JSON-like
dictionary supplied by the caller. It SHOULD contain only strings, numbers,
booleans, null values, lists, and dictionaries.

The Starlark environment MUST be hermetic by default:

- No filesystem access.
- No network access.
- No environment access.
- No process spawning.
- No unrestricted imports or module loading.
- No host-specific objects except the host API functions defined below.

Hosts MAY support additional languages in the future by adding new
`metaskill_language` values. A metaskill that wants to be portable across hosts SHOULD use `starlark`.

## 7. Host API

A metaskill exposes three global functions to the program:

| Function                 | Purpose                                        |
| ------------------------ | ---------------------------------------------- |
| `ask(prompt, opts={})`   | Run a bounded nested agent turn.               |
| `command(argv, opts={})` | Run an argv-style command through host policy. |
| `trace(kind, data={})`   | Append structured trace data.                  |

Portable metaskills MUST use these global functions. A host MAY also expose an
equivalent namespaced object such as `agent.ask`, but that is an extension and
MUST NOT be required by portable metaskills.

### `ask(prompt, opts={})`

`ask` asks the host agent to do one nested piece of work. It uses the host's
normal model, tools, sandbox, cancellation, and reporting behavior.

Parameters:

| Name             | Type    | Meaning                                                                  |
| ---------------- | ------- | ------------------------------------------------------------------------ |
| `prompt`         | string  | Prompt for the nested agent turn.                                        |
| `opts.purpose`   | string  | Optional label for traces and reports.                                   |
| `opts.max_turns` | integer | Optional nested turn cap. Hosts MUST clamp this to a documented maximum. |

Return value:

```json
{
  "answer": "text returned by the nested agent",
  "exhausted": false,
  "turns": 6,
  "truncated": false
}
```

`answer` MUST be a string. `exhausted` SHOULD indicate whether the nested turn
hit a host limit. `turns` SHOULD be the number of turns used when known; hosts
MAY report the granted cap when exact usage is unavailable. `truncated` MUST be
true when `answer` was shortened before returning to the metaskill.

### `command(argv, opts={})`

`command` runs a command through the host's normal command policy. It MUST NOT
bypass allowlists, approvals, middleware, sandbox roots, lifecycle hooks, or
reporting. Shell strings are not part of the portable API.

Parameters:

| Name           | Type            | Meaning                                                                     |
| -------------- | --------------- | --------------------------------------------------------------------------- |
| `argv`         | list of strings | Command and arguments. MUST be non-empty.                                   |
| `opts.timeout` | integer         | Optional timeout in seconds. Hosts MUST clamp this to a documented maximum. |

Return value:

```json
{
  "ok": true,
  "exit_code": 0,
  "result": "stdout and stderr or host error text",
  "truncated": false
}
```

`ok` MUST be true only when the command was allowed, started, did not time out,
and exited successfully. `exit_code` SHOULD be the process exit code when known.

`result` MUST be a string. Host-level failures SHOULD use an `error:` prefix in
`result`. `truncated` MUST be true when `result` was shortened before returning
to the metaskill.

### `trace(kind, data={})`

`trace` records what happened. Use it for small facts that help a user or
implementer understand the run later.

Parameters:

| Name   | Type       | Meaning                        |
| ------ | ---------- | ------------------------------ |
| `kind` | string     | Short event label.             |
| `data` | dictionary | Optional JSON-like event data. |

Trace calls MUST NOT consume ask or command budgets. Hosts MUST cap the number
and size of trace entries.

## 8. Budgets and limits

A conforming host MUST enforce finite budgets. Recommended defaults:

| Limit                             | Default     |
| --------------------------------- | ----------- |
| Nested `ask` calls                | 5           |
| `command` calls                   | 10          |
| Wall-clock execution timeout      | 300 seconds |
| Trace entries                     | 100         |
| Characters per nested answer      | 20,000      |
| Characters per command result     | 20,000      |
| Characters in final result string | 20,000      |

The `run_metaskill` operation MAY allow callers to set `max_ask_calls` and
`max_command_calls` within host-defined bounds. Hosts SHOULD check cancellation
and timeout before every host API call. Hosts SHOULD also interrupt or isolate
pure script computation so large loops cannot evade wall-clock limits.

## 9. Result contract

Metaskill execution returns a string at the host tool boundary.

On success, the string SHOULD contain a short completion header followed by one
JSON object:

```text
[Metaskill: test-fix completed]
{"status":"accepted","answer":"Tests pass.","trace":[...]}
```

The JSON object is the result envelope. The top-level `answer` is the primary
text a calling model should use. Other fields are structured metadata.

`run(input)` return values are normalized by the host:

| Script return value | Normalized envelope                                                                                        |
| ------------------- | ---------------------------------------------------------------------------------------------------------- |
| `None`              | `{"status":"ok","answer":""}`                                                                              |
| string              | `{"status":"ok","answer":<string>}`                                                                        |
| dictionary          | The dictionary, with missing `status` defaulted to `ok` and missing `answer` defaulted to an empty string. |

Any other return type MUST fail with an `error:` result.

On failure, the host MUST return a string beginning with `error:`. Failures
include unknown metaskill name, static skill passed to `run_metaskill`,
disallowed external metaskill, unavailable runtime, unreadable program file,
syntax errors, runtime errors, exhausted budgets, timeout, cancellation, and
invalid return values.

When the success envelope exceeds the host's result limit, the host SHOULD
truncate trace first and answer last. Truncation MUST be explicit in the JSON,
for example with `answer_truncated: true` or a trace entry whose kind is
`truncated`.

## 10. Standard tool schema

Hosts that expose tools to a model SHOULD expose metaskill execution with this
JSON-schema-shaped function:

```json
{
  "name": "run_metaskill",
  "description": "Run a dynamic skill workflow by name.",
  "parameters": {
    "type": "object",
    "properties": {
      "name": {
        "type": "string",
        "description": "The metaskill name to execute."
      },
      "input": {
        "type": "object",
        "description": "Input data passed to the metaskill program."
      },
      "max_ask_calls": {
        "type": "integer",
        "minimum": 1,
        "description": "Maximum nested model calls."
      },
      "max_command_calls": {
        "type": "integer",
        "minimum": 0,
        "description": "Maximum command calls."
      }
    },
    "required": ["name", "input"]
  }
}
```

When the host knows the executable metaskill names, it SHOULD add an `enum` to
the `name` property.

If the caller passes an empty or missing `input`, the host SHOULD return a
helpful error that includes the skill instructions so the caller can learn the
expected keys and retry without an additional round trip.

## 11. Practical examples

### Run tests and ask for fixes

This metaskill runs a test command. If the command fails, it asks the host agent
to fix the failure, then tries again.

```python
def run(input):
    test_command = input.get("test_command", ["pytest", "-q"])

    for i in range(3):
        result = command(test_command, {"timeout": 120})

        trace("test-run", {
            "attempt": i + 1,
            "ok": result["ok"],
        })

        if result["ok"]:
            return {
                "status": "accepted",
                "answer": "Tests passed.",
                "attempts": i + 1,
            }

        ask(
            "The tests failed. Fix the problem, then stop.\n\n" +
            result["result"],
            {
                "purpose": "fix-tests",
                "max_turns": 10,
            },
        )

    return {
        "status": "exhausted",
        "answer": result["result"],
        "warning": "Tests still failed after 3 attempts.",
    }
```

Example input:

```json
{
  "test_command": ["pytest", "tests/test_parser.py", "-q"]
}
```

### Require a checklist

This metaskill asks for a short checklist and retries until all required words
appear.

```python
def run(input):
    task = input["task"]
    required = input.get("required", ["install", "test", "rollback"])

    for i in range(3):
        result = ask(task, {
            "purpose": "draft-checklist",
            "max_turns": 4,
        })

        missing = []
        for word in required:
            if word not in result["answer"].lower():
                missing.append(word)

        if not missing:
            return {
                "status": "accepted",
                "answer": result["answer"],
                "attempts": i + 1,
            }

        trace("missing-words", {
            "attempt": i + 1,
            "missing": missing,
        })

        task = (
            "Revise the checklist. It must include these words: " +
            ", ".join(missing) +
            "\n\nPrevious checklist:\n" +
            result["answer"]
        )

    return {
        "status": "exhausted",
        "answer": result["answer"],
        "missing": missing,
    }
```

Example input:

```json
{
  "task": "Write a deployment checklist for a small web app.",
  "required": ["install", "test", "rollback"]
}
```

### Review a diff once

This metaskill keeps the workflow simple: get a diff through host policy, ask
for a focused review, and return the answer.

```python
def run(input):
    diff = command(["git", "diff"], {"timeout": 30})

    if not diff["ok"]:
        return {
            "status": "error",
            "answer": diff["result"],
        }

    review = ask(
        "Review this diff for bugs only. Ignore style nits.\n\n" +
        diff["result"],
        {
            "purpose": "bug-review",
            "max_turns": 6,
        },
    )

    return {
        "status": "ok",
        "answer": review["answer"],
    }
```

## 12. Security model

Metaskills are code, so they need an explicit trust model.

Default policy:

- Local metaskills MAY be executable by default when skills are enabled.
- External or global metaskills SHOULD require an explicit trust opt-in before
  execution.
- Static activation of external skills MAY remain available without enabling
  external metaskill execution.
- Direct script access to files, processes, environment variables, network,
  clocks, randomness, and imports MUST be unavailable unless a host documents a
  non-portable extension.
- All side effects MUST go through `ask` or `command`.

Prompt-injection posture:

- `input`, nested model answers, command output, and external tool output MUST
  be treated as untrusted data.
- Metaskill authors SHOULD validate outputs using deterministic checks whenever
  possible.
- Hosts SHOULD preserve normal untrusted-content labeling and reporting for
  nested tool results.

Sandbox posture:

- `command` MUST run under the same command policy and filesystem sandbox as
  normal host command tools.
- A metaskill MUST NOT expand the host's file, command, network, or tool access.
- Hosts SHOULD defensively copy or freeze `input`, host API return values, and
  trace data before exposing them to the script.

## 13. Agent behavior guidelines

An AI agent using a metaskill-capable host SHOULD:

- When a metaskill applies to the user's request, call `run_metaskill` directly
  with the appropriate `input` keys. The skill's SKILL.md body documents
  what keys the program expects.
- If the agent does not know what input keys a metaskill needs, it MAY call
  `use_skill` first to read the instructions, then call `run_metaskill`.
- Pass a small JSON-like `input` object containing the user's task, relevant
  constraints, and any explicit limits.
- Avoid budget overrides unless the task clearly needs them.
- Treat the returned `answer` field as primary and use `trace` as supporting
  evidence.
- Report `error:` results directly or use them to choose a fallback plan.

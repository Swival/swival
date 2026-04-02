# Semantic Routing

Semantic routing automatically selects the best profile for a given task. Instead of picking a model yourself, a lightweight router model reads your task and chooses from your configured profiles.

This is useful when you have profiles with very different strengths — a fast local model for quick edits, a large reasoning model for hard problems — and want the right one picked without thinking about it every time.

## Quick Setup

```toml
routing_enabled = true
semantic_routing_profile = "router"

[profiles.router]
provider = "lmstudio"
model = "qwen3.5-0.8b-mlx"

[profiles.fast]
provider = "lmstudio"
model = "qwen3-coder-next"
description = "Fast local edits, tests, and routine coding."

[profiles.heavy]
provider = "chatgpt"
model = "gpt-5.4"
reasoning_effort = "high"
description = "Hard debugging, design work, and complex reasoning."
```

With this config, every single-shot run asks the router model which profile fits best, then runs the task with that profile.

## How It Works

1. You run `swival "fix the off-by-one in parse.py"`.
2. Before anything else, Swival sends your task text to the router profile's model with the list of candidate profiles and their descriptions.
3. The router responds with a profile name and a short reason.
4. Swival selects that profile and continues as if you had typed `--profile <name>`.

The router call is cheap — small input, capped at 256 output tokens, temperature 0, no tools. It adds one fast LLM round-trip before the main run.

## Config Keys

### Top-Level

| Key                        | Type   | Default | Purpose                                                                              |
| -------------------------- | ------ | ------- | ------------------------------------------------------------------------------------ |
| `routing_enabled`          | bool   | `false` | Feature switch. Must be `true` for routing to run.                                   |
| `semantic_routing_profile` | string | —       | Name of the profile used to make routing decisions.                                  |
| `routing_strict`           | bool   | `false` | `true` = abort on routing failure. `false` = warn and fall back to `active_profile`. |

### Profile-Level

| Key           | Type   | Purpose                                                                                                    |
| ------------- | ------ | ---------------------------------------------------------------------------------------------------------- |
| `description` | string | Human-readable label. Profiles with a description are routing candidates. Also shown by `--list-profiles`. |

A profile without `description` is not a routing candidate. It can still be used via `--profile` or `active_profile`, but the router will never select it.

## Precedence

Profile selection follows this order:

1. `--profile NAME` (strongest — skips routing entirely)
2. Routing result (when `routing_enabled = true`)
3. `active_profile` from config
4. No profile

Routing behaves like a dynamic default. It never overrides an explicit `--profile` flag.

## Router vs Target Profiles

The `router profile` is the model that makes the decision. It does not need a `description` unless you want the router to also be selectable as an execution target. A small, fast model works well here — routing is a classification task, not a coding task.

`Target profiles` are profiles with a `description`. The router chooses among these based on the task text and descriptions. If only one target exists, Swival selects it directly without making a router call.

The router profile can select itself if it has a `description`. This is intentional — if the router doubles as a general-purpose model, give it a description that says so.

## Failure Handling

When the router model returns an unparseable or invalid response:

1. Try lenient parsing: accept bare profile names, extract names from short prose.
2. Retry once with a corrective prompt.
3. If still invalid:
   - ``routing_strict = false`` (default): warn and fall back to `active_profile`, or the router profile if no `active_profile` is set.
   - ``routing_strict = true``: abort with an error.

If the router profile's provider is unreachable, the same strict/fallback logic applies.

Fallback events are visible in verbose output and in `--report` JSON.

## Verbose Output

In verbose mode (the default unless `--quiet` is set), routing logs a status line:

```text
Routing: mode=semantic router=router selected=heavy duration=0.8s parse=json reason=Task needs stronger reasoning.
```

The provider info line also shows the routing decision:

```text
provider=chatgpt  model=gpt-5.4  profile=heavy  routed (via router)  context=1,050,000
```

## Short-Circuit

If only one profile has a `description`, routing selects it directly without calling the router model:

```text
Routing: single target fast, skipping router call
```

This avoids paying for a router call when there is no actual choice to make.

## Reports

When `--report` is used, the report JSON includes a `semantic_routing` block in `settings`:

```json
{
  "semantic_routing": {
    "router_profile": "router",
    "selected_profile": "heavy",
    "reason": "Task needs stronger reasoning.",
    "fallback_used": false,
    "parse_mode": "json"
  }
}
```

## Outbound Protections

The router call goes through the same outbound protections as normal agent calls:

- `LLM filter` (`llm_filter`): the filter script receives the router call with `call_kind` set to `"router"`. You can use this to apply different filtering rules for routing calls vs normal agent calls.
- `Secret encryption` (`encrypt_secrets`): if encryption is enabled, the router call is encrypted the same way.

## Scope

Semantic routing currently runs for:

- CLI single-shot runs
- Piped stdin tasks
- `--objective` mode (routes once at startup, pins the profile for the full run)

It does not yet run for REPL sessions, `Session.run()` / `Session.ask()`, or `--serve` / A2A contexts. These are planned for a future release.

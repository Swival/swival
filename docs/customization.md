# Customization

## Project instruction files

Swival supports two project instruction files, loaded from the base directory at
startup:

- `CLAUDE.md` -- project-level instructions, wrapped in `<project-instructions>` tags
- `AGENT.md` -- agent-specific instructions, wrapped in `<agent-instructions>` tags

If both exist, both are included (CLAUDE.md first). Each file is capped at
10,000 characters. These are appended to the default system prompt, so the
agent sees them alongside its built-in instructions.

This is useful for telling the agent about your project's conventions, preferred
patterns, or things to watch out for. A typical CLAUDE.md might say:

```markdown
This is a Go project using Chi for routing. Tests use testify.
Always run `go test ./...` after making changes.
Don't add dependencies without asking.
```

### Disabling instruction files

For untrusted repositories where you don't want the agent reading project
instructions:

```sh
uvx swival --no-instructions "task"
```

Instruction files are also skipped when you provide a custom system prompt with
`--system-prompt`.

## System prompt control

The default system prompt lives in `swival/system_prompt.txt` and describes the
agent's workflow, tools, and coding standards. You can replace it entirely:

```sh
uvx swival --system-prompt "You are a security auditor. Only report vulnerabilities." "Audit src/"
```

Or omit it:

```sh
uvx swival --no-system-prompt "Just answer: what is 2+2?"
```

These two flags are mutually exclusive. When using `--system-prompt`,
CLAUDE.md/AGENT.md files and the skill catalog are not appended (since you're
providing the full prompt yourself).

The current date and time are always appended to the system message, regardless
of which prompt is used.

## Tuning parameters

### Temperature and top-p

```sh
uvx swival --temperature 0.3 --top-p 0.9 "task"
```

The default temperature is 0.55, which gives a good balance between creativity
and consistency for coding tasks. Lower values make the agent more deterministic;
higher values make it more creative (and more likely to hallucinate).

Top-p defaults to 1.0 (no nucleus sampling). Reducing it limits the token pool
the model samples from.

### Seed

```sh
uvx swival --seed 42 "task"
```

Sets a random seed for reproducible outputs. When given, the seed is passed
through to the model provider. Not all models support this, and even those that
do may not guarantee identical outputs across different hardware or software
versions. Omit it (the default) to let the model sample normally.

### Max turns

```sh
uvx swival --max-turns 10 "quick task"
```

Limits the number of agent loop iterations. Each turn is one LLM call that may
include multiple tool calls. The default is 50. If the agent hits the limit
without producing a final answer, it exits with code 2.

### Output tokens

```sh
uvx swival --max-output-tokens 16384 "task"
```

Maximum tokens the model can generate per response. Defaults to 32768. Swival
automatically clamps this down if the prompt is large enough that prompt +
output would exceed the context window.

### Context length

```sh
uvx swival --max-context-tokens 65536 "task"
```

For LM Studio, this can trigger a model reload with the new context size. The
value must be at least as large as `--max-output-tokens`. If not specified,
Swival uses whatever context length the model is currently loaded with.

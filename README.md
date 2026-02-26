![Swival Logo](.media/logo.png)

# Swival

A coding agent for open models. [Documentation](https://swival.github.io/swival/)

Swival connects to [LM Studio](https://lmstudio.ai/),
[HuggingFace Inference API](https://huggingface.co/inference-api), or
[OpenRouter](https://openrouter.ai/), sends your task, and runs an autonomous
tool loop until it produces an answer. With LM Studio it auto-discovers your
loaded model, so there's nothing to configure. A few thousand lines of Python,
no framework.

## Quickstart

### LM Studio

1. Install [LM Studio](https://lmstudio.ai/) and load a model with tool-calling
   support. Recommended first model:
   [qwen3-coder-next](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF)
   (great quality/speed tradeoff on local hardware).
   Crank the context size as high as your hardware allows.
2. Start the LM Studio server.
3. Install Swival:

```sh
uv tool install swival
```

4. Run:

```sh
swival "Refactor the error handling in src/api.py"
```

That's it. Swival finds the model, connects, and goes to work.

### HuggingFace

```sh
export HF_TOKEN=hf_...
uv tool install swival
swival "Refactor the error handling in src/api.py" \
    --provider huggingface --model zai-org/GLM-5
```

You can also point it at a dedicated endpoint with `--base-url` and `--api-key`.

### OpenRouter

```sh
export OPENROUTER_API_KEY=sk_or_...
uv tool install swival
swival "Refactor the error handling in src/api.py" \
    --provider openrouter --model openrouter/free
```

### Interactive sessions

```sh
swival --repl
```

The REPL carries conversation history across questions, which makes it good for
exploratory work and longer tasks.

### Updates and uninstall

```sh
uv tool upgrade swival    # update
uv tool uninstall swival  # remove
```

## What makes it different

**Your models, your way.** Swival works with LM Studio and HuggingFace
Inference API. With LM Studio, it auto-discovers whatever model you have
loaded. With HuggingFace, point it at any supported model or your own dedicated
endpoint. You pick the model and the infrastructure.

**Small enough to read and hack.** The whole agent is a few thousand lines of
Python across a handful of files, with no framework underneath. You can read the
entire thing in an afternoon. If something doesn't work the way you want, you
can change it.

**Structured thinking for any model.** The built-in think tool gives any model
(including local ones) multi-step reasoning with revisions, branches, and
persistent notes that survive context compaction. It's not locked to a specific
provider.

**Built for benchmarking.** Pass `--report report.json` and Swival writes a
machine-readable evaluation report with per-call LLM timing, tool
success/failure counts, context compaction events, and guardrail interventions.
Good for comparing models systematically on real coding tasks.

**CLI-native.** stdout is exclusively the final answer. All diagnostics go to
stderr. You can pipe Swival's output straight into another command or a file.

## Documentation

Full documentation is available at [swival.github.io/swival](https://swival.github.io/swival/).

- [Getting Started](docs.md/getting-started.md) -- installation, first run, what
  happens under the hood
- [Usage](docs.md/usage.md) -- one-shot mode, REPL mode, CLI flags, piping,
  exit codes
- [Tools](docs.md/tools.md) -- what the agent can do: file ops, search, editing,
  web fetching, thinking, command execution
- [Safety and Sandboxing](docs.md/safety-and-sandboxing.md) -- path resolution,
  symlink protection, command whitelisting, YOLO mode
- [Skills](docs.md/skills.md) -- creating and using SKILL.md-based agent skills
- [Customization](docs.md/customization.md) -- project instructions, system prompt
  overrides, tuning parameters
- [Providers](docs.md/providers.md) -- LM Studio, HuggingFace, and OpenRouter
  configuration
- [Reports](docs.md/reports.md) -- JSON reports for benchmarking and evaluation
- [Reviews](docs.md/reviews.md) -- external reviewer scripts for automated QA
  and LLM-as-judge evaluation
- [Using Swival with AgentFS](docs.md/agentfs.md) -- copy-on-write filesystem
  sandboxing for safe agent runs

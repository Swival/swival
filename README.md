![Swival Logo](.media/logo.png)

# Swival

A coding agent for any model. [Documentation](https://swival.dev/)

Swival is a CLI coding agent built to be practical, reliable, and easy to use.
It works with frontier models, but its main goal is to be as reliable as
possible with smaller models, including local ones. It is designed from the
ground up to handle tight context windows and limited resources without falling
apart.

It connects to [LM Studio](https://lmstudio.ai/),
[HuggingFace Inference API](https://huggingface.co/inference-api),
[OpenRouter](https://openrouter.ai/),
[ChatGPT Plus/Pro](https://chatgpt.com/), or any OpenAI-compatible server (ollama,
llama.cpp, mlx_lm.server, vLLM, etc.), sends your task, and runs an autonomous
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
    --provider openrouter --model z-ai/glm-5
```

### ChatGPT Plus/Pro

Use OpenAI models through your existing ChatGPT Plus or Pro subscription -- no
API key needed.

```sh
uv tool install swival
swival "Refactor the error handling in src/api.py" \
    --provider chatgpt --model gpt-5.4
```

On first use, a device code and URL are printed to your terminal. Open the URL,
enter the code, and authorize with your ChatGPT account. Tokens are cached
locally for subsequent runs.

### Generic (OpenAI-compatible)

```sh
swival "Refactor the error handling in src/api.py" \
    --provider generic \
    --base-url http://127.0.0.1:8080 \
    --model my-model
```

Works with ollama, llama.cpp, mlx_lm.server, vLLM, and anything else that
speaks the OpenAI chat completions protocol. No API key required for local
servers.

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

**Reliable with small models.** Context management is one of Swival's strengths.
It keeps things clean and focused, which is especially important when you are
working with models that have tight context windows. Graduated compaction,
persistent thinking notes, and a todo checklist all survive context resets, so
the agent doesn't lose track of multi-step plans even under pressure.

**Your models, your way.** Works with LM Studio, HuggingFace Inference API,
OpenRouter, ChatGPT Plus/Pro, and any OpenAI-compatible server. With LM Studio,
it auto-discovers whatever model you have loaded. With HuggingFace or
OpenRouter, point it at any supported model. With ChatGPT Plus/Pro, authenticate
through your browser and use OpenAI's models through your existing subscription. With the generic provider, connect to ollama,
llama.cpp, mlx_lm.server, vLLM, or any other compatible server. You pick the
model and the infrastructure.

**Review loop and LLM-as-a-judge.** Swival has a configurable review loop that
can run external reviewer scripts or use a built-in LLM-as-judge to
automatically evaluate and retry agent output. Good for quality assurance on
tasks that matter.

**Built for benchmarking.** Pass `--report report.json` and Swival writes a
machine-readable evaluation report with per-call LLM timing, tool
success/failure counts, context compaction events, and guardrail interventions.
Useful for comparing models, settings, skills, and MCP servers systematically
on real coding tasks.

**Skills and MCP.** Extend the agent with SKILL.md-based skills for reusable
workflows, and connect to external tools via the Model Context Protocol.

**Small enough to read and hack.** A few thousand lines of Python across a
handful of files, with no framework underneath. Read the whole agent in an
afternoon. If something doesn't work the way you want, change it.

**CLI-native.** stdout is exclusively the final answer. All diagnostics go to
stderr. Pipe Swival's output straight into another command or a file.

## Documentation

Full documentation is available at [swival.dev](https://swival.dev/).

- [Getting Started](docs.md/getting-started.md) -- installation, first run, what
  happens under the hood
- [Usage](docs.md/usage.md) -- one-shot mode, REPL mode, CLI flags, piping,
  exit codes
- [Tools](docs.md/tools.md) -- what the agent can do: file ops, search, editing,
  web fetching, thinking, task tracking, command execution
- [Safety and Sandboxing](docs.md/safety-and-sandboxing.md) -- path resolution,
  symlink protection, command whitelisting, YOLO mode
- [Skills](docs.md/skills.md) -- creating and using SKILL.md-based agent skills
- [Customization](docs.md/customization.md) -- config files, project instructions,
  system prompt overrides, tuning parameters
- [Providers](docs.md/providers.md) -- LM Studio, HuggingFace, OpenRouter,
  ChatGPT Plus/Pro, and generic OpenAI-compatible server configuration
- [MCP](docs.md/mcp.md) -- connecting external tool servers via the Model Context
  Protocol
- [Reports](docs.md/reports.md) -- JSON reports for benchmarking and evaluation
- [Reviews](docs.md/reviews.md) -- external reviewer scripts for automated QA
  and LLM-as-judge evaluation
- [Using Swival with AgentFS](docs.md/agentfs.md) -- copy-on-write filesystem
  sandboxing for safe agent runs

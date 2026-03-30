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
[Google Gemini](https://ai.google.dev/),
[ChatGPT Plus/Pro](https://chatgpt.com/), any OpenAI-compatible server (ollama,
llama.cpp, mlx_lm.server, vLLM, etc.), or any external command
(`codex exec`, custom wrappers, etc.), sends your task, and runs an autonomous tool loop until
it produces an answer. With LM Studio it auto-discovers your
loaded model, so there's nothing to configure. A few thousand lines of Python,
no framework.

## Quickstart

Pick the provider that matches how you want to run models:

| Provider         | Auth                                                | Required flags                                    | First command                                                                        |
| ---------------- | --------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------ |
| LM Studio        | none                                                | none                                              | `swival "Refactor src/api.py"`                                                       |
| HuggingFace      | `HF_TOKEN` or `--api-key`                           | `--provider huggingface --model ORG/MODEL`        | `swival --provider huggingface --model zai-org/GLM-5 "task"`                         |
| OpenRouter       | `OPENROUTER_API_KEY` or `--api-key`                 | `--provider openrouter --model MODEL`             | `swival --provider openrouter --model z-ai/glm-5 "task"`                             |
| Google Gemini    | `GEMINI_API_KEY`, `OPENAI_API_KEY`, or `--api-key`  | `--provider google --model MODEL`                 | `swival --provider google --model gemini-2.5-flash "task"`                           |
| ChatGPT Plus/Pro | browser auth on first run or `CHATGPT_API_KEY`      | `--provider chatgpt --model MODEL`                | `swival --provider chatgpt --model gpt-5.4 "task"`                                   |
| Generic          | optional `OPENAI_API_KEY`                           | `--provider generic --base-url URL --model MODEL` | `swival --provider generic --base-url http://127.0.0.1:8080 --model my-model "task"` |
| AWS Bedrock      | AWS credential chain (`AWS_PROFILE`, env vars, IAM) | `--provider bedrock --model MODEL`                | `swival --provider bedrock --model global.anthropic.claude-opus-4-6-v1 "task"`       |
| Command          | none                                                | `--provider command --model "COMMAND"`            | `swival --provider command --model "codex exec --full-auto" "task"`                  |

Run `swival --help` for the grouped CLI reference and copy-paste examples.

### LM Studio

1. Install [LM Studio](https://lmstudio.ai/) and load a model with tool-calling
   support. Recommended first model:
   [qwen3-coder-next](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF)
   (great quality/speed tradeoff on local hardware).
   Crank the context size as high as your hardware allows.
2. Start the LM Studio server.
3. Install Swival (requires Python 3.13+):

```sh
uv tool install swival
```

On macOS you can also use Homebrew: `brew install swival/tap/swival`

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

### Google Gemini

```sh
export GEMINI_API_KEY=...
uv tool install swival
swival "Refactor the error handling in src/api.py" \
    --provider google --model gemini-2.5-flash
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

Works with ollama, llama.cpp, mlx_lm.server, vLLM, DeepSeek API, and anything
else that speaks the OpenAI chat completions protocol. No API key required for
local servers.

### Interactive sessions

```sh
swival
```

The REPL carries conversation history across questions, which makes it good for
exploratory work and longer tasks.

### Task Input From Stdin

If you omit the positional task and pipe stdin, Swival reads the task from
stdin.

```sh
swival -q < objective.md

cat prompts/review.md | swival --provider huggingface --model zai-org/GLM-5
```

Useful for long prompts, shell-quoting avoidance, and scripted workflows.

### Updates and uninstall

```sh
uv tool upgrade swival    # update (uv)
uv tool uninstall swival  # remove (uv)
brew upgrade swival       # update (Homebrew)
brew uninstall swival     # remove (Homebrew)
```

## What makes it different

**Reliable with small models.** Context management is one of Swival's strengths.
It keeps things clean and focused, which is especially important when you are
working with models that have tight context windows. Graduated compaction,
persistent thinking notes, and a todo checklist all survive context resets, so
the agent doesn't lose track of multi-step plans even under pressure.

**Your models, your way.** Works with LM Studio, HuggingFace Inference API,
OpenRouter, Google Gemini, ChatGPT Plus/Pro, any OpenAI-compatible server, and
any external command. With LM Studio, it auto-discovers whatever model you have
loaded. With HuggingFace or OpenRouter, point it at any supported model. With
Google Gemini, use Gemini models through Google's native API. With ChatGPT
Plus/Pro, authenticate through your browser and use OpenAI's models through your
existing subscription. With the generic provider, connect to ollama, llama.cpp,
mlx_lm.server, vLLM, or any other compatible server. With the command provider,
shell out to any program that reads a prompt on stdin and writes a response on
stdout. You pick the model and the infrastructure.

**Review loop and LLM-as-a-judge.** Swival has a configurable review loop that
can run external reviewer scripts or use a built-in LLM-as-judge to
automatically evaluate and retry agent output. Good for quality assurance on
tasks that matter.

**Built for benchmarking.** Pass `--report report.json` and Swival writes a
machine-readable evaluation report with per-call LLM timing, tool
success/failure counts, context compaction events, and guardrail interventions.
Useful for comparing models, settings, skills, and MCP servers systematically
on real coding tasks.

**Secrets stay on your machine.** Swival automatically detects API keys and
credential tokens in LLM messages and encrypts them before they leave your
machine. The LLM never sees the real values. Decryption happens locally when
the response comes back, so tools still work normally. No configuration needed.

**Cross-session memory.** The agent remembers things across sessions. It stores
notes in a local memory file and retrieves the most relevant entries for each
new conversation using BM25 ranking, so context from past work carries forward
without bloating the prompt. Use `/learn` in the REPL to teach it something
on the spot.

**Pick up where you left off.** When a session is interrupted — Ctrl+C, max
turns, context overflow — Swival saves its state to disk. Next time you run it
in the same directory, it picks up where it left off: what it was doing, what
it had figured out, and what was left.

**A2A server mode.** Run `swival --serve` and your agent becomes an A2A
endpoint that other agents can call over HTTP. Multi-turn context, streaming,
rate limiting, and bearer auth are built in.

**Skills, MCP, and A2A.** Extend the agent with SKILL.md-based skills for
reusable workflows, connect to external tools via the Model Context Protocol,
and talk to remote agents via the Agent-to-Agent (A2A) protocol.

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
  symlink protection, filesystem access modes, command execution modes
- [Skills](docs.md/skills.md) -- creating and using SKILL.md-based agent skills
- [Customization](docs.md/customization.md) -- config files, project instructions,
  system prompt overrides, tuning parameters
- [Context Management](docs.md/context-management.md) -- compaction, snapshots,
  knowledge survival, and how Swival handles tight context windows
- [Providers](docs.md/providers.md) -- LM Studio, HuggingFace, OpenRouter,
  Google Gemini, ChatGPT Plus/Pro, generic OpenAI-compatible server, and
  command (external program) configuration
- [MCP](docs.md/mcp.md) -- connecting external tool servers via the Model Context
  Protocol
- [A2A](docs.md/a2a.md) -- connecting to remote agents via the Agent-to-Agent
  protocol
- [Reports](docs.md/reports.md) -- JSON reports for benchmarking and evaluation
- [Web Browsing](docs.md/web-browsing.md) -- Chrome DevTools MCP, Lightpanda
  MCP, and agent-browser for web interaction
- [Reviews](docs.md/reviews.md) -- external reviewer scripts for automated QA
  and LLM-as-judge evaluation
- [Secret Encryption](docs.md/secrets.md) -- transparent encryption of
  credentials before they reach the LLM provider
- [Outbound LLM Filter](docs.md/llm-filter.md) -- user-defined scripts to
  redact or block outbound LLM requests
- [Lifecycle Hooks](docs.md/lifecycle-hooks.md) -- startup/exit hooks for
  syncing state to remote storage
- [Custom Commands](docs.md/custom-commands.md) -- REPL custom command setup
  and execution
- [Python API](docs.md/python-api.md) -- library API for embedding Swival in
  Python applications
- [Using Swival with AgentFS](docs.md/agentfs.md) -- copy-on-write filesystem
  sandboxing for safe agent runs

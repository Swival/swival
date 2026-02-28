# Getting Started

## Prerequisites

Swival requires Python 3.13 or newer and [uv](https://docs.astral.sh/uv/). If `uv` is not installed yet, you can install it with the command below.

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation

Install the CLI with `uv tool install swival`. This places the `swival` command on your `PATH`, so you can run it from any directory.

```sh
uv tool install swival
```

## Upgrading

To upgrade an existing installation to the newest release, run `uv tool upgrade swival`.

```sh
uv tool upgrade swival
```

If you ever want to remove it, run `uv tool uninstall swival`.

```sh
uv tool uninstall swival
```

## Running with LM Studio

LM Studio is the default provider and usually the fastest way to get started. Install LM Studio from [lmstudio.ai](https://lmstudio.ai/), load a tool-calling model, and start the local server from the Local Server tab. If your machine can handle it, increase the context window, because larger context gives the agent more room to reason over your codebase.

Once LM Studio is running, this is enough to start:

```sh
swival "Hello world"
```

By default, Swival connects to `http://127.0.0.1:1234`, queries LM Studio for the currently loaded model, and uses that model automatically.

## What Happens Internally

When you run a task against LM Studio, Swival first calls `/api/v1/models` to discover the loaded model and context size. It then builds a system prompt that includes tool definitions and workspace context, sends your task through LiteLLM, and enters the agent loop where the model can read files, edit files, search, and continue tool-calling until it finishes. When the model returns a final text answer with no more tool calls, Swival prints that answer to standard output and exits.

Diagnostic logs such as turn headers, tool traces, and timing information are written to standard error, which keeps standard output clean for piping into other tools.

## Running with HuggingFace

If you prefer hosted inference over running models locally, you can use the HuggingFace Inference API. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) if you don't have one already, then export it so Swival can authenticate.

```sh
export HF_TOKEN=hf_your_token_here
```

Pick a model that supports tool calling and pass it in `org/model` format.

```sh
swival "Hello world" --provider huggingface --model zai-org/GLM-5
```

This uses HuggingFace's serverless inference, which is the fastest way to try hosted models without provisioning anything. Serverless endpoints often have smaller context windows than local deployments, so long multi-turn sessions can hit context pressure sooner.

If you need more headroom, you can provision a dedicated HuggingFace endpoint and pass its URL directly. Dedicated endpoints let you use the full deployed context window.

```sh
swival "Hello world" \
    --provider huggingface \
    --model zai-org/GLM-5 \
    --base-url https://xyz.endpoints.huggingface.cloud \
    --api-key hf_your_key
```

For a deeper look at HuggingFace-specific options, see [Providers](providers.md).

## Running with OpenRouter

OpenRouter gives you access to models from many providers through a single API key. Sign up at [openrouter.ai](https://openrouter.ai/) and grab your API key from the dashboard.

```sh
export OPENROUTER_API_KEY=sk_or_your_token_here
```

Then pass the model you want to use. OpenRouter has both free and paid tiers.

```sh
swival "Hello world" --provider openrouter --model z-ai/glm-5
```

OpenRouter models vary widely in context limits, so you should set `--max-context-tokens` to match the model you chose. Without it, Swival falls back to a conservative default that may not use the full window your model supports.

```sh
swival "Hello world" \
    --provider openrouter \
    --model z-ai/glm-5 \
    --max-context-tokens 131072
```

For a deeper look at OpenRouter-specific options, see [Providers](providers.md).

## Running with Any OpenAI-Compatible Server

If you're running ollama, llama.cpp, mlx_lm.server, vLLM, or any other server that exposes an OpenAI-compatible API, use the generic provider.

```sh
swival "Hello world" \
    --provider generic \
    --base-url http://127.0.0.1:8080 \
    --model my-model
```

Both `--model` and `--base-url` are required. No API key is needed for most local servers. If your server requires one, pass `--api-key` or set `OPENAI_API_KEY`.

For a deeper look at generic provider options and server-specific examples, see [Providers](providers.md).

## Where To Go Next

If you want the full command surface and mode behavior, continue with [Usage](usage.md). If you want a deeper look at built-in capabilities, read [Tools](tools.md). If you need to understand trust boundaries before enabling stronger actions, read [Safety and Sandboxing](safety-and-sandboxing.md).

If you want to understand how Swival fits large tasks into small context windows, read [Context Management](context-management.md). If you want to connect external tool servers via MCP, see [MCP](mcp.md). If you want copy-on-write isolation so you can review and apply changes only when ready, read [Using Swival with AgentFS](agentfs.md).

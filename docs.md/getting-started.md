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

### Homebrew (macOS)

On macOS you can also install through the Homebrew tap:

```sh
brew install swival/tap/swival
```

This builds from source and pulls in Python 3.13 automatically. The first install takes a while because several dependencies compile native extensions.

## Upgrading

To upgrade an existing installation to the newest release:

```sh
uv tool upgrade swival    # if installed with uv
brew upgrade swival       # if installed with Homebrew
```

To remove it:

```sh
uv tool uninstall swival  # if installed with uv
brew uninstall swival     # if installed with Homebrew
```

## Provider Quick Reference

| Provider      | Auth                                                | Required flags                                    |
| ------------- | --------------------------------------------------- | ------------------------------------------------- |
| `lmstudio`    | none                                                | none                                              |
| `llamacpp`    | none                                                | `--provider llamacpp`                             |
| `huggingface` | `HF_TOKEN` or `--api-key`                           | `--provider huggingface --model ORG/MODEL`        |
| `openrouter`  | `OPENROUTER_API_KEY` or `--api-key`                 | `--provider openrouter --model MODEL`             |
| `google`      | `--api-key`, `GEMINI_API_KEY`, or `OPENAI_API_KEY`  | `--provider google --model MODEL`                 |
| `chatgpt`     | browser auth on first run or `CHATGPT_API_KEY`      | `--provider chatgpt --model MODEL`                |
| `generic`     | optional `OPENAI_API_KEY`                           | `--provider generic --base-url URL --model MODEL` |
| `bedrock`     | AWS credential chain (`AWS_PROFILE`, env vars, IAM) | `--provider bedrock --model MODEL`                |
| `command`     | none                                                | `--provider command --model "COMMAND"`            |

The sections below expand each provider with copy-paste commands.

## Running with LM Studio

LM Studio is the default provider and usually the fastest way to get started. Install LM Studio from [lmstudio.ai](https://lmstudio.ai/), load a tool-calling model, and start the local server from the Local Server tab. If your machine can handle it, increase the context window, because larger context gives the agent more room to reason over your codebase.

Once LM Studio is running, this is enough to start:

```sh
swival "Hello world"
```

By default, Swival connects to `http://127.0.0.1:1234`, queries LM Studio for the currently loaded model, and uses that model automatically.

## What Happens Internally

When you run a task against LM Studio, Swival first calls `/api/v1/models` to discover the loaded model and context size. It then builds a system prompt that includes tool definitions and workspace context, sends your task to the model, and enters the agent loop where the model can read files, edit files, search, and continue tool-calling until it finishes. When the model returns a final text answer with no more tool calls, Swival prints that answer to standard output and exits.

Diagnostic logs such as turn headers, tool traces, and timing information are written to standard error, which keeps standard output clean for piping into other tools.

## Passing The Task On Stdin

If you omit the positional task and pipe stdin, Swival reads the task from stdin. On an interactive terminal with no task, Swival enters REPL mode automatically.

```sh
swival -q < objective.md

cat prompts/review.md | swival --provider huggingface --model zai-org/GLM-5.1
```

This is useful for longer prompts, reusable task files, and avoiding shell quoting.

## Running with HuggingFace

If you prefer hosted inference over running models locally, you can use the HuggingFace Inference API. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) if you don't have one already, then export it so Swival can authenticate.

```sh
export HF_TOKEN=hf_your_token_here
```

Pick a model that supports tool calling and pass it in `org/model` format.

```sh
swival "Hello world" --provider huggingface --model zai-org/GLM-5.1
```

This uses HuggingFace's serverless inference, which is the fastest way to try hosted models without provisioning anything. Serverless endpoints often have smaller context windows than local deployments, so long multi-turn sessions can hit context pressure sooner.

If you need more headroom, you can provision a dedicated HuggingFace endpoint and pass its URL directly. Dedicated endpoints let you use the full deployed context window.

```sh
swival "Hello world" \
    --provider huggingface \
    --model zai-org/GLM-5.1 \
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

## Running with Google Gemini

If you want to use Gemini through Google's API, use the `google` provider. Authentication comes from `--api-key`, `GEMINI_API_KEY`, or `OPENAI_API_KEY`.

```sh
export GEMINI_API_KEY=...
swival "Hello world" --provider google --model gemini-2.5-flash
```

Swival routes this through Google's OpenAI-compatible endpoint and will try to auto-detect the context window when `--max-context-tokens` is not set.

For a deeper look at Google-specific options, see [Providers](providers.md).

## Running with Any OpenAI-Compatible Server

If you're running llama.cpp, use the `llamacpp` provider — it auto-discovers the loaded model and defaults to `http://127.0.0.1:8080`:

```sh
swival --provider llamacpp "Hello world"
```

For ollama, mlx_lm.server, vLLM, DeepSeek API, or any other server that exposes an OpenAI-compatible API, use the generic provider. Both `--model` and `--base-url` are required:

```sh
swival "Hello world" \
    --provider generic \
    --base-url http://127.0.0.1:8080 \
    --model my-model
```

No API key is needed for most local servers. If your server requires one, pass `--api-key` or set `OPENAI_API_KEY`.

For a deeper look at provider options and server-specific examples, see [Providers](providers.md).

## Running with ChatGPT Plus/Pro

If you have a ChatGPT Plus or ChatGPT Pro subscription and want to use OpenAI's models without a separate API key, the `chatgpt` provider authenticates through an OAuth device-code flow using your existing subscription.

```sh
swival "Hello world" --provider chatgpt --model gpt-5.4
```

On the first run, Swival will print a URL and a code. Open the URL in your browser, enter the code, and authorize. After that, tokens are cached locally and you won't be prompted again.

`--model` is required -- there is no default. Supported model names can change over time, so check [Providers](providers.md) if you need the current naming.

For a deeper look at ChatGPT Plus/Pro-specific options, see [Providers](providers.md).

## Running with AWS Bedrock

If you use AWS, the `bedrock` provider connects to models hosted on Bedrock using your existing AWS credentials. No API key is needed — Swival picks up credentials from environment variables, `~/.aws/credentials`, IAM roles, and SSO.

```sh
export AWS_REGION_NAME=us-east-2
swival "Hello world" --provider bedrock --model global.anthropic.claude-opus-4-6-v1
```

To use a named profile, pass `--aws-profile` or set `AWS_PROFILE`. Note: the region env var is `AWS_REGION_NAME`, not `AWS_DEFAULT_REGION`.

For a deeper look at Bedrock-specific options, see [Providers](providers.md).

## Where To Go Next

If you want the full command surface and mode behavior, continue with [Usage](usage.md). If you want a deeper look at built-in capabilities, read [Tools](tools.md). If you need to understand trust boundaries before enabling stronger actions, read [Safety and Sandboxing](safety-and-sandboxing.md).

If you want to understand how Swival fits large tasks into small context windows, read [Context Management](context-management.md). If you want to connect external tool servers via MCP, see [MCP](mcp.md). If you want to browse the web, see [Web Browsing](web-browsing.md). If you want to connect to remote agents via the A2A protocol, or expose Swival as an A2A server, see [A2A](a2a.md). If you want to drive Swival from an ACP-aware editor such as Zed or the `agent-client-protocol.nvim` plugin, see [ACP](acp.md). If you want copy-on-write isolation so you can review and apply changes only when ready, read [Using Swival with AgentFS](agentfs.md). If you want to run a staged security audit over your codebase, see [Security Audit](audit.md).

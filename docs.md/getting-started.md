# Getting Started

## Prerequisites

You need Python 3.13+ and [uv](https://docs.astral.sh/uv/). If you don't have
uv yet:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation

```sh
uv tool install swival
```

This puts the `swival` command on your PATH. You can now run it from anywhere.

To update to the latest version:

```sh
uv tool upgrade swival
```

To uninstall:

```sh
uv tool uninstall swival
```

## Running with LM Studio (recommended)

LM Studio is the easiest path. Install it from
[lmstudio.ai](https://lmstudio.ai/), then download a model that supports tool
calling. My pick is
[qwen3-coder-next](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF) -- on
a 64 GB MacBook Pro with a 100k context window, it's genuinely excellent.

Load the model, start the server (the "Local Server" tab in LM Studio), and
increase the context size as high as your RAM allows. Bigger context means the
agent can hold more of your codebase in its head at once.

Then run:

```sh
swival "Hello world"
```

Swival connects to LM Studio at `http://127.0.0.1:1234`, queries the API to
discover which model is loaded, and sends your prompt. No API keys, no config
files, no environment variables. It just works.

### What happens under the hood

When you run that command, Swival:

1. Hits LM Studio's `/api/v1/models` endpoint to find the loaded model and its
   context size.
2. Builds a system prompt that describes the available tools (file reading,
   editing, search, etc.) and the working directory.
3. Sends your question to the model via LiteLLM.
4. Enters the agent loop: the model can call tools, read files, make edits, and
   reason through the problem across as many turns as it needs.
5. When the model produces a final text answer (no more tool calls), Swival
   prints it to stdout and exits.

All diagnostic output (turn numbers, tool calls, timing) goes to stderr. The
final answer goes to stdout. This means you can pipe Swival's output cleanly
into other tools.

## Running with HuggingFace

If you'd rather use a hosted model instead of running one locally:

```sh
export HF_TOKEN=hf_your_token_here
swival "Hello world" --provider huggingface --model Qwen/Qwen3.5-397B-A17B
```

You need a HuggingFace token with Inference API access. The model must be
specified in `org/model` format. For dedicated endpoints, add `--base-url` and
`--api-key`:

```sh
swival "Hello world" \
    --provider huggingface \
    --model Qwen/Qwen3.5-397B-A17B \
    --base-url https://xyz.endpoints.huggingface.cloud \
    --api-key hf_your_key
```

See [Providers](providers.md) for more detail on provider configuration.

## Next steps

- [Usage](usage.md) covers the two operating modes and all CLI flags.
- [Tools](tools.md) explains what the agent can do out of the box.
- [Safety and Sandboxing](safety-and-sandboxing.md) explains how file access is restricted.
- [Using Swival with AgentFS](agentfs.md) shows how to run the agent in a copy-on-write sandbox so you can review and test changes before applying them.

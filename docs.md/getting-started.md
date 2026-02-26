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

If you prefer hosted inference, set a HuggingFace token and provide a model in `org/model` format.

```sh
export HF_TOKEN=hf_your_token_here
swival "Hello world" --provider huggingface --model zai-org/GLM-5
```

If you are using a dedicated HuggingFace endpoint, pass both `--base-url` and `--api-key`.

```sh
swival "Hello world" \
    --provider huggingface \
    --model zai-org/GLM-5 \
    --base-url https://xyz.endpoints.huggingface.cloud \
    --api-key hf_your_key
```

## Where To Go Next

If you want the full command surface and mode behavior, continue with [Usage](usage.md). If you want a deeper look at built-in capabilities, read [Tools](tools.md). If you need to understand trust boundaries before enabling stronger actions, read [Safety and Sandboxing](safety-and-sandboxing.md). If you want copy-on-write isolation so you can review and apply changes only when ready, read [Using Swival with AgentFS](agentfs.md).

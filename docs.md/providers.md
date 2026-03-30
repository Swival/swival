# Providers

Swival supports local, hosted, and API-based model providers:

- [LM Studio](#lm-studio) — local inference
- [HuggingFace Inference API](#huggingface-inference-api) — hosted inference
- [OpenRouter](#openrouter) — multi-provider access through a single API
- [Generic (OpenAI-compatible)](#generic-openai-compatible) — any OpenAI-compatible server
- [Google Gemini API](#google-gemini-api) — Google's models
- [ChatGPT Plus/Pro](#chatgpt-pluspro) — OpenAI models via your existing subscription
- [AWS Bedrock](#aws-bedrock) — models hosted on AWS
- [Command (External Program)](#command-external-program) — shell out to an external program

## LM Studio

LM Studio is the default provider and usually requires no flags when the local server is already running with a loaded model.

At startup, Swival calls `http://127.0.0.1:1234/api/v1/models` unless you override `--base-url`. It looks for the first model entry with `type: "llm"` and a non-empty `loaded_instances` array, then extracts the model identifier and current context length from that payload. If no loaded model is found, Swival exits and asks you to load a model or pass `--model` explicitly.

If LM Studio is running on another host or port, set `--base-url`.

```sh
swival --base-url http://192.168.1.100:1234 "task"
```

If you want to bypass auto-discovery, pass `--model`.

```sh
swival --model "qwen3-coder-next" "task"
```

If you pass `--max-context-tokens`, Swival may reload the model through LM Studio's `/api/v1/models/load` endpoint.

```sh
swival --max-context-tokens 131072 "task"
```

If the requested value already matches the loaded context length, no reload happens.

When a reload is required, it can take noticeable time depending on model size and hardware.

## HuggingFace Inference API

For HuggingFace, `--model` is required and must be in `org/model` format. Authentication comes from `HF_TOKEN` by default or `--api-key` if you pass one explicitly.

```sh
export HF_TOKEN=hf_your_token_here
swival --provider huggingface --model zai-org/GLM-5 "task"
```

Serverless HuggingFace endpoints often expose smaller context windows than local deployments, so long multi-turn coding sessions can hit context pressure sooner.

For dedicated endpoints, keep the same model identifier and pass your endpoint URL and key.

```sh
swival --provider huggingface \
    --model zai-org/GLM-5 \
    --base-url https://xyz.endpoints.huggingface.cloud \
    --api-key hf_your_key \
    "task"
```

### HuggingFace Inference Endpoints

HuggingFace [Inference Endpoints](https://huggingface.co/inference-endpoints) let you deploy any supported model on dedicated infrastructure. Create an endpoint from the HuggingFace UI, then point Swival at it with `--base-url`.

```sh
swival --provider huggingface \
    --model Qwen/Qwen3.5-35B-A3B \
    --base-url https://tfg1ghx03o7xuv5p.us-east-1.aws.endpoints.huggingface.cloud
```

Most inference endpoints use vLLM as the serving backend. For tool calling to work, you must add the following to the **Container Arguments** field in your endpoint's configuration on HuggingFace:

```text
--enable-auto-tool-choice --tool-call-parser qwen3_xml
```

The `--tool-call-parser` value depends on the model you deploy. For Qwen models use `qwen3_xml`, for other model families check the [vLLM tool calling documentation](https://docs.vllm.ai/en/latest/features/tool_calling.html) for the correct parser name. Without these arguments, the endpoint will not return structured tool calls and Swival will not be able to use its tools.

For recently released models, the default vLLM version configured in Inference Endpoints may not support them yet. If you hit errors during model loading, set the **Engine URI** in your endpoint configuration to `vllm/vllm-openai:latest` to use a more recent build.

Some models served through vLLM leak internal reasoning markers like `<think>` tags into their responses. If you see these in the output, enable `--sanitize-thinking` to strip them. See [Thinking Tag Sanitization](customization.md#thinking-tag-sanitization) for details.

Dedicated endpoints usually let you use the full deployed model context window rather than tighter serverless limits.

## OpenRouter

For OpenRouter, `--model` is required and authentication comes from `OPENROUTER_API_KEY` or `--api-key`.

```sh
export OPENROUTER_API_KEY=sk_or_your_token_here
swival --provider openrouter --model z-ai/glm-5 "task"
```

If you use an OpenRouter-compatible custom endpoint, set `--base-url`.

```sh
swival --provider openrouter \
    --model z-ai/glm-5 \
    --base-url https://custom.openrouter.endpoint \
    --api-key sk_or_key \
    "task"
```

OpenRouter models vary widely in context limits, so you should set `--max-context-tokens` to match the model you chose.

```sh
swival --provider openrouter --model z-ai/glm-5 \
    --max-context-tokens 131072 "task"
```

Pass bare model identifiers like `z-ai/glm-5`. If you accidentally include a provider prefix (e.g. `openrouter/z-ai/glm-5`), Swival detects and corrects the double prefix.

## Generic (OpenAI-compatible)

The generic provider works with any server that exposes an OpenAI-compatible chat completions endpoint. This covers mlx_lm.server, ollama, llama.cpp, vLLM, LocalAI, text-generation-webui, DeepSeek API, and similar tools.

Both `--model` and `--base-url` are required. Pass the server's root URL without `/v1` — Swival appends it automatically. If your URL already ends in `/v1`, that's fine too.

```sh
# mlx_lm.server
swival --provider generic \
    --base-url http://127.0.0.1:8080 \
    --model mlx-community/Qwen3-Coder-480B-A35B-4bit \
    "task"
```

```sh
# ollama
swival --provider generic \
    --base-url http://127.0.0.1:11434 \
    --model qwen3:32b \
    "task"
```

```sh
# llama.cpp server
swival --provider generic \
    --base-url http://127.0.0.1:8080 \
    --model default \
    "task"
```

```sh
# DeepSeek API
export DEEPSEEK_API_KEY=sk-...
swival --provider generic \
    --base-url https://api.deepseek.com \
    --model deepseek-chat \
    --api-key "$DEEPSEEK_API_KEY" \
    --max-output-tokens 8192 \
    "task"
```

No API key is required for most local servers. If your server needs one, pass `--api-key` or set `OPENAI_API_KEY`.

```sh
export OPENAI_API_KEY=sk-...
swival --provider generic \
    --base-url https://my-server.example.com \
    --model my-model \
    "task"
```

There is no model auto-discovery and no context window reload. Set `--max-context-tokens` manually if you need Swival to know the window size.

## Google Gemini API

The `google` provider connects to Google's Gemini API through its OpenAI-compatible endpoint (`/v1beta/openai`).

`--model` is required. Authentication comes from `--api-key`, `GEMINI_API_KEY`, or `OPENAI_API_KEY`.

```sh
export GEMINI_API_KEY=...
swival --provider google \
    --model gemini-2.5-flash \
    "task"
```

When `--max-context-tokens` is not set, Swival auto-detects the context window from the model's known limits. If detection fails, context length is unknown and compaction may not trigger at the right time — set `--max-context-tokens` explicitly if you hit issues.

`--base-url` overrides the default endpoint if you need a custom one.

## ChatGPT Plus/Pro

The `chatgpt` provider lets you use OpenAI models through your existing ChatGPT Plus or ChatGPT Pro subscription, without needing a separate API key.

Authentication uses an OAuth device-code flow — on first use, Swival prints a device code and a verification URL to your terminal. Open the URL, enter the code, and authorize with your ChatGPT account. The resulting tokens are cached locally and refreshed automatically on subsequent runs.

If you need to pass an API key explicitly (for example, when using `--self-review` which passes credentials via environment variables), set `CHATGPT_API_KEY` or use `--api-key`.

`--model` is required. There is no default model.

```sh
swival --provider chatgpt --model gpt-5.4 "task"
```

On the first run, you will see a device-code prompt with a URL and a code to enter in your browser. Once you complete the flow, the OAuth tokens are stored locally and refreshed automatically.

Supported model names may change over time. Check OpenAI's documentation for the current model list and naming conventions.

Two environment variables are available for advanced use. `CHATGPT_TOKEN_DIR` overrides the default token storage directory. `CHATGPT_API_BASE` overrides the API base URL.

```sh
export CHATGPT_TOKEN_DIR=/path/to/tokens
swival --provider chatgpt --model gpt-5.4 "task"
```

The `--top-p`, `--seed`, and `tool_choice` parameters are not supported by the ChatGPT Plus/Pro backend. Swival drops them automatically when using this provider.

Models like `gpt-5.4` support tunable reasoning effort. Use `--reasoning-effort` to control how much the model thinks before responding:

```sh
swival --provider chatgpt --model gpt-5.4 --reasoning-effort high "task"
```

No other configuration is needed.

## AWS Bedrock

The `bedrock` provider connects to AWS Bedrock. It uses your existing AWS credentials — `--api-key` is not supported.

`--model` is required. Use the Bedrock model ID.

```sh
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION_NAME=us-east-2
swival --provider bedrock --model global.anthropic.claude-opus-4-6-v1 "task"
```

Swival picks up credentials from environment variables, `~/.aws/credentials`, `~/.aws/config`, IAM roles, and SSO. To use a named profile, pass `--aws-profile` or set `AWS_PROFILE`.

```sh
swival --provider bedrock --model global.anthropic.claude-opus-4-6-v1 \
    --aws-profile bedrock "task"
```

Use `--base-url` to set the region, or pass a custom endpoint URL.

```sh
swival --provider bedrock --model global.anthropic.claude-opus-4-6-v1 \
    --base-url us-east-2 "task"
```

Note: the region env var is `AWS_REGION_NAME`, not `AWS_DEFAULT_REGION`.

For cross-region inference, use the regional prefix in the model ID (e.g. `us.`, `eu.`, `apac.`).

```sh
swival --provider bedrock --model us.anthropic.claude-opus-4-6-v1 "task"
```

## Extra Provider Parameters

Some models and servers accept parameters that go beyond the standard OpenAI API. Use `--extra-body` to pass them through. The value is a JSON object that gets forwarded directly to the API call.

For example, Qwen models served through vLLM can disable internal thinking mode:

```sh
swival --provider generic \
    --base-url http://127.0.0.1:8000 \
    --model Qwen/Qwen3.5-35B-A3B \
    --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}' \
    "task"
```

You can also set this in config so you don't repeat it every time:

```toml
provider = "generic"
base_url = "http://127.0.0.1:8000"
model = "Qwen/Qwen3.5-35B-A3B"
extra_body = { chat_template_kwargs = { enable_thinking = false } }
```

The dictionary is forwarded as `extra_body` to the provider's API. Refer to your model or server documentation for supported parameters.

When using vLLM as the inference backend, models may leak internal reasoning markers like `<think>` tags into their output even with thinking disabled. Use `--sanitize-thinking` to strip them. See [Thinking Tag Sanitization](customization.md#thinking-tag-sanitization) for details.

For reasoning effort specifically, Swival provides a dedicated `--reasoning-effort` flag instead of requiring `extra_body`. See [Customization](customization.md) for details.

## Command (External Program)

The `command` provider shells out to an external program instead of calling an API. This is useful when you want to wrap an existing CLI tool — such as `codex exec --skip-git-repo-check --full-auto --ephemeral`, `ollama run`, or a custom script — as Swival's backend.

The conversation transcript is written to the program's stdin, and the program's stdout is read back as the model response. `--model` holds the command string, which is split with `shlex`. `--base-url` and `--api-key` are ignored.

Tool calling is supported when command execution is enabled (the default, or via `--commands`). The external program can request tool execution by emitting `<swival:call>` XML blocks in its output. Swival parses these, dispatches the tool calls, appends results to the transcript, and re-invokes the command. This loop continues (up to 20 rounds) until the program responds without tool calls. When no commands are whitelisted, the command provider runs without tools.

```sh
swival --provider command --model "codex exec --skip-git-repo-check --full-auto --ephemeral" "task"
```

Or in config:

```toml
provider = "command"
model = "codex exec --skip-git-repo-check --full-auto --ephemeral"
```

Because the external program handles all model routing, there is no auto-discovery. If you set `--max-context-tokens`, Swival will apply output clamping and graduated compaction as usual; otherwise context management is left to the command itself.

## Prompt Caching

For providers that support explicit cache annotations, Swival automatically marks the system message as cacheable each turn. This avoids re-processing the system prompt and tool schemas on every call, which typically saves 30–60% of input token costs in long sessions.

| Provider                     | Caching mechanism                           | Notes                                                           |
| ---------------------------- | ------------------------------------------- | --------------------------------------------------------------- |
| Anthropic (via OpenRouter)   | Explicit `cache_control` injected by Swival | System message cached; tool schemas not cached in Phase 1       |
| Google Gemini                | Explicit `cache_control` injected by Swival | Via `openrouter/google/...` or native `google` provider         |
| AWS Bedrock                  | Explicit `cache_control` injected by Swival | Supported for Anthropic models on Bedrock                       |
| OpenAI / Deepseek            | Automatic (provider-side)                   | No annotation needed; prompts >1024 tokens cached automatically |
| LM Studio                    | None                                        | Local inference, no server-side cache                           |
| Generic with custom base_url | None                                        | Swival cannot identify the upstream provider from the URL alone |

Cache annotation is applied automatically when the model is known to support it. If the call succeeds with an unexpected provider, the extra annotation is silently ignored.

When `--verbose` is active, Swival prints cache hit and write stats to stderr after each turn:

```text
Prompt cache: 4821 tokens cached
Prompt cache: 6103 tokens written to cache
```

To opt out of explicit cache annotations, pass `--no-prompt-cache`. This only suppresses the injection; providers that cache automatically are unaffected.

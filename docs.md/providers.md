# Providers

Swival supports LM Studio for local inference, HuggingFace Inference API for hosted inference, OpenRouter for multi-provider access through a single API, a ChatGPT Plus/Pro provider for using OpenAI models through your existing subscription via OAuth, and a generic provider for any OpenAI-compatible server. All provider calls are normalized through [LiteLLM](https://docs.litellm.ai/), so the runtime loop stays consistent while credential and model routing change per provider.

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

Internally, LM Studio calls are routed through LiteLLM as an OpenAI-compatible endpoint. Swival sends the model as `openai/<model_id>`, sets `api_base` to `<base_url>/v1`, and uses the placeholder API key `lm-studio`.

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
    --base-url https://tfg1ghx03o7xuv5p.us-east-1.aws.endpoints.huggingface.cloud \
    --repl
```

Most inference endpoints use vLLM as the serving backend. For tool calling to work, you must add the following to the **Container Arguments** field in your endpoint's configuration on HuggingFace:

```
--enable-auto-tool-choice --tool-call-parser qwen3_xml
```

The `--tool-call-parser` value depends on the model you deploy. For Qwen models use `qwen3_xml`, for other model families check the [vLLM tool calling documentation](https://docs.vllm.ai/en/latest/features/tool_calling.html) for the correct parser name. Without these arguments, the endpoint will not return structured tool calls and Swival will not be able to use its tools.

For recently released models, the default vLLM version configured in Inference Endpoints may not support them yet. If you hit errors during model loading, set the **Engine URI** in your endpoint configuration to `vllm/vllm-openai:latest` to use a more recent build.

Internally, Swival normalizes the model to `huggingface/<model_id>` for LiteLLM and strips an existing `huggingface/` prefix if you already included it. If `--base-url` is set, it is forwarded as `api_base`.

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

Internally, Swival normalizes OpenRouter models to LiteLLM's `openrouter/...` format. If the model identifier starts with the literal double prefix `openrouter/openrouter/`, Swival strips the redundant prefix so LiteLLM does not see a stutter.

Any other `openrouter/` prefix (e.g. `openrouter/z-ai/glm-5`) is treated as part of the model path and gets prefixed normally, so you should pass bare identifiers like `z-ai/glm-5`.

## Generic (OpenAI-compatible)

The generic provider works with any server that exposes an OpenAI-compatible chat completions endpoint. This covers mlx_lm.server, ollama, llama.cpp, vLLM, LocalAI, text-generation-webui, and similar tools.

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

No API key is required for most local servers. If your server needs one, pass `--api-key` or set `OPENAI_API_KEY`.

```sh
export OPENAI_API_KEY=sk-...
swival --provider generic \
    --base-url https://my-server.example.com \
    --model my-model \
    "task"
```

There is no model auto-discovery and no context window reload. Set `--max-context-tokens` manually if you need Swival to know the window size.

Internally, generic calls are routed through LiteLLM as `openai/<model_id>` with `api_base` pointing at your server's `/v1` path.

## Google Gemini API

The `google` provider connects to Google Gemini API through LiteLLM's native Gemini support.

`--model` is required. Authentication comes from `GEMINI_API_KEY`, `OPENAI_API_KEY`, or `--api-key`.

```sh
export GEMINI_API_KEY=...
swival --provider google \
    --model gemini-2.5-flash \
    "task"
```

When `--max-context-tokens` is not set, Swival auto-detects the context window via `litellm.get_model_info()`. If detection fails, context length is unknown and compaction may not trigger at the right time — set `--max-context-tokens` explicitly if you hit issues.

Internally, the model is normalized to `gemini/<model_id>` and routed through LiteLLM's native Gemini provider. `--base-url` is optional and only needed if you are targeting a custom endpoint.

## ChatGPT Plus/Pro

The `chatgpt` provider lets you use OpenAI models through your existing ChatGPT Plus or ChatGPT Pro subscription, without needing a separate API key.

Authentication uses an OAuth device-code flow handled by LiteLLM -- on first use, LiteLLM prints a device code and a verification URL to your terminal. Open the URL, enter the code, and authorize with your ChatGPT account. The resulting tokens are cached locally and refreshed automatically on subsequent runs.

If you need to pass an API key explicitly (for example, when using `--self-review` which passes credentials via environment variables), set `CHATGPT_API_KEY` or use `--api-key`.

`--model` is required. There is no default model.

```sh
swival --provider chatgpt --model gpt-5.4 "task"
```

On the first run, you will see a device-code prompt with a URL and a code to enter in your browser. Once you complete the flow, the OAuth tokens are stored at `~/.config/litellm/chatgpt/auth.json` and refreshed automatically.

Currently available models are `gpt-5.4`, `gpt-5.3-codex`, and `gpt-5.3-codex-spark`. Use whichever your ChatGPT Plus or Pro plan has access to.

```sh
swival --provider chatgpt --model gpt-5.4 "task"
```

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

All OAuth handling happens inside LiteLLM. Swival normalizes the model to `chatgpt/<model_id>` and passes it through. No other configuration is needed.

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

The dictionary is forwarded as `extra_body` to LiteLLM, which passes it through to the server. Refer to your model or server documentation for supported parameters.

For reasoning effort specifically, Swival provides a dedicated `--reasoning-effort` flag instead of requiring `extra_body`. See [Customization](customization.md) for details.

## Adding More Providers Later

Because API calls are already abstracted behind LiteLLM, adding a provider is mostly a matter of argument validation, model normalization, and credential wiring. The provider-specific branch in `call_llm()` is intentionally compact so new providers can be added without changing the rest of the agent loop.

In practice, each provider branch is only about ten lines of routing logic.

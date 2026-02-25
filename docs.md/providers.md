# Providers

Swival supports three providers: LM Studio (local), HuggingFace Inference API
(hosted), and OpenRouter (multi-provider API). Under the hood, all LLM calls go
through [LiteLLM](https://docs.litellm.ai/), which normalizes the API
differences.

## LM Studio

This is the default and requires no configuration beyond having LM Studio
running with a model loaded.

### Auto-discovery

Swival queries `http://127.0.0.1:1234/api/v1/models` at startup to find the
loaded model. It looks for the first entry with `type: "llm"` and a non-empty
`loaded_instances` array. The model identifier and context length are extracted
automatically.

If no model is loaded, Swival exits with an error telling you to load one.

### Custom base URL

If LM Studio is running on a different host or port:

```sh
swival --base-url http://192.168.1.100:1234 "task"
```

### Manual model selection

If auto-discovery doesn't find the right model (e.g., multiple models loaded),
you can specify it:

```sh
swival --model "qwen3-coder-next" "task"
```

### Context size configuration

You can request a specific context length, which may trigger LM Studio to
reload the model:

```sh
swival --max-context-tokens 131072 "task"
```

Swival calls LM Studio's `/api/v1/models/load` endpoint with the new context
size. If the requested size matches what's already loaded, no reload happens.
Reloads can be slow depending on the model and hardware.

### How the LiteLLM call works

For LM Studio, Swival prefixes the model identifier with `openai/` and sets
`api_base` to `{base_url}/v1`. The API key is set to `"lm-studio"` (LM Studio
doesn't require a real key). This tells LiteLLM to use the OpenAI-compatible
API format.

## HuggingFace Inference API

For hosted inference without running a local model.

### Basic usage

```sh
export HF_TOKEN=hf_your_token_here
swival --provider huggingface --model zai-org/GLM-5 "task"
```

The `--model` flag is required and must be in `org/model` format. Authentication
comes from `HF_TOKEN` in the environment or `--api-key` on the command line
(which takes precedence).

Serverless endpoints typically have a 32k token context limit, which can be
restrictive for agentic workloads that accumulate
tool calls and file contents over many turns. If you're hitting context limits,
consider a dedicated endpoint instead.

### Dedicated endpoints

For HuggingFace dedicated inference endpoints (private deployments):

```sh
swival --provider huggingface \
    --model zai-org/GLM-5 \
    --base-url https://xyz.endpoints.huggingface.cloud \
    --api-key hf_your_key \
    "task"
```

The `--base-url` points to your endpoint. The model identifier still needs to
match what's deployed there. Dedicated endpoints don't have the context size
limits of serverless -- you control the deployment, so the full model context
window is available.

### How the LiteLLM call works

For HuggingFace, Swival prefixes the model with `huggingface/` (stripping any
existing prefix first) and passes the API key directly. If `--base-url` is set,
it's passed as `api_base` to LiteLLM.

## OpenRouter

For access to dozens of models from different providers (OpenAI, Anthropic,
Meta, Google, and others) through a single API.

### Basic usage

```sh
export OPENROUTER_API_KEY=sk_or_your_token_here
swival --provider openrouter --model openrouter/free "task"
```

The `--model` flag is required. Use the model identifier from OpenRouter's
catalog (e.g., `openrouter/free`, `anthropic/claude-3-5-sonnet`,
`openrouter/free`). Authentication comes from
`OPENROUTER_API_KEY` in the environment or `--api-key` on the command line
(which takes precedence).

### Custom base URL

For custom OpenRouter-compatible endpoints:

```sh
swival --provider openrouter \
    --model openrouter/free \
    --base-url https://custom.openrouter.endpoint \
    --api-key sk_or_key \
    "task"
```

### Context size

OpenRouter models vary widely in context size (8K to 200K+ tokens). Swival
defaults to whatever `--max-context-tokens` is set to; use it to match your
model's actual limit:

```sh
swival --provider openrouter --model openrouter/free \
    --max-context-tokens 131072 "task"
```

### How the LiteLLM call works

For OpenRouter, Swival prefixes the model with `openrouter/` (stripping any
existing prefix first) and passes the API key directly. If `--base-url` is set,
it's passed as `api_base` to LiteLLM.

## Future providers

Since Swival uses LiteLLM for the actual API call, adding new providers is
straightforward -- it's mostly a matter of building the right model string and
passing the right credentials. The provider-specific logic in `call_llm()` is
about 10 lines per provider.

OpenRouter already provides access to dozens of models from different providers
through a single API, so it's a good option if you need flexibility without
configuring multiple credentials.

# Providers

Swival supports two providers: LM Studio (local) and HuggingFace Inference API
(hosted). Under the hood, all LLM calls go through
[LiteLLM](https://docs.litellm.ai/), which normalizes the API differences.

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
swival --provider huggingface --model Qwen/Qwen3.5-397B-A17B "task"
```

The `--model` flag is required and must be in `org/model` format. Authentication
comes from `HF_TOKEN` in the environment or `--api-key` on the command line
(which takes precedence).

### Dedicated endpoints

For HuggingFace dedicated inference endpoints (private deployments):

```sh
swival --provider huggingface \
    --model Qwen/Qwen3.5-397B-A17B \
    --base-url https://xyz.endpoints.huggingface.cloud \
    --api-key hf_your_key \
    "task"
```

The `--base-url` points to your endpoint. The model identifier still needs to
match what's deployed there.

### How the LiteLLM call works

For HuggingFace, Swival prefixes the model with `huggingface/` (stripping any
existing prefix first) and passes the API key directly. If `--base-url` is set,
it's passed as `api_base` to LiteLLM.

## Future providers

Since Swival uses LiteLLM for the actual API call, adding new providers is
straightforward -- it's mostly a matter of building the right model string and
passing the right credentials. The provider-specific logic in `call_llm()` is
about 10 lines per provider.

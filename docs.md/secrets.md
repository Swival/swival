# Secret Encryption

When sending messages to remote LLM providers, secrets like API keys and tokens can appear in user messages, tool results, or system prompts. Swival's secret encryption replaces recognized credential tokens with realistic-looking fakes before they leave your machine. The LLM can still reason about them — it sees a plausible `ghp_...` token, not garbled ciphertext — and Swival decrypts them back to real values before tool dispatch or final output.

The encryption is format-preserving: a GitHub PAT stays the same length and keeps its `ghp_` prefix, so the model treats it like a normal token. A provenance registry tracks which ciphertext values Swival actually emitted, so only those are decrypted on the way back. Model-invented token-shaped strings pass through untouched.

## Enabling Encryption

On the command line:

```sh
swival --encrypt-secrets "deploy using the token in .env"
```

Or in config:

```toml
encrypt_secrets = true
```

Or in the library API:

```python
from swival import Session

session = Session(encrypt_secrets=True)
result = session.run("read the API key from .env and use it")
```

Encryption is off by default.

## How It Works

The encryption pipeline has two phases.

**Outbound (before the LLM call):** Swival deep-copies the message list, scans all `system`, `user`, `tool`, and `assistant` message content for recognized token patterns, encrypts each match in place using format-preserving encryption, and records a ciphertext-to-plaintext mapping in the session registry. The original messages are not modified. When encryption is active, response caching is automatically disabled to avoid cross-session cache key conflicts.

**Inbound (after the LLM response):** Swival scans the model's response content and tool call arguments for any ciphertext strings present in the registry, and replaces them with the original plaintext. Only registry-tracked ciphertext is decrypted — if the model invents a token-shaped string, it passes through unchanged.

## Encryption Keys

By default, a random 256-bit key is generated each session and discarded on exit. This means ciphertext is ephemeral and not reproducible across runs.

To use a persistent key (for example, for stable ciphertext across sessions or for debugging):

```sh
swival --encrypt-secrets --encrypt-secrets-key "$(openssl rand -hex 32)" "task"
```

Or in config:

```toml
encrypt_secrets = true
encrypt_secrets_key = "aabbccdd..."   # 64 hex chars = 32 bytes
```

Or in the library API:

```python
session = Session(encrypt_secrets=True, encrypt_secrets_key="aabbccdd...")
```

## Built-In Token Patterns

Swival recognizes the following token types out of the box:

| Pattern                   | Prefix          | Example                                          |
| ------------------------- | --------------- | ------------------------------------------------ |
| GitHub PAT                | `ghp_`          | `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`       |
| GitHub OAuth              | `gho_`          | `gho_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`       |
| GitHub User-to-Server     | `ghu_`          | `ghu_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`       |
| GitHub Server-to-Server   | `ghs_`          | `ghs_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`       |
| GitHub Refresh            | `ghr_`          | `ghr_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`       |
| OpenAI (project)          | `sk-proj-`      | `sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`       |
| OpenAI (legacy)           | `sk-`           | `sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`    |
| Anthropic                 | `sk-ant-api03-` | `sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`  |
| AWS Access Key            | `AKIA`          | `AKIAIOSFODNN7EXAMPLE`                           |
| AWS Secret Key            | (heuristic)     | 40-char base64 strings near AWS context          |
| Google API                | `AIza`          | `AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`        |
| HuggingFace               | `hf_`           | `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`        |
| Stripe Secret (live)      | `sk_live_`      | `sk_live_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`       |
| Stripe Publishable (live) | `pk_live_`      | `pk_live_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`       |
| Stripe Secret (test)      | `sk_test_`      | `sk_test_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`       |
| Stripe Publishable (test) | `pk_test_`      | `pk_test_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`       |
| Slack Bot                 | `xoxb-`         | `xoxb-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`      |
| Slack User                | `xoxp-`         | `xoxp-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`      |
| Vercel                    | `vercel_`       | `vercel_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`    |
| GitLab PAT                | `glpat-`        | `glpat-xxxxxxxxxxxxxxxxxxxx`                     |
| Datadog                   | `ddapi_`        | `ddapi_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`         |
| PyPI                      | `pypi-`         | `pypi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`          |
| npm                       | `npm_`          | `npm_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`       |
| Supabase                  | `sbp_`          | `sbp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`           |
| Grafana                   | `glc_`          | `glc_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`           |
| SendGrid                  | `SG.`           | `SG.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`            |
| Twilio                    | `SK`            | `SKxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`             |
| Fastly                    | (heuristic)     | 32-char alphanumeric strings near Fastly context |

## Custom Patterns

If your project uses credential formats that aren't covered by the built-in list, add custom patterns in config:

```toml
[[encrypt_secrets_patterns]]
name = "myapp-key"
prefix = "myapp_"
body_regex = "[A-Za-z0-9]{32}"

[[encrypt_secrets_patterns]]
name = "internal-token"
prefix = "int_tok_"
body_regex = "[A-Za-z0-9_-]{40,60}"
```

Each pattern requires a `name` field. The `prefix` and `body_regex` fields define what to match. The body alphabet defaults to alphanumeric characters and minimum body length defaults to `len(prefix) + 8`.

In the library API:

```python
session = Session(
    encrypt_secrets=True,
    encrypt_secrets_patterns=[
        {"name": "myapp-key", "prefix": "myapp_", "body_regex": "[A-Za-z0-9]{32}"},
    ],
)
```

Project-level `encrypt_secrets_patterns` in `swival.toml` replace global-level patterns entirely (no per-pattern merging), consistent with how `serve_skills` merging works.

## Provider Bypass

The `command` provider (local subprocess) bypasses encryption entirely since it runs locally and there is no remote provider to protect against.

## Reviewer Integration

When `--encrypt-secrets` is active and a reviewer is configured, Swival passes the encryption key to the reviewer subprocess via the `SWIVAL_ENCRYPT_KEY` environment variable. This allows `--self-review` and `swival --reviewer-mode` to decrypt any encrypted tokens that appear in the agent's answer.

If you write a custom reviewer script that needs to handle encrypted tokens, you can read `SWIVAL_ENCRYPT_KEY` from the environment and use it with the `fast-cipher` library directly.

## Threat Model

This feature protects against the LLM provider logging or storing real credentials. When encryption is active, the provider never sees actual token values — only format-preserving fakes that look plausible but decrypt to nothing without the session key.

It does **not** protect against:

- **Unrecognized credential formats.** If a token doesn't match any built-in or custom pattern, it passes through in plaintext. Add custom patterns for internal credential formats.
- **Contextual inference by the model.** The model might infer what a credential is for based on surrounding context (filenames, comments, URLs), even if the token itself is encrypted.
- **Python memory safety.** Plaintext secrets exist in Python process memory during the session. The key material is zeroized on session exit, but Python's garbage collector does not guarantee immediate memory clearing.
- **Local storage.** Secrets may appear in tool results written to `.swival/cmd_output_*.txt` files, history, or continue-here files. Encryption only covers the LLM provider boundary.

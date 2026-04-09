# Outbound LLM Filter

Swival can run a user-defined script before every outbound LLM request. The script receives the message list as JSON, and can modify messages, redact content, or block the request entirely. This is useful for stripping internal URLs, project names, customer identifiers, or any other text that should not reach an external provider.

The filter covers all Swival-managed LLM call paths: normal agent turns, REPL turns, the Session API, and compaction summaries.

## Enabling a Filter

On the command line:

```sh
swival --llm-filter "./scripts/redact.py" "task"
```

In a config file:

```toml
llm_filter = "./scripts/redact.py"
```

In the library API:

```python
from swival import Session

session = Session(llm_filter="./scripts/redact.py")
result = session.run("task")
```

The value is a shell command string. It is split with `shlex.split`, and path-like first tokens — anything starting with `/`, `~`, or containing a `/` (e.g. `./`, `../`, `.rtk/`, `scripts/`) — resolve against the config file's parent directory, consistent with `reviewer` and other command-valued config keys.

## Script Contract

Swival sends a JSON object to the script's stdin:

```json
{
  "provider": "openrouter",
  "model": "qwen/qwen3-coder-next",
  "call_kind": "agent",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me about https://internal.corp.example/secret-project"}
  ],
  "tools": [...]
}
```

`messages` is the exact message list about to be sent. `tools` is included as read-only context so the filter can make informed decisions. `call_kind` is `"agent"` for normal turns and `"summary"` for compaction summaries.

The script writes a JSON object to stdout. Two response shapes are supported:

**Allow (with optional modifications):**

```json
{"messages": [...]}
```

**Block:**

```json
{"allow": false, "reason": "contains internal hostname"}
```

## Behavior Rules

| Condition                        | Result                              |
| -------------------------------- | ----------------------------------- |
| Exit 0 + `{"messages": [...]}`   | Use the returned messages           |
| Exit 0 + `{"allow": false, ...}` | Abort the LLM call, show the reason |
| Non-zero exit                    | Abort the LLM call                  |
| Malformed JSON on stdout         | Abort the LLM call                  |
| Timeout (30 seconds)             | Abort the LLM call                  |

The filter fails closed. If the script errors or rejects the request, Swival does not send anything to the provider. Stderr from the script is forwarded to Swival's stderr for debugging.

When a filter blocks a request, Swival raises a hard error and the agent loop terminates. The block reason is shown to the user as an error message. No synthetic assistant reply is generated and nothing is written to history.

## Returned Message Validation

Swival validates the structure of messages returned by the filter:

- The top-level output must be a JSON object with a `messages` key
- `messages` must be a list of dicts
- Each message must have a `role` field with a valid value (`system`, `user`, `assistant`, or `tool`)
- `tool` messages must have a `tool_call_id` field
- `tool_calls` entries on assistant messages must have an `id` and a `function.name`

If validation fails, the request is aborted.

## Ordering

The filter runs **before** secret encryption. This means the filter sees human-readable plaintext, not encrypted tokens. After filtering, the existing `--encrypt-secrets` pipeline still runs on the filtered copy, so credentials that survive filtering are still protected at the provider boundary.

The filter also runs before the LLM response cache lookup. When a filter is active, caching is automatically disabled to avoid stale responses if the filter script changes.

## Example: Redacting Internal URLs

```python
#!/usr/bin/env python3
import json
import re
import sys

payload = json.load(sys.stdin)
for msg in payload["messages"]:
    content = msg.get("content")
    if isinstance(content, str):
        msg["content"] = re.sub(
            r"https://[^\s]*corp\.example[^\s]*",
            "[internal-url]",
            content,
        )
json.dump({"messages": payload["messages"]}, sys.stdout)
```

## Example: Blocking on Keywords

```python
#!/usr/bin/env python3
import json
import sys

payload = json.load(sys.stdin)
text = json.dumps(payload["messages"])
if "PROJECT_CODENAME" in text:
    json.dump({"allow": False, "reason": "message contains project codename"}, sys.stdout)
else:
    json.dump({"messages": payload["messages"]}, sys.stdout)
```

## Interaction with Other Features

**Secret encryption:** The filter runs first, then encryption. The filter sees plaintext; the provider sees encrypted tokens (if encryption is enabled).

**Command provider:** Filtered messages flow through to the command provider the same way they flow to API-based providers. No separate filter path is needed.

**Compaction summaries:** Internal summarization calls also pass through the filter with `call_kind` set to `"summary"`, so sensitive content is redacted even during context management.

**Cache:** Caching is disabled when a filter is active.

**Reviewer:** The reviewer subprocess is a separate Swival invocation and can use its own `llm_filter` setting independently.

## Limitations

- Only one filter command is supported. If you need multiple filters, chain them inside a wrapper script.
- The filter only covers outbound messages. Model responses are not filtered.
- Tool schemas are included in the payload as read-only context but cannot be modified by the filter.
- The filter is stateless across calls. Each invocation is a fresh subprocess.

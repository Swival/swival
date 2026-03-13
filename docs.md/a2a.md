# A2A

Swival can connect to remote agents via the [Agent-to-Agent (A2A) protocol](https://google.github.io/A2A/) v1.0. When A2A agents are configured, Swival fetches their Agent Cards at startup, discovers their skills, and exposes each skill as a tool in the agent loop.

Unlike MCP tools, which have custom parameter schemas defined by the server, A2A skills always use the same generic shape: a `message` string (required), plus optional `context_id` and `task_id` for multi-turn conversations. The model talks to remote agents in natural language.

Each A2A tool is namespaced as `a2a__<agent_name>__<skill_id>` to avoid collisions with built-in tools and across agents. The model calls them like any other tool. Swival routes the call to the correct agent and returns the result.

If an A2A agent fails to connect at startup, Swival logs a warning and continues without that agent's tools. If an agent fails mid-session, its tools are marked as degraded and return an error message instead of blocking the agent loop.

Tool name collisions across agents cause the colliding agent's tools to be skipped entirely, with a warning.

## Agent Card Discovery

At startup, Swival fetches each agent's Agent Card from `<url>/.well-known/agent-card.json` (or a custom `card_url` if configured). The card declares the agent's name, description, skills, and endpoint URL.

If the card declares skills, each skill becomes a separate tool. If no skills are declared, Swival creates a single generic `ask` tool for that agent.

## Multi-Turn Conversations

A2A supports multi-turn conversations through `contextId`. When an agent returns a result, the response includes a `contextId` that groups related interactions. To continue the conversation, the model passes that `contextId` back in the next call.

Some agents may return an `input-required` state, meaning they need more information before completing the task. In that case, the response includes both a `contextId` and a `taskId`. To resume, the model passes both back in the next call.

During context compaction, Swival preserves these IDs so the model can continue multi-turn interactions even after older messages are dropped.

## TOML Configuration

Add `[a2a_servers.<name>]` tables to `swival.toml`. Each agent needs a `url` pointing to its A2A endpoint.

```toml
[a2a_servers.research-agent]
url = "https://research.example.com"

[a2a_servers.code-review]
url = "https://review.example.com"
auth_type = "bearer"
auth_token = "sk-..."
timeout = 180
```

Agent names must match `[a-zA-Z0-9_-]+` and cannot contain double underscores (since `__` is used as the namespacing separator in tool names).

### Configuration Fields

| Field        | Required | Description                                                                       |
| ------------ | -------- | --------------------------------------------------------------------------------- |
| `url`        | Yes      | Base URL of the A2A agent                                                         |
| `card_url`   | No       | Override for the Agent Card URL (defaults to `<url>/.well-known/agent-card.json`) |
| `auth_type`  | No       | Authentication type: `bearer` or `api_key`                                        |
| `auth_token` | No       | Authentication token or key                                                       |
| `timeout`    | No       | Request timeout in seconds (default: 120)                                         |

### Authentication

Two authentication methods are supported:

Bearer token -- sends `Authorization: Bearer <token>` on all requests:

```toml
[a2a_servers.my-agent]
url = "https://agent.example.com"
auth_type = "bearer"
auth_token = "sk-..."
```

API key -- sends `X-API-Key: <token>` on all requests:

```toml
[a2a_servers.my-agent]
url = "https://agent.example.com"
auth_type = "api_key"
auth_token = "key-..."
```

## Standalone Config File

You can also put A2A configuration in a separate TOML file and pass it with `--a2a-config`:

```sh
swival --a2a-config agents.toml "task"
```

The file uses the same `[a2a_servers.*]` format as `swival.toml`:

```toml
[a2a_servers.research-agent]
url = "https://research.example.com"

[a2a_servers.code-review]
url = "https://review.example.com"
```

When both `--a2a-config` and `swival.toml` define agents, the TOML config takes precedence by agent name.

## Config Precedence

When the project and global config both define A2A agents, project-level agents win by name, and global-only agents are merged in.

`--no-a2a` disables A2A agent connections entirely, even if agents are configured.

## CLI Flags

`--no-a2a` disables A2A agent connections entirely.

`--a2a-config FILE` provides an explicit path to an A2A TOML config file.

## Output Handling

A2A tool outputs are size-guarded the same way as MCP tools. Results up to 20 KB are returned inline. Larger results are saved to `.swival/cmd_output_*.txt` and the model receives a pointer message to use `read_file` for paginated access.

When saving large A2A output to file, Swival preserves continuation metadata (the `[contextId=...]` or `[input-required]` header line) so the model can still continue multi-turn conversations even when the response body is too large for inline display.

Error outputs are kept inline but truncated at 20 KB.

During context compaction, A2A tool results receive structured summaries that preserve `input-required` headers with their `contextId` and `taskId`, so multi-turn state survives compaction.

## Library API

The `Session` class accepts `a2a_servers` as a constructor argument. Pass a dictionary mapping agent names to config dicts:

```python
from swival import Session

session = Session(
    a2a_servers={
        "research-agent": {
            "url": "https://research.example.com",
            "auth_type": "bearer",
            "auth_token": "sk-...",
        }
    }
)
result = session.run("Ask the research agent to summarize recent papers on RAG")
```

Use `Session` as a context manager to ensure A2A connections are cleaned up:

```python
with Session(a2a_servers={"agent": {"url": "https://..."}}) as session:
    result = session.run("task")
```

## Example: Local Documentation Agent

This walkthrough sets up two swival instances — one serving project documentation over A2A, and another querying it. The server agent has access to source code and docs via its base directory, so it can read files and answer questions grounded in the actual codebase.

### The project

Suppose you have a project with an API and some docs:

```
acme-api/
  README.md       # endpoint reference, auth, error codes
  app.py          # FastAPI source
  CHANGELOG.md
  swival.toml     # server config (see below)
```

### Server config

Create a `swival.toml` in the project directory. This tells swival which model to use when serving, and defines how the agent advertises itself to clients:

```toml
model = "qwen3.5-9b"
max_turns = 10
max_output_tokens = 4096

serve_name = "Acme Docs"
serve_description = "Answers questions about the Acme Widget API using project documentation and source code"

[[serve_skills]]
id = "lookup"
name = "Documentation Lookup"
description = "Look up API endpoints, error codes, authentication, and changelog entries"
examples = ["How do I create a widget?", "What error codes can POST /widgets return?"]
```

The server agent's base directory is the project root, so it can `read_file` and `grep` across the docs and source code to answer questions.

### Start the server

From the project directory:

```sh
cd acme-api/
swival --serve --serve-port 9100
```

You can verify the agent card:

```sh
curl -s http://127.0.0.1:9100/.well-known/agent-card.json | python3 -m json.tool
```

### Client config

From wherever you want to run the client, create an A2A config file (`a2a.toml`):

```toml
[a2a_servers.acme-docs]
url = "http://127.0.0.1:9100"
```

### Send a query

```sh
swival --a2a-config a2a.toml "Ask the Acme Docs agent: how do I create a new widget? Include a curl example."
```

The client agent sees `a2a__acme-docs__lookup` as a tool, calls it with the question, and the server agent reads the project files to build an answer. The response comes back with endpoint details, required fields, and a working curl command — all grounded in the actual README and source code, not hallucinated.

### What happens under the hood

1. The client discovers the server's agent card and registers `a2a__acme-docs__lookup` as a tool.
2. The client model decides to call that tool with a natural-language message.
3. The server receives the message, creates a Session with the project directory as its workspace, and runs an agent loop that reads files and builds an answer.
4. The server returns the result as a completed A2A task with a `contextId`.
5. The client model incorporates the answer and presents it to the user.

If the client needs to ask a follow-up, it can pass the `contextId` back to continue the conversation with the same server session.

## Protocol Details

Swival implements the A2A v1.0 JSON-RPC binding. It sends `SendMessage` requests with `returnImmediately=false` (blocking mode) as the primary path. If the server returns a non-terminal task instead of blocking, Swival falls back to polling with `GetTask` using exponential backoff.

Responses can be either Task-shaped (with `id`, `status`, and optional `artifacts`) or Message-shaped (with `role` and `parts`). Swival handles both.

Task states are categorized as:

- Terminal: `completed`, `failed`, `canceled`, `rejected`
- Interrupted: `input-required`, `auth-required`

Terminal tasks return their result immediately. Interrupted tasks return with continuation metadata so the model can resume.

## Serving As An A2A Agent

Swival can also run as an A2A server, exposing a Session as an endpoint that other agents can call. This lets you wrap any swival configuration (provider, model, tools, skills, MCP servers) as a remote A2A agent.

```sh
swival --serve --provider openrouter --model z-ai/glm-5
```

This starts an HTTP server at `0.0.0.0:8080` that accepts A2A JSON-RPC requests and serves an Agent Card at `/.well-known/agent-card.json`.

### Server CLI Flags

`--serve` starts the A2A server instead of running a one-shot task or REPL.

`--serve-host HOST` sets the bind address (default: `0.0.0.0`).

`--serve-port PORT` sets the port (default: `8080`).

`--serve-auth-token TOKEN` enables bearer token authentication. When set, all JSON-RPC requests must include an `Authorization: Bearer <token>` header.

All other flags (`--provider`, `--model`, `--allowed-commands`, `--yolo`, `--mcp-config`, etc.) configure the underlying Session that handles incoming tasks.

### How It Works

Each incoming `SendMessage` request is routed to a Session instance keyed by `contextId`. If no `contextId` is provided, the server generates one. The server uses `Session.ask()` for each message, preserving conversation state across calls within the same context.

Sessions are cleaned up after a configurable TTL (default: 1 hour). If the session limit is reached (default: 100), the least-recently-used session is evicted. Per-context locks ensure sequential processing of messages within the same context.

The server supports three JSON-RPC methods:

- **SendMessage** — sends a message to a session and returns the task result
- **GetTask** — retrieves the current state of a task by ID
- **ListTasks** — lists tasks, optionally filtered by `contextId`

Task outcomes map from Session results: a successful `ask()` produces a `completed` task, an exhausted run with no answer produces `input-required` (needs more information), and an exhausted run with a partial answer or an exception produces `failed`.

### Agent Card

The server auto-generates an Agent Card from the session configuration. The card includes the server name (derived from provider and model), capabilities, and endpoint URL. When `--serve-auth-token` is set, the card declares a bearer security scheme.

#### Custom Name and Description

Override the auto-generated name and description with `--serve-name` and `--serve-description`:

```sh
swival --serve --serve-name "Code Review Bot" \
  --serve-description "Reviews Python code for bugs, security issues, and style"
```

Or in `swival.toml`:

```toml
serve_name = "Code Review Bot"
serve_description = "Reviews Python code for bugs, security issues, and style"
```

#### Defining Skills

Skills tell client agents what your server is good at. Define them in `swival.toml`:

```toml
[[serve_skills]]
id = "review"
name = "Code Review"
description = "Analyze code for correctness, security, and style"
examples = ["Review this pull request", "Check this function for bugs"]

[[serve_skills]]
id = "explain"
name = "Code Explanation"
description = "Explain how a piece of code works"
examples = ["What does this function do?"]
```

Each skill requires an `id` field. The `id` must be a stable identifier that matches `[a-zA-Z0-9_-]+` with no double underscores or leading/trailing `_-` characters. The `name`, `description`, and `examples` fields are optional but recommended for client-side routing.

Project-level `serve_skills` in `swival.toml` replace global-level skills entirely (no per-skill merging).

### Library API

You can also create and run the server programmatically:

```python
from swival.a2a_server import A2aServer

server = A2aServer(
    session_kwargs={"provider": "openrouter", "model": "z-ai/glm-5"},
    host="0.0.0.0",
    port=8080,
    auth_token="sk-...",
    name="Code Review Bot",
    description="Reviews Python code for bugs and style",
    skills=[{"id": "review", "name": "Code Review", "description": "Analyze code"}],
)
server.serve()
```

The `A2aServer.app` property returns a Starlette ASGI application, which can be mounted in larger applications or used with any ASGI server.

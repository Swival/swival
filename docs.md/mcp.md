# MCP

Swival can connect to external tool servers via the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP). When MCP servers are configured, their tools are discovered at startup, converted to the same function-calling format as built-in tools, and exposed alongside them in the agent loop.

Each MCP tool is namespaced as `mcp__<server_name>__<tool_name>` to avoid collisions with built-in tools and across servers. The model calls them like any other tool. Swival routes the call to the correct server and returns the result as a string following the same conventions as built-in tools.

MCP servers can use stdio transport (local subprocess) or SSE transport (remote HTTP). Both are configured through `swival.toml` or `.mcp.json`.

If an MCP server fails to connect at startup, Swival logs a warning and continues without that server's tools. If a server crashes mid-session, its tools are marked as degraded and return an error message instead of blocking the agent loop. Tool name collisions within a server cause that server's tools to be skipped entirely, with a warning.

When MCP tool schemas consume more than 30% of the context window, Swival warns. At 50%, it iteratively drops the most expensive server's tools until usage is under budget.

## TOML Configuration

Add `[mcp_servers.<name>]` tables to `swival.toml`. Each server needs either `command` (for stdio transport) or `url` (for SSE transport), but not both.

```toml
[mcp_servers.filesystem]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]

[mcp_servers.remote-api]
url = "https://api.example.com/mcp"
headers = { Authorization = "Bearer token123" }
```

Server names must match `[a-zA-Z0-9_-]+` and cannot contain double underscores (since `__` is used as the namespacing separator in tool names).

For stdio servers, `command` is the executable name and `args` is an optional list of arguments. You can also set `env` to pass environment variables to the subprocess.

```toml
[mcp_servers.my-server]
command = "node"
args = ["server.js"]
env = { API_KEY = "secret" }
```

For SSE servers, `url` is the endpoint and `headers` is an optional dictionary of HTTP headers.

## JSON Configuration

Swival also reads `.mcp.json` from the project root directory. This uses the same format as other MCP-compatible tools:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    }
  }
}
```

The JSON format supports the same fields as TOML: `command`, `args`, `env` for stdio, and `url`, `headers` for SSE.

## Config Precedence

When both `swival.toml` and `.mcp.json` define servers, the TOML config takes precedence by server name. Servers defined only in `.mcp.json` are merged in.

The full precedence order is:

`--no-mcp` (disables all) > `--mcp-config FILE` (explicit JSON path) > `swival.toml [mcp_servers.*]` > `.mcp.json`

When the project and global config both define MCP servers, project-level servers win by name, and global-only servers are merged in.

## CLI Flags

`--no-mcp` disables MCP server connections entirely, even if servers are configured in `swival.toml` or `.mcp.json`.

`--mcp-config FILE` provides an explicit path to an MCP JSON config file. When set, this replaces the default `.mcp.json` lookup in the project root.

## Output Handling

MCP tool outputs are size-guarded similarly to `run_command`, with higher thresholds to accommodate richer external tool results. Results up to 20 KB are returned inline.

Larger results are saved to `.swival/cmd_output_*.txt` and the model receives a pointer message telling it to use `read_file` for paginated access. Output is hard-capped at 10 MB before writing to disk, so a misbehaving server cannot consume unbounded memory or storage.

Error outputs from MCP tools are kept inline but truncated at 20 KB. Errors are diagnostic, not data the model needs to page through, so they are never saved to file.

During context compaction, MCP tool results receive head-preserving summaries that retain the first 300 characters of content, unlike the generic fallback which discards content entirely.

## Library API

The `Session` class accepts `mcp_servers` as a constructor argument. Pass a dictionary mapping server names to config dicts:

```python
from swival import Session

session = Session(
    mcp_servers={
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        }
    }
)
result = session.run("List files in /tmp")
```

Use `Session` as a context manager to ensure MCP connections are cleaned up:

```python
with Session(mcp_servers={"fs": {"command": "..."}}) as session:
    result = session.run("task")
```

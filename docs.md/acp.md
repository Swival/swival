# ACP

Swival speaks the [Agent Client Protocol](https://agentclientprotocol.com) on `stdio`, so editors that act as ACP clients (Zed, the `agent-client-protocol.nvim` plugin, and similar) can drive a swival agent the same way they drive Claude Code, Gemini CLI, or codex.

The editor is the client; swival is the agent. Pass `--acp` and swival reads newline-delimited JSON-RPC 2.0 requests on stdin and writes responses and notifications on stdout. Nothing else gets written to stdout, ever, so the JSON-RPC framing stays intact.

## Quick start

Launch swival in ACP mode against your usual provider:

```sh
swival --acp --provider lmstudio
```

There is no question argument. The editor opens sessions and sends prompts; swival replies with text and tool-call updates as the agent loop runs.

If you accidentally launch this from a real terminal, swival prints a warning to stderr because nothing will read its stdout. Pipe stdout into a JSON-RPC client, or use the editor configurations below.

To capture the wire traffic and any internal diagnostics for debugging, add `--acp-log`:

```sh
swival --acp --acp-log /tmp/swival-acp.log
```

The log file is the only place swival writes anything other than JSON-RPC frames.

## Editor setup

### Zed

Add an `agent_servers` entry to `~/.config/zed/settings.json`:

```json
{
  "agent_servers": {
    "swival": {
      "type": "custom",
      "command": "swival",
      "args": ["--acp"]
    }
  }
}
```

Open the agent panel with `cmd-?`, and then use the `+` button in the top right to start a new Swival thread.

Zed spawns one swival process per agent connection and tears it down when the panel closes. Use any provider/model flags you would normally pass to swival, including `--yolo`, `--files`, `--commands`, `--mcp-config`, and `--a2a-config`. The CLI surface is the same as one-shot mode, with a few exceptions: `--repl`, `--serve`, and a positional question are rejected because they conflict with ACP.

### Neovim (`agent-client-protocol.nvim`)

If you use the `agent-client-protocol.nvim` plugin, register swival as a server in your config:

```lua
require("agent-client-protocol").setup({
  servers = {
    swival = {
      command = "swival",
      args = { "--acp" },
    },
  },
})
```

The exact field names depend on your plugin version; consult its README. The shape is always the same: a command and a list of arguments that produce a process speaking ACP on `stdio`.

## What is currently supported

- `initialize`, with version negotiation and capability echo
- `authenticate` (no-op; provider credentials live in your swival config, not in ACP)
- `session/new` with a working directory
- `session/prompt` for a single text turn, returning when the model finishes or hits the turn limit
- `session/cancel` to interrupt a running prompt; the prompt response then carries `stopReason: "cancelled"`
- `session/update` notifications: `agent_message_chunk` for assistant text, `tool_call` and `tool_call_update` for tool activity (with kinds `read`, `edit`, `execute`, `search`, `think`, `delete`, `fetch`, `other`)

Each ACP session is backed by a swival `Session`, so all your normal swival features are active: skills, MCP servers, A2A clients, the safety policy you chose with `--files` and `--commands`, snapshots, todos, and so on. The editor sees them as tool calls and renders them however it likes.

## Troubleshooting

If the editor says "agent not responding" or hangs on initialize, run swival the same way the editor does and check stderr:

```sh
swival --acp --acp-log /tmp/swival-acp.log < /dev/null
echo "$?"
cat /tmp/swival-acp.log
```

A clean exit with no log entries means swival never received a request. A log with a parse error means the editor is sending something other than newline-delimited JSON-RPC. A log with a Python traceback is a swival bug; please file an issue with the log attached.

If you want to drive swival by hand to confirm the protocol works, three lines of JSON are enough:

```sh
{
  echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{}}}'
  echo '{"jsonrpc":"2.0","id":2,"method":"session/new","params":{"cwd":"'"$PWD"'","mcpServers":[]}}'
  echo '{"jsonrpc":"2.0","id":3,"method":"session/prompt","params":{"sessionId":"REPLACE_AFTER_NEW","prompt":[{"type":"text","text":"hello"}]}}'
} | swival --acp --provider lmstudio
```

You will need to fill in the session id from the `session/new` response before sending the prompt; the easiest way is to do it in two passes or use a small script.

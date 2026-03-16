# Web Browsing

Swival can browse the web, interact with pages, fill forms, take screenshots, and extract content. There are four ways to set this up, each with different tradeoffs.

| Approach                                                      | How it works                                   | Best for                                      |
| ------------------------------------------------------------- | ---------------------------------------------- | --------------------------------------------- |
| [Chrome DevTools MCP](#chrome-devtools-mcp)                   | MCP server controlling Chrome                  | Full browser fidelity, debugging, screenshots |
| [Lightpanda MCP](#lightpanda-mcp)                             | MCP server with a lightweight headless browser | Fast content extraction, scraping, CI/CD      |
| [agent-browser](#agent-browser)                               | CLI tool called via `run_command`              | Token-efficient interactive browsing          |
| [agent-browser + Lightpanda](#using-lightpanda-as-the-engine) | Same CLI, Lightpanda engine                    | Low-resource interactive browsing             |

## Chrome DevTools MCP

[Chrome DevTools MCP](https://github.com/ChromeDevTools/chrome-devtools-mcp) is Google's official MCP server for browser automation. It gives Swival access to the full Chrome DevTools feature set, including navigation, clicking, form filling, screenshots, console messages, network inspection, and performance tracing.

### Setup

Add it to your `swival.toml`:

```toml
[mcp_servers.chrome]
command = "npx"
args = ["-y", "chrome-devtools-mcp@latest"]
```

This launches Chrome with a visible window. For headless operation (no UI), add `--headless`:

```toml
[mcp_servers.chrome]
command = "npx"
args = ["-y", "chrome-devtools-mcp@latest", "--headless"]
```

Requires Node.js v20.19+ and Chrome (stable channel).

### What it gives Swival

Once configured, Swival gets MCP tools like `mcp__chrome__navigate_page`, `mcp__chrome__click`, `mcp__chrome__fill`, `mcp__chrome__take_screenshot`, and `mcp__chrome__list_console_messages`. The model calls them like any other tool.

### Example

```sh
swival --repl
> Navigate to https://news.ycombinator.com, find the top story, and summarize it
```

Swival will call `navigate_page`, then `snapshot` or `get_page_content` to read the page, and return a summary.

### Headless with isolation

For throwaway sessions that leave no browser state behind:

```toml
[mcp_servers.chrome]
command = "npx"
args = ["-y", "chrome-devtools-mcp@latest", "--headless", "--isolated"]
```

The `--isolated` flag creates a temporary profile that gets cleaned up when the session ends.

### Slim mode

If you only need basic navigation and don't want the full DevTools toolset (performance, emulation, etc.), use `--slim`:

```toml
[mcp_servers.chrome]
command = "npx"
args = ["-y", "chrome-devtools-mcp@latest", "--headless", "--slim"]
```

This reduces the number of tools exposed to the model, which saves context window space.

## Lightpanda MCP

[Lightpanda](https://lightpanda.io/) is a headless browser built from scratch for AI agents. It skips pixel rendering entirely and focuses on DOM processing and JavaScript execution via V8. The result is roughly 10x faster page loads and 10x less memory than headless Chrome.

Lightpanda has a built-in MCP server that you can connect to Swival directly. It exposes seven tools focused on content extraction: `goto` for navigation, `markdown` for page content as markdown, `links` for extracting all links, `semantic_tree` for an AI-friendly DOM representation, `interactiveElements` for listing buttons and inputs, `structuredData` for JSON-LD and OpenGraph metadata, and `evaluate` for running JavaScript.

These are read-oriented tools. Lightpanda MCP can navigate pages, read content, and run JavaScript, but it can't click buttons or fill forms. If you need interaction, use [Chrome DevTools MCP](#chrome-devtools-mcp) or [agent-browser](#agent-browser) instead.

### Install Lightpanda

Download the binary for your platform.

**macOS (Apple Silicon):**

```sh
curl -L -o lightpanda https://github.com/lightpanda-io/browser/releases/download/nightly/lightpanda-aarch64-macos
chmod a+x ./lightpanda
sudo mv ./lightpanda /usr/local/bin/
```

**Linux (x86_64):**

```sh
curl -L -o lightpanda https://github.com/lightpanda-io/browser/releases/download/nightly/lightpanda-x86_64-linux
chmod a+x ./lightpanda
sudo mv ./lightpanda /usr/local/bin/
```

Verify it works:

```sh
lightpanda fetch --dump html https://example.com
```

### Setup

Add it to your `swival.toml`:

```toml
[mcp_servers.lightpanda]
command = "lightpanda"
args = ["mcp"]
```

That's the entire setup. No Node.js, no Chrome download, no browser process lingering in the background. Lightpanda starts in milliseconds and uses about 24 MB of memory per instance compared to Chrome's 207 MB.

### What it gives Swival

Once configured, Swival gets tools like `mcp__lightpanda__markdown`, `mcp__lightpanda__links`, `mcp__lightpanda__semantic_tree`, and `mcp__lightpanda__evaluate`. The `markdown` tool is particularly useful since it returns page content in a format that's already compact and easy for the model to reason about.

### Example

```sh
swival --repl
> Fetch https://news.ycombinator.com and summarize the top 5 stories
```

Swival will call `mcp__lightpanda__markdown` with the URL and get back clean markdown content it can work with directly.

### When to use Lightpanda MCP

Lightpanda MCP is the best choice when you just need to read web content. It starts faster, uses less memory, and produces fewer tokens than Chrome DevTools MCP. It works well for documentation lookup, research, scraping, and CI/CD pipelines where you want to keep resource usage low.

The tradeoff is that Lightpanda is still in beta. Most websites work, but you may hit gaps in Web API coverage. It also can't take screenshots or interact with page elements beyond running JavaScript. Use Chrome DevTools MCP when you need full browser fidelity.

### Disabling telemetry

Lightpanda collects usage telemetry by default. To disable it, set this in your environment:

```sh
export LIGHTPANDA_DISABLE_TELEMETRY=true
```

## agent-browser

[agent-browser](https://github.com/vercel-labs/agent-browser) is a CLI by Vercel Labs that controls a browser from the command line. It produces compact text output optimized for AI agents and uses reference-based element selection that burns fewer tokens than DOM selectors.

Instead of connecting via MCP, agent-browser works through Swival's `run_command` tool. You whitelist the `agent-browser` command and the model calls it directly.

### Install

```sh
npm install -g agent-browser
agent-browser install        # downloads Chrome for Testing (first time only)
```

Or with Homebrew on macOS:

```sh
brew install agent-browser
agent-browser install
```

### Configure Swival

Allow the `agent-browser` command in `swival.toml`:

```toml
allowed_commands = ["agent-browser"]
```

Or pass it on the command line:

```sh
swival --allowed-commands agent-browser "Open example.com and tell me what's on the page"
```

### How the model uses it

The model calls `run_command` with agent-browser subcommands. First it opens a URL, then takes a snapshot to get an accessibility tree with element refs like `@e1` and `@e2`, then interacts with those elements, and finally closes the browser when it's done:

```sh
agent-browser open <url>
agent-browser snapshot -i
agent-browser click @e1
agent-browser fill @e2 "search query"
agent-browser screenshot page.png
agent-browser close
```

The snapshot output looks like this:

```text
- none
  - heading "Example Domain" [ref=e1]
    - StaticText "Example Domain"
  - paragraph
    - StaticText "This domain is for use in..."
  - paragraph
    - link "Learn more" [ref=e2]
      - StaticText "Learn more"
```

A typical snapshot is 200-400 tokens compared to 3,000-5,000 for a full DOM dump. This matters when you're browsing multiple pages in a single session.

### Example session

```sh
swival --allowed-commands agent-browser --repl
> Go to https://news.ycombinator.com, click on the top story, and summarize the article
```

The model will run `agent-browser open`, `agent-browser snapshot -i`, `agent-browser click @e1`, and so on, chaining commands until it has the information it needs.

### More commands

agent-browser has 50+ commands. Some useful ones:

```sh
agent-browser get text @e1          # get text content of an element
agent-browser get title             # page title
agent-browser get url               # current URL
agent-browser tab new https://...   # open a new tab
agent-browser tab 2                 # switch tabs
agent-browser scroll down 500       # scroll the page
agent-browser eval "document.title" # run JavaScript
agent-browser pdf report.pdf        # save page as PDF
```

### Using Lightpanda as the engine

agent-browser uses Chrome by default, but you can swap in Lightpanda for faster, lighter execution. Set the `AGENT_BROWSER_ENGINE` environment variable and all commands work the same way:

```sh
export AGENT_BROWSER_ENGINE=lightpanda
```

This requires Lightpanda to be installed on your system. See [Install Lightpanda](#install-lightpanda) above for instructions.

Once the environment variable is set, agent-browser will use Lightpanda transparently:

```sh
agent-browser open https://example.com
agent-browser snapshot -i
agent-browser close
```

You can also pass the engine per-command instead of setting it globally:

```sh
agent-browser --engine lightpanda open https://example.com
```

To use it with Swival, add the environment variable to your shell profile (`.zshrc`, `.bashrc`, etc.) and configure Swival as usual:

```sh
export AGENT_BROWSER_ENGINE=lightpanda
swival --allowed-commands agent-browser "Open example.com and describe the page"
```

Note that Lightpanda doesn't support screenshots, so commands like `agent-browser screenshot` won't work with this engine. Everything else, including snapshots, clicking, form filling, and JavaScript evaluation, works the same as with Chrome.

## Which approach should I use?

Pick **Chrome DevTools MCP** when you need the full browser: screenshots, network inspection, performance profiling, or sites that require complete rendering fidelity. It has the most tools and the best compatibility, but it's also the heaviest.

Pick **Lightpanda MCP** when you just need to read web pages. It's the simplest to set up (one binary, two lines of config), the fastest to start, and the lightest on resources. It can't click or fill forms, but for research, scraping, and documentation lookup it's hard to beat.

Pick **agent-browser** when you need interactive browsing with low token overhead. The ref-based snapshot system keeps context usage down, which helps when you need to visit many pages or do complex multi-step interactions. You can run it with Chrome for full fidelity or with Lightpanda for speed.

You can also combine approaches. For example, use Lightpanda MCP for quick content lookups and Chrome DevTools MCP for tasks that need screenshots or form filling.

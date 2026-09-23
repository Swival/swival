![Swival Logo](.media/logo.png)

# Swival

A coding agent for any model.

Swival makes it easy to work with local models and online services in the same tool.
Smaller models get the same attention as the largest ones, with careful use of the space available for conversation and working notes.
A short setup guide helps connect the model, and switching models doesn't mean changing agents.

For online services, secret encryption can be enabled with a single option to hide supported API keys and tokens before they're sent to the model.
Filters can also be added to remove private details, such as customer names and internal URLs.

Longer jobs don't need constant reminders: Swival keeps notes on what it's learned and what's left to do.
An interrupted session can pick up where it left off, and setting a goal keeps Swival working toward it.

Checking the work is built in too.
A single option adds a review step, so Swival can check its work and try again when the review finds problems.
Another model can handle the review instead, and a built-in security audit checks existing code for bugs.

Swival is free, open source, and written in Python, so the code is available to read and change.
It also connects to other tools and agents and can run as part of a Python program.

The [introduction article](https://00f.net/2026/04/13/swival-ai-agent/) explains the ideas behind Swival.
The full documentation is at [swival.dev](https://swival.dev/).

## Quick start

With [uv](https://docs.astral.sh/uv/), one command installs Swival:

```sh
uv tool install --python 3.14 swival
```

Or with Homebrew on macOS:

```sh
brew trust swival/tap
brew install swival/tap/swival
```

Then, from the project directory:

```sh
swival
```

On the first run, Swival offers a short setup guide.

For a single task, one command is enough:

```sh
swival "Find the cause of the failing tests and fix it"
```

The [setup guide](https://swival.dev/pages/getting-started.html) covers installation and the first run in more detail.
The rest of the documentation is at [swival.dev](https://swival.dev/).

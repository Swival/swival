# Swival

A small, powerful, open-source CLI coding agent that works with open models.

Swival connects to [LM Studio](https://lmstudio.ai/) or
[HuggingFace Inference API](https://huggingface.co/inference-api), sends your
task, and runs an autonomous tool-calling loop until it produces an answer. No
configuration needed.

It auto-discovers your loaded model, gives it sandboxed file access, and gets out of the way.

It's what I use every day. Try it and see if you like it too.

## Quickstart

1. Install [LM Studio](https://lmstudio.ai/) and load a model with tool-calling
   support (I recommend
   [qwen3-coder-next](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF)).
   Crank the context size as high as your hardware allows.
2. Start the LM Studio server.
3. Run:

```sh
uvx swival "Refactor the error handling in src/api.py"
```

That's it. Swival finds the model, connects, and goes to work.

For interactive sessions:

```sh
uvx swival --repl
```

## Documentation

- [Getting Started](docs/getting-started.md) -- installation, first run, what
  happens under the hood
- [Usage](docs/usage.md) -- one-shot mode, REPL mode, CLI flags, piping,
  exit codes
- [Tools](docs/tools.md) -- what the agent can do: file ops, search, editing,
  web fetching, thinking, command execution
- [Safety and Sandboxing](docs/safety-and-sandboxing.md) -- path resolution,
  symlink protection, command whitelisting, YOLO mode
- [Skills](docs/skills.md) -- creating and using SKILL.md-based agent skills
- [Customization](docs/customization.md) -- project instructions, system prompt
  overrides, tuning parameters
- [Providers](docs/providers.md) -- LM Studio and HuggingFace configuration

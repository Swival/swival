# Tool-description rare-parameter corpus

This corpus guards compact tool descriptions against losing uncommon argument
guidance. Run it before trimming built-in tool descriptions or changing
`--tool-descriptions`.

Example:

```sh
uv run python -m swival.benchmark run path/to/bench.toml
```

Use variants that compare `--tool-descriptions full`, `brief`, and
`progressive` against the same pinned model/profile and clean workspaces.

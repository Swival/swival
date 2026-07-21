# Simplification

The `/simplify` command runs three complementary code reviews in parallel, gives their plain-text reports to the main agent, and asks the main agent to apply only small, safe simplifications and validate the result.

```text
/simplify [focus]
```

The focus is free-form text, not a path schema or an option language:

```text
swival> /simplify
swival> /simplify swival/edit.py
swival> /simplify the command parser and its callers
swival> /simplify the code changed for issue 123
```

Without a focus, the reviewers inspect the current workspace broadly. They see the workspace as it exists, including uncommitted changes, and are encouraged to inspect existing implementation code instead of stopping at the first obvious file.

## How It Works

Three read-only subagents start together, each with a non-overlapping job:

- **Local** finds small deletions and direct rewrites within one function or file.
- **Reuse** finds code that an existing helper or an already-computed result can safely replace; it may not invent a new abstraction.
- **Contracts** inspects callers and tests to identify behavior, edge cases, and risks that every proposed change must preserve.

Each reviewer returns at most three ranked findings under short, consistent headings. This explicit structure is easy for smaller models to follow while remaining ordinary prose rather than a machine-parsed protocol.

The reviewers discover the project's languages, layout, build system, and conventions from the workspace. `/simplify` does not classify files by extension, require particular filenames, or restrict the review to committed Git source.

While the reviewers work, a live status line shows each one's turn count and the tool it is currently running, so a long review never looks stalled. As each reviewer completes, a summary line reports how many turns it took, how long it ran, and how much advice it produced. A reviewer that fails is called out with a warning; its report becomes an explicit gap in the advice instead of aborting the command.

Each reviewer returns ordinary prose with concrete locations, reasoning, risks, and suggested validation. There is no JSON response format, candidate schema, parser, deterministic shortlist, or ratifier. A reviewer can report a promising but uncertain idea instead of suppressing it; the main agent decides whether it survives inspection.

After collecting all three reports, the main agent:

1. reads the referenced code and relevant callers or tests;
2. reconciles proposals against the contracts review and project instructions;
3. ranks ideas by safety and code removed;
4. applies only small, high-confidence changes, normally within one file and never as an unrelated cleanup bundle; and
5. runs the formatter, tests, lint, build, or other checks defined by the project.

The main agent is explicitly told not to introduce new abstractions, dependencies, or architectural layers, change public APIs, or weaken tests. A successful run may make no changes. The final answer summarizes applied changes, validation, and notable suggestions that were rejected. The reports are advice, not an instruction to make three changes or to accept every finding.

## Safety and Permissions

Reviewer tool access is enforced as read-only. Their sessions cannot edit files, run project commands, or use the network. They can therefore inspect the live workspace in parallel without racing to change it.

The main agent makes changes through the normal Swival tool and sandbox rules. Read-before-write enforcement, allowed-directory guards, command policy, project instructions, and user approvals work exactly as they do for any other task. `/simplify` does not create a shadow patch or a separate apply prompt; edits remain ordinary working-tree changes for you to review and commit.

The command is programming-language agnostic and does not require a Git repository. It works in the REPL and in one-shot command mode when `--oneshot-commands` is enabled.

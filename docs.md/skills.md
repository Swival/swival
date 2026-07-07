# Skills

Skills are reusable instruction packages that let Swival load detailed guidance only when a task needs it. Instead of injecting every instruction into the base prompt, Swival uses progressive disclosure: the model sees a compact catalog first and loads a specific skill body on demand through `use_skill` or automatically via `$skill-name` mentions in user messages.

## Creating A Skill

A skill lives in its own directory and must include `SKILL.md`. The file begins with YAML frontmatter containing `name` and `description`, followed by the full instruction body.

```
skills/
  deploy/
    SKILL.md
    scripts/
      deploy.sh
```

A typical `SKILL.md` file looks like this:

```markdown
---
name: deploy
description: Deploy the application to production using the deploy script.
---

Run `scripts/deploy.sh` from the skill directory, check the output for errors,
and verify the deployment through the health endpoint.

The deploy script expects `DEPLOY_ENV` to be set. Use `production` for prod
and `staging` for staging.
```

The `name` field must be lowercase alphanumeric with hyphens, must match the directory name, cannot contain leading or trailing hyphens, cannot contain consecutive hyphens, and cannot exceed 64 characters. The `description` field is what the model sees in the catalog and cannot exceed 1,024 characters.

The instruction body after frontmatter can be up to 20,000 characters, and longer bodies are truncated.

## Skill Discovery Locations

Swival checks these locations for skills, in precedence order:

1. `.swival/skills/` — Swival-specific project skills (highest precedence)
2. `.agents/skills/` — common cross-agent standard ([OpenCode](https://opencode.ai/docs/skills), [OpenHands](https://docs.openhands.dev/overview/skills), etc.)
3. `--skills-dir` paths — explicit extra directories
4. `~/.config/swival/skills/` (or `$XDG_CONFIG_HOME/swival/skills/`) — global Swival skills
5. `~/.agents/skills/` — global cross-agent skills (lowest precedence)

Every immediate subdirectory that contains `SKILL.md` is treated as a skill. If the same skill name exists in multiple locations, the first one in the precedence order wins.

Skills in project-local locations are normally shown with file paths in the catalog and need no allowlist entries. The exception is symlinks: if `.agents` or a skill directory symlinks to a path outside the project root, those skills resolve as external and follow external-skill access rules instead.

### Global skills

Place skill directories in `~/.config/swival/skills/` (or `$XDG_CONFIG_HOME/swival/skills/`) for Swival-specific global skills, or in `~/.agents/skills/` for skills shared across agents. Both are scanned automatically.

Global skills have the lowest precedence, so any project-local skill or `--skills-dir` skill with the same name takes priority.

Global skills are typically outside the project, so they resolve as non-local: they don't show file paths in the catalog and are auto-added to the read-only allowlist at setup time. In the unusual case where a global path resolves inside the project (e.g. via symlinks), they behave as local skills instead.

### Extra skill directories

You can add additional skill locations with `--skills-dir` or `skills_dir` in config.

```sh
swival --skills-dir ~/my-skills "task"
```

Each `--skills-dir` path can point directly at one skill directory that contains `SKILL.md`, or at a parent directory where nested subdirectories contain skill files. If duplicate skill names exist across extra directories, first discovery wins. Extra directories override global skills of the same name.

If you do not want skill loading at all, use `--no-skills`.

## Managing Skills From The CLI

The `swival skills` command installs and removes skills so you don't have to copy directories by hand:

```sh
swival skills add    [--global] [--as NAME] [--ref REF] [--force] <name-or-URL>
swival skills delete [--global] [--library] <name>
swival skills list   [--library]
```

There are three places skills can live. Two are active (the agent discovers them); the third is a staging shelf:

- `.swival/skills/` — active, this project.
- `~/.config/swival/skills/` — active, every project.
- `~/.config/swival/library/skills/` — the **library**: a shelf of collections you've downloaded but not necessarily turned on. The agent does not load skills from here.

The quickest path is to point `add` at a git repository, which clones it and installs the skills under its `skills/` directory straight into the current project:

```sh
swival skills add https://github.com/DietrichGebert/ponytail
```

For the full set of options, including staging downloaded collections for review before activating them, pinning a ref, installing globally, and removing skills, see [The Library](library.md).

## How Progressive Disclosure Works

At startup, Swival builds a compact skill catalog that includes names, descriptions, and file paths (for local skills). That catalog is appended to the system prompt under a `## Skills` heading, and the `use_skill` tool is exposed.

### $skill-name mentions (automatic activation)

When a user message contains `$skill-name` (e.g. "please $deploy"), Swival automatically activates matching skills before the model's turn. Each mentioned skill produces a synthetic `use_skill` tool-call/result pair injected into the conversation history. This teaches the model the correct single-skill-per-call API shape while giving it the full instructions without requiring an extra round-trip.

Because injections use assistant+tool messages (not user messages), compaction can shrink or drop them when context pressure grows.

### Manual activation

When the model decides a skill is relevant on its own, it calls `use_skill` with the skill name. Swival reads the full body from `SKILL.md` and returns it inside `<skill-instructions>` tags along with the skill directory path. For local skills, the model can also read the `SKILL.md` file directly using the path shown in the catalog.

### Usage guidance in the prompt

The catalog ends with a short "How to use skills" section: call `use_skill` with the skill name to receive detailed instructions, read local skills directly from the file paths shown, and never search the filesystem for skills that show no path. The `use_skill` tool description also tells the model to use the tool instead of hunting for `SKILL.md` files. This reduces the chance of the model ignoring available skills.

## File Access For External Skills

Project-local skills are already inside normal sandbox roots, so they use standard file access rules. External skills (including global skills) are automatically added as read-only roots at session setup time. That means the model can read helper files under those skill directories with absolute paths, but cannot write into those external skill directories.

## Agent MetaSKILLs

A skill directory can also contain a program file (`SKILL.star`) that turns it into an Agent MetaSKILL — a dynamic workflow that runs bounded loops with nested model calls, command execution, and structured tracing. When a `SKILL.star` file is present (or the `metaskill` frontmatter field points to one), Swival exposes the `run_metaskill` tool alongside `use_skill`.

MetaSKILLs use Starlark as their runtime language and expose three host functions: `ask()`, `command()`, and `trace()`. Execution requires the optional Starlark runtime (installed with `uv tool install 'swival[metaskills]'`); without it, these skills behave as ordinary static skills. Local metaskills execute by default; external metaskills require `--metaskills all` or `metaskills = "all"` in config.

See the [Agent MetaSKILLs specification](metaskills.md) for the full format, host API, budgets, and authoring guide.

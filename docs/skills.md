# Skills

Skills are reusable instruction sets that teach the agent how to handle
specific tasks. Instead of cramming everything into the system prompt, Swival
uses progressive disclosure: the agent sees a compact catalog of available
skills, and loads the full instructions only when it actually needs one.

## Creating a skill

A skill lives in a directory with a `SKILL.md` file. The file starts with YAML
frontmatter that defines the skill's name and description, followed by the full
instructions:

```
skills/
  deploy/
    SKILL.md
    scripts/
      deploy.sh
```

Here's what `skills/deploy/SKILL.md` might look like:

```markdown
---
name: deploy
description: Deploy the application to production using the deploy script.
---

To deploy the application:

1. Run `scripts/deploy.sh` from the skill directory.
2. Check the output for any errors.
3. Verify the deployment by hitting the health endpoint.

The deploy script expects the `DEPLOY_ENV` environment variable to be set.
Use `production` for prod deploys, `staging` for staging.
```

### Frontmatter rules

The frontmatter requires exactly two fields:

- `name`: lowercase alphanumeric with hyphens (like `my-skill`). Must match
  the directory name. Max 64 characters. No leading, trailing, or consecutive
  hyphens.
- `description`: a short summary (max 1024 characters). This is what the agent
  sees in the catalog to decide whether to activate the skill.

The body after the frontmatter closing `---` is the full instruction text. It
can be up to 20,000 characters. Anything beyond that is truncated.

## Where to put skills

Swival looks for skills in two places:

`Project-local skills` live in a `skills/` directory inside the base
directory. Each subdirectory with a `SKILL.md` file is a skill. These take
precedence over external skills.

`External skills` are specified with `--skills-dir`:

```sh
uvx swival --skills-dir ~/my-skills "task"
```

The `--skills-dir` flag is repeatable. Each path can be either a directory
containing `SKILL.md` directly (a single skill) or a parent directory whose
subdirectories contain `SKILL.md` files.

If the same skill name appears in both project-local and external locations,
the project-local one wins. Among external directories, first occurrence wins.

Disable skill discovery entirely with `--no-skills`.

## How progressive disclosure works

At startup, Swival scans for skills and builds a catalog. The catalog -- just
names and descriptions, one line each -- is appended to the system prompt inside
`<available-skills>` tags. The agent also gets a `use_skill` tool.

When the agent encounters a task that matches a skill's description, it calls
`use_skill` with the skill name. Swival loads the full `SKILL.md` body and
returns it as a tool result wrapped in `<skill-instructions>` tags, along with
the skill's directory path.

This means the full instructions only enter the context window when actually
needed. For a project with many skills, this keeps the system prompt lean.

## File access for external skills

Project-local skills (under `base_dir/skills/`) can access files through the
normal sandbox -- they're already inside the base directory.

External skills get their directory added to a read-only allowlist when
activated. The agent can read supporting files in the skill directory (scripts,
templates, examples) using absolute paths, but can't write to them.

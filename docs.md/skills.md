# Skills

Skills are reusable instruction packages that let Swival load detailed guidance only when a task needs it. Instead of injecting every instruction into the base prompt, Swival uses progressive disclosure: the model sees a compact catalog first and loads a specific skill body on demand through `use_skill`.

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

Swival checks project-local skills first under `.swival/skills/` inside the base directory. Every immediate subdirectory that contains `SKILL.md` is treated as a local skill. Local skills take precedence if a name collision happens.

You can also add external skill locations with `--skills-dir`.

```sh
swival --skills-dir ~/my-skills "task"
```

Each `--skills-dir` path can point directly at one skill directory that contains `SKILL.md`, or at a parent directory where nested subdirectories contain skill files. If duplicate skill names exist across external directories, first discovery wins. If both local and external skills define the same name, the local definition wins.

If you do not want skill loading at all, use `--no-skills`.

## How Progressive Disclosure Works

At startup, Swival builds a compact skill catalog that includes only names and descriptions. That catalog is appended to the system prompt inside `<available-skills>` tags, and the `use_skill` tool is exposed.

When the model decides a skill is relevant, it calls `use_skill` with the skill name. Swival then reads the full body from `SKILL.md` and returns it inside `<skill-instructions>` tags along with the skill directory path. This keeps base prompt size small while still allowing deep instructions when needed.

## File Access For External Skills

Project-local skills are already inside normal sandbox roots, so they use standard file access rules. External skills are added as read-only roots when activated. That means the model can read helper files under those skill directories with absolute paths, but cannot write into those external skill directories.

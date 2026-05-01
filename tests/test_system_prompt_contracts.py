"""Golden tests pinning the load-bearing contracts in system_prompt.txt.

These tests intentionally do not assert prompt length or exact wording.
They lock down the *semantic* content that must survive any future
trim or rewrite:

  - Path sandboxing
  - Instruction priority hierarchy and the always-binding safety rule
  - Edit semantics (verbatim old_string, line_number disambiguator,
    one edit per call, no re-read after edit)
  - The literal <learned>...</learned> tag (downstream tooling parses it)
  - Explicit when-to-use triggers for think / todo / snapshot — Swival is
    expected to work with small models, so workflow guidance must remain
    direct and tool-named, not reduced to vague "reason carefully" prose.

Hard contracts use normalized-text regex. Workflow triggers use
co-occurrence within a character window so future rewrites have room
to rephrase without losing the trigger word.
"""

import re

from swival.agent import (
    DEFAULT_SYSTEM_PROMPT_FILE,
    _apply_capability_substitutions,
    _apply_interaction_policy,
)


def _read_prompt(
    policy: str = "autonomous",
    *,
    no_memory: bool = False,
    files_mode: str = "some",
) -> str:
    """Return the assembled prompt the way build_system_prompt() would.

    Defaults match the common-case ("memory on, file tools usable"), so the
    hard-contract tests below see the full default prompt.
    """
    raw = DEFAULT_SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")
    raw = _apply_capability_substitutions(
        raw, no_memory=no_memory, files_mode=files_mode
    )
    return _apply_interaction_policy(raw, policy)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _has_within(text: str, anchor: str, triggers: list[str], window: int = 200) -> bool:
    """True if `anchor` appears within `window` chars of any `trigger` regex."""
    norm = _normalize(text)
    anchor_re = re.escape(anchor.lower())
    for m in re.finditer(anchor_re, norm):
        start = max(0, m.start() - window)
        end = min(len(norm), m.end() + window)
        slice_ = norm[start:end]
        if any(re.search(t, slice_) for t in triggers):
            return True
    return False


# ---------------------------------------------------------------------------
# Hard contracts
# ---------------------------------------------------------------------------


class TestPathSandboxContract:
    def test_path_sandboxing_present(self):
        text = _normalize(_read_prompt())
        # The rule must say that paths outside the working directory are blocked.
        # Allow rephrasing as long as those three concepts appear close together.
        assert re.search(
            r"paths?[^.]{0,80}outside[^.]{0,80}working directory[^.]{0,40}block",
            text,
        ), "path sandboxing rule missing or weakened"

    def test_path_sandboxing_in_both_policies(self):
        for policy in ("autonomous", "interactive"):
            text = _normalize(_read_prompt(policy))
            assert "outside" in text and "working directory" in text and "block" in text


class TestInstructionPriorityContract:
    def test_user_messages_override(self):
        text = _normalize(_read_prompt())
        assert re.search(r"user messages? override", text), (
            "user-messages-override rule missing"
        )

    def test_claude_md_and_agents_md_named(self):
        # Both files must be named so the model knows project-level instruction
        # files exist. Their relative ordering is not part of the contract.
        text = _normalize(_read_prompt())
        assert "claude.md" in text, "CLAUDE.md must be named in the prompt"
        assert "agents.md" in text, "AGENTS.md must be named in the prompt"

    def test_safety_always_binding(self):
        text = _normalize(_read_prompt())
        # Safety constraints must be called out as always-on, regardless of other
        # instructions. We accept either "always binding" or "regardless".
        assert re.search(
            r"safet[^.]{0,80}(always[^.]{0,40}bind|bind[^.]{0,40}regardless|regardless of other)",
            text,
        ), "always-binding safety rule missing or weakened"


class TestEditContract:
    def test_old_string_verbatim(self):
        text = _normalize(_read_prompt())
        assert re.search(
            r"old_string[^.]{0,80}verbatim|verbatim[^.]{0,80}old_string", text
        ), "verbatim old_string rule missing"

    def test_line_number_disambiguator(self):
        text = _normalize(_read_prompt())
        assert "line_number" in text, "line_number must be named explicitly"
        assert re.search(r"multiple matches?", text), (
            "multiple-matches case must be addressed"
        )

    def test_one_edit_per_call(self):
        text = _normalize(_read_prompt())
        # Either "each call ... one edit" or "multiple calls" is acceptable.
        assert re.search(
            r"(each call[^.]{0,40}one edit|one edit[^.]{0,40}per call|multiple changes?[^.]{0,40}multiple calls)",
            text,
        ), "one-edit-per-call rule missing"

    def test_no_reread_after_edit(self):
        text = _normalize(_read_prompt())
        # Small-model guardrail: don't burn turns re-verifying.
        assert re.search(
            r"(do not|don't|never)[^.]{0,40}re-?read[^.]{0,40}(after )?edit",
            text,
        ), "no-re-read-after-edit guardrail missing"


class TestLearnedTagContract:
    def test_learned_open_tag_literal(self):
        # Case-sensitive literal — downstream tooling parses this.
        assert "<learned>" in _read_prompt()

    def test_learned_close_tag_literal(self):
        assert "</learned>" in _read_prompt()


# ---------------------------------------------------------------------------
# Workflow triggers (co-occurrence — flexible for future rewrites)
# ---------------------------------------------------------------------------


class TestExplicitWorkflowToolTriggers:
    """Workflow tools must be named directly with when-to-use triggers nearby.

    Swival is expected to work with small models. Vague guidance like
    "reason carefully about your approach" is not acceptable here — the
    prompt must say *think* / *todo* / *snapshot* with explicit triggers.
    """

    def test_think_named_with_trigger(self):
        text = _read_prompt()
        triggers = [
            r"\bbefore\b",
            r"multi-?step",
            r"debug(ging)?",
            r"\bdecision",
            r"\bediting?\b",
            r"\bplan",
        ]
        assert _has_within(text, "think", triggers), (
            "`think` must appear close to a when-to-use trigger"
        )

    def test_todo_named_with_trigger(self):
        text = _read_prompt()
        triggers = [
            r"multi-?step",
            r"track",
            r"checklist",
            r"\bitems?\b",
            r"work items?",
            r"compaction",
        ]
        assert _has_within(text, "todo", triggers), (
            "`todo` must appear close to a when-to-use trigger"
        )

    def test_snapshot_named_with_trigger(self):
        text = _read_prompt()
        triggers = [
            r"\bafter\b",
            r"explor(e|ation|ing)",
            r"summar",
            r"investigat",
            r"reading",
            r"collapse",
        ]
        assert _has_within(text, "snapshot", triggers), (
            "`snapshot` must appear close to a when-to-use trigger"
        )


# ---------------------------------------------------------------------------
# Tool economy
# ---------------------------------------------------------------------------


class TestToolEconomyContract:
    def test_direct_answers_before_tools(self):
        text = _normalize(_read_prompt())
        assert re.search(
            r"(answer directly[^.]{0,100}(already know|simple math)|do not call tools[^.]{0,100}simple math)",
            text,
        ), "prompt must allow direct answers when tools are unnecessary"

    def test_goal_tools_not_in_default_prompt(self):
        text = _read_prompt()
        assert "complete_goal" not in text
        assert "create_goal" not in text
        assert "update_goal" not in text
        assert "get_goal" not in text

    def test_no_blind_searching_for_ambiguity(self):
        text = _normalize(_read_prompt())
        assert re.search(
            r"ambiguous[^.]{0,140}(clarifying question|instead of searching blindly)",
            text,
        ), "prompt must discourage blind file searches for ambiguous requests"


# ---------------------------------------------------------------------------
# Smoke test: interaction-policy substitution stays well-formed
# ---------------------------------------------------------------------------


class TestInteractionPolicySubstitution:
    def test_no_unsubstituted_placeholders(self):
        for policy in ("autonomous", "interactive"):
            text = _read_prompt(policy)
            assert "{{AUTONOMY_DIRECTIVE}}" not in text
            assert "{{AMBIGUITY_DIRECTIVE}}" not in text

    def test_think_survives_substitution_in_both_policies(self):
        # Both directive variants currently mention `think`; if a future
        # rewrite drops it from one, the workflow-trigger test above still
        # has to pass under both policies.
        for policy in ("autonomous", "interactive"):
            text = _normalize(_read_prompt(policy))
            assert "think" in text

    def test_no_unsubstituted_capability_placeholders(self):
        # All four flag combinations must produce a fully-substituted prompt.
        for no_memory in (False, True):
            for files_mode in ("some", "all", "none"):
                text = _read_prompt(no_memory=no_memory, files_mode=files_mode)
                assert "{{MEMORY_GUIDANCE}}" not in text, (
                    f"MEMORY_GUIDANCE leaked with no_memory={no_memory}, files_mode={files_mode}"
                )
                assert "{{EDITING_GUIDANCE}}" not in text, (
                    f"EDITING_GUIDANCE leaked with no_memory={no_memory}, files_mode={files_mode}"
                )


# ---------------------------------------------------------------------------
# Capability gates: prompt content must adapt to active features
# ---------------------------------------------------------------------------


class TestMemoryGate:
    def test_memory_guidance_present_by_default(self):
        text = _normalize(_read_prompt(no_memory=False))
        # The memory bullet names MEMORY.md explicitly — the model needs to
        # know where to write durable lessons.
        assert "memory.md" in text, "memory guidance missing in default prompt"

    def test_memory_guidance_absent_when_no_memory(self):
        text = _normalize(_read_prompt(no_memory=True))
        # With --no-memory, MEMORY.md isn't loaded; telling the model to
        # write to it would be misleading.
        assert "memory.md" not in text, (
            "memory guidance must be dropped when no_memory=True"
        )

    def test_history_guidance_present_in_both_modes(self):
        # History is independent of the memory flag.
        for no_memory in (False, True):
            text = _normalize(_read_prompt(no_memory=no_memory))
            assert "history.md" in text, (
                f"history guidance missing with no_memory={no_memory}"
            )


class TestEditingGate:
    def test_editing_section_present_by_default(self):
        # files_mode="some" is the default; editing rules must be in scope.
        text = _read_prompt(files_mode="some")
        assert "# Editing files" in text
        # Hard-contract substrings must be present (smoke check; the
        # TestEditContract class above does the real work).
        norm = _normalize(text)
        assert "verbatim" in norm
        assert "line_number" in norm

    def test_editing_section_present_with_files_all(self):
        text = _read_prompt(files_mode="all")
        assert "# Editing files" in text

    def test_editing_section_absent_when_files_none(self):
        # In --files none the file tools error outside .swival/, so the
        # editing rules are unreachable. Drop them; the post-template
        # "Filesystem access is restricted" sentence still informs the model.
        text = _read_prompt(files_mode="none")
        assert "# Editing files" not in text
        norm = _normalize(text)
        assert "verbatim" not in norm, (
            "edit guidance must be dropped when files_mode=none"
        )

    def test_default_path_passes_all_hard_contracts(self):
        # Sanity: the contract regexes above all run against _read_prompt()
        # with defaults. This test pins the default tuple so a future change
        # to defaults can't silently weaken the hard-contract coverage.
        assert _read_prompt() == _read_prompt(no_memory=False, files_mode="some")

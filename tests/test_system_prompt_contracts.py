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

from swival.agent import DEFAULT_SYSTEM_PROMPT_FILE, _apply_interaction_policy


def _read_prompt(policy: str = "autonomous") -> str:
    raw = DEFAULT_SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")
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

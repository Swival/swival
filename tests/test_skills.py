"""Tests for skills.py — skill discovery, activation, and safe_resolve integration."""

import pytest

from pathlib import Path
from swival.skills import (
    SkillInfo,
    parse_frontmatter,
    validate_skill_name,
    discover_skills,
    activate_skill,
    format_skill_catalog,
    MAX_SKILL_BODY_CHARS,
    MAX_SKILL_NAME_CHARS,
)
from swival.tools import safe_resolve, _read_file, dispatch


def _make_skill(
    parent: Path,
    name: str,
    description: str = "A test skill.",
    body: str = "# Instructions\nDo stuff.",
):
    """Create a skill directory with a SKILL.md file."""
    skill_dir = parent / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    content = f"---\nname: {name}\ndescription: {description}\n---\n\n{body}"
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    return skill_dir


# =========================================================================
# Frontmatter parsing
# =========================================================================


class TestParseFrontmatter:
    def test_valid_plain_scalars(self):
        text = "---\nname: my-skill\ndescription: A simple skill.\n---\n\n# Body"
        result = parse_frontmatter(text)
        assert isinstance(result, dict)
        assert result["name"] == "my-skill"
        assert result["description"] == "A simple skill."
        assert result["body"] == "# Body"

    def test_valid_double_quoted(self):
        text = '---\nname: "my-skill"\ndescription: "Deploy to https://example.com"\n---\n\nBody'
        result = parse_frontmatter(text)
        assert result["name"] == "my-skill"
        assert result["description"] == "Deploy to https://example.com"

    def test_valid_single_quoted(self):
        text = "---\nname: 'my-skill'\ndescription: 'A skill with \\'quotes\\''\n---\n\nBody"
        result = parse_frontmatter(text)
        assert result["name"] == "my-skill"
        assert result["description"] == "A skill with 'quotes'"

    def test_missing_closing_quote(self):
        text = '---\nname: "my-skill\ndescription: A skill.\n---\n\nBody'
        result = parse_frontmatter(text)
        assert isinstance(result, str)
        assert "closing" in result

    def test_trailing_garbage_after_quote(self):
        text = '---\nname: "my-skill" extra\ndescription: A skill.\n---\n\nBody'
        result = parse_frontmatter(text)
        assert isinstance(result, str)
        assert "trailing" in result

    def test_multiline_folded(self):
        text = "---\nname: my-skill\ndescription: This is a long\n  description that spans\n  multiple lines.\n---\n\nBody"
        result = parse_frontmatter(text)
        assert (
            result["description"]
            == "This is a long description that spans multiple lines."
        )

    def test_multiline_literal(self):
        text = (
            "---\nname: my-skill\ndescription: |\n  Line one.\n  Line two.\n---\n\nBody"
        )
        result = parse_frontmatter(text)
        assert result["description"] == "Line one.\nLine two."

    def test_missing_opening_delimiter(self):
        text = "name: my-skill\ndescription: A skill.\n---\n\nBody"
        result = parse_frontmatter(text)
        assert isinstance(result, str)
        assert "opening" in result

    def test_missing_closing_delimiter(self):
        text = "---\nname: my-skill\ndescription: A skill.\n\nBody without closing"
        result = parse_frontmatter(text)
        assert isinstance(result, str)
        assert "closing" in result

    def test_missing_name(self):
        text = "---\ndescription: A skill.\n---\n\nBody"
        result = parse_frontmatter(text)
        assert isinstance(result, str)
        assert "name" in result

    def test_missing_description(self):
        text = "---\nname: my-skill\n---\n\nBody"
        result = parse_frontmatter(text)
        assert isinstance(result, str)
        assert "description" in result

    def test_empty_name(self):
        text = "---\nname:\ndescription: A skill.\n---\n\nBody"
        result = parse_frontmatter(text)
        assert isinstance(result, str)
        assert "empty" in result

    def test_empty_description(self):
        text = "---\nname: my-skill\ndescription:\n---\n\nBody"
        result = parse_frontmatter(text)
        assert isinstance(result, str)
        assert "empty" in result

    def test_unknown_fields_skipped(self):
        text = "---\nname: my-skill\nmetadata: ignore-me\n  nested: also-ignored\ndescription: A skill.\nlicense: MIT\n---\n\nBody"
        result = parse_frontmatter(text)
        assert isinstance(result, dict)
        assert result["name"] == "my-skill"
        assert result["description"] == "A skill."
        assert "metadata" not in result
        assert "license" not in result

    def test_colon_in_description_value(self):
        text = '---\nname: my-skill\ndescription: "Deploy to https://example.com:8080/path"\n---\n\nBody'
        result = parse_frontmatter(text)
        assert result["description"] == "Deploy to https://example.com:8080/path"

    def test_empty_body(self):
        text = "---\nname: my-skill\ndescription: A skill.\n---\n"
        result = parse_frontmatter(text)
        assert isinstance(result, dict)
        assert result["body"] == ""


# =========================================================================
# Name validation
# =========================================================================


class TestValidateSkillName:
    def test_valid_simple(self):
        assert validate_skill_name("pdf", "pdf") is None

    def test_valid_hyphenated(self):
        assert validate_skill_name("code-review", "code-review") is None

    def test_valid_single_char(self):
        assert validate_skill_name("a", "a") is None

    def test_valid_max_length(self):
        name = "a" * MAX_SKILL_NAME_CHARS
        assert validate_skill_name(name, name) is None

    def test_invalid_uppercase(self):
        err = validate_skill_name("PDF", "PDF")
        assert err is not None
        assert "lowercase" in err

    def test_invalid_leading_hyphen(self):
        err = validate_skill_name("-pdf", "-pdf")
        assert err is not None

    def test_invalid_trailing_hyphen(self):
        err = validate_skill_name("pdf-", "pdf-")
        assert err is not None

    def test_invalid_consecutive_hyphens(self):
        err = validate_skill_name("pdf--review", "pdf--review")
        assert err is not None
        assert "consecutive" in err

    def test_invalid_too_long(self):
        name = "a" * (MAX_SKILL_NAME_CHARS + 1)
        err = validate_skill_name(name, name)
        assert err is not None
        assert "exceeds" in err

    def test_invalid_underscore(self):
        err = validate_skill_name("pdf_review", "pdf_review")
        assert err is not None

    def test_invalid_dot(self):
        err = validate_skill_name("pdf.review", "pdf.review")
        assert err is not None

    def test_invalid_empty(self):
        err = validate_skill_name("", "")
        assert err is not None

    def test_directory_mismatch(self):
        err = validate_skill_name("pdf", "pdf-tool")
        assert err is not None
        assert "does not match" in err


# =========================================================================
# Discovery
# =========================================================================


class TestDiscoverSkills:
    def test_no_skills_dir(self, tmp_path):
        """No .swival/skills/ directory — empty catalog, no error."""
        catalog = discover_skills(str(tmp_path))
        assert catalog == {}

    def test_empty_skills_dir(self, tmp_path):
        (tmp_path / ".swival" / "skills").mkdir(parents=True)
        catalog = discover_skills(str(tmp_path))
        assert catalog == {}

    def test_valid_skills_found(self, tmp_path):
        skills_dir = tmp_path / ".swival" / "skills"
        _make_skill(skills_dir, "pdf", "Extract text from PDFs.")
        _make_skill(skills_dir, "deploy", "Deploy the application.")

        catalog = discover_skills(str(tmp_path))
        assert len(catalog) == 2
        assert "pdf" in catalog
        assert "deploy" in catalog
        assert catalog["pdf"].description == "Extract text from PDFs."
        assert catalog["pdf"].is_local is True

    def test_invalid_skills_skipped(self, tmp_path, capsys):
        skills_dir = tmp_path / ".swival" / "skills"
        # Invalid: name mismatch
        skill_dir = skills_dir / "wrong-name"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: different\ndescription: Bad.\n---\n\nBody"
        )

        catalog = discover_skills(str(tmp_path), verbose=True)
        assert catalog == {}
        captured = capsys.readouterr()
        assert "Warning:" in captured.err

    def test_extra_skills_dir(self, tmp_path):
        # Use a separate directory outside base_dir for external skills
        import tempfile

        with tempfile.TemporaryDirectory() as ext_tmp:
            extra = Path(ext_tmp) / "global-skills"
            _make_skill(extra, "lint", "Lint code.")

            catalog = discover_skills(str(tmp_path), extra_dirs=[str(extra)])
            assert "lint" in catalog
            assert catalog["lint"].is_local is False

    def test_project_local_beats_extra(self, tmp_path, capsys):
        local_skills = tmp_path / ".swival" / "skills"
        _make_skill(local_skills, "pdf", "Local PDF skill.")

        extra = tmp_path / "global-skills"
        _make_skill(extra, "pdf", "Global PDF skill.")

        catalog = discover_skills(str(tmp_path), extra_dirs=[str(extra)], verbose=True)
        assert len(catalog) == 1
        assert catalog["pdf"].description == "Local PDF skill."
        captured = capsys.readouterr()
        assert "ignored; already loaded" in captured.err

    def test_first_extra_dir_wins(self, tmp_path, capsys):
        extra1 = tmp_path / "extra1"
        _make_skill(extra1, "lint", "First lint.")
        extra2 = tmp_path / "extra2"
        _make_skill(extra2, "lint", "Second lint.")

        catalog = discover_skills(
            str(tmp_path), extra_dirs=[str(extra1), str(extra2)], verbose=True
        )
        assert catalog["lint"].description == "First lint."
        captured = capsys.readouterr()
        assert "ignored; already loaded" in captured.err

    def test_nonexistent_extra_dir(self, tmp_path, capsys):
        catalog = discover_skills(
            str(tmp_path), extra_dirs=["/nonexistent/path"], verbose=True
        )
        assert catalog == {}
        captured = capsys.readouterr()
        assert "does not exist" in captured.err

    def test_extra_dir_is_file(self, tmp_path, capsys):
        f = tmp_path / "not-a-dir"
        f.write_text("oops")
        catalog = discover_skills(str(tmp_path), extra_dirs=[str(f)], verbose=True)
        assert catalog == {}
        captured = capsys.readouterr()
        assert "not a directory" in captured.err

    def test_non_utf8_skill_md_skipped(self, tmp_path, capsys):
        """A SKILL.md with invalid UTF-8 should be skipped, not crash."""
        skills_dir = tmp_path / ".swival" / "skills"
        skill_dir = skills_dir / "bad"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_bytes(
            b"---\nname: bad\ndescription: \xff\xfe broken\n---\n\nBody"
        )

        catalog = discover_skills(str(tmp_path), verbose=True)
        assert "bad" not in catalog
        captured = capsys.readouterr()
        assert "Warning:" in captured.err

    def test_skills_dir_pointing_directly_at_skill(self, tmp_path):
        """--skills-dir can point directly at a directory containing SKILL.md."""
        skill_dir = tmp_path / "some" / "path" / "fastly-cli"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: fastly-cli\ndescription: Manage Fastly services.\n---\n\n# Usage\nRun commands.",
        )

        catalog = discover_skills(str(tmp_path), extra_dirs=[str(skill_dir)])
        assert "fastly-cli" in catalog
        assert catalog["fastly-cli"].description == "Manage Fastly services."
        assert catalog["fastly-cli"].path == skill_dir.resolve()

    def test_skills_dir_direct_and_parent_mixed(self, tmp_path):
        """Can mix direct skill dirs and parent dirs in --skills-dir."""
        import tempfile

        with tempfile.TemporaryDirectory() as ext_tmp:
            # A direct skill dir
            direct = Path(ext_tmp) / "direct-skill"
            direct.mkdir()
            (direct / "SKILL.md").write_text(
                "---\nname: direct-skill\ndescription: Direct.\n---\n\nBody",
            )

            # A parent dir with children
            parent = Path(ext_tmp) / "parent"
            _make_skill(parent, "child-a", "Child A.")
            _make_skill(parent, "child-b", "Child B.")

            catalog = discover_skills(
                str(tmp_path),
                extra_dirs=[str(direct), str(parent)],
            )
            assert "direct-skill" in catalog
            assert "child-a" in catalog
            assert "child-b" in catalog

    def test_skills_dir_nested_structure(self, tmp_path):
        """--skills-dir recurses into subdirectories to find SKILL.md files."""
        # Mimic plugins/<plugin>/skills/<skill>/SKILL.md layout
        plugins = tmp_path / "plugins"
        plugin_a = plugins / "plugin-a" / "skills" / "skill-one"
        plugin_a.mkdir(parents=True)
        (plugin_a / "SKILL.md").write_text(
            "---\nname: skill-one\ndescription: First.\n---\n\nBody",
        )
        plugin_b = plugins / "plugin-b" / "skills" / "skill-two"
        plugin_b.mkdir(parents=True)
        (plugin_b / "SKILL.md").write_text(
            "---\nname: skill-two\ndescription: Second.\n---\n\nBody",
        )

        catalog = discover_skills(str(tmp_path), extra_dirs=[str(plugins)])
        assert "skill-one" in catalog
        assert "skill-two" in catalog

    def test_skills_dir_direct_with_subdirs_ignored(self, tmp_path):
        """When --skills-dir points at a SKILL.md dir, its subdirectories are not scanned."""
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: Top-level.\n---\n\nBody",
        )
        # A nested skill that should NOT be discovered
        nested = skill_dir / "nested"
        nested.mkdir()
        (nested / "SKILL.md").write_text(
            "---\nname: nested\ndescription: Nested.\n---\n\nBody",
        )

        catalog = discover_skills(str(tmp_path), extra_dirs=[str(skill_dir)])
        assert "my-skill" in catalog
        assert "nested" not in catalog


# =========================================================================
# Activation (use_skill handler)
# =========================================================================


class TestActivateSkill:
    def test_successful_load(self, tmp_path):
        skills_dir = tmp_path / ".swival" / "skills"
        _make_skill(skills_dir, "deploy", "Deploy.", "# Deployment\n\nStep 1: Build.")
        catalog = discover_skills(str(tmp_path))
        read_roots: list[Path] = []

        result = activate_skill("deploy", catalog, read_roots)
        assert "[Skill: deploy activated]" in result
        assert "<skill-instructions>" in result
        assert "# Deployment" in result
        assert "Step 1: Build." in result
        assert "Skill directory:" in result

    def test_unknown_skill(self):
        result = activate_skill("nonexistent", {}, [])
        assert result.startswith("error:")
        assert "nonexistent" in result

    def test_body_truncation(self, tmp_path):
        skills_dir = tmp_path / ".swival" / "skills"
        long_body = "x" * (MAX_SKILL_BODY_CHARS + 1000)
        _make_skill(skills_dir, "big", "A big skill.", long_body)
        catalog = discover_skills(str(tmp_path))
        read_roots: list[Path] = []

        result = activate_skill("big", catalog, read_roots)
        assert "truncated" in result

    def test_external_skill_appends_read_root(self, tmp_path):
        import tempfile

        with tempfile.TemporaryDirectory() as ext_tmp:
            extra = Path(ext_tmp) / "global"
            _make_skill(extra, "lint", "Lint code.", "# Lint\nRun linter.")

            catalog = discover_skills(str(tmp_path), extra_dirs=[str(extra)])
            read_roots: list[Path] = []

            activate_skill("lint", catalog, read_roots)
            assert len(read_roots) == 1
            assert read_roots[0] == (extra / "lint").resolve()

    def test_local_skill_no_read_root(self, tmp_path):
        skills_dir = tmp_path / ".swival" / "skills"
        _make_skill(skills_dir, "pdf", "PDF skill.", "# PDF")
        catalog = discover_skills(str(tmp_path))
        read_roots: list[Path] = []

        activate_skill("pdf", catalog, read_roots)
        assert len(read_roots) == 0

    def test_reactivate_no_duplicate(self, tmp_path):
        import tempfile

        with tempfile.TemporaryDirectory() as ext_tmp:
            extra = Path(ext_tmp) / "global"
            _make_skill(extra, "lint", "Lint.", "# Lint")
            catalog = discover_skills(str(tmp_path), extra_dirs=[str(extra)])
            read_roots: list[Path] = []

            activate_skill("lint", catalog, read_roots)
            activate_skill("lint", catalog, read_roots)
            assert len(read_roots) == 1


# =========================================================================
# safe_resolve with extra_read_roots
# =========================================================================


class TestSafeResolveExtraRoots:
    def test_read_under_skill_root(self, tmp_path):
        skill_root = tmp_path / "ext-skill"
        skill_root.mkdir()
        (skill_root / "helper.py").write_text("print('hi')")

        resolved = safe_resolve(
            str(skill_root / "helper.py"),
            str(tmp_path / "project"),
            extra_read_roots=[skill_root.resolve()],
        )
        assert resolved == (skill_root / "helper.py").resolve()

    def test_traversal_blocked(self, tmp_path):
        skill_root = tmp_path / "ext-skill"
        skill_root.mkdir()
        (tmp_path / "secret.txt").write_text("secret")

        with pytest.raises(ValueError):
            safe_resolve(
                str(skill_root / ".." / "secret.txt"),
                str(tmp_path / "project"),
                extra_read_roots=[skill_root.resolve()],
            )

    def test_symlink_escape_blocked(self, tmp_path):
        skill_root = tmp_path / "ext-skill"
        skill_root.mkdir()
        (tmp_path / "secret.txt").write_text("secret")
        (skill_root / "link").symlink_to(tmp_path / "secret.txt")

        with pytest.raises(ValueError):
            safe_resolve(
                str(skill_root / "link"),
                str(tmp_path / "project"),
                extra_read_roots=[skill_root.resolve()],
            )

    def test_empty_extra_roots_unchanged(self, tmp_path):
        """With no extra roots, behaves like original safe_resolve."""
        (tmp_path / "file.txt").write_text("hello")
        resolved = safe_resolve("file.txt", str(tmp_path), extra_read_roots=[])
        assert resolved == (tmp_path / "file.txt").resolve()

    def test_write_to_skill_root_blocked(self, tmp_path):
        """Write/edit operations don't receive extra_read_roots — they only use base_dir."""
        skill_root = tmp_path / "ext-skill"
        skill_root.mkdir()
        (skill_root / "file.txt").write_text("original")

        # Simulate what write_file does: call safe_resolve without extra_read_roots
        with pytest.raises(ValueError):
            safe_resolve(str(skill_root / "file.txt"), str(tmp_path / "project"))

    def test_multiple_roots(self, tmp_path):
        root1 = tmp_path / "skill-a"
        root1.mkdir()
        (root1 / "a.py").write_text("a")

        root2 = tmp_path / "skill-b"
        root2.mkdir()
        (root2 / "b.py").write_text("b")

        project = tmp_path / "project"
        project.mkdir()

        roots = [root1.resolve(), root2.resolve()]
        assert (
            safe_resolve(str(root1 / "a.py"), str(project), extra_read_roots=roots).name
            == "a.py"
        )
        assert (
            safe_resolve(str(root2 / "b.py"), str(project), extra_read_roots=roots).name
            == "b.py"
        )


# =========================================================================
# read_file with extra_read_roots (via dispatch)
# =========================================================================


class TestReadFileSkillRoots:
    def test_read_file_under_activated_skill_root(self, tmp_path):
        skill_root = tmp_path / "ext-skill"
        skill_root.mkdir()
        (skill_root / "helper.py").write_text("print('hello')\n")

        project = tmp_path / "project"
        project.mkdir()

        result = _read_file(
            str(skill_root / "helper.py"),
            str(project),
            extra_read_roots=[skill_root.resolve()],
        )
        assert "print('hello')" in result

    def test_read_file_via_dispatch(self, tmp_path):
        skill_root = tmp_path / "ext-skill"
        skill_root.mkdir()
        (skill_root / "data.txt").write_text("line1\nline2\n")

        project = tmp_path / "project"
        project.mkdir()

        result = dispatch(
            "read_file",
            {"file_path": str(skill_root / "data.txt")},
            str(project),
            skill_read_roots=[skill_root.resolve()],
        )
        assert "line1" in result
        assert "line2" in result


# =========================================================================
# use_skill via dispatch
# =========================================================================


class TestUseSkillDispatch:
    def test_dispatch_use_skill(self, tmp_path):
        skills_dir = tmp_path / ".swival" / "skills"
        _make_skill(skills_dir, "deploy", "Deploy the app.", "# Deploy\nStep 1.")
        catalog = discover_skills(str(tmp_path))
        read_roots: list[Path] = []

        result = dispatch(
            "use_skill",
            {"name": "deploy"},
            str(tmp_path),
            skills_catalog=catalog,
            skill_read_roots=read_roots,
        )
        assert "[Skill: deploy activated]" in result

    def test_dispatch_use_skill_unknown(self, tmp_path):
        result = dispatch(
            "use_skill",
            {"name": "nope"},
            str(tmp_path),
            skills_catalog={},
            skill_read_roots=[],
        )
        assert result.startswith("error:")


# =========================================================================
# Catalog formatting
# =========================================================================


class TestFormatCatalog:
    def test_empty_catalog(self):
        assert format_skill_catalog({}) == ""

    def test_single_skill(self, tmp_path):
        catalog = {
            "pdf": SkillInfo(
                name="pdf",
                description="Extract text from PDFs.",
                path=tmp_path,
                is_local=True,
            ),
        }
        text = format_skill_catalog(catalog)
        assert "<available-skills>" in text
        assert "- pdf: Extract text from PDFs." in text
        assert "use_skill" in text

    def test_sorted_output(self, tmp_path):
        catalog = {
            "deploy": SkillInfo(
                name="deploy", description="Deploy.", path=tmp_path, is_local=True
            ),
            "analyze": SkillInfo(
                name="analyze", description="Analyze.", path=tmp_path, is_local=True
            ),
        }
        text = format_skill_catalog(catalog)
        assert text.index("analyze") < text.index("deploy")


# =========================================================================
# Integration: catalog in system prompt, --no-skills
# =========================================================================


class TestIntegration:
    def test_use_skill_tool_added_when_skills_exist(self, tmp_path):
        """USE_SKILL_TOOL is added to tools list only when catalog is non-empty."""
        from swival.tools import USE_SKILL_TOOL

        skills_dir = tmp_path / ".swival" / "skills"
        _make_skill(skills_dir, "pdf", "PDF processing.")
        catalog = discover_skills(str(tmp_path))

        tools = [{"type": "function", "function": {"name": "read_file"}}]
        if catalog:
            tools.append(USE_SKILL_TOOL)

        tool_names = [t["function"]["name"] for t in tools]
        assert "use_skill" in tool_names

    def test_use_skill_tool_not_added_when_no_skills(self, tmp_path):
        from swival.tools import USE_SKILL_TOOL

        catalog = discover_skills(str(tmp_path))
        assert catalog == {}

        tools = [{"type": "function", "function": {"name": "read_file"}}]
        if catalog:
            tools.append(USE_SKILL_TOOL)

        tool_names = [t["function"]["name"] for t in tools]
        assert "use_skill" not in tool_names

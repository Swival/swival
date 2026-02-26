#!/usr/bin/env python3
"""Build script: converts docs.md/*.md to docs/pages/*.html, generates docs hub,
copies logo, generates favicon. Exits non-zero on broken links."""

import re
import shutil
import sys
from pathlib import Path

import markdown
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = "https://swival.github.io/swival"
REPO_URL = "https://github.com/swival/swival"

ROOT = Path(__file__).parent
DOCS_SRC = ROOT / "docs.md"
WWW = ROOT / "docs"
WWW_DOCS = WWW / "pages"
WWW_IMG = WWW / "img"
MEDIA = ROOT / ".media"

NAV = [
    (
        "Basics",
        [
            (
                "getting-started",
                "Getting Started",
                "Installation, first run, what happens under the hood",
            ),
            (
                "usage",
                "Usage",
                "One-shot mode, REPL mode, CLI flags, piping, exit codes",
            ),
            (
                "tools",
                "Tools",
                "File ops, search, editing, web fetching, thinking, command execution",
            ),
        ],
    ),
    (
        "Configuration & Deployment",
        [
            (
                "safety-and-sandboxing",
                "Safety & Sandboxing",
                "Path resolution, symlink protection, command whitelisting",
            ),
            ("skills", "Skills", "Creating and using SKILL.md-based agent skills"),
            (
                "customization",
                "Customization",
                "Project instructions, system prompt overrides, tuning parameters",
            ),
            ("providers", "Providers", "LM Studio and HuggingFace configuration"),
            ("reports", "Reports", "JSON reports for benchmarking and evaluation"),
            ("reviews", "Reviews", "External reviewer scripts for automated QA gates"),
            ("agentfs", "AgentFS", "Copy-on-write filesystem sandboxing"),
        ],
    ),
]

MD_EXTENSIONS = ["fenced_code", "tables", "toc"]

# ---------------------------------------------------------------------------
# HTML templates
# ---------------------------------------------------------------------------


def sidebar_html(active_slug: str) -> str:
    parts = []
    for group_name, pages in NAV:
        parts.append('<div class="sidebar-section">')
        parts.append(f"<h4>{group_name}</h4>")
        parts.append("<ul>")
        for slug, title, _desc in pages:
            cls = ' class="active"' if slug == active_slug else ""
            parts.append(f'<li><a href="{slug}.html"{cls}>{title}</a></li>')
        parts.append("</ul>")
        parts.append("</div>")
    return "\n".join(parts)


def docs_page_html(title: str, body: str, slug: str) -> str:
    nav = sidebar_html(slug)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title} — Swival</title>
    <meta name="description" content="{title} — Swival documentation">
    <meta property="og:type" content="article">
    <meta property="og:title" content="{title} — Swival">
    <meta property="og:description" content="{title} — Swival documentation">
    <meta property="og:image" content="{BASE_URL}/img/logo.png">
    <meta property="og:url" content="{BASE_URL}/pages/{slug}.html">
    <meta name="twitter:card" content="summary">
    <meta name="twitter:title" content="{title} — Swival">
    <meta name="twitter:description" content="{title} — Swival documentation">
    <meta name="twitter:image" content="{BASE_URL}/img/logo.png">
    <link rel="icon" href="../favicon.ico">
    <link rel="stylesheet" href="../css/style.css">
</head>
<body>
    <header class="site-header">
        <div class="header-inner">
            <a href="../" class="header-logo">
                <img src="../img/logo.png" alt="Swival">
            </a>
            <nav class="header-nav">
                <a href="./">Docs</a>
                <a href="{REPO_URL}">GitHub</a>
            </nav>
        </div>
    </header>
    <div class="docs-layout">
        <aside class="sidebar">
            {nav}
        </aside>
        <article class="docs-content">
            {body}
        </article>
    </div>
    <footer class="site-footer">
        MIT License &middot;
        <a href="{REPO_URL}">GitHub</a>
    </footer>
</body>
</html>"""


def docs_hub_html() -> str:
    nav = sidebar_html("")
    body_parts = ["<h1>Documentation</h1>"]
    for group_name, pages in NAV:
        body_parts.append('<div class="docs-hub-group">')
        body_parts.append(f"<h2>{group_name}</h2>")
        body_parts.append('<ul class="docs-hub-list">')
        for slug, title, desc in pages:
            body_parts.append(
                f'<li><a href="{slug}.html">{title}</a>'
                f'<span class="desc">{desc}</span></li>'
            )
        body_parts.append("</ul>")
        body_parts.append("</div>")
    body = "\n".join(body_parts)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Documentation — Swival</title>
    <meta name="description" content="Swival documentation hub">
    <meta property="og:type" content="website">
    <meta property="og:title" content="Documentation — Swival">
    <meta property="og:description" content="Swival documentation hub">
    <meta property="og:image" content="{BASE_URL}/img/logo.png">
    <meta property="og:url" content="{BASE_URL}/pages/">
    <meta name="twitter:card" content="summary">
    <meta name="twitter:title" content="Documentation — Swival">
    <meta name="twitter:description" content="Swival documentation hub">
    <meta name="twitter:image" content="{BASE_URL}/img/logo.png">
    <link rel="icon" href="../favicon.ico">
    <link rel="stylesheet" href="../css/style.css">
</head>
<body>
    <header class="site-header">
        <div class="header-inner">
            <a href="../" class="header-logo">
                <img src="../img/logo.png" alt="Swival">
            </a>
            <nav class="header-nav">
                <a href="./">Docs</a>
                <a href="{REPO_URL}">GitHub</a>
            </nav>
        </div>
    </header>
    <div class="docs-layout">
        <aside class="sidebar">
            {nav}
        </aside>
        <article class="docs-content">
            {body}
        </article>
    </div>
    <footer class="site-footer">
        MIT License &middot;
        <a href="{REPO_URL}">GitHub</a>
    </footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Link rewriting
# ---------------------------------------------------------------------------

LINK_RE = re.compile(r'href="([^"]*\.md)(#[^"]*)??"')


def rewrite_md_links(html: str) -> str:
    """Rewrite href="...*.md" to href="...*.html", preserving fragments."""

    def replace(m: re.Match) -> str:
        path = m.group(1)
        fragment = m.group(2) or ""
        # Only rewrite relative links (not http:// etc.)
        if "://" in path:
            return m.group(0)
        html_path = re.sub(r"\.md$", ".html", path)
        return f'href="{html_path}{fragment}"'

    return LINK_RE.sub(replace, html)


# ---------------------------------------------------------------------------
# Broken-link checker
# ---------------------------------------------------------------------------


def extract_ids(html: str) -> set[str]:
    """Extract all id attributes from HTML."""
    return set(re.findall(r'id="([^"]+)"', html))


def check_links(pages: dict[str, str]) -> list[str]:
    """Check all local href links in generated docs pages.
    pages: {filename: html_content} for files in docs/pages/.
    Returns list of error messages. Empty = all good."""
    errors = []
    href_re = re.compile(r'href="([^"]*)"')

    for filename, html in pages.items():
        for m in href_re.finditer(html):
            href = m.group(1)
            # Skip external links, mailto, javascript
            if "://" in href or href.startswith("mailto:") or href.startswith("#"):
                # Check same-page fragment
                if href.startswith("#"):
                    frag = href[1:]
                    if frag and frag not in extract_ids(html):
                        errors.append(f"{filename}: broken fragment {href}")
                continue
            # Skip parent-relative links (to landing page, css, img, etc.)
            if href.startswith("../") or href.startswith("/"):
                continue

            # Split target and fragment
            if "#" in href:
                target, fragment = href.split("#", 1)
            else:
                target, fragment = href, ""

            # Resolve target file
            if target == "" or target == "./":
                target_html = pages.get("index.html", "")
            else:
                target_html = pages.get(target, None)
                if target_html is None:
                    # Check if file exists on disk
                    target_path = WWW_DOCS / target
                    if not target_path.exists():
                        errors.append(f"{filename}: broken link to {href}")
                    continue

            # Check fragment
            if fragment and fragment not in extract_ids(target_html):
                errors.append(
                    f"{filename}: broken fragment #{fragment} in {target or 'index.html'}"
                )

    return errors


# ---------------------------------------------------------------------------
# Favicon generation
# ---------------------------------------------------------------------------


def generate_favicon(logo_path: Path, out_path: Path) -> None:
    """Generate a favicon.ico from the logo PNG."""
    img = Image.open(logo_path)
    img = img.resize((32, 32), Image.LANCZOS)
    # Convert to RGBA if not already
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    img.save(out_path, format="ICO", sizes=[(32, 32)])


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build() -> bool:
    """Run the full build. Returns True on success, False on broken links."""
    md_converter = markdown.Markdown(extensions=MD_EXTENSIONS)

    # Collect all known slugs
    known_slugs = set()
    for _group, pages in NAV:
        for slug, _title, _desc in pages:
            known_slugs.add(slug)

    # Ensure output dirs exist
    WWW_DOCS.mkdir(parents=True, exist_ok=True)
    WWW_IMG.mkdir(parents=True, exist_ok=True)

    # Remove stale generated HTML from docs/pages/
    expected_files = {"index.html"} | {f"{s}.html" for s in known_slugs}
    for existing in WWW_DOCS.glob("*.html"):
        if existing.name not in expected_files:
            existing.unlink()
            print(f"  removed stale {existing.name}")

    # Track generated pages for link checking
    generated: dict[str, str] = {}

    # Convert each docs page
    for _group, pages in NAV:
        for slug, title, _desc in pages:
            md_path = DOCS_SRC / f"{slug}.md"
            if not md_path.exists():
                print(
                    f"ERROR: {md_path} not found (referenced in NAV config)",
                    file=sys.stderr,
                )
                return False

            md_converter.reset()
            md_text = md_path.read_text(encoding="utf-8")
            body_html = md_converter.convert(md_text)
            body_html = rewrite_md_links(body_html)
            full_html = docs_page_html(title, body_html, slug)

            out_path = WWW_DOCS / f"{slug}.html"
            out_path.write_text(full_html, encoding="utf-8")
            generated[f"{slug}.html"] = full_html
            print(f"  {slug}.md -> {slug}.html")

    # Generate docs hub
    hub_html = docs_hub_html()
    (WWW_DOCS / "index.html").write_text(hub_html, encoding="utf-8")
    generated["index.html"] = hub_html
    print("  docs/index.html (hub)")

    # Copy logo
    logo_src = MEDIA / "logo.png"
    logo_dst = WWW_IMG / "logo.png"
    if logo_src.exists():
        shutil.copy2(logo_src, logo_dst)
        print("  logo.png -> docs/img/logo.png")
    else:
        print(f"WARNING: {logo_src} not found, skipping logo copy", file=sys.stderr)

    # Generate favicon
    if logo_src.exists():
        favicon_path = WWW / "favicon.ico"
        generate_favicon(logo_src, favicon_path)
        print("  favicon.ico generated")

    # Check links
    print("\nChecking links...")
    errors = check_links(generated)
    if errors:
        print(f"\nBroken links found ({len(errors)}):", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)
        return False

    print("All links OK.")
    return True


if __name__ == "__main__":
    print("Building Swival website...\n")
    ok = build()
    if not ok:
        sys.exit(1)
    print("\nDone.")

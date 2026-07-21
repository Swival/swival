#!/usr/bin/env python3
"""Build script: converts docs.md/*.md to docs/pages/*.html, generates docs hub,
copies logo, generates favicon. Exits non-zero on broken links."""

import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import markdown
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = "https://swival.dev"
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
            (
                "audit",
                "Security Audit",
                "Multi-phase security audit with triage, verification, and patch generation",
            ),
            (
                "simplify",
                "Simplification",
                "Four plain-text reviewers find simplifications for the main agent to apply and validate",
            ),
            (
                "goal",
                "Goals",
                "Persistent goal mode that keeps the agent on task across turns until the objective is done",
            ),
            (
                "context-management",
                "Context Management",
                "How Swival fits large tasks into small context windows",
            ),
            (
                "open-models",
                "Not Just for Frontier Models",
                "Why Swival is built to work well with small and open models too",
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
            (
                "secrets",
                "Secret Encryption",
                "Format-preserving encryption of credentials before they reach the LLM provider",
            ),
            (
                "llm-filter",
                "Outbound LLM Filter",
                "User-defined scripts that redact or block content before it reaches the LLM provider",
            ),
            (
                "command-middleware",
                "Command Middleware",
                "Intercept, rewrite, or block shell commands before execution — RTK integration included",
            ),
            ("skills", "Skills", "Creating and using SKILL.md-based agent skills"),
            (
                "library",
                "The Library",
                "Staging downloaded skill collections and promoting them into projects",
            ),
            (
                "metaskills",
                "Metaskills",
                "Portable standard for dynamic SKILL.md workflow programs",
            ),
            (
                "web-browsing",
                "Web Browsing",
                "Chrome DevTools MCP, agent-browser, and Lightpanda",
            ),
            (
                "mcp",
                "MCP",
                "Connecting external tool servers via Model Context Protocol",
            ),
            (
                "a2a",
                "A2A",
                "Connecting to remote agents via the Agent-to-Agent protocol",
            ),
            (
                "acp",
                "ACP",
                "Driving Swival from ACP-aware editors like Zed and agent-client-protocol.nvim",
            ),
            (
                "lifecycle-hooks",
                "Lifecycle Hooks",
                "Run commands at startup and exit to sync .swival/ state with remote storage",
            ),
            (
                "custom-commands",
                "Custom Commands",
                "Run external scripts from the REPL and inject their output into the conversation",
            ),
            (
                "customization",
                "Customization",
                "Config files, project instructions, system prompt overrides, tuning parameters",
            ),
            (
                "providers",
                "Providers",
                "LM Studio, HuggingFace, OpenRouter, Google Gemini, Gemini Enterprise Agent Platform (Vertex AI), ChatGPT Plus/Pro, AWS Bedrock, and generic server configuration",
            ),
            ("reports", "Reports", "JSON reports for benchmarking and evaluation"),
            ("reviews", "Reviews", "External reviewer scripts for automated QA gates"),
            ("agentfs", "AgentFS", "Copy-on-write filesystem sandboxing"),
            (
                "nono",
                "nono",
                "Capability-based OS sandboxing with Landlock/Seatbelt, network filtering, and rollback",
            ),
        ],
    ),
    (
        "Reference",
        [
            (
                "python-api",
                "Python API",
                "Session, Result, run(), and the exception contract",
            ),
        ],
    ),
]

MD_EXTENSIONS = ["fenced_code", "tables", "toc"]

# ---------------------------------------------------------------------------
# Admonition preprocessor
# ---------------------------------------------------------------------------
#
# Recognise GitHub-style admonition blockquotes of the form:
#
#     > [!TIP] Optional title
#     > Body line 1
#     > Body line 2
#
# and rewrite them into a blockquote with class="callout callout-{kind}".
# The CSS provides styling for: tip, note, warning, danger. Unknown kinds
# fall through unchanged.

ADMONITION_KINDS = {
    "TIP": "tip",
    "NOTE": "note",
    "INFO": "note",
    "IMPORTANT": "warning",
    "WARNING": "warning",
    "CAUTION": "warning",
    "DANGER": "danger",
}

ADMONITION_KIND_TITLES = {
    "tip": "Tip",
    "note": "Note",
    "warning": "Warning",
    "danger": "Danger",
}

# Inline SVGs sized to fit a 16px-tall callout-title. Kept inline so we
# do not depend on extra asset files.
ADMONITION_ICONS = {
    "tip": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
        'aria-hidden="true"><path d="M9 18h6"></path>'
        '<path d="M10 22h4"></path>'
        '<path d="M12 2a7 7 0 0 0-4 12.7c.6.5 1 1.2 1 2v.3h6v-.3c0-.8.4-1.5 1-2A7 7 0 0 0 12 2z">'
        "</path></svg>"
    ),
    "note": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
        'aria-hidden="true"><circle cx="12" cy="12" r="10">'
        '</circle><path d="M12 16v-4"></path><path d="M12 8h.01"></path></svg>'
    ),
    "warning": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
        'aria-hidden="true"><path d="M10.3 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z">'
        '</path><path d="M12 9v4"></path><path d="M12 17h.01"></path></svg>'
    ),
    "danger": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
        'aria-hidden="true"><polygon points="12 2 22 22 2 22"></polygon>'
        '<path d="M12 9v4"></path><path d="M12 17h.01"></path></svg>'
    ),
}

# Matches the first line of an admonition, e.g. "[!TIP] Title" or
# "[!TIP]" (title optional). The leading "> " of a Markdown blockquote is
# consumed by the blockquote syntax and is not present in the rendered
# HTML, so we do not require it here.
ADMONITION_HEADER_RE = re.compile(
    r"^\[!([A-Z]+)\][ ]*(.*?)\s*$"
)


def transform_admonitions(html: str) -> str:
    """Walk the HTML and rewrite `> [!KIND] Title\\n> ...` blockquote bodies
    into `<blockquote class="callout callout-{kind}">` with a callout-title.

    Markdown already converts `> ...` into `<blockquote>...</blockquote>`.
    Inside that blockquote the first paragraph or the literal text holds
    the first line. We detect admonitions by looking for the first
    paragraph of a blockquote, matching the header text in the format
    `[!KIND] Title`, and rewriting both the wrapper class and that
    paragraph's content.
    """

    def parse_kinds(text: str) -> tuple[str, str] | None:
        m = ADMONITION_HEADER_RE.match(text)
        if not m:
            return None
        kind_token = m.group(1).upper()
        if kind_token not in ADMONITION_KINDS:
            return None
        return ADMONITION_KINDS[kind_token], m.group(2)

    # When a blockquote has multiple lines, python-markdown collapses them
    # into a single <p> separated by "\n". Split that single <p> into a
    # header paragraph and a body paragraph so the admonition can be
    # rendered with a styled title.
    def split_paragraph(p_match: re.Match) -> str:
        attrs, body = p_match.group(1), p_match.group(2)
        lines = body.split("\n")
        if not lines:
            return p_match.group(0)
        first = lines[0].rstrip()
        rest = "\n".join(lines[1:]).strip()
        if rest:
            return f"<p{attrs}>{first}</p>\n<p{attrs}>{rest}</p>"
        return f"<p{attrs}>{first}</p>"

    split_p = re.compile(r"<p([^>]*)>([\s\S]*?)</p>")

    def rewrite(match: re.Match) -> str:
        attrs = match.group(1) or ""
        inner = match.group(2) or ""

        # First, split any <p>...</p> with a header line at the top.
        inner = split_p.sub(split_paragraph, inner, count=1)

        # Pull off the first <p>...</p> if it is the admonition header.
        first_p = re.match(r"\s*(<p>(.*?)</p>)\s*", inner, flags=re.DOTALL)
        if not first_p:
            return match.group(0)
        parsed = parse_kinds(first_p.group(2).strip())
        if not parsed:
            return match.group(0)
        kind, user_title = parsed
        # Strip leading/trailing emphasis markers like <em> or trailing colons
        title = user_title.strip().rstrip(":").strip()
        if not title:
            title = ADMONITION_KIND_TITLES[kind]
        title_html = (
            f'<div class="callout-title">{ADMONITION_ICONS[kind]}<span>{title}</span></div>'
        )
        rest = inner[first_p.end():].lstrip()
        # Add our classes onto the wrapper
        new_class = f'callout callout-{kind}'
        if 'class="' in attrs:
            attrs = re.sub(
                r'class="([^"]*)"',
                lambda m: f'class="{m.group(1)} {new_class}"',
                attrs,
                count=1,
            )
        else:
            attrs = f' class="{new_class}"' + attrs
        return f"<blockquote{attrs}>{title_html}{rest}</blockquote>"

    pattern = re.compile(r"<blockquote([^>]*)>([\s\S]*?)</blockquote>")
    return pattern.sub(rewrite, html)


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


# Flat slug list, in display order, for prev/next navigation.
_FLAT_SLUGS: list[str] | None = None


def flat_slugs() -> list[str]:
    global _FLAT_SLUGS
    if _FLAT_SLUGS is None:
        _FLAT_SLUGS = [slug for _g, pages in NAV for slug, _t, _d in pages]
    return _FLAT_SLUGS


def page_title_for(slug: str) -> str:
    for _g, pages in NAV:
        for s, t, _d in pages:
            if s == slug:
                return t
    return slug


def prev_next_html(slug: str) -> str:
    slugs = flat_slugs()
    try:
        idx = slugs.index(slug)
    except ValueError:
        return ""
    prev_slug = slugs[idx - 1] if idx > 0 else None
    next_slug = slugs[idx + 1] if idx + 1 < len(slugs) else None
    parts = ['<nav class="page-nav" aria-label="Page navigation">']
    if prev_slug:
        parts.append(
            f'<a class="nav-prev" href="{prev_slug}.html">'
            f'<span class="nav-label">&larr; Previous</span>'
            f'<span class="nav-title">{page_title_for(prev_slug)}</span>'
            f"</a>"
        )
    else:
        parts.append('<span class="nav-prev nav-empty" aria-hidden="true"></span>')
    if next_slug:
        parts.append(
            f'<a class="nav-next" href="{next_slug}.html">'
            f'<span class="nav-label">Next &rarr;</span>'
            f'<span class="nav-title">{page_title_for(next_slug)}</span>'
            f"</a>"
        )
    else:
        parts.append('<span class="nav-next nav-empty" aria-hidden="true"></span>')
    parts.append("</nav>")
    return "".join(parts)


def docs_page_html(title: str, desc: str, body: str, slug: str) -> str:
    nav = sidebar_html(slug)
    page_url = f"{BASE_URL}/pages/{slug}.html"
    meta_desc = f"{title}: {desc}" if desc else f"{title} — Swival documentation"
    page_nav = prev_next_html(slug)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title} — Swival</title>
    <meta name="description" content="{meta_desc}">
    <link rel="canonical" href="{page_url}">
    <meta property="og:type" content="article">
    <meta property="og:title" content="{title} — Swival">
    <meta property="og:description" content="{meta_desc}">
    <meta property="og:image" content="{BASE_URL}/img/og.png">
    <meta property="og:url" content="{page_url}">
    <meta name="twitter:card" content="summary">
    <meta name="twitter:title" content="{title} — Swival">
    <meta name="twitter:description" content="{meta_desc}">
    <meta name="twitter:image" content="{BASE_URL}/img/og.png">
    <script type="application/ld+json">
    {{
        "@context": "https://schema.org",
        "@type": "TechArticle",
        "headline": "{title} — Swival",
        "description": "{meta_desc}",
        "url": "{page_url}",
        "publisher": {{
            "@type": "Organization",
            "name": "Swival",
            "logo": {{ "@type": "ImageObject", "url": "{BASE_URL}/img/logo.png" }}
        }},
        "isPartOf": {{
            "@type": "WebSite",
            "name": "Swival",
            "url": "{BASE_URL}/"
        }}
    }}
    </script>
    <link rel="icon" href="../favicon.ico">
    <link rel="stylesheet" href="../css/style.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
</head>
<body>
    <a class="skip-link" href="#main">Skip to content</a>
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
    <main id="main" class="docs-layout-3col">
        <aside class="sidebar">
            {nav}
        </aside>
        <article class="docs-content">
            {body}
            {page_nav}
        </article>
        <aside class="page-toc" aria-label="On this page">
            <p class="page-toc-title">On this page</p>
            <div class="page-toc-list"></div>
        </aside>
    </main>
    <footer class="site-footer">
        <div class="footer-inner">
            <div class="footer-grid">
                <div class="footer-col">
                    <img src="../img/logo.png" alt="Swival" class="footer-logo">
                    <p class="footer-tagline">A coding agent for any model.<br>Free, open-source, and easy to set up.</p>
                </div>
                <div class="footer-col">
                    <h4>Resources</h4>
                    <ul>
                        <li><a href="getting-started.html">Getting Started</a></li>
                        <li><a href="usage.html">Usage</a></li>
                        <li><a href="tools.html">Tools</a></li>
                        <li><a href="context-management.html">Context Management</a></li>
                    </ul>
                </div>
                <div class="footer-col">
                    <h4>Guides</h4>
                    <ul>
                        <li><a href="providers.html">Providers</a></li>
                        <li><a href="safety-and-sandboxing.html">Safety &amp; Sandboxing</a></li>
                        <li><a href="python-api.html">Python API</a></li>
                        <li><a href="open-models.html">Open Models</a></li>
                    </ul>
                </div>
                <div class="footer-col">
                    <h4>Community</h4>
                    <ul>
                        <li><a href="{REPO_URL}">GitHub</a></li>
                        <li><a href="https://calibra.swival.dev">Calibra</a></li>
                    </ul>
                </div>
            </div>
            <div class="footer-bottom">
                <p>&copy; 2025&ndash;2026 Swival &middot; MIT License</p>
            </div>
        </div>
    </footer>
    <button class="back-to-top" aria-label="Back to top">&uarr;</button>
    <script src="../js/site.js"></script>
</body>
</html>"""


def docs_hub_html() -> str:
    nav = sidebar_html("")
    body_parts = ['<h1 id="documentation">Documentation</h1>']
    body_parts.append(
        '<p class="docs-hub-lede">Everything you need to install, configure, and get the most out of Swival.</p>'
    )
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
    <meta name="description" content="Swival documentation: guides for installation, configuration, providers, MCP, A2A, security audits, and the Python API.">
    <link rel="canonical" href="{BASE_URL}/pages/">
    <meta property="og:type" content="website">
    <meta property="og:title" content="Documentation — Swival">
    <meta property="og:description" content="Swival documentation: guides for installation, configuration, providers, MCP, A2A, security audits, and the Python API.">
    <meta property="og:image" content="{BASE_URL}/img/og.png">
    <meta property="og:url" content="{BASE_URL}/pages/">
    <meta name="twitter:card" content="summary">
    <meta name="twitter:title" content="Documentation — Swival">
    <meta name="twitter:description" content="Swival documentation: guides for installation, configuration, providers, MCP, A2A, security audits, and the Python API.">
    <meta name="twitter:image" content="{BASE_URL}/img/og.png">
    <link rel="icon" href="../favicon.ico">
    <link rel="stylesheet" href="../css/style.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
</head>
<body>
    <a class="skip-link" href="#main">Skip to content</a>
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
    <main id="main" class="docs-layout-3col">
        <aside class="sidebar">
            {nav}
        </aside>
        <article class="docs-content">
            {body}
        </article>
        <aside class="page-toc" aria-label="On this page">
            <p class="page-toc-title">On this page</p>
            <div class="page-toc-list"></div>
        </aside>
    </main>
    <footer class="site-footer">
        <div class="footer-inner">
            <div class="footer-grid">
                <div class="footer-col">
                    <img src="../img/logo.png" alt="Swival" class="footer-logo">
                    <p class="footer-tagline">A coding agent for any model.<br>Free, open-source, and easy to set up.</p>
                </div>
                <div class="footer-col">
                    <h4>Resources</h4>
                    <ul>
                        <li><a href="getting-started.html">Getting Started</a></li>
                        <li><a href="usage.html">Usage</a></li>
                        <li><a href="tools.html">Tools</a></li>
                        <li><a href="context-management.html">Context Management</a></li>
                    </ul>
                </div>
                <div class="footer-col">
                    <h4>Guides</h4>
                    <ul>
                        <li><a href="providers.html">Providers</a></li>
                        <li><a href="safety-and-sandboxing.html">Safety &amp; Sandboxing</a></li>
                        <li><a href="python-api.html">Python API</a></li>
                        <li><a href="open-models.html">Open Models</a></li>
                    </ul>
                </div>
                <div class="footer-col">
                    <h4>Community</h4>
                    <ul>
                        <li><a href="{REPO_URL}">GitHub</a></li>
                        <li><a href="https://calibra.swival.dev">Calibra</a></li>
                    </ul>
                </div>
            </div>
            <div class="footer-bottom">
                <p>&copy; 2025&ndash;2026 Swival &middot; MIT License</p>
            </div>
        </div>
    </footer>
    <button class="back-to-top" aria-label="Back to top">&uarr;</button>
    <script src="../js/site.js"></script>
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


def generate_og_image(logo_path: Path, out_path: Path) -> None:
    """Generate a 1200x630 Open Graph card with a gradient background,
    the brand mark, and the tagline. Pillow only — no system fonts required."""
    from PIL import ImageDraw, ImageFont

    W, H = 1200, 630
    grad = Image.new("RGB", (W, H), (15, 23, 42))
    px = grad.load()
    for y in range(H):
        t = y / (H - 1)
        r = int(15 + (29 - 15) * t)
        g = int(23 + (78 - 23) * t)
        b = int(42 + (216 - 42) * t)
        for x in range(W):
            px[x, y] = (r, g, b)

    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for r, a in [(600, 25), (450, 40), (300, 60), (150, 90)]:
        gd.ellipse((-200, -200, r, r), fill=(96, 165, 250, a))
    for r, a in [(600, 25), (450, 40), (300, 55), (150, 80)]:
        gd.ellipse((W - r, H - r, W + 200, H + 200), fill=(251, 146, 60, a))
    img = Image.alpha_composite(grad.convert("RGBA"), glow).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Pick a bold + regular font from common system locations. Falls back
    # silently to no text if none are available; the gradient alone is still
    # a usable OG image.
    font_pairs = [
        (
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Helvetica.ttc",
        ),
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ),
        (
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ),
        (
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
        ),
    ]
    title_font = None
    sub_font = None
    for bold, regular in font_pairs:
        try:
            title_font = ImageFont.truetype(bold, 96)
            sub_font = ImageFont.truetype(regular, 36)
            break
        except OSError:
            continue

    pad = 80
    if logo_path.exists():
        try:
            logo = Image.open(logo_path).convert("RGBA")
            logo.thumbnail((220, 220), Image.LANCZOS)
            img.paste(logo, (pad, pad), logo)
        except Exception:
            pass

    if title_font is not None:
        draw.text((pad, H - 260), "Swival", fill=(255, 255, 255), font=title_font)
        draw.text(
            (pad, H - 140),
            "A coding agent for any model.",
            fill=(226, 232, 240),
            font=sub_font,
        )
        draw.text(
            (pad, H - 90),
            "Free, open-source, and easy to set up.",
            fill=(148, 163, 184),
            font=sub_font,
        )

    img.save(out_path, format="PNG", optimize=True)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _git_lastmod(path: Path | None = None) -> str:
    """Return the last-modified date for *path* (or the repo root) as YYYY-MM-DD.
    Falls back to today if git is unavailable or the file is untracked."""
    cmd = ["git", "log", "-1", "--format=%aI"]
    if path is not None:
        cmd.append(str(path))
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        if out:
            return out[:10]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


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
    (WWW / "js").mkdir(parents=True, exist_ok=True)

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
        for slug, title, desc in pages:
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
            body_html = transform_admonitions(body_html)
            full_html = docs_page_html(title, desc, body_html, slug)

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

    # Generate Open Graph card
    og_path = WWW_IMG / "og.png"
    try:
        generate_og_image(logo_src, og_path)
        print("  og.png generated (1200x630)")
    except Exception as exc:
        print(f"WARNING: og.png generation failed: {exc}", file=sys.stderr)

    # Generate sitemap.xml
    lastmod = _git_lastmod()
    sitemap_urls = [f"  <url><loc>{BASE_URL}/</loc><lastmod>{lastmod}</lastmod></url>"]
    sitemap_urls.append(
        f"  <url><loc>{BASE_URL}/pages/</loc><lastmod>{lastmod}</lastmod></url>"
    )
    for _group, pages in NAV:
        for slug, _title, _desc in pages:
            md_path = DOCS_SRC / f"{slug}.md"
            page_mod = _git_lastmod(md_path)
            sitemap_urls.append(
                f"  <url><loc>{BASE_URL}/pages/{slug}.html</loc>"
                f"<lastmod>{page_mod}</lastmod></url>"
            )
    sitemap_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "\n".join(sitemap_urls)
        + "\n</urlset>\n"
    )
    (WWW / "sitemap.xml").write_text(sitemap_xml, encoding="utf-8")
    print("  sitemap.xml generated")

    # Generate robots.txt
    robots_txt = f"User-agent: *\nAllow: /\n\nSitemap: {BASE_URL}/sitemap.xml\n"
    (WWW / "robots.txt").write_text(robots_txt, encoding="utf-8")
    print("  robots.txt generated")

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

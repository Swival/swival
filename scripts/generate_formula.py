#!/usr/bin/env python3
"""Generate Homebrew formula resource blocks from resolved dependencies.

Usage:
    uv run python scripts/generate_formula.py [--version VERSION]

Writes homebrew-tap/Formula/swival.rb with up-to-date resource stanzas and SHA256.
Requires the sdist to exist in dist/ (run `make dist` first).
"""

import hashlib
import json
import re
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORMULA = ROOT / "homebrew-tap" / "Formula" / "swival.rb"

SKIP = {"swival"}

# Packages whose sdists don't build in Homebrew (need maturin, etc.).
# Values are filename substrings used to pick the right wheel on PyPI.
USE_WHEEL = {
    "hf-xet": "cp37-abi3-macosx",
    "rank-bm25": "py3-none-any",
}


def get_version():
    text = (ROOT / "pyproject.toml").read_text()
    m = re.search(r'^version\s*=\s*"(.+?)"', text, re.MULTILINE)
    if not m:
        raise SystemExit("Could not find version in pyproject.toml")
    return m.group(1)


def resolve_deps():
    result = subprocess.run(
        ["uv", "pip", "compile", "pyproject.toml", "--no-header"],
        capture_output=True, text=True, cwd=ROOT,
    )
    if result.returncode != 0:
        raise SystemExit(f"uv pip compile failed:\n{result.stderr}")

    deps = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z0-9_.-]+)==(.+)$", line)
        if m:
            name, version = m.group(1), m.group(2)
            if name.lower() not in SKIP:
                deps.append((name, version))
    return deps


def pypi_url_and_sha(name, version):
    """Fetch the best download URL and SHA256 from PyPI JSON API."""
    api_url = f"https://pypi.org/pypi/{name}/{version}/json"
    with urllib.request.urlopen(api_url) as resp:
        data = json.load(resp)

    normalized = name.lower().replace("_", "-")
    if normalized in USE_WHEEL:
        tag_prefix = USE_WHEEL[normalized]
        for entry in data["urls"]:
            fn = entry["filename"]
            if tag_prefix in fn and ("arm64" in fn or "none-any" in fn):
                return entry["url"], entry["digests"]["sha256"]

    for entry in data["urls"]:
        if entry["packagetype"] == "sdist":
            return entry["url"], entry["digests"]["sha256"]

    return data["urls"][0]["url"], data["urls"][0]["digests"]["sha256"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def format_resource(name, url, sha):
    is_wheel = url.endswith(".whl")
    url_line = f'    url "{url}", using: :nounzip' if is_wheel else f'    url "{url}"'
    return (
        f'  resource "{name}" do\n'
        f'{url_line}\n'
        f'    sha256 "{sha}"\n'
        f"  end\n"
    )


def main():
    if "--version" in sys.argv:
        idx = sys.argv.index("--version")
        version = sys.argv[idx + 1]
    else:
        version = get_version()

    print(f"Generating formula for swival {version}")

    sdist = ROOT / "dist" / f"swival-{version}.tar.gz"
    our_sha = sha256_file(sdist) if sdist.exists() else "PLACEHOLDER"
    if our_sha == "PLACEHOLDER":
        print(f"Warning: {sdist} not found, using PLACEHOLDER for sha256")

    deps = resolve_deps()
    print(f"Resolved {len(deps)} dependencies")

    def fetch_one(item):
        i, (name, ver) = item
        print(f"  [{i+1}/{len(deps)}] {name}=={ver}")
        try:
            url, sha = pypi_url_and_sha(name, ver)
        except Exception as e:
            print(f"    WARNING: could not fetch {name} from PyPI: {e}")
            url, sha = f"FIXME:{name}-{ver}", "FIXME"
        return i, name, url, sha

    results = [None] * len(deps)
    with ThreadPoolExecutor(max_workers=16) as pool:
        for i, name, url, sha in pool.map(fetch_one, enumerate(deps)):
            results[i] = (name, url, sha)

    resource_text = "\n".join(format_resource(n, u, s) for n, u, s in results)

    template = FORMULA.read_text()
    template = re.sub(r'version ".*?"', f'version "{version}"', template, count=1)
    template = re.sub(r'sha256 ".*?"', f'sha256 "{our_sha}"', template, count=1)
    template = re.sub(
        r"  # RESOURCES_START\n.*?  # RESOURCES_END",
        f"  # RESOURCES_START\n{resource_text}  # RESOURCES_END",
        template,
        flags=re.DOTALL,
    )

    FORMULA.write_text(template)
    print(f"\nWrote {FORMULA}")


if __name__ == "__main__":
    main()

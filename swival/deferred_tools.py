"""Deferred MCP tool schemas: names up front, schemas on demand.

A large MCP catalog costs its full schemas on every request, whether or not
the task needs them. When the catalog is deferred, the model sees the tool
names in the ``tool_search`` description and loads the schemas it wants by
keyword; loaded schemas join every later request of that conversation.

Routing is unaffected: ``McpManager`` still knows every tool, so a call to a
deferred tool works and loads its schema for the requests that follow.
"""

import json
import re

from .tokens import count_tokens

TOOL_SEARCH_NAME = "tool_search"

# About ten average tools on the one server measured (chrome-devtools, 211
# tokens per schema). Below this, a search round trip and the prompt-cache
# miss it causes cost more than sending the schemas every time.
DEFER_MIN_TOKENS = 2000
MAX_LOADS = 5
MAX_OVERFLOW_NAMES = 20
_NAME_HIT = 2
_DESCRIPTION_HIT = 1
_QUERY_SPLIT_RE = re.compile(r"[^0-9a-z]+")


def _schema_name(schema: dict) -> str:
    return (schema.get("function") or {}).get("name") or ""


def _split_name(name: str) -> tuple[str, str]:
    """``mcp__server__tool`` -> ``(server, tool)``."""
    parts = name.split("__", 2)
    if len(parts) == 3 and parts[0] == "mcp":
        return parts[1], parts[2]
    return "", name


def _haystack(schema: dict) -> str:
    fn = schema.get("function") or {}
    words = [fn.get("description") or ""]
    words.extend((fn.get("parameters") or {}).get("properties") or {})
    return " ".join(words).lower()


class DeferredTools:
    """Hidden MCP schemas of one conversation and the ones it has loaded."""

    def __init__(self, schemas: list[dict]):
        self._schemas = {_schema_name(s): s for s in schemas}
        self._loaded: dict[str, None] = {}
        self._search_schema = _search_schema(list(self._schemas))

    def fresh(self) -> "DeferredTools":
        """Same catalog, nothing loaded: for a subagent's own conversation."""
        return DeferredTools(list(self._schemas.values()))

    def copy(self) -> "DeferredTools":
        """Same catalog and loads, then independent: for a forked conversation."""
        clone = self.fresh()
        clone._loaded = dict(self._loaded)
        return clone

    def reset(self) -> None:
        self._loaded.clear()

    @property
    def search_schema(self) -> dict:
        return self._search_schema

    def __contains__(self, name: str) -> bool:
        return name in self._schemas

    def load(self, name: str) -> bool:
        if name not in self._schemas or name in self._loaded:
            return False
        self._loaded[name] = None
        return True

    def with_loaded(self, tools: list | None) -> list | None:
        """*tools* plus every loaded schema it does not already carry."""
        if tools is None or not self._loaded:
            return tools
        present = {_schema_name(t) for t in tools}
        extra = [self._schemas[n] for n in self._loaded if n not in present]
        return tools + extra if extra else tools

    def search(self, query) -> str:
        """Rank deferred tools against *query* and load the best matches.

        An exact name loads only itself; otherwise name hits rank above
        description and parameter hits, then name order, so equal queries
        always load the same tools.
        """
        q = query.strip().lower() if isinstance(query, str) else ""
        words = [w for w in _QUERY_SPLIT_RE.split(q) if w]
        if not words:
            return f"error: {TOOL_SEARCH_NAME}: query must not be empty"
        ranked = []
        for name, schema in self._schemas.items():
            lowered = name.lower()
            exact = q in (lowered, _split_name(lowered)[1])
            hay = _haystack(schema)
            score = sum(
                _NAME_HIT if w in lowered else _DESCRIPTION_HIT if w in hay else 0
                for w in words
            )
            if exact or score:
                ranked.append((not exact, -score, name))
        if not ranked:
            return (
                f"No deferred MCP tools matched {query!r}. "
                "Try other keywords or an exact name from the catalog."
            )
        ranked.sort()
        # An exact name would otherwise pull in its namespace neighbours,
        # since words like "mcp" and the server name match every tool.
        if not ranked[0][0]:
            ranked = [entry for entry in ranked if not entry[0]]
        hits = [name for _, _, name in ranked[:MAX_LOADS]]
        overflow = [name for _, _, name in ranked[MAX_LOADS:]]
        new = [name for name in hits if self.load(name)]
        lines = []
        if new:
            plural = "tool" if len(new) == 1 else "tools"
            lines.append(
                f"Loaded {len(new)} {plural}, callable from your next message:"
            )
            lines.extend(f"- `{name}`" for name in new)
        already = [name for name in hits if name not in new]
        if already:
            lines.append("Already loaded: " + ", ".join(f"`{n}`" for n in already))
        if overflow:
            shown = ", ".join(f"`{n}`" for n in overflow[:MAX_OVERFLOW_NAMES])
            more = len(overflow) - MAX_OVERFLOW_NAMES
            if more > 0:
                shown += f" and {more} more"
            lines.append(
                f"Also matched but not loaded: {shown}. "
                "Search an exact tool name to load it."
            )
        return "\n".join(lines)

    def summary_line(self) -> str | None:
        if not self._loaded:
            return None
        return f"MCP schemas: {len(self._loaded)} of {len(self._schemas)} loaded"


def _search_schema(names: list[str]) -> dict:
    by_server: dict[str, list[str]] = {}
    for name in names:
        server, tool = _split_name(name)
        by_server.setdefault(server, []).append(tool)
    catalog = "\n".join(
        f"{server}: {', '.join(tools)}" for server, tools in by_server.items()
    )
    return {
        "type": "function",
        "function": {
            "name": TOOL_SEARCH_NAME,
            "description": (
                "Load deferred MCP tools, callable as mcp__<server>__<tool> from your "
                "next message. An exact name loads that tool; keywords match names, "
                f"descriptions and parameters and load up to {MAX_LOADS}. "
                f"Deferred tools:\n{catalog}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords or an exact tool name.",
                    }
                },
                "required": ["query"],
            },
        },
    }


def defer_mcp_tools(
    mcp_tools: list[dict], *, provider: str | None, enabled: bool = True
) -> DeferredTools | None:
    """Return a :class:`DeferredTools` when *mcp_tools* are worth deferring.

    The command provider renders its catalog into the system prompt once, so
    it keeps the full schemas.
    """
    if not enabled or not mcp_tools or provider == "command":
        return None
    if count_tokens(json.dumps(mcp_tools)) <= DEFER_MIN_TOKENS:
        return None
    return DeferredTools(mcp_tools)

"""Inline interactive model selector.

The picker renders a filterable, scrollable list of :class:`ModelEntry` rows
on stderr using a small prompt_toolkit application: type to filter, arrows to
move, Enter to select, ``*`` to toggle a favorite, ``^t`` to toggle the
tools-only filter, ``^r`` to refresh the catalog, ``^s`` to search all of
HuggingFace, Esc to cancel. A numbered-list fallback covers terminals that
cannot host the application.

All selection/ranking logic lives in pure functions and :class:`PickerState`
so it is testable without a terminal. One ``PickerState`` is shared across
picker rounds, so filter text, the tools toggle, and favorite changes carry
over refreshes and searches.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from functools import lru_cache

from . import fmt
from .model_catalog import ModelEntry, _fmt_ctx, _fmt_price

MAX_VISIBLE_ROWS = 12
_BASIC_LIST_LIMIT = 20


@dataclass
class PickResult:
    """Outcome of one picker round.

    ``kind`` is one of ``selected`` (entry is set), ``manual`` (user wants to
    type an id), ``refresh``, ``search`` (HuggingFace-wide search requested),
    or ``cancel``.
    """

    kind: str
    entry: ModelEntry | None = None


@lru_cache(maxsize=4096)
def _match_texts(entry: ModelEntry) -> tuple[str, ...]:
    texts = [entry.id.lower()]
    if entry.display_name:
        texts.append(entry.display_name.lower())
    return tuple(texts)


def match_score(query: str, entry: ModelEntry) -> int | None:
    """Rank how well *entry* matches *query*; lower is better, None is no match."""
    if not query:
        return 0
    q = query.lower()
    best = None
    for t in _match_texts(entry):
        if q == t:
            return 0
        if t.startswith(q):
            best = min(best, 1) if best is not None else 1
        elif q in t:
            best = min(best, 2) if best is not None else 2
        elif best is None and _is_subsequence(q, t):
            best = 3
    return best


def _is_subsequence(needle: str, haystack: str) -> bool:
    it = iter(haystack)
    return all(ch in it for ch in needle)


def rank_entries(
    entries: list[ModelEntry],
    *,
    query: str = "",
    favorites: frozenset | set = frozenset(),
    recents: tuple | list = (),
    current: str | None = None,
) -> list[ModelEntry]:
    """Filter by *query* and order: favorites, current, recents, catalog order."""
    recent_pos = {m: i for i, m in enumerate(recents)}
    ranked = []
    for idx, e in enumerate(entries):
        score = match_score(query, e)
        if score is None:
            continue
        if e.id in favorites:
            group, sub = 0, 0
        elif e.id == current:
            group, sub = 1, 0
        elif e.id in recent_pos:
            group, sub = 2, recent_pos[e.id]
        else:
            group, sub = 3, 0
        ranked.append((group, score, sub, idx, e))
    ranked.sort(key=lambda t: t[:4])
    return [t[4] for t in ranked]


def _passes_tools_filter(entry: ModelEntry) -> bool:
    # Hide only models known NOT to support tools; unknown stays visible so
    # local providers (which report nothing) are never blanked out.
    return entry.supports_tools is not False


@dataclass
class PickerState:
    """All mutable picker state, kept UI-free for headless testing.

    Every prompt_toolkit redraw re-reads the ranked list several times, so
    :meth:`matching` is memoized; the mutating methods (``set_query``,
    ``set_entries``, ``set_favorite``) invalidate the cache.
    """

    entries: list[ModelEntry]
    favorites: set = field(default_factory=set)
    recents: list = field(default_factory=list)
    current: str | None = None
    query: str = ""
    cursor: int = 0
    offset: int = 0
    tools_only: bool = False
    _matching: list[ModelEntry] | None = field(default=None, repr=False)

    def _invalidate(self) -> None:
        self._matching = None

    def matching(self) -> list[ModelEntry]:
        if self._matching is None:
            self._matching = rank_entries(
                self.entries,
                query=self.query,
                favorites=self.favorites,
                recents=self.recents,
                current=self.current,
            )
        return self._matching

    def visible(self) -> list[ModelEntry]:
        matching = self.matching()
        if self.tools_only:
            return [e for e in matching if _passes_tools_filter(e)]
        return matching

    def hidden_by_tools(self) -> int:
        if not self.tools_only:
            return 0
        matching = self.matching()
        return len(matching) - sum(1 for e in matching if _passes_tools_filter(e))

    def rows(self) -> list[ModelEntry | None]:
        """Selectable rows: visible entries plus the manual-entry row (None)."""
        rows: list[ModelEntry | None] = list(self.visible())
        rows.append(None)
        return rows

    def clamp(self) -> None:
        rows = self.rows()
        self.cursor = max(0, min(self.cursor, len(rows) - 1)) if rows else 0
        if self.cursor < self.offset:
            self.offset = self.cursor
        elif self.cursor >= self.offset + MAX_VISIBLE_ROWS:
            self.offset = self.cursor - MAX_VISIBLE_ROWS + 1

    def move(self, delta: int) -> None:
        self.cursor += delta
        self.clamp()

    def set_query(self, query: str) -> None:
        if query != self.query:
            self.query = query
            self.cursor = 0
            self.offset = 0
            self._invalidate()

    def set_entries(self, entries: list[ModelEntry]) -> None:
        self.entries = entries
        self._invalidate()
        self.clamp()

    def set_favorite(self, model_id: str, is_favorite: bool) -> None:
        if is_favorite:
            self.favorites.add(model_id)
        else:
            self.favorites.discard(model_id)
        self._invalidate()
        self.keep_cursor_on(model_id)

    def selected(self) -> ModelEntry | None:
        rows = self.rows()
        if 0 <= self.cursor < len(rows):
            return rows[self.cursor]
        return None

    def on_manual_row(self) -> bool:
        rows = self.rows()
        return bool(rows) and self.cursor == len(rows) - 1

    def toggle_tools_only(self) -> None:
        self.tools_only = not self.tools_only
        self.cursor = 0
        self.offset = 0

    def keep_cursor_on(self, model_id: str) -> None:
        for i, row in enumerate(self.rows()):
            if row is not None and row.id == model_id:
                self.cursor = i
                break
        self.clamp()


def format_entry(
    entry: ModelEntry,
    *,
    is_favorite: bool = False,
    is_current: bool = False,
    id_width: int = 40,
) -> str:
    """One plain-text list row (used by the fallback picker and tests)."""
    star = "★" if is_favorite else " "
    name = entry.id + (" (current)" if is_current else "")
    cols = [f"{star} {name:<{id_width}}"]
    cols.append(
        f"{_fmt_ctx(entry.context_length):>6}" if entry.context_length else " " * 6
    )
    cols.append(f"{_fmt_price(entry.price_in, entry.price_out):>13}")
    cols.append("tools" if entry.supports_tools else "     ")
    meta = []
    if entry.loaded is True:
        meta.append("loaded")
    meta.extend(entry.tags)
    if meta:
        cols.append(" ".join(meta))
    return "  ".join(cols).rstrip()


def _id_width(rows: list[ModelEntry | None]) -> int:
    widths = [len(r.id) for r in rows if r is not None]
    return min(max(widths, default=20) + 10, 54)


def pick_model(
    state: PickerState,
    *,
    title: str,
    allow_search: bool = False,
    on_toggle_favorite=None,
    input=None,
    output=None,
) -> PickResult:
    """Run one interactive picker round over *state* and return the outcome.

    Falls back to a numbered list when the inline application cannot run,
    unless an explicit *input*/*output* pair was supplied (tests).
    """
    from .fmt import suspend_live

    with suspend_live():
        explicit_io = input is not None or output is not None
        try:
            return _pick_with_app(
                state,
                title=title,
                allow_search=allow_search,
                on_toggle_favorite=on_toggle_favorite,
                input=input,
                output=output,
            )
        except (KeyboardInterrupt, EOFError):
            return PickResult(kind="cancel")
        except Exception:
            if explicit_io:
                raise
            return _pick_basic(state, title=title)


def _pick_with_app(
    state: PickerState,
    *,
    title: str,
    allow_search: bool,
    on_toggle_favorite,
    input=None,
    output=None,
) -> PickResult:
    from prompt_toolkit.application import Application
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
    from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
    from prompt_toolkit.layout.dimension import Dimension
    from prompt_toolkit.output import create_output
    from prompt_toolkit.styles import Style

    query_buffer = Buffer(multiline=False)
    if state.query:
        query_buffer.text = state.query
        query_buffer.cursor_position = len(state.query)

    def _on_text_changed(_buf) -> None:
        state.set_query(query_buffer.text)

    query_buffer.on_text_changed += _on_text_changed

    def _title_fragments():
        count = len(state.visible())
        total = len(state.entries)
        counter = f"{count} of {total}" if count != total else f"{total} models"
        return [
            ("class:picker.title", f"{title}"),
            ("class:picker.meta", f" · {counter} · type to filter"),
        ]

    def _list_fragments():
        state.clamp()
        rows = state.rows()
        if not rows:
            return [("class:picker.meta", "  (no matches)")]
        fragments = []
        width = _id_width(rows)
        window = rows[state.offset : state.offset + MAX_VISIBLE_ROWS]
        for i, row in enumerate(window, start=state.offset):
            is_cursor = i == state.cursor
            prefix = "class:picker.cursor " if is_cursor else ""
            pointer = "❯ " if is_cursor else "  "
            if row is None:
                fragments.append(
                    (
                        prefix + "class:picker.meta",
                        f"{pointer}(type a model id manually)\n",
                    )
                )
                continue
            line = format_entry(
                row,
                is_favorite=row.id in state.favorites,
                is_current=row.id == state.current,
                id_width=width,
            )
            style = prefix
            if row.id in state.favorites:
                style += " class:picker.star"
            elif row.id == state.current:
                style += " class:picker.current"
            fragments.append((style.strip(), f"{pointer}{line}\n"))
        if state.offset + MAX_VISIBLE_ROWS < len(rows):
            remaining = len(rows) - state.offset - MAX_VISIBLE_ROWS
            fragments.append(("class:picker.meta", f"  … {remaining} more\n"))
        return fragments

    def _detail_fragments():
        row = state.selected()
        if row is None or not row.detail:
            return []
        return [("class:picker.detail", f"  {row.detail}")]

    def _help_fragments():
        hidden = state.hidden_by_tools()
        tools_bit = (
            f"^t tools-only ({hidden} hidden)" if state.tools_only else "^t tools-only"
        )
        parts = ["↑↓ move", "enter select", "* favorite", tools_bit, "^r refresh"]
        if allow_search:
            parts.append("^s search all of HF")
        parts.append("esc cancel")
        return [("class:picker.help", "  " + " · ".join(parts))]

    kb = KeyBindings()

    @kb.add("up")
    def _up(event):
        state.move(-1)

    @kb.add("down")
    def _down(event):
        state.move(1)

    @kb.add("pageup")
    def _pgup(event):
        state.move(-MAX_VISIBLE_ROWS)

    @kb.add("pagedown")
    def _pgdn(event):
        state.move(MAX_VISIBLE_ROWS)

    @kb.add("enter")
    def _enter(event):
        if state.on_manual_row() or not state.rows():
            event.app.exit(result=PickResult(kind="manual"))
            return
        row = state.selected()
        if row is not None:
            event.app.exit(result=PickResult(kind="selected", entry=row))

    @kb.add("escape", eager=True)
    @kb.add("c-c")
    @kb.add("c-g")
    def _cancel(event):
        event.app.exit(result=PickResult(kind="cancel"))

    @kb.add("c-t")
    def _tools(event):
        state.toggle_tools_only()

    @kb.add("c-r")
    def _refresh(event):
        event.app.exit(result=PickResult(kind="refresh"))

    if allow_search:

        @kb.add("c-s")
        def _search(event):
            event.app.exit(result=PickResult(kind="search"))

    @kb.add("*", eager=True)
    def _favorite(event):
        row = state.selected()
        if row is None or on_toggle_favorite is None:
            return
        state.set_favorite(row.id, on_toggle_favorite(row.id))

    def _list_height():
        rows = len(state.rows())
        return Dimension(
            min=1, max=MAX_VISIBLE_ROWS + 1, preferred=min(rows, MAX_VISIBLE_ROWS) + 1
        )

    layout = Layout(
        HSplit(
            [
                Window(FormattedTextControl(_title_fragments), height=1),
                VSplit(
                    [
                        Window(
                            FormattedTextControl([("class:picker.prompt", "❯ ")]),
                            width=2,
                        ),
                        Window(BufferControl(buffer=query_buffer), height=1),
                    ]
                ),
                Window(FormattedTextControl(_list_fragments), height=_list_height),
                Window(FormattedTextControl(_detail_fragments), height=1),
                Window(FormattedTextControl(_help_fragments), height=1),
            ]
        )
    )

    style = Style.from_dict(
        {
            "picker.title": "bold",
            "picker.prompt": "bold fg:ansicyan",
            "picker.cursor": "reverse",
            "picker.star": "fg:ansiyellow",
            "picker.current": "fg:ansigreen",
            "picker.meta": "fg:ansibrightblack",
            "picker.detail": "fg:ansicyan",
            "picker.help": "fg:ansibrightblack",
        }
    )

    app = Application(
        layout=layout,
        key_bindings=kb,
        style=style,
        erase_when_done=True,
        mouse_support=False,
        input=input,
        output=output if output is not None else create_output(sys.stderr),
    )
    result = app.run()
    if result is None:
        return PickResult(kind="cancel")
    return result


def _mini_prompt(label: str, *, default: str = "", input=None, output=None) -> str:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.output import create_output

    from .fmt import suspend_live

    session = PromptSession(
        input=input,
        output=output if output is not None else create_output(sys.stderr),
    )
    try:
        with suspend_live():
            return session.prompt(f"{label}: ", default=default).strip()
    except (KeyboardInterrupt, EOFError):
        return ""


def choose_model(
    provider: str,
    base_url: str | None = None,
    api_key: str | None = None,
    *,
    current: str | None = None,
    initial_query: str = "",
    input=None,
    output=None,
) -> tuple[str, ModelEntry | None] | None:
    """Drive the full explorer flow for *provider* until the user decides.

    Handles picker rounds, catalog refresh, manual entry, and (for the public
    HuggingFace endpoint) hub-wide search. Returns ``(model_id, entry)`` where
    ``entry`` is None for manually typed ids, or None when the user cancels.
    Raises :class:`~swival.model_catalog.CatalogUnavailable` when the initial
    catalog cannot be fetched at all.
    """
    from .fmt import suspend_live

    # One suspension for the whole flow, so the REPL prompt does not flash
    # between picker rounds.
    with suspend_live():
        from .model_catalog import CatalogUnavailable, is_hf_router, list_models
        from .model_prefs import load_prefs, toggle_favorite

        hf_explorer = is_hf_router(provider, base_url)
        catalog = list_models(provider, base_url, api_key)

        prefs = load_prefs()
        if prefs.warning:
            fmt.warning(prefs.warning)

        state = PickerState(
            entries=catalog.entries,
            favorites=set(prefs.favorites_for(provider)),
            recents=prefs.recents_for(provider),
            current=current,
            query=initial_query,
            # Swival is a tool-calling agent, so the explorer starts on the
            # tools-capable subset; the hidden count keeps that visible and ^t
            # lifts it.
            tools_only=hf_explorer,
        )
        search_query: str | None = None  # None = browsing the main catalog

        while True:
            if search_query is None:
                title = f"Select a model · {catalog.source}"
            else:
                title = f"HuggingFace search results for {search_query!r}"

            res = pick_model(
                state,
                title=title,
                allow_search=hf_explorer,
                on_toggle_favorite=lambda mid: toggle_favorite(provider, mid),
                input=input,
                output=output,
            )

            if res.kind == "selected":
                return res.entry.id, res.entry

            if res.kind == "manual":
                text = _mini_prompt(
                    "Model id (blank to go back)",
                    default=state.query,
                    input=input,
                    output=output,
                )
                if text:
                    return text, None
                continue

            if res.kind == "refresh":
                try:
                    if search_query is None:
                        catalog = list_models(provider, base_url, api_key, refresh=True)
                        state.set_entries(catalog.entries)
                    else:
                        from .model_catalog import search_hf_models

                        state.set_entries(search_hf_models(search_query, api_key))
                except CatalogUnavailable as e:
                    fmt.warning(f"refresh failed: {e.reason}")
                continue

            if res.kind == "search":
                q = _mini_prompt(
                    "Search all of Hugging Face (blank to go back)",
                    default=state.query,
                    input=input,
                    output=output,
                )
                if not q:
                    continue
                from .model_catalog import search_hf_models

                try:
                    results = search_hf_models(q, api_key)
                except CatalogUnavailable as e:
                    fmt.warning(f"search failed: {e.reason}")
                    continue
                if not results:
                    fmt.info(f"no provider-served chat models match {q!r} on the hub")
                    continue
                search_query = q
                state.set_entries(results)
                state.set_query("")
                continue

            # cancel: inside search results, back out to the main catalog first.
            if search_query is not None:
                search_query = None
                state.set_entries(catalog.entries)
                state.set_query("")
                continue
            return None


def _pick_basic(state: PickerState, *, title: str) -> PickResult:
    """Numbered-list fallback for terminals that cannot host the inline app.

    Typing a number selects, ``m`` switches to manual entry, plain text
    refines the filter, and a blank line cancels.
    """
    from prompt_toolkit import PromptSession
    from prompt_toolkit.output import create_output

    session = PromptSession(output=create_output(sys.stderr))
    err = sys.stderr

    while True:
        rows = [r for r in state.rows() if r is not None][:_BASIC_LIST_LIMIT]
        print(f"\n{title}", file=err)
        if not rows:
            print("  (no matches)", file=err)
        width = _id_width(list(rows))
        for i, row in enumerate(rows, 1):
            line = format_entry(
                row,
                is_favorite=row.id in state.favorites,
                is_current=row.id == state.current,
                id_width=width,
            )
            print(f"  {i:>2}. {line}", file=err)
        total = len(state.visible())
        if total > len(rows):
            print(f"  … {total - len(rows)} more, type text to filter", file=err)
        print(file=err)

        if rows:
            hint = f"1-{len(rows)}, text to filter, m for manual, blank to cancel"
        else:
            hint = "text to filter, m for manual, blank to cancel"
        try:
            raw = session.prompt(f"Model [{hint}]: ").strip()
        except (KeyboardInterrupt, EOFError):
            return PickResult(kind="cancel")

        if not raw:
            return PickResult(kind="cancel")
        if raw.lower() == "m":
            return PickResult(kind="manual")
        try:
            n = int(raw)
        except ValueError:
            state.set_query(raw)
            continue
        if 1 <= n <= len(rows):
            return PickResult(kind="selected", entry=rows[n - 1])
        print(f"  Please enter a number between 1 and {len(rows)}.", file=err)

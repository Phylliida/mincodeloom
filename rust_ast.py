"""
rust_ast.py — Minimal-token Rust codebase explorer & editor, with .todo/.md support.

Install:
    pip install tree-sitter tree-sitter-verus

Reading (Rust files):
    tree.query()                                    → ls project root
    tree.query("src")                               → ls src/
    tree.query("src/main.rs")                       → item names
    tree.query("src/main.rs", "impl Server")        → child names or source
    tree.query("src/main.rs", "impl Server", "new") → full source
    tree.query(..., offset=15)                       → pagination

    Multi-segment filesystem paths are auto-joined:
    tree.query("crates", "vcad-geometry", "src", "lib.rs")
      is equivalent to:
    tree.query("crates/vcad-geometry/src/lib.rs")

    Macro invocations (e.g. verus!) are expanded by re-parsing their
    token-tree body:
    tree.query("src/lib.rs", "verus!")              → list items inside macro
    tree.query("src/lib.rs", "verus!", "fn foo")    → source of fn foo
    tree.query("src/lib.rs", "verus!", "impl Bar")  → list methods of impl Bar

Reading (.todo / .md files):
    tree.query("plan.todo")                            → list section headers
    tree.query("plan.todo", "Section Title")           → list items or raw body
    tree.query("plan.todo", "Section Title", "Item…")  → item detail + notes

Searching:
    tree.search("function_name")                    → find by substring
    tree.search("function_name", exact=True)        → exact match only
    tree.search("function_name", kind="function_item") → filter by node type

Editing (Rust files):
    tree.edit("src/new.rs", action="create_file", content="// new file")
    tree.edit("src/old.rs", action="delete_file")
    tree.edit("src/lib.rs", action="append", content="fn foo() {}")
    tree.edit("src/lib.rs", "old_fn", action="replace", content="fn new_fn() {}")
    tree.edit("src/lib.rs", "old_fn", action="delete")
    tree.edit("src/lib.rs", "some_fn", action="insert_before", content="/// doc")
    tree.edit("src/lib.rs", "some_fn", action="insert_after", content="fn after() {}")
    tree.edit("src/lib.rs", "impl Server", action="append_child", content="    fn m(&self) {}")
    tree.edit("src/lib.rs", "impl Server", "handle", action="replace", content="fn handle() {}")
    tree.edit("src/lib.rs", "verus!", action="append_child", content="    fn new_fn() {}")
    tree.edit("src/lib.rs", "verus!", "fn old", action="replace", content="fn new() {}")

Editing (.todo / .md files):
    tree.edit("plan.todo", action="add_header", content="New Section")
    tree.edit("plan.todo", "Section", action="add_item", content="Task text")
    tree.edit("plan.todo", "Section", "Item text", action="check")
    tree.edit("plan.todo", "Section", "Item text", action="uncheck")
    tree.edit("plan.todo", "Section", "Item text", action="add_note", content="Note text")
    tree.edit("plan.todo", "Section", "Item text", action="delete")
    tree.edit("plan.todo", "Section", action="delete")
    tree.edit("plan.todo", "Section", "Item text", action="replace", content="New text")

Attributes:
    Preceding #[…] attribute nodes are attached to their following item.
    Listings show attributes inline, e.g.:
        #[cfg(test)] mod tests
    Both the display name (with attrs) and the base name (without) can be
    used to address an item in queries and edits.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import tree_sitter_verus as ts_verus
from tree_sitter import Language, Parser, Node

RUST_LANGUAGE = Language(ts_verus.language())
LIMIT = 15

TOP_LEVEL_TYPES = {
    "function_item", "struct_item", "enum_item", "trait_item",
    "impl_item", "type_item", "const_item", "static_item",
    "mod_item", "use_declaration", "macro_definition",
    "macro_invocation",
}

CONTAINER_TYPES = {"impl_item", "trait_item", "struct_item", "enum_item", "mod_item"}

BODY_NODE_TYPES: dict[str, tuple[str, ...]] = {
    "function_item":    ("block",),
    "impl_item":        ("declaration_list",),
    "trait_item":       ("declaration_list",),
    "struct_item":      ("field_declaration_list",),
    "enum_item":        ("enum_variant_list",),
    "mod_item":         ("declaration_list",),
    "macro_invocation": ("token_tree",),
}

EDIT_ACTIONS = {
    "create_file", "delete_file", "append",
    "replace", "delete", "insert_before", "insert_after",
    "append_child",
    # todo-specific
    "add_header", "add_item", "check", "uncheck", "add_note",
}

TODO_EXTENSIONS = {".todo", ".md"}
TODO_ONLY_ACTIONS = {"add_header", "add_item", "check", "uncheck", "add_note"}

_TODO_ITEM_RE = re.compile(r"^- \[([ xX])\] ?(.*)")
_SECTION_PROGRESS_RE = re.compile(r"\s*\(\d+/\d+\s+done\)\s*$")
_CHECKBOX_PREFIX_RE = re.compile(r"^\[[ xX]\]\s*")
_FULL_CHECKBOX_PREFIX_RE = re.compile(r"^-\s*\[[ xX]\]\s*")


# ── Attributed node wrapper ──────────────────────────────────

@dataclass
class _AttrNode:
    """An AST node bundled with its preceding #[...] attribute nodes.

    ``start_byte`` covers the first attribute (or the node itself if none).
    ``end_byte`` is the inner node's end.
    """
    node: Node
    attr_start: int  # first byte including leading attributes

    @property
    def start_byte(self) -> int:
        return self.attr_start

    @property
    def end_byte(self) -> int:
        return self.node.end_byte

    @property
    def type(self) -> str:
        return self.node.type


@dataclass
class _OffsetAttrNode:
    """Proxy for an _AttrNode whose byte positions are shifted so they
    refer to the *original* file rather than a re-parsed sub-buffer.

    Used when addressing children inside a macro invocation body that
    was re-parsed independently.
    """
    _inner: _AttrNode
    _byte_offset: int

    @property
    def start_byte(self) -> int:
        return self._inner.start_byte + self._byte_offset

    @property
    def end_byte(self) -> int:
        return self._inner.end_byte + self._byte_offset

    @property
    def type(self) -> str:
        return self._inner.type

    @property
    def node(self) -> Node:
        return self._inner.node


def _collect_items(
    named_children: list[Node],
    valid_types: set[str] | None = None,
) -> list[_AttrNode]:
    """Group consecutive ``attribute_item`` nodes with their following
    declaration.

    If *valid_types* is given, only children whose type is in that set
    are kept (attributes before a skipped child are discarded).  If
    ``None``, every non-attribute named child is kept.
    """
    result: list[_AttrNode] = []
    pending_attr_start: int | None = None
    for child in named_children:
        if child.type == "attribute_item":
            if pending_attr_start is None:
                pending_attr_start = child.start_byte
        elif valid_types is None or child.type in valid_types:
            start = (
                pending_attr_start
                if pending_attr_start is not None
                else child.start_byte
            )
            result.append(_AttrNode(node=child, attr_start=start))
            pending_attr_start = None
        else:
            pending_attr_start = None
    return result


# ── Normalizers for round-tripping query output → edit input ──

def _normalize_section_key(raw: str) -> str:
    """Strip leading '#'s and trailing '(N/M done)' so formatted query
    output like '## Title (2/5 done)' matches the stored title 'Title'."""
    s = raw.strip()
    while s.startswith("#"):
        s = s[1:]
    s = s.strip()
    s = _SECTION_PROGRESS_RE.sub("", s)
    return s.strip()


def _normalize_item_key(raw: str) -> str:
    """Strip leading '[x] ' or '[ ] ' so formatted query output like
    '[x] Do the thing' matches the stored text 'Do the thing'."""
    s = raw.strip()
    s = _CHECKBOX_PREFIX_RE.sub("", s)
    return s.strip()


def _strip_checkbox_content(raw: str) -> str:
    """Strip any leading checkbox markup from user-supplied content
    so that add_item / replace don't double up: '- [ ] [ ] text'."""
    s = raw.strip()
    s = _FULL_CHECKBOX_PREFIX_RE.sub("", s)
    s = _CHECKBOX_PREFIX_RE.sub("", s)
    return s.strip()


# ── Rust AST helpers ──────────────────────────────────────────

def _text(node: Node, source: bytes) -> str:
    """Raw source text of a single tree-sitter Node."""
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _attr_text(anode: _AttrNode, source: bytes) -> str:
    """Full source text of a node *including* its preceding attributes."""
    return source[anode.start_byte:anode.end_byte].decode("utf-8", errors="replace")


def _find_body(node: Node) -> Optional[Node]:
    for child in node.children:
        if child.type in BODY_NODE_TYPES.get(node.type, ()):
            return child
    return None


def _body_children(anode: _AttrNode) -> list[_AttrNode]:
    # For macro invocations the token_tree children are raw tokens,
    # not meaningful Rust items.  Use _parse_macro_body instead.
    if anode.type == "macro_invocation":
        return []
    body = _find_body(anode.node)
    if not body:
        return []
    return _collect_items(list(body.named_children))


def _extract_name(node: Node, source: bytes) -> str:
    """Short structural name of a node (no preceding attributes)."""
    if node.type == "impl_item":
        type_node = node.child_by_field_name("type")
        trait_node = node.child_by_field_name("trait")
        type_params = node.child_by_field_name("type_parameters")
        parts = ["impl"]
        if type_params:
            parts.append(_text(type_params, source))
        if trait_node and type_node:
            parts.extend([_text(trait_node, source), "for", _text(type_node, source)])
        elif type_node:
            parts.append(_text(type_node, source))
        return " ".join(parts)

    if node.type == "use_declaration":
        return _text(node, source).strip().rstrip(";")

    if node.type == "attribute_item":
        return _text(node, source).strip()

    if node.type == "macro_invocation":
        macro_node = node.child_by_field_name("macro")
        if macro_node:
            return _text(macro_node, source) + "!"
        if node.named_children:
            return _text(node.named_children[0], source) + "!"
        return _text(node, source).strip()[:50]

    if node.type in ("field_declaration", "enum_variant"):
        name_node = node.child_by_field_name("name")
        return (
            _text(name_node, source)
            if name_node
            else _text(node, source).strip().rstrip(",")
        )

    name_node = node.child_by_field_name("name")
    return _text(name_node, source) if name_node else _text(node, source).strip()


def _display_name(anode: _AttrNode, source: bytes) -> str:
    """Name for display, including any preceding attributes collapsed
    to a single line."""
    base = _extract_name(anode.node, source)
    if anode.attr_start < anode.node.start_byte:
        attr_bytes = source[anode.attr_start : anode.node.start_byte]
        attr_text = " ".join(
            attr_bytes.decode("utf-8", errors="replace").split()
        )
        if attr_text:
            return f"{attr_text} {base}"
    return base


def _child_count_suffix(anode: _AttrNode, source: bytes, tree: "CodebaseTree | None" = None) -> str:
    if anode.type in CONTAINER_TYPES:
        children = _body_children(anode)
        if children:
            return f" ({len(children)} children)"
    if anode.type == "macro_invocation" and tree is not None:
        _, items, _ = tree._parse_macro_body(anode, source)
        if items:
            return f" ({len(items)} children)"
    return ""


def _dedup_names(names: list[str], suffixes: list[str] | None = None) -> list[str]:
    counts: Counter[str] = Counter(names)
    seen: Counter[str] = Counter()
    out = []
    for i, name in enumerate(names):
        seen[name] += 1
        deduped = f"{name} #{seen[name]}" if counts[name] > 1 else name
        if suffixes:
            deduped += suffixes[i]
        out.append(deduped)
    return out


def _paginate(items: list[str], offset: int, path: list[str] | None = None, search_hint: dict | None = None) -> str:
    page = items[offset:offset + LIMIT]
    out = "\n".join(page)
    remaining = len(items) - offset - len(page)
    if remaining > 0:
        next_offset = offset + LIMIT
        if path is not None:
            out += f"\n... +{remaining} more: browse(path={path!r}, offset={next_offset})"
        elif search_hint is not None:
            parts = [f"name={search_hint['name']!r}"]
            if search_hint.get("exact"):
                parts.append("exact=true")
            if search_hint.get("kind"):
                parts.append(f"kind={search_hint['kind']!r}")
            parts.append(f"offset={next_offset}")
            out += f"\n... +{remaining} more: search({', '.join(parts)})"
        else:
            out += f"\n... +{remaining} more (offset={next_offset})"
    return out


# ── Todo structures & parser ─────────────────────────────────

@dataclass
class _TodoItem:
    checked: bool
    text: str
    notes: list[str] = field(default_factory=list)
    line_start: int = 0
    line_end: int = 0   # inclusive


@dataclass
class _TodoSection:
    title: str
    level: int
    items: list[_TodoItem] = field(default_factory=list)
    line_start: int = 0
    line_end: int = 0   # inclusive


def _parse_todo(text: str) -> tuple[list[str], list[_TodoSection]]:
    """Parse a .todo/.md file into (lines, sections)."""
    lines = text.split("\n")
    sections: list[_TodoSection] = []
    cur_section: Optional[_TodoSection] = None
    cur_item: Optional[_TodoItem] = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        # ── header ─────────────────────────────────────────
        if stripped.startswith("#"):
            level = 0
            for ch in stripped:
                if ch == "#":
                    level += 1
                else:
                    break
            if level < len(stripped) and stripped[level] == " ":
                if cur_section is not None:
                    cur_section.line_end = i - 1
                cur_item = None
                title = stripped[level:].strip()
                cur_section = _TodoSection(
                    title=title, level=level, line_start=i, line_end=i,
                )
                sections.append(cur_section)
                continue

        # ── checkbox item ──────────────────────────────────
        m = _TODO_ITEM_RE.match(stripped)
        if m:
            checked = m.group(1) in ("x", "X")
            item_text = m.group(2)
            cur_item = _TodoItem(
                checked=checked, text=item_text, line_start=i, line_end=i,
            )
            if cur_section is None:
                cur_section = _TodoSection(
                    title="(top)", level=0, line_start=i, line_end=i,
                )
                sections.append(cur_section)
            cur_section.items.append(cur_item)
            continue

        # ── note (indented continuation) ───────────────────
        if (cur_item is not None and stripped
                and (line.startswith("  ") or line.startswith("\t"))):
            cur_item.notes.append(stripped)
            cur_item.line_end = i
            continue

    # finalize last section
    if cur_section is not None:
        cur_section.line_end = len(lines) - 1

    return lines, sections


def _find_section(sections: list[_TodoSection], title: str) -> _TodoSection:
    """Find a section by title.  Accepts both the raw title and the
    formatted query output (e.g. '## Title (2/5 done)')."""
    titles = [s.title for s in sections]
    deduped = _dedup_names(titles)

    # exact match first
    for section, name in zip(sections, deduped):
        if name == title:
            return section

    # normalized match: strip '#' prefix and '(N/M done)' suffix
    norm_input = _normalize_section_key(title)
    if norm_input:
        for section, name in zip(sections, deduped):
            if name == norm_input or _normalize_section_key(name) == norm_input:
                return section

    raise KeyError(
        f"Section {title!r} not found. Available:\n" + "\n".join(deduped)
    )


def _find_todo_item(items: list[_TodoItem], item_text: str) -> _TodoItem:
    """Find an item by text.  Accepts both the raw text and the
    formatted query output (e.g. '[x] Task description')."""
    texts = [it.text for it in items]
    deduped = _dedup_names(texts)

    # exact match first
    for item, name in zip(items, deduped):
        if name == item_text:
            return item

    # normalized match: strip '[x] ' or '[ ] ' prefix
    norm_input = _normalize_item_key(item_text)
    if norm_input:
        for item, name in zip(items, deduped):
            if name == norm_input or _normalize_item_key(name) == norm_input:
                return item

    raise KeyError(
        f"Item {item_text!r} not found. Available:\n" + "\n".join(deduped)
    )


# ── Search result ─────────────────────────────────────────────

@dataclass
class SearchResult:
    """One match from a recursive search."""
    file: str                   # relative path to the .rs file
    path: tuple[str, ...]       # query-ready path segments after the file
    node_type: str              # tree-sitter node type
    line: int                   # 1-based line number

    @property
    def query_path(self) -> tuple[str, ...]:
        """Full path segments usable with ``tree.query(...)``."""
        return (self.file,) + self.path

    def __str__(self) -> str:
        segments = list((self.file,) + self.path)
        name = self.path[-1] if self.path else self.file
        return f"{name} — browse(path={segments!r})"


# ── CodebaseTree ──────────────────────────────────────────────

class CodebaseTree:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self._parser = Parser(RUST_LANGUAGE)
        self._cache: dict[Path, tuple[bytes, Any]] = {}

    # ── path resolution ────────────────────────────────────────

    def _resolve_segments(self, path: tuple[str, ...]) -> tuple[str, ...]:
        """Join leading path segments that form a filesystem path.

        Tries the longest possible prefix first: join all N segments and
        check if it's a file, then N-1, etc.  Falls back to a filename
        heuristic (last segment containing '.') for not-yet-existing files
        so that create_file / add_header still work with split paths.
        """
        if not path:
            return path

        # 1) Longest prefix that resolves to a file on disk.
        for i in range(len(path), 0, -1):
            candidate = "/".join(path[:i])
            if (self.root / candidate).is_file():
                if i == len(path):
                    return (candidate,)
                return (candidate,) + path[i:]

        # 2) Longest prefix that resolves to a directory on disk.
        for i in range(len(path), 0, -1):
            candidate = "/".join(path[:i])
            if (self.root / candidate).is_dir():
                if i == len(path):
                    return (candidate,)
                return (candidate,) + path[i:]

        # 3) Nothing exists on disk yet. Heuristic: join through the
        #    last segment that looks like a filename (contains '.').
        for i in range(len(path), 0, -1):
            if "." in path[i - 1]:
                if i == len(path):
                    return ("/".join(path),)
                return ("/".join(path[:i]),) + path[i:]

        return path

    # ── parsing & helpers ──────────────────────────────────────

    def _parse(self, filepath: Path) -> tuple[bytes, Any]:
        filepath = filepath.resolve()
        if filepath not in self._cache:
            source = filepath.read_bytes()
            self._cache[filepath] = (source, self._parser.parse(source))
        return self._cache[filepath]

    def invalidate(self, filepath: str | Path | None = None):
        if filepath is None:
            self._cache.clear()
        else:
            self._cache.pop(Path(filepath).resolve(), None)

    def _resolve(self, relpath: str) -> Path:
        p = self.root / relpath
        if not p.exists():
            raise FileNotFoundError(f"{relpath} not found under {self.root}")
        return p

    def _is_todo(self, relpath: str) -> bool:
        return Path(relpath).suffix in TODO_EXTENSIONS

    def _top_nodes(self, relpath: str) -> tuple[bytes, list[_AttrNode]]:
        source, tree = self._parse(self._resolve(relpath))
        return source, _collect_items(
            list(tree.root_node.named_children), TOP_LEVEL_TYPES,
        )

    def _find(
        self, nodes: list[_AttrNode], source: bytes, target: str,
    ) -> _AttrNode:
        """Find a node by name.  Tries display names (with attrs) first,
        then falls back to base names (without attrs)."""
        display_names = [_display_name(an, source) for an in nodes]
        base_names = [_extract_name(an.node, source) for an in nodes]
        display_ids = _dedup_names(display_names)
        base_ids = _dedup_names(base_names)

        # exact match on display name (includes attrs)
        for anode, nid in zip(nodes, display_ids):
            if nid == target:
                return anode

        # fallback: match on base name (without attrs)
        for anode, nid in zip(nodes, base_ids):
            if nid == target:
                return anode

        raise KeyError(
            f"{target!r} not found. Available:\n" + "\n".join(display_ids)
        )

    # ── macro body re-parsing ──────────────────────────────────

    def _parse_macro_body(
        self, anode: _AttrNode, source: bytes,
    ) -> tuple[bytes, list[_AttrNode], int]:
        """Re-parse the token-tree contents of a ``macro_invocation``
        as top-level Rust.

        Returns ``(inner_source, items, byte_offset)`` where
        *inner_source* is the bytes inside the braces, *items* are
        the successfully-parsed top-level nodes, and *byte_offset* is
        the position of inner_source[0] in *source* (so that
        ``byte_offset + inner_position == original_position``).
        """
        token_tree = None
        for child in anode.node.children:
            if child.type == "token_tree":
                token_tree = child
                break
        if not token_tree:
            return b"", [], 0

        inner_start = token_tree.start_byte + 1   # skip opening brace
        inner_end = token_tree.end_byte - 1        # skip closing brace
        if inner_start >= inner_end:
            return b"", [], inner_start

        inner_source = source[inner_start:inner_end]
        tree = self._parser.parse(inner_source)
        items = _collect_items(
            list(tree.root_node.named_children), TOP_LEVEL_TYPES,
        )
        return inner_source, items, inner_start

    # ── shared expand-or-source logic ──────────────────────────

    def _render_node(
        self, anode: _AttrNode, source: bytes, offset: int = 0,
        path: list[str] | None = None,
    ) -> str:
        """If *anode* is a container or macro invocation, list its
        children.  Otherwise return full source text."""
        # Rust containers (impl, trait, struct, enum, mod)
        if anode.type in CONTAINER_TYPES:
            children = _body_children(anode)
            if children:
                raw = [_display_name(c, source) for c in children]
                suffixes = [_child_count_suffix(c, source, self) for c in children]
                return _paginate(_dedup_names(raw, suffixes), offset, path=path)

        # Macro invocations — re-parse body as Rust
        if anode.type == "macro_invocation":
            inner_source, items, _ = self._parse_macro_body(anode, source)
            if items:
                raw = [_display_name(an, inner_source) for an in items]
                suffixes = [_child_count_suffix(an, inner_source, self) for an in items]
                return _paginate(_dedup_names(raw, suffixes), offset, path=path)

        return _attr_text(anode, source)

    def _find_node(
        self, path: tuple[str, ...],
    ) -> tuple[Path, bytes, _AttrNode | _OffsetAttrNode]:
        """Resolve path segments to (filepath, source_bytes, node).

        Requires at least 2 segments (file + item). Supports arbitrary
        depth for nested containers and macro invocations.

        For children inside a macro invocation the returned node is an
        ``_OffsetAttrNode`` whose byte positions refer to the original
        file so that edits splice correctly.
        """
        if len(path) < 2:
            raise ValueError("Need at least 2 path segments to address a node")
        filepath = self._resolve(path[0])
        source, top = self._top_nodes(path[0])
        current: _AttrNode | _OffsetAttrNode = self._find(top, source, path[1])
        current_source = source
        byte_offset = 0

        for seg in path[2:]:
            if current.type == "macro_invocation":
                # Unwrap proxy if needed to get the real _AttrNode
                real = current._inner if isinstance(current, _OffsetAttrNode) else current
                inner_source, items, off = self._parse_macro_body(real, current_source)
                child = self._find(items, inner_source, seg)
                byte_offset += off
                current = _OffsetAttrNode(_inner=child, _byte_offset=byte_offset)
                current_source = inner_source
            elif current.type in CONTAINER_TYPES:
                real = current._inner if isinstance(current, _OffsetAttrNode) else current
                children = _body_children(real)
                current = self._find(children, current_source, seg)
                if byte_offset:
                    current = _OffsetAttrNode(_inner=current, _byte_offset=byte_offset)
            else:
                raise ValueError(f"Cannot drill into '{path[len(path) - len(path[2:]) + path[2:].index(seg)]}' — not a container")

        return filepath, source, current

    def _write(self, filepath: Path, new_source: bytes) -> None:
        filepath.write_bytes(new_source)
        self.invalidate(filepath)

    # ── search (recursive find) ────────────────────────────────

    def search(
        self,
        name: str,
        *,
        exact: bool = False,
        kind: str | None = None,
        offset: int = 0,
    ) -> str:
        """Search all .rs files recursively for items matching *name*.

        Parameters:
            name:   The string to search for.
            exact:  If True, match the full item name. Otherwise substring.
            kind:   Optional tree-sitter node type filter, e.g.
                    ``"function_item"``, ``"struct_item"``, ``"impl_item"``.
            offset: Pagination offset into the results list.

        Returns a formatted string of matches.  Each line shows the
        query-ready path, the node type, and the line number.
        """
        results = self._search_results(name, exact=exact, kind=kind,
                                       limit=offset + LIMIT)
        if not results:
            return f"No items matching '{name}' found."
        lines = [str(r) for r in results]
        hint: dict = {"name": name}
        if exact:
            hint["exact"] = True
        if kind:
            hint["kind"] = kind
        return _paginate(lines, offset, search_hint=hint)

    def search_results(
        self,
        name: str,
        *,
        exact: bool = False,
        kind: str | None = None,
    ) -> list[SearchResult]:
        """Programmatic variant — returns a list of ``SearchResult``."""
        return self._search_results(name, exact=exact, kind=kind)

    def _search_results(
        self,
        name: str,
        *,
        exact: bool = False,
        kind: str | None = None,
        limit: int = 0,
    ) -> list[SearchResult]:
        results: list[SearchResult] = []
        # Pre-compute lowercase query once for substring matching
        name_lower = name.lower()

        def matches(item_name: str) -> bool:
            if exact:
                return item_name == name
            return name_lower in item_name.lower()

        def kind_ok(node_type: str) -> bool:
            return kind is None or node_type == kind

        def at_limit() -> bool:
            return limit > 0 and len(results) >= limit

        for rs_file in sorted(self.root.rglob("*.rs")):
            if at_limit():
                # Append sentinel so pagination knows there are more
                results.append(SearchResult(file="", path=("",), node_type="", line=0))
                break
            # skip target/ and hidden directories
            try:
                rel = rs_file.relative_to(self.root)
            except ValueError:
                continue
            if any(p.startswith(".") or p == "target" for p in rel.parts):
                continue
            relpath = str(rel)

            # Pre-filter: skip files whose source doesn't contain the
            # query string at all (case-insensitive).  Avoids tree-sitter
            # parsing for non-matching files.
            try:
                filepath = self._resolve(relpath)
                source_bytes = filepath.read_bytes()
            except Exception:
                continue
            if name_lower not in source_bytes.decode("utf-8", errors="replace").lower():
                continue

            try:
                source, nodes = self._top_nodes(relpath)
            except Exception:
                continue

            for anode in nodes:
                if at_limit():
                    break
                item_name = _extract_name(anode.node, source)
                item_line = anode.node.start_point[0] + 1

                # check the top-level item itself
                if matches(item_name) and kind_ok(anode.type):
                    results.append(SearchResult(
                        file=relpath,
                        path=(item_name,),
                        node_type=anode.type,
                        line=item_line,
                    ))

                # children of Rust containers (impl, trait, …)
                if anode.type in CONTAINER_TYPES:
                    for child in _body_children(anode):
                        if at_limit():
                            break
                        child_name = _extract_name(child.node, source)
                        child_line = child.node.start_point[0] + 1
                        if matches(child_name) and kind_ok(child.type):
                            results.append(SearchResult(
                                file=relpath,
                                path=(item_name, child_name),
                                node_type=child.type,
                                line=child_line,
                            ))

                # children inside macro invocations
                if anode.type == "macro_invocation":
                    try:
                        inner_src, items, _ = self._parse_macro_body(
                            anode, source,
                        )
                    except Exception:
                        continue
                    macro_name = item_name
                    for inner in items:
                        if at_limit():
                            break
                        inner_name = _extract_name(inner.node, inner_src)
                        inner_line = (
                            anode.node.start_point[0]
                            + inner.node.start_point[0]
                            + 1
                        )
                        if matches(inner_name) and kind_ok(inner.type):
                            results.append(SearchResult(
                                file=relpath,
                                path=(macro_name, inner_name),
                                node_type=inner.type,
                                line=inner_line,
                            ))
                        # containers inside macros
                        if inner.type in CONTAINER_TYPES:
                            for child in _body_children(inner):
                                if at_limit():
                                    break
                                child_name = _extract_name(
                                    child.node, inner_src,
                                )
                                child_line = (
                                    anode.node.start_point[0]
                                    + child.node.start_point[0]
                                    + 1
                                )
                                if matches(child_name) and kind_ok(child.type):
                                    results.append(SearchResult(
                                        file=relpath,
                                        path=(macro_name, inner_name, child_name),
                                        node_type=child.type,
                                        line=child_line,
                                    ))

        return results

    # ── query (read) ───────────────────────────────────────────

    def query(self, *path: str, offset: int = 0) -> str:
        path = self._resolve_segments(path)
        depth = len(path)
        if depth == 0:
            return self._list_dir("", offset)
        if depth == 1:
            p = self.root / path[0]
            if p.is_dir():
                return self._list_dir(path[0], offset)
        if depth >= 1 and self._is_todo(path[0]):
            return self._query_todo(*path, offset=offset)
        if depth == 1:
            return self._list_items(path[0], offset)

        # Depth >= 2: walk the tree of items
        source, nodes = self._top_nodes(path[0])
        current = self._find(nodes, source, path[1])
        current_path = [path[0], path[1]]

        for seg in path[2:]:
            # Drill into macro invocations
            if current.type == "macro_invocation":
                source, items, _ = self._parse_macro_body(current, source)
                current = self._find(items, source, seg)
            # Drill into containers
            elif current.type in CONTAINER_TYPES:
                children = _body_children(current)
                current = self._find(children, source, seg)
            else:
                raise ValueError(f"'{current_path[-1]}' has no children to drill into")
            current_path.append(seg)

        return self._render_node(current, source, offset, path=current_path)

    def _list_dir(self, relpath: str, offset: int) -> str:
        dirpath = (self.root / relpath) if relpath else self.root
        entries = []
        for p in sorted(dirpath.iterdir()):
            if p.name.startswith(".") or p.name == "target":
                continue
            entries.append(p.name + "/" if p.is_dir() else p.name)
        path = [relpath] if relpath else []
        return _paginate(entries, offset, path=path)

    def _list_items(self, relpath: str, offset: int) -> str:
        source, nodes = self._top_nodes(relpath)
        raw = [_display_name(an, source) for an in nodes]
        suffixes = [_child_count_suffix(an, source, self) for an in nodes]
        return _paginate(_dedup_names(raw, suffixes), offset, path=[relpath])

    # ── query todo ─────────────────────────────────────────────

    def _query_todo(self, *path: str, offset: int = 0) -> str:
        filepath = self._resolve(path[0])
        text = filepath.read_text(encoding="utf-8")
        lines, sections = _parse_todo(text)

        if len(path) == 1:
            # list section headers with progress
            entries = []
            for s in sections:
                done = sum(1 for it in s.items if it.checked)
                total = len(s.items)
                entries.append(f"{s.title} ({done}/{total} done)")
            if not entries:
                return "(no sections)"
            header = "Sections:\n" if offset == 0 else ""
            return header + _paginate(entries, offset, path=[path[0]])

        if len(path) == 2:
            section = _find_section(sections, path[1])
            if section.items:
                entries = []
                for it in section.items:
                    mark = "x" if it.checked else " "
                    entries.append(f"[{mark}] {it.text}")
                return _paginate(entries, offset, path=[path[0], path[1]])
            # no items → show raw section body
            body = lines[section.line_start + 1 : section.line_end + 1]
            raw = "\n".join(body).strip()
            return raw if raw else "(empty section)"

        if len(path) == 3:
            section = _find_section(sections, path[1])
            item = _find_todo_item(section.items, path[2])
            mark = "x" if item.checked else " "
            result = f"[{mark}] {item.text}"
            if item.notes:
                result += "\n  Notes:"
                for note in item.notes:
                    result += f"\n    {note}"
            return result

        raise ValueError("todo query accepts 1–3 path segments")

    # ── edit (write) ───────────────────────────────────────────

    def edit(self, *path: str, action: str, content: str = "") -> str:
        if action not in EDIT_ACTIONS:
            raise ValueError(
                f"Unknown action {action!r}. "
                f"Choose from: {', '.join(sorted(EDIT_ACTIONS))}"
            )

        # ── create_file: join ALL segments (file may not exist yet) ─

        if action == "create_file":
            relpath = "/".join(path)
            p = self.root / relpath
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"created {relpath}"

        # ── delete_file: join ALL segments ─────────────────

        if action == "delete_file":
            relpath = "/".join(path)
            p = self._resolve(relpath)
            p.unlink()
            self.invalidate(p)
            return f"deleted {relpath}"

        # ── append: join ALL segments (pure file-level op) ─

        if action == "append":
            relpath = "/".join(path)
            p = self._resolve(relpath)
            source = p.read_bytes()
            nl = b"" if (not source or source.endswith(b"\n")) else b"\n"
            self._write(p, source + nl + content.encode("utf-8") + b"\n")
            return f"appended to {relpath}"

        # ── resolve multi-segment filesystem paths ─────────

        path = self._resolve_segments(path)

        # ── route todo / md files ──────────────────────────

        if len(path) >= 1 and self._is_todo(path[0]):
            return self._edit_todo(*path, action=action, content=content)

        # guard todo-only actions on non-todo files
        if action in TODO_ONLY_ACTIONS:
            raise ValueError(
                f"Action {action!r} is only valid for .todo/.md files"
            )

        # ── Rust container action ──────────────────────────

        if action == "append_child":
            if len(path) != 2:
                raise ValueError(
                    "append_child: path must be (file, container_item)"
                )
            filepath, source, anode = self._find_node(path)
            body = _find_body(anode.node)
            if not body:
                raise ValueError(f"'{path[1]}' has no body to append into")
            pos = body.end_byte - 1
            snippet = b"\n" + content.encode("utf-8") + b"\n"
            self._write(filepath, source[:pos] + snippet + source[pos:])
            return f"appended child in {path[1]}"

        # ── Rust node-level actions ────────────────────────

        if len(path) < 2:
            raise ValueError(f"{action}: need at least (file, item)")

        filepath, source, anode = self._find_node(path)
        start, end = anode.start_byte, anode.end_byte

        if action == "replace":
            self._write(
                filepath,
                source[:start] + content.encode("utf-8") + source[end:],
            )
            return f"replaced {path[-1]}"

        if action == "delete":
            if end < len(source) and source[end : end + 1] == b"\n":
                end += 1
            self._write(filepath, source[:start] + source[end:])
            return f"deleted {path[-1]}"

        if action == "insert_before":
            snippet = content.encode("utf-8") + b"\n"
            self._write(filepath, source[:start] + snippet + source[start:])
            return f"inserted before {path[-1]}"

        if action == "insert_after":
            pos = end
            if pos < len(source) and source[pos : pos + 1] == b"\n":
                pos += 1
            snippet = content.encode("utf-8") + b"\n"
            self._write(filepath, source[:pos] + snippet + source[pos:])
            return f"inserted after {path[-1]}"

        raise ValueError(f"Unhandled action: {action!r}")

    # ── edit todo ──────────────────────────────────────────────

    def _edit_todo(self, *path: str, action: str, content: str = "") -> str:
        relpath = path[0]
        filepath = self.root / relpath

        # auto-create for add_header
        if action == "add_header" and not filepath.exists():
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text("", encoding="utf-8")
        elif not filepath.exists():
            raise FileNotFoundError(f"{relpath} not found under {self.root}")

        text = filepath.read_text(encoding="utf-8")
        lines, sections = _parse_todo(text)

        def _save() -> None:
            out = "\n".join(lines)
            if not out.endswith("\n"):
                out += "\n"
            filepath.write_text(out, encoding="utf-8")

        # ── add_header (path: file) ────────────────────────

        if action == "add_header":
            if len(path) != 1:
                raise ValueError("add_header: path must be (file,)")
            # trim trailing blank lines, then add separator
            while lines and not lines[-1].strip():
                lines.pop()
            if lines:
                lines.append("")
            lines.append(f"# {content}")
            _save()
            return f"added header: {content}"

        # ── add_item (path: file, section) ─────────────────

        if action == "add_item":
            if len(path) != 2:
                raise ValueError("add_item: path must be (file, section)")
            section = _find_section(sections, path[1])
            clean = _strip_checkbox_content(content)
            if section.items:
                insert_at = section.items[-1].line_end + 1
            else:
                insert_at = section.line_start + 1
            lines.insert(insert_at, f"- [ ] {clean}")
            _save()
            return f"added item: {clean}"

        # ── check (path: file, section, item) ──────────────

        if action == "check":
            if len(path) != 3:
                raise ValueError("check: path must be (file, section, item)")
            section = _find_section(sections, path[1])
            item = _find_todo_item(section.items, path[2])
            lines[item.line_start] = lines[item.line_start].replace(
                "- [ ] ", "- [x] ", 1
            )
            _save()
            return f"checked: {item.text}"

        # ── uncheck (path: file, section, item) ────────────

        if action == "uncheck":
            if len(path) != 3:
                raise ValueError("uncheck: path must be (file, section, item)")
            section = _find_section(sections, path[1])
            item = _find_todo_item(section.items, path[2])
            line = lines[item.line_start]
            lines[item.line_start] = (
                line.replace("- [x] ", "- [ ] ", 1)
                    .replace("- [X] ", "- [ ] ", 1)
            )
            _save()
            return f"unchecked: {item.text}"

        # ── add_note (path: file, section, item) ───────────

        if action == "add_note":
            if len(path) != 3:
                raise ValueError(
                    "add_note: path must be (file, section, item)"
                )
            section = _find_section(sections, path[1])
            item = _find_todo_item(section.items, path[2])
            lines.insert(item.line_end + 1, f"  {content}")
            _save()
            return f"added note to: {item.text}"

        # ── delete (path: file, section  OR  file, section, item)

        if action == "delete":
            if len(path) == 2:
                section = _find_section(sections, path[1])
                del lines[section.line_start : section.line_end + 1]
                _save()
                return f"deleted section: {section.title}"
            if len(path) == 3:
                section = _find_section(sections, path[1])
                item = _find_todo_item(section.items, path[2])
                del lines[item.line_start : item.line_end + 1]
                _save()
                return f"deleted item: {item.text}"
            raise ValueError(
                "delete: path must be (file, section) "
                "or (file, section, item)"
            )

        # ── replace item text (path: file, section, item) ──

        if action == "replace":
            if len(path) != 3:
                raise ValueError(
                    "replace on todo: path must be (file, section, item)"
                )
            section = _find_section(sections, path[1])
            item = _find_todo_item(section.items, path[2])
            clean = _strip_checkbox_content(content)
            mark = "x" if item.checked else " "
            lines[item.line_start] = f"- [{mark}] {clean}"
            _save()
            return f"replaced: {item.text} -> {clean}"

        raise ValueError(
            f"Action {action!r} not supported for todo files. "
            f"Supported: add_header, add_item, check, uncheck, "
            f"add_note, delete, replace"
        )


# ── CLI ────────────────────────────────────────────────────────

def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Rust codebase explorer & editor (with .todo support)",
    )
    sub = ap.add_subparsers(dest="cmd")
    sub.default = "query"

    q = sub.add_parser("query", help="Read / browse")
    q.add_argument("root")
    q.add_argument("path", nargs="*")
    q.add_argument("--offset", type=int, default=0)

    s = sub.add_parser("search", help="Search for items by name")
    s.add_argument("root")
    s.add_argument("name", help="Name or substring to search for")
    s.add_argument("--exact", action="store_true", help="Exact match only")
    s.add_argument(
        "--kind",
        default=None,
        help="Filter by node type (e.g. function_item, struct_item)",
    )
    s.add_argument("--offset", type=int, default=0)

    e = sub.add_parser("edit", help="Edit")
    e.add_argument("root")
    e.add_argument("action", choices=sorted(EDIT_ACTIONS))
    e.add_argument("path", nargs="*")
    e.add_argument("--content", default=None)
    e.add_argument("--stdin", action="store_true", help="Read content from stdin")

    args = ap.parse_args()
    tree = CodebaseTree(args.root)

    try:
        if args.cmd == "search":
            print(tree.search(
                args.name,
                exact=args.exact,
                kind=args.kind,
                offset=args.offset,
            ))
        elif args.cmd == "edit":
            content = args.content or ""
            if args.stdin:
                content = sys.stdin.read()
            print(tree.edit(*args.path, action=args.action, content=content))
        else:
            print(
                tree.query(*args.path, offset=getattr(args, "offset", 0))
            )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
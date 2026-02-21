from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
import httpx
from openai import OpenAI

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
EVENT_LOG_PATH = DATA_DIR / "events.jsonl"
TOOL_WORKDIR = APP_DIR.parent
DEFAULT_WORKSPACE_PATH = str(TOOL_WORKDIR)
BROWSE_TOOL_NAME = "browse"
EDIT_TOOL_NAME = "edit"

BUILTIN_TOOL_SPECS: Dict[str, Dict[str, Any]] = {
    "pwd": {
        "description": "Return the current workspace directory path.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    "ls": {
        "description": "List files in a workspace path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path. Default is '.'"},
                "show_all": {"type": "boolean", "description": "Include hidden files."},
                "long": {"type": "boolean", "description": "Use long listing format."},
            },
        },
    },
    "rg": {
        "description": "Search text using ripgrep in the workspace.",
        "parameters": {
            "type": "object",
            "required": ["pattern"],
            "properties": {
                "pattern": {"type": "string", "description": "Regex/text pattern to search for."},
                "path": {"type": "string", "description": "Search path. Default is '.'"},
                "glob": {"type": "string", "description": "Optional include/exclude glob filter."},
                "max_matches": {"type": "integer", "description": "Maximum matches per file."},
                "ignore_case": {"type": "boolean", "description": "Case-insensitive search."},
            },
        },
    },
    "rg_files": {
        "description": "List files tracked by ripgrep (similar to rg --files).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Root path. Default is '.'"},
                "glob": {"type": "string", "description": "Optional include/exclude glob filter."},
                "max_files": {"type": "integer", "description": "Maximum files to return."},
            },
        },
    },
    
    BROWSE_TOOL_NAME: {
        "description": (
            "Browse the Rust codebase and todo/md files. Pass path segments to drill down to any depth. "
            "[] lists project root; [dir] lists directory; [file.rs] lists top-level items; "
            "[file.rs,item] lists children or source; add more segments to drill deeper into "
            "nested containers (impl blocks, macros like verus!, traits, etc). "
            "For .todo/.md files: [file] lists headers, [file,section] lists items, "
            "[file,section,item] shows detail. Use offset to paginate."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "segments to drill down. Examples: [], ['src'], "
                        "['src/main.rs'], ['src/main.rs','impl Server'], "
                        "['src/main.rs','impl Server','new'], "
                        "['src/main.rs','verus!','impl Foo','my_method'], "
                        "['plan.todo'], ['plan.todo','Backend']"
                    ),
                },
                "offset": {"type": "integer", "description": "Pagination offset for long listings. Default 0."},
            },
        },
    },
    "search": {
        "description": "Recursively search all .rs files for items (functions, structs, impls, traits, etc.) matching a name. Returns query-ready paths, node types, and line numbers.",
        "parameters": {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string", "description": "Name or substring to search for, matched case-insensitively."},
                "exact": {"type": "boolean", "description": "If true, match the full item name exactly instead of substring."},
                "kind": {"type": "string", "description": "Optional node type filter (e.g. function_item, struct_item, impl_item, trait_item, enum_item, const_item, mod_item)."},
                "offset": {"type": "integer", "description": "Pagination offset. Each page returns up to 15 results."}
            }
        }
    },
    EDIT_TOOL_NAME: {
        "description": (
            "Edit Rust source files and todo/md files. Rust actions: create_file, delete_file, append, "
            "replace/delete/insert_before/insert_after on nodes, append_child for containers. "
            "Todo actions: add_header, add_item, check, uncheck, add_note, delete, replace."
        ),
        "parameters": {
            "type": "object",
            "required": ["action", "path"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "create_file",
                        "delete_file",
                        "append",
                        "replace",
                        "delete",
                        "insert_before",
                        "insert_after",
                        "append_child",
                        "add_header",
                        "add_item",
                        "check",
                        "uncheck",
                        "add_note",
                    ],
                    "description": "The edit action to perform.",
                },
                "path": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "segments addressing the target. Examples: ['src/new.rs'], "
                        "['src/lib.rs','old_fn'], ['src/lib.rs','impl Server','handle'], "
                        "['plan.todo','Backend','Add auth']"
                    ),
                },
                "content": {
                    "type": "string",
                    "description": (
                        "The content for the action. Required for create_file, append, replace, "
                        "insert_before, insert_after, append_child, add_header, add_item, add_note. "
                        "Not used for delete, delete_file, check, uncheck."
                    ),
                },
            },
        },
    },
}

DEFAULT_BUILTIN_TOOLS = {name: False for name in BUILTIN_TOOL_SPECS}

DEFAULT_STATE: Dict[str, Any] = {
    "config": {
        "base_url": "http://localhost:8000/v1",
        "api_key": "",
        "model": "Qwen3-Coder-Next",
        "workspace_path": DEFAULT_WORKSPACE_PATH,
        "builtin_tools": copy.deepcopy(DEFAULT_BUILTIN_TOOLS),
    },
    "tools": {},
    "history": [],
}

app = Flask(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_state_identity(state: Dict[str, Any], prev_state_hash: str, next_event_index: int) -> Tuple[str, str]:
    packed = json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    state_hash = hashlib.sha256(packed.encode("utf-8")).hexdigest()
    snapshot_seed = f"{prev_state_hash}:{state_hash}:{next_event_index}"
    snapshot_hash = hashlib.sha256(snapshot_seed.encode("utf-8")).hexdigest()
    state_uuid = str(uuid.UUID(snapshot_hash[:32]))
    return state_hash, state_uuid


def read_all_events() -> List[Dict[str, Any]]:
    if not EVENT_LOG_PATH.exists():
        return []
    events: List[Dict[str, Any]] = []
    with EVENT_LOG_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def append_action_record(action: str, **kwargs: Any) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    record = {"action": action, "timestamp": utc_now_iso(), **kwargs}
    with EVENT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
        handle.write("\n")


def append_state_snapshot(
    state: Dict[str, Any],
    action: str,
    detail: Dict[str, Any],
    parent_state_uuid: str | None = None,
) -> Dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    events = read_all_events()
    prev_state_hash = ""
    for ev in reversed(events):
        if "state_hash" in ev:
            prev_state_hash = ev["state_hash"]
            break
    next_event_index = len(events) + 1
    state_hash, state_uuid = compute_state_identity(state, prev_state_hash, next_event_index)
    event = {
        "event_id": str(uuid.uuid4()),
        "timestamp": utc_now_iso(),
        "action": action,
        "detail": detail,
        "prev_state_hash": prev_state_hash,
        "state_hash": state_hash,
        "state_uuid": state_uuid,
        "parent_state_uuid": parent_state_uuid,
        "state": state,
    }
    with EVENT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True, sort_keys=True))
        handle.write("\n")
    return event


def ensure_store() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if EVENT_LOG_PATH.exists() and EVENT_LOG_PATH.stat().st_size > 0:
        return
    append_state_snapshot(copy.deepcopy(DEFAULT_STATE), "init", {"note": "initial state"})


def resolve_workspace_dir(raw_path: str | None) -> Path:
    candidate = (raw_path or "").strip()
    if not candidate:
        path = Path(DEFAULT_WORKSPACE_PATH).resolve()
    else:
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = (Path(DEFAULT_WORKSPACE_PATH) / path).resolve()
        else:
            path = path.resolve()

    if not path.exists():
        raise ValueError(f"workspace_path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"workspace_path is not a directory: {path}")
    return path


def normalize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    config = state.get("config")
    if not isinstance(config, dict):
        config = {}
        state["config"] = config

    for key in ["base_url", "api_key", "model", "workspace_path"]:
        value = config.get(key)
        if not isinstance(value, str):
            config[key] = str(DEFAULT_STATE["config"][key])

    incoming_builtin = config.get("builtin_tools")
    normalized_builtin = copy.deepcopy(DEFAULT_BUILTIN_TOOLS)
    if isinstance(incoming_builtin, dict):
        if "query" in incoming_builtin and "browse" not in incoming_builtin:
            incoming_builtin = {**incoming_builtin, "browse": incoming_builtin.get("query")}
        if "rust_ast_query" in incoming_builtin and "browse" not in incoming_builtin:
            incoming_builtin = {**incoming_builtin, "browse": incoming_builtin.get("rust_ast_query")}
        for name in DEFAULT_BUILTIN_TOOLS:
            if name in incoming_builtin:
                normalized_builtin[name] = bool(incoming_builtin[name])
    config["builtin_tools"] = normalized_builtin

    tools = state.get("tools")
    if not isinstance(tools, dict):
        state["tools"] = {}

    history = state.get("history")
    if not isinstance(history, list):
        state["history"] = []

    return state


def build_tree_index(events: List[Dict[str, Any]]) -> Tuple[
    Dict[str, Dict[str, Any]],
    Dict[str, List[str]],
    Dict[str, str],
]:
    by_uuid: Dict[str, Dict[str, Any]] = {}
    children_map: Dict[str, List[Tuple[str, str]]] = {}  # parent -> [(timestamp, child_uuid)]
    last_child_map: Dict[str, str] = {}

    for ev in events:
        action = ev.get("action")
        if action == "last_child_visited":
            parent = ev.get("parent_state_uuid")
            child = ev.get("child_state_uuid")
            if parent and child:
                last_child_map[parent] = child
            continue

        suuid = ev.get("state_uuid")
        if not suuid:
            continue
        by_uuid[suuid] = ev
        parent = ev.get("parent_state_uuid")
        if parent:
            children_map.setdefault(parent, []).append((ev.get("timestamp", ""), suuid))

    # Sort children by timestamp for consistent ordering
    sorted_children: Dict[str, List[str]] = {}
    for parent, kids in children_map.items():
        kids.sort()
        sorted_children[parent] = [k[1] for k in kids]

    return by_uuid, sorted_children, last_child_map


def load_state(state_uuid: str | None = None) -> Tuple[Dict[str, Any], Dict[str, Any], int, Dict[str, Any]]:
    ensure_store()
    events = read_all_events()
    by_uuid, children_map, last_child_map = build_tree_index(events)

    if not state_uuid:
        # Find the latest actual state event (not an action record)
        target = None
        for ev in reversed(events):
            if ev.get("state_uuid"):
                target = ev
                break
        if target is None:
            raise RuntimeError("No state events found")
    else:
        target = by_uuid.get(state_uuid)
        if target is None:
            raise KeyError(f"state_uuid not found: {state_uuid}")

    current_uuid = target["state_uuid"]
    parent_uuid = target.get("parent_state_uuid")

    # Children of current node
    children = children_map.get(current_uuid, [])
    preferred_child = last_child_map.get(current_uuid)
    if preferred_child and preferred_child not in children:
        preferred_child = None
    if not preferred_child and children:
        preferred_child = children[0]

    # Siblings (other children of same parent)
    prev_sibling = None
    next_sibling = None
    if parent_uuid:
        siblings = children_map.get(parent_uuid, [])
        if current_uuid in siblings:
            idx = siblings.index(current_uuid)
            if idx > 0:
                prev_sibling = siblings[idx - 1]
            if idx < len(siblings) - 1:
                next_sibling = siblings[idx + 1]

    nav = {
        "parent": parent_uuid,
        "child": preferred_child,
        "prev_sibling": prev_sibling,
        "next_sibling": next_sibling,
    }

    return normalize_state(copy.deepcopy(target["state"])), target, len(events), nav


def to_jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def compile_tool(tool_name: str, implementation_code: str):
    scope: Dict[str, Any] = {"__builtins__": __builtins__}
    exec(implementation_code, scope, scope)
    func = scope.get(tool_name)
    if not callable(func):
        raise ValueError(f"Tool code must define a callable named '{tool_name}'")
    return func


def coerce_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def clip_text(text: str, max_chars: int = 12000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... output truncated to {max_chars} chars ..."


def rough_token_count_text(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def estimate_message_tokens(message: Dict[str, Any]) -> int:
    total = 4
    role = message.get("role")
    if isinstance(role, str):
        total += rough_token_count_text(role)

    for key in ["name", "tool_call_id"]:
        value = message.get(key)
        if isinstance(value, str):
            total += rough_token_count_text(value)

    total += rough_token_count_text(coerce_content(message.get("content")))

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                total += rough_token_count_text(coerce_content(tool_call))
                continue
            total += 6
            tool_call_id = tool_call.get("id")
            if isinstance(tool_call_id, str):
                total += rough_token_count_text(tool_call_id)
            fn = tool_call.get("function")
            if isinstance(fn, dict):
                name = fn.get("name")
                arguments = fn.get("arguments")
                if isinstance(name, str):
                    total += rough_token_count_text(name)
                if isinstance(arguments, str):
                    total += rough_token_count_text(arguments)
            else:
                total += rough_token_count_text(coerce_content(fn))

    return total


def estimate_context_tokens(history: Any) -> int:
    if not isinstance(history, list):
        return 0
    total = 2
    for message in history:
        if isinstance(message, dict):
            total += estimate_message_tokens(message)
        else:
            total += rough_token_count_text(coerce_content(message))
    return total


def workspace_dir_from_runtime(runtime_ctx: Dict[str, Any]) -> Path:
    ensure_runtime_context(runtime_ctx)
    raw_workspace = runtime_ctx.get("workspace_dir", DEFAULT_WORKSPACE_PATH)
    if isinstance(raw_workspace, Path):
        return raw_workspace
    if isinstance(raw_workspace, str):
        workspace = resolve_workspace_dir(raw_workspace)
        runtime_ctx["workspace_dir"] = workspace
        return workspace
    workspace = resolve_workspace_dir(DEFAULT_WORKSPACE_PATH)
    runtime_ctx["workspace_dir"] = workspace
    return workspace


def resolve_workspace_path(raw_path: str | None, runtime_ctx: Dict[str, Any]) -> Path:
    workspace_dir = workspace_dir_from_runtime(runtime_ctx)
    candidate = (raw_path or ".").strip() or "."
    path = Path(candidate)
    if not path.is_absolute():
        path = (workspace_dir / path).resolve()
    else:
        path = path.resolve()
    if workspace_dir not in path.parents and path != workspace_dir:
        raise ValueError(f"path must stay inside workspace: {workspace_dir}")
    return path


def run_shell_tool(command: List[str], runtime_ctx: Dict[str, Any], allow_nonzero: bool = True) -> Dict[str, Any]:
    workspace_dir = workspace_dir_from_runtime(runtime_ctx)
    try:
        completed = subprocess.run(
            command,
            cwd=workspace_dir,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Command not found: {command[0]}") from exc

    if completed.returncode != 0 and not allow_nonzero:
        raise RuntimeError(completed.stderr.strip() or f"Command failed with code {completed.returncode}")

    return {
        "command": command,
        "cwd": str(workspace_dir),
        "returncode": completed.returncode,
        "stdout": clip_text(completed.stdout),
        "stderr": clip_text(completed.stderr),
    }


def ensure_runtime_context(runtime_ctx: Dict[str, Any]) -> None:
    if "workspace_dir" not in runtime_ctx:
        runtime_ctx["workspace_dir"] = DEFAULT_WORKSPACE_PATH
    try:
        runtime_ctx["response_index"] = int(runtime_ctx.get("response_index", 0))
    except (TypeError, ValueError):
        runtime_ctx["response_index"] = 0


_BROWSE_ANNOTATION_RE = re.compile(r"\s*\(\d+(?:/\d+)?\s+(?:children|done)\)\s*$")


def parse_path_segments(raw_path: Any, allow_empty: bool = True) -> List[str]:
    if raw_path is None:
        return []
    if not isinstance(raw_path, list):
        raise ValueError("path must be an array of strings")

    segments: List[str] = []
    for index, value in enumerate(raw_path):
        if not isinstance(value, str):
            raise ValueError(f"path[{index}] must be a string")
        clean = _BROWSE_ANNOTATION_RE.sub("", value).strip()
        if not clean:
            raise ValueError(f"path[{index}] must not be empty")
        segments.append(clean)

    if not allow_empty and not segments:
        raise ValueError("path must contain at least one segment")
    return segments


def run_browse_tool(parsed_args: Dict[str, Any], runtime_ctx: Dict[str, Any]) -> str:
    ensure_runtime_context(runtime_ctx)

    path_segments = parse_path_segments(parsed_args.get("path"))
    offset_arg = parsed_args.get("offset", 0)

    try:
        offset = int(offset_arg)
    except (TypeError, ValueError):
        raise ValueError("offset must be an integer") from None
    if offset < 0:
        raise ValueError("offset must be >= 0")

    workspace_dir = workspace_dir_from_runtime(runtime_ctx)
    if path_segments:
        first = resolve_workspace_path(path_segments[0], runtime_ctx)
        supported_file_exts = {".rs", ".todo", ".md"}

    args = [
        sys.executable,
        str(APP_DIR / "rust_ast.py"),
        "query",
        str(workspace_dir),
        *path_segments,
        "--offset",
        str(offset),
    ]

    completed = subprocess.run(
        args,
        cwd=workspace_dir,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        error_text = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(error_text or f"rust_ast.py failed with code {completed.returncode}")

    raw_output = completed.stdout.rstrip("\n")
    output_text = clip_text(raw_output, max_chars=20000)

    return output_text


def run_search_tool(parsed_args: Dict[str, Any], runtime_ctx: Dict[str, Any]) -> str:
    ensure_runtime_context(runtime_ctx)

    name = parsed_args.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("search requires a non-empty 'name' string")
    name = name.strip()

    exact = bool(parsed_args.get("exact", False))

    kind_raw = parsed_args.get("kind")
    kind: str | None
    if kind_raw is None:
        kind = None
    elif isinstance(kind_raw, str) and kind_raw.strip():
        kind = kind_raw.strip()
    else:
        raise ValueError("kind must be a non-empty string when provided")

    offset_arg = parsed_args.get("offset", 0)
    try:
        offset = int(offset_arg)
    except (TypeError, ValueError):
        raise ValueError("offset must be an integer") from None
    if offset < 0:
        raise ValueError("offset must be >= 0")

    workspace_dir = workspace_dir_from_runtime(runtime_ctx)
    args = [
        sys.executable,
        str(APP_DIR / "rust_ast.py"),
        "search",
        str(workspace_dir),
        name,
        "--offset",
        str(offset),
    ]
    if exact:
        args.append("--exact")
    if kind:
        args.extend(["--kind", kind])

    completed = subprocess.run(
        args,
        cwd=workspace_dir,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        error_text = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(error_text or f"rust_ast.py search failed with code {completed.returncode}")

    return clip_text(completed.stdout.rstrip("\n"), max_chars=20000)


def run_edit_tool(parsed_args: Dict[str, Any], runtime_ctx: Dict[str, Any]) -> str:
    ensure_runtime_context(runtime_ctx)

    action = parsed_args.get("action")
    if not isinstance(action, str) or not action.strip():
        raise ValueError("action must be a non-empty string")
    action = action.strip()
    valid_actions = {
        "create_file",
        "delete_file",
        "append",
        "replace",
        "delete",
        "insert_before",
        "insert_after",
        "append_child",
        "add_header",
        "add_item",
        "check",
        "uncheck",
        "add_note",
    }
    if action not in valid_actions:
        raise ValueError(f"Unknown action '{action}'")

    path_segments = parse_path_segments(parsed_args.get("path"), allow_empty=False)
    content = parsed_args.get("content")
    file_ext = Path(path_segments[0]).suffix.lower()
    rust_ext = ".rs"
    todo_exts = {".todo", ".md"}

    actions_requiring_content = {
        "create_file",
        "append",
        "replace",
        "insert_before",
        "insert_after",
        "append_child",
        "add_header",
        "add_item",
        "add_note",
    }
    if action in actions_requiring_content:
        if not isinstance(content, str):
            raise ValueError(f"content is required for action '{action}'")
        content_text = content
    else:
        if content is not None and not isinstance(content, str):
            raise ValueError("content must be a string when provided")
        content_text = content if isinstance(content, str) else ""

    if action in {"create_file", "delete_file", "append", "add_header"} and len(path_segments) != 1:
        raise ValueError(f"{action} requires path to contain exactly 1 segment")
    if action in {"append_child", "add_item"} and len(path_segments) != 2:
        raise ValueError(f"{action} requires path to contain exactly 2 segments")
    if action in {"check", "uncheck", "add_note"} and len(path_segments) != 3:
        raise ValueError(f"{action} requires path to contain exactly 3 segments")
    if action in {"replace", "delete", "insert_before", "insert_after"} and len(path_segments) not in {2, 3}:
        raise ValueError(f"{action} requires path to contain 2 or 3 segments")

    rust_only_actions = {"append", "insert_before", "insert_after", "append_child"}
    todo_only_actions = {"add_header", "add_item", "check", "uncheck", "add_note"}
    if action in rust_only_actions and file_ext != rust_ext:
        raise ValueError(f"action '{action}' requires a .rs file")
    if action in todo_only_actions and file_ext not in todo_exts:
        raise ValueError(f"action '{action}' requires a .todo or .md file")

    if action in {"replace", "delete"} and file_ext in todo_exts:
        if action == "replace" and len(path_segments) != 3:
            raise ValueError("todo replace requires path to contain exactly 3 segments")
        if action == "delete" and len(path_segments) not in {2, 3}:
            raise ValueError("todo delete requires path to contain 2 or 3 segments")

    if action in {"create_file", "delete_file"} and file_ext not in {rust_ext, *todo_exts}:
        raise ValueError("create_file/delete_file support .rs, .todo, and .md files")

    file_path = resolve_workspace_path(path_segments[0], runtime_ctx)

    if action == "create_file":
        if file_path.exists():
            raise ValueError(f"File already exists: {path_segments[0]}")
    else:
        if not file_path.exists():
            raise ValueError(f"File not found: {path_segments[0]}")
        if not file_path.is_file():
            raise ValueError(f"path[0] is not a file: {path_segments[0]}")

    workspace_dir = workspace_dir_from_runtime(runtime_ctx)
    args = [
        sys.executable,
        str(APP_DIR / "rust_ast.py"),
        "edit",
        str(workspace_dir),
        action,
        *path_segments,
    ]

    stdin_text: str | None = None
    if action in actions_requiring_content:
        args.append("--stdin")
        stdin_text = content_text

    completed = subprocess.run(
        args,
        cwd=workspace_dir,
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        error_text = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(error_text or f"rust_ast.py edit failed with code {completed.returncode}")

    return clip_text(completed.stdout.rstrip("\n"), max_chars=20000)


def shell_result_output(result: Dict[str, Any]) -> str:
    stdout = str(result.get("stdout", ""))
    stderr = str(result.get("stderr", ""))
    if stdout.strip():
        return stdout
    if stderr.strip():
        return stderr
    return ""


def execute_builtin_tool(tool_name: str, parsed_args: Dict[str, Any], runtime_ctx: Dict[str, Any]) -> Dict[str, Any]:
    ensure_runtime_context(runtime_ctx)
    workspace_dir = workspace_dir_from_runtime(runtime_ctx)

    if tool_name == "pwd":
        return {"output": str(workspace_dir)}

    if tool_name == "ls":
        path = str(parsed_args.get("path", "."))
        resolve_workspace_path(path, runtime_ctx)
        show_all = bool(parsed_args.get("show_all", False))
        long_format = bool(parsed_args.get("long", False))

        command = ["ls"]
        if show_all:
            command.append("-a")
        if long_format:
            command.append("-l")
        command.append(path)
        result = run_shell_tool(command, runtime_ctx, allow_nonzero=False)
        return {"output": shell_result_output(result)}

    if tool_name == "rg":
        pattern = parsed_args.get("pattern")
        if not isinstance(pattern, str) or not pattern.strip():
            raise ValueError("rg requires a non-empty 'pattern' string")

        path = str(parsed_args.get("path", "."))
        resolve_workspace_path(path, runtime_ctx)
        glob = parsed_args.get("glob")
        ignore_case = bool(parsed_args.get("ignore_case", False))

        max_matches_raw = parsed_args.get("max_matches", 100)
        try:
            max_matches = max(1, min(int(max_matches_raw), 2000))
        except (TypeError, ValueError):
            max_matches = 100

        command = ["rg", "--line-number", "--color", "never", "--max-count", str(max_matches)]
        if ignore_case:
            command.append("-i")
        if isinstance(glob, str) and glob.strip():
            command.extend(["-g", glob.strip()])
        command.extend([pattern, path])
        result = run_shell_tool(command, runtime_ctx, allow_nonzero=True)
        return {"output": shell_result_output(result)}

    if tool_name == "rg_files":
        path = str(parsed_args.get("path", "."))
        resolve_workspace_path(path, runtime_ctx)
        glob = parsed_args.get("glob")

        max_files_raw = parsed_args.get("max_files", 300)
        try:
            max_files = max(1, min(int(max_files_raw), 5000))
        except (TypeError, ValueError):
            max_files = 300

        command = ["rg", "--files"]
        if isinstance(glob, str) and glob.strip():
            command.extend(["-g", glob.strip()])
        command.append(path)

        result = run_shell_tool(command, runtime_ctx, allow_nonzero=True)
        files = [line for line in result["stdout"].splitlines() if line.strip()]
        truncated = len(files) > max_files
        files = files[:max_files]
        output = "\n".join(files)
        if truncated:
            output = output + f"\n... truncated to {max_files} files ..."
        return {"output": output}

    if tool_name == BROWSE_TOOL_NAME:
        return {"output": run_browse_tool(parsed_args, runtime_ctx)}

    if tool_name == "search":
        return {"output": run_search_tool(parsed_args, runtime_ctx)}

    if tool_name == EDIT_TOOL_NAME:
        return {"output": run_edit_tool(parsed_args, runtime_ctx)}

    raise ValueError(f"Unknown builtin tool '{tool_name}'")


def build_tools_payload(tools: Dict[str, Dict[str, Any]], builtin_flags: Dict[str, bool]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for name, tool in tools.items():
        payload.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
        )
    for name, spec in BUILTIN_TOOL_SPECS.items():
        if not builtin_flags.get(name, False):
            continue
        if name in tools:
            continue
        payload.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": spec["description"],
                    "parameters": spec["parameters"],
                },
            }
        )
    return payload


def execute_tool_call(
    tool_name: str,
    raw_args: str,
    tools: Dict[str, Dict[str, Any]],
    builtin_flags: Dict[str, bool],
    runtime_ctx: Dict[str, Any],
) -> Dict[str, Any]:
    ensure_runtime_context(runtime_ctx)

    try:
        parsed_args = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError as exc:
        return {"output": f"ERROR: Tool args are not valid JSON: {exc}"}

    if not isinstance(parsed_args, dict):
        return {"output": "ERROR: Tool args must be a JSON object"}

    tool = tools.get(tool_name)
    if tool is None:
        if tool_name in BUILTIN_TOOL_SPECS:
            enabled = builtin_flags.get(tool_name, False)
            if not enabled:
                return {"output": f"ERROR: Builtin tool '{tool_name}' is disabled"}
            try:
                return execute_builtin_tool(tool_name, parsed_args, runtime_ctx)
            except Exception as exc:  # noqa: BLE001
                return {"output": f"ERROR: {exc}"}
        return {"output": f"ERROR: Unknown tool '{tool_name}'"}

    try:
        tool_fn = compile_tool(tool_name, tool["implementation_code"])
        result = tool_fn(**parsed_args)
        if isinstance(result, dict) and set(result.keys()) == {"output"}:
            return {"output": to_jsonable(result.get("output"))}
        return {"output": to_jsonable(result)}
    except Exception as exc:  # noqa: BLE001
        return {"output": f"ERROR: {exc}"}


def stream_line(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


def update_partial_tool_calls(partials: List[Dict[str, Any]], delta_tool_calls: Any) -> None:
    for tool_call_delta in delta_tool_calls or []:
        index = getattr(tool_call_delta, "index", 0)
        if not isinstance(index, int):
            index = 0
        while len(partials) <= index:
            partials.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})

        current = partials[index]
        delta_id = getattr(tool_call_delta, "id", None)
        if isinstance(delta_id, str) and delta_id:
            current["id"] = delta_id

        fn = getattr(tool_call_delta, "function", None)
        if fn is None:
            continue

        delta_name = getattr(fn, "name", None)
        if isinstance(delta_name, str) and delta_name:
            current["function"]["name"] += delta_name

        delta_arguments = getattr(fn, "arguments", None)
        if isinstance(delta_arguments, str) and delta_arguments:
            current["function"]["arguments"] += delta_arguments


def finalize_partial_tool_calls(partials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    finalized: List[Dict[str, Any]] = []
    for partial in partials:
        function_data = partial.get("function", {})
        name = str(function_data.get("name", "")).strip()
        if not name:
            continue
        arguments = function_data.get("arguments", "")
        if not isinstance(arguments, str) or not arguments:
            arguments = "{}"
        tool_call_id = partial.get("id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            tool_call_id = str(uuid.uuid4())
        finalized.append(
            {
                "id": tool_call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return finalized


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/state")
def get_state():
    requested_state_uuid = str(request.args.get("state_uuid", "")).strip()
    try:
        state, latest, event_count, nav = load_state(requested_state_uuid or None)
    except KeyError:
        return jsonify({"ok": False, "error": f"state_uuid not found: {requested_state_uuid}"}), 404
    return jsonify(
        {
            "ok": True,
            "state_uuid": latest["state_uuid"],
            "state_hash": latest["state_hash"],
            "event_count": event_count,
            "parent_state_uuid": nav["parent"],
            "child_state_uuid": nav["child"],
            "prev_sibling_uuid": nav["prev_sibling"],
            "next_sibling_uuid": nav["next_sibling"],
            "context_tokens_estimate": estimate_context_tokens(state.get("history", [])),
            "state": state,
        }
    )


@app.post("/api/state/visit_child")
def visit_child():
    payload = request.get_json(silent=True) or {}
    parent = str(payload.get("parent", "")).strip()
    child = str(payload.get("child", "")).strip()
    if not parent or not child:
        return jsonify({"ok": False, "error": "parent and child are required"}), 400
    append_action_record("last_child_visited", parent_state_uuid=parent, child_state_uuid=child)
    return jsonify({"ok": True})


@app.post("/api/config")
def save_config():
    payload = request.get_json(silent=True) or {}
    state, parent_event, _, _ = load_state()
    config = state["config"]

    updated_fields: List[str] = []
    for key in ["base_url", "api_key", "model"]:
        if key in payload and isinstance(payload[key], str):
            config[key] = payload[key].strip()
            updated_fields.append(key)

    if "workspace_path" in payload:
        if not isinstance(payload["workspace_path"], str):
            return jsonify({"ok": False, "error": "workspace_path must be a string"}), 400
        try:
            workspace_dir = resolve_workspace_dir(payload["workspace_path"])
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        config["workspace_path"] = str(workspace_dir)
        updated_fields.append("workspace_path")

    if "builtin_tools" in payload:
        incoming_builtin = payload["builtin_tools"]
        if not isinstance(incoming_builtin, dict):
            return jsonify({"ok": False, "error": "builtin_tools must be an object"}), 400
        normalized_builtin = copy.deepcopy(DEFAULT_BUILTIN_TOOLS)
        current_builtin = config.get("builtin_tools", {})
        for name in normalized_builtin:
            if name in incoming_builtin:
                normalized_builtin[name] = bool(incoming_builtin[name])
            else:
                normalized_builtin[name] = bool(current_builtin.get(name, False))
        config["builtin_tools"] = normalized_builtin
        updated_fields.append("builtin_tools")

    event = append_state_snapshot(state, "set_config", {"updated_fields": updated_fields}, parent_state_uuid=parent_event["state_uuid"])
    return jsonify(
        {
            "ok": True,
            "state_uuid": event["state_uuid"],
            "state_hash": event["state_hash"],
        }
    )


@app.post("/api/tools")
def add_or_update_tool():
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name", "")).strip()
    description = str(payload.get("description", "")).strip()
    parameters_raw = payload.get("parameters", {})
    implementation_code = str(payload.get("implementation_code", ""))

    if not name:
        return jsonify({"ok": False, "error": "Tool name is required"}), 400
    if not description:
        return jsonify({"ok": False, "error": "Tool description is required"}), 400
    if not implementation_code.strip():
        return jsonify({"ok": False, "error": "Tool implementation_code is required"}), 400

    if isinstance(parameters_raw, str):
        try:
            parameters = json.loads(parameters_raw)
        except json.JSONDecodeError as exc:
            return jsonify({"ok": False, "error": f"parameters is not valid JSON: {exc}"}), 400
    elif isinstance(parameters_raw, dict):
        parameters = parameters_raw
    else:
        return jsonify({"ok": False, "error": "parameters must be JSON object or JSON string"}), 400

    if not isinstance(parameters, dict):
        return jsonify({"ok": False, "error": "parameters must resolve to a JSON object"}), 400

    try:
        compile_tool(name, implementation_code)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"Tool compile error: {exc}"}), 400

    state, parent_event, _, _ = load_state()
    existed = name in state["tools"]
    state["tools"][name] = {
        "name": name,
        "description": description,
        "parameters": parameters,
        "implementation_code": implementation_code,
        "updated_at": utc_now_iso(),
    }

    action = "update_tool" if existed else "add_tool"
    event = append_state_snapshot(state, action, {"tool_name": name}, parent_state_uuid=parent_event["state_uuid"])
    return jsonify(
        {
            "ok": True,
            "state_uuid": event["state_uuid"],
            "state_hash": event["state_hash"],
        }
    )


@app.delete("/api/tools/<path:tool_name>")
def delete_tool(tool_name: str):
    name = tool_name.strip()
    if not name:
        return jsonify({"ok": False, "error": "Tool name is required"}), 400

    state, parent_event, _, _ = load_state()
    if name not in state["tools"]:
        return jsonify({"ok": False, "error": f"Custom tool not found: {name}"}), 404

    del state["tools"][name]
    event = append_state_snapshot(state, "remove_tool", {"tool_name": name}, parent_state_uuid=parent_event["state_uuid"])
    return jsonify(
        {
            "ok": True,
            "state_uuid": event["state_uuid"],
            "state_hash": event["state_hash"],
        }
    )


@app.post("/api/history/clear")
def clear_history():
    payload = request.get_json(silent=True) or {}
    requested_uuid = str(payload.get("state_uuid", "")).strip() or None
    state, parent_event, _, _ = load_state(requested_uuid)
    state["history"] = []
    event = append_state_snapshot(state, "clear_history", {}, parent_state_uuid=parent_event["state_uuid"])
    return jsonify(
        {
            "ok": True,
            "state_uuid": event["state_uuid"],
            "state_hash": event["state_hash"],
        }
    )


@app.post("/api/history/append")
def append_history():
    payload = request.get_json(silent=True) or {}
    messages_raw = payload.get("messages")
    if not isinstance(messages_raw, list) or not messages_raw:
        return jsonify({"ok": False, "error": "messages must be a non-empty array"}), 400

    accepted_roles = {"user", "assistant", "system", "tool"}
    normalized: List[Dict[str, Any]] = []

    for index, msg in enumerate(messages_raw):
        if not isinstance(msg, dict):
            return jsonify({"ok": False, "error": f"messages[{index}] must be an object"}), 400

        role = msg.get("role")
        if not isinstance(role, str) or role not in accepted_roles:
            return jsonify({"ok": False, "error": f"messages[{index}].role is invalid"}), 400

        content = msg.get("content", "")
        if not isinstance(content, str):
            content = coerce_content(content)

        entry: Dict[str, Any] = {"role": role, "content": content}
        if role == "tool":
            tool_call_id = msg.get("tool_call_id")
            name = msg.get("name")
            if isinstance(tool_call_id, str) and tool_call_id:
                entry["tool_call_id"] = tool_call_id
            if isinstance(name, str) and name:
                entry["name"] = name
        normalized.append(entry)

    merge_last = bool(payload.get("merge_last", False))
    state, parent_event, _, _ = load_state()
    if merge_last and normalized and state["history"] and state["history"][-1].get("role") == normalized[0].get("role"):
        state["history"][-1]["content"] = (state["history"][-1].get("content") or "") + (normalized[0].get("content") or "")
        state["history"].extend(normalized[1:])
    else:
        state["history"].extend(normalized)
    event = append_state_snapshot(state, "append_history", {"count": len(normalized), "merge_last": merge_last}, parent_state_uuid=parent_event["state_uuid"])
    return jsonify(
        {
            "ok": True,
            "state_uuid": event["state_uuid"],
            "state_hash": event["state_hash"],
            "context_tokens_estimate": estimate_context_tokens(state["history"]),
        }
    )


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    prompt = str(payload.get("message", "")).strip()
    continue_only = bool(payload.get("continue", False))
    if not prompt and not continue_only:
        return jsonify({"ok": False, "error": "message is required unless continue=true"}), 400

    requested_uuid = str(payload.get("state_uuid", "")).strip() or None
    state, parent_event, _, _ = load_state(requested_uuid)
    config = state["config"]
    tools = state["tools"]
    builtin_flags = config.get("builtin_tools", copy.deepcopy(DEFAULT_BUILTIN_TOOLS))
    tools_payload = build_tools_payload(tools, builtin_flags)
    try:
        workspace_dir = resolve_workspace_dir(str(config.get("workspace_path", DEFAULT_WORKSPACE_PATH)))
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    messages = copy.deepcopy(state["history"])
    merge_continue = False
    if prompt:
        messages.append({"role": "user", "content": prompt})
    elif continue_only and not messages:
        return jsonify({"ok": False, "error": "cannot continue without prior chat history"}), 400
    elif continue_only and messages and messages[-1].get("role") == "assistant":
        merge_continue = True
    tool_trace: List[Dict[str, Any]] = []
    final_reply = ""
    runtime_ctx: Dict[str, Any] = {}
    ensure_runtime_context(runtime_ctx)
    runtime_ctx["workspace_dir"] = workspace_dir

    try:
        api_key = config.get("api_key", "")
        client = OpenAI(base_url=config["base_url"], api_key=api_key or "EMPTY", timeout=httpx.Timeout(None, connect=10.0))
        while True:
            request_args: Dict[str, Any] = {
                "messages": messages,
                "model": config["model"],
            }
            if tools_payload:
                request_args["tools"] = tools_payload

            completion = client.chat.completions.create(**request_args)
            message = completion.choices[0].message
            content = coerce_content(message.content)
            tool_calls = message.tool_calls or []
            runtime_ctx["response_index"] = int(runtime_ctx["response_index"]) + 1

            assistant_message: Dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_message["tool_calls"] = []
                for tool_call in tool_calls:
                    assistant_message["tool_calls"].append(
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments or "{}",
                            },
                        }
                    )
            if merge_continue:
                prev = messages[-1]
                prev["content"] = (prev.get("content") or "") + content
                if tool_calls:
                    prev["tool_calls"] = prev.get("tool_calls", []) + assistant_message["tool_calls"]
                merge_continue = False
            else:
                messages.append(assistant_message)

            if not tool_calls:
                final_reply = content
                break

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                raw_args = tool_call.function.arguments or "{}"
                tool_result = execute_tool_call(tool_name, raw_args, tools, builtin_flags, runtime_ctx)

                tool_trace.append(
                    {
                        "tool_name": tool_name,
                        "arguments": raw_args,
                        "result": tool_result,
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )

        state["history"] = messages
        event = append_state_snapshot(
            state,
            "chat",
            {
                "prompt": prompt,
                "continue": continue_only,
                "tool_calls_executed": len(tool_trace),
            },
            parent_state_uuid=parent_event["state_uuid"],
        )

        return jsonify(
            {
                "ok": True,
                "reply": final_reply,
                "history": messages,
                "tool_trace": tool_trace,
                "context_tokens_estimate": estimate_context_tokens(messages),
                "state_uuid": event["state_uuid"],
                "state_hash": event["state_hash"],
            }
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify(
            {
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=5),
            }
        ), 500


@app.post("/api/chat/stream")
def chat_stream():
    payload = request.get_json(silent=True) or {}
    prompt = str(payload.get("message", "")).strip()
    continue_only = bool(payload.get("continue", False))
    if not prompt and not continue_only:
        return jsonify({"ok": False, "error": "message is required unless continue=true"}), 400

    requested_uuid = str(payload.get("state_uuid", "")).strip() or None
    state, parent_event, _, _ = load_state(requested_uuid)
    config = state["config"]
    tools = state["tools"]
    builtin_flags = config.get("builtin_tools", copy.deepcopy(DEFAULT_BUILTIN_TOOLS))
    tools_payload = build_tools_payload(tools, builtin_flags)
    try:
        workspace_dir = resolve_workspace_dir(str(config.get("workspace_path", DEFAULT_WORKSPACE_PATH)))
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    messages = copy.deepcopy(state["history"])
    merge_continue = False
    if prompt:
        messages.append({"role": "user", "content": prompt})
    elif continue_only and not messages:
        return jsonify({"ok": False, "error": "cannot continue without prior chat history"}), 400
    elif continue_only and messages and messages[-1].get("role") == "assistant":
        merge_continue = True
    tool_trace: List[Dict[str, Any]] = []
    final_reply = ""
    runtime_ctx: Dict[str, Any] = {}
    ensure_runtime_context(runtime_ctx)
    runtime_ctx["workspace_dir"] = workspace_dir

    @stream_with_context
    def generate():
        nonlocal final_reply, merge_continue
        try:
            api_key = config.get("api_key", "")
            client = OpenAI(base_url=config["base_url"], api_key=api_key or "EMPTY", timeout=httpx.Timeout(None, connect=10.0))
            current_parent_uuid = parent_event["state_uuid"]

            def save_progress(action_detail: Dict[str, Any]) -> Dict[str, Any]:
                nonlocal current_parent_uuid
                state["history"] = messages
                ev = append_state_snapshot(
                    state, "chat_progress", action_detail,
                    parent_state_uuid=current_parent_uuid,
                )
                current_parent_uuid = ev["state_uuid"]
                return ev

            # Save initial snapshot with user message added
            if prompt and not continue_only:
                ev = save_progress({"prompt": prompt, "stage": "user_message"})
                yield stream_line({"type": "snapshot", "state_uuid": ev["state_uuid"]})

            while True:
                request_args: Dict[str, Any] = {
                    "messages": messages,
                    "model": config["model"],
                    "stream": True,
                }
                if tools_payload:
                    request_args["tools"] = tools_payload

                completion_stream = client.chat.completions.create(**request_args)
                content_parts: List[str] = []
                partial_tool_calls: List[Dict[str, Any]] = []

                for chunk in completion_stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta is None:
                        continue

                    delta_content = getattr(delta, "content", None)
                    if delta_content:
                        if isinstance(delta_content, str):
                            text_piece = delta_content
                        else:
                            text_piece = coerce_content(delta_content)
                        content_parts.append(text_piece)
                        yield stream_line({"type": "delta", "content": text_piece})

                    update_partial_tool_calls(partial_tool_calls, getattr(delta, "tool_calls", None))

                content = "".join(content_parts)
                tool_calls = finalize_partial_tool_calls(partial_tool_calls)
                runtime_ctx["response_index"] = int(runtime_ctx["response_index"]) + 1

                assistant_message: Dict[str, Any] = {"role": "assistant", "content": content}
                if tool_calls:
                    assistant_message["tool_calls"] = tool_calls
                if merge_continue:
                    prev = messages[-1]
                    prev["content"] = (prev.get("content") or "") + content
                    if tool_calls:
                        prev["tool_calls"] = prev.get("tool_calls", []) + assistant_message["tool_calls"]
                    merge_continue = False
                else:
                    messages.append(assistant_message)

                if not tool_calls:
                    final_reply = content
                    break

                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    raw_args = tool_call["function"]["arguments"] or "{}"
                    yield stream_line(
                        {
                            "type": "tool_call",
                            "tool_name": tool_name,
                            "arguments": raw_args,
                        }
                    )

                    tool_result = execute_tool_call(tool_name, raw_args, tools, builtin_flags, runtime_ctx)
                    tool_trace.append(
                        {
                            "tool_name": tool_name,
                            "arguments": raw_args,
                            "result": tool_result,
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": tool_name,
                            "content": json.dumps(tool_result, ensure_ascii=False),
                        }
                    )
                    yield stream_line(
                        {
                            "type": "tool_result",
                            "tool_name": tool_name,
                            "result": tool_result,
                        }
                    )

                # Save progress after each tool-use round
                ev = save_progress({
                    "prompt": prompt,
                    "stage": "tool_round",
                    "tool_calls_executed": len(tool_trace),
                })
                yield stream_line({"type": "snapshot", "state_uuid": ev["state_uuid"]})

            state["history"] = messages
            event = append_state_snapshot(
                state,
                "chat",
                {
                    "prompt": prompt,
                    "continue": continue_only,
                    "tool_calls_executed": len(tool_trace),
                },
                parent_state_uuid=current_parent_uuid,
            )
            yield stream_line(
                {
                    "type": "done",
                    "reply": final_reply,
                    "history": messages,
                    "tool_trace": tool_trace,
                    "context_tokens_estimate": estimate_context_tokens(messages),
                    "state_uuid": event["state_uuid"],
                    "state_hash": event["state_hash"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            yield stream_line(
                {
                    "type": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=5),
                }
            )

    return Response(generate(), mimetype="application/x-ndjson")


if __name__ == "__main__":
    ensure_store()
    app.run(host="127.0.0.1", port=5050, debug=True)

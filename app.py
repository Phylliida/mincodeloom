from __future__ import annotations

import atexit
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
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
        "description": "List files in a workspace path, one per line.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path. Default is '.'"},
                "show_all": {"type": "boolean", "description": "Include hidden files."},
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
    "read_file": {
        "description": "Read a file's contents with line numbers. Optionally specify a line range. Prefer lookup_source to save context.",
        "parameters": {
            "type": "object",
            "required": ["path"],
            "properties": {
                "path": {"type": "string", "description": "File path relative to workspace."},
                "start_line": {"type": "integer", "description": "First line to read (1-based). Default is 1."},
                "end_line": {"type": "integer", "description": "Last line to read (inclusive). Default is end of file."},
            },
        },
    },
    "file_edit": {
        "description": (
            "Edit a file by exact string replacement. Provide old_string (must match exactly once) "
            "and new_string. To create a new file, set create=true and put content in new_string."
        ),
        "parameters": {
            "type": "object",
            "required": ["path"],
            "properties": {
                "path": {"type": "string", "description": "File path relative to workspace."},
                "old_string": {"type": "string", "description": "Exact text to find (must appear exactly once)."},
                "new_string": {"type": "string", "description": "Replacement text, or full content when create=true."},
                "create": {"type": "boolean", "description": "If true, create/overwrite the file with new_string."},
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
        "mcp_servers": {},
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
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
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
        if "edit" in incoming_builtin and "file_edit" not in incoming_builtin:
            incoming_builtin = {**incoming_builtin, "file_edit": incoming_builtin.get("edit")}
        for name in DEFAULT_BUILTIN_TOOLS:
            if name in incoming_builtin:
                normalized_builtin[name] = bool(incoming_builtin[name])
    config["builtin_tools"] = normalized_builtin

    mcp_servers = config.get("mcp_servers")
    if not isinstance(mcp_servers, dict):
        config["mcp_servers"] = {}
    else:
        cleaned: Dict[str, Any] = {}
        for srv_name, srv_cfg in mcp_servers.items():
            if isinstance(srv_cfg, dict) and isinstance(srv_cfg.get("command"), str):
                tools_flags = srv_cfg.get("tools", {})
                cleaned[srv_name] = {
                    "command": srv_cfg["command"],
                    "args": srv_cfg.get("args", []) if isinstance(srv_cfg.get("args"), list) else [],
                    "env": srv_cfg.get("env", {}) if isinstance(srv_cfg.get("env"), dict) else {},
                    "enabled": srv_cfg.get("enabled", True),
                    "tools": tools_flags if isinstance(tools_flags, dict) else {},
                }
        config["mcp_servers"] = cleaned

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


# ---------------------------------------------------------------------------
# MCP (Model Context Protocol) client
# ---------------------------------------------------------------------------

_mcp_clients: Dict[str, "McpClient"] = {}
_mcp_lock = threading.Lock()


class McpClient:
    """Minimal MCP client over stdio (newline-delimited JSON-RPC 2.0)."""

    def __init__(self, name: str, command: str, args: List[str] | None = None, env: Dict[str, str] | None = None, cwd: str | None = None):
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.cwd = cwd
        self._process: subprocess.Popen | None = None
        self._next_id = 1
        self._tools_cache: List[Dict[str, Any]] | None = None

    def _ensure_running(self):
        if self._process is not None and self._process.poll() is None:
            return
        merged_env = {**os.environ, **self.env} if self.env else None
        self._process = subprocess.Popen(
            [self.command] + self.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=merged_env,
            cwd=self.cwd,
        )
        self._handshake()

    def _send(self, method: str, params: Any = None, *, notification: bool = False) -> int | None:
        msg: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        msg_id = None
        if not notification:
            msg_id = self._next_id
            self._next_id += 1
            msg["id"] = msg_id
        if params is not None:
            msg["params"] = params
        raw = json.dumps(msg, ensure_ascii=False) + "\n"
        self._process.stdin.write(raw.encode("utf-8"))
        self._process.stdin.flush()
        return msg_id

    def _recv(self, msg_id: int) -> Any:
        while True:
            line = self._process.stdout.readline()
            if not line:
                stderr_tail = ""
                if self._process.stderr:
                    try:
                        stderr_tail = self._process.stderr.read(2000).decode("utf-8", errors="replace")
                    except Exception:
                        pass
                raise RuntimeError(f"MCP server '{self.name}' closed (stderr: {stderr_tail})")
            try:
                resp = json.loads(line)
            except json.JSONDecodeError:
                continue
            if resp.get("id") == msg_id:
                if "error" in resp:
                    err = resp["error"]
                    raise RuntimeError(f"MCP error: {err.get('message', str(err))}")
                return resp.get("result")

    def _handshake(self):
        msg_id = self._send("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "mincodeloom", "version": "0.1.0"},
        })
        self._recv(msg_id)
        self._send("notifications/initialized", notification=True)
        self._tools_cache = None

    def list_tools(self) -> List[Dict[str, Any]]:
        if self._tools_cache is not None:
            return self._tools_cache
        self._ensure_running()
        msg_id = self._send("tools/list")
        result = self._recv(msg_id)
        self._tools_cache = result.get("tools", [])
        return self._tools_cache

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        self._ensure_running()
        msg_id = self._send("tools/call", {"name": tool_name, "arguments": arguments})
        result = self._recv(msg_id)
        parts = result.get("content", []) if isinstance(result, dict) else []
        texts = []
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text", ""))
            elif isinstance(part, dict):
                texts.append(json.dumps(part, ensure_ascii=False))
        return "\n".join(texts) if texts else json.dumps(result, ensure_ascii=False)

    def stop(self):
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None
        self._tools_cache = None


def get_mcp_client(name: str, server_config: Dict[str, Any], workspace_path: str | None = None) -> McpClient:
    cwd = server_config.get("cwd") or workspace_path
    with _mcp_lock:
        client = _mcp_clients.get(name)
        if client is not None:
            if (client.command == server_config["command"]
                    and client.args == server_config.get("args", [])
                    and client.cwd == cwd):
                return client
            client.stop()
        client = McpClient(
            name=name,
            command=server_config["command"],
            args=server_config.get("args", []),
            env=server_config.get("env", {}),
            cwd=cwd,
        )
        _mcp_clients[name] = client
        return client


def stop_all_mcp_clients():
    with _mcp_lock:
        for client in _mcp_clients.values():
            client.stop()
        _mcp_clients.clear()


atexit.register(stop_all_mcp_clients)


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

        command = ["ls", "-1"]
        if show_all:
            command.append("-a")
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

    if tool_name == "read_file":
        path_str = parsed_args.get("path")
        if not isinstance(path_str, str) or not path_str.strip():
            raise ValueError("path is required")
        resolved = resolve_workspace_path(path_str.strip(), runtime_ctx)
        if not resolved.is_file():
            raise ValueError(f"Not a file: {path_str}")
        text = resolved.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines(keepends=False)
        total = len(lines)
        try:
            start = max(1, int(parsed_args.get("start_line", 1)))
        except (TypeError, ValueError):
            start = 1
        try:
            end = min(total, int(parsed_args.get("end_line", total)))
        except (TypeError, ValueError):
            end = total
        explicit_range = parsed_args.get("start_line") is not None or parsed_args.get("end_line") is not None
        selected = lines[start - 1 : end]
        numbered = [f"{i:>6}\t{line}" for i, line in enumerate(selected, start=start)]
        header = f"{path_str} ({total} lines)"
        if start > 1 or end < total:
            header += f" [showing {start}-{end}]"
        body = header + "\n" + "\n".join(numbered)
        max_chars = 20000
        if not explicit_range and len(body) > max_chars:
            # Truncate at line boundary
            truncated = body[:max_chars]
            last_nl = truncated.rfind("\n")
            if last_nl > 0:
                truncated = truncated[:last_nl]
            # Figure out the last displayed line number
            shown_lines = truncated.count("\n")  # header is first line
            last_shown = start + shown_lines - 1
            truncated += (
                f"\n\n... truncated ({total - last_shown} more lines). "
                f"Use read_file with start_line={last_shown + 1} to see the rest."
            )
            body = truncated
        return {"output": body}

    if tool_name == "file_edit":
        path_str = parsed_args.get("path")
        if not isinstance(path_str, str) or not path_str.strip():
            raise ValueError("path is required")
        resolved = resolve_workspace_path(path_str.strip(), runtime_ctx)
        create = bool(parsed_args.get("create", False))
        old_string = parsed_args.get("old_string")
        new_string = parsed_args.get("new_string")

        if create:
            if not isinstance(new_string, str):
                raise ValueError("new_string is required when create=true")
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(new_string, encoding="utf-8")
            return {"output": f"Created {path_str} ({len(new_string)} chars)"}

        if not isinstance(old_string, str):
            raise ValueError("old_string is required")
        if not isinstance(new_string, str):
            raise ValueError("new_string is required")
        if not resolved.is_file():
            raise ValueError(f"File not found: {path_str}")
        text = resolved.read_text(encoding="utf-8")
        count = text.count(old_string)
        if count == 0:
            raise ValueError("old_string not found in file")
        if count > 1:
            raise ValueError(f"old_string matches {count} locations; provide more surrounding context to be unique")
        start_line = text[:text.index(old_string)].count("\n") + 1
        new_text = text.replace(old_string, new_string, 1)
        resolved.write_text(new_text, encoding="utf-8")
        return {
            "output": f"Edited {path_str} (line {start_line}, {len(old_string)} -> {len(new_string)} chars)",
            "start_line": start_line,
        }

    raise ValueError(f"Unknown builtin tool '{tool_name}'")


def build_tools_payload(
    tools: Dict[str, Dict[str, Any]],
    builtin_flags: Dict[str, bool],
    mcp_servers: Dict[str, Dict[str, Any]] | None = None,
    workspace_path: str | None = None,
) -> List[Dict[str, Any]]:
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

    used_names = {entry["function"]["name"] for entry in payload}
    for srv_name, srv_cfg in (mcp_servers or {}).items():
        if not srv_cfg.get("enabled", True):
            continue
        tool_flags = srv_cfg.get("tools", {})
        try:
            client = get_mcp_client(srv_name, srv_cfg, workspace_path=workspace_path)
            for mcp_tool in client.list_tools():
                tname = mcp_tool.get("name", "")
                if not tname or tname in used_names:
                    continue
                if tool_flags and not tool_flags.get(tname, True):
                    continue
                used_names.add(tname)
                payload.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tname,
                            "description": mcp_tool.get("description", ""),
                            "parameters": mcp_tool.get("inputSchema", {"type": "object", "properties": {}}),
                        },
                    }
                )
        except Exception:
            pass

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

        mcp_servers = runtime_ctx.get("mcp_servers", {})
        for srv_name, srv_cfg in mcp_servers.items():
            if not srv_cfg.get("enabled", True):
                continue
            tool_flags = srv_cfg.get("tools", {})
            if tool_flags and not tool_flags.get(tool_name, True):
                continue
            try:
                ws = str(runtime_ctx.get("workspace_dir", DEFAULT_WORKSPACE_PATH))
                client = get_mcp_client(srv_name, srv_cfg, workspace_path=ws)
                srv_tool_names = [t.get("name") for t in client.list_tools()]
                if tool_name in srv_tool_names:
                    result_text = client.call_tool(tool_name, parsed_args)
                    return {"output": clip_text(result_text)}
            except Exception as exc:  # noqa: BLE001
                return {"output": f"ERROR: MCP '{srv_name}' tool '{tool_name}': {exc}"}

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
        stop_all_mcp_clients()
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

    if "mcp_servers" in payload:
        incoming_mcp = payload["mcp_servers"]
        if not isinstance(incoming_mcp, dict):
            return jsonify({"ok": False, "error": "mcp_servers must be an object"}), 400
        current_mcp = config.get("mcp_servers", {})
        current_mcp.update(incoming_mcp)
        # Re-normalize through the state normalizer
        config["mcp_servers"] = current_mcp
        normalize_state(state)
        updated_fields.append("mcp_servers")

    event = append_state_snapshot(state, "set_config", {"updated_fields": updated_fields}, parent_state_uuid=parent_event["state_uuid"])
    return jsonify(
        {
            "ok": True,
            "state_uuid": event["state_uuid"],
            "state_hash": event["state_hash"],
        }
    )


@app.get("/api/mcp_tools")
def list_mcp_tools():
    state, _, _, _ = load_state()
    workspace_path = state["config"].get("workspace_path", DEFAULT_WORKSPACE_PATH)
    mcp_servers = state["config"].get("mcp_servers", {})
    result: Dict[str, Any] = {}
    for srv_name, srv_cfg in mcp_servers.items():
        if not srv_cfg.get("enabled", True):
            result[srv_name] = {"error": "disabled"}
            continue
        try:
            client = get_mcp_client(srv_name, srv_cfg, workspace_path=workspace_path)
            tools_list = client.list_tools()
            result[srv_name] = {
                "tools": [
                    {"name": t.get("name", ""), "description": t.get("description", "")}
                    for t in tools_list
                ]
            }
        except Exception as exc:
            result[srv_name] = {"error": str(exc)}
    return jsonify({"ok": True, "mcp_tools": result})


@app.post("/api/mcp_servers")
def add_mcp_server():
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name", "")).strip()
    command = str(payload.get("command", "")).strip()
    if not name:
        return jsonify({"ok": False, "error": "name is required"}), 400
    if not command:
        return jsonify({"ok": False, "error": "command is required"}), 400
    args = payload.get("args", [])
    if not isinstance(args, list):
        return jsonify({"ok": False, "error": "args must be an array"}), 400
    env = payload.get("env", {})
    if not isinstance(env, dict):
        return jsonify({"ok": False, "error": "env must be an object"}), 400
    enabled = payload.get("enabled", True)

    state, parent_event, _, _ = load_state()
    mcp = state["config"].setdefault("mcp_servers", {})
    existed = name in mcp
    tool_flags = payload.get("tools", {})
    if not isinstance(tool_flags, dict):
        tool_flags = {}
    mcp[name] = {"command": command, "args": args, "env": env, "enabled": bool(enabled), "tools": tool_flags}
    event = append_state_snapshot(
        state, "update_mcp_server" if existed else "add_mcp_server",
        {"server_name": name}, parent_state_uuid=parent_event["state_uuid"],
    )
    return jsonify({"ok": True, "state_uuid": event["state_uuid"], "state_hash": event["state_hash"]})


@app.delete("/api/mcp_servers/<path:server_name>")
def delete_mcp_server(server_name: str):
    name = server_name.strip()
    if not name:
        return jsonify({"ok": False, "error": "server name is required"}), 400
    state, parent_event, _, _ = load_state()
    mcp = state["config"].get("mcp_servers", {})
    if name not in mcp:
        return jsonify({"ok": False, "error": f"MCP server not found: {name}"}), 404
    del mcp[name]
    with _mcp_lock:
        client = _mcp_clients.pop(name, None)
        if client:
            client.stop()
    event = append_state_snapshot(
        state, "remove_mcp_server",
        {"server_name": name}, parent_state_uuid=parent_event["state_uuid"],
    )
    return jsonify({"ok": True, "state_uuid": event["state_uuid"], "state_hash": event["state_hash"]})


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


@app.post("/api/compact")
def compact():
    payload = request.get_json(silent=True) or {}
    requested_uuid = str(payload.get("state_uuid", "")).strip() or None
    state, parent_event, _, _ = load_state(requested_uuid)
    config = state["config"]
    history = state["history"]
    if not history:
        return jsonify({"ok": False, "error": "nothing to compact"}), 400

    compaction_prompt = (
        "Summarize this conversation so far. Include: what the user asked for, "
        "key decisions made, current state of the work, file paths and code discussed, "
        "and anything needed to continue seamlessly. Be concise but thorough."
    )
    messages = copy.deepcopy(history) + [{"role": "user", "content": compaction_prompt}]

    @stream_with_context
    def generate():
        try:
            api_key = config.get("api_key", "")
            client = OpenAI(
                base_url=config["base_url"],
                api_key=api_key or "EMPTY",
                timeout=httpx.Timeout(None, connect=10.0),
            )
            completion_stream = client.chat.completions.create(
                messages=messages,
                model=config["model"],
                stream=True,
                stream_options={"include_usage": True},
            )
            content_parts: List[str] = []
            reasoning_parts: List[str] = []
            for chunk in completion_stream:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    yield stream_line({
                        "type": "usage",
                        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                    })
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                delta_reasoning = getattr(delta, "reasoning_content", None)
                if delta_reasoning and isinstance(delta_reasoning, str):
                    reasoning_parts.append(delta_reasoning)
                    yield stream_line({"type": "reasoning_delta", "content": delta_reasoning})
                delta_content = getattr(delta, "content", None)
                if delta_content and isinstance(delta_content, str):
                    content_parts.append(delta_content)
                    yield stream_line({"type": "delta", "content": delta_content})

            summary = "".join(content_parts)

            # Collect lookup tool result outputs into one system message
            lookup_tc_ids: set = set()
            for msg in history:
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        fn = tc.get("function", {})
                        if fn.get("name") == "lookup":
                            lookup_tc_ids.add(tc.get("id"))
            lookup_outputs: List[str] = []
            for msg in history:
                if msg.get("role") == "tool" and msg.get("tool_call_id") in lookup_tc_ids:
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        try:
                            parsed = json.loads(content)
                            text = parsed.get("output", content) if isinstance(parsed, dict) else content
                        except (json.JSONDecodeError, TypeError):
                            text = content
                    else:
                        text = coerce_content(content)
                    if text.strip():
                        lookup_outputs.append(text.strip())

            # Trim from the front to fit within ~15000 tokens
            max_lookup_tokens = 15000
            while lookup_outputs and rough_token_count_text("\n\n".join(lookup_outputs)) > max_lookup_tokens:
                lookup_outputs.pop(0)

            new_history: List[Dict[str, Any]] = []
            if lookup_outputs:
                new_history.append({"role": "system", "content": "Lookup results from prior context:\n\n" + "\n\n".join(lookup_outputs)})
            new_history.append({"role": "assistant", "content": summary})
            state["history"] = new_history
            event = append_state_snapshot(state, "compact", {}, parent_state_uuid=parent_event["state_uuid"])
            yield stream_line({
                "type": "done",
                "state_uuid": event["state_uuid"],
                "state_hash": event["state_hash"],
                "context_tokens_estimate": estimate_context_tokens(state["history"]),
            })
        except Exception as exc:  # noqa: BLE001
            yield stream_line({
                "type": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(limit=5),
            })

    return Response(generate(), mimetype="application/x-ndjson")


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
    mcp_servers = config.get("mcp_servers", {})
    try:
        workspace_dir = resolve_workspace_dir(str(config.get("workspace_path", DEFAULT_WORKSPACE_PATH)))
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    tools_payload = build_tools_payload(tools, builtin_flags, mcp_servers, workspace_path=str(workspace_dir))

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
    runtime_ctx["mcp_servers"] = mcp_servers

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
            reasoning = coerce_content(getattr(message, "reasoning_content", None))
            tool_calls = message.tool_calls or []
            runtime_ctx["response_index"] = int(runtime_ctx["response_index"]) + 1

            assistant_message: Dict[str, Any] = {"role": "assistant", "content": content}
            if reasoning:
                assistant_message["reasoning_content"] = reasoning
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
    mcp_servers = config.get("mcp_servers", {})
    try:
        workspace_dir = resolve_workspace_dir(str(config.get("workspace_path", DEFAULT_WORKSPACE_PATH)))
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    tools_payload = build_tools_payload(tools, builtin_flags, mcp_servers, workspace_path=str(workspace_dir))

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
    runtime_ctx["mcp_servers"] = mcp_servers

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

            last_usage: Dict[str, Any] = {}

            while True:
                request_args: Dict[str, Any] = {
                    "messages": messages,
                    "model": config["model"],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                if tools_payload:
                    request_args["tools"] = tools_payload

                completion_stream = client.chat.completions.create(**request_args)
                content_parts: List[str] = []
                reasoning_parts: List[str] = []
                partial_tool_calls: List[Dict[str, Any]] = []
                announced_tool_indices: set = set()

                for chunk in completion_stream:
                    # Extract usage from final chunk
                    usage = getattr(chunk, "usage", None)
                    if usage is not None:
                        last_usage = {
                            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                        }
                        yield stream_line({"type": "usage", **last_usage})
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta is None:
                        continue

                    delta_reasoning = getattr(delta, "reasoning_content", None)
                    if delta_reasoning and isinstance(delta_reasoning, str):
                        reasoning_parts.append(delta_reasoning)
                        yield stream_line({"type": "reasoning_delta", "content": delta_reasoning})

                    delta_content = getattr(delta, "content", None)
                    if delta_content:
                        if isinstance(delta_content, str):
                            text_piece = delta_content
                        else:
                            text_piece = coerce_content(delta_content)
                        content_parts.append(text_piece)
                        yield stream_line({"type": "delta", "content": text_piece})

                    delta_tool_calls = getattr(delta, "tool_calls", None)
                    update_partial_tool_calls(partial_tool_calls, delta_tool_calls)

                    for idx, partial in enumerate(partial_tool_calls):
                        if idx in announced_tool_indices:
                            continue
                        name = partial.get("function", {}).get("name", "").strip()
                        if name:
                            announced_tool_indices.add(idx)
                            yield stream_line({"type": "tool_call_start", "tool_name": name, "index": idx})

                    for tc_delta in (delta_tool_calls or []):
                        idx = getattr(tc_delta, "index", 0)
                        fn = getattr(tc_delta, "function", None)
                        if fn:
                            args_frag = getattr(fn, "arguments", None)
                            if isinstance(args_frag, str) and args_frag:
                                yield stream_line({"type": "tool_call_delta", "index": idx, "arguments_delta": args_frag})

                content = "".join(content_parts)
                reasoning = "".join(reasoning_parts)
                tool_calls = finalize_partial_tool_calls(partial_tool_calls)
                runtime_ctx["response_index"] = int(runtime_ctx["response_index"]) + 1

                assistant_message: Dict[str, Any] = {"role": "assistant", "content": content}
                if reasoning:
                    assistant_message["reasoning_content"] = reasoning
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

                # Save after each assistant response finishes streaming
                ev = save_progress({
                    "prompt": prompt,
                    "stage": "assistant_response",
                    "has_tool_calls": bool(tool_calls),
                })
                yield stream_line({"type": "snapshot", "state_uuid": ev["state_uuid"]})

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

                    # Save after each tool call + result
                    ev = save_progress({
                        "prompt": prompt,
                        "stage": "tool_result",
                        "tool_name": tool_name,
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
            context_tokens = last_usage.get("total_tokens") or estimate_context_tokens(messages)
            yield stream_line(
                {
                    "type": "done",
                    "reply": final_reply,
                    "history": messages,
                    "tool_trace": tool_trace,
                    "context_tokens_estimate": context_tokens,
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

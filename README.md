# Minimal Flask Tool-Calling Web UI

This is a minimal Flask app that:

- Sends chat requests to an OpenAI-compatible server (like your `llama-server` endpoint).
- Supports tool-calling (`tools=[...]`) with Python function execution.
- Streams model output token-by-token in the chat panel.
- Does not set app-level tool-loop or `max_tokens` caps (model/backend controls stop conditions).
- Lets you add/update/delete custom tools from a web UI.
- Stores all state as append-only JSONL snapshots in `data/events.jsonl`.
- Uses a write-once `state_uuid` per snapshot event and keeps a content `state_hash`.
- Uses URL hash as `#<state_uuid>` so you can load/share a specific snapshot.
- Includes checkbox toggles for built-in tools: `pwd`, `ls`, `rg`, `rg_files`, `browse`, `edit`.
- You can set `workspace_path` in the UI; built-in tools run against that directory.

## Run

```bash
cd mincode
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open: `http://127.0.0.1:5050`

## Notes

- Tool code is executed with `exec`. Only use trusted tool code.
- State file is append-only: each mutation writes a new JSON object line.
- `browse` and `edit` call `mincode/rust_ast.py` and require:
  - `pip install tree-sitter tree-sitter-rust`
- `browse` list outputs are capped at 15 entries per call; use `offset` to paginate.
- Tool call results are normalized to a single shape: `{"output": ...}`.
- UI shows `context_tokens~N`, a rough token estimate for current chat history.

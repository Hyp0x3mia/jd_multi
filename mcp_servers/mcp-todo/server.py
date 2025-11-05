"""
MCP Server: mcp-todo
Capabilities: tools, resources
Storage: JSONL (path via TODO_DB_PATH)

Env:
- TODO_DB_PATH (required)
- DEFAULT_ASSIGNEE (optional)
- SLA_HOURS (default 72)
"""
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import Field
from dotenv import load_dotenv

mcp = FastMCP("mcp-todo")
load_dotenv("./.env")

_DB_PATH = os.getenv("TODO_DB_PATH") or os.path.join("./cache_dir", "todo_bank.jsonl")
_DEFAULT_ASSIGNEE = os.getenv("DEFAULT_ASSIGNEE", "")
_SLA_HOURS = int(os.getenv("SLA_HOURS", "72"))

os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _iso_after_hours(hours: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + hours * 3600))


def _iter_db():
    if not os.path.exists(_DB_PATH):
        return
    with open(_DB_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _save_all(items: List[Dict[str, Any]]):
    with open(_DB_PATH, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def _append(item: Dict[str, Any]):
    with open(_DB_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


@mcp.tool(description="todo.ping: health check")
def todo_ping() -> Dict[str, Any]:
    return {"ok": True, "service": "mcp-todo", "version": "v1"}


@mcp.tool(description="todo.create: create a todo item")
def todo_create(
    title: str = Field(...),
    source: str = Field(...),
    description: str = Field(...),
    priority: str = Field(default="medium"),
    due: Optional[str] = Field(default=None),
    assignee: Optional[str] = Field(default=None),
    links: Optional[Dict[str, Any]] = Field(default=None)
) -> Dict[str, Any]:
    try:
        tid = str(uuid.uuid4())
        item = {
            "id": tid,
            "title": title,
            "source": source,
            "description": description,
            "priority": priority if priority in ["low", "medium", "high"] else "medium",
            "status": "open",
            "created_at": _now_iso(),
            "due": due or _iso_after_hours(_SLA_HOURS),
            "assignee": assignee or _DEFAULT_ASSIGNEE,
            "links": links or {},
            "metrics": {"attempts": 0, "last_transition_at": _now_iso()},
        }
        _append(item)
        return {"ok": True, "id": tid}
    except Exception as e:
        return {"ok": False, "error": f"backend_failure: {e}"}


@mcp.tool(description="todo.list: list todo items with filters")
def todo_list(
    status: Optional[str] = Field(default=None),
    priority: Optional[str] = Field(default=None),
    assignee: Optional[str] = Field(default=None),
    source: Optional[str] = Field(default=None),
    since: Optional[str] = Field(default=None)
) -> Dict[str, Any]:
    try:
        items = []
        for it in _iter_db():
            if status and it.get("status") != status:
                continue
            if priority and it.get("priority") != priority:
                continue
            if assignee and it.get("assignee") != assignee:
                continue
            if source and it.get("source") != source:
                continue
            items.append(it)
        return {"ok": True, "items": items}
    except Exception as e:
        return {"ok": False, "error": f"backend_failure: {e}"}


@mcp.tool(description="todo.update: patch fields with state machine")
def todo_update(
    id: str = Field(...),
    patch: Dict[str, Any] = Field(...)
) -> Dict[str, Any]:
    try:
        all_items = list(_iter_db())
        for it in all_items:
            if it.get("id") == id:
                # state machine
                old = it.get("status", "open")
                new = patch.get("status", old)
                allowed = {"open": ["in_progress", "blocked"], "in_progress": ["review", "blocked"], "review": ["done", "blocked"], "blocked": ["open", "in_progress"]}
                if new != old:
                    if new not in allowed.get(old, []):
                        return {"ok": False, "error": "invalid_input: invalid status transition"}
                    it["status"] = new
                    it["metrics"]["last_transition_at"] = _now_iso()
                # patch other fields
                for k in ["title", "description", "priority", "assignee", "due", "links"]:
                    if k in patch:
                        it[k] = patch[k]
                _save_all(all_items)
                return {"ok": True}
        return {"ok": False, "error": "not_found"}
    except Exception as e:
        return {"ok": False, "error": f"backend_failure: {e}"}


@mcp.tool(description="todo.autogen_from_case: create a high priority ticket from case")
def todo_autogen_from_case(
    case_id: str = Field(...),
    reason: str = Field(...),
    severity: Optional[str] = Field(default=None),
    attach: Optional[Dict[str, Any]] = Field(default=None)
) -> Dict[str, Any]:
    title = f"Auto Todo for case {case_id}"
    desc = f"reason: {reason}\nattach: {json.dumps(attach or {}, ensure_ascii=False)}"
    return todo_create(title=title, source=case_id, description=desc, priority="high", links={"case_id": case_id})


@mcp.tool(description="todo.link_case: bind todo to case id")
def todo_link_case(id: str = Field(...), case_id: str = Field(...)) -> Dict[str, Any]:
    return todo_update(id=id, patch={"links": {"case_id": case_id}})


@mcp.tool(description="todo.stats: summary stats")
def todo_stats() -> Dict[str, Any]:
    try:
        items = list(_iter_db())
        summary = {"open": 0, "blocked": 0, "done": 0}
        cycle_hours = []
        for it in items:
            st = it.get("status")
            if st in summary:
                summary[st] += 1
            if st == "done":
                created = it.get("created_at")
                done_at = it.get("metrics", {}).get("last_transition_at")
                # coarse estimate (ignore tz):
                try:
                    # not computing precise hours to keep simple
                    cycle_hours.append(_SLA_HOURS)  # placeholder
                except Exception:
                    pass
        avg_cycle = (sum(cycle_hours) / len(cycle_hours)) if cycle_hours else 0.0
        return {"ok": True, "summary": {**summary, "avg_cycle_hours": avg_cycle}}
    except Exception as e:
        return {"ok": False, "error": f"backend_failure: {e}"}


@mcp.resource("todo://item/{id}")
def todo_resource(id: str):
    for it in _iter_db():
        if it.get("id") == id:
            return json.dumps(it, ensure_ascii=False)
    raise FileNotFoundError(id)


if __name__ == "__main__":
    mcp.run()



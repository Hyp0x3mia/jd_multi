"""
MCP Server: mcp-case-bank
Capabilities: tools, resources
Storage: JSONL (path via CASE_DB_PATH)

Env:
- CASE_DB_PATH (required)
- EMBEDDING_PROVIDER (optional; if absent, keyword match)
- SEARCH_TOPK (default 3)
- SIM_THRESHOLD (default 0.85)
- MAX_PLAN_SIZE_KB (default 64)
- CASEBANK_BOOTSTRAP_FILE (optional JSONL to import at startup)
"""
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import Field
from dotenv import load_dotenv

mcp = FastMCP("mcp-case-bank")
load_dotenv("./.env")


# ---------------- Storage (simple JSONL) ----------------
_DB_PATH = os.getenv("CASE_DB_PATH") or os.path.join("./cache_dir", "case_bank.jsonl")
_SEARCH_TOPK = int(os.getenv("SEARCH_TOPK", "3"))
_SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.85"))
_MAX_PLAN_SIZE_KB = int(os.getenv("MAX_PLAN_SIZE_KB", "64"))
_BOOTSTRAP = os.getenv("CASEBANK_BOOTSTRAP_FILE")

os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def _append_db(obj: Dict[str, Any]):
    with open(_DB_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _save_replace(record: Dict[str, Any]):
    # upsert by case_id
    all_items: List[Dict[str, Any]] = list(_iter_db())
    found = False
    for i, it in enumerate(all_items):
        if it.get("case_id") == record.get("case_id"):
            all_items[i] = record
            found = True
            break
    with open(_DB_PATH, "w", encoding="utf-8") as f:
        for it in all_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
        if not found:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _length_kb(s: str) -> int:
    return max(0, len(s.encode("utf-8")) // 1024)


def _basic_score(q: str, text: str) -> float:
    # naive keyword overlap scoring as fallback
    if not q or not text:
        return 0.0
    qs = set(w.lower() for w in q.split())
    ts = set(w.lower() for w in text.split())
    if not qs:
        return 0.0
    inter = len(qs & ts)
    return min(1.0, inter / max(1, len(qs)))


# ---------------- Tools ----------------

@mcp.tool(description="case_bank.ping: health check")
def casebank_ping() -> Dict[str, Any]:
    return {"ok": True, "service": "mcp-case-bank", "version": "v1"}


@mcp.tool(description="case.save: save or upsert a reusable case plan")
def case_save(
    query: str = Field(..., description="sanitized query"),
    signature: Dict[str, Any] = Field(..., description="task signature/meta"),
    plan: Dict[str, Any] = Field(..., description="reusable plan (no private inputs)"),
    tools_used: List[str] = Field(default_factory=list, description="tools used"),
    final_answer: Any = Field(..., description="final answer value"),
    validation: Dict[str, Any] = Field(default_factory=dict, description="validation schema"),
    env: Dict[str, Any] = Field(default_factory=dict, description="env versions/hashes"),
) -> Dict[str, Any]:
    try:
        plan_json = json.dumps(plan, ensure_ascii=False)
        if _length_kb(plan_json) > _MAX_PLAN_SIZE_KB:
            return {"ok": False, "error": "invalid_input: plan too large"}

        # dedupe by signature+plan
        sig_key = json.dumps(signature, sort_keys=True, ensure_ascii=False)
        for it in _iter_db():
            if json.dumps(it.get("signature", {}), sort_keys=True, ensure_ascii=False) == sig_key and json.dumps(
                it.get("plan", {}), sort_keys=True, ensure_ascii=False
            ) == plan_json:
                # update timestamps & minimal fields
                it["updated_at"] = _now_iso()
                _save_replace(it)
                return {"ok": True, "case_id": it.get("case_id")}

        record = {
            "case_id": str(uuid.uuid4()),
            "query": query,
            "signature": signature,
            "plan": plan,
            "tools_used": tools_used,
            "final_answer": final_answer,
            "validation": validation,
            "metrics": {"success_rate": 0.0, "counts": 0, "avg_latency_ms": 0.0},
            "env": env,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        _append_db(record)
        return {"ok": True, "case_id": record["case_id"]}
    except Exception as e:
        return {"ok": False, "error": f"backend_failure: {e}"}


@mcp.tool(description="case.search: semantic+keyword search cases")
def case_search(
    query: str = Field(..., description="query text"),
    signature: Optional[Dict[str, Any]] = Field(default=None, description="optional filters"),
    top_k: int = Field(default=_SEARCH_TOPK, description="top k"),
    require_answer_type: Optional[str] = Field(default=None, description="filter by signature.answer_type")
) -> Dict[str, Any]:
    try:
        items = []
        for it in _iter_db():
            if require_answer_type:
                if it.get("signature", {}).get("answer_type") != require_answer_type:
                    continue
            # basic score by keyword overlap (embedding provider not implemented)
            score = _basic_score(query, it.get("query", ""))
            if score >= _SIM_THRESHOLD:
                items.append(
                    {
                        "case_id": it.get("case_id"),
                        "query": it.get("query"),
                        "signature": it.get("signature"),
                        "plan": it.get("plan"),
                        "tools_used": it.get("tools_used", []),
                        "validation": it.get("validation", {}),
                        "final_answer": it.get("final_answer"),
                        "score": score,
                    }
                )
        items.sort(key=lambda x: x["score"], reverse=True)
        return {"ok": True, "items": items[: max(1, top_k)]}
    except Exception as e:
        return {"ok": False, "error": f"backend_failure: {e}"}


@mcp.tool(description="case.update_score: update metrics after success/failure")
def case_update_score(
    case_id: str = Field(..., description="case id"),
    success: bool = Field(..., description="success or failure"),
    latency_ms: Optional[float] = Field(default=None, description="latency in ms")
) -> Dict[str, Any]:
    try:
        all_items = list(_iter_db())
        found = False
        for it in all_items:
            if it.get("case_id") == case_id:
                m = it.setdefault("metrics", {"success_rate": 0.0, "counts": 0, "avg_latency_ms": 0.0})
                c = m.get("counts", 0) + 1
                sr = m.get("success_rate", 0.0)
                # update running success rate
                sr = (sr * m.get("counts", 0) + (1.0 if success else 0.0)) / c
                m["counts"] = c
                m["success_rate"] = sr
                if latency_ms is not None:
                    # update running average
                    m["avg_latency_ms"] = (m.get("avg_latency_ms", 0.0) * (c - 1) + float(latency_ms)) / c
                it["updated_at"] = _now_iso()
                _save_replace(it)
                found = True
                break
        if not found:
            return {"ok": False, "error": "not_found"}
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": f"backend_failure: {e}"}


@mcp.tool(description="case.get: get a case by id")
def case_get(case_id: str = Field(...)) -> Dict[str, Any]:
    for it in _iter_db():
        if it.get("case_id") == case_id:
            return {"ok": True, "item": it}
    return {"ok": False, "error": "not_found"}


@mcp.resource("case-bank://case/{case_id}")
def case_resource(case_id: str):
    for it in _iter_db():
        if it.get("case_id") == case_id:
            return json.dumps(it, ensure_ascii=False)
    raise FileNotFoundError(case_id)


def _bootstrap_if_needed():
    if not _BOOTSTRAP or not os.path.exists(_BOOTSTRAP):
        return
    seen = {it.get("case_id") for it in _iter_db()}
    imported = 0
    skipped = 0
    with open(_BOOTSTRAP, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            case_id = obj.get("task_id")
            if not case_id or case_id in seen:
                skipped += 1
                continue
            query = obj.get("query", "")
            answer = obj.get("answer")
            steps = obj.get("steps")
            level = obj.get("level")
            file_name = obj.get("file_name")
            # infer answer_type and comparator
            atype = "number" if isinstance(answer, (int, float)) else "string"
            cmp = "number" if atype == "number" else "exact"
            record = {
                "case_id": case_id,
                "query": query,
                "signature": {"level": level, "answer_type": atype},
                "plan": {"steps": steps or [] , "metadata": {"source": file_name}},
                "tools_used": [],
                "final_answer": answer,
                "validation": {"cmp": cmp},
                "metrics": {"success_rate": 0.0, "counts": 0, "avg_latency_ms": 0.0},
                "env": {},
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
            }
            _append_db(record)
            imported += 1
    print(f"[mcp-case-bank] bootstrap imported={imported}, skipped={skipped}")


if __name__ == "__main__":
    _bootstrap_if_needed()
    mcp.run()



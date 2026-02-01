from __future__ import annotations
from typing import Dict, Any, List
import json, os

# Expected: data/pubmedqa/dev.json or dev.jsonl.
# We normalize into items with:
#  - id
#  - question
#  - context_sentences: list[str]
#  - label in {yes,no,maybe}

def load_pubmedqa(split: str, root: str = "data/pubmedqa") -> List[Dict[str, Any]]:
    json_path = os.path.join(root, f"{split}.json")
    jsonl_path = os.path.join(root, f"{split}.jsonl")

    # 1) Prefer .json if present
    if os.path.exists(json_path):
        raw = _load_json_file(json_path)
        return _normalize_raw_pubmedqa(raw)

    # 2) Else fall back to .jsonl
    if os.path.exists(jsonl_path):
        raw = _load_jsonl_or_json(jsonl_path)  # robust: supports JSONL or JSON array/object
        return _normalize_raw_pubmedqa(raw)

    raise FileNotFoundError(
        f"PubMedQA split file not found. Expected one of:\n"
        f"  - {json_path}\n"
        f"  - {jsonl_path}"
    )

def _load_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_jsonl_or_json(path: str) -> Any:
    """
    Supports:
      - JSONL: one JSON object per line
      - JSON: a single JSON object or a JSON array
    This is necessary because many PubMedQA downloads are JSON arrays but sometimes named *.jsonl.
    """
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)

        # If it begins with '[' or '{', treat as a JSON file (array or object)
        if first in ("[", "{"):
            return json.load(f)

        # Otherwise treat as JSONL
        items = []
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {i} of {path}. "
                    f"Expected JSONL (1 JSON object per line) OR a JSON array/object.\n"
                    f"Offending line (truncated): {line[:200]}"
                ) from e
        return items

def _normalize_raw_pubmedqa(raw: Any) -> List[Dict[str, Any]]:
    """
    PubMedQA appears in several shapes:
      - dict keyed by id -> record
      - list of records (each may have id)
      - single record dict (rare)
    Return normalized list of items.
    """
    items: List[Dict[str, Any]] = []

    if isinstance(raw, dict):
        # Could be:
        #  (A) dict keyed by id -> record
        #  (B) a single record dict with fields like QUESTION/CONTEXTS
        looks_like_record = any(k in raw for k in ["QUESTION", "question", "query", "final_decision", "label", "answer", "CONTEXTS", "contexts", "context"])
        if looks_like_record:
            items.append(_normalize_item(str(raw.get("id", "0")), raw))
        else:
            for k, v in raw.items():
                items.append(_normalize_item(str(k), v))

    elif isinstance(raw, list):
        for v in raw:
            if isinstance(v, dict):
                items.append(_normalize_item(str(v.get("id", len(items))), v))

    else:
        raise ValueError(f"Unsupported PubMedQA format: {type(raw)}")

    return items

def _normalize_item(item_id: str, v: Dict[str, Any]) -> Dict[str, Any]:
    q = v.get("QUESTION") or v.get("question") or v.get("query") or ""
    label = v.get("final_decision") or v.get("label") or v.get("answer") or ""
    label = str(label).lower().strip()

    if label in ["yes", "no", "maybe"]:
        pass
    elif label in ["true", "supports", "support"]:
        label = "yes"
    elif label in ["false", "refutes", "refute"]:
        label = "no"
    else:
        label = "maybe"

    ctx = v.get("CONTEXTS") or v.get("contexts") or v.get("context") or []

    # ctx might be a dict with "abstract" or "sentences"
    if isinstance(ctx, dict):
        ctx = ctx.get("abstract", []) or ctx.get("sentences", []) or []

    # sometimes context is a string
    if isinstance(ctx, str):
        ctx = [s.strip() for s in ctx.split(".") if s.strip()]

    # ensure list[str]
    if not isinstance(ctx, list):
        ctx = []
    ctx = [str(s) for s in ctx if str(s).strip()]

    return {"id": item_id, "question": q, "context_sentences": ctx, "label": label}

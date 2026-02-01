from __future__ import annotations
from typing import Dict, Any, List, Optional
import csv, os

def load_matscibench(
    split: str = "test",
    root: str = "data/matscibench",
    only_num: bool = True,
    require_no_image: bool = True,
    max_steps: Optional[int] = 4,
) -> List[Dict[str, Any]]:
    """
    Expects: data/matscibench/qa.csv
    Returns items normalized to:
      - id
      - question
      - answer
      - unit
      - type
      - domain
      - solution (optional, for analysis only)
    """
    path = os.path.join(root, "qa.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Put MatSciBench qa.csv there.")

    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            qtype = (r.get("type") or "").strip()
            img = (r.get("image") or "").strip()
            steps = r.get("steps_count") or r.get("steps") or ""
            try:
                steps_i = int(float(steps)) if str(steps).strip() else None
            except Exception:
                steps_i = None

            if only_num and qtype.upper() != "NUM":
                continue
            if require_no_image and img not in ["", "None", "none", "null", "NULL"]:
                continue
            if max_steps is not None and steps_i is not None and steps_i > max_steps:
                continue

            qid = str(r.get("qid") or r.get("id") or len(items))
            question = (r.get("question") or "").strip()
            answer = (r.get("answer") or "").strip()
            unit = (r.get("unit") or "").strip()
            domain = (r.get("domain") or "").strip()
            solution = (r.get("solution") or "").strip()

            items.append({
                "id": qid,
                "question": question,
                "answer": answer,
                "unit": unit,
                "type": qtype,
                "domain": domain,
                "solution": solution,  # keep for error analysis, not inference
            })

    return items

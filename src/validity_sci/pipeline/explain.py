from __future__ import annotations
from typing import List, Dict, Any
from ..utils import ConditionAudit, EvidenceSentence

def build_explanation(audits: List[ConditionAudit], max_sents: int = 4) -> Dict[str, Any]:
    """Extractive explanation composed only of cited evidence sentences."""
    used = []
    seen = set()
    for a in sorted(audits, key=lambda x: x.score, reverse=True):
        ev = a.best_evidence
        if ev is None:
            continue
        key = (ev.doc_id, ev.sent_id)
        if key in seen:
            continue
        seen.add(key)
        used.append({"doc_id": ev.doc_id, "sent_id": ev.sent_id, "text": ev.text, "condition": a.condition.text, "status": a.status, "score": a.score})
        if len(used) >= max_sents:
            break
    return {"sentences": used}

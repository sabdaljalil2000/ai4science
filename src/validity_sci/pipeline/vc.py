from __future__ import annotations
from typing import Any, Dict, List, Optional
from ..utils import ConditionAudit

def build_vc(
    item_id: str,
    input_text: str,
    prediction: str,
    confidence: float,
    audits: List[ConditionAudit],
    evidence_sentences: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Validity Certificate JSON."""
    vc = {
        "id": item_id,
        "input": input_text,
        "prediction": prediction,
        "confidence": confidence,
        "conditions": [],
        "evidence": evidence_sentences,  # full list used by the run (so VC can cite ids)
    }
    for a in audits:
        ev = a.best_evidence
        vc["conditions"].append({
            "text": a.condition.text,
            "critical": a.condition.critical,
            "status": a.status,
            "score": a.score,
            "nli_label": a.nli_label,
            "best_evidence": None if ev is None else {"doc_id": ev.doc_id, "sent_id": ev.sent_id},
        })
    return vc

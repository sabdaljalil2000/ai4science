from __future__ import annotations
from typing import List, Dict, Any, Callable, Tuple
from copy import deepcopy
from ..utils import EvidenceSentence, Condition, ConditionAudit
from .audit import audit_conditions

def greedy_minimal_sufficient_evidence(
    nli,
    conditions: List[Condition],
    evidence: List[EvidenceSentence],
    decide_fn: Callable[[List[ConditionAudit]], Tuple[str, float]],
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    """Greedy removal: keep removing evidence sentences that don't change decision."""
    if not evidence:
        audits = audit_conditions(nli, conditions, evidence, **thresholds)
        pred, conf = decide_fn(audits)
        return {"kept": [], "removed": [], "pred": pred, "conf": conf, "sufficient": True, "kept_frac": 0.0}

    cur = evidence[:]
    removed = []
    # baseline decision
    audits0 = audit_conditions(nli, conditions, cur, **thresholds)
    pred0, conf0 = decide_fn(audits0)

    changed = True
    while changed and len(cur) > 1:
        changed = False
        # try remove each sentence, keep the first that preserves decision
        for i in range(len(cur)):
            trial = cur[:i] + cur[i+1:]
            audits = audit_conditions(nli, conditions, trial, **thresholds)
            pred, _ = decide_fn(audits)
            if pred == pred0:
                removed.append(cur[i])
                cur = trial
                changed = True
                break

    return {
        "kept": [{"doc_id": e.doc_id, "sent_id": e.sent_id, "text": e.text} for e in cur],
        "removed": [{"doc_id": e.doc_id, "sent_id": e.sent_id, "text": e.text} for e in removed],
        "pred": pred0,
        "conf": conf0,
        "sufficient": True,  # by construction we preserve decision
        "kept_frac": len(cur) / max(1, len(evidence)),
        "kept_n": len(cur),
        "orig_n": len(evidence),
    }

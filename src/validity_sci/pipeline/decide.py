from __future__ import annotations
from typing import List, Tuple
from ..utils import ConditionAudit

def decide_verification(audits: List[ConditionAudit], abstain: bool = True) -> Tuple[str, float]:
    """Return (label, confidence).
    Confidence is a simple aggregate score; you can replace with calibration.
    """
    if not audits:
        return ("NEI", 0.0)

    if any(a.status == "contradicted" for a in audits):
        conf = max(a.score for a in audits if a.status == "contradicted")
        return ("REFUTES", conf)

    if abstain and any(a.condition.critical and a.status == "missing" for a in audits):
        conf = 1.0 - max(a.score for a in audits if a.status == "missing")
        return ("NEI", conf)

    # supports if no contradictions and no missing critical
    conf = min(1.0, sum(a.score for a in audits if a.status == "supported") / max(1, sum(1 for a in audits if a.status=="supported")))
    return ("SUPPORTS", conf)

def decide_qa(audits: List[ConditionAudit], abstain: bool = True) -> Tuple[str, float]:
    """PubMedQA-style: yes/no/maybe.
    - contradiction => 'no'
    - missing critical => 'maybe'
    - else => 'yes'
    """
    if not audits:
        return ("maybe", 0.0)
    if any(a.status == "contradicted" for a in audits):
        conf = max(a.score for a in audits if a.status == "contradicted")
        return ("no", conf)
    if abstain and any(a.condition.critical and a.status == "missing" for a in audits):
        conf = 1.0 - max(a.score for a in audits if a.status == "missing")
        return ("maybe", conf)
    conf = min(1.0, sum(a.score for a in audits if a.status == "supported") / max(1, sum(1 for a in audits if a.status=="supported")))
    return ("yes", conf)

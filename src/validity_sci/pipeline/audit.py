from __future__ import annotations
from typing import List, Tuple, Optional
from ..utils import Condition, EvidenceSentence, ConditionAudit
from ..models.nli import NLIModel

def audit_conditions(
    nli: NLIModel,
    conditions: List[Condition],
    evidence: List[EvidenceSentence],
    support_entailment: float,
    refute_contradiction: float,
    missing_neutral_max: float,
) -> List[ConditionAudit]:
    if not evidence:
        return [ConditionAudit(c, status="missing", best_evidence=None, score=0.0, nli_label=None) for c in conditions]

    audits: List[ConditionAudit] = []
    for cond in conditions:
        premises = [e.text for e in evidence]
        hypotheses = [cond.text] * len(evidence)
        scores = nli.score(premises, hypotheses)
        # pick best by max(entailment, contradiction)
        best_i = 0
        best_val = -1.0
        best_lab = "neutral"
        best_trip = (0.0,0.0,0.0)
        for i, s in enumerate(scores):
            m = max(s.entailment, s.contradiction, s.neutral)
            # prioritize informative labels (ent/contrad) if close
            cand_val = max(s.entailment, s.contradiction)
            if cand_val > best_val:
                best_val = cand_val
                best_i = i
                if s.entailment >= s.contradiction and s.entailment >= s.neutral:
                    best_lab = "entailment"
                elif s.contradiction >= s.entailment and s.contradiction >= s.neutral:
                    best_lab = "contradiction"
                else:
                    best_lab = "neutral"
                best_trip = (s.entailment, s.contradiction, s.neutral)

        best_e = evidence[best_i]
        ent, con, neu = best_trip

        # status rules
        if con >= refute_contradiction:
            status = "contradicted"
            score = con
            nli_label = "contradiction"
        elif ent >= support_entailment:
            status = "supported"
            score = ent
            nli_label = "entailment"
        else:
            # treat as missing if neutral-ish / low confidence
            status = "uncertain" if max(ent, con) <= missing_neutral_max else "missing"
            # status = "missing" if max(ent, con) <= missing_neutral_max else "missing"
            score = max(ent, con)
            nli_label = "neutral"

        audits.append(ConditionAudit(condition=cond, status=status, best_evidence=best_e, score=score, nli_label=nli_label))
    return audits

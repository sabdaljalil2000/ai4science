from __future__ import annotations
from typing import Dict, Any, List, Tuple
import json, os
from ..utils import EvidenceSentence

# Expected layout (official SciFact):
# data/scifact/claims_train.jsonl, claims_dev.jsonl, claims_test.jsonl
# and corpus.jsonl with abstracts split into sentences (or a separate sentence file).
#
# This loader is deliberately tolerant: if your files differ, adapt here.

def load_scifact(split: str, root: str = "data/scifact") -> List[Dict[str, Any]]:
    path = os.path.join(root, f"claims_{split}.jsonl")
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def load_scifact_corpus(root: str = "data/scifact") -> Dict[str, List[str]]:
    # corpus.jsonl lines: {"doc_id": ..., "abstract": [...sentences...]} or similar
    path = os.path.join(root, "corpus.jsonl")
    corpus = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            r = json.loads(line)
            doc_id = str(r.get("doc_id", r.get("docid", r.get("paper_id"))))
            sents = r.get("abstract") or r.get("sentences") or []
            corpus[doc_id] = sents
    return corpus

def get_evidence_sentences(
    item: Dict[str, Any],
    corpus: Dict[str, List[str]],
    max_sents: int = 20
) -> List[EvidenceSentence]:
    """
    Supports multiple SciFact-like schemas, including your schema:
      evidence: { "<doc_id>": [ {"sentences":[...], "label":"SUPPORT|CONTRADICT"}, ... ] }
    """
    evidence: List[EvidenceSentence] = []
    ev = item.get("evidence") or item.get("evidence_sets") or {}

    # Case A) Your schema: evidence is a dict keyed by doc_id
    if isinstance(ev, dict):
        for doc_id, ann_list in ev.items():
            doc_id = str(doc_id)
            sents = corpus.get(doc_id, [])
            if not isinstance(ann_list, list):
                continue
            for ann in ann_list:
                if not isinstance(ann, dict):
                    continue
                for sent_id in ann.get("sentences", []) or []:
                    try:
                        sent_id = int(sent_id)
                    except Exception:
                        continue
                    if 0 <= sent_id < len(sents):
                        evidence.append(EvidenceSentence(doc_id=doc_id, sent_id=sent_id, text=sents[sent_id]))
        return evidence[:max_sents]

    # Case B) Older/other schema: evidence is a list of sets/pairs
    ev_sets = ev if isinstance(ev, list) else []
    for ev_item in ev_sets:
        # ev_item could be list of pairs or dict with "sentences"
        pairs = ev_item.get("sentences") if isinstance(ev_item, dict) and "sentences" in ev_item else ev_item
        if not isinstance(pairs, list):
            continue
        for pair in pairs:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                doc_id, sent_id = str(pair[0]), int(pair[1])
                sents = corpus.get(doc_id, [])
                if 0 <= sent_id < len(sents):
                    evidence.append(EvidenceSentence(doc_id=doc_id, sent_id=sent_id, text=sents[sent_id]))

    return evidence[:max_sents]

def get_gold_label(item: Dict[str, Any]) -> str:
    """
    Derive a claim label from evidence annotations:
      - any SUPPORT => SUPPORTS
      - else any CONTRADICT => REFUTES
      - else => NOT_ENOUGH_INFO
    """
    ev = item.get("evidence") or {}
    has_support = False
    has_contra = False

    if isinstance(ev, dict):
        for _, ann_list in ev.items():
            if not isinstance(ann_list, list):
                continue
            for ann in ann_list:
                if not isinstance(ann, dict):
                    continue
                lab = (ann.get("label") or "").upper()
                if lab == "SUPPORT":
                    has_support = True
                elif lab == "CONTRADICT":
                    has_contra = True

    if has_support:
        return "SUPPORTS"
    if has_contra:
        return "REFUTES"
    return "NOT_ENOUGH_INFO"



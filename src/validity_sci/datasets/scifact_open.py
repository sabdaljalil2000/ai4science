from __future__ import annotations
from typing import Dict, Any, List, Tuple
import json, os
from ..utils import EvidenceSentence

# This expects a corpus of abstracts/sentences for retrieval.
# Provide a JSONL: {"doc_id": "...", "sentences": ["...", "...", ...]}
# Put it at data/scifact_open/corpus_sentences.jsonl

def load_scifact_open(split: str, root: str = "data/scifact_open") -> List[Dict[str, Any]]:
    path = os.path.join(root, f"claims_{split}.jsonl")
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

# def iter_corpus_sentences(root: str = "data/scifact_open"):
#     path = os.path.join(root, "corpus_sentences.jsonl")
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             if not line.strip(): 
#                 continue
#             r = json.loads(line)
#             doc_id = str(r["doc_id"])
#             sents = r.get("sentences", [])
#             for i, s in enumerate(sents):
#                 yield (doc_id, i, s)
def iter_corpus_sentences(root: str = "data/scifact_open"):
    path = os.path.join(root, "corpus.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            doc_id = str(r.get("doc_id"))
            sents = r.get("abstract", []) or []
            for i, s in enumerate(sents):
                s = (s or "").strip()
                if not s:
                    continue
                yield EvidenceSentence(doc_id=doc_id, sent_id=i, text=s)

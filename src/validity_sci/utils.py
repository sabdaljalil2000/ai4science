from __future__ import annotations
import os, random, json
import numpy as np
import torch
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple, Optional

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

@dataclass
class EvidenceSentence:
    doc_id: str
    sent_id: int
    text: str

@dataclass
class Condition:
    text: str
    critical: bool = True

@dataclass
class ConditionAudit:
    condition: Condition
    status: str  # supported|contradicted|missing
    best_evidence: Optional[EvidenceSentence]
    score: float
    nli_label: Optional[str] = None  # entailment|contradiction|neutral

LABELS_VERIF = ["SUPPORTS", "REFUTES", "NEI"]  # claim verification
LABELS_QA = ["yes", "no", "maybe"]

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# MNLI label mapping differs per model; we normalize to {entailment, contradiction, neutral}
def _infer_label_map(model_name: str, id2label: Dict[int, str]) -> Dict[str, int]:
    lowered = {k: v.lower() for k, v in id2label.items()}
    # common variants
    def find(target: str):
        for i, lab in lowered.items():
            if target in lab:
                return i
        return None
    ent = find("entail")
    con = find("contrad")
    neu = find("neutral")
    if ent is None or con is None or neu is None:
        # fallback to MNLI standard ordering for many models: contradiction=0, neutral=1, entailment=2
        return {"contradiction": 0, "neutral": 1, "entailment": 2}
    return {"entailment": ent, "contradiction": con, "neutral": neu}

@dataclass
class NLIScore:
    entailment: float
    contradiction: float
    neutral: float

class NLIModel:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.label_map = _infer_label_map(model_name, self.model.config.id2label)

    @torch.no_grad()
    def score(self, premises: List[str], hypotheses: List[str], batch_size: int = 16) -> List[NLIScore]:
        assert len(premises) == len(hypotheses)
        out: List[NLIScore] = []
        for i in range(0, len(premises), batch_size):
            p = premises[i:i+batch_size]
            h = hypotheses[i:i+batch_size]
            enc = self.tok(p, h, padding=True, truncation=True, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu()
            for row in probs:
                ent = float(row[self.label_map["entailment"]])
                con = float(row[self.label_map["contradiction"]])
                neu = float(row[self.label_map["neutral"]])
                out.append(NLIScore(entailment=ent, contradiction=con, neutral=neu))
        return out

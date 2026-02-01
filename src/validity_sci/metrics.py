from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def classification_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }

def risk_coverage_curve(correct: List[int], conf: List[float], num_points: int = 50) -> List[Dict[str, float]]:
    """Return points with threshold -> coverage, risk.
    - coverage: fraction of answered (conf >= thr)
    - risk: 1 - accuracy among answered
    """
    correct = np.asarray(correct)
    conf = np.asarray(conf)
    thrs = np.linspace(0.0, 1.0, num_points)
    out = []
    for t in thrs:
        mask = conf >= t
        cov = float(mask.mean()) if mask.size else 0.0
        if mask.sum() == 0:
            risk = 0.0
        else:
            acc = float(correct[mask].mean())
            risk = 1.0 - acc
        out.append({"threshold": float(t), "coverage": cov, "risk": risk})
    return out

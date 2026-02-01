from __future__ import annotations
import yaml
from typing import Any, Dict
from .config import Config, ModelConfig, ThresholdConfig, RetrievalConfig, RunConfig

def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)
    cfg = Config()
    # shallow merge by sections
    if raw.get("task"): cfg.task = raw["task"]
    if "model" in raw: cfg.model = ModelConfig(**{**cfg.model.__dict__, **raw["model"]})
    if "thresholds" in raw: cfg.thresholds = ThresholdConfig(**{**cfg.thresholds.__dict__, **raw["thresholds"]})
    if "retrieval" in raw: cfg.retrieval = RetrievalConfig(**{**cfg.retrieval.__dict__, **raw["retrieval"]})
    if "run" in raw: cfg.run = RunConfig(**{**cfg.run.__dict__, **raw["run"]})
    return cfg

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class ModelConfig:
    decomposer_model: str = "google/flan-t5-base"
    nli_model: str = "microsoft/deberta-v3-base-mnli"
    # embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cuda"  # or cpu
    
    decomposer_backend: str = "hf_seq2seq"
    decomposer_max_new_tokens: int = 128
    decomposer_temperature: float = 0.0
    decomposer_api_base_url: str = ""

@dataclass
class ThresholdConfig:
    support_entailment: float = 0.55
    refute_contradiction: float = 0.55
    missing_neutral_max: float = 0.50  # if best is neutral-ish, treat missing

@dataclass
class RetrievalConfig:
    top_k_docs: int = 5
    top_k_sents: int = 10

@dataclass
class RunConfig:
    seed: int = 13
    output_dir: str = "runs"
    max_instances: Optional[int] = None

@dataclass
class Config:
    task: str = "scifact"
    model: ModelConfig = field(default_factory=ModelConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    run: RunConfig = field(default_factory=RunConfig)

def as_dict(cfg: Config) -> Dict[str, Any]:
    from dataclasses import asdict
    return asdict(cfg)

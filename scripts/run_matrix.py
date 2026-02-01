import argparse
import copy
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


import yaml

TASKS_DEFAULT = ["scifact", "scifact_open", "pubmedqa"]

# Define the generator (LLM) set you want to test.
# backend must match how you implemented it in ConditionDecomposer:
# hf_seq2seq | hf_chat | openai | deepseek
GENERATORS = [
    # Local baseline (fast)
    {
        "name": "flan_t5_large",
        "backend": "hf_seq2seq",
        "model": "google/flan-t5-large",
        "max_new_tokens": 128,
        "temperature": 0.0,
    },

    # SOTA / strong models via OpenRouter (unified API)
    {
        "name": "gpt4o_mini",
        "backend": "openrouter",
        "model": "openai/gpt-4o-mini",
        "max_new_tokens": 256,
        "temperature": 0.0,
    },
    {
        "name": "gpt-5.2-chat",
        "backend": "openrouter",
        "model": "openai/gpt-5.2-chat",
        "max_new_tokens": 256,
        "temperature": 0.0,
    },
    
    {
        "name": "llama33_70b",
        "backend": "openrouter",
        "model": "meta-llama/llama-3.3-70b-instruct",
        "max_new_tokens": 256,
        "temperature": 0.0,
    },
    {
        "name": "mistral_large",
        "backend": "openrouter",
        "model": "mistralai/mistral-large",
        "max_new_tokens": 256,
        "temperature": 0.0,
    },
    {
        "name": "deepseek_chat",
        "backend": "openrouter",
        "model": "deepseek/deepseek-chat",
        "max_new_tokens": 256,
        "temperature": 0.0,
    },
    # {
    # "name": "palm2_chat_bison",
    # "backend": "openrouter",
    # "model": "google/palm-2-chat-bison",
    # "max_new_tokens": 256,
    # "temperature": 0.0,
    # },
    
    # {
    # "name": "gemma-3-12b-it",
    # "backend": "openrouter",
    # "model": "google/gemma-3-12b-it",
    # "max_new_tokens": 256,
    # "temperature": 0.0,
    # },
]


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def dump_yaml(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def patch_config(base_cfg: dict, gen: dict, device: Optional[str]) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("model", {})
    cfg["model"]["decomposer_backend"] = gen["backend"]
    cfg["model"]["decomposer_model"] = gen["model"]
    cfg["model"]["decomposer_max_new_tokens"] = gen.get("max_new_tokens", 128)
    cfg["model"]["decomposer_temperature"] = gen.get("temperature", 0.0)
    if "api_base_url" in gen:
        cfg["model"]["decomposer_api_base_url"] = gen["api_base_url"]
    if device is not None:
        cfg["model"]["device"] = device
    return cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs-dir", default="configs", help="Directory containing base dataset YAMLs")
    ap.add_argument("--tasks", nargs="*", default=TASKS_DEFAULT, help="Tasks to run")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max", type=int, default=200)
    ap.add_argument("--device", default=None, help="Override device (cpu/cuda/mps), else use config")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    configs_dir = Path(args.configs_dir)

    # Map task -> base config path (expects e.g. configs/scifact.yaml)
    task_to_cfg = {t: configs_dir / f"{t}.yaml" for t in args.tasks}

    for task, cfg_path in task_to_cfg.items():
        if not cfg_path.exists():
            print(f"[ERROR] Missing config for task '{task}': {cfg_path}", file=sys.stderr)
            sys.exit(1)

    for task, cfg_path in task_to_cfg.items():
        base_cfg = load_yaml(cfg_path)

        for gen in GENERATORS:
            # Build an “effective” config
            eff_cfg = patch_config(base_cfg, gen, args.device)

            # Put temp config in a deterministic location
            tmp_dir = Path(".tmp_effective_configs") / task
            tmp_cfg_path = tmp_dir / f"{task}__{gen['name']}.yaml"
            dump_yaml(eff_cfg, tmp_cfg_path)

            cmd = [
                sys.executable, "run_experiment.py",
                "--task", task,
                "--config", str(tmp_cfg_path),
                "--split", args.split,
                "--max", str(args.max),
            ]

            print("\n==>", " ".join(cmd))
            if args.dry_run:
                continue

            # Run
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import os
import json
import re
from datetime import datetime
from typing import Any, Dict, List

from rich.console import Console
from rich.table import Table

from .io import load_config
from .utils import set_seed, ensure_dir, EvidenceSentence
from .models.nli import NLIModel
from .models.llm_condition import ConditionDecomposer
from .pipeline.audit import audit_conditions
from .pipeline.decide import decide_verification, decide_qa
from .pipeline.vc import build_vc
from .pipeline.explain import build_explanation
from .pipeline.mse import greedy_minimal_sufficient_evidence
from .metrics import classification_metrics, risk_coverage_curve

from .datasets.scifact import load_scifact, load_scifact_corpus, get_evidence_sentences
from .datasets.pubmedqa import load_pubmedqa
from .datasets.scifact_open import load_scifact_open, iter_corpus_sentences
from .datasets.matbench import load_matscibench

from .retrieval.faiss_retriever import FaissSentenceRetriever

console = Console()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, required=True, choices=["scifact", "scifact_open", "pubmedqa", "matbench"])
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--split", type=str, default="dev")
    ap.add_argument("--build-index", action="store_true")
    ap.add_argument("--no-decompose", action="store_true")
    ap.add_argument("--no-audit", action="store_true")
    ap.add_argument("--no-abstain", action="store_true")
    ap.add_argument("--no-mse", action="store_true")
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument(
    "--experiment-tag",
    type=str,
    default="main",
    help="Run grouping tag (e.g., main, ablations)"
)

    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.task = args.task

    set_seed(cfg.run.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = getattr(cfg.model, "decomposer_model", "unknown_model")
    model_tag = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name)
    
    base_dir = cfg.run.output_dir

    # Route ablations separately
    if args.experiment_tag != "main":
        base_dir = os.path.join(base_dir, "ablations")

    run_dir = os.path.join(
        base_dir,
        args.task,
        f"{model_tag}__{args.experiment_tag}__{ts}",
    )


    # run_dir = os.path.join(cfg.run.output_dir, args.task, f"{model_tag}__{ts}")
    ensure_dir(run_dir)

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(_cfg_to_json(cfg), f, indent=2)

    # ---------------- models / thresholds / run buffers ----------------
    nli = NLIModel(cfg.model.nli_model, device=cfg.model.device)

    decomposer = None if args.no_decompose else ConditionDecomposer(
        model_name=cfg.model.decomposer_model,
        device=cfg.model.device,
        backend=getattr(cfg.model, "decomposer_backend", "hf_seq2seq"),
        temperature=getattr(cfg.model, "decomposer_temperature", 0.0),
        api_base_url=(getattr(cfg.model, "decomposer_api_base_url", "") or None),
    )

    thresholds = {
        "support_entailment": cfg.thresholds.support_entailment,
        "refute_contradiction": cfg.thresholds.refute_contradiction,
        "missing_neutral_max": cfg.thresholds.missing_neutral_max,
    }

    preds: List[str] = []
    vc_rows: List[Dict[str, Any]] = []
    mse_rows: List[Dict[str, Any]] = []
    y_true: List[Any] = []
    y_pred: List[Any] = []
    correct: List[int] = []
    confs: List[float] = []
    # -------------------------------------------------------------------

    # =========================
    # MatSciBench (numeric QA)
    # =========================
    if cfg.task == "matbench":
        if decomposer is None:
            raise RuntimeError("matbench requires a decomposer backend (OpenRouter / HF). Remove --no-decompose.")

        items = load_matscibench(split="test")
        if args.max or getattr(cfg.run, "max_instances", None):
            items = items[: (args.max or cfg.run.max_instances)]

        for it in _progress(items, "MatSciBench"):
            item_id = str(it.get("id", len(vc_rows)))
            q = it.get("question", "")
            gold = it.get("answer", "")
            unit_hint = it.get("unit", "")

            pred_line = decomposer.answer_numeric(q, unit_hint=unit_hint)

            vc_rows.append(
                {
                    "id": item_id,
                    "input": q,
                    "gold": gold,
                    "prediction": pred_line,
                    "confidence": 1.0,
                    "meta": {
                        "unit_hint": unit_hint,
                        "domain": it.get("domain", ""),
                        "type": it.get("type", ""),
                        "source": it.get("source", ""),
                    },
                }
            )

        def _extract_float(answer_line: str):
            if not answer_line:
                return None
            up = answer_line.upper()
            if "ANSWER:" not in up:
                return None
            part = answer_line.split("ANSWER:", 1)[1].strip()
            part = part.split("|", 1)[0].strip()
            if part.upper().startswith("UNKNOWN"):
                return None
            try:
                return float(part)
            except Exception:
                return None

        def _numeric_correct(pred_val, gold_val, rel_tol=0.01, abs_tol=1e-6) -> int:
            if pred_val is None:
                return 0
            try:
                g = float(gold_val)
            except Exception:
                return 0
            return int(abs(pred_val - g) <= max(abs_tol, rel_tol * abs(g)))

        preds_num = [_extract_float(r.get("prediction", "")) for r in vc_rows]
        correct = [_numeric_correct(p, r.get("gold", "")) for p, r in zip(preds_num, vc_rows)]
        confs = [1.0] * len(correct)

        acc = sum(correct) / max(1, len(correct))
        m = {"accuracy": acc, "n": len(correct)}

        rc = risk_coverage_curve(correct, confs, num_points=51)

        _write_jsonl(os.path.join(run_dir, "vc_logs.jsonl"), vc_rows)
        _write_csv(os.path.join(run_dir, "risk_coverage.csv"), rc)

        with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)

        _print_summary(m, rc, run_dir)
        return

    # =========================
    # SciFact (closed evidence)
    # =========================
    if cfg.task == "scifact":
        items = load_scifact(args.split)
        corpus = load_scifact_corpus()
        if args.max or getattr(cfg.run, "max_instances", None):
            items = items[: (args.max or cfg.run.max_instances)]

        for it in _progress(items, "SciFact"):
            item_id = str(it.get("id", it.get("claim_id", len(preds))))
            claim = it.get("claim") or it.get("text") or ""
            gold = _normalize_scifact_label(it.get("label", "NEI"))
            evidence = get_evidence_sentences(it, corpus, max_sents=cfg.retrieval.top_k_sents)

            pred_row = _run_one(
                item_id=item_id,
                input_text=claim,
                gold=gold,
                evidence=evidence,
                decomposer=decomposer,
                nli=nli,
                thresholds=thresholds,
                abstain=not args.no_abstain,
                task="verif",
                do_audit=not args.no_audit,
                do_mse=not args.no_mse,
            )
            preds.append(pred_row["pred"])
            vc_rows.append(pred_row["vc"])
            if pred_row.get("mse") is not None:
                mse_rows.append(pred_row["mse"])
            y_true.append(gold)
            y_pred.append(pred_row["pred"])
            correct.append(1 if pred_row["pred"] == gold else 0)
            confs.append(float(pred_row["conf"]))

    # =========================
    # PubMedQA (context-as-evidence)
    # =========================
    elif cfg.task == "pubmedqa":
        items = load_pubmedqa(args.split)
        if args.max or getattr(cfg.run, "max_instances", None):
            items = items[: (args.max or cfg.run.max_instances)]

        for it in _progress(items, "PubMedQA"):
            item_id = str(it["id"])
            q = it["question"]
            gold = it["label"]
            evidence = [
                EvidenceSentence(doc_id=item_id, sent_id=i, text=s)
                for i, s in enumerate(it["context_sentences"][: cfg.retrieval.top_k_sents])
            ]
            pred_row = _run_one(
                item_id=item_id,
                input_text=q,
                gold=gold,
                evidence=evidence,
                decomposer=decomposer,
                nli=nli,
                thresholds=thresholds,
                abstain=not args.no_abstain,
                task="qa",
                do_audit=not args.no_audit,
                do_mse=not args.no_mse,
            )
            vc_rows.append(pred_row["vc"])
            if pred_row.get("mse") is not None:
                mse_rows.append(pred_row["mse"])
            y_true.append(gold)
            y_pred.append(pred_row["pred"])
            correct.append(1 if pred_row["pred"] == gold else 0)
            confs.append(float(pred_row["conf"]))

    # =========================
    # SciFact-Open (retrieval)
    # =========================
    elif cfg.task == "scifact_open":
        items = load_scifact_open(args.split)
        if args.max or getattr(cfg.run, "max_instances", None):
            items = items[: (args.max or cfg.run.max_instances)]

        index_path = os.path.join("data/scifact_open", f"faiss_{_safe(cfg.model.embed_model)}.index")
        retriever = FaissSentenceRetriever(
            cfg.model.embed_model, index_path=index_path, device=("cuda" if cfg.model.device == "cuda" else "cpu")
        )

        if args.build_index or (not os.path.exists(index_path)):
            console.print("[bold]Building FAISS sentence index...[/bold]")
            sents = list(iter_corpus_sentences())
            retriever.build(sents)
            console.print("Index built at", index_path)
            if args.build_index:
                return
        else:
            retriever.load()

        for it in _progress(items, "SciFact-Open"):
            item_id = str(it.get("id", len(vc_rows)))
            claim = it.get("claim") or it.get("text") or ""
            gold = _normalize_scifact_label(it.get("label", "NEI"))
            retrieved = retriever.search(claim, top_k=cfg.retrieval.top_k_sents)
            evidence = [EvidenceSentence(doc_id=r.doc_id, sent_id=r.sent_id, text=r.text) for r in retrieved]
            pred_row = _run_one(
                item_id=item_id,
                input_text=claim,
                gold=gold,
                evidence=evidence,
                decomposer=decomposer,
                nli=nli,
                thresholds=thresholds,
                abstain=not args.no_abstain,
                task="verif",
                do_audit=not args.no_audit,
                do_mse=not args.no_mse,
            )
            vc_rows.append(pred_row["vc"])
            if pred_row.get("mse") is not None:
                mse_rows.append(pred_row["mse"])
            y_true.append(gold)
            y_pred.append(pred_row["pred"])
            correct.append(1 if pred_row["pred"] == gold else 0)
            confs.append(float(pred_row["conf"]))

    # ---------------- Save artifacts (classification tasks) ----------------
    # Record abstention decision at a default threshold for analysis
    DEFAULT_ABSTAIN_THRESHOLD = 0.5
    for r in vc_rows:
        try:
            r["abstained"] = float(r.get("confidence", 0.0)) < DEFAULT_ABSTAIN_THRESHOLD
        except Exception:
            r["abstained"] = True

    _write_jsonl(os.path.join(run_dir, "vc_logs.jsonl"), vc_rows)
    if mse_rows:
        _write_jsonl(os.path.join(run_dir, "mse.jsonl"), mse_rows)

    # Metrics (classification)
    labels = ["SUPPORTS", "REFUTES", "NEI"] if cfg.task in ["scifact", "scifact_open"] else ["yes", "no", "maybe"]
    m = classification_metrics(y_true, y_pred, labels=labels)
    m["n"] = len(y_true)

    # Risk–coverage
    rc = risk_coverage_curve(correct, confs, num_points=51)
    _write_csv(os.path.join(run_dir, "risk_coverage.csv"), rc)

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)

    _print_summary(m, rc, run_dir)


def _run_one(item_id, input_text, gold, evidence, decomposer, nli, thresholds, abstain, task, do_audit, do_mse):
    if decomposer is None:
        conditions = [{"text": input_text, "critical": True}]
        conditions_obj = None
    else:
        conditions_obj = decomposer.decompose(input_text)
        conditions = [{"text": c.text, "critical": c.critical} for c in conditions_obj]

    if not do_audit:
        audits = []
        pred, conf = ("NEI", 0.0) if task == "verif" else ("maybe", 0.0)
    else:
        audits = audit_conditions(nli, conditions_obj or [_cond_from_dict(conditions[0])], evidence, **thresholds)
        decide_fn = (lambda a: decide_verification(a, abstain=abstain)) if task == "verif" else (lambda a: decide_qa(a, abstain=abstain))
        pred, conf = decide_fn(audits)

    expl = build_explanation(audits)
    evidence_dump = [{"doc_id": e.doc_id, "sent_id": e.sent_id, "text": e.text} for e in evidence]
    vc = build_vc(item_id, input_text, pred, float(conf), audits, evidence_dump)
    vc["explanation"] = expl

    mse = None
    if do_mse and do_audit and conditions_obj is not None:
        decide_fn = (lambda a: decide_verification(a, abstain=abstain)) if task == "verif" else (lambda a: decide_qa(a, abstain=abstain))
        mse = greedy_minimal_sufficient_evidence(nli, conditions_obj, evidence, decide_fn, thresholds)

    return {"id": item_id, "gold": gold, "pred": pred, "conf": float(conf), "vc": vc, "mse": mse}


def _cond_from_dict(d):
    from .utils import Condition
    return Condition(text=d["text"], critical=bool(d.get("critical", True)))


def _progress(items, name):
    from tqdm import tqdm
    return tqdm(items, desc=name)


def _normalize_scifact_label(lbl: str) -> str:
    lbl = str(lbl).upper().strip()
    if lbl in ["SUPPORTS", "REFUTES", "NEI"]:
        return lbl
    if lbl in ["SUPPORTED", "SUPPORT"]:
        return "SUPPORTS"
    if lbl in ["REFUTED", "REFUTE"]:
        return "REFUTES"
    return "NEI"


def _safe(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", s)


def _cfg_to_json(cfg):
    from .config import as_dict
    return as_dict(cfg)


def _write_jsonl(path, rows):
    import orjson
    with open(path, "wb") as f:
        for r in rows:
            f.write(orjson.dumps(r) + b"\n")


def _write_csv(path, rows):
    import csv
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _print_summary(metrics, rc, run_dir):
    t = Table(title="Run summary")
    t.add_column("Metric")
    t.add_column("Value")
    for k, v in metrics.items():
        t.add_row(k, str(v))
    console.print(t)
    console.print(f"Artifacts saved to: [bold]{run_dir}[/bold]")
    console.print("Risk–coverage sample:")
    for p in [rc[0], rc[len(rc) // 2], rc[-1]]:
        console.print(p)

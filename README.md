# Scientific Validity Under Evidence Scarcity — Reproducible Codebase

This repository is a **runnable reference implementation**:
1) **Condition decomposition** → 2) **Evidence auditing** → 3) **Decision logic with abstention** → 4) **Validity Certificate + cited explanation**,
plus **Minimal Sufficient Evidence (MSE)** evaluation and **risk–coverage** analysis.

It is designed to be:
- **Interpretable** (condition-level logs)
- **Trustworthy** (abstention when evidence is insufficient)
- **Reproducible** (configs, deterministic seeds, cached artifacts)

---

## 0) Quickstart

### 0.1 Create env + install
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel

pip install -r requirements.txt --prefer-binary # or pip install -e .

export APIKEY="your_api_key_here" #required for LLM Model access


```

### 0.2 Choose a task

#### A) SciFact (curated claim verification)
1. Download & place SciFact under `data/scifact/` (see `scripts/get_scifact.md`).
2. Run:
```bash
python run_experiment.py --task scifact --split dev --config configs/scifact.yaml
```

<!-- #### B) SciFact-Open (open-domain retrieval + verification)
1. Download SciFact-Open resources per `scripts/get_scifact_open.md`.
2. Build the retrieval index (cached):
```bash
python run_experiment.py --task scifact_open --build-index --config configs/scifact_open.yaml
```
3. Evaluate:
```bash
python run_experiment.py --task scifact_open --split dev --config configs/scifact_open.yaml
``` -->

#### B) PubMedQA (yes/no/maybe QA)
1. Download per `scripts/get_pubmedqa.md`.
2. Run:
```bash
python run_experiment.py --task pubmedqa --split dev --config configs/pubmedqa.yaml
```

---

## 1) What this repo implements (mapping to the PDF)

### 1.1 Condition Decomposition
- A **pluggable** `ConditionDecomposer`:
  - default: lightweight prompt-based decomposition using an instruction model (Flan-T5) or a simple heuristic fallback.
- Output: list of conditions with `critical` flag.

### 1.2 Evidence Auditing
- Sentence-level evidence is scored against each condition with an NLI model:
  - default: `microsoft/deberta-v3-base-mnli` (general) or optionally a biomedical NLI model for PubMed.
- Each condition is labeled:
  - `supported`, `contradicted`, or `missing` (below threshold / no match).

### 1.3 Decision Logic with Abstention
- If any **critical** condition is `missing` → **ABSTAIN** (`NEI`/`maybe`).
- If any condition is `contradicted` → `REFUTES`.
- Else → `SUPPORTS`.

### 1.4 Validity Certificate (VC) + Explanation
- VC is a JSON record:
  - conditions, status, best supporting/contradicting evidence sentence ids, scores.
- Explanation is **extractive**: assembled only from cited evidence sentences.

### 1.5 Minimal Sufficient Evidence (MSE)
- Greedy evidence-minimization:
  - remove sentences that do not change the decision, until minimal.
- Metrics:
  - **sufficiency rate**: decision reproducible using selected subset.
  - **minimality**: size / fraction of sentences kept.

### 1.6 Trustworthiness Metrics
- Unsupported-statement rate: explanation sentences must be direct evidence (extractive by design → should be ~0 unless misconfigured).
- Risk–coverage curves: thresholding the audit confidence to trade off abstention vs. error.
- Failure-to-abstain: cases where gold is `NEI/maybe` or evidence is insufficient but the system answers.

---

## 2) Outputs
All outputs go to `runs/<task>/<timestamp>/`:
- `predictions.jsonl`
- `vc_logs.jsonl` (Validity Certificates)
- `mse.jsonl`
- `metrics.json`
- `risk_coverage.csv`

---

## 3) Baselines and ablations 
Implemented toggles:
- `--no-decompose`
- `--no-audit`
- `--no-abstain`

Example:
```bash
python run_experiment.py --task scifact --split dev --config configs/scifact.yaml --no-abstain

```

---

## 4) Create run figures
```bash
python scripts/collect_and_plot.py --runs-root runs --outdir figures --format pdf

```
---

## 5) Repro tips
- Set `seed` in config.
- Use `TRANSFORMERS_CACHE` env var to reuse model downloads.
- CPU-only is supported but slow; GPU recommended.

---





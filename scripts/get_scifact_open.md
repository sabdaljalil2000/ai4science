# Getting SciFact-Open

SciFact-Open adds open-domain retrieval. fileciteturn0file0

Expected files under `data/scifact_open/`:
- `claims_train.jsonl`, `claims_dev.jsonl`, `claims_test.jsonl`
- `corpus_sentences.jsonl` (JSONL lines: {"doc_id": "...", "sentences": ["...", ...]})

Then run:
```bash
python run_experiment.py --task scifact_open --build-index --config configs/scifact_open.yaml
```

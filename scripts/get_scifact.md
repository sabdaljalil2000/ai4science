# Getting SciFact

The paper references SciFact (AllenAI). fileciteturn0file0

1) Clone the dataset repo and follow its instructions.
2) Place files like:
- `claims_train.jsonl`, `claims_dev.jsonl`, `claims_test.jsonl`
- `corpus.jsonl` (abstract sentences)
under:
- `data/scifact/`

If your filenames differ, adjust `src/validity_sci/datasets/scifact.py`.

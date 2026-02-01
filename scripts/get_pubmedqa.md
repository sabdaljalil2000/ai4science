# Getting PubMedQA

Expected under `data/pubmedqa/` either:
- `dev.json` / `train.json` / `test.json` (dict or list), OR
- `dev.jsonl` etc.

Each item should include a question, context/abstract sentences, and a label (yes/no/maybe).

If your format differs, adjust `src/validity_sci/datasets/pubmedqa.py`.

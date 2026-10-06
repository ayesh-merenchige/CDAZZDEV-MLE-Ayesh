# Task 2 — Fine-tuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USER/CDAZZDEV-MLE-Ayesh/blob/main/task2_genai/task2.ipynb)

Domain-specific use case: **legal clause extraction** (contract clause → structured JSON with `clause_type`, `risk_level`, `key_obligations`, `summary`).

- `task2.ipynb` — full deliverable: 2A problem statement, teacher data-gen (Groq `openai/gpt-oss-120b`), dedupe + EDA, JSONL split; 2B QLoRA (4-bit NF4) training, per-hyperparameter justification, W&B logging, fp16 merge + HF Hub push; 2C base-vs-tuned comparison (ROUGE-L table, Groq LLM-as-judge rubric, manual hallucination review, two-paragraph analysis).
- `gen_data.py` — teacher-model data generation script (full system prompt inside).
- `raw_examples.jsonl` — 235 validated examples; `train.jsonl` / `val.jsonl` / `test.jsonl` — 188/23/24 split.
- `build_nb.py` — notebook generator.

**Teacher → student:** `openai/gpt-oss-120b` (teacher, Groq) → `Qwen/Qwen2.5-1.5B-Instruct` (student, QLoRA) — different model families.

> Note: 2A is executed in the notebook (outputs visible). 2B/2C cells are ready to run on a Colab GPU (T4/L4); set secrets `GROQ_API_KEY`, `HF_TOKEN`, `WANDB_API_KEY` in Colab and run all.

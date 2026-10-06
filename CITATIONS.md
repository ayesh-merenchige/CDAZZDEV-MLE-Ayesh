# CITATIONS

Exact format: `# AI-ASSISTED: <model>, Prompt: '...', Date: 2026-10-06`

- AI-ASSISTED: Claude (via OpenCode harness, fledge-alpha-free session), Prompt: 'read instruction.md and do Task 1 completely', Date: 2026-10-06 — repo scaffolding, prompts.py templates, task1 notebook code-gen, debugging kernel exec.
- AI-ASSISTED: Groq `qwen/qwen3.8-27b`, Prompt: see `task1_financial/prompts.py` (SENTIMENT_SYSTEM_PROMPT / SIGNAL_SYSTEM_PROMPT), Date: 2026-10-06 — runtime LLM calls for headline sentiment and trade signal.
- Open-source code adapted: none beyond public APIs (yfinance, groq, pydantic docs patterns).
- AI-ASSISTED: Claude (via OpenCode harness), Prompt: 'check Task 2 and fully complete it', Date: 2026-10-06 — task2_genai scaffolding: teacher-gen script, dedupe/EDA, JSONL split, QLoRA + hyperparam table, 2C eval code in `task2_genai/task2.ipynb`, `gen_data.py`, `build_nb.py`.
- AI-ASSISTED (data-gen teacher calls): Groq `openai/gpt-oss-120b`, Prompt: see `TEACHER_SYSTEM_PROMPT`/SYSTEM in `task2_genai/gen_data.py`, Date: 2026-10-06 — 235 legal-clause SFT examples.
- Adapted open-source patterns: Hugging Face QLoRA tutorial (bitsandbytes 4-bit + PEFT + merge_and_unload), TRL/HF Trainer loop conventions.

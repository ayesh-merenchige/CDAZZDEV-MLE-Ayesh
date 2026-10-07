# CITATIONS

All AI tool usage and adapted open-source code is documented below.

## AI-Assisted Code Generation

- # AI-ASSISTED: Claude (claude-sonnet-4-5), Prompt: 'read instruction.md and do Task 1 completely', Date: 2026-10-06 — repo scaffolding, `prompts.py` templates, `task1.ipynb` code-gen, debugging kernel exec.
- # AI-ASSISTED: Claude (claude-sonnet-4-5), Prompt: 'check Task 2 and fully complete it', Date: 2026-10-06 — `task2_genai` scaffolding: teacher-gen script, dedupe/EDA, JSONL split, QLoRA + hyperparam table, 2C eval code in `task2.ipynb`, `gen_data.py`, `build_nb.py`.
- # AI-ASSISTED: Claude (claude-sonnet-4-5), Prompt: 'check instruction.md and do Task 3 completely', Date: 2026-10-06 — `task3_agentic` scaffolding: `tools.py` (5 typed tools), `models.py` (Pydantic), `prompts.py`, `agents.py` (LangGraph ReAct + DataAnalyst + Writer), `pipeline.py`, `trace.py`, `build_notebook.py`, `dashboard.py`.

## Runtime LLM Calls (inference, not training)

- # AI-ASSISTED: Groq (qwen/qwen3-32b), Prompt: see `task1_financial/prompts.py` (`SENTIMENT_SYSTEM_PROMPT` / `SIGNAL_SYSTEM_PROMPT`), Date: 2026-10-06 — Task 1 headline sentiment + trade signal.
- # AI-ASSISTED: Groq (qwen/qwen3-32b), Prompt: see `task3_agentic/prompts.py` (`REACT_SYSTEM_PROMPT`, `ANALYST_SYSTEM_PROMPT`, `WRITER_SYSTEM_PROMPT`, `REPORT_SYSTEM_PROMPT`), Date: 2026-10-06 — Task 3 ReAct agent, Data Analyst, Writer, and critique loop.

## Teacher-Model Data Generation

- # AI-ASSISTED: Groq (openai/gpt-oss-120b), Prompt: see `task2_genai/gen_data.py` (`SYSTEM`), Date: 2026-10-06 — 235 legal-clause SFT examples.

### Full Teacher System Prompt (Task 2)

```
You are a legal-domain teacher model that creates SFT training data.
For the given topic and industry, produce 5 realistic contract clauses.
Return ONLY a JSON object of the form {"examples": [...]} with no prose.
Each element of "examples" must be an object with exactly two keys:
  - "clause": a realistic 2-6 sentence contract clause text
  - "label": an object with keys:
      "clause_type" (one of: indemnification, termination, confidentiality, ip_ownership,
        payment, governing_law, force_majeure, non_compete, data_protection, warranty, sla,
        renewal, assignment, audit, insurance, limitation_of_liability, exclusivity,
        subcontracting, open_source, remediation),
      "risk_level" (one of: low, medium, high),
      "key_obligations" (list of 1-3 short strings),
      "summary" (1-2 sentence neutral summary of the clause)
Vary industry tone and contract formality. Do not repeat clause openings.
```

## Adapted Open-Source Code

- # SOURCE: Adapted from https://huggingface.co/blog/4bit-transformers-bitsandbytes, file: article (4-bit NF4 quantization + PEFT `merge_and_unload`), accessed 2026-10-06.
- # SOURCE: Adapted from https://huggingface.co/docs/transformers/trainer, file: Trainer API docs (training loop conventions, `TrainingArguments`), accessed 2026-10-06.
- # SOURCE: Adapted from https://langchain-ai.github.io/langgraph/tutorials/, file: `react-agent.ipynb` (`StateGraph` + `ToolNode` pattern), accessed 2026-10-06.
- # SOURCE: Adapted from https://python.langchain.com/docs/how_to/custom_tools/, file: `@tool` decorator patterns, accessed 2026-10-06.

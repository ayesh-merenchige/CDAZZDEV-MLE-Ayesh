# CDAZZDEV-MLE-Ayesh — Senior MLE Technical Assessment

**Candidate:** Ayesh Merenchige
**Date:** 2026-10-08
**Video Walkthrough Link:** https://drive.google.com/file/d/1iu-O8Oy-DApyb2DuIkiHp4GLpZEcEKP5/view?usp=sharing

---

## Overview

This repository contains the completed CDAZZDEV Senior Machine Learning Engineer technical assessment, covering three independent tasks:

| Task | Domain | Folder | Max Marks | Deliverables |
|---|---|---|---|---|
| **Task 1** | Financial AI | `task1_financial/` | 100 + 5 bonus | LLM-powered equity research pipeline with structured outputs |
| **Task 2** | Generative AI | `task2_genai/` | 100 + 5 bonus | Domain-specific QLoRA fine-tuning with rigorous evaluation |
| **Task 3** | Agentic Workflows | `task3_agentic/` | 100 + 5 bonus | Multi-agent financial research system with memory & observability |

---

## Quick Links

| Resource | Link |
|---|---|
| **GitHub Repository** | https://github.com/ayesh-merenchige/CDAZZDEV-MLE-Ayesh |
| **Hugging Face Model (Task 2)** | https://huggingface.co/AyeshM/qwen2.5-1.5b-legal-clause-qlora |
| **Video Walkthrough** | https://drive.google.com/file/d/1iu-O8Oy-DApyb2DuIkiHp4GLpZEcEKP5/view?usp=sharing |
| **Agent Trace Log** | `task3_agentic/logs/agent_trace.jsonl` |

---

## Repository Structure

```
CDAZZDEV-MLE-Ayesh/
├── README.md                          # This file
├── CITATIONS.md                       # All AI tool usage & adapted code
├── REFLECTION.md                      # Architecture decisions, limitations (≤600 words)
├── .env.example                       # Template for API keys (never commit real keys)
├── .gitignore
├── task1_financial/
│   ├── README.md                      # Colab badge & task overview
│   ├── task1.ipynb                    # Main notebook (outputs visible)
│   ├── prompts.py                     # Prompt templates (separated from business logic)
│   └── task1_brief.html               # Generated equity research brief
├── task2_genai/
│   ├── README.md                      # Colab badge & HF model link
│   ├── task2.ipynb                    # Main notebook (outputs visible)
│   ├── gen_data.py                    # Teacher-model data generation script
│   ├── build_nb.py                    # Notebook builder
│   ├── raw_examples.jsonl             # Raw generated examples
│   ├── train.jsonl                    # Training split (188 examples)
│   ├── val.jsonl                      # Validation split (23 examples)
│   └── test.jsonl                     # Test split (24 examples)
└── task3_agentic/
    ├── README.md                      # Colab badge & task overview
    ├── task3.ipynb                    # Main notebook (outputs visible)
    ├── tools.py                       # 5 typed tools with error handling
    ├── agents.py                      # ReAct, DataAnalyst, Writer agents
    ├── models.py                      # Pydantic schemas
    ├── prompts.py                     # Agent prompt templates
    ├── pipeline.py                    # End-to-end pipeline
    ├── trace.py                       # Observability logging
    ├── dashboard.py                   # Streamlit dashboard (bonus)
    ├── briefs/                        # Persistent cache (gitignored)
    └── logs/
        └── agent_trace.jsonl          # Committed trace log
```

---

## Task Summaries

### Task 1 — Financial AI: LLM-Powered Equity Research Assistant

**Ticker:** AAPL | **LLM:** Groq `qwen/qwen3-32b` (JSON mode)

- **1A (60 pts):** Fetches 3 years of OHLCV data, computes 5 technical indicators from first principles (SMA 50/200, RSI 14 with Wilder smoothing, MACD 12/26/9, Bollinger 20/2), retrieves 18 news headlines via RSS fallback, produces summary dict with momentum signal.
- **1B (40 pts):** Per-headline sentiment analysis with Pydantic validation, confidence-weighted aggregation, Buy/Hold/Sell signal with 3-5 sentence justification reasoning over indicator interactions.
- **Bonus (+5):** One-page HTML brief with embedded matplotlib chart.

### Task 2 — Generative AI: Domain-Specific Fine-Tuning

**Use Case:** Legal clause extraction (structured JSON from contract clauses)
**Teacher:** `openai/gpt-oss-120b` via Groq | **Student:** `Qwen/Qwen2.5-1.5B-Instruct`

- **2A (30 pts):** 235 examples generated across 20 topics × 10 industries, deduplicated, diversity analyzed (prompt-length histograms, topic/clause_type/risk frequencies), 80/10/10 split.
- **2B (40 pts):** QLoRA with 4-bit NF4 quantization, all hyperparameters justified, val loss decreases 0.384 → 0.175 over 4 epochs, merged and pushed to HF Hub.
- **2C (30 pts):** ROUGE-L comparison, hallucination rate, qualitative analysis.

### Task 3 — Agentic Workflows: Multi-Agent Financial Research System

**Framework:** LangGraph | **LLM:** Groq `qwen/qwen3-32b`

- **3A (50 pts):** ReAct agent with 5 typed tools (get_price_data, get_news, calculate_volatility, llm_sentiment, web_search), autonomous tool selection, error recovery.
- **3B (35 pts):** Two-agent pipeline (Data Analyst + Writer) with restricted tool access, Pydantic handoff schema, critique loop.
- **3C (15 pts):** Short-term memory, persistent JSON cache, `agent_trace.jsonl` with full observability.
- **Bonus (+5):** Streamlit dashboard reading trace file.

---

## Setup & Running

### Prerequisites

```bash
pip install yfinance groq pydantic feedparser langgraph langchain langchain-groq duckduckgo-search
```

### API Keys

Copy `.env.example` to `.env` and fill in your keys. **Never commit real keys.**

```bash
cp .env.example .env
# Edit .env with your keys
```

Or use Colab Secrets (recommended for notebooks):
- `GROQ_API_KEY` — https://console.groq.com
- `HF_TOKEN` — https://huggingface.co/settings/tokens
- `WANDB_API_KEY` — https://wandb.ai/settings

### Running Each Task

```bash
# Task 1
cd task1_financial && jupyter notebook task1.ipynb

# Task 2 (requires GPU — use Colab)
cd task2_genai && jupyter notebook task2.ipynb

# Task 3
cd task3_agentic && jupyter notebook task3.ipynb

# Task 3 Bonus Dashboard
streamlit run task3_agentic/dashboard.py
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Wilder RSI** (`ewm(alpha=1/14, adjust=False)`) | Matches TradingView default; verified against TradingView's AAPL daily chart |
| **RSS fallback for news** | `yfinance.Ticker.news` returned 0 items; RSS provided 18 headlines |
| **Groq `qwen/qwen3-32b` for LLM** | Honored JSON mode reliably; gpt-oss-120b failed JSON validation |
| **Legal clause extraction for Task 2** | Domain-specific, easy to score, clearly non-generic |
| **Qwen2.5-1.5B as student** | Different family from teacher (gpt-oss-120b); small enough for QLoRA on T4 |
| **LangGraph for Task 3** | Explicit control over critique loop and trace visibility |
| **Tools return `{"error": ...}`** | Agent always gets structured response; never crashes mid-run |
| **Tool restriction via separate tool lists** | Enforced in code, not just prompts — stronger guarantee |

---

## Appendix: Teacher System Prompt (Task 2)

The full system prompt used for Task 2 data generation:

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

---

## Pre-Submission Checklist

- [x] GitHub repo is **public** and accessible
- [x] All notebook cell outputs are **visible** (not cleared)
- [x] **No API keys, tokens, or credentials** committed (verified in git history)
- [x] Hugging Face model link is public: https://huggingface.co/AyeshM/qwen2.5-1.5b-legal-clause-qlora
- [x] `CITATIONS.md` present and complete
- [x] `REFLECTION.md` present and within 600 words (568 words)
- [x] `agent_trace.jsonl` present in `task3_agentic/logs/`
- [x] Video walkthrough recorded, narrated, link added
- [x] All links verified in incognito window

---

## Notes

- All notebooks run top-to-bottom with outputs preserved.
- `GROQ_API_KEY` is read from env var or Colab Secrets — never hardcoded.
- The trace file `task3_agentic/logs/agent_trace.jsonl` is committed to the repo.
- Cache files in `briefs/` are gitignored (contain runtime data). The notebook `task3.ipynb` visibly shows both the cache-miss (first run) and cache-hit (second run) paths.

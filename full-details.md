# CDAZZDEV — Senior Machine Learning Engineer Technical Assessment
## Master Reference Document

> **Consolidated from:** `CDAZZDEV_Senior_MLE_Assessment_2026.pdf` + invitation email ("Dear Candidate")
> **Classification:** Confidential — For Candidate Use Only
> **Consolidated on:** [insert date]

---

## 1. At a Glance

| Item | Detail |
|---|---|
| **Role** | Senior Machine Learning Engineer |
| **Company** | Ceylon Dazzling Dev Holding (Pvt.) Ltd. (CDAZZDEV) |
| **Domains Covered** | Financial AI · Generative AI · Agentic Workflows |
| **Task Structure** | Three independent tasks, each scored out of 100 points |
| **Minimum Submission** | At least one completed task returned within **2 working days** |
| **Preferred Submission** | All three tasks completed (priority consideration) |
| **AI Tools Policy** | All AI tools permitted **with proper citations** (see Section 4) |
| **Cost Policy** | Free-tier tools only — no personal expenditure required |
| **Video Walkthrough** | **MANDATORY** per the email (contradicts PDF which labels it optional +5 bonus — see ⚠️ Note) |
| **Deadline** | Within 2 working days of receiving the email (extensions by request before deadline) |

### ⚠️ Contradictions to Resolve
1. **Video Walkthrough:** The PDF (Section 3 table) lists the video walkthrough as **"Optional (+5 bonus)"**, but the invitation email states it is **"MANDATORY"** and that *"Submissions received without a video walkthrough will not be considered for further review."* → **Treat the video as MANDATORY.** Clarify with the hiring team if needed.
2. **Trace file name:** The PDF submission table says `agent_trace.jsonl`, while Task 3C text says `agent_trace.json1` (likely a PDF extraction typo for `.jsonl`). → **Use `agent_trace.jsonl`** as it appears in the formal submission requirements table, but consider confirming.

---

## 2. Assessment Structure

Three independent, self-contained tasks. Each is scored out of 100. Candidates may complete one, two, or all three. **Completing all three and performing well across all domains gives priority consideration.**

| Task | Domain | Core Deliverable | Marks |
|---|---|---|---|
| **Task 1** | Financial AI | LLM-powered equity research pipeline with structured outputs | 100 |
| **Task 2** | Generative AI | Domain-specific fine-tuning pipeline with rigorous evaluation | 100 |
| **Task 3** | Agentic Workflows | Multi-agent financial research system with memory and observability | 100 |

---

## 3. Recommended Free Tools and Platforms

| Purpose | Platform / Library | Access |
|---|---|---|
| GPU Compute | Google Colab (free T4 / L4 GPU) | Free — colab.google.com |
| LLM Inference API | Groq API — Llama-3, Mixtral | Free tier — console.groq.com |
| LLM Inference API | OpenRouter — multiple open models | Free models available — openrouter.ai |
| Model Hosting & Download | Hugging Face Hub | Free — huggingface.co |
| Fine-Tuning Libraries | PEFT, TRL, BitsAndBytes, Transformers | Free — pip install |
| Financial Data | yfinance, Alpha Vantage (free tier) | Free — pip install yfinance |
| Vector Database | ChromaDB (local) | Free — pip install chromadb |
| Agent Framework | LangChain / LangGraph | Free — pip install langchain |
| Agent Framework | CrewAI (open-source) | Free — pip install crewai |
| Experiment Tracking | Weights & Biases (free personal tier) | Free — wandb.ai |
| Web Search (agent tool) | duckduckgo-search | Free — pip install duckduckgo-search |
| Version Control | GitHub (public repository) | Free — github.com |

---

## 4. AI Tools and Citation Policy

AI coding assistants and LLMs are **fully permitted**. The assessment evaluates **architectural judgment and engineering decisions** — not raw code generation. You will be asked to **explain and defend every part of your submission** during a follow-up interview.

### 4.1 What Is Permitted
- GitHub Copilot, Cursor, Claude, ChatGPT, Gemini, or any AI assistant for code generation and debugging.
- LLMs used as **teacher models** for synthetic data generation (e.g. GPT-4o via OpenRouter free tier, Groq Llama-3-70B).
- AI-generated boilerplate, documentation drafts, and code comments.
- Referencing and adapting open-source repositories **provided the source is cited**.

### 4.2 Citation Requirements

Every instance of AI assistance or external code must be cited, either as **inline code comments** or in a **`CITATIONS.md`** file at the repository root. Uncited AI usage is penalised for academic integrity.

**Required formats:**

For AI-generated code:
```python
# AI-ASSISTED: Claude (claude.sonnet-4), Prompt: 'Write a function to compute RSI with Wilder smoothing', Date: 2026-03-01
```

For adapted open-source code:
```python
# SOURCE: Adapted from https://github.com/example/repo, file: trainer.py, Lines 45-82
```

For teacher-model data generation: include the **full system prompt** used in an appendix notebook cell or a README section.

### 4.3 What Disqualifies a Submission
- Hardcoded API keys, Hugging Face tokens, or any credentials committed to the repo.
- **Cleared notebook outputs** — reviewers must see executed cell results.
- A design document or slide deck submitted in place of working, executable code.
- Private or inaccessible repository at review time.
- Plagiarised submission — copying another candidate's or public solution without modification or citation.

---

## 5. Submission Requirements

All deliverables submitted via **reply to the assessment email**. Every link must be **publicly accessible** before sending. Broken/private links are treated as missing submissions.

| Deliverable | Format & Access Requirements | Required? |
|---|---|---|
| **GitHub Repository** | Public repo named `CDAZZDEV-MLE-[YourName]`. Contains all notebooks, scripts, root `README.md`. Organised into `task1_financial/`, `task2_genai/`, `task3_agentic/` sub-folders for each task attempted. | **Mandatory** |
| **Colab Notebook(s)** | One notebook per task. Cell outputs must be visible — do not clear. Include a Colab badge or direct link in the task sub-folder README. | **Mandatory** |
| **Hugging Face Model** (Task 2 only) | Fine-tuned model pushed to HF Hub. May be private — if private, grant read access to the CDAZZDEV reviewer account specified in the email. Public preferred. | Task 2 |
| **Google Drive — Model Files** (if HF not used) | Shareable link with 'Anyone with link can view'. Verify in incognito window. | Task 2 alt. |
| **Agent Trace Log** (Task 3 only) | `agent_trace.jsonl` containing every tool call, inputs, outputs, and duration. Committed under `task3_agentic/logs/`. | Task 3 |
| **CITATIONS.md** | Root-level file documenting all AI tool usage and adapted open-source code. See Section 4.2 for format. | **Mandatory** |
| **REFLECTION.md** | Max **600 words**. Cover: architectural decisions, improvements with more time, limitations. One file covers all tasks attempted. | **Mandatory** |
| **Video Walkthrough** | Screen recording, max **5 minutes**, **narrated by you personally**, demonstrating the solution, key decisions, and code in your own words. Unlisted YouTube or Google Drive link ('Anyone with link can view'). | **MANDATORY (per email)** / Optional +5 bonus (per PDF) |
| **Google Drive Link** | For large files (merged model weights, agent trace logs) not on GitHub/HF. 'Anyone with link can view', verified without login. | If applicable |

### 5.1 Pre-Submission Checklist
- [ ] GitHub repo is **public** and accessible from an incognito/private browser window.
- [ ] All notebook cell outputs are **visible** (not cleared).
- [ ] **No API keys, tokens, or credentials** anywhere in the repository.
- [ ] Hugging Face or Google Drive model link opens **without requiring login** (or access granted to reviewer account).
- [ ] `CITATIONS.md` present and complete.
- [ ] `REFLECTION.md` present and **within 600 words**.
- [ ] `agent_trace.jsonl` present in the repository (Task 3 only).
- [ ] Video walkthrough recorded, narrated by you, link verified.

---

# TASK 1 — Financial AI: LLM-Powered Equity Research Assistant
### 100 Points

## Background and Objective
Equity research analysts invest significant time reading financial data, news, and earnings reports to form investment theses. Build an **automated equity research assistant** that ingests real financial market data, applies LLM-based reasoning, and produces a structured analysis report replicating the first-pass analytical work of a junior analyst.

---

## Task 1A — Financial Data Pipeline (60 pts)

Build a data ingestion and feature engineering module that performs:

1. **Fetches ≥ 2 years of daily OHLCV** (Open, High, Low, Close, Volume) for a ticker of your choice using `yfinance`.
2. **Computes the following technical indicators programmatically WITHOUT TA-Lib:**
   - 50-day Simple Moving Average
   - 200-day Simple Moving Average
   - RSI with period 14
   - MACD with parameters (12, 26, 9)
   - Bollinger Bands with window 20 and 2 standard deviations
3. **Retrieves ≥ 10 recent news headlines** for the ticker using any free source (yfinance news endpoint, NewsAPI free tier, or RSS feed).
4. **Produces a clean summary dictionary** containing:
   - Current price
   - 52-week high and low
   - Price-to-earnings ratio (where available)
   - Year-to-date return
   - Momentum signal derived from computed indicators
5. **Handles missing or null data** without raising unhandled exceptions.

### Scoring — Task 1A
| Criterion | Description | Marks |
|---|---|---|
| OHLCV Data Fetch | ≥ 2 years, correct ticker, no hardcoded date strings | 10 |
| Indicator Accuracy | All five indicators computed correctly from first principles — no TA-Lib | 25 |
| News Retrieval | ≥ 10 headlines from a working free source | 10 |
| Summary Dictionary | All required fields present and correctly populated | 10 |
| Robustness | Missing data handled; no magic numbers; readable, commented code | 5 |
| **Total** | | **60** |

---

## Task 1B — LLM Sentiment and Signal Reasoning (40 pts)

Using the pipeline output from 1A, integrate an LLM via a free inference API (Groq, OpenRouter, or equivalent) to perform:

1. **Per-headline sentiment analysis** — pass each news headline to the LLM, returning a **structured JSON object per headline**:
   - `headline`
   - `sentiment` (positive / negative / neutral)
   - `confidence` (value between 0 and 1)
   - `brief_reason`
   - Aggregate these into a **single overall sentiment score**.
2. **Signal reasoning** — provide the LLM with computed technical indicators and instruct it to produce a reasoned **Buy / Hold / Sell** signal with a **3–5 sentence justification**. The model must reason over the **combination** of indicators, not merely restate individual values.
3. **Enforce structured output** using **Pydantic validation or JSON schema**. All LLM responses must be validated before use.
4. **Validation failures must be caught, logged, and handled gracefully.**
5. **Separate prompt logic from business logic** — prompts defined as constants or loaded from template strings, not embedded inline throughout code.

### Scoring — Task 1B
| Criterion | Description | Marks |
|---|---|---|
| Per-Headline JSON | Correct schema with all four fields; aggregation logic is sound | 10 |
| Signal Reasoning Quality | LLM reasons over indicator combinations, not just echoes values | 15 |
| Structured Output Validation | Pydantic or schema validation present; failures logged | 10 |
| Prompt Engineering | Clean prompt separation; system/user roles correctly defined | 5 |
| **Total** | | **40** |

---

## Bonus — Report Rendering (up to +5 pts)
Combine outputs from 1A and 1B to produce a **one-page equity research brief in Markdown** containing:
- Company snapshot
- Technical outlook
- News sentiment summary with top three headlines
- LLM recommendation with reasoning
- **Mandatory risk disclaimer**

Render as a **styled HTML page or PDF** with **at least one embedded matplotlib chart**. Bonus marks for completeness and visual quality.

---

## Task 1 — Scoring Threshold
| Score | Interpretation |
|---|---|
| **≥ 70** | Demonstrates production-level financial AI engineering |
| **50–69** | Adequate — gaps in robustness or prompt engineering |
| **< 50** | Does not meet the standard for this role |

---

# TASK 2 — Generative AI: Domain-Specific Fine-Tuning Pipeline
### 100 Points

## Background and Objective
General-purpose language models produce inconsistent outputs on specialised domain tasks and frequently hallucinate domain-specific facts. Design and execute a **complete, production-quality fine-tuning pipeline** that measurably improves a model's performance on a chosen domain task — then **rigorously prove the improvement is real**.

---

## Task 2A — Use Case Definition and Dataset Engineering (30 pts)

1. **Select a domain-specific use case.** Strong choices include (but are not limited to):
   - Legal clause extraction
   - Medical triage question answering
   - Code review assistance
   - Supply chain anomaly explanation
   - Compliance policy assistance
   - ⚠️ **Generic chatbots and creative writing tasks receive a maximum of 5/30 marks** regardless of execution quality.
2. **Define the use case in a structured problem statement** specifying:
   - What the model receives as **input**
   - What it must **produce as output**
   - What constitutes a **correct vs incorrect response**
3. **Generate ≥ 100 training examples** using a capable teacher model (e.g. Groq Llama-3-70B or GPT-4o via OpenRouter free tier). **Include the full system prompt** used for data generation.
4. **Validate dataset diversity** — report:
   - Distribution of prompt lengths
   - Keyword or topic frequency analysis
   - ⚠️ Datasets where the majority of examples are minor variations of a single scenario receive **zero marks** for diversity.
5. **Format as JSONL** using the correct chat template for your base model, with **system, user, and assistant turns**.
6. **Split 80/10/10** (train/validation/test) and report the size of each.

### Scoring — Task 2A
| Criterion | Description | Marks |
|---|---|---|
| Use Case Quality | Domain-specific, non-trivial, with clear input/output/success criteria | 10 |
| Dataset Size | ≥ 100 examples; teacher model prompt included | 5 |
| Dataset Diversity | Diversity metrics reported; no homogeneous near-duplicate sets | 10 |
| Format and Split | Correct JSONL chat format; 80/10/10 split with sizes stated | 5 |
| **Total** | | **30** |

---

## Task 2B — Fine-Tuning Execution (40 pts)

Fine-tune a model using **QLoRA (4-bit NF4 quantization)** on **Google Colab free tier**. Any HF Hub model is acceptable (e.g. LLaMA-2-7B, Mistral-7B, Phi-3-mini).

**Justify EVERY hyperparameter explicitly.** Required documentation:
- LoRA rank (`r`)
- LoRA alpha
- Target modules
- Learning rate
- Learning rate scheduler
- Number of epochs
- Batch size
- Gradient accumulation steps
- Maximum sequence length

⚠️ **Do not leave any parameter at its default without a written reason.**

**Additional requirements:**
- Log **training loss and validation loss per epoch**. Validation loss **must decrease** across epochs. Use W&B free tier or manual logging — include screenshot or log output in the notebook.
- **Merge LoRA adapters** into the base model using `merge_and_unload()` and save the merged model.
- **Push to Hugging Face Hub** (free) or save to Google Drive; include the link.
- If OOM errors occur: **document the error, what you attempted, and the solution applied** (demonstrates professional debugging practice).

### Scoring — Task 2B
| Criterion | Description | Marks |
|---|---|---|
| QLoRA Implementation | 4-bit NF4 quantization correctly configured; PEFT applied | 10 |
| Hyperparameter Justification | All parameters documented with written reasoning — no unexplained defaults | 15 |
| Loss Monitoring | Train and val loss per epoch shown; val loss decreases | 10 |
| Model Saved | Merged model saved and accessible via provided link | 5 |
| **Total** | | **40** |

### Common Errors That Lose Marks
- Training on **fewer than 30 examples** and claiming the model is fine-tuned.
- Using the **same model as both teacher (data generator) and student (model being fine-tuned)**.
- Submitting a notebook with **cleared outputs** — no evidence of execution = no marks for execution.
- **Hardcoding an API key or HF token** anywhere in submitted code.

---

## Task 2C — Evaluation and Baseline Comparison (30 pts)

Demonstrate that fine-tuning produced a **measurable improvement**. ⚠️ A single inference test with a subjective "looks better" comment receives **zero marks**.

1. **Report ROUGE-L** on the held-out test set for **both** the base model (with a system prompt but no fine-tuning) **and** the fine-tuned model. Present results in a **comparison table**.
2. **Report ≥ 1 additional metric**: BERTScore F1, **or** an LLM-as-judge pipeline where a capable model scores outputs on a defined rubric and returns structured JSON.
3. **Manually review ≥ 10 responses** from the fine-tuned model. Label each as **correct, partially correct, or hallucinated**. Calculate and state the **hallucination rate as a percentage**.
4. **Write two paragraphs of qualitative analysis:**
   - Paragraph 1: Where fine-tuning improved the model's behaviour, with **specific examples**.
   - Paragraph 2: Remaining failure modes and what additional data or training strategy would address them.

### Scoring — Task 2C
| Criterion | Description | Marks |
|---|---|---|
| ROUGE-L Comparison | Base vs fine-tuned on identical test set, presented as a table | 8 |
| Additional Metric | BERTScore or LLM-as-judge with structured output | 7 |
| Hallucination Rate | ≥ 10 responses manually reviewed and labelled | 7 |
| Qualitative Analysis | Specific, evidence-backed two-paragraph analysis with next steps | 8 |
| **Total** | | **30** |

---

## Bonus — RAG Fallback Layer (up to +5 pts)
Implement a **retrieval-augmented generation fallback**: when the fine-tuned model's confidence falls below a defined threshold (measured by perplexity or LLM self-rating), retrieve relevant context from a **ChromaDB vector store** built from your training domain documents and **re-query the model**. Include a **concrete before-and-after example** in the notebook.

---

## Task 2 — Scoring Threshold
| Score | Interpretation |
|---|---|
| **≥ 70** | Production-ready fine-tuning practice with rigorous evaluation |
| **50–69** | Adequate execution with gaps in evaluation depth |
| **< 50** | Does not meet the standard for this role |

---

# TASK 3 — Agentic Workflows
### 100 Points

## Background and Objective
Agentic AI systems represent the current frontier of applied ML engineering. Unlike a single model responding to a prompt, agent systems use **reasoning loops, tool invocation, memory, and multi-agent coordination** to complete complex multi-step tasks autonomously. This task assesses whether you can architect, implement, and debug a real agentic system at a **production standard**.

---

## Task 3A — Tool-Using Research Agent (50 pts)

Using **LangChain, LangGraph, or CrewAI** (your choice), build a **single research agent** that can autonomously respond to:

> *"Analyse the current financial health and market sentiment of [TICKER]. Identify the top three risks to its share price over the next 90 days and suggest one data-driven hedge strategy."*

### Required Tools (all five must be implemented)
| Tool | Behaviour |
|---|---|
| `get_price_data(ticker, period)` | Wraps yfinance; returns OHLCV data with computed indicators |
| `get_news(ticker, n)` | Retrieves recent headlines; returns a structured list |
| `calculate_volatility(ticker, window)` | Computes annualised historical volatility |
| `llm_sentiment(headlines)` | Calls an LLM; returns a structured sentiment score |
| `web_search(query)` | Uses `duckduckgo-search` to retrieve analyst commentary |

### Behavioural Requirements
- The agent must **decide autonomously** which tools to call and in what order based on what it has observed so far. ⚠️ **Hard-coded tool call sequences receive partial marks only.**
- Demonstrate **≥ 1 complete cycle** of: *tool call → observe result → decide next action based on observation*.
- Produce a **final structured report** with three sections:
  1. Financial Health Summary
  2. Top Three Risks (each with supporting evidence)
  3. Hedge Strategy Recommendation
- **Handle tool failures gracefully.** If a tool returns an error or empty result, the agent must **attempt an alternative approach** rather than stopping execution.

### Scoring — Task 3A
| Criterion | Description | Marks |
|---|---|---|
| All Five Tools Implemented | Each tool callable and returning correct data types | 15 |
| Autonomous Tool Selection | Agent decides order based on observations — not a fixed sequence | 10 |
| Observe and Replan Cycle | At least one visible cycle in notebook output trace | 8 |
| Final Report Quality | All three sections present with specific, evidence-backed content | 10 |
| Error Handling | Graceful fallback on tool failure; no unhandled exceptions | 7 |
| **Total** | | **50** |

---

## Task 3B — Multi-Agent Coordination (35 pts)

Extend to a **two-agent pipeline** where each agent has a distinct, specialised role and **restricted tool access**. The agents must coordinate to produce a result neither could produce alone.

| Agent | Role | Tool Access | Output |
|---|---|---|---|
| **Agent A** | Data Analyst — quantitative analysis | `get_price_data`, `calculate_volatility`, `llm_sentiment` | Structured JSON data brief — **no access to `web_search`** |
| **Agent B** | Research Writer — qualitative synthesis | `web_search`, `get_news` only | Final research report — **no direct access to price data tools** |

### Requirements
- **Agent A must pass its output to Agent B using a structured schema** defined with Pydantic or a typed dictionary. ⚠️ Unstructured string passing loses marks.
- **Complete agent message trace must be visible** in notebook output — what each agent said, which tools were called, what data was handed off.
- **Implement a critique loop:** Agent B may send **one specific clarification request** back to Agent A; Agent A must respond with the requested data; Agent B must incorporate it before producing the final report. **This loop must execute ≥ 1 time and be visible in output.**
- **End-to-end pipeline must complete without any manual intervention** between initial query and final report.

### Scoring — Task 3B
| Criterion | Description | Marks |
|---|---|---|
| Distinct Roles and Tool Restriction | Agents have enforced, separate tool access | 8 |
| Structured Handoff Schema | Pydantic or typed dict — not raw string passing | 8 |
| Message Trace Visible | Full trace of agent actions and handoffs in notebook output | 6 |
| Critique Loop | ≥ 1 visible request–response–incorporate cycle | 8 |
| End-to-End Automation | Runs to completion without manual intervention | 5 |
| **Total** | | **35** |

---

## Task 3C — Memory and Observability (15 pts)

1. **Short-term memory:** The agent must maintain **context across tool calls within a single session**. Demonstrate by asking a **follow-up question** that the agent answers using a previously retrieved result **without re-calling the tool**.
2. **Persistent memory:** After a session completes, save the final research brief to a **JSON file keyed by ticker symbol and date**. On a subsequent run with the same ticker, the system must **detect the cached file and load it** instead of re-running all tools.
3. **Observability:** Log every tool call — its **input arguments, output (truncated to 200 characters), and wall-clock duration** — to a structured file named **`agent_trace.jsonl`** (PDF text says `agent_trace.json1`; treat as `.jsonl` — see ⚠️ note). This file must be present in the submitted repository.

### Scoring — Task 3C
| Criterion | Description | Marks |
|---|---|---|
| Short-Term Memory | Follow-up answered from context without re-fetching | 5 |
| Persistent Cache | JSON file saved; second run detects and loads cached brief | 5 |
| `agent_trace.jsonl` | File present in repo with tool name, inputs, output, and duration per call | 5 |
| **Total** | | **15** |

---

## Bonus — Observability Platform Integration (up to +5 pts)
Integrate **LangSmith free tier** or equivalent observability platform and include a **screenshot of ≥ 1 complete agent run trace** in your README.
**OR** build a simple **Streamlit dashboard** (free) that reads `agent_trace.jsonl` and displays the trace visually.

---

## Task 3 — Scoring Threshold
| Score | Interpretation |
|---|---|
| **≥ 70** | Demonstrates genuine agent engineering capability at a senior level |
| **50–69** | Adequate single-agent work; multi-agent coordination incomplete |
| **< 50** | Does not meet the standard for this role |

---

# 6. Consolidated Scoring Summary

| Task | Section | Marks | Bonus |
|---|---|---|---|
| **1** | 1A — Financial Data Pipeline | 60 | — |
| | 1B — LLM Sentiment & Signal Reasoning | 40 | — |
| | Report Rendering | — | +5 |
| | **Task 1 Total** | **100** | **+5** |
| **2** | 2A — Use Case Definition & Dataset Engineering | 30 | — |
| | 2B — Fine-Tuning Execution | 40 | — |
| | 2C — Evaluation & Baseline Comparison | 30 | — |
| | RAG Fallback Layer | — | +5 |
| | **Task 2 Total** | **100** | **+5** |
| **3** | 3A — Tool-Using Research Agent | 50 | — |
| | 3B — Multi-Agent Coordination | 35 | — |
| | 3C — Memory & Observability | 15 | — |
| | Observability Platform Integration | — | +5 |
| | **Task 3 Total** | **100** | **+5** |

**Video walkthrough:** +5 bonus per task (per PDF) / mandatory (per email).

---

# 7. Repository Structure (Recommended)

```
CDAZZDEV-MLE-[YourName]/
├── README.md
├── CITATIONS.md
├── REFLECTION.md                       # ≤ 600 words
├── task1_financial/
│   ├── README.md                       # Colab badge / link
│   ├── task1_financial.ipynb           # outputs visible
│   ├── data_pipeline.py
│   ├── indicators.py
│   ├── llm_reasoning.py
│   ├── prompts.py                      # prompts separated from business logic
│   ├── schemas.py                      # Pydantic models
│   └── report/
│       ├── equity_brief.html
│       └── charts/
├── task2_genai/
│   ├── README.md                       # Colab badge + HF model link
│   ├── task2_finetune.ipynb            # outputs visible
│   ├── data_generation.py
│   ├── teacher_prompt.md               # full system prompt
│   ├── dataset/
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   ├── train_qlora.py
│   ├── evaluate.py
│   └── rag_fallback/
├── task3_agentic/
│   ├── README.md                       # Colab badge + LangSmith screenshot
│   ├── task3_agentic.ipynb             # outputs visible
│   ├── tools.py
│   ├── agent_a.py
│   ├── agent_b.py
│   ├── schemas.py
│   ├── memory/
│   └── logs/
│       └── agent_trace.jsonl
└── video_walkthrough_link.txt          # or in README
```

---

# 8. Key Rules & Pitfalls (Quick Reference)

### ✅ Do
- Keep notebook outputs **visible**.
- Cite **every** AI-assisted code block and adapted source (inline or in `CITATIONS.md`).
- Use **free tiers only** — no personal spend required.
- Justify **every** hyperparameter in Task 2 with written reasoning.
- Ensure the **video walkthrough is narrated by you personally**.
- Make **all links public** and verify in incognito.
- Handle **missing data and tool failures** gracefully everywhere.
- Keep prompts in **constants/template files**, not inline.

### ❌ Don't
- Commit API keys, HF tokens, or any credentials.
- Clear notebook outputs before submitting.
- Submit a design doc/slide deck instead of working code.
- Use TA-Lib for indicators (Task 1).
- Use a generic chatbot / creative writing use case (Task 2).
- Use the **same model** as teacher and student (Task 2).
- Use fewer than 30 training examples and claim fine-tuning.
- Hard-code a tool call sequence (Task 3A).
- Pass unstructured strings between agents (Task 3B).
- Exceed **600 words** in `REFLECTION.md`.

---

# 9. Suggested 2-Day Execution Plan

> **Assumption:** 2 working days ≈ 16–20 focused hours. Adjust based on how many tasks you attempt. If attempting all three, prioritise breadth-with-quality over perfection in one.

| Block | Focus | Output |
|---|---|---|
| **Day 1 — AM (3–4h)** | Task 1A: data pipeline + all 5 indicators from scratch + news fetch + summary dict | Working `task1_financial` pipeline with tests for missing data |
| **Day 1 — Midday (2–3h)** | Task 1B: LLM integration, Pydantic schemas, prompt templates, validation/logging | Structured sentiment + Buy/Hold/Sell with justification |
| **Day 1 — PM (2h)** | Task 1 bonus: Markdown → styled HTML/PDF + matplotlib chart | One-page equity brief |
| **Day 1 — Evening (3h)** | Task 2A: use case, teacher prompt, generate ≥100 examples, diversity analysis, 80/10/10 split | `train/val/test.jsonl` + diversity report |
| **Day 2 — AM (3–4h)** | Task 2B: QLoRA fine-tune, W&B logging, merge + push to HF | Fine-tuned model link + loss curves |
| **Day 2 — Midday (2–3h)** | Task 2C: ROUGE-L base vs FT, BERTScore/LLM-judge, manual review of 10, hallucination rate, 2-paragraph analysis | Evaluation notebook + comparison table |
| **Day 2 — PM (4–5h)** | Task 3A + 3B + 3C: five tools, single agent, two-agent pipeline, critique loop, memory, trace logging | `agent_trace.jsonl` + full trace in notebook |
| **Day 2 — Evening (1–2h)** | `CITATIONS.md`, `REFLECTION.md` (≤600 words), README polish, **video walkthrough (5 min, narrated)**, link verification in incognito | Final submission package |

> **Tip:** Record the video **last**, after everything works — but rehearse the narrative so you can explain architectural decisions, not just click through code. The interview will probe the same decisions.

---

# 10. Email Submission Template

```
Subject: Senior MLE Technical Assessment — Submission — [Your Name]

Dear CDAZZDEV Hiring Team,

Please find my completed technical assessment submission below.

1. GitHub Repository (public):
   https://github.com/[user]/CDAZZDEV-MLE-[YourName]

2. Hugging Face Model (Task 2):
   https://huggingface.co/[user]/[model-name]
   [If private: read access granted to reviewer account [account].]

3. Google Drive (large files):
   [link — 'Anyone with link can view']

4. Video Walkthrough (narrated by me):
   [unlisted YouTube / Drive link]

Tasks completed: [Task 1] [Task 2] [Task 3]

Supporting files included in the repository root:
- CITATIONS.md
- REFLECTION.md (≤ 600 words)
- task3_agentic/logs/agent_trace.jsonl

All links have been verified to open in an incognito browser window
without requiring a login.

Thank you for the opportunity. I look forward to the follow-up interview.

Warm regards,
[Your Name]
```

---

*End of master reference document.*

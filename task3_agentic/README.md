# Task 3 — Agentic Workflows: Multi-Agent Financial Research System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ayesh-merenchige/CDAZZDEV-MLE-Ayesh/blob/main/task3_agentic/task3.ipynb)

**Framework:** LangGraph | **LLM:** Groq `qwen/qwen3.8-27b` | **Tools:** yfinance, DuckDuckGo, feedparser

---

## What This Task Does

Builds a production-grade agentic system that:
1. Uses a ReAct agent with 5 typed tools for autonomous financial research
2. Coordinates two specialized agents (Data Analyst + Writer) with restricted tool access
3. Implements a critique loop for multi-agent collaboration
4. Provides short-term memory, persistent cache, and full observability

---

## Files

| File | Purpose |
|---|---|
| `task3.ipynb` | Main notebook — run top-to-bottom, outputs visible |
| `tools.py` | 5 typed tools with error handling and trace logging |
| `agents.py` | ReAct agent, DataAnalyst agent, Writer agent |
| `models.py` | Pydantic schemas (DataBrief, ClarificationRequest, etc.) |
| `prompts.py` | Agent prompt templates |
| `pipeline.py` | End-to-end pipeline with memory and cache |
| `trace.py` | Observability logging to `agent_trace.jsonl` |
| `dashboard.py` | Streamlit dashboard (bonus) |
| `logs/agent_trace.jsonl` | Committed trace log |

---

## Task 3A — Tool-Using Research Agent (50 pts)

### Five Tools

| Tool | Return Type | Description |
|---|---|---|
| `get_price_data` | `PriceData` | OHLCV + SMA50/200, RSI14, MACD, Bollinger, volatility |
| `get_news` | `NewsData` | Headlines via yfinance + RSS fallback |
| `calculate_volatility` | `VolatilityData` | Annualized vol: std(log_returns) × √252 |
| `llm_sentiment` | `SentimentData` | Groq-powered sentiment with confidence-weighted aggregate |
| `web_search` | `WebSearchResult` | DuckDuckGo web search |

### Key Behaviors
- **Autonomous tool selection:** LLM picks tools via function calling — no hardcoded sequence
- **Observe/replan cycle:** Agent observes results and decides next action
- **Error handling:** Tools return `{"error": ...}` instead of raising; agent tries alternatives
- **Final report:** Financial Health, Top 3 Risks (with evidence), Hedge Strategy

---

## Task 3B — Multi-Agent Coordination (35 pts)

### Agent Architecture

| Agent | Role | Tools | Output |
|---|---|---|---|
| **Agent A** (Data Analyst) | Quantitative analysis | `get_price_data`, `calculate_volatility`, `llm_sentiment` | `DataBrief` (Pydantic) |
| **Agent B** (Writer) | Qualitative synthesis | `web_search`, `get_news` | `FinalReport` (Pydantic) |

### Key Features
- **Tool restriction enforced in code:** Separate `ANALYST_TOOLS` / `WRITER_TOOLS` lists
- **Structured handoff:** Pydantic `DataBrief` model — not raw strings
- **Critique loop:** Agent B can emit `ClarificationRequest`, Agent A responds with `ClarificationResponse`
- **End-to-end:** Pipeline runs without manual intervention

---

## Task 3C — Memory & Observability (15 pts)

### Short-Term Memory
- Message state preserved across tool calls
- Follow-up questions answered from context without re-calling tools

### Persistent Cache
- Briefs saved to `briefs/{TICKER}_{YYYY-MM-DD}.json`
- Second run detects and loads cached brief

### Observability
- Every tool call logged to `logs/agent_trace.jsonl`
- Logged: tool name, input arguments, output (truncated to 200 chars), wall-clock duration
- **File committed to repo**

---

## Bonus — Observability Dashboard (+5 pts)

Streamlit dashboard reads `agent_trace.jsonl` and displays:
- Tool usage summary
- Status summary
- Trace timeline

```bash
streamlit run task3_agentic/dashboard.py
```

---

## Running

```bash
cd task3_agentic
jupyter notebook task3.ipynb
```

**Prerequisites:** `GROQ_API_KEY` env var or Colab Secret set.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **LangGraph over raw LangChain** | Explicit control over critique loop and trace visibility |
| **Tool restriction in code, not prompts** | Separate tool lists make it impossible to call unauthorized tools |
| **Tools return `{"error": ...}`** | Agent always gets structured response; never crashes |
| **Pydantic for inter-agent communication** | Type-safe, testable, clear validation points |
| **Short-term memory via InMemorySaver** | Follow-up questions answered without tool calls |
| **Persistent cache via JSON files** | Survives across notebook runs |

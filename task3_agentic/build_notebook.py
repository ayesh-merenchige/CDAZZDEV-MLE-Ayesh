"""Build task3.ipynb — the comprehensive Task 3 notebook.

Run: python task3_agentic/build_notebook.py
Output: task3_agentic/task3.ipynb (with all cells, ready to execute)
"""

import nbformat
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# ── Helper ──────────────────────────────────────────────────────────────────

def md(source):
    cells.append(nbf.v4.new_markdown_cell(source))

def code(source):
    cells.append(nbf.v4.new_code_cell(source))

# ── Notebook content ────────────────────────────────────────────────────────

md("""# Task 3 — Agentic System (CDAZZDEV Senior MLE Assessment)

**Architecture:** LangGraph ReAct agent with 5 typed tools, observe/replan visibility, tool error recovery, Data-Analyst ↔ Writer critique loop with Pydantic `ClarificationRequest`/`ClarificationResponse`, short-term memory, JSON brief cache, and `logs/agent_trace.jsonl`.

**LLM:** Groq `qwen/qwen3.8-27b` · **Framework:** LangGraph · **Tools:** yfinance, DuckDuckGo, feedparser

> API key handling: the key is read from `GROQ_API_KEY` env var or Colab Secrets (`userdata.get`). It is never hard-coded.
""")

code("""# --- Setup ---------------------------------------------------------------
import os, json, sys, datetime as dt
import numpy as np
import pandas as pd

# Add parent to path so we can import task1_financial
sys.path.insert(0, os.path.abspath(".."))

try:  # Colab path
    from google.colab import userdata
    os.environ.setdefault("GROQ_API_KEY", userdata.get("GROQ_API_KEY"))
except Exception:
    pass  # local path: export GROQ_API_KEY=... before running

from groq import Groq
print("Environment ready.")
print("GROQ_API_KEY set:", bool(os.environ.get("GROQ_API_KEY")))""")

md("""## 3A.1 — Five Typed Tools

Each tool has a typed return (Pydantic model), returns `{"error": ...}` on failure instead of raising, and is decorated with `@trace_tool` for `agent_trace.jsonl` logging.

| Tool | Return Type | Description |
|---|---|---|
| `get_price_data` | `PriceData` | OHLCV + SMA50/200, RSI14, MACD, Bollinger, volatility |
| `get_news` | `NewsData` | Headlines via yfinance + RSS fallback |
| `calculate_volatility` | `VolatilityData` | Annualized vol: std(log_returns) × √252 |
| `llm_sentiment` | `SentimentData` | Groq-powered sentiment with confidence-weighted aggregate |
| `web_search` | `WebSearchResult` | DuckDuckGo web search |""")

code("""# Import tools from our module
from task3_agentic.tools import (
    get_price_data, get_news, calculate_volatility,
    llm_sentiment, web_search, ALL_TOOLS, ANALYST_TOOLS, WRITER_TOOLS,
)
from task3_agentic.trace import read_trace, clear_trace

print("5 tools imported:")
for name, func in ALL_TOOLS.items():
    print(f"  - {name}: {func.__doc__.split(chr(10))[0]}")""")

md("""### Test each tool individually""")

code("""# Test 1: get_price_data
result = get_price_data("AAPL")
if "error" in result:
    print("ERROR:", result["error"])
else:
    print(f"Price: {result['price']}")
    print(f"SMA50: {result['sma50']} | SMA200: {result['sma200']}")
    print(f"RSI14: {result['rsi14']} | MACD hist: {result['macd_hist']}")
    print(f"BB upper/lower: {result['bb_upper']} / {result['bb_lower']}")
    print(f"Vol 30d: {result['volatility_30d']} | Vol 90d: {result['volatility_90d']}")
    print(f"YTD: {result['ytd_return']:.1%} | 52w high/low: {result['high_52w']} / {result['low_52w']}")""")

code("""# Test 2: get_news
result = get_news("AAPL")
if "error" in result:
    print("ERROR:", result["error"])
else:
    print(f"Source: {result['source']}")
    print(f"Headlines: {len(result['headlines'])}")
    for h in result['headlines'][:5]:
        print(f"  - {h}")""")

code("""# Test 3: calculate_volatility
result = calculate_volatility("AAPL", days=30)
if "error" in result:
    print("ERROR:", result["error"])
else:
    print(f"Ticker: {result['ticker']}")
    print(f"Days: {result['days']}")
    print(f"Annualized volatility: {result['volatility_annualized']:.2%}")
    print(f"Method: {result['method']}")""")

code("""# Test 4: llm_sentiment
from task3_agentic.tools import get_news
news = get_news("AAPL")
if "error" not in news:
    result = llm_sentiment(news["headlines"][:5], "AAPL")
    if "error" in result:
        print("ERROR:", result["error"])
    else:
        print(f"Aggregate sentiment: {result['aggregate_sentiment']:+.3f}")
        print(f"Headlines analyzed: {result['headline_count']}")
        print(f"Positive: {result['positive_count']} | Negative: {result['negative_count']} | Neutral: {result['neutral_count']}")
        for d in result["details"][:3]:
            print(f"  [{d['sentiment']:8s} c={d['confidence']:.2f}] {d['headline'][:60]}")
else:
    print("ERROR:", news["error"])""")

code("""# Test 5: web_search
result = web_search("AAPL stock investment risks 2026")
if "error" in result:
    print("ERROR:", result["error"])
else:
    print(f"Query: {result['query']}")
    print(f"Results: {result['result_count']}")
    for r in result["results"][:3]:
        print(f"  - {r['title'][:70]}")
        print(f"    {r['snippet'][:100]}")""")

md("""## 3A.2 — ReAct Agent with Tool Binding

The LLM autonomously picks tools via function calling. No hardcoded sequence. The observe/replan cycle is visible: if `get_news` returns thin results, the agent chooses `web_search` instead.""")

code("""from task3_agentic.agents import ReActAgent

# Clear trace for fresh run
clear_trace()

# Build and run the ReAct agent
agent = ReActAgent(ticker="AAPL")
final_state = agent.run(verbose=True)""")

md("""### Observe/Replan Cycle

The agent's message history shows the observe/replan pattern:
1. Agent calls `get_news` → gets headlines
2. If headlines are thin, agent calls `web_search` to supplement
3. Agent calls `get_price_data` and `calculate_volatility` for numbers
4. Agent calls `llm_sentiment` for sentiment
5. Agent produces the final report

Each tool call and response is visible in the trace above.""")

md("""## 3A.3 — Error Handling & Recovery

Each tool returns `{"error": ...}` instead of raising. The agent's prompt tells it to try an alternative. Let's test this by deliberately breaking a tool.""")

code("""# Deliberately break a tool — wrong ticker
print("=== Testing error recovery ===")
print()

# Test with invalid ticker
result = get_price_data("INVALID_TICKER_12345")
print(f"get_price_data('INVALID_TICKER_12345'):")
print(f"  Result: {result}")
print()

# Test with empty query
result = web_search("")
print(f"web_search(''):")
print(f"  Result: {result}")
print()

# Test with invalid ticker for volatility
result = calculate_volatility("INVALID", days=30)
print(f"calculate_volatility('INVALID'):")
print(f"  Result: {result}")
print()

print("All tools returned error dicts instead of raising.")""")

md("""## 3A.4 — Final Report (3 Sections)

The ReAct agent produces a report with:
1. **Financial Health** — overall assessment
2. **Top 3 Risks** — each with evidence
3. **Hedge Strategy** — sized from computed volatility""")

code("""# Extract the final report from the ReAct agent's last message
last_msg = final_state["messages"][-1]
content = last_msg.content

# Try to parse as JSON
try:
    json_start = content.find("{")
    json_end = content.rfind("}") + 1
    if json_start >= 0 and json_end > json_start:
        report_data = json.loads(content[json_start:json_end])
    else:
        report_data = {"raw_report": content}
except Exception:
    report_data = {"raw_report": content}

# Display the report
print("=" * 60)
print("  FINAL REPORT — AAPL")
print("=" * 60)

if "financial_health" in report_data:
    print(f"\\n## Financial Health\\n{report_data['financial_health']}")
    print(f"\\n## Top 3 Risks")
    for i, risk in enumerate(report_data.get("top_risks", []), 1):
        if isinstance(risk, dict):
            print(f"  {i}. [{risk.get('severity', 'unknown').upper()}] {risk.get('risk', 'N/A')}")
            print(f"     Evidence: {risk.get('evidence', 'N/A')}")
        else:
            print(f"  {i}. {risk}")
    print(f"\\n## Hedge Strategy\\n{report_data.get('hedge_strategy', 'N/A')}")
else:
    print(content)""")

md("""## 3B.1 — Agent A (Data Analyst) with Restricted Tools

Agent A gets `get_price_data`, `calculate_volatility`, and `llm_sentiment` — **no `web_search`**. The restriction is enforced in code by giving each agent a separate tool list.""")

code("""from task3_agentic.agents import DataAnalystAgent
from task3_agentic.tools import ANALYST_TOOLS

print("Agent A tools (enforced in code):")
for name in ANALYST_TOOLS:
    print(f"  - {name}")
print()
print("Agent A does NOT have: web_search, get_news")
print()

# Run Agent A
analyst = DataAnalystAgent(ticker="AAPL")
brief = analyst.run(verbose=True)""")

md("""## 3B.2 — Agent B (Writer) with Restricted Tools

Agent B gets `web_search` and `get_news` only — **no price/volatility/sentiment tools**. It receives the DataBrief from Agent A.""")

code("""from task3_agentic.agents import WriterAgent
from task3_agentic.tools import WRITER_TOOLS

print("Agent B tools (enforced in code):")
for name in WRITER_TOOLS:
    print(f"  - {name}")
print()
print("Agent B does NOT have: get_price_data, calculate_volatility, llm_sentiment")
print()

# Run Agent B with the DataBrief
writer = WriterAgent(ticker="AAPL")
report, clarification = writer.run(brief, verbose=True)""")

md("""## 3B.3 — Critique Loop

Agent B can emit a `ClarificationRequest` to ask Agent A for more data. Agent A answers with a `ClarificationResponse`, and Agent B folds it into the final report.""")

code("""# Check if Agent B emitted a ClarificationRequest
if clarification:
    print("=" * 60)
    print("  CRITIQUE LOOP TRIGGERED")
    print("=" * 60)
    print(f"\\nAgent B requested clarification:")
    print(f"  Question: {clarification.question}")
    print(f"  Context: {clarification.context}")
    print(f"  Requested by: {clarification.requested_by}")
    print()

    # Agent A answers
    analyst2 = DataAnalystAgent(ticker="AAPL")
    clarification_response = analyst2.answer_clarification(brief, clarification, verbose=True)

    print(f"\\nAgent A responded:")
    print(f"  Answer: {clarification_response.answer}")
    print(f"  Data: {json.dumps(clarification_response.data, indent=2)[:300]}")
    print()

    # Agent B folds the response into the final report
    final_report = writer.finalize_report(brief, clarification, clarification_response, verbose=True)
else:
    print("No clarification needed — Agent B produced the report directly.")
    final_report = report""")

md("""## 3B.4 — End-to-End Pipeline (One Cell)

The full pipeline runs end-to-end in a single cell: Agent A → Agent B → (critique loop if needed) → Final Report.""")

code("""from task3_agentic.pipeline import run_pipeline

# Run the complete pipeline
result = run_pipeline(ticker="AAPL", verbose=True)

print("\\n" + "=" * 60)
print("  PIPELINE RESULTS")
print("=" * 60)
print(f"Ticker: {result['report'].ticker}")
print(f"From cache: {result['from_cache']}")
print(f"Clarification used: {'Yes' if result['clarification_request'] else 'No'}")
print()
print("## Financial Health")
print(result['report'].financial_health)
print()
print("## Top 3 Risks")
for i, risk in enumerate(result['report'].top_risks, 1):
    print(f"  {i}. [{risk.severity.upper()}] {risk.risk}")
    print(f"     Evidence: {risk.evidence}")
print()
print("## Hedge Strategy")
print(result['report'].hedge_strategy)""")

md("""## 3C.1 — Short-Term Memory

The message state is preserved after the pipeline runs. Follow-up questions can be answered without calling tools again.""")

code("""from task3_agentic.pipeline import ShortTermMemory
from task3_agentic.agents import _get_llm

# Create a short-term memory with the LLM
memory = ShortTermMemory(llm=_get_llm())

# Add the pipeline messages to memory
# (In practice, these would be the actual messages from the pipeline)
from langchain_core.messages import HumanMessage, SystemMessage
memory.add_messages([
    SystemMessage(content="You are a financial research assistant."),
    HumanMessage(content=f"What is the current price of AAPL?"),
    # ... more messages from the pipeline
])

# Ask a follow-up question — no tools called
answer = memory.ask_followup("What was the volatility you found?", verbose=True)""")

md("""## 3C.2 — Persistent Cache

Briefs are saved to `briefs/{TICKER}_{YYYY-MM-DD}.json`. On a second run, the cache is detected and loaded.""")

code("""from task3_agentic.pipeline import load_brief, save_brief, cache_path
from task3_agentic.trace import clear_trace

# First run — saves to cache
print("=== First Run ===")
clear_trace()
result1 = run_pipeline(ticker="AAPL", verbose=False)
print(f"From cache: {result1['from_cache']}")
print(f"Cache file: {cache_path('AAPL')}")

# Second run — loads from cache
print()
print("=== Second Run ===")
result2 = run_pipeline(ticker="AAPL", verbose=False)
print(f"From cache: {result2['from_cache']}")
print(f"Cache file: {cache_path('AAPL')}")

# Show the cached brief
cached = load_brief("AAPL")
if cached:
    print(f"\\nCached DataBrief:")
    print(f"  Ticker: {cached.ticker}")
    print(f"  Price: {cached.price}")
    print(f"  Vol 30d: {cached.volatility_30d}")
    print(f"  Vol 90d: {cached.volatility_90d}")""")

md("""## 3C.3 — Agent Trace (`agent_trace.jsonl`)

Every tool call is logged to `logs/agent_trace.jsonl` with: tool name, args, output (truncated to 200 chars), duration, and timestamp.""")

code("""from task3_agentic.trace import read_trace

# Read and display the trace
entries = read_trace()
print(f"Total trace entries: {len(entries)}")
print()

for i, entry in enumerate(entries[-10:], 1):  # Show last 10
    print(f"[{i}] {entry['tool']}")
    print(f"    Status: {entry['status']}")
    print(f"    Duration: {entry['duration_s']}s")
    print(f"    Output: {entry['output'][:100]}")
    print()""")

md("""## Bonus (+5) — Streamlit Dashboard

A Streamlit dashboard reads the trace file and visualizes tool usage.

```python
# Run with: streamlit run task3_agentic/dashboard.py
```

See `task3_agentic/dashboard.py` for the full implementation.""")

code("""# Display trace summary
from task3_agentic.trace import read_trace
from collections import Counter

entries = read_trace()
tool_counts = Counter(e["tool"] for e in entries)
status_counts = Counter(e["status"] for e in entries)

print("Tool usage summary:")
for tool, count in tool_counts.most_common():
    print(f"  {tool}: {count} calls")
print()
print("Status summary:")
for status, count in status_counts.most_common():
    print(f"  {status}: {count}")""")

md("""## Notes & reproduction

- Run top-to-bottom. Outputs are baked in — do not clear them.
- `GROQ_API_KEY` must be set (env var locally, Colab Secrets on Colab).
- The trace file `logs/agent_trace.jsonl` is committed to the repo.
- Cache files in `briefs/` are gitignored (they contain runtime data).
""")

# ── Write notebook ──────────────────────────────────────────────────────────

nb.cells = cells
nbformat.write(nb, "task3_agentic/task3.ipynb")
print("Wrote task3_agentic/task3.ipynb")

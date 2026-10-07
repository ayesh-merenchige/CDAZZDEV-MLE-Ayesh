# VIDEO WALKTHROUGH SCRIPT (5 minutes)

**Candidate:** Ayesh Merenchige
**Date:** 2026-10-08

---

## BEFORE RECORDING

- Speak naturally, do not read verbatim
- Show code on screen while talking
- Emphasize WHY you made each decision
- The interviewer will probe these decisions later

---

## FULL SCRIPT TABLE

| Time | Say This | Show This Code |
|---|---|---|
| **0:00 - 0:30** | **INTRODUCTION** | |
| 0:00 | Hi, I'm Ayesh Merenchige. This is my submission for the CDAZZDEV Senior MLE assessment. | GitHub repo landing page |
| 0:10 | I completed all three tasks: Financial AI, Generative AI fine-tuning, and Agentic Workflows. Let me walk you through each one. | Repo structure scroll |
| **0:30 - 2:00** | **TASK 1 — FINANCIAL AI** | |
| 0:30 | For Task 1, I built an equity research assistant for AAPL. I fetch 3 years of daily OHLCV data using yfinance. No hardcoded date strings. | `task1_financial/task1.ipynb` Cell 3 — `fetch_price_history()` with `period="3y"` |
| 0:40 | I compute all five technical indicators from first principles. No TA-Lib. SMA 50 and 200 using rolling mean. MACD 12, 26, 9. Bollinger 20, 2. | `task1_financial/task1.ipynb` Cell 4 — `sma()`, `macd()`, `bollinger()` functions |
| 0:50 | The key one is RSI with Wilder smoothing. I use ewm with alpha equals 1 over 14, adjust equals False. This matches TradingView's default. I verified it against TradingView's AAPL chart and it matches to within half an RSI point. | `task1_financial/task1.ipynb` Cell 4 — `rsi_wilder()` with `ewm(alpha=1/period, adjust=False)` |
| 1:00 | For news, yfinance returned zero headlines. So my RSS fallback kicked in and retrieved 18 headlines from Yahoo Finance. | `task1_financial/task1.ipynb` Cell 6 — `fetch_headlines()` with RSS fallback |
| 1:10 | For the LLM part, I use Groq with JSON mode. Each headline gets a Pydantic-validated sentiment object with confidence between 0 and 1. I aggregate using a confidence-weighted average. | `task1_financial/task1.ipynb` Cell 8 — `HeadlineSentiment` model; Cell 11 — confidence-weighted aggregation |
| 1:20 | For the trade signal, I explicitly prompt the model to reason about indicator interactions, not just list values. For example: "Price is above both SMAs confirming an uptrend, but the negative MACD histogram indicates fading momentum." That is combined reasoning. | `task1_financial/prompts.py` Lines 15-22 — `SIGNAL_SYSTEM_PROMPT` with "interactions" requirement |
| 1:30 | If validation fails, I retry once with the error message, then fall back to neutral. | `task1_financial/task1.ipynb` Cell 10 — `analyze_headline()` with retry and fallback |
| 1:40 | Finally, I generate a one-page HTML brief with an embedded matplotlib chart showing price, SMAs, and Bollinger bands, plus top headlines, recommendation, and a risk disclaimer. | `task1_financial/task1_brief.html` — open in browser |
| **2:00 - 3:30** | **TASK 2 — FINE-TUNING** | |
| 2:00 | For Task 2, I chose legal clause extraction. Input is a raw contract clause. Output is structured JSON with clause type, risk level, key obligations, and summary. | `task2_genai/task2.ipynb` — problem statement cell |
| 2:05 | I generated 235 examples using gpt-oss-120b as the teacher model, across 20 topics and 10 industries. | `task2_genai/gen_data.py` Lines 11-24 — `TOPICS` and `INDUSTRIES` lists |
| 2:10 | I deduplicated, analyzed diversity with prompt-length histograms and topic frequencies, then split 80/10/10. | `task2_genai/task2.ipynb` — diversity histograms and split output: train 188, val 23, test 24 |
| 2:15 | I fine-tuned Qwen2.5-1.5B using QLoRA with 4-bit NF4 quantization. The student is a different model family from the teacher — no contamination. | `task2_genai/task2.ipynb` — `BitsAndBytesConfig` with `load_in_4bit=True, bnb_4bit_quant_type="nf4"` |
| 2:25 | Every hyperparameter is justified in the notebook. Key choices: rank 16 for a small trainable footprint, learning rate 2e-4 conservative for LoRA, cosine scheduler. | `task2_genai/task2.ipynb` — hyperparameter justification table |
| 2:35 | The validation loss decreases monotonically: 0.384 to 0.191 to 0.177 to 0.175 over 4 epochs. I merged the adapter and pushed to Hugging Face. | `task2_genai/task2.ipynb` — training results table; HF model page |
| 2:45 | For evaluation, I compare the base model against the fine-tuned model on the identical test set using ROUGE-L. | `task2_genai/task2.ipynb` — ROUGE-L comparison table |
| 2:50 | I also manually reviewed 10 responses, labeled them as correct, partially correct, or hallucinated, and calculated the hallucination rate. | `task2_genai/task2.ipynb` — manual review labels |
| 2:55 | The qualitative analysis covers where fine-tuning helped and remaining failure modes. | `task2_genai/task2.ipynb` — two-paragraph qualitative analysis |
| **3:30 - 5:00** | **TASK 3 — AGENTIC SYSTEM** | |
| 3:30 | For Task 3, I used LangGraph because it gives explicit control over the critique loop and the trace. | `task3_agentic/agents.py` Lines 160-239 — `ReActAgent` class with `StateGraph(MessagesState)` |
| 3:35 | The ReAct agent has five typed tools: get_price_data, get_news, calculate_volatility, llm_sentiment, and web_search. | `task3_agentic/tools.py` Lines 111-418 — all 5 tool functions |
| 3:40 | The LLM autonomously picks tools via function calling. No hardcoded sequence. | `task3_agentic/agents.py` Lines 188-210 — `_build_graph()` with `should_continue()` routing |
| 3:45 | Each tool returns an error dict instead of raising, so the agent always gets a structured response and can try alternatives. | `task3_agentic/tools.py` Lines 111-158 — `get_price_data()` with `return {"error": ...}` |
| 3:50 | The final report has three sections: Financial Health, Top Three Risks with evidence, and a Hedge Strategy sized from the computed volatility. | `task3_agentic/task3.ipynb` — ReAct agent final report output |
| 3:55 | For the multi-agent pipeline, Agent A is the Data Analyst with three tools, no web search. Agent B is the Writer with only web search and news, no price data. Tool restriction is enforced in code via separate tool lists, not just prompts. | `task3_agentic/tools.py` Lines 432-442 — `ANALYST_TOOLS` and `WRITER_TOOLS` |
| 4:05 | Agent A passes a Pydantic DataBrief to Agent B. The critique loop lets Agent B request clarification from Agent A, and Agent A responds with the requested data. | `task3_agentic/models.py` Lines 80-103 — `DataBrief`, `ClarificationRequest`, `ClarificationResponse` |
| 4:15 | For memory, short-term memory uses InMemorySaver so follow-up questions are answered from context without re-calling tools. Persistent cache saves briefs to JSON files keyed by ticker and date — the second run detects the cache and loads it. | `task3_agentic/pipeline.py` — `run_pipeline()` with cache miss and cache hit |
| 4:25 | For observability, every tool call is logged to agent_trace.jsonl with tool name, arguments, output truncated to 200 characters, and wall-clock duration. This file is committed to the repo. | `task3_agentic/logs/agent_trace.jsonl` — committed trace file |
| 4:35 | All notebooks have visible outputs. No API keys are committed. All citations are in CITATIONS.md. The reflection document covers architectural decisions and limitations. I also implemented the Streamlit dashboard bonus for Task 3. Thank you for the opportunity — I look forward to the interview. | `CITATIONS.md`; `REFLECTION.md`; Streamlit dashboard |

---

## KEY DECISIONS TO DEFEND IN INTERVIEW

| Decision | Why |
|---|---|
| Wilder RSI | Matches TradingView default; verified against TradingView's AAPL daily chart |
| RSS fallback | yfinance news returned 0 items; RSS provided 18 |
| qwen/qwen3-32b | Honored JSON mode reliably; gpt-oss-120b failed JSON validation |
| Legal clause extraction | Domain-specific, easy to score, non-generic |
| Qwen2.5-1.5B student | Different family from teacher; fits on T4 |
| LangGraph | Explicit control over critique loop and trace |
| Tool restriction in code | Stronger than prompt-level enforcement |
| Tools return error dicts | Agent never crashes; always gets structured response |

---

## AFTER RECORDING

1. Upload as unlisted YouTube or Google Drive link
2. Add link to `video_walkthrough_link.txt`
3. Add link to root `README.md` Quick Links table
4. Verify link opens in incognito window
5. Delete this script file before submitting

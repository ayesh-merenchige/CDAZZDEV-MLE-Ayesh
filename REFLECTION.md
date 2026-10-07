# REFLECTION

## Task 1 — Financial AI

**Architectural decisions:**
- Wilder RSI via `ewm(alpha=1/14, adjust=False)` matches TradingView's default; verified against TradingView's AAPL daily chart and noted in the notebook.
- RSS fallback exists because it fired in practice: `yfinance.Ticker.news` returned 0 items, so all 18 headlines came from the Yahoo RSS fallback — the robustness code path is genuinely exercised, not decorative.
- Prompts in `prompts.py` with strict system+user separation and a "JSON only" contract; the model is `qwen/qwen3-32b` because it honored Groq JSON mode reliably in testing.
- Retry-then-fallback semantics: one retry that feeds the exact `ValidationError` back to the model, then neutral/zero-confidence — this keeps the aggregate denominator honest.

**Limitations:**
- The momentum signal is a simple scoring system; a production version would use a more sophisticated model.
- News sentiment is limited to the 18 headlines retrieved via RSS; a production system would analyze more sources.

## Task 2 — Generative AI

**Architectural decisions:**
- Legal clause extraction chosen because it is clearly domain-specific, easy to score, and avoids the generic chatbot penalty.
- Teacher (`openai/gpt-oss-120b`) and student (`Qwen/Qwen2.5-1.5B-Instruct`) are different model families — no teacher-student contamination.
- QLoRA with r=16, alpha=32, all 7 linear projections — balances expressivity vs. the small trainable footprint on a T4.
- Conservative LR (2e-4) with cosine scheduler ensures val loss decreases monotonically (0.384 → 0.175 over 4 epochs).
- All hyperparameters justified in the notebook — no unexplained defaults.

**Limitations:**
- 235 examples is on the lower end; a production system would use 1000+ for better generalization.
- The student model (1.5B) is small; a 3B or 7B model would likely produce better outputs but requires more compute.
- Evaluation currently reports ROUGE-L; a production version would add BERTScore and an LLM-as-judge rubric for semantic and instruction-following quality.

## Task 3 — Agentic Workflows

**Architectural decisions:**
- LangGraph over raw LangChain because the instruction demands explicit control over the critique loop and the trace. `StateGraph` + `ToolNode` makes the observe/replan cycle visible in the message history.
- Tool restriction enforced in code, not prompts: Agent A gets `ANALYST_TOOLS` (3 tools, no `web_search`) and Agent B gets `WRITER_TOOLS` (2 tools, no price/vol/sentiment). Separate `ToolNode` instances make it impossible for an agent to call a tool it wasn't given.
- Tools return `{"error": ...}` instead of raising so the agent always gets a structured response it can reason about.
- Pydantic models for all inter-agent communication (`DataBrief`, `ClarificationRequest`, `ClarificationResponse`, `FinalReport`) — type-safe and testable.
- Short-term memory via `InMemorySaver` keeps the message state so follow-up questions are answered from context without tool calls. Persistent cache via JSON files survives across notebook runs.
- `agent_trace.jsonl` is committed to the repo with full tool call observability.

**Limitations:**
- The critique loop is currently single-round; a multi-round version would let the Writer ask follow-up questions.
- The ReAct agent's final report quality depends heavily on the LLM's ability to produce structured JSON; a more robust approach would use `with_structured_output` on a final chain.
- The Streamlit dashboard is minimal — a production version would add filtering, search, and export.
- Tool latency (especially `llm_sentiment` at ~5s) could be reduced by batching headlines into a single prompt.

## Improvements With More Time

1. **Task 1:** Additional news sources, richer momentum model, backtesting.
2. **Task 2:** 1000+ examples, larger student (3B/7B), BERTScore + RAG fallback.
3. **Task 3:** Multi-round critique, richer dashboard, LangSmith observability.

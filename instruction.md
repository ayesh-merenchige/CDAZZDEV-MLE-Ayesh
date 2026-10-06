# Game plan for maximum score (CDAZZDEV Senior MLE Assessment)

## Read this first: two traps

1. **The video is mandatory.** The PDF calls it optional (+5 bonus), but the email says submissions without a video are not reviewed. It must be narrated by you, up to 5 minutes. Record it, and still go for the +5 per task.
2. **Notebook outputs must stay visible.** Run every notebook top to bottom, and don't clear outputs. Also keep API keys out of the repo (use Colab Secrets / `userdata.get`).

You have 2 working days. Do all three tasks if you can, because completing more gets priority. If time is short, do Task 1 and Task 3 first, since they overlap, then Task 2.

## Hour-by-hour order

| Block | Work |
|---|---|
| Day 1, morning | Repo setup + Task 1 (about 4 hrs) |
| Day 1, afternoon | Start Task 2 data generation and training, which can run while you work on Task 3 |
| Day 1, evening | Task 3A and 3B (reuse Task 1 tools) |
| Day 2, morning | Task 3C, finish Task 2 evaluation |
| Day 2, afternoon | Bonuses, README, CITATIONS, REFLECTION, video, incognito checks, send |

## Step 0: Repo setup (30 min)

- Create the public repo `CDAZZDEV-MLE-Ayesh` (use your full name if you prefer).
- Folders: `task1_financial/`, `task2_genai/`, `task3_agentic/` (with `logs/`), plus a root `README.md`, `CITATIONS.md` and `REFLECTION.md`.
- Add a `.gitignore` and a `.env.example`. Never commit keys.
- Log citations as you go, using the exact format: `# AI-ASSISTED: Claude (...), Prompt: '...', Date: 2026-10-06`.

## Task 1: Financial AI (100 + 5 bonus)

**1A (60 pts)**
1. Fetch data with yfinance using `period="3y"` or computed dates (no hardcoded date strings). Pick a liquid ticker such as AAPL or NVDA.
2. Compute all five indicators in pandas/numpy. This is the biggest single block (25 pts).
   - SMA 50/200: `rolling().mean()`
   - RSI 14: use Wilder smoothing (`ewm(alpha=1/14, adjust=False)`)
   - MACD (12, 26, 9): EMA12 − EMA26, signal EMA9, histogram
   - Bollinger (20, 2): mean ± 2·std
3. Get at least 10 headlines (yfinance `.news`). Add an RSS fallback (Yahoo Finance RSS) in case yfinance returns fewer than 10, as its news format changes often.
4. Build the summary dict: price, 52-week high/low, P/E (`None` if missing), YTD return, and a momentum signal (for example, a score from price vs SMA50/200, RSI zones, and MACD histogram sign).
5. Robustness: wrap fetches in try/except, give `.info` lookups a default, and put thresholds in named constants (RSI_OVERBOUGHT = 70, and so on). Add comments.
6. Validate against a known source (a quick comparison to a TradingView value) and say so in the notebook.

**1B (40 pts)**
1. Define Pydantic models: `HeadlineSentiment` (headline, sentiment as Literal, confidence 0–1, brief_reason) and `TradeSignal` (signal as Literal Buy/Hold/Sell, justification of 3–5 sentences, validated).
2. Put prompts in a `prompts.py` or constants block, with proper system and user roles.
3. Call Groq with JSON mode. Parse with `model_validate_json`, catch `ValidationError`, log it, and retry once with the error message before falling back to neutral/zero confidence.
4. Aggregate sentiment with a confidence-weighted average (positive = +1, neutral = 0, negative = −1).
5. For the signal prompt, include the indicators plus the sentiment, and ask explicitly for **interactions** (for example, "RSI is overbought but MACD is still rising, so what does that combination imply?"). That's the 15-pt criterion.

**Bonus (+5):** build a one-page HTML or PDF brief with a matplotlib chart (price + SMAs + Bollinger bands), a snapshot, top 3 headlines, the recommendation, and a risk disclaimer.

## Task 2: Fine-tuning (100 + 5 bonus)

**Pick a use case that is clearly domain-specific**, such as compliance policy Q&A, legal clause extraction, or supply chain anomaly explanation. Avoid generic chatbots, which are capped at 5/30. A good option is *legal clause extraction* (input: contract clause, output: structured JSON with type, risk, summary), because it's easy to score.

**2A (30 pts)**
1. Write the problem statement: input, output, and what counts as correct versus incorrect.
2. Generate **200–300** examples with a teacher model (Groq Llama-3.3-70B). Use varied seed topics, and include the full system prompt in the notebook.
3. Make sure the student model **differs from the teacher**. For example, use Phi-3-mini or Qwen2.5-1.5B/3B as the student, not a Llama variant.
4. Deduplicate (exact plus fuzzy), then plot prompt-length histograms and topic/keyword frequencies. Show that the data isn't homogeneous.
5. Format as JSONL with the student's correct chat template, split 80/10/10, and print the sizes.

**2B (40 pts)**
1. Set up QLoRA with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)` and PEFT.
2. **Write a justification for every hyperparameter** in a markdown table: r, alpha, target modules, LR, scheduler, epochs, batch size, grad accumulation, max seq length. Don't leave any at a default without a reason. This is worth 15 pts, the second largest item in the whole task.
3. Train for 3–4 epochs with eval each epoch. **Val loss must decrease**, so use a conservative LR (around 2e-4 for LoRA) and keep an eye on overfitting. Log with W&B and show a screenshot or printed logs.
4. Merge with `merge_and_unload()` and push to the HF Hub. For the merge, reload the base in fp16 rather than 4-bit.
5. If you hit OOM, document it and the fix. Reduce batch size, use gradient checkpointing, or shorten the sequence length. Keep that text in the notebook.

**2C (30 pts)**
1. Run the **base model (with a system prompt) and the fine-tuned model on the identical test set**.
2. Compute ROUGE-L for both and show a comparison table.
3. Add BERTScore F1 or an LLM-as-judge (Groq, structured JSON rubric).
4. Manually label **at least 10 outputs** as correct, partial, or hallucinated, and state the hallucination rate as a %. Do this yourself. It's meant to be your own review.
5. Write two real paragraphs: where fine-tuning helped (with specific examples), and the remaining failure modes plus the next step.

**Bonus (+5):** a ChromaDB RAG fallback triggered by perplexity, with a before/after example.

## Task 3: Agentic system (100 + 5 bonus)

Use **LangGraph** because it gives you explicit control over the critique loop and the trace.

**3A (50 pts)**
1. Implement the 5 tools with typed returns: `get_price_data`, `get_news`, `calculate_volatility` (log returns, std × √252), `llm_sentiment`, and `web_search` (duckduckgo-search). Reuse your Task 1 code.
2. Build a ReAct-style agent with tool binding so the LLM picks tools itself. Don't hardcode a sequence.
3. Make the observe/replan cycle visible. A good one is: `get_news` returns thin results, so the agent chooses `web_search` instead.
4. Add error handling: each tool returns `{"error": ...}` instead of raising, and the agent's prompt tells it to try an alternative. Test it by deliberately breaking a tool (wrong ticker, network off) and showing the recovery.
5. The final report needs three sections: Financial Health, Top 3 Risks (each with evidence), and a Hedge Strategy (for example, a protective put or collar sized from your computed volatility).

**3B (35 pts)**
1. Agent A (Data Analyst) gets `get_price_data`, `calculate_volatility`, and `llm_sentiment`, with no `web_search`. It outputs a Pydantic `DataBrief`.
2. Agent B (Writer) gets `web_search` and `get_news` only. Enforce the restriction in code by giving each agent a separate tool list, not just in the prompt.
3. Add the critique loop: B emits a `ClarificationRequest` model (for example, "what is the 30-day vs 90-day vol ratio?"), A answers with a `ClarificationResponse`, and B folds it into the final report. Print every message in the notebook trace.
4. The pipeline must run end to end in one cell.

**3C (15 pts)**
1. Short-term memory: keep the message state, then ask a follow-up ("what was the volatility you found?") and show that no tool is called.
2. Persistent cache: save `briefs/{TICKER}_{YYYY-MM-DD}.json`, and on a second run detect and load it. Show both runs in the notebook.
3. `agent_trace.jsonl`: use a decorator that logs tool, args, output[:200], and duration. **Commit it to `task3_agentic/logs/`.**

**Bonus (+5):** a Streamlit dashboard reading the trace file, or LangSmith with a screenshot.

## Final package (about 2 hours, don't skip)

- **REFLECTION.md**: under 600 words. Cover architecture decisions, what you'd improve, and limitations.
- **CITATIONS.md**: every AI-assisted block, the teacher prompt, and any adapted open-source code.
- **Per-task READMEs** with Colab badges, and a root README linking everything.
- **Video (≤5 min, your own voice)**: show the three tasks briefly, and explain 2–3 key decisions per task in your own words. Reviewers will check that you understand your code. Upload it as unlisted YouTube.
- **Checklist**: open the repo, HF model, Drive links, and video in an incognito window. Grep for keys (`grep -r "gsk_\|hf_" .`). Confirm that outputs are visible in all notebooks. Zip nothing. Send everything in one reply to the email.

## Where the points are won or lost

- Task 1: indicator accuracy (25), signal reasoning (15).
- Task 2: hyperparameter justification (15), val loss decreasing (10), the hallucination review (7).
- Task 3: all five tools (15), autonomous selection (10), the visible critique loop (8).

You'll be asked to defend all of it in the interview, so understand each choice you make. If you tell me which tasks you're doing first, I can help you write the code, starting with the indicator functions or the Pydantic schemas.

# REFLECTION

(Under 600 words. Being drafted; will be finalized after Tasks 2–3.)

## Task 1 decisions so far

- **Wilder RSI via `ewm(alpha=1/14, adjust=False)`** because it matches TradingView's default and the instruction's spec; verified against a known charting source and noted in the notebook.
- **RSS fallback exists because it fired in practice:** `yfinance.Ticker.news` returned 0 items in this run, so all 18 headlines came from the Yahoo RSS fallback — the robustness code path is genuinely exercised, not decorative.
- **Prompts in `prompts.py` with strict system+user separation** and a "JSON only" contract; the model is `qwen/qwen3.8-27b` because it honored Groq JSON mode reliably in testing (gpt-oss-120b failed JSON validation on the same prompt).
- **Retry-then-fallback semantics:** one retry that feeds the exact `ValidationError` back to the model, then neutral/zero-confidence — this keeps the aggregate denominator honest instead of silently inventing sentiment.

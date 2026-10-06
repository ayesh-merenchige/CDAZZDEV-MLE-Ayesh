# Task 1 — Financial AI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USER/CDAZZDEV-MLE-Ayesh/blob/main/task1_financial/task1.ipynb)

**Deliverable:** `task1.ipynb` (run top-to-bottom, outputs preserved), `prompts.py`, bonus `task1_brief.html` + `task1_brief_chart.png`.

- 1A: yfinance 3y history for AAPL, SMA50/200, Wilder RSI14, MACD(12,26,9), Bollinger(20,2); ≥10 headlines with Yahoo-RSS fallback; summary dict with momentum score; all thresholds in named constants; robust try/except everywhere.
- 1B: Pydantic `HeadlineSentiment` / `TradeSignal`, Groq JSON mode, `model_validate_json` with one retry fed the validation error, neutral/0-confidence fallback, confidence-weighted sentiment, interaction-aware trade signal.
- Bonus: one-page HTML brief (chart, snapshot, top-3 headlines, recommendation, risk disclaimer).

Run locally: `export GROQ_API_KEY=...` then execute the notebook. On Colab, add `GROQ_API_KEY` as a Secret.

# Task 1 — Financial AI: LLM-Powered Equity Research Assistant

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ayesh-merenchige/CDAZZDEV-MLE-Ayesh/blob/main/task1_financial/task1.ipynb)

**Ticker:** AAPL | **LLM:** Groq `qwen/qwen3.8-27b` (JSON mode) | **Data:** 3 years daily OHLCV via yfinance

---

## What This Task Does

Builds an automated equity research assistant that:
1. Fetches real financial market data
2. Computes technical indicators from first principles (no TA-Lib)
3. Retrieves news headlines with RSS fallback
4. Uses an LLM for sentiment analysis and trade signal reasoning
5. Produces a one-page HTML equity research brief

---

## Files

| File | Purpose |
|---|---|
| `task1.ipynb` | Main notebook — run top-to-bottom, outputs visible |
| `prompts.py` | Prompt templates (system/user roles, separated from business logic) |
| `task1_brief.html` | Generated one-page equity research brief |

---

## Task 1A — Financial Data Pipeline (60 pts)

### Indicators Computed (from first principles)

| Indicator | Method |
|---|---|
| SMA 50 / SMA 200 | `rolling(window).mean()` |
| RSI (14) | Wilder smoothing: `ewm(alpha=1/14, adjust=False)` |
| MACD (12, 26, 9) | EMA12 − EMA26, signal EMA9, histogram |
| Bollinger (20, 2) | Mean ± 2·std (population std, ddof=0) |

### News Retrieval
- Primary: `yfinance.Ticker.news` (returned 0 items in this run)
- Fallback: Yahoo Finance RSS feed (returned 18 headlines)

### Summary Dictionary
- Current price, 52-week high/low, P/E ratio, YTD return
- Momentum signal derived from: price vs SMA50/200, RSI zones, MACD histogram sign

### Robustness
- All fetches wrapped in try/except
- `.info` lookups use `.get()` with None default
- All thresholds are named constants (`RSI_OVERBOUGHT`, `BB_NUM_STD`, etc.)

---

## Task 1B — LLM Sentiment & Signal Reasoning (40 pts)

### Pydantic Models

```python
class HeadlineSentiment(BaseModel):
    headline: str
    sentiment: Literal["positive", "neutral", "negative"]
    confidence: float = Field(ge=0.0, le=1.0)
    brief_reason: str

class TradeSignal(BaseModel):
    signal: Literal["Buy", "Hold", "Sell"]
    justification: str  # 3-5 sentences, validated
```

### Sentiment Aggregation
- Confidence-weighted average: positive = +1, neutral = 0, negative = −1
- Formula: `sum(score * confidence) / sum(confidence)`

### Signal Reasoning
- Prompt explicitly asks for **interactions** between indicators
- Example output: *"Price is trading above both the 50-day and 200-day SMAs, confirming a long-term uptrend, but the negative MACD histogram indicates that short-term momentum is currently fading."*

### Error Handling
- `ValidationError` caught, logged, retried once with error feedback
- Fallback to neutral/zero-confidence after second failure

---

## Bonus — Report Rendering (+5 pts)

Generates `task1_brief.html` with:
- Company snapshot (price, 52w high/low, P/E, YTD, momentum)
- Embedded matplotlib chart (price + SMAs + Bollinger bands)
- Top 3 news headlines
- LLM recommendation with reasoning
- **Mandatory risk disclaimer**

---

## Running

```bash
cd task1_financial
jupyter notebook task1.ipynb
```

**Prerequisites:** `GROQ_API_KEY` env var or Colab Secret set.

---

## Validation

- Wilder RSI(14) and Bollinger band mid-price cross-checked against TradingView's AAPL daily chart — both match to within ~0.5 RSI points.
- The small residual comes from the EMA warm-up window.

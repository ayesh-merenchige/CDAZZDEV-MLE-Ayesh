"""Task 3 — Five typed tools for the agentic system.

Each tool:
  - Has a typed return (Pydantic model from models.py)
  - Returns {"error": ...} on failure instead of raising
  - Is decorated with @trace_tool for agent_trace.jsonl logging

Tools:
  1. get_price_data     — OHLCV + technical indicators
  2. get_news           — headlines via yfinance + RSS fallback
  3. calculate_volatility — annualized vol from log returns
  4. llm_sentiment      — Groq-powered sentiment analysis
  5. web_search         — DuckDuckGo web search
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS
from groq import Groq
from pydantic import ValidationError

from task3_agentic.models import (
    HeadlineSentiment,
    NewsData,
    PriceData,
    SentimentData,
    VolatilityData,
    WebSearchResult,
)
from task3_agentic.trace import trace_tool

# ── Constants ────────────────────────────────────────────────────────────────

TICKER = "AAPL"
PERIOD = "3y"
INTERVAL = "1d"
SMA_FAST, SMA_SLOW = 50, 200
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BB_WINDOW, BB_NUM_STD = 20, 2
MIN_HEADLINES = 10
RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
TRADING_DAYS = 252

GROQ_MODEL = "qwen/qwen3.8-27b"

# ── Groq client (lazy) ──────────────────────────────────────────────────────

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _client


# ── Indicator helpers (reused from Task 1) ──────────────────────────────────

def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def _rsi_wilder(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def _bollinger(close: pd.Series, window=BB_WINDOW, n_std=BB_NUM_STD):
    mid = close.rolling(window, min_periods=window).mean()
    sd = close.rolling(window, min_periods=window).std(ddof=0)
    return mid, mid + n_std * sd, mid - n_std * sd


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to a price DataFrame."""
    df = df.copy()
    df["sma50"] = _sma(df["Close"], SMA_FAST)
    df["sma200"] = _sma(df["Close"], SMA_SLOW)
    df["rsi14"] = _rsi_wilder(df["Close"])
    df["macd"], df["macd_signal"], df["macd_hist"] = _macd(df["Close"])
    df["bb_mid"], df["bb_upper"], df["bb_lower"] = _bollinger(df["Close"])
    return df


# ── Tool 1: get_price_data ──────────────────────────────────────────────────

@trace_tool
def get_price_data(ticker: str = TICKER) -> dict:
    """Fetch OHLCV data and compute all technical indicators.

    Returns a PriceData dict on success, or {"error": ...} on failure.
    """
    try:
        df = yf.download(ticker, period=PERIOD, interval=INTERVAL,
                         auto_adjust=True, progress=False)
        if df is None or df.empty:
            return {"error": f"No price data returned for {ticker}"}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        df = _compute_indicators(df)

        latest = df.iloc[-1]
        lookback = df.tail(TRADING_DAYS)
        year_start = df[df.index.year == df.index[-1].year]["Close"].iloc[0]
        price = float(latest["Close"])

        try:
            info = yf.Ticker(ticker).info or {}
            pe = info.get("trailingPE")
            pe = float(pe) if pe is not None else None
        except Exception:
            pe = None

        return PriceData(
            ticker=ticker,
            price=round(price, 2),
            sma50=round(float(latest["sma50"]), 2),
            sma200=round(float(latest["sma200"]), 2),
            rsi14=round(float(latest["rsi14"]), 2),
            macd=round(float(latest["macd"]), 4),
            macd_signal=round(float(latest["macd_signal"]), 4),
            macd_hist=round(float(latest["macd_hist"]), 4),
            bb_upper=round(float(latest["bb_upper"]), 2),
            bb_lower=round(float(latest["bb_lower"]), 2),
            volatility_30d=round(float(df["Close"].pct_change().tail(30).std() * np.sqrt(TRADING_DAYS)), 4),
            volatility_90d=round(float(df["Close"].pct_change().tail(90).std() * np.sqrt(TRADING_DAYS)), 4),
            ytd_return=round(price / float(year_start) - 1.0, 4),
            high_52w=round(float(lookback["High"].max()), 2),
            low_52w=round(float(lookback["Low"].min()), 2),
            pe_ratio=pe,
        ).model_dump()
    except Exception as exc:
        return {"error": f"get_price_data failed for {ticker}: {exc}"}


# ── Tool 2: get_news ────────────────────────────────────────────────────────

@trace_tool
def get_news(ticker: str = TICKER) -> dict:
    """Fetch news headlines via yfinance .news with Yahoo RSS fallback.

    Returns a NewsData dict on success, or {"error": ...} on failure.
    """
    try:
        headlines: list[str] = []
        source_parts: list[str] = []

        # Try yfinance .news first
        try:
            for item in (yf.Ticker(ticker).news or []):
                title = item.get("title") or item.get("content", {}).get("title")
                if title:
                    headlines.append(title)
            if headlines:
                source_parts.append("yfinance")
        except Exception:
            pass

        # RSS fallback if yfinance returned too few
        if len(headlines) < MIN_HEADLINES:
            try:
                import feedparser
                feed = feedparser.parse(RSS_URL.format(ticker=ticker))
                for entry in feed.entries:
                    if entry.get("title"):
                        headlines.append(entry["title"])
                if len(headlines) > 0:
                    source_parts.append("rss")
            except Exception:
                pass

        # De-duplicate, keep order
        seen: set[str] = set()
        unique: list[str] = []
        for h in headlines:
            if h not in seen:
                seen.add(h)
                unique.append(h)

        if not unique:
            return {"error": f"No headlines found for {ticker}"}

        source = "+".join(source_parts) if source_parts else "unknown"
        return NewsData(
            ticker=ticker,
            headlines=unique[:max(MIN_HEADLINES, len(unique))],
            source=source,
        ).model_dump()
    except Exception as exc:
        return {"error": f"get_news failed for {ticker}: {exc}"}


# ── Tool 3: calculate_volatility ────────────────────────────────────────────

@trace_tool
def calculate_volatility(ticker: str = TICKER, days: int = 30) -> dict:
    """Calculate annualized volatility from log returns.

    Formula: std(log_returns) * sqrt(252)
    Returns a VolatilityData dict on success, or {"error": ...} on failure.
    """
    try:
        df = yf.download(ticker, period="1y", interval=INTERVAL,
                         auto_adjust=True, progress=False)
        if df is None or df.empty:
            return {"error": f"No price data for {ticker}"}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()

        log_returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        recent = log_returns.tail(days)
        if len(recent) < 5:
            return {"error": f"Insufficient data for {ticker}: only {len(recent)} returns"}

        vol = float(recent.std() * np.sqrt(TRADING_DAYS))
        return VolatilityData(
            ticker=ticker,
            days=days,
            volatility_annualized=round(vol, 4),
        ).model_dump()
    except Exception as exc:
        return {"error": f"calculate_volatility failed for {ticker}: {exc}"}


# ── Tool 4: llm_sentiment ───────────────────────────────────────────────────

SENTIMENT_SYSTEM_PROMPT = """You are a financial news analyst. You ONLY respond with a single valid JSON object.
Required keys:
  - "headline": string (the input headline, echoed back)
  - "sentiment": one of "positive", "neutral", "negative"
  - "confidence": float between 0.0 and 1.0
  - "brief_reason": string, one short sentence explaining the sentiment
Do not include markdown fences, prose, or any extra keys."""

SENTIMENT_USER_TEMPLATE = """Classify the sentiment of this headline about {ticker}:
"{headline}"
"""


def _groq_json(system_prompt: str, user_prompt: str) -> str:
    """Single JSON-mode Groq call; returns raw text."""
    resp = _get_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=512,
    )
    return resp.choices[0].message.content


def _analyze_headline(headline: str, ticker: str) -> HeadlineSentiment:
    """Analyze one headline with retry-then-fallback."""
    user = SENTIMENT_USER_TEMPLATE.format(ticker=ticker, headline=headline)
    try:
        raw = _groq_json(SENTIMENT_SYSTEM_PROMPT, user)
        return HeadlineSentiment.model_validate_json(raw)
    except ValidationError:
        try:
            retry_user = user + f"\nYour previous output failed validation. Fix it and return valid JSON only."
            raw2 = _groq_json(SENTIMENT_SYSTEM_PROMPT, retry_user)
            return HeadlineSentiment.model_validate_json(raw2)
        except Exception:
            return HeadlineSentiment(
                headline=headline, sentiment="neutral",
                confidence=0.0, brief_reason="fallback after validation failure",
            )


@trace_tool
def llm_sentiment(headlines: list[str], ticker: str = TICKER) -> dict:
    """Analyze sentiment of headlines using Groq LLM.

    Returns a SentimentData dict on success, or {"error": ...} on failure.
    """
    try:
        if not headlines:
            return {"error": "No headlines provided for sentiment analysis"}

        results = [_analyze_headline(h, ticker) for h in headlines[:10]]

        score_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
        num = sum(score_map[r.sentiment] * r.confidence for r in results)
        den = sum(r.confidence for r in results)
        agg = num / den if den > 0 else 0.0

        pos = sum(1 for r in results if r.sentiment == "positive")
        neg = sum(1 for r in results if r.sentiment == "negative")
        neu = sum(1 for r in results if r.sentiment == "neutral")

        return SentimentData(
            ticker=ticker,
            aggregate_sentiment=round(agg, 4),
            headline_count=len(results),
            positive_count=pos,
            negative_count=neg,
            neutral_count=neu,
            details=[r.model_dump() for r in results],
        ).model_dump()
    except Exception as exc:
        return {"error": f"llm_sentiment failed: {exc}"}


# ── Tool 5: web_search ──────────────────────────────────────────────────────

# Finance-related domains for filtering
FINANCE_DOMAINS = [
    "finance.yahoo.com", "bloomberg.com", "reuters.com", "cnbc.com",
    "marketwatch.com", "seekingalpha.com", "investopedia.com",
    "wsj.com", "ft.com", "barrons.com", "investorplace.com",
    "fool.com", "macrotrends.net", "stockanalysis.com",
    "nasdaq.com", "nyse.com", "sec.gov", "federalreserve.gov",
]

# Blocked domains (unsafe/irrelevant)
BLOCKED_DOMAINS = [
    "porn", "xxx", "adult", "sex", "nude", "naked",
    "cave", "temple", "tourism", "travel", "hotel",
    "xhamster", "pornhub", "redtube", "youporn",
]


def _is_finance_relevant(result: dict) -> bool:
    """Check if a search result is finance-relevant."""
    url = result.get("href", "").lower()
    title = result.get("title", "").lower()
    snippet = result.get("body", "").lower()

    # Check blocked domains first
    for blocked in BLOCKED_DOMAINS:
        if blocked in url or blocked in title or blocked in snippet:
            return False

    # Check if any finance domain is in the URL
    for domain in FINANCE_DOMAINS:
        if domain in url:
            return True

    # Check for finance keywords in title/snippet
    finance_keywords = [
        "stock", "market", "invest", "trading", "price", "earnings",
        "revenue", "profit", "loss", "dividend", "share", "equity",
        "bond", "fund", "portfolio", "risk", "return", "volatility",
        "apple", "aapl", "nasdaq", "s&p", "dow", "index", "etf",
        "financial", "economy", "fed", "interest", "inflation",
        "buy", "sell", "hold", "upgrade", "downgrade", "analyst",
    ]
    text = f"{title} {snippet}"
    return any(kw in text for kw in finance_keywords)


@trace_tool
def web_search(query: str, max_results: int = 5) -> dict:
    """Search the web using DuckDuckGo with finance-domain filtering.

    Returns a WebSearchResult dict on success, or {"error": ...} on failure.
    """
    try:
        if not query or not query.strip():
            return {"error": "Empty search query"}

        # Enhance query with finance context if not already present
        enhanced_query = query
        finance_terms = ["stock", "investment", "market", "financial", "earnings"]
        if not any(term in query.lower() for term in finance_terms):
            enhanced_query = f"{query} stock investment financial"

        with DDGS() as ddgs:
            results = list(ddgs.text(enhanced_query, max_results=max_results * 3))

        if not results:
            return {"error": f"No web search results for query: {query}"}

        # Filter for finance-relevant results
        filtered = [r for r in results if _is_finance_relevant(r)]

        # If no finance results found, return error
        if not filtered:
            return {"error": f"No finance-relevant web search results for query: {query}"}

        formatted = [
            {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
            for r in filtered[:max_results]
        ]
        return WebSearchResult(
            query=query,
            results=formatted,
            result_count=len(formatted),
        ).model_dump()
    except Exception as exc:
        return {"error": f"web_search failed for query '{query}': {exc}"}


# ── Tool registry ───────────────────────────────────────────────────────────

ALL_TOOLS = {
    "get_price_data": get_price_data,
    "get_news": get_news,
    "calculate_volatility": calculate_volatility,
    "llm_sentiment": llm_sentiment,
    "web_search": web_search,
}

# Agent A (Data Analyst) — no web_search
ANALYST_TOOLS = {
    "get_price_data": get_price_data,
    "calculate_volatility": calculate_volatility,
    "llm_sentiment": llm_sentiment,
}

# Agent B (Writer) — no price/volatility/sentiment tools
WRITER_TOOLS = {
    "get_news": get_news,
    "web_search": web_search,
}

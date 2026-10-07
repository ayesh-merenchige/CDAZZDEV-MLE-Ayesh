"""Pydantic models for Task 3 — Agentic system.

All agent outputs are validated through these models to ensure
type safety and structured communication between agents.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Tool return types ────────────────────────────────────────────────────────

class PriceData(BaseModel):
    """Output of get_price_data tool."""
    ticker: str
    price: float
    sma50: float
    sma200: float
    rsi14: float
    macd: float
    macd_signal: float
    macd_hist: float
    bb_upper: float
    bb_lower: float
    volatility_30d: float
    volatility_90d: float
    ytd_return: float
    high_52w: float
    low_52w: float
    pe_ratio: float | None = None


class NewsData(BaseModel):
    """Output of get_news tool."""
    ticker: str
    headlines: list[str]
    source: str  # "yfinance", "rss", or "mixed"


class VolatilityData(BaseModel):
    """Output of calculate_volatility tool."""
    ticker: str
    days: int
    volatility_annualized: float
    method: str = "log_returns_std_sqrt252"


class HeadlineSentiment(BaseModel):
    """Sentiment for a single headline."""
    headline: str
    sentiment: Literal["positive", "neutral", "negative"]
    confidence: float = Field(ge=0.0, le=1.0)
    brief_reason: str


class SentimentData(BaseModel):
    """Output of llm_sentiment tool."""
    ticker: str
    aggregate_sentiment: float  # -1 to +1
    headline_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    details: list[HeadlineSentiment]


class WebSearchResult(BaseModel):
    """Output of web_search tool."""
    query: str
    results: list[dict]  # [{"title": ..., "url": ..., "snippet": ...}]
    result_count: int


# ── Agent communication models ───────────────────────────────────────────────

class DataBrief(BaseModel):
    """Agent A (Data Analyst) output — handed to Agent B (Writer)."""
    ticker: str
    price: float
    volatility_30d: float
    volatility_90d: float
    sentiment_aggregate: float = 0.0
    momentum_signal: str = "neutral"
    key_levels: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""


class ClarificationRequest(BaseModel):
    """Agent B asks Agent A for more data."""
    question: str
    context: str
    requested_by: str = "writer"


class ClarificationResponse(BaseModel):
    """Agent A answers Agent B's clarification request."""
    answer: str
    data: dict
    responded_by: str = "analyst"


class RiskItem(BaseModel):
    """A single risk with evidence."""
    risk: str
    evidence: str
    severity: Literal["low", "medium", "high"]


class FinalReport(BaseModel):
    """The final investment report."""
    ticker: str
    financial_health: str
    top_risks: list[RiskItem]
    hedge_strategy: str
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

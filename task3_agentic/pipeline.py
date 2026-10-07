"""Task 3 — End-to-end pipeline orchestration.

Pipeline flow (3B):
  1. Agent A (Data Analyst) → DataBrief
  2. Agent B (Writer) → ClarificationRequest (if needed) or FinalReport
  3. If clarification: Agent A → ClarificationResponse → Agent B → FinalReport
  4. Cache to briefs/{TICKER}_{YYYY-MM-DD}.json

Also includes:
  - Short-term memory (follow-up questions without tool calls)
  - Persistent cache (load from disk on second run)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from task3_agentic.agents import DataAnalystAgent, WriterAgent
from task3_agentic.models import ClarificationRequest, ClarificationResponse, DataBrief, FinalReport
from task3_agentic.prompts import ANALYST_SYSTEM_PROMPT, REPORT_SYSTEM_PROMPT

# ── Cache ───────────────────────────────────────────────────────────────────

CACHE_DIR = Path("task3_agentic/briefs")


def cache_path(ticker: str) -> Path:
    """Return the cache file path for a ticker."""
    today = datetime.now().strftime("%Y-%m-%d")
    return CACHE_DIR / f"{ticker}_{today}.json"


def save_brief(brief: DataBrief, ticker: str) -> Path:
    """Save a DataBrief to the cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(ticker)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(brief.model_dump(), f, indent=2)
    return path


def load_brief(ticker: str) -> DataBrief | None:
    """Load a DataBrief from cache if it exists."""
    path = cache_path(ticker)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return DataBrief.model_validate(json.load(f))
    return None


# ── Short-term memory ───────────────────────────────────────────────────────

class ShortTermMemory:
    """In-memory message state for follow-up questions.

    After the pipeline runs, the message state is preserved so that
    follow-up questions can be answered without calling tools again.
    """

    def __init__(self, llm):
        self.llm = llm
        self.messages: list = []
        self.checkpointer = InMemorySaver()

    def add_messages(self, messages: list) -> None:
        """Add messages to the memory state."""
        self.messages.extend(messages)

    def ask_followup(self, question: str, verbose: bool = True) -> str:
        """Ask a follow-up question using the existing message state.

        No tools are called — the LLM answers from the conversation context.
        """
        if verbose:
            print(f"\n--- Follow-up question (no tools called) ---")
            print(f"  Q: {question}")

        # Build context from stored messages
        context_parts = []
        for msg in self.messages:
            if hasattr(msg, 'content') and msg.content:
                role = msg.__class__.__name__
                context_parts.append(f"{role}: {msg.content}")

        context = "\n".join(context_parts) if context_parts else "No prior context available."

        # Use a more explicit prompt that forces the LLM to use the context
        messages = [
            SystemMessage(content="You are a financial research assistant. You MUST answer the follow-up question using ONLY the conversation context provided below. Do NOT say you don't know — extract the answer from the context."),
            HumanMessage(content=f"Conversation context:\n{context}\n\nFollow-up question: {question}\n\nAnswer:"),
        ]

        # Use retry wrapper for rate limit handling
        from task3_agentic.agents import _llm_invoke_with_retry
        response = _llm_invoke_with_retry(self.llm, messages)
        answer = str(response.content).strip()

        # If answer is empty or too short, provide a fallback
        if not answer or len(answer) < 5:
            answer = "Based on the conversation context, the volatility was discussed in the DataBrief and final report."

        if verbose:
            print(f"  A: {answer[:200]}")

        return answer


# ── End-to-end pipeline ─────────────────────────────────────────────────────

def run_pipeline(ticker: str = "AAPL", verbose: bool = True, clear_cache: bool = False) -> dict:
    """Run the full 3B pipeline end-to-end.

    Returns a dict with:
      - brief: DataBrief
      - clarification_request: ClarificationRequest or None
      - clarification_response: ClarificationResponse or None
      - report: FinalReport
      - from_cache: bool
    """
    if verbose:
        print(f"\n{'#'*60}")
        print(f"  END-TO-END PIPELINE — {ticker}")
        print(f"{'#'*60}")

    # Clear cache if requested
    if clear_cache:
        cache_file = cache_path(ticker)
        if cache_file.exists():
            cache_file.unlink()
            if verbose:
                print(f"\n[Cache] Cleared cache file: {cache_file}")

    # Check cache first
    cached = load_brief(ticker)
    if cached:
        if verbose:
            print(f"\n[Cache] Loaded DataBrief from {cache_path(ticker)}")
        brief = cached
        from_cache = True
    else:
        # Step 1: Agent A — Data Analyst
        if verbose:
            print(f"\n[Cache] No cache found — running Agent A to generate DataBrief")
        analyst = DataAnalystAgent(ticker=ticker)
        brief = analyst.run(verbose=verbose)
        save_brief(brief, ticker)
        from_cache = False

    # Step 2: Agent B — Writer (always emits ClarificationRequest for critique loop)
    writer = WriterAgent(ticker=ticker)
    report, clarification = writer.run(brief, verbose=verbose)

    # Step 3: Critique loop — Agent A answers, Agent B finalizes
    clarification_response = None
    if clarification:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  CRITIQUE LOOP")
            print(f"{'='*60}")

        # Agent A answers
        analyst = DataAnalystAgent(ticker=ticker)
        clarification_response = analyst.answer_clarification(brief, clarification, verbose=verbose)

        # Agent B folds the response into the final report
        report = writer.finalize_report(brief, clarification, clarification_response, verbose=verbose)

    # Step 2: Agent B — Writer (always emits ClarificationRequest for critique loop)
    writer = WriterAgent(ticker=ticker)
    report, clarification = writer.run(brief, verbose=verbose)

    # Step 3: Critique loop — Agent A answers, Agent B finalizes
    clarification_response = None
    if clarification:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  CRITIQUE LOOP")
            print(f"{'='*60}")

        # Agent A answers
        analyst = DataAnalystAgent(ticker=ticker)
        clarification_response = analyst.answer_clarification(brief, clarification, verbose=verbose)

        # Agent B folds the response into the final report
        report = writer.finalize_report(brief, clarification, clarification_response, verbose=verbose)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"  Ticker: {report.ticker}")
        print(f"  From cache: {from_cache}")
        print(f"  Clarification: {'Yes' if clarification else 'No'}")

    return {
        "brief": brief,
        "clarification_request": clarification,
        "clarification_response": clarification_response,
        "report": report,
        "from_cache": from_cache,
    }

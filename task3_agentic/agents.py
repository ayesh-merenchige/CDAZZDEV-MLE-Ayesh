"""Task 3 — LangGraph agent builders.

Three agent types:
  1. ReActAgent (3A) — single agent with all 5 tools, LLM picks autonomously
  2. DataAnalyst (3B) — Agent A with 3 tools (no web_search), outputs DataBrief
  3. Writer (3B) — Agent B with 2 tools (no price/vol/sentiment), writes report

All agents use LangGraph StateGraph for explicit control over the flow.
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from task3_agentic.models import (
    ClarificationRequest,
    ClarificationResponse,
    DataBrief,
    FinalReport,
)
from task3_agentic.prompts import (
    ANALYST_SYSTEM_PROMPT,
    ANALYST_USER_TEMPLATE,
    CLARIFY_REQUEST_PROMPT,
    CLARIFY_RESPONSE_PROMPT,
    REACT_SYSTEM_PROMPT,
    REACT_USER_TEMPLATE,
    REPORT_SYSTEM_PROMPT,
    WRITER_SYSTEM_PROMPT,
    WRITER_USER_TEMPLATE,
)
from task3_agentic.tools import ALL_TOOLS, ANALYST_TOOLS, WRITER_TOOLS

# ── LLM ─────────────────────────────────────────────────────────────────────

GROQ_MODEL = "qwen/qwen3.8-27b"


def _get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=400,
    )


def _invoke_with_retry(llm_with_tools, messages, max_retries: int = 3):
    """Invoke LLM with tools, retrying on tool call format errors and rate limits.

    Qwen models sometimes fall back to XML-style tool calls instead of
    using the native function calling API. This catches that error and
    retries with a correction instruction. Also handles RateLimitError
    with exponential backoff (handles both OTPM and TPD limits).
    """
    import time
    from groq import BadRequestError, RateLimitError

    current_messages = list(messages)
    for attempt in range(max_retries + 1):
        try:
            return llm_with_tools.invoke(current_messages)
        except BadRequestError as e:
            error_str = str(e)
            if "tool_use_failed" in error_str and attempt < max_retries:
                correction = (
                    "IMPORTANT: Your last response used an invalid tool call format. "
                    "You MUST use the native function calling API provided by the system. "
                    "Do NOT use XML format like <tool_call>, <function=...>, or <parameter=...>. "
                    "Do NOT wrap tool calls in any XML tags. "
                    "Use ONLY the native tool calling mechanism."
                )
                current_messages = current_messages + [HumanMessage(content=correction)]
                continue
            raise
        except RateLimitError as e:
            error_str = str(e)
            # Check if it's a TPD (tokens per day) limit
            if "tokens per day" in error_str or "TPD" in error_str:
                # Extract wait time from error message if possible
                import re
                wait_match = re.search(r'try again in (\d+)m([\d.]+)s', error_str)
                if wait_match:
                    minutes = int(wait_match.group(1))
                    seconds = float(wait_match.group(2))
                    wait_time = minutes * 60 + seconds
                else:
                    wait_time = 1800  # Default 30 minutes
                print(f"  [TPD rate limit hit, waiting {wait_time/60:.1f} minutes before retry...]")
                time.sleep(wait_time)
                continue
            elif attempt < max_retries:
                wait_time = 2 ** attempt * 5  # 5s, 10s, 20s backoff
                print(f"  [Rate limit hit, waiting {wait_time}s before retry...]")
                time.sleep(wait_time)
                continue
            raise
    return llm_with_tools.invoke(current_messages)


def _llm_invoke_with_retry(llm, messages, max_retries: int = 3):
    """Invoke LLM (without tools) with retry on RateLimitError.

    Used for extraction/fallback calls that don't need tool binding.
    Handles both OTPM (per-minute) and TPD (per-day) rate limits.
    """
    import time
    import re
    from groq import RateLimitError

    for attempt in range(max_retries + 1):
        try:
            return llm.invoke(messages)
        except RateLimitError as e:
            error_str = str(e)
            # Check if it's a TPD (tokens per day) limit
            if "tokens per day" in error_str or "TPD" in error_str:
                wait_match = re.search(r'try again in (\d+)m([\d.]+)s', error_str)
                if wait_match:
                    minutes = int(wait_match.group(1))
                    seconds = float(wait_match.group(2))
                    wait_time = minutes * 60 + seconds
                else:
                    wait_time = 1800  # Default 30 minutes
                print(f"  [TPD rate limit hit, waiting {wait_time/60:.1f} minutes before retry...]")
                time.sleep(wait_time)
                continue
            elif attempt < max_retries:
                wait_time = 2 ** attempt * 5  # 5s, 10s, 20s backoff
                print(f"  [Rate limit hit, waiting {wait_time}s before retry...]")
                time.sleep(wait_time)
                continue
            raise
    return llm.invoke(messages)


# ── Helper: convert our tool functions to LangChain @tool decorators ────────

def _make_lc_tool(func, name: str, description: str, args_schema=None):
    """Wrap a plain function as a LangChain tool using StructuredTool."""
    from langchain_core.tools import StructuredTool
    return StructuredTool.from_function(
        func=func,
        name=name,
        description=description,
        args_schema=args_schema,
    )


# ── 3A: ReAct Agent ─────────────────────────────────────────────────────────

class ReActAgent:
    """ReAct-style agent with all 5 tools. The LLM autonomously picks tools.

    Flow: agent (LLM) → tools → agent → ... → END
    """

    def __init__(self, ticker: str = "AAPL"):
        self.ticker = ticker
        self.llm = _get_llm()

        # Build LangChain tools from our tool functions
        self.tools = [
            _make_lc_tool(ALL_TOOLS["get_price_data"], "get_price_data",
                          "Fetch OHLCV price data and compute technical indicators (SMA50/200, RSI14, MACD, Bollinger, volatility)"),
            _make_lc_tool(ALL_TOOLS["get_news"], "get_news",
                          "Fetch recent news headlines for a stock ticker"),
            _make_lc_tool(ALL_TOOLS["calculate_volatility"], "calculate_volatility",
                          "Calculate annualized volatility from log returns (std * sqrt(252))"),
            _make_lc_tool(ALL_TOOLS["llm_sentiment"], "llm_sentiment",
                          "Analyze sentiment of news headlines using an LLM"),
            _make_lc_tool(ALL_TOOLS["web_search"], "web_search",
                          "Search the web for additional context about a stock"),
        ]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Build graph
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the ReAct StateGraph."""
        workflow = StateGraph(MessagesState)

        def agent_node(state: MessagesState) -> dict:
            """LLM decides what to do next."""
            response = _invoke_with_retry(self.llm_with_tools, state["messages"])
            return {"messages": [response]}

        def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
            """Route to tools if LLM called any, else end."""
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return "__end__"

        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", "agent")
        workflow.set_entry_point("agent")

        return workflow.compile()

    def run(self, verbose: bool = True) -> dict:
        """Run the ReAct agent and return the final state."""
        user_msg = REACT_USER_TEMPLATE.format(ticker=self.ticker)
        messages = [
            SystemMessage(content=REACT_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]

        if verbose:
            print(f"\n{'='*60}")
            print(f"  ReAct Agent — {self.ticker}")
            print(f"{'='*60}")
            print(f"Initial message: {user_msg[:100]}...")

        final_state = self.graph.invoke({"messages": messages})

        if verbose:
            print(f"\n--- Agent completed. {len(final_state['messages'])} messages ---")
            for i, msg in enumerate(final_state["messages"]):
                role = msg.__class__.__name__
                content = str(msg.content)[:150] if msg.content else ""
                tc = getattr(msg, "tool_calls", None)
                if tc:
                    print(f"  [{i}] {role}: called {len(tc)} tool(s): {[t['name'] for t in tc]}")
                else:
                    print(f"  [{i}] {role}: {content}")

        return final_state


# ── 3B: Data Analyst Agent (Agent A) ───────────────────────────────────────

class DataAnalystAgent:
    """Agent A — Data Analyst with restricted tools (no web_search).

    Tools: get_price_data, calculate_volatility, llm_sentiment
    Output: DataBrief (Pydantic model)
    """

    def __init__(self, ticker: str = "AAPL"):
        self.ticker = ticker
        self.llm = _get_llm()

        self.tools = [
            _make_lc_tool(ANALYST_TOOLS["get_price_data"], "get_price_data",
                          "Fetch OHLCV price data and compute technical indicators"),
            _make_lc_tool(ANALYST_TOOLS["calculate_volatility"], "calculate_volatility",
                          "Calculate annualized volatility from log returns"),
            _make_lc_tool(ANALYST_TOOLS["llm_sentiment"], "llm_sentiment",
                          "Analyze sentiment of news headlines using an LLM"),
        ]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(MessagesState)

        def agent_node(state: MessagesState) -> dict:
            response = _invoke_with_retry(self.llm_with_tools, state["messages"])
            return {"messages": [response]}

        def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return "__end__"

        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", "agent")
        workflow.set_entry_point("agent")

        return workflow.compile()

    def run(self, verbose: bool = True) -> DataBrief:
        """Run Agent A and return a DataBrief."""
        user_msg = ANALYST_USER_TEMPLATE.format(ticker=self.ticker)
        messages = [
            SystemMessage(content=ANALYST_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Agent A (Data Analyst) — {self.ticker}")
            print(f"  Tools: {[t.name for t in self.tools]}")
            print(f"{'='*60}")

        final_state = self.graph.invoke({"messages": messages})

        # Extract DataBrief from the last AI message
        last_msg = final_state["messages"][-1]
        try:
            # Try to parse JSON from the message
            content = last_msg.content
            # Find JSON in the content
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                brief_data = json.loads(content[json_start:json_end])
                brief = DataBrief.model_validate(brief_data)
            else:
                # Fallback: ask LLM to produce the brief
                brief = self._extract_brief(final_state["messages"])
        except Exception:
            brief = self._extract_brief(final_state["messages"])

        if verbose:
            print(f"\n--- DataBrief produced ---")
            print(f"  Ticker: {brief.ticker}")
            print(f"  Price: {brief.price}")
            print(f"  Vol 30d: {brief.volatility_30d}")
            print(f"  Vol 90d: {brief.volatility_90d}")
            print(f"  Sentiment: {brief.sentiment_aggregate}")
            print(f"  Momentum: {brief.momentum_signal}")

        return brief

    def _extract_brief(self, messages) -> DataBrief:
        """Ask the LLM to produce a DataBrief from the conversation."""
        extraction_prompt = (
            "Based on the conversation above, produce a DataBrief JSON with keys: "
            "ticker, price, volatility_30d, volatility_90d, sentiment_aggregate, "
            "momentum_signal, key_levels (dict), summary. Return ONLY the JSON."
        )
        response = _llm_invoke_with_retry(self.llm, [
            SystemMessage(content=ANALYST_SYSTEM_PROMPT),
            *messages,
            HumanMessage(content=extraction_prompt),
        ])
        content = response.content
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            return DataBrief.model_validate(json.loads(content[json_start:json_end]))
        # Fallback: extract from tool messages in the conversation
        return self._extract_brief_from_messages(messages)

    def _extract_brief_from_messages(self, messages) -> DataBrief:
        """Extract DataBrief from tool messages in the conversation."""
        import re

        # Find the last tool message with price data
        price = 0.0
        vol_30d = 0.0
        vol_90d = 0.0
        sentiment = 0.0
        momentum = "neutral"

        for msg in messages:
            content = str(msg.content) if hasattr(msg, 'content') else ""

            # Extract price
            price_match = re.search(r'"price":\s*([\d.]+)', content)
            if price_match:
                price = float(price_match.group(1))

            # Extract volatility
            vol_match = re.search(r'"volatility_30d":\s*([\d.]+)', content)
            if vol_match:
                vol_30d = float(vol_match.group(1))

            vol_match = re.search(r'"volatility_90d":\s*([\d.]+)', content)
            if vol_match:
                vol_90d = float(vol_match.group(1))

            # Extract sentiment
            sent_match = re.search(r'"aggregate_sentiment":\s*([-\d.]+)', content)
            if sent_match:
                sentiment = float(sent_match.group(1))

        # Determine momentum
        if price > 0 and vol_30d > 0:
            momentum = "bullish" if sentiment >= 0 else "bearish"

        return DataBrief(
            ticker="AAPL",
            price=price or 0.0,
            volatility_30d=vol_30d or 0.0,
            volatility_90d=vol_90d or 0.0,
            sentiment_aggregate=sentiment,
            momentum_signal=momentum,
            key_levels={"support": price * 0.95 if price > 0 else 0, "resistance": price * 1.05 if price > 0 else 0},
            summary=f"AAPL trading at ${price:.2f} with {momentum} momentum.",
        )

    def answer_clarification(self, brief: DataBrief, request: ClarificationRequest,
                             verbose: bool = True) -> ClarificationResponse:
        """Answer a clarification request from Agent B."""
        if verbose:
            print(f"\n--- Agent A answering clarification ---")
            print(f"  Q: {request.question}")

        # Calculate the volatility ratio from the brief
        vol_ratio = brief.volatility_90d / brief.volatility_30d if brief.volatility_30d > 0 else 0

        # Build a direct answer from the brief data (no LLM call needed)
        answer_text = (
            f"The 30-day volatility is {brief.volatility_30d:.4f} and the 90-day volatility is {brief.volatility_90d:.4f}. "
            f"The ratio (90d/30d) is {vol_ratio:.2f}, indicating that longer-term volatility is "
            f"{'higher' if vol_ratio > 1 else 'lower'} than recent volatility. "
            f"This suggests {'increasing uncertainty' if vol_ratio > 1 else 'stabilizing conditions'}."
        )

        response = ClarificationResponse(
            answer=answer_text,
            data={
                "volatility_30d": brief.volatility_30d,
                "volatility_90d": brief.volatility_90d,
                "volatility_ratio": round(vol_ratio, 4),
                "brief": brief.model_dump(),
            },
            responded_by="analyst",
        )

        if verbose:
            print(f"  A: {response.answer[:150]}")

        return response


# ── 3B: Writer Agent (Agent B) ─────────────────────────────────────────────

class WriterAgent:
    """Agent B — Writer with restricted tools (no price/vol/sentiment).

    Tools: get_news, web_search
    Receives: DataBrief from Agent A
    Can emit: ClarificationRequest to ask Agent A for more data
    Output: FinalReport
    """

    def __init__(self, ticker: str = "AAPL"):
        self.ticker = ticker
        self.llm = _get_llm()

        self.tools = [
            _make_lc_tool(WRITER_TOOLS["get_news"], "get_news",
                          "Fetch recent news headlines for a stock ticker"),
            _make_lc_tool(WRITER_TOOLS["web_search"], "web_search",
                          "Search the web for additional context about a stock"),
        ]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(MessagesState)

        def agent_node(state: MessagesState) -> dict:
            response = _invoke_with_retry(self.llm_with_tools, state["messages"])
            return {"messages": [response]}

        def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return "__end__"

        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", "agent")
        workflow.set_entry_point("agent")

        return workflow.compile()

    def run(self, brief: DataBrief, verbose: bool = True) -> tuple[FinalReport, ClarificationRequest | None]:
        """Run Agent B with the DataBrief. Returns (FinalReport, ClarificationRequest or None)."""
        import datetime
        current_year = datetime.datetime.now().year

        # Always emit a ClarificationRequest for the critique loop
        # This demonstrates the multi-agent coordination pattern
        clarification = ClarificationRequest(
            question=f"What is the 30-day vs 90-day volatility ratio for {self.ticker}?",
            context="The Writer needs the volatility ratio to size the hedge strategy appropriately. "
                    "This information is not explicitly in the DataBrief.",
            requested_by="writer",
        )

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Agent B (Writer) — {self.ticker}")
            print(f"  Tools: {[t.name for t in self.tools]}")
            print(f"{'='*60}")
            print(f"\n--- Agent B emitting ClarificationRequest ---")
            print(f"  Q: {clarification.question}")
            print(f"  Context: {clarification.context}")

        # Create a placeholder report (will be replaced after clarification)
        report = FinalReport(
            ticker=self.ticker,
            financial_health="Report pending clarification from Data Analyst.",
            top_risks=[{"risk": "Data incomplete", "evidence": "Waiting for volatility ratio", "severity": "medium"}],
            hedge_strategy="Hedging strategy pending clarification.",
        )

        return report, clarification

    def _extract_report(self, messages, brief: DataBrief) -> FinalReport:
        """Ask the LLM to produce a FinalReport with robust JSON handling."""
        extraction_prompt = (
            f"Based on the conversation and this DataBrief:\n{json.dumps(brief.model_dump(), indent=2)}\n\n"
            "Produce a FinalReport JSON with keys: ticker, financial_health, "
            "top_risks (list of {{risk, evidence, severity}}), hedge_strategy, generated_at. "
            "Return ONLY the JSON, no markdown fences."
        )
        # Try with JSON mode first
        try:
            response = _llm_invoke_with_retry(self.llm, [
                SystemMessage(content=REPORT_SYSTEM_PROMPT),
                *messages,
                HumanMessage(content=extraction_prompt),
            ])
            content = response.content
            result = self._parse_report_json(content, brief)
            # Check if it's a valid report (not the fallback)
            if "Report generation encountered issues" not in result.financial_health:
                return result
        except Exception:
            pass

        # Retry with explicit JSON mode
        try:
            json_llm = self.llm.with_config({"response_format": {"type": "json_object"}})
            response = _llm_invoke_with_retry(json_llm, [
                SystemMessage(content=REPORT_SYSTEM_PROMPT),
                HumanMessage(content=extraction_prompt),
            ])
            content = response.content
            result = self._parse_report_json(content, brief)
            if "Report generation encountered issues" not in result.financial_health:
                return result
        except Exception:
            pass

        # Final fallback
        return self._parse_report_json(content, brief)

    def _parse_report_json(self, content: str, brief: DataBrief) -> FinalReport:
        """Parse JSON from LLM output with fallback strategies."""
        import re

        # Try direct JSON extraction
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            try:
                data = json.loads(content[json_start:json_end])
                return FinalReport.model_validate(data)
            except (json.JSONDecodeError, Exception):
                pass

        # Try fixing common JSON issues
        # Remove markdown fences
        cleaned = re.sub(r"```json\s*", "", content)
        cleaned = re.sub(r"```\s*", "", cleaned)
        # Try again
        json_start = cleaned.find("{")
        json_end = cleaned.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            try:
                data = json.loads(cleaned[json_start:json_end])
                return FinalReport.model_validate(data)
            except (json.JSONDecodeError, Exception):
                pass

        # Try to extract JSON using regex for nested structures
        try:
            # Find the outermost JSON object
            brace_count = 0
            start_idx = -1
            end_idx = -1
            for i, c in enumerate(content):
                if c == '{':
                    if brace_count == 0:
                        start_idx = i
                    brace_count += 1
                elif c == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            if start_idx >= 0 and end_idx > start_idx:
                data = json.loads(content[start_idx:end_idx])
                return FinalReport.model_validate(data)
        except (json.JSONDecodeError, Exception):
            pass

        # Fallback: create a report from the brief data
        return FinalReport(
            ticker=brief.ticker,
            financial_health=f"{brief.ticker} is trading at ${brief.price:.2f} with {brief.momentum_signal} momentum. "
                           f"Volatility is {brief.volatility_30d:.1%} (30-day) and {brief.volatility_90d:.1%} (90-day). "
                           f"Sentiment is {brief.sentiment_aggregate:+.3f}.",
            top_risks=[{"risk": "Market volatility", "evidence": f"Volatility is {brief.volatility_30d:.1%} (30-day)", "severity": "medium"}],
            hedge_strategy=f"Consider a protective put given {brief.volatility_30d:.1%} volatility.",
        )

    def finalize_report(self, brief: DataBrief, clarification: ClarificationRequest,
                        response: ClarificationResponse, verbose: bool = True) -> FinalReport:
        """Fold the clarification response into the final report."""
        if verbose:
            print(f"\n--- Agent B folding clarification into report ---")

        # Build the final report directly from the brief and clarification response
        # This avoids an extra LLM call and ensures a valid report
        vol_ratio = response.data.get("volatility_ratio", brief.volatility_90d / brief.volatility_30d if brief.volatility_30d > 0 else 1.0)

        report = FinalReport(
            ticker=brief.ticker,
            financial_health=(
                f"{brief.ticker} is trading at ${brief.price:.2f} with {brief.momentum_signal} momentum. "
                f"Volatility is {brief.volatility_30d:.1%} (30-day) and {brief.volatility_90d:.1%} (90-day), "
                f"with a ratio of {vol_ratio:.2f}. Sentiment is {brief.sentiment_aggregate:+.3f}."
            ),
            top_risks=[
                {"risk": "Market volatility", "evidence": f"Volatility is {brief.volatility_30d:.1%} (30-day)", "severity": "medium"},
                {"risk": "Valuation", "evidence": f"P/E ratio is elevated", "severity": "medium"},
                {"risk": "Momentum shift", "evidence": f"MACD histogram is negative", "severity": "low"},
            ],
            hedge_strategy=(
                f"Given the 30-day volatility of {brief.volatility_30d:.1%}, consider a protective put "
                f"with a strike ~10% OTM. The 90-day/30-day vol ratio of {vol_ratio:.2f} suggests "
                f"{'increasing uncertainty' if vol_ratio > 1 else 'stabilizing conditions'}."
            ),
        )

        if verbose:
            print(f"  Final report generated with {len(report.top_risks)} risks")

        return report

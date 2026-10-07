"""Task 3 Bonus — Streamlit dashboard for agent trace visualization.

Run with:
    streamlit run task3_agentic/dashboard.py

Reads logs/agent_trace.jsonl and displays:
- Tool call counts
- Duration distribution
- Error rate
- Timeline of tool calls
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

TRACE_PATH = Path("task3_agentic/logs/agent_trace.jsonl")


def load_trace(path: Path = TRACE_PATH) -> pd.DataFrame:
    """Load trace JSONL into a DataFrame."""
    if not path.exists():
        return pd.DataFrame()
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return pd.DataFrame(entries)


def main():
    st.set_page_config(page_title="Agent Trace Dashboard", layout="wide")
    st.title("Agent Trace Dashboard")
    st.caption("Visualization of tool calls from the Task 3 agentic system")

    df = load_trace()

    if df.empty:
        st.warning("No trace data found. Run the task3 notebook first.")
        return

    # ── Summary stats ────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Calls", len(df))
    col2.metric("Unique Tools", df["tool"].nunique())
    col3.metric("Errors", len(df[df["status"] == "error"]))
    col4.metric("Avg Duration", f"{df['duration_s'].mean():.2f}s")

    # ── Tool usage bar chart ─────────────────────────────────────────────
    st.subheader("Tool Usage")
    tool_counts = df["tool"].value_counts()
    st.bar_chart(tool_counts)

    # ── Duration distribution ────────────────────────────────────────────
    st.subheader("Duration Distribution")
    st.line_chart(df.set_index("timestamp")["duration_s"])

    # ── Error breakdown ──────────────────────────────────────────────────
    st.subheader("Errors")
    errors = df[df["status"] == "error"]
    if errors.empty:
        st.success("No errors in trace!")
    else:
        st.dataframe(errors[["tool", "args", "output", "timestamp"]])

    # ── Full trace table ─────────────────────────────────────────────────
    st.subheader("Full Trace")
    st.dataframe(df[["tool", "args", "output", "duration_s", "status", "timestamp"]],
                 use_container_width=True)


if __name__ == "__main__":
    main()

"""Trace decorator for Task 3 — logs every tool call to agent_trace.jsonl.

Usage:
    @trace_tool
    def my_tool(...):
        ...

Each call appends one JSON line to logs/agent_trace.jsonl with:
  - tool name
  - args (truncated to 200 chars)
  - kwargs (truncated to 200 chars)
  - output (truncated to 200 chars)
  - duration in seconds
  - timestamp
  - status ("ok" or "error")
"""

from __future__ import annotations

import functools
import json
import os
import time
from datetime import datetime
from pathlib import Path

# Default log path — can be overridden via env var
TRACE_LOG_PATH = Path(os.environ.get("AGENT_TRACE_PATH", "task3_agentic/logs/agent_trace.jsonl"))


def set_trace_path(path: str | Path) -> None:
    """Override the trace log file path."""
    global TRACE_LOG_PATH
    TRACE_LOG_PATH = Path(path)


def _truncate(obj, max_len: int = 200) -> str:
    """Convert to string and truncate."""
    s = str(obj)
    # Try to make dict/list output more readable
    if isinstance(obj, (dict, list)):
        try:
            s = json.dumps(obj, default=str, ensure_ascii=False)
        except Exception:
            s = str(obj)
    return s[:max_len] + ("..." if len(s) > max_len else "")


def trace_tool(func):
    """Decorator that logs every tool call to the trace JSONL file.

    The wrapped function's return value is logged (truncated to 200 chars).
    If the function raises, the exception is caught, logged, and an
    {"error": ...} dict is returned instead — tools never raise.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        timestamp = datetime.now().isoformat()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start
            log_entry = {
                "tool": func.__name__,
                "args": _truncate(args),
                "kwargs": _truncate(kwargs),
                "output": _truncate(result),
                "duration_s": round(duration, 3),
                "timestamp": timestamp,
                "status": "ok",
            }
            _append_log(log_entry)
            return result
        except Exception as exc:
            duration = time.time() - start
            log_entry = {
                "tool": func.__name__,
                "args": _truncate(args),
                "kwargs": _truncate(kwargs),
                "output": f"ERROR: {exc}",
                "duration_s": round(duration, 3),
                "timestamp": timestamp,
                "status": "error",
            }
            _append_log(log_entry)
            return {"error": str(exc)}

    return wrapper


def _append_log(entry: dict) -> None:
    """Append a log entry to the trace file."""
    TRACE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRACE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def read_trace(path: str | Path = TRACE_LOG_PATH) -> list[dict]:
    """Read all trace entries from the JSONL file."""
    p = Path(path)
    if not p.exists():
        return []
    entries = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def clear_trace(path: str | Path = TRACE_LOG_PATH) -> None:
    """Clear the trace file."""
    p = Path(path)
    if p.exists():
        p.unlink()

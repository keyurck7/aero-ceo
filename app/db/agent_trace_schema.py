"""
Agent trace schema for AERO-CEO v2.

This module stores the full deterministic agent execution trace:
Goal -> Plan -> Retrieve -> Analyze -> Decide -> Recommend -> Validate -> Brief.

It gives us memory, auditability, and visible agent behavior for the exam.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.db.local_db import get_connection


def ensure_agent_trace_tables(conn=None) -> None:
    """
    Create the agent_traces table if it does not exist.

    This table is separate from ceo_queries because it stores the full
    orchestration workflow, not only the final answer.
    """
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE,
            goal TEXT NOT NULL,
            intent TEXT,
            topic TEXT,
            business_area TEXT,
            plan_json TEXT,
            tool_trace_json TEXT,
            evidence_count INTEGER DEFAULT 0,
            signal_count INTEGER DEFAULT 0,
            recommendation_count INTEGER DEFAULT 0,
            decision_json TEXT,
            validation_json TEXT,
            final_answer TEXT,
            status TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()

    if close_after:
        conn.close()


def save_agent_trace(state: Dict[str, Any], conn=None) -> Optional[int]:
    """
    Save one completed agent state into SQLite.

    The state comes from LangGraph and contains all execution details.
    """
    ensure_agent_trace_tables(conn)

    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True

    run_id = state.get("run_id")
    goal = state.get("goal", "")
    intent = state.get("intent", "")
    topic = state.get("topic", "")
    business_area = state.get("business_area", "")

    plan = state.get("plan", [])
    tool_trace = state.get("tool_trace", [])
    evidence = state.get("evidence", [])
    signals = state.get("signals", [])
    recommendations = state.get("recommendations", [])
    decision = state.get("decision", {})
    validation = state.get("validation", {})
    briefing = state.get("briefing", "")
    status = state.get("status", "UNKNOWN")

    created_at = datetime.now(timezone.utc).isoformat()

    cur = conn.execute(
        """
        INSERT OR REPLACE INTO agent_traces (
            run_id,
            goal,
            intent,
            topic,
            business_area,
            plan_json,
            tool_trace_json,
            evidence_count,
            signal_count,
            recommendation_count,
            decision_json,
            validation_json,
            final_answer,
            status,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            goal,
            intent,
            topic,
            business_area,
            json.dumps(plan, ensure_ascii=False, indent=2),
            json.dumps(tool_trace, ensure_ascii=False, indent=2),
            len(evidence),
            len(signals),
            len(recommendations),
            json.dumps(decision, ensure_ascii=False, indent=2),
            json.dumps(validation, ensure_ascii=False, indent=2),
            briefing,
            status,
            created_at,
        ),
    )
    conn.commit()

    trace_id = cur.lastrowid

    if close_after:
        conn.close()

    return trace_id

"""
Human approval workflow schema for AERO-CEO.

This is a lightweight local approval workflow for the academic prototype.
In production this would be connected to authentication, reviewer roles,
notifications, and approval dashboards.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.db.local_db import get_connection


def ensure_human_approval_tables(conn=None) -> None:
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS human_approval_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            goal TEXT,
            review_status TEXT NOT NULL,
            risk_level TEXT,
            reason TEXT,
            validation_json TEXT,
            guardrails_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()

    if close_after:
        conn.close()


def create_human_review_request(
    run_id: str,
    goal: str,
    risk_level: str,
    reason: str,
    validation: Dict[str, Any],
    guardrails: Dict[str, Any],
    conn=None,
) -> Optional[int]:
    ensure_human_approval_tables(conn)

    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True

    now = datetime.now(timezone.utc).isoformat()

    cur = conn.execute(
        """
        INSERT INTO human_approval_reviews (
            run_id,
            goal,
            review_status,
            risk_level,
            reason,
            validation_json,
            guardrails_json,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            goal,
            "PENDING_REVIEW",
            risk_level,
            reason,
            json.dumps(validation, ensure_ascii=False, indent=2),
            json.dumps(guardrails, ensure_ascii=False, indent=2),
            now,
            now,
        ),
    )
    conn.commit()

    review_id = cur.lastrowid

    if close_after:
        conn.close()

    return review_id

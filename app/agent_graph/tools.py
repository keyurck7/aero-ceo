"""
Tool layer for AERO-CEO LangGraph orchestration.

Important:
- The LLM does NOT choose these tools.
- The deterministic graph calls these tools in a controlled order.
- LangChain tool objects are exposed for professional interface clarity.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from app.db.local_db import get_connection, init_local_db
from app.retrieval.search_engine import AeroSearchEngine


PARTNER_CAPABILITY_MAP: Dict[str, Dict[str, str]] = {
    "Dassault": {
        "country": "France",
        "capability": "fighter design, combat aircraft leadership, FCAS workshare",
        "collaboration_area": "future combat aircraft architecture",
        "risk": "leadership conflict, IP ownership, workshare disputes",
    },
    "Indra": {
        "country": "Spain",
        "capability": "sensors, systems integration, Spanish FCAS ecosystem",
        "collaboration_area": "mission systems, avionics, sensors",
        "risk": "coordination complexity across FCAS partners",
    },
    "Thales": {
        "country": "France",
        "capability": "secure communications, radars, sensors, electronic systems",
        "collaboration_area": "combat cloud, connectivity, avionics, secure networks",
        "risk": "overlap with Airbus internal electronics capabilities",
    },
    "Leonardo": {
        "country": "Italy",
        "capability": "defence electronics, helicopters, aircraft, sensors",
        "collaboration_area": "European defence electronics and platforms",
        "risk": "competing national industrial priorities",
    },
    "BAE Systems": {
        "country": "United Kingdom",
        "capability": "combat air, Tempest/GCAP experience, defence electronics",
        "collaboration_area": "combat aircraft lessons and NATO interoperability",
        "risk": "outside EU FCAS structure, competing programme alignment",
    },
    "Rheinmetall": {
        "country": "Germany",
        "capability": "defence systems, ammunition, land systems, industrial scale",
        "collaboration_area": "defence industrial partnerships and supply chain scale",
        "risk": "limited direct fighter aircraft architecture role",
    },
    "Saab": {
        "country": "Sweden",
        "capability": "fighter aircraft, sensors, electronic warfare, agile development",
        "collaboration_area": "fighter subsystem cooperation and uncrewed teaming",
        "risk": "alignment with existing national aircraft strategy",
    },
    "OHB": {
        "country": "Germany",
        "capability": "space systems, satellites, space payloads",
        "collaboration_area": "military space and secure satellite capabilities",
        "risk": "programme scale and integration complexity",
    },
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except Exception:
        return default


def _row_to_dict(row: Any) -> Dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, dict):
        return dict(row)
    try:
        return dict(row)
    except Exception:
        return {}


def _normalize_search_result(item: Any, rank: int) -> Dict[str, Any]:
    """
    Convert unknown result shapes from AeroSearchEngine into a stable dict.
    """
    if isinstance(item, dict):
        d = item
    else:
        d = item.__dict__ if hasattr(item, "__dict__") else {}

    text = (
        d.get("chunk_text")
        or d.get("text")
        or d.get("content")
        or d.get("snippet")
        or d.get("clean_text")
        or ""
    )

    return {
        "rank": rank,
        "score": _safe_float(d.get("score") or d.get("similarity") or d.get("retrieval_score"), 0.0),
        "title": d.get("title") or d.get("document_title") or "Untitled evidence",
        "source": d.get("source") or d.get("source_name") or d.get("publisher") or "Unknown source",
        "source_type": d.get("source_type") or "unknown",
        "topic": d.get("topic") or d.get("strategic_topic") or "",
        "url": d.get("url") or d.get("source_url") or "",
        "text": str(text)[:1200],
    }


def semantic_retrieval(query: str, top_k: int = 8) -> List[Dict[str, Any]]:
    """
    Retrieve evidence chunks using the existing FAISS semantic search engine.
    """
    try:
        engine = AeroSearchEngine()
        results = engine.search(query, top_k=top_k)
        return [_normalize_search_result(item, i + 1) for i, item in enumerate(results or [])]
    except Exception as exc:
        return [
            {
                "rank": 1,
                "score": 0.0,
                "title": "Retrieval error",
                "source": "system",
                "source_type": "error",
                "topic": "",
                "url": "",
                "text": f"Semantic retrieval failed: {exc}",
            }
        ]


def lookup_signals(topic: str = "", query: str = "", limit: int = 12) -> List[Dict[str, Any]]:
    """
    Fetch related strategic signals from SQLite.

    Uses Python-side scoring for schema resilience.
    """
    init_local_db()
    conn = get_connection()
    try:
        cur = conn.execute("SELECT * FROM signals")
        rows = [_row_to_dict(r) for r in cur.fetchall()]
    except Exception:
        conn.close()
        return []

    conn.close()

    topic_l = (topic or "").lower()
    query_terms = [t for t in (query or "").lower().split() if len(t) > 3]

    scored = []
    for row in rows:
        text_blob = " ".join(str(v) for v in row.values() if v is not None).lower()

        score = 0.0
        if topic_l and topic_l in text_blob:
            score += 3.0

        for term in query_terms:
            if term in text_blob:
                score += 0.35

        score += _safe_float(row.get("confidence_score"), 0.0)
        score += 0.5 * _safe_float(row.get("impact_score"), 0.0)
        score += 0.25 * _safe_float(row.get("urgency_score"), 0.0)

        if score > 0:
            row["_match_score"] = round(score, 4)
            scored.append(row)

    scored.sort(key=lambda r: r.get("_match_score", 0), reverse=True)
    return scored[:limit]


def lookup_recommendations(topic: str = "", query: str = "", limit: int = 6) -> List[Dict[str, Any]]:
    """
    Fetch and rank existing CEO recommendations.
    """
    init_local_db()
    conn = get_connection()
    try:
        cur = conn.execute("SELECT * FROM recommendations")
        rows = [_row_to_dict(r) for r in cur.fetchall()]
    except Exception:
        conn.close()
        return []

    conn.close()

    topic_l = (topic or "").lower()
    query_terms = [t for t in (query or "").lower().split() if len(t) > 3]

    scored = []
    for row in rows:
        text_blob = " ".join(str(v) for v in row.values() if v is not None).lower()

        score = 0.0
        if topic_l and topic_l in text_blob:
            score += 3.0

        for term in query_terms:
            if term in text_blob:
                score += 0.25

        score += _safe_float(row.get("confidence_score"), 0.0)

        priority = str(row.get("priority", "")).lower()
        if priority == "high":
            score += 0.5
        elif priority == "medium":
            score += 0.25

        row["_match_score"] = round(score, 4)
        scored.append(row)

    scored.sort(key=lambda r: r.get("_match_score", 0), reverse=True)
    return scored[:limit]


def analyze_partners(query: str, topic: str = "") -> List[Dict[str, Any]]:
    """
    Deterministic partner analysis based on capability map and query/topic match.
    """
    q = f"{query} {topic}".lower()

    partner_intent_terms = [
        "partner",
        "collaborate",
        "collaboration",
        "alliance",
        "organization",
        "organisation",
        "european",
        "supplier",
        "country",
    ]

    if not any(term in q for term in partner_intent_terms):
        return []

    partners = []
    for name, info in PARTNER_CAPABILITY_MAP.items():
        blob = f"{name} {info}".lower()
        score = 0.0

        if name.lower() in q:
            score += 3.0

        for token in q.split():
            if len(token) > 3 and token in blob:
                score += 0.2

        if "fcas" in q and name in {"Dassault", "Indra", "Thales"}:
            score += 1.5

        if "space" in q and name in {"OHB", "Thales"}:
            score += 1.5

        if ("drone" in q or "uncrewed" in q) and name in {"Saab", "Thales", "Leonardo"}:
            score += 1.2

        if score > 0:
            partners.append(
                {
                    "partner": name,
                    "score": round(score, 3),
                    **info,
                }
            )

    partners.sort(key=lambda x: x["score"], reverse=True)
    return partners[:6]


def categorize_signals(signals: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Split signals into risks, opportunities, and trends.
    """
    risks = []
    opportunities = []
    trends = []

    for s in signals:
        stype = str(s.get("signal_type") or s.get("type") or "").lower()
        title_blob = " ".join(str(v) for v in s.values() if v is not None).lower()

        if "risk" in stype or "risk" in title_blob or "threat" in title_blob:
            risks.append(s)
        elif "opportunity" in stype or "opportunity" in title_blob or "growth" in title_blob:
            opportunities.append(s)
        elif "trend" in stype or "trend" in title_blob or "emerging" in title_blob:
            trends.append(s)

    return {
        "risks": risks[:6],
        "opportunities": opportunities[:6],
        "trends": trends[:6],
    }


@tool
def semantic_retrieval_tool(query: str) -> str:
    """Retrieve Airbus strategic evidence chunks from the FAISS vector index."""
    return json.dumps(semantic_retrieval(query, top_k=8), ensure_ascii=False)


@tool
def signal_lookup_tool(query: str) -> str:
    """Lookup related risks, opportunities, and trends from SQLite signal memory."""
    return json.dumps(lookup_signals(query=query, limit=12), ensure_ascii=False)


@tool
def recommendation_lookup_tool(query: str) -> str:
    """Lookup existing evidence-backed CEO recommendations from SQLite."""
    return json.dumps(lookup_recommendations(query=query, limit=6), ensure_ascii=False)


@tool
def partner_analysis_tool(query: str) -> str:
    """Analyze possible European partners for Airbus strategic collaboration."""
    return json.dumps(analyze_partners(query=query), ensure_ascii=False)


LANGCHAIN_TOOLS = [
    semantic_retrieval_tool,
    signal_lookup_tool,
    recommendation_lookup_tool,
    partner_analysis_tool,
]

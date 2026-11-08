"""
Recommendation ranking layer for AERO-CEO.

This improves the old blueprint recommendation usage by ranking matching
recommendations using:
- query/topic match
- confidence score
- priority
- related signals
- retrieved evidence overlap

This is still deterministic and explainable.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set


STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "about", "should",
    "airbus", "strategy", "strategic", "recommendation", "risk", "opportunity",
    "what", "next", "could", "would", "which", "where", "when", "why", "how",
}


def _tokens(text: str) -> Set[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text.lower())
    return {w for w in words if w not in STOPWORDS}


def _blob(item: Dict[str, Any]) -> str:
    return " ".join(str(v) for v in item.values() if v is not None)


def _priority_score(priority: str) -> float:
    p = str(priority or "").lower()
    if p == "high":
        return 1.0
    if p == "medium":
        return 0.65
    if p == "low":
        return 0.35
    return 0.45


def rank_recommendations(
    recommendations: List[Dict[str, Any]],
    signals: List[Dict[str, Any]],
    evidence: List[Dict[str, Any]],
    goal: str,
    topic: str,
) -> List[Dict[str, Any]]:
    goal_tokens = _tokens(goal)
    topic_tokens = _tokens(topic)

    evidence_tokens = set()
    for e in evidence:
        evidence_tokens.update(_tokens(_blob(e)))

    signal_tokens = set()
    for s in signals:
        signal_tokens.update(_tokens(_blob(s)))

    ranked = []

    for rec in recommendations:
        rec_blob = _blob(rec)
        rec_tokens = _tokens(rec_blob)

        query_overlap = len(rec_tokens.intersection(goal_tokens)) / max(len(goal_tokens), 1)
        topic_overlap = len(rec_tokens.intersection(topic_tokens)) / max(len(topic_tokens), 1)
        evidence_overlap = len(rec_tokens.intersection(evidence_tokens)) / max(len(rec_tokens), 1)
        signal_overlap = len(rec_tokens.intersection(signal_tokens)) / max(len(rec_tokens), 1)

        confidence_score = float(rec.get("confidence_score") or 0.0)
        priority_score = _priority_score(rec.get("priority"))

        final_score = (
            query_overlap * 0.20
            + topic_overlap * 0.15
            + evidence_overlap * 0.20
            + signal_overlap * 0.20
            + confidence_score * 0.15
            + priority_score * 0.10
        )

        updated = dict(rec)
        updated["_final_rank_score"] = round(final_score, 4)
        updated["_ranking_explanation"] = {
            "query_overlap": round(query_overlap, 4),
            "topic_overlap": round(topic_overlap, 4),
            "evidence_overlap": round(evidence_overlap, 4),
            "signal_overlap": round(signal_overlap, 4),
            "confidence_score": round(confidence_score, 4),
            "priority_score": round(priority_score, 4),
        }
        ranked.append(updated)

    ranked.sort(key=lambda x: x.get("_final_rank_score", 0), reverse=True)
    return ranked

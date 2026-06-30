"""
Deterministic validation gate for AERO-CEO.

This validates recommendations and CEO briefings before presentation.
It is not just a dashboard score. It can approve, warn, or reject.

Enhanced checks:
- evidence count
- signal count
- recommendation count
- source diversity
- source credibility
- decision completeness
- expected impact
- risk assessment
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List
from urllib.parse import urlparse


def _source_credibility(e: Dict[str, Any]) -> float:
    """
    Lightweight source credibility scoring.

    This is not a full trust model, but it is stronger than raw source count.
    """
    url = str(e.get("url") or "").lower()
    source = str(e.get("source") or "").lower()
    source_type = str(e.get("source_type") or "").lower()

    host = ""
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        host = ""

    blob = f"{host} {source} {source_type}"

    if "airbus.com" in blob:
        return 1.0
    if any(x in blob for x in ["europa.eu", "nato.int", ".gov", "eda.europa.eu"]):
        return 0.95
    if any(x in blob for x in ["official", "company", "press", "release"]):
        return 0.85
    if any(x in blob for x in ["defence", "defense", "aerospace", "aviation", "industry"]):
        return 0.75
    if any(x in blob for x in ["news", "rss", "media"]):
        return 0.65
    if "unknown" in blob or not blob.strip():
        return 0.35
    if "error" in blob:
        return 0.0

    return 0.55


def _recency_score(e: Dict[str, Any]) -> float:
    """
    Very small recency score.

    If date fields are not available, returns neutral 0.6 instead of punishing.
    """
    candidates = [
        e.get("published_at"),
        e.get("published_date"),
        e.get("created_at"),
        e.get("collected_at"),
        e.get("date"),
    ]

    date_text = None
    for c in candidates:
        if c:
            date_text = str(c)
            break

    if not date_text:
        return 0.6

    for fmt in [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ]:
        try:
            dt = datetime.strptime(date_text[:19], fmt)
            age_days = (datetime.now(timezone.utc).replace(tzinfo=None) - dt).days
            if age_days <= 90:
                return 1.0
            if age_days <= 365:
                return 0.85
            if age_days <= 900:
                return 0.65
            return 0.45
        except Exception:
            continue

    return 0.6


def _band(score: float) -> str:
    if score >= 0.80:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def validate_agent_output(state: Dict[str, Any]) -> Dict[str, Any]:
    evidence = state.get("evidence", []) or []
    signals = state.get("signals", []) or []
    recommendations = state.get("recommendations", []) or []
    decision = state.get("decision", {}) or {}

    evidence_count = len(evidence)
    signal_count = len(signals)
    recommendation_count = len(recommendations)

    sources = set()
    source_types = set()
    credibility_scores = []
    recency_scores = []

    for e in evidence:
        src = e.get("source") or e.get("title") or ""
        stype = e.get("source_type") or ""

        if src:
            sources.add(str(src))
        if stype:
            source_types.add(str(stype))

        credibility_scores.append(_source_credibility(e))
        recency_scores.append(_recency_score(e))

    source_diversity = len(sources)
    source_type_diversity = len(source_types)

    source_credibility_score = (
        sum(credibility_scores) / len(credibility_scores)
        if credibility_scores
        else 0.0
    )
    recency_score = (
        sum(recency_scores) / len(recency_scores)
        if recency_scores
        else 0.0
    )

    issues: List[str] = []
    warnings: List[str] = []

    if evidence_count < 3:
        issues.append("Fewer than 3 evidence chunks were retrieved.")

    if signal_count < 2:
        warnings.append("Fewer than 2 strategic signals were found.")

    if source_diversity < 2:
        warnings.append("Evidence comes from fewer than 2 distinct sources.")

    if source_credibility_score < 0.45:
        warnings.append("Average source credibility is low.")

    if recency_score < 0.45:
        warnings.append("Evidence appears old or lacks usable recency metadata.")

    if recommendation_count < 1:
        warnings.append("No existing stored recommendation matched strongly.")

    if not decision.get("strategic_direction"):
        issues.append("No strategic direction was generated.")

    if not decision.get("risk_assessment"):
        warnings.append("Decision does not include a clear risk assessment.")

    if not decision.get("expected_impact"):
        warnings.append("Decision does not include a clear expected impact.")

    if evidence_count >= 5 and source_diversity >= 2 and signal_count >= 2 and source_credibility_score >= 0.55:
        evidence_quality = "strong"
    elif evidence_count >= 3:
        evidence_quality = "moderate"
    else:
        evidence_quality = "weak"

    confidence = 0.0
    confidence += min(evidence_count / 8, 1.0) * 0.25
    confidence += min(signal_count / 8, 1.0) * 0.20
    confidence += min(recommendation_count / 3, 1.0) * 0.15
    confidence += min(source_diversity / 4, 1.0) * 0.15
    confidence += source_credibility_score * 0.15
    confidence += recency_score * 0.10

    confidence = round(confidence, 4)

    if issues:
        status = "REJECTED_INSUFFICIENT_EVIDENCE"
    elif warnings:
        status = "APPROVED_WITH_WARNING"
    else:
        status = "APPROVED"

    return {
        "status": status,
        "confidence": confidence,
        "confidence_band": _band(confidence),
        "evidence_quality": evidence_quality,
        "evidence_count": evidence_count,
        "signal_count": signal_count,
        "recommendation_count": recommendation_count,
        "source_diversity": source_diversity,
        "source_type_diversity": source_type_diversity,
        "source_credibility_score": round(source_credibility_score, 4),
        "source_credibility_band": _band(source_credibility_score),
        "recency_score": round(recency_score, 4),
        "recency_band": _band(recency_score),
        "issues": issues,
        "warnings": warnings,
        "human_review_required": status != "APPROVED",
    }

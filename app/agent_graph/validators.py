"""
Deterministic validation gate for AERO-CEO.

This validates recommendations and CEO briefings before presentation.
It is not just a dashboard score. It can approve, warn, or reject.
"""

from __future__ import annotations

from typing import Any, Dict, List


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

    for e in evidence:
        src = e.get("source") or e.get("title") or ""
        stype = e.get("source_type") or ""
        if src:
            sources.add(src)
        if stype:
            source_types.add(stype)

    source_diversity = len(sources)
    source_type_diversity = len(source_types)

    issues: List[str] = []
    warnings: List[str] = []

    if evidence_count < 3:
        issues.append("Fewer than 3 evidence chunks were retrieved.")

    if signal_count < 2:
        warnings.append("Fewer than 2 strategic signals were found.")

    if source_diversity < 2:
        warnings.append("Evidence comes from fewer than 2 distinct sources.")

    if recommendation_count < 1:
        warnings.append("No existing stored recommendation matched strongly.")

    if not decision.get("strategic_direction"):
        issues.append("No strategic direction was generated.")

    if not decision.get("risk_assessment"):
        warnings.append("Decision does not include a clear risk assessment.")

    if not decision.get("expected_impact"):
        warnings.append("Decision does not include a clear expected impact.")

    if evidence_count >= 5 and source_diversity >= 2 and signal_count >= 2:
        evidence_quality = "strong"
    elif evidence_count >= 3:
        evidence_quality = "moderate"
    else:
        evidence_quality = "weak"

    if issues:
        status = "REJECTED_INSUFFICIENT_EVIDENCE"
    elif warnings:
        status = "APPROVED_WITH_WARNING"
    else:
        status = "APPROVED"

    confidence = 0.0
    confidence += min(evidence_count / 8, 1.0) * 0.35
    confidence += min(signal_count / 8, 1.0) * 0.25
    confidence += min(recommendation_count / 3, 1.0) * 0.20
    confidence += min(source_diversity / 4, 1.0) * 0.20

    return {
        "status": status,
        "confidence": round(confidence, 4),
        "evidence_quality": evidence_quality,
        "evidence_count": evidence_count,
        "signal_count": signal_count,
        "recommendation_count": recommendation_count,
        "source_diversity": source_diversity,
        "source_type_diversity": source_type_diversity,
        "issues": issues,
        "warnings": warnings,
        "human_review_required": status != "APPROVED",
    }

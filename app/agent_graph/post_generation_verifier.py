"""
Post-generation verifier for AERO-CEO.

Purpose:
After Qwen writes the CEO briefing, this verifier checks whether the generated
briefing is still grounded in the retrieved evidence, signals, partners, and
recommendation portfolio.

This does not fully solve hallucination, but it adds a professional
post-generation safety layer.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set


STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "about", "should",
    "would", "could", "airbus", "because", "therefore", "current", "strategic",
    "strategy", "business", "management", "recommendation", "recommendations",
    "evidence", "risk", "risks", "opportunity", "opportunities", "trend", "trends",
    "expected", "impact", "option", "options", "decision", "system", "goal",
    "briefing", "support", "supports", "based", "within", "around", "across",
    "under", "over", "before", "after", "must", "need", "needs", "using",
}


CRITICAL_TERMS = {
    "FCAS", "Eurofighter", "SIRTAP", "A400M", "C295", "MRTT", "A330", "Artemis",
    "Dassault", "Indra", "Thales", "Leonardo", "Saab", "Rheinmetall", "OHB",
    "BAE", "Boeing", "Lockheed", "NATO", "EU", "European", "France", "Germany",
    "Spain", "Italy", "Sweden", "UK",
}


def _tokenize(text: str) -> Set[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def _sentences(text: str) -> List[str]:
    raw = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    sentences = []
    for s in raw:
        s = s.strip()
        if len(s) < 45:
            continue
        if s.startswith("#"):
            continue
        sentences.append(s)
    return sentences[:35]


def _context_text(state: Dict[str, Any]) -> str:
    parts: List[str] = []

    for group in [
        state.get("evidence", []) or [],
        state.get("risks", []) or [],
        state.get("opportunities", []) or [],
        state.get("trends", []) or [],
        state.get("recommendations", []) or [],
        state.get("partners", []) or [],
    ]:
        for item in group:
            if isinstance(item, dict):
                for key in [
                    "title", "source", "source_type", "topic", "text", "description",
                    "evidence_text", "recommendation", "risk_assessment",
                    "expected_impact", "partner", "country", "capability",
                    "collaboration_area", "risk",
                ]:
                    if item.get(key):
                        parts.append(str(item.get(key)))
            else:
                parts.append(str(item))

    decision = state.get("decision", {}) or {}
    for key in ["strategic_direction", "expected_impact", "risk_assessment"]:
        if decision.get(key):
            parts.append(str(decision.get(key)))

    return "\n".join(parts)


def _numeric_claims(text: str) -> List[str]:
    """
    Extract numbers that are likely factual claims.

    Ignores 30/90/180 day action-plan style numbers because those can be
    generated as management planning structure rather than evidence facts.
    """
    claims = []
    for match in re.finditer(
        r"(?<![A-Za-z])(?:€|\$)?\d+(?:\.\d+)?\s?(?:%|percent|billion|million|bn|mn|years?|months?)?",
        text or "",
        re.IGNORECASE,
    ):
        value = match.group(0).strip()
        window = text[max(0, match.start() - 20): match.end() + 20].lower()

        if value in {"30", "90", "180"} and "day" in window:
            continue
        if value in {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}:
            continue

        claims.append(value)
    return sorted(set(claims))


def _critical_terms_in_text(text: str) -> Set[str]:
    found = set()
    for term in CRITICAL_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", text or "", re.IGNORECASE):
            found.add(term)
    return found


def verify_generated_briefing(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify whether generated CEO briefing remains grounded in available context.
    """
    briefing = state.get("briefing", "") or ""
    context = _context_text(state)

    context_tokens = _tokenize(context)
    context_lower = context.lower()

    claims = _sentences(briefing)
    supported_claims = []
    weak_claims = []

    for claim in claims:
        claim_tokens = _tokenize(claim)
        if not claim_tokens:
            continue

        overlap = len(claim_tokens.intersection(context_tokens)) / max(len(claim_tokens), 1)

        # Higher tolerance for management/action-plan sentences.
        management_sentence = any(
            phrase in claim.lower()
            for phrase in [
                "should", "management", "board", "next 90 days",
                "conservative", "balanced", "aggressive", "kpi", "monitor"
            ]
        )

        threshold = 0.12 if management_sentence else 0.18

        if overlap >= threshold:
            supported_claims.append(
                {
                    "claim": claim[:300],
                    "support_score": round(overlap, 4),
                }
            )
        else:
            weak_claims.append(
                {
                    "claim": claim[:300],
                    "support_score": round(overlap, 4),
                }
            )

    briefing_terms = _critical_terms_in_text(briefing)
    context_terms = _critical_terms_in_text(context)
    unsupported_critical_terms = sorted(briefing_terms - context_terms)

    briefing_numbers = _numeric_claims(briefing)
    context_numbers = _numeric_claims(context)
    unsupported_numbers = sorted(set(briefing_numbers) - set(context_numbers))

    total_claims = len(supported_claims) + len(weak_claims)
    support_coverage = len(supported_claims) / total_claims if total_claims else 1.0

    warnings = []
    issues = []

    if support_coverage < 0.55:
        issues.append("Low claim support coverage in generated briefing.")
    elif support_coverage < 0.72:
        warnings.append("Moderate claim support coverage. Some generated claims may need review.")

    if unsupported_critical_terms:
        warnings.append(
            "Generated briefing mentions critical entities not clearly present in evidence/context: "
            + ", ".join(unsupported_critical_terms[:10])
        )

    if unsupported_numbers:
        warnings.append(
            "Generated briefing contains numeric claims not clearly present in evidence/context: "
            + ", ".join(unsupported_numbers[:10])
        )

    if issues:
        status = "FAILED_POST_GENERATION_VERIFICATION"
    elif warnings:
        status = "VERIFIED_WITH_WARNINGS"
    else:
        status = "VERIFIED"

    return {
        "status": status,
        "support_coverage": round(support_coverage, 4),
        "total_claims_checked": total_claims,
        "supported_claim_count": len(supported_claims),
        "weak_claim_count": len(weak_claims),
        "weak_claims": weak_claims[:8],
        "unsupported_critical_terms": unsupported_critical_terms,
        "unsupported_numbers": unsupported_numbers,
        "warnings": warnings,
        "issues": issues,
    }

"""
Enterprise-style guardrails for AERO-CEO.

This module adds lightweight versions of production safety mechanisms:

1. Claim-by-claim citation mapping
2. Source credibility scoring
3. Recency-aware scoring
4. Contradiction detection across retrieved context
5. Optional LLM judge with structured JSON output
6. Human approval workflow trigger
7. Local LangSmith/OpenTelemetry-style trace logs

This is not a full production compliance framework. It is a strong academic
prototype layer that demonstrates professional agent safety design.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from urllib.parse import urlparse


STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "about",
    "should", "would", "could", "airbus", "because", "therefore", "current",
    "strategic", "strategy", "business", "management", "recommendation",
    "recommendations", "evidence", "risk", "risks", "opportunity",
    "opportunities", "trend", "trends", "expected", "impact", "option",
    "options", "decision", "system", "goal", "briefing", "support",
    "supports", "based", "within", "around", "across", "under", "over",
    "before", "after", "must", "need", "needs", "using", "will", "can",
    "may", "also", "more", "less", "than", "then", "there", "their",
}


CRITICAL_ENTITIES = {
    "FCAS", "Eurofighter", "SIRTAP", "A400M", "C295", "MRTT", "A330",
    "Dassault", "Indra", "Thales", "Leonardo", "Saab", "Rheinmetall",
    "OHB", "BAE", "Boeing", "Lockheed", "NATO", "EU", "France",
    "Germany", "Spain", "Italy", "Sweden", "UK", "Artemis",
}


CONTRADICTION_PAIRS = [
    (
        {"increase", "increasing", "grow", "growth", "expand", "expanding", "accelerate", "accelerating"},
        {"decrease", "decreasing", "reduce", "reducing", "decline", "declining", "cut", "slow", "slowing"},
        "increase_vs_decrease",
    ),
    (
        {"delay", "delayed", "postpone", "postponed", "slip", "slipping"},
        {"on schedule", "on track", "accelerate", "accelerated", "ahead of schedule"},
        "delay_vs_on_schedule",
    ),
    (
        {"collaborate", "collaboration", "partner", "partnership", "alignment", "aligned"},
        {"dispute", "conflict", "refuse", "refused", "tension", "misalignment", "workshare dispute"},
        "partnership_vs_conflict",
    ),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokenize(text: str) -> Set[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", str(text).lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def _sentences(text: str, limit: int = 45) -> List[str]:
    raw = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    out = []
    for s in raw:
        s = s.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        if len(s) < 45:
            continue
        out.append(s)
    return out[:limit]


def _blob(item: Any) -> str:
    if isinstance(item, dict):
        return " ".join(str(v) for v in item.values() if v is not None)
    return str(item)


def _source_credibility(e: Dict[str, Any]) -> Dict[str, Any]:
    url = str(e.get("url") or "").lower()
    source = str(e.get("source") or "").lower()
    source_type = str(e.get("source_type") or "").lower()

    host = ""
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        host = ""

    blob = f"{host} {source} {source_type}"

    score = 0.55
    reason = "Generic source"

    if "airbus.com" in blob:
        score = 1.0
        reason = "Official Airbus source"
    elif any(x in blob for x in ["europa.eu", "nato.int", ".gov", "eda.europa.eu"]):
        score = 0.95
        reason = "Government, EU, NATO, or public institution source"
    elif any(x in blob for x in ["official", "company", "press", "release"]):
        score = 0.85
        reason = "Official company or press source"
    elif any(x in blob for x in ["defence", "defense", "aerospace", "aviation", "industry"]):
        score = 0.75
        reason = "Industry or domain-specialist source"
    elif any(x in blob for x in ["news", "rss", "media"]):
        score = 0.65
        reason = "News or media source"
    elif "unknown" in blob or not blob.strip():
        score = 0.35
        reason = "Unknown or missing source metadata"
    elif "error" in blob:
        score = 0.0
        reason = "Retrieval or source error"

    return {
        "score": round(score, 4),
        "reason": reason,
        "host": host,
        "source": source or "unknown",
        "source_type": source_type or "unknown",
    }


def _recency_score(e: Dict[str, Any]) -> Dict[str, Any]:
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
        return {
            "score": 0.6,
            "reason": "No usable date metadata, neutral recency score applied",
            "date": None,
        }

    for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
        try:
            dt = datetime.strptime(date_text[:19], fmt)
            age_days = (datetime.now(timezone.utc).replace(tzinfo=None) - dt).days

            if age_days <= 90:
                return {"score": 1.0, "reason": "Recent evidence", "date": date_text}
            if age_days <= 365:
                return {"score": 0.85, "reason": "Current-year evidence", "date": date_text}
            if age_days <= 900:
                return {"score": 0.65, "reason": "Older but still potentially useful", "date": date_text}
            return {"score": 0.45, "reason": "Old evidence", "date": date_text}
        except Exception:
            continue

    return {
        "score": 0.6,
        "reason": "Date metadata exists but could not be parsed",
        "date": date_text,
    }


def _build_evidence_catalog(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    evidence = state.get("evidence", []) or []
    catalog = []

    for i, e in enumerate(evidence[:12], start=1):
        text = str(e.get("text") or e.get("snippet") or _blob(e))
        credibility = _source_credibility(e)
        recency = _recency_score(e)

        catalog.append(
            {
                "evidence_id": f"E{i}",
                "title": e.get("title") or "Untitled evidence",
                "source": e.get("source") or "Unknown source",
                "url": e.get("url") or "",
                "source_type": e.get("source_type") or "unknown",
                "retrieval_score": e.get("score") or 0,
                "text": text[:1600],
                "tokens": list(_tokenize(text)),
                "credibility": credibility,
                "recency": recency,
            }
        )

    return catalog


def _map_claims_to_citations(state: Dict[str, Any]) -> Dict[str, Any]:
    briefing = state.get("briefing", "") or ""
    claims = _sentences(briefing)
    evidence_catalog = _build_evidence_catalog(state)

    mapped = []

    for idx, claim in enumerate(claims, start=1):
        claim_tokens = _tokenize(claim)
        candidates = []

        for ev in evidence_catalog:
            ev_tokens = set(ev.get("tokens", []))
            overlap = len(claim_tokens.intersection(ev_tokens)) / max(len(claim_tokens), 1)

            # Boost if critical entity appears in both.
            entity_boost = 0.0
            for ent in CRITICAL_ENTITIES:
                if re.search(rf"\b{re.escape(ent)}\b", claim, re.I) and re.search(
                    rf"\b{re.escape(ent)}\b", ev.get("text", ""), re.I
                ):
                    entity_boost += 0.05

            support_score = min(overlap + entity_boost, 1.0)

            if support_score > 0:
                candidates.append(
                    {
                        "evidence_id": ev["evidence_id"],
                        "title": ev["title"],
                        "source": ev["source"],
                        "url": ev["url"],
                        "support_score": round(support_score, 4),
                        "source_credibility": ev["credibility"]["score"],
                        "recency_score": ev["recency"]["score"],
                    }
                )

        candidates.sort(
            key=lambda x: (
                x["support_score"],
                x["source_credibility"],
                x["recency_score"],
            ),
            reverse=True,
        )

        top = candidates[:3]
        best_score = top[0]["support_score"] if top else 0.0

        management_sentence = any(
            phrase in claim.lower()
            for phrase in [
                "should", "management", "board", "kpi", "monitor", "90 days",
                "conservative", "balanced", "aggressive", "action plan",
            ]
        )
        threshold = 0.08 if management_sentence else 0.14

        mapped.append(
            {
                "claim_id": f"C{idx}",
                "claim": claim,
                "support_status": "supported" if best_score >= threshold else "weak_or_unsupported",
                "best_support_score": round(best_score, 4),
                "suggested_citations": top,
            }
        )

    supported = [m for m in mapped if m["support_status"] == "supported"]
    weak = [m for m in mapped if m["support_status"] != "supported"]

    coverage = len(supported) / len(mapped) if mapped else 1.0

    return {
        "support_coverage": round(coverage, 4),
        "total_claims": len(mapped),
        "supported_claims": len(supported),
        "weak_claims": len(weak),
        "claim_map": mapped,
        "evidence_catalog": [
            {
                "evidence_id": e["evidence_id"],
                "title": e["title"],
                "source": e["source"],
                "url": e["url"],
                "source_type": e["source_type"],
                "retrieval_score": e["retrieval_score"],
                "credibility": e["credibility"],
                "recency": e["recency"],
            }
            for e in evidence_catalog
        ],
    }


def _critical_terms(text: str) -> Set[str]:
    found = set()
    for ent in CRITICAL_ENTITIES:
        if re.search(rf"\b{re.escape(ent)}\b", text or "", re.I):
            found.add(ent)
    return found


def _numeric_claims(text: str) -> List[str]:
    claims = []
    for match in re.finditer(
        r"(?<![A-Za-z])(?:€|\$)?\d+(?:\.\d+)?\s?(?:%|percent|billion|million|bn|mn|years?|months?)?",
        text or "",
        re.I,
    ):
        value = match.group(0).strip()
        window = text[max(0, match.start() - 20): match.end() + 20].lower()

        if value in {"30", "90", "180"} and "day" in window:
            continue
        if value in {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}:
            continue

        claims.append(value)

    return sorted(set(claims))


def _detect_contradictions(state: Dict[str, Any]) -> Dict[str, Any]:
    evidence_text = "\n".join(_blob(e) for e in state.get("evidence", []) or [])
    signal_text = "\n".join(_blob(s) for s in state.get("signals", []) or [])
    briefing = state.get("briefing", "") or ""
    combined_context = f"{evidence_text}\n{signal_text}"

    findings = []

    context_sentences = _sentences(combined_context, limit=80)
    briefing_sentences = _sentences(briefing, limit=40)

    entities = sorted(_critical_terms(combined_context + "\n" + briefing))

    for entity in entities:
        entity_context = [
            s for s in context_sentences
            if re.search(rf"\b{re.escape(entity)}\b", s, re.I)
        ]
        entity_brief = [
            s for s in briefing_sentences
            if re.search(rf"\b{re.escape(entity)}\b", s, re.I)
        ]

        if not entity_context or not entity_brief:
            continue

        context_blob = " ".join(entity_context).lower()
        brief_blob = " ".join(entity_brief).lower()

        for pos_set, neg_set, label in CONTRADICTION_PAIRS:
            context_pos = any(p in context_blob for p in pos_set)
            context_neg = any(n in context_blob for n in neg_set)
            brief_pos = any(p in brief_blob for p in pos_set)
            brief_neg = any(n in brief_blob for n in neg_set)

            if (context_pos and brief_neg) or (context_neg and brief_pos):
                findings.append(
                    {
                        "entity": entity,
                        "type": label,
                        "context_signal": "positive" if context_pos else "negative",
                        "briefing_signal": "positive" if brief_pos else "negative",
                        "example_context": entity_context[:2],
                        "example_briefing": entity_brief[:2],
                    }
                )

    return {
        "contradictions_found": len(findings),
        "findings": findings[:8],
        "status": "CONTRADICTIONS_FOUND" if findings else "NO_CONTRADICTIONS_DETECTED",
    }


def _run_llm_judge(state: Dict[str, Any], claim_map: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optional LLM judge.

    Disabled by default. Enable with:
    LLM_JUDGE_ENABLED=true

    The judge is not trusted blindly. It provides another structured signal.
    """
    enabled = os.getenv("LLM_JUDGE_ENABLED", "false").lower() == "true"
    if not enabled:
        return {
            "enabled": False,
            "status": "SKIPPED",
            "reason": "LLM_JUDGE_ENABLED is false",
        }

    try:
        from app.agents.local_llm import get_local_llm

        evidence_pack = {
            "goal": state.get("goal"),
            "decision": state.get("decision"),
            "validation": state.get("validation"),
            "claim_map_summary": {
                "support_coverage": claim_map.get("support_coverage"),
                "weak_claims": claim_map.get("weak_claims"),
                "total_claims": claim_map.get("total_claims"),
            },
            "weak_claim_examples": [
                c for c in claim_map.get("claim_map", [])
                if c.get("support_status") != "supported"
            ][:6],
            "briefing": state.get("briefing", "")[:6000],
        }

        system_prompt = """
You are a strict verification judge for an Airbus strategic intelligence agent.

Return ONLY valid JSON with this schema:
{
  "status": "PASS" | "WARN" | "FAIL",
  "risk_level": "low" | "medium" | "high",
  "unsupported_claims": ["..."],
  "missing_citations": ["..."],
  "contradiction_notes": ["..."],
  "summary": "..."
}

Do not add markdown.
Do not add text outside JSON.
Judge whether the CEO briefing is sufficiently grounded in the provided claim map.
"""

        user_prompt = json.dumps(evidence_pack, ensure_ascii=False, indent=2)

        llm = get_local_llm()
        raw = llm.generate(
            [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ]
        )

        raw = (raw or "").strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("LLM judge did not return JSON object")

        parsed = json.loads(raw[start:end + 1])
        parsed["enabled"] = True
        return parsed

    except Exception as exc:
        return {
            "enabled": True,
            "status": "ERROR",
            "risk_level": "medium",
            "error": str(exc),
            "summary": "LLM judge failed. Deterministic verifier remains active.",
        }


def _write_observability_logs(state: Dict[str, Any], guardrails: Dict[str, Any]) -> Dict[str, Any]:
    """
    Local LangSmith/OpenTelemetry-style traces.

    This does not require external accounts. It writes JSONL logs that mimic
    trace/span concepts.
    """
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    run_id = state.get("run_id") or str(uuid.uuid4())
    trace_id = str(uuid.uuid4())

    event = {
        "timestamp": _now(),
        "trace_id": trace_id,
        "run_id": run_id,
        "goal": state.get("goal"),
        "intent": state.get("intent"),
        "topic": state.get("topic"),
        "validation_status": (state.get("validation") or {}).get("status"),
        "guardrail_status": guardrails.get("status"),
        "support_coverage": guardrails.get("claim_citation_map", {}).get("support_coverage"),
        "human_review_required": guardrails.get("human_approval", {}).get("required"),
    }

    spans = [
        {
            "trace_id": trace_id,
            "span_id": str(uuid.uuid4()),
            "name": "claim_citation_mapping",
            "timestamp": _now(),
            "attributes": {
                "claims": guardrails.get("claim_citation_map", {}).get("total_claims"),
                "support_coverage": guardrails.get("claim_citation_map", {}).get("support_coverage"),
            },
        },
        {
            "trace_id": trace_id,
            "span_id": str(uuid.uuid4()),
            "name": "contradiction_detection",
            "timestamp": _now(),
            "attributes": guardrails.get("contradiction_detection", {}),
        },
        {
            "trace_id": trace_id,
            "span_id": str(uuid.uuid4()),
            "name": "human_approval_routing",
            "timestamp": _now(),
            "attributes": guardrails.get("human_approval", {}),
        },
    ]

    with (logs_dir / "agent_guardrails.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

    with (logs_dir / "otel_like_spans.jsonl").open("a", encoding="utf-8") as f:
        for span in spans:
            f.write(json.dumps(span, ensure_ascii=False) + "\n")

    return {
        "trace_id": trace_id,
        "event_log": "logs/agent_guardrails.jsonl",
        "span_log": "logs/otel_like_spans.jsonl",
    }


def _human_approval_decision(
    state: Dict[str, Any],
    claim_map: Dict[str, Any],
    contradictions: Dict[str, Any],
    llm_judge: Dict[str, Any],
) -> Dict[str, Any]:
    validation = state.get("validation", {}) or {}

    reasons = []
    risk_level = "low"

    if validation.get("status") not in {"APPROVED", "VERIFIED"}:
        reasons.append(f"Validation status is {validation.get('status')}")
        risk_level = "medium"

    if claim_map.get("support_coverage", 1.0) < 0.65:
        reasons.append("Claim support coverage is below 0.65")
        risk_level = "high"

    if claim_map.get("weak_claims", 0) >= 5:
        reasons.append("Multiple weak or unsupported claims detected")
        risk_level = "high"

    if contradictions.get("contradictions_found", 0) > 0:
        reasons.append("Possible contradictions detected across evidence and briefing")
        risk_level = "high"

    if llm_judge.get("status") in {"FAIL", "ERROR"}:
        reasons.append(f"LLM judge status is {llm_judge.get('status')}")
        risk_level = "high" if llm_judge.get("status") == "FAIL" else risk_level

    if validation.get("confidence_band") == "low":
        reasons.append("Validation confidence band is low")
        risk_level = "high"

    required = bool(reasons)

    review_id = None
    if required:
        try:
            from app.db.approval_schema import create_human_review_request

            guardrail_snapshot = {
                "claim_support_coverage": claim_map.get("support_coverage"),
                "weak_claims": claim_map.get("weak_claims"),
                "contradictions": contradictions.get("contradictions_found"),
                "llm_judge_status": llm_judge.get("status"),
            }

            review_id = create_human_review_request(
                run_id=state.get("run_id", ""),
                goal=state.get("goal", ""),
                risk_level=risk_level,
                reason="; ".join(reasons),
                validation=validation,
                guardrails=guardrail_snapshot,
            )
        except Exception as exc:
            reasons.append(f"Could not create human review request: {exc}")

    return {
        "required": required,
        "risk_level": risk_level,
        "reasons": reasons,
        "review_id": review_id,
        "status": "PENDING_REVIEW" if required else "NOT_REQUIRED",
    }


def run_enterprise_guardrails(state: Dict[str, Any]) -> Dict[str, Any]:
    claim_map = _map_claims_to_citations(state)
    contradictions = _detect_contradictions(state)
    llm_judge = _run_llm_judge(state, claim_map)
    human_approval = _human_approval_decision(state, claim_map, contradictions, llm_judge)

    warnings = []
    issues = []

    if claim_map.get("support_coverage", 1.0) < 0.65:
        warnings.append("Claim support coverage is below target threshold.")

    if contradictions.get("contradictions_found", 0) > 0:
        warnings.append("Potential contradiction detected between evidence and generated briefing.")

    if llm_judge.get("status") in {"WARN", "FAIL", "ERROR"}:
        warnings.append(f"LLM judge returned {llm_judge.get('status')}.")

    if human_approval.get("required"):
        warnings.append("Human approval is required before treating this as a final board decision.")

    if claim_map.get("support_coverage", 1.0) < 0.45:
        issues.append("Claim support coverage is critically low.")

    status = "PASSED"
    if issues:
        status = "FAILED"
    elif warnings:
        status = "PASSED_WITH_WARNINGS"

    guardrails = {
        "status": status,
        "claim_citation_map": claim_map,
        "contradiction_detection": contradictions,
        "llm_judge": llm_judge,
        "human_approval": human_approval,
        "warnings": warnings,
        "issues": issues,
    }

    observability = _write_observability_logs(state, guardrails)
    guardrails["observability"] = observability

    return guardrails

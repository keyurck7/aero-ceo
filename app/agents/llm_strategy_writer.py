import json
from typing import Dict, List

from app.agents.local_llm import get_local_llm


def _short(text: str, limit: int = 700) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def build_evidence_pack(
    question: str,
    route: Dict,
    evidence: List[Dict],
    signals: List[Dict],
    recommendations: List[Dict],
    partners: List[Dict],
    confidence: float,
) -> str:
    evidence_items = []
    for idx, item in enumerate(evidence[:8], start=1):
        evidence_items.append(
            {
                "rank": idx,
                "title": item.get("title"),
                "source": item.get("source_name"),
                "source_type": item.get("source_type"),
                "topic": item.get("topic"),
                "url": item.get("url"),
                "retrieval_score": round(float(item.get("score") or 0), 3),
                "snippet": _short(item.get("chunk_text"), 500),
            }
        )

    signal_items = []
    for signal in signals[:8]:
        signal_items.append(
            {
                "signal_type": signal.get("signal_type"),
                "topic": signal.get("topic"),
                "title": signal.get("title"),
                "description": signal.get("description"),
                "confidence": signal.get("confidence_score"),
                "impact": signal.get("impact_score"),
                "urgency": signal.get("urgency_score"),
            }
        )

    recommendation_items = []
    for recommendation in recommendations[:5]:
        recommendation_items.append(
            {
                "title": recommendation.get("title"),
                "priority": recommendation.get("priority"),
                "confidence": recommendation.get("confidence_score"),
                "recommendation": recommendation.get("recommendation"),
                "expected_impact": recommendation.get("expected_impact"),
                "risk_assessment": recommendation.get("risk_assessment"),
            }
        )

    pack = {
        "question": question,
        "query_route": route,
        "answer_confidence_from_pipeline": confidence,
        "retrieved_evidence": evidence_items,
        "strategic_signals": signal_items,
        "existing_recommendations": recommendation_items,
        "partner_options": partners[:8],
    }

    return json.dumps(pack, indent=2, ensure_ascii=False)


def generate_llm_strategy_answer(
    question: str,
    route: Dict,
    evidence: List[Dict],
    signals: List[Dict],
    recommendations: List[Dict],
    partners: List[Dict],
    confidence: float,
    template_answer: str,
) -> str:
    evidence_pack = build_evidence_pack(
        question=question,
        route=route,
        evidence=evidence,
        signals=signals,
        recommendations=recommendations,
        partners=partners,
        confidence=confidence,
    )

    system_message = """
You are AERO-CEO, an evidence-grounded Strategic Intelligence Agent for Airbus SE.

Your role:
- Advise the CEO using retrieved evidence.
- Use only the evidence pack and clearly marked inference.
- Do not invent facts, numbers, events, or partnerships.
- If evidence is weak, say so clearly.
- Tie every recommendation to evidence.
- Write like an executive strategy advisor, not a news summarizer.

Required response structure:
1. Executive Answer
2. Strategic Recommendation
3. Evidence Basis
4. Decision Options: Conservative / Balanced / Aggressive
5. Risks and Mitigations
6. Partner or Capability Implications, if relevant
7. Confidence Explanation
8. Next 6-Month Actions
""".strip()

    user_message = f"""
CEO question:
{question}

Evidence pack:
{evidence_pack}

Baseline deterministic answer from the current AERO-CEO system:
{template_answer}

Now produce the final CEO-ready answer. Make it natural, strategic, and decision-oriented, but do not add unsupported facts.
""".strip()

    llm = get_local_llm()

    generated = llm.generate(
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
    )

    return f"""# AERO-CEO Strategic Answer

**Generation mode:** Local open-source LLM grounded by retrieval, strategic signals, recommendations, and evidence.

{generated}
"""

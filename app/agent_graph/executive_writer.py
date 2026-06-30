"""
Executive briefing writer for AERO-CEO Mission Control.

This is where the local LLM is used correctly:
- The graph retrieves evidence first.
- The graph analyzes risks/opportunities/trends.
- The graph creates a decision.
- The validation gate checks quality.
- Only then does Qwen write the CEO-facing briefing.

The LLM is not the orchestrator. It is the executive writer.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


def _short(value: Any, limit: int = 900) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def _compact_evidence(evidence: List[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    rows = []
    for item in evidence[:limit]:
        rows.append(
            {
                "rank": item.get("rank"),
                "score": item.get("score"),
                "title": item.get("title"),
                "source": item.get("source"),
                "source_type": item.get("source_type"),
                "topic": item.get("topic"),
                "url": item.get("url"),
                "snippet": _short(item.get("text"), 700),
            }
        )
    return rows


def _compact_items(items: List[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    compact = []
    for item in items[:limit]:
        compact.append(
            {
                "title": item.get("title") or item.get("signal_title") or item.get("recommendation"),
                "type": item.get("signal_type") or item.get("type"),
                "topic": item.get("topic") or item.get("strategic_topic"),
                "confidence": item.get("confidence_score"),
                "impact": item.get("impact_score"),
                "urgency": item.get("urgency_score"),
                "summary": _short(
                    item.get("description")
                    or item.get("evidence_text")
                    or item.get("recommendation")
                    or item,
                    600,
                ),
            }
        )
    return compact


def build_executive_evidence_pack(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a clean structured evidence pack for the LLM.

    This prevents dumping raw database rows into the model.
    """
    return {
        "ceo_goal": state.get("goal", ""),
        "agent_understanding": {
            "intent": state.get("intent", ""),
            "intent_confidence": state.get("intent_confidence", 0),
            "topic": state.get("topic", ""),
            "topic_confidence": state.get("topic_confidence", 0),
            "business_area": state.get("business_area", ""),
        },
        "decision": state.get("decision", {}) or {},
        "validation": state.get("validation", {}) or {},
        "retrieved_evidence": _compact_evidence(state.get("evidence", []) or []),
        "risks": _compact_items(state.get("risks", []) or []),
        "opportunities": _compact_items(state.get("opportunities", []) or []),
        "trends": _compact_items(state.get("trends", []) or []),
        "recommendation_portfolio": _compact_items(state.get("recommendations", []) or []),
        "partner_options": state.get("partners", []) or [],
    }


def generate_executive_briefing_with_llm(state: Dict[str, Any]) -> str:
    """
    Generate a detailed CEO-facing briefing using local Qwen.

    This function assumes retrieval, analysis, decision, and validation have already happened.
    """
    from app.agents.local_llm import get_local_llm

    evidence_pack = build_executive_evidence_pack(state)

    system_prompt = """
You are AERO-CEO, an evidence-grounded strategic intelligence advisor for the executive board of Airbus SE.

You are NOT allowed to invent facts.
Use only the evidence pack provided by the system.
If evidence is weak, say so clearly.
Do not explain internal workflow, tools, LangGraph, FAISS, embeddings, database, or implementation.
The CEO does not want system mechanics. The CEO wants strategic decision support.

Write like a senior strategy consultant briefing the CEO:
- direct
- detailed
- practical
- evidence-grounded
- decision-oriented
- no generic filler
- no software explanation

Your answer must be structured with these sections:

# CEO Strategic Briefing

## 1. Executive recommendation
Give the main recommendation in 3-5 strong sentences.

## 2. Strategic situation
Explain what is happening and why it matters for Airbus.

## 3. Evidence-backed reasoning
Use the retrieved evidence and signals. Mention source titles or source types where useful.

## 4. Recommended management actions
Give concrete actions. Make them specific, not vague.

## 5. 30 / 90 / 180 day action plan
Give a practical phased plan.

## 6. Partner and capability choices
If partners are relevant, compare them. If not relevant, say what capability Airbus should prioritize.

## 7. Risk assessment and mitigation
Explain key risks and how management should reduce them.

## 8. Expected impact
Explain strategic, operational, market, and capability impact.

## 9. Decision options
Give conservative, balanced, and aggressive options.

## 10. KPIs to monitor
Give measurable indicators management should track.

## 11. Board-level caution
State what should not be decided yet if evidence is incomplete.

## 12. Suggested CEO follow-up questions
Give 4 useful follow-up questions.

Rules:
- Do not say "the agent retrieved" or "the workflow shows".
- Do not mention validation mechanics except as governance confidence.
- Do not expose internal tool names.
- Do not write a short answer.
- Do not simply restate the recommendation portfolio.
- Synthesize the evidence into an actual strategic answer.
"""

    user_prompt = f"""
CEO goal:
{state.get("goal", "")}

Evidence pack:
{json.dumps(evidence_pack, ensure_ascii=False, indent=2)}

Now write the CEO-facing briefing.
"""

    llm = get_local_llm()
    answer = llm.generate(
        [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]
    )

    answer = (answer or "").strip()

    if len(answer) < 500:
        raise RuntimeError("LLM executive briefing was too short or empty.")

    return answer

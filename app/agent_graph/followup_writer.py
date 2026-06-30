"""
CEO follow-up insight writer.

Purpose:
- Do NOT rerun the full LangGraph workflow.
- Do NOT show validation, evidence tables, or workflow again.
- Use the existing briefing context and answer the CEO's follow-up directly.
"""

from __future__ import annotations

import json
from typing import Any, Dict


def _short(value: Any, limit: int = 5000) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def build_followup_context(state: Dict[str, Any], followup_question: str) -> Dict[str, Any]:
    """
    Build compact context for a CEO follow-up answer.

    This is not a new graph run. It uses the already produced agent state.
    """
    decision = state.get("decision", {}) or {}
    validation = state.get("validation", {}) or {}

    return {
        "original_ceo_goal": state.get("goal", ""),
        "followup_question": followup_question,
        "topic": state.get("topic", ""),
        "intent": state.get("intent", ""),
        "business_area": state.get("business_area", ""),
        "current_strategic_direction": decision.get("strategic_direction", ""),
        "expected_impact": decision.get("expected_impact", ""),
        "risk_assessment": decision.get("risk_assessment", ""),
        "decision_options": decision.get("decision_options", {}),
        "validation_status": validation.get("status", ""),
        "evidence_quality": validation.get("evidence_quality", ""),
        "existing_ceo_briefing": _short(state.get("briefing", ""), 7000),
        "partners": state.get("partners", [])[:6],
        "risks": state.get("risks", [])[:5],
        "opportunities": state.get("opportunities", [])[:5],
        "trends": state.get("trends", [])[:5],
    }


def deterministic_followup_answer(state: Dict[str, Any], followup_question: str) -> str:
    """
    Safe fallback if local LLM is unavailable.
    """
    decision = state.get("decision", {}) or {}
    topic = state.get("topic", "the strategic topic")

    strategic_direction = decision.get(
        "strategic_direction",
        "continue with an evidence-backed, staged strategic response",
    )
    risk_assessment = decision.get(
        "risk_assessment",
        "the main risk is acting before evidence, partners, and execution capacity are clear",
    )

    q = followup_question.strip()

    if "lower-risk" in q.lower() or "low risk" in q.lower():
        return f"""### Follow-up insight

A lower-risk version would be to keep the strategic direction but reduce commitment size and increase stage gates.

Airbus should **{strategic_direction[0].lower() + strategic_direction[1:]}**, but execute it in phases instead of making a large immediate commitment. The first phase should focus on evidence validation, partner alignment, technical feasibility, and procurement signals. The second phase should expand only if the early indicators are positive.

The main risk to control is: {risk_assessment}

A practical lower-risk move is to create a 90-day executive task force, validate the top two partnership or capability options, define clear exit criteria, and avoid locking Airbus into one partner or one architecture too early."""

    if "90" in q or "next 90" in q.lower():
        return f"""### Follow-up insight

In the next 90 days, Airbus should turn the recommendation into an executable management plan.

1. Assign an executive owner for {topic}.
2. Validate the strongest evidence and identify what is still uncertain.
3. Compare the top capability or partner options.
4. Define a staged investment path with clear go/no-go criteria.
5. Prepare a board-level decision memo with expected impact, risk, cost, and strategic urgency.

The goal of the next 90 days is not to make the biggest possible decision. It is to reduce uncertainty enough that Airbus can make the next decision with confidence."""

    if "partner" in q.lower():
        partners = state.get("partners", []) or []
        if partners:
            top = partners[0]
            return f"""### Follow-up insight

The first partner to prioritize should be **{top.get('partner')}** because its capability fit is strongest for the current strategic question.

Its main contribution would be: {top.get('capability')}

The best collaboration area is: {top.get('collaboration_area')}

However, Airbus should avoid single-partner dependency. The better approach is to define a lead partner for the first workstream while keeping a second option active for negotiation leverage and technical optionality."""
        return """### Follow-up insight

The partner choice should not be finalized yet. Airbus should first clarify which capability gap it wants to solve: combat aircraft architecture, sensors, secure communications, uncrewed systems, or military space. Once the capability gap is clear, partner selection becomes much safer."""

    return f"""### Follow-up insight

The key point is that Airbus should treat this as a staged strategic decision, not a one-shot commitment.

The current direction is: **{strategic_direction}**

For the follow-up question, the most important management action is to convert the recommendation into a practical decision path: clarify the capability gap, validate the evidence, compare options, define risk controls, and decide the next 90-day move.

The main caution remains: {risk_assessment}"""


def generate_followup_answer(state: Dict[str, Any], followup_question: str) -> str:
    """
    Generate focused CEO follow-up answer using existing state.

    The answer should be conversational and strategic, not a full new briefing.
    """
    followup_question = (followup_question or "").strip()
    if not followup_question:
        return "Please enter a follow-up question."

    try:
        from app.agents.local_llm import get_local_llm

        context = build_followup_context(state, followup_question)

        system_prompt = """
You are AERO-CEO, a senior strategic intelligence advisor to the Airbus CEO.

You are answering a follow-up question based on an existing CEO briefing.
Do NOT rerun or describe the workflow.
Do NOT include validation tables.
Do NOT list all evidence again.
Do NOT create a full CEO briefing template again.
Do NOT mention LangGraph, FAISS, SQLite, retrieval, embeddings, tools, or implementation.

Answer only the follow-up question.
Use the existing briefing context.
Be strategic, practical, and executive-facing.

Preferred answer style:
- Start with the direct answer.
- Then explain the strategic reasoning.
- Then give concrete management action.
- Include trade-offs or cautions only if useful.
- Keep it focused.
- No generic filler.
"""

        user_prompt = f"""
Existing briefing context:
{json.dumps(context, ensure_ascii=False, indent=2)}

CEO follow-up question:
{followup_question}

Write a focused follow-up answer only.
"""

        llm = get_local_llm()
        answer = llm.generate(
            [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ]
        )

        answer = (answer or "").strip()

        if len(answer) < 120:
            raise RuntimeError("Follow-up answer too short.")

        return answer

    except Exception as exc:
        fallback = deterministic_followup_answer(state, followup_question)
        fallback += f"\n\n*Note: Local LLM follow-up writer was unavailable, so a deterministic fallback was used. Error: {exc}*"
        return fallback

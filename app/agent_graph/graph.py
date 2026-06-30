"""
AERO-CEO deterministic LangGraph orchestration.

This is the new main agent workflow:

Goal -> Plan -> Retrieve -> Analyze -> Decide -> Recommend -> Validate -> Brief -> Save Trace

The graph controls the tools. The LLM does not randomly choose tools.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, List

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from app.agent_graph.intent_classifier import classify_goal
from app.agent_graph.state import AgentState
from app.agent_graph.tools import (
    analyze_partners,
    categorize_signals,
    lookup_recommendations,
    lookup_signals,
    semantic_retrieval,
)
from app.agent_graph.validators import validate_agent_output
from app.db.agent_trace_schema import ensure_agent_trace_tables, save_agent_trace
from app.db.local_db import get_connection, init_local_db

load_dotenv()


def _append_trace(state: AgentState, step: str, tool: str, status: str, observation: str) -> List[Dict[str, Any]]:
    trace = list(state.get("tool_trace", []) or [])
    trace.append(
        {
            "step": step,
            "tool": tool,
            "status": status,
            "observation": observation[:1000] if observation else "",
        }
    )
    return trace


def node_classify_goal(state: AgentState) -> Dict[str, Any]:
    goal = state.get("goal", "").strip()
    result = classify_goal(goal)

    return {
        "intent": result["intent"],
        "intent_confidence": result["intent_confidence"],
        "topic": result["topic"],
        "topic_confidence": result["topic_confidence"],
        "business_area": result["business_area"],
        "tool_trace": _append_trace(
            state,
            "Goal Understanding",
            "embedding_intent_classifier",
            "completed",
            json.dumps(result, ensure_ascii=False),
        ),
    }


def node_build_plan(state: AgentState) -> Dict[str, Any]:
    intent = state.get("intent", "ceo_briefing")
    topic = state.get("topic", "Airbus Strategy")

    base_plan = [
        {
            "step": 1,
            "name": "Classify CEO goal",
            "purpose": "Understand intent, topic, and business area.",
            "status": "completed",
        },
        {
            "step": 2,
            "name": "Retrieve evidence",
            "purpose": "Search FAISS vector memory for relevant Airbus evidence.",
            "status": "planned",
        },
        {
            "step": 3,
            "name": "Analyze strategic signals",
            "purpose": "Identify related risks, opportunities, and trends from SQLite signal memory.",
            "status": "planned",
        },
        {
            "step": 4,
            "name": "Decide strategic direction",
            "purpose": "Choose a management direction based on evidence and signals.",
            "status": "planned",
        },
        {
            "step": 5,
            "name": "Generate recommendation",
            "purpose": "Use existing evidence-backed recommendations or create a cautious decision recommendation.",
            "status": "planned",
        },
        {
            "step": 6,
            "name": "Validate before presenting",
            "purpose": "Check evidence sufficiency, source diversity, risks, impact, and confidence.",
            "status": "planned",
        },
        {
            "step": 7,
            "name": "Write CEO briefing",
            "purpose": "Produce executive answer only after retrieval, analysis, decision, and validation.",
            "status": "planned",
        },
    ]

    if intent == "partnership_strategy":
        base_plan.insert(
            4,
            {
                "step": 4.5,
                "name": "Analyze partner options",
                "purpose": "Evaluate European partner candidates and partnership risks.",
                "status": "planned",
            },
        )

    if intent == "scenario_analysis":
        base_plan.insert(
            4,
            {
                "step": 4.5,
                "name": "Analyze scenario implications",
                "purpose": "Convert scenario into risk, opportunity, and strategic response options.",
                "status": "planned",
            },
        )

    return {
        "plan": base_plan,
        "tool_trace": _append_trace(
            state,
            "Planning",
            "deterministic_plan_builder",
            "completed",
            f"Built plan for intent={intent}, topic={topic}",
        ),
    }


def node_retrieve_evidence(state: AgentState) -> Dict[str, Any]:
    goal = state.get("goal", "")
    topic = state.get("topic", "")
    business_area = state.get("business_area", "")

    retrieval_query = f"{goal} Airbus {topic} {business_area}".strip()
    evidence = semantic_retrieval(retrieval_query, top_k=8)

    return {
        "evidence": evidence,
        "tool_trace": _append_trace(
            state,
            "Retrieve",
            "semantic_retrieval_tool_FAISS",
            "completed",
            f"Retrieved {len(evidence)} evidence chunks for query: {retrieval_query}",
        ),
    }


def node_analyze_context(state: AgentState) -> Dict[str, Any]:
    goal = state.get("goal", "")
    topic = state.get("topic", "")
    intent = state.get("intent", "")

    signals = lookup_signals(topic=topic, query=goal, limit=12)
    categories = categorize_signals(signals)

    partners = []
    if intent == "partnership_strategy":
        partners = analyze_partners(query=goal, topic=topic)

    return {
        "signals": signals,
        "risks": categories["risks"],
        "opportunities": categories["opportunities"],
        "trends": categories["trends"],
        "partners": partners,
        "tool_trace": _append_trace(
            state,
            "Analyze",
            "signal_analysis_tool_SQLite",
            "completed",
            (
                f"Signals={len(signals)}, "
                f"risks={len(categories['risks'])}, "
                f"opportunities={len(categories['opportunities'])}, "
                f"trends={len(categories['trends'])}, "
                f"partners={len(partners)}"
            ),
        ),
    }


def node_decide_strategy(state: AgentState) -> Dict[str, Any]:
    goal = state.get("goal", "")
    intent = state.get("intent", "")
    topic = state.get("topic", "")
    risks = state.get("risks", []) or []
    opportunities = state.get("opportunities", []) or []
    trends = state.get("trends", []) or []
    partners = state.get("partners", []) or []

    recommendations = lookup_recommendations(topic=topic, query=goal, limit=6)

    top_rec = recommendations[0] if recommendations else {}

    if top_rec:
        strategic_direction = (
            top_rec.get("title")
            or top_rec.get("recommendation")
            or "Prioritize evidence-backed strategic action."
        )
        expected_impact = top_rec.get("expected_impact") or "Expected impact should be validated through follow-up analysis."
        risk_assessment = top_rec.get("risk_assessment") or "Risks should be monitored through the risk dashboard."
    else:
        if intent == "risk_analysis":
            strategic_direction = f"Reduce Airbus exposure in {topic} by prioritizing mitigation actions around the strongest retrieved risks."
        elif intent == "opportunity_analysis":
            strategic_direction = f"Prioritize Airbus growth options in {topic} where evidence shows strategic upside."
        elif intent == "partnership_strategy":
            if partners:
                strategic_direction = f"Use a selective partnership strategy for {topic}, prioritizing {partners[0]['partner']} while avoiding single-partner dependency."
            else:
                strategic_direction = f"Explore European partnership options for {topic}, but require stronger evidence before commitment."
        elif intent == "scenario_analysis":
            strategic_direction = f"Preserve strategic optionality in {topic} and prepare a phased response under scenario uncertainty."
        else:
            strategic_direction = f"Focus Airbus management attention on {topic} based on retrieved evidence, signals, and recommendation portfolio."

        expected_impact = "Improved strategic focus, stronger evidence-based prioritization, and clearer management response."
        risk_assessment = "Execution risk remains if evidence is incomplete, partner alignment is weak, or market signals change."

    decision = {
        "intent": intent,
        "topic": topic,
        "strategic_direction": strategic_direction,
        "expected_impact": expected_impact,
        "risk_assessment": risk_assessment,
        "decision_basis": {
            "risks_used": len(risks),
            "opportunities_used": len(opportunities),
            "trends_used": len(trends),
            "partners_used": len(partners),
            "recommendations_used": len(recommendations),
        },
        "decision_options": {
            "conservative": "Monitor and strengthen evidence before major commitment.",
            "balanced": "Move forward with staged investment and risk controls.",
            "aggressive": "Accelerate investment or partnership ahead of competitors, accepting higher execution risk.",
        },
    }

    return {
        "recommendations": recommendations,
        "decision": decision,
        "tool_trace": _append_trace(
            state,
            "Decide",
            "deterministic_decision_engine",
            "completed",
            json.dumps(decision, ensure_ascii=False),
        ),
    }


def node_validate_output(state: AgentState) -> Dict[str, Any]:
    validation = validate_agent_output(state)

    return {
        "validation": validation,
        "status": validation["status"],
        "tool_trace": _append_trace(
            state,
            "Validate",
            "evidence_quality_gate",
            validation["status"],
            json.dumps(validation, ensure_ascii=False),
        ),
    }


def _format_items(items: List[Dict[str, Any]], label_keys: List[str], max_items: int = 5) -> str:
    lines = []
    for i, item in enumerate(items[:max_items], start=1):
        label = None
        for key in label_keys:
            if item.get(key):
                label = item.get(key)
                break
        if not label:
            label = str(item)[:160]
        lines.append(f"{i}. {label}")
    return "\n".join(lines) if lines else "None found."


def _deterministic_briefing(state: AgentState) -> str:
    goal = state.get("goal", "")
    intent = state.get("intent", "")
    topic = state.get("topic", "")
    business_area = state.get("business_area", "")
    evidence = state.get("evidence", []) or []
    risks = state.get("risks", []) or []
    opportunities = state.get("opportunities", []) or []
    trends = state.get("trends", []) or []
    partners = state.get("partners", []) or []
    recommendations = state.get("recommendations", []) or []
    decision = state.get("decision", {}) or {}
    validation = state.get("validation", {}) or {}
    plan = state.get("plan", []) or []

    evidence_lines = []
    for e in evidence[:5]:
        title = e.get("title", "Untitled")
        source = e.get("source", "Unknown")
        score = e.get("score", 0)
        url = e.get("url", "")
        evidence_lines.append(f"- **{title}** | Source: {source} | Score: {score} | {url}")

    if not evidence_lines:
        evidence_lines.append("- No evidence retrieved.")

    partner_section = ""
    if partners:
        partner_rows = []
        for p in partners[:5]:
            partner_rows.append(
                f"- **{p.get('partner')}** ({p.get('country')}): {p.get('capability')}. "
                f"Use case: {p.get('collaboration_area')}. Risk: {p.get('risk')}."
            )
        partner_section = "\n\n## Partner options\n" + "\n".join(partner_rows)

    validation_status = validation.get("status", "UNKNOWN")
    validation_warning = ""
    if validation.get("issues") or validation.get("warnings"):
        validation_warning = "\n\n## Validation notes\n"
        for issue in validation.get("issues", []):
            validation_warning += f"- Issue: {issue}\n"
        for warning in validation.get("warnings", []):
            validation_warning += f"- Warning: {warning}\n"

    plan_lines = "\n".join(
        [
            f"{p.get('step')}. {p.get('name')} - {p.get('purpose')}"
            for p in plan
        ]
    )

    briefing = f"""# AERO-CEO Mission Control Briefing

## CEO goal
{goal}

## Agent understanding
- Intent: **{intent}**
- Topic: **{topic}**
- Business area: **{business_area}**

## Execution plan
{plan_lines}

## What happened?
The agent retrieved Airbus-related evidence, analyzed strategic signals, checked existing recommendations, and produced a decision through a deterministic LangGraph workflow.

## Why does it matter?
The evidence and signal memory indicate that **{topic}** is strategically relevant for Airbus management. The decision should be treated as evidence-grounded but still subject to validation status: **{validation_status}**.

## What should management do next?
**{decision.get("strategic_direction", "No strategic direction generated.")}**

## Expected impact
{decision.get("expected_impact", "No expected impact generated.")}

## Risk assessment
{decision.get("risk_assessment", "No risk assessment generated.")}

## Decision options
- **Conservative:** {decision.get("decision_options", {}).get("conservative", "Not available.")}
- **Balanced:** {decision.get("decision_options", {}).get("balanced", "Not available.")}
- **Aggressive:** {decision.get("decision_options", {}).get("aggressive", "Not available.")}

## Risks detected
{_format_items(risks, ["title", "signal_title", "description", "evidence_text"])}

## Opportunities detected
{_format_items(opportunities, ["title", "signal_title", "description", "evidence_text"])}

## Trends detected
{_format_items(trends, ["title", "signal_title", "description", "evidence_text"])}

## Matching recommendation portfolio
{_format_items(recommendations, ["title", "recommendation"])}

{partner_section}

## Supporting evidence
{chr(10).join(evidence_lines)}

## Validation result
- Status: **{validation_status}**
- Confidence: **{validation.get("confidence", 0)}**
- Evidence quality: **{validation.get("evidence_quality", "unknown")}**
- Human review required: **{validation.get("human_review_required", True)}**
{validation_warning}
"""
    return briefing


def node_write_briefing(state: AgentState) -> Dict[str, Any]:
    """
    Write final CEO briefing.

    By default, this is deterministic markdown.
    If AGENT_GRAPH_LLM_ENABLED=true, we optionally let the local LLM polish the briefing,
    but only after evidence, decision, and validation exist.
    """
    briefing = _deterministic_briefing(state)

    llm_enabled = os.getenv("AGENT_GRAPH_LLM_ENABLED", "false").lower() == "true"

    if llm_enabled:
        try:
            from app.agents.local_llm import get_local_llm

            llm = get_local_llm()
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are AERO-CEO, an evidence-grounded strategic intelligence agent. "
                        "Rewrite the provided deterministic briefing into a polished CEO briefing. "
                        "Do not add facts not present in the briefing. Keep validation warnings."
                    ),
                },
                {
                    "role": "user",
                    "content": briefing,
                },
            ]
            polished = llm.generate(messages)
            if polished and len(polished.strip()) > 100:
                briefing = polished.strip()
        except Exception as exc:
            briefing += f"\n\n## LLM polishing note\nLocal LLM polishing failed, so deterministic briefing was used. Error: {exc}\n"

    return {
        "briefing": briefing,
        "tool_trace": _append_trace(
            state,
            "Brief",
            "ceo_briefing_writer",
            "completed",
            f"Generated briefing. LLM polishing enabled={llm_enabled}",
        ),
    }


def node_save_trace(state: AgentState) -> Dict[str, Any]:
    init_local_db()
    conn = get_connection()
    ensure_agent_trace_tables(conn)
    trace_id = save_agent_trace(state, conn)
    conn.close()

    return {
        "tool_trace": _append_trace(
            state,
            "Memory",
            "sqlite_agent_trace_memory",
            "completed",
            f"Saved agent trace id={trace_id}",
        ),
    }


def build_aero_ceo_graph():
    """
    Build and compile the deterministic LangGraph workflow.
    """
    graph = StateGraph(AgentState)

    graph.add_node("classify_goal", node_classify_goal)
    graph.add_node("build_plan", node_build_plan)
    graph.add_node("retrieve_evidence", node_retrieve_evidence)
    graph.add_node("analyze_context", node_analyze_context)
    graph.add_node("decide_strategy", node_decide_strategy)
    graph.add_node("validate_output", node_validate_output)
    graph.add_node("write_briefing", node_write_briefing)
    graph.add_node("save_trace", node_save_trace)

    graph.add_edge(START, "classify_goal")
    graph.add_edge("classify_goal", "build_plan")
    graph.add_edge("build_plan", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "analyze_context")
    graph.add_edge("analyze_context", "decide_strategy")
    graph.add_edge("decide_strategy", "validate_output")
    graph.add_edge("validate_output", "write_briefing")
    graph.add_edge("write_briefing", "save_trace")
    graph.add_edge("save_trace", END)

    return graph.compile()


def run_agent_goal(goal: str) -> AgentState:
    """
    Run the full deterministic AERO-CEO graph for one CEO goal.
    """
    app = build_aero_ceo_graph()

    initial_state: AgentState = {
        "goal": goal,
        "run_id": str(uuid.uuid4()),
        "tool_trace": [],
        "status": "STARTED",
    }

    final_state = app.invoke(initial_state)
    return final_state

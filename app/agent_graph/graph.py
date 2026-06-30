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
        "scope_status": result.get("scope_status", "in_scope"),
        "domain_relevance": result.get("domain_relevance", 0.0),
        "rejection_reason": result.get("rejection_reason", ""),
        "tool_trace": _append_trace(
            state,
            "Goal Understanding",
            "Goal Understanding Tool",
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
            "Planning Tool",
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
            "Semantic Retrieval Tool",
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
            "Strategic Signal Analysis Tool",
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
            "Decision Engine",
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
            "Evidence Quality Validation Gate",
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
    """
    Create a CEO-facing briefing.

    This is for the CEO, not for debugging. Internal workflow details belong
    in the Agent Workflow tab.
    """
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

    strategic_direction = decision.get(
        "strategic_direction",
        "Prioritize an evidence-backed strategic response."
    )
    expected_impact = decision.get(
        "expected_impact",
        "Expected impact should be validated through follow-up analysis."
    )
    risk_assessment = decision.get(
        "risk_assessment",
        "Risks should be monitored through the risk dashboard."
    )

    options = decision.get("decision_options", {}) or {}

    validation_status = validation.get("status", "UNKNOWN")
    confidence = validation.get("confidence", 0)
    evidence_quality = validation.get("evidence_quality", "unknown")
    human_review = validation.get("human_review_required", True)

    def pick_label(item):
        for key in ["title", "signal_title", "recommendation", "description", "evidence_text"]:
            value = item.get(key)
            if value:
                return str(value)
        return str(item)[:180]

    def numbered(items, max_items=4):
        if not items:
            return "No strong items found in current memory."
        lines = []
        for i, item in enumerate(items[:max_items], start=1):
            lines.append(f"{i}. {pick_label(item)}")
        return "\n".join(lines)

    evidence_lines = []
    for e in evidence[:5]:
        title = e.get("title", "Untitled evidence")
        source = e.get("source", "Unknown source")
        score = e.get("score", 0)
        url = e.get("url", "")

        line = f"- **{title}** | Source: {source} | Relevance: {score}"
        if url:
            line += f" | {url}"
        evidence_lines.append(line)

    if not evidence_lines:
        evidence_lines.append("- No evidence retrieved.")

    partner_section = ""
    if partners:
        partner_lines = []
        for p in partners[:5]:
            partner_lines.append(
                "- **{partner}** ({country}): {capability}. Best use: {area}. Main risk: {risk}.".format(
                    partner=p.get("partner", "Unknown partner"),
                    country=p.get("country", "Unknown country"),
                    capability=p.get("capability", "Capability not available"),
                    area=p.get("collaboration_area", "Collaboration area not available"),
                    risk=p.get("risk", "Risk not available"),
                )
            )
        partner_section = "\n## Partnership options\n" + "\n".join(partner_lines)

    recommendation_portfolio = ""
    if recommendations:
        rec_lines = []
        for i, rec in enumerate(recommendations[:5], start=1):
            rec_title = rec.get("title") or rec.get("recommendation") or "Untitled recommendation"
            rec_lines.append(f"{i}. {rec_title}")
        recommendation_portfolio = "\n## Related recommendation portfolio\n" + "\n".join(rec_lines)

    validation_note = ""
    if validation.get("issues") or validation.get("warnings"):
        notes = ["\n## Management caution"]
        for issue in validation.get("issues", []):
            notes.append(f"- Evidence issue: {issue}")
        for warning in validation.get("warnings", []):
            notes.append(f"- Warning: {warning}")
        validation_note = "\n".join(notes)

    opening_action = strategic_direction
    if opening_action:
        opening_action = opening_action[0].lower() + opening_action[1:]

    sections = [
        "# CEO Strategic Briefing",
        "",
        "## CEO question",
        goal,
        "",
        "## Executive answer",
        f"Airbus should **{opening_action}**.",
        "",
        (
            f"The strategic context is **{intent}** around **{topic}** in **{business_area}**. "
            f"The recommendation currently has validation status **{validation_status}**, "
            f"confidence **{confidence}**, and evidence quality **{evidence_quality}**."
        ),
        "",
        "## What management should do next",
        f"**{strategic_direction}**",
        "",
        "## Why this matters now",
        (
            f"{topic} is strategically relevant for Airbus because it affects future defence positioning, "
            "industrial execution, partner alignment, and European defence competitiveness."
        ),
        "",
        "## Expected impact",
        expected_impact,
        "",
        "## Risk assessment",
        risk_assessment,
        "",
        "## Decision options",
        f"- **Conservative:** {options.get('conservative', 'Monitor the situation and strengthen evidence before major commitment.')}",
        f"- **Balanced:** {options.get('balanced', 'Move forward with staged investment and risk controls.')}",
        f"- **Aggressive:** {options.get('aggressive', 'Accelerate investment or partnership ahead of competitors while accepting higher execution risk.')}",
        "",
        "## Key risks to monitor",
        numbered(risks),
        "",
        "## Key opportunities",
        numbered(opportunities),
        "",
        "## Relevant trends",
        numbered(trends),
        partner_section,
        recommendation_portfolio,
        "",
        "## Supporting evidence",
        "\n".join(evidence_lines),
        "",
        "## Confidence and governance",
        f"- Validation status: **{validation_status}**",
        f"- Evidence quality: **{evidence_quality}**",
        f"- Human review required: **{human_review}**",
        "",
        "The full execution plan, tool calls, and validation trace are available in the **Agent Workflow** tab.",
        validation_note,
    ]

    return "\n".join([s for s in sections if s is not None])



def node_write_briefing(state: AgentState) -> Dict[str, Any]:
    """
    Write the final CEO briefing.

    Product rule:
    - Workflow/proof stays in Agent Workflow tab.
    - CEO Briefing tab gets a board-level strategic answer.
    - If AGENT_GRAPH_LLM_ENABLED=true, Qwen writes the final answer from the evidence pack.
    - If the LLM fails or is disabled, we fall back to deterministic briefing.
    """
    llm_enabled = os.getenv("AGENT_GRAPH_LLM_ENABLED", "false").lower() == "true"

    if llm_enabled:
        try:
            from app.agent_graph.executive_writer import generate_executive_briefing_with_llm

            briefing = generate_executive_briefing_with_llm(state)

            return {
                "briefing": briefing,
                "tool_trace": _append_trace(
                    state,
                    "Brief",
                    "Executive Writer Tool",
                    "completed",
                    "Generated CEO-facing briefing from structured evidence pack.",
                ),
            }

        except Exception as exc:
            fallback = _deterministic_briefing(state)
            fallback += (
                "\n\n## Generation fallback note\n"
                f"Local LLM executive writing failed, so deterministic briefing was used. Error: {exc}\n"
            )

            return {
                "briefing": fallback,
                "tool_trace": _append_trace(
                    state,
                    "Brief",
                    "Executive Writer Fallback",
                    "warning",
                    f"LLM executive writer failed: {exc}",
                ),
            }

    briefing = _deterministic_briefing(state)

    return {
        "briefing": briefing,
        "tool_trace": _append_trace(
            state,
            "Brief",
            "Deterministic Executive Writer",
            "completed",
            "Generated deterministic CEO-facing briefing. LLM disabled.",
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
            "Trace Memory Tool",
            "completed",
            f"Saved agent trace id={trace_id}",
        ),
    }




def node_out_of_scope_response(state: AgentState) -> Dict[str, Any]:
    """
    Stop the workflow for prompts outside Airbus strategic intelligence scope.
    This prevents forced retrieval and fake approvals for unrelated prompts.
    """
    goal = state.get("goal", "")
    reason = state.get("rejection_reason", "Goal is outside AERO-CEO scope.")
    domain_relevance = state.get("domain_relevance", 0.0)

    plan = [
        {
            "step": 1,
            "name": "Classify CEO goal",
            "purpose": "Understand whether the request belongs to Airbus strategic intelligence scope.",
            "status": "completed",
        },
        {
            "step": 2,
            "name": "Domain Scope Guard",
            "purpose": "Reject unrelated prompts before retrieval, analysis, or LLM generation.",
            "status": "rejected",
        },
    ]

    validation = {
        "status": "REJECTED_OUT_OF_SCOPE",
        "confidence": 0.0,
        "evidence_quality": "not_applicable",
        "evidence_count": 0,
        "signal_count": 0,
        "recommendation_count": 0,
        "source_diversity": 0,
        "source_type_diversity": 0,
        "issues": [reason],
        "warnings": [],
        "human_review_required": False,
    }

    briefing = f"""# Request Outside AERO-CEO Scope

## CEO goal received
{goal}

## Decision
This request was **not processed** because it is outside the scope of AERO-CEO.

## Why it was rejected
{reason}

## Scope of this agent
AERO-CEO is designed for Airbus strategic intelligence and CEO decision support, especially:

- Airbus Defence and Space
- FCAS and future combat systems
- Eurofighter
- Uncrewed systems and drones
- Military space and secure communications
- Aerial refuelling and military transport
- European defence autonomy
- Supply chain and delivery risk
- Strategic risks, opportunities, trends, partnerships, and recommendations

## Example valid questions
- What should Airbus do next for FCAS and uncrewed systems?
- Which European organization should Airbus collaborate with for sixth-generation fighter systems?
- What are the biggest risks for Airbus Defence and Space?
- What opportunities should Airbus prioritize in military space?

## Governance note
The workflow stopped before retrieval and before LLM briefing generation. This prevents AERO-CEO from producing unsupported answers for unrelated prompts.

Domain relevance score: **{domain_relevance}**
"""

    return {
        "plan": plan,
        "evidence": [],
        "signals": [],
        "risks": [],
        "opportunities": [],
        "trends": [],
        "recommendations": [],
        "partners": [],
        "decision": {
            "strategic_direction": "Reject out-of-scope request.",
            "expected_impact": "Prevents unsupported or misleading strategic answers.",
            "risk_assessment": "Low risk because no unrelated recommendation is generated.",
            "decision_options": {
                "conservative": "Reject unrelated prompt and ask for an Airbus strategic question.",
                "balanced": "Provide examples of valid Airbus strategic questions.",
                "aggressive": "Do not answer unrelated prompts under AERO-CEO branding.",
            },
        },
        "validation": validation,
        "status": "REJECTED_OUT_OF_SCOPE",
        "briefing": briefing,
        "tool_trace": _append_trace(
            state,
            "Scope Check",
            "Domain Scope Guard",
            "REJECTED_OUT_OF_SCOPE",
            reason,
        ),
    }


def _route_after_scope_check(state: AgentState) -> str:
    """
    Conditional graph route after goal classification.
    """
    if state.get("scope_status") == "out_of_scope":
        return "out_of_scope"
    return "in_scope"



def build_aero_ceo_graph():
    """
    Build and compile the deterministic LangGraph workflow.
    """
    graph = StateGraph(AgentState)

    graph.add_node("classify_goal", node_classify_goal)
    graph.add_node("out_of_scope_response", node_out_of_scope_response)
    graph.add_node("build_plan", node_build_plan)
    graph.add_node("retrieve_evidence", node_retrieve_evidence)
    graph.add_node("analyze_context", node_analyze_context)
    graph.add_node("decide_strategy", node_decide_strategy)
    graph.add_node("validate_output", node_validate_output)
    graph.add_node("write_briefing", node_write_briefing)
    graph.add_node("save_trace", node_save_trace)

    graph.add_edge(START, "classify_goal")
    graph.add_conditional_edges(
        "classify_goal",
        _route_after_scope_check,
        {
            "out_of_scope": "out_of_scope_response",
            "in_scope": "build_plan",
        },
    )
    graph.add_edge("out_of_scope_response", "save_trace")
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

"""
AERO-CEO Mission Control Dashboard.

This is the new executive-facing entry point for the deterministic LangGraph agent.

Main flow:
CEO Goal -> Plan -> Retrieve -> Analyze -> Decide -> Recommend -> Validate -> Brief
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from app.agent_graph.graph import run_agent_goal
from app.db.local_db import get_connection, init_local_db
from app.db.agent_trace_schema import ensure_agent_trace_tables


st.set_page_config(
    page_title="AERO-CEO Mission Control",
    page_icon="🛩️",
    layout="wide",
)


DEMO_GOALS = [
    "If you were Airbus CEO today, what should we do next for FCAS and uncrewed systems?",
    "Which European organization should Airbus collaborate with for sixth-generation fighter systems?",
    "What should Airbus do if FCAS is delayed by five years and drone demand accelerates?",
    "What are the biggest risks for Airbus Defence and Space right now?",
    "What opportunities should Airbus prioritize in military space and secure communications?",
    "What should Airbus do to reduce supply chain and delivery execution risk?",
]


def _safe_json_loads(value: str):
    try:
        return json.loads(value or "[]")
    except Exception:
        return []


def get_system_counts() -> Dict[str, int]:
    init_local_db()
    conn = get_connection()
    ensure_agent_trace_tables(conn)

    counts = {}
    tables = [
        "documents",
        "sources",
        "document_chunks",
        "signals",
        "recommendations",
        "recommendation_evidence",
        "ceo_queries",
        "agent_traces",
    ]

    for table in tables:
        try:
            cur = conn.execute(f"SELECT COUNT(*) AS n FROM {table}")
            counts[table] = int(cur.fetchone()["n"])
        except Exception:
            counts[table] = 0

    conn.close()
    return counts


def get_recent_traces(limit: int = 5) -> List[Dict[str, Any]]:
    init_local_db()
    conn = get_connection()
    ensure_agent_trace_tables(conn)

    try:
        cur = conn.execute(
            """
            SELECT id, run_id, goal, intent, topic, business_area, status,
                   evidence_count, signal_count, recommendation_count,
                   validation_json, created_at
            FROM agent_traces
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = [dict(r) for r in cur.fetchall()]
    except Exception:
        rows = []

    conn.close()
    return rows


def render_plan(plan: List[Dict[str, Any]]) -> None:
    st.subheader("1. Agent plan")

    if not plan:
        st.info("No plan generated yet.")
        return

    for step in plan:
        with st.container(border=True):
            st.markdown(f"**Step {step.get('step')}: {step.get('name')}**")
            st.caption(step.get("purpose", ""))
            st.write(f"Status: `{step.get('status', 'planned')}`")


def render_tool_trace(tool_trace: List[Dict[str, Any]]) -> None:
    st.subheader("2. Tool execution trace")

    if not tool_trace:
        st.info("No tool trace generated yet.")
        return

    for item in tool_trace:
        status = item.get("status", "")
        step = item.get("step", "")
        tool = item.get("tool", "")
        observation = item.get("observation", "")

        with st.expander(f"{step} → {tool} [{status}]", expanded=False):
            st.write(observation)


def render_validation(validation: Dict[str, Any]) -> None:
    st.subheader("3. Validation gate")

    if not validation:
        st.info("No validation result yet.")
        return

    status = validation.get("status", "UNKNOWN")
    confidence = validation.get("confidence", 0)
    evidence_quality = validation.get("evidence_quality", "unknown")

    if status == "APPROVED":
        st.success(f"Validation status: {status}")
    elif status == "APPROVED_WITH_WARNING":
        st.warning(f"Validation status: {status}")
    else:
        st.error(f"Validation status: {status}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confidence", confidence)
    c2.metric("Evidence quality", evidence_quality)
    c3.metric("Evidence count", validation.get("evidence_count", 0))
    c4.metric("Source diversity", validation.get("source_diversity", 0))

    issues = validation.get("issues", []) or []
    warnings = validation.get("warnings", []) or []

    if issues:
        st.markdown("**Issues**")
        for issue in issues:
            st.error(issue)

    if warnings:
        st.markdown("**Warnings**")
        for warning in warnings:
            st.warning(warning)

    st.write("Human review required:", validation.get("human_review_required", True))


def render_evidence(evidence: List[Dict[str, Any]]) -> None:
    st.subheader("4. Retrieved evidence")

    if not evidence:
        st.info("No evidence retrieved.")
        return

    table_rows = []
    for e in evidence:
        table_rows.append(
            {
                "rank": e.get("rank"),
                "score": e.get("score"),
                "title": e.get("title"),
                "source": e.get("source"),
                "source_type": e.get("source_type"),
                "topic": e.get("topic"),
                "url": e.get("url"),
            }
        )

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    for e in evidence[:5]:
        with st.expander(f"Evidence {e.get('rank')}: {e.get('title')}"):
            st.write("Source:", e.get("source"))
            st.write("Score:", e.get("score"))
            st.write("URL:", e.get("url"))
            st.write(e.get("text", ""))


def render_signals(state: Dict[str, Any]) -> None:
    st.subheader("5. Strategic analysis")

    risks = state.get("risks", []) or []
    opportunities = state.get("opportunities", []) or []
    trends = state.get("trends", []) or []
    partners = state.get("partners", []) or []

    t1, t2, t3, t4 = st.tabs(["Risks", "Opportunities", "Trends", "Partners"])

    with t1:
        if risks:
            st.dataframe(pd.DataFrame(risks), use_container_width=True)
        else:
            st.info("No risk signals found.")

    with t2:
        if opportunities:
            st.dataframe(pd.DataFrame(opportunities), use_container_width=True)
        else:
            st.info("No opportunity signals found.")

    with t3:
        if trends:
            st.dataframe(pd.DataFrame(trends), use_container_width=True)
        else:
            st.info("No trend signals found.")

    with t4:
        if partners:
            st.dataframe(pd.DataFrame(partners), use_container_width=True)
        else:
            st.info("No partner analysis produced for this goal.")


def render_decision(decision: Dict[str, Any]) -> None:
    st.subheader("6. Decision layer")

    if not decision:
        st.info("No decision generated.")
        return

    st.markdown("### Strategic direction")
    st.write(decision.get("strategic_direction", "Not available."))

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Expected impact")
        st.write(decision.get("expected_impact", "Not available."))

    with c2:
        st.markdown("### Risk assessment")
        st.write(decision.get("risk_assessment", "Not available."))

    st.markdown("### Decision options")
    options = decision.get("decision_options", {}) or {}
    st.write("**Conservative:**", options.get("conservative", "Not available."))
    st.write("**Balanced:**", options.get("balanced", "Not available."))
    st.write("**Aggressive:**", options.get("aggressive", "Not available."))


def render_followup_chat(state: Dict[str, Any]) -> None:
    """
    CEO-facing follow-up insight chat.

    Important:
    This does not rerun the full LangGraph workflow.
    It uses the current briefing context to answer a focused follow-up question.
    """
    from app.agent_graph.followup_writer import generate_followup_answer

    st.divider()
    st.subheader("CEO follow-up")

    st.caption(
        "Ask for more insight based on this briefing. This is a focused CEO follow-up, "
        "not a new full workflow run."
    )

    suggested_questions = [
        "What is the lower-risk version of this strategy?",
        "What should Airbus do in the next 90 days?",
        "Which partner should Airbus prioritize first and why?",
        "What could go wrong with this recommendation?",
        "What decision should the board make first?",
        "What should we avoid doing too early?",
    ]

    selected = st.selectbox(
        "Suggested follow-up",
        suggested_questions,
        key="followup_suggested_question",
    )

    followup = st.text_area(
        "Follow-up question",
        value=selected,
        height=90,
        key="ceo_followup_question",
    )

    if st.button("Ask follow-up", key="ask_ceo_followup"):
        if not followup.strip():
            st.error("Please enter a follow-up question.")
            return

        with st.spinner("Generating focused CEO follow-up insight..."):
            answer = generate_followup_answer(state, followup.strip())

        st.session_state.setdefault("ceo_followup_history", [])
        st.session_state["ceo_followup_history"].append(
            {
                "question": followup.strip(),
                "answer": answer,
            }
        )

    history = st.session_state.get("ceo_followup_history", [])
    if history:
        st.markdown("### Follow-up insights")
        for i, item in enumerate(reversed(history[-5:]), start=1):
            with st.expander(f"Follow-up {i}: {item.get('question')}", expanded=i == 1):
                st.markdown(item.get("answer", "No answer generated."))



def main() -> None:
    st.title("AERO-CEO Mission Control")
    st.caption("Deterministic LangGraph Strategic Intelligence Agent for Airbus SE")

    counts = get_system_counts()

    with st.sidebar:
        st.header("System status")
        st.metric("Documents", counts.get("documents", 0))
        st.metric("Signals", counts.get("signals", 0))
        st.metric("Recommendations", counts.get("recommendations", 0))
        st.metric("Agent traces", counts.get("agent_traces", 0))

        st.divider()
        st.markdown("### Required workflow")
        st.code("Goal → Plan → Retrieve → Analyze → Decide → Recommend → Validate")

        st.divider()
        st.markdown("### Architecture")
        st.write("Orchestration: LangGraph")
        st.write("Retrieval: FAISS + BGE")
        st.write("Memory: SQLite")
        st.write("LLM: Qwen optional final writer")

    st.markdown(
        """
        This page is the new main cockpit. The CEO enters a strategic goal, and the system runs
        a deterministic agent workflow with explicit planning, tool usage, evidence retrieval,
        analysis, decision-making, validation, and briefing.
        """
    )

    selected_goal = st.selectbox("Choose a demo CEO goal", DEMO_GOALS)
    goal = st.text_area("CEO strategic goal", value=selected_goal, height=100)

    run = st.button("Run AERO-CEO Mission Control", type="primary")

    if run:
        if not goal.strip():
            st.error("Please enter a CEO goal.")
            return

        with st.spinner("Running deterministic LangGraph agent..."):
            state = run_agent_goal(goal.strip())

        st.session_state["mission_control_state"] = state

    state = st.session_state.get("mission_control_state")

    if not state:
        st.info("Enter a CEO goal and run the agent to see the full workflow trace.")
    else:
        st.success("Agent run completed.")

        top1, top2, top3, top4 = st.columns(4)
        top1.metric("Intent", state.get("intent", ""))
        top2.metric("Topic", state.get("topic", ""))
        top3.metric("Validation", state.get("validation", {}).get("status", ""))
        top4.metric("Run ID", str(state.get("run_id", ""))[:8])

        st.divider()

        tab_brief, tab_workflow, tab_analysis, tab_evidence, tab_memory = st.tabs(
            [
                "CEO Briefing",
                "Agent Workflow",
                "Analysis & Decision",
                "Evidence",
                "Trace Memory",
            ]
        )

        with tab_brief:
            st.markdown(state.get("briefing", ""))
            render_followup_chat(state)

        with tab_workflow:
            render_plan(state.get("plan", []))
            st.divider()
            render_tool_trace(state.get("tool_trace", []))
            st.divider()
            render_validation(state.get("validation", {}))

        with tab_analysis:
            render_signals(state)
            st.divider()
            render_decision(state.get("decision", {}))

        with tab_evidence:
            render_evidence(state.get("evidence", []))

        with tab_memory:
            st.subheader("Recent agent traces")
            rows = get_recent_traces(limit=10)
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("No saved agent traces found.")


if __name__ == "__main__":
    main()

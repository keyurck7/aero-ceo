import os
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from app.agents.ceo_chat_agent import CEOChatAgent
from app.agents.recommendation_drilldown import DRILLDOWN_ACTIONS, build_drilldown_question

load_dotenv()

DB_PATH = os.getenv("SQLITE_DB_PATH", "data/aero_ceo.sqlite")


st.set_page_config(
    page_title="AERO-CEO | Airbus Strategic Intelligence",
    page_icon="✈️",
    layout="wide",
)


def get_connection():
    if not Path(DB_PATH).exists():
        st.error(f"Database not found: {DB_PATH}")
        st.stop()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def read_df(query: str, params=None) -> pd.DataFrame:
    conn = get_connection()
    try:
        return pd.read_sql_query(query, conn, params=params or {})
    finally:
        conn.close()


@st.cache_resource(show_spinner=False)
def get_ceo_agent():
    return CEOChatAgent()


def metric_value(query: str):
    conn = get_connection()
    try:
        cur = conn.cursor()
        return cur.execute(query).fetchone()[0]
    finally:
        conn.close()


def sentiment_label(text: str) -> str:
    text = (text or "").lower()

    positive_terms = [
        "launch", "contract", "signed", "growth", "opportunity", "investment",
        "partnership", "modernization", "secure", "success", "expand", "advance",
        "accelerate", "award", "selected"
    ]

    negative_terms = [
        "risk", "delay", "dispute", "collapse", "uncertainty", "shortage",
        "tension", "overrun", "cut", "problem", "threat", "concern", "fragmentation"
    ]

    pos = sum(1 for word in positive_terms if word in text)
    neg = sum(1 for word in negative_terms if word in text)

    if pos > neg:
        return "Positive"
    if neg > pos:
        return "Negative"
    return "Neutral"


def priority_rank(priority: str) -> int:
    return {"High": 1, "Medium": 2, "Low": 3}.get(priority, 4)


st.title("AERO-CEO: Strategic Intelligence Agent for Airbus SE")
st.caption(
    "Executive intelligence dashboard focused on Airbus Defence and Space, "
    "fighter systems, uncrewed aircraft, FCAS, Eurofighter, military space, and European defence autonomy."
)

# Load core tables
documents_df = read_df("""
    SELECT d.*, s.name AS source_name
    FROM documents d
    LEFT JOIN sources s ON d.source_id = s.id
""")

sources_df = read_df("SELECT * FROM sources")

signals_df = read_df("""
    SELECT sig.*, d.title AS document_title, d.url, d.source_type, s.name AS source_name
    FROM signals sig
    LEFT JOIN documents d ON sig.document_id = d.id
    LEFT JOIN sources s ON d.source_id = s.id
""")

recommendations_df = read_df("""
    SELECT *
    FROM recommendations
""")

evidence_df = read_df("""
    SELECT
        re.id,
        re.recommendation_id,
        re.evidence_strength,
        r.title AS recommendation_title,
        d.title AS document_title,
        d.url,
        d.topic,
        d.source_type,
        s.name AS source_name
    FROM recommendation_evidence re
    JOIN recommendations r ON re.recommendation_id = r.id
    JOIN documents d ON re.document_id = d.id
    LEFT JOIN sources s ON d.source_id = s.id
""")

if not recommendations_df.empty:
    recommendations_df["priority_rank"] = recommendations_df["priority"].apply(priority_rank)
    recommendations_df = recommendations_df.sort_values(
        ["priority_rank", "confidence_score"],
        ascending=[True, False]
    )

tab_overview, tab_market, tab_opportunity, tab_risk, tab_sentiment, tab_recommendations, tab_ask, tab_evidence, tab_system = st.tabs(
    [
        "Executive Overview",
        "Market Intelligence",
        "Opportunity Monitor",
        "Risk Monitor",
        "Sentiment Analysis",
        "CEO Recommendations",
        "Ask AERO-CEO",
        "Evidence Explorer",
        "System Audit",
    ]
)

with tab_overview:
    st.subheader("Company Overview")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Company", "Airbus SE")
    c2.metric("Strategic Focus", "Defence & Space")
    c3.metric("Documents", len(documents_df))
    c4.metric("Sources", len(sources_df))
    c5.metric("Signals", len(signals_df))

    c6, c7, c8 = st.columns(3)
    c6.metric("Recommendations", len(recommendations_df))
    c7.metric("Evidence Links", len(evidence_df))
    latest_update = documents_df["collected_at"].max() if "collected_at" in documents_df.columns else "N/A"
    c8.metric("Last Update", str(latest_update)[:19])

    st.markdown("### Strategic Alert Snapshot")

    high_risk_count = len(
        signals_df[
            (signals_df["signal_type"] == "risk")
            & (signals_df["confidence_score"] >= 0.75)
        ]
    )

    high_opp_count = len(
        signals_df[
            (signals_df["signal_type"] == "opportunity")
            & (signals_df["confidence_score"] >= 0.75)
        ]
    )

    a, b, c = st.columns(3)
    a.metric("High-confidence Opportunities", high_opp_count)
    b.metric("High-confidence Risks", high_risk_count)
    c.metric("Strategic Alert Level", "Elevated" if high_risk_count >= 5 else "Moderate")

    st.markdown("### Intelligence Distribution")

    left, right = st.columns(2)

    with left:
        if not documents_df.empty:
            topic_counts = documents_df["topic"].value_counts().reset_index()
            topic_counts.columns = ["Topic", "Documents"]
            fig = px.bar(topic_counts, x="Topic", y="Documents", title="Documents by Strategic Topic")
            st.plotly_chart(fig, use_container_width=True)

    with right:
        if not signals_df.empty:
            signal_counts = signals_df["signal_type"].value_counts().reset_index()
            signal_counts.columns = ["Signal Type", "Count"]
            fig = px.pie(signal_counts, names="Signal Type", values="Count", title="Strategic Signal Mix")
            st.plotly_chart(fig, use_container_width=True)

with tab_market:
    st.subheader("Market Intelligence")

    st.markdown("### Recent Intelligence Feed")
    recent_docs = documents_df.sort_values("id", ascending=False).head(25)

    st.dataframe(
        recent_docs[["id", "title", "source_name", "source_type", "topic", "trust_score", "url"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Competitor and Policy Radar")

    competitor_terms = [
        "Dassault", "Boeing", "Lockheed", "BAE", "Leonardo", "Saab",
        "Thales", "Rheinmetall", "NATO", "European Defence"
    ]

    radar_rows = []
    for term in competitor_terms:
        mask = documents_df["clean_text"].fillna("").str.contains(term, case=False, regex=False) | \
               documents_df["title"].fillna("").str.contains(term, case=False, regex=False)

        radar_rows.append(
            {
                "Entity": term,
                "Mentions": int(mask.sum()),
                "Strategic Reading": (
                    "High relevance" if mask.sum() >= 10 else
                    "Moderate relevance" if mask.sum() >= 3 else
                    "Low current signal"
                )
            }
        )

    radar_df = pd.DataFrame(radar_rows).sort_values("Mentions", ascending=False)
    st.dataframe(radar_df, use_container_width=True, hide_index=True)

    fig = px.bar(radar_df, x="Entity", y="Mentions", title="Competitor / Policy Mentions")
    st.plotly_chart(fig, use_container_width=True)

with tab_opportunity:
    st.subheader("Opportunity Monitor")

    opportunities = signals_df[signals_df["signal_type"] == "opportunity"].copy()

    if opportunities.empty:
        st.warning("No opportunity signals found.")
    else:
        grouped = (
            opportunities
            .groupby(["title", "topic"])
            .agg(
                evidence_count=("id", "count"),
                avg_confidence=("confidence_score", "mean"),
                avg_impact=("impact_score", "mean"),
                avg_urgency=("urgency_score", "mean"),
            )
            .reset_index()
            .sort_values(["avg_confidence", "evidence_count"], ascending=[False, False])
        )

        st.dataframe(
            grouped,
            use_container_width=True,
            hide_index=True,
            column_config={
                "avg_confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
                "avg_impact": st.column_config.ProgressColumn("Impact", min_value=0, max_value=1),
                "avg_urgency": st.column_config.ProgressColumn("Urgency", min_value=0, max_value=1),
            }
        )

        st.markdown("### Top Opportunity Evidence")
        for _, row in opportunities.sort_values("confidence_score", ascending=False).head(8).iterrows():
            with st.expander(f"{row['title']} | {row['topic']} | confidence {row['confidence_score']:.3f}"):
                st.write(row["description"])
                st.info(row["evidence_text"])
                st.write(f"Source: {row['source_name']} | Type: {row['source_type']}")
                st.write(row["url"])

with tab_risk:
    st.subheader("Risk Monitor")

    risks = signals_df[signals_df["signal_type"] == "risk"].copy()

    if risks.empty:
        st.warning("No risk signals found.")
    else:
        grouped = (
            risks
            .groupby(["title", "topic"])
            .agg(
                evidence_count=("id", "count"),
                avg_confidence=("confidence_score", "mean"),
                avg_impact=("impact_score", "mean"),
                avg_urgency=("urgency_score", "mean"),
            )
            .reset_index()
            .sort_values(["avg_confidence", "evidence_count"], ascending=[False, False])
        )

        st.dataframe(
            grouped,
            use_container_width=True,
            hide_index=True,
            column_config={
                "avg_confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
                "avg_impact": st.column_config.ProgressColumn("Severity / Impact", min_value=0, max_value=1),
                "avg_urgency": st.column_config.ProgressColumn("Urgency", min_value=0, max_value=1),
            }
        )

        st.markdown("### Top Risk Evidence")
        for _, row in risks.sort_values("confidence_score", ascending=False).head(8).iterrows():
            with st.expander(f"{row['title']} | {row['topic']} | confidence {row['confidence_score']:.3f}"):
                st.write(row["description"])
                st.warning(row["evidence_text"])
                st.write(f"Source: {row['source_name']} | Type: {row['source_type']}")
                st.write(row["url"])

with tab_sentiment:
    st.subheader("Sentiment Analysis")

    st.caption(
        "Current version uses a lightweight lexical sentiment proxy over titles and cleaned text. "
        "This can be replaced with a transformer sentiment model in the production profile."
    )

    sentiment_df = documents_df.copy()
    sentiment_df["sentiment"] = sentiment_df.apply(
        lambda r: sentiment_label(f"{r.get('title', '')} {r.get('clean_text', '')}"),
        axis=1,
    )

    left, right = st.columns(2)

    with left:
        sent_counts = sentiment_df["sentiment"].value_counts().reset_index()
        sent_counts.columns = ["Sentiment", "Documents"]
        fig = px.pie(sent_counts, names="Sentiment", values="Documents", title="Overall Intelligence Sentiment")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        sent_topic = (
            sentiment_df
            .groupby(["topic", "sentiment"])
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(
            sent_topic,
            x="topic",
            y="count",
            color="sentiment",
            title="Sentiment by Strategic Topic",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Sentiment Table")
    st.dataframe(
        sentiment_df[["title", "source_name", "source_type", "topic", "sentiment", "url"]].head(50),
        use_container_width=True,
        hide_index=True,
    )

with tab_recommendations:
    st.subheader("Strategic Recommendations")

    if recommendations_df.empty:
        st.warning("No recommendations found. Run: python -m app.agents.ceo_recommendation_engine")
    else:
        for _, rec in recommendations_df.iterrows():
            title_line = (
                f"{rec['priority']} Priority | Confidence {rec['confidence_score']:.3f} | "
                f"{rec['title']}"
            )

            with st.expander(title_line, expanded=rec["priority"] == "High"):
                st.markdown("#### Recommendation")
                st.write(rec["recommendation"])

                st.markdown("#### Expected Impact")
                st.success(rec["expected_impact"])

                st.markdown("#### Risk Assessment")
                st.warning(rec["risk_assessment"])

                st.markdown("#### Supporting Evidence")
                ev = evidence_df[evidence_df["recommendation_id"] == rec["id"]].sort_values(
                    "evidence_strength", ascending=False
                )

                for i, (_, item) in enumerate(ev.head(6).iterrows(), start=1):
                    st.markdown(
                        f"**{i}. Evidence strength {item['evidence_strength']:.3f}**  \n"
                        f"Source: {item['source_name']} | Type: {item['source_type']} | Topic: {item['topic']}  \n"
                        f"Document: {item['document_title']}  \n"
                        f"URL: {item['url']}"
                    )


    st.markdown("---")
    st.markdown("### Recommendation Drill-down")

    st.caption(
        "Use this section to interrogate a recommendation. "
        "The CEO can ask why, inspect evidence, request a lower-risk version, test budget constraints, "
        "or explore partnership options."
    )

    if recommendations_df.empty:
        st.info("No recommendations available for drill-down.")
    else:
        rec_options = {
            f"{int(row['id'])} | {row['priority']} | confidence {row['confidence_score']:.3f} | {row['title']}": int(row["id"])
            for _, row in recommendations_df.iterrows()
        }

        selected_rec_label = st.selectbox(
            "Select recommendation for drill-down",
            list(rec_options.keys()),
            key="drilldown_rec_select",
        )

        selected_action = st.selectbox(
            "Select drill-down action",
            list(DRILLDOWN_ACTIONS.keys()),
            key="drilldown_action_select",
        )

        custom_condition = st.text_input(
            "Optional CEO condition",
            placeholder="Example: limited budget, low political risk, focus on Spain, fast 6-month execution",
            key="drilldown_custom_condition",
        )

        run_drilldown = st.button("Run Recommendation Drill-down", type="primary")

        if run_drilldown:
            selected_rec_id = rec_options[selected_rec_label]
            selected_rec_rows = recommendations_df[recommendations_df["id"] == selected_rec_id]

            if selected_rec_rows.empty:
                st.error("Selected recommendation was not found.")
            else:
                selected_rec = selected_rec_rows.iloc[0].to_dict()
                drilldown_question = build_drilldown_question(
                    recommendation=selected_rec,
                    action=selected_action,
                    custom_condition=custom_condition,
                )

                with st.spinner("AERO-CEO is drilling into the recommendation with evidence..."):
                    try:
                        agent = get_ceo_agent()
                        result = agent.answer(drilldown_question, top_k=8)

                        st.session_state["latest_drilldown_result"] = result
                        st.session_state["latest_drilldown_action"] = selected_action
                        st.session_state["latest_drilldown_rec_label"] = selected_rec_label
                    except Exception as exc:
                        st.error(f"Recommendation drill-down failed: {exc}")

        if "latest_drilldown_result" in st.session_state:
            result = st.session_state["latest_drilldown_result"]
            route = result.get("route", {})

            st.markdown("#### Drill-down Result")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Action", st.session_state.get("latest_drilldown_action", ""))
            d2.metric("Intent", route.get("intent", "unknown"))
            d3.metric("Confidence", result.get("confidence", 0))
            d4.metric("Evidence Items", result.get("evidence_count", 0))

            st.markdown(result["answer_markdown"])


with tab_ask:
    st.subheader("Ask AERO-CEO")

    st.caption(
        "Ask strategic follow-up questions about Airbus SE, Airbus Defence and Space, FCAS, Eurofighter, "
        "uncrewed systems, partnerships, risks, competitors, and evidence-backed recommendations."
    )

    q1, q2, q3 = st.columns(3)
    q1.metric("Mode", "Evidence-grounded Q&A")
    q2.metric("Repository", "FAISS + SQLite")
    q3.metric("Agent", "CEO Strategic Advisor")

    example_questions = [
        "Should Airbus collaborate with another European organization for sixth-generation fighter systems?",
        "What are the biggest risks if FCAS is delayed?",
        "What should Airbus do if budget is limited but it still wants to compete in uncrewed combat aircraft?",
        "Explain the recommendation about military space and secure communications with evidence.",
        "Which partner could help Airbus in Spain for SIRTAP and uncrewed systems?",
        "What is the lower-risk version of investing in future combat systems?",
    ]

    selected_example = st.selectbox(
        "Choose an example CEO question",
        [""] + example_questions,
    )

    default_question = selected_example if selected_example else ""

    ceo_question = st.text_area(
        "CEO question",
        value=default_question,
        height=90,
        placeholder="Ask AERO-CEO a strategic question, for example: What should Airbus do if FCAS is delayed?",
    )

    col_a, col_b, col_c = st.columns([1, 1, 2])
    ask_button = col_a.button("Ask AERO-CEO", type="primary")
    clear_button = col_b.button("Clear current answer")

    if clear_button:
        st.session_state.pop("latest_ceo_answer", None)
        st.session_state.pop("latest_ceo_route", None)
        st.session_state.pop("latest_ceo_confidence", None)
        st.session_state.pop("latest_ceo_evidence_count", None)

    if ask_button:
        if not ceo_question.strip():
            st.warning("Please enter a CEO question first.")
        else:
            with st.spinner("AERO-CEO is retrieving evidence and generating a strategic answer..."):
                try:
                    agent = get_ceo_agent()
                    result = agent.answer(ceo_question.strip(), top_k=8)

                    st.session_state["latest_ceo_answer"] = result["answer_markdown"]
                    st.session_state["latest_ceo_route"] = result["route"]
                    st.session_state["latest_ceo_confidence"] = result["confidence"]
                    st.session_state["latest_ceo_evidence_count"] = result["evidence_count"]
                except Exception as exc:
                    st.error(f"CEO agent failed: {exc}")

    if "latest_ceo_answer" in st.session_state:
        route = st.session_state.get("latest_ceo_route", {})
        confidence = st.session_state.get("latest_ceo_confidence", 0)
        evidence_count = st.session_state.get("latest_ceo_evidence_count", 0)

        st.markdown("### Answer Summary")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Intent", route.get("intent", "unknown"))
        s2.metric("Topic", route.get("topic") or "General")
        s3.metric("Confidence", confidence)
        s4.metric("Evidence Items", evidence_count)

        st.markdown("### AERO-CEO Answer")
        st.markdown(st.session_state["latest_ceo_answer"])

    st.markdown("### Recent CEO Q&A Sessions")

    try:
        recent_qna = read_df("""
            SELECT id, question, intent, topic, business_area, region,
                   confidence_score, evidence_count, created_at
            FROM ceo_queries
            ORDER BY id DESC
            LIMIT 20
        """)

        if recent_qna.empty:
            st.info("No CEO Q&A sessions stored yet. Ask a question above.")
        else:
            st.dataframe(
                recent_qna,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "confidence_score": st.column_config.ProgressColumn(
                        "Confidence",
                        min_value=0,
                        max_value=1,
                    )
                },
            )

            selected_qna_id = st.number_input(
                "Open saved Q&A by ID",
                min_value=1,
                step=1,
                value=int(recent_qna.iloc[0]["id"]),
            )

            selected_qna = read_df(
                """
                SELECT question, answer_markdown, confidence_score, evidence_count, created_at
                FROM ceo_queries
                WHERE id = ?
                """,
                params=(selected_qna_id,),
            )

            if not selected_qna.empty:
                row = selected_qna.iloc[0]
                with st.expander(f"Saved answer for Q&A #{selected_qna_id}", expanded=False):
                    st.markdown(f"**Question:** {row['question']}")
                    st.markdown(f"**Confidence:** {row['confidence_score']} | **Evidence count:** {row['evidence_count']} | **Created:** {row['created_at']}")
                    st.markdown(row["answer_markdown"])

    except Exception as exc:
        st.info(
            "CEO Q&A table is not initialized yet. Ask one question first or run "
            "`bash scripts/ask_ceo.sh \"your question\"`."
        )
        st.caption(str(exc))



with tab_evidence:
    st.subheader("Evidence Explorer")

    query = st.text_input("Search evidence by keyword", value="")
    topic_filter = st.selectbox(
        "Filter by topic",
        ["All"] + sorted(documents_df["topic"].dropna().unique().tolist())
    )

    evidence_view = documents_df.copy()

    if query:
        mask = evidence_view["title"].fillna("").str.contains(query, case=False, regex=False) | \
               evidence_view["clean_text"].fillna("").str.contains(query, case=False, regex=False)
        evidence_view = evidence_view[mask]

    if topic_filter != "All":
        evidence_view = evidence_view[evidence_view["topic"] == topic_filter]

    st.dataframe(
        evidence_view[["id", "title", "source_name", "source_type", "topic", "trust_score", "url"]].head(100),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Document Preview")
    selected_id = st.number_input("Document ID", min_value=1, value=1, step=1)

    selected = documents_df[documents_df["id"] == selected_id]
    if not selected.empty:
        row = selected.iloc[0]
        st.markdown(f"**{row['title']}**")
        st.write(f"Source: {row['source_name']} | Type: {row['source_type']} | Topic: {row['topic']}")
        st.write(row["url"])
        st.text_area("Clean text", row["clean_text"], height=300)

with tab_system:
    st.subheader("System Audit")

    st.markdown("### Pipeline Status")

    status = pd.DataFrame(
        [
            {"Layer": "Data Collection", "Status": "Complete", "Evidence": f"{len(documents_df)} documents"},
            {"Layer": "Knowledge Repository", "Status": "Complete", "Evidence": "SQLite local repository"},
            {"Layer": "Semantic Retrieval", "Status": "Complete", "Evidence": "FAISS index + BGE embeddings"},
            {"Layer": "Signal Extraction", "Status": "Complete", "Evidence": f"{len(signals_df)} strategic signals"},
            {"Layer": "CEO Recommendations", "Status": "Complete", "Evidence": f"{len(recommendations_df)} recommendations"},
            {"Layer": "Evidence Linking", "Status": "Complete", "Evidence": f"{len(evidence_df)} evidence links"},
            {"Layer": "Production Profile", "Status": "Designed", "Evidence": "PostgreSQL + pgvector + Docker Compose"},
        ]
    )

    st.dataframe(status, use_container_width=True, hide_index=True)

    st.markdown("### Source Audit")
    st.dataframe(
        sources_df.sort_values("trust_score", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Design Principle")
    st.info(
        "No recommendation without evidence. No evidence without source metadata. "
        "No strategic score without explainable criteria."
    )

import os
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

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

tab_overview, tab_market, tab_opportunity, tab_risk, tab_sentiment, tab_recommendations, tab_evidence, tab_system = st.tabs(
    [
        "Executive Overview",
        "Market Intelligence",
        "Opportunity Monitor",
        "Risk Monitor",
        "Sentiment Analysis",
        "CEO Recommendations",
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

import os
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from app.agents.ceo_chat_agent import CEOChatAgent
from app.intelligence.business_unit_strategy import BUSINESS_UNITS, build_business_unit_profile, generate_profile_markdown
from app.intelligence.data_quality_audit import get_core_tables, calculate_quality_scores, source_type_summary, topic_coverage_summary, recommendation_evidence_audit, data_gaps, generate_audit_markdown
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

tab_overview, tab_market, tab_units, tab_quality, tab_opportunity, tab_risk, tab_sentiment, tab_recommendations, tab_ask, tab_scenario, tab_evidence, tab_system = st.tabs(
    [
        "Executive Overview",
        "Market Intelligence",
        "Business Unit Strategy",
        "Data Quality Audit",
        "Opportunity Monitor",
        "Risk Monitor",
        "Sentiment Analysis",
        "CEO Recommendations",
        "Ask AERO-CEO",
        "Scenario Analyzer",
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

with tab_units:
    st.subheader("Business Unit Strategy")

    st.caption(
        "Select a section of Airbus and inspect its documents, strategic signals, risks, opportunities, "
        "recommendations, and evidence. This gives CEO-level coverage beyond one static recommendation."
    )

    selected_unit = st.selectbox(
        "Select Airbus section",
        list(BUSINESS_UNITS.keys()),
    )

    try:
        profile = build_business_unit_profile(selected_unit)

        st.markdown(f"### {selected_unit}")
        st.write(profile["description"])

        u1, u2, u3, u4, u5 = st.columns(5)
        u1.metric("Documents", profile["document_count"])
        u2.metric("Signals", profile["signal_count"])
        u3.metric("Opportunities", profile["opportunity_count"])
        u4.metric("Risks", profile["risk_count"])
        u5.metric("Trends", profile["trend_count"])

        st.markdown("### Executive Profile")
        st.markdown(generate_profile_markdown(selected_unit))

        left_col, right_col = st.columns(2)

        with left_col:
            st.markdown("### Signal Summary")
            st.dataframe(
                profile["signal_summary"],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "avg_confidence": st.column_config.ProgressColumn("Avg Confidence", min_value=0, max_value=1),
                    "avg_impact": st.column_config.ProgressColumn("Avg Impact", min_value=0, max_value=1),
                    "avg_urgency": st.column_config.ProgressColumn("Avg Urgency", min_value=0, max_value=1),
                },
            )

        with right_col:
            st.markdown("### Source Summary")
            st.dataframe(
                profile["source_summary"],
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### Related Recommendations")
        if profile["recommendations"].empty:
            st.info("No related recommendations found.")
        else:
            st.dataframe(
                profile["recommendations"][["id", "priority", "confidence_score", "title", "recommendation"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "confidence_score": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
                },
            )

        st.markdown("### Top Opportunities")
        if profile["opportunities"].empty:
            st.info("No opportunity signals found for this section.")
        else:
            st.dataframe(
                profile["opportunities"][[
                    "signal_type", "topic", "title", "confidence_score",
                    "impact_score", "urgency_score", "source_name", "url"
                ]].head(12),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "confidence_score": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
                    "impact_score": st.column_config.ProgressColumn("Impact", min_value=0, max_value=1),
                    "urgency_score": st.column_config.ProgressColumn("Urgency", min_value=0, max_value=1),
                },
            )

        st.markdown("### Top Risks")
        if profile["risks"].empty:
            st.info("No risk signals found for this section.")
        else:
            st.dataframe(
                profile["risks"][[
                    "signal_type", "topic", "title", "confidence_score",
                    "impact_score", "urgency_score", "source_name", "url"
                ]].head(12),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "confidence_score": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
                    "impact_score": st.column_config.ProgressColumn("Impact", min_value=0, max_value=1),
                    "urgency_score": st.column_config.ProgressColumn("Urgency", min_value=0, max_value=1),
                },
            )

        st.markdown("### Recent Evidence Documents")
        if profile["documents"].empty:
            st.info("No documents found for this section.")
        else:
            st.dataframe(
                profile["documents"][[
                    "id", "title", "source_name", "source_type", "topic", "trust_score", "url"
                ]].head(30),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### Ask AERO-CEO About This Section")

        section_question = st.text_area(
            "Business-unit CEO question",
            value=f"What should Airbus CEO prioritize next for {selected_unit}, and why?",
            height=90,
            key="business_unit_question",
        )

        if st.button("Ask About Selected Section", type="primary"):
            with st.spinner("AERO-CEO is generating a section-specific answer..."):
                try:
                    agent = get_ceo_agent()
                    result = agent.answer(section_question, top_k=8)
                    st.session_state["latest_business_unit_answer"] = result
                except Exception as exc:
                    st.error(f"Business unit Q&A failed: {exc}")

        if "latest_business_unit_answer" in st.session_state:
            result = st.session_state["latest_business_unit_answer"]
            route = result.get("route", {})

            st.markdown("#### Section-Specific CEO Answer")
            a, b, c = st.columns(3)
            a.metric("Intent", route.get("intent", "unknown"))
            b.metric("Confidence", result.get("confidence", 0))
            c.metric("Evidence Items", result.get("evidence_count", 0))

            st.markdown(result["answer_markdown"])

    except Exception as exc:
        st.error(f"Business Unit Strategy tab failed: {exc}")



with tab_quality:
    st.subheader("Data Quality Audit")

    st.caption(
        "This audit explains whether AERO-CEO has enough source diversity, document volume, "
        "chunk coverage, strategic signals, and evidence links to support executive recommendations."
    )

    try:
        audit_tables = get_core_tables()
        quality_scores = calculate_quality_scores(audit_tables)

        q1, q2, q3, q4, q5 = st.columns(5)
        q1.metric("Documents", quality_scores["document_count"])
        q2.metric("Sources", quality_scores["source_count"])
        q3.metric("Chunks", quality_scores["chunk_count"])
        q4.metric("Signals", quality_scores["signal_count"])
        q5.metric("Quality Score", quality_scores["overall_quality_score"])

        q6, q7, q8, q9 = st.columns(4)
        q6.metric("Recommendations", quality_scores["recommendation_count"])
        q7.metric("Evidence Links", quality_scores["evidence_count"])
        q8.metric("Evidence / Recommendation", quality_scores["evidence_per_recommendation"])
        q9.metric("Avg Trust", quality_scores["avg_document_trust"])

        st.markdown("### Quality Score Breakdown")

        score_rows = [
            {"Metric": "Source Diversity", "Score": quality_scores["source_diversity_score"], "Meaning": "Number of independent sources"},
            {"Metric": "Document Volume", "Score": quality_scores["document_volume_score"], "Meaning": "Corpus size versus 100-document target"},
            {"Metric": "Chunking", "Score": quality_scores["chunking_score"], "Meaning": "Documents converted into searchable chunks"},
            {"Metric": "Signal Extraction", "Score": quality_scores["signal_score"], "Meaning": "Strategic signals per document"},
            {"Metric": "Evidence Linking", "Score": quality_scores["evidence_link_score"], "Meaning": "Evidence links per recommendation"},
            {"Metric": "Trust", "Score": quality_scores["trust_score"], "Meaning": "Average source trust score"},
            {"Metric": "Topic Coverage", "Score": quality_scores["topic_coverage_score"], "Meaning": "Strategic topic breadth"},
        ]

        score_df = pd.DataFrame(score_rows)

        st.dataframe(
            score_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1),
            },
        )

        st.markdown("### Source Type Summary")
        st.dataframe(
            source_type_summary(audit_tables["documents"]),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Topic Coverage Summary")
        st.dataframe(
            topic_coverage_summary(audit_tables["documents"], audit_tables["signals"]),
            use_container_width=True,
            hide_index=True,
            column_config={
                "avg_signal_confidence": st.column_config.ProgressColumn("Avg Signal Confidence", min_value=0, max_value=1),
                "avg_impact": st.column_config.ProgressColumn("Avg Impact", min_value=0, max_value=1),
            },
        )

        st.markdown("### Recommendation Evidence Audit")
        st.dataframe(
            recommendation_evidence_audit(audit_tables["recommendations"], audit_tables["evidence"]),
            use_container_width=True,
            hide_index=True,
            column_config={
                "confidence_score": st.column_config.ProgressColumn("Recommendation Confidence", min_value=0, max_value=1),
                "avg_evidence_strength": st.column_config.ProgressColumn("Avg Evidence Strength", min_value=0, max_value=1),
            },
        )

        st.markdown("### Data Gaps")
        gaps = data_gaps(audit_tables["documents"])
        if gaps.empty:
            st.success("No major data gaps found.")
        else:
            st.dataframe(gaps, use_container_width=True, hide_index=True)

        st.markdown("### Audit Summary")
        st.markdown(generate_audit_markdown())

    except Exception as exc:
        st.error(f"Data Quality Audit failed: {exc}")



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

    st.markdown("### Semantic Evidence Search")

    st.caption(
        "Search the FAISS vector memory directly. This lets the CEO, examiner, or analyst inspect the evidence "
        "behind recommendations, CEO answers, risks, opportunities, and scenarios."
    )

    sem_col1, sem_col2, sem_col3 = st.columns([2, 1, 1])

    with sem_col1:
        semantic_query = st.text_input(
            "Semantic search query",
            value="FCAS partner risk Dassault future fighter",
            placeholder="Example: SIRTAP uncrewed aircraft opportunity for Airbus",
            key="semantic_evidence_query",
        )

    with sem_col2:
        semantic_topic = st.selectbox(
            "Topic filter",
            ["All"] + sorted(documents_df["topic"].dropna().unique().tolist()),
            key="semantic_topic_filter",
        )

    with sem_col3:
        semantic_source_type = st.selectbox(
            "Source type filter",
            ["All"] + sorted(documents_df["source_type"].dropna().unique().tolist()),
            key="semantic_source_filter",
        )

    sem_col4, sem_col5, sem_col6 = st.columns([1, 1, 2])

    with sem_col4:
        semantic_top_k = st.slider(
            "Top K",
            min_value=3,
            max_value=20,
            value=8,
            step=1,
            key="semantic_top_k",
        )

    with sem_col5:
        min_score = st.slider(
            "Min score",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            key="semantic_min_score",
        )

    search_semantic_button = st.button("Search Semantic Memory", type="primary")

    if search_semantic_button:
        if not semantic_query.strip():
            st.warning("Enter a semantic search query first.")
        else:
            with st.spinner("Searching FAISS evidence memory..."):
                try:
                    agent = get_ceo_agent()

                    if not agent.search_engine:
                        st.error("FAISS search engine is not available. Rebuild with: python -m app.retrieval.build_faiss_index")
                    else:
                        semantic_results = agent.search_engine.search(
                            query=semantic_query.strip(),
                            top_k=semantic_top_k,
                            topic=None if semantic_topic == "All" else semantic_topic,
                            source_type=None if semantic_source_type == "All" else semantic_source_type,
                        )

                        semantic_results = [
                            item for item in semantic_results
                            if float(item.get("score") or 0) >= min_score
                        ]

                        st.session_state["latest_semantic_results"] = semantic_results
                        st.session_state["latest_semantic_query"] = semantic_query.strip()

                except Exception as exc:
                    st.error(f"Semantic search failed: {exc}")

    if "latest_semantic_results" in st.session_state:
        semantic_results = st.session_state["latest_semantic_results"]
        semantic_query_used = st.session_state.get("latest_semantic_query", "")

        st.markdown("### Semantic Search Results")
        st.write(f"Query: **{semantic_query_used}**")
        st.metric("Results", len(semantic_results))

        if not semantic_results:
            st.info("No semantic results matched the query and filters.")
        else:
            results_table = pd.DataFrame([
                {
                    "rank": idx + 1,
                    "score": round(float(item.get("score") or 0), 3),
                    "topic": item.get("topic"),
                    "source_type": item.get("source_type"),
                    "source_name": item.get("source_name"),
                    "title": item.get("title"),
                    "url": item.get("url"),
                }
                for idx, item in enumerate(semantic_results)
            ])

            st.dataframe(
                results_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1),
                },
            )

            st.markdown("### Evidence Chunks")

            for idx, item in enumerate(semantic_results, start=1):
                title = item.get("title") or "Untitled evidence"
                score = float(item.get("score") or 0)
                topic = item.get("topic") or "Unknown topic"
                source_name = item.get("source_name") or "Unknown source"

                with st.expander(f"{idx}. score {score:.3f} | {topic} | {source_name} | {title[:90]}"):
                    st.write(f"**Source type:** {item.get('source_type')}")
                    st.write(f"**URL:** {item.get('url')}")
                    st.write(f"**Document ID:** {item.get('document_id')} | **Chunk ID:** {item.get('chunk_id')}")
                    st.markdown("**Evidence text:**")
                    st.info(item.get("chunk_text", ""))

            st.markdown("### Ask AERO-CEO Using This Evidence Topic")

            followup_question = st.text_area(
                "Follow-up CEO question based on search",
                value=f"Based on evidence about {semantic_query_used}, what should Airbus do next and why?",
                height=90,
                key="semantic_followup_question",
            )

            if st.button("Ask AERO-CEO From Evidence Search"):
                with st.spinner("AERO-CEO is answering using the evidence topic..."):
                    try:
                        agent = get_ceo_agent()
                        result = agent.answer(followup_question, top_k=8)
                        st.session_state["latest_semantic_followup_answer"] = result
                    except Exception as exc:
                        st.error(f"Evidence-based follow-up failed: {exc}")

            if "latest_semantic_followup_answer" in st.session_state:
                result = st.session_state["latest_semantic_followup_answer"]
                route = result.get("route", {})

                st.markdown("#### Evidence-Based CEO Follow-up Answer")
                f1, f2, f3 = st.columns(3)
                f1.metric("Intent", route.get("intent", "unknown"))
                f2.metric("Confidence", result.get("confidence", 0))
                f3.metric("Evidence Items", result.get("evidence_count", 0))

                st.markdown(result["answer_markdown"])

    st.markdown("---")
    st.markdown("### Keyword Evidence Browser")
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

import os
import sqlite3
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("SQLITE_DB_PATH", "data/aero_ceo.sqlite")
REPORT_PATH = "reports/aero_ceo_executive_dashboard.html"


def read_df(query: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


def fig_to_html(fig):
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def table_html(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "<p>No data available.</p>"
    return df.head(max_rows).to_html(index=False, escape=False, classes="data-table")


def main():
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    Path("reports").mkdir(parents=True, exist_ok=True)

    documents = read_df("""
        SELECT d.*, s.name AS source_name
        FROM documents d
        LEFT JOIN sources s ON d.source_id = s.id
    """)

    signals = read_df("""
        SELECT sig.*, d.title AS document_title, d.url, d.source_type, s.name AS source_name
        FROM signals sig
        LEFT JOIN documents d ON sig.document_id = d.id
        LEFT JOIN sources s ON d.source_id = s.id
    """)

    recommendations = read_df("""
        SELECT *
        FROM recommendations
        ORDER BY
            CASE priority
                WHEN 'High' THEN 1
                WHEN 'Medium' THEN 2
                ELSE 3
            END,
            confidence_score DESC
    """)

    evidence = read_df("""
        SELECT
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
        ORDER BY re.evidence_strength DESC
    """)

    sources = read_df("SELECT * FROM sources")

    topic_counts = documents["topic"].value_counts().reset_index()
    topic_counts.columns = ["Topic", "Documents"]
    topic_fig = px.bar(topic_counts, x="Topic", y="Documents", title="Documents by Strategic Topic")

    signal_counts = signals["signal_type"].value_counts().reset_index()
    signal_counts.columns = ["Signal Type", "Count"]
    signal_fig = px.pie(signal_counts, names="Signal Type", values="Count", title="Strategic Signal Mix")

    rec_fig = px.bar(
        recommendations,
        x="title",
        y="confidence_score",
        color="priority",
        title="CEO Recommendations by Confidence",
    )

    opportunity_summary = (
        signals[signals["signal_type"] == "opportunity"]
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

    risk_summary = (
        signals[signals["signal_type"] == "risk"]
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

    rec_sections = ""
    for _, rec in recommendations.iterrows():
        rec_evidence = evidence[evidence["recommendation_id"] == rec["id"]].head(6)

        evidence_items = ""
        for _, ev in rec_evidence.iterrows():
            evidence_items += f"""
            <li>
                <b>Strength:</b> {ev['evidence_strength']:.3f}<br>
                <b>Source:</b> {ev['source_name']} | {ev['source_type']} | {ev['topic']}<br>
                <b>Document:</b> {ev['document_title']}<br>
                <a href="{ev['url']}" target="_blank">{ev['url']}</a>
            </li>
            """

        rec_sections += f"""
        <div class="card">
            <h3>{rec['title']}</h3>
            <p><b>Priority:</b> {rec['priority']} | <b>Confidence:</b> {rec['confidence_score']:.3f}</p>
            <h4>Recommendation</h4>
            <p>{rec['recommendation']}</p>
            <h4>Expected Impact</h4>
            <p class="success">{rec['expected_impact']}</p>
            <h4>Risk Assessment</h4>
            <p class="warning">{rec['risk_assessment']}</p>
            <h4>Supporting Evidence</h4>
            <ol>{evidence_items}</ol>
        </div>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>AERO-CEO Executive Dashboard</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 32px;
                background: #0f172a;
                color: #e5e7eb;
            }}
            h1, h2, h3 {{
                color: #f8fafc;
            }}
            .subtitle {{
                color: #cbd5e1;
                font-size: 18px;
            }}
            .metrics {{
                display: grid;
                grid-template-columns: repeat(5, 1fr);
                gap: 16px;
                margin: 24px 0;
            }}
            .metric {{
                background: #1e293b;
                padding: 18px;
                border-radius: 12px;
                border: 1px solid #334155;
            }}
            .metric .num {{
                font-size: 30px;
                font-weight: bold;
                color: #38bdf8;
            }}
            .card {{
                background: #1e293b;
                padding: 22px;
                border-radius: 14px;
                margin: 22px 0;
                border: 1px solid #334155;
            }}
            .success {{
                background: #052e16;
                padding: 12px;
                border-radius: 8px;
                color: #bbf7d0;
            }}
            .warning {{
                background: #451a03;
                padding: 12px;
                border-radius: 8px;
                color: #fed7aa;
            }}
            .data-table {{
                border-collapse: collapse;
                width: 100%;
                background: #111827;
                color: #e5e7eb;
                font-size: 14px;
            }}
            .data-table th, .data-table td {{
                border: 1px solid #374151;
                padding: 8px;
                text-align: left;
                vertical-align: top;
            }}
            .data-table th {{
                background: #1f2937;
                color: #f9fafb;
            }}
            a {{
                color: #7dd3fc;
            }}
        </style>
    </head>
    <body>
        <h1>AERO-CEO: Strategic Intelligence Agent for Airbus SE</h1>
        <p class="subtitle">
            Executive intelligence dashboard focused on Airbus Defence and Space, fighter systems,
            FCAS, Eurofighter, uncrewed aircraft, military space, and European defence autonomy.
        </p>

        <div class="metrics">
            <div class="metric"><div>Documents</div><div class="num">{len(documents)}</div></div>
            <div class="metric"><div>Sources</div><div class="num">{len(sources)}</div></div>
            <div class="metric"><div>Signals</div><div class="num">{len(signals)}</div></div>
            <div class="metric"><div>Recommendations</div><div class="num">{len(recommendations)}</div></div>
            <div class="metric"><div>Evidence Links</div><div class="num">{len(evidence)}</div></div>
        </div>

        <div class="card">
            <h2>CEO Briefing</h2>
            <p><b>What happened?</b> AERO-CEO collected live public intelligence about Airbus SE and Airbus Defence and Space, then extracted strategic signals across FCAS, Eurofighter, uncrewed aircraft, military space, aerial refuelling, military transport, and European defence autonomy.</p>
            <p><b>Why does it matter?</b> Airbus operates in a defence environment shaped by European sovereignty demand, future fighter uncertainty, autonomous aircraft, military space resilience, procurement politics, and competitor pressure.</p>
            <p><b>What should management do next?</b> Prioritize high-confidence strategic actions where evidence supports both near-term execution and long-term positioning: uncrewed combat systems, FCAS modular architecture, Eurofighter modernization, military space, and execution resilience.</p>
        </div>

        <h2>Market Intelligence</h2>
        <div class="card">{fig_to_html(topic_fig)}</div>
        <div class="card">{fig_to_html(signal_fig)}</div>

        <h2>Strategic Recommendations</h2>
        <div class="card">{fig_to_html(rec_fig)}</div>
        {rec_sections}

        <h2>Opportunity Monitor</h2>
        <div class="card">{table_html(opportunity_summary, 20)}</div>

        <h2>Risk Monitor</h2>
        <div class="card">{table_html(risk_summary, 20)}</div>

        <h2>Recent Intelligence Feed</h2>
        <div class="card">{table_html(documents[['id', 'title', 'source_name', 'source_type', 'topic', 'trust_score', 'url']].sort_values('id', ascending=False), 40)}</div>

        <h2>System Audit</h2>
        <div class="card">
            <ul>
                <li>Data collection: complete</li>
                <li>Knowledge repository: SQLite local profile, PostgreSQL + pgvector production profile</li>
                <li>Vector retrieval: FAISS + BGE embeddings</li>
                <li>Signal extraction: opportunity, risk, and trend detection</li>
                <li>Recommendations: evidence-linked CEO recommendations</li>
                <li>Dashboard: Streamlit primary, static HTML fallback</li>
            </ul>
            <p><b>Design principle:</b> No recommendation without evidence. No evidence without source metadata. No strategic score without explainable criteria.</p>
        </div>

        <p>Generated at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </body>
    </html>
    """

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Static executive dashboard generated: {REPORT_PATH}")


if __name__ == "__main__":
    main()

import argparse
from typing import Dict, List

import pandas as pd

from app.db.local_db import get_connection, init_local_db


BUSINESS_UNITS = {
    "Airbus Defence and Space": {
        "description": "Overall defence and space strategy across military aircraft, space systems, secure communications, and future combat.",
        "topics": [
            "FCAS",
            "Eurofighter",
            "Uncrewed Combat Aircraft",
            "Military Space",
            "Military Transport",
            "Aerial Refuelling",
            "European Defence Autonomy",
            "Policy and Procurement",
            "Supply Chain",
        ],
        "keywords": [
            "defence", "defense", "military", "airbus defence and space",
            "fighter", "satellite", "secure communications", "nato"
        ],
    },
    "Future Combat / FCAS": {
        "description": "Future Combat Air System, sixth-generation fighter architecture, combat cloud, and future air combat positioning.",
        "topics": ["FCAS", "European Defence Autonomy"],
        "keywords": [
            "fcas", "future combat air system", "sixth-generation", "sixth generation",
            "future fighter", "combat cloud", "next generation fighter"
        ],
    },
    "Eurofighter": {
        "description": "Eurofighter Typhoon modernization, fighter relevance, near-term capability bridge, and competitive pressure.",
        "topics": ["Eurofighter"],
        "keywords": ["eurofighter", "typhoon", "fighter modernization", "radar upgrade"],
    },
    "Uncrewed Systems": {
        "description": "SIRTAP, Eurodrone, uncrewed aircraft, crewed-uncrewed teaming, and collaborative combat aircraft.",
        "topics": ["Uncrewed Combat Aircraft"],
        "keywords": [
            "uncrewed", "unmanned", "uav", "uas", "drone", "sirtap",
            "eurodrone", "collaborative combat", "loyal wingman",
            "crewed-uncrewed", "manned-unmanned"
        ],
    },
    "Military Space": {
        "description": "Military satellites, secure communications, space resilience, and command-and-control infrastructure.",
        "topics": ["Military Space"],
        "keywords": [
            "space", "satellite", "secure communications", "earth observation",
            "military satellite", "space resilience"
        ],
    },
    "Military Transport": {
        "description": "A400M, C295, CN235, tactical transport, mission support, and delivery reliability.",
        "topics": ["Military Transport"],
        "keywords": ["a400m", "c295", "cn235", "military transport", "transport aircraft"],
    },
    "Aerial Refuelling": {
        "description": "A330 MRTT, tanker aircraft, refuelling capability, and mobility support.",
        "topics": ["Aerial Refuelling"],
        "keywords": ["a330 mrtt", "mrtt", "tanker", "refuelling", "refueling"],
    },
    "Supply Chain & Execution": {
        "description": "Supplier risk, delivery risk, production resilience, cost pressure, and programme execution.",
        "topics": ["Supply Chain", "Military Transport", "Aerial Refuelling"],
        "keywords": [
            "supply chain", "supplier", "delivery", "shortage", "production",
            "cost overrun", "execution", "manufacturing"
        ],
    },
    "Policy & Procurement": {
        "description": "Government procurement, NATO demand, export policy, defence budgets, and European strategic autonomy.",
        "topics": ["Policy and Procurement", "European Defence Autonomy"],
        "keywords": [
            "procurement", "contract", "ministry", "government", "budget",
            "export", "regulation", "policy", "nato", "sovereign",
            "strategic autonomy", "european defence"
        ],
    },
    "Competitor Radar": {
        "description": "Competitor activity from Dassault, Boeing, Lockheed Martin, BAE Systems, Leonardo, Saab, Thales, and Rheinmetall.",
        "topics": ["Competitor Activity", "FCAS", "Eurofighter"],
        "keywords": [
            "dassault", "boeing", "lockheed", "bae systems", "leonardo",
            "saab", "thales", "rheinmetall", "f-35", "gcap", "tempest"
        ],
    },
}


def read_df(query: str, params=None) -> pd.DataFrame:
    conn = get_connection()
    try:
        return pd.read_sql_query(query, conn, params=params or [])
    finally:
        conn.close()


def _row_blob(df: pd.DataFrame, columns: List[str]) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=str)

    available = [c for c in columns if c in df.columns]
    if not available:
        return pd.Series([""] * len(df), index=df.index)

    return df[available].fillna("").astype(str).agg(" ".join, axis=1).str.lower()


def filter_for_unit(df: pd.DataFrame, unit_name: str, text_columns: List[str]) -> pd.DataFrame:
    if unit_name not in BUSINESS_UNITS:
        raise ValueError(f"Unknown business unit: {unit_name}")

    if df.empty:
        return df

    config = BUSINESS_UNITS[unit_name]
    topics = set(config["topics"])
    keywords = [k.lower() for k in config["keywords"]]

    mask = pd.Series(False, index=df.index)

    if "topic" in df.columns:
        mask = mask | df["topic"].fillna("").isin(topics)

    blob = _row_blob(df, text_columns)
    for keyword in keywords:
        mask = mask | blob.str.contains(keyword, regex=False)

    return df[mask].copy()


def get_documents() -> pd.DataFrame:
    return read_df("""
        SELECT
            d.id,
            d.title,
            d.url,
            d.source_type,
            d.topic,
            d.trust_score,
            d.published_at,
            d.collected_at,
            d.clean_text,
            s.name AS source_name
        FROM documents d
        LEFT JOIN sources s ON d.source_id = s.id
        ORDER BY d.id DESC
    """)


def get_signals() -> pd.DataFrame:
    return read_df("""
        SELECT
            sig.id,
            sig.document_id,
            sig.signal_type,
            sig.topic,
            sig.title,
            sig.description,
            sig.entities,
            sig.impact_score,
            sig.urgency_score,
            sig.confidence_score,
            sig.evidence_text,
            d.title AS document_title,
            d.url,
            d.source_type,
            s.name AS source_name
        FROM signals sig
        LEFT JOIN documents d ON sig.document_id = d.id
        LEFT JOIN sources s ON d.source_id = s.id
        ORDER BY sig.confidence_score DESC, sig.impact_score DESC
    """)


def get_recommendations() -> pd.DataFrame:
    return read_df("""
        SELECT
            id,
            title,
            recommendation,
            priority,
            expected_impact,
            risk_assessment,
            confidence_score,
            created_at
        FROM recommendations
        ORDER BY
            CASE priority
                WHEN 'High' THEN 1
                WHEN 'Medium' THEN 2
                ELSE 3
            END,
            confidence_score DESC
    """)


def get_related_recommendations(unit_name: str, recommendations: pd.DataFrame) -> pd.DataFrame:
    if recommendations.empty:
        return recommendations

    config = BUSINESS_UNITS[unit_name]
    keywords = [k.lower() for k in config["keywords"]]
    topics = [t.lower() for t in config["topics"]]

    blob = _row_blob(
        recommendations,
        ["title", "recommendation", "expected_impact", "risk_assessment", "priority"],
    )

    mask = pd.Series(False, index=recommendations.index)

    for keyword in keywords + topics:
        mask = mask | blob.str.contains(keyword, regex=False)

    related = recommendations[mask].copy()

    if related.empty:
        related = recommendations.head(3).copy()

    return related


def build_business_unit_profile(unit_name: str) -> Dict:
    init_local_db()

    if unit_name not in BUSINESS_UNITS:
        raise ValueError(f"Unknown business unit: {unit_name}")

    documents = get_documents()
    signals = get_signals()
    recommendations = get_recommendations()

    unit_documents = filter_for_unit(
        documents,
        unit_name,
        ["title", "clean_text", "source_type", "source_name", "topic"],
    )

    unit_signals = filter_for_unit(
        signals,
        unit_name,
        ["title", "description", "evidence_text", "entities", "document_title", "source_name", "topic"],
    )

    unit_recommendations = get_related_recommendations(unit_name, recommendations)

    opportunities = unit_signals[unit_signals["signal_type"] == "opportunity"].copy()
    risks = unit_signals[unit_signals["signal_type"] == "risk"].copy()
    trends = unit_signals[unit_signals["signal_type"] == "trend"].copy()

    if unit_signals.empty:
        signal_summary = pd.DataFrame(columns=["signal_type", "count", "avg_confidence", "avg_impact", "avg_urgency"])
    else:
        signal_summary = (
            unit_signals
            .groupby("signal_type")
            .agg(
                count=("id", "count"),
                avg_confidence=("confidence_score", "mean"),
                avg_impact=("impact_score", "mean"),
                avg_urgency=("urgency_score", "mean"),
            )
            .reset_index()
            .sort_values("count", ascending=False)
        )

    if unit_documents.empty:
        source_summary = pd.DataFrame(columns=["source_type", "documents", "avg_trust"])
    else:
        source_summary = (
            unit_documents
            .groupby("source_type")
            .agg(
                documents=("id", "count"),
                avg_trust=("trust_score", "mean"),
            )
            .reset_index()
            .sort_values("documents", ascending=False)
        )

    profile = {
        "unit_name": unit_name,
        "description": BUSINESS_UNITS[unit_name]["description"],
        "documents": unit_documents.drop(columns=["clean_text"], errors="ignore"),
        "signals": unit_signals,
        "opportunities": opportunities,
        "risks": risks,
        "trends": trends,
        "recommendations": unit_recommendations,
        "signal_summary": signal_summary,
        "source_summary": source_summary,
        "document_count": len(unit_documents),
        "signal_count": len(unit_signals),
        "opportunity_count": len(opportunities),
        "risk_count": len(risks),
        "trend_count": len(trends),
        "recommendation_count": len(unit_recommendations),
    }

    return profile


def generate_profile_markdown(unit_name: str) -> str:
    profile = build_business_unit_profile(unit_name)

    lines = []
    lines.append(f"# Business Unit Strategy Profile: {unit_name}\n")
    lines.append(profile["description"])
    lines.append("")
    lines.append("## Snapshot")
    lines.append(f"- Documents: {profile['document_count']}")
    lines.append(f"- Strategic signals: {profile['signal_count']}")
    lines.append(f"- Opportunities: {profile['opportunity_count']}")
    lines.append(f"- Risks: {profile['risk_count']}")
    lines.append(f"- Trends: {profile['trend_count']}")
    lines.append(f"- Related recommendations: {profile['recommendation_count']}")

    recs = profile["recommendations"]
    if not recs.empty:
        lines.append("\n## Related CEO Recommendations")
        for _, rec in recs.head(5).iterrows():
            lines.append(
                f"- **{rec['priority']} | confidence {rec['confidence_score']}:** {rec['title']}"
            )

    opps = profile["opportunities"]
    if not opps.empty:
        lines.append("\n## Top Opportunities")
        for _, row in opps.head(5).iterrows():
            lines.append(
                f"- **confidence {row['confidence_score']} | impact {row['impact_score']}:** {row['title']}"
            )

    risks = profile["risks"]
    if not risks.empty:
        lines.append("\n## Top Risks")
        for _, row in risks.head(5).iterrows():
            lines.append(
                f"- **confidence {row['confidence_score']} | impact {row['impact_score']}:** {row['title']}"
            )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate Airbus business unit strategy profile.")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--unit", type=str, default="Airbus Defence and Space")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable business units:")
        for name in BUSINESS_UNITS:
            print(f"- {name}")
        return

    print(generate_profile_markdown(args.unit))


if __name__ == "__main__":
    main()

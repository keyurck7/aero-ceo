import argparse
from typing import Dict

import pandas as pd

from app.db.local_db import get_connection, init_local_db


def read_df(query: str, params=None) -> pd.DataFrame:
    conn = get_connection()
    try:
        return pd.read_sql_query(query, conn, params=params or [])
    finally:
        conn.close()


def get_core_tables() -> Dict[str, pd.DataFrame]:
    init_local_db()

    sources = read_df("SELECT * FROM sources")

    documents = read_df("""
        SELECT
            d.*,
            s.name AS source_name,
            s.trust_score AS source_registry_trust
        FROM documents d
        LEFT JOIN sources s ON d.source_id = s.id
    """)

    chunks = read_df("SELECT * FROM document_chunks")
    signals = read_df("SELECT * FROM signals")
    recommendations = read_df("SELECT * FROM recommendations")
    evidence = read_df("SELECT * FROM recommendation_evidence")

    ceo_queries = read_df("""
        SELECT *
        FROM ceo_queries
        ORDER BY id DESC
    """) if table_exists("ceo_queries") else pd.DataFrame()

    return {
        "sources": sources,
        "documents": documents,
        "chunks": chunks,
        "signals": signals,
        "recommendations": recommendations,
        "evidence": evidence,
        "ceo_queries": ceo_queries,
    }


def table_exists(table_name: str) -> bool:
    conn = get_connection()
    try:
        cur = conn.cursor()
        row = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def calculate_quality_scores(tables: Dict[str, pd.DataFrame]) -> Dict:
    sources = tables["sources"]
    documents = tables["documents"]
    chunks = tables["chunks"]
    signals = tables["signals"]
    recommendations = tables["recommendations"]
    evidence = tables["evidence"]

    doc_count = len(documents)
    source_count = len(sources)
    chunk_count = len(chunks)
    signal_count = len(signals)
    rec_count = len(recommendations)
    evidence_count = len(evidence)

    source_diversity_score = min(source_count / 25, 1.0)
    document_volume_score = min(doc_count / 100, 1.0)
    chunking_score = min(chunk_count / max(doc_count, 1), 1.0)
    signal_score = min(signal_count / max(doc_count, 1), 1.0)

    if rec_count > 0:
        evidence_per_rec = evidence_count / rec_count
        evidence_link_score = min(evidence_per_rec / 5, 1.0)
    else:
        evidence_per_rec = 0
        evidence_link_score = 0

    if not documents.empty and "trust_score" in documents.columns:
        avg_trust = float(documents["trust_score"].fillna(0.5).mean())
    else:
        avg_trust = 0.0

    trust_score = min(avg_trust, 1.0)

    if not documents.empty and "topic" in documents.columns:
        topic_count = int(documents["topic"].fillna("Unknown").nunique())
    else:
        topic_count = 0

    topic_coverage_score = min(topic_count / 8, 1.0)

    overall_score = (
        0.18 * source_diversity_score
        + 0.18 * document_volume_score
        + 0.14 * chunking_score
        + 0.16 * signal_score
        + 0.18 * evidence_link_score
        + 0.08 * trust_score
        + 0.08 * topic_coverage_score
    )

    return {
        "document_count": doc_count,
        "source_count": source_count,
        "chunk_count": chunk_count,
        "signal_count": signal_count,
        "recommendation_count": rec_count,
        "evidence_count": evidence_count,
        "evidence_per_recommendation": round(evidence_per_rec, 2),
        "avg_document_trust": round(avg_trust, 3),
        "topic_count": topic_count,
        "source_diversity_score": round(source_diversity_score, 3),
        "document_volume_score": round(document_volume_score, 3),
        "chunking_score": round(chunking_score, 3),
        "signal_score": round(signal_score, 3),
        "evidence_link_score": round(evidence_link_score, 3),
        "trust_score": round(trust_score, 3),
        "topic_coverage_score": round(topic_coverage_score, 3),
        "overall_quality_score": round(overall_score, 3),
    }


def source_type_summary(documents: pd.DataFrame) -> pd.DataFrame:
    if documents.empty:
        return pd.DataFrame()

    return (
        documents
        .groupby("source_type")
        .agg(
            documents=("id", "count"),
            avg_trust=("trust_score", "mean"),
            unique_sources=("source_name", "nunique"),
        )
        .reset_index()
        .sort_values("documents", ascending=False)
    )


def topic_coverage_summary(documents: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    if documents.empty:
        return pd.DataFrame()

    doc_summary = (
        documents
        .groupby("topic")
        .agg(
            documents=("id", "count"),
            avg_trust=("trust_score", "mean"),
            source_types=("source_type", lambda x: ", ".join(sorted(set(str(v) for v in x if pd.notna(v))))),
        )
        .reset_index()
    )

    if not signals.empty:
        signal_summary = (
            signals
            .groupby("topic")
            .agg(
                signals=("id", "count"),
                avg_signal_confidence=("confidence_score", "mean"),
                avg_impact=("impact_score", "mean"),
            )
            .reset_index()
        )

        merged = doc_summary.merge(signal_summary, on="topic", how="left")
    else:
        merged = doc_summary
        merged["signals"] = 0
        merged["avg_signal_confidence"] = 0
        merged["avg_impact"] = 0

    merged["signals"] = merged["signals"].fillna(0).astype(int)
    merged["avg_signal_confidence"] = merged["avg_signal_confidence"].fillna(0)
    merged["avg_impact"] = merged["avg_impact"].fillna(0)

    return merged.sort_values("documents", ascending=False)


def recommendation_evidence_audit(recommendations: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    if recommendations.empty:
        return pd.DataFrame()

    if evidence.empty:
        out = recommendations.copy()
        out["evidence_links"] = 0
        out["avg_evidence_strength"] = 0
        return out

    ev_summary = (
        evidence
        .groupby("recommendation_id")
        .agg(
            evidence_links=("id", "count"),
            avg_evidence_strength=("evidence_strength", "mean"),
        )
        .reset_index()
    )

    out = recommendations.merge(
        ev_summary,
        left_on="id",
        right_on="recommendation_id",
        how="left",
    )

    out["evidence_links"] = out["evidence_links"].fillna(0).astype(int)
    out["avg_evidence_strength"] = out["avg_evidence_strength"].fillna(0)

    return out[[
        "id",
        "priority",
        "confidence_score",
        "evidence_links",
        "avg_evidence_strength",
        "title",
    ]].sort_values(["evidence_links", "confidence_score"], ascending=[False, False])


def data_gaps(documents: pd.DataFrame) -> pd.DataFrame:
    if documents.empty:
        return pd.DataFrame()

    checks = []

    checks.append({
        "check": "Documents without URL",
        "count": int(documents["url"].isna().sum() + (documents["url"].fillna("") == "").sum()),
        "severity": "Medium",
    })

    checks.append({
        "check": "Documents with short clean text under 250 characters",
        "count": int((documents["clean_text"].fillna("").str.len() < 250).sum()),
        "severity": "Low",
    })

    checks.append({
        "check": "Documents without topic",
        "count": int(documents["topic"].isna().sum() + (documents["topic"].fillna("") == "").sum()),
        "severity": "Medium",
    })

    checks.append({
        "check": "Documents with low trust score below 0.6",
        "count": int((documents["trust_score"].fillna(0.5) < 0.6).sum()),
        "severity": "Low",
    })

    if "content_hash" in documents.columns:
        duplicate_hashes = documents["content_hash"].fillna("").duplicated().sum()
    else:
        duplicate_hashes = 0

    checks.append({
        "check": "Possible duplicate content hashes",
        "count": int(duplicate_hashes),
        "severity": "Medium",
    })

    return pd.DataFrame(checks)


def generate_audit_markdown() -> str:
    tables = get_core_tables()
    scores = calculate_quality_scores(tables)

    lines = []
    lines.append("# AERO-CEO Data Quality Audit\n")
    lines.append("## Core Counts")
    lines.append(f"- Documents: {scores['document_count']}")
    lines.append(f"- Sources: {scores['source_count']}")
    lines.append(f"- Chunks: {scores['chunk_count']}")
    lines.append(f"- Strategic signals: {scores['signal_count']}")
    lines.append(f"- Recommendations: {scores['recommendation_count']}")
    lines.append(f"- Recommendation evidence links: {scores['evidence_count']}")
    lines.append(f"- Evidence per recommendation: {scores['evidence_per_recommendation']}")

    lines.append("\n## Quality Scores")
    lines.append(f"- Overall quality score: {scores['overall_quality_score']}")
    lines.append(f"- Source diversity score: {scores['source_diversity_score']}")
    lines.append(f"- Document volume score: {scores['document_volume_score']}")
    lines.append(f"- Chunking score: {scores['chunking_score']}")
    lines.append(f"- Signal extraction score: {scores['signal_score']}")
    lines.append(f"- Evidence link score: {scores['evidence_link_score']}")
    lines.append(f"- Average trust score: {scores['avg_document_trust']}")
    lines.append(f"- Topic coverage score: {scores['topic_coverage_score']}")

    gaps = data_gaps(tables["documents"])
    if not gaps.empty:
        lines.append("\n## Data Gaps")
        for _, row in gaps.iterrows():
            lines.append(f"- {row['severity']}: {row['check']} = {row['count']}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run AERO-CEO data quality audit.")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    if args.markdown:
        print(generate_audit_markdown())
        return

    tables = get_core_tables()
    scores = calculate_quality_scores(tables)

    print("\n=== AERO-CEO Data Quality Audit ===")
    for key, value in scores.items():
        print(f"{key}: {value}")

    print("\nSource type summary:")
    print(source_type_summary(tables["documents"]).to_string(index=False))

    print("\nTopic coverage summary:")
    print(topic_coverage_summary(tables["documents"], tables["signals"]).head(15).to_string(index=False))

    print("\nRecommendation evidence audit:")
    print(recommendation_evidence_audit(tables["recommendations"], tables["evidence"]).to_string(index=False))

    print("\nData gaps:")
    print(data_gaps(tables["documents"]).to_string(index=False))


if __name__ == "__main__":
    main()

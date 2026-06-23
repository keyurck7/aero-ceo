import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from app.db.local_db import get_connection, init_local_db


MIN_DOCUMENTS = 100
MIN_SOURCES = 3
MIN_RECOMMENDATIONS = 3
MIN_EVIDENCE_PER_RECOMMENDATION = 3
MIN_SIGNALS = 50


RETRIEVAL_PROBES = [
    {
        "query": "FCAS partner risk Dassault future fighter",
        "expected_terms": ["fcas", "dassault", "fighter"],
    },
    {
        "query": "SIRTAP uncrewed aircraft opportunity Airbus",
        "expected_terms": ["sirtap", "uncrewed", "airbus"],
    },
    {
        "query": "military space secure communications Airbus",
        "expected_terms": ["space", "satellite", "communications"],
    },
    {
        "query": "Eurofighter modernization uncrewed teaming",
        "expected_terms": ["eurofighter", "typhoon", "uncrewed"],
    },
    {
        "query": "Airbus defence supply chain delivery risk",
        "expected_terms": ["supply", "delivery", "risk"],
    },
]


def read_df(query: str, params=None) -> pd.DataFrame:
    conn = get_connection()
    try:
        return pd.read_sql_query(query, conn, params=params or [])
    finally:
        conn.close()


def table_exists(table_name: str) -> bool:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def check_pass(name: str, passed: bool, value, threshold, details: str) -> Dict:
    return {
        "check": name,
        "status": "PASS" if passed else "FAIL",
        "value": value,
        "threshold": threshold,
        "details": details,
    }


def run_database_checks() -> List[Dict]:
    init_local_db()

    documents = read_df("SELECT * FROM documents")
    sources = read_df("SELECT * FROM sources")
    chunks = read_df("SELECT * FROM document_chunks")
    signals = read_df("SELECT * FROM signals")
    recommendations = read_df("SELECT * FROM recommendations")
    evidence = read_df("SELECT * FROM recommendation_evidence")

    checks = []

    checks.append(
        check_pass(
            "Minimum document volume",
            len(documents) >= MIN_DOCUMENTS,
            len(documents),
            f">= {MIN_DOCUMENTS}",
            "The corpus must be large enough to support strategic analysis.",
        )
    )

    checks.append(
        check_pass(
            "Minimum source diversity",
            len(sources) >= MIN_SOURCES,
            len(sources),
            f">= {MIN_SOURCES}",
            "The system should not depend on only one source.",
        )
    )

    checks.append(
        check_pass(
            "Documents are chunked",
            len(chunks) >= len(documents),
            len(chunks),
            f">= document count {len(documents)}",
            "Chunking is required for reliable retrieval.",
        )
    )

    checks.append(
        check_pass(
            "Strategic signals exist",
            len(signals) >= MIN_SIGNALS,
            len(signals),
            f">= {MIN_SIGNALS}",
            "The system should extract opportunities, risks, and trends.",
        )
    )

    checks.append(
        check_pass(
            "CEO recommendations exist",
            len(recommendations) >= MIN_RECOMMENDATIONS,
            len(recommendations),
            f">= {MIN_RECOMMENDATIONS}",
            "The CEO agent needs multiple executive recommendations.",
        )
    )

    if len(recommendations) > 0:
        evidence_per_rec = len(evidence) / len(recommendations)
    else:
        evidence_per_rec = 0

    checks.append(
        check_pass(
            "Evidence per recommendation",
            evidence_per_rec >= MIN_EVIDENCE_PER_RECOMMENDATION,
            round(evidence_per_rec, 2),
            f">= {MIN_EVIDENCE_PER_RECOMMENDATION}",
            "Recommendations should be evidence-backed.",
        )
    )

    if not recommendations.empty:
        rec_evidence = read_df("""
            SELECT
                r.id,
                r.title,
                COUNT(re.id) AS evidence_links
            FROM recommendations r
            LEFT JOIN recommendation_evidence re ON r.id = re.recommendation_id
            GROUP BY r.id, r.title
        """)

        unsupported = int((rec_evidence["evidence_links"] == 0).sum())
    else:
        unsupported = 0

    checks.append(
        check_pass(
            "No unsupported recommendations",
            unsupported == 0,
            unsupported,
            "0",
            "Every recommendation should have at least one evidence link.",
        )
    )

    if table_exists("ceo_queries"):
        ceo_queries = read_df("SELECT * FROM ceo_queries")
        weak_answers = int((ceo_queries["evidence_count"].fillna(0) == 0).sum()) if not ceo_queries.empty else 0
        ceo_count = len(ceo_queries)
    else:
        ceo_count = 0
        weak_answers = 0

    checks.append(
        check_pass(
            "CEO Q&A sessions stored",
            ceo_count >= 1,
            ceo_count,
            ">= 1",
            "The CEO should be able to interrogate the system.",
        )
    )

    checks.append(
        check_pass(
            "CEO answers use evidence",
            weak_answers == 0,
            weak_answers,
            "0 weak answers",
            "Stored CEO answers should normally include retrieved evidence.",
        )
    )

    return checks


def run_retrieval_checks(top_k: int = 5) -> List[Dict]:
    results = []

    try:
        from app.retrieval.search_engine import AeroSearchEngine

        engine = AeroSearchEngine()

        for probe in RETRIEVAL_PROBES:
            retrieved = engine.search(probe["query"], top_k=top_k)

            combined_text = " ".join(
                f"{item.get('title', '')} {item.get('chunk_text', '')}".lower()
                for item in retrieved
            )

            matched_terms = [
                term for term in probe["expected_terms"]
                if term.lower() in combined_text
            ]

            avg_score = 0.0
            if retrieved:
                avg_score = sum(float(item.get("score") or 0) for item in retrieved) / len(retrieved)

            passed = len(retrieved) > 0 and len(matched_terms) >= 1

            results.append(
                check_pass(
                    f"Retrieval probe: {probe['query']}",
                    passed,
                    f"{len(retrieved)} results, avg score {round(avg_score, 3)}, matched {matched_terms}",
                    ">= 1 relevant result",
                    "Semantic memory should retrieve evidence relevant to strategic queries.",
                )
            )

    except Exception as exc:
        results.append(
            check_pass(
                "Retrieval engine available",
                False,
                str(exc),
                "FAISS index + metadata available",
                "Retrieval failed. Rebuild with: python -m app.retrieval.build_faiss_index",
            )
        )

    return results


def summarize_results(checks: List[Dict]) -> Dict:
    total = len(checks)
    passed = sum(1 for check in checks if check["status"] == "PASS")
    failed = total - passed

    score = round(passed / total, 3) if total else 0.0

    if score >= 0.90:
        grade = "Production-demo ready"
    elif score >= 0.75:
        grade = "Strong prototype"
    elif score >= 0.60:
        grade = "Needs improvement"
    else:
        grade = "Not ready"

    return {
        "total_checks": total,
        "passed": passed,
        "failed": failed,
        "evaluation_score": score,
        "readiness": grade,
    }


def run_full_evaluation(include_retrieval: bool = True) -> Dict:
    checks = run_database_checks()

    if include_retrieval:
        checks.extend(run_retrieval_checks())

    summary = summarize_results(checks)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "checks": checks,
    }


def save_evaluation_report(report: Dict, path: str = "reports/evaluation_report.json") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def report_to_markdown(report: Dict) -> str:
    summary = report["summary"]

    lines = []
    lines.append("# AERO-CEO Evaluation & Guardrails Report\n")
    lines.append(f"Generated at: {report['generated_at']}\n")
    lines.append("## Summary")
    lines.append(f"- Total checks: {summary['total_checks']}")
    lines.append(f"- Passed: {summary['passed']}")
    lines.append(f"- Failed: {summary['failed']}")
    lines.append(f"- Evaluation score: {summary['evaluation_score']}")
    lines.append(f"- Readiness: {summary['readiness']}")

    lines.append("\n## Checks")
    for check in report["checks"]:
        icon = "✅" if check["status"] == "PASS" else "❌"
        lines.append(f"- {icon} **{check['check']}**: {check['status']}")
        lines.append(f"  - Value: {check['value']}")
        lines.append(f"  - Threshold: {check['threshold']}")
        lines.append(f"  - Details: {check['details']}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate AERO-CEO system readiness.")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    report = run_full_evaluation(include_retrieval=not args.skip_retrieval)

    if args.save:
        save_evaluation_report(report)

    if args.markdown:
        print(report_to_markdown(report))
        return

    print("\n=== AERO-CEO Evaluation & Guardrails ===")
    print(json.dumps(report["summary"], indent=2))

    print("\nChecks:")
    for check in report["checks"]:
        print(f"{check['status']} | {check['check']} | value={check['value']} | threshold={check['threshold']}")


if __name__ == "__main__":
    main()

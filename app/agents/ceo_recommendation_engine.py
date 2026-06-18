import argparse
from typing import Dict, List, Tuple

from app.db.local_db import get_connection, init_local_db


RECOMMENDATION_BLUEPRINTS = [
    {
        "title": "Accelerate Airbus-led uncrewed combat aircraft and combat cloud capabilities",
        "signal_titles": [
            "Scale uncrewed combat aircraft and collaborative combat systems",
            "Strengthen Airbus position in FCAS and future combat architecture",
            "Defence aviation is shifting toward crewed-uncrewed teaming",
        ],
        "recommendation": (
            "Airbus should accelerate investment in uncrewed combat aircraft, SIRTAP/Eurodrone-related capabilities, "
            "crewed-uncrewed teaming, and combat cloud integration. This creates a near-term capability bridge while "
            "larger sixth-generation fighter programs continue to mature."
        ),
        "expected_impact": (
            "High strategic impact. This can strengthen Airbus Defence and Space in European sovereign defence, "
            "create exportable modular capabilities, improve Eurofighter relevance, and position Airbus as a lead "
            "systems integrator for future air combat."
        ),
        "risk_assessment": (
            "Risk level: Medium-High. Main risks include technical integration complexity, procurement timing, "
            "airspace certification issues, export controls, and dependency on national defence budgets."
        ),
        "priority": "High",
    },
    {
        "title": "Protect Airbus' strategic position in FCAS through modular architecture leadership",
        "signal_titles": [
            "Strengthen Airbus position in FCAS and future combat architecture",
            "FCAS fragmentation and partner misalignment risk",
            "Defence aviation is shifting toward crewed-uncrewed teaming",
        ],
        "recommendation": (
            "Airbus should reduce dependence on fragile programme-level alignment by leading modular FCAS architecture, "
            "mission systems, combat cloud, sensors, and crewed-uncrewed integration. The strategic focus should be on "
            "controllable system layers where Airbus can preserve influence even if partner politics shift."
        ),
        "expected_impact": (
            "High strategic impact. This improves Airbus' resilience in future fighter strategy, protects intellectual "
            "property, and strengthens negotiating power in European defence programmes."
        ),
        "risk_assessment": (
            "Risk level: High. FCAS is exposed to political disagreement, workshare disputes, national industrial policy, "
            "long procurement cycles, and competitor pressure."
        ),
        "priority": "High",
    },
    {
        "title": "Use Eurofighter modernization as a near-term revenue and capability bridge",
        "signal_titles": [
            "Expand Eurofighter modernization as a near-term defence growth bridge",
            "Competitor acceleration in fighter and defence systems",
            "Defence aviation is shifting toward crewed-uncrewed teaming",
        ],
        "recommendation": (
            "Airbus should prioritize Eurofighter modernization packages that connect radar, electronic warfare, "
            "mission software, and uncrewed teaming. This keeps Eurofighter strategically relevant while FCAS timelines "
            "remain long and uncertain."
        ),
        "expected_impact": (
            "Medium-High impact. This can generate nearer-term defence revenue, retain customer confidence, and create "
            "a practical migration path toward future air combat systems."
        ),
        "risk_assessment": (
            "Risk level: Medium. Key risks include customer budget limits, competition from F-35 and other fighter systems, "
            "integration complexity, and national procurement variation."
        ),
        "priority": "High",
    },
    {
        "title": "Expand military space and secure communications as a defence growth pillar",
        "signal_titles": [
            "Grow military space and secure communications business",
            "Military space resilience is growing in strategic importance",
            "Capture NATO and European defence autonomy demand",
        ],
        "recommendation": (
            "Airbus should expand its military satellite, secure communications, and space resilience portfolio. "
            "The company should package space capabilities with defence aviation, command-and-control, and sovereign "
            "European security requirements."
        ),
        "expected_impact": (
            "High impact. Military space can diversify defence revenue, support secure communications demand, and "
            "strengthen Airbus' role in European strategic autonomy."
        ),
        "risk_assessment": (
            "Risk level: Medium. Risks include competition from space-native firms, cybersecurity exposure, launch dependency, "
            "procurement delays, and fast technology cycles."
        ),
        "priority": "High",
    },
    {
        "title": "Build a European defence autonomy go-to-market strategy",
        "signal_titles": [
            "Capture NATO and European defence autonomy demand",
            "European defence autonomy is becoming a strategic procurement driver",
            "Procurement and export policy uncertainty",
        ],
        "recommendation": (
            "Airbus should create a unified go-to-market strategy around European defence autonomy, NATO interoperability, "
            "and sovereign industrial capacity. This should connect military aircraft, uncrewed systems, space, secure "
            "communications, and lifecycle services into one executive-level offer."
        ),
        "expected_impact": (
            "Medium-High impact. This can improve strategic positioning with European governments, increase cross-selling "
            "between business lines, and make Airbus more attractive in multinational procurement."
        ),
        "risk_assessment": (
            "Risk level: Medium. Risks include national procurement politics, export approval constraints, budget cycles, "
            "and potential friction between European partners."
        ),
        "priority": "Medium",
    },
    {
        "title": "Strengthen execution resilience in defence aerospace supply chains",
        "signal_titles": [
            "Defence aerospace supply chain and delivery risk",
            "Procurement and export policy uncertainty",
        ],
        "recommendation": (
            "Airbus should strengthen supply chain visibility, supplier risk monitoring, and production resilience for "
            "defence programmes such as A400M, A330 MRTT, C295, satellites, and uncrewed aircraft. Strategic growth will "
            "only convert into value if execution reliability is protected."
        ),
        "expected_impact": (
            "Medium impact. Better execution resilience can protect margins, improve delivery confidence, and reduce "
            "programme risk in defence and aerospace contracts."
        ),
        "risk_assessment": (
            "Risk level: Medium. Risks include supplier dependency, inflation, skilled labour availability, component "
            "shortages, and defence customer schedule pressure."
        ),
        "priority": "Medium",
    },
]


def reset_recommendations(conn):
    cur = conn.cursor()
    cur.execute("DELETE FROM recommendation_evidence")
    cur.execute("DELETE FROM recommendations")
    conn.commit()


def fetch_matching_signals(conn, signal_titles: List[str], limit: int = 12) -> List[Dict]:
    placeholders = ",".join(["?"] * len(signal_titles))

    query = f"""
        SELECT
            sig.id AS signal_id,
            sig.document_id,
            sig.signal_type,
            sig.topic,
            sig.title,
            sig.description,
            sig.impact_score,
            sig.urgency_score,
            sig.confidence_score,
            sig.evidence_text,
            d.title AS document_title,
            d.url,
            d.source_type,
            d.trust_score,
            s.name AS source_name
        FROM signals sig
        JOIN documents d ON sig.document_id = d.id
        LEFT JOIN sources s ON d.source_id = s.id
        WHERE sig.title IN ({placeholders})
        ORDER BY
            sig.confidence_score DESC,
            sig.impact_score DESC,
            sig.urgency_score DESC
        LIMIT ?
    """

    cur = conn.cursor()
    rows = cur.execute(query, (*signal_titles, limit)).fetchall()
    return [dict(row) for row in rows]


def calculate_recommendation_confidence(signals: List[Dict]) -> float:
    if not signals:
        return 0.0

    avg_conf = sum(float(s["confidence_score"] or 0) for s in signals) / len(signals)
    avg_impact = sum(float(s["impact_score"] or 0) for s in signals) / len(signals)

    unique_sources = len(set(s.get("source_name") or "unknown" for s in signals))
    unique_topics = len(set(s.get("topic") or "unknown" for s in signals))

    source_diversity = min(unique_sources / 5, 1.0)
    topic_diversity = min(unique_topics / 4, 1.0)

    score = (
        0.35 * avg_conf
        + 0.25 * avg_impact
        + 0.25 * source_diversity
        + 0.15 * topic_diversity
    )

    return round(min(score, 1.0), 3)


def insert_recommendation(conn, blueprint: Dict, signals: List[Dict]) -> int:
    confidence = calculate_recommendation_confidence(signals)

    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO recommendations (
            title,
            recommendation,
            priority,
            expected_impact,
            risk_assessment,
            confidence_score
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            blueprint["title"],
            blueprint["recommendation"],
            blueprint["priority"],
            blueprint["expected_impact"],
            blueprint["risk_assessment"],
            confidence,
        ),
    )
    conn.commit()

    recommendation_id = int(cur.lastrowid)

    for s in signals[:8]:
        evidence_strength = round(
            0.45 * float(s["confidence_score"] or 0)
            + 0.35 * float(s["impact_score"] or 0)
            + 0.20 * float(s["urgency_score"] or 0),
            3,
        )

        cur.execute(
            """
            INSERT INTO recommendation_evidence (
                recommendation_id,
                document_id,
                chunk_id,
                evidence_strength
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                recommendation_id,
                s["document_id"],
                None,
                evidence_strength,
            ),
        )

    conn.commit()
    return recommendation_id


def generate_recommendations(reset: bool = True):
    init_local_db()
    conn = get_connection()

    if reset:
        reset_recommendations(conn)

    created = []

    for blueprint in RECOMMENDATION_BLUEPRINTS:
        signals = fetch_matching_signals(conn, blueprint["signal_titles"], limit=14)

        if len(signals) < 2:
            continue

        recommendation_id = insert_recommendation(conn, blueprint, signals)

        created.append(
            {
                "id": recommendation_id,
                "title": blueprint["title"],
                "priority": blueprint["priority"],
                "confidence": calculate_recommendation_confidence(signals),
                "evidence_count": len(signals),
            }
        )

    conn.close()
    return created


def print_recommendations():
    conn = get_connection()
    cur = conn.cursor()

    print("\n=== CEO Recommendations ===")

    recs = cur.execute(
        """
        SELECT id, title, priority, confidence_score, recommendation, expected_impact, risk_assessment
        FROM recommendations
        ORDER BY
            CASE priority
                WHEN 'High' THEN 1
                WHEN 'Medium' THEN 2
                ELSE 3
            END,
            confidence_score DESC
        """
    ).fetchall()

    for rec in recs:
        print("\n" + "=" * 110)
        print(f"ID: {rec['id']}")
        print(f"Title: {rec['title']}")
        print(f"Priority: {rec['priority']} | Confidence: {rec['confidence_score']}")
        print("\nRecommendation:")
        print(rec["recommendation"])
        print("\nExpected impact:")
        print(rec["expected_impact"])
        print("\nRisk assessment:")
        print(rec["risk_assessment"])

        evidence = cur.execute(
            """
            SELECT
                re.evidence_strength,
                d.title AS document_title,
                d.url,
                d.topic,
                d.source_type,
                s.name AS source_name
            FROM recommendation_evidence re
            JOIN documents d ON re.document_id = d.id
            LEFT JOIN sources s ON d.source_id = s.id
            WHERE re.recommendation_id = ?
            ORDER BY re.evidence_strength DESC
            LIMIT 5
            """,
            (rec["id"],),
        ).fetchall()

        print("\nSupporting evidence:")
        for i, ev in enumerate(evidence, start=1):
            print(
                f"{i}. strength={ev['evidence_strength']} | "
                f"{ev['source_type']} | {ev['topic']} | {ev['source_name']} | {ev['document_title'][:90]}"
            )
            print(f"   {ev['url']}")

    print("\nTotal recommendations:", len(recs))
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Generate CEO-level strategic recommendations.")
    parser.add_argument("--no-reset", action="store_true")
    args = parser.parse_args()

    created = generate_recommendations(reset=not args.no_reset)

    print(f"Recommendations created: {len(created)}")
    for item in created:
        print(
            f"{item['id']} | {item['priority']} | conf={item['confidence']} | "
            f"evidence={item['evidence_count']} | {item['title']}"
        )

    print_recommendations()


if __name__ == "__main__":
    main()

import argparse
import re
from typing import Dict, List

from app.db.local_db import get_connection, init_local_db
from app.intelligence.scoring import confidence_score, impact_score, urgency_score


ENTITY_PATTERNS = [
    "Airbus",
    "Airbus Defence and Space",
    "Dassault",
    "Boeing",
    "Lockheed Martin",
    "BAE Systems",
    "Leonardo",
    "Saab",
    "Thales",
    "Rheinmetall",
    "NATO",
    "European Union",
    "European Defence Agency",
    "German Air Force",
    "Spanish Ministry of Defence",
    "French Armed Forces",
    "Eurofighter",
    "FCAS",
    "SIRTAP",
    "Eurodrone",
    "A400M",
    "A330 MRTT",
    "C295",
]


OPPORTUNITY_RULES = [
    {
        "title": "Scale uncrewed combat aircraft and collaborative combat systems",
        "topic_match": ["Uncrewed Combat Aircraft"],
        "keywords": ["uncrewed", "unmanned", "uav", "uas", "sirtap", "eurodrone", "collaborative combat", "drone"],
        "description": "Airbus has an opportunity to expand sovereign European uncrewed combat aircraft and collaborative combat capabilities.",
        "evidence_strength": 0.86,
    },
    {
        "title": "Strengthen Airbus position in FCAS and future combat architecture",
        "topic_match": ["FCAS"],
        "keywords": ["fcas", "future combat air system", "sixth-generation", "combat cloud", "fighter"],
        "description": "Airbus can increase strategic influence in future combat aviation by leading modular combat cloud, systems integration, and crewed-uncrewed teaming capabilities.",
        "evidence_strength": 0.84,
    },
    {
        "title": "Expand Eurofighter modernization as a near-term defence growth bridge",
        "topic_match": ["Eurofighter"],
        "keywords": ["eurofighter", "typhoon", "modernization", "upgrade", "radar", "combat"],
        "description": "Eurofighter modernization can provide near-term revenue and capability relevance while longer-term sixth-generation programs mature.",
        "evidence_strength": 0.78,
    },
    {
        "title": "Grow military space and secure communications business",
        "topic_match": ["Military Space"],
        "keywords": ["satellite", "space", "secure communications", "earth observation", "military satellite"],
        "description": "Airbus can expand in military satellites, secure communications, and sovereign space infrastructure as defence customers increase resilience requirements.",
        "evidence_strength": 0.80,
    },
    {
        "title": "Capture NATO and European defence autonomy demand",
        "topic_match": ["European Defence Autonomy", "Policy and Procurement"],
        "keywords": ["nato", "european defence", "strategic autonomy", "sovereign", "procurement", "defence spending"],
        "description": "Rising European defence autonomy and NATO procurement demand create opportunities for Airbus across aircraft, space, and secure communications.",
        "evidence_strength": 0.82,
    },
]


RISK_RULES = [
    {
        "title": "FCAS fragmentation and partner misalignment risk",
        "topic_match": ["FCAS"],
        "keywords": ["dispute", "delay", "collapse", "tension", "dassault", "workshare", "fragmentation", "uncertainty"],
        "description": "Political or industrial disagreement in FCAS could weaken Airbus' future fighter strategy and delay capability development.",
        "evidence_strength": 0.88,
    },
    {
        "title": "Competitor acceleration in fighter and defence systems",
        "topic_match": ["Eurofighter", "FCAS", "Competitor Activity"],
        "keywords": ["lockheed", "boeing", "bae systems", "dassault", "leonardo", "saab", "gcAP", "tempest", "f-35"],
        "description": "Competitor activity in fighter aircraft, sensors, drones, and combat systems may pressure Airbus' market position.",
        "evidence_strength": 0.78,
    },
    {
        "title": "Defence aerospace supply chain and delivery risk",
        "topic_match": ["Supply Chain", "Military Transport", "Aerial Refuelling"],
        "keywords": ["supply chain", "delay", "shortage", "production", "delivery", "cost overrun"],
        "description": "Supply chain disruption, delivery pressure, or cost overruns could reduce execution reliability in Airbus defence programs.",
        "evidence_strength": 0.76,
    },
    {
        "title": "Procurement and export policy uncertainty",
        "topic_match": ["Policy and Procurement", "European Defence Autonomy"],
        "keywords": ["export", "regulation", "procurement", "budget", "ministry", "parliament", "approval"],
        "description": "Defence procurement cycles, export approvals, and national budget politics may create uncertainty for Airbus strategic planning.",
        "evidence_strength": 0.72,
    },
]


TREND_RULES = [
    {
        "title": "Defence aviation is shifting toward crewed-uncrewed teaming",
        "topic_match": ["Uncrewed Combat Aircraft", "Eurofighter", "FCAS"],
        "keywords": ["crewed-uncrewed", "manned-unmanned", "teaming", "collaborative combat", "loyal wingman", "uncrewed"],
        "description": "Combat aviation is moving toward mixed teams of piloted aircraft, drones, sensors, and cloud-connected mission systems.",
        "evidence_strength": 0.84,
    },
    {
        "title": "European defence autonomy is becoming a strategic procurement driver",
        "topic_match": ["European Defence Autonomy", "Policy and Procurement"],
        "keywords": ["european defence", "sovereign", "strategic autonomy", "nato", "procurement"],
        "description": "European customers increasingly value sovereign and interoperable defence capabilities.",
        "evidence_strength": 0.82,
    },
    {
        "title": "Military space resilience is growing in strategic importance",
        "topic_match": ["Military Space"],
        "keywords": ["space", "satellite", "secure communications", "resilience", "earth observation"],
        "description": "Military space, satellite communications, and resilient information infrastructure are becoming central to defence strategy.",
        "evidence_strength": 0.78,
    },
]


def extract_entities(text: str) -> List[str]:
    found = []
    lower = text.lower()

    for entity in ENTITY_PATTERNS:
        if entity.lower() in lower:
            found.append(entity)

    return sorted(set(found))


def keyword_match(text: str, keywords: List[str]) -> bool:
    lower = text.lower()
    return any(keyword.lower() in lower for keyword in keywords)


def topic_match(document_topic: str, allowed_topics: List[str]) -> bool:
    return document_topic in allowed_topics


def get_evidence_snippet(text: str, keywords: List[str], max_chars: int = 550) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    lower = clean.lower()

    positions = [lower.find(k.lower()) for k in keywords if lower.find(k.lower()) >= 0]
    if not positions:
        return clean[:max_chars]

    start = max(min(positions) - 140, 0)
    end = min(start + max_chars, len(clean))

    return clean[start:end]


def detect_signals_for_doc(doc: Dict) -> List[Dict]:
    text = f"{doc.get('title') or ''}. {doc.get('clean_text') or ''}"
    document_topic = doc.get("topic") or "General Airbus Strategy"
    published_at = doc.get("published_at")
    trust_score = float(doc.get("trust_score") or 0.5)
    source_type = doc.get("source_type") or "unknown"

    output = []

    rule_groups = [
        ("opportunity", OPPORTUNITY_RULES),
        ("risk", RISK_RULES),
        ("trend", TREND_RULES),
    ]

    for signal_type, rules in rule_groups:
        for rule in rules:
            topic_ok = topic_match(document_topic, rule["topic_match"])
            keyword_ok = keyword_match(text, rule["keywords"])

            if not (topic_ok or keyword_ok):
                continue

            evidence = get_evidence_snippet(text, rule["keywords"])
            conf = confidence_score(
                trust_score=trust_score,
                published_at=published_at,
                evidence_strength=rule["evidence_strength"],
                source_type=source_type,
            )

            output.append(
                {
                    "document_id": doc["id"],
                    "signal_type": signal_type,
                    "topic": document_topic,
                    "title": rule["title"],
                    "description": rule["description"],
                    "entities": ", ".join(extract_entities(text)),
                    "impact_score": impact_score(document_topic, signal_type, text),
                    "urgency_score": urgency_score(text, published_at),
                    "confidence_score": conf,
                    "evidence_text": evidence,
                }
            )

    return output


def reset_signals(conn):
    cur = conn.cursor()
    cur.execute("DELETE FROM signals")
    conn.commit()


def fetch_documents(conn) -> List[Dict]:
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT id, title, clean_text, source_type, topic, trust_score, published_at
        FROM documents
        WHERE clean_text IS NOT NULL
          AND LENGTH(clean_text) > 40
        ORDER BY id
        """
    ).fetchall()

    return [dict(row) for row in rows]


def insert_signal(conn, signal: Dict):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO signals (
            document_id,
            signal_type,
            topic,
            title,
            description,
            entities,
            impact_score,
            urgency_score,
            confidence_score,
            evidence_text
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            signal["document_id"],
            signal["signal_type"],
            signal["topic"],
            signal["title"],
            signal["description"],
            signal["entities"],
            signal["impact_score"],
            signal["urgency_score"],
            signal["confidence_score"],
            signal["evidence_text"],
        ),
    )
    conn.commit()


def print_summary(conn):
    cur = conn.cursor()

    print("\n=== Strategic Signal Summary ===")
    print("Signals:", cur.execute("SELECT COUNT(*) FROM signals").fetchone()[0])

    print("\nBy signal type:")
    for row in cur.execute("SELECT signal_type, COUNT(*) AS n FROM signals GROUP BY signal_type ORDER BY n DESC"):
        print(row["signal_type"], row["n"])

    print("\nTop signal titles:")
    for row in cur.execute("""
        SELECT title, signal_type, COUNT(*) AS n, ROUND(AVG(confidence_score), 3) AS avg_conf
        FROM signals
        GROUP BY title, signal_type
        ORDER BY n DESC
        LIMIT 12
    """):
        print(f"{row['signal_type']} | {row['n']} | conf={row['avg_conf']} | {row['title']}")

    print("\nHighest confidence signals:")
    for row in cur.execute("""
        SELECT signal_type, title, topic, confidence_score, impact_score
        FROM signals
        ORDER BY confidence_score DESC, impact_score DESC
        LIMIT 10
    """):
        print(
            f"{row['confidence_score']} | impact={row['impact_score']} | "
            f"{row['signal_type']} | {row['topic']} | {row['title']}"
        )


def main():
    parser = argparse.ArgumentParser(description="Extract strategic signals from Airbus intelligence corpus.")
    parser.add_argument("--reset", action="store_true", help="Reset existing signals before extraction.")
    args = parser.parse_args()

    init_local_db()
    conn = get_connection()

    if args.reset:
        reset_signals(conn)

    docs = fetch_documents(conn)
    print(f"Documents scanned: {len(docs)}")

    inserted = 0

    for doc in docs:
        signals = detect_signals_for_doc(doc)
        for signal in signals:
            insert_signal(conn, signal)
            inserted += 1

    print(f"Signals inserted: {inserted}")
    print_summary(conn)

    conn.close()


if __name__ == "__main__":
    main()

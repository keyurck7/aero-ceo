import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from app.agents.query_router import route_query
from app.db.local_db import get_connection, init_local_db
from app.db.chat_schema import ensure_ceo_chat_tables
from app.retrieval.search_engine import AeroSearchEngine
from app.agents.llm_strategy_writer import generate_llm_strategy_answer


PARTNER_CAPABILITY_MAP = {
    "Indra": {
        "country": "Spain",
        "capability": "systems integration, sensors, defence electronics, Spanish FCAS ecosystem",
        "collaboration_area": "mission systems, combat cloud, national programme alignment",
        "risk": "national workshare politics and programme governance complexity",
    },
    "Dassault": {
        "country": "France",
        "capability": "fighter aircraft design and combat aviation leadership",
        "collaboration_area": "future fighter platform interfaces and European air combat alignment",
        "risk": "leadership disputes, IP ownership tension, industrial workshare conflict",
    },
    "Leonardo": {
        "country": "Italy",
        "capability": "avionics, sensors, helicopters, aircraft systems, defence electronics",
        "collaboration_area": "sensors, electronic warfare, avionics, aircraft subsystems",
        "risk": "overlapping industrial interests and procurement politics",
    },
    "BAE Systems": {
        "country": "United Kingdom",
        "capability": "combat aircraft, GCAP/Tempest experience, mission systems",
        "collaboration_area": "sixth-generation aircraft concepts, mission systems, interoperability",
        "risk": "UK-EU alignment, export-control complexity, competing programme priorities",
    },
    "Thales": {
        "country": "France",
        "capability": "radars, sensors, secure communications, cyber-secure mission systems",
        "collaboration_area": "combat cloud, secure communications, sensors, cyber resilience",
        "risk": "supplier dependency and sensitive technology governance",
    },
    "Rheinmetall": {
        "country": "Germany",
        "capability": "land systems, battlefield integration, defence electronics",
        "collaboration_area": "multi-domain integration, battlefield networks, German defence ecosystem",
        "risk": "portfolio mismatch if collaboration is not focused on multi-domain integration",
    },
    "Saab": {
        "country": "Sweden",
        "capability": "fighter aircraft, agile defence development, sensors",
        "collaboration_area": "fighter interoperability, smaller-nation defence requirements, rapid development",
        "risk": "scale limitations and programme alignment uncertainty",
    },
    "OHB": {
        "country": "Germany",
        "capability": "space systems and satellites",
        "collaboration_area": "military space, secure communications, satellite resilience",
        "risk": "space programme procurement cycles and launch dependency",
    },
    "GMV": {
        "country": "Spain",
        "capability": "space, navigation, defence software, mission systems",
        "collaboration_area": "space command systems, navigation, military software",
        "risk": "integration complexity across space and defence aviation systems",
    },
}


RISK_LIBRARY = {
    "FCAS": [
        "Political and industrial workshare disputes",
        "Long development timelines",
        "IP ownership and leadership tension",
        "Dependency on multinational alignment",
        "Competition from alternative fighter programmes",
    ],
    "Uncrewed Combat Aircraft": [
        "Autonomy integration complexity",
        "Certification and operational doctrine uncertainty",
        "Export-control restrictions",
        "Cybersecurity and datalink resilience",
        "Procurement timing risk",
    ],
    "Military Space": [
        "Cybersecurity exposure",
        "Launch dependency",
        "Competition from space-native firms",
        "Fast technology cycles",
        "Procurement delays",
    ],
    "Eurofighter": [
        "Competition from F-35 and other fighter ecosystems",
        "Customer budget constraints",
        "Integration complexity for modernization packages",
        "Long-term relevance risk if future combat migration is slow",
    ],
    "Supply Chain": [
        "Supplier dependency",
        "Component shortages",
        "Inflation pressure",
        "Skilled labour availability",
        "Delivery schedule risk",
    ],
    "General": [
        "Procurement uncertainty",
        "Execution complexity",
        "Competitor pressure",
        "Regulatory and export-control risk",
        "Budget and political alignment risk",
    ],
}


OPTION_LIBRARY = {
    "conservative": "Prioritize lower-risk upgrades, evidence monitoring, supplier resilience, and modular investments that preserve optionality.",
    "balanced": "Invest in near-term capability improvements while building modular future-combat and space capabilities with selected partners.",
    "aggressive": "Move faster into high-impact strategic bets such as combat cloud, uncrewed aircraft, and future fighter architecture leadership.",
}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _short(text: str, limit: int = 420) -> str:
    text = _clean(text)
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


class CEOChatAgent:
    def __init__(self):
        init_local_db()
        self.conn = get_connection()
        ensure_ceo_chat_tables(self.conn)
        self.search_engine = self._load_search_engine()

    def _load_search_engine(self) -> Optional[AeroSearchEngine]:
        index_path = os.getenv("FAISS_INDEX_PATH", "data/cache/aero_ceo.faiss")
        metadata_path = os.getenv("FAISS_METADATA_PATH", "data/cache/faiss_metadata.json")

        if not Path(index_path).exists() or not Path(metadata_path).exists():
            return None

        return AeroSearchEngine()

    def close(self):
        self.conn.close()

    def answer(self, question: str, top_k: int = 8) -> Dict:
        route = route_query(question)

        evidence = self.retrieve_evidence(question, route, top_k=top_k)
        signals = self.fetch_related_signals(route, limit=10)
        recommendations = self.fetch_related_recommendations(route, question, limit=6)
        partners = self.analyze_partners(question, evidence)

        confidence = self.calculate_answer_confidence(evidence, signals, recommendations)
        answer_markdown = self.compose_answer(
            question=question,
            route=route,
            evidence=evidence,
            signals=signals,
            recommendations=recommendations,
            partners=partners,
            confidence=confidence,
        )

        if os.getenv("LLM_GENERATION_ENABLED", "false").lower() == "true":
            try:
                answer_markdown = generate_llm_strategy_answer(
                    question=question,
                    route=route,
                    evidence=evidence,
                    signals=signals,
                    recommendations=recommendations,
                    partners=partners,
                    confidence=confidence,
                    template_answer=answer_markdown,
                )
            except Exception as exc:
                answer_markdown = (
                    answer_markdown
                    + "\n\n---\n"
                    + f"**Local LLM generation warning:** {exc}\n\n"
                    + "The system returned the deterministic evidence-grounded answer instead."
                )

        if len(evidence) < 2 or confidence < 0.45:
            guardrail_note = (
                "\n\n---\n"
                "## Evidence Sufficiency Guardrail\n"
                "AERO-CEO found limited supporting evidence for this answer. "
                "Treat the response as exploratory rather than decision-ready. "
                "Collect more sources or broaden the query before making an executive decision.\n"
            )
            answer_markdown = answer_markdown + guardrail_note

        self.save_query(
            question=question,
            route=route,
            answer_markdown=answer_markdown,
            confidence=confidence,
            evidence_count=len(evidence),
        )

        return {
            "question": question,
            "route": route,
            "confidence": confidence,
            "evidence_count": len(evidence),
            "answer_markdown": answer_markdown,
        }

    def retrieve_evidence(self, question: str, route: Dict, top_k: int = 8) -> List[Dict]:
        if not self.search_engine:
            return []

        search_query = question

        if route.get("topic"):
            search_query += f" Airbus {route['topic']}"

        if route.get("business_area"):
            search_query += f" {route['business_area']}"

        results = self.search_engine.search(
            query=search_query,
            top_k=top_k,
            topic=None,
            source_type=None,
        )

        return results

    def fetch_related_signals(self, route: Dict, limit: int = 10) -> List[Dict]:
        cur = self.conn.cursor()

        topic = route.get("topic")
        intent = route.get("intent")

        params = []
        where = []

        if topic:
            where.append("topic = ?")
            params.append(topic)

        if intent == "risk_analysis":
            where.append("signal_type = 'risk'")
        elif intent == "opportunity_analysis":
            where.append("signal_type = 'opportunity'")
        elif intent == "partnership_strategy":
            where.append("signal_type IN ('opportunity', 'risk', 'trend')")
        elif intent == "scenario_analysis":
            where.append("signal_type IN ('risk', 'trend', 'opportunity')")

        where_sql = "WHERE " + " AND ".join(where) if where else ""

        rows = cur.execute(
            f"""
            SELECT *
            FROM signals
            {where_sql}
            ORDER BY confidence_score DESC, impact_score DESC, urgency_score DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()

        return [dict(row) for row in rows]

    def fetch_related_recommendations(self, route: Dict, question: str, limit: int = 6) -> List[Dict]:
        cur = self.conn.cursor()

        topic = (route.get("topic") or "").lower()
        question_lower = question.lower()

        rows = cur.execute(
            """
            SELECT *
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

        recs = [dict(row) for row in rows]

        if not topic and not question_lower:
            return recs[:limit]

        scored = []
        for rec in recs:
            blob = f"{rec.get('title', '')} {rec.get('recommendation', '')} {rec.get('expected_impact', '')}".lower()
            score = 0

            if topic and topic in blob:
                score += 3

            for term in question_lower.split():
                if len(term) >= 5 and term in blob:
                    score += 1

            scored.append((score, rec))

        scored.sort(key=lambda x: (x[0], x[1].get("confidence_score") or 0), reverse=True)

        selected = [rec for score, rec in scored if score > 0]
        if not selected:
            selected = recs

        return selected[:limit]

    def analyze_partners(self, question: str, evidence: List[Dict]) -> List[Dict]:
        lower_question = question.lower()
        evidence_text = " ".join(
            f"{item.get('title', '')} {item.get('chunk_text', '')}"
            for item in evidence
        ).lower()

        should_analyze = any(
            term in lower_question
            for term in ["partner", "collab", "collaboration", "alliance", "country", "organisation", "organization"]
        )

        if not should_analyze:
            return []

        partners = []

        cur = self.conn.cursor()

        for name, info in PARTNER_CAPABILITY_MAP.items():
            mention_count = cur.execute(
                """
                SELECT COUNT(*)
                FROM documents
                WHERE LOWER(title || ' ' || clean_text) LIKE ?
                """,
                (f"%{name.lower()}%",),
            ).fetchone()[0]

            if name.lower() in lower_question or name.lower() in evidence_text or mention_count > 0:
                relevance = min(1.0, 0.55 + 0.07 * mention_count)

                partners.append(
                    {
                        "name": name,
                        "country": info["country"],
                        "capability": info["capability"],
                        "collaboration_area": info["collaboration_area"],
                        "risk": info["risk"],
                        "evidence_mentions": mention_count,
                        "relevance": round(relevance, 3),
                    }
                )

        partners.sort(key=lambda x: (x["relevance"], x["evidence_mentions"]), reverse=True)
        return partners[:8]

    def calculate_answer_confidence(
        self,
        evidence: List[Dict],
        signals: List[Dict],
        recommendations: List[Dict],
    ) -> float:
        if not evidence and not signals and not recommendations:
            return 0.0

        evidence_score = 0.0
        if evidence:
            evidence_score = sum(float(item.get("score") or 0) for item in evidence) / len(evidence)

        signal_score = 0.0
        if signals:
            signal_score = sum(float(item.get("confidence_score") or 0) for item in signals) / len(signals)

        rec_score = 0.0
        if recommendations:
            rec_score = sum(float(item.get("confidence_score") or 0) for item in recommendations) / len(recommendations)

        source_diversity = 0.0
        if evidence:
            source_diversity = min(len(set(item.get("source_name") for item in evidence)) / 5, 1.0)

        score = (
            0.35 * evidence_score
            + 0.30 * signal_score
            + 0.20 * rec_score
            + 0.15 * source_diversity
        )

        return round(min(max(score, 0.0), 1.0), 3)

    def compose_answer(
        self,
        question: str,
        route: Dict,
        evidence: List[Dict],
        signals: List[Dict],
        recommendations: List[Dict],
        partners: List[Dict],
        confidence: float,
    ) -> str:
        intent = route.get("intent")
        topic = route.get("topic") or "Airbus strategy"
        business_area = route.get("business_area") or "Airbus"
        region = route.get("region") or "relevant markets"
        constraints = route.get("constraints") or []

        answer = []

        answer.append(f"# AERO-CEO Strategic Answer\n")
        answer.append(f"**Question:** {question}\n")
        answer.append("## Query Understanding")
        answer.append(f"- **Intent:** {intent}")
        answer.append(f"- **Business area:** {business_area}")
        answer.append(f"- **Topic:** {topic}")
        answer.append(f"- **Region:** {region}")
        answer.append(f"- **Constraints:** {', '.join(constraints) if constraints else 'None detected'}")
        answer.append(f"- **Answer confidence:** {confidence}\n")

        answer.append("## Executive Answer")

        if intent == "partnership_strategy":
            answer.append(
                f"Airbus should consider partnership for **{topic}**, but the safest strategy is not to depend on one large all-or-nothing programme structure. "
                f"The stronger move is to collaborate around modular capability layers: mission systems, combat cloud, uncrewed teaming, sensors, secure communications, and space-enabled command systems. "
                f"This gives Airbus strategic flexibility while reducing exposure to political or industrial workshare disputes."
            )
        elif intent == "risk_analysis":
            answer.append(
                f"The main risks around **{topic}** are execution complexity, procurement uncertainty, partner alignment, competitor pressure, and technology integration. "
                f"Airbus should treat these as strategic management risks, not only technical risks."
            )
        elif intent == "opportunity_analysis":
            answer.append(
                f"The strongest opportunity around **{topic}** is to convert current market signals into a focused capability roadmap. "
                f"Airbus should prioritize areas where evidence shows customer demand, sovereign defence relevance, and reuse across multiple programmes."
            )
        elif intent == "scenario_analysis":
            answer.append(
                f"Under this scenario, Airbus should preserve optionality. "
                f"The best response is to separate near-term executable actions from long-term strategic bets, then invest in modular capabilities that remain valuable even if programme timelines change."
            )
        elif intent == "competitor_analysis":
            answer.append(
                f"Competitor pressure should be treated as a signal for faster differentiation. "
                f"Airbus should focus on areas where it can combine aircraft, space, secure communications, and systems integration into a defensible European value proposition."
            )
        else:
            answer.append(
                f"Based on the retrieved Airbus intelligence, the strategic answer is to prioritize evidence-backed actions in **{business_area}** around **{topic}**. "
                f"The recommendation should balance near-term execution, long-term positioning, partner dependency, and risk exposure."
            )

        answer.append("\n## Strategic Recommendation")

        if recommendations:
            top_rec = recommendations[0]
            answer.append(f"**Primary recommendation:** {top_rec['title']}")
            answer.append("")
            answer.append(top_rec["recommendation"])
            answer.append("")
            answer.append(f"**Priority:** {top_rec['priority']} | **Recommendation confidence:** {top_rec['confidence_score']}")
        else:
            answer.append(
                "Generate a focused executive action plan from the retrieved evidence before making a final investment decision."
            )

        if partners:
            answer.append("\n## Partnership Options")
            answer.append("| Partner | Country | Capability | Best collaboration area | Risk | Evidence mentions |")
            answer.append("|---|---|---|---|---|---|")
            for partner in partners:
                answer.append(
                    f"| {partner['name']} | {partner['country']} | {partner['capability']} | "
                    f"{partner['collaboration_area']} | {partner['risk']} | {partner['evidence_mentions']} |"
                )

        answer.append("\n## Decision Options")
        answer.append(f"- **Conservative option:** {OPTION_LIBRARY['conservative']}")
        answer.append(f"- **Balanced option:** {OPTION_LIBRARY['balanced']}")
        answer.append(f"- **Aggressive option:** {OPTION_LIBRARY['aggressive']}")

        answer.append("\n## Risk Assessment")
        risk_items = RISK_LIBRARY.get(route.get("topic") or "", RISK_LIBRARY["General"])
        for risk in risk_items:
            answer.append(f"- {risk}")

        if signals:
            answer.append("\n## Strategic Signals Used")
            for signal in signals[:6]:
                answer.append(
                    f"- **{signal['signal_type'].title()} | {signal['topic']} | confidence {signal['confidence_score']}:** "
                    f"{signal['title']}"
                )

        answer.append("\n## Evidence Used")
        if evidence:
            for idx, item in enumerate(evidence[:8], start=1):
                answer.append(
                    f"{idx}. **{item.get('title')}**  \n"
                    f"   Source: {item.get('source_name')} | Type: {item.get('source_type')} | Topic: {item.get('topic')} | Retrieval score: {float(item.get('score') or 0):.3f}  \n"
                    f"   URL: {item.get('url')}  \n"
                    f"   Evidence snippet: {_short(item.get('chunk_text'), 360)}"
                )
        else:
            answer.append(
                "No vector evidence was available. Rebuild the FAISS index with `python -m app.retrieval.build_faiss_index`."
            )

        answer.append("\n## Confidence Explanation")
        if confidence >= 0.75:
            answer.append("Confidence is **High** because the answer is supported by retrieved evidence, strategic signals, and existing recommendations.")
        elif confidence >= 0.50:
            answer.append("Confidence is **Medium** because there is useful evidence, but the decision should be validated with more source diversity or fresher documents.")
        else:
            answer.append("Confidence is **Low** because the available evidence is limited or weak. The system should collect more documents before making a strong recommendation.")

        answer.append("\n## Follow-up Questions the CEO Could Ask")
        answer.append("- What is the lower-risk version of this recommendation?")
        answer.append("- Which evidence is strongest and which is weakest?")
        answer.append("- What changes if budget is limited?")
        answer.append("- Which partner gives Airbus the best strategic leverage?")
        answer.append("- What should Airbus do in the next 6 months?")

        return "\n".join(answer)

    def save_query(
        self,
        question: str,
        route: Dict,
        answer_markdown: str,
        confidence: float,
        evidence_count: int,
    ) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO ceo_queries (
                question,
                intent,
                topic,
                business_area,
                region,
                answer_markdown,
                confidence_score,
                evidence_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                question,
                route.get("intent"),
                route.get("topic"),
                route.get("business_area"),
                route.get("region"),
                answer_markdown,
                confidence,
                evidence_count,
            ),
        )
        self.conn.commit()


def save_answer_to_markdown(answer_markdown: str, path: str = "reports/last_ceo_answer.md") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(answer_markdown)


def main():
    parser = argparse.ArgumentParser(description="Ask AERO-CEO a strategic question.")
    parser.add_argument("question", nargs="*", help="Question to ask the CEO agent.")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    agent = CEOChatAgent()

    try:
        if args.interactive:
            print("AERO-CEO interactive mode. Type 'exit' to stop.\n")
            while True:
                question = input("CEO> ").strip()
                if question.lower() in {"exit", "quit", "q"}:
                    break

                result = agent.answer(question, top_k=args.top_k)
                print("\n" + result["answer_markdown"] + "\n")
                save_answer_to_markdown(result["answer_markdown"])
                print("Saved latest answer to reports/last_ceo_answer.md\n")
        else:
            question = " ".join(args.question).strip()
            if not question:
                raise ValueError("Please provide a question or use --interactive.")

            result = agent.answer(question, top_k=args.top_k)
            print(result["answer_markdown"])
            save_answer_to_markdown(result["answer_markdown"])
            print("\nSaved latest answer to reports/last_ceo_answer.md")
    finally:
        agent.close()


if __name__ == "__main__":
    main()

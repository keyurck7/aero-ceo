import argparse
from typing import Dict, List, Optional

from app.db.local_db import get_connection, init_local_db
from app.agents.ceo_chat_agent import CEOChatAgent


DRILLDOWN_ACTIONS = {
    "Why this recommendation?": (
        "Explain why Airbus should prioritize this recommendation. "
        "Include strategic logic, evidence, risks, and expected impact."
    ),
    "Show strongest evidence": (
        "Show the strongest evidence supporting this recommendation. "
        "Rank the evidence and explain why it matters."
    ),
    "Lower-risk version": (
        "Create a lower-risk version of this recommendation. "
        "Preserve strategic value but reduce financial, operational, and political risk."
    ),
    "Budget-constrained version": (
        "Assume Airbus has limited budget. "
        "What should management do first, what should be delayed, and what should be avoided?"
    ),
    "Partnership options": (
        "Identify possible partner organizations and countries that could help Airbus execute this recommendation. "
        "Include capability fit, collaboration area, and risks."
    ),
    "Next 6 months action plan": (
        "Convert this recommendation into a practical 6-month action plan for Airbus leadership."
    ),
    "What could go wrong?": (
        "Stress-test this recommendation. "
        "Identify failure modes, early warning signs, and mitigation actions."
    ),
}


def fetch_recommendations() -> List[Dict]:
    init_local_db()
    conn = get_connection()
    cur = conn.cursor()

    rows = cur.execute(
        """
        SELECT id, title, recommendation, priority, expected_impact,
               risk_assessment, confidence_score, created_at
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

    conn.close()
    return [dict(row) for row in rows]


def fetch_recommendation_by_id(recommendation_id: int) -> Optional[Dict]:
    init_local_db()
    conn = get_connection()
    cur = conn.cursor()

    row = cur.execute(
        """
        SELECT id, title, recommendation, priority, expected_impact,
               risk_assessment, confidence_score, created_at
        FROM recommendations
        WHERE id = ?
        """,
        (recommendation_id,),
    ).fetchone()

    conn.close()
    return dict(row) if row else None


def fetch_recommendation_evidence(recommendation_id: int, limit: int = 8) -> List[Dict]:
    init_local_db()
    conn = get_connection()
    cur = conn.cursor()

    rows = cur.execute(
        """
        SELECT
            re.evidence_strength,
            d.title AS document_title,
            d.url,
            d.topic,
            d.source_type,
            d.clean_text,
            s.name AS source_name
        FROM recommendation_evidence re
        JOIN documents d ON re.document_id = d.id
        LEFT JOIN sources s ON d.source_id = s.id
        WHERE re.recommendation_id = ?
        ORDER BY re.evidence_strength DESC
        LIMIT ?
        """,
        (recommendation_id, limit),
    ).fetchall()

    conn.close()
    return [dict(row) for row in rows]


def build_drilldown_question(
    recommendation: Dict,
    action: str,
    custom_condition: str = "",
) -> str:
    action_instruction = DRILLDOWN_ACTIONS.get(action, DRILLDOWN_ACTIONS["Why this recommendation?"])

    condition_text = ""
    if custom_condition.strip():
        condition_text = f"\nAdditional CEO condition: {custom_condition.strip()}"

    return f"""
CEO wants a recommendation drill-down.

Recommendation title:
{recommendation.get("title")}

Current recommendation:
{recommendation.get("recommendation")}

Priority:
{recommendation.get("priority")}

Expected impact:
{recommendation.get("expected_impact")}

Risk assessment:
{recommendation.get("risk_assessment")}

Drill-down task:
{action_instruction}
{condition_text}

Answer as a strategic Airbus CEO advisor. Use evidence, decision options, risks, confidence, and concrete next actions.
""".strip()


def run_drilldown(
    recommendation_id: int,
    action: str,
    custom_condition: str = "",
    top_k: int = 8,
) -> Dict:
    recommendation = fetch_recommendation_by_id(recommendation_id)

    if not recommendation:
        raise ValueError(f"Recommendation not found: {recommendation_id}")

    question = build_drilldown_question(
        recommendation=recommendation,
        action=action,
        custom_condition=custom_condition,
    )

    agent = CEOChatAgent()

    try:
        result = agent.answer(question, top_k=top_k)
    finally:
        agent.close()

    result["recommendation_id"] = recommendation_id
    result["drilldown_action"] = action
    result["drilldown_question"] = question

    return result


def main():
    parser = argparse.ArgumentParser(description="Drill down into a CEO recommendation.")
    parser.add_argument("--id", type=int, required=False, help="Recommendation ID")
    parser.add_argument("--action", type=str, default="Why this recommendation?")
    parser.add_argument("--condition", type=str, default="")
    parser.add_argument("--list", action="store_true", help="List available recommendations")
    args = parser.parse_args()

    if args.list:
        recs = fetch_recommendations()
        print("\nAvailable recommendations:")
        for rec in recs:
            print(
                f"{rec['id']} | {rec['priority']} | conf={rec['confidence_score']} | {rec['title']}"
            )
        return

    if not args.id:
        raise ValueError("Provide --id or use --list.")

    result = run_drilldown(
        recommendation_id=args.id,
        action=args.action,
        custom_condition=args.condition,
    )

    print(result["answer_markdown"])


if __name__ == "__main__":
    main()

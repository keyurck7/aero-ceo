import os
import sqlite3
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("SQLITE_DB_PATH", "data/aero_ceo.sqlite")


DEMO_QUESTIONS = [
    "Should Airbus collaborate with another European organization for sixth-generation fighter systems?",
    "What are the biggest risks if FCAS is delayed?",
    "What should Airbus do if budget is limited but it still wants to compete in uncrewed combat aircraft?",
    "Explain the recommendation about military space and secure communications with evidence.",
    "Which partner could help Airbus in Spain for SIRTAP and uncrewed systems?",
    "What is the lower-risk version of investing in future combat systems?",
    "What should Airbus prioritize next in military space and secure communications?",
    "How should Airbus respond if drone demand grows faster than future fighter procurement?",
]


DEMO_FLOW = [
    "Executive Overview",
    "Data Quality Audit",
    "Evidence Explorer",
    "Opportunity Monitor",
    "Risk Monitor",
    "CEO Recommendations",
    "Ask AERO-CEO",
    "Scenario Analyzer",
    "Evaluation & Guardrails",
]


def _count_table(conn: sqlite3.Connection, table_name: str) -> int:
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    except Exception:
        return 0


def get_system_status() -> Dict:
    status = {
        "database_exists": Path(DB_PATH).exists(),
        "db_path": DB_PATH,
        "faiss_exists": Path(os.getenv("FAISS_INDEX_PATH", "data/cache/aero_ceo.faiss")).exists(),
        "faiss_metadata_exists": Path(os.getenv("FAISS_METADATA_PATH", "data/cache/faiss_metadata.json")).exists(),
        "llm_enabled": os.getenv("LLM_GENERATION_ENABLED", "false").lower() == "true",
        "llm_model": os.getenv("LLM_MODEL_ID", "not configured"),
        "documents": 0,
        "sources": 0,
        "chunks": 0,
        "signals": 0,
        "recommendations": 0,
        "evidence_links": 0,
        "ceo_queries": 0,
    }

    if not status["database_exists"]:
        return status

    conn = sqlite3.connect(DB_PATH)
    try:
        status["documents"] = _count_table(conn, "documents")
        status["sources"] = _count_table(conn, "sources")
        status["chunks"] = _count_table(conn, "document_chunks")
        status["signals"] = _count_table(conn, "signals")
        status["recommendations"] = _count_table(conn, "recommendations")
        status["evidence_links"] = _count_table(conn, "recommendation_evidence")
        status["ceo_queries"] = _count_table(conn, "ceo_queries")
    finally:
        conn.close()

    return status


def readiness_label(status: Dict) -> str:
    if not status["database_exists"]:
        return "Database missing"

    checks = [
        status["documents"] >= 100,
        status["sources"] >= 3,
        status["chunks"] >= status["documents"],
        status["signals"] >= 50,
        status["recommendations"] >= 3,
        status["evidence_links"] >= status["recommendations"],
        status["faiss_exists"],
        status["faiss_metadata_exists"],
    ]

    passed = sum(1 for item in checks if item)

    if passed == len(checks):
        return "Demo ready"
    if passed >= 6:
        return "Mostly ready"
    if passed >= 4:
        return "Needs rebuild"
    return "Not ready"


def demo_script_short() -> List[str]:
    return [
        "Start with Executive Overview to show scope and corpus.",
        "Open Data Quality Audit to prove source reliability.",
        "Use Evidence Explorer to search FCAS or SIRTAP evidence.",
        "Show Opportunity and Risk monitors.",
        "Open CEO Recommendations and drill into one recommendation.",
        "Ask AERO-CEO a strategic question.",
        "Run a Scenario Analyzer what-if case.",
        "Finish with Evaluation & Guardrails.",
    ]

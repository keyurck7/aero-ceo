import html
import os
import sqlite3
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("SQLITE_DB_PATH", "data/aero_ceo.sqlite")
REPORT_PATH = "reports/aero_ceo_qna_report.html"


def get_conn():
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def safe(value):
    return html.escape(str(value or ""))


def markdown_block_to_html(markdown_text: str) -> str:
    """
    Lightweight renderer for demo reports.
    It preserves the structured CEO answer without needing extra dependencies.
    """
    text = safe(markdown_text)
    text = text.replace("\n", "<br>")
    text = text.replace("**", "")
    return text


def main():
    Path("reports").mkdir(parents=True, exist_ok=True)

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS ceo_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            intent TEXT,
            topic TEXT,
            business_area TEXT,
            region TEXT,
            answer_markdown TEXT NOT NULL,
            confidence_score REAL DEFAULT 0,
            evidence_count INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)

    rows = cur.execute("""
        SELECT id, question, intent, topic, business_area, region,
               answer_markdown, confidence_score, evidence_count, created_at
        FROM ceo_queries
        ORDER BY id DESC
        LIMIT 20
    """).fetchall()

    total_queries = cur.execute("SELECT COUNT(*) FROM ceo_queries").fetchone()[0]

    avg_conf = cur.execute("""
        SELECT ROUND(AVG(confidence_score), 3)
        FROM ceo_queries
    """).fetchone()[0] or 0

    total_evidence = cur.execute("""
        SELECT SUM(evidence_count)
        FROM ceo_queries
    """).fetchone()[0] or 0

    intent_rows = cur.execute("""
        SELECT intent, COUNT(*) AS n
        FROM ceo_queries
        GROUP BY intent
        ORDER BY n DESC
    """).fetchall()

    conn.close()

    intent_html = ""
    for row in intent_rows:
        intent_html += f"""
        <tr>
            <td>{safe(row["intent"])}</td>
            <td>{row["n"]}</td>
        </tr>
        """

    cards = ""

    if not rows:
        cards = """
        <div class="card">
            <h2>No CEO Q&A sessions found</h2>
            <p>Run <code>bash scripts/seed_ceo_questions.sh</code> or ask a question using <code>bash scripts/ask_ceo.sh "your question"</code>.</p>
        </div>
        """
    else:
        for row in rows:
            answer_html = markdown_block_to_html(row["answer_markdown"])

            cards += f"""
            <div class="card">
                <h2>Question #{row["id"]}</h2>
                <p class="question">{safe(row["question"])}</p>

                <div class="meta">
                    <span>Intent: {safe(row["intent"])}</span>
                    <span>Topic: {safe(row["topic"])}</span>
                    <span>Business Area: {safe(row["business_area"])}</span>
                    <span>Region: {safe(row["region"])}</span>
                    <span>Confidence: {safe(row["confidence_score"])}</span>
                    <span>Evidence Count: {safe(row["evidence_count"])}</span>
                    <span>Created: {safe(row["created_at"])}</span>
                </div>

                <div class="answer">
                    {answer_html}
                </div>
            </div>
            """

    report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>AERO-CEO Q&A Report</title>
        <style>
            body {{
                margin: 32px;
                font-family: Arial, sans-serif;
                background: #0f172a;
                color: #e5e7eb;
            }}
            h1, h2, h3 {{
                color: #f8fafc;
            }}
            .subtitle {{
                color: #cbd5e1;
                font-size: 18px;
                max-width: 1100px;
            }}
            .metrics {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 16px;
                margin: 24px 0;
            }}
            .metric {{
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 14px;
                padding: 18px;
            }}
            .metric .num {{
                font-size: 34px;
                font-weight: bold;
                color: #38bdf8;
            }}
            .card {{
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 14px;
                padding: 24px;
                margin: 24px 0;
            }}
            .question {{
                font-size: 20px;
                font-weight: bold;
                color: #bae6fd;
            }}
            .meta {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin: 16px 0;
            }}
            .meta span {{
                background: #0f172a;
                border: 1px solid #334155;
                padding: 8px 10px;
                border-radius: 8px;
                color: #cbd5e1;
            }}
            .answer {{
                background: #111827;
                border: 1px solid #374151;
                border-radius: 12px;
                padding: 18px;
                line-height: 1.55;
                white-space: normal;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                background: #111827;
            }}
            th, td {{
                border: 1px solid #374151;
                padding: 10px;
                text-align: left;
            }}
            th {{
                background: #1f2937;
                color: #f9fafb;
            }}
            code {{
                background: #111827;
                color: #7dd3fc;
                padding: 2px 5px;
                border-radius: 4px;
            }}
            a {{
                color: #7dd3fc;
            }}
        </style>
    </head>
    <body>
        <h1>AERO-CEO: CEO Strategic Q&A Report</h1>
        <p class="subtitle">
            This report shows how the CEO can interrogate AERO-CEO beyond static recommendations.
            Each answer is routed by intent/topic, grounded in retrieved evidence, and stored for audit.
        </p>

        <div class="metrics">
            <div class="metric">
                <div>Total CEO Questions</div>
                <div class="num">{total_queries}</div>
            </div>
            <div class="metric">
                <div>Average Confidence</div>
                <div class="num">{avg_conf}</div>
            </div>
            <div class="metric">
                <div>Total Evidence References</div>
                <div class="num">{total_evidence}</div>
            </div>
        </div>

        <div class="card">
            <h2>Intent Distribution</h2>
            <table>
                <tr>
                    <th>Intent</th>
                    <th>Questions</th>
                </tr>
                {intent_html}
            </table>
        </div>

        <div class="card">
            <h2>How this satisfies the CEO-agent requirement</h2>
            <p>
                AERO-CEO is not limited to pre-generated recommendations. The CEO can ask follow-up questions about
                partnership strategy, risks, scenarios, competitors, evidence, and constrained decisions. The system
                retrieves relevant evidence, connects it with extracted strategic signals, and produces an executive
                answer with options, risks, confidence, and sources.
            </p>
        </div>

        {cards}

        <p>Generated at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </body>
    </html>
    """

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"CEO Q&A report generated: {REPORT_PATH}")


if __name__ == "__main__":
    main()

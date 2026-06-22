import sqlite3


def ensure_ceo_chat_tables(conn: sqlite3.Connection) -> None:
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

    conn.commit()

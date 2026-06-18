import os
import sqlite3
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("SQLITE_DB_PATH", "data/aero_ceo.sqlite")


def get_connection():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_local_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        source_type TEXT NOT NULL,
        base_url TEXT,
        trust_score REAL DEFAULT 0.5,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id INTEGER,
        title TEXT,
        url TEXT UNIQUE,
        published_at TEXT,
        collected_at TEXT DEFAULT CURRENT_TIMESTAMP,
        raw_text TEXT,
        clean_text TEXT,
        language TEXT DEFAULT 'en',
        source_type TEXT,
        topic TEXT,
        content_hash TEXT,
        trust_score REAL DEFAULT 0.5,
        FOREIGN KEY(source_id) REFERENCES sources(id)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS document_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER,
        chunk_index INTEGER,
        chunk_text TEXT NOT NULL,
        token_count INTEGER,
        faiss_id INTEGER,
        FOREIGN KEY(document_id) REFERENCES documents(id)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER,
        signal_type TEXT NOT NULL,
        topic TEXT,
        title TEXT,
        description TEXT,
        entities TEXT,
        impact_score REAL DEFAULT 0,
        urgency_score REAL DEFAULT 0,
        confidence_score REAL DEFAULT 0,
        evidence_text TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(document_id) REFERENCES documents(id)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS recommendations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        recommendation TEXT NOT NULL,
        priority TEXT,
        expected_impact TEXT,
        risk_assessment TEXT,
        confidence_score REAL DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS recommendation_evidence (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recommendation_id INTEGER,
        document_id INTEGER,
        chunk_id INTEGER,
        evidence_strength REAL DEFAULT 0.5,
        FOREIGN KEY(recommendation_id) REFERENCES recommendations(id),
        FOREIGN KEY(document_id) REFERENCES documents(id),
        FOREIGN KEY(chunk_id) REFERENCES document_chunks(id)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS analysis_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_type TEXT,
        status TEXT,
        started_at TEXT DEFAULT CURRENT_TIMESTAMP,
        completed_at TEXT,
        notes TEXT
    );
    """)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_local_db()
    print(f"Local SQLite database initialized at: {DB_PATH}")

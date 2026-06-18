import argparse
import html
import sqlite3
import time
from datetime import datetime
from typing import Optional
from urllib.parse import quote_plus

import feedparser
from bs4 import BeautifulSoup

from app.db.local_db import get_connection, init_local_db
from app.processing.cleaner import clean_text, content_hash, infer_topic


AIRBUS_INTELLIGENCE_QUERIES = [
    "Airbus Defence and Space",
    "Airbus FCAS",
    "Airbus Future Combat Air System",
    "Airbus Eurofighter",
    "Airbus Typhoon fighter",
    "Airbus A400M",
    "Airbus A330 MRTT",
    "Airbus C295",
    "Airbus Eurodrone",
    "Airbus SIRTAP",
    "Airbus military satellite",
    "Airbus secure communications defence",
    "Airbus NATO defence",
    "Airbus European defence autonomy",
    "Airbus defence procurement",
    "Airbus Dassault FCAS",
    "Airbus fighter jet programme",
    "Airbus uncrewed combat aircraft",
    "Airbus crewed uncrewed teaming",
    "Airbus combat cloud",
    "Airbus BAE Systems fighter",
    "Airbus Boeing defence competition",
    "Airbus Lockheed Martin competition",
    "Airbus Leonardo defence",
    "Airbus supply chain aerospace defence",
]

TRUSTED_SOURCE_KEYWORDS = {
    "reuters": 0.90,
    "airbus": 0.95,
    "defense news": 0.84,
    "breaking defense": 0.84,
    "aviation week": 0.84,
    "flightglobal": 0.82,
    "janes": 0.86,
    "nato": 0.90,
    "defence industry europe": 0.78,
    "army recognition": 0.72,
    "defence blog": 0.70,
    "the defense post": 0.72,
    "aerotime": 0.70,
}


def google_news_rss_url(query: str) -> str:
    encoded = quote_plus(query)
    return f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"


def ensure_source(conn: sqlite3.Connection, name: str, source_type: str, base_url: str, trust_score: float) -> int:
    cur = conn.cursor()
    cur.execute("SELECT id FROM sources WHERE name = ?", (name,))
    row = cur.fetchone()

    if row:
        return int(row["id"])

    cur.execute(
        """
        INSERT INTO sources (name, source_type, base_url, trust_score)
        VALUES (?, ?, ?, ?)
        """,
        (name, source_type, base_url, trust_score),
    )
    conn.commit()
    return int(cur.lastrowid)


def estimate_trust(source_name: str) -> float:
    lower = source_name.lower()
    for keyword, score in TRUSTED_SOURCE_KEYWORDS.items():
        if keyword in lower:
            return score
    return 0.65


def insert_rss_item(
    conn: sqlite3.Connection,
    source_id: int,
    title: str,
    url: str,
    published_at: Optional[str],
    summary: str,
    query: str,
    source_name: str,
    trust_score: float,
) -> bool:
    title = clean_text(html.unescape(title))
    summary = clean_text(summary)

    intelligence_text = clean_text(
        f"""
        Title: {title}

        Source: {source_name}
        Search context: {query}
        Published: {published_at or "unknown"}

        Strategic intelligence note:
        This public news/RSS item mentions Airbus in the context of {query}.
        The item should be considered as an external market signal and should be
        verified against the linked source before final executive action.

        Available summary:
        {summary}
        """
    )

    if len(title) < 5:
        return False

    doc_hash = content_hash(f"{title} {url} {summary}")
    topic = infer_topic(f"{title} {summary} {query}")

    cur = conn.cursor()

    cur.execute("SELECT id FROM documents WHERE url = ? OR content_hash = ?", (url, doc_hash))
    if cur.fetchone():
        return False

    try:
        cur.execute(
            """
            INSERT INTO documents (
                source_id, title, url, published_at, raw_text, clean_text,
                source_type, topic, content_hash, trust_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_id,
                title,
                url,
                published_at,
                intelligence_text,
                intelligence_text,
                "news",
                topic,
                doc_hash,
                trust_score,
            ),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def collect_rss_intelligence(per_query: int = 20, target: int = 150) -> int:
    init_local_db()
    conn = get_connection()

    inserted = 0

    for query in AIRBUS_INTELLIGENCE_QUERIES:
        if inserted >= target:
            break

        feed_url = google_news_rss_url(query)
        feed = feedparser.parse(feed_url)

        print(f"\nQuery: {query} | RSS items: {len(feed.entries)}")

        for entry in feed.entries[:per_query]:
            if inserted >= target:
                break

            title = entry.get("title", "").strip()
            link = entry.get("link", "").strip()
            published_at = entry.get("published", None)

            raw_summary = entry.get("summary", "")
            summary = BeautifulSoup(raw_summary, "html.parser").get_text(" ", strip=True)

            source_name = "Google News RSS"
            try:
                if hasattr(entry, "source") and entry.source:
                    source_name = entry.source.get("title", source_name)
            except Exception:
                pass

            trust_score = estimate_trust(source_name)

            source_id = ensure_source(
                conn=conn,
                name=source_name,
                source_type="news",
                base_url=link,
                trust_score=trust_score,
            )

            ok = insert_rss_item(
                conn=conn,
                source_id=source_id,
                title=title,
                url=link,
                published_at=published_at,
                summary=summary,
                query=query,
                source_name=source_name,
                trust_score=trust_score,
            )

            if ok:
                inserted += 1
                print(f"[rss] inserted {inserted}: {source_name} | {title[:90]}")

            time.sleep(0.05)

    conn.close()
    return inserted


def print_summary():
    conn = get_connection()
    cur = conn.cursor()

    print("\n=== AERO-CEO Corpus Summary ===")
    print("Documents:", cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
    print("Sources:", cur.execute("SELECT COUNT(*) FROM sources").fetchone()[0])

    print("\nSource types:")
    for row in cur.execute("SELECT source_type, COUNT(*) FROM documents GROUP BY source_type"):
        print(row[0], row[1])

    print("\nTop topics:")
    for row in cur.execute("SELECT topic, COUNT(*) FROM documents GROUP BY topic ORDER BY COUNT(*) DESC LIMIT 12"):
        print(row[0], row[1])

    print("\nTop news sources:")
    for row in cur.execute("""
        SELECT s.name, COUNT(*) AS n
        FROM documents d
        JOIN sources s ON d.source_id = s.id
        GROUP BY s.name
        ORDER BY n DESC
        LIMIT 12
    """):
        print(row["name"], row["n"])

    conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-query", type=int, default=20)
    parser.add_argument("--target", type=int, default=180)
    args = parser.parse_args()

    inserted = collect_rss_intelligence(per_query=args.per_query, target=args.target)
    print(f"\nInserted RSS intelligence documents: {inserted}")
    print_summary()


if __name__ == "__main__":
    main()

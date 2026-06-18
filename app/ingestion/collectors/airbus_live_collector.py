import argparse
import html
import sqlite3
import time
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import quote_plus, urljoin, urlparse

import feedparser
import requests
import trafilatura
from bs4 import BeautifulSoup

from app.db.local_db import get_connection, init_local_db
from app.ingestion.source_registry import SOURCES
from app.processing.cleaner import clean_text, content_hash, infer_topic


HEADERS = {
    "User-Agent": (
        "AERO-CEO Strategic Intelligence Research Bot "
        "(academic project; contact: repository owner)"
    )
}

OFFICIAL_SEED_URLS = [
    "https://www.airbus.com/en/newsroom",
    "https://www.airbus.com/en/products-services/defence",
    "https://www.airbus.com/en/products-services/defence/military-aircraft",
    "https://www.airbus.com/en/products-services/defence/future-combat-air-system-fcas",
    "https://www.airbus.com/en/products-services/defence/uav",
    "https://www.airbus.com/en/products-services/defence/space",
]

NEWS_QUERIES = [
    "Airbus Defence and Space FCAS",
    "Airbus Eurofighter Typhoon",
    "Airbus A400M military aircraft",
    "Airbus A330 MRTT tanker",
    "Airbus Eurodrone SIRTAP uncrewed aircraft",
    "Airbus military satellite secure communications",
    "Airbus European defence autonomy",
    "Airbus Dassault FCAS fighter jet",
    "Airbus NATO defence procurement",
    "Airbus defence supply chain risk",
]


def google_news_rss_url(query: str) -> str:
    encoded = quote_plus(query)
    return f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"


def fetch_url(url: str, timeout: int = 20) -> Optional[str]:
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        if response.status_code >= 400:
            return None
        return response.text
    except Exception:
        return None


def extract_text_from_url(url: str) -> Dict[str, str]:
    downloaded = None

    try:
        downloaded = trafilatura.fetch_url(url)
    except Exception:
        downloaded = None

    if downloaded:
        extracted = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
        if extracted and len(extracted) > 300:
            return {
                "text": clean_text(extracted),
                "title": "",
            }

    html_text = fetch_url(url)

    if not html_text:
        return {
            "text": "",
            "title": "",
        }

    soup = BeautifulSoup(html_text, "html.parser")

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = clean_text(" ".join(paragraphs))

    return {
        "text": text,
        "title": title,
    }


def discover_airbus_links(seed_url: str, max_links: int = 20) -> List[str]:
    html_text = fetch_url(seed_url)
    if not html_text:
        return [seed_url]

    soup = BeautifulSoup(html_text, "html.parser")
    links = [seed_url]
    seen = {seed_url}

    for a in soup.find_all("a", href=True):
        href = a["href"]
        absolute = urljoin(seed_url, href)
        parsed = urlparse(absolute)

        if "airbus.com" not in parsed.netloc:
            continue

        lowered = absolute.lower()
        useful = any(
            token in lowered
            for token in [
                "defence",
                "newsroom",
                "fcas",
                "eurofighter",
                "a400m",
                "mrtt",
                "space",
                "military",
                "uav",
                "drone",
            ]
        )

        if useful and absolute not in seen:
            seen.add(absolute)
            links.append(absolute)

        if len(links) >= max_links:
            break

    return links


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


def insert_document(
    conn: sqlite3.Connection,
    source_id: int,
    title: str,
    url: str,
    published_at: Optional[str],
    raw_text: str,
    source_type: str,
    trust_score: float,
) -> bool:
    cleaned = clean_text(raw_text)

    if len(cleaned) < 250:
        return False

    doc_hash = content_hash(cleaned)
    topic = infer_topic(f"{title} {cleaned}")

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
                raw_text,
                cleaned,
                source_type,
                topic,
                doc_hash,
                trust_score,
            ),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def collect_official_airbus(conn: sqlite3.Connection, limit: int = 40) -> int:
    source_id = ensure_source(
        conn,
        name="Airbus Official Web",
        source_type="official",
        base_url="https://www.airbus.com",
        trust_score=0.95,
    )

    inserted = 0
    urls = []

    for seed in OFFICIAL_SEED_URLS:
        urls.extend(discover_airbus_links(seed, max_links=20))

    unique_urls = list(dict.fromkeys(urls))

    for url in unique_urls:
        if inserted >= limit:
            break

        extracted = extract_text_from_url(url)
        text = extracted["text"]
        title = extracted["title"] or "Airbus official page"

        ok = insert_document(
            conn=conn,
            source_id=source_id,
            title=title,
            url=url,
            published_at=None,
            raw_text=text,
            source_type="official",
            trust_score=0.95,
        )

        if ok:
            inserted += 1
            print(f"[official] inserted {inserted}: {title[:80]}")

        time.sleep(0.5)

    return inserted


def collect_google_news(conn: sqlite3.Connection, per_query: int = 20, total_limit: int = 160) -> int:
    inserted = 0

    for query in NEWS_QUERIES:
        if inserted >= total_limit:
            break

        feed_url = google_news_rss_url(query)
        feed = feedparser.parse(feed_url)

        for entry in feed.entries[:per_query]:
            if inserted >= total_limit:
                break

            title = html.unescape(entry.get("title", "")).strip()
            link = entry.get("link", "").strip()
            published_at = entry.get("published", None)
            summary = BeautifulSoup(entry.get("summary", ""), "html.parser").get_text(" ", strip=True)

            source_name = "Google News RSS"
            if hasattr(entry, "source") and entry.source:
                source_name = entry.source.get("title", source_name)

            source_type = "news"
            trust_score = 0.72

            lower_source = source_name.lower()
            if any(x in lower_source for x in ["reuters", "airbus", "nato", "defense news", "breaking defense"]):
                trust_score = 0.88

            source_id = ensure_source(
                conn,
                name=source_name,
                source_type=source_type,
                base_url=link,
                trust_score=trust_score,
            )

            extracted = extract_text_from_url(link)
            full_text = extracted["text"]

            if len(full_text) < 300:
                full_text = f"{title}. {summary}. Search query: {query}"

            ok = insert_document(
                conn=conn,
                source_id=source_id,
                title=title,
                url=link,
                published_at=published_at,
                raw_text=full_text,
                source_type=source_type,
                trust_score=trust_score,
            )

            if ok:
                inserted += 1
                print(f"[news] inserted {inserted}: {source_name} | {title[:80]}")

            time.sleep(0.3)

    return inserted


def register_configured_sources(conn: sqlite3.Connection) -> None:
    for source in SOURCES:
        ensure_source(
            conn,
            name=source["name"],
            source_type=source["source_type"],
            base_url=source["base_url"],
            trust_score=source["trust_score"],
        )


def print_summary(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    print("\n=== AERO-CEO Corpus Summary ===")

    cur.execute("SELECT COUNT(*) AS n FROM documents")
    print(f"Documents: {cur.fetchone()['n']}")

    cur.execute("SELECT COUNT(*) AS n FROM sources")
    print(f"Sources: {cur.fetchone()['n']}")

    print("\nDocuments by source type:")
    for row in cur.execute("SELECT source_type, COUNT(*) AS n FROM documents GROUP BY source_type ORDER BY n DESC"):
        print(f"  {row['source_type']}: {row['n']}")

    print("\nTop topics:")
    for row in cur.execute("SELECT topic, COUNT(*) AS n FROM documents GROUP BY topic ORDER BY n DESC LIMIT 10"):
        print(f"  {row['topic']}: {row['n']}")


def main():
    parser = argparse.ArgumentParser(description="Collect live Airbus strategic intelligence documents.")
    parser.add_argument("--official-limit", type=int, default=40)
    parser.add_argument("--news-limit", type=int, default=160)
    parser.add_argument("--per-query", type=int, default=20)
    args = parser.parse_args()

    init_local_db()
    conn = get_connection()

    register_configured_sources(conn)

    print("Starting Airbus official collection...")
    official_count = collect_official_airbus(conn, limit=args.official_limit)

    print("\nStarting live news collection...")
    news_count = collect_google_news(conn, per_query=args.per_query, total_limit=args.news_limit)

    print(f"\nInserted official documents: {official_count}")
    print(f"Inserted news documents: {news_count}")

    print_summary(conn)
    conn.close()


if __name__ == "__main__":
    main()

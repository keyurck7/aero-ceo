"""Import a v1 corpus (``data/aero_ceo.sqlite``) into the v2 store.

Nothing collected so far is thrown away, but the v1 artefacts are repaired on the way in:

* **Template stripping (finding B1).** v1 wrapped every RSS item in a template that
  contained the search query ("…mentions Airbus in the context of {query}…"), and
  its Google-News fallback appended "Search query: {query}". Both are detected,
  the publisher's title + summary are kept, and the query moves to
  ``ingestion_context``.
* Dates are normalised to UTC (finding B4); unparseable dates become NULL, not 0.55.
* Duplicates across redirect URLs and syndicated copies collapse (finding B9).
* Topics are re-inferred (multi-label, word boundaries); chunks get spans.

The v1 database is opened read-only and never modified.
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aero.domain import SourceIn
from aero.ingestion.collectors import SourcePlan
from aero.ingestion.pipeline import MIN_WORDS, Ingestor, RawItem
from aero.log import get_logger

log = get_logger(__name__)

# rss_intelligence_collector.insert_rss_item(), after v1 clean_text() collapsed whitespace.
_TEMPLATE_A = re.compile(
    r"^Title:\s*(?P<title>.*?)\s+Source:\s*(?P<source>.*?)\s+Search context:\s*(?P<query>.*?)\s+"
    r"Published:\s*(?P<published>.*?)\s+Strategic intelligence note:.*?"
    r"Available summary:\s*(?P<summary>.*)$",
    re.DOTALL,
)
# airbus_live_collector.collect_google_news() fallback: f"{title}. {summary}. Search query: {query}"
_TEMPLATE_B = re.compile(r"^(?P<head>.*)\.\s*Search query:\s*(?P<query>[^.]*?)\s*$", re.DOTALL)

_V1_TYPES = {"official", "news", "policy", "competitor", "community"}


@dataclass
class V1Row:
    id: int
    title: str
    url: str
    published_at: str | None
    text: str
    source_name: str
    source_type: str
    trust_score: float


def parse_v1_text(title: str, text: str) -> tuple[str, str, dict[str, Any]]:
    """Return ``(body, content_level, context)`` with any v1 template removed."""
    m = _TEMPLATE_A.match(text)
    if m:
        return m.group("summary").strip(), "headline_summary", {"v1_template": "rss_intelligence_note",
                                                                 "query": m.group("query").strip()}
    m = _TEMPLATE_B.match(text)
    if m:
        head = m.group("head").strip()
        if head.startswith(title):
            head = head[len(title):].lstrip(". ").strip()
        return head, "headline_summary", {"v1_template": "google_news_fallback", "query": m.group("query").strip()}
    return text, "full_text", {}


def read_v1(path: str | Path) -> list[V1Row]:
    uri = f"file:{Path(path).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT d.id, d.title, d.url, d.published_at, COALESCE(NULLIF(d.clean_text, ''), d.raw_text) AS text,
                      COALESCE(s.name, 'Unknown publisher') AS source_name,
                      COALESCE(s.source_type, d.source_type, 'unknown') AS source_type,
                      COALESCE(d.trust_score, s.trust_score, 0.5) AS trust_score
               FROM documents d LEFT JOIN sources s ON s.id = d.source_id
               WHERE COALESCE(d.clean_text, d.raw_text) IS NOT NULL
               ORDER BY d.id"""
        ).fetchall()
    finally:
        conn.close()
    return [V1Row(**dict(r)) for r in rows]


def migrate_v1(path: str | Path, ingestor: Ingestor, plan: SourcePlan, run_id: int | None = None) -> Counter[str]:
    stats: Counter[str] = Counter()
    for row in read_v1(path):
        body, level, context = parse_v1_text(row.title or "", row.text or "")
        if context:
            stats[f"stripped_{context['v1_template']}"] += 1
        if level == "full_text" and row.source_type == "official" and not row.published_at:
            level = "page"
        # v1 accepted anything >= 250 characters as an "article"; short texts are snippets, not full text.
        if level != "headline_summary" and len(f"{row.title} {body}".split()) < MIN_WORDS[level]:
            level = "headline_summary"
            stats["downgraded_short_text"] += 1
        stype = row.source_type if row.source_type in _V1_TYPES else "unknown"
        known = next((s for s in plan.sources if s.name.lower() == row.source_name.lower()), None)
        if known:
            source = known
        elif stype == "news":
            source = plan.news_source(row.source_name)
        else:
            source = SourceIn(name=row.source_name, source_type=stype,  # type: ignore[arg-type]
                              credibility_prior=min(max(float(row.trust_score), 0.0), 1.0),
                              publisher_group=re.sub(r"[^a-z0-9]+", "_", row.source_name.lower()).strip("_"))
        result = ingestor.ingest(
            RawItem(source=source, title=row.title or "", url=row.url or f"v1://document/{row.id}", body=body,
                    content_level=level, published_at=row.published_at,  # type: ignore[arg-type]
                    ingestion_context={"collector": "v1_import", "v1_document_id": row.id, **context}),
            run_id=run_id,
        )
        stats[result.outcome if result.outcome != "duplicate" else f"duplicate_{result.detail}"] += 1
    return stats

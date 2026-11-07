from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from aero.domain import SearchFilters, SourceIn
from aero.ingestion.collectors import collect_news, collect_official, discover_links, load_source_plan
from aero.ingestion.extract import extract_article, parse_feed
from aero.ingestion.http import FetchResult
from aero.ingestion.migrate_v1 import migrate_v1, parse_v1_text
from aero.ingestion.pipeline import Ingestor, RawItem, compose_text
from aero.retrieval import HybridRetriever
from aero.storage import open_store
from tests.conftest import FIXTURES, ROOT, V1_RSS_TEMPLATE  # noqa: F401


class FakeFetcher:
    """Maps URL -> (final_url, body). Unknown URLs return None (like a 404)."""

    def __init__(self, pages: dict[str, tuple[str, str]]):
        self.pages = pages
        self.calls: list[str] = []

    def get(self, url):
        self.calls.append(url)
        if url not in self.pages:
            return None
        final, body = self.pages[url]
        return FetchResult(url, final, 200, "text/html", body)


@pytest.fixture
def plan():
    return load_source_plan(ROOT / "seeds" / "sources.yaml")


@pytest.fixture
def ingestor(settings, embedder, taxonomy):
    store = open_store(settings, migrate=True)
    yield Ingestor(store, embedder, taxonomy, settings)
    store.close()


# ----------------------------------------------------------------- feeds & extraction
def test_parse_google_news_rss_is_xxe_safe():
    items = parse_feed((FIXTURES / "google_news.xml").read_bytes())
    assert len(items) == 2  # the empty item is dropped
    first, second = items
    assert first.title == "Dassault and Airbus clash over FCAS leadership"  # " - Reuters" suffix removed
    assert first.publisher == "Reuters" and first.publisher_url == "https://www.reuters.com"
    assert first.published_at == datetime(2025, 10, 14, 8, 30, tzinfo=UTC)
    assert "<" not in first.summary
    assert "root:" not in second.title  # external entity not resolved


def test_parse_atom():
    (entry,) = parse_feed((FIXTURES / "atom.xml").read_text())
    assert entry.link == "https://eda.europa.eu/news/drone-study"
    assert entry.publisher == "European Defence Agency"
    assert entry.published_at == datetime(2025, 9, 1, 7, 0, tzinfo=UTC)
    assert entry.summary == "The agency launched a study on crewed-uncrewed teaming."


def test_extract_article_metadata_and_boilerplate():
    art = extract_article((FIXTURES / "article.html").read_text(), "https://www.airbus.com/x")
    assert art.title == "SIRTAP tactical UAS completes first flight"
    assert art.published_at == datetime(2025, 10, 20, 7, 0, tzinfo=UTC)  # from JSON-LD, v1 stored None
    assert art.canonical_link.endswith("/sirtap-first-flight")
    assert "Rota test range" in art.text and "Cookie Policy" not in art.text
    assert "\n\n" in art.text  # paragraphs preserved


def test_discover_links_stays_on_allowed_hosts():
    html = (FIXTURES / "article.html").read_text()
    links = discover_links(html, "https://www.airbus.com/en/newsroom", ["airbus.com"], ["newsroom", "defence"], 10)
    assert "https://www.airbus.com/en/products-services/defence/uav" in links
    assert not any("twitter" in u or u.endswith(".pdf") or "careers" in u for u in links)


# ----------------------------------------------------------------- pipeline (finding B1)
def test_compose_text_does_not_duplicate_title():
    assert compose_text("Title here", "Title here and more") == "Title here and more"
    assert compose_text("Title", "Body") == "Title\n\nBody"


def test_query_never_enters_document_text(ingestor):
    src = SourceIn(name="Reuters", source_type="news", credibility_prior=0.9, publisher_group="reuters")
    res = ingestor.ingest(RawItem(
        source=src, title="Dassault and Airbus clash over FCAS leadership",
        url="https://reuters.com/a", body="Workshare dispute continues.", content_level="headline_summary",
        published_at="Tue, 14 Oct 2025 08:30:00 GMT",
        ingestion_context={"collector": "news_feed", "query": "Airbus Dassault FCAS"},
    ))
    doc = ingestor.store.get_document(res.document_id)
    assert res.outcome == "inserted"
    assert "Search" not in doc.clean_text and "query" not in doc.clean_text.lower()
    assert doc.ingestion_context["query"] == "Airbus Dassault FCAS"
    assert doc.published_at == datetime(2025, 10, 14, 8, 30, tzinfo=UTC)


def test_short_and_duplicate_items(ingestor):
    src = SourceIn(name="X", source_type="news")
    short = ingestor.ingest(
        RawItem(source=src, title="Hi", url="https://x/1", body="too short", content_level="full_text")
    )
    assert short.outcome == "too_short"
    body = " ".join(["FCAS workshare negotiation"] * 40)
    first = ingestor.ingest(RawItem(source=src, title="T", url="https://x/2?utm_source=a", body=body))
    again = ingestor.ingest(RawItem(source=src, title="T", url="https://www.x/2", body=body))
    assert first.outcome == "inserted" and again.outcome == "duplicate"
    assert ingestor.counts["inserted"] == 1 and ingestor.counts["too_short"] == 1


def test_collect_official_then_ingest_and_search(ingestor, plan):
    html = (FIXTURES / "article.html").read_text()
    seed = plan.official_seed_urls[0]
    fetcher = FakeFetcher({
        seed: (seed, html),
        "https://www.airbus.com/en/products-services/defence/uav": ("https://www.airbus.com/en/x", html),
    })
    items = list(collect_official(fetcher, plan, limit=5))
    assert items and items[0].content_level == "full_text" and items[0].source.source_type == "official"
    results = [ingestor.ingest(i) for i in items]
    assert results[0].outcome == "inserted"
    assert all(r.outcome == "duplicate" for r in results[1:])  # same page through two URLs
    doc = ingestor.store.get_document(results[0].document_id)
    assert "ignore_instructions" in doc.injection_flags
    assert doc.url == "https://www.airbus.com/en/newsroom/press-releases/sirtap-first-flight"  # canonical link
    assert ingestor.embed_pending() > 0 and ingestor.embed_pending() == 0  # incremental
    res = HybridRetriever(ingestor.store, ingestor.embedder).search(
        "SIRTAP maiden flight Rota", top_k=2, filters=SearchFilters(topics=["Uncrewed Combat Aircraft"])
    )
    assert res.hits and res.hits[0].document_id == doc.id


def test_collect_news_respects_licence_policy(plan):
    feed = (FIXTURES / "google_news.xml").read_text()
    article = (FIXTURES / "article.html").read_text()
    q = plan.news_queries[0]
    from aero.ingestion.collectors import feed_url

    pages = {
        feed_url(plan, q): (feed_url(plan, q), feed),
        "https://news.google.com/rss/articles/CBMiBBB?oc=5": ("https://www.defensenews.com/sirtap", article),
    }
    research = list(collect_news(FakeFetcher(pages), plan, queries=[q]))
    assert [i.content_level for i in research] == ["headline_summary", "full_text"]
    assert research[0].ingestion_context["unresolved_aggregator_link"] is True
    assert research[1].url == "https://www.defensenews.com/sirtap"  # resolved past the aggregator
    assert research[1].source.name == "Defense News" and research[1].source.credibility_prior > 0.8
    assert all(i.ingestion_context["query"] == q for i in research)

    plan.full_text_policy = "respect_licence"
    strict = list(collect_news(FakeFetcher(pages), plan, queries=[q]))
    assert [i.content_level for i in strict] == ["headline_summary", "headline_summary"]


# ----------------------------------------------------------------- v1 migration
def test_v1_template_is_still_what_v1_writes():
    """Guards the fixture: if v1's template changes, this test must be updated with it."""
    src = (ROOT / "app/ingestion/collectors/rss_intelligence_collector.py").read_text()
    assert "This public news/RSS item mentions Airbus in the context of {query}." in src
    assert 'full_text = f"{title}. {summary}. Search query: {query}"' in (
        ROOT / "app/ingestion/collectors/airbus_live_collector.py").read_text()


def test_parse_v1_text_strips_both_templates():
    body, level, ctx = parse_v1_text("T", "Title: T Source: R Search context: Airbus FCAS Published: x "
                                          "Strategic intelligence note: blah. Available summary: The real summary.")
    assert (body, level, ctx["query"]) == ("The real summary.", "headline_summary", "Airbus FCAS")
    body, level, ctx = parse_v1_text("SIRTAP drone flies", "SIRTAP drone flies. Flew in Spain. Search query: SIRTAP")
    assert body == "Flew in Spain" and ctx["v1_template"] == "google_news_fallback"
    assert parse_v1_text("T", "A normal article.")[1] == "full_text"


def test_migrate_v1_end_to_end(v1_db, ingestor, plan):
    before = Path(v1_db).read_bytes()
    stats = migrate_v1(v1_db, ingestor, plan)
    assert Path(v1_db).read_bytes() == before  # read-only
    assert stats["inserted"] == 3 and stats["duplicate_content_hash"] == 1
    assert stats["stripped_rss_intelligence_note"] == 2 and stats["stripped_google_news_fallback"] == 1
    store = ingestor.store
    texts = [store.get_document(i).clean_text for i in store.list_document_ids()]
    assert not any("Search query" in t or "Search context" in t or "Strategic intelligence note" in t for t in texts)
    fcas = next(store.get_document(i) for i in store.list_document_ids()
                if "Dassault" in store.get_document(i).title)
    assert fcas.ingestion_context["query"] == "Airbus Dassault FCAS"
    assert fcas.published_at == datetime(2025, 10, 14, 8, 30, tzinfo=UTC)
    assert fcas.content_level == "headline_summary"
    space = next(store.get_document(i) for i in store.list_document_ids() if "space" in store.get_document(i).url)
    assert space.content_level == "page" and "Military Space" in space.topics

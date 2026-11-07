"""Collectors: turn the collection plan in ``seeds/sources.yaml`` into ``RawItem`` streams.

Ports of v1's ``airbus_live_collector.py`` and ``rss_intelligence_collector.py``, with:

* no search query or template text inside the document (finding B1);
* aggregator links (Google News) resolved to the publisher URL when possible,
  so dedup works across feeds (finding B9);
* a full-text policy per source licence (finding P3), see ``FullTextPolicy``;
* publication dates from page metadata, not ``None``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote_plus, urljoin, urlsplit

import yaml
from bs4 import BeautifulSoup

from aero.domain import SourceIn
from aero.ingestion.extract import FeedItem, extract_article, parse_feed
from aero.ingestion.http import PoliteFetcher
from aero.ingestion.pipeline import RawItem
from aero.log import get_logger
from aero.text import is_aggregator_url, url_host

log = get_logger(__name__)

# research_mode     : store full article text for non-commercial research (v1 behaviour).
# respect_licence   : full text only where the source's licence allows it; otherwise headline + snippet.
FullTextPolicy = Literal["research_mode", "respect_licence"]
FULL_TEXT_LICENCES = {"full_text_permitted", "licensed", "own_content"}


@dataclass
class SourcePlan:
    version: str
    sources: list[SourceIn]
    publisher_priors: dict[str, float]
    default_news_prior: float
    official_seed_urls: list[str]
    official_allowed_hosts: list[str]
    official_path_tokens: list[str]
    news_queries: list[str]
    news_feed_template: str
    full_text_policy: FullTextPolicy = "research_mode"
    extra: dict[str, Any] = field(default_factory=dict)

    def prior_for(self, publisher: str) -> float:
        lower = publisher.lower()
        best = [v for k, v in self.publisher_priors.items() if k.lower() in lower]
        return max(best) if best else self.default_news_prior

    def news_source(self, publisher: str) -> SourceIn:
        for s in self.sources:
            if s.name.lower() == publisher.lower():
                return s
        group = re.sub(r"[^a-z0-9]+", "_", publisher.lower()).strip("_") or "unknown"
        return SourceIn(name=publisher, source_type="news", credibility_prior=self.prior_for(publisher),
                        publisher_group=group, licence="unknown")

    def may_store_full_text(self, source: SourceIn) -> bool:
        return self.full_text_policy == "research_mode" or source.licence in FULL_TEXT_LICENCES


def load_source_plan(path: str | Path) -> SourcePlan:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    col = data.get("collection", {})
    return SourcePlan(
        version=str(data.get("version", "unversioned")),
        sources=[SourceIn(**s) for s in data.get("sources", [])],
        publisher_priors={str(k): float(v) for k, v in data.get("publisher_priors", {}).items()},
        default_news_prior=float(data.get("default_news_prior", 0.65)),
        official_seed_urls=list(col.get("official_seed_urls", [])),
        official_allowed_hosts=list(col.get("official_allowed_hosts", [])),
        official_path_tokens=[t.lower() for t in col.get("official_path_tokens", [])],
        news_queries=list(col.get("news_queries", [])),
        news_feed_template=col.get("news_feed_template", ""),
        full_text_policy=data.get("full_text_policy", "research_mode"),
    )


# --------------------------------------------------------------------------- official pages


def discover_links(html: str, base_url: str, allowed_hosts: list[str], path_tokens: list[str], limit: int) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    out: dict[str, None] = {base_url: None}
    for a in soup.find_all("a", href=True):
        absolute = urljoin(base_url, str(a["href"])).split("#")[0]
        host = url_host(absolute)
        if not any(host == h or host.endswith("." + h) for h in allowed_hosts):
            continue
        path = urlsplit(absolute).path.lower()
        if any(path.endswith(ext) for ext in (".pdf", ".jpg", ".png", ".zip", ".mp4")):
            continue
        if path_tokens and not any(tok in path for tok in path_tokens):
            continue
        out.setdefault(absolute, None)
        if len(out) >= limit:
            break
    return list(out)


def collect_official(fetcher: PoliteFetcher, plan: SourcePlan, limit: int = 40,
                     links_per_seed: int = 20) -> Iterator[RawItem]:
    official = next((s for s in plan.sources if s.source_type == "official"), None) or SourceIn(
        name="Airbus Official Web", source_type="official", credibility_prior=0.95, publisher_group="airbus"
    )
    urls: dict[str, None] = {}
    for seed in plan.official_seed_urls:
        page = fetcher.get(seed)
        if page is None:
            continue
        for link in discover_links(page.text, page.final_url, plan.official_allowed_hosts, plan.official_path_tokens,
                                   links_per_seed):
            urls.setdefault(link, None)
    yielded = 0
    for url in urls:
        if yielded >= limit:
            return
        page = fetcher.get(url)
        if page is None:
            continue
        art = extract_article(page.text, page.final_url)
        yield RawItem(
            source=official,
            title=art.title or "Airbus official page",
            url=art.canonical_link or page.final_url,
            body=art.text,
            content_level="full_text" if art.published_at else "page",
            published_at=art.published_at,
            ingestion_context={"collector": "official_crawl", "seed_plan": plan.version, "extractor": art.extractor,
                               "fetched_url": url},
        )
        yielded += 1


# --------------------------------------------------------------------------- news feeds


def feed_url(plan: SourcePlan, query: str) -> str:
    return plan.news_feed_template.format(query=quote_plus(query))


def _news_item(fetcher: PoliteFetcher, plan: SourcePlan, item: FeedItem, query: str, fetch_articles: bool) -> RawItem:
    publisher = item.publisher or url_host(item.link) or "Unknown publisher"
    source = plan.news_source(publisher)
    context: dict[str, Any] = {"collector": "news_feed", "query": query, "feed_link": item.link,
                               "publisher_url": item.publisher_url, "seed_plan": plan.version}
    url, body, level = item.link, item.summary, "headline_summary"
    if fetch_articles and plan.may_store_full_text(source):
        page = fetcher.get(item.link)  # follows redirects; aggregator links may resolve to the publisher
        if page is not None and not is_aggregator_url(page.final_url):
            url = page.final_url
            art = extract_article(page.text, page.final_url)
            if len(art.text.split()) >= 120:
                body, level = art.text, "full_text"
                context["extractor"] = art.extractor
    if is_aggregator_url(url):
        context["unresolved_aggregator_link"] = True
    return RawItem(source=source, title=item.title, url=url, body=body, content_level=level,  # type: ignore[arg-type]
                   published_at=item.published_at, ingestion_context=context)


def collect_news(fetcher: PoliteFetcher, plan: SourcePlan, per_query: int = 20, limit: int = 200,
                 fetch_articles: bool = True, queries: list[str] | None = None) -> Iterator[RawItem]:
    yielded = 0
    for query in queries or plan.news_queries:
        page = fetcher.get(feed_url(plan, query))
        if page is None:
            continue
        for item in parse_feed(page.text)[:per_query]:
            if yielded >= limit:
                return
            yield _news_item(fetcher, plan, item, query, fetch_articles)
            yielded += 1

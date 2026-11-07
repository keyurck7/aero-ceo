"""Feed parsing and article extraction.

Feeds: RSS 2.0 and Atom are parsed with lxml, with entity resolution and network
access disabled (XXE-safe). Removes the ``feedparser`` dependency.

Articles: ``trafilatura`` when installed (best boilerplate removal); otherwise a
BeautifulSoup fallback. Either way, the publication date comes from page metadata
(``article:published_time``, JSON-LD ``datePublished``, ``<time datetime>``). v1
stored ``published_at=None`` for every official page.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime

from bs4 import BeautifulSoup
from lxml import etree

from aero.dates import parse_published_at
from aero.text import clean_text

_ATOM = "{http://www.w3.org/2005/Atom}"


@dataclass(frozen=True)
class FeedItem:
    title: str
    link: str
    published_at: datetime | None
    summary: str
    publisher: str | None = None
    publisher_url: str | None = None


@dataclass(frozen=True)
class Article:
    title: str
    text: str
    published_at: datetime | None
    canonical_link: str | None = None
    extractor: str = "bs4"
    meta: dict[str, str] = field(default_factory=dict)


def _html_to_text(fragment: str) -> str:
    if not fragment:
        return ""
    if "<" not in fragment:
        return clean_text(fragment)
    text = BeautifulSoup(fragment, "html.parser").get_text(" ", strip=True)
    return clean_text(re.sub(r"\s+([.,;:!?)])", r"\1", text))


def _strip_publisher_suffix(title: str, publisher: str | None) -> str:
    """Google News titles end with " - Publisher"; keep the headline only."""
    if publisher and title.endswith(f" - {publisher}"):
        return title[: -len(publisher) - 3].strip()
    return title


def parse_feed(xml: str | bytes) -> list[FeedItem]:
    parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=True, huge_tree=False)
    data = xml.encode("utf-8") if isinstance(xml, str) else xml
    root = etree.fromstring(data, parser=parser)
    if root is None:
        return []
    items: list[FeedItem] = []
    for node in root.iter("item"):  # RSS 2.0
        source = node.find("source")
        publisher = source.text.strip() if source is not None and source.text else None
        title = _html_to_text(node.findtext("title") or "")
        items.append(FeedItem(
            title=_strip_publisher_suffix(title, publisher),
            link=(node.findtext("link") or "").strip(),
            published_at=parse_published_at(node.findtext("pubDate") or node.findtext("{*}date")),
            summary=_html_to_text(node.findtext("description") or ""),
            publisher=publisher,
            publisher_url=source.get("url") if source is not None else None,
        ))
    for node in root.iter(f"{_ATOM}entry"):  # Atom
        link_el = node.find(f"{_ATOM}link[@rel='alternate']")
        if link_el is None:
            link_el = node.find(f"{_ATOM}link")
        author = node.findtext(f"{_ATOM}author/{_ATOM}name")
        items.append(FeedItem(
            title=_html_to_text(node.findtext(f"{_ATOM}title") or ""),
            link=(link_el.get("href") if link_el is not None else "") or "",
            published_at=parse_published_at(node.findtext(f"{_ATOM}published") or node.findtext(f"{_ATOM}updated")),
            summary=_html_to_text(node.findtext(f"{_ATOM}summary") or node.findtext(f"{_ATOM}content") or ""),
            publisher=author,
        ))
    return [i for i in items if i.title and i.link]


# --------------------------------------------------------------------------- articles

_DATE_META = (
    ("property", "article:published_time"),
    ("property", "og:published_time"),
    ("name", "pubdate"),
    ("name", "publishdate"),
    ("name", "date"),
    ("itemprop", "datePublished"),
    ("name", "dc.date"),
    ("name", "DC.date.issued"),
)


def _jsonld_dates(soup: BeautifulSoup) -> list[str]:
    found: list[str] = []
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        stack = [data]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                if isinstance(cur.get("datePublished"), str):
                    found.append(cur["datePublished"])
                stack.extend(cur.values())
            elif isinstance(cur, list):
                stack.extend(cur)
    return found


def _page_date(soup: BeautifulSoup) -> datetime | None:
    candidates: list[str] = []
    for attr, value in _DATE_META:
        tag = soup.find("meta", attrs={attr: re.compile(f"^{re.escape(value)}$", re.I)})
        if tag and tag.get("content"):
            candidates.append(str(tag["content"]))
    candidates += _jsonld_dates(soup)
    time_tag = soup.find("time", attrs={"datetime": True})
    if time_tag:
        candidates.append(str(time_tag["datetime"]))
    for c in candidates:
        parsed = parse_published_at(c)
        if parsed:
            return parsed
    return None


def _bs4_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript", "svg"]):
        tag.decompose()
    container = soup.find("article") or soup.find("main") or soup.body or soup
    blocks = [el.get_text(" ", strip=True) for el in container.find_all(["h1", "h2", "h3", "p", "li"])]
    return clean_text("\n\n".join(b for b in blocks if len(b.split()) >= 4))


def extract_article(html: str, url: str | None = None) -> Article:
    soup = BeautifulSoup(html, "lxml")
    og_title = soup.find("meta", attrs={"property": "og:title"})
    title = (og_title.get("content") if og_title else None) or (soup.title.string if soup.title else "") or ""
    canonical = soup.find("link", rel="canonical")
    canonical_link = str(canonical["href"]) if canonical and canonical.get("href") else None
    published = _page_date(soup)

    text, extractor = "", "bs4"
    try:
        import trafilatura  # optional extra

        extracted = trafilatura.extract(html, url=url, include_comments=False, include_tables=False,
                                        favor_precision=True, output_format="txt")
        if extracted and len(extracted.split()) >= 60:
            text, extractor = clean_text(extracted), "trafilatura"
    except ImportError:
        pass
    if not text:
        text = _bs4_text(soup)
    return Article(title=clean_text(str(title)), text=text, published_at=published,
                   canonical_link=canonical_link, extractor=extractor)

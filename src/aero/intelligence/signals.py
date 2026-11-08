"""Strategic signal extraction and aggregation (replaces ``app/intelligence/signal_extractor.py``).

v1 fired a rule when the document's topic matched **or** any keyword appeared, so one
document triggered many rules (595 signals from 190 documents). Its "evidence_strength"
was a constant, and no passage was recorded (finding B5).

``RuleSignalExtractor`` (extractor id ``rule:v1-strict``) keeps v1's rules
(``seeds/signal_rules_v1.yaml``) but:

* requires a keyword occurrence in the text, plus either a topic match or two
  distinct keywords;
* ignores keyword hits inside company names (``taxonomy.ignore_phrases``), so
  "Airbus Defence and **Space**" is not evidence about space;
* emits at most one signal per (document, rule), from its best chunk;
* stores the exact sentence and its character span, so every signal is citable.

An LLM extractor (roadmap phase 5) will implement the same ``extract`` signature.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from aero.dates import utcnow
from aero.domain import Document, EvidenceChunk, Signal, SignalIn, SignalType
from aero.storage import Store
from aero.taxonomy import Taxonomy

DEFAULT_IMPACT = 0.62
MAX_QUOTE_CHARS = 400


@dataclass(frozen=True)
class SignalRule:
    signal_type: SignalType
    title: str
    description: str
    topics: tuple[str, ...]
    keywords: tuple[str, ...]
    pattern: re.Pattern[str]


def _keyword_pattern(keywords: Iterable[str]) -> re.Pattern[str]:
    escaped = sorted((re.escape(k.strip()) for k in keywords if k.strip()), key=len, reverse=True)
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(escaped) + r")(?![a-z0-9])", re.IGNORECASE)


def load_signal_rules(path: str | Path) -> list[SignalRule]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    rules: list[SignalRule] = []
    for signal_type in ("opportunity", "risk", "trend"):
        for r in data.get(signal_type, []):
            rules.append(SignalRule(
                signal_type=signal_type, title=r["title"], description=r["description"],
                topics=tuple(r.get("topic_match", [])), keywords=tuple(r["keywords"]),
                pattern=_keyword_pattern(r["keywords"]),
            ))
    return rules


def sentence_span(text: str, pos: int) -> tuple[int, int]:
    """Bounds of the sentence (or line) containing ``pos``, whitespace-trimmed, at most MAX_QUOTE_CHARS."""
    start = max(text.rfind(c, 0, pos) for c in ".!?\n") + 1
    ends = [i for i in (text.find(c, pos) for c in ".!?\n") if i != -1]
    end = (min(ends) + 1) if ends else len(text)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if end - start > MAX_QUOTE_CHARS:
        start = max(start, pos - MAX_QUOTE_CHARS // 2)
        end = min(end, start + MAX_QUOTE_CHARS)
    return start, end


class RuleSignalExtractor:
    extractor_id = "rule:v1-strict"

    def __init__(self, rules: Sequence[SignalRule], taxonomy: Taxonomy) -> None:
        self.rules = list(rules)
        self.taxonomy = taxonomy
        self._ignore = _keyword_pattern(taxonomy.ignore_phrases) if taxonomy.ignore_phrases else None
        self._impact = dict(taxonomy.topic_impact_weights)

    def _hits(self, rule: SignalRule, text: str) -> list[re.Match[str]]:
        ignored = [m.span() for m in self._ignore.finditer(text)] if self._ignore else []
        return [m for m in rule.pattern.finditer(text)
                if not any(a <= m.start() < b for a, b in ignored)]

    def extract(self, document: Document, chunks: Sequence[EvidenceChunk]) -> list[SignalIn]:
        doc_topics = set(document.topics)
        out: list[SignalIn] = []
        for rule in self.rules:
            topical = bool(doc_topics.intersection(rule.topics))
            best: tuple[int, EvidenceChunk, re.Match[str]] | None = None
            for chunk in chunks:
                hits = self._hits(rule, chunk.text)
                distinct = len({h.group(0).lower() for h in hits})
                if not hits or (not topical and distinct < 2):
                    continue
                if best is None or distinct > best[0]:
                    best = (distinct, chunk, hits[0])
            if best is None:
                continue
            distinct, chunk, hit = best
            q_start, q_end = sentence_span(chunk.text, hit.start())
            topic = next((t for t in rule.topics if t in doc_topics), rule.topics[0] if rule.topics else "General")
            confidence = min(0.95, 0.35 + 0.15 * distinct + (0.15 if topical else 0.0))
            if chunk.content_level == "headline_summary":
                confidence *= 0.85  # a headline supports "X was reported", little more
            out.append(SignalIn(
                document_id=document.id, chunk_id=chunk.chunk_id, signal_type=rule.signal_type, title=rule.title,
                description=rule.description, topic=topic,
                business_unit=self.taxonomy.business_unit_for(topic, chunk.text),
                evidence_quote=chunk.text[q_start:q_end], quote_start=chunk.char_start + q_start,
                quote_end=chunk.char_start + q_end, impact=self._impact.get(topic, DEFAULT_IMPACT),
                confidence=round(confidence, 3), extractor=self.extractor_id,
            ))
        return out


def run_extraction(
    store: Store, extractor: RuleSignalExtractor, document_ids: Iterable[int] | None = None
) -> Counter[str]:
    stats: Counter[str] = Counter()
    for doc_id in document_ids if document_ids is not None else store.list_document_ids():
        doc = store.get_document(doc_id)
        if doc is None:
            continue
        signals = extractor.extract(doc, store.document_chunks(doc_id))
        store.replace_signals(doc_id, extractor.extractor_id, signals)
        stats["documents"] += 1
        stats["signals"] += len(signals)
        for s in signals:
            stats[s.signal_type] += 1
    return stats


# ----------------------------------------------------------------------------- aggregation


class TimeseriesPoint(BaseModel):
    period: str
    topic: str
    signal_type: SignalType
    count: int
    avg_impact: float


class TopMover(BaseModel):
    topic: str
    current: int
    previous: int
    change: int
    signal_types: dict[str, int]


def signal_timeseries(
    signals: Iterable[Signal], bucket: Literal["month", "quarter"] = "month"
) -> list[TimeseriesPoint]:
    """Counts per period/topic/type by publication date; undated signals are left out."""
    groups: dict[tuple[str, str, SignalType], list[float]] = defaultdict(list)
    for s in signals:
        if s.published_at is None:
            continue
        d = s.published_at
        period = f"{d.year}-{d.month:02d}" if bucket == "month" else f"{d.year}-Q{(d.month - 1) // 3 + 1}"
        groups[(period, s.topic, s.signal_type)].append(s.impact)
    return [
        TimeseriesPoint(period=p, topic=t, signal_type=st, count=len(v), avg_impact=round(sum(v) / len(v), 3))
        for (p, t, st), v in sorted(groups.items())
    ]


def top_movers(signals: Iterable[Signal], window_days: int = 90, now: datetime | None = None) -> list[TopMover]:
    """Topics whose signal count changed most between the last window and the one before it."""
    now = now or utcnow()
    current_start, previous_start = now - timedelta(days=window_days), now - timedelta(days=2 * window_days)
    cur: Counter[str] = Counter()
    prev: Counter[str] = Counter()
    types: dict[str, Counter[str]] = defaultdict(Counter)
    for s in signals:
        if s.published_at is None or s.published_at > now:
            continue
        if s.published_at >= current_start:
            cur[s.topic] += 1
            types[s.topic][s.signal_type] += 1
        elif s.published_at >= previous_start:
            prev[s.topic] += 1
    movers = [TopMover(topic=t, current=cur[t], previous=prev[t], change=cur[t] - prev[t],
                       signal_types=dict(types[t])) for t in set(cur) | set(prev)]
    movers.sort(key=lambda m: (-abs(m.change), -m.current, m.topic))
    return movers

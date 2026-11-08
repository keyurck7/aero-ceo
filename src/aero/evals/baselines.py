"""Keyword-rule signal baseline (v1 rules, ``rule:v1-strict``). EVALUATION ONLY (ADR 0008).

The rules cannot read direction ("the failed fighter deal" fires an FCAS *opportunity* rule), and their numbers
were invented, so they no longer produce stored signals. They remain as the baseline the LLM extractor must beat
on the signal gold set: ``aero eval signals`` reports both.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel

from aero.domain import Document, EvidenceChunk, SignalType
from aero.storage import Store
from aero.taxonomy import Taxonomy

MAX_QUOTE_CHARS = 400


class BaselineSignal(BaseModel):
    document_id: int
    chunk_id: int
    signal_type: SignalType
    topic: str
    evidence_quote: str




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

    def _hits(self, rule: SignalRule, text: str) -> list[re.Match[str]]:
        ignored = [m.span() for m in self._ignore.finditer(text)] if self._ignore else []
        return [m for m in rule.pattern.finditer(text)
                if not any(a <= m.start() < b for a, b in ignored)]

    def extract(self, document: Document, chunks: Sequence[EvidenceChunk]) -> list[BaselineSignal]:
        doc_topics = set(document.topics)
        out: list[BaselineSignal] = []
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
            _, chunk, hit = best
            q_start, q_end = sentence_span(chunk.text, hit.start())
            topic = next((t for t in rule.topics if t in doc_topics), rule.topics[0] if rule.topics else "General")
            out.append(BaselineSignal(document_id=document.id, chunk_id=chunk.chunk_id, signal_type=rule.signal_type,
                                      topic=topic, evidence_quote=chunk.text[q_start:q_end]))
        return out


def run_baseline(store: Store, extractor: RuleSignalExtractor,
                 document_ids: Iterable[int] | None = None) -> list[BaselineSignal]:
    """Baseline predictions for evaluation; nothing is written to the database."""
    out: list[BaselineSignal] = []
    for doc_id in document_ids if document_ids is not None else store.list_document_ids():
        doc = store.get_document(doc_id)
        if doc is not None:
            out += extractor.extract(doc, store.document_chunks(doc_id))
    return out

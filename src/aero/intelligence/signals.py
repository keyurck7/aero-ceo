"""Strategic signal extraction (LLM, ADR 0008) and aggregation.

Replaces v1's keyword rules (``app/intelligence/signal_extractor.py``) and v2's interim ``rule:v1-strict``,
which could not read direction: "what the failed fighter deal means" counted as an FCAS *opportunity*.

Per document, ``LLMSignalExtractor``:

1. asks the model ``N`` times (self-consistency samples) for signals in a JSON schema whose enums come from the
   taxonomy: type, topic, ``direction`` for Airbus, ``event_status``, ``materiality`` (written rubric) and an
   exact quote with its chunk id;
2. rejects deterministically: unknown chunk, quote not found verbatim in that chunk, or an incoherent
   type/direction pair (an opportunity cannot be bad news for Airbus, a risk cannot be good news);
3. keeps a signal only if a strict majority of samples produced it (same chunk, type and topic). Field values
   are the modal ones, and fields on which samples disagreed are recorded in ``disputed_fields``;
4. asks an independent verifier whether the chunk supports the signal's claim, and stores only ``supported``.

Nothing is scored: every stored field is a category, a quote, or a count. Rejections are reported with reasons
so the extraction can be audited and evaluated (``aero eval signals``).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, create_model

from aero.dates import utcnow
from aero.domain import (
    ALLOWED_DIRECTIONS,
    CorroborationItem,
    Direction,
    Document,
    EventStatus,
    EvidenceChunk,
    Materiality,
    Signal,
    SignalFilters,
    SignalIn,
    SignalType,
)
from aero.intelligence.verification import Verifier, find_quote, information_credibility
from aero.llm import LLM, PromptRegistry, default_registry, generate
from aero.llm.concurrency import pmap
from aero.log import get_logger
from aero.retrieval import HybridRetriever
from aero.storage import Store
from aero.taxonomy import DEFAULT_TOPIC, Taxonomy

log = get_logger(__name__)

MAX_QUOTE_CHARS = 400


def load_materiality_rubric(path: str | Path) -> str:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    lines = []
    for level, spec in data["levels"].items():
        lines.append(f"  - {level}: {spec['definition']}")
        lines += [f"      e.g. {ex}" for ex in spec.get("examples", [])]
    return "\n".join(lines)


@dataclass
class Candidate:
    sample: int
    chunk_id: int
    signal_type: SignalType
    topic: str
    title: str
    claim: str
    direction: Direction
    event_status: EventStatus
    materiality: Materiality
    materiality_rationale: str
    quote: str
    span: tuple[int, int] = (0, 0)

    @property
    def key(self) -> tuple[int, str, str]:
        return self.chunk_id, self.signal_type, self.topic


@dataclass
class Rejection:
    document_id: int
    reason: str
    detail: dict[str, Any]


@dataclass
class DocumentResult:
    document_id: int
    signals: list[SignalIn] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    skipped: str | None = None


def _mode(values: Sequence[str]) -> tuple[str, bool]:
    """Most common value (first-seen wins a tie) and whether the samples disagreed at all."""
    counts = Counter(values)
    best = max(counts.values())
    winner = next(v for v in values if counts[v] == best)
    return winner, len(counts) > 1


class LLMSignalExtractor:
    def __init__(self, llm: LLM, verifier: Verifier, taxonomy: Taxonomy, materiality_rubric: str, *,
                 samples: int = 3, temperature: float = 0.7, registry: PromptRegistry | None = None) -> None:
        if samples < 1:
            raise ValueError("samples must be >= 1")
        self.llm = llm
        self.verifier = verifier
        self.taxonomy = taxonomy
        self.rubric = materiality_rubric
        self.samples = samples
        self.temperature = temperature if samples > 1 else 0.0
        self.prompt = (registry or default_registry()).get("signal_extract")

    @property
    def extractor_id(self) -> str:
        return self.prompt.ref

    @cached_property
    def topics(self) -> list[str]:
        return [*self.taxonomy.topic_names(), DEFAULT_TOPIC]

    @cached_property
    def output_model(self) -> type[BaseModel]:
        """The response schema, with the taxonomy's topics as an enum so decoding cannot invent topics."""
        topic_type: Any = Literal[tuple(self.topics)]
        item = create_model(
            "ExtractedSignal", chunk_id=(int, ...), signal_type=(SignalType, ...), topic=(topic_type, ...),
            title=(str, ...), claim=(str, ...), direction_for_airbus=(Direction, ...),
            event_status=(EventStatus, ...), materiality=(Materiality, ...), materiality_rationale=(str, ...),
            quote=(str, ...),
        )
        return create_model("SignalExtraction", document_topics=(list[topic_type], ...), signals=(list[item], ...))

    def _variables(self, document: Document, chunks: Sequence[EvidenceChunk], publisher: str) -> dict[str, str]:
        body = "\n".join(f'<chunk id="{c.chunk_id}">\n{c.text}\n</chunk>' for c in chunks)
        return {
            "materiality_rubric": self.rubric, "topics": ", ".join(self.topics),
            "business_units": ", ".join(self.taxonomy.business_units), "publisher": publisher,
            "published_at": document.published_at.date().isoformat() if document.published_at else "unknown",
            "content_level": document.content_level, "title": document.title, "chunks": body,
        }

    def _sample(self, variables: dict[str, str], sample: int) -> tuple[list[str], list[dict[str, Any]]]:
        out, _ = generate(self.llm, self.prompt, self.output_model, variables,
                          temperature=self.temperature, seed=sample if self.samples > 1 else None,
                          top_p=0.8 if self.samples > 1 else None)
        data = out.model_dump()
        return list(data["document_topics"]), list(data["signals"])

    def extract(self, document: Document, chunks: Sequence[EvidenceChunk]) -> DocumentResult:
        result = DocumentResult(document_id=document.id)
        if not chunks:
            result.skipped = "no_chunks"
            return result
        if document.injection_flags:
            result.skipped = "quarantined"  # never feed instruction-like text to the model
            return result
        by_id = {c.chunk_id: c for c in chunks}
        variables = self._variables(document, chunks, chunks[0].publisher)
        outputs = [self._sample(variables, s) for s in range(self.samples)]

        # 1-2. deterministic checks per candidate
        candidates: list[Candidate] = []
        for s, (_, raw_signals) in enumerate(outputs):
            for raw in raw_signals:
                cand = Candidate(sample=s, chunk_id=raw["chunk_id"], signal_type=raw["signal_type"],
                                 topic=raw["topic"], title=raw["title"].strip()[:160], claim=raw["claim"].strip(),
                                 direction=raw["direction_for_airbus"], event_status=raw["event_status"],
                                 materiality=raw["materiality"],
                                 materiality_rationale=raw["materiality_rationale"].strip(),
                                 quote=raw["quote"].strip())
                reason = self._check(cand, by_id)
                if reason:
                    result.rejections.append(Rejection(document.id, reason, {"sample": s, **raw}))
                else:
                    candidates.append(cand)

        # 3. self-consistency: strict majority of samples, modal field values
        groups: dict[tuple[int, str, str], list[Candidate]] = defaultdict(list)
        for cand in candidates:
            if all(c.sample != cand.sample for c in groups[cand.key]):  # one vote per sample
                groups[cand.key].append(cand)
        majority = self.samples // 2 + 1
        kept: list[tuple[Candidate, int, list[str]]] = []
        for key, members in groups.items():
            if len(members) < majority:
                result.rejections.append(Rejection(document.id, "no_majority", {
                    "key": list(key), "samples_agreeing": len(members), "samples_total": self.samples}))
                continue
            disputed: list[str] = []
            chosen: dict[str, str] = {}
            for fname in ("direction", "event_status", "materiality"):
                value, split = _mode([getattr(m, fname) for m in members])
                chosen[fname] = value
                if split:
                    disputed.append(fname)
            rep = next((m for m in members if all(getattr(m, f) == v for f, v in chosen.items())), members[0])
            kept.append((rep, len(members), disputed))

        # 4. independent verification against the chunk
        for rep, agreeing, disputed in kept:
            verdict = self.verifier.verify(rep.claim, by_id[rep.chunk_id])
            if verdict.verdict != "supported":
                result.rejections.append(Rejection(document.id, f"verification_{verdict.verdict}", {
                    "claim": rep.claim, "chunk_id": rep.chunk_id, "reason": verdict.reason}))
                continue
            chunk = by_id[rep.chunk_id]
            result.signals.append(SignalIn(
                document_id=document.id, chunk_id=rep.chunk_id, signal_type=rep.signal_type, title=rep.title,
                description=rep.claim, topic=rep.topic, business_unit=self.taxonomy.business_unit_for(rep.topic),
                direction=rep.direction, event_status=rep.event_status, materiality=rep.materiality,
                materiality_rationale=rep.materiality_rationale,
                evidence_quote=chunk.text[rep.span[0]:rep.span[1]],
                quote_start=chunk.char_start + rep.span[0], quote_end=chunk.char_start + rep.span[1],
                verification_note=verdict.reason, samples_agreeing=agreeing, samples_total=self.samples,
                disputed_fields=disputed, extractor=self.extractor_id, model=self.llm.model,
            ))

        topic_votes = Counter(t for topics, _ in outputs for t in dict.fromkeys(topics))
        result.topics = [t for t in self.topics if topic_votes[t] >= majority and t != DEFAULT_TOPIC]
        return result

    @staticmethod
    def _check(cand: Candidate, by_id: dict[int, EvidenceChunk]) -> str | None:
        chunk = by_id.get(cand.chunk_id)
        if chunk is None:
            return "unknown_chunk"
        if not cand.quote or len(cand.quote) > MAX_QUOTE_CHARS * 2:
            return "bad_quote_length"
        span = find_quote(chunk.text, cand.quote)
        if span is None:
            return "ungrounded_quote"
        cand.span = span
        if cand.direction not in ALLOWED_DIRECTIONS[cand.signal_type]:
            return "incoherent_direction"
        if len(cand.claim) < 3 or len(cand.title) < 3:
            return "empty_claim"
        return None


@dataclass
class ExtractionReport:
    extractor: str
    model: str
    stats: Counter[str] = field(default_factory=Counter)
    rejections: list[Rejection] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {"extractor": self.extractor, "model": self.model, **dict(self.stats)}


def run_extraction(store: Store, extractor: LLMSignalExtractor, document_ids: Iterable[int] | None = None,
                   max_workers: int = 4, update_topics: bool = True) -> ExtractionReport:
    """Extract, verify and store signals per document (idempotent per document and extractor version)."""
    report = ExtractionReport(extractor=extractor.extractor_id, model=extractor.llm.model)
    ids = list(document_ids if document_ids is not None else store.list_document_ids())

    def one(doc_id: int) -> DocumentResult | None:
        doc = store.get_document(doc_id)
        if doc is None:
            return None
        res = extractor.extract(doc, store.document_chunks(doc_id))
        if res.skipped is None:
            store.replace_signals(doc_id, extractor.extractor_id, res.signals)
            if update_topics:
                store.set_document_topics(doc_id, res.topics, extractor.extractor_id)
        return res

    for res in pmap(one, ids, max_workers):
        if res is None:
            continue
        report.stats["documents"] += 1
        if res.skipped:
            report.stats[f"skipped_{res.skipped}"] += 1
            continue
        report.stats["signals"] += len(res.signals)
        for s in res.signals:
            report.stats[s.signal_type] += 1
            report.stats[f"direction_{s.direction}"] += 1
        for r in res.rejections:
            report.stats[f"rejected_{r.reason}"] += 1
        report.rejections += res.rejections
    log.info("signal extraction", extra=report.summary())
    return report


# ----------------------------------------------------------------------------- corroboration


def corroborate(store: Store, retriever: HybridRetriever, verifier: Verifier, signal: Signal,
                depth: int = 5) -> tuple[list[CorroborationItem], int]:
    """Judge the closest passages from *other* publisher groups against the signal's claim."""
    related = retriever.find_related(signal.chunk_id, top_k=depth, other_publishers_only=True)
    items: list[CorroborationItem] = []
    for hit in related:
        v = verifier.verify(signal.description, hit)
        if v.verdict != "not_enough_info":
            items.append(CorroborationItem(chunk_id=hit.chunk_id, verdict=v.verdict,
                                           publisher_group=(hit.publisher_group or hit.publisher).lower()))
    grade = information_credibility(items)
    store.set_signal_corroboration(signal.id, items, grade)
    return items, grade


def run_corroboration(store: Store, retriever: HybridRetriever, verifier: Verifier,
                      filters: SignalFilters | None = None, depth: int = 5, max_workers: int = 4) -> Counter[str]:
    signals = store.list_signals(filters or SignalFilters(), limit=100_000)
    stats: Counter[str] = Counter()
    for _, grade in pmap(lambda s: corroborate(store, retriever, verifier, s, depth), signals, max_workers):
        stats[f"credibility_{grade}"] += 1
    stats["signals"] = len(signals)
    return stats


# ----------------------------------------------------------------------------- aggregation (counts only)


class TimeseriesPoint(BaseModel):
    period: str
    topic: str
    signal_type: SignalType
    count: int
    by_materiality: dict[str, int]
    by_direction: dict[str, int]


class TopMover(BaseModel):
    topic: str
    current: int
    previous: int
    change: int
    signal_types: dict[str, int]


def signal_timeseries(
    signals: Iterable[Signal], bucket: Literal["month", "quarter"] = "month"
) -> list[TimeseriesPoint]:
    """Counts per period/topic/type by publication date, split by materiality and direction; undated left out."""
    groups: dict[tuple[str, str, SignalType], list[Signal]] = defaultdict(list)
    for s in signals:
        if s.published_at is None:
            continue
        d = s.published_at
        period = f"{d.year}-{d.month:02d}" if bucket == "month" else f"{d.year}-Q{(d.month - 1) // 3 + 1}"
        groups[(period, s.topic, s.signal_type)].append(s)
    return [
        TimeseriesPoint(period=p, topic=t, signal_type=st, count=len(v),
                        by_materiality=dict(Counter(s.materiality for s in v)),
                        by_direction=dict(Counter(s.direction for s in v)))
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

"""Ingestion pipeline: raw item -> clean -> dedup -> topics -> chunks -> embeddings.

The rule that fixes finding B1: the document text is *only* what the publisher
wrote (title + body, or title + feed summary). Why we collected the item (search
query, feed, collector) goes to ``ingestion_context``, which is never embedded,
indexed or shown as evidence.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from aero.config import Settings
from aero.dates import parse_published_at
from aero.domain import ChunkIn, ContentLevel, DocumentIn, SourceIn
from aero.embeddings import Embedder
from aero.log import get_logger
from aero.storage import DuplicateDocumentError, Store
from aero.taxonomy import Taxonomy, infer_topics
from aero.text import canonical_url, chunk_text, clean_text, content_hash, scan_injection, simhash64

log = get_logger(__name__)

MIN_WORDS: dict[str, int] = {"full_text": 80, "page": 60, "headline_summary": 5}


@dataclass
class RawItem:
    source: SourceIn
    title: str
    url: str
    body: str
    content_level: ContentLevel = "full_text"
    published_at: str | datetime | None = None
    ingestion_context: dict[str, Any] = field(default_factory=dict)


Outcome = Literal["inserted", "duplicate", "too_short", "empty"]


@dataclass
class IngestResult:
    outcome: Outcome
    document_id: int | None = None
    detail: str | None = None


def compose_text(title: str, body: str) -> str:
    """Title + body. The body never repeats a title it already starts with."""
    title, body = clean_text(title), clean_text(body)
    if body.lower().startswith(title.lower()):
        return body
    return f"{title}\n\n{body}" if body else title


class Ingestor:
    def __init__(self, store: Store, embedder: Embedder, taxonomy: Taxonomy, settings: Settings) -> None:
        self.store = store
        self.embedder = embedder
        self.taxonomy = taxonomy
        self.settings = settings
        self._source_ids: dict[str, int] = {}
        self.counts: Counter[str] = Counter()

    def _source_id(self, src: SourceIn) -> int:
        if src.name not in self._source_ids:
            self._source_ids[src.name] = self.store.upsert_source(src).id
        return self._source_ids[src.name]

    def ingest(self, item: RawItem, run_id: int | None = None) -> IngestResult:
        text = compose_text(item.title, item.body)
        if not text:
            self.counts["empty"] += 1
            return IngestResult("empty")
        if len(text.split()) < MIN_WORDS[item.content_level]:
            self.counts["too_short"] += 1
            return IngestResult("too_short", detail=f"{len(text.split())} words")
        doc_in = DocumentIn(
            source_id=self._source_id(item.source),
            title=clean_text(item.title) or "(untitled)",
            url=item.url,
            canonical_url=canonical_url(item.url),
            published_at=parse_published_at(item.published_at),
            clean_text=text,
            content_level=item.content_level,
            topics=infer_topics(text, self.taxonomy),
            content_hash=content_hash(text),
            simhash=simhash64(text),
            ingestion_context=item.ingestion_context,
            injection_flags=scan_injection(f"{item.title}\n{item.body}"),
        )
        try:
            doc = self.store.insert_document(doc_in, pipeline_run_id=run_id)
        except DuplicateDocumentError as dup:
            self.counts[f"duplicate_{dup.reason}"] += 1
            return IngestResult("duplicate", dup.existing_id, dup.reason)
        chunks = chunk_text(text, self.settings.chunk_max_words, self.settings.chunk_overlap_words)
        self.store.replace_chunks(doc.id, [
            ChunkIn(chunk_index=c.index, text=c.text, char_start=c.char_start, char_end=c.char_end,
                    word_count=c.word_count)
            for c in chunks
        ])
        self.counts["inserted"] += 1
        if doc.injection_flags:
            self.counts["injection_flagged"] += 1
            log.warning("document %s flagged %s: %s", doc.id, doc.injection_flags, doc.url)
        return IngestResult("inserted", doc.id)

    def embed_pending(self, batch_size: int | None = None) -> int:
        """Embed every chunk missing a vector for the current model. Incremental: never a full rebuild."""
        batch = batch_size or max(self.settings.embedding_batch_size * 4, 64)
        total = 0
        seen: set[int] = set()
        while True:
            pending = self.store.chunks_needing_embedding(self.embedder.model_name, limit=batch)
            if not pending:
                return total
            if seen.intersection(cid for cid, _ in pending):  # writes not sticking: fail, don't spin
                raise RuntimeError("embeddings were not persisted; aborting to avoid an infinite loop")
            seen.update(cid for cid, _ in pending)
            vectors = self.embedder.embed_documents([t for _, t in pending])
            self.store.set_embeddings([(cid, vectors[i]) for i, (cid, _) in enumerate(pending)],
                                      self.embedder.model_name)
            total += len(pending)
            log.info("embedded %d chunks (total %d)", len(pending), total)

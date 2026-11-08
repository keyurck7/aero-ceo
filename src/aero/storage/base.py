"""Storage port: the one interface pipelines, retrieval and MCP servers depend on.

v1 had SQLite for rows, a FAISS file for vectors and a JSON file holding a copy of
every chunk's text (findings P5/B10). v2 keeps rows, vectors and the lexical index
in one database behind this interface, with two adapters:

* ``PostgresStore`` (production): pgvector HNSW + tsvector/GIN, filters in SQL.
* ``SQLiteStore`` (local/dev/CI): FTS5 + numpy brute-force cosine. Fine to ~10^5 chunks.

Both adapters pass the same contract test suite (``tests/test_store_contract.py``).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

from aero.domain import (
    ALLOWED_TRANSITIONS,
    ChunkIn,
    CorroborationItem,
    Document,
    DocumentIn,
    EvidenceChunk,
    EvidenceLinkIn,
    Recommendation,
    RecommendationEvidence,
    RecommendationIn,
    RecommendationReview,
    RecommendationStatus,
    SearchFilters,
    Signal,
    SignalFilters,
    SignalIn,
    Source,
    SourceIn,
    StoreStats,
)
from aero.text import from_signed64, hamming64, to_signed64

# Measured on 3-word-shingle SimHash: a byline/footer moves a 300-word article by <= ~11 bits
# (mean ~4); unrelated texts are >= ~19 bits apart. 7 catches typical syndication safely.
NEAR_DUPLICATE_MAX_HAMMING = 7

# Kept as one string: easier to maintain than a list literal.
_STOPWORD_TEXT = (
    "a an and are as at be but by can could did do does for from had has have how i if in into is it its "
    "may me might more most my no not of on or our should so than that the their them then there these they "
    "this those to was we were what when where which while who why will with would you your about against "
    "airbus"
)
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def simhash_bands(simhash: int) -> tuple[int, ...]:
    """Eight 8-bit bands. Pigeonhole: two hashes within Hamming distance 7 share >= 1 band,
    so near-duplicate lookup is an indexed equality query, not a table scan."""
    value = from_signed64(simhash)
    return tuple((value >> (8 * i)) & 0xFF for i in range(8))


def lexical_terms(query: str, max_terms: int = 24) -> list[str]:
    """Safe search terms for FTS5 / to_tsquery: alphanumerics only, stopwords removed.

    "airbus" is a stopword here because nearly every document mentions it; as a
    lexical term it adds noise, not signal. Dense retrieval still sees the full query.
    """
    terms: dict[str, None] = {}
    for tok in re.findall(r"[A-Za-z0-9]+", query.lower()):
        if tok not in _STOPWORDS and (len(tok) > 1 or tok.isdigit()):
            terms.setdefault(tok, None)
    return list(terms)[:max_terms]


def is_near_duplicate(a: int, b: int) -> bool:
    return hamming64(a, b) <= NEAR_DUPLICATE_MAX_HAMMING


# ---- signal SQL shared by both adapters (ADR 0008: no impact/confidence floats) -----------------
SIGNAL_COLUMNS: tuple[str, ...] = (
    "document_id", "chunk_id", "signal_type", "title", "description", "topic", "business_unit", "direction",
    "event_status", "materiality", "materiality_rationale", "evidence_quote", "quote_start", "quote_end",
    "verification_note", "samples_agreeing", "samples_total", "disputed_fields", "extractor", "model",
)
SIGNAL_ORDER = (
    "CASE g.materiality WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, "
    "d.published_at DESC NULLS LAST, g.id"
)


def signal_filter_columns(filters: SignalFilters) -> list[tuple[str, list[Any]]]:
    """(column, allowed values) pairs for the IN-list filters that are set."""
    pairs: list[tuple[str, Sequence[Any] | None]] = [
        ("g.signal_type", filters.signal_types), ("g.topic", filters.topics),
        ("g.business_unit", filters.business_units), ("g.direction", filters.directions),
        ("g.materiality", filters.materiality), ("g.information_credibility", filters.information_credibility),
    ]
    return [(col, list(vals)) for col, vals in pairs if vals]


class DuplicateDocumentError(Exception):
    def __init__(self, existing_id: int, reason: str) -> None:
        super().__init__(f"duplicate of document {existing_id} ({reason})")
        self.existing_id = existing_id
        self.reason = reason


class InvalidTransitionError(ValueError):
    pass


def check_transition(current: str, target: str) -> None:
    if target not in ALLOWED_TRANSITIONS.get(current, set()):
        allowed = sorted(ALLOWED_TRANSITIONS.get(current, set())) or ["none (terminal)"]
        raise InvalidTransitionError(f"cannot move a {current} recommendation to {target}; allowed: {allowed}")


class Store(ABC):
    """Synchronous storage port. Implementations must be safe to call from worker threads."""

    dialect: str

    # ---- lifecycle --------------------------------------------------------------------
    @abstractmethod
    def migrate(self) -> list[str]:
        """Apply pending migrations; return the versions applied."""

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- sources ----------------------------------------------------------------------
    @abstractmethod
    def upsert_source(self, source: SourceIn) -> Source:
        """Insert or update by unique ``name``."""

    @abstractmethod
    def get_source(self, source_id: int) -> Source | None: ...

    @abstractmethod
    def list_sources(self, source_type: str | None = None) -> list[Source]: ...

    # ---- documents & chunks -----------------------------------------------------------
    @abstractmethod
    def find_duplicate(self, canonical_url: str, content_hash: str, simhash: int) -> tuple[int, str] | None:
        """Return ``(document_id, reason)`` if an equivalent document exists."""

    @abstractmethod
    def insert_document(self, doc: DocumentIn, pipeline_run_id: int | None = None) -> Document:
        """Insert; raises ``DuplicateDocumentError`` on URL/hash/near-duplicate match."""

    @abstractmethod
    def get_document(self, document_id: int) -> Document | None: ...

    @abstractmethod
    def list_document_ids(self) -> list[int]: ...

    @abstractmethod
    def replace_chunks(self, document_id: int, chunks: Sequence[ChunkIn]) -> list[int]:
        """Atomically replace a document's chunks; embeddings are cleared."""

    # ---- embeddings -------------------------------------------------------------------
    @abstractmethod
    def chunks_needing_embedding(self, model: str, limit: int = 256) -> list[tuple[int, str]]:
        """Chunks with no embedding, or one from a different model."""

    @abstractmethod
    def set_embeddings(self, items: Sequence[tuple[int, np.ndarray]], model: str) -> None: ...

    @abstractmethod
    def get_embedding(self, chunk_id: int) -> tuple[np.ndarray, str] | None: ...

    # ---- search -----------------------------------------------------------------------
    @abstractmethod
    def dense_search(
        self, vector: np.ndarray, k: int, filters: SearchFilters, model: str
    ) -> list[tuple[int, float]]:
        """``(chunk_id, cosine)`` best first, filtered before ranking."""

    @abstractmethod
    def lexical_search(self, query: str, k: int, filters: SearchFilters) -> list[tuple[int, float]]:
        """``(chunk_id, score)`` best first (higher = better), filtered before ranking."""

    @abstractmethod
    def get_evidence(self, chunk_ids: Sequence[int]) -> dict[int, EvidenceChunk]: ...

    @abstractmethod
    def get_neighbours(self, chunk_id: int, window: int = 1) -> list[EvidenceChunk]:
        """The chunk plus up to ``window`` chunks either side, in document order."""

    @abstractmethod
    def document_chunks(self, document_id: int) -> list[EvidenceChunk]:
        """All chunks of a document in order, with citation metadata."""

    # ---- signals (migration 0002) ----------------------------------------------------
    @abstractmethod
    def set_document_topics(self, document_id: int, topics: Sequence[str], tagger: str) -> None:
        """Replace a document's topic labels and record who assigned them (e.g. the LLM prompt ref)."""

    @abstractmethod
    def replace_signals(self, document_id: int, extractor: str, signals: Sequence[SignalIn]) -> int:
        """Atomically replace one extractor's signals for one document (idempotent re-extraction)."""

    @abstractmethod
    def list_signals(self, filters: SignalFilters, limit: int = 50) -> list[Signal]:
        """Ordered by materiality (high first), then most recent, then id."""

    @abstractmethod
    def set_signal_corroboration(self, signal_id: int, items: Sequence[CorroborationItem],
                                 information_credibility: int) -> None:
        """Record the corroboration facts and the Admiralty information credibility they imply."""

    @abstractmethod
    def get_signal(self, signal_id: int) -> Signal | None: ...

    # ---- recommendations (migration 0002) --------------------------------------------
    @abstractmethod
    def upsert_recommendation(self, rec: RecommendationIn) -> Recommendation:
        """Insert by (key, version). An existing row is updated only while it is still a draft."""

    @abstractmethod
    def get_recommendation(self, recommendation_id: int) -> Recommendation | None: ...

    @abstractmethod
    def list_recommendations(
        self,
        status: RecommendationStatus | None = None,
        topic: str | None = None,
        business_unit: str | None = None,
        key: str | None = None,
        limit: int = 50,
    ) -> list[Recommendation]: ...

    @abstractmethod
    def set_recommendation_status(
        self, recommendation_id: int, status: RecommendationStatus, reviewer: str, reason: str
    ) -> Recommendation:
        """Validated lifecycle transition plus an audit row, in one transaction."""

    @abstractmethod
    def link_recommendation_evidence(
        self, recommendation_id: int, links: Sequence[EvidenceLinkIn], overwrite: bool = False
    ) -> int:
        """Attach chunk-level evidence. With ``overwrite=False`` existing links are kept (auto-linkers
        must never downgrade a human- or verifier-set relation)."""

    @abstractmethod
    def recommendation_evidence(self, recommendation_id: int) -> list[RecommendationEvidence]: ...

    @abstractmethod
    def recommendation_reviews(self, recommendation_id: int) -> list[RecommendationReview]: ...

    # ---- pipeline runs & stats --------------------------------------------------------
    @abstractmethod
    def start_run(self, flow: str, params: dict[str, Any]) -> int: ...

    @abstractmethod
    def finish_run(self, run_id: int, status: str, stats: dict[str, Any], error: str | None = None) -> None: ...

    @abstractmethod
    def stats(self) -> StoreStats: ...


def doc_simhash_columns(doc: DocumentIn) -> dict[str, int]:
    signed = to_signed64(doc.simhash)
    return {"simhash": signed, **{f"simhash_b{i}": band for i, band in enumerate(simhash_bands(signed))}}

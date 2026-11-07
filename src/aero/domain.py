"""Domain models shared by pipelines, storage, retrieval and the MCP servers.

These Pydantic models are the contract between layers. MCP tools return them
directly, so the JSON shape a client sees is defined here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SourceType = Literal["official", "news", "policy", "competitor", "community", "aggregator", "unknown"]

# How much of the publisher's content we hold. Retrieval and the verifier treat a
# headline+summary differently from a full article.
ContentLevel = Literal["full_text", "headline_summary", "page"]


def _require_aware(v: datetime | None) -> datetime | None:
    if v is not None and v.tzinfo is None:
        raise ValueError("datetimes must be timezone-aware (UTC)")
    return v


class SourceIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    source_type: SourceType = "unknown"
    base_url: str | None = None
    credibility_prior: float = Field(0.5, ge=0.0, le=1.0)
    # Syndicated copies of one story count once for "independent sources".
    publisher_group: str | None = None
    # What we may store: e.g. "full_text_permitted", "headline_link_only", "unknown".
    licence: str = "unknown"


class Source(SourceIn):
    id: int
    created_at: datetime


class ChunkIn(BaseModel):
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    word_count: int = Field(ge=1)


class DocumentIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: int
    title: str
    url: str
    canonical_url: str
    published_at: datetime | None = None
    clean_text: str = Field(min_length=1)
    content_level: ContentLevel = "full_text"
    language: str = "en"
    topics: list[str] = Field(default_factory=list)
    content_hash: str
    simhash: int
    # Provenance that must NEVER be embedded or shown as evidence (fixes finding B1):
    # e.g. {"collector": "news_rss", "query": "Airbus FCAS", "feed": "..."}.
    ingestion_context: dict[str, Any] = Field(default_factory=dict)
    injection_flags: list[str] = Field(default_factory=list)
    raw_object_key: str | None = None

    @field_validator("published_at")
    @classmethod
    def _published_aware(cls, v: datetime | None) -> datetime | None:
        return _require_aware(v)


class Document(DocumentIn):
    id: int
    collected_at: datetime


class EvidenceChunk(BaseModel):
    """A chunk joined with everything needed to cite it. No evidence without source metadata."""

    chunk_id: int
    document_id: int
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    title: str
    url: str
    canonical_url: str
    publisher: str
    publisher_group: str | None = None
    source_type: SourceType
    licence: str
    credibility_prior: float
    published_at: datetime | None
    collected_at: datetime
    topics: list[str]
    content_level: ContentLevel
    injection_flags: list[str] = Field(default_factory=list)


class SearchFilters(BaseModel):
    """Filters are applied inside the database query, before ranking (fixes finding B6)."""

    model_config = ConfigDict(extra="forbid")

    topics: list[str] | None = None
    source_types: list[SourceType] | None = None
    published_after: datetime | None = None
    published_before: datetime | None = None
    min_credibility: float = Field(0.0, ge=0.0, le=1.0)
    # Reproducibility: only evidence collected on or before this instant.
    as_of: datetime | None = None
    exclude_document_ids: list[int] | None = None

    @field_validator("published_after", "published_before", "as_of")
    @classmethod
    def _filters_aware(cls, v: datetime | None) -> datetime | None:
        return _require_aware(v)


class ScoreBreakdown(BaseModel):
    dense: float | None = None
    dense_rank: int | None = None
    lexical: float | None = None
    lexical_rank: int | None = None
    rrf: float
    rerank: float | None = None
    credibility: float
    recency: float
    final: float


class EvidenceHit(EvidenceChunk):
    rank: int
    scores: ScoreBreakdown


class SearchResult(BaseModel):
    query: str
    filters: SearchFilters
    hits: list[EvidenceHit]
    independent_publishers: int
    note: str | None = None


class StoreStats(BaseModel):
    sources: int
    documents: int
    chunks: int
    embedded_chunks: int
    embedding_models: list[str]
    latest_collected_at: datetime | None

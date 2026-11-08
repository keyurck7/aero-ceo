"""Domain models shared by pipelines, storage, retrieval and the MCP servers.

These Pydantic models are the contract between layers. MCP tools return them
directly, so the JSON shape a client sees is defined here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

SourceType = Literal["official", "news", "policy", "competitor", "community", "aggregator", "unknown"]

# How much of the publisher's content we hold. Retrieval and the verifier treat a
# headline+summary differently from a full article.
ContentLevel = Literal["full_text", "headline_summary", "page"]

# ----------------------------------------------------------------------------- source grading (ADR 0008)
# NATO Admiralty code: source reliability A-F, assessed separately from the information itself.
SourceReliability = Literal["A", "B", "C", "D", "E", "F"]
RELIABILITY_GRADES: tuple[SourceReliability, ...] = ("A", "B", "C", "D", "E", "F")
RELIABILITY_LABELS: dict[str, str] = {
    "A": "Completely reliable", "B": "Usually reliable", "C": "Fairly reliable",
    "D": "Not usually reliable", "E": "Unreliable", "F": "Reliability cannot be judged",
}
# The analyst assigns a category (a checkable fact about the publisher); this table is the policy that turns it
# into a grade. Changing it is a reviewed code change, not a tuning knob.
SourceCategory = Literal["official_primary", "established_newsroom", "specialist_or_commentary",
                         "aggregator_or_ugc", "off_topic_or_spam", "unreviewed"]
CATEGORY_RELIABILITY: dict[str, SourceReliability] = {
    "official_primary": "A",          # an organisation's own channel, about its own decisions
    "established_newsroom": "B",      # named editors and a corrections policy
    "specialist_or_commentary": "C",  # smaller trade titles, analyst blogs, commentary
    "aggregator_or_ugc": "D",         # press-release redistribution, finance communities, social posts
    "off_topic_or_spam": "E",         # spam or off-topic content observed in this corpus
    "unreviewed": "F",                # default until an analyst assigns a category
}
# Admiralty information credibility 1-6. The system only assigns 1, 5 and 6 (from corroboration facts);
# 2-4 need an analyst's judgement of logic and consistency.
InformationCredibility = Literal[1, 2, 3, 4, 5, 6]
CREDIBILITY_LABELS: dict[int, str] = {
    1: "Confirmed by other sources", 2: "Probably true", 3: "Possibly true", 4: "Doubtful", 5: "Improbable",
    6: "Truth cannot be judged",
}


def _require_aware(v: datetime | None) -> datetime | None:
    if v is not None and v.tzinfo is None:
        raise ValueError("datetimes must be timezone-aware (UTC)")
    return v


class SourceIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    source_type: SourceType = "unknown"
    base_url: str | None = None
    category: SourceCategory = "unreviewed"
    reliability_note: str = ""
    # Syndicated copies of one story count once for "independent sources".
    publisher_group: str | None = None
    # What we may store: e.g. "full_text_permitted", "headline_link_only", "unknown".
    licence: str = "unknown"


    @computed_field  # type: ignore[prop-decorator]
    @property
    def reliability(self) -> SourceReliability:
        """Admiralty grade, derived from the category by policy (never set directly)."""
        return CATEGORY_RELIABILITY[self.category]


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
    topics_tagger: str = Field("keyword:v1", description="keyword:v1 (provisional) or the LLM prompt that set topics")


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
    source_reliability: SourceReliability
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
    # Allowed Admiralty source grades. None = the server's default policy (everything except E).
    source_reliability: list[SourceReliability] | None = None
    # Documents with prompt-injection flags are excluded unless explicitly requested (ADR 0008: quarantine,
    # never down-weight).
    include_quarantined: bool = False
    # Reproducibility: only evidence collected on or before this instant.
    as_of: datetime | None = None
    exclude_document_ids: list[int] | None = None

    @field_validator("published_after", "published_before", "as_of")
    @classmethod
    def _filters_aware(cls, v: datetime | None) -> datetime | None:
        return _require_aware(v)


class ScoreBreakdown(BaseModel):
    """Relevance only. Trust is a filter and a displayed grade, never a term in this score (ADR 0008)."""

    dense: float | None = None
    dense_rank: int | None = None
    lexical: float | None = None
    lexical_rank: int | None = None
    rrf: float
    rerank: float | None = None
    final: float = Field(description="The value the hits are ordered by: rerank score if reranking ran, else RRF")


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


# ----------------------------------------------------------------------------- signals (migration 0002)

SignalType = Literal["opportunity", "risk", "trend"]
# How the reported development bears on Airbus. Reading this is what keyword rules could not do (ADR 0008, H7).
Direction = Literal["positive", "negative", "mixed", "neutral"]
EventStatus = Literal["announced", "planned", "proposed", "under_negotiation", "awarded", "in_progress",
                      "completed", "delayed", "cancelled", "failed", "speculative", "commentary"]
# Ordinal judged against the written rubric in seeds/materiality_rubric.yaml; never a float.
Materiality = Literal["high", "medium", "low"]
MATERIALITY_ORDER: dict[str, int] = {"high": 0, "medium": 1, "low": 2}
# Coherence policy: an opportunity cannot be bad news for Airbus, a risk cannot be good news.
ALLOWED_DIRECTIONS: dict[str, set[str]] = {
    "opportunity": {"positive", "mixed"},
    "risk": {"negative", "mixed"},
    "trend": {"positive", "negative", "mixed", "neutral"},
}


class CorroborationItem(BaseModel):
    chunk_id: int
    publisher_group: str
    verdict: Literal["supported", "contradicted"]


class SignalIn(BaseModel):
    """A strategic signal: an LLM judgement grounded in an exact quote, verified, and agreed across samples."""

    model_config = ConfigDict(extra="forbid")

    document_id: int
    chunk_id: int
    signal_type: SignalType
    title: str = Field(min_length=3, max_length=160)
    description: str = Field(min_length=3, description="The claim, in one sentence, that the quote must support")
    topic: str
    business_unit: str | None = None
    direction: Direction
    event_status: EventStatus
    materiality: Materiality
    materiality_rationale: str = ""
    evidence_quote: str = Field(min_length=1)
    quote_start: int = Field(ge=0, description="Character offset in the document's clean_text")
    quote_end: int = Field(gt=0)
    verification_note: str = Field("", description="The verifier's reason for 'supported'")
    samples_agreeing: int = Field(ge=1, description="Self-consistency: samples that produced this signal")
    samples_total: int = Field(ge=1)
    disputed_fields: list[str] = Field(default_factory=list, description="Fields on which the samples disagreed")
    extractor: str = Field(description="prompt_id@version that produced it")
    model: str


class Signal(SignalIn):
    id: int
    extracted_at: datetime
    document_title: str
    url: str
    publisher: str
    publisher_group: str | None = None
    source_type: SourceType
    source_reliability: SourceReliability
    content_level: ContentLevel
    published_at: datetime | None
    information_credibility: InformationCredibility = 6
    corroboration: list[CorroborationItem] = Field(default_factory=list)
    corroborated_at: datetime | None = None


class SignalFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal_types: list[SignalType] | None = None
    topics: list[str] | None = None
    business_units: list[str] | None = None
    directions: list[Direction] | None = None
    materiality: list[Materiality] | None = None
    information_credibility: list[InformationCredibility] | None = None
    extractor: str | None = None
    published_after: datetime | None = None
    published_before: datetime | None = None

    @field_validator("published_after", "published_before")
    @classmethod
    def _signal_dates_aware(cls, v: datetime | None) -> datetime | None:
        return _require_aware(v)


# ----------------------------------------------------------------------------- recommendations (migration 0002)

RecommendationStatus = Literal["draft", "approved", "retired"]
EvidenceRelation = Literal["supports", "contradicts", "candidate"]
Priority = Literal["High", "Medium", "Low"]

# draft -> approved | retired; approved -> retired. Retired is terminal; a change means a new version.
ALLOWED_TRANSITIONS: dict[str, set[str]] = {"draft": {"approved", "retired"}, "approved": {"retired"}, "retired": set()}


class RecommendationIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=3, max_length=80, pattern=r"^[a-z0-9_]+$")
    version: int = Field(1, ge=1)
    status: RecommendationStatus = "draft"
    title: str = Field(min_length=5)
    business_unit: str | None = None
    topics: list[str] = Field(default_factory=list)
    priority: Priority = "Medium"
    recommendation: str = Field(min_length=10)
    expected_impact: str = ""
    risk_assessment: str = ""
    signal_titles: list[str] = Field(default_factory=list)
    created_by: str = Field(min_length=1, description="e.g. 'seed:v1', 'agent:<run id>', 'user:<name>'")


class Recommendation(RecommendationIn):
    id: int
    created_at: datetime
    updated_at: datetime
    approved_by: str | None = None
    approved_at: datetime | None = None
    status_reason: str | None = None


class EvidenceLinkIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: int
    relation: EvidenceRelation = "candidate"
    score: float | None = None
    linked_by: str = Field(min_length=1, description="e.g. 'auto:retrieval', 'agent:<run>', 'user:<name>'")
    note: str | None = None


class RecommendationEvidence(BaseModel):
    evidence: EvidenceChunk
    relation: EvidenceRelation
    score: float | None
    linked_by: str
    note: str | None


class RecommendationReview(BaseModel):
    recommendation_id: int
    from_status: RecommendationStatus
    to_status: RecommendationStatus
    reviewer: str
    reason: str
    created_at: datetime

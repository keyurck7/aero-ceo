"""Typed state of one briefing run. Every number in the header is a count (ADR 0008)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from aero.domain import EvidenceHit

BriefingStatus = Literal["out_of_scope", "published", "needs_review", "blocked"]


class ScopeDecision(BaseModel):
    in_scope: bool
    intent: str
    reason: str
    sub_questions: list[str]


class ResearchResult(BaseModel):
    sub_question: str
    hits: list[EvidenceHit]
    independent_publishers: int
    rounds: int


class Claim(BaseModel):
    text: str
    kind: Literal["fact", "assessment"]
    citations: list[int]


class ClaimCheck(BaseModel):
    claim: Claim
    status: Literal["supported", "contradicted", "unsupported"]
    supporting: list[int] = Field(default_factory=list)
    contradicting: list[int] = Field(default_factory=list)
    missing_literals: list[str] = Field(default_factory=list, description="Numbers/dates absent from cited text")
    reasons: list[str] = Field(default_factory=list)


class BriefingHeader(BaseModel):
    claims_total: int
    claims_verified: int
    claims_removed: int
    contradicted_claims: int
    independent_publishers: int
    cited_source_grades: dict[str, int]
    thin_coverage: list[str] = Field(description="Sub-questions with evidence from fewer than 2 publishers")


class TraceEvent(BaseModel):
    node: str
    started_at: datetime
    duration_ms: int
    detail: dict[str, Any] = Field(default_factory=dict)


class Briefing(BaseModel):
    question: str
    status: BriefingStatus
    scope: ScopeDecision
    verified_claims: list[ClaimCheck] = Field(default_factory=list)
    removed_claims: list[ClaimCheck] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    evidence: list[EvidenceHit] = Field(default_factory=list, description="Every passage a verified claim cites")
    header: BriefingHeader | None = None
    review_reasons: list[str] = Field(default_factory=list)
    prompts: list[str] = Field(default_factory=list)
    model: str = ""
    trace: list[TraceEvent] = Field(default_factory=list)

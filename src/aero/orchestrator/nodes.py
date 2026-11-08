"""Briefing nodes. Each is a plain function; decisions are categorical rules or validated model judgements.

Replaces the local agent's thresholds (ADR 0008): the cosine scope guard (H13), the 8-14% word-overlap claim
support (H11) and the volume-based confidence (H12).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from functools import cache
from typing import Any, Literal

from pydantic import BaseModel, create_model

from aero.domain import EvidenceHit
from aero.intelligence.verification import Verifier
from aero.llm import LLM, PromptRegistry, default_registry, generate
from aero.orchestrator.state import (
    Briefing,
    BriefingHeader,
    Claim,
    ClaimCheck,
    ResearchResult,
    ScopeDecision,
)
from aero.orchestrator.tools import EvidenceTools

MIN_INDEPENDENT_PUBLISHERS = 2  # editorial two-source rule (policy, plan §4)
MAX_RESEARCH_ROUNDS = 2
MAX_EVIDENCE_PASSAGES = 24  # context budget for the writer

_LITERAL = re.compile(r"(?<![\w.])(?:€|\$|£)?\d(?:[\d,.]*\d)?(?:\s?(?:%|bn\b|billion\b|million\b|m\b))?",
                      re.IGNORECASE)


# ----------------------------------------------------------------------------- plan
@cache
def _scope_model(intents: tuple[str, ...]) -> type[BaseModel]:
    intent_type: Any = Literal[intents]
    return create_model("ScopePlan", in_scope=(bool, ...), intent=(intent_type, ...), reason=(str, ...),
                        sub_questions=(list[str], ...))


def plan(question: str, llm: LLM, intents: Sequence[str], registry: PromptRegistry | None = None) -> ScopeDecision:
    prompt = (registry or default_registry()).get("scope_plan")
    allowed = tuple(dict.fromkeys([*intents, "other"]))
    out, _ = generate(llm, prompt, _scope_model(allowed), {"question": question, "intents": ", ".join(allowed)})
    data = out.model_dump()
    subs = [q.strip() for q in data["sub_questions"] if q.strip()][:4] if data["in_scope"] else []
    if data["in_scope"] and not subs:
        subs = [question]
    return ScopeDecision(in_scope=data["in_scope"], intent=data["intent"], reason=data["reason"], sub_questions=subs)


# ----------------------------------------------------------------------------- research
def _groups(hits: Sequence[EvidenceHit]) -> set[str]:
    return {(h.publisher_group or h.publisher).lower() for h in hits}


def research(sub_questions: Sequence[str], tools: EvidenceTools, top_k: int = 6) -> list[ResearchResult]:
    """One search per sub-question; a second, deeper round when fewer than 2 publishers answered."""
    out = []
    for q in sub_questions:
        result = tools.search(q, top_k)
        hits, rounds = list(result.hits), 1
        if len(_groups(hits)) < MIN_INDEPENDENT_PUBLISHERS and rounds < MAX_RESEARCH_ROUNDS:
            deeper = tools.search(q, top_k * 2)
            seen = {h.chunk_id for h in hits}
            hits += [h for h in deeper.hits if h.chunk_id not in seen]
            rounds += 1
        out.append(ResearchResult(sub_question=q, hits=hits, independent_publishers=len(_groups(hits)),
                                  rounds=rounds))
    return out


def pooled_evidence(results: Sequence[ResearchResult]) -> list[EvidenceHit]:
    """Distinct passages, interleaved across sub-questions in rank order, capped for the writer's context."""
    pooled: dict[int, EvidenceHit] = {}
    depth = max((len(r.hits) for r in results), default=0)
    for i in range(depth):
        for r in results:
            if i < len(r.hits) and r.hits[i].chunk_id not in pooled and len(pooled) < MAX_EVIDENCE_PASSAGES:
                pooled[r.hits[i].chunk_id] = r.hits[i]
    return list(pooled.values())


# ----------------------------------------------------------------------------- write
class _Draft(BaseModel):
    claims: list[Claim]
    open_questions: list[str]


def write(question: str, evidence: Sequence[EvidenceHit], llm: LLM,
          registry: PromptRegistry | None = None) -> tuple[list[Claim], list[str]]:
    prompt = (registry or default_registry()).get("brief_write")
    rendered = "\n".join(
        f'<passage id="C{h.chunk_id}" publisher="{h.publisher}" grade="{h.source_reliability}" '
        f'date="{h.published_at.date().isoformat() if h.published_at else "undated"}" '
        f'content="{h.content_level}">{h.text}</passage>' for h in evidence)
    out, _ = generate(llm, prompt, _Draft, {"question": question, "evidence": rendered or "(no evidence found)"})
    return out.claims, out.open_questions


# ----------------------------------------------------------------------------- verify
def literals(text: str) -> list[str]:
    return [m.group(0).strip() for m in _LITERAL.finditer(text)]


def _norm(s: str) -> str:
    return re.sub(r"[\s,]", "", s.lower())


def verify_claims(claims: Sequence[Claim], evidence: dict[int, EvidenceHit], verifier: Verifier) -> list[ClaimCheck]:
    """Every claim against every passage it cites. Fail closed: no valid citation means unsupported."""
    checks = []
    for claim in claims:
        cited = [evidence[c] for c in dict.fromkeys(claim.citations) if c in evidence]
        check = ClaimCheck(claim=claim, status="unsupported")
        unknown = [c for c in claim.citations if c not in evidence]
        if unknown:
            check.reasons.append(f"cites passages that were not provided: {unknown}")
        if not cited:
            check.reasons.append("no valid citation")
            checks.append(check)
            continue
        joined = _norm(" ".join(h.text for h in cited))
        check.missing_literals = [lit for lit in literals(claim.text) if _norm(lit) not in joined]
        for hit in cited:
            v = verifier.verify(claim.text, hit)
            if v.verdict == "supported":
                check.supporting.append(hit.chunk_id)
            elif v.verdict == "contradicted":
                check.contradicting.append(hit.chunk_id)
            check.reasons.append(f"C{hit.chunk_id}: {v.verdict}: {v.reason}")
        if check.missing_literals:
            check.status = "unsupported"
            check.reasons.append(f"numbers/dates not in the cited passages: {check.missing_literals}")
        elif check.supporting:
            check.status = "supported"
        elif check.contradicting:
            check.status = "contradicted"
        checks.append(check)
    return checks


# ----------------------------------------------------------------------------- gate
def gate(question: str, scope: ScopeDecision, results: Sequence[ResearchResult], checks: Sequence[ClaimCheck],
         evidence: dict[int, EvidenceHit], open_questions: Sequence[str]) -> Briefing:
    """Categorical publication rules (no score):

    * unsupported or contradicted claims never reach the briefing; they are listed as removed;
    * no verified claim at all -> ``blocked``;
    * any contradicted claim, or a sub-question answered by fewer than 2 publishers -> ``needs_review``;
    * otherwise ``published``.
    """
    verified = [c for c in checks if c.status == "supported"]
    removed = [c for c in checks if c.status != "supported"]
    cited_ids = list(dict.fromkeys(i for c in verified for i in c.supporting))
    cited = [evidence[i] for i in cited_ids]
    thin = [r.sub_question for r in results if r.independent_publishers < MIN_INDEPENDENT_PUBLISHERS]
    contradicted = [c for c in checks if c.status == "contradicted"]
    reasons = []
    if contradicted:
        reasons.append(f"{len(contradicted)} claim(s) contradicted by their own cited passages")
    if thin:
        reasons.append(f"{len(thin)} sub-question(s) answered by fewer than {MIN_INDEPENDENT_PUBLISHERS} publishers")
    status = "blocked" if not verified else "needs_review" if reasons else "published"
    if not verified:
        reasons.insert(0, "no claim survived verification")
    header = BriefingHeader(
        claims_total=len(checks), claims_verified=len(verified), claims_removed=len(removed),
        contradicted_claims=len(contradicted), independent_publishers=len(_groups(cited)),
        cited_source_grades=dict(sorted(Counter(h.source_reliability for h in cited).items())), thin_coverage=thin)
    return Briefing(question=question, status=status, scope=scope, verified_claims=verified, removed_claims=removed,
                    open_questions=list(open_questions), evidence=cited, header=header, review_reasons=reasons)

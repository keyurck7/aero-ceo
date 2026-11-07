"""Evidence credibility (replaces ``app/intelligence/scoring.py``).

v1 mixed source trust, recency and a per-rule constant called "evidence_strength"
into one number, then averaged those into the answer's "confidence" (finding B7).
v2 keeps the factors separate and explainable:

* ``evidence_credibility``: how far to trust *this passage as a source*
  (publisher prior, how much of the content we hold, injection flags).
* ``dates.recency_weight``: how fresh it is.
* Answer-level confidence is computed later by the verifier from claim support.
  It is deliberately absent here.
"""

from __future__ import annotations

from collections.abc import Iterable

from aero.domain import ContentLevel

CONTENT_LEVEL_FACTOR: dict[str, float] = {
    "full_text": 1.0,
    "page": 0.95,
    # A headline + feed snippet supports "X was reported" much better than any detail.
    "headline_summary": 0.8,
}
INJECTION_PENALTY = 0.5


def evidence_credibility(prior: float, content_level: ContentLevel, injection_flags: Iterable[str] = ()) -> float:
    score = prior * CONTENT_LEVEL_FACTOR.get(content_level, 0.9)
    if any(True for _ in injection_flags):
        score *= INJECTION_PENALTY
    return round(min(max(score, 0.0), 1.0), 4)


def blend_final(relevance: float, credibility: float, recency: float, w_cred: float, w_rec: float) -> float:
    """Final rank score. ``relevance`` must already be in [0, 1]."""
    w_rel = 1.0 - w_cred - w_rec
    if w_rel <= 0:
        raise ValueError("credibility + recency weights must leave room for relevance")
    return round(w_rel * relevance + w_cred * credibility + w_rec * recency, 6)

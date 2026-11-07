"""Hybrid evidence retrieval (replaces ``app/retrieval/search_engine.py``).

Pipeline per query:

1. Dense top-N (embedding similarity) and lexical top-N (BM25 / ts_rank). Both
   use the *same* filters inside the database query (finding B6).
2. Reciprocal Rank Fusion: score-scale-free, robust, no tuning needed.
3. Optional cross-encoder rerank of the fused candidates.
4. Final score = relevance blended with source credibility and recency. Every
   factor is returned in ``ScoreBreakdown``, so nothing is hidden.

Also reports ``independent_publishers``: syndicated copies count once. The
orchestrator's coverage check ("≥ 2 independent publishers") needs this number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aero.config import Settings
from aero.dates import recency_weight, utcnow
from aero.domain import EvidenceHit, ScoreBreakdown, SearchFilters, SearchResult
from aero.embeddings import Embedder
from aero.rerank import Reranker
from aero.scoring import blend_final, evidence_credibility
from aero.storage import Store


@dataclass(frozen=True)
class RetrievalConfig:
    candidate_k: int = 50
    rrf_k: int = 60
    credibility_weight: float = 0.10
    recency_weight: float = 0.05
    recency_half_life_days: float = 180.0

    @classmethod
    def from_settings(cls, s: Settings) -> RetrievalConfig:
        return cls(
            candidate_k=s.retrieval_candidate_k,
            rrf_k=s.retrieval_rrf_k,
            credibility_weight=s.retrieval_credibility_weight,
            recency_weight=s.retrieval_recency_weight,
            recency_half_life_days=s.recency_half_life_days,
        )


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion (Cormack et al., 2009): sum of 1 / (k + rank)."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return fused


def independent_publishers(hits: list[EvidenceHit]) -> int:
    return len({(h.publisher_group or h.publisher).lower() for h in hits})


class HybridRetriever:
    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        reranker: Reranker | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.config = config or RetrievalConfig()

    def search(self, query: str, top_k: int = 8, filters: SearchFilters | None = None) -> SearchResult:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        filters = filters or SearchFilters()
        cfg = self.config
        n = max(cfg.candidate_k, top_k)

        qvec = self.embedder.embed_query(query)
        dense = self.store.dense_search(qvec, n, filters, self.embedder.model_name)
        lexical = self.store.lexical_search(query, n, filters)

        dense_rank = {cid: (i + 1, s) for i, (cid, s) in enumerate(dense)}
        lex_rank = {cid: (i + 1, s) for i, (cid, s) in enumerate(lexical)}
        fused = rrf_fuse([[c for c, _ in dense], [c for c, _ in lexical]], k=cfg.rrf_k)
        if not fused:
            return SearchResult(query=query, filters=filters, hits=[], independent_publishers=0,
                                note="No evidence matched the query and filters.")

        candidates = sorted(fused, key=lambda c: (-fused[c], c))[:n]
        evidence = self.store.get_evidence(candidates)
        candidates = [c for c in candidates if c in evidence]

        rerank_scores: dict[int, float] = {}
        if self.reranker is not None and candidates:
            scores = self.reranker.score(query, [evidence[c].text for c in candidates])
            rerank_scores = dict(zip(candidates, scores, strict=True))

        # Relevance in [0, 1]: rerank probability if available, else RRF normalised to the best candidate.
        best_rrf = max(fused[c] for c in candidates) if candidates else 1.0
        now = filters.as_of or utcnow()
        hits: list[EvidenceHit] = []
        for cid in candidates:
            ev = evidence[cid]
            relevance = rerank_scores[cid] if rerank_scores else fused[cid] / best_rrf
            cred = evidence_credibility(ev.credibility_prior, ev.content_level, ev.injection_flags)
            rec = recency_weight(ev.published_at, now=now, half_life_days=cfg.recency_half_life_days)
            final = blend_final(relevance, cred, rec, cfg.credibility_weight, cfg.recency_weight)
            d, lx = dense_rank.get(cid), lex_rank.get(cid)
            hits.append(EvidenceHit(
                **ev.model_dump(), rank=0,
                scores=ScoreBreakdown(
                    dense=round(d[1], 6) if d else None, dense_rank=d[0] if d else None,
                    lexical=round(lx[1], 6) if lx else None, lexical_rank=lx[0] if lx else None,
                    rrf=round(fused[cid], 6), rerank=round(rerank_scores[cid], 6) if rerank_scores else None,
                    credibility=cred, recency=round(rec, 4), final=final,
                ),
            ))
        hits.sort(key=lambda h: (-h.scores.final, h.chunk_id))
        hits = hits[:top_k]
        for i, h in enumerate(hits, start=1):
            h.rank = i
        return SearchResult(
            query=query, filters=filters, hits=hits, independent_publishers=independent_publishers(hits)
        )

    def find_related(self, chunk_id: int, top_k: int = 5, other_publishers_only: bool = True) -> list[EvidenceHit]:
        """Corroboration: passages semantically closest to ``chunk_id``, from other documents/publishers."""
        anchor = self.store.get_evidence([chunk_id]).get(chunk_id)
        if anchor is None:
            raise KeyError(f"chunk {chunk_id} not found")
        stored = self.store.get_embedding(chunk_id)
        if stored is not None and stored[1] == self.embedder.model_name:
            vec = stored[0]
        else:
            vec = self.embedder.embed_documents([anchor.text])[0]
        filters = SearchFilters(exclude_document_ids=[anchor.document_id])
        raw = self.store.dense_search(np.asarray(vec), max(top_k * 4, 20), filters, self.embedder.model_name)
        evidence = self.store.get_evidence([c for c, _ in raw])
        anchor_group = (anchor.publisher_group or anchor.publisher).lower()
        out: list[EvidenceHit] = []
        for cid, score in raw:
            ev = evidence.get(cid)
            if ev is None:
                continue
            if other_publishers_only and (ev.publisher_group or ev.publisher).lower() == anchor_group:
                continue
            cred = evidence_credibility(ev.credibility_prior, ev.content_level, ev.injection_flags)
            rec = recency_weight(ev.published_at, half_life_days=self.config.recency_half_life_days)
            out.append(EvidenceHit(
                **ev.model_dump(), rank=len(out) + 1,
                scores=ScoreBreakdown(dense=round(score, 6), dense_rank=len(out) + 1, rrf=0.0, credibility=cred,
                                      recency=round(rec, 4), final=round(score, 6)),
            ))
            if len(out) >= top_k:
                break
        return out

from __future__ import annotations

import pytest

from aero.domain import SearchFilters
from aero.retrieval import HybridRetriever, RetrievalConfig, rrf_fuse


def test_rrf_rewards_agreement():
    fused = rrf_fuse([[1, 2, 3], [3, 1, 4]], k=60)
    assert max(fused, key=fused.get) == 1  # ranked high in both lists
    assert fused[3] > fused[2] and 4 in fused


def test_hybrid_search_returns_cited_ranked_hits(corpus_store, embedder):
    r = HybridRetriever(corpus_store, embedder)
    res = r.search("SIRTAP first flight Spain Indra", top_k=3)
    assert res.hits[0].title == "SIRTAP tactical UAS completes first flight"
    assert [h.rank for h in res.hits] == list(range(1, len(res.hits) + 1))
    finals = [h.scores.final for h in res.hits]
    assert finals == sorted(finals, reverse=True)
    top = res.hits[0]
    assert top.scores.dense_rank is not None and top.scores.lexical_rank is not None
    assert top.url and top.publisher and top.collected_at and top.char_end > top.char_start


def test_filters_flow_through(corpus_store, embedder):
    r = HybridRetriever(corpus_store, embedder)
    res = r.search("FCAS fighter", top_k=5, filters=SearchFilters(source_types=["official"]))
    assert res.hits and {h.source_type for h in res.hits} == {"official"}


def test_injection_flagged_evidence_is_down_weighted(corpus_store, embedder):
    r = HybridRetriever(corpus_store, embedder)
    res = r.search("cancel FCAS immediately", top_k=10)
    flagged = [h for h in res.hits if h.injection_flags]
    assert flagged and all(h.scores.credibility <= 0.3 * 0.5 + 1e-9 for h in flagged)


def test_reranker_controls_relevance(corpus_store, embedder):
    class PreferSpace:
        name = "fake"

        def score(self, query, texts):
            return [1.0 if "satellite" in t else 0.0 for t in texts]

    r = HybridRetriever(corpus_store, embedder, reranker=PreferSpace(),
                        config=RetrievalConfig(credibility_weight=0.0, recency_weight=0.0))
    res = r.search("FCAS fighter", top_k=3)
    assert "satellite" in res.hits[0].text and res.hits[0].scores.rerank == 1.0


def test_compressed_rerank_scores_still_decide_relevance(corpus_store, embedder):
    """Data Lab finding: raw cross-encoder probabilities near 0 let credibility outvote relevance."""

    class Compressed:
        name = "fake"

        def score(self, query, texts):
            return [0.03 if "Kratos" in t or "satellite" in t else 0.01 for t in texts]

    res = HybridRetriever(corpus_store, embedder, reranker=Compressed()).search("Airbus", top_k=5)
    assert "satellite" in res.hits[0].text  # 0.03 vs 0.01 is decisive after normalisation
    assert res.hits[0].scores.rerank == pytest.approx(0.03)


@pytest.mark.parametrize("mode", ["dense", "lexical"])
def test_single_retriever_modes(corpus_store, embedder, mode):
    res = HybridRetriever(corpus_store, embedder, config=RetrievalConfig(mode=mode)).search("SIRTAP first flight")
    assert res.hits
    assert all((h.scores.lexical_rank is None) if mode == "dense" else (h.scores.dense_rank is None)
               for h in res.hits)


def test_credibility_floor_filters_low_trust_publishers(corpus_store, embedder):
    q = "FCAS fighter programme partners"
    default = HybridRetriever(corpus_store, embedder).search(q, top_k=10)
    assert "Unknown Blog" in {h.publisher for h in default.hits}  # prior 0.3 > default floor 0.15
    strict = HybridRetriever(corpus_store, embedder, config=RetrievalConfig(min_credibility=0.5)).search(q, top_k=10)
    assert strict.hits and "Unknown Blog" not in {h.publisher for h in strict.hits}
    assert strict.filters.min_credibility == 0.5


def test_independent_publisher_count(corpus_store, embedder):
    res = HybridRetriever(corpus_store, embedder).search("Airbus", top_k=10)
    assert res.independent_publishers == len({h.publisher_group for h in res.hits})


def test_empty_results_and_validation(corpus_store, embedder):
    r = HybridRetriever(corpus_store, embedder)
    res = r.search("zzzz", top_k=3, filters=SearchFilters(topics=["Nonexistent"]))
    assert res.hits == [] and res.note
    with pytest.raises(ValueError):
        r.search("   ")


def test_find_related_excludes_same_document_and_publisher(corpus_store, embedder):
    r = HybridRetriever(corpus_store, embedder)
    anchor = r.search("Dassault Airbus FCAS leadership dispute", top_k=1).hits[0]
    related = r.find_related(anchor.chunk_id, top_k=3)
    assert related and all(h.document_id != anchor.document_id for h in related)
    assert all(h.publisher_group != anchor.publisher_group for h in related)
    with pytest.raises(KeyError):
        r.find_related(999_999)

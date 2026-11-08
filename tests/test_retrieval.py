from __future__ import annotations

import pytest

from aero.domain import SearchFilters, SourceIn
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


def test_injection_flagged_evidence_is_quarantined_not_down_weighted(corpus_store, embedder):
    r = HybridRetriever(corpus_store, embedder)
    assert not [h for h in r.search("cancel FCAS immediately", top_k=10).hits if h.injection_flags]
    audit = r.search("cancel FCAS immediately", top_k=10, filters=SearchFilters(include_quarantined=True))
    assert [h for h in audit.hits if h.injection_flags]


def test_reranker_controls_relevance(corpus_store, embedder):
    class PreferSpace:
        name = "fake"

        def score(self, query, texts):
            return [1.0 if "satellite" in t else 0.0 for t in texts]

    r = HybridRetriever(corpus_store, embedder, reranker=PreferSpace())
    res = r.search("FCAS fighter", top_k=3)
    assert "satellite" in res.hits[0].text and res.hits[0].scores.rerank == 1.0


def test_compressed_rerank_scores_still_decide_relevance(corpus_store, embedder):
    """Compressed cross-encoder probabilities near 0 still order the hits: nothing else is in the score."""

    class Compressed:
        name = "fake"

        def score(self, query, texts):
            return [0.03 if "Kratos" in t or "satellite" in t else 0.01 for t in texts]

    res = HybridRetriever(corpus_store, embedder, reranker=Compressed()).search("Airbus", top_k=5)
    assert "satellite" in res.hits[0].text
    assert res.hits[0].scores.rerank == pytest.approx(0.03) == res.hits[0].scores.final


@pytest.mark.parametrize("mode", ["dense", "lexical"])
def test_single_retriever_modes(corpus_store, embedder, mode):
    res = HybridRetriever(corpus_store, embedder, config=RetrievalConfig(mode=mode)).search("SIRTAP first flight")
    assert res.hits
    assert all((h.scores.lexical_rank is None) if mode == "dense" else (h.scores.dense_rank is None)
               for h in res.hits)


def test_grade_e_excluded_by_policy_and_never_weighted(corpus_store, embedder):
    q = "FCAS fighter leadership dispute"
    r = HybridRetriever(corpus_store, embedder)
    before = r.search(q, top_k=10)
    assert "Reuters" in {h.publisher for h in before.hits}
    assert before.filters.source_reliability == ["A", "B", "C", "D", "F"]
    assert all(h.scores.final == (h.scores.rerank if h.scores.rerank is not None else h.scores.rrf)
               for h in before.hits)  # relevance only
    corpus_store.upsert_source(SourceIn(name="Reuters", source_type="news", category="off_topic_or_spam",
                                        publisher_group="reuters"))
    assert "Reuters" not in {h.publisher for h in r.search(q, top_k=10).hits}
    explicit = r.search(q, top_k=10, filters=SearchFilters(source_reliability=["E"]))
    assert {h.publisher for h in explicit.hits} == {"Reuters"} and explicit.hits[0].source_reliability == "E"
    stricter = HybridRetriever(corpus_store, embedder, config=RetrievalConfig(exclude_reliability=("D", "E", "F")))
    assert stricter.search(q).filters.source_reliability == ["A", "B", "C"]


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

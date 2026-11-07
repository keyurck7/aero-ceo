"""Contract tests every Store adapter must pass (SQLite always; Postgres when configured)."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from aero.domain import ChunkIn, DocumentIn, SearchFilters, SourceIn
from aero.storage.base import DuplicateDocumentError
from aero.text import canonical_url, clean_text, content_hash, simhash64
from tests.conftest import CORPUS, DIM, NOW, make_document

LONG_ARTICLE = "\n\n".join(
    f"Paragraph {i}: Airbus, Dassault and Indra negotiators met again in Paris about FCAS pillar {i}, "
    f"discussing workshare item {i * 7}, flight demonstrator milestone {i + 2029} and the combat cloud "
    f"architecture that partners want to define before the next phase decision number {i * 3}."
    for i in range(12)
)


def test_migrate_is_idempotent(store):
    assert store.migrate() == []


def test_source_upsert_updates_in_place(store):
    a = store.upsert_source(SourceIn(name="Reuters", source_type="news", credibility_prior=0.8))
    b = store.upsert_source(SourceIn(name="Reuters", source_type="news", credibility_prior=0.9, licence="licensed"))
    assert a.id == b.id and b.credibility_prior == pytest.approx(0.9) and b.licence == "licensed"
    assert [s.name for s in store.list_sources("news")] == ["Reuters"]
    assert store.get_source(a.id).name == "Reuters"


def test_document_roundtrip_and_ingestion_context_kept_out_of_text(store, taxonomy):
    doc_id = make_document(store, CORPUS[0], taxonomy)
    doc = store.get_document(doc_id)
    assert doc.canonical_url == canonical_url(CORPUS[0]["url"])
    assert doc.published_at == CORPUS[0]["published"]
    assert doc.ingestion_context == {"collector": "test"}
    assert "collector" not in doc.clean_text
    assert "FCAS" in doc.topics
    assert doc.simhash == simhash64(doc.clean_text)


@pytest.mark.parametrize("variant", ["canonical_url", "content_hash", "near_duplicate"])
def test_duplicates_are_rejected(store, taxonomy, variant):
    article = dict(CORPUS[0], text=LONG_ARTICLE)  # article-length: SimHash is calibrated for real articles
    make_document(store, article, taxonomy)
    src = store.list_sources()[0]
    text = clean_text(LONG_ARTICLE)
    url = "https://other.example/syndicated-copy"
    if variant == "canonical_url":
        url = CORPUS[0]["url"].replace("utm_source=rss", "utm_medium=x")
        text = "completely different words here for hashing purposes only"
    elif variant == "near_duplicate":
        text = text + "\n\nReporting by Reuters staff; editing by Jane Doe."
    with pytest.raises(DuplicateDocumentError) as err:
        store.insert_document(DocumentIn(
            source_id=src.id, title="copy", url=url, canonical_url=canonical_url(url), clean_text=text,
            content_hash=content_hash(text), simhash=simhash64(text),
        ))
    assert err.value.reason == variant


def test_replace_chunks_and_embedding_lifecycle(store, taxonomy, embedder):
    doc_id = make_document(store, CORPUS[1], taxonomy)
    pending = store.chunks_needing_embedding(embedder.model_name)
    assert pending and all(isinstance(t, str) for _, t in pending)
    vecs = embedder.embed_documents([t for _, t in pending])
    store.set_embeddings([(cid, vecs[i]) for i, (cid, _) in enumerate(pending)], embedder.model_name)
    assert store.chunks_needing_embedding(embedder.model_name) == []
    # A different model means everything needs re-embedding (safe model upgrades).
    assert len(store.chunks_needing_embedding("other-model")) == len(pending)
    vec, model = store.get_embedding(pending[0][0])
    assert model == embedder.model_name and np.allclose(vec, vecs[0], atol=1e-6)
    with pytest.raises(ValueError, match="dim"):
        store.set_embeddings([(pending[0][0], np.ones(DIM + 1))], embedder.model_name)
    # Re-chunking clears embeddings for that document.
    store.replace_chunks(doc_id, [ChunkIn(chunk_index=0, text="new text", char_start=0, char_end=8, word_count=2)])
    assert [t for _, t in store.chunks_needing_embedding(embedder.model_name)] == ["new text"]


def test_dense_and_lexical_search(corpus_store, embedder):
    f = SearchFilters()
    dense = corpus_store.dense_search(embedder.embed_query("SIRTAP uncrewed first flight Spain"), 3, f,
                                      embedder.model_name)
    lexical = corpus_store.lexical_search("SIRTAP first flight", 3, f)
    ev = corpus_store.get_evidence([dense[0][0], lexical[0][0]])
    assert "SIRTAP" in ev[dense[0][0]].text and "SIRTAP" in ev[lexical[0][0]].text
    assert dense == sorted(dense, key=lambda x: -x[1])
    assert corpus_store.lexical_search("the and of", 3, f) == []  # stopwords only
    assert corpus_store.lexical_search('"); DROP TABLE chunks; --', 3, f) == []  # sanitised


@pytest.mark.parametrize(
    "filters, allowed",
    [
        (SearchFilters(topics=["Military Space"]), {"Airbus wins military satellite communications contract"}),
        (SearchFilters(source_types=["official"]), {"SIRTAP tactical UAS completes first flight"}),
        (SearchFilters(min_credibility=0.9), {"Dassault and Airbus clash over FCAS leadership",
                                              "SIRTAP tactical UAS completes first flight"}),
        (SearchFilters(published_after=NOW - timedelta(days=30)), {"Dassault and Airbus clash over FCAS leadership"}),
        (SearchFilters(published_before=NOW - timedelta(days=365)),
         {"Eurofighter electronic warfare upgrade moves ahead"}),
    ],
)
def test_filters_apply_before_ranking(corpus_store, embedder, filters, allowed):
    """v1 filtered after taking the top-k*5 neighbours (finding B6). Here the filter is inside the query:
    dense search returns every allowed chunk (never an empty page), lexical only allowed matches."""
    q = "Airbus FCAS fighter satellite drone"
    dense = corpus_store.dense_search(embedder.embed_query(q), 50, filters, embedder.model_name)
    dense_titles = {e.title for e in corpus_store.get_evidence([c for c, _ in dense]).values()}
    assert dense_titles == allowed
    lexical = corpus_store.lexical_search(q, 50, filters)
    assert {e.title for e in corpus_store.get_evidence([c for c, _ in lexical]).values()} <= allowed


def test_as_of_and_exclusion_filters(corpus_store, embedder):
    q = embedder.embed_query("FCAS")
    assert corpus_store.dense_search(q, 10, SearchFilters(as_of=NOW - timedelta(days=3650)), embedder.model_name) == []
    first = corpus_store.dense_search(q, 1, SearchFilters(), embedder.model_name)[0][0]
    doc_id = corpus_store.get_evidence([first])[first].document_id
    rest = corpus_store.dense_search(q, 50, SearchFilters(exclude_document_ids=[doc_id]), embedder.model_name)
    assert doc_id not in {e.document_id for e in corpus_store.get_evidence([c for c, _ in rest]).values()}


def test_evidence_carries_citation_metadata(corpus_store):
    hits = corpus_store.lexical_search("SIRTAP", 1, SearchFilters())
    ev = corpus_store.get_evidence([hits[0][0]])[hits[0][0]]
    doc = corpus_store.get_document(ev.document_id)
    assert doc.clean_text[ev.char_start : ev.char_end] == ev.text
    assert ev.publisher == "Airbus Newsroom" and ev.source_type == "official" and ev.licence == "unknown"
    assert ev.published_at is not None and ev.collected_at is not None and ev.topics


def test_neighbours(corpus_store):
    hit = corpus_store.lexical_search("workshare dispute", 1, SearchFilters())[0][0]
    around = corpus_store.get_neighbours(hit, window=1)
    assert hit in {e.chunk_id for e in around}
    assert [e.chunk_index for e in around] == sorted(e.chunk_index for e in around)


def test_injection_flags_survive(corpus_store):
    hit = corpus_store.lexical_search("cancel FCAS immediately", 1, SearchFilters())[0][0]
    assert "ignore_instructions" in corpus_store.get_evidence([hit])[hit].injection_flags


def test_runs_and_stats(corpus_store):
    run = corpus_store.start_run("test", {"a": 1})
    corpus_store.finish_run(run, "succeeded", {"inserted": 5})
    st = corpus_store.stats()
    assert st.documents == len(CORPUS) and st.sources == len(CORPUS)
    assert st.embedded_chunks == st.chunks > 0 and st.embedding_models == [f"hashing-{DIM}"]

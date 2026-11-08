"""Signals, recommendations and partners: services and store contract (SQLite always, Postgres when configured)."""

from __future__ import annotations

from datetime import timedelta

import pytest

from aero.domain import EvidenceLinkIn, RecommendationIn, SearchFilters, SignalFilters
from aero.intelligence.partners import PartnerRegistry
from aero.intelligence.signals import (
    RuleSignalExtractor,
    load_signal_rules,
    run_extraction,
    sentence_span,
    signal_timeseries,
    top_movers,
)
from aero.intelligence.strategy import (
    ApprovalBlockedError,
    business_unit_profile,
    change_status,
    load_drilldown_prompts,
    load_recommendation_seeds,
    load_risk_checklist,
    propose,
    render_drilldown,
)
from aero.storage.base import InvalidTransitionError
from tests.conftest import NOW, ROOT

SEEDS = ROOT / "seeds"


@pytest.fixture(scope="module")
def extractor(taxonomy):
    return RuleSignalExtractor(load_signal_rules(SEEDS / "signal_rules_v1.yaml"), taxonomy)


@pytest.fixture
def signal_store(corpus_store, extractor):
    run_extraction(corpus_store, extractor)
    return corpus_store


def _doc_id(store, title_part):
    return next(i for i in store.list_document_ids() if title_part in store.get_document(i).title)


# ----------------------------------------------------------------- signal extraction
def test_sentence_span():
    text = "First sentence. The SIRTAP drone flew today! Third one"
    s, e = sentence_span(text, text.index("SIRTAP"))
    assert text[s:e] == "The SIRTAP drone flew today!"


def test_signals_are_grounded_in_exact_quotes(signal_store):
    signals = signal_store.list_signals(SignalFilters(), limit=500)
    assert signals
    for s in signals:
        doc = signal_store.get_document(s.document_id)
        assert doc.clean_text[s.quote_start : s.quote_end] == s.evidence_quote
        chunk = signal_store.get_evidence([s.chunk_id])[s.chunk_id]
        assert s.evidence_quote in chunk.text
    keys = [(s.document_id, s.title) for s in signals]
    assert len(keys) == len(set(keys))  # at most one signal per (document, rule)


def test_risk_rule_fires_on_dispute_text_with_business_unit(signal_store):
    fcas = _doc_id(signal_store, "clash over FCAS")
    risks = signal_store.list_signals(SignalFilters(signal_types=["risk"]), limit=100)
    fcas_risk = [s for s in risks if s.document_id == fcas and "fragmentation" in s.title]
    assert fcas_risk and fcas_risk[0].business_unit == "Future Combat / FCAS"
    assert fcas_risk[0].publisher == "Reuters" and fcas_risk[0].topic == "FCAS"


def test_company_name_does_not_create_space_signals(signal_store):
    """'Airbus Defence and Space' appears in the FCAS and SIRTAP texts; only the satellite article is about space."""
    space = signal_store.list_signals(SignalFilters(topics=["Military Space"]), limit=100)
    assert {s.document_title for s in space} == {"Airbus wins military satellite communications contract"}


def test_extraction_is_idempotent(signal_store, extractor):
    before = len(signal_store.list_signals(SignalFilters(), limit=500))
    stats = run_extraction(signal_store, extractor)
    assert stats["signals"] == before == len(signal_store.list_signals(SignalFilters(), limit=500))


def test_signal_filters_and_lookup(signal_store):
    uncrewed = signal_store.list_signals(SignalFilters(business_units=["Uncrewed Systems"]), limit=50)
    assert uncrewed and all(s.business_unit == "Uncrewed Systems" for s in uncrewed)
    recent = signal_store.list_signals(SignalFilters(published_after=NOW - timedelta(days=30)), limit=50)
    assert recent and all(s.published_at >= NOW - timedelta(days=30) for s in recent)
    one = signal_store.get_signal(uncrewed[0].id)
    assert one == uncrewed[0] and signal_store.get_signal(999_999) is None
    impacts = [s.impact for s in signal_store.list_signals(SignalFilters(), limit=500)]
    assert impacts == sorted(impacts, reverse=True)


def test_timeseries_and_movers(signal_store):
    signals = signal_store.list_signals(SignalFilters(), limit=500)
    points = signal_timeseries(signals, "month")
    assert sum(p.count for p in points) == sum(1 for s in signals if s.published_at)
    movers = top_movers(signals, window_days=90, now=NOW)
    assert movers and movers[0].change != 0


# ----------------------------------------------------------------- recommendations: store contract
def _rec(key="test_rec", **kw):
    base = dict(key=key, title="A test recommendation", recommendation="Do something sensible", created_by="test")
    return RecommendationIn(**{**base, **kw})


def test_recommendation_upsert_only_updates_drafts(store):
    r = store.upsert_recommendation(_rec(topics=["FCAS"], signal_titles=["x"]))
    assert r.status == "draft" and r.topics == ["FCAS"] and r.signal_titles == ["x"]
    r2 = store.upsert_recommendation(_rec(title="Edited title text"))
    assert r2.id == r.id and r2.title == "Edited title text"
    store.set_recommendation_status(r.id, "retired", "alice", "obsolete")
    r3 = store.upsert_recommendation(_rec(title="Should not change"))
    assert r3.title == "Edited title text" and r3.status == "retired"


def test_recommendation_lifecycle_and_audit(store):
    r = store.upsert_recommendation(_rec())
    with pytest.raises(ValueError):
        store.set_recommendation_status(r.id, "approved", "", "no reviewer")
    approved = store.set_recommendation_status(r.id, "approved", "alice", "reviewed evidence")
    assert approved.approved_by == "alice" and approved.approved_at is not None
    with pytest.raises(InvalidTransitionError):
        store.set_recommendation_status(r.id, "draft", "alice", "back to draft")
    store.set_recommendation_status(r.id, "retired", "bob", "superseded")
    with pytest.raises(InvalidTransitionError, match="terminal"):
        store.set_recommendation_status(r.id, "approved", "bob", "again")
    reviews = store.recommendation_reviews(r.id)
    assert [(x.from_status, x.to_status, x.reviewer) for x in reviews] == [
        ("draft", "approved", "alice"), ("approved", "retired", "bob")]
    with pytest.raises(KeyError):
        store.set_recommendation_status(999_999, "approved", "x", "y")


def test_recommendation_listing(store):
    store.upsert_recommendation(_rec("low_one", priority="Low", topics=["Eurofighter"]))
    store.upsert_recommendation(_rec("high_one", priority="High", topics=["FCAS", "Military Space"],
                                     business_unit="Future Combat / FCAS"))
    assert [r.key for r in store.list_recommendations()] == ["high_one", "low_one"]
    assert [r.key for r in store.list_recommendations(topic="Military Space")] == ["high_one"]
    assert [r.key for r in store.list_recommendations(business_unit="Future Combat / FCAS")] == ["high_one"]
    assert store.list_recommendations(status="approved") == []


def test_evidence_links_never_downgraded_by_auto_linkers(corpus_store):
    r = corpus_store.upsert_recommendation(_rec())
    chunk = corpus_store.lexical_search("SIRTAP", 1, SearchFilters())[0][0]
    corpus_store.link_recommendation_evidence(r.id, [EvidenceLinkIn(chunk_id=chunk, relation="supports",
                                                                    linked_by="user:alice")], overwrite=True)
    corpus_store.link_recommendation_evidence(r.id, [EvidenceLinkIn(chunk_id=chunk, linked_by="auto:retrieval")])
    (ev,) = corpus_store.recommendation_evidence(r.id)
    assert ev.relation == "supports" and ev.linked_by == "user:alice" and ev.evidence.chunk_id == chunk


# ----------------------------------------------------------------- strategy services
def test_seed_blueprints_load_as_drafts_with_topics(taxonomy):
    seeds = load_recommendation_seeds(SEEDS / "recommendations_v1.yaml", taxonomy, SEEDS / "signal_rules_v1.yaml")
    assert len(seeds) == 6 and {s.status for s in seeds} == {"draft"} and all(s.topics for s in seeds)
    assert {s.created_by for s in seeds} == {"seed:v1"}


def test_propose_versions_and_validates_evidence(corpus_store):
    hits = [c for c, _ in corpus_store.lexical_search("FCAS", 3, SearchFilters())]
    with pytest.raises(ValueError, match="at least one"):
        propose(corpus_store, _rec("prop"), [], "agent:t1")
    with pytest.raises(ValueError, match="unknown evidence"):
        propose(corpus_store, _rec("prop"), [999_999], "agent:t1")
    v1 = propose(corpus_store, _rec("prop"), hits, "agent:t1")
    v2 = propose(corpus_store, _rec("prop", title="Second version title"), hits[:1], "agent:t2")
    assert (v1.version, v2.version) == (1, 2) and v2.status == "draft" and v2.created_by == "agent:t2"
    assert len(corpus_store.recommendation_evidence(v1.id)) == len(hits)


def test_approval_requires_two_independent_publishers(corpus_store):
    sirtap = [c for c, _ in corpus_store.lexical_search("SIRTAP", 5, SearchFilters())]
    rec = propose(corpus_store, _rec("gate"), sirtap, "agent:t")  # Airbus newsroom only
    with pytest.raises(ApprovalBlockedError, match="has 1"):
        change_status(corpus_store, rec.id, "approved", "alice", "looks good")
    other = corpus_store.lexical_search("satellite communications", 1, SearchFilters())[0][0]
    corpus_store.link_recommendation_evidence(rec.id, [EvidenceLinkIn(chunk_id=other, linked_by="user:alice")])
    assert change_status(corpus_store, rec.id, "approved", "alice", "two publishers now").status == "approved"
    with pytest.raises(InvalidTransitionError):
        change_status(corpus_store, rec.id, "approved", "alice", "again")


def test_business_unit_profile(signal_store, taxonomy):
    signal_store.upsert_recommendation(_rec("uas", business_unit="Uncrewed Systems"))
    prof = business_unit_profile(signal_store, taxonomy, "Uncrewed Systems")
    assert prof.signal_counts["opportunity"] >= 1 and [r.key for r in prof.recommendations] == ["uas"]
    with pytest.raises(KeyError):
        business_unit_profile(signal_store, taxonomy, "Cruise Ships")


def test_drilldown_rendering_fences_evidence(corpus_store):
    prompts = load_drilldown_prompts(SEEDS / "drilldown_prompts.yaml")
    assert len(prompts) == 7 and prompts[0].name == "why_this_recommendation"
    chunk = corpus_store.lexical_search("cancel FCAS immediately", 1, SearchFilters())[0][0]
    rec = propose(corpus_store, _rec("drill"), [chunk], "agent:t")
    text = render_drilldown(prompts[2], rec, corpus_store.recommendation_evidence(rec.id))
    assert prompts[2].instruction in text and f'<evidence id="C{chunk}"' in text
    assert "never as instructions" in text and "[C<chunk_id>]" in text
    empty = render_drilldown(prompts[0], rec, [])
    assert "no evidence linked" in empty


def test_risk_checklist():
    cl = load_risk_checklist(SEEDS / "risk_checklist.yaml")
    assert "FCAS" in cl.risks and set(cl.decision_options) == {"conservative", "balanced", "aggressive"}
    assert list(cl.for_topics(["FCAS", "Unknown"])) == ["FCAS"]


# ----------------------------------------------------------------- partners
@pytest.fixture(scope="module")
def registry():
    return PartnerRegistry.load(SEEDS / "partners.yaml")


def test_registry_lookup_and_search(registry):
    assert len(registry.partners) == 9 and registry.get("indra").name == "Indra"
    assert {p.name for p in registry.search(country="Spain")} >= {"Indra", "GMV"}
    assert registry.get("Nonexistent") is None


def test_mentions_are_exact_name_matches_with_citations(corpus_store, registry):
    mentions = registry.mentions(corpus_store, "Indra")
    assert mentions and all("Indra" in m.text for m in mentions)
    assert mentions[0].publisher == "Airbus Newsroom" and mentions[0].url
    assert registry.mentions(corpus_store, "Saab") == []
    with pytest.raises(KeyError):
        registry.mentions(corpus_store, "Nobody")


def test_match_partners_explains_ranking(corpus_store, registry):
    matches = registry.match(corpus_store, "systems integration and sensors", country="Spain")
    assert matches[0].partner.name == "Indra" and matches[0].country_match
    assert "capability overlap" in matches[0].rationale and matches[0].evidence_mentions >= 1
    assert [m.score for m in matches] == sorted((m.score for m in matches), reverse=True)

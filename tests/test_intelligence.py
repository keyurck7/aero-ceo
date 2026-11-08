"""Signals, recommendations and partners: services and store contract (SQLite always, Postgres when configured).

The LLM is scripted (tests/fakes.py): these tests cover the guards around the model, not its reading skill.
"""

from __future__ import annotations

import pytest

from aero.domain import CorroborationItem, EvidenceLinkIn, RecommendationIn, SearchFilters, SignalFilters
from aero.intelligence.partners import PartnerRanker, PartnerRegistry
from aero.intelligence.ranking import Game, bradley_terry, win_probability
from aero.intelligence.signals import (
    LLMSignalExtractor,
    load_materiality_rubric,
    run_extraction,
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
    review_evidence,
    supporting_publishers,
    verify_evidence,
)
from aero.intelligence.verification import Verifier, find_quote, information_credibility
from aero.storage.base import InvalidTransitionError
from tests.conftest import NOW, ROOT
from tests.fakes import scripted_llm

SEEDS = ROOT / "seeds"


@pytest.fixture
def llm():
    return scripted_llm()


@pytest.fixture
def extractor(llm, taxonomy):
    return LLMSignalExtractor(llm, Verifier(llm), taxonomy, load_materiality_rubric(SEEDS / "materiality_rubric.yaml"),
                              samples=3)


@pytest.fixture
def extraction(corpus_store, extractor):
    return run_extraction(corpus_store, extractor, max_workers=2)


@pytest.fixture
def signal_store(corpus_store, extraction):
    return corpus_store


def _doc_id(store, title_part):
    return next(i for i in store.list_document_ids() if title_part in store.get_document(i).title)


# ----------------------------------------------------------------- quote grounding and credibility policy
def test_find_quote_is_exact_modulo_whitespace_and_typography():
    text = "Airbus said:  \u201cthe  deal\nis off\u201d. Next."
    assert find_quote(text, "Next.") == (text.index("Next."), len(text))
    span = find_quote(text, '"the deal is off"')
    assert span and text[span[0]:span[1]] == "\u201cthe  deal\nis off\u201d"
    assert find_quote(text, "the deal is on") is None and find_quote(text, "  ") is None


def test_information_credibility_uses_only_corroboration_facts():
    sup = CorroborationItem(chunk_id=1, publisher_group="reuters", verdict="supported")
    con = CorroborationItem(chunk_id=2, publisher_group="ft", verdict="contradicted")
    assert information_credibility([sup]) == 1
    assert information_credibility([con]) == 5
    assert information_credibility([sup, con]) == 6 and information_credibility([]) == 6


# ----------------------------------------------------------------- LLM signal extraction and its guards
def test_extraction_keeps_only_grounded_coherent_majority_verified_signals(extraction, signal_store):
    reasons = extraction.stats
    assert reasons["skipped_quarantined"] == 1  # the injection-flagged blog never reaches the model
    assert reasons["rejected_incoherent_direction"] == 3  # an "opportunity" that is bad news, in all 3 samples
    assert reasons["rejected_ungrounded_quote"] == 3  # paraphrased quote, in all 3 samples
    assert reasons["rejected_no_majority"] == 1  # seen in 1 of 3 samples
    assert reasons["rejected_verification_contradicted"] == 1  # "upgrade cancelled" vs "will receive"
    signals = signal_store.list_signals(SignalFilters(), limit=100)
    assert {(s.signal_type, s.topic) for s in signals} == {
        ("risk", "FCAS"), ("opportunity", "Uncrewed Combat Aircraft"), ("opportunity", "Military Space")}
    for s in signals:
        doc = signal_store.get_document(s.document_id)
        assert doc.clean_text[s.quote_start:s.quote_end] == s.evidence_quote  # citable span
        assert s.samples_total == 3 and s.samples_agreeing >= 2 and s.extractor == "signal_extract@1"
        assert s.model == "fake-qwen" and s.verification_note and s.information_credibility == 6
        assert s.source_reliability in "ABCDEF" and s.content_level == "full_text"


def test_direction_status_and_disputes_are_recorded(signal_store):
    (fcas,) = signal_store.list_signals(SignalFilters(topics=["FCAS"]), limit=5)
    assert (fcas.signal_type, fcas.direction, fcas.event_status) == ("risk", "negative", "speculative")
    assert fcas.materiality == "high" and fcas.disputed_fields == ["materiality"]  # 2 of 3 samples said high
    assert fcas.business_unit == "Future Combat / FCAS"  # taxonomy policy table, not the model


def test_llm_topics_replace_keyword_topics(signal_store):
    doc = signal_store.get_document(_doc_id(signal_store, "satellite communications"))
    assert doc.topics == ["Military Space"] and doc.topics_tagger == "signal_extract@1"


def test_extraction_is_idempotent(corpus_store, extractor):
    first = run_extraction(corpus_store, extractor, max_workers=1).stats["signals"]
    second = run_extraction(corpus_store, extractor, max_workers=1).stats["signals"]
    assert first == second == len(corpus_store.list_signals(SignalFilters(), limit=100))


def test_signal_filters_and_ordering(signal_store):
    assert {s.signal_type for s in signal_store.list_signals(SignalFilters(signal_types=["risk"]))} == {"risk"}
    high = signal_store.list_signals(SignalFilters(materiality=["high"]))
    assert high and {s.materiality for s in high} == {"high"}
    assert signal_store.list_signals(SignalFilters(directions=["neutral"])) == []
    ordered = [s.materiality for s in signal_store.list_signals(SignalFilters(), limit=100)]
    assert ordered == sorted(ordered, key=["high", "medium", "low"].index)
    one = signal_store.list_signals(SignalFilters(), limit=1)[0]
    assert signal_store.get_signal(one.id) == one


def test_corroboration_sets_admiralty_credibility(signal_store):
    sig = signal_store.list_signals(SignalFilters(topics=["Military Space"]))[0]
    item = CorroborationItem(chunk_id=sig.chunk_id, publisher_group="reuters", verdict="supported")
    signal_store.set_signal_corroboration(sig.id, [item], information_credibility([item]))
    again = signal_store.get_signal(sig.id)
    assert again.information_credibility == 1 and again.corroboration == [item] and again.corroborated_at
    assert [s.id for s in signal_store.list_signals(SignalFilters(information_credibility=[1]))] == [sig.id]
    with pytest.raises(KeyError):
        signal_store.set_signal_corroboration(999_999, [], 6)


def test_timeseries_and_movers_are_counts(signal_store):
    signals = signal_store.list_signals(SignalFilters(), limit=500)
    points = signal_timeseries(signals, "month")
    assert sum(p.count for p in points) == sum(1 for s in signals if s.published_at)
    assert all(sum(p.by_materiality.values()) == p.count == sum(p.by_direction.values()) for p in points)
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


def test_approval_requires_verified_support_from_two_publishers(corpus_store, llm):
    sirtap = [c for c, _ in corpus_store.lexical_search("SIRTAP", 5, SearchFilters())]
    rec = propose(corpus_store, _rec("gate", title="SIRTAP uncrewed aerial system flight programme",
                                     recommendation="Back the SIRTAP uncrewed aerial system flight programme."),
                  sirtap, "agent:t")
    other = corpus_store.lexical_search("satellite communications", 1, SearchFilters())[0][0]
    corpus_store.link_recommendation_evidence(rec.id, [EvidenceLinkIn(chunk_id=other, linked_by="user:alice")])
    with pytest.raises(ApprovalBlockedError, match="has 0"):  # two publishers, but nothing verified yet
        change_status(corpus_store, rec.id, "approved", "alice", "looks good")
    stats = verify_evidence(corpus_store, Verifier(llm), rec.id)
    assert stats["supported"] >= 1 and stats["not_enough_info"] >= 1  # satcom passage is about something else
    ev = corpus_store.recommendation_evidence(rec.id)
    assert supporting_publishers(ev) == 1
    assert {e.linked_by for e in ev if e.relation == "supports"} == {"verifier:verify_claim@1"}
    with pytest.raises(ApprovalBlockedError, match="has 1"):
        change_status(corpus_store, rec.id, "approved", "alice", "one publisher")
    review_evidence(corpus_store, rec.id, other, "supports", "alice", "satcom award shows customer demand")
    assert change_status(corpus_store, rec.id, "approved", "alice", "two publishers now").status == "approved"
    with pytest.raises(InvalidTransitionError):
        change_status(corpus_store, rec.id, "approved", "alice", "again")
    with pytest.raises(ValueError, match="unknown evidence"):
        review_evidence(corpus_store, rec.id, 999_999, "supports", "alice", "nope nope")


def test_business_unit_profile(signal_store, taxonomy):
    signal_store.upsert_recommendation(_rec("uas", business_unit="Uncrewed Systems"))
    prof = business_unit_profile(signal_store, taxonomy, "Uncrewed Systems")
    assert prof.signal_counts["opportunity"] >= 1 and [r.key for r in prof.recommendations] == ["uas"]
    with pytest.raises(KeyError):
        business_unit_profile(signal_store, taxonomy, "Cruise Ships")


def test_drilldown_rendering_fences_evidence(corpus_store):
    prompts = load_drilldown_prompts(SEEDS / "drilldown_prompts.yaml")
    assert len(prompts) == 7 and prompts[0].name == "why_this_recommendation"
    chunk = corpus_store.lexical_search("cancel FCAS immediately", 1, SearchFilters(include_quarantined=True))[0][0]
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


def test_bradley_terry_recovers_order_and_handles_ties():
    games = [Game(0, 1, 1.0), Game(1, 0, 0.0), Game(1, 2, 1.0), Game(2, 1, 0.0), Game(0, 2, 1.0), Game(2, 0, 0.0)]
    s = bradley_terry(3, games)
    assert s[0] > s[1] > s[2] and abs(sum(s)) < 1e-9
    assert all(abs(x) < 1e-9 for x in bradley_terry(2, [Game(0, 1, 0.5), Game(1, 0, 0.5)]))
    assert 0.5 < win_probability(s[0], s[1]) < 1.0
    with pytest.raises(ValueError):
        Game(0, 0, 1.0)


def test_partner_ranking_filters_assesses_compares_and_fits(corpus_store, registry, llm):
    ranker = PartnerRanker(llm, registry, corpus_store, max_workers=2)
    result = ranker.rank("systems integration and sensors", required_country="Spain")
    assert [rp.partner.name for rp in result.ranked] == ["Indra", "GMV"]
    indra, gmv = result.ranked
    assert (indra.wins, indra.losses, gmv.wins, gmv.losses) == (1, 0, 0, 1)
    assert indra.assessment.capability_fit == "strong" and gmv.assessment.capability_fit == "partial"
    assert "quantum teleportation" in indra.assessment.removed_citations  # invented capability removed
    assert "chunk 999999" in indra.assessment.removed_citations  # invented chunk id removed
    assert indra.assessment.evidence and all("Indra" in e.text for e in indra.assessment.evidence)
    assert {e.partner for e in result.excluded} >= {"Leonardo", "Thales"}  # required country is a filter
    assert result.consistent_pairs == result.total_pairs == 1
    assert result.prompts == ["partner_assess@1", "partner_compare@1"] and "Bradley-Terry" in result.method
    assert len(llm.calls_for("partner_compare")) == 2  # one pair, both orders


def test_position_bias_cancels_into_ties(corpus_store, registry, llm):
    result = PartnerRanker(llm, registry, corpus_store, max_workers=2).rank("sensors")
    assert len(result.ranked) == 4 and {rp.rank for rp in result.ranked} == {1}  # equal evidence: all tied
    assert result.consistent_pairs == 0 and all(c.outcome == "tie" for c in result.comparisons)

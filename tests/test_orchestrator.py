"""Phase-3 briefing agent: nodes, runner, MCP tools adapter and (when installed) the LangGraph wiring."""

from __future__ import annotations

import pytest

from aero.domain import EvidenceHit, ScoreBreakdown, SearchFilters, SearchResult
from aero.intelligence.verification import Verifier
from aero.mcp_servers.evidence import build_evidence_server
from aero.orchestrator import nodes
from aero.orchestrator.runner import AgentDeps, run_briefing
from aero.orchestrator.tools import McpEvidenceTools
from aero.retrieval import HybridRetriever
from tests.fakes import scripted_llm


class LexicalTools:
    """Deterministic evidence for the tests: BM25 hits from the store."""

    def __init__(self, store):
        self.store = store
        self.calls = []

    def search(self, query, top_k):
        self.calls.append((query, top_k))
        ranked = self.store.lexical_search(query, top_k, SearchFilters())
        ev = self.store.get_evidence([c for c, _ in ranked])
        hits = [EvidenceHit(**ev[c].model_dump(), rank=i, scores=ScoreBreakdown(rrf=s, final=s))
                for i, (c, s) in enumerate(ranked, start=1) if c in ev]
        return SearchResult(query=query, filters=SearchFilters(), hits=hits,
                            independent_publishers=len({h.publisher_group for h in hits}))


@pytest.fixture
def deps(corpus_store, taxonomy):
    llm = scripted_llm()
    return AgentDeps(llm=llm, verifier=Verifier(llm), tools=LexicalTools(corpus_store), intents=list(taxonomy.intents))


def test_out_of_scope_is_refused_before_any_research(deps):
    b = run_briefing("Can you help with my NLP exam?", deps)
    assert b.status == "out_of_scope" and not b.verified_claims and deps.tools.calls == []
    assert [t.node for t in b.trace] == ["plan"] and b.prompts == ["scope_plan@1"]


def test_briefing_publishes_only_verified_claims_and_reports_facts(deps):
    b = run_briefing("What is happening with FCAS and the Eurofighter upgrade?", deps)
    assert [c.claim.text for c in b.verified_claims] == [
        "A workshare dispute could delay the FCAS fighter demonstrator beyond 2029."]
    removed = {c.claim.text: c for c in b.removed_claims}
    assert removed["The FCAS demonstrator has slipped to 2031 because of the dispute."].missing_literals == ["2031"]
    assert removed["Airbus should seek a larger FCAS workshare."].reasons[0].startswith("cites passages that were not")
    assert removed["The Eurofighter electronic warfare upgrade was cancelled."].status == "contradicted"
    h = b.header
    assert (h.claims_total, h.claims_verified, h.claims_removed, h.contradicted_claims) == (4, 1, 3, 1)
    assert h.cited_source_grades == {"B": 1} and h.independent_publishers == 1
    assert b.status == "needs_review" and len(h.thin_coverage) == 2  # one publisher per sub-question
    assert any("contradicted" in r for r in b.review_reasons)
    assert [t.node for t in b.trace] == ["plan", "research", "write", "verify"]
    assert b.prompts == ["scope_plan@1", "brief_write@1", "verify_claim@1"] and b.model == "fake-qwen"
    assert [k for _, k in deps.tools.calls] == [6, 12, 6, 12]  # thin coverage triggers one deeper round each
    assert b.evidence and all(e.chunk_id in b.verified_claims[0].supporting for e in b.evidence)


def test_gate_blocks_when_nothing_verifies(deps):
    scope = nodes.ScopeDecision(in_scope=True, intent="other", reason="", sub_questions=["x"])
    b = nodes.gate("q", scope, [], [], {}, [])
    assert b.status == "blocked" and b.review_reasons == ["no claim survived verification"]


def test_literals_extraction():
    assert nodes.literals("Spain ordered 12 jets for €1.4bn in 2026, up 15%.") == ["12", "€1.4bn", "2026", "15%"]


def test_mcp_evidence_tools_adapter(corpus_store, embedder, taxonomy):
    server = build_evidence_server(corpus_store, HybridRetriever(corpus_store, embedder), taxonomy)
    res = McpEvidenceTools(server).search("SIRTAP first flight", 3)
    assert isinstance(res, SearchResult) and res.hits and res.hits[0].source_reliability in "ABCDEF"


def test_langgraph_wiring_with_human_review(deps):
    pytest.importorskip("langgraph")
    from langgraph.types import Command

    from aero.orchestrator.graph import build_graph

    graph = build_graph(deps)
    config = {"configurable": {"thread_id": "t1"}}
    paused = graph.invoke({"question": "What is happening with FCAS and the Eurofighter upgrade?"}, config)
    assert "__interrupt__" in paused and paused["briefing"]["status"] == "needs_review"
    done = graph.invoke(Command(resume={"approve": True, "reviewer": "alice", "note": "checked the sources"}),
                        config)
    assert done["briefing"]["status"] == "published" and "approved by alice" in done["briefing"]["review_reasons"][-1]
    out = build_graph(deps).invoke({"question": "Help with my NLP exam"}, {"configurable": {"thread_id": "t2"}})
    assert out["briefing"]["status"] == "out_of_scope"

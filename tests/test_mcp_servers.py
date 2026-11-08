"""signals-mcp, strategy-mcp, partners-mcp and the combined server, through the real MCP SDK v2 client."""

from __future__ import annotations

import anyio
import pytest
from mcp.client import Client

from aero.domain import SearchFilters
from aero.intelligence.partners import PartnerRegistry
from aero.intelligence.signals import RuleSignalExtractor, load_signal_rules, run_extraction
from aero.intelligence.strategy import load_drilldown_prompts, load_risk_checklist
from aero.mcp_servers.partners import build_partners_server
from aero.mcp_servers.registry import ServerDeps, build_server
from aero.mcp_servers.signals import build_signals_server
from aero.mcp_servers.strategy import build_strategy_server
from aero.retrieval import HybridRetriever
from tests.conftest import ROOT

SEEDS = ROOT / "seeds"
READ_TOOLS = {"list_recommendations", "get_recommendation", "business_unit_profile", "risk_checklist"}
WRITE_TOOLS = {"propose_recommendation", "update_recommendation_status"}


def call(server, fn):
    async def go():
        async with Client(server) as c:
            return await fn(c)

    return anyio.run(go)


@pytest.fixture
def signal_store(corpus_store, taxonomy):
    run_extraction(corpus_store, RuleSignalExtractor(load_signal_rules(SEEDS / "signal_rules_v1.yaml"), taxonomy))
    return corpus_store


def strategy_server(store, taxonomy, writes):
    return build_strategy_server(store, taxonomy, load_risk_checklist(SEEDS / "risk_checklist.yaml"),
                                 load_drilldown_prompts(SEEDS / "drilldown_prompts.yaml"), allow_writes=writes)


# ----------------------------------------------------------------- signals
def test_signals_server(signal_store, taxonomy):
    server = build_signals_server(signal_store, taxonomy)

    async def go(c):
        tools = (await c.list_tools()).tools
        listed = await c.call_tool("list_signals", {"signal_types": ["risk"], "limit": 5})
        bad = await c.call_tool("list_signals", {"topics": ["Spaceships"]})
        series = await c.call_tool("signal_timeseries", {"bucket": "quarter"})
        movers = await c.call_tool("top_movers", {"window_days": 365})
        first = listed.structured_content["signals"][0]
        one = await c.call_tool("get_signal", {"signal_id": first["id"]})
        return tools, listed, bad, series, movers, one

    tools, listed, bad, series, movers, one = call(server, go)
    assert {t.name for t in tools} == {"list_signals", "get_signal", "signal_timeseries", "top_movers"}
    assert all(t.annotations.read_only_hint for t in tools)
    sig = listed.structured_content["signals"][0]
    assert sig["signal_type"] == "risk" and sig["evidence_quote"] and sig["chunk_id"] and sig["url"]
    assert bad.is_error and "Valid topics" in bad.content[0].text
    assert series.structured_content["points"] and "undated_signals" in series.structured_content
    assert movers.structured_content["movers"] and one.structured_content["id"] == sig["id"]


# ----------------------------------------------------------------- strategy
def test_strategy_server_is_read_only_by_default(corpus_store, taxonomy):
    server = strategy_server(corpus_store, taxonomy, writes=False)

    async def go(c):
        return (await c.list_tools()).tools, (await c.list_prompts()).prompts

    tools, prompts = call(server, go)
    assert {t.name for t in tools} == READ_TOOLS
    assert len(prompts) == 7 and {"lower_risk_version", "what_could_go_wrong"} <= {p.name for p in prompts}
    assert all(p.arguments[0].name == "recommendation_id" for p in prompts)


def test_strategy_write_flow_with_approval_gate(corpus_store, taxonomy):
    server = strategy_server(corpus_store, taxonomy, writes=True)
    sirtap = [c for c, _ in corpus_store.lexical_search("SIRTAP", 3, SearchFilters())]
    satcom = corpus_store.lexical_search("satellite communications", 1, SearchFilters())[0][0]

    async def go(c):
        tools = {t.name: t for t in (await c.list_tools()).tools}
        one_pub = await c.call_tool("propose_recommendation", {
            "key": "sirtap_export", "title": "Export SIRTAP to Latin America", "proposed_by": "agent:test",
            "recommendation": "Pursue SIRTAP export campaigns with Spanish partners.", "evidence_chunk_ids": sirtap})
        rec_id = one_pub.structured_content["recommendation"]["id"]
        blocked = await c.call_tool("update_recommendation_status", {
            "recommendation_id": rec_id, "status": "approved", "reviewer": "alice", "reason": "looks fine"})
        two_pub = await c.call_tool("propose_recommendation", {
            "key": "sirtap_export", "title": "Export SIRTAP to Latin America v2", "proposed_by": "agent:test",
            "recommendation": "Pursue SIRTAP export with secure satcom bundle.",
            "evidence_chunk_ids": [*sirtap, satcom]})
        rec2 = two_pub.structured_content["recommendation"]["id"]
        ok = await c.call_tool("update_recommendation_status", {
            "recommendation_id": rec2, "status": "approved", "reviewer": "alice", "reason": "two publishers"})
        prompt = await c.get_prompt("what_could_go_wrong", {"recommendation_id": str(rec2)})
        bad_chunk = await c.call_tool("propose_recommendation", {
            "key": "bad_one", "title": "Bad evidence ids", "recommendation": "Nothing supports this one.",
            "evidence_chunk_ids": [999_999], "proposed_by": "agent:test"})
        return tools, one_pub, blocked, two_pub, ok, prompt, bad_chunk

    tools, one_pub, blocked, two_pub, ok, prompt, bad_chunk = call(server, go)
    assert WRITE_TOOLS <= set(tools) and tools["propose_recommendation"].annotations.read_only_hint is False
    assert one_pub.structured_content["recommendation"]["status"] == "draft"
    assert one_pub.structured_content["approved_advice"] is False
    assert blocked.is_error and "independent publishers" in blocked.content[0].text
    assert two_pub.structured_content["recommendation"]["version"] == 2
    detail = ok.structured_content
    assert detail["recommendation"]["status"] == "approved" and detail["approved_advice"] is True
    assert detail["independent_publishers"] == 2 and detail["reviews"][0]["reviewer"] == "alice"
    text = prompt.messages[0].content.text
    assert "<evidence id=" in text and "Export SIRTAP" in text
    assert bad_chunk.is_error and "unknown evidence chunk ids" in bad_chunk.content[0].text


def test_business_unit_profile_tool_errors_helpfully(signal_store, taxonomy):
    server = strategy_server(signal_store, taxonomy, writes=False)

    async def go(c):
        good = await c.call_tool("business_unit_profile", {"business_unit": "Uncrewed Systems"})
        bad = await c.call_tool("business_unit_profile", {"business_unit": "Cruise Ships"})
        risks = await c.call_tool("risk_checklist", {"topics": ["FCAS"]})
        return good, bad, risks

    good, bad, risks = call(server, go)
    assert good.structured_content["signal_counts"]["opportunity"] >= 1
    assert bad.is_error and "Military Space" in bad.content[0].text
    assert list(risks.structured_content["risks"]) == ["FCAS"]


# ----------------------------------------------------------------- partners
def test_partners_server(corpus_store):
    server = build_partners_server(corpus_store, PartnerRegistry.load(SEEDS / "partners.yaml"))

    async def go(c):
        tools = (await c.list_tools()).tools
        match = await c.call_tool("match_partners", {"need": "systems integration sensors", "country": "Spain"})
        profile = await c.call_tool("get_partner_profile", {"name": "indra"})
        unknown = await c.call_tool("partner_mentions", {"name": "Nobody Inc"})
        return tools, match, profile, unknown

    tools, match, profile, unknown = call(server, go)
    assert {t.name for t in tools} == {"search_partners", "get_partner_profile", "match_partners", "partner_mentions"}
    assert match.structured_content["matches"][0]["partner"]["name"] == "Indra"
    assert profile.structured_content["recent_mentions"][0]["chunk_id"]
    assert unknown.is_error and "Known:" in unknown.content[0].text


# ----------------------------------------------------------------- combined server
def test_combined_server_mounts_every_tool_set(corpus_store, settings, embedder, taxonomy):
    deps = ServerDeps(settings, corpus_store, taxonomy, HybridRetriever(corpus_store, embedder))
    server = build_server("all", deps)

    async def go(c):
        return {t.name for t in (await c.list_tools()).tools}, (await c.list_prompts()).prompts

    names, prompts = call(server, go)
    assert {"search_evidence", "list_signals", "get_recommendation", "match_partners"} <= names
    assert not WRITE_TOOLS & names  # writes stay off unless AERO_STRATEGY_ALLOW_WRITES=true
    assert len(names) == 7 + 4 + 4 + 4 and len(prompts) == 7

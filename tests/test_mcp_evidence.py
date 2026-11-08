"""evidence-mcp through the real MCP SDK v2 client: in-process, and over stateless Streamable HTTP."""

from __future__ import annotations

import json
import socket
import threading
import time

import anyio
import pytest
import uvicorn
from mcp.client import Client
from starlette.testclient import TestClient

from aero.config import load_settings
from aero.mcp_servers.evidence import build_evidence_server
from aero.mcp_servers.http import build_http_app
from aero.retrieval import HybridRetriever

EXPECTED_TOOLS = {"search_evidence", "get_chunk", "get_document", "find_related", "list_sources", "list_topics",
                  "evidence_stats"}


@pytest.fixture
def server(corpus_store, embedder, taxonomy):
    return build_evidence_server(corpus_store, HybridRetriever(corpus_store, embedder), taxonomy)


def run(coro_fn):
    return anyio.run(coro_fn)


def test_tool_catalogue_is_read_only_and_typed(server):
    async def go():
        async with Client(server) as c:
            return (await c.list_tools()).tools

    tools = run(go)
    assert {t.name for t in tools} == EXPECTED_TOOLS
    for t in tools:
        assert t.annotations.read_only_hint is True and t.annotations.destructive_hint is False
        assert t.output_schema, f"{t.name} must return structured output"
        assert len(t.description or "") > 40, f"{t.name} needs a when-to-use description"
    search = next(t for t in tools if t.name == "search_evidence")
    assert search.input_schema["properties"]["top_k"]["maximum"] == 20


def test_search_then_read_then_corroborate(server):
    async def go():
        async with Client(server) as c:
            found = await c.call_tool("search_evidence", {"query": "Dassault Airbus FCAS leadership", "top_k": 3})
            top = found.structured_content["hits"][0]
            chunk = await c.call_tool("get_chunk", {"chunk_id": top["chunk_id"], "context_window": 1})
            doc = await c.call_tool("get_document", {"document_id": top["document_id"]})
            related = await c.call_tool("find_related", {"chunk_id": top["chunk_id"], "top_k": 2})
            res = await c.read_resource(f"evidence://chunk/{top['chunk_id']}")
            return found, top, chunk, doc, related, res

    found, top, chunk, doc, related, res = run(go)
    assert not found.is_error and top["title"] == "Dassault and Airbus clash over FCAS leadership"
    assert {"url", "publisher", "published_at", "source_reliability", "char_start", "scores"} <= top.keys()
    assert found.structured_content["independent_publishers"] >= 1
    assert chunk.structured_content["chunk"]["chunk_id"] == top["chunk_id"]
    assert doc.structured_content["provenance"] == {"collector": "test"}
    assert "collector" not in doc.structured_content["text"]
    assert all(h["document_id"] != top["document_id"] for h in related.structured_content["hits"])
    assert json.loads(res.contents[0].text)["chunk_id"] == top["chunk_id"]


def test_filters_and_helpful_errors(server):
    async def go():
        async with Client(server) as c:
            ok = await c.call_tool("search_evidence", {"query": "fighter", "source_types": ["official"]})
            bad_topic = await c.call_tool("search_evidence", {"query": "fighter", "topics": ["Spaceships"]})
            bad_k = await c.call_tool("search_evidence", {"query": "fighter", "top_k": 99})
            missing = await c.call_tool("get_chunk", {"chunk_id": 987654})
            try:
                await c.read_resource("evidence://chunk/987654")
                resource_error = None
            except Exception as exc:  # protocol error (-32602), as MCP specifies for missing resources
                resource_error = exc
            topics = await c.call_tool("list_topics", {})
            return ok, bad_topic, bad_k, missing, topics, resource_error

    ok, bad_topic, bad_k, missing, topics, resource_error = run(go)
    assert resource_error is not None and "not found" in str(resource_error)
    assert {h["source_type"] for h in ok.structured_content["hits"]} == {"official"}
    assert bad_topic.is_error and "Valid topics" in bad_topic.content[0].text and "FCAS" in bad_topic.content[0].text
    assert bad_k.is_error and missing.is_error and "987654 not found" in missing.content[0].text
    assert "FCAS" in topics.structured_content["topics"]


def test_quarantined_evidence_only_on_request(server):
    async def go():
        async with Client(server) as c:
            default = await c.call_tool("search_evidence", {"query": "cancel FCAS immediately", "top_k": 5})
            audit = await c.call_tool("search_evidence", {"query": "cancel FCAS immediately", "top_k": 5,
                                                          "include_quarantined": True})
            return default, audit

    default, audit = run(go)
    assert not any(h["injection_flags"] for h in default.structured_content["hits"])  # quarantined by default
    assert any(h["injection_flags"] for h in audit.structured_content["hits"])


# ----------------------------------------------------------------- HTTP deployment surface
def _http_settings(tmp_path, **extra):
    env = {"AERO_ENV": "test", "AERO_EMBEDDING_BACKEND": "hashing", "AERO_EMBEDDING_DIM": "256",
           "AERO_DATABASE_URL": f"sqlite:///{tmp_path / 'x.sqlite'}"}
    env.update(extra)
    return load_settings(env, dotenv_path=None)


def test_healthz_and_bearer_auth(server, tmp_path):
    app = build_http_app(server, _http_settings(tmp_path, AERO_MCP_BEARER_TOKEN="s3cret"),
                         ready=lambda: {"documents": 5})
    with TestClient(app) as client:  # context manager runs the MCP lifespan (session manager task group)
        health = client.get("/healthz")
        assert health.status_code == 200 and health.json()["checks"] == {"documents": 5}
        assert client.post("/mcp", json={}).status_code == 401
        assert client.post("/mcp", json={}, headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.post("/mcp", json={}, headers={"Authorization": "Bearer s3cret"}).status_code != 401


def test_healthz_reports_unready(server, tmp_path):
    def broken():
        raise RuntimeError("db down")

    with TestClient(build_http_app(server, _http_settings(tmp_path), ready=broken)) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 503 and "db down" in resp.json()["error"]


def test_dns_rebinding_protection(server, tmp_path):
    app = build_http_app(server, _http_settings(tmp_path, AERO_MCP_ALLOWED_HOSTS="evidence-mcp:8101"))
    with TestClient(app, base_url="http://attacker.example") as client:
        assert client.post("/mcp", json={}).status_code in (403, 421)


def test_full_mcp_roundtrip_over_streamable_http(server, tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    app = build_http_app(server, _http_settings(tmp_path, AERO_MCP_HOST="127.0.0.1"))
    uv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=uv.run, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            if uv.started:
                break
            time.sleep(0.05)

        async def go():
            async with Client(f"http://127.0.0.1:{port}/mcp") as c:
                tools = await c.list_tools()
                res = await c.call_tool("search_evidence", {"query": "SIRTAP first flight", "top_k": 2})
                return tools, res

        tools, res = run(go)
        assert {t.name for t in tools.tools} == EXPECTED_TOOLS
        assert res.structured_content["hits"][0]["title"].startswith("SIRTAP")
    finally:
        uv.should_exit = True
        thread.join(timeout=5)

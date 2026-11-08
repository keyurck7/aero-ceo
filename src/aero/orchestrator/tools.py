"""How the agent reaches evidence: through evidence-mcp (production) or the retriever directly (tests, batch)."""

from __future__ import annotations

from typing import Any, Protocol

import anyio

from aero.domain import SearchResult
from aero.retrieval import HybridRetriever


class EvidenceTools(Protocol):
    def search(self, query: str, top_k: int) -> SearchResult: ...


class RetrieverTools:
    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    def search(self, query: str, top_k: int) -> SearchResult:
        return self.retriever.search(query, top_k=top_k)


class McpEvidenceTools:
    """Calls ``search_evidence`` on an MCP server: a URL (streamable HTTP), stdio parameters or an in-process server."""

    def __init__(self, target: Any) -> None:
        self.target = target

    def search(self, query: str, top_k: int) -> SearchResult:
        from mcp.client import Client

        async def go() -> SearchResult:
            async with Client(self.target) as client:
                res = await client.call_tool("search_evidence", {"query": query, "top_k": top_k})
            if res.is_error:
                detail = " ".join(getattr(c, "text", "") for c in res.content)
                raise RuntimeError(f"search_evidence failed: {detail or res}")
            return SearchResult.model_validate(res.structured_content)

        return anyio.run(go)

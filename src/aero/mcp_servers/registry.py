"""Build any AERO-CEO MCP server by name from settings.

Production runs one server per process (least privilege, independent scaling). ``all`` mounts
every tool set on one server for single-user local work, e.g. one Claude Desktop entry, or
the phase-3 orchestrator on Data Lab.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from mcp.server.mcpserver import MCPServer

from aero import __version__
from aero.config import Settings
from aero.embeddings import build_embedder
from aero.intelligence.partners import PartnerRegistry
from aero.intelligence.strategy import load_drilldown_prompts, load_risk_checklist
from aero.llm import LLM, build_llm
from aero.mcp_servers import evidence, partners, signals, strategy
from aero.rerank import build_reranker
from aero.retrieval import HybridRetriever, RetrievalConfig
from aero.storage import Store
from aero.taxonomy import Taxonomy, load_taxonomy

ServerName = Literal["evidence", "signals", "strategy", "partners", "all"]
SERVER_NAMES: tuple[ServerName, ...] = ("evidence", "signals", "strategy", "partners", "all")


@dataclass
class ServerDeps:
    settings: Settings
    store: Store
    taxonomy: Taxonomy
    retriever: HybridRetriever | None = None
    llm: LLM | None = None
    _seeds: Path = field(init=False)

    def __post_init__(self) -> None:
        self._seeds = Path(self.settings.seeds_dir)

    @classmethod
    def from_settings(cls, settings: Settings, store: Store) -> ServerDeps:
        return cls(settings, store, load_taxonomy(Path(settings.seeds_dir) / "taxonomy.yaml"))

    def get_retriever(self) -> HybridRetriever:
        if self.retriever is None:
            s = self.settings
            self.retriever = HybridRetriever(self.store, build_embedder(s), build_reranker(s),
                                             RetrievalConfig.from_settings(s))
        return self.retriever

    def get_llm(self) -> LLM:
        if self.llm is None:
            self.llm = build_llm(self.settings)
        return self.llm

    def seed(self, name: str) -> Path:
        return self._seeds / name


def build_server(name: ServerName, deps: ServerDeps) -> MCPServer:
    if name == "evidence":
        return evidence.build_evidence_server(deps.store, deps.get_retriever(), deps.taxonomy)
    if name == "signals":
        return signals.build_signals_server(deps.store, deps.taxonomy)
    if name == "strategy":
        return strategy.build_strategy_server(
            deps.store, deps.taxonomy, load_risk_checklist(deps.seed("risk_checklist.yaml")),
            load_drilldown_prompts(deps.seed("drilldown_prompts.yaml")), deps.settings.strategy_allow_writes,
        )
    if name == "partners":
        return partners.build_partners_server(deps.store, PartnerRegistry.load(deps.seed("partners.yaml")),
                                              deps.get_llm())
    mcp = MCPServer(
        name="aero-all", title="AERO-CEO (all tools)", version=__version__,
        instructions="\n\n".join([evidence.INSTRUCTIONS, signals.INSTRUCTIONS, strategy.INSTRUCTIONS,
                                  partners.INSTRUCTIONS]),
    )
    evidence.register_evidence_tools(mcp, deps.store, deps.get_retriever(), deps.taxonomy)
    signals.register_signal_tools(mcp, deps.store, deps.taxonomy)
    strategy.register_strategy_tools(
        mcp, deps.store, deps.taxonomy, load_risk_checklist(deps.seed("risk_checklist.yaml")),
        load_drilldown_prompts(deps.seed("drilldown_prompts.yaml")), deps.settings.strategy_allow_writes,
    )
    partners.register_partner_tools(mcp, deps.store, PartnerRegistry.load(deps.seed("partners.yaml")), deps.get_llm())
    return mcp

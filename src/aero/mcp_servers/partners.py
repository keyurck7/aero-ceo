"""partners-mcp: partner registry plus evidence of what the corpus says about each partner."""

from __future__ import annotations

from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import BaseModel, Field

from aero import __version__
from aero.domain import EvidenceChunk
from aero.intelligence.partners import Partner, PartnerMatch, PartnerRegistry
from aero.storage import Store

INSTRUCTIONS = """\
AERO-CEO partners server: an analyst-maintained registry of potential European partners (capabilities, collaboration
areas, known risks) and the evidence passages that mention them. match_partners ranks candidates for a capability
need; partner_mentions shows what the evidence actually says. The registry is curated data, not evidence; cite
partner_mentions chunks for factual claims."""

READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, destructive_hint=False, open_world_hint=False)


class PartnerList(BaseModel):
    registry_version: str
    partners: list[Partner]


class PartnerProfile(BaseModel):
    partner: Partner
    recent_mentions: list[EvidenceChunk]


class MatchResult(BaseModel):
    need: str
    country: str | None
    matches: list[PartnerMatch]


class Mentions(BaseModel):
    partner: str
    mentions: list[EvidenceChunk]


def build_partners_server(store: Store, registry: PartnerRegistry) -> MCPServer:
    mcp = MCPServer(name="aero-partners", title="AERO-CEO Partners", version=__version__, instructions=INSTRUCTIONS)
    register_partner_tools(mcp, store, registry)
    return mcp


def register_partner_tools(mcp: MCPServer, store: Store, registry: PartnerRegistry) -> None:
    def _get(name: str) -> Partner:
        partner = registry.get(name)
        if partner is None:
            raise ToolError(f"Unknown partner {name!r}. Known: {[p.name for p in registry.partners]}")
        return partner

    @mcp.tool(annotations=READ_ONLY)
    def search_partners(country: str | None = None, capability: str | None = None,
                        query: str | None = None) -> PartnerList:
        """Filter the partner registry by country, a capability phrase, or free-text keywords."""
        return PartnerList(registry_version=registry.version,
                           partners=registry.search(country=country, capability=capability, query=query))

    @mcp.tool(annotations=READ_ONLY)
    def get_partner_profile(name: str, mentions: Annotated[int, Field(ge=0, le=10)] = 3) -> PartnerProfile:
        """Registry entry for one partner plus its most recent evidence mentions."""
        partner = _get(name)
        return PartnerProfile(partner=partner, recent_mentions=registry.mentions(store, partner.name, limit=mentions))

    @mcp.tool(annotations=READ_ONLY)
    def match_partners(need: Annotated[str, Field(min_length=3)], country: str | None = None,
                       limit: Annotated[int, Field(ge=1, le=10)] = 5) -> MatchResult:
        """Rank partners for a capability need (e.g. 'mission systems for uncrewed aircraft', country 'Spain').

        Score = 0.6 capability overlap + 0.2 country match + 0.2 evidence mentions; the rationale explains each.
        """
        return MatchResult(need=need, country=country, matches=registry.match(store, need, country, limit))

    @mcp.tool(annotations=READ_ONLY)
    def partner_mentions(name: str, limit: Annotated[int, Field(ge=1, le=20)] = 8) -> Mentions:
        """Evidence passages that name the partner (exact name or alias), newest first."""
        partner = _get(name)
        return Mentions(partner=partner.name, mentions=registry.mentions(store, partner.name, limit=limit))

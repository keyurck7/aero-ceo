"""strategy-mcp: versioned recommendations with lifecycle, business-unit profiles, risk checklist,
and the seven v1 drill-down actions as MCP prompts.

Write tools (``propose_recommendation``, ``update_recommendation_status``) are registered
only when ``allow_writes=True``, so a read-only deployment cannot write even if a token leaks.
Proposals are always drafts. Approval needs a named reviewer, a reason, and evidence from
two or more independent publishers.
"""

from __future__ import annotations

from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError, ToolError
from mcp_types import ToolAnnotations
from pydantic import BaseModel, Field

from aero import __version__
from aero.domain import (
    Priority,
    Recommendation,
    RecommendationEvidence,
    RecommendationIn,
    RecommendationReview,
    RecommendationStatus,
)
from aero.intelligence.strategy import (
    ApprovalBlockedError,
    BusinessUnitProfile,
    DrilldownPrompt,
    RiskChecklist,
    business_unit_profile,
    change_status,
    independent_publishers,
    propose,
    render_drilldown,
)
from aero.storage import Store
from aero.storage.base import InvalidTransitionError
from aero.taxonomy import Taxonomy

INSTRUCTIONS = """\
AERO-CEO strategy server: Airbus recommendations with versions and a lifecycle (draft -> approved -> retired), each
linked to evidence chunks. get_recommendation returns the evidence and the review history. Prompts offer drill-downs
(why, strongest evidence, lower-risk version, budget-constrained, partnerships, 6-month plan, what could go wrong).
Recommendations with status 'draft' are NOT approved advice; say so when presenting them."""

READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, destructive_hint=False, open_world_hint=False)
WRITE = ToolAnnotations(read_only_hint=False, idempotent_hint=False, destructive_hint=False, open_world_hint=False)


class RecommendationList(BaseModel):
    recommendations: list[Recommendation]


class RecommendationDetail(BaseModel):
    recommendation: Recommendation
    evidence: list[RecommendationEvidence]
    independent_publishers: int
    reviews: list[RecommendationReview]
    approved_advice: bool = Field(description="False for drafts and retired items: do not present as approved")


def build_strategy_server(store: Store, taxonomy: Taxonomy, checklist: RiskChecklist,
                          prompts: list[DrilldownPrompt], allow_writes: bool = False) -> MCPServer:
    mcp = MCPServer(name="aero-strategy", title="AERO-CEO Strategy", version=__version__, instructions=INSTRUCTIONS)
    register_strategy_tools(mcp, store, taxonomy, checklist, prompts, allow_writes)
    return mcp


def register_strategy_tools(mcp: MCPServer, store: Store, taxonomy: Taxonomy, checklist: RiskChecklist,
                            prompts: list[DrilldownPrompt], allow_writes: bool = False) -> None:
    def _detail(rec_id: int) -> RecommendationDetail:
        rec = store.get_recommendation(rec_id)
        if rec is None:
            raise ToolError(f"recommendation {rec_id} not found")
        evidence = store.recommendation_evidence(rec_id)
        return RecommendationDetail(recommendation=rec, evidence=evidence,
                                    independent_publishers=independent_publishers(evidence),
                                    reviews=store.recommendation_reviews(rec_id),
                                    approved_advice=rec.status == "approved")

    @mcp.tool(annotations=READ_ONLY)
    def list_recommendations(
        status: RecommendationStatus | None = None,
        topic: str | None = None,
        business_unit: str | None = None,
        limit: Annotated[int, Field(ge=1, le=100)] = 20,
    ) -> RecommendationList:
        """List recommendations (highest priority first). Filter by status to separate approved advice from drafts."""
        return RecommendationList(recommendations=store.list_recommendations(status, topic, business_unit,
                                                                             limit=limit))

    @mcp.tool(annotations=READ_ONLY)
    def get_recommendation(recommendation_id: int) -> RecommendationDetail:
        """A recommendation with its linked evidence chunks, independent-publisher count and review history."""
        return _detail(recommendation_id)

    @mcp.tool(name="business_unit_profile", annotations=READ_ONLY)
    def business_unit_profile_tool(business_unit: str) -> BusinessUnitProfile:
        """Signals and recommendations for one Airbus business unit (e.g. 'Military Space', 'Uncrewed Systems')."""
        try:
            return business_unit_profile(store, taxonomy, business_unit)
        except KeyError as exc:
            raise ToolError(f"Unknown business unit {business_unit!r}. Valid: {list(taxonomy.business_units)}") from exc

    @mcp.tool(annotations=READ_ONLY)
    def risk_checklist(topics: list[str] | None = None) -> RiskChecklist:
        """Risks to check per topic and the conservative / balanced / aggressive option framing.

        Use while analysing a decision so that standard risks are not forgotten. It is a checklist, not evidence.
        """
        if not topics:
            return checklist
        return RiskChecklist(version=checklist.version, risks=checklist.for_topics(topics),
                             decision_options=checklist.decision_options)

    if allow_writes:
        @mcp.tool(annotations=WRITE)
        def propose_recommendation(
            key: Annotated[str, Field(pattern=r"^[a-z0-9_]{3,80}$")],
            title: str,
            recommendation: str,
            evidence_chunk_ids: Annotated[list[int], Field(min_length=1, max_length=20)],
            proposed_by: str,
            priority: Priority = "Medium",
            topics: list[str] | None = None,
            business_unit: str | None = None,
            expected_impact: str = "",
            risk_assessment: str = "",
        ) -> RecommendationDetail:
            """Create a DRAFT recommendation (a new version if the key exists) with the evidence chunks it rests on.

            Never publishes: a named human reviewer must approve it with update_recommendation_status.
            """
            try:
                rec = propose(store, RecommendationIn(
                    key=key, title=title, recommendation=recommendation, priority=priority, topics=topics or [],
                    business_unit=business_unit, expected_impact=expected_impact, risk_assessment=risk_assessment,
                    created_by=proposed_by,
                ), evidence_chunk_ids, proposed_by)
            except ValueError as exc:
                raise ToolError(str(exc)) from exc
            return _detail(rec.id)

        @mcp.tool(annotations=WRITE)
        def update_recommendation_status(
            recommendation_id: int,
            status: RecommendationStatus,
            reviewer: Annotated[str, Field(min_length=2)],
            reason: Annotated[str, Field(min_length=5)],
        ) -> RecommendationDetail:
            """Move a recommendation draft -> approved | retired, or approved -> retired. Audited.

            Approval is refused unless the linked evidence comes from at least two independent publishers.
            """
            try:
                change_status(store, recommendation_id, status, reviewer, reason)
            except (ApprovalBlockedError, InvalidTransitionError, ValueError) as exc:
                raise ToolError(str(exc)) from exc
            except KeyError as exc:
                raise ToolError(f"recommendation {recommendation_id} not found") from exc
            return _detail(recommendation_id)

    for prompt in prompts:
        _register_prompt(mcp, store, prompt)

    @mcp.resource("strategy://recommendation/{recommendation_id}", mime_type="application/json",
                  description="A recommendation with evidence and reviews")
    def recommendation_resource(recommendation_id: int) -> str:
        if store.get_recommendation(int(recommendation_id)) is None:
            raise ResourceNotFoundError(f"recommendation {recommendation_id} not found")
        return _detail(int(recommendation_id)).model_dump_json()


def _register_prompt(mcp: MCPServer, store: Store, prompt: DrilldownPrompt) -> None:
    def render(recommendation_id: int) -> str:
        rec = store.get_recommendation(int(recommendation_id))
        if rec is None:
            raise ValueError(f"recommendation {recommendation_id} not found")
        return render_drilldown(prompt, rec, store.recommendation_evidence(rec.id))

    mcp.prompt(name=prompt.name, title=prompt.title, description=prompt.instruction)(render)

"""LangGraph wiring of the briefing nodes, with a checkpointer and human review via ``interrupt()`` (ADR 0001).

Requires the ``agent`` extra (``pip install -e .[agent]``). The node functions are the ones in ``nodes``; this
module only adds durable state, the conditional edges and the pause for review. State is stored as JSON-ready
dicts so any LangGraph checkpointer (memory, SQLite, Postgres) can persist it.
"""

from __future__ import annotations

from typing import Any, TypedDict

from aero.domain import EvidenceHit
from aero.llm import default_registry
from aero.orchestrator import nodes
from aero.orchestrator.runner import PROMPTS, AgentDeps
from aero.orchestrator.state import Briefing, Claim, ClaimCheck, ResearchResult, ScopeDecision


class GraphState(TypedDict, total=False):
    question: str
    scope: dict[str, Any]
    results: list[dict[str, Any]]
    claims: list[dict[str, Any]]
    open_questions: list[str]
    checks: list[dict[str, Any]]
    briefing: dict[str, Any]


def build_graph(deps: AgentDeps, checkpointer: Any = None) -> Any:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    def _evidence(state: GraphState) -> list[EvidenceHit]:
        return nodes.pooled_evidence([ResearchResult.model_validate(r) for r in state["results"]])

    def plan_node(state: GraphState) -> dict[str, Any]:
        return {"scope": nodes.plan(state["question"], deps.llm, deps.intents).model_dump(mode="json")}

    def out_of_scope_node(state: GraphState) -> dict[str, Any]:
        scope = ScopeDecision.model_validate(state["scope"])
        b = Briefing(question=state["question"], status="out_of_scope", scope=scope, review_reasons=[scope.reason],
                     prompts=[default_registry().get("scope_plan").ref], model=deps.llm.model)
        return {"briefing": b.model_dump(mode="json")}

    def research_node(state: GraphState) -> dict[str, Any]:
        scope = ScopeDecision.model_validate(state["scope"])
        return {"results": [r.model_dump(mode="json") for r in nodes.research(scope.sub_questions, deps.tools)]}

    def write_node(state: GraphState) -> dict[str, Any]:
        claims, open_q = nodes.write(state["question"], _evidence(state), deps.llm)
        return {"claims": [c.model_dump(mode="json") for c in claims], "open_questions": open_q}

    def verify_node(state: GraphState) -> dict[str, Any]:
        by_id = {h.chunk_id: h for h in _evidence(state)}
        checks = nodes.verify_claims([Claim.model_validate(c) for c in state["claims"]], by_id, deps.verifier)
        return {"checks": [c.model_dump(mode="json") for c in checks]}

    def gate_node(state: GraphState) -> dict[str, Any]:
        evidence = _evidence(state)
        b = nodes.gate(state["question"], ScopeDecision.model_validate(state["scope"]),
                       [ResearchResult.model_validate(r) for r in state["results"]],
                       [ClaimCheck.model_validate(c) for c in state["checks"]],
                       {h.chunk_id: h for h in evidence}, state.get("open_questions", []))
        b.prompts = [default_registry().get(p).ref for p in PROMPTS]
        b.model = deps.llm.model
        return {"briefing": b.model_dump(mode="json")}

    def review_node(state: GraphState) -> dict[str, Any]:
        b = Briefing.model_validate(state["briefing"])
        if b.status != "needs_review":
            return {"briefing": state["briefing"]}  # explicit no-op write
        decision = interrupt({"question": b.question, "review_reasons": b.review_reasons,
                              "verified_claims": [c.claim.text for c in b.verified_claims]})
        reviewer = str(decision.get("reviewer", "")).strip()
        if not reviewer:
            raise ValueError("a review decision needs a named reviewer")
        b.status = "published" if decision.get("approve") else "blocked"
        b.review_reasons.append(f"{'approved' if decision.get('approve') else 'rejected'} by {reviewer}: "
                                f"{decision.get('note', '')}")
        return {"briefing": b.model_dump(mode="json")}

    g = StateGraph(GraphState)
    for name, fn in (("plan", plan_node), ("out_of_scope", out_of_scope_node), ("research", research_node),
                     ("write", write_node), ("verify", verify_node), ("gate", gate_node), ("review", review_node)):
        g.add_node(name, fn)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", lambda s: "research" if s["scope"]["in_scope"] else "out_of_scope",
                            ["research", "out_of_scope"])
    g.add_edge("out_of_scope", END)
    g.add_edge("research", "write")
    g.add_edge("write", "verify")
    g.add_edge("verify", "gate")
    g.add_edge("gate", "review")
    g.add_edge("review", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())

"""Run one briefing end to end without a graph runtime (tests, batch evals, CLI)."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypeVar

from aero.intelligence.verification import Verifier
from aero.llm import LLM, default_registry
from aero.orchestrator import nodes
from aero.orchestrator.state import Briefing, TraceEvent
from aero.orchestrator.tools import EvidenceTools

T = TypeVar("T")

OUT_OF_SCOPE_PROMPTS = ("scope_plan",)
PROMPTS = ("scope_plan", "brief_write", "verify_claim")


@dataclass
class AgentDeps:
    llm: LLM
    verifier: Verifier
    tools: EvidenceTools
    intents: Sequence[str]


def _timed(trace: list[TraceEvent], node: str, fn: Callable[[], T], detail: Callable[[T], dict[str, Any]]) -> T:
    started = datetime.now(UTC)
    t0 = time.perf_counter()
    result = fn()
    trace.append(TraceEvent(node=node, started_at=started, duration_ms=int((time.perf_counter() - t0) * 1000),
                            detail=detail(result)))
    return result


def run_briefing(question: str, deps: AgentDeps) -> Briefing:
    trace: list[TraceEvent] = []
    registry = default_registry()
    scope = _timed(trace, "plan", lambda: nodes.plan(question, deps.llm, deps.intents),
                   lambda s: {"in_scope": s.in_scope, "intent": s.intent, "sub_questions": s.sub_questions})
    if not scope.in_scope:
        return Briefing(question=question, status="out_of_scope", scope=scope, review_reasons=[scope.reason],
                        prompts=[registry.get(p).ref for p in OUT_OF_SCOPE_PROMPTS], model=deps.llm.model,
                        trace=trace)
    results = _timed(trace, "research", lambda: nodes.research(scope.sub_questions, deps.tools),
                     lambda rs: {"per_sub_question": [(r.sub_question, len(r.hits), r.independent_publishers,
                                                       r.rounds) for r in rs]})
    evidence = nodes.pooled_evidence(results)
    claims, open_questions = _timed(trace, "write", lambda: nodes.write(question, evidence, deps.llm),
                                    lambda out: {"claims": len(out[0]), "open_questions": len(out[1])})
    by_id = {h.chunk_id: h for h in evidence}
    checks = _timed(trace, "verify", lambda: nodes.verify_claims(claims, by_id, deps.verifier),
                    lambda cs: {"statuses": [c.status for c in cs]})
    briefing = nodes.gate(question, scope, results, checks, by_id, open_questions)
    briefing.prompts = [registry.get(p).ref for p in PROMPTS]
    briefing.model = deps.llm.model
    briefing.trace = trace
    return briefing

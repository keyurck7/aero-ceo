"""
Typed state for the AERO-CEO deterministic LangGraph workflow.

Every node receives this state and returns updates to it.
This makes the agent workflow inspectable and deterministic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # Input
    goal: str
    run_id: str

    # Goal understanding
    intent: str
    intent_confidence: float
    topic: str
    topic_confidence: float
    business_area: str
    scope_status: str
    domain_relevance: float
    rejection_reason: str

    # Planning
    plan: List[Dict[str, Any]]

    # Tool execution trace
    tool_trace: List[Dict[str, Any]]

    # Retrieved and analyzed information
    evidence: List[Dict[str, Any]]
    signals: List[Dict[str, Any]]
    risks: List[Dict[str, Any]]
    opportunities: List[Dict[str, Any]]
    trends: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    partners: List[Dict[str, Any]]

    # Decision and output
    decision: Dict[str, Any]
    validation: Dict[str, Any]
    briefing: str
    status: str

    # Error handling
    error: Optional[str]

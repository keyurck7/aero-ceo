from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


INTENT_RULES = {
    "partnership_strategy": [
        "collab", "collaboration", "collaborate", "partner", "partnership",
        "joint venture", "alliance", "work with", "team up"
    ],
    "risk_analysis": [
        "risk", "threat", "danger", "weakness", "problem", "delay",
        "uncertainty", "what could go wrong", "concern"
    ],
    "opportunity_analysis": [
        "opportunity", "growth", "expand", "investment", "invest",
        "market", "new business", "advantage"
    ],
    "recommendation_drilldown": [
        "explain recommendation", "why recommend", "recommendation",
        "support this", "evidence for", "why should"
    ],
    "scenario_analysis": [
        "what if", "scenario", "suppose", "if fcas", "if budget",
        "if dassault", "if delayed", "under condition"
    ],
    "competitor_analysis": [
        "competitor", "dassault", "boeing", "lockheed", "bae",
        "leonardo", "saab", "thales", "rheinmetall"
    ],
    "evidence_request": [
        "evidence", "sources", "proof", "show me", "where did",
        "cite", "supporting documents"
    ],
}


TOPIC_RULES = {
    "FCAS": [
        "fcas", "future combat air system", "sixth-generation",
        "sixth generation", "future fighter", "fighter jet"
    ],
    "Eurofighter": [
        "eurofighter", "typhoon"
    ],
    "Uncrewed Combat Aircraft": [
        "uncrewed", "unmanned", "uav", "uas", "drone", "eurodrone",
        "sirtap", "collaborative combat aircraft", "loyal wingman"
    ],
    "Military Space": [
        "space", "satellite", "secure communications", "earth observation",
        "military satellite"
    ],
    "Aerial Refuelling": [
        "a330 mrtt", "mrtt", "tanker", "refuelling", "refueling"
    ],
    "Military Transport": [
        "a400m", "c295", "cn235", "transport aircraft"
    ],
    "Supply Chain": [
        "supply chain", "supplier", "delivery", "shortage", "production",
        "manufacturing"
    ],
    "European Defence Autonomy": [
        "european defence", "european defense", "sovereign", "sovereignty",
        "strategic autonomy", "nato", "europe"
    ],
    "Policy and Procurement": [
        "procurement", "contract", "ministry", "government", "budget",
        "export", "regulation", "policy"
    ],
}


BUSINESS_AREA_RULES = {
    "Airbus Defence and Space": [
        "defence", "defense", "fighter", "military", "space", "satellite",
        "fcas", "eurofighter", "a400m", "mrtt", "drone", "sirtap"
    ],
    "Airbus Commercial Aircraft": [
        "commercial", "airline", "a320", "a350", "a330neo", "passenger",
        "aircraft delivery"
    ],
    "Airbus Helicopters": [
        "helicopter", "h145", "h160", "rotorcraft"
    ],
    "Airbus Corporate / Group Strategy": [
        "airbus as a company", "group", "corporate", "overall", "company"
    ],
}


REGION_RULES = {
    "Europe": ["europe", "european", "eu", "nato"],
    "Germany": ["germany", "german", "bundeswehr"],
    "France": ["france", "french"],
    "Spain": ["spain", "spanish"],
    "United Kingdom": ["uk", "britain", "british", "united kingdom"],
    "Italy": ["italy", "italian"],
    "United States": ["united states", "usa", "us ", "american"],
}


CONSTRAINT_RULES = {
    "low_budget": ["low budget", "limited budget", "cheap", "cost effective", "lower cost"],
    "low_risk": ["low risk", "safe", "conservative", "less risky"],
    "urgent": ["urgent", "quick", "immediate", "short term", "near term", "now"],
    "long_term": ["long term", "future", "2030", "2035", "sixth-generation"],
    "partnership_required": ["with partner", "collaborate", "collaboration", "partner"],
}


@dataclass
class QueryRoute:
    question: str
    intent: str
    topic: Optional[str]
    business_area: Optional[str]
    region: Optional[str]
    constraints: List[str]
    needs_recommendation: bool
    needs_risk_assessment: bool
    needs_evidence: bool


def _first_match(text: str, rules: Dict[str, List[str]]) -> Optional[str]:
    lower = text.lower()
    for label, keywords in rules.items():
        if any(keyword in lower for keyword in keywords):
            return label
    return None


def _all_matches(text: str, rules: Dict[str, List[str]]) -> List[str]:
    lower = text.lower()
    matches = []
    for label, keywords in rules.items():
        if any(keyword in lower for keyword in keywords):
            matches.append(label)
    return matches


def route_query(question: str) -> Dict:
    lower = question.lower()

    intent = _first_match(question, INTENT_RULES) or "general_strategy"
    topic = _first_match(question, TOPIC_RULES)
    business_area = _first_match(question, BUSINESS_AREA_RULES) or "Airbus Defence and Space"
    region = _first_match(question, REGION_RULES)
    constraints = _all_matches(question, CONSTRAINT_RULES)

    needs_risk = intent in {
        "risk_analysis",
        "scenario_analysis",
        "partnership_strategy",
        "recommendation_drilldown",
        "general_strategy",
    } or "risk" in lower

    needs_evidence = True

    needs_recommendation = intent in {
        "general_strategy",
        "partnership_strategy",
        "opportunity_analysis",
        "risk_analysis",
        "scenario_analysis",
        "competitor_analysis",
        "recommendation_drilldown",
    }

    route = QueryRoute(
        question=question,
        intent=intent,
        topic=topic,
        business_area=business_area,
        region=region,
        constraints=constraints,
        needs_recommendation=needs_recommendation,
        needs_risk_assessment=needs_risk,
        needs_evidence=needs_evidence,
    )

    return asdict(route)

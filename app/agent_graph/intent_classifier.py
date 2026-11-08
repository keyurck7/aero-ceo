"""
Deterministic embedding-based goal classifier for AERO-CEO.

This avoids a simple keyword router. It classifies the CEO goal by comparing
the query embedding with intent and topic prototype embeddings.

Same query + same model + same prototypes = deterministic classification.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()


INTENT_PROTOTYPES: Dict[str, List[str]] = {
    "ceo_briefing": [
        "If you were the CEO today what would you do next and why?",
        "Give me an executive briefing with what happened why it matters and what management should do.",
        "Summarize the strategic situation and recommend CEO actions.",
    ],
    "risk_analysis": [
        "What are the biggest risks for Airbus?",
        "What threats should management monitor?",
        "What could go wrong for this business area?",
        "Analyze delivery, partner, supply chain, regulatory, and competitive risks.",
    ],
    "opportunity_analysis": [
        "What opportunities should Airbus prioritize?",
        "Where should Airbus invest for growth?",
        "Which technologies or markets create upside?",
        "Identify product, market, partnership, and technology opportunities.",
    ],
    "partnership_strategy": [
        "Which European organization should Airbus collaborate with?",
        "Should Airbus partner with Dassault, Indra, Thales, Leonardo, Saab, or Rheinmetall?",
        "Analyze strategic partnership options for future combat systems.",
    ],
    "scenario_analysis": [
        "What if FCAS is delayed?",
        "What if drone demand accelerates?",
        "What if defence budgets are limited?",
        "Analyze a future scenario and recommend actions.",
    ],
    "recommendation_drilldown": [
        "Explain this recommendation in detail.",
        "Give a lower-risk version of the recommendation.",
        "Show strongest evidence for the recommendation.",
        "What could go wrong with this recommendation?",
    ],
    "evidence_request": [
        "Show me evidence.",
        "Which sources support this?",
        "Find documents about this topic.",
        "Retrieve evidence from the knowledge base.",
    ],
}


TOPIC_PROTOTYPES: Dict[str, List[str]] = {
    "FCAS": [
        "Future Combat Air System",
        "sixth generation fighter system",
        "combat cloud future fighter FCAS",
        "France Germany Spain future air combat programme",
    ],
    "Eurofighter": [
        "Eurofighter modernization",
        "Typhoon fighter upgrade",
        "combat aircraft production bridge",
    ],
    "Uncrewed Systems": [
        "uncrewed aircraft",
        "drones",
        "UAS",
        "SIRTAP",
        "collaborative combat aircraft",
        "crewed uncrewed teaming",
    ],
    "Military Space": [
        "military satellite",
        "secure communications",
        "space defence",
        "earth observation defence",
    ],
    "Military Transport": [
        "A400M",
        "C295",
        "military transport aircraft",
        "air mobility",
    ],
    "Aerial Refuelling": [
        "A330 MRTT",
        "aerial refuelling",
        "multi-role tanker transport",
        "tanker aircraft",
    ],
    "Supply Chain": [
        "supply chain",
        "delivery delays",
        "production ramp-up",
        "industrial execution",
        "supplier risk",
    ],
    "European Defence Autonomy": [
        "European defence autonomy",
        "European sovereignty",
        "EU defence procurement",
        "NATO Europe defence industrial base",
    ],
    "Competitor Radar": [
        "competitor activity",
        "Dassault",
        "Leonardo",
        "BAE Systems",
        "Rheinmetall",
        "Saab",
        "Thales",
        "Boeing",
        "Lockheed Martin",
    ],
}


BUSINESS_AREA_BY_TOPIC = {
    "FCAS": "Airbus Defence and Space",
    "Eurofighter": "Airbus Defence and Space",
    "Uncrewed Systems": "Airbus Defence and Space",
    "Military Space": "Airbus Defence and Space",
    "Military Transport": "Airbus Defence and Space",
    "Aerial Refuelling": "Airbus Defence and Space",
    "Supply Chain": "Airbus Group / Defence and Space",
    "European Defence Autonomy": "Airbus Defence and Space",
    "Competitor Radar": "Airbus Group Strategy",
}


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load the embedding model once.

    Uses the same EMBEDDING_MODEL from .env when available.
    """
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    return SentenceTransformer(model_name)


def _embed(texts: List[str]) -> np.ndarray:
    model = get_embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype="float32")


def _best_match(query: str, prototypes: Dict[str, List[str]]) -> Tuple[str, float]:
    labels = list(prototypes.keys())

    prototype_texts = []
    prototype_labels = []

    for label, examples in prototypes.items():
        for example in examples:
            prototype_texts.append(example)
            prototype_labels.append(label)

    all_texts = [query] + prototype_texts
    vectors = _embed(all_texts)

    query_vec = vectors[0]
    prototype_vecs = vectors[1:]

    scores = prototype_vecs @ query_vec

    best_idx = int(np.argmax(scores))
    best_label = prototype_labels[best_idx]
    best_score = float(scores[best_idx])

    # Aggregate by label for more stable classification.
    label_scores: Dict[str, List[float]] = {label: [] for label in labels}
    for label, score in zip(prototype_labels, scores):
        label_scores[label].append(float(score))

    averaged = {
        label: max(values) * 0.7 + (sum(values) / len(values)) * 0.3
        for label, values in label_scores.items()
    }

    best_label = max(averaged, key=averaged.get)
    best_score = float(averaged[best_label])

    return best_label, round(best_score, 4)



DOMAIN_SCOPE_PROTOTYPES = [
    "Airbus Defence and Space strategic decision",
    "Airbus CEO should decide on defence strategy",
    "FCAS future combat air system strategy",
    "Eurofighter modernization opportunity and risk",
    "uncrewed aircraft drones SIRTAP military systems",
    "military space secure communications Airbus",
    "A400M C295 MRTT military transport and refuelling",
    "European defence autonomy and aerospace strategy",
    "Airbus supply chain delivery production risk",
    "defence partnership with Dassault Indra Thales Leonardo Saab Rheinmetall",
]

OFF_DOMAIN_PROTOTYPES = [
    "NLP exam",
    "machine learning homework",
    "write a poem",
    "weather forecast",
    "football match",
    "cooking recipe",
    "personal birthday party",
    "random university exam question",
    "general chatbot question unrelated to Airbus",
    "software installation problem",
]

DOMAIN_KEYWORDS = {
    "airbus", "defence", "defense", "aerospace", "aviation", "fcas", "eurofighter",
    "fighter", "drone", "drones", "uncrewed", "uas", "sirtap", "space", "satellite",
    "military", "a400m", "c295", "mrtt", "refuelling", "refueling", "supply", "chain",
    "procurement", "european", "autonomy", "sovereignty", "partner", "partnership",
    "dassault", "indra", "thales", "leonardo", "saab", "rheinmetall", "ohb",
    "risk", "opportunity", "strategy", "strategic", "ceo", "executive",
}


def _max_similarity(query: str, examples: List[str]) -> float:
    texts = [query] + examples
    vectors = _embed(texts)
    query_vec = vectors[0]
    example_vecs = vectors[1:]
    scores = example_vecs @ query_vec
    return float(np.max(scores))


def _domain_scope_check(goal: str) -> Dict[str, object]:
    goal_l = (goal or "").lower()
    tokens = {t.strip(".,?!:;()[]{}") for t in goal_l.split()}
    keyword_hits = sorted(tokens.intersection(DOMAIN_KEYWORDS))

    domain_score = _max_similarity(goal, DOMAIN_SCOPE_PROTOTYPES)
    off_domain_score = _max_similarity(goal, OFF_DOMAIN_PROTOTYPES)

    # Deterministic decision:
    # 1. Domain keyword hit is strong evidence of scope.
    # 2. Otherwise, semantic similarity to Airbus/defence scope must beat off-domain examples.
    # 3. Short vague prompts are rejected unless they contain domain terms.
    is_short = len(goal_l.split()) <= 3

    if keyword_hits:
        scope_status = "in_scope"
        reason = f"Domain keywords found: {', '.join(keyword_hits[:8])}"
    elif is_short:
        scope_status = "out_of_scope"
        reason = "Goal is too short and has no Airbus/defence/strategy domain terms."
    elif domain_score >= 0.42 and domain_score >= off_domain_score:
        scope_status = "in_scope"
        reason = "Goal is semantically close to Airbus strategic intelligence scope."
    else:
        scope_status = "out_of_scope"
        reason = (
            "Goal does not appear related to Airbus, aerospace, defence, military systems, "
            "strategic recommendations, risks, opportunities, or CEO decision support."
        )

    return {
        "scope_status": scope_status,
        "domain_relevance": round(domain_score, 4),
        "off_domain_similarity": round(off_domain_score, 4),
        "domain_keyword_hits": keyword_hits,
        "rejection_reason": reason,
    }


def classify_goal(goal: str) -> Dict[str, object]:
    """
    Classify CEO goal into scope, intent, topic, and business area.

    First checks whether the query belongs to AERO-CEO's Airbus strategic
    intelligence domain. This prevents forced mapping of out-of-scope prompts
    like "NLP exam" into irrelevant Airbus topics.
    """
    goal = (goal or "").strip()
    if not goal:
        goal = "Give me a CEO briefing for Airbus."

    scope = _domain_scope_check(goal)

    # If out of scope, still return safe defaults so downstream code has fields.
    if scope["scope_status"] == "out_of_scope":
        return {
            "scope_status": "out_of_scope",
            "domain_relevance": scope["domain_relevance"],
            "off_domain_similarity": scope["off_domain_similarity"],
            "domain_keyword_hits": scope["domain_keyword_hits"],
            "rejection_reason": scope["rejection_reason"],
            "intent": "out_of_scope",
            "intent_confidence": 0.0,
            "topic": "Out of scope",
            "topic_confidence": 0.0,
            "business_area": "AERO-CEO Scope Guard",
        }

    intent, intent_confidence = _best_match(goal, INTENT_PROTOTYPES)
    topic, topic_confidence = _best_match(goal, TOPIC_PROTOTYPES)

    business_area = BUSINESS_AREA_BY_TOPIC.get(topic, "Airbus Group Strategy")

    return {
        "scope_status": "in_scope",
        "domain_relevance": scope["domain_relevance"],
        "off_domain_similarity": scope["off_domain_similarity"],
        "domain_keyword_hits": scope["domain_keyword_hits"],
        "rejection_reason": scope["rejection_reason"],
        "intent": intent,
        "intent_confidence": intent_confidence,
        "topic": topic,
        "topic_confidence": topic_confidence,
        "business_area": business_area,
    }

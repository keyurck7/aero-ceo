from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Optional


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def recency_score(published_at: Optional[str]) -> float:
    if not published_at:
        return 0.55

    try:
        dt = parsedate_to_datetime(published_at)
        if dt.tzinfo:
            dt = dt.replace(tzinfo=None)

        age_days = max((datetime.utcnow() - dt).days, 0)

        if age_days <= 30:
            return 1.0
        if age_days <= 90:
            return 0.85
        if age_days <= 180:
            return 0.70
        if age_days <= 365:
            return 0.55
        return 0.40
    except Exception:
        return 0.55


def confidence_score(
    trust_score: float,
    published_at: Optional[str],
    evidence_strength: float,
    source_type: str,
) -> float:
    source_bonus = {
        "official": 0.10,
        "news": 0.03,
        "policy": 0.08,
        "competitor": 0.04,
        "community": -0.05,
    }.get(source_type, 0.0)

    score = (
        0.40 * clamp(trust_score)
        + 0.25 * recency_score(published_at)
        + 0.25 * clamp(evidence_strength)
        + source_bonus
    )

    return round(clamp(score), 3)


def impact_score(topic: str, signal_type: str, text: str) -> float:
    topic_weights = {
        "FCAS": 0.92,
        "Uncrewed Combat Aircraft": 0.90,
        "Eurofighter": 0.84,
        "Military Space": 0.82,
        "Aerial Refuelling": 0.76,
        "Military Transport": 0.72,
        "European Defence Autonomy": 0.88,
        "Policy and Procurement": 0.78,
        "Competitor Activity": 0.80,
        "Supply Chain": 0.74,
    }

    base = topic_weights.get(topic, 0.62)

    lower = text.lower()
    boost_terms = [
        "contract",
        "procurement",
        "nato",
        "ministry of defence",
        "sixth-generation",
        "future combat",
        "autonomous",
        "satellite",
        "strategic",
        "sovereign",
        "combat cloud",
    ]

    boost = sum(0.02 for term in boost_terms if term in lower)
    type_bonus = 0.03 if signal_type in ["opportunity", "risk"] else 0.0

    return round(clamp(base + boost + type_bonus), 3)


def urgency_score(text: str, published_at: Optional[str]) -> float:
    lower = text.lower()

    urgency_terms = [
        "urgent",
        "delay",
        "risk",
        "dispute",
        "contract",
        "announced",
        "signed",
        "launches",
        "procurement",
        "deadline",
        "2026",
        "2027",
        "2028",
        "2029",
    ]

    score = 0.45 + sum(0.04 for term in urgency_terms if term in lower)
    score += 0.20 * recency_score(published_at)

    return round(clamp(score), 3)

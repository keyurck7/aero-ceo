"""Strategic taxonomy loaded from ``seeds/taxonomy.yaml``.

v1 hard-coded the topic, intent, business-unit and region keyword lists in
``query_router.py`` and ``cleaner.py`` (and the two copies had drifted apart).
v2 keeps one versioned file. ``infer_topics`` is multi-label and matches on word
boundaries: v1 took the first substring match, so "us " matched inside "focus ".
The orchestrator's LLM classifier (phase 3) outputs labels from this same file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

DEFAULT_TOPIC = "General Airbus Strategy"


@dataclass(frozen=True)
class Taxonomy:
    version: str
    topics: dict[str, list[str]]
    business_units: dict[str, list[str]] = field(default_factory=dict)
    regions: dict[str, list[str]] = field(default_factory=dict)
    intents: dict[str, list[str]] = field(default_factory=dict)
    constraints: dict[str, list[str]] = field(default_factory=dict)
    # Names removed before topic matching: "Airbus Defence and Space" is a company,
    # not evidence about military space (v1 tagged every Airbus article "Military Space").
    ignore_phrases: list[str] = field(default_factory=list)
    # Prior impact per topic (v1 scoring.impact_score). A prior, not a measurement.
    topic_impact_weights: dict[str, float] = field(default_factory=dict)
    business_unit_topics: dict[str, list[str]] = field(default_factory=dict)
    umbrella_business_units: list[str] = field(default_factory=list)

    def topic_names(self) -> list[str]:
        return list(self.topics)

    def business_unit_for(self, topic: str, text: str = "") -> str | None:
        """Most specific unit owning ``topic``; else best keyword match on ``text`` (umbrella units last)."""
        owners = [bu for bu, topics in self.business_unit_topics.items() if topic in topics]
        specific = [bu for bu in owners if bu not in self.umbrella_business_units]
        if specific:
            return specific[0]
        labels = [label for label, _ in match_labels(strip_ignored(text, self), self.business_units)] if text else []
        labels.sort(key=lambda bu: bu in self.umbrella_business_units)  # stable: specific first
        return labels[0] if labels else (owners[0] if owners else None)


def _compile(keywords: list[str]) -> re.Pattern[str]:
    escaped = sorted((re.escape(k.strip().lower()) for k in keywords if k.strip()), key=len, reverse=True)
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(escaped) + r")(?![a-z0-9])")


@lru_cache(maxsize=64)
def _pattern(keywords: tuple[str, ...]) -> re.Pattern[str]:
    return _compile(list(keywords))


def match_labels(text: str, rules: dict[str, list[str]]) -> list[tuple[str, int]]:
    """All labels whose keywords occur in ``text``, ordered by hit count (desc)."""
    lower = text.lower()
    scored = []
    for label, keywords in rules.items():
        if not keywords:
            continue
        hits = len(_pattern(tuple(keywords)).findall(lower))
        if hits:
            scored.append((label, hits))
    scored.sort(key=lambda item: (-item[1], list(rules).index(item[0])))
    return scored


def strip_ignored(text: str, taxonomy: Taxonomy) -> str:
    if not taxonomy.ignore_phrases:
        return text
    return _pattern(tuple(taxonomy.ignore_phrases)).sub(" ", text.lower())


def infer_topics(text: str, taxonomy: Taxonomy, max_topics: int = 3) -> list[str]:
    labels = [label for label, _ in match_labels(strip_ignored(text, taxonomy), taxonomy.topics)[:max_topics]]
    return labels or [DEFAULT_TOPIC]


def load_taxonomy(path: str | Path) -> Taxonomy:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Taxonomy(
        version=str(data.get("version", "unversioned")),
        topics=data["topics"],
        business_units=data.get("business_units", {}),
        regions=data.get("regions", {}),
        intents=data.get("intents", {}),
        constraints=data.get("constraints", {}),
        ignore_phrases=data.get("ignore_phrases", []),
        topic_impact_weights={str(k): float(v) for k, v in data.get("topic_impact_weights", {}).items()},
        business_unit_topics=data.get("business_unit_topics", {}),
        umbrella_business_units=data.get("umbrella_business_units", []),
    )

"""Partner registry (replaces ``PARTNER_CAPABILITY_MAP`` hard-coded in ``ceo_chat_agent.py``).

The registry itself is analyst-maintained data (``seeds/partners.yaml``; git is the audit
trail). What v2 adds is **evidence**: v1 counted ``LIKE '%name%'`` matches; here
mentions are retrieved passages in which a partner alias occurs as a whole word,
returned with citation metadata.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from aero.domain import EvidenceChunk, SearchFilters
from aero.storage import Store

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"and", "the", "of", "for", "in", "with", "to", "a", "an", "systems", "system"}


class Partner(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    country: str
    capabilities: list[str]
    collaboration_areas: list[str] = Field(default_factory=list)
    known_risks: list[str] = Field(default_factory=list)
    provenance: str = ""


class PartnerMatch(BaseModel):
    partner: Partner
    score: float
    capability_overlap: list[str]
    country_match: bool
    evidence_mentions: int
    rationale: str


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


class PartnerRegistry:
    def __init__(self, partners: Sequence[Partner], version: str = "") -> None:
        self.partners = list(partners)
        self.version = version
        self._by_name = {a.lower(): p for p in self.partners for a in {p.name, *p.aliases}}

    @classmethod
    def load(cls, path: str | Path) -> PartnerRegistry:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls([Partner(**p) for p in data["partners"]], str(data.get("version", "")))

    def get(self, name: str) -> Partner | None:
        return self._by_name.get(name.strip().lower())

    def search(
        self, country: str | None = None, capability: str | None = None, query: str | None = None
    ) -> list[Partner]:
        out = []
        for p in self.partners:
            if country and p.country.lower() != country.lower():
                continue
            blob = " ".join([p.name, *p.capabilities, *p.collaboration_areas]).lower()
            if capability and capability.lower() not in blob:
                continue
            if query and not _tokens(query) & _tokens(blob):
                continue
            out.append(p)
        return out

    def mentions(self, store: Store, name: str, limit: int = 10, min_credibility: float = 0.15) -> list[EvidenceChunk]:
        partner = self.get(name)
        if partner is None:
            raise KeyError(name)
        aliases = sorted({partner.name, *partner.aliases}, key=len, reverse=True)
        pattern = re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(re.escape(a) for a in aliases) + r")(?![A-Za-z0-9])",
                             re.IGNORECASE)
        ids: dict[int, None] = {}
        for alias in aliases:
            for chunk_id, _ in store.lexical_search(alias, 50, SearchFilters(min_credibility=min_credibility)):
                ids.setdefault(chunk_id, None)
        evidence = store.get_evidence(list(ids))
        hits = [e for e in evidence.values() if pattern.search(e.text)]  # FTS stems; require the exact name
        hits.sort(key=lambda e: (e.published_at is None, -(e.published_at.timestamp() if e.published_at else 0)))
        return hits[:limit]

    def match(self, store: Store | None, need: str, country: str | None = None, limit: int = 5) -> list[PartnerMatch]:
        """Rank partners for a capability need: 0.6 capability overlap + 0.2 country + 0.2 evidence mentions."""
        need_tokens = _tokens(need)
        matches: list[PartnerMatch] = []
        for p in self.partners:
            offered = _tokens(" ".join([*p.capabilities, *p.collaboration_areas]))
            overlap = sorted(need_tokens & offered)
            cap = len(overlap) / max(len(need_tokens), 1)
            country_ok = bool(country) and p.country.lower() == (country or "").lower()
            mentions = len(self.mentions(store, p.name, limit=50)) if store is not None else 0
            score = 0.6 * cap + 0.2 * float(country_ok) + 0.2 * min(mentions / 5, 1.0)
            if score == 0:
                continue
            why = [f"capability overlap {overlap}" if overlap else "no direct capability overlap"]
            if country_ok:
                why.append(f"based in {p.country}")
            why.append(f"{mentions} evidence passages mention {p.name}")
            matches.append(PartnerMatch(partner=p, score=round(score, 3), capability_overlap=overlap,
                                        country_match=country_ok, evidence_mentions=mentions, rationale="; ".join(why)))
        matches.sort(key=lambda m: (-m.score, m.partner.name))
        return matches[:limit]

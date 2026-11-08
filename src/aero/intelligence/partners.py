"""Partner registry and evidence-based partner ranking (ADR 0008, replaces the 0.6/0.2/0.2 formula).

The registry is analyst-maintained data (``seeds/partners.yaml``; git is the audit trail). Ranking a need:

1. **Filter**: a *required* country removes partners outright. It is a constraint, not a weight. A preferred
   country or any other preference is passed to the judges as text.
2. **Assess** each partner with a rubric (LLM, versioned prompt ``partner_assess``): capability fit
   strong/partial/none citing the registry's own capability strings, track record documented/mentioned/none
   citing evidence chunk ids, and the risks that matter for this need. Citations are checked: an invented
   capability or chunk id is removed and reported. Fit "none" is excluded.
3. **Compare** every remaining pair twice, A-vs-B and B-vs-A (``partner_compare``), so position bias cancels.
4. **Rank** with Bradley-Terry fitted to those games (``aero.intelligence.ranking``).

What the caller sees are facts: rank, win/loss/tie record, how many pairs were judged consistently in both
orders, the cited evidence and the judges' reasons. Validated against analyst rankings with Kendall's tau
(``aero eval partners``).
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from aero.domain import RELIABILITY_GRADES, EvidenceChunk, SearchFilters
from aero.intelligence.ranking import Game, bradley_terry
from aero.llm import LLM, PromptRegistry, default_registry, generate
from aero.llm.concurrency import pmap
from aero.storage import Store

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"and", "the", "of", "for", "in", "with", "to", "a", "an", "systems", "system"}
DEFAULT_GRADES = [g for g in RELIABILITY_GRADES if g != "E"]

CapabilityFit = Literal["strong", "partial", "none"]
TrackRecord = Literal["documented", "mentioned", "none"]


class Partner(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    country: str
    capabilities: list[str]
    collaboration_areas: list[str] = Field(default_factory=list)
    known_risks: list[str] = Field(default_factory=list)
    provenance: str = ""


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
        """Filter the registry (exact country, capability substring, any shared word). No scoring."""
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

    def mentions(self, store: Store, name: str, limit: int = 10, query: str | None = None) -> list[EvidenceChunk]:
        """Passages in which the partner's name or an alias occurs as a whole word.

        Ordered by BM25 relevance to ``"<alias> <query>"`` when a query is given, else most recent first.
        Grade-E sources and quarantined documents are excluded (default policy).
        """
        partner = self.get(name)
        if partner is None:
            raise KeyError(name)
        aliases = sorted({partner.name, *partner.aliases}, key=len, reverse=True)
        pattern = re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(re.escape(a) for a in aliases) + r")(?![A-Za-z0-9])",
                             re.IGNORECASE)
        filters = SearchFilters(source_reliability=DEFAULT_GRADES)
        ids: dict[int, None] = {}
        for alias in aliases:
            for chunk_id, _ in store.lexical_search(f"{alias} {query}" if query else alias, 50, filters):
                ids.setdefault(chunk_id, None)
        evidence = store.get_evidence(list(ids))
        hits = [evidence[c] for c in ids if c in evidence and pattern.search(evidence[c].text)]  # exact name
        if not query:
            hits.sort(key=lambda e: (e.published_at is None, -(e.published_at.timestamp() if e.published_at else 0)))
        return hits[:limit]


# ----------------------------------------------------------------------------- LLM wire models


class _AssessOutput(BaseModel):
    capability_fit: CapabilityFit
    matched_capabilities: list[str]
    track_record: TrackRecord
    evidence_chunk_ids: list[int]
    risks: list[str]
    rationale: str


class _CompareOutput(BaseModel):
    better: Literal["A", "B", "tie"]
    reason: str


# ----------------------------------------------------------------------------- results


class PartnerAssessment(BaseModel):
    partner: str
    capability_fit: CapabilityFit
    matched_capabilities: list[str]
    track_record: TrackRecord
    evidence: list[EvidenceChunk]
    risks: list[str]
    rationale: str
    removed_citations: list[str] = Field(default_factory=list,
                                         description="Capabilities or chunk ids the judge cited that do not exist")


class PairJudgement(BaseModel):
    first: str
    second: str
    better: Literal["first", "second", "tie"]
    reason: str


class Comparison(BaseModel):
    a: str
    b: str
    judgements: list[PairJudgement] = Field(description="The same pair judged in both orders")
    outcome: str = Field(description="Partner name, or 'tie' (includes order-inconsistent judgements)")
    consistent: bool


class RankedPartner(BaseModel):
    rank: int
    partner: Partner
    assessment: PartnerAssessment
    wins: float
    losses: float
    ties: float
    bt_log_strength: float = Field(description="Bradley-Terry log-strength fitted to the pairwise games")


class Exclusion(BaseModel):
    partner: str
    reason: str


class PartnerRanking(BaseModel):
    need: str
    required_country: str | None
    preferences: str
    method: str
    ranked: list[RankedPartner]
    excluded: list[Exclusion]
    comparisons: list[Comparison]
    consistent_pairs: int
    total_pairs: int
    prompts: list[str]
    model: str
    registry_version: str


class PartnerRanker:
    METHOD = ("required-country filter -> rubric assessment (cited) -> exclude fit 'none' -> every pair judged in "
              "both orders -> Bradley-Terry fit")

    def __init__(self, llm: LLM, registry: PartnerRegistry, store: Store, *, prompts: PromptRegistry | None = None,
                 evidence_per_partner: int = 6, max_workers: int = 8) -> None:
        self.llm = llm
        self.registry = registry
        self.store = store
        prompts = prompts or default_registry()
        self.assess_prompt = prompts.get("partner_assess")
        self.compare_prompt = prompts.get("partner_compare")
        self.evidence_per_partner = evidence_per_partner
        self.max_workers = max_workers

    @staticmethod
    def _constraints(required_country: str | None, preferences: str) -> str:
        parts = []
        if required_country:
            parts.append(f"Required: partner based in {required_country}.")
        if preferences.strip():
            parts.append(f"Preferences: {preferences.strip()}")
        return "\n".join(parts) or "No further constraints."

    def assess(self, need: str, partner: Partner, constraints: str) -> PartnerAssessment:
        evidence = self.registry.mentions(self.store, partner.name, limit=self.evidence_per_partner, query=need)
        rendered = "\n".join(
            f'<passage chunk_id="{e.chunk_id}" publisher="{e.publisher}" grade="{e.source_reliability}" '
            f'published="{e.published_at.date().isoformat() if e.published_at else "unknown"}">{e.text}</passage>'
            for e in evidence) or "(no passages mention this partner)"
        out, _ = generate(self.llm, self.assess_prompt, _AssessOutput, {
            "need": need, "constraints": constraints, "partner": partner.name, "country": partner.country,
            "capabilities": "; ".join(partner.capabilities),
            "collaboration_areas": "; ".join(partner.collaboration_areas) or "none listed",
            "known_risks": "; ".join(partner.known_risks) or "none listed", "evidence": rendered,
        })
        known_caps = {c.lower(): c for c in partner.capabilities}
        by_id = {e.chunk_id: e for e in evidence}
        removed = [c for c in out.matched_capabilities if c.lower() not in known_caps]
        removed += [f"chunk {i}" for i in out.evidence_chunk_ids if i not in by_id]
        caps = [known_caps[c.lower()] for c in out.matched_capabilities if c.lower() in known_caps]
        cited = [by_id[i] for i in dict.fromkeys(out.evidence_chunk_ids) if i in by_id]
        fit = out.capability_fit
        if fit != "none" and not caps:
            fit = "none"  # fail closed: a fit claim must cite at least one real capability
            removed.append("fit downgraded to 'none': no existing capability cited")
        track = out.track_record if cited else "none"
        return PartnerAssessment(partner=partner.name, capability_fit=fit, matched_capabilities=caps,
                                 track_record=track, evidence=cited, risks=out.risks, rationale=out.rationale,
                                 removed_citations=removed)

    @staticmethod
    def _dossier(p: Partner, a: PartnerAssessment) -> str:
        ev = "\n".join(f"  [chunk {e.chunk_id}, {e.publisher}, grade {e.source_reliability}] {e.text[:600]}"
                       for e in a.evidence) or "  none"
        return (f"Name: {p.name} ({p.country})\nCapability fit: {a.capability_fit}; matched: "
                f"{'; '.join(a.matched_capabilities) or 'none'}\nTrack record: {a.track_record}\nCited evidence:\n"
                f"{ev}\nRisks: {'; '.join(a.risks) or 'none stated'}\nAssessor's rationale: {a.rationale}")

    def _judge(self, need: str, constraints: str, first: tuple[Partner, PartnerAssessment],
               second: tuple[Partner, PartnerAssessment]) -> PairJudgement:
        out, _ = generate(self.llm, self.compare_prompt, _CompareOutput, {
            "need": need, "constraints": constraints, "candidate_a": self._dossier(*first),
            "candidate_b": self._dossier(*second)})
        better: Literal["first", "second", "tie"] = {"A": "first", "B": "second"}.get(out.better, "tie")  # type: ignore[assignment]
        return PairJudgement(first=first[0].name, second=second[0].name, better=better, reason=out.reason)

    def rank(self, need: str, required_country: str | None = None, preferences: str = "",
             limit: int = 5) -> PartnerRanking:
        constraints = self._constraints(required_country, preferences)
        excluded: list[Exclusion] = []
        pool: list[Partner] = []
        for p in self.registry.partners:
            if required_country and p.country.lower() != required_country.lower():
                excluded.append(Exclusion(partner=p.name, reason=f"not based in {required_country} (required)"))
            else:
                pool.append(p)
        assessments = pmap(lambda p: self.assess(need, p, constraints), pool, self.max_workers)
        eligible: list[tuple[Partner, PartnerAssessment]] = []
        for p, a in zip(pool, assessments, strict=True):
            if a.capability_fit == "none":
                excluded.append(Exclusion(partner=p.name, reason="capability fit 'none' for this need"))
            else:
                eligible.append((p, a))

        pairs = list(itertools.combinations(range(len(eligible)), 2))
        ordered = [(i, j) for i, j in pairs] + [(j, i) for i, j in pairs]
        judgements = pmap(lambda ij: self._judge(need, constraints, eligible[ij[0]], eligible[ij[1]]), ordered,
                          self.max_workers)
        verdict = dict(zip(ordered, judgements, strict=True))
        games: list[Game] = []
        comparisons: list[Comparison] = []
        record = {i: [0.0, 0.0, 0.0] for i in range(len(eligible))}  # wins, losses, ties (pair level)
        for i, j in pairs:
            fwd, rev = verdict[(i, j)], verdict[(j, i)]
            for first, second, jdg in ((i, j, fwd), (j, i, rev)):
                games.append(Game(first, second, {"first": 1.0, "second": 0.0, "tie": 0.5}[jdg.better]))
            i_wins = (fwd.better == "first") + (rev.better == "second")
            j_wins = (fwd.better == "second") + (rev.better == "first")
            consistent = (fwd.better == "tie" and rev.better == "tie") or i_wins == 2 or j_wins == 2
            if consistent and i_wins != j_wins:
                winner, loser = (i, j) if i_wins > j_wins else (j, i)
                outcome = eligible[winner][0].name
                record[winner][0] += 1
                record[loser][1] += 1
            else:
                outcome = "tie"
                record[i][2] += 1
                record[j][2] += 1
            comparisons.append(Comparison(a=eligible[i][0].name, b=eligible[j][0].name, judgements=[fwd, rev],
                                          outcome=outcome, consistent=consistent))

        strengths = bradley_terry(len(eligible), games)
        order = sorted(range(len(eligible)), key=lambda k: (-strengths[k], eligible[k][0].name))
        ranked: list[RankedPartner] = []
        for pos, k in enumerate(order):
            tied_with_previous = pos > 0 and abs(strengths[k] - strengths[order[pos - 1]]) < 1e-9
            rank = ranked[-1].rank if tied_with_previous else pos + 1
            ranked.append(RankedPartner(rank=rank, partner=eligible[k][0], assessment=eligible[k][1],
                                        wins=record[k][0], losses=record[k][1], ties=record[k][2],
                                        bt_log_strength=round(strengths[k], 4)))
        return PartnerRanking(
            need=need, required_country=required_country, preferences=preferences, method=self.METHOD,
            ranked=ranked[:limit], excluded=excluded, comparisons=comparisons,
            consistent_pairs=sum(c.consistent for c in comparisons), total_pairs=len(comparisons),
            prompts=[self.assess_prompt.ref, self.compare_prompt.ref], model=self.llm.model,
            registry_version=self.registry.version,
        )

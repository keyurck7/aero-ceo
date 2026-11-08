"""Claim verification: does a passage support a claim? (ADR 0008, H10/H11)

One judge, four uses:
- signal extraction (a signal is stored only if its own passage supports it);
- corroboration (do *other* publishers support or contradict it? -> Admiralty information credibility);
- recommendation evidence (candidate links become ``supports`` / ``contradicts``);
- briefing claims in the phase-3 orchestrator (unsupported claims block publication).

The judge is an LLM with a versioned prompt and a JSON schema. Its verdict counts only if the quote it gives
is found verbatim in the passage (fail closed). Agreement with human labels is measured by
``aero eval judge`` (Cohen's kappa) before the verdicts gate anything in production.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from aero.domain import CorroborationItem, EvidenceChunk
from aero.llm import LLM, PromptRegistry, default_registry, generate

Verdict = Literal["supported", "contradicted", "not_enough_info"]

_QUOTE_CHARS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-",
                              "—": "-", " ": " "})


class _JudgeOutput(BaseModel):
    verdict: Verdict
    quote: str
    reason: str


class Verification(BaseModel):
    verdict: Verdict
    quote: str
    reason: str
    grounded: bool
    prompt: str
    model: str


def find_quote(text: str, quote: str) -> tuple[int, int] | None:
    """Character span of ``quote`` in ``text``: exact, else modulo whitespace runs and typographic quotes/dashes."""
    quote = quote.strip()
    if not quote:
        return None
    i = text.find(quote)
    if i >= 0:
        return i, i + len(quote)
    norm_chars: list[str] = []
    index: list[int] = []
    prev_space = False
    for pos, ch in enumerate(text.translate(_QUOTE_CHARS)):
        if ch.isspace():
            if prev_space:
                continue
            ch, prev_space = " ", True
        else:
            prev_space = False
        norm_chars.append(ch)
        index.append(pos)
    norm_text = "".join(norm_chars)
    norm_quote = " ".join(quote.translate(_QUOTE_CHARS).split())
    j = norm_text.find(norm_quote)
    if j < 0:
        return None
    return index[j], index[j + len(norm_quote) - 1] + 1


class Verifier:
    def __init__(self, llm: LLM, registry: PromptRegistry | None = None) -> None:
        self.llm = llm
        self.prompt = (registry or default_registry()).get("verify_claim")

    def verify(self, claim: str, passage: EvidenceChunk) -> Verification:
        out, completion = generate(self.llm, self.prompt, _JudgeOutput, {
            "claim": claim, "passage": passage.text, "publisher": passage.publisher,
            "published_at": passage.published_at.date().isoformat() if passage.published_at else "unknown",
            "content_level": passage.content_level,
        })
        if out.verdict != "not_enough_info" and find_quote(passage.text, out.quote) is None:
            return Verification(verdict="not_enough_info", quote="", grounded=False, prompt=self.prompt.ref,
                                model=completion.model,
                                reason=f"judge said {out.verdict} but its quote is not in the passage: {out.reason}")
        return Verification(verdict=out.verdict, quote=out.quote, reason=out.reason, grounded=True,
                            prompt=self.prompt.ref, model=completion.model)


def information_credibility(items: Sequence[CorroborationItem]) -> int:
    """Admiralty 1-6 from corroboration facts only (plan §3): 1 confirmed, 5 contradicted, otherwise 6.

    Grades 2-4 require an analyst's judgement of logic and consistency and are never assigned automatically.
    """
    supported = {i.publisher_group for i in items if i.verdict == "supported"}
    contradicted = {i.publisher_group for i in items if i.verdict == "contradicted"}
    if supported and not contradicted:
        return 1
    if contradicted and not supported:
        return 5
    return 6

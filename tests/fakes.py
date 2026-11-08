"""A scripted stand-in for Qwen, for unit tests of the plumbing around the LLM (ADR 0008).

It does NOT test whether a real model reads direction correctly: that is measured on Data Lab against the
signal gold set (``aero eval signals``). It does test that every guard works: schema validation, quote
grounding, coherence, self-consistency majority, verification, citation checks and the pairwise ranking.
"""

from __future__ import annotations

import re
from typing import Any

from aero.llm.client import LLMRequest
from aero.llm.fake import FakeLLM, variables_of

_CHUNK = re.compile(r'<chunk id="(\d+)">\n(.*?)\n</chunk>', re.DOTALL)
_PASSAGE = re.compile(r"<passage[^>]*>\n(.*?)\n</passage>", re.DOTALL)
_WORD = re.compile(r"[a-z]{4,}")

# trigger phrase -> the signal a careful analyst would extract
SCRIPT: list[tuple[str, dict[str, Any]]] = [
    ("workshare dispute could delay", {
        "signal_type": "risk", "topic": "FCAS", "direction_for_airbus": "negative", "event_status": "speculative",
        "materiality": "high", "title": "FCAS workshare dispute threatens demonstrator schedule",
        "claim": "A workshare dispute could delay the FCAS fighter demonstrator beyond 2029.",
        "quote": "Industry sources said the workshare dispute could delay the sixth-generation fighter "
                 "demonstrator beyond 2029.",
        "materiality_rationale": "Schedule risk on a programme Airbus co-leads."}),
    ("maiden flight", {
        "signal_type": "opportunity", "topic": "Uncrewed Combat Aircraft", "direction_for_airbus": "positive",
        "event_status": "completed", "materiality": "medium", "title": "SIRTAP completes maiden flight",
        "claim": "The SIRTAP uncrewed aerial system completed its maiden flight in Spain.",
        "quote": "The SIRTAP tactical uncrewed aerial system completed its maiden flight in Spain.",
        "materiality_rationale": "Programme milestone."}),
    ("secure military satellite communications", {
        "signal_type": "opportunity", "topic": "Military Space", "direction_for_airbus": "positive",
        "event_status": "awarded", "materiality": "high", "title": "Bundeswehr secure satcom contract",
        "claim": "Airbus signed a secure military satellite communications contract for the Bundeswehr.",
        "quote": "Airbus Defence and Space signed a contract for secure military satellite communications for the "
                 "Bundeswehr.",
        "materiality_rationale": "Contract award."}),
    ("electronic warfare suite", {
        "signal_type": "trend", "topic": "Eurofighter", "direction_for_airbus": "neutral",
        "event_status": "cancelled", "materiality": "low", "title": "Eurofighter upgrade cancelled",
        "claim": "The Eurofighter electronic warfare upgrade was cancelled.",  # wrong: the verifier must catch it
        "quote": "The Eurofighter Typhoon will receive a new electronic warfare suite and radar upgrades.",
        "materiality_rationale": "Upgrade."}),
]


def _chunk_for(chunks: list[tuple[str, str]], phrase: str) -> int | None:
    return next((int(cid) for cid, text in chunks if phrase in text), None)


def extract(req: LLMRequest) -> dict[str, Any]:
    text = variables_of(req)
    chunks = _CHUNK.findall(text)
    sample = req.seed or 0
    signals: list[dict[str, Any]] = []
    topics: list[str] = []
    for phrase, spec in SCRIPT:
        cid = _chunk_for(chunks, phrase)
        if cid is not None:
            signals.append({"chunk_id": cid, "business_unit": None, **spec})
            topics.append(spec["topic"])
    fcas = _chunk_for(chunks, "workshare dispute could delay")
    if fcas is not None:
        # incoherent: an "opportunity" that is bad news for Airbus
        signals.append({**SCRIPT[0][1], "chunk_id": fcas, "signal_type": "opportunity", "topic": "FCAS"})
        # paraphrased quote: not in the text
        signals.append({**SCRIPT[0][1], "chunk_id": fcas, "signal_type": "trend", "topic": "European Defence Autonomy",
                        "quote": "Germany and Spain support Airbus.", "direction_for_airbus": "positive"})
    space = _chunk_for(chunks, "sovereign space capabilities")
    if space is not None and sample == 0:  # only one of three samples sees it -> no majority
        signals.append({**SCRIPT[2][1], "chunk_id": space, "signal_type": "risk", "topic": "Supply Chain",
                        "direction_for_airbus": "negative",
                        "quote": "The award strengthens European sovereign space capabilities."})
    if sample == 1 and signals:  # one sample disagrees on materiality of the first signal
        signals[0] = {**signals[0], "materiality": "medium"}
    for s in signals:
        s.pop("business_unit", None)
    return {"document_topics": topics, "signals": signals}


def verify(req: LLMRequest) -> dict[str, Any]:
    text = variables_of(req)
    claim = text.split("CLAIM:", 1)[1].split("\n", 1)[0].strip().lower()
    passage = _PASSAGE.search(text)
    body = passage.group(1) if passage else ""
    first = body.split(". ")[0].strip()
    if "cancel" in claim and "will receive" in body:
        return {"verdict": "contradicted", "quote": first, "reason": "The passage says the upgrade will happen."}
    overlap = set(_WORD.findall(claim)) & set(_WORD.findall(body.lower()))
    if len(overlap) >= 3:
        return {"verdict": "supported", "quote": first, "reason": f"Shared facts: {sorted(overlap)[:4]}"}
    return {"verdict": "not_enough_info", "quote": "", "reason": "Different subject."}


def _need_words(text: str) -> set[str]:
    need = text.split("NEED:", 1)[1].split("\n", 1)[0].lower()
    return set(_WORD.findall(need))


def assess(req: LLMRequest) -> dict[str, Any]:
    text = variables_of(req)
    caps_line = re.search(r"Capabilities: (.*)", text)
    caps = caps_line.group(1).split("; ") if caps_line else []
    need = _need_words(text)
    matched = [c for c in caps if set(_WORD.findall(c.lower())) & need]
    ids = [int(i) for i in re.findall(r'chunk_id="(\d+)"', text)]
    name = re.search(r'<dossier name="([^"]+)"', text)
    hallucinated = ["quantum teleportation"] if name and name.group(1) == "Indra" else []
    fit = "strong" if len(matched) >= 2 else "partial" if matched else "none"
    return {"capability_fit": fit, "matched_capabilities": matched + hallucinated,
            "track_record": "documented" if ids else "none", "evidence_chunk_ids": [*ids, 999_999],
            "risks": [], "rationale": f"{len(matched)} capabilities match the need."}


def compare(req: LLMRequest) -> dict[str, Any]:
    text = variables_of(req)
    a, b = text.split('<candidate id="B">', 1)

    def n_matched(block: str) -> int:
        m = re.search(r"matched: (.*)", block)
        return 0 if not m or m.group(1).strip() == "none" else len(m.group(1).split("; "))

    na, nb = n_matched(a), n_matched(b)
    if na == nb:
        return {"better": "A", "reason": "position bias (equal evidence)"}  # the swap must neutralise this
    return {"better": "A" if na > nb else "B", "reason": f"{max(na, nb)} vs {min(na, nb)} matched capabilities"}


def scope(req: LLMRequest) -> dict[str, Any]:
    question = variables_of(req).split("QUESTION:", 1)[1].strip().lower()
    if "exam" in question or "homework" in question:
        return {"in_scope": False, "intent": "other", "reason": "Not about Airbus strategy.", "sub_questions": []}
    return {"in_scope": True, "intent": "other", "reason": "Airbus programme question.",
            "sub_questions": ["workshare dispute fighter demonstrator", "Eurofighter electronic warfare suite"]}


_EVIDENCE = re.compile(r'<passage id="C(\d+)"[^>]*>(.*?)</passage>', re.DOTALL)


def brief(req: LLMRequest) -> dict[str, Any]:
    passages = {int(i): t for i, t in _EVIDENCE.findall(variables_of(req))}
    fcas = next((i for i, t in passages.items() if "workshare dispute could delay" in t), None)
    ew = next((i for i, t in passages.items() if "electronic warfare suite" in t), None)
    claims: list[dict[str, Any]] = []
    if fcas is not None:
        claims += [
            {"text": "A workshare dispute could delay the FCAS fighter demonstrator beyond 2029.", "kind": "fact",
             "citations": [fcas]},
            {"text": "The FCAS demonstrator has slipped to 2031 because of the dispute.", "kind": "fact",
             "citations": [fcas]},  # invented number
            {"text": "Airbus should seek a larger FCAS workshare.", "kind": "assessment", "citations": [999_999]},
        ]
    if ew is not None:
        claims.append({"text": "The Eurofighter electronic warfare upgrade was cancelled.", "kind": "fact",
                       "citations": [ew]})
    return {"claims": claims, "open_questions": ["The cost of a delay is not in the evidence."]}


ROUTES = {"signal_extract": extract, "verify_claim": verify, "partner_assess": assess, "partner_compare": compare,
          "scope_plan": scope, "brief_write": brief}


def scripted_llm() -> FakeLLM:
    return FakeLLM(lambda req: ROUTES[req.prompt_id](req), model="fake-qwen")

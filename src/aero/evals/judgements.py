"""Evaluations for every model judgement (ADR 0008 release gates, plan §7).

Gold files live in ``evals/datasets/``. Only rows with ``reviewed_by`` set (a human) count towards the gates;
draft rows can be included for smoke tests with ``include_unreviewed=True``.

* signals: ``signal_gold.jsonl``, one row per document keyed by ``canonical_url`` (stable across databases).
  Reports P/R/F1 of (document, type, topic) against gold for the LLM extractor *and* the keyword baseline,
  direction and event-status accuracy on matched signals, and weighted kappa for materiality.
* judge: ``verification_gold.jsonl`` with (claim, passage, label). Reports Cohen's kappa and a confusion matrix.
* partners: ``partner_rankings.jsonl`` with an analyst's ordering per need. Reports Kendall's tau-b per need.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aero.domain import EvidenceChunk, SignalFilters
from aero.evals.baselines import BaselineSignal
from aero.evals.metrics import cohen_kappa, prf, ranking_tau
from aero.intelligence.partners import PartnerRanker
from aero.intelligence.verification import Verifier
from aero.storage import Store

MATERIALITY_LEVELS = ["low", "medium", "high"]
VERDICTS = ["supported", "contradicted", "not_enough_info"]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _reviewed(rows: Sequence[dict[str, Any]], include_unreviewed: bool) -> list[dict[str, Any]]:
    return [r for r in rows if include_unreviewed or r.get("reviewed_by")]


def _doc_index(store: Store) -> dict[str, int]:
    out = {}
    for doc_id in store.list_document_ids():
        doc = store.get_document(doc_id)
        if doc is not None:
            out[doc.canonical_url] = doc_id
    return out


# ----------------------------------------------------------------------------- signals


def evaluate_signals(store: Store, gold_rows: Sequence[dict[str, Any]], extractor: str,
                     baseline: Sequence[BaselineSignal] | None = None,
                     include_unreviewed: bool = False) -> dict[str, Any]:
    rows = _reviewed(gold_rows, include_unreviewed)
    docs = _doc_index(store)
    missing = [r["canonical_url"] for r in rows if r["canonical_url"] not in docs]
    rows = [r for r in rows if r["canonical_url"] in docs]
    doc_ids = {docs[r["canonical_url"]] for r in rows}
    url_of = {v: k for k, v in docs.items()}

    gold: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in rows:
        for g in r.get("signals", []):
            gold[(r["canonical_url"], g["signal_type"], g["topic"])] = g
    predicted: dict[tuple[str, str, str], Any] = {}
    for s in store.list_signals(SignalFilters(extractor=extractor), limit=1_000_000):
        if s.document_id in doc_ids:
            predicted.setdefault((url_of[s.document_id], s.signal_type, s.topic), s)

    matched = sorted(set(gold) & set(predicted))
    report: dict[str, Any] = {
        "documents": len(rows), "documents_missing_from_db": missing, "gold_signals": len(gold),
        "extractor": extractor, "llm": prf(set(predicted), set(gold)).__dict__, "matched": len(matched),
    }
    if matched:
        report["direction_accuracy"] = round(
            sum(predicted[k].direction == gold[k]["direction"] for k in matched) / len(matched), 4)
        report["event_status_accuracy"] = round(
            sum(predicted[k].event_status == gold[k]["event_status"] for k in matched) / len(matched), 4)
        report["materiality_kappa_quadratic"] = round(cohen_kappa(
            [predicted[k].materiality for k in matched], [gold[k]["materiality"] for k in matched],
            labels=MATERIALITY_LEVELS, weights="quadratic"), 4)
        report["direction_confusion"] = dict(Counter(
            f"{gold[k]['direction']}->{predicted[k].direction}" for k in matched))
    if baseline is not None:
        base_keys = {(url_of[b.document_id], b.signal_type, b.topic) for b in baseline if b.document_id in doc_ids}
        report["rule_baseline"] = prf(base_keys, set(gold)).__dict__
    return report


def sample_signal_documents(store: Store, n: int, seed: int = 20261108) -> list[dict[str, Any]]:
    """Stratified sample for labelling: round-robin over (content_level, first topic) strata, seeded."""
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    for doc_id in store.list_document_ids():
        doc = store.get_document(doc_id)
        if doc is not None and not doc.injection_flags:
            strata[(doc.content_level, doc.topics[0] if doc.topics else "General")].append(doc_id)
    rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not security
    for ids in strata.values():
        rng.shuffle(ids)
    keys = sorted(strata)
    chosen: list[int] = []
    while len(chosen) < n and any(strata.values()):
        for key in keys:
            if strata[key] and len(chosen) < n:
                chosen.append(strata[key].pop())
    out: list[dict[str, Any]] = []
    for doc_id in chosen:
        doc = store.get_document(doc_id)
        assert doc is not None
        out.append({"canonical_url": doc.canonical_url, "title": doc.title, "content_level": doc.content_level,
                    "published_at": doc.published_at.isoformat() if doc.published_at else None,
                    "text": doc.clean_text[:3000], "signals": [], "reviewed_by": None,
                    "label_help": "signals: [{signal_type, topic, direction, event_status, materiality}]"})
    return out


# ----------------------------------------------------------------------------- judge (claim verification)


def adhoc_chunk(text: str, publisher: str = "unknown", content_level: str = "full_text") -> EvidenceChunk:
    now = datetime.now(UTC)
    return EvidenceChunk(chunk_id=0, document_id=0, chunk_index=0, text=text, char_start=0, char_end=len(text),
                         title="", url="", canonical_url="", publisher=publisher, source_type="unknown",
                         licence="unknown", source_reliability="F", published_at=None, collected_at=now, topics=[],
                         content_level=content_level)


def evaluate_judge(verifier: Verifier, gold_rows: Sequence[dict[str, Any]],
                   include_unreviewed: bool = False) -> dict[str, Any]:
    rows = _reviewed(gold_rows, include_unreviewed)
    if not rows:
        return {"items": 0, "note": "no reviewed rows"}
    human, judge = [], []
    for r in rows:
        v = verifier.verify(r["claim"], adhoc_chunk(r["passage"], r.get("publisher", "unknown"),
                                                    r.get("content_level", "full_text")))
        human.append(r["label"])
        judge.append(v.verdict)
    return {
        "items": len(rows), "prompt": verifier.prompt.ref, "model": verifier.llm.model,
        "cohen_kappa": round(cohen_kappa(judge, human, labels=VERDICTS), 4),
        "accuracy": round(sum(a == b for a, b in zip(judge, human, strict=True)) / len(rows), 4),
        "confusion_human_to_judge": dict(Counter(f"{h}->{j}" for h, j in zip(human, judge, strict=True))),
    }


# ----------------------------------------------------------------------------- partners


def evaluate_partners(ranker: PartnerRanker, gold_rows: Sequence[dict[str, Any]],
                      include_unreviewed: bool = False) -> dict[str, Any]:
    rows = _reviewed(gold_rows, include_unreviewed)
    per_need = []
    for r in rows:
        result = ranker.rank(r["need"], required_country=r.get("required_country"),
                             preferences=r.get("preferences", ""), limit=len(ranker.registry.partners))
        system = [rp.partner.name for rp in result.ranked]
        per_need.append({"need": r["need"], "tau_b": round(ranking_tau(system, r["ranking"]), 4), "system": system,
                         "gold": r["ranking"], "consistent_pairs": result.consistent_pairs,
                         "total_pairs": result.total_pairs})
    taus = [p["tau_b"] for p in per_need]
    return {"needs": len(per_need), "mean_tau_b": round(sum(taus) / len(taus), 4) if taus else None,
            "per_need": per_need}

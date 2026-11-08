"""Retrieval evaluation against labelled relevance judgements (qrels).

v1's "retrieval probes" passed when a query word appeared in the results. The v1
templates had written query words into the documents, so the probes measured
nothing (finding P10). v2 measures ranking quality against human labels:

* Labels are per **document** (keyed by canonical URL): they survive re-chunking
  and re-embedding, unlike chunk IDs.
* Metrics: capped Recall@k, Precision@k, MRR, nDCG@k with graded gains (0/1/2).
* Labelling workflow (TREC-style pooling): ``aero eval pool`` writes the top-k
  candidates per query to a JSONL file. You set ``grade`` (0 = irrelevant,
  1 = partially, 2 = directly relevant) and run ``aero eval retrieval``.

qrels line: {"qid": "rq001", "query": "...", "filters": {...}, "relevant": {"https://...": 2}}
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aero.domain import SearchFilters
from aero.retrieval import HybridRetriever


@dataclass
class QueryJudgement:
    qid: str
    query: str
    relevant: dict[str, int]
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryScore:
    qid: str
    query: str
    recall: float
    precision: float
    mrr: float
    ndcg: float
    retrieved: list[str]
    missed: list[str]


def load_qrels(path: str | Path) -> list[QueryJudgement]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out.append(QueryJudgement(row["qid"], row["query"], {k: int(v) for k, v in row["relevant"].items()},
                                      row.get("filters", {})))
    return out


def ranked_documents(retriever: HybridRetriever, query: str, k: int, filters: dict[str, Any]) -> list[str]:
    """Top-k distinct documents (canonical URLs) in rank order."""
    res = retriever.search(query, top_k=max(k * 4, 20), filters=SearchFilters(**filters))
    docs: dict[str, None] = {}
    for hit in res.hits:
        docs.setdefault(hit.canonical_url, None)
        if len(docs) >= k:
            break
    return list(docs)


def score_ranking(ranked: list[str], relevant: dict[str, int], k: int) -> tuple[float, float, float, float]:
    rel = {d for d, g in relevant.items() if g > 0}
    top = ranked[:k]
    hits = [d for d in top if d in rel]
    # Capped recall (BEIR "recall_cap"): hits / min(|relevant|, k), so a query with 30 relevant
    # documents can still reach 1.0 at k=10 and the CI gate stays meaningful.
    recall = len(hits) / min(len(rel), k) if rel else 0.0
    precision = len(hits) / k
    mrr = next((1.0 / (i + 1) for i, d in enumerate(top) if d in rel), 0.0)
    dcg = sum((2 ** relevant.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(top))
    ideal = sorted((g for g in relevant.values() if g > 0), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return recall, precision, mrr, (dcg / idcg if idcg else 0.0)


def evaluate(retriever: HybridRetriever, qrels: Iterable[QueryJudgement], k: int = 10) -> dict[str, Any]:
    scores: list[QueryScore] = []
    for q in qrels:
        ranked = ranked_documents(retriever, q.query, k, q.filters)
        r, p, m, n = score_ranking(ranked, q.relevant, k)
        missed = [d for d, g in q.relevant.items() if g > 0 and d not in ranked]
        scores.append(QueryScore(q.qid, q.query, r, p, m, n, ranked, missed))
    count = len(scores) or 1
    summary = {
        f"recall@{k}": sum(s.recall for s in scores) / count,
        f"precision@{k}": sum(s.precision for s in scores) / count,
        "mrr": sum(s.mrr for s in scores) / count,
        f"ndcg@{k}": sum(s.ndcg for s in scores) / count,
        "queries": len(scores),
    }
    return {"k": k, "summary": {key: round(v, 4) if isinstance(v, float) else v for key, v in summary.items()},
            "per_query": [asdict(s) for s in scores]}


def build_pool(retriever: HybridRetriever, queries: Iterable[dict[str, Any]], k: int = 20) -> list[dict[str, Any]]:
    """Candidates to label. Each query's top-k documents, grade left as null."""
    pool = []
    for q in queries:
        res = retriever.search(q["query"], top_k=k * 3, filters=SearchFilters(**q.get("filters", {})))
        seen: dict[str, dict[str, Any]] = {}
        for h in res.hits:
            if h.canonical_url not in seen and len(seen) < k:
                seen[h.canonical_url] = {"canonical_url": h.canonical_url, "title": h.title, "publisher": h.publisher,
                                         "published_at": h.published_at.isoformat() if h.published_at else None,
                                         "snippet": h.text[:300], "grade": None}
        pool.append({"qid": q["qid"], "query": q["query"], "filters": q.get("filters", {}),
                     "candidates": list(seen.values())})
    return pool


def qrels_from_pool(pool_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in pool_rows:
        graded = {c["canonical_url"]: int(c["grade"]) for c in row["candidates"] if c.get("grade") is not None}
        if graded:
            out.append(
                {"qid": row["qid"], "query": row["query"], "filters": row.get("filters", {}), "relevant": graded}
            )
    return out

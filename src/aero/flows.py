"""Pipeline entry points (replace the step list in ``scripts/rebuild_demo.sh``).

Plain functions with explicit parameters, each recorded as a ``pipeline_runs``
row with stats and errors. Prefect (roadmap phase 5) will wrap these as tasks.
They are deliberately not written against Prefect today, so they stay testable
and runnable from the CLI.
"""

from __future__ import annotations

import traceback
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from aero.config import Settings
from aero.domain import SourceIn
from aero.embeddings import Embedder
from aero.ingestion.collectors import SourcePlan, collect_news, collect_official, load_source_plan
from aero.ingestion.http import PoliteFetcher
from aero.ingestion.migrate_v1 import migrate_v1
from aero.ingestion.pipeline import Ingestor, RawItem
from aero.intelligence import strategy
from aero.intelligence.signals import (
    ExtractionReport,
    LLMSignalExtractor,
    load_materiality_rubric,
    run_corroboration,
    run_extraction,
)
from aero.intelligence.verification import Verifier
from aero.llm import LLM, build_llm
from aero.log import get_logger
from aero.retrieval import HybridRetriever
from aero.storage import Store
from aero.taxonomy import load_taxonomy

log = get_logger(__name__)


def _plan(settings: Settings) -> SourcePlan:
    return load_source_plan(Path(settings.seeds_dir) / "sources.yaml")


def seed_sources(store: Store, settings: Settings) -> int:
    plan = _plan(settings)
    for src in plan.sources:
        store.upsert_source(plan.graded(src))
    return len(plan.sources)


def grade_sources(store: Store, settings: Settings) -> Counter[str]:
    """Apply the analyst's Admiralty categories (seeds/sources.yaml) to every source in the database.

    The file is the source of truth: a publisher it does not list becomes 'unreviewed' (grade F).
    """
    plan = _plan(settings)
    configured = {s.name.lower(): s for s in plan.sources}
    stats: Counter[str] = Counter()
    for src in store.list_sources():
        current = SourceIn(**src.model_dump(include=set(SourceIn.model_fields)))
        base = configured.get(src.name.lower()) or current.model_copy(
            update={"category": "unreviewed", "reliability_note": ""})
        updated = store.upsert_source(plan.graded(base))
        stats[f"grade_{updated.reliability}"] += 1
    return stats


def _run(store: Store, settings: Settings, embedder: Embedder, flow: str, params: dict[str, Any],
         items: Iterable[RawItem] | None = None, v1_path: str | None = None) -> dict[str, Any]:
    run_id = store.start_run(flow, params)
    ingestor = Ingestor(store, embedder, load_taxonomy(Path(settings.seeds_dir) / "taxonomy.yaml"), settings)
    stats: Counter[str] = Counter()
    try:
        if v1_path is not None:
            stats.update(migrate_v1(v1_path, ingestor, _plan(settings), run_id=run_id))
        for item in items or []:
            res = ingestor.ingest(item, run_id=run_id)
            stats[res.outcome if res.outcome != "duplicate" else f"duplicate_{res.detail}"] += 1
        stats["embedded_chunks"] = ingestor.embed_pending()
        # Signal extraction needs the LLM and is a separate, explicit job: `aero signals extract`.
        stats["injection_flagged"] = ingestor.counts["injection_flagged"]
        store.finish_run(run_id, "succeeded", dict(stats))
    except Exception as exc:
        store.finish_run(run_id, "failed", dict(stats), error="".join(traceback.format_exception(exc))[-4000:])
        raise
    log.info("%s finished: %s", flow, dict(stats))
    return {"run_id": run_id, **stats}


def ingest_official(store: Store, settings: Settings, embedder: Embedder, limit: int = 40) -> dict[str, Any]:
    plan = _plan(settings)
    items = collect_official(PoliteFetcher.from_settings(settings), plan, limit=limit)
    return _run(store, settings, embedder, "ingest_official", {"limit": limit, "plan": plan.version}, items)


def ingest_news(store: Store, settings: Settings, embedder: Embedder, per_query: int = 20, limit: int = 200,
                fetch_articles: bool = True) -> dict[str, Any]:
    plan = _plan(settings)
    items = collect_news(PoliteFetcher.from_settings(settings), plan, per_query=per_query, limit=limit,
                         fetch_articles=fetch_articles)
    params = {"per_query": per_query, "limit": limit, "fetch_articles": fetch_articles, "plan": plan.version,
              "full_text_policy": plan.full_text_policy}
    return _run(store, settings, embedder, "ingest_news", params, items)


def import_v1(store: Store, settings: Settings, embedder: Embedder, v1_path: str) -> dict[str, Any]:
    return _run(store, settings, embedder, "import_v1", {"path": v1_path}, v1_path=v1_path)


def signal_extractor(settings: Settings, llm: LLM | None = None, judge: LLM | None = None) -> LLMSignalExtractor:
    seeds = Path(settings.seeds_dir)
    return LLMSignalExtractor(
        llm or build_llm(settings), Verifier(judge or build_llm(settings, judge=True)),
        load_taxonomy(seeds / "taxonomy.yaml"), load_materiality_rubric(seeds / "materiality_rubric.yaml"),
        samples=settings.signal_samples, temperature=settings.signal_sample_temperature,
    )


def extract_signals(store: Store, settings: Settings, document_ids: Iterable[int] | None = None,
                    llm: LLM | None = None, judge: LLM | None = None,
                    rejections_path: str | Path | None = None) -> ExtractionReport:
    """LLM extraction + verification for every document (or ``document_ids``). Audited as a pipeline run."""
    extractor = signal_extractor(settings, llm, judge)
    run_id = store.start_run("extract_signals", {"extractor": extractor.extractor_id, "model": extractor.llm.model,
                                                 "samples": extractor.samples})
    try:
        report = run_extraction(store, extractor, document_ids, max_workers=settings.llm_max_concurrency)
    except Exception as exc:
        store.finish_run(run_id, "failed", {}, error="".join(traceback.format_exception(exc))[-4000:])
        raise
    store.finish_run(run_id, "succeeded", report.summary())
    if rejections_path:
        import json

        Path(rejections_path).parent.mkdir(parents=True, exist_ok=True)
        Path(rejections_path).write_text("".join(
            json.dumps({"document_id": r.document_id, "reason": r.reason, **r.detail}, ensure_ascii=False,
                       default=str) + "\n" for r in report.rejections), encoding="utf-8")
    return report


def corroborate_signals(store: Store, settings: Settings, retriever: HybridRetriever,
                        judge: LLM | None = None) -> Counter[str]:
    verifier = Verifier(judge or build_llm(settings, judge=True))
    return run_corroboration(store, retriever, verifier, depth=settings.corroboration_depth,
                             max_workers=settings.llm_max_concurrency)


def verify_recommendation(store: Store, settings: Settings, recommendation_id: int,
                          judge: LLM | None = None) -> Counter[str]:
    return strategy.verify_evidence(store, Verifier(judge or build_llm(settings, judge=True)), recommendation_id)


def seed_recommendations(store: Store, settings: Settings, retriever: HybridRetriever | None) -> Counter[str]:
    seeds = Path(settings.seeds_dir)
    recs = strategy.load_recommendation_seeds(seeds / "recommendations_v1.yaml",
                                              load_taxonomy(seeds / "taxonomy.yaml"), seeds / "signal_rules_v1.yaml")
    return strategy.seed_recommendations(store, retriever, recs)


def embed_pending(store: Store, settings: Settings, embedder: Embedder) -> int:
    ingestor = Ingestor(store, embedder, load_taxonomy(Path(settings.seeds_dir) / "taxonomy.yaml"), settings)
    return ingestor.embed_pending()

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
from aero.embeddings import Embedder
from aero.ingestion.collectors import SourcePlan, collect_news, collect_official, load_source_plan
from aero.ingestion.http import PoliteFetcher
from aero.ingestion.migrate_v1 import migrate_v1
from aero.ingestion.pipeline import Ingestor, RawItem
from aero.log import get_logger
from aero.storage import Store
from aero.taxonomy import load_taxonomy

log = get_logger(__name__)


def _plan(settings: Settings) -> SourcePlan:
    return load_source_plan(Path(settings.seeds_dir) / "sources.yaml")


def seed_sources(store: Store, settings: Settings) -> int:
    plan = _plan(settings)
    for src in plan.sources:
        store.upsert_source(src)
    return len(plan.sources)


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


def embed_pending(store: Store, settings: Settings, embedder: Embedder) -> int:
    ingestor = Ingestor(store, embedder, load_taxonomy(Path(settings.seeds_dir) / "taxonomy.yaml"), settings)
    return ingestor.embed_pending()

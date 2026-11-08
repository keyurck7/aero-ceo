"""``aero``: one command-line entry point for v2 (replaces the nine ``scripts/*.sh`` wrappers).

    aero db migrate | stats
    aero seed sources
    aero ingest official|news|v1 PATH
    aero embed
    aero search "query" [--topic T] [--source-type official] [--json]
    aero signals extract
    aero seed recommendations [--no-link]
    aero rec list | show ID | approve ID | retire ID  --reviewer NAME --reason TEXT
    aero serve evidence|signals|strategy|partners|all [--stdio]
    aero eval pool --queries Q.jsonl --out pool.jsonl
    aero eval labels --pool pool.jsonl --out qrels.jsonl
    aero eval retrieval --qrels qrels.jsonl [--min-recall 0.8]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from aero.config import Settings, get_settings
from aero.log import configure_logging


def _store(settings: Settings, migrate: bool = False) -> Any:
    from aero.storage import open_store

    return open_store(settings, migrate=migrate)


def _retriever(settings: Settings, store: Any) -> Any:
    from aero.embeddings import build_embedder
    from aero.rerank import build_reranker
    from aero.retrieval import HybridRetriever, RetrievalConfig

    return HybridRetriever(store, build_embedder(settings), build_reranker(settings),
                           RetrievalConfig.from_settings(settings))


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str, ensure_ascii=False))


def cmd_db(args: argparse.Namespace, s: Settings) -> int:
    with _store(s) as store:
        if args.action == "migrate":
            applied = store.migrate()
            print(f"applied migrations: {applied or 'none (up to date)'}")
        else:
            _print(store.stats().model_dump())
    return 0


def cmd_seed(args: argparse.Namespace, s: Settings) -> int:
    from aero.flows import seed_sources

    with _store(s, migrate=True) as store:
        print(f"upserted {seed_sources(store, s)} sources from {s.seeds_dir}/sources.yaml")
    return 0


def cmd_ingest(args: argparse.Namespace, s: Settings) -> int:
    from aero import flows
    from aero.embeddings import build_embedder

    embedder = build_embedder(s)
    with _store(s, migrate=True) as store:
        flows.seed_sources(store, s)
        if args.kind == "official":
            result = flows.ingest_official(store, s, embedder, limit=args.limit)
        elif args.kind == "news":
            result = flows.ingest_news(store, s, embedder, per_query=args.per_query, limit=args.limit,
                                       fetch_articles=not args.headlines_only)
        else:
            result = flows.import_v1(store, s, embedder, args.path)
    _print(result)
    return 0


def cmd_embed(args: argparse.Namespace, s: Settings) -> int:
    from aero.embeddings import build_embedder
    from aero.flows import embed_pending

    with _store(s) as store:
        print(f"embedded {embed_pending(store, s, build_embedder(s))} chunks")
    return 0


def cmd_search(args: argparse.Namespace, s: Settings) -> int:
    from aero.domain import SearchFilters

    with _store(s) as store:
        res = _retriever(s, store).search(
            args.query, top_k=args.top_k,
            filters=SearchFilters(topics=args.topic or None, source_types=args.source_type or None),
        )
    if args.json:
        print(res.model_dump_json(indent=2))
        return 0
    print(f"{len(res.hits)} hits · {res.independent_publishers} independent publishers")
    for h in res.hits:
        date = h.published_at.date().isoformat() if h.published_at else "undated"
        print(f"\n#{h.rank} [{h.scores.final:.3f}] {h.title} — {h.publisher} ({h.source_type}, {date})")
        print(f"   chunk {h.chunk_id} · {h.url}")
        if h.injection_flags:
            print(f"   ⚠ injection flags: {h.injection_flags}")
        print("   " + h.text[:300].replace("\n", " ") + ("…" if len(h.text) > 300 else ""))
    return 0


def cmd_signals(args: argparse.Namespace, s: Settings) -> int:
    from aero.flows import extract_signals

    with _store(s, migrate=True) as store:
        _print(dict(extract_signals(store, s)))
    return 0


def cmd_seed_recommendations(args: argparse.Namespace, s: Settings) -> int:
    from aero.flows import seed_recommendations

    with _store(s, migrate=True) as store:
        retriever = None if args.no_link else _retriever(s, store)
        _print(dict(seed_recommendations(store, s, retriever)))
    return 0


def cmd_rec(args: argparse.Namespace, s: Settings) -> int:
    from aero.domain import RecommendationStatus
    from aero.intelligence.strategy import ApprovalBlockedError, change_status, independent_publishers
    from aero.storage.base import InvalidTransitionError

    with _store(s, migrate=True) as store:
        if args.action == "list":
            for r in store.list_recommendations(limit=200):
                print(f"#{r.id:<4} v{r.version} {r.status:<9} {r.priority:<6} {r.title}")
            return 0
        if args.id is None:
            print("an id is required", file=sys.stderr)
            return 2
        if args.action in ("approve", "retire"):
            if not (args.reviewer and args.reason):
                print("--reviewer and --reason are required", file=sys.stderr)
                return 2
            target: RecommendationStatus = "approved" if args.action == "approve" else "retired"
            try:
                change_status(store, args.id, target, args.reviewer, args.reason)
            except (ApprovalBlockedError, InvalidTransitionError, KeyError) as exc:
                print(f"refused: {exc}", file=sys.stderr)
                return 1
        rec = store.get_recommendation(args.id)
        if rec is None:
            print(f"recommendation {args.id} not found", file=sys.stderr)
            return 1
        ev = store.recommendation_evidence(rec.id)
        print(f"#{rec.id} v{rec.version} [{rec.status}] {rec.title}\n{rec.recommendation}\n")
        print(f"evidence: {len(ev)} chunks from {independent_publishers(ev)} independent publishers")
        for e in ev:
            print(f"  [{e.relation}] C{e.evidence.chunk_id} {e.evidence.publisher}: {e.evidence.title[:90]}")
        for review in store.recommendation_reviews(rec.id):
            print(f"  review: {review.from_status} -> {review.to_status} by {review.reviewer}: {review.reason}")
    return 0


def cmd_serve(args: argparse.Namespace, s: Settings) -> int:
    from aero.mcp_servers.registry import ServerDeps, build_server

    store = _store(s, migrate=True)
    server = build_server(args.server, ServerDeps.from_settings(s, store))
    if args.stdio:
        server.run("stdio")
        return 0
    import uvicorn

    from aero.mcp_servers.http import build_http_app

    app = build_http_app(server, s, ready=lambda: {"documents": store.stats().documents})
    uvicorn.run(app, host=s.mcp_host, port=s.mcp_port, log_level=s.log_level.lower())
    return 0


def cmd_eval(args: argparse.Namespace, s: Settings) -> int:
    from aero.evals import retrieval as ev

    if args.action == "labels":
        rows = [json.loads(line) for line in Path(args.pool).read_text().splitlines() if line.strip()]
        qrels = ev.qrels_from_pool(rows)
        Path(args.out).write_text("".join(json.dumps(q) + "\n" for q in qrels))
        print(f"wrote {len(qrels)} labelled queries to {args.out}")
        return 0
    with _store(s) as store:
        retriever = _retriever(s, store)
        if args.action == "pool":
            queries = [json.loads(line) for line in Path(args.queries).read_text().splitlines() if line.strip()]
            pool = ev.build_pool(retriever, queries, k=args.k)
            Path(args.out).write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pool))
            print(f"wrote {len(pool)} queries to {args.out}; set each candidate's grade to 0, 1 or 2")
            return 0
        report = ev.evaluate(retriever, ev.load_qrels(args.qrels), k=args.k)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    _print(report["summary"])
    recall = report["summary"][f"recall@{args.k}"]
    if args.min_recall is not None and recall < args.min_recall:
        print(f"FAIL: recall@{args.k}={recall} < {args.min_recall}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aero", description="AERO-CEO v2")
    sub = p.add_subparsers(dest="command", required=True)

    db = sub.add_parser("db", help="database migrations and stats")
    db.add_argument("action", choices=["migrate", "stats"])
    db.set_defaults(fn=cmd_db)

    seed = sub.add_parser("seed", help="load seed data")
    seed_sub = seed.add_subparsers(dest="what", required=True)
    seed_sub.add_parser("sources").set_defaults(fn=cmd_seed)
    seed_rec = seed_sub.add_parser("recommendations", help="v1 blueprints as drafts + auto-linked evidence")
    seed_rec.add_argument("--no-link", action="store_true", help="skip evidence auto-linking (no embedder needed)")
    seed_rec.set_defaults(fn=cmd_seed_recommendations)

    ing = sub.add_parser("ingest", help="collect and index evidence")
    ing_sub = ing.add_subparsers(dest="kind", required=True)
    off = ing_sub.add_parser("official")
    off.add_argument("--limit", type=int, default=40)
    news = ing_sub.add_parser("news")
    news.add_argument("--per-query", type=int, default=20)
    news.add_argument("--limit", type=int, default=200)
    news.add_argument("--headlines-only", action="store_true", help="do not fetch article pages")
    v1 = ing_sub.add_parser("v1", help="import a v1 SQLite corpus (read-only)")
    v1.add_argument("path")
    ing.set_defaults(fn=cmd_ingest)

    emb = sub.add_parser("embed", help="embed chunks missing vectors for the configured model")
    emb.set_defaults(fn=cmd_embed)

    se = sub.add_parser("search", help="hybrid evidence search")
    se.add_argument("query")
    se.add_argument("--top-k", type=int, default=8)
    se.add_argument("--topic", action="append")
    se.add_argument("--source-type", action="append")
    se.add_argument("--json", action="store_true")
    se.set_defaults(fn=cmd_search)

    sig = sub.add_parser("signals", help="strategic signal extraction")
    sig.add_argument("action", choices=["extract"])
    sig.set_defaults(fn=cmd_signals)

    rec = sub.add_parser("rec", help="review recommendations (human-in-the-loop)")
    rec.add_argument("action", choices=["list", "show", "approve", "retire"])
    rec.add_argument("id", type=int, nargs="?")
    rec.add_argument("--reviewer")
    rec.add_argument("--reason")
    rec.set_defaults(fn=cmd_rec)

    sv = sub.add_parser("serve", help="run an MCP server")
    sv.add_argument("server", choices=["evidence", "signals", "strategy", "partners", "all"])
    sv.add_argument("--stdio", action="store_true", help="stdio transport (Claude Desktop, MCP Inspector)")
    sv.set_defaults(fn=cmd_serve)

    ev = sub.add_parser("eval", help="retrieval evaluation")
    ev_sub = ev.add_subparsers(dest="action", required=True)
    pool = ev_sub.add_parser("pool", help="write top-k candidates per query for labelling")
    pool.add_argument("--queries", default="evals/datasets/retrieval_queries.jsonl")
    pool.add_argument("--out", default="evals/datasets/retrieval_pool.jsonl")
    pool.add_argument("--k", type=int, default=20)
    labels = ev_sub.add_parser("labels", help="convert a graded pool into qrels")
    labels.add_argument("--pool", default="evals/datasets/retrieval_pool.jsonl")
    labels.add_argument("--out", default="evals/datasets/retrieval_qrels.jsonl")
    ret = ev_sub.add_parser("retrieval", help="score retrieval against qrels")
    ret.add_argument("--qrels", default="evals/datasets/retrieval_qrels.jsonl")
    ret.add_argument("--k", type=int, default=10)
    ret.add_argument("--min-recall", type=float)
    ret.add_argument("--out")
    ev.set_defaults(fn=cmd_eval)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    return int(args.fn(args, settings))


if __name__ == "__main__":
    raise SystemExit(main())

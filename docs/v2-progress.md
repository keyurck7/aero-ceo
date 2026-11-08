# AERO-CEO v2: progress tracker

Plan: [`docs/architecture.md`](architecture.md) (roadmap §18, findings §3, migration map §15).
Decisions: [`docs/adr/`](adr/).

## Phase status

| Phase | Scope | Status |
|---|---|---|
| 0 · Harden baseline | settings, deps, tests, CI, critical bug fixes | ✅ done in v2 code (v1 `app/` left frozen) |
| 1 · Data platform | Postgres+pgvector / SQLite adapters, migrations, hybrid search + rerank, incremental embedding, v1 importer, eval harness | ✅ code + tests · ⏳ Postgres run in CI · ⏳ qrels need labels |
| 2 · MCP servers | evidence ✅ · signals ⏳ · strategy (+ drill-down prompts) ⏳ · partners ⏳ | 🟡 1 of 4 |
| 3 · Agent | LangGraph graph, vLLM + LiteLLM, API gateway, thin UI | ⬜ next |
| 4 · Verification & HITL | verify-mcp, repair loop, approvals | ⬜ |
| 5 · Pipelines, observability, eval gates | Prefect, LLM signal extraction, OTel + Langfuse, DeepEval/promptfoo | ⬜ (flows are Prefect-ready) |
| 6 · Production readiness | OIDC, MCP gateway/OAuth, Helm/Terraform, runbooks | ⬜ |

## Findings closed by v2 code

| ID | Finding | Where fixed | Test |
|---|---|---|---|
| B1 | Search query written into evidence text | `ingestion/pipeline.py`, `collectors.py`, `migrate_v1.py` (strips both v1 templates) | `test_query_never_enters_document_text`, `test_migrate_v1_end_to_end` |
| B3 | Recommendation evidence had no chunk link | chunks carry `char_start/char_end`; every evidence object is citable | `test_evidence_carries_citation_metadata` |
| B4 | Non-RFC dates silently scored 0.55 | `dates.py` (RFC 2822, ISO 8601, human formats, plausibility window) | `TestDates` |
| B6 | Filters applied after nearest-neighbour cut | filters inside SQL in both adapters | `test_filters_apply_before_ranking` |
| B9 | Redirect URLs / syndication defeated dedup | canonical URLs + SimHash bands + aggregator resolution | `test_duplicates_are_rejected`, `test_collect_news_respects_licence_policy` |
| B10 | Two drifting schemas | one migration set per dialect, one contract suite | `test_store_contract.py` |
| B11 | `.env.example` didn't configure the code | `config.py` validated settings; unknown vars are errors | `TestConfig`, `test_cli_rejects_bad_config` |
| P2 | Prompt injection from scraped pages | `text.scan_injection` flags at ingest, down-weights in ranking, surfaced to agents | `test_injection_*` |
| P5 | Full FAISS rebuild + JSON copy of corpus | incremental embedding per chunk/model | `test_replace_chunks_and_embedding_lifecycle` |
| P6 | No retries / rate limits / robots | `ingestion/http.py` | (network-free unit coverage via fakes) |
| P7 | Hard-coded GPU, unpinned deps, no CI | backends by config; `pyproject.toml`; `.github/workflows/ci.yml` | CI |
| P9 | `print` logging | `log.py` JSON logs | — |
| P10 | Evaluation measured volume only | `evals/retrieval.py`: Recall/Precision/MRR/nDCG on qrels with CI gate | `test_cli_end_to_end_v1_import_search_and_eval` |
| — | Topic tagger labelled every "Airbus Defence and Space" article "Military Space" (also in v1) | `taxonomy.ignore_phrases` | `test_company_name_is_not_a_topic` |

Still open from §3: B2 (LLM truncation; fixed by vLLM in phase 3), B5 (LLM signal extraction, phase 2/5),
B7 (calibrated confidence, phase 4), B8 (generated recommendations, phase 2), P1 (UI auth, phase 3),
P3 (licence review: **owner decision**), P4 (shared model serving: TEI wired, vLLM phase 3), P8 (UI split, phase 3).

## Verified vs. not yet verified

- ✅ Executed: 90 tests on Python 3.13 with SQLite, MCP SDK 2.3.0 (in-process, Streamable HTTP over
  uvicorn, stdio subprocess), v1 importer against a DB created by v1's own `init_local_db()`.
- ✅ Postgres adapter SQL executed against **real PostgreSQL 16.15**: all 33 Postgres-parametrised tests
  (contract, retrieval, MCP) pass. The authoring sandbox had no PyPI and no pgvector, so they ran through a
  validation-only psql-backed driver, with `real[]` plus a SQL `<#>` operator standing in for pgvector.
  That run caught a real bug: `credibility_prior REAL` made `0.9 >= 0.9` false. The column is now `DOUBLE PRECISION`.
- ✅ Wheel builds and contains both migration sets and the `aero` entry point.
- ✅ CI (GitHub Actions) green: SQLite suite on Python 3.11/3.12/3.13, **33/33 tests on real Postgres +
  pgvector 0.8**, `ruff` clean, `mypy --strict` clean (both now blocking), Docker image builds. The first runs
  caught an undeclared `psycopg-pool` and pgvector returning `Vector` objects; both are fixed.
- ⏳ Not yet executed: `docker-compose.yml` as a whole stack (TEI images, healthchecks).

## Real corpus (v1 `aero_ceo.sqlite`, imported 2026-11-07)

| Finding | Number | Consequence |
|---|---|---|
| Documents carrying v1's query-in-text template (B1) | **180 / 190 (95%)** | stripped on import; query kept as provenance |
| News items that are headline-only (Google News RSS has no article text) | 180 / 181 | median v2 document = **10 words**; v1 answers rested on headlines plus template text |
| Duplicates v1 stored twice (its template made the content hashes differ) | 3 | collapsed |
| Official pages titled "Airbus official page" | 9 / 9 | titled from URL |
| SEO-spam publishers at the default 0.65 credibility (Mshale, Fathom Journal…) | 6 docs | `publisher_overrides` in `seeds/sources.yaml` (0.05) |
| Undated documents | 9 (all official pages) | recency = neutral 0.5 |
| v1 `agent_traces` / `human_approval_reviews` | 25 / 4 | **local agent code is not in the repo**; one review shows `confidence 0.92 "high"` while the guardrail found claim support < 0.65 plus contradictions (finding B7) |

**Retrieval baseline** on 15 queries × 186 docs (draft qrels, `evals/datasets/`), hashing embedder + BM25 (sandbox floor):
nDCG@10 **0.42** · capped Recall@10 **0.49** · MRR **0.77**. Best: FCAS/Dassault (0.90). Worst: abstract strategy
questions such as "drone demand grows faster than fighter procurement" (0.00): the gap BGE is meant to close.
**Data Lab, 2026-11-08** (RTX A6000, BGE-large-en-v1.5, same qrels):

| Setup | nDCG@10 | capped R@10 | MRR |
|---|---|---|---|
| hashing + BM25 (floor) | 0.42 | 0.49 | 0.77 |
| **BGE + BM25 hybrid** | **0.49** | **0.58** | **0.85** |
| BGE + BM25 + bge-reranker-v2-m3 (raw scores blended) | 0.35 | 0.41 | 0.74 |

The reranker *hurt*: it prefers long generic official pages over 10-word headlines, and its compressed raw
probabilities let the credibility term outvote relevance. Rerank scores are now min-max normalised per query;
`AERO_RETRIEVAL_MODE` (hybrid/dense/lexical) and the weight settings allow a clean ablation.

## Owner decisions (2026-11-07)
Academic use → `research_mode` (ADR 0004) · Qwen only, local → Qwen3.6-27B on vLLM (ADR 0006) · Data Lab nodes per phase (runbook).

## Next session

1. Data Lab session (A6000, about 30 min): BGE ± reranker baseline on the draft qrels; review the qrels.
2. Fix the corpus itself: add direct publisher RSS feeds (Breaking Defense, Defense News, Janes, Aviation Week,
   FlightGlobal) so news items carry article text, not just headlines.
3. `signals-mcp` (migration 0002: signals with chunk spans; rule:v1 extractor from
   `seeds/signal_rules_v1.yaml`, LLM extractor behind an interface), then `strategy-mcp` and
   `partners-mcp` from the seeds.
4. Phase 3 orchestrator skeleton.

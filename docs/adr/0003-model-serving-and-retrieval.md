# ADR 0003: Embedding/rerank backends and hybrid retrieval

**Status:** accepted · 2026-11-07

## Decision
- `Embedder` / `Reranker` interfaces with three backends: **TEI over HTTP** (shared GPU copy, prod),
  **sentence-transformers in-process** (single GPU server) and **hashing** (tests only; refused when
  `AERO_ENV=prod`). GPU selection belongs to the scheduler or the environment. It is no longer
  hard-coded `CUDA_VISIBLE_DEVICES=1` (finding P7).
- BGE v1.5 query instruction is applied to queries only, as the model card recommends.
- Retrieval = dense top-N + lexical top-N, both filtered in SQL → **Reciprocal Rank Fusion** (k=60) →
  optional cross-encoder rerank → final = 0.85·relevance + 0.10·credibility + 0.05·recency. Every
  component is returned in `ScoreBreakdown`.
- Credibility = publisher prior × content-level factor (headline+summary 0.8) × 0.5 if the text
  carries prompt-injection flags.
- Each chunk row stores its `embedding_model`. Changing the model re-embeds incrementally and never
  mixes vector spaces.

## Measured (Data Lab, 2026-11-08; 15 queries × 186 docs, draft qrels)

| Run | nDCG@10 | capped R@10 | MRR |
|---|---|---|---|
| hybrid, relevance only | 0.509 | 0.585 | 0.917 |
| hybrid, credibility 0.10 + recency 0.05 blend | 0.493 | 0.579 | 0.850 |
| hybrid + bge-reranker-v2-m3, relevance only | 0.383 | 0.456 | 0.741 |
| hybrid + reranker, blended | 0.362 | 0.414 | 0.741 |
| dense (BGE) only | 0.536 | 0.556 | 0.943 |
| lexical (BM25) only | 0.406 | 0.505 | 0.710 |

**Decisions**
- Reranker off by default. On a headline-only corpus it prefers long generic official pages. Re-test once
  articles carry full text.
- Rank on relevance only (weights 0). Trust becomes a floor (`AERO_RETRIEVAL_MIN_CREDIBILITY=0.15`, which drops
  content-farm publishers) and a per-hit field the agent uses in its coverage check.
- Keep hybrid as the default. Dense-only is +0.03 nDCG, within noise for 15 draft-labelled queries, and BM25
  matters for exact programme names once documents are longer. Revisit with reviewed labels.

## Consequences
Change any of these only with an `aero eval retrieval` run as evidence.

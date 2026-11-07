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

## Consequences
Weights are defaults, not tuned values. Change them only with an `aero eval retrieval` run as
evidence.

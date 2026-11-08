# ADR 0008: No heuristic weights. Standards-based grades, validated model judgements and fitted rankings

**Status:** accepted · 2026-11-08 · supersedes the scoring parts of ADR 0003 and ADR 0007

## Context
v2 still contained hand-picked numbers that looked like measurements:
- publisher credibility priors (0.05–0.95);
- content and injection multipliers;
- an optional credibility/recency blend in retrieval;
- per-topic "impact" priors;
- a rule-confidence formula;
- the partner score 0.6/0.2/0.2.

Two of these were known product limits. Keyword rules cannot read direction, and the partner score is a formula
with no training or validation. The owner asked for all of them to be removed.

## Decision
1. Executive-facing numbers are restricted to facts, grades from a published standard, validated model judgements,
   and calibrated probabilities ([plan §1](../plan-no-heuristics.md)).
2. **Sources** carry a NATO Admiralty reliability grade A–F. It is derived from an analyst-assigned category
   through a fixed table, with `reliability_basis` recorded. Unreviewed sources are F. Retrieval excludes E by
   default. Injection-flagged documents are quarantined (excluded unless asked for).
3. **Retrieval** has no blend: RRF order, or reranker order when enabled. Recency is a filter.
4. **Signals** come from an LLM with JSON-schema output (direction, event status, materiality rubric). Each must
   pass four checks to be stored: exact-quote grounding, a coherence rule, an entailment check, and a majority of
   N self-consistency samples. Information credibility (Admiralty 1/5/6) comes from corroboration by other
   publisher groups. Keyword rules survive only as an eval baseline.
5. **Partners** are ranked by Bradley–Terry fitted to pairwise LLM judgements, each pair judged in both orders,
   among candidates that pass a cited rubric assessment. A required country is a filter.
6. **Probabilities** are shown only after isotonic calibration on ≥ 200 labels with ECE ≤ 0.05, worded on the
   ICD 203 scale.
7. Every LLM call goes through `aero.llm`, which provides:
   - versioned prompts, with `prompt_id@version` and the model stored next to every output;
   - JSON-schema structured outputs (vLLM `response_format`);
   - retries;
   - record/replay cassettes so CI is deterministic.
8. Schema change by **forward migration 0003**. Applied migrations are never edited. Signals are derived data:
   the table is recreated and re-extracted.

## Consequences
- Signals, partner rankings and verification need a served LLM. Without one, these tools return an explicit
  "not available" error rather than a heuristic fallback.
- Labelling work is now on the critical path ([plan §6](../plan-no-heuristics.md)). The release gates in plan §7
  decide when the agent is executive-ready.
- LLM cost per document goes from 0 to about (N + verifications) calls. On vLLM with batching this is minutes
  for the current 186-document corpus.

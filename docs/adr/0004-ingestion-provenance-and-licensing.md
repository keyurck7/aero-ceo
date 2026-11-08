# ADR 0004: Evidence text is publisher text only; licence-aware full-text policy

**Status:** accepted · 2026-11-07. Owner decision 2026-11-07: **academic use**, so `research_mode` stays.

## Context
v1 wrote the search query into stored evidence ("…mentions Airbus in the context of {query}…" and
"Search query: {query}"). Retrieval then matched the query wording and the evaluation probes passed
by construction (findings B1, P10). v1 also scraped full articles regardless of licence (P3).

## Decision
- Document text = title + publisher body (article, or feed summary). Collection context (query, feed,
  collector, extractor) goes to `documents.ingestion_context`. It is never embedded, never indexed,
  and only exposed as `provenance` by `get_document`.
- `content_level` records how much we hold: `full_text`, `page` or `headline_summary`.
- `seeds/sources.yaml → full_text_policy`:
  - `research_mode` (current default, matches v1): store full text for non-commercial research.
  - `respect_licence`: full text only where `licence ∈ {full_text_permitted, licensed, own_content}`;
    otherwise headline + snippet.
- Every source has a `licence` field, currently `unknown` for all of them, pending review.
- The v1 importer strips both v1 templates and moves the query into provenance.

## Consequences
- Current use is academic (owner decision), so `full_text_policy: research_mode` is correct today.
Before any commercial or shared deployment, every active source needs a reviewed licence and
`full_text_policy: respect_licence`.

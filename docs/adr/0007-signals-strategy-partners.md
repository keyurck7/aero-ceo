# ADR 0007: Signals, recommendations and partners as MCP servers

**Status:** accepted · 2026-11-08

## Decisions

**Signals (signals-mcp)**
- Stored in the database (migration 0002), one row per (document, extractor, rule), each holding the
  exact quoted sentence plus `chunk_id` and character span. Re-extraction replaces one extractor's rows
  per document, so it is idempotent.
- `rule:v1-strict` reuses v1's rules but needs a keyword *in the text*, plus either a topic match or two
  distinct keywords. It also ignores hits inside company names. On the v1 corpus: **191 signals on 116 of
  186 documents (v1: 595)**.
- The business unit comes from the topic via v1's `BUSINESS_UNITS` table (`taxonomy.business_unit_topics`).
  The umbrella unit "Airbus Defence and Space" applies only when nothing specific fits. Keyword-only
  assignment put 144 of 191 signals on the umbrella.
- Known limit: keyword rules cannot read direction ("the *failed* fighter deal" counts as an FCAS
  *opportunity*). The LLM extractor (phase 5) implements the same `extract(document, chunks)`
  signature, and its output is compared against the rules on labelled samples.

**Recommendations (strategy-mcp)**
- Versioned rows `(key, version)` with lifecycle `draft → approved | retired`, `approved → retired`.
  Retired is terminal; a change means a new version. Every transition records reviewer and reason in
  `recommendation_reviews`.
- Evidence links are chunk-level (`recommendation_evidence.chunk_id NOT NULL`, fixes B3), with relation
  `supports | contradicts | candidate`. Auto-linkers write `candidate` and never overwrite
  human- or verifier-set links.
- **Approval gate:** linked evidence (excluding contradicting links) must come from at least two
  independent publishers. On the v1 corpus, the "military space" recommendation is blocked: all 6
  candidate chunks are Airbus's own pages.
- v1's six blueprints import as **drafts**, never as approved advice.
- Write tools are registered only with `AERO_STRATEGY_ALLOW_WRITES=true`. Proposals are always drafts.
- The seven v1 drill-down actions are MCP **prompts** (argument `recommendation_id`). Rendered prompts
  fence evidence as untrusted data and require `[C<chunk_id>]` citations.

**Partners (partners-mcp)**
- The registry stays an analyst-maintained YAML file (`seeds/partners.yaml`), with git as the audit
  trail and edits by pull request. No database table yet: there is no write path to justify one.
- Mentions are evidence passages where a partner name or alias appears as a whole word. v1 used
  `LIKE '%name%'`, which also matched inside other words.
- `match_partners` = 0.6 capability-token overlap + 0.2 country + 0.2 evidence mentions, with a
  per-partner rationale. It is a transparent heuristic, not a learned ranker.

**Deployment**
- One process per server in production (compose: ports 8101–8104). `aero serve all` mounts every tool
  set on one server for local single-user work and the phase-3 orchestrator.

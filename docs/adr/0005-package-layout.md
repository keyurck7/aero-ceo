# ADR 0005: One `aero` package with optional extras (not a multi-package workspace)

**Status:** accepted · 2026-11-07 (simplifies docs/architecture.md §16)

## Decision
v2 is one installable package, `src/aero/`, with extras per deployable: `postgres`, `mcp`, `extract`,
`local-models`, `dev`. Each Dockerfile build installs only the extras its service needs, so the
evidence MCP image never ships torch.

```
src/aero/
  config.py  log.py  domain.py  dates.py  text.py  scoring.py  taxonomy.py
  embeddings.py  rerank.py  retrieval.py  flows.py  cli.py
  storage/      base.py  sqlite.py  postgres.py  migrations/{sqlite,postgres}/
  ingestion/    http.py  extract.py  collectors.py  pipeline.py  migrate_v1.py
  mcp_servers/  evidence.py  http.py            (signals, strategy, partners, verify, … next)
  evals/        retrieval.py
seeds/          taxonomy, sources, partners, risks, recommendations_v1, drilldown prompts, signal rules
evals/datasets/ golden-question and retrieval-query seeds; qrels once labelled
```

## Consequences
One version and one test suite; less ceremony for a small team. Revisit with a `uv` workspace if
services start to need conflicting dependency versions.

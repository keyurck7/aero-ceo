# ADR 0002: One evidence store (Postgres + pgvector), a SQLite adapter, plain-SQL migrations

**Status:** accepted · 2026-11-07 (supersedes "SQLAlchemy + Alembic" in docs/architecture.md §7)

## Context
v1 kept rows in SQLite, vectors in a flat FAISS file, and a second copy of every chunk's text in a
JSON file. It rebuilt everything from scratch on each run. Two schemas (`schema.sql` and `local_db.py`)
had drifted apart (findings B10, P5).

## Decision
1. **Storage port** (`aero.storage.base.Store`) with two adapters that pass one contract-test suite:
   - `PostgresStore`: pgvector HNSW (inner product on normalised vectors) plus a generated `tsvector`
     with GIN. Filters run inside SQL. `hnsw.iterative_scan = relaxed_order` keeps filtered queries
     from coming back short.
   - `SQLiteStore`: FTS5 (BM25, porter stemming) plus brute-force numpy cosine, with the vector matrix
     cached per `vector_generation`. Used for single-box/GPU-server work, CI and tests. Comfortable up
     to about 10^5 chunks.
2. **Forward-only SQL migrations** (`storage/migrations/<dialect>/NNNN_*.sql`) applied by a small
   runner with a `schema_migrations` table. On Postgres an advisory lock serialises concurrent
   migrators. Alembic is not used: the schema is mostly pgvector/FTS DDL that Alembic cannot
   autogenerate, there is no ORM, and plain SQL is easier to review.
3. **Near-duplicates:** 64-bit SimHash over 3-word shingles, stored with eight 8-bit band columns. By
   the pigeonhole principle, any pair within Hamming distance 7 shares at least one band, so the
   lookup is an indexed equality query. The threshold of 7 was measured: a byline moves a 300-word
   article by at most about 11 bits (mean about 4), while unrelated texts stay 19 or more bits apart.

## Consequences
- The Postgres SQL was executed against PostgreSQL 16.15 with a stand-in for pgvector. pgvector itself
  (type, HNSW, psycopg adapters) is covered by CI job `test-postgres` against `pgvector/pgvector:0.8.0-pg16`.
- `credibility_prior` is `DOUBLE PRECISION`: with `REAL`, 0.9 is stored as 0.8999999762 and fails
  `>= 0.9` filters.
- Adding a table means adding a migration file for both dialects.

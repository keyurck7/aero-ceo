-- AERO-CEO v2 · 0001 · evidence core (Postgres 16 + pgvector >= 0.8)
-- Replaces app/db/schema.sql and app/db/local_db.py (finding B10: two drifting schemas).
-- {{EMBEDDING_DIM}} is substituted by the migration runner from AERO_EMBEDDING_DIM.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE sources (
    id                 BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name               TEXT NOT NULL UNIQUE,
    source_type        TEXT NOT NULL DEFAULT 'unknown'
                       CHECK (source_type IN ('official','news','policy','competitor','community','aggregator','unknown')),
    base_url           TEXT,
    credibility_prior  DOUBLE PRECISION NOT NULL DEFAULT 0.5 CHECK (credibility_prior BETWEEN 0 AND 1),  -- float8: REAL made 0.9 < 0.9
    publisher_group    TEXT,
    licence            TEXT NOT NULL DEFAULT 'unknown',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE pipeline_runs (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    flow         TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running','succeeded','failed')),
    params       JSONB NOT NULL DEFAULT '{}'::jsonb,
    stats        JSONB NOT NULL DEFAULT '{}'::jsonb,
    error        TEXT,
    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at  TIMESTAMPTZ
);

CREATE TABLE documents (
    id                 BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_id          BIGINT NOT NULL REFERENCES sources(id),
    title              TEXT NOT NULL,
    url                TEXT NOT NULL,
    canonical_url      TEXT NOT NULL UNIQUE,
    published_at       TIMESTAMPTZ,
    collected_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    clean_text         TEXT NOT NULL,
    content_level      TEXT NOT NULL DEFAULT 'full_text'
                       CHECK (content_level IN ('full_text','headline_summary','page')),
    language           TEXT NOT NULL DEFAULT 'en',
    topics             TEXT[] NOT NULL DEFAULT '{}',
    content_hash       TEXT NOT NULL UNIQUE,
    simhash            BIGINT NOT NULL,
    -- 8-bit SimHash bands: with Hamming distance <= 7 at least one of 8 bands matches exactly,
    -- so near-duplicate lookup is an indexed equality query instead of a full scan.
    simhash_b0         SMALLINT NOT NULL,
    simhash_b1         SMALLINT NOT NULL,
    simhash_b2         SMALLINT NOT NULL,
    simhash_b3         SMALLINT NOT NULL,
    simhash_b4         SMALLINT NOT NULL,
    simhash_b5         SMALLINT NOT NULL,
    simhash_b6         SMALLINT NOT NULL,
    simhash_b7         SMALLINT NOT NULL,
    ingestion_context  JSONB NOT NULL DEFAULT '{}'::jsonb,   -- never embedded (finding B1)
    injection_flags    TEXT[] NOT NULL DEFAULT '{}',
    raw_object_key     TEXT,
    pipeline_run_id    BIGINT REFERENCES pipeline_runs(id)
);
CREATE INDEX documents_source_idx     ON documents (source_id);
CREATE INDEX documents_published_idx  ON documents (published_at);
CREATE INDEX documents_collected_idx  ON documents (collected_at);
CREATE INDEX documents_topics_gin     ON documents USING gin (topics);
CREATE INDEX documents_b0_idx ON documents (simhash_b0);
CREATE INDEX documents_b1_idx ON documents (simhash_b1);
CREATE INDEX documents_b2_idx ON documents (simhash_b2);
CREATE INDEX documents_b3_idx ON documents (simhash_b3);
CREATE INDEX documents_b4_idx ON documents (simhash_b4);
CREATE INDEX documents_b5_idx ON documents (simhash_b5);
CREATE INDEX documents_b6_idx ON documents (simhash_b6);
CREATE INDEX documents_b7_idx ON documents (simhash_b7);

CREATE TABLE chunks (
    id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id      BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index      INTEGER NOT NULL,
    text             TEXT NOT NULL,
    char_start       INTEGER NOT NULL,
    char_end         INTEGER NOT NULL CHECK (char_end > char_start),
    word_count       INTEGER NOT NULL,
    embedding        vector({{EMBEDDING_DIM}}),
    embedding_model  TEXT,
    tsv              tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    UNIQUE (document_id, chunk_index)
);
CREATE INDEX chunks_document_idx  ON chunks (document_id);
CREATE INDEX chunks_tsv_gin       ON chunks USING gin (tsv);
CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_ip_ops)
    WITH (m = 16, ef_construction = 128);
CREATE INDEX chunks_model_idx     ON chunks (embedding_model);

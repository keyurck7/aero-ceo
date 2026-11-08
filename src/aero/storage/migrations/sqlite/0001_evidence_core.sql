-- AERO-CEO v2 · 0001 · evidence core (SQLite >= 3.35 with FTS5)
-- Local/dev adapter. Mirrors postgres/0001 table-for-table; differences:
--   * topics live in document_topics (no array type)
--   * embeddings are float32 BLOBs, searched with numpy
--   * lexical search uses an FTS5 index kept in sync by triggers
-- Timestamps are fixed-width UTC text (YYYY-MM-DDTHH:MM:SSZ), so text order = time order.

CREATE TABLE sources (
    id                 INTEGER PRIMARY KEY,
    name               TEXT NOT NULL UNIQUE,
    source_type        TEXT NOT NULL DEFAULT 'unknown'
                       CHECK (source_type IN ('official','news','policy','competitor','community','aggregator','unknown')),
    base_url           TEXT,
    credibility_prior  REAL NOT NULL DEFAULT 0.5 CHECK (credibility_prior BETWEEN 0 AND 1),
    publisher_group    TEXT,
    licence            TEXT NOT NULL DEFAULT 'unknown',
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE TABLE pipeline_runs (
    id           INTEGER PRIMARY KEY,
    flow         TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running','succeeded','failed')),
    params       TEXT NOT NULL DEFAULT '{}',
    stats        TEXT NOT NULL DEFAULT '{}',
    error        TEXT,
    started_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    finished_at  TEXT
);

CREATE TABLE documents (
    id                 INTEGER PRIMARY KEY,
    source_id          INTEGER NOT NULL REFERENCES sources(id),
    title              TEXT NOT NULL,
    url                TEXT NOT NULL,
    canonical_url      TEXT NOT NULL UNIQUE,
    published_at       TEXT,
    collected_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    clean_text         TEXT NOT NULL,
    content_level      TEXT NOT NULL DEFAULT 'full_text'
                       CHECK (content_level IN ('full_text','headline_summary','page')),
    language           TEXT NOT NULL DEFAULT 'en',
    content_hash       TEXT NOT NULL UNIQUE,
    simhash            INTEGER NOT NULL,
    simhash_b0         INTEGER NOT NULL,
    simhash_b1         INTEGER NOT NULL,
    simhash_b2         INTEGER NOT NULL,
    simhash_b3         INTEGER NOT NULL,
    simhash_b4         INTEGER NOT NULL,
    simhash_b5         INTEGER NOT NULL,
    simhash_b6         INTEGER NOT NULL,
    simhash_b7         INTEGER NOT NULL,
    ingestion_context  TEXT NOT NULL DEFAULT '{}',
    injection_flags    TEXT NOT NULL DEFAULT '[]',
    raw_object_key     TEXT,
    pipeline_run_id    INTEGER REFERENCES pipeline_runs(id)
);
CREATE INDEX documents_source_idx    ON documents (source_id);
CREATE INDEX documents_published_idx ON documents (published_at);
CREATE INDEX documents_collected_idx ON documents (collected_at);
CREATE INDEX documents_b0_idx ON documents (simhash_b0);
CREATE INDEX documents_b1_idx ON documents (simhash_b1);
CREATE INDEX documents_b2_idx ON documents (simhash_b2);
CREATE INDEX documents_b3_idx ON documents (simhash_b3);
CREATE INDEX documents_b4_idx ON documents (simhash_b4);
CREATE INDEX documents_b5_idx ON documents (simhash_b5);
CREATE INDEX documents_b6_idx ON documents (simhash_b6);
CREATE INDEX documents_b7_idx ON documents (simhash_b7);

CREATE TABLE document_topics (
    document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    topic        TEXT NOT NULL,
    PRIMARY KEY (document_id, topic)
);
CREATE INDEX document_topics_topic_idx ON document_topics (topic);

CREATE TABLE chunks (
    id               INTEGER PRIMARY KEY,
    document_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index      INTEGER NOT NULL,
    text             TEXT NOT NULL,
    char_start       INTEGER NOT NULL,
    char_end         INTEGER NOT NULL CHECK (char_end > char_start),
    word_count       INTEGER NOT NULL,
    embedding        BLOB,
    embedding_model  TEXT,
    UNIQUE (document_id, chunk_index)
);
CREATE INDEX chunks_document_idx ON chunks (document_id);
CREATE INDEX chunks_model_idx    ON chunks (embedding_model);

CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text, content='chunks', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER chunks_au AFTER UPDATE OF text ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

-- Bumped on every chunk/embedding write so other processes (e.g. the MCP server)
-- know to refresh their in-memory vector matrix.
CREATE TABLE store_meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
INSERT INTO store_meta (key, value) VALUES ('vector_generation', 0);

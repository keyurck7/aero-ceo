-- AERO-CEO v2 · 0002 · signals, recommendations (versioned, lifecycle, audit), chunk-level evidence links
-- Mirrors postgres/0002. Lists are JSON text (topics, signal_titles); filter with json_each().

CREATE TABLE signals (
    id              INTEGER PRIMARY KEY,
    document_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id        INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    signal_type     TEXT NOT NULL CHECK (signal_type IN ('opportunity','risk','trend')),
    title           TEXT NOT NULL,
    description     TEXT NOT NULL,
    topic           TEXT NOT NULL,
    business_unit   TEXT,
    evidence_quote  TEXT NOT NULL,
    quote_start     INTEGER NOT NULL,
    quote_end       INTEGER NOT NULL CHECK (quote_end > quote_start),
    impact          REAL NOT NULL CHECK (impact BETWEEN 0 AND 1),
    confidence      REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    extractor       TEXT NOT NULL,
    extracted_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    UNIQUE (document_id, extractor, title)
);
CREATE INDEX signals_type_topic_idx ON signals (signal_type, topic);
CREATE INDEX signals_bu_idx ON signals (business_unit);

CREATE TABLE recommendations (
    id               INTEGER PRIMARY KEY,
    key              TEXT NOT NULL,
    version          INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),
    status           TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft','approved','retired')),
    title            TEXT NOT NULL,
    business_unit    TEXT,
    topics           TEXT NOT NULL DEFAULT '[]',
    priority         TEXT NOT NULL DEFAULT 'Medium' CHECK (priority IN ('High','Medium','Low')),
    recommendation   TEXT NOT NULL,
    expected_impact  TEXT NOT NULL DEFAULT '',
    risk_assessment  TEXT NOT NULL DEFAULT '',
    signal_titles    TEXT NOT NULL DEFAULT '[]',
    created_by       TEXT NOT NULL,
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    updated_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    approved_by      TEXT,
    approved_at      TEXT,
    status_reason    TEXT,
    UNIQUE (key, version)
);
CREATE INDEX recommendations_status_idx ON recommendations (status);

CREATE TABLE recommendation_evidence (
    recommendation_id  INTEGER NOT NULL REFERENCES recommendations(id) ON DELETE CASCADE,
    chunk_id           INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    relation           TEXT NOT NULL CHECK (relation IN ('supports','contradicts','candidate')),
    score              REAL,
    linked_by          TEXT NOT NULL,
    note               TEXT,
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    PRIMARY KEY (recommendation_id, chunk_id)
);

CREATE TABLE recommendation_reviews (
    id                 INTEGER PRIMARY KEY,
    recommendation_id  INTEGER NOT NULL REFERENCES recommendations(id) ON DELETE CASCADE,
    from_status        TEXT NOT NULL,
    to_status          TEXT NOT NULL,
    reviewer           TEXT NOT NULL,
    reason             TEXT NOT NULL,
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE INDEX recommendation_reviews_rec_idx ON recommendation_reviews (recommendation_id);

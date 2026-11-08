-- AERO-CEO v2 · 0003 · Admiralty source grading and LLM-verified signals (ADR 0008)
-- Removes every heuristic number from the schema: credibility_prior, signal impact and confidence.
-- Signals are derived data: the table is recreated and re-extracted with `aero signals extract`.

ALTER TABLE sources DROP COLUMN credibility_prior;
ALTER TABLE sources ADD COLUMN category TEXT NOT NULL DEFAULT 'unreviewed'
    CHECK (category IN ('official_primary','established_newsroom','specialist_or_commentary',
                        'aggregator_or_ugc','off_topic_or_spam','unreviewed'));
ALTER TABLE sources ADD COLUMN reliability CHAR(1) NOT NULL DEFAULT 'F'
    CHECK (reliability IN ('A','B','C','D','E','F'));
ALTER TABLE sources ADD COLUMN reliability_note TEXT NOT NULL DEFAULT '';
CREATE INDEX sources_reliability_idx ON sources (reliability);

-- Who assigned the topic labels: 'keyword:v1' (provisional, ingest-time) or the LLM prompt that replaced them.
ALTER TABLE documents ADD COLUMN topics_tagger TEXT NOT NULL DEFAULT 'keyword:v1';

DROP TABLE signals;
CREATE TABLE signals (
    id                      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id             BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id                BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    signal_type             TEXT NOT NULL CHECK (signal_type IN ('opportunity','risk','trend')),
    title                   TEXT NOT NULL,
    description             TEXT NOT NULL,
    topic                   TEXT NOT NULL,
    business_unit           TEXT,
    direction               TEXT NOT NULL CHECK (direction IN ('positive','negative','mixed','neutral')),
    event_status            TEXT NOT NULL,
    materiality             TEXT NOT NULL CHECK (materiality IN ('high','medium','low')),
    materiality_rationale   TEXT NOT NULL DEFAULT '',
    evidence_quote          TEXT NOT NULL,
    quote_start             INTEGER NOT NULL,
    quote_end               INTEGER NOT NULL CHECK (quote_end > quote_start),
    verification_note       TEXT NOT NULL DEFAULT '',
    samples_agreeing        INTEGER NOT NULL CHECK (samples_agreeing >= 1),
    samples_total           INTEGER NOT NULL CHECK (samples_total >= samples_agreeing),
    disputed_fields         JSONB NOT NULL DEFAULT '[]'::jsonb,
    extractor               TEXT NOT NULL,
    model                   TEXT NOT NULL,
    extracted_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    information_credibility SMALLINT NOT NULL DEFAULT 6 CHECK (information_credibility BETWEEN 1 AND 6),
    corroboration           JSONB NOT NULL DEFAULT '[]'::jsonb,
    corroborated_at         TIMESTAMPTZ,
    UNIQUE (document_id, extractor, chunk_id, signal_type, topic)
);
CREATE INDEX signals_type_topic_idx ON signals (signal_type, topic);
CREATE INDEX signals_bu_idx ON signals (business_unit);
CREATE INDEX signals_materiality_idx ON signals (materiality);

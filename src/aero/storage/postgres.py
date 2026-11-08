"""Postgres 16 + pgvector adapter (production).

* Dense: HNSW over normalised vectors with inner-product ops (``<#>``). With
  pgvector >= 0.8, ``hnsw.iterative_scan`` keeps filtered queries from coming
  back short, so filters can run *inside* the query (finding B6).
* Lexical: generated ``tsvector`` + GIN, ranked with ``ts_rank_cd``.
* Connections: a psycopg pool in autocommit mode; writes use explicit transactions.

Requires the ``postgres`` extra: ``pip install 'aero-ceo[postgres]'``.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any, cast

import numpy as np

from aero.domain import (
    ChunkIn,
    Document,
    DocumentIn,
    EvidenceChunk,
    EvidenceLinkIn,
    Recommendation,
    RecommendationEvidence,
    RecommendationIn,
    RecommendationReview,
    RecommendationStatus,
    SearchFilters,
    Signal,
    SignalFilters,
    SignalIn,
    Source,
    SourceIn,
    StoreStats,
)
from aero.storage.base import (
    DuplicateDocumentError,
    Store,
    check_transition,
    doc_simhash_columns,
    is_near_duplicate,
    lexical_terms,
    simhash_bands,
)
from aero.storage.migrations import discover
from aero.text import from_signed64, to_signed64

try:  # optional extra
    import psycopg
    from pgvector.psycopg import register_vector
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
    from psycopg_pool import ConnectionPool
except ImportError as exc:  # pragma: no cover
    raise ImportError("PostgresStore needs: pip install 'aero-ceo[postgres]'") from exc

_MIGRATION_LOCK_KEY = 7_270_101  # arbitrary, constant: serialises concurrent migrators

_EVIDENCE_SELECT = """
SELECT c.id AS chunk_id, c.document_id, c.chunk_index, c.text, c.char_start, c.char_end,
       d.title, d.url, d.canonical_url, d.published_at, d.collected_at, d.content_level, d.topics,
       d.injection_flags, s.name AS publisher, s.publisher_group, s.source_type, s.licence,
       s.credibility_prior
FROM chunks c
JOIN documents d ON d.id = c.document_id
JOIN sources s ON s.id = d.source_id
"""


class PostgresStore(Store):
    dialect = "postgres"

    def __init__(self, dsn: str, embedding_dim: int, min_size: int = 1, max_size: int = 10) -> None:
        self.dsn = dsn
        self.embedding_dim = embedding_dim
        self._min, self._max = min_size, max_size
        self._pool: ConnectionPool | None = None
        self._pool_lock = threading.Lock()

    # ------------------------------------------------------------------ connections
    @staticmethod
    def _configure(conn: psycopg.Connection) -> None:
        register_vector(conn)
        try:
            conn.execute("SET hnsw.iterative_scan = relaxed_order")
            conn.execute("SET hnsw.ef_search = 100")
        except psycopg.Error:  # pgvector < 0.8: no iterative scans; still correct, may return fewer rows
            pass

    @property
    def pool(self) -> ConnectionPool:
        if self._pool is None:
            with self._pool_lock:
                if self._pool is None:
                    with psycopg.connect(self.dsn, autocommit=True) as probe:
                        has_vector = probe.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone()
                    if not has_vector:
                        raise RuntimeError("pgvector extension missing: run `aero db migrate` first")
                    self._pool = ConnectionPool(
                        self.dsn, min_size=self._min, max_size=self._max, open=True,
                        kwargs={"autocommit": True, "row_factory": dict_row}, configure=self._configure,
                    )
        return self._pool

    def _q(self, sql: str, params: dict[str, Any] | Sequence[Any] | None = None) -> list[dict[str, Any]]:
        with self.pool.connection() as conn:
            return cast(list[dict[str, Any]], conn.execute(sql, params).fetchall())

    # ------------------------------------------------------------------ lifecycle
    def migrate(self) -> list[str]:
        applied: list[str] = []
        with psycopg.connect(self.dsn, autocommit=True) as conn, conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK_KEY,))
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY, name TEXT NOT NULL,"
                " applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
            done = {r[0] for r in conn.execute("SELECT version FROM schema_migrations").fetchall()}
            for mig in discover("postgres", self.embedding_dim):
                if mig.version in done:
                    continue
                # No parameters -> psycopg uses the simple protocol, so multi-statement files work.
                conn.execute(mig.sql)
                conn.execute("INSERT INTO schema_migrations (version, name) VALUES (%s, %s)", (mig.version, mig.name))
                applied.append(mig.version)
        return applied

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    # ------------------------------------------------------------------ sources
    @staticmethod
    def _source(row: dict[str, Any]) -> Source:
        return Source(**{k: row[k] for k in Source.model_fields})

    def upsert_source(self, source: SourceIn) -> Source:
        rows = self._q(
            """INSERT INTO sources (name, source_type, base_url, credibility_prior, publisher_group, licence)
               VALUES (%(name)s, %(source_type)s, %(base_url)s, %(credibility_prior)s, %(publisher_group)s, %(licence)s)
               ON CONFLICT (name) DO UPDATE SET source_type = EXCLUDED.source_type, base_url = EXCLUDED.base_url,
                 credibility_prior = EXCLUDED.credibility_prior, publisher_group = EXCLUDED.publisher_group,
                 licence = EXCLUDED.licence
               RETURNING *""",
            source.model_dump(),
        )
        return self._source(rows[0])

    def get_source(self, source_id: int) -> Source | None:
        rows = self._q("SELECT * FROM sources WHERE id = %s", (source_id,))
        return self._source(rows[0]) if rows else None

    def list_sources(self, source_type: str | None = None) -> list[Source]:
        if source_type:
            rows = self._q("SELECT * FROM sources WHERE source_type = %s ORDER BY name", (source_type,))
        else:
            rows = self._q("SELECT * FROM sources ORDER BY name")
        return [self._source(r) for r in rows]

    # ------------------------------------------------------------------ documents
    def find_duplicate(self, canonical_url: str, content_hash: str, simhash: int) -> tuple[int, str] | None:
        rows = self._q(
            "SELECT id, canonical_url FROM documents WHERE canonical_url = %s OR content_hash = %s LIMIT 1",
            (canonical_url, content_hash),
        )
        if rows:
            return rows[0]["id"], "canonical_url" if rows[0]["canonical_url"] == canonical_url else "content_hash"
        bands = simhash_bands(to_signed64(simhash))
        for row in self._q(
            "SELECT id, simhash FROM documents WHERE "
            + " OR ".join(f"simhash_b{i} = %s" for i in range(8)),
            bands,
        ):
            if is_near_duplicate(from_signed64(row["simhash"]), simhash):
                return row["id"], "near_duplicate"
        return None

    @staticmethod
    def _document(row: dict[str, Any]) -> Document:
        data = {k: row[k] for k in Document.model_fields if k in row}
        data["simhash"] = from_signed64(row["simhash"])
        data["topics"] = list(row["topics"] or [])
        data["injection_flags"] = list(row["injection_flags"] or [])
        return Document(**data)

    def insert_document(self, doc: DocumentIn, pipeline_run_id: int | None = None) -> Document:
        dup = self.find_duplicate(doc.canonical_url, doc.content_hash, doc.simhash)
        if dup:
            raise DuplicateDocumentError(*dup)
        params = {
            **doc.model_dump(exclude={"simhash", "ingestion_context"}),
            **doc_simhash_columns(doc),
            "ingestion_context": Jsonb(doc.ingestion_context),
            "pipeline_run_id": pipeline_run_id,
        }
        try:
            rows = self._q(
                """INSERT INTO documents (source_id, title, url, canonical_url, published_at, clean_text,
                     content_level, language, topics, content_hash, simhash, simhash_b0, simhash_b1, simhash_b2,
                     simhash_b3, simhash_b4, simhash_b5, simhash_b6, simhash_b7, ingestion_context, injection_flags,
                     raw_object_key, pipeline_run_id)
                   VALUES (%(source_id)s, %(title)s, %(url)s, %(canonical_url)s, %(published_at)s, %(clean_text)s,
                     %(content_level)s, %(language)s, %(topics)s, %(content_hash)s, %(simhash)s, %(simhash_b0)s,
                     %(simhash_b1)s, %(simhash_b2)s, %(simhash_b3)s, %(simhash_b4)s,
                     %(simhash_b5)s, %(simhash_b6)s, %(simhash_b7)s,
                     %(ingestion_context)s, %(injection_flags)s,
                     %(raw_object_key)s, %(pipeline_run_id)s)
                   RETURNING *""",
                params,
            )
        except psycopg.errors.UniqueViolation as exc:  # raced with another writer
            raise DuplicateDocumentError(-1, f"integrity: {exc}") from exc
        return self._document(rows[0])

    def get_document(self, document_id: int) -> Document | None:
        rows = self._q("SELECT * FROM documents WHERE id = %s", (document_id,))
        return self._document(rows[0]) if rows else None

    def list_document_ids(self) -> list[int]:
        return [r["id"] for r in self._q("SELECT id FROM documents ORDER BY id")]

    def replace_chunks(self, document_id: int, chunks: Sequence[ChunkIn]) -> list[int]:
        ids: list[int] = []
        with self.pool.connection() as conn, conn.transaction():
            conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
            for ch in chunks:
                row = conn.execute(
                    """INSERT INTO chunks (document_id, chunk_index, text, char_start, char_end, word_count)
                       VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
                    (document_id, ch.chunk_index, ch.text, ch.char_start, ch.char_end, ch.word_count),
                ).fetchone()
                assert row is not None
                ids.append(cast(dict[str, Any], row)["id"])
        return ids

    # ------------------------------------------------------------------ embeddings
    def chunks_needing_embedding(self, model: str, limit: int = 256) -> list[tuple[int, str]]:
        rows = self._q(
            "SELECT id, text FROM chunks WHERE embedding IS NULL OR embedding_model IS DISTINCT FROM %s "
            "ORDER BY id LIMIT %s",
            (model, limit),
        )
        return [(r["id"], r["text"]) for r in rows]

    def set_embeddings(self, items: Sequence[tuple[int, np.ndarray]], model: str) -> None:
        payload = []
        for chunk_id, vec in items:
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            if arr.shape[0] != self.embedding_dim:
                raise ValueError(f"embedding for chunk {chunk_id} has dim {arr.shape[0]}, store expects "
                                 f"{self.embedding_dim}")
            payload.append((arr, model, chunk_id))
        with self.pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            cur.executemany("UPDATE chunks SET embedding = %s, embedding_model = %s WHERE id = %s", payload)

    def get_embedding(self, chunk_id: int) -> tuple[np.ndarray, str] | None:
        rows = self._q("SELECT embedding, embedding_model FROM chunks WHERE id = %s AND embedding IS NOT NULL",
                       (chunk_id,))
        if not rows:
            return None
        value = rows[0]["embedding"]
        # pgvector >= 0.4 returns a pgvector.Vector object, older versions a numpy array.
        arr = value.to_numpy() if hasattr(value, "to_numpy") else value
        return np.asarray(arr, dtype=np.float32), rows[0]["embedding_model"]

    # ------------------------------------------------------------------ search
    @staticmethod
    def _filter_sql(f: SearchFilters) -> tuple[str, dict[str, Any]]:
        where: list[str] = []
        p: dict[str, Any] = {}
        if f.topics:
            where.append("d.topics && %(f_topics)s::text[]")
            p["f_topics"] = list(f.topics)
        if f.source_types:
            where.append("s.source_type = ANY(%(f_source_types)s::text[])")
            p["f_source_types"] = list(f.source_types)
        if f.published_after:
            where.append("d.published_at >= %(f_after)s")
            p["f_after"] = f.published_after
        if f.published_before:
            where.append("d.published_at < %(f_before)s")
            p["f_before"] = f.published_before
        if f.min_credibility > 0:
            where.append("s.credibility_prior >= %(f_min_cred)s")
            p["f_min_cred"] = f.min_credibility
        if f.as_of:
            where.append("d.collected_at <= %(f_as_of)s")
            p["f_as_of"] = f.as_of
        if f.exclude_document_ids:
            where.append("NOT (d.id = ANY(%(f_exclude)s::bigint[]))")
            p["f_exclude"] = list(f.exclude_document_ids)
        return (" AND ".join(where) if where else "TRUE"), p

    def dense_search(self, vector: np.ndarray, k: int, filters: SearchFilters, model: str) -> list[tuple[int, float]]:
        where, p = self._filter_sql(filters)
        rows = self._q(
            f"""SELECT c.id, -(c.embedding <#> %(qvec)s::vector) AS score
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                JOIN sources s ON s.id = d.source_id
                WHERE c.embedding IS NOT NULL AND c.embedding_model = %(model)s AND {where}
                ORDER BY c.embedding <#> %(qvec)s::vector
                LIMIT %(k)s""",
            {**p, "qvec": np.asarray(vector, dtype=np.float32).reshape(-1), "model": model, "k": k},
        )
        return [(int(r["id"]), float(r["score"])) for r in rows]

    def lexical_search(self, query: str, k: int, filters: SearchFilters) -> list[tuple[int, float]]:
        terms = lexical_terms(query)
        if not terms:
            return []
        where, p = self._filter_sql(filters)
        rows = self._q(
            f"""SELECT c.id, ts_rank_cd(c.tsv, q) AS score
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                JOIN sources s ON s.id = d.source_id
                CROSS JOIN to_tsquery('english', %(tsq)s) AS q
                WHERE c.tsv @@ q AND {where}
                ORDER BY score DESC, c.id
                LIMIT %(k)s""",
            {**p, "tsq": " | ".join(terms), "k": k},
        )
        return [(int(r["id"]), float(r["score"])) for r in rows]

    @staticmethod
    def _evidence(row: dict[str, Any]) -> EvidenceChunk:
        data = dict(row)
        data["topics"] = list(row["topics"] or [])
        data["injection_flags"] = list(row["injection_flags"] or [])
        return EvidenceChunk(**data)

    def get_evidence(self, chunk_ids: Sequence[int]) -> dict[int, EvidenceChunk]:
        if not chunk_ids:
            return {}
        rows = self._q(f"{_EVIDENCE_SELECT} WHERE c.id = ANY(%s::bigint[])", (list(chunk_ids),))
        return {r["chunk_id"]: self._evidence(r) for r in rows}

    def get_neighbours(self, chunk_id: int, window: int = 1) -> list[EvidenceChunk]:
        rows = self._q(
            f"""WITH t AS (SELECT document_id, chunk_index FROM chunks WHERE id = %(id)s)
                {_EVIDENCE_SELECT}
                JOIN t ON t.document_id = c.document_id
                WHERE c.chunk_index BETWEEN t.chunk_index - %(w)s AND t.chunk_index + %(w)s
                ORDER BY c.chunk_index""",
            {"id": chunk_id, "w": window},
        )
        return [self._evidence(r) for r in rows]

    def document_chunks(self, document_id: int) -> list[EvidenceChunk]:
        rows = self._q(f"{_EVIDENCE_SELECT} WHERE c.document_id = %s ORDER BY c.chunk_index", (document_id,))
        return [self._evidence(r) for r in rows]

    # ------------------------------------------------------------------ signals
    def replace_signals(self, document_id: int, extractor: str, signals: Sequence[SignalIn]) -> int:
        with self.pool.connection() as conn, conn.transaction():
            conn.execute("DELETE FROM signals WHERE document_id = %s AND extractor = %s", (document_id, extractor))
            for sig in signals:
                if sig.document_id != document_id or sig.extractor != extractor:
                    raise ValueError("signal does not belong to this document/extractor")
                conn.execute(
                    """INSERT INTO signals (document_id, chunk_id, signal_type, title, description, topic,
                         business_unit, evidence_quote, quote_start, quote_end, impact, confidence, extractor)
                       VALUES (%(document_id)s, %(chunk_id)s, %(signal_type)s, %(title)s, %(description)s,
                         %(topic)s, %(business_unit)s, %(evidence_quote)s, %(quote_start)s, %(quote_end)s,
                         %(impact)s, %(confidence)s, %(extractor)s)""",
                    sig.model_dump(),
                )
        return len(signals)

    _SIGNAL_SELECT = """
        SELECT g.*, d.title AS document_title, d.url, d.published_at, s.name AS publisher, s.source_type,
               s.credibility_prior
        FROM signals g JOIN documents d ON d.id = g.document_id JOIN sources s ON s.id = d.source_id"""

    def list_signals(self, filters: SignalFilters, limit: int = 50) -> list[Signal]:
        where: list[str] = []
        p: dict[str, Any] = {"limit": limit}
        for column, name, values in (("g.signal_type", "types", filters.signal_types),
                                     ("g.topic", "topics", filters.topics),
                                     ("g.business_unit", "bus", filters.business_units)):
            if values:
                where.append(f"{column} = ANY(%({name})s::text[])")
                p[name] = list(values)
        if filters.extractor:
            where.append("g.extractor = %(extractor)s")
            p["extractor"] = filters.extractor
        if filters.published_after:
            where.append("d.published_at >= %(after)s")
            p["after"] = filters.published_after
        if filters.published_before:
            where.append("d.published_at < %(before)s")
            p["before"] = filters.published_before
        if filters.min_impact > 0:
            where.append("g.impact >= %(min_impact)s")
            p["min_impact"] = filters.min_impact
        clause = " AND ".join(where) if where else "TRUE"
        rows = self._q(
            f"{self._SIGNAL_SELECT} WHERE {clause} "
            "ORDER BY g.impact DESC, g.confidence DESC, d.published_at DESC NULLS LAST, g.id LIMIT %(limit)s",
            p,
        )
        return [Signal(**r) for r in rows]

    def get_signal(self, signal_id: int) -> Signal | None:
        rows = self._q(f"{self._SIGNAL_SELECT} WHERE g.id = %s", (signal_id,))
        return Signal(**rows[0]) if rows else None

    # ------------------------------------------------------------------ recommendations
    @staticmethod
    def _recommendation(row: dict[str, Any]) -> Recommendation:
        data = dict(row)
        data["topics"] = list(row["topics"] or [])
        data["signal_titles"] = list(row["signal_titles"] or [])
        return Recommendation(**data)

    def upsert_recommendation(self, rec: RecommendationIn) -> Recommendation:
        params = {**rec.model_dump(), "signal_titles": Jsonb(rec.signal_titles)}
        with self.pool.connection() as conn, conn.transaction():
            conn.execute(
                """INSERT INTO recommendations (key, version, status, title, business_unit, topics, priority,
                     recommendation, expected_impact, risk_assessment, signal_titles, created_by)
                   VALUES (%(key)s, %(version)s, %(status)s, %(title)s, %(business_unit)s, %(topics)s,
                     %(priority)s, %(recommendation)s, %(expected_impact)s, %(risk_assessment)s,
                     %(signal_titles)s, %(created_by)s)
                   ON CONFLICT (key, version) DO UPDATE SET title = EXCLUDED.title,
                     business_unit = EXCLUDED.business_unit, topics = EXCLUDED.topics,
                     priority = EXCLUDED.priority, recommendation = EXCLUDED.recommendation,
                     expected_impact = EXCLUDED.expected_impact, risk_assessment = EXCLUDED.risk_assessment,
                     signal_titles = EXCLUDED.signal_titles, updated_at = now()
                   WHERE recommendations.status = 'draft'""",
                params,
            )
            row = conn.execute("SELECT * FROM recommendations WHERE key = %s AND version = %s",
                               (rec.key, rec.version)).fetchone()
        assert row is not None
        return self._recommendation(cast(dict[str, Any], row))

    def get_recommendation(self, recommendation_id: int) -> Recommendation | None:
        rows = self._q("SELECT * FROM recommendations WHERE id = %s", (recommendation_id,))
        return self._recommendation(rows[0]) if rows else None

    def list_recommendations(
        self,
        status: RecommendationStatus | None = None,
        topic: str | None = None,
        business_unit: str | None = None,
        key: str | None = None,
        limit: int = 50,
    ) -> list[Recommendation]:
        where: list[str] = []
        p: dict[str, Any] = {"limit": limit}
        if status:
            where.append("r.status = %(status)s")
            p["status"] = status
        if topic:
            where.append("%(topic)s = ANY(r.topics)")
            p["topic"] = topic
        if business_unit:
            where.append("r.business_unit = %(bu)s")
            p["bu"] = business_unit
        if key:
            where.append("r.key = %(key)s")
            p["key"] = key
        clause = " AND ".join(where) if where else "TRUE"
        rows = self._q(
            f"""SELECT * FROM recommendations r WHERE {clause}
                ORDER BY CASE r.priority WHEN 'High' THEN 0 WHEN 'Medium' THEN 1 ELSE 2 END, r.key, r.version DESC
                LIMIT %(limit)s""",
            p,
        )
        return [self._recommendation(r) for r in rows]

    def set_recommendation_status(
        self, recommendation_id: int, status: RecommendationStatus, reviewer: str, reason: str
    ) -> Recommendation:
        if not reviewer.strip() or not reason.strip():
            raise ValueError("reviewer and reason are required for every status change")
        with self.pool.connection() as conn, conn.transaction():
            row = conn.execute("SELECT status FROM recommendations WHERE id = %s FOR UPDATE",
                               (recommendation_id,)).fetchone()
            if row is None:
                raise KeyError(f"recommendation {recommendation_id} not found")
            current = cast(dict[str, Any], row)["status"]
            check_transition(current, status)
            approved = status == "approved"
            conn.execute(
                """UPDATE recommendations SET status = %(status)s, status_reason = %(reason)s, updated_at = now(),
                     approved_by = CASE WHEN %(approved)s THEN %(reviewer)s ELSE approved_by END,
                     approved_at = CASE WHEN %(approved)s THEN now() ELSE approved_at END
                   WHERE id = %(id)s""",
                {"status": status, "reason": reason, "approved": approved, "reviewer": reviewer,
                 "id": recommendation_id},
            )
            conn.execute(
                """INSERT INTO recommendation_reviews (recommendation_id, from_status, to_status, reviewer, reason)
                   VALUES (%s, %s, %s, %s, %s)""",
                (recommendation_id, current, status, reviewer, reason),
            )
        result = self.get_recommendation(recommendation_id)
        assert result is not None
        return result

    def link_recommendation_evidence(
        self, recommendation_id: int, links: Sequence[EvidenceLinkIn], overwrite: bool = False
    ) -> int:
        conflict = (
            "DO UPDATE SET relation = EXCLUDED.relation, score = EXCLUDED.score, linked_by = EXCLUDED.linked_by, "
            "note = EXCLUDED.note" if overwrite else "DO NOTHING"
        )
        written = 0
        with self.pool.connection() as conn, conn.transaction():
            for link in links:
                cur = conn.execute(
                    f"""INSERT INTO recommendation_evidence (recommendation_id, chunk_id, relation, score, linked_by,
                          note) VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (recommendation_id, chunk_id) {conflict}""",
                    (recommendation_id, link.chunk_id, link.relation, link.score, link.linked_by, link.note),
                )
                written += max(cur.rowcount, 0)
        return written

    def recommendation_evidence(self, recommendation_id: int) -> list[RecommendationEvidence]:
        rows = self._q(
            """SELECT chunk_id, relation, score, linked_by, note FROM recommendation_evidence
               WHERE recommendation_id = %s
               ORDER BY CASE relation WHEN 'supports' THEN 0 WHEN 'candidate' THEN 1 ELSE 2 END,
                        score DESC NULLS LAST, chunk_id""",
            (recommendation_id,),
        )
        evidence = self.get_evidence([r["chunk_id"] for r in rows])
        return [
            RecommendationEvidence(evidence=evidence[r["chunk_id"]], relation=r["relation"], score=r["score"],
                                   linked_by=r["linked_by"], note=r["note"])
            for r in rows if r["chunk_id"] in evidence
        ]

    def recommendation_reviews(self, recommendation_id: int) -> list[RecommendationReview]:
        rows = self._q(
            """SELECT recommendation_id, from_status, to_status, reviewer, reason, created_at
               FROM recommendation_reviews WHERE recommendation_id = %s ORDER BY id""",
            (recommendation_id,),
        )
        return [RecommendationReview(**r) for r in rows]

    # ------------------------------------------------------------------ runs & stats
    def start_run(self, flow: str, params: dict[str, Any]) -> int:
        rows = self._q("INSERT INTO pipeline_runs (flow, params) VALUES (%s, %s) RETURNING id", (flow, Jsonb(params)))
        return int(rows[0]["id"])

    def finish_run(self, run_id: int, status: str, stats: dict[str, Any], error: str | None = None) -> None:
        self._q(
            "UPDATE pipeline_runs SET status = %s, stats = %s, error = %s, finished_at = now() WHERE id = %s "
            "RETURNING id",
            (status, Jsonb(stats), error, run_id),
        )

    def stats(self) -> StoreStats:
        row = self._q(
            """SELECT (SELECT COUNT(*) FROM sources) AS sources,
                      (SELECT COUNT(*) FROM documents) AS documents,
                      (SELECT COUNT(*) FROM chunks) AS chunks,
                      (SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL) AS embedded_chunks,
                      (SELECT COALESCE(array_agg(DISTINCT embedding_model), '{}') FROM chunks
                         WHERE embedding_model IS NOT NULL) AS embedding_models,
                      (SELECT MAX(collected_at) FROM documents) AS latest_collected_at"""
        )[0]
        row["embedding_models"] = sorted(row["embedding_models"] or [])
        return StoreStats(**row)

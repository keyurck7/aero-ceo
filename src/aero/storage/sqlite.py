"""SQLite adapter: local development, CI and single-box demos.

Lexical search uses FTS5 (BM25). Dense search keeps one float32 matrix in memory
and refreshes it when ``store_meta.vector_generation`` changes. That counter is
bumped by any process that writes chunks or embeddings, so a running MCP server
sees new data without a restart.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from aero.dates import from_db, to_db, utcnow
from aero.domain import (
    ChunkIn,
    CorroborationItem,
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
    SIGNAL_COLUMNS,
    SIGNAL_ORDER,
    DuplicateDocumentError,
    Store,
    check_transition,
    doc_simhash_columns,
    is_near_duplicate,
    lexical_terms,
    signal_filter_columns,
    simhash_bands,
)
from aero.storage.migrations import discover
from aero.text import from_signed64, to_signed64

_EVIDENCE_SELECT = """
SELECT c.id AS chunk_id, c.document_id, c.chunk_index, c.text, c.char_start, c.char_end,
       d.title, d.url, d.canonical_url, d.published_at, d.collected_at, d.content_level,
       d.injection_flags, s.name AS publisher, s.publisher_group, s.source_type, s.licence,
       s.reliability AS source_reliability
FROM chunks c
JOIN documents d ON d.id = c.document_id
JOIN sources s ON s.id = d.source_id
"""


class SQLiteStore(Store):
    dialect = "sqlite"

    def __init__(self, path: str | Path, embedding_dim: int) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA busy_timeout = 10000")
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._lock = threading.RLock()
        self._matrix_cache: tuple[int, str, np.ndarray, np.ndarray] | None = None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _ts(value: str) -> datetime:
        """Non-null timestamp column."""
        parsed = from_db(value)
        assert parsed is not None
        return parsed

    def _q(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def _tx(self) -> _Tx:
        return _Tx(self._conn, self._lock)

    def _bump_vectors(self, cur: sqlite3.Connection) -> None:
        cur.execute("UPDATE store_meta SET value = value + 1 WHERE key = 'vector_generation'")

    # ------------------------------------------------------------------ lifecycle
    def migrate(self) -> list[str]:
        applied: list[str] = []
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY, name TEXT NOT NULL,"
                " applied_at TEXT NOT NULL)"
            )
            done = {r[0] for r in self._conn.execute("SELECT version FROM schema_migrations")}
            for mig in discover("sqlite", self.embedding_dim):
                if mig.version in done:
                    continue
                script = (
                    "BEGIN;\n" + mig.sql + "\nINSERT INTO schema_migrations (version, name, applied_at) VALUES ("
                    f"'{mig.version}', '{mig.name}', '{to_db(utcnow())}');\nCOMMIT;"
                )
                try:
                    self._conn.executescript(script)
                except Exception:
                    if self._conn.in_transaction:
                        self._conn.execute("ROLLBACK")
                    raise
                applied.append(mig.version)
        return applied

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ sources
    @staticmethod
    def _source(row: sqlite3.Row) -> Source:
        return Source(
            id=row["id"], name=row["name"], source_type=row["source_type"], base_url=row["base_url"],
            category=row["category"], reliability_note=row["reliability_note"],
            publisher_group=row["publisher_group"], licence=row["licence"],
            created_at=SQLiteStore._ts(row["created_at"]),
        )

    def upsert_source(self, source: SourceIn) -> Source:
        with self._tx() as cur:
            cur.execute(
                """INSERT INTO sources (name, source_type, base_url, category, reliability, reliability_note,
                                        publisher_group, licence)
                   VALUES (:name, :source_type, :base_url, :category, :reliability, :reliability_note,
                           :publisher_group, :licence)
                   ON CONFLICT(name) DO UPDATE SET source_type=excluded.source_type, base_url=excluded.base_url,
                     category=excluded.category, reliability=excluded.reliability,
                     reliability_note=excluded.reliability_note, publisher_group=excluded.publisher_group,
                     licence=excluded.licence""",
                source.model_dump(),
            )
            row = cur.execute("SELECT * FROM sources WHERE name = ?", (source.name,)).fetchone()
        return self._source(row)

    def get_source(self, source_id: int) -> Source | None:
        rows = self._q("SELECT * FROM sources WHERE id = ?", (source_id,))
        return self._source(rows[0]) if rows else None

    def list_sources(self, source_type: str | None = None) -> list[Source]:
        if source_type:
            rows = self._q("SELECT * FROM sources WHERE source_type = ? ORDER BY name", (source_type,))
        else:
            rows = self._q("SELECT * FROM sources ORDER BY name")
        return [self._source(r) for r in rows]

    # ------------------------------------------------------------------ documents
    def find_duplicate(self, canonical_url: str, content_hash: str, simhash: int) -> tuple[int, str] | None:
        rows = self._q(
            "SELECT id, canonical_url, content_hash FROM documents WHERE canonical_url = ? OR content_hash = ?",
            (canonical_url, content_hash),
        )
        for row in rows:
            return row["id"], "canonical_url" if row["canonical_url"] == canonical_url else "content_hash"
        bands = simhash_bands(to_signed64(simhash))
        candidates = self._q(
            "SELECT id, simhash FROM documents WHERE " + " OR ".join(f"simhash_b{i} = ?" for i in range(8)),
            bands,
        )
        for row in candidates:
            if is_near_duplicate(from_signed64(row["simhash"]), simhash):
                return row["id"], "near_duplicate"
        return None

    def _document(self, row: sqlite3.Row) -> Document:
        topics = [r[0] for r in self._q("SELECT topic FROM document_topics WHERE document_id = ? ORDER BY rowid",
                                        (row["id"],))]
        return Document(
            id=row["id"], source_id=row["source_id"], title=row["title"], url=row["url"],
            canonical_url=row["canonical_url"], published_at=from_db(row["published_at"]),
            collected_at=self._ts(row["collected_at"]), clean_text=row["clean_text"],
            content_level=row["content_level"], language=row["language"], topics=topics,
            content_hash=row["content_hash"], simhash=from_signed64(row["simhash"]),
            ingestion_context=json.loads(row["ingestion_context"]),
            injection_flags=json.loads(row["injection_flags"]), raw_object_key=row["raw_object_key"],
            topics_tagger=row["topics_tagger"],
        )

    def set_document_topics(self, document_id: int, topics: Sequence[str], tagger: str) -> None:
        with self._tx() as cur:
            if cur.execute("UPDATE documents SET topics_tagger = ? WHERE id = ?", (tagger, document_id)).rowcount == 0:
                raise KeyError(document_id)
            cur.execute("DELETE FROM document_topics WHERE document_id = ?", (document_id,))
            cur.executemany("INSERT OR IGNORE INTO document_topics (document_id, topic) VALUES (?, ?)",
                            [(document_id, t) for t in topics])

    def insert_document(self, doc: DocumentIn, pipeline_run_id: int | None = None) -> Document:
        dup = self.find_duplicate(doc.canonical_url, doc.content_hash, doc.simhash)
        if dup:
            raise DuplicateDocumentError(*dup)
        params = {
            **doc.model_dump(exclude={"topics", "simhash", "ingestion_context", "injection_flags", "published_at"}),
            **doc_simhash_columns(doc),
            "published_at": to_db(doc.published_at),
            "ingestion_context": json.dumps(doc.ingestion_context, ensure_ascii=False, default=str),
            "injection_flags": json.dumps(doc.injection_flags),
            "pipeline_run_id": pipeline_run_id,
        }
        with self._tx() as cur:
            try:
                new_id = cur.execute(
                    """INSERT INTO documents (source_id, title, url, canonical_url, published_at, clean_text,
                         content_level, language, content_hash, simhash, simhash_b0, simhash_b1, simhash_b2,
                         simhash_b3, simhash_b4, simhash_b5, simhash_b6, simhash_b7, ingestion_context, injection_flags,
                         raw_object_key, pipeline_run_id)
                       VALUES (:source_id, :title, :url, :canonical_url, :published_at, :clean_text,
                         :content_level, :language, :content_hash, :simhash, :simhash_b0, :simhash_b1, :simhash_b2,
                         :simhash_b3, :simhash_b4, :simhash_b5, :simhash_b6, :simhash_b7,
                         :ingestion_context, :injection_flags,
                         :raw_object_key, :pipeline_run_id)""",
                    params,
                ).lastrowid
            except sqlite3.IntegrityError as exc:  # raced with another writer
                raise DuplicateDocumentError(-1, f"integrity: {exc}") from exc
            cur.executemany(
                "INSERT OR IGNORE INTO document_topics (document_id, topic) VALUES (?, ?)",
                [(new_id, t) for t in doc.topics],
            )
        result = self.get_document(int(new_id))  # type: ignore[arg-type]
        assert result is not None
        return result

    def get_document(self, document_id: int) -> Document | None:
        rows = self._q("SELECT * FROM documents WHERE id = ?", (document_id,))
        return self._document(rows[0]) if rows else None

    def list_document_ids(self) -> list[int]:
        return [r[0] for r in self._q("SELECT id FROM documents ORDER BY id")]

    def replace_chunks(self, document_id: int, chunks: Sequence[ChunkIn]) -> list[int]:
        with self._tx() as cur:
            cur.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            ids = []
            for ch in chunks:
                ids.append(
                    cur.execute(
                        """INSERT INTO chunks (document_id, chunk_index, text, char_start, char_end, word_count)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (document_id, ch.chunk_index, ch.text, ch.char_start, ch.char_end, ch.word_count),
                    ).lastrowid
                )
            self._bump_vectors(cur)
        return [int(i) for i in ids if i is not None]

    # ------------------------------------------------------------------ embeddings
    def chunks_needing_embedding(self, model: str, limit: int = 256) -> list[tuple[int, str]]:
        rows = self._q(
            "SELECT id, text FROM chunks WHERE embedding IS NULL OR embedding_model IS NOT ? ORDER BY id LIMIT ?",
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
            payload.append((arr.tobytes(), model, chunk_id))
        with self._tx() as cur:
            cur.executemany("UPDATE chunks SET embedding = ?, embedding_model = ? WHERE id = ?", payload)
            self._bump_vectors(cur)

    def get_embedding(self, chunk_id: int) -> tuple[np.ndarray, str] | None:
        rows = self._q("SELECT embedding, embedding_model FROM chunks WHERE id = ? AND embedding IS NOT NULL",
                       (chunk_id,))
        if not rows:
            return None
        return np.frombuffer(rows[0]["embedding"], dtype=np.float32).copy(), rows[0]["embedding_model"]

    # ------------------------------------------------------------------ search
    def _filter_sql(self, f: SearchFilters) -> tuple[str, list[Any]]:
        where: list[str] = []
        params: list[Any] = []
        if f.topics:
            where.append(
                f"d.id IN (SELECT document_id FROM document_topics WHERE topic IN ({','.join('?' * len(f.topics))}))"
            )
            params += f.topics
        if f.source_types:
            where.append(f"s.source_type IN ({','.join('?' * len(f.source_types))})")
            params += f.source_types
        if f.published_after:
            where.append("d.published_at >= ?")
            params.append(to_db(f.published_after))
        if f.published_before:
            where.append("d.published_at < ?")
            params.append(to_db(f.published_before))
        if f.source_reliability is not None:
            where.append(f"s.reliability IN ({','.join('?' * len(f.source_reliability)) or 'NULL'})")
            params += list(f.source_reliability)
        if not f.include_quarantined:
            where.append("d.injection_flags = '[]'")
        if f.as_of:
            where.append("d.collected_at <= ?")
            params.append(to_db(f.as_of))
        if f.exclude_document_ids:
            where.append(f"d.id NOT IN ({','.join('?' * len(f.exclude_document_ids))})")
            params += f.exclude_document_ids
        return (" AND ".join(where) if where else "1=1"), params

    def _vector_matrix(self, model: str) -> tuple[np.ndarray, np.ndarray]:
        generation = int(self._q("SELECT value FROM store_meta WHERE key='vector_generation'")[0][0])
        cached = self._matrix_cache
        if cached and cached[0] == generation and cached[1] == model:
            return cached[2], cached[3]
        rows = self._q("SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL AND embedding_model = ? "
                       "ORDER BY id", (model,))
        ids = np.fromiter((r["id"] for r in rows), dtype=np.int64, count=len(rows))
        matrix = (
            np.frombuffer(b"".join(r["embedding"] for r in rows), dtype=np.float32).reshape(len(rows), -1)
            if rows else np.zeros((0, self.embedding_dim), dtype=np.float32)
        )
        self._matrix_cache = (generation, model, ids, matrix)
        return ids, matrix

    def dense_search(self, vector: np.ndarray, k: int, filters: SearchFilters, model: str) -> list[tuple[int, float]]:
        ids, matrix = self._vector_matrix(model)
        if len(ids) == 0:
            return []
        where, params = self._filter_sql(filters)
        allowed = self._q(
            f"SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id JOIN sources s ON s.id = d.source_id "
            f"WHERE c.embedding_model = ? AND {where}",
            [model, *params],
        )
        mask = np.isin(ids, np.fromiter((r[0] for r in allowed), dtype=np.int64, count=len(allowed)))
        if not mask.any():
            return []
        sub_ids, sub = ids[mask], matrix[mask]
        scores = sub @ np.asarray(vector, dtype=np.float32).reshape(-1)
        top = np.argsort(-scores, kind="stable")[:k]
        return [(int(sub_ids[i]), float(scores[i])) for i in top]

    def lexical_search(self, query: str, k: int, filters: SearchFilters) -> list[tuple[int, float]]:
        terms = lexical_terms(query)
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        where, params = self._filter_sql(filters)
        rows = self._q(
            f"""SELECT c.id, -bm25(chunks_fts) AS score
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.rowid
                JOIN documents d ON d.id = c.document_id
                JOIN sources s ON s.id = d.source_id
                WHERE chunks_fts MATCH ? AND {where}
                ORDER BY bm25(chunks_fts) LIMIT ?""",
            [match, *params, k],
        )
        return [(int(r[0]), float(r[1])) for r in rows]

    def _evidence(self, row: sqlite3.Row, topics: dict[int, list[str]]) -> EvidenceChunk:
        return EvidenceChunk(
            chunk_id=row["chunk_id"], document_id=row["document_id"], chunk_index=row["chunk_index"],
            text=row["text"], char_start=row["char_start"], char_end=row["char_end"], title=row["title"],
            url=row["url"], canonical_url=row["canonical_url"], publisher=row["publisher"],
            publisher_group=row["publisher_group"], source_type=row["source_type"], licence=row["licence"],
            source_reliability=row["source_reliability"], published_at=from_db(row["published_at"]),
            collected_at=self._ts(row["collected_at"]), topics=topics.get(row["document_id"], []),
            content_level=row["content_level"], injection_flags=json.loads(row["injection_flags"]),
        )

    def _topics_for(self, doc_ids: set[int]) -> dict[int, list[str]]:
        if not doc_ids:
            return {}
        rows = self._q(
            f"SELECT document_id, topic FROM document_topics WHERE document_id IN ({','.join('?' * len(doc_ids))}) "
            "ORDER BY rowid",
            list(doc_ids),
        )
        out: dict[int, list[str]] = {}
        for r in rows:
            out.setdefault(r[0], []).append(r[1])
        return out

    def get_evidence(self, chunk_ids: Sequence[int]) -> dict[int, EvidenceChunk]:
        if not chunk_ids:
            return {}
        rows = self._q(f"{_EVIDENCE_SELECT} WHERE c.id IN ({','.join('?' * len(chunk_ids))})", list(chunk_ids))
        topics = self._topics_for({r["document_id"] for r in rows})
        return {r["chunk_id"]: self._evidence(r, topics) for r in rows}

    def get_neighbours(self, chunk_id: int, window: int = 1) -> list[EvidenceChunk]:
        rows = self._q(
            f"""{_EVIDENCE_SELECT}
                WHERE c.document_id = (SELECT document_id FROM chunks WHERE id = ?)
                  AND c.chunk_index BETWEEN (SELECT chunk_index FROM chunks WHERE id = ?) - ?
                                        AND (SELECT chunk_index FROM chunks WHERE id = ?) + ?
                ORDER BY c.chunk_index""",
            (chunk_id, chunk_id, window, chunk_id, window),
        )
        topics = self._topics_for({r["document_id"] for r in rows})
        return [self._evidence(r, topics) for r in rows]

    def document_chunks(self, document_id: int) -> list[EvidenceChunk]:
        rows = self._q(f"{_EVIDENCE_SELECT} WHERE c.document_id = ? ORDER BY c.chunk_index", (document_id,))
        topics = self._topics_for({document_id})
        return [self._evidence(r, topics) for r in rows]

    # ------------------------------------------------------------------ signals
    def replace_signals(self, document_id: int, extractor: str, signals: Sequence[SignalIn]) -> int:
        with self._tx() as cur:
            cur.execute("DELETE FROM signals WHERE document_id = ? AND extractor = ?", (document_id, extractor))
            for sig in signals:
                if sig.document_id != document_id or sig.extractor != extractor:
                    raise ValueError("signal does not belong to this document/extractor")
                cur.execute(
                    f"INSERT INTO signals ({', '.join(SIGNAL_COLUMNS)}) "
                    f"VALUES ({', '.join(':' + c for c in SIGNAL_COLUMNS)})",
                    {**sig.model_dump(), "disputed_fields": json.dumps(sig.disputed_fields)},
                )
        return len(signals)

    _SIGNAL_SELECT = """
        SELECT g.*, d.title AS document_title, d.url, d.published_at, d.content_level, s.name AS publisher,
               s.publisher_group, s.source_type, s.reliability AS source_reliability
        FROM signals g JOIN documents d ON d.id = g.document_id JOIN sources s ON s.id = d.source_id"""

    def _signal(self, row: sqlite3.Row) -> Signal:
        data = {k: row[k] for k in row.keys()}  # noqa: SIM118 - sqlite3.Row is not a dict
        data["extracted_at"] = self._ts(row["extracted_at"])
        data["published_at"] = from_db(row["published_at"])
        data["corroborated_at"] = from_db(row["corroborated_at"])
        data["disputed_fields"] = json.loads(row["disputed_fields"])
        data["corroboration"] = json.loads(row["corroboration"])
        return Signal(**data)

    def list_signals(self, filters: SignalFilters, limit: int = 50) -> list[Signal]:
        where: list[str] = []
        params: list[Any] = []
        for column, values in signal_filter_columns(filters):
            where.append(f"{column} IN ({','.join('?' * len(values))})")
            params += list(values)
        if filters.extractor:
            where.append("g.extractor = ?")
            params.append(filters.extractor)
        if filters.published_after:
            where.append("d.published_at >= ?")
            params.append(to_db(filters.published_after))
        if filters.published_before:
            where.append("d.published_at < ?")
            params.append(to_db(filters.published_before))
        clause = " AND ".join(where) if where else "1=1"
        rows = self._q(
            f"{self._SIGNAL_SELECT} WHERE {clause} ORDER BY {SIGNAL_ORDER} LIMIT ?",
            [*params, limit],
        )
        return [self._signal(r) for r in rows]

    def set_signal_corroboration(self, signal_id: int, items: Sequence[CorroborationItem],
                                 information_credibility: int) -> None:
        with self._tx() as cur:
            updated = cur.execute(
                "UPDATE signals SET corroboration = ?, information_credibility = ?, corroborated_at = ? WHERE id = ?",
                (json.dumps([i.model_dump() for i in items]), information_credibility, to_db(utcnow()), signal_id),
            ).rowcount
            if updated == 0:
                raise KeyError(signal_id)

    def get_signal(self, signal_id: int) -> Signal | None:
        rows = self._q(f"{self._SIGNAL_SELECT} WHERE g.id = ?", (signal_id,))
        return self._signal(rows[0]) if rows else None

    # ------------------------------------------------------------------ recommendations
    def _recommendation(self, row: sqlite3.Row) -> Recommendation:
        data = {k: row[k] for k in row.keys()}  # noqa: SIM118
        data["topics"] = json.loads(row["topics"])
        data["signal_titles"] = json.loads(row["signal_titles"])
        for col in ("created_at", "updated_at"):
            data[col] = self._ts(row[col])
        data["approved_at"] = from_db(row["approved_at"])
        return Recommendation(**data)

    def upsert_recommendation(self, rec: RecommendationIn) -> Recommendation:
        params = {**rec.model_dump(), "topics": json.dumps(rec.topics), "signal_titles": json.dumps(rec.signal_titles),
                  "now": to_db(utcnow())}
        with self._tx() as cur:
            cur.execute(
                """INSERT INTO recommendations (key, version, status, title, business_unit, topics, priority,
                     recommendation, expected_impact, risk_assessment, signal_titles, created_by)
                   VALUES (:key, :version, :status, :title, :business_unit, :topics, :priority, :recommendation,
                     :expected_impact, :risk_assessment, :signal_titles, :created_by)
                   ON CONFLICT (key, version) DO UPDATE SET title = excluded.title,
                     business_unit = excluded.business_unit, topics = excluded.topics,
                     priority = excluded.priority, recommendation = excluded.recommendation,
                     expected_impact = excluded.expected_impact, risk_assessment = excluded.risk_assessment,
                     signal_titles = excluded.signal_titles, updated_at = :now
                   WHERE recommendations.status = 'draft'""",
                params,
            )
            row = cur.execute("SELECT * FROM recommendations WHERE key = ? AND version = ?",
                              (rec.key, rec.version)).fetchone()
        return self._recommendation(row)

    def get_recommendation(self, recommendation_id: int) -> Recommendation | None:
        rows = self._q("SELECT * FROM recommendations WHERE id = ?", (recommendation_id,))
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
        params: list[Any] = []
        if status:
            where.append("r.status = ?")
            params.append(status)
        if topic:
            where.append("EXISTS (SELECT 1 FROM json_each(r.topics) WHERE value = ?)")
            params.append(topic)
        if business_unit:
            where.append("r.business_unit = ?")
            params.append(business_unit)
        if key:
            where.append("r.key = ?")
            params.append(key)
        clause = " AND ".join(where) if where else "1=1"
        rows = self._q(
            f"""SELECT * FROM recommendations r WHERE {clause}
                ORDER BY CASE r.priority WHEN 'High' THEN 0 WHEN 'Medium' THEN 1 ELSE 2 END, r.key, r.version DESC
                LIMIT ?""",
            [*params, limit],
        )
        return [self._recommendation(r) for r in rows]

    def set_recommendation_status(
        self, recommendation_id: int, status: RecommendationStatus, reviewer: str, reason: str
    ) -> Recommendation:
        if not reviewer.strip() or not reason.strip():
            raise ValueError("reviewer and reason are required for every status change")
        now = to_db(utcnow())
        with self._tx() as cur:
            row = cur.execute("SELECT status FROM recommendations WHERE id = ?", (recommendation_id,)).fetchone()
            if row is None:
                raise KeyError(f"recommendation {recommendation_id} not found")
            check_transition(row["status"], status)
            approved = status == "approved"
            cur.execute(
                """UPDATE recommendations SET status = ?, status_reason = ?, updated_at = ?,
                     approved_by = CASE WHEN ? THEN ? ELSE approved_by END,
                     approved_at = CASE WHEN ? THEN ? ELSE approved_at END
                   WHERE id = ?""",
                (status, reason, now, approved, reviewer, approved, now, recommendation_id),
            )
            cur.execute(
                """INSERT INTO recommendation_reviews (recommendation_id, from_status, to_status, reviewer, reason)
                   VALUES (?, ?, ?, ?, ?)""",
                (recommendation_id, row["status"], status, reviewer, reason),
            )
        result = self.get_recommendation(recommendation_id)
        assert result is not None
        return result

    def link_recommendation_evidence(
        self, recommendation_id: int, links: Sequence[EvidenceLinkIn], overwrite: bool = False
    ) -> int:
        conflict = (
            "DO UPDATE SET relation = excluded.relation, score = excluded.score, linked_by = excluded.linked_by, "
            "note = excluded.note" if overwrite else "DO NOTHING"
        )
        written = 0
        with self._tx() as cur:
            for link in links:
                written += cur.execute(
                    f"""INSERT INTO recommendation_evidence (recommendation_id, chunk_id, relation, score, linked_by,
                          note) VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT (recommendation_id, chunk_id) {conflict}""",
                    (recommendation_id, link.chunk_id, link.relation, link.score, link.linked_by, link.note),
                ).rowcount
        return written

    def recommendation_evidence(self, recommendation_id: int) -> list[RecommendationEvidence]:
        rows = self._q(
            """SELECT chunk_id, relation, score, linked_by, note FROM recommendation_evidence
               WHERE recommendation_id = ?
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
        rows = self._q("SELECT * FROM recommendation_reviews WHERE recommendation_id = ? ORDER BY id",
                       (recommendation_id,))
        return [
            RecommendationReview(recommendation_id=r["recommendation_id"], from_status=r["from_status"],
                                 to_status=r["to_status"], reviewer=r["reviewer"], reason=r["reason"],
                                 created_at=self._ts(r["created_at"]))
            for r in rows
        ]

    # ------------------------------------------------------------------ runs & stats
    def start_run(self, flow: str, params: dict[str, Any]) -> int:
        with self._tx() as cur:
            run_id = cur.execute("INSERT INTO pipeline_runs (flow, params) VALUES (?, ?)",
                                 (flow, json.dumps(params, default=str))).lastrowid
        return int(run_id)  # type: ignore[arg-type]

    def finish_run(self, run_id: int, status: str, stats: dict[str, Any], error: str | None = None) -> None:
        with self._tx() as cur:
            cur.execute(
                "UPDATE pipeline_runs SET status = ?, stats = ?, error = ?, finished_at = ? WHERE id = ?",
                (status, json.dumps(stats, default=str), error, to_db(utcnow()), run_id),
            )

    def stats(self) -> StoreStats:
        def one(sql: str) -> Any:
            return self._q(sql)[0][0]

        models = [
            r[0] for r in self._q("SELECT DISTINCT embedding_model FROM chunks WHERE embedding_model IS NOT NULL")
        ]
        return StoreStats(
            sources=one("SELECT COUNT(*) FROM sources"),
            documents=one("SELECT COUNT(*) FROM documents"),
            chunks=one("SELECT COUNT(*) FROM chunks"),
            embedded_chunks=one("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"),
            embedding_models=sorted(models),
            latest_collected_at=from_db(one("SELECT MAX(collected_at) FROM documents")),
        )


class _Tx:
    """``with store._tx() as cur:`` → BEGIN IMMEDIATE … COMMIT/ROLLBACK under the store lock."""

    def __init__(self, conn: sqlite3.Connection, lock: threading.RLock) -> None:
        self.conn, self.lock = conn, lock

    def __enter__(self) -> sqlite3.Connection:
        self.lock.acquire()
        self.conn.execute("BEGIN IMMEDIATE")
        return self.conn

    def __exit__(self, exc_type: object, *_: object) -> None:
        try:
            self.conn.execute("ROLLBACK" if exc_type else "COMMIT")
        finally:
            self.lock.release()

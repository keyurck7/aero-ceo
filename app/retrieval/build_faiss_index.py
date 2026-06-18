import json
import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

# Important in Datalab: use free GPU if configured.
cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES")
if cuda_visible:
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from app.db.local_db import get_connection, init_local_db
from app.processing.cleaner import chunk_text


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/cache/aero_ceo.faiss")
FAISS_METADATA_PATH = os.getenv("FAISS_METADATA_PATH", "data/cache/faiss_metadata.json")


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def reset_chunks(conn):
    cur = conn.cursor()
    cur.execute("DELETE FROM document_chunks")
    conn.commit()


def fetch_documents(conn) -> List[Dict]:
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT 
            d.id,
            d.title,
            d.url,
            d.clean_text,
            d.source_type,
            d.topic,
            d.trust_score,
            d.published_at,
            s.name AS source_name
        FROM documents d
        LEFT JOIN sources s ON d.source_id = s.id
        WHERE d.clean_text IS NOT NULL
          AND LENGTH(d.clean_text) > 50
        ORDER BY d.id
        """
    ).fetchall()

    return [dict(row) for row in rows]


def insert_chunk(conn, document_id: int, chunk_index: int, chunk: str, faiss_id: int) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO document_chunks (
            document_id, chunk_index, chunk_text, token_count, faiss_id
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            document_id,
            chunk_index,
            chunk,
            len(chunk.split()),
            faiss_id,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def build_chunks(conn, documents: List[Dict]) -> List[Dict]:
    all_chunks = []
    faiss_id = 0

    for doc in tqdm(documents, desc="Chunking documents"):
        chunks = chunk_text(doc["clean_text"], max_chars=1800, overlap=250)

        for idx, chunk in enumerate(chunks):
            if len(chunk.strip()) < 40:
                continue

            chunk_id = insert_chunk(
                conn=conn,
                document_id=doc["id"],
                chunk_index=idx,
                chunk=chunk,
                faiss_id=faiss_id,
            )

            all_chunks.append(
                {
                    "faiss_id": faiss_id,
                    "chunk_id": chunk_id,
                    "document_id": doc["id"],
                    "chunk_text": chunk,
                    "title": doc["title"],
                    "url": doc["url"],
                    "source_type": doc["source_type"],
                    "source_name": doc["source_name"],
                    "topic": doc["topic"],
                    "trust_score": doc["trust_score"],
                    "published_at": doc["published_at"],
                }
            )

            faiss_id += 1

    return all_chunks


def build_index(chunks: List[Dict]):
    if not chunks:
        raise ValueError("No chunks available to index.")

    device = get_device()
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    print(f"Device: {device}")

    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    texts = [item["chunk_text"] for item in chunks]

    print(f"Encoding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    dim = embeddings.shape[1]
    print(f"Embedding dimension: {dim}")

    # Normalized embeddings + inner product = cosine similarity.
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, embeddings.shape[0], dim


def save_index_and_metadata(index, chunks: List[Dict], dim: int):
    Path(FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(FAISS_METADATA_PATH).parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, FAISS_INDEX_PATH)

    metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": dim,
        "total_chunks": len(chunks),
        "items": chunks,
    }

    with open(FAISS_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"FAISS index saved to: {FAISS_INDEX_PATH}")
    print(f"Metadata saved to: {FAISS_METADATA_PATH}")


def main():
    init_local_db()
    conn = get_connection()

    print("Resetting old chunks...")
    reset_chunks(conn)

    documents = fetch_documents(conn)
    print(f"Documents available: {len(documents)}")

    chunks = build_chunks(conn, documents)
    print(f"Chunks created: {len(chunks)}")

    index, total_vectors, dim = build_index(chunks)
    print(f"Vectors indexed: {total_vectors}")

    save_index_and_metadata(index, chunks, dim)

    conn.close()


if __name__ == "__main__":
    main()

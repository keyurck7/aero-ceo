import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES")
if cuda_visible:
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible

import faiss
import torch
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/cache/aero_ceo.faiss")
FAISS_METADATA_PATH = os.getenv("FAISS_METADATA_PATH", "data/cache/faiss_metadata.json")


class AeroSearchEngine:
    def __init__(self):
        if not Path(FAISS_INDEX_PATH).exists():
            raise FileNotFoundError(f"FAISS index not found: {FAISS_INDEX_PATH}")

        if not Path(FAISS_METADATA_PATH).exists():
            raise FileNotFoundError(f"FAISS metadata not found: {FAISS_METADATA_PATH}")

        self.index = faiss.read_index(FAISS_INDEX_PATH)

        with open(FAISS_METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.items = self.metadata["items"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading search model: {EMBEDDING_MODEL}")
        print(f"Device: {device}")

        self.model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    def search(
        self,
        query: str,
        top_k: int = 8,
        topic: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> List[Dict]:
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        # Retrieve more than needed so filters still have room.
        search_k = max(top_k * 5, 25)
        scores, indices = self.index.search(query_embedding, search_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            item = self.items[int(idx)]

            if topic and item.get("topic") != topic:
                continue

            if source_type and item.get("source_type") != source_type:
                continue

            result = dict(item)
            result["score"] = float(score)
            results.append(result)

            if len(results) >= top_k:
                break

        return results


def print_results(results: List[Dict]):
    for i, r in enumerate(results, start=1):
        print("\n" + "=" * 100)
        print(f"Rank: {i}")
        print(f"Score: {r['score']:.4f}")
        print(f"Topic: {r.get('topic')}")
        print(f"Source type: {r.get('source_type')}")
        print(f"Source: {r.get('source_name')}")
        print(f"Title: {r.get('title')}")
        print(f"URL: {r.get('url')}")
        print("-" * 100)
        print(r.get("chunk_text", "")[:900])


def main():
    parser = argparse.ArgumentParser(description="Search AERO-CEO strategic intelligence memory.")
    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--topic", type=str, default=None)
    parser.add_argument("--source-type", type=str, default=None)
    args = parser.parse_args()

    engine = AeroSearchEngine()
    results = engine.search(
        query=args.query,
        top_k=args.top_k,
        topic=args.topic,
        source_type=args.source_type,
    )
    print_results(results)


if __name__ == "__main__":
    main()

"""Cross-encoder reranking: usually the largest precision gain per euro in RAG.

v1 had no reranker. v2 reranks the fused candidate list with a cross-encoder,
either served by TEI (``POST /rerank``) or loaded in-process.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

import requests

from aero.config import Settings


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Relevance in [0, 1] for each text, in input order."""
        ...


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class TEIReranker:
    def __init__(self, base_url: str, name: str, timeout: float = 30.0, session: requests.Session | None = None):
        self.base_url = base_url.rstrip("/")
        self.name = name
        self.timeout = timeout
        self._session = session or requests.Session()

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        resp = self._session.post(
            f"{self.base_url}/rerank",
            json={"query": query, "texts": list(texts), "raw_scores": False, "truncate": True},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        scores = [0.0] * len(texts)
        for item in resp.json():
            scores[int(item["index"])] = float(item["score"])
        return scores


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError("Install the 'local-models' extra for in-process reranking") from exc
        self.name = model_name
        self._model = CrossEncoder(model_name)

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        raw = [float(x) for x in self._model.predict([(query, t) for t in texts])]
        # sentence-transformers versions differ on whether a sigmoid is already applied.
        if all(0.0 <= x <= 1.0 for x in raw):
            return raw
        return [_sigmoid(x) for x in raw]


def build_reranker(settings: Settings) -> Reranker | None:
    if settings.reranker_backend == "none":
        return None
    if settings.reranker_backend == "tei":
        assert settings.tei_rerank_url
        return TEIReranker(settings.tei_rerank_url, settings.reranker_model)
    return CrossEncoderReranker(settings.reranker_model)

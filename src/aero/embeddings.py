"""Embedding backends behind one interface.

v1 loaded a SentenceTransformer into every process (CLI, Streamlit, index builder)
and pinned the GPU with ``CUDA_VISIBLE_DEVICES=1`` in eleven places (finding P4/P7).
v2 picks a backend by configuration:

* ``tei``: Hugging Face Text Embeddings Inference over HTTP. Production: one
  GPU-resident copy shared by all services.
* ``sentence-transformers``: in-process. Single-box development (your GPU server).
* ``hashing``: deterministic feature hashing. Tests and CI only; refused in prod.

All backends return L2-normalised float32 vectors, so inner product = cosine.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from itertools import pairwise
from typing import Protocol

import numpy as np
import requests

from aero.config import Settings
from aero.log import get_logger

log = get_logger(__name__)


class Embedder(Protocol):
    model_name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalise(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out: np.ndarray = matrix / norms
    return out


class HashingEmbedder:
    """Signed feature hashing over unigrams + bigrams. Lexical, deterministic, dependency-free."""

    _word = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.model_name = f"hashing-{dim}"

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        words = self._word.findall(text.lower())
        feats = words + [f"{a}_{b}" for a, b in pairwise(words)]
        for feat in feats:
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            h = int.from_bytes(digest, "big")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return vec

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalise(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        vec: np.ndarray = self.embed_documents([text])[0]
        return vec


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, dim: int, query_instruction: str = "", batch_size: int = 32) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("Install the 'local-models' extra: pip install 'aero-ceo[local-models]'") from exc
        self.model_name = model_name
        self.dim = dim
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name)  # device chosen by torch / CUDA_VISIBLE_DEVICES
        actual = self._model.get_sentence_embedding_dimension()
        if actual != dim:
            raise ValueError(f"{model_name} produces {actual}-d vectors but AERO_EMBEDDING_DIM={dim}")

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = self._model.encode(
            list(texts), batch_size=self.batch_size, convert_to_numpy=True, normalize_embeddings=True
        )
        return _normalise(out)

    def embed_query(self, text: str) -> np.ndarray:
        vec: np.ndarray = self.embed_documents([self.query_instruction + text])[0]
        return vec


class TEIEmbedder:
    """Client for Hugging Face Text Embeddings Inference (``POST /embed``)."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        dim: int,
        query_instruction: str = "",
        batch_size: int = 32,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.dim = dim
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self.timeout = timeout
        self._session = session or requests.Session()

    def _post(self, inputs: list[str]) -> np.ndarray:
        resp = self._session.post(
            f"{self.base_url}/embed",
            json={"inputs": inputs, "normalize": True, "truncate": True},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        matrix = np.asarray(resp.json(), dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.dim:
            raise ValueError(f"TEI returned shape {matrix.shape}; expected (n, {self.dim})")
        return matrix

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        parts = [self._post(list(texts[i : i + self.batch_size])) for i in range(0, len(texts), self.batch_size)]
        return _normalise(np.vstack(parts))

    def embed_query(self, text: str) -> np.ndarray:
        vec: np.ndarray = self.embed_documents([self.query_instruction + text])[0]
        return vec


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "hashing":
        return HashingEmbedder(settings.embedding_dim)
    if settings.embedding_backend == "tei":
        assert settings.tei_embed_url  # guaranteed by Settings validation
        return TEIEmbedder(
            settings.tei_embed_url,
            settings.embedding_model,
            settings.embedding_dim,
            settings.embedding_query_instruction,
            settings.embedding_batch_size,
        )
    return SentenceTransformerEmbedder(
        settings.embedding_model,
        settings.embedding_dim,
        settings.embedding_query_instruction,
        settings.embedding_batch_size,
    )

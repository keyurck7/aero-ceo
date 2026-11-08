"""Typed, validated configuration for every AERO-CEO v2 process.

v1 read ad-hoc ``os.getenv`` calls scattered across modules, and ``.env.example``
did not match the names the code actually read (finding B11). v2 has exactly one
settings object. Every variable is prefixed ``AERO_`` and validated at startup,
so a typo or an impossible value fails immediately instead of silently falling
back to a default.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ENV_PREFIX = "AERO_"


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    env: Literal["local", "test", "staging", "prod"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = False

    # --- storage -------------------------------------------------------------------
    # sqlite:///path/to.db  -> local/dev adapter (FTS5 + numpy)
    # postgresql://...      -> production adapter (pgvector + tsvector)
    database_url: str = "sqlite:///data/aero_v2.sqlite"

    # --- embeddings / reranking ----------------------------------------------------
    embedding_backend: Literal["sentence-transformers", "tei", "hashing"] = "sentence-transformers"
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = Field(1024, ge=8, le=4096)
    # BGE v1.5 retrieves better when short queries carry this instruction (passages do not).
    embedding_query_instruction: str = "Represent this sentence for searching relevant passages: "
    embedding_batch_size: int = Field(32, ge=1, le=1024)
    tei_embed_url: str | None = None
    reranker_backend: Literal["none", "tei", "cross-encoder"] = "none"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    tei_rerank_url: str | None = None

    # --- retrieval -----------------------------------------------------------------
    # hybrid = dense + lexical (default); dense / lexical exist for ablations.
    retrieval_mode: Literal["hybrid", "dense", "lexical"] = "hybrid"
    retrieval_candidate_k: int = Field(50, ge=5, le=500)
    retrieval_rrf_k: int = Field(60, ge=1)
    # Defaults set by the 2026-11-08 ablation (docs/v2-progress.md): ranking on relevance alone beat the
    # credibility/recency blend (nDCG@10 0.509 vs 0.493, MRR 0.917 vs 0.850). Trust is handled by a floor
    # (drops content-farm publishers) and reported per hit for the agent's coverage check, not mixed into rank.
    retrieval_credibility_weight: float = Field(0.0, ge=0.0, le=0.5)
    retrieval_recency_weight: float = Field(0.0, ge=0.0, le=0.5)
    retrieval_min_credibility: float = Field(0.15, ge=0.0, le=1.0)
    recency_half_life_days: float = Field(180.0, gt=0)

    # --- chunking ------------------------------------------------------------------
    chunk_max_words: int = Field(300, ge=50, le=2000)
    chunk_overlap_words: int = Field(45, ge=0)

    # --- ingestion -----------------------------------------------------------------
    # A generic client name. Never put a personal e-mail address here.
    http_user_agent: str = "AERO-CEO-research-crawler/2.0 (+https://github.com/keyurck7/aero-ceo)"
    http_timeout_seconds: float = Field(20.0, gt=0)
    http_max_retries: int = Field(3, ge=0, le=10)
    http_per_domain_delay_seconds: float = Field(1.0, ge=0)
    http_respect_robots: bool = True
    seeds_dir: Path = Path("seeds")

    # --- MCP servers ---------------------------------------------------------------
    mcp_host: str = "127.0.0.1"
    mcp_port: int = Field(8101, ge=1, le=65535)
    mcp_path: str = "/mcp"
    # Comma-separated Host headers accepted (DNS-rebinding protection), e.g. "evidence-mcp:8101,localhost:*".
    # Empty + host 127.0.0.1 -> localhost only. Empty + host 0.0.0.0 -> protection off (dev only; refused in prod).
    mcp_allowed_hosts: str = ""
    # Interim shared-secret auth until the OAuth 2.1 MCP gateway exists (roadmap phase 6). Empty = no auth.
    mcp_bearer_token: str = ""

    @property
    def mcp_allowed_host_list(self) -> list[str]:
        return [h.strip() for h in self.mcp_allowed_hosts.split(",") if h.strip()]

    @field_validator("database_url")
    @classmethod
    def _known_scheme(cls, v: str) -> str:
        if not (v.startswith("sqlite:///") or v.startswith(("postgresql://", "postgres://"))):
            raise ValueError("AERO_DATABASE_URL must start with sqlite:/// or postgresql://")
        return v

    @model_validator(mode="after")
    def _cross_checks(self) -> Settings:
        if self.chunk_overlap_words >= self.chunk_max_words:
            raise ValueError("AERO_CHUNK_OVERLAP_WORDS must be smaller than AERO_CHUNK_MAX_WORDS")
        if self.embedding_backend == "tei" and not self.tei_embed_url:
            raise ValueError("AERO_EMBEDDING_BACKEND=tei requires AERO_TEI_EMBED_URL")
        if self.reranker_backend == "tei" and not self.tei_rerank_url:
            raise ValueError("AERO_RERANKER_BACKEND=tei requires AERO_TEI_RERANK_URL")
        if self.env == "prod" and self.embedding_backend == "hashing":
            raise ValueError("The hashing embedder is for tests only; it is refused in prod")
        if self.env == "prod" and self.mcp_host not in ("127.0.0.1", "localhost") and not (
            self.mcp_allowed_hosts and self.mcp_bearer_token
        ):
            raise ValueError("In prod, a non-local MCP server needs AERO_MCP_ALLOWED_HOSTS and AERO_MCP_BEARER_TOKEN")
        return self

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith(("postgresql://", "postgres://"))

    @property
    def sqlite_path(self) -> Path:
        if self.is_postgres:
            raise ValueError("database_url is not a SQLite URL")
        return Path(self.database_url.removeprefix("sqlite:///"))


def _env_overrides(environ: dict[str, str]) -> dict[str, Any]:
    fields = Settings.model_fields
    out: dict[str, Any] = {}
    for key, value in environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        name = key[len(ENV_PREFIX) :].lower()
        if name not in fields:
            raise ValueError(f"Unknown setting {key}; valid names: {sorted(ENV_PREFIX + f.upper() for f in fields)}")
        out[name] = value
    return out


def load_settings(environ: dict[str, str] | None = None, dotenv_path: str | Path | None = ".env") -> Settings:
    """Build settings from ``AERO_*`` variables (and ``.env`` if present)."""
    if environ is None:
        if dotenv_path and Path(dotenv_path).exists():
            load_dotenv(dotenv_path, override=False)
        environ = dict(os.environ)
    return Settings(**_env_overrides(environ))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()

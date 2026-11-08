"""Forward-only SQL migrations (Flyway-style), one directory per dialect.

Files are named ``NNNN_description.sql`` and applied in order inside a
transaction. Applied versions are recorded in ``schema_migrations``. The only
templating is ``{{EMBEDDING_DIM}}``. Why plain SQL rather than Alembic: see
docs/adr/0002-postgres-pgvector-and-sql-migrations.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources

_NAME = re.compile(r"^(\d{4})_([a-z0-9_]+)\.sql$")


@dataclass(frozen=True)
class Migration:
    version: str
    name: str
    sql: str


def discover(dialect: str, embedding_dim: int) -> list[Migration]:
    if dialect not in ("sqlite", "postgres"):
        raise ValueError(dialect)
    folder = resources.files(__package__).joinpath(dialect)
    found = []
    for entry in folder.iterdir():
        m = _NAME.match(entry.name)
        if not m:
            continue
        sql = entry.read_text(encoding="utf-8").replace("{{EMBEDDING_DIM}}", str(int(embedding_dim)))
        found.append(Migration(m.group(1), m.group(2), sql))
    found.sort(key=lambda mig: mig.version)
    versions = [mig.version for mig in found]
    if len(versions) != len(set(versions)):
        raise RuntimeError(f"duplicate migration versions in {dialect}: {versions}")
    return found

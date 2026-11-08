"""Storage adapters. Use ``open_store(settings)``; never import an adapter directly."""

from __future__ import annotations

from aero.config import Settings
from aero.storage.base import DuplicateDocumentError, Store

__all__ = ["DuplicateDocumentError", "Store", "open_store"]


def open_store(settings: Settings, *, migrate: bool = False) -> Store:
    store: Store
    if settings.is_postgres:
        from aero.storage.postgres import PostgresStore  # optional extra

        store = PostgresStore(settings.database_url, settings.embedding_dim)
    else:
        from aero.storage.sqlite import SQLiteStore

        store = SQLiteStore(settings.sqlite_path, settings.embedding_dim)
    if migrate:
        store.migrate()
    return store

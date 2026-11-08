"""Date normalisation (fixes finding B4). Recency is a filter, never a weight (ADR 0008).

v1 parsed ``published_at`` only as RFC 2822 (RSS style). ISO-8601 dates from web
pages raised inside a bare ``except`` and silently scored 0.55. v2 normalises
every timestamp to timezone-aware UTC once, at ingestion, and rejects impossible
values instead of guessing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime

from dateutil import parser as dateutil_parser

MIN_PLAUSIBLE = datetime(1990, 1, 1, tzinfo=UTC)
FUTURE_TOLERANCE = timedelta(days=2)

_DB_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def utcnow() -> datetime:
    return datetime.now(UTC)


def as_utc(dt: datetime) -> datetime:
    """Naive datetimes are interpreted as UTC; aware ones are converted."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def parse_published_at(value: str | datetime | None, *, now: datetime | None = None) -> datetime | None:
    """Parse RFC 2822, ISO 8601 and common human formats into aware UTC.

    Returns ``None`` for empty, unparseable or implausible values (before 1990 or
    more than two days in the future) so bad dates never masquerade as fresh news.
    """
    if value is None:
        return None
    now = now or utcnow()
    parsed: datetime | None = None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = value.strip()
        if not text:
            return None
        try:
            parsed = parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError):
            parsed = None
        if parsed is None:
            try:
                parsed = dateutil_parser.isoparse(text)
            except (ValueError, OverflowError):
                parsed = None
        if parsed is None:
            try:
                parsed = dateutil_parser.parse(text, fuzzy=False)
            except (ValueError, OverflowError, dateutil_parser.ParserError):
                return None
    parsed = as_utc(parsed)
    if parsed < MIN_PLAUSIBLE or parsed > now + FUTURE_TOLERANCE:
        return None
    return parsed


def to_db(dt: datetime | None) -> str | None:
    """Fixed-width UTC text so SQLite string comparison equals time comparison."""
    return None if dt is None else as_utc(dt).strftime(_DB_FORMAT)


def from_db(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return as_utc(value)
    return datetime.strptime(value, _DB_FORMAT).replace(tzinfo=UTC)

"""Text processing: cleaning, canonical URLs, near-duplicate hashing, chunking, injection scan.

Replaces ``app/processing/cleaner.py``. The main behavioural changes:

* ``clean_text`` keeps paragraph breaks. v1 collapsed all whitespace into single
  spaces, so chunking could not respect document structure.
* ``chunk_text`` packs whole paragraphs and sentences up to a word budget, and
  returns character spans. Citations can then point at the exact passage
  (needed for findings B3/B7).
* ``canonical_url`` and ``simhash64`` give two dedup keys, so the same story from
  different redirect links or syndicated copies is stored once (finding B9).
* ``scan_injection`` flags instruction-like text in scraped pages before it can
  reach an LLM prompt (finding P2).
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# --------------------------------------------------------------------------- cleaning

_ZERO_WIDTH = re.compile("[​-‏⁠﻿]")
_INLINE_WS = re.compile(r"[ \t\f\v ]+")
_BOILERPLATE_LINE = re.compile(
    r"^(cookie (policy|settings|preferences)|privacy policy|terms of use|accept (all )?cookies"
    r"|subscribe to our newsletter|share this article|skip to (main )?content)\.?$",
    re.IGNORECASE,
)


def clean_text(text: str | None) -> str:
    """Normalise whitespace while keeping paragraph boundaries as blank lines."""
    if not text:
        return ""
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for raw in text.split("\n"):
        line = _INLINE_WS.sub(" ", raw).strip()
        if line and _BOILERPLATE_LINE.match(line):
            continue
        lines.append(line)
    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def normalise_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalise_for_hash(text).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- URLs

_TRACKING_PREFIXES = ("utm_", "mc_", "pk_", "hsa_", "_hs")
_TRACKING_KEYS = {
    "gclid", "fbclid", "dclid", "msclkid", "yclid", "igshid", "ocid", "cmpid", "ref", "ref_src",
    "spm", "_ga", "_gl", "cid", "smid", "taid", "partner", "src",
}
_AGGREGATOR_HOSTS = {"news.google.com", "news.yahoo.com", "www.bing.com", "bing.com", "flipboard.com"}


def canonical_url(url: str) -> str:
    """Stable identity for a URL: lower-case host, https, no www/fragment/tracking params."""
    raw = url.strip()
    parts = urlsplit(raw)
    if not parts.netloc:
        return raw
    scheme = "https" if parts.scheme in ("http", "https", "") else parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    port = parts.port
    netloc = host if port in (None, 80, 443) else f"{host}:{port}"
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=False)
        if k.lower() not in _TRACKING_KEYS and not k.lower().startswith(_TRACKING_PREFIXES)
    ]
    query.sort()
    return urlunsplit((scheme, netloc, path, urlencode(query), ""))


def is_aggregator_url(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return host in _AGGREGATOR_HOSTS


def url_host(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


# --------------------------------------------------------------------------- near-duplicates

_WORD = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?")
_MASK64 = (1 << 64) - 1


def simhash64(text: str, shingle: int = 3) -> int:
    """64-bit SimHash over word shingles. Near-identical texts differ in few bits."""
    words = _WORD.findall(text.lower())
    if not words:
        return 0
    grams = (
        [" ".join(words[i : i + shingle]) for i in range(len(words) - shingle + 1)]
        if len(words) >= shingle
        else [" ".join(words)]
    )
    acc = [0] * 64
    for gram in grams:
        h = int.from_bytes(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(64):
            acc[bit] += 1 if (h >> bit) & 1 else -1
    value = 0
    for bit in range(64):
        if acc[bit] > 0:
            value |= 1 << bit
    return value


def hamming64(a: int, b: int) -> int:
    return ((a ^ b) & _MASK64).bit_count()


def to_signed64(value: int) -> int:
    """Store an unsigned 64-bit hash in a signed BIGINT / SQLite INTEGER column."""
    value &= _MASK64
    return value - (1 << 64) if value >= (1 << 63) else value


def from_signed64(value: int) -> int:
    return value & _MASK64


# --------------------------------------------------------------------------- chunking


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    words: int


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str
    char_start: int
    char_end: int
    word_count: int


_PARAGRAPH = re.compile(r"[^\n]+(?:\n(?!\n)[^\n]+)*")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“(])")
_TOKEN = re.compile(r"\S+")


def _word_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    return [(start + m.start(), start + m.end()) for m in _TOKEN.finditer(text[start:end])]


def _split_long(text: str, start: int, end: int, max_words: int) -> list[Span]:
    """Split a paragraph that is over budget into sentences, then into word windows."""
    pieces: list[Span] = []
    cursor = start
    boundaries = [start + m.start() for m in _SENTENCE_END.finditer(text[start:end])]
    for cut in [*boundaries, end]:
        seg_start, seg_end = cursor, cut
        stripped = text[seg_start:seg_end].strip()
        if stripped:
            lead = len(text[seg_start:seg_end]) - len(text[seg_start:seg_end].lstrip())
            seg_start += lead
            seg_end = seg_start + len(stripped)
            words = _word_spans(text, seg_start, seg_end)
            if len(words) <= max_words:
                pieces.append(Span(seg_start, seg_end, len(words)))
            else:
                for i in range(0, len(words), max_words):
                    window = words[i : i + max_words]
                    pieces.append(Span(window[0][0], window[-1][1], len(window)))
        cursor = cut
    return pieces


def _units(text: str, max_words: int) -> list[Span]:
    units: list[Span] = []
    for m in _PARAGRAPH.finditer(text):
        start, end = m.start(), m.end()
        words = len(_TOKEN.findall(m.group(0)))
        if words == 0:
            continue
        if words <= max_words:
            units.append(Span(start, end, words))
        else:
            units.extend(_split_long(text, start, end, max_words))
    return units


def chunk_text(text: str, max_words: int = 300, overlap_words: int = 45, min_words: int = 8) -> list[TextChunk]:
    """Pack paragraphs/sentences into chunks of at most ``max_words`` words.

    Chunks overlap by whole units (at most ``overlap_words`` words) so a fact that
    straddles a boundary stays retrievable. Every chunk carries its character
    span in ``text``, so ``text[c.char_start:c.char_end] == c.text``.
    """
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")
    units = _units(text, max_words)
    if not units:
        return []
    chunks: list[TextChunk] = []
    i = 0
    while i < len(units):
        j, words = i, 0
        while j < len(units) and words + units[j].words <= max_words:
            words += units[j].words
            j += 1
        if j == i:  # single unit already capped at max_words by _units; defensive
            j, words = i + 1, units[i].words
        start, end = units[i].start, units[j - 1].end
        chunks.append(TextChunk(len(chunks), text[start:end], start, end, words))
        if j >= len(units):
            break
        # Step back over trailing units for overlap, but always make progress.
        back, back_words = j, 0
        while back - 1 > i and back_words + units[back - 1].words <= overlap_words:
            back -= 1
            back_words += units[back].words
        i = back
    if len(chunks) > 1:
        kept = [c for c in chunks if c.word_count >= min_words]
        chunks = [
            TextChunk(n, c.text, c.char_start, c.char_end, c.word_count) for n, c in enumerate(kept or chunks)
        ]
    return chunks


# --------------------------------------------------------------------------- injection scan

_INJECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "ignore_instructions": re.compile(
        r"\b(ignore|disregard|forget)\b.{0,40}\b(previous|prior|above|earlier|all)\b.{0,20}"
        r"\b(instructions?|prompts?|rules|directions)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "role_override": re.compile(
        r"\byou are now\b|\bnew instructions?\s*:|\bact as (an?|the) (ai|assistant|system)\b", re.I
    ),
    "prompt_markup": re.compile(r"<\|?(im_start|im_end|system|assistant)\|?>|</?(system|assistant)>|\[/?INST\]", re.I),
    "system_prompt_probe": re.compile(r"\b(reveal|print|show|repeat)\b.{0,30}\bsystem prompt\b", re.I),
    "exfiltration": re.compile(r"\b(send|post|forward|email)\b.{0,40}\b(api key|password|credentials|token)s?\b", re.I),
    "hidden_unicode": _ZERO_WIDTH,
}


def scan_injection(raw_text: str) -> list[str]:
    """Return the names of instruction-like patterns found in untrusted text.

    Flags do not delete content. They are stored on the document so retrieval
    can down-weight or quarantine it, and so the writer's prompt can warn.
    """
    return [name for name, pattern in _INJECTION_PATTERNS.items() if pattern.search(raw_text or "")]

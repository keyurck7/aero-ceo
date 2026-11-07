"""Polite HTTP fetching (replaces bare ``requests.get`` + ``time.sleep`` in v1 collectors; finding P6).

* Retries with exponential backoff on 429/5xx (urllib3 ``Retry``), honouring Retry-After.
* Minimum delay between requests to the same host, across threads.
* robots.txt respected and cached per host.
* Response size cap and content-type check, so a 2 GB file never reaches the extractor.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from aero.config import Settings
from aero.log import get_logger

log = get_logger(__name__)

MAX_BYTES = 5 * 1024 * 1024
TEXT_TYPES = ("text/html", "application/xhtml", "application/xml", "text/xml", "application/rss", "application/atom",
              "text/plain")


@dataclass(frozen=True)
class FetchResult:
    url: str
    final_url: str
    status: int
    content_type: str
    text: str


class PoliteFetcher:
    def __init__(
        self,
        user_agent: str,
        timeout: float = 20.0,
        max_retries: int = 3,
        per_host_delay: float = 1.0,
        respect_robots: bool = True,
        session: requests.Session | None = None,
    ) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self.per_host_delay = per_host_delay
        self.respect_robots = respect_robots
        self.session = session or requests.Session()
        if session is None:
            retry = Retry(
                total=max_retries, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=("GET", "HEAD"), respect_retry_after_header=True, raise_on_status=False,
            )
            adapter = HTTPAdapter(max_retries=retry, pool_maxsize=16)
            self.session.mount("https://", adapter)
            self.session.mount("http://", adapter)
        self.session.headers.update({"User-Agent": user_agent, "Accept-Language": "en;q=1.0, *;q=0.5"})
        self._last_hit: dict[str, float] = {}
        self._robots: dict[str, RobotFileParser | None] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, s: Settings) -> PoliteFetcher:
        return cls(s.http_user_agent, s.http_timeout_seconds, s.http_max_retries, s.http_per_domain_delay_seconds,
                   s.http_respect_robots)

    # ------------------------------------------------------------------ politeness
    def _wait_turn(self, host: str) -> None:
        with self._lock:
            now = time.monotonic()
            ready_at = self._last_hit.get(host, 0.0) + self.per_host_delay
            self._last_hit[host] = max(now, ready_at)
        delay = ready_at - now
        if delay > 0:
            time.sleep(delay)

    def allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            parser: RobotFileParser | None = RobotFileParser()
            try:
                resp = self.session.get(f"{origin}/robots.txt", timeout=self.timeout)
                if resp.status_code >= 400:
                    parser = None  # no robots.txt => allowed
                else:
                    assert parser is not None
                    parser.parse(resp.text.splitlines())
            except requests.RequestException:
                parser = None
            self._robots[origin] = parser
        parser = self._robots[origin]
        return True if parser is None else parser.can_fetch(self.user_agent, url)

    # ------------------------------------------------------------------ fetch
    def get(self, url: str) -> FetchResult | None:
        if not self.allowed(url):
            log.info("robots.txt disallows %s", url)
            return None
        self._wait_turn(urlsplit(url).netloc)
        try:
            with self.session.get(url, timeout=self.timeout, stream=True, allow_redirects=True) as resp:
                ctype = resp.headers.get("Content-Type", "").lower()
                if resp.status_code >= 400:
                    log.info("HTTP %s for %s", resp.status_code, url)
                    return None
                if ctype and not ctype.startswith(TEXT_TYPES):
                    log.info("skipping non-text %s (%s)", url, ctype)
                    return None
                body = b""
                for part in resp.iter_content(64 * 1024):
                    body += part
                    if len(body) > MAX_BYTES:
                        log.warning("response over %d bytes, truncated: %s", MAX_BYTES, url)
                        break
                encoding = resp.encoding or resp.apparent_encoding or "utf-8"
                return FetchResult(url, resp.url, resp.status_code, ctype, body.decode(encoding, errors="replace"))
        except requests.RequestException as exc:
            log.info("fetch failed for %s: %s", url, exc)
            return None

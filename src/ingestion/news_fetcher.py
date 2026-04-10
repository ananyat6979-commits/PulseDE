"""Multi-source financial news ingestion with content-level deduplication.

Sources:
  1. NewsAPI (primary, structured)
  2. RSS feeds (FT, WSJ, Bloomberg, Yahoo Finance)
  3. Alpha Vantage News Sentiment API (optional)

Deduplication strategy:
  - Exact: SHA-256 content hash stored in Redis (24 h window)
  - Near-duplicate: SimHash with Hamming distance < 4 (catches reworded duplicates)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator

import feedparser
import httpx
import redis as redis_lib
from simhash import Simhash

from config.settings import settings
from src.ingestion.schema import RawArticle

logger = logging.getLogger(__name__)

# SimHash bit-vector prefix for Redis key namespace
_SIMHASH_NS = "pulsede:simhash:"
_HASH_NS = "pulsede:hash:"


@dataclass
class FetchStats:
    source: str
    fetched: int = 0
    deduplicated: int = 0
    errors: int = 0

    @property
    def yield_rate(self) -> float:
        if self.fetched == 0:
            return 0.0
        return (self.fetched - self.deduplicated) / self.fetched


class NewsFetcher:
    """Fetches and deduplicates financial news from multiple sources."""

    def __init__(self) -> None:
        self._http = httpx.Client(
            timeout=10.0,
            headers={"User-Agent": "PulseDE/2.0 (+https://github.com/ananyat6979-commits/PulseDE)"},
        )
        self._redis = redis_lib.Redis(
            host=settings.redis.host,
            port=settings.redis.port,
            db=settings.redis.db,
            password=settings.redis.password.get_secret_value() if settings.redis.password else None,
            decode_responses=True,
        )
        self._stats: list[FetchStats] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def fetch_all(self) -> list[RawArticle]:
        """Fetch from all sources; return deduplicated, sorted-by-time articles."""
        articles: list[RawArticle] = []
        articles.extend(self._fetch_newsapi())
        articles.extend(self._fetch_rss_feeds())
        if settings.news.alpha_vantage_key:
            articles.extend(self._fetch_alpha_vantage())

        unique = list(self._deduplicate(articles))
        unique.sort(key=lambda a: a.published_at, reverse=True)

        for stat in self._stats:
            logger.info(
                "fetch_stats",
                extra={
                    "source": stat.source,
                    "fetched": stat.fetched,
                    "deduped": stat.deduplicated,
                    "yield_rate": f"{stat.yield_rate:.1%}",
                },
            )
        return unique[: settings.news.max_articles_per_fetch]

    # ── Private source adapters ────────────────────────────────────────────────

    def _fetch_newsapi(self) -> list[RawArticle]:
        stat = FetchStats(source="newsapi")
        self._stats.append(stat)
        articles: list[RawArticle] = []

        queries = ["stock market", "earnings", "Fed rates", "inflation", "S&P 500", "crypto"]
        api_key = settings.news.news_api_key.get_secret_value()

        for query in queries:
            try:
                resp = self._http.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        "q": query,
                        "language": "en",
                        "sortBy": "publishedAt",
                        "pageSize": 20,
                        "apiKey": api_key,
                    },
                )
                resp.raise_for_status()
                for item in resp.json().get("articles", []):
                    if not item.get("title") or item["title"] == "[Removed]":
                        continue
                    article = RawArticle(
                        source=f"newsapi:{item.get('source', {}).get('name', 'unknown')}",
                        headline=item["title"].strip(),
                        url=item.get("url", ""),
                        published_at=datetime.fromisoformat(
                            item["publishedAt"].replace("Z", "+00:00")
                        ),
                        body=item.get("content") or item.get("description") or "",
                        author=item.get("author") or "",
                    )
                    articles.append(article)
                    stat.fetched += 1
            except httpx.HTTPError as exc:
                logger.warning("newsapi_fetch_error", extra={"query": query, "error": str(exc)})
                stat.errors += 1
            time.sleep(0.1)  # respect rate limit

        return articles

    def _fetch_rss_feeds(self) -> list[RawArticle]:
        stat = FetchStats(source="rss")
        self._stats.append(stat)
        articles: list[RawArticle] = []

        for feed_url in settings.news.rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                source_name = feed.feed.get("title", feed_url)
                for entry in feed.entries[:20]:
                    pub = entry.get("published_parsed") or entry.get("updated_parsed")
                    published_at = (
                        datetime(*pub[:6], tzinfo=timezone.utc) if pub else datetime.now(timezone.utc)
                    )
                    article = RawArticle(
                        source=f"rss:{source_name}",
                        headline=(entry.get("title") or "").strip(),
                        url=entry.get("link") or "",
                        published_at=published_at,
                        body=entry.get("summary") or "",
                        author=entry.get("author") or "",
                    )
                    if article.headline:
                        articles.append(article)
                        stat.fetched += 1
            except Exception as exc:
                logger.warning("rss_fetch_error", extra={"feed": feed_url, "error": str(exc)})
                stat.errors += 1

        return articles

    def _fetch_alpha_vantage(self) -> list[RawArticle]:
        """Alpha Vantage News Sentiment API — provides pre-labeled articles (ignored here)."""
        stat = FetchStats(source="alpha_vantage")
        self._stats.append(stat)
        articles: list[RawArticle] = []

        try:
            resp = self._http.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "NEWS_SENTIMENT",
                    "topics": "financial_markets,earnings",
                    "limit": 50,
                    "apikey": settings.news.alpha_vantage_key.get_secret_value(),  # type: ignore[union-attr]
                },
            )
            resp.raise_for_status()
            for item in resp.json().get("feed", []):
                pub_str = item.get("time_published", "")
                try:
                    published_at = datetime.strptime(pub_str, "%Y%m%dT%H%M%S").replace(
                        tzinfo=timezone.utc
                    )
                except ValueError:
                    published_at = datetime.now(timezone.utc)

                article = RawArticle(
                    source=f"av:{item.get('source', 'unknown')}",
                    headline=(item.get("title") or "").strip(),
                    url=item.get("url") or "",
                    published_at=published_at,
                    body=item.get("summary") or "",
                    author=item.get("authors", [None])[0] or "",
                )
                if article.headline:
                    articles.append(article)
                    stat.fetched += 1
        except httpx.HTTPError as exc:
            logger.warning("av_fetch_error", extra={"error": str(exc)})
            stat.errors += 1

        return articles

    # ── Deduplication ──────────────────────────────────────────────────────────

    def _deduplicate(self, articles: list[RawArticle]) -> Iterator[RawArticle]:
        """Yield only articles unseen in the 24 h deduplication window.

        Two-stage:
          1. Exact match via Redis SET with content_hash (O(1) lookup).
          2. Near-duplicate via SimHash stored as integer, Hamming distance < 4.
        """
        pipeline = self._redis.pipeline()
        ttl = settings.redis.dedup_window_seconds

        for article in articles:
            exact_key = f"{_HASH_NS}{article.content_hash}"

            # Stage 1: exact hash check
            if self._redis.exists(exact_key):
                continue

            # Stage 2: SimHash near-duplicate
            sh = Simhash(article.headline)
            sh_int = sh.value
            is_near_dup = self._is_near_duplicate(sh_int)

            if is_near_dup:
                continue

            # Mark as seen
            pipeline.setex(exact_key, ttl, "1")
            self._store_simhash(sh_int, ttl)
            pipeline.execute()

            yield article

    def _is_near_duplicate(self, simhash_int: int) -> bool:
        """Check Hamming distance against stored SimHashes (naïve scan; fine at <10k/day)."""
        keys = self._redis.keys(f"{_SIMHASH_NS}*")
        for key in keys:
            stored_val = self._redis.get(key)
            if stored_val is None:
                continue
            stored_int = int(stored_val)
            hamming = bin(simhash_int ^ stored_int).count("1")
            if hamming < 4:
                return True
        return False

    def _store_simhash(self, simhash_int: int, ttl: int) -> None:
        key = f"{_SIMHASH_NS}{simhash_int}"
        self._redis.setex(key, ttl, str(simhash_int))

    def close(self) -> None:
        self._http.close()
        self._redis.close()

    def __enter__(self) -> "NewsFetcher":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

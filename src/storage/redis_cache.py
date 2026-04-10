"""Redis caching layer — low-latency dashboard reads, rate limiting, pub/sub.

Responsibilities:
  - Cache latest N sentiment results per source (sorted set, TTL 1h).
  - Cache aggregated sentiment scores per ticker (hash, TTL 5m).
  - Sliding-window rate limiter for the FastAPI layer.
  - Pub/Sub channel for real-time dashboard WebSocket pushes.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import redis as redis_lib

from config.settings import settings
from src.ingestion.schema import SentimentResult

logger = logging.getLogger(__name__)

_LATEST_KEY = "pulsede:latest"          # ZSET scored by published_at timestamp
_TICKER_KEY = "pulsede:ticker:{ticker}"  # HASH
_RATE_KEY = "pulsede:rate:{client_id}"  # STRING (counter)
_PUBSUB_CHANNEL = "pulsede:realtime"


class RedisCache:
    def __init__(self) -> None:
        self._r = redis_lib.Redis(
            host=settings.redis.host,
            port=settings.redis.port,
            db=settings.redis.db,
            password=(
                settings.redis.password.get_secret_value()
                if settings.redis.password
                else None
            ),
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )

    # ── Caching ────────────────────────────────────────────────────────────────

    def cache_results(self, results: list[SentimentResult]) -> None:
        """Cache results in a sorted set keyed by timestamp for O(log n) range queries."""
        pipe = self._r.pipeline()
        ttl = settings.redis.ttl_seconds

        for r in results:
            score = r.published_at.timestamp()
            value = json.dumps(r.to_dict())
            pipe.zadd(_LATEST_KEY, {value: score})
            # Publish to WebSocket subscribers
            pipe.publish(_PUBSUB_CHANNEL, value)

        # Keep only the latest 1000 entries in the sorted set
        pipe.zremrangebyrank(_LATEST_KEY, 0, -1001)
        pipe.expire(_LATEST_KEY, ttl)
        pipe.execute()

        # Update per-ticker aggregated scores
        self._update_ticker_aggregates(results)

    def get_latest(self, n: int = 100) -> list[dict[str, Any]]:
        """Fetch the n most recent results (high-score = most recent)."""
        raw = self._r.zrevrange(_LATEST_KEY, 0, n - 1, withscores=False)
        return [json.loads(v) for v in raw]

    def get_ticker_summary(self, ticker: str) -> dict[str, Any] | None:
        key = _TICKER_KEY.format(ticker=ticker)
        data = self._r.hgetall(key)
        if not data:
            return None
        return {
            "ticker": ticker,
            "positive_pct": float(data.get("positive_pct", 0)),
            "negative_pct": float(data.get("negative_pct", 0)),
            "neutral_pct": float(data.get("neutral_pct", 0)),
            "article_count": int(data.get("article_count", 0)),
            "avg_confidence": float(data.get("avg_confidence", 0)),
            "avg_uncertainty": float(data.get("avg_uncertainty", 0)),
        }

    def _update_ticker_aggregates(self, results: list[SentimentResult]) -> None:
        from collections import defaultdict

        ticker_data: dict[str, list[SentimentResult]] = defaultdict(list)
        for r in results:
            for ticker in r.tickers:
                ticker_data[ticker].append(r)

        pipe = self._r.pipeline()
        for ticker, ticker_results in ticker_data.items():
            key = _TICKER_KEY.format(ticker=ticker)
            n = len(ticker_results)
            pos = sum(1 for r in ticker_results if r.ensemble_sentiment.value == "positive")
            neg = sum(1 for r in ticker_results if r.ensemble_sentiment.value == "negative")
            neu = sum(1 for r in ticker_results if r.ensemble_sentiment.value == "neutral")
            avg_conf = sum(r.ensemble_confidence for r in ticker_results) / n
            avg_unc = sum(r.ensemble_uncertainty for r in ticker_results) / n

            existing_count = int(self._r.hget(key, "article_count") or 0)
            pipe.hset(key, mapping={
                "positive_pct": round(pos / n, 4),
                "negative_pct": round(neg / n, 4),
                "neutral_pct": round(neu / n, 4),
                "article_count": existing_count + n,
                "avg_confidence": round(avg_conf, 4),
                "avg_uncertainty": round(avg_unc, 4),
            })
            pipe.expire(key, 300)  # 5 min TTL; refreshed on each write

        pipe.execute()

    # ── Rate limiting (sliding window counter) ─────────────────────────────────

    def is_rate_limited(self, client_id: str, limit: int = 60, window_s: int = 60) -> bool:
        key = _RATE_KEY.format(client_id=client_id)
        pipe = self._r.pipeline()
        pipe.incr(key)
        pipe.expire(key, window_s)
        count, _ = pipe.execute()
        return int(count) > limit

    # ── Pub/Sub ────────────────────────────────────────────────────────────────

    def get_pubsub(self) -> redis_lib.client.PubSub:
        ps = self._r.pubsub()
        ps.subscribe(_PUBSUB_CHANNEL)
        return ps

    def close(self) -> None:
        self._r.close()

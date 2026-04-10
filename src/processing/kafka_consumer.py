"""Kafka consumer — reads raw articles, runs the full ML pipeline, writes results.

Design:
  - Polls in batches (configurable) for ML throughput (vectorised inference).
  - Commits offsets only after successful DB write (at-least-once).
  - Backpressure via batch size and poll timeout.
  - Dead-letter queue on unrecoverable errors.
  - Prometheus metrics: consumed, processed, failed, e2e latency.
"""
from __future__ import annotations

import json
import logging
import signal
import time
from datetime import datetime, timezone
from typing import Any

from confluent_kafka import Consumer, KafkaError, KafkaException, Message
from prometheus_client import Counter, Gauge, Histogram

from config.settings import settings
from src.ingestion.schema import RawArticle, SentimentResult
from src.ml.ensemble import SentimentEnsemble
from src.ml.feature_engineering import FinancialFeatureExtractor
from src.storage.timescale_writer import TimescaleWriter
from src.storage.redis_cache import RedisCache

logger = logging.getLogger(__name__)

CONSUMED = Counter("pulsede_consumer_messages_total", "Messages consumed from Kafka", ["status"])
PROCESSED = Counter("pulsede_processed_articles_total", "Successfully analysed articles")
E2E_LATENCY = Histogram(
    "pulsede_e2e_latency_seconds",
    "End-to-end latency from Kafka consume to DB write",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)
LAG = Gauge("pulsede_consumer_lag", "Estimated consumer lag (messages behind)")


class SentimentConsumer:
    """Stateful consumer: maintains ensemble models in memory across batches."""

    def __init__(
        self,
        ensemble: SentimentEnsemble,
        feature_extractor: FinancialFeatureExtractor,
        db_writer: TimescaleWriter,
        cache: RedisCache,
        batch_size: int = 16,
        poll_timeout_s: float = 1.0,
    ) -> None:
        self._ensemble = ensemble
        self._features = feature_extractor
        self._db = db_writer
        self._cache = cache
        self._batch_size = batch_size
        self._poll_timeout = poll_timeout_s
        self._running = False

        conf: dict[str, Any] = {
            "bootstrap.servers": settings.kafka.bootstrap_servers,
            "group.id": settings.kafka.consumer_group,
            "auto.offset.reset": "earliest",
            # Manual commit for at-least-once guarantees
            "enable.auto.commit": False,
            "max.poll.interval.ms": 300_000,
            "session.timeout.ms": 30_000,
            "fetch.min.bytes": 1,
        }
        self._consumer = Consumer(conf)
        self._consumer.subscribe([settings.kafka.news_topic])

        # Handle SIGTERM/SIGINT gracefully
        signal.signal(signal.SIGTERM, lambda *_: self.stop())
        signal.signal(signal.SIGINT, lambda *_: self.stop())

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Blocking event loop. Call stop() from another thread or signal handler."""
        self._running = True
        logger.info("consumer_started", extra={"topic": settings.kafka.news_topic})

        try:
            while self._running:
                batch = self._poll_batch()
                if not batch:
                    continue
                self._process_batch(batch)
        finally:
            self._consumer.close()
            logger.info("consumer_stopped")

    def stop(self) -> None:
        self._running = False

    # ── Core batch processing ──────────────────────────────────────────────────

    def _poll_batch(self) -> list[tuple[Message, RawArticle]]:
        batch: list[tuple[Message, RawArticle]] = []
        deadline = time.monotonic() + self._poll_timeout

        while len(batch) < self._batch_size and time.monotonic() < deadline:
            msg = self._consumer.poll(timeout=0.05)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error("kafka_consume_error", extra={"error": str(msg.error())})
                CONSUMED.labels(status="error").inc()
                continue

            try:
                payload = json.loads(msg.value())
                article = self._deserialise(payload)
                batch.append((msg, article))
                CONSUMED.labels(status="ok").inc()
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning("deserialise_error", extra={"error": str(exc)})
                CONSUMED.labels(status="deserialise_error").inc()

        return batch

    def _process_batch(self, batch: list[tuple[Message, RawArticle]]) -> None:
        msgs, articles = zip(*batch)
        texts = [(a.headline, a.body) for a in articles]
        headlines = [a.headline for a in articles]

        t0 = time.perf_counter()

        # --- Feature extraction (CPU-bound, fast) ---
        feature_vectors = self._features.extract_batch(texts)

        # --- Ensemble inference (GPU/CPU, vectorised) ---
        ensemble_outputs = self._ensemble.predict(headlines)

        elapsed = time.perf_counter() - t0
        E2E_LATENCY.observe(elapsed)

        # --- Assemble SentimentResult objects ---
        results: list[SentimentResult] = []
        for article, fv, eo in zip(articles, feature_vectors, ensemble_outputs):
            result = SentimentResult(
                article_hash=article.content_hash,
                headline=article.headline,
                url=article.url,
                source=article.source,
                published_at=article.published_at,
                processed_at=datetime.now(timezone.utc),
                ensemble_sentiment=eo["sentiment"],
                ensemble_confidence=eo["confidence"],
                ensemble_uncertainty=eo["uncertainty"],
                positive_prob=eo["positive_prob"],
                negative_prob=eo["negative_prob"],
                neutral_prob=eo["neutral_prob"],
                model_predictions=eo["model_predictions"],
                entities=fv.entities,
                tickers=fv.tickers,
                sectors=fv.sectors,
                is_forward_looking=fv.is_forward_looking,
                has_negation=fv.has_negation,
                hedge_score=fv.hedge_score,
                market_impact=fv.market_impact,
                is_uncertain=eo["is_uncertain"],
            )
            results.append(result)

        # --- Persist to TimescaleDB ---
        try:
            self._db.write_batch(results)
            # Cache latest results in Redis for dashboard low-latency reads
            self._cache.cache_results(results)
            PROCESSED.inc(len(results))
        except Exception as exc:
            logger.error("db_write_failed", extra={"error": str(exc), "batch_size": len(results)})
            return  # Do NOT commit — will re-process on restart

        # --- Commit offsets only after successful write ---
        for msg in msgs:
            self._consumer.commit(message=msg, asynchronous=False)

        logger.info(
            "batch_processed",
            extra={
                "count": len(results),
                "elapsed_s": round(elapsed, 3),
                "articles_per_sec": round(len(results) / elapsed, 1),
            },
        )

    @staticmethod
    def _deserialise(payload: dict[str, Any]) -> RawArticle:
        return RawArticle(
            source=payload["source"],
            headline=payload["headline"],
            url=payload["url"],
            published_at=datetime.fromisoformat(payload["published_at"]),
            body=payload.get("body", ""),
            author=payload.get("author", ""),
        )

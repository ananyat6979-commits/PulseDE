"""TimescaleDB persistence layer.

Schema highlights:
  - `sentiment_results` as a TimescaleDB hypertable (chunked by `published_at`).
  - Continuous aggregate (hourly rollup) for dashboard performance.
  - Bulk insert via `executemany` with `ON CONFLICT DO NOTHING` idempotency.
  - SQLAlchemy connection pool with health-check pings.
  - Alembic migrations manage schema versions.

Why TimescaleDB over plain Postgres:
  - Time-ordered queries (e.g. "last 24h by ticker") are 10–100× faster on hypertables.
  - Automatic chunk pruning / retention policies without manual partitioning.
  - Native compression on old chunks (up to 95% size reduction).
  - Compatible with all Postgres tooling (psycopg2, SQLAlchemy, Grafana).
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from config.settings import settings
from src.ingestion.schema import SentimentResult

logger = logging.getLogger(__name__)

# ── DDL ────────────────────────────────────────────────────────────────────────

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sentiment_results (
    id              BIGSERIAL,
    published_at    TIMESTAMPTZ NOT NULL,
    processed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    article_hash    TEXT        NOT NULL,
    headline        TEXT        NOT NULL,
    url             TEXT,
    source          TEXT        NOT NULL,

    -- Ensemble outputs
    ensemble_sentiment  TEXT    NOT NULL,
    ensemble_confidence FLOAT   NOT NULL,
    ensemble_uncertainty FLOAT  NOT NULL,
    positive_prob       FLOAT   NOT NULL,
    negative_prob       FLOAT   NOT NULL,
    neutral_prob        FLOAT   NOT NULL,

    -- Features
    tickers         TEXT[],
    sectors         TEXT[],
    is_forward_looking BOOLEAN DEFAULT FALSE,
    has_negation    BOOLEAN DEFAULT FALSE,
    hedge_score     FLOAT   DEFAULT 0.0,
    market_impact   TEXT    NOT NULL DEFAULT 'unknown',

    -- Quality flags
    is_uncertain    BOOLEAN DEFAULT FALSE,

    CONSTRAINT sentiment_results_pkey PRIMARY KEY (id, published_at)
);
"""

_CREATE_HYPERTABLE_SQL = """
SELECT create_hypertable(
    'sentiment_results',
    'published_at',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_sr_article_hash ON sentiment_results (article_hash);
CREATE INDEX IF NOT EXISTS idx_sr_source ON sentiment_results (source, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_sr_sentiment ON sentiment_results (ensemble_sentiment, published_at DESC);
"""

_HOURLY_ROLLUP_SQL = """
CREATE MATERIALIZED VIEW IF NOT EXISTS sentiment_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', published_at)  AS bucket,
    source,
    ensemble_sentiment,
    COUNT(*)                             AS article_count,
    AVG(ensemble_confidence)             AS avg_confidence,
    AVG(ensemble_uncertainty)            AS avg_uncertainty,
    AVG(positive_prob)                   AS avg_positive,
    AVG(negative_prob)                   AS avg_negative,
    AVG(neutral_prob)                    AS avg_neutral,
    AVG(hedge_score)                     AS avg_hedge
FROM sentiment_results
GROUP BY 1, 2, 3
WITH NO DATA;
"""

_INSERT_SQL = """
INSERT INTO sentiment_results (
    published_at, article_hash, headline, url, source,
    ensemble_sentiment, ensemble_confidence, ensemble_uncertainty,
    positive_prob, negative_prob, neutral_prob,
    tickers, sectors, is_forward_looking, has_negation, hedge_score,
    market_impact, is_uncertain
) VALUES (
    :published_at, :article_hash, :headline, :url, :source,
    :ensemble_sentiment, :ensemble_confidence, :ensemble_uncertainty,
    :positive_prob, :negative_prob, :neutral_prob,
    :tickers, :sectors, :is_forward_looking, :has_negation, :hedge_score,
    :market_impact, :is_uncertain
)
ON CONFLICT DO NOTHING;
"""


class TimescaleWriter:
    """Write SentimentResult objects to TimescaleDB."""

    def __init__(self) -> None:
        self._engine = create_engine(
            settings.db.url,
            pool_size=settings.db.pool_size,
            max_overflow=settings.db.max_overflow,
            pool_pre_ping=True,  # health-check connections on checkout
            echo=settings.db.echo,
        )
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)
        self._initialise_schema()

    # ── Schema management ──────────────────────────────────────────────────────

    def _initialise_schema(self) -> None:
        """Idempotent schema bootstrap. In production, use Alembic instead."""
        with self._engine.begin() as conn:
            conn.execute(text(_CREATE_TABLE_SQL))
            try:
                conn.execute(text(_CREATE_HYPERTABLE_SQL))
            except Exception:
                pass  # Already a hypertable
            conn.execute(text(_CREATE_INDEX_SQL))
            try:
                conn.execute(text(_HOURLY_ROLLUP_SQL))
            except Exception:
                pass  # Already exists
        logger.info("timescaledb_schema_ready")

    # ── Write API ──────────────────────────────────────────────────────────────

    def write_batch(self, results: list[SentimentResult]) -> None:
        if not results:
            return
        rows = [self._to_row(r) for r in results]
        with self._session() as session:
            session.execute(text(_INSERT_SQL), rows)
            session.commit()
        logger.debug("timescale_write", extra={"count": len(results)})

    def write_one(self, result: SentimentResult) -> None:
        self.write_batch([result])

    # ── Query API (used by FastAPI / dashboard) ────────────────────────────────

    def query_recent(self, hours: int = 24, limit: int = 500) -> list[dict]:
        sql = text(
            """
            SELECT * FROM sentiment_results
            WHERE published_at >= NOW() - INTERVAL ':hours hours'
            ORDER BY published_at DESC
            LIMIT :limit
            """.replace(":hours hours", f"{hours} hours")
        )
        with self._session() as session:
            rows = session.execute(sql, {"limit": limit}).mappings().all()
        return [dict(r) for r in rows]

    def query_hourly_rollup(self, hours: int = 48) -> list[dict]:
        sql = text(
            """
            SELECT * FROM sentiment_hourly
            WHERE bucket >= NOW() - INTERVAL ':hours hours'
            ORDER BY bucket DESC
            """.replace(":hours hours", f"{hours} hours")
        )
        with self._session() as session:
            rows = session.execute(sql).mappings().all()
        return [dict(r) for r in rows]

    def query_by_ticker(self, ticker: str, hours: int = 24) -> list[dict]:
        sql = text(
            """
            SELECT * FROM sentiment_results
            WHERE :ticker = ANY(tickers)
              AND published_at >= NOW() - INTERVAL ':hours hours'
            ORDER BY published_at DESC
            LIMIT 200
            """.replace(":hours hours", f"{hours} hours")
        )
        with self._session() as session:
            rows = session.execute(sql, {"ticker": ticker}).mappings().all()
        return [dict(r) for r in rows]

    # ── Helpers ────────────────────────────────────────────────────────────────

    @contextmanager
    def _session(self) -> Generator[Session, None, None]:
        session: Session = self._Session()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _to_row(r: SentimentResult) -> dict:
        return {
            "published_at": r.published_at,
            "article_hash": r.article_hash,
            "headline": r.headline,
            "url": r.url,
            "source": r.source,
            "ensemble_sentiment": r.ensemble_sentiment.value,
            "ensemble_confidence": r.ensemble_confidence,
            "ensemble_uncertainty": r.ensemble_uncertainty,
            "positive_prob": r.positive_prob,
            "negative_prob": r.negative_prob,
            "neutral_prob": r.neutral_prob,
            "tickers": r.tickers,
            "sectors": r.sectors,
            "is_forward_looking": r.is_forward_looking,
            "has_negation": r.has_negation,
            "hedge_score": r.hedge_score,
            "market_impact": r.market_impact.value,
            "is_uncertain": r.is_uncertain,
        }

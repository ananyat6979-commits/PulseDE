"""Strongly-typed domain models and Avro schema definitions.

Using Python dataclasses (not Pydantic) for Kafka serialisation compatibility,
with a matching Pydantic model for the API layer in serving/api.py.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class MarketImpact(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class RawArticle:
    """Represents a raw financial news article from any source."""

    source: str
    headline: str
    url: str
    published_at: datetime
    body: str = ""
    author: str = ""
    # Stable content fingerprint for deduplication
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        self.content_hash = hashlib.sha256(
            f"{self.headline}{self.url}".encode()
        ).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "headline": self.headline,
            "url": self.url,
            "published_at": self.published_at.isoformat(),
            "body": self.body,
            "author": self.author,
            "content_hash": self.content_hash,
        }


@dataclass
class EntityMention:
    """A named entity extracted from article text."""

    text: str
    entity_type: str  # ORG, PER, TICKER, LOC
    confidence: float
    start_char: int
    end_char: int


@dataclass
class ModelPrediction:
    """Single-model sentiment prediction."""

    model_name: str
    sentiment: Sentiment
    positive_prob: float
    negative_prob: float
    neutral_prob: float
    latency_ms: float


@dataclass
class SentimentResult:
    """Full enriched sentiment result persisted to TimescaleDB."""

    article_hash: str
    headline: str
    url: str
    source: str
    published_at: datetime
    processed_at: datetime

    # Ensemble outputs
    ensemble_sentiment: Sentiment
    ensemble_confidence: float          # max class probability after temperature scaling
    ensemble_uncertainty: float         # predictive entropy across MC dropout passes
    positive_prob: float
    negative_prob: float
    neutral_prob: float

    # Per-model predictions for audit trail
    model_predictions: list[ModelPrediction]

    # Feature engineering outputs
    entities: list[EntityMention]
    tickers: list[str]
    sectors: list[str]
    is_forward_looking: bool
    has_negation: bool
    hedge_score: float                  # 0-1 how hedged the language is
    market_impact: MarketImpact

    # Calibration
    is_uncertain: bool
    temperature_scaled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "article_hash": self.article_hash,
            "headline": self.headline,
            "url": self.url,
            "source": self.source,
            "published_at": self.published_at.isoformat(),
            "processed_at": self.processed_at.isoformat(),
            "ensemble_sentiment": self.ensemble_sentiment.value,
            "ensemble_confidence": round(self.ensemble_confidence, 4),
            "ensemble_uncertainty": round(self.ensemble_uncertainty, 4),
            "positive_prob": round(self.positive_prob, 4),
            "negative_prob": round(self.negative_prob, 4),
            "neutral_prob": round(self.neutral_prob, 4),
            "tickers": self.tickers,
            "sectors": self.sectors,
            "is_forward_looking": self.is_forward_looking,
            "has_negation": self.has_negation,
            "hedge_score": round(self.hedge_score, 4),
            "market_impact": self.market_impact.value,
            "is_uncertain": self.is_uncertain,
            "model_count": len(self.model_predictions),
        }


# ── Avro schemas (used by Schema Registry) ────────────────────────────────────

RAW_ARTICLE_AVRO_SCHEMA: dict[str, Any] = {
    "type": "record",
    "name": "RawArticle",
    "namespace": "com.pulsede.ingestion",
    "fields": [
        {"name": "source", "type": "string"},
        {"name": "headline", "type": "string"},
        {"name": "url", "type": "string"},
        {"name": "published_at", "type": "string"},
        {"name": "body", "type": "string", "default": ""},
        {"name": "author", "type": "string", "default": ""},
        {"name": "content_hash", "type": "string"},
    ],
}

SENTIMENT_RESULT_AVRO_SCHEMA: dict[str, Any] = {
    "type": "record",
    "name": "SentimentResult",
    "namespace": "com.pulsede.processing",
    "fields": [
        {"name": "article_hash", "type": "string"},
        {"name": "headline", "type": "string"},
        {"name": "url", "type": "string"},
        {"name": "source", "type": "string"},
        {"name": "published_at", "type": "string"},
        {"name": "processed_at", "type": "string"},
        {"name": "ensemble_sentiment", "type": "string"},
        {"name": "ensemble_confidence", "type": "float"},
        {"name": "ensemble_uncertainty", "type": "float"},
        {"name": "positive_prob", "type": "float"},
        {"name": "negative_prob", "type": "float"},
        {"name": "neutral_prob", "type": "float"},
        {"name": "tickers", "type": {"type": "array", "items": "string"}},
        {"name": "sectors", "type": {"type": "array", "items": "string"}},
        {"name": "is_forward_looking", "type": "boolean"},
        {"name": "has_negation", "type": "boolean"},
        {"name": "hedge_score", "type": "float"},
        {"name": "market_impact", "type": "string"},
        {"name": "is_uncertain", "type": "boolean"},
    ],
}

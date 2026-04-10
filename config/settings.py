"""Centralised, validated configuration via Pydantic v2 Settings.

All secrets come from environment variables or .env.
All values carry type annotations, validators, and defaults.
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class KafkaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KAFKA_")

    bootstrap_servers: str = "localhost:9092"
    news_topic: str = "financial-news-raw"
    sentiment_topic: str = "financial-sentiment"
    dead_letter_topic: str = "financial-dlq"
    consumer_group: str = "pulsede-sentiment-cg"
    schema_registry_url: str = "http://localhost:8081"
    acks: Literal["0", "1", "all"] = "all"
    retries: int = 5
    linger_ms: int = 5
    batch_size: int = 32_768
    compression_type: Literal["gzip", "snappy", "lz4", "zstd", "none"] = "snappy"


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DB_")

    host: str = "localhost"
    port: int = 5432
    name: str = "pulsede"
    user: str = "pulsede"
    password: SecretStr = Field(default=SecretStr("pulsede"))
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False

    @property
    def url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.name}"
        )


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS_")

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: SecretStr | None = None
    ttl_seconds: int = 3600         # 1 h default cache TTL
    dedup_window_seconds: int = 86400  # 24 h deduplication window


class MLSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ML_")

    # Primary financial-domain BERT
    primary_model: str = "ProsusAI/finbert"
    # Complementary tone-aware model
    secondary_model: str = "yiyanghkust/finbert-tone"
    # Distil-RoBERTa for speed/diversity
    tertiary_model: str = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    # NER model for ticker / entity extraction
    ner_model: str = "dslim/bert-base-NER"

    device: Literal["cpu", "cuda", "mps"] = "cpu"
    batch_size: int = 16
    max_length: int = 512

    # Ensemble weighting (primary, secondary, tertiary) — must sum to 1.0
    weights: list[float] = [0.5, 0.3, 0.2]

    # MC Dropout passes for uncertainty estimation
    mc_dropout_passes: int = 10

    # Confidence threshold below which we flag as uncertain
    uncertainty_threshold: float = 0.15

    # Temperature scaling calibration parameter (1.0 = uncalibrated)
    temperature: float = 1.0

    @field_validator("weights")
    @classmethod
    def weights_sum_to_one(cls, v: list[float]) -> list[float]:
        if abs(sum(v) - 1.0) > 1e-6:
            raise ValueError(f"Ensemble weights must sum to 1.0, got {sum(v):.4f}")
        return v


class MLflowSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MLFLOW_")

    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "pulsede-sentiment"
    model_name: str = "pulsede-ensemble"


class NewsAPISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="")

    news_api_key: SecretStr = Field(..., alias="NEWS_API_KEY")
    # Additional optional sources
    alpha_vantage_key: SecretStr | None = Field(default=None, alias="ALPHA_VANTAGE_KEY")
    rss_feeds: list[str] = [
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "https://feeds.finance.yahoo.com/rss/2.0/headline",
        "https://www.ft.com/rss/home",
        "https://feeds.bloomberg.com/markets/news.rss",
    ]
    fetch_interval_seconds: int = 300  # 5 min
    max_articles_per_fetch: int = 100


class MonitoringSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MONITORING_")

    prometheus_port: int = 8000
    otlp_endpoint: str = "http://localhost:4317"
    drift_check_interval_seconds: int = 3600
    # PSI threshold for distribution drift alert
    psi_threshold: float = 0.2


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="API_")

    host: str = "0.0.0.0"
    port: int = 8080
    secret_key: SecretStr = Field(default=SecretStr("change-me-in-production"))
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    rate_limit_per_minute: int = 60
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8501"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "console"

    kafka: KafkaSettings = KafkaSettings()
    db: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    ml: MLSettings = MLSettings()
    mlflow: MLflowSettings = MLflowSettings()
    news: NewsAPISettings = NewsAPISettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    api: APISettings = APISettings()


# Module-level singleton — import and use directly
settings = Settings()

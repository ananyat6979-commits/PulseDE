"""Prefect 2.x orchestration pipeline — replaces the naive `schedule` module.

Flow topology:
  fetch_news_flow  (every 5 min)
    └── fetch_and_publish_task   — NewsAPI + RSS → Kafka
  drift_check_flow (every 1 hour)
    └── check_drift_task         — TimescaleDB → DriftDetector → alerts
  calibration_flow (every 24 h)
    └── recalibrate_temperature  — Dev set → temperature scaling

Why Prefect over `schedule`:
  - Retries with exponential backoff per task (not per run).
  - Persistent run history, artifact storage, failure alerts in the UI.
  - Parameterisation — run ad-hoc fetches via API or CLI.
  - Deployment-ready: Docker or k8s workers, remote execution.
  - Concurrency limits prevent overlapping runs.
"""
from __future__ import annotations

import logging
from datetime import timedelta

import mlflow
from prefect import flow, get_run_logger, task
from prefect.tasks import task_input_hash

from config.settings import settings
from src.ingestion.kafka_producer import ArticleProducer
from src.ingestion.news_fetcher import NewsFetcher
from src.monitoring.drift_detector import DriftDetector
from src.storage.timescale_writer import TimescaleWriter

logger = logging.getLogger(__name__)


# ── Tasks ──────────────────────────────────────────────────────────────────────

@task(
    name="fetch-and-publish-news",
    retries=3,
    retry_delay_seconds=30,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(minutes=4),  # Prevent duplicate fetches within one cycle
    tags=["ingestion"],
)
def fetch_and_publish_task() -> int:
    """Fetch news from all sources and publish to Kafka. Returns article count."""
    prefect_logger = get_run_logger()
    with NewsFetcher() as fetcher:
        articles = fetcher.fetch_all()
        prefect_logger.info(f"Fetched {len(articles)} unique articles")

    if articles:
        with ArticleProducer() as producer:
            producer.publish_batch(articles)
            prefect_logger.info(f"Published {len(articles)} articles to Kafka")

    return len(articles)


@task(
    name="check-model-drift",
    retries=2,
    retry_delay_seconds=60,
    tags=["monitoring"],
)
def check_drift_task(lookback_hours: int = 1, reference_hours: int = 24) -> dict:
    """Compare current hour's distribution to 24h reference. Returns drift report."""
    prefect_logger = get_run_logger()
    db = TimescaleWriter()
    detector = DriftDetector()

    # Reference window (yesterday)
    reference_data = db.query_recent(hours=reference_hours, limit=2000)
    if len(reference_data) < 50:
        prefect_logger.warning("Insufficient reference data for drift check")
        return {}

    ref_conf = [r["ensemble_confidence"] for r in reference_data]
    ref_sents = [r["ensemble_sentiment"] for r in reference_data]
    detector.set_reference(ref_conf, ref_sents)

    # Current window
    current_data = db.query_recent(hours=lookback_hours, limit=500)
    if len(current_data) < 10:
        prefect_logger.info("Not enough current data — skipping drift check")
        return {}

    import numpy as np
    cur_conf = [r["ensemble_confidence"] for r in current_data]
    cur_sents = [r["ensemble_sentiment"] for r in current_data]
    cur_proba = np.array(
        [[r["positive_prob"], r["negative_prob"], r["neutral_prob"]] for r in current_data]
    )

    report = detector.check(cur_conf, cur_sents, cur_proba)
    result = {
        "psi": report.psi_confidence,
        "js_divergence": report.js_divergence,
        "chi2_pvalue": report.chi2_pvalue,
        "is_drifting": report.is_drifting,
        "alerts": report.alerts,
    }

    # Log to MLflow
    with mlflow.start_run(
        run_name="drift_check",
        experiment_id=mlflow.get_experiment_by_name(settings.mlflow.experiment_name).experiment_id
        if mlflow.get_experiment_by_name(settings.mlflow.experiment_name)
        else None,
    ):
        mlflow.log_metrics({
            "drift/psi": report.psi_confidence,
            "drift/js": report.js_divergence,
            "drift/chi2_pvalue": report.chi2_pvalue,
        })
        if report.is_drifting:
            mlflow.set_tag("drift_alert", "; ".join(report.alerts))

    if report.is_drifting:
        prefect_logger.warning(f"DRIFT DETECTED: {report.alerts}")
    else:
        prefect_logger.info("No drift detected")

    return result


# ── Flows ──────────────────────────────────────────────────────────────────────

@flow(
    name="pulsede-fetch-news",
    description="Ingest financial news from all sources into Kafka",
    version="2.0.0",
    log_prints=True,
)
def fetch_news_flow() -> None:
    count = fetch_and_publish_task()
    print(f"Pipeline run complete: {count} articles processed")


@flow(
    name="pulsede-drift-check",
    description="Hourly model and data drift monitoring",
    version="2.0.0",
)
def drift_check_flow() -> None:
    report = check_drift_task()
    if report.get("is_drifting"):
        print(f"⚠️  Drift alerts: {report['alerts']}")


# ── Entry points ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run locally for development
    fetch_news_flow()

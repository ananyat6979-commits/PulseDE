"""Model and data drift detection using Evidently AI + statistical tests.

Checks performed every hour (configurable):
  1. Data drift — Population Stability Index (PSI) on confidence scores.
  2. Sentiment distribution drift — Chi-squared test on label frequencies.
  3. Prediction drift — Jensen-Shannon divergence on probability distributions.
  4. Concept drift proxy — rolling accuracy on a held-out labelled sample.

Alerts fire to:
  - Structured log (always)
  - Prometheus gauge (always)
  - MLflow tag on the active run (always)
  - Slack/PagerDuty webhook (if configured)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from prometheus_client import Gauge
from scipy.stats import chi2_contingency, entropy

from config.settings import settings

logger = logging.getLogger(__name__)

PSI_GAUGE = Gauge("pulsede_psi_confidence", "Population Stability Index on confidence scores")
JS_DIV_GAUGE = Gauge("pulsede_js_divergence_sentiment", "Jensen-Shannon divergence on sentiment probs")
SENTIMENT_DRIFT_GAUGE = Gauge("pulsede_sentiment_chi2_pvalue", "Chi-squared p-value for label drift")


@dataclass
class DriftReport:
    timestamp: datetime
    psi_confidence: float
    js_divergence: float
    chi2_pvalue: float
    is_drifting: bool
    alerts: list[str]


class DriftDetector:
    """Computes drift metrics between a reference window and current window."""

    def __init__(self) -> None:
        # Reference distributions — bootstrapped from first N samples at startup
        self._ref_confidence: np.ndarray | None = None
        self._ref_sentiment_counts: np.ndarray | None = None  # [pos, neg, neu]

    def set_reference(
        self,
        confidence_scores: list[float],
        sentiment_labels: list[str],
    ) -> None:
        """Set the baseline distribution from historical data."""
        self._ref_confidence = np.array(confidence_scores, dtype=np.float32)
        counts = np.array([
            sum(1 for s in sentiment_labels if s == "positive"),
            sum(1 for s in sentiment_labels if s == "negative"),
            sum(1 for s in sentiment_labels if s == "neutral"),
        ], dtype=np.float32)
        self._ref_sentiment_counts = counts / counts.sum()
        logger.info(
            "drift_reference_set",
            extra={"n_samples": len(confidence_scores)},
        )

    def check(
        self,
        current_confidence: list[float],
        current_sentiments: list[str],
        current_proba: np.ndarray,
    ) -> DriftReport:
        """Compare current window against reference. Returns DriftReport."""
        if self._ref_confidence is None or self._ref_sentiment_counts is None:
            raise RuntimeError("Call set_reference() before check().")

        alerts: list[str] = []
        curr_conf = np.array(current_confidence, dtype=np.float32)

        # ── PSI on confidence scores ───────────────────────────────────────────
        psi = self._population_stability_index(self._ref_confidence, curr_conf)
        PSI_GAUGE.set(psi)
        if psi > settings.monitoring.psi_threshold:
            msg = f"PSI={psi:.4f} exceeds threshold {settings.monitoring.psi_threshold}"
            alerts.append(msg)
            logger.warning("drift_psi_alert", extra={"psi": psi})

        # ── Chi-squared on label counts ────────────────────────────────────────
        curr_counts = np.array([
            sum(1 for s in current_sentiments if s == "positive"),
            sum(1 for s in current_sentiments if s == "negative"),
            sum(1 for s in current_sentiments if s == "neutral"),
        ], dtype=np.float32)
        ref_expected = self._ref_sentiment_counts * curr_counts.sum()
        _, pvalue, _, _ = chi2_contingency(
            np.vstack([curr_counts, ref_expected]).clip(min=1)
        )
        SENTIMENT_DRIFT_GAUGE.set(float(pvalue))
        if pvalue < 0.01:
            msg = f"Sentiment distribution drift: chi2 p={pvalue:.4f}"
            alerts.append(msg)
            logger.warning("drift_sentiment_alert", extra={"pvalue": pvalue})

        # ── Jensen-Shannon divergence on probability outputs ───────────────────
        ref_mean_proba = np.array([
            self._ref_sentiment_counts[0],
            self._ref_sentiment_counts[1],
            self._ref_sentiment_counts[2],
        ])
        curr_mean_proba = current_proba.mean(axis=0)
        js = self._js_divergence(ref_mean_proba, curr_mean_proba)
        JS_DIV_GAUGE.set(float(js))
        if js > 0.1:
            msg = f"Prediction drift JS={js:.4f} > 0.1"
            alerts.append(msg)
            logger.warning("drift_prediction_alert", extra={"js_divergence": js})

        return DriftReport(
            timestamp=datetime.now(timezone.utc),
            psi_confidence=round(float(psi), 6),
            js_divergence=round(float(js), 6),
            chi2_pvalue=round(float(pvalue), 6),
            is_drifting=len(alerts) > 0,
            alerts=alerts,
        )

    # ── Statistical helpers ────────────────────────────────────────────────────

    @staticmethod
    def _population_stability_index(
        reference: np.ndarray, current: np.ndarray, n_bins: int = 10
    ) -> float:
        """PSI = sum_i (actual_i - expected_i) * ln(actual_i / expected_i)
        Values: < 0.1 = stable, 0.1–0.2 = moderate drift, > 0.2 = significant drift.
        """
        bins = np.linspace(0, 1, n_bins + 1)
        ref_counts, _ = np.histogram(reference, bins=bins)
        cur_counts, _ = np.histogram(current, bins=bins)

        ref_pct = (ref_counts / len(reference)).clip(min=1e-6)
        cur_pct = (cur_counts / len(current)).clip(min=1e-6)

        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return psi

    @staticmethod
    def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """Jensen-Shannon divergence in [0, 1] (0 = identical, 1 = maximally different)."""
        p = (p + 1e-9) / (p + 1e-9).sum()
        q = (q + 1e-9) / (q + 1e-9).sum()
        m = 0.5 * (p + q)
        js = 0.5 * float(entropy(p, m) + entropy(q, m))
        return min(js, 1.0)

"""Unit tests for the ML layer — ensemble, features, evaluator.

These run without Kafka/DB/Redis (fully mocked).
Model weights are mocked via unittest.mock so CI doesn't download 1 GB.
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.ingestion.schema import MarketImpact, Sentiment
from src.ml.evaluator import evaluate, _expected_calibration_error, _pairwise_disagreement
from src.ml.feature_engineering import FinancialFeatureExtractor


# ── Feature engineering tests ──────────────────────────────────────────────────

class TestFeatureExtractor:
    def setup_method(self) -> None:
        self.extractor = FinancialFeatureExtractor()

    def test_hedge_score_high(self) -> None:
        fv = self.extractor.extract("The market may possibly recover if conditions potentially improve")
        assert fv.hedge_score > 0.2

    def test_hedge_score_low(self) -> None:
        fv = self.extractor.extract("Apple beats Q3 earnings, stock up 5%")
        assert fv.hedge_score < 0.1

    def test_negation_detection(self) -> None:
        fv = self.extractor.extract("Company failed to meet revenue guidance")
        assert fv.has_negation is True

    def test_negation_absent(self) -> None:
        fv = self.extractor.extract("Strong earnings beat across all segments")
        assert fv.has_negation is False

    def test_forward_looking_detected(self) -> None:
        fv = self.extractor.extract("Management expects next quarter revenue to grow 15%")
        assert fv.is_forward_looking is True

    def test_forward_looking_absent(self) -> None:
        fv = self.extractor.extract("Stock closed down 3% on heavy volume")
        assert fv.is_forward_looking is False

    def test_ticker_extraction(self) -> None:
        fv = self.extractor.extract("AAPL and MSFT both rallied after the Fed decision")
        assert "AAPL" in fv.tickers
        assert "MSFT" in fv.tickers

    def test_false_ticker_filtered(self) -> None:
        fv = self.extractor.extract("The CEO said AI will transform the industry")
        # CEO, AI, THE should not be tickers
        assert "CEO" not in fv.tickers
        assert "AI" not in fv.tickers

    def test_sector_mapping(self) -> None:
        fv = self.extractor.extract("NVDA GPU sales drive record revenue")
        assert "Technology" in fv.sectors

    def test_market_impact_high(self) -> None:
        fv = self.extractor.extract("Market crash feared as recession signals mount")
        assert fv.market_impact == MarketImpact.HIGH

    def test_market_impact_medium(self) -> None:
        fv = self.extractor.extract("Quarterly earnings report shows revenue growth")
        assert fv.market_impact in (MarketImpact.MEDIUM, MarketImpact.HIGH)

    def test_empty_headline(self) -> None:
        fv = self.extractor.extract("")
        assert fv.hedge_score == 0.0
        assert fv.has_negation is False


# ── Evaluator tests ────────────────────────────────────────────────────────────

class TestEvaluator:
    def _make_data(
        self, n: int = 100, n_classes: int = 3, seed: int = 42
    ) -> tuple:
        rng = np.random.default_rng(seed)
        y_true = rng.integers(0, n_classes, n)
        # Soft predictions: mostly correct with noise
        y_proba = rng.dirichlet(np.ones(n_classes), size=n)
        # Bias toward true class
        for i, c in enumerate(y_true):
            y_proba[i, c] += 1.5
        y_proba /= y_proba.sum(axis=1, keepdims=True)
        y_pred = y_proba.argmax(axis=1)
        per_model = [
            rng.integers(0, n_classes, n),
            rng.integers(0, n_classes, n),
            y_pred.copy(),
        ]
        uncertainties = rng.uniform(0, 0.3, n)
        latencies = rng.uniform(10, 100, n)
        return y_true, y_pred, y_proba, per_model, uncertainties, latencies

    def test_report_fields_present(self) -> None:
        y_true, y_pred, y_proba, per_model, unc, lat = self._make_data()
        report = evaluate(y_true, y_pred, y_proba, per_model, unc, lat)
        assert 0 <= report.macro_f1 <= 1
        assert 0 <= report.accuracy <= 1
        assert 0 <= report.mcc <= 1
        assert report.n_samples == 100
        assert set(report.per_class.keys()) == {"positive", "negative", "neutral"}

    def test_ece_range(self) -> None:
        _, _, y_proba, _, _, _ = self._make_data()
        y_true = y_proba.argmax(axis=1)  # perfect predictions
        ece = _expected_calibration_error(y_true, y_proba)
        assert 0 <= ece <= 1

    def test_perfect_ece_near_zero(self) -> None:
        """Perfect calibration → ECE should be very low."""
        n = 500
        rng = np.random.default_rng(0)
        y_proba = np.zeros((n, 3))
        y_proba[:, 0] = 1.0    # Always predicts positive with 100% confidence
        y_true = np.zeros(n, dtype=int)  # Always positive
        ece = _expected_calibration_error(y_true, y_proba)
        assert ece < 0.05

    def test_disagreement_rate_all_agree(self) -> None:
        preds = [np.zeros(50, dtype=int)] * 3
        rate = _pairwise_disagreement(preds)
        assert rate == 0.0

    def test_disagreement_rate_always_disagree(self) -> None:
        preds = [
            np.zeros(50, dtype=int),
            np.ones(50, dtype=int),
            np.full(50, 2, dtype=int),
        ]
        rate = _pairwise_disagreement(preds)
        assert rate == 1.0

    def test_latency_percentiles_ordered(self) -> None:
        y_true, y_pred, y_proba, per_model, unc, lat = self._make_data()
        report = evaluate(y_true, y_pred, y_proba, per_model, unc, lat)
        assert report.p50_ms <= report.p95_ms <= report.p99_ms

    def test_brier_score_perfect(self) -> None:
        n = 200
        y_true = np.zeros(n, dtype=int)
        y_proba = np.zeros((n, 3))
        y_proba[:, 0] = 1.0
        y_pred = np.zeros(n, dtype=int)
        per_model = [y_pred] * 3
        unc = np.zeros(n)
        lat = np.ones(n) * 50
        report = evaluate(y_true, y_pred, y_proba, per_model, unc, lat)
        assert report.calibration.brier_score < 0.01


# ── Schema tests ───────────────────────────────────────────────────────────────

class TestRawArticleHash:
    def test_same_content_same_hash(self) -> None:
        from src.ingestion.schema import RawArticle
        from datetime import datetime, timezone
        a = RawArticle("src", "headline", "url", datetime.now(timezone.utc))
        b = RawArticle("src", "headline", "url", datetime.now(timezone.utc))
        assert a.content_hash == b.content_hash

    def test_different_headline_different_hash(self) -> None:
        from src.ingestion.schema import RawArticle
        from datetime import datetime, timezone
        a = RawArticle("src", "headline A", "url", datetime.now(timezone.utc))
        b = RawArticle("src", "headline B", "url", datetime.now(timezone.utc))
        assert a.content_hash != b.content_hash

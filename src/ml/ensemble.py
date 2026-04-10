"""Multi-model FinBERT ensemble with uncertainty quantification.

Architecture:
  - Three independently fine-tuned financial NLP models
  - Weighted probability averaging (calibrated via held-out dev set)
  - Monte Carlo Dropout for epistemic uncertainty estimation
  - Temperature scaling for calibration (learns T on a calibration set)
  - Predictive entropy as the uncertainty signal

Research references:
  - Araci (2019) FinBERT: https://arxiv.org/abs/1908.10063
  - Gal & Ghahramani (2016) MC Dropout: https://arxiv.org/abs/1506.02142
  - Guo et al. (2017) Temperature Scaling: https://arxiv.org/abs/1706.04599
  - Lakshminarayanan et al. (2017) Deep Ensembles: https://arxiv.org/abs/1612.01474
"""
from __future__ import annotations

import contextlib
import logging
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import entr
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Pipeline, pipeline

from config.settings import settings
from src.ingestion.schema import ModelPrediction, Sentiment

logger = logging.getLogger(__name__)

# Label mapping from HuggingFace model outputs to our canonical Sentiment enum.
# Each model may use different label strings — normalise here.
_LABEL_MAPS: dict[str, dict[str, Sentiment]] = {
    settings.ml.primary_model: {
        "positive": Sentiment.POSITIVE,
        "negative": Sentiment.NEGATIVE,
        "neutral": Sentiment.NEUTRAL,
    },
    settings.ml.secondary_model: {
        "Positive": Sentiment.POSITIVE,
        "Negative": Sentiment.NEGATIVE,
        "Neutral": Sentiment.NEUTRAL,
    },
    settings.ml.tertiary_model: {
        "positive": Sentiment.POSITIVE,
        "negative": Sentiment.NEGATIVE,
        "neutral": Sentiment.NEUTRAL,
    },
}

_SENTIMENT_IDX = {Sentiment.POSITIVE: 0, Sentiment.NEGATIVE: 1, Sentiment.NEUTRAL: 2}


class _MCDropoutModel:
    """Wraps a HuggingFace model for inference with dropout kept active."""

    def __init__(self, model_name: str) -> None:
        self.name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(settings.ml.device)
        self._label_map = _LABEL_MAPS[model_name]
        self._id2label: dict[int, str] = self.model.config.id2label

    @torch.no_grad()
    def predict(
        self, texts: list[str], enable_dropout: bool = False
    ) -> list[dict[str, float]]:
        """Return list of {sentiment_name: probability} dicts."""
        if enable_dropout:
            self.model.train()  # activates dropout
        else:
            self.model.eval()

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=settings.ml.max_length,
            return_tensors="pt",
        ).to(settings.ml.device)

        logits = self.model(**inputs).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()

        results = []
        for prob_row in probs:
            row: dict[str, float] = {}
            for idx, prob in enumerate(prob_row):
                raw_label = self._id2label[idx]
                canonical = self._label_map.get(raw_label, Sentiment.NEUTRAL)
                row[canonical.value] = float(prob)
            results.append(row)

        return results


class SentimentEnsemble:
    """Production-grade ensemble sentiment classifier.

    Instantiation loads three models into memory; subsequent calls are
    fully vectorised per batch. Use as a singleton.
    """

    def __init__(self) -> None:
        logger.info("Loading ensemble models…")
        self._models = [
            _MCDropoutModel(settings.ml.primary_model),
            _MCDropoutModel(settings.ml.secondary_model),
            _MCDropoutModel(settings.ml.tertiary_model),
        ]
        self._weights = np.array(settings.ml.weights, dtype=np.float32)
        self._temperature = settings.ml.temperature
        logger.info("Ensemble ready", extra={"models": len(self._models)})

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        """Full pipeline: ensemble + MC Dropout + temperature scaling.

        Returns a list of dicts with keys:
          sentiment, confidence, uncertainty, positive_prob, negative_prob,
          neutral_prob, model_predictions.
        """
        if not texts:
            return []

        # --- Deterministic predictions for each model ---
        per_model_probs: list[np.ndarray] = []
        model_predictions_per_text: list[list[ModelPrediction]] = [[] for _ in texts]

        for i, model in enumerate(self._models):
            t0 = time.perf_counter()
            preds = model.predict(texts, enable_dropout=False)
            latency_ms = (time.perf_counter() - t0) * 1000 / len(texts)

            probs_matrix = np.array(
                [
                    [p.get("positive", 0.0), p.get("negative", 0.0), p.get("neutral", 0.0)]
                    for p in preds
                ],
                dtype=np.float32,
            )
            per_model_probs.append(probs_matrix)

            for j, probs in enumerate(probs_matrix):
                argmax = int(probs.argmax())
                sentiment = [Sentiment.POSITIVE, Sentiment.NEGATIVE, Sentiment.NEUTRAL][argmax]
                model_predictions_per_text[j].append(
                    ModelPrediction(
                        model_name=model.name,
                        sentiment=sentiment,
                        positive_prob=float(probs[0]),
                        negative_prob=float(probs[1]),
                        neutral_prob=float(probs[2]),
                        latency_ms=latency_ms,
                    )
                )

        # --- Weighted ensemble average ---
        # Shape: (n_models, n_texts, 3)
        stacked = np.stack(per_model_probs, axis=0)
        # (n_texts, 3)
        ensemble_probs = (stacked * self._weights[:, None, None]).sum(axis=0)

        # --- Temperature scaling (calibration) ---
        ensemble_logits = np.log(ensemble_probs + 1e-9) / self._temperature
        calibrated_probs = self._softmax(ensemble_logits)

        # --- MC Dropout uncertainty ---
        mc_uncertainty = self._mc_dropout_uncertainty(texts)

        # --- Assemble results ---
        results = []
        for i in range(len(texts)):
            probs = calibrated_probs[i]
            argmax = int(probs.argmax())
            sentiment = [Sentiment.POSITIVE, Sentiment.NEGATIVE, Sentiment.NEUTRAL][argmax]
            confidence = float(probs[argmax])
            uncertainty = float(mc_uncertainty[i])
            is_uncertain = uncertainty > settings.ml.uncertainty_threshold

            results.append(
                {
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "uncertainty": uncertainty,
                    "is_uncertain": is_uncertain,
                    "positive_prob": float(probs[0]),
                    "negative_prob": float(probs[1]),
                    "neutral_prob": float(probs[2]),
                    "model_predictions": model_predictions_per_text[i],
                }
            )

        return results

    # ── Calibration ────────────────────────────────────────────────────────────

    def calibrate_temperature(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        n_epochs: int = 50,
        lr: float = 0.01,
    ) -> float:
        """Learn scalar temperature T via NLL minimisation on a calibration set.

        Args:
            logits: (N, 3) array of raw model logits.
            labels: (N,) array of integer class labels {0, 1, 2}.
            n_epochs: Gradient steps.
            lr: Learning rate.

        Returns:
            Learned temperature T.
        """
        T = torch.nn.Parameter(torch.ones(1))
        optimiser = torch.optim.LBFGS([T], lr=lr, max_iter=n_epochs)
        logits_t = torch.from_numpy(logits).float()
        labels_t = torch.from_numpy(labels).long()

        def _closure() -> torch.Tensor:
            optimiser.zero_grad()
            scaled = logits_t / T
            loss = F.cross_entropy(scaled, labels_t)
            loss.backward()
            return loss

        optimiser.step(_closure)
        self._temperature = float(T.item())
        logger.info("calibration_done", extra={"temperature": round(self._temperature, 4)})
        return self._temperature

    # ── MC Dropout uncertainty ─────────────────────────────────────────────────

    def _mc_dropout_uncertainty(self, texts: list[str]) -> np.ndarray:
        """Run T stochastic forward passes; return predictive entropy per text.

        Predictive entropy H[p] = -sum_c p_c * log(p_c)
        High entropy → model is uncertain across classes.
        """
        T = settings.ml.mc_dropout_passes
        mc_samples: list[np.ndarray] = []

        primary = self._models[0]  # Only primary for cost reasons
        for _ in range(T):
            preds = primary.predict(texts, enable_dropout=True)
            probs = np.array(
                [[p.get("positive", 0.0), p.get("negative", 0.0), p.get("neutral", 0.0)]
                 for p in preds],
                dtype=np.float32,
            )
            mc_samples.append(probs)

        # Mean predictive distribution: (n_texts, 3)
        mean_probs = np.stack(mc_samples, axis=0).mean(axis=0)
        # Predictive entropy (nats): (n_texts,)
        entropy = entr(mean_probs).sum(axis=1)
        # Normalise to [0, 1] by max entropy = ln(3)
        return entropy / np.log(3)

    # ── Utilities ──────────────────────────────────────────────────────────────

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

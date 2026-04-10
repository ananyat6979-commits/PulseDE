"""Research-grade ML evaluation for financial sentiment classification.

Metrics implemented:
  - Per-class Precision, Recall, F1 (macro, micro, weighted)
  - PR-AUC (area under precision-recall curve) per class vs. rest
  - ROC-AUC (macro OVR)
  - Matthews Correlation Coefficient (handles class imbalance)
  - Expected Calibration Error (ECE) — 15-bin reliability diagram data
  - Brier Score (proper scoring rule for probabilistic outputs)
  - Confusion matrix (normalised and raw)
  - Cohen's Kappa
  - Model-level latency percentiles (p50, p95, p99)
  - Ensemble diversity (pairwise disagreement rate)

All results are MLflow-ready: flat dict of scalars + artifact paths.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

CLASS_NAMES = ["positive", "negative", "neutral"]


@dataclass
class PerClassMetrics:
    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float
    support: int


@dataclass
class CalibrationMetrics:
    ece: float                      # Expected Calibration Error (lower = better)
    brier_score: float              # Proper scoring rule (lower = better)
    reliability_diagram: list[dict[str, float]]  # fraction_positive, mean_predicted_prob


@dataclass
class EvaluationReport:
    # Aggregate
    macro_f1: float
    micro_f1: float
    weighted_f1: float
    macro_roc_auc: float
    mcc: float
    cohen_kappa: float
    accuracy: float

    # Per-class
    per_class: dict[str, PerClassMetrics]

    # Calibration
    calibration: CalibrationMetrics

    # Confusion matrix (raw counts)
    confusion_matrix: list[list[int]]
    confusion_matrix_normalised: list[list[float]]

    # Ensemble-level
    ensemble_disagreement_rate: float   # fraction of samples where models disagree
    uncertain_fraction: float           # fraction flagged as uncertain

    # Latency
    p50_ms: float
    p95_ms: float
    p99_ms: float

    # Metadata
    n_samples: int
    class_distribution: dict[str, float]
    model_names: list[str] = field(default_factory=list)


def evaluate(
    y_true: np.ndarray,                 # (N,) int labels 0=pos,1=neg,2=neu
    y_pred: np.ndarray,                 # (N,) int labels
    y_proba: np.ndarray,                # (N, 3) probabilities
    per_model_preds: list[np.ndarray],  # list of (N,) per-model int labels
    uncertainties: np.ndarray,          # (N,) float in [0,1]
    latencies_ms: np.ndarray,           # (N,) float
    model_names: list[str] | None = None,
) -> EvaluationReport:
    """Compute full evaluation report.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Ensemble hard predictions.
        y_proba: Ensemble soft probabilities (N, 3).
        per_model_preds: One (N,) array per sub-model for diversity metrics.
        uncertainties: MC Dropout entropy per sample.
        latencies_ms: Per-sample inference time.
        model_names: Optional labels for sub-models.

    Returns:
        EvaluationReport dataclass with all metrics.
    """
    n = len(y_true)
    assert n > 0, "Empty evaluation set"

    # ── Aggregate classification metrics ──────────────────────────────────────
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    accuracy = float((y_true == y_pred).mean())

    macro_roc_auc = float(
        roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    )

    # ── Per-class metrics ─────────────────────────────────────────────────────
    per_class: dict[str, PerClassMetrics] = {}
    for i, cls_name in enumerate(CLASS_NAMES):
        binary_true = (y_true == i).astype(int)
        binary_pred = (y_pred == i).astype(int)

        p = float(precision_score(binary_true, binary_pred, zero_division=0))
        r = float(recall_score(binary_true, binary_pred, zero_division=0))
        f = float(f1_score(binary_true, binary_pred, zero_division=0))

        prec_curve, rec_curve, _ = precision_recall_curve(binary_true, y_proba[:, i])
        pr_auc = float(average_precision_score(binary_true, y_proba[:, i]))

        roc = float(roc_auc_score(binary_true, y_proba[:, i]))

        support = int(binary_true.sum())
        per_class[cls_name] = PerClassMetrics(
            precision=p, recall=r, f1=f, pr_auc=pr_auc, roc_auc=roc, support=support
        )

    # ── Calibration ───────────────────────────────────────────────────────────
    ece = _expected_calibration_error(y_true, y_proba, n_bins=15)
    # Brier score: average over one-vs-rest
    brier = float(
        np.mean(
            [
                brier_score_loss((y_true == i).astype(int), y_proba[:, i])
                for i in range(3)
            ]
        )
    )
    # Reliability diagram (positive class, index 0)
    frac_pos, mean_pred = calibration_curve((y_true == 0).astype(int), y_proba[:, 0], n_bins=10)
    reliability = [
        {"mean_predicted_prob": float(mp), "fraction_positive": float(fp)}
        for mp, fp in zip(mean_pred, frac_pos)
    ]

    calib = CalibrationMetrics(
        ece=round(ece, 6), brier_score=round(brier, 6), reliability_diagram=reliability
    )

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    # ── Ensemble diversity ────────────────────────────────────────────────────
    disagreement_rate = _pairwise_disagreement(per_model_preds)
    uncertain_frac = float((uncertainties > 0.15).mean())

    # ── Latency ───────────────────────────────────────────────────────────────
    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    p99 = float(np.percentile(latencies_ms, 99))

    # ── Class distribution ────────────────────────────────────────────────────
    class_dist = {
        cls_name: float((y_true == i).mean())
        for i, cls_name in enumerate(CLASS_NAMES)
    }

    return EvaluationReport(
        macro_f1=round(macro_f1, 6),
        micro_f1=round(micro_f1, 6),
        weighted_f1=round(weighted_f1, 6),
        macro_roc_auc=round(macro_roc_auc, 6),
        mcc=round(mcc, 6),
        cohen_kappa=round(kappa, 6),
        accuracy=round(accuracy, 6),
        per_class=per_class,
        calibration=calib,
        confusion_matrix=cm.tolist(),
        confusion_matrix_normalised=cm_norm.round(4).tolist(),
        ensemble_disagreement_rate=round(disagreement_rate, 4),
        uncertain_fraction=round(uncertain_frac, 4),
        p50_ms=round(p50, 2),
        p95_ms=round(p95, 2),
        p99_ms=round(p99, 2),
        n_samples=n,
        class_distribution=class_dist,
        model_names=model_names or [],
    )


def log_to_mlflow(report: EvaluationReport, run_id: str | None = None) -> None:
    """Log all metrics and artifacts to the active (or specified) MLflow run."""
    with mlflow.start_run(run_id=run_id, nested=True):
        flat = _flatten_report(report)
        mlflow.log_metrics(flat)

        # Log confusion matrix as JSON artifact
        cm_path = Path("/tmp/confusion_matrix.json")
        cm_path.write_text(
            json.dumps(
                {
                    "raw": report.confusion_matrix,
                    "normalised": report.confusion_matrix_normalised,
                    "class_names": CLASS_NAMES,
                },
                indent=2,
            )
        )
        mlflow.log_artifact(str(cm_path), artifact_path="evaluation")

        # Log reliability diagram data
        calib_path = Path("/tmp/calibration.json")
        calib_path.write_text(json.dumps(report.calibration.reliability_diagram, indent=2))
        mlflow.log_artifact(str(calib_path), artifact_path="evaluation")

    logger.info("mlflow_eval_logged", extra=flat)


def _expected_calibration_error(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 15
) -> float:
    """ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Bins by confidence (max probability across classes).
    """
    confidences = y_proba.max(axis=1)
    correct = (y_proba.argmax(axis=1) == y_true).astype(float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(acc - conf)

    return float(ece)


def _pairwise_disagreement(per_model_preds: list[np.ndarray]) -> float:
    """Average fraction of samples where at least two models disagree."""
    if len(per_model_preds) < 2:
        return 0.0
    stacked = np.stack(per_model_preds, axis=1)  # (N, M)
    # A sample disagrees if not all models predict the same class
    disagree = (stacked != stacked[:, 0:1]).any(axis=1)
    return float(disagree.mean())


def _flatten_report(report: EvaluationReport) -> dict[str, float]:
    flat: dict[str, float] = {
        "eval/macro_f1": report.macro_f1,
        "eval/micro_f1": report.micro_f1,
        "eval/weighted_f1": report.weighted_f1,
        "eval/macro_roc_auc": report.macro_roc_auc,
        "eval/mcc": report.mcc,
        "eval/cohen_kappa": report.cohen_kappa,
        "eval/accuracy": report.accuracy,
        "eval/ece": report.calibration.ece,
        "eval/brier_score": report.calibration.brier_score,
        "eval/ensemble_disagreement": report.ensemble_disagreement_rate,
        "eval/uncertain_fraction": report.uncertain_fraction,
        "eval/p50_ms": report.p50_ms,
        "eval/p95_ms": report.p95_ms,
        "eval/p99_ms": report.p99_ms,
        "eval/n_samples": float(report.n_samples),
    }
    for cls_name, m in report.per_class.items():
        flat[f"eval/{cls_name}/precision"] = m.precision
        flat[f"eval/{cls_name}/recall"] = m.recall
        flat[f"eval/{cls_name}/f1"] = m.f1
        flat[f"eval/{cls_name}/pr_auc"] = m.pr_auc
        flat[f"eval/{cls_name}/roc_auc"] = m.roc_auc
        flat[f"eval/{cls_name}/support"] = float(m.support)
    return flat

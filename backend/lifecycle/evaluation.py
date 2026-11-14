"""Advanced diagnostics for a trained run, computed on demand.

The base metrics already live in ``ExperimentRun``; this module adds the
*diagnostic* layer — curves, calibration, threshold sweeps, and the
train-versus-test diagnosis — by reloading the fitted pipeline. It needs no
pandas column knowledge: the pipeline re-encodes its own features, and the
stored test split (row labels + predictions + probabilities) is the ground
truth for every curve.

Outputs are JSON-safe: arrays are down-sampled to a fixed number of points
so a plot never ships 10,000 coordinates.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

#: How many points each curve reports. Enough to look smooth, small to ship.
CURVE_POINTS = 60
#: Thresholds swept by the threshold analysis, in this order.
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class DiagnosticsError(Exception):
    """A user-readable reason the diagnostics could not be computed."""


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _downsample(values: np.ndarray, count: int = CURVE_POINTS) -> list[float]:
    """Evenly sample ``count`` points from an array, keeping the ends."""
    values = np.asarray(values, dtype=float)
    if values.size <= count:
        result = [v for v in values.tolist() if math.isfinite(v)]
        return result
    indices = np.linspace(0, values.size - 1, num=count).astype(int)
    return [v for v in values[indices].tolist() if math.isfinite(v)]


def _positive_class(y_true: np.ndarray, classes: list[str]) -> int:
    """The code the threshold/ROC analyses treat as 'the class of interest'."""
    if len(classes) == 2:
        return 1
    return int(max(y_true)) if len(y_true) else 0


# --------------------------------------------------------------------------
# classification diagnostics
# --------------------------------------------------------------------------


def classification_diagnostics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
) -> dict[str, Any]:
    """ROC, precision-recall, calibration and threshold tables for a classifier."""
    result: dict[str, Any] = {
        "roc": None,
        "precision_recall": None,
        "calibration": None,
        "threshold_analysis": None,
    }
    positive = _positive_class(y_true, classes)

    roc_curve_ = None
    if len(classes) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, positive])
        try:
            auc = _finite(roc_auc_score(y_true, y_proba[:, positive]))
        except (ValueError, IndexError):
            auc = None
        result["roc"] = {
            "fpr": _downsample(fpr),
            "tpr": _downsample(tpr),
            "auc": auc,
        }
        roc_curve_ = (fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_proba[:, positive])
    result["precision_recall"] = {
        "recall": _downsample(recall),
        "precision": _downsample(precision),
    }

    try:
        proba_true, proba_pred = calibration_curve(
            y_true, y_proba[:, positive], strategy="quantile", n_bins=10
        )
        if len(proba_true):
            result["calibration"] = {
                "prob_true_mean": _downsample(proba_true),
                "prob_pred_mean": _downsample(proba_pred),
                "brier": _finite(float(np.mean((y_proba[:, positive] - y_true) ** 2))),
            }
    except (ValueError, IndexError, TypeError):
        pass

    # Threshold analysis: how accuracy/precision/recall/f1 trade off as the
    # decision threshold moves away from 0.5.
    binary = y_true.astype(int)
    rows: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        predicted = (y_proba[:, positive] >= threshold).astype(int)
        tp = int(np.sum((predicted == 1) & (binary == 1)))
        fp = int(np.sum((predicted == 1) & (binary == 0)))
        fn = int(np.sum((predicted == 0) & (binary == 1)))
        tn = int(np.sum((predicted == 0) & (binary == 0)))
        total = tp + fp + fn + tn
        precision_v = tp / (tp + fp) if (tp + fp) else None
        recall_v = tp / (tp + fn) if (tp + fn) else None
        denom = (2 * tp + fp + fn) or 0
        f1_v = (2 * tp / denom) if denom else None
        accuracy_v = (tp + tn) / total if total else None
        rows.append(
            {
                "threshold": threshold,
                "accuracy": _finite(accuracy_v),
                "precision": _finite(precision_v),
                "recall": _finite(recall_v),
                "f1": _finite(f1_v),
                "predicted_positive_rate": _finite((tp + fp) / total if total else None),
            }
        )
    result["threshold_analysis"] = rows
    return result


# --------------------------------------------------------------------------
# regression diagnostics
# --------------------------------------------------------------------------


def regression_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    residual_sample: list[dict] | None = None,
) -> dict[str, Any]:
    """Error tables and worst-prediction lists for a regressor."""
    true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)
    residual = pred_arr - true_arr
    absolute = np.abs(residual)

    worst = [
        {
            "actual": _finite(true_arr[i]),
            "predicted": _finite(pred_arr[i]),
            "error": _finite(residual[i]),
        }
        for i in np.argsort(-absolute)[:20]
    ]

    # Percentage error, guarded against a zero actual (NaN where undefined).
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(true_arr != 0, np.abs(residual / true_arr), np.nan)
    pct_finite = pct[np.isfinite(pct)]

    return {
        "residuals": {
            "mean": _finite(float(residual.mean())) if residual.size else None,
            "std": _finite(float(residual.std())) if residual.size else None,
            "bias": _finite(float(residual.sum())) if residual.size else None,
        },
        "absolute_error": {
            "mean": _finite(float(absolute.mean())) if absolute.size else None,
            "median": _finite(float(np.median(absolute))) if absolute.size else None,
            "max": _finite(float(absolute.max())) if absolute.size else None,
        },
        "percentage_error": _finite(float(np.nanmean(pct))) if pct_finite.size else None,
        "worst_predictions": worst,
        "residual_distribution": _downsample(absolute),
    }


# --------------------------------------------------------------------------
# the train/test diagnosis
# --------------------------------------------------------------------------

#: A train metric far above the test metric is the overfitting signature.
_OVERFIT_GAP = 0.2


def diagnose_fit(
    task_type: str,
    metrics: dict[str, Any],
    train_metrics: dict[str, Any],
    leakage_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify the run as good-fit / overfit / underfit / leaky, with reasons.

    The verdict uses only numbers that always exist (the primary metric and its
    train-side twin), so a regression run and a classification run can share
    the one code path; extra keys are added where the task allows.
    """
    diagnosis: dict[str, Any] = {
        "verdict": "good_fit",
        "confidence": "low",
        "reasons": [],
        "train_metric": None,
        "test_metric": None,
        "gap": None,
    }

    if task_type == "regression":
        train_key, test_key = "r2", "r2"
        scale = 1.0  # R2 is not bounded the way accuracy is
    else:
        train_key, test_key = "f1", "accuracy"
        # Prefer the metric the experiment optimises for the gap, fall back.
        train_key = next((k for k in ("f1", "accuracy") if train_metrics.get(k) is not None), "accuracy")
        test_key = train_key
        scale = 1.0

    train_value = _finite(train_metrics.get(train_key))
    test_value = _finite((metrics or {}).get(test_key))
    diagnosis["train_metric"], diagnosis["test_metric"] = train_value, test_value

    if train_value is None or test_value is None:
        diagnosis["verdict"] = "unknown"
        diagnosis["reasons"].append("Not enough metric data to diagnose the fit.")
        return diagnosis

    gap = train_value - test_value
    diagnosis["gap"] = _finite(gap)
    diagnosis["metric_name"] = train_key

    leakage_note = ((leakage_summary or {}).get("severity") or "info").lower()

    if leakage_note == "critical" and gap > 0.3:
        diagnosis["verdict"] = "potential_leakage"
        diagnosis["confidence"] = "high"
        diagnosis["reasons"].append(
            "Training performance is far above test performance *and* a "
            "critical leakage check fired. A feature is very likely encoding "
            "the target; fix the data before trusting any number here."
        )
        return diagnosis

    if gap > _OVERFIT_GAP:
        diagnosis["verdict"] = "overfitting"
        diagnosis["confidence"] = "medium"
        diagnosis["reasons"].append(
            f"Training {train_key} ({train_value:.3f}) is well above test "
            f"{test_key} ({test_value:.3f}). The model fits the training data "
            "much better than it generalises; consider more data, less "
            "complexity, or stronger regularisation."
        )
    elif test_value < (0.6 if task_type == "classification" else 0.2):
        diagnosis["verdict"] = "underfitting"
        diagnosis["confidence"] = "medium"
        diagnosis["reasons"].append(
            "Both train and test performance are low, so the model is not "
            "capturing the pattern at all. Try a more expressive model or "
            "more features."
        )
    else:
        diagnosis["verdict"] = "good_fit"
        diagnosis["confidence"] = "medium"
        diagnosis["reasons"].append(
            "Training and test performance are close, which is the sign of a "
            "well-generalising model."
        )
        if leakage_note == "warning":
            diagnosis["reasons"].append(
                "A leakage check raised a warning; review the findings before "
                "trusting this result."
            )
    return diagnosis


def build_diagnostics(
    task_type: str,
    y_true: np.ndarray,
    y_proba: np.ndarray | None,
    y_pred: np.ndarray,
    classes: list[str] | None,
    metrics: dict,
    train_metrics: dict,
    leakage_summary: dict | None = None,
    residual_sample: list[dict] | None = None,
) -> dict[str, Any]:
    """One call the API uses: task-appropriate curves plus the diagnosis."""
    payload: dict[str, Any] = {"diagnosis": diagnose_fit(task_type, metrics, train_metrics, leakage_summary)}
    if task_type == "classification":
        if y_proba is None:
            payload["note"] = "This model does not expose probabilities, so ROC, precision-recall, calibration and threshold tables are unavailable."
            return payload
        payload.update(classification_diagnostics(np.asarray(y_true), np.asarray(y_proba, dtype=float), np.asarray(y_pred), classes or []))
    else:
        payload.update(regression_diagnostics(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float), residual_sample))
    return payload

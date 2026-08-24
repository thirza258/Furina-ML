"""Markdown evaluation report for a training run.

``markdown_report`` is deliberately duck-typed: it reads a ``TrainingRun`` with
plain attribute access and imports nothing from Django, so it can be rendered
for any object (a model instance, a namespace, a dict-backed stub) and unit
tested on its own.  Every field is optional -- a missing, ``None`` or empty
value is rendered as ``n/a`` instead of raising.
"""

from __future__ import annotations

import datetime as _datetime
import math
from typing import Any, Iterable

#: Metric order used for the metrics tables, per task type.
CLASSIFICATION_METRICS: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "f2",
    "roc_auc",
    "log_loss",
    "matthews_corrcoef",
)
REGRESSION_METRICS: tuple[str, ...] = (
    "r2",
    "mae",
    "mse",
    "rmse",
    "mape",
    "explained_variance",
    "max_error",
)

METRIC_LABELS: dict[str, str] = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "precision": "Precision (weighted)",
    "recall": "Recall (weighted)",
    "f1": "F1 (weighted)",
    "f2": "F2 (weighted)",
    "roc_auc": "ROC AUC",
    "log_loss": "Log loss",
    "matthews_corrcoef": "Matthews correlation",
    "r2": "R2",
    "mae": "Mean absolute error",
    "mse": "Mean squared error",
    "rmse": "Root mean squared error",
    "mape": "Mean absolute percentage error",
    "explained_variance": "Explained variance",
    "max_error": "Max error",
}

MAX_FEATURE_IMPORTANCE_ROWS: int = 20
MAX_RESIDUAL_ROWS: int = 10
NOT_AVAILABLE: str = "n/a"


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------


def _get(run: Any, name: str, default: Any = None) -> Any:
    """Attribute access that never raises, whatever ``run`` is."""
    try:
        value = getattr(run, name, default)
    except Exception:  # noqa: BLE001 - a lazy relation may explode
        return default
    return default if value is None else value


def _format_number(value: Any) -> str:
    """Format a number with four significant digits and no exponent."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return NOT_AVAILABLE
    if number == 0:
        return "0"
    magnitude = math.floor(math.log10(abs(number)))
    if -5 <= magnitude < 15:
        decimals = max(0, 3 - magnitude)
        text = f"{number:,.{decimals}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text
    return f"{number:.4g}"


def _format_value(value: Any) -> str:
    """Render any value for a table cell, using ``n/a`` for missing data."""
    if value is None:
        return NOT_AVAILABLE
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _format_number(value)
    if isinstance(value, (_datetime.datetime, _datetime.date)):
        return value.isoformat()
    if isinstance(value, (list, tuple, set)):
        items = [_format_value(item) for item in value]
        return ", ".join(items) if items else NOT_AVAILABLE
    if isinstance(value, dict):
        if not value:
            return NOT_AVAILABLE
        return ", ".join(f"{key}={_format_value(item)}" for key, item in value.items())
    text = str(value).strip()
    return text or NOT_AVAILABLE


def _cell(value: Any) -> str:
    """Escape a value so it cannot break out of a Markdown table cell."""
    text = _format_value(value)
    return text.replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _int(value: Any, default: int = 0) -> int:
    """Best-effort integer conversion for row counts and supports."""
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_or_none(value: Any) -> float | None:
    """Return a finite float, or None for missing/NaN/Inf values."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _table(headers: list[str], rows: Iterable[Iterable[Any]]) -> list[str]:
    """Render a Markdown table, or nothing when there are no rows."""
    body = [[_cell(cell) for cell in row] for row in rows]
    if not body:
        return []
    lines = ["| " + " | ".join(_cell(head) for head in headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in body:
        padded = list(row) + [NOT_AVAILABLE] * (len(headers) - len(row))
        lines.append("| " + " | ".join(padded[: len(headers)]) + " |")
    return lines


def _metric_label(key: str) -> str:
    """Human label for a metric key."""
    return METRIC_LABELS.get(key, str(key).replace("_", " ").capitalize())


def _as_dict(value: Any) -> dict:
    """Coerce a value to a dict, tolerating None and junk."""
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list:
    """Coerce a value to a list, tolerating None, junk and numpy arrays."""
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, (str, bytes, dict)) or value is None:
        return []
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        try:
            converted = to_list()
        except Exception:  # noqa: BLE001 - fall through to the generic path
            converted = None
        if isinstance(converted, list):
            return converted
    try:
        return list(value)
    except TypeError:
        return []


def _duration(milliseconds: Any) -> str:
    """Render a duration in ms with a seconds hint for long runs."""
    value = _float_or_none(milliseconds)
    if value is None:
        return NOT_AVAILABLE
    if value >= 1000:
        return f"{_format_number(value)} ms ({_format_number(value / 1000)} s)"
    return f"{_format_number(value)} ms"


def _run_name(run: Any) -> str:
    """The best available display name for the run."""
    for attribute in ("display_name", "name"):
        value = _get(run, attribute)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    algorithm = _format_value(_get(run, "algorithm_label") or _get(run, "algorithm"))
    target = _format_value(_get(run, "target_column"))
    algorithm = " ".join(algorithm.split())
    target = " ".join(target.split())
    if algorithm != NOT_AVAILABLE and target != NOT_AVAILABLE:
        return f"{algorithm} on {target}"
    return "Training run"


def _dataset_name(run: Any) -> str:
    """Name of the dataset the run used, without touching a broken relation."""
    dataset = _get(run, "dataset")
    if dataset is None:
        return _format_value(_get(run, "dataset_name"))
    name = _get(dataset, "name")
    if name is None:
        return _format_value(dataset)
    return _format_value(name)


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


def _overview_section(run: Any) -> list[str]:
    """Metadata table: dataset, algorithm, task, split, timing."""
    task_type = _format_value(_get(run, "task_type"))
    features = _as_list(_get(run, "feature_columns"))
    if features:
        shown = [str(name) for name in features[:30]]
        if len(features) > 30:
            shown.append(f"and {len(features) - 30} more")
        feature_text = ", ".join(shown)
    else:
        feature_text = NOT_AVAILABLE

    test_size = _float_or_none(_get(run, "test_size"))
    split = (
        f"{_format_number(test_size)} ({_format_number(test_size * 100)}% held out)"
        if test_size is not None
        else NOT_AVAILABLE
    )
    n_train = _int(_get(run, "n_train"))
    n_test = _int(_get(run, "n_test"))
    rows = f"{n_train:,} train + {n_test:,} test = {n_train + n_test:,} rows"

    primary_name = _get(run, "primary_metric_name")
    if not primary_name:
        primary_name = "r2" if task_type == "regression" else "accuracy"
    primary_value = _get(run, "primary_metric")
    if primary_value is None:
        primary_value = _as_dict(_get(run, "metrics")).get(primary_name)
    primary = f"{_metric_label(str(primary_name))}: {_format_value(_float_or_none(primary_value))}"

    classes = _as_list(_get(run, "classes"))
    entries: list[tuple[str, Any]] = [
        ("Run id", _get(run, "id")),
        ("Status", _get(run, "status", "unknown")),
        ("Dataset", _dataset_name(run)),
        ("Algorithm", _get(run, "algorithm_label") or _get(run, "algorithm")),
        ("Algorithm key", _get(run, "algorithm")),
        ("Task", task_type),
        ("Target column", _get(run, "target_column")),
        (f"Features ({len(features)})", feature_text),
        ("Headline result", primary),
        ("Test split", split),
        ("Random seed", _get(run, "random_state")),
        ("Stratified split", _get(run, "stratify")),
        ("Scaled features", _get(run, "scale_features")),
        ("Cross-validation folds", _get(run, "cv_folds")),
        ("Rows used", rows),
        ("Training duration", _duration(_get(run, "duration_ms"))),
        ("Created", _get(run, "created_at")),
    ]
    if classes:
        entries.insert(8, (f"Classes ({len(classes)})", classes))

    lines = ["## Overview", ""]
    lines.extend(_table(["Field", "Value"], [[key, value] for key, value in entries]))
    lines.append("")
    return lines


def _metrics_section(run: Any) -> list[str]:
    """Test metrics with the matching train metrics beside them."""
    test = _as_dict(_get(run, "metrics"))
    train = _as_dict(_get(run, "train_metrics"))
    if not test and not train:
        return ["## Metrics", "", "No metrics were recorded for this run.", ""]

    task_type = str(_get(run, "task_type", ""))
    order = REGRESSION_METRICS if task_type == "regression" else CLASSIFICATION_METRICS
    keys = [key for key in order if key in test or key in train]
    keys.extend(sorted((set(test) | set(train)) - set(keys), key=str))

    rows = [
        [
            _metric_label(key),
            _format_value(_float_or_none(test.get(key))),
            _format_value(_float_or_none(train.get(key))),
        ]
        for key in keys
    ]
    lines = ["## Metrics", ""]
    lines.extend(_table(["Metric", "Test set", "Training set"], rows))
    lines.append("")
    lines.append(
        "A large gap between the test and training columns is a sign the model is "
        "memorising the training rows rather than learning the pattern."
    )
    lines.append("")
    return lines


def _cv_section(run: Any) -> list[str]:
    """Cross-validation summary, only when the run recorded one."""
    cv = _as_dict(_get(run, "cv"))
    if not cv:
        return []
    scores = [_float_or_none(score) for score in _as_list(cv.get("scores"))]
    lines = ["## Cross-validation", ""]
    summary = [
        ["Folds", _int(cv.get("folds"), 0) or NOT_AVAILABLE],
        ["Scoring", cv.get("scoring")],
        ["Mean score", _format_value(_float_or_none(cv.get("mean")))],
        ["Standard deviation", _format_value(_float_or_none(cv.get("std")))],
    ]
    lines.extend(_table(["Field", "Value"], summary))
    lines.append("")
    if scores:
        lines.extend(
            _table(
                ["Fold", "Score"],
                [[f"Fold {index}", _format_value(score)] for index, score in enumerate(scores, start=1)],
            )
        )
        lines.append("")
    return lines


def _confusion_section(run: Any) -> list[str]:
    """Confusion matrix rendered with class labels on both axes."""
    matrix_data = _as_dict(_get(run, "confusion_matrix"))
    matrix = [_as_list(row) for row in _as_list(matrix_data.get("matrix"))]
    if not matrix:
        return []
    labels = [str(label) for label in _as_list(matrix_data.get("labels"))]
    size = max(len(matrix), max((len(row) for row in matrix), default=0), len(labels))
    while len(labels) < size:
        labels.append(f"class {len(labels) + 1}")
    labels = labels[:size]

    header = ["Actual \\ Predicted"] + labels + ["Total"]
    rows = []
    for index in range(size):
        row = matrix[index] if index < len(matrix) else []
        counts = [_int(row[position]) if position < len(row) else 0 for position in range(size)]
        rows.append([f"**{labels[index]}**"] + [f"{count:,}" for count in counts] + [f"{sum(counts):,}"])
    lines = ["## Confusion matrix", "", "Rows are the true class, columns are what the model predicted.", ""]
    lines.extend(_table(header, rows))
    lines.append("")
    return lines


def _per_class_section(run: Any) -> list[str]:
    """Per-class precision / recall / F1 / support."""
    per_class = [item for item in _as_list(_get(run, "per_class")) if isinstance(item, dict)]
    if not per_class:
        return []
    rows = [
        [
            entry.get("label"),
            _format_value(_float_or_none(entry.get("precision"))),
            _format_value(_float_or_none(entry.get("recall"))),
            _format_value(_float_or_none(entry.get("f1"))),
            f"{_int(entry.get('support')):,}",
        ]
        for entry in per_class
    ]
    lines = ["## Per-class performance", ""]
    lines.extend(_table(["Class", "Precision", "Recall", "F1", "Support"], rows))
    lines.append("")
    return lines


def _feature_importance_section(run: Any) -> list[str]:
    """Top feature importances, strongest first."""
    items = [item for item in _as_list(_get(run, "feature_importance")) if isinstance(item, dict)]
    if not items:
        return []
    ranked = sorted(
        items,
        key=lambda entry: abs(_float_or_none(entry.get("importance")) or 0.0),
        reverse=True,
    )
    top = ranked[:MAX_FEATURE_IMPORTANCE_ROWS]
    rows = [
        [str(index), entry.get("feature"), _format_value(_float_or_none(entry.get("importance")))]
        for index, entry in enumerate(top, start=1)
    ]
    lines = ["## Feature importance", ""]
    if len(ranked) > len(top):
        lines.append(f"Showing the top {len(top)} of {len(ranked)} features.")
        lines.append("")
    lines.extend(_table(["#", "Feature", "Importance"], rows))
    lines.append("")
    return lines


def _residual_section(run: Any) -> list[str]:
    """A small sample of actual vs predicted values for regression runs."""
    sample = [item for item in _as_list(_get(run, "residual_sample")) if isinstance(item, dict)]
    if not sample:
        return []
    rows = []
    for entry in sample[:MAX_RESIDUAL_ROWS]:
        actual = _float_or_none(entry.get("actual"))
        predicted = _float_or_none(entry.get("predicted"))
        error = None if actual is None or predicted is None else predicted - actual
        rows.append(
            [
                _format_value(actual),
                _format_value(predicted),
                _format_value(error),
            ]
        )
    lines = ["## Residual sample", ""]
    if len(sample) > len(rows):
        lines.append(f"Showing {len(rows)} of {len(sample)} sampled test rows.")
        lines.append("")
    lines.extend(_table(["Actual", "Predicted", "Error (predicted - actual)"], rows))
    lines.append("")
    return lines


def _hyperparameters_section(run: Any) -> list[str]:
    """Effective hyperparameters, flagging the ones the user chose."""
    effective = _as_dict(_get(run, "effective_hyperparameters"))
    supplied = _as_dict(_get(run, "hyperparameters"))
    lines = ["## Hyperparameters", ""]
    if not effective and not supplied:
        lines.append("This run used the algorithm's default hyperparameters.")
        lines.append("")
        return lines
    keys = sorted(set(effective) | set(supplied), key=str)
    rows = [
        [
            str(key),
            _format_value(effective.get(key, supplied.get(key))),
            "chosen by you" if key in supplied else "default",
        ]
        for key in keys
    ]
    lines.extend(_table(["Parameter", "Value", "Source"], rows))
    lines.append("")
    return lines


def _warnings_section(run: Any) -> list[str]:
    """Warnings collected while training."""
    warnings = [_format_value(item) for item in _as_list(_get(run, "warnings"))]
    warnings = [item for item in warnings if item and item != NOT_AVAILABLE]
    lines = ["## Warnings", ""]
    if not warnings:
        lines.append("No warnings were recorded for this run.")
    else:
        lines.extend(f"- {item}" for item in warnings)
    lines.append("")
    return lines


def _error_section(run: Any) -> list[str]:
    """Failure details, only when the run did not succeed."""
    status = str(_get(run, "status", "success"))
    error = _format_value(_get(run, "error"))
    if status == "success" and error == NOT_AVAILABLE:
        return []
    lines = ["## What went wrong", ""]
    lines.append(error if error != NOT_AVAILABLE else "The run did not finish successfully.")
    lines.append("")
    return lines


def markdown_report(run) -> str:
    """Render a full human-readable Markdown evaluation report for a run.

    ``run`` is a ``core.models.TrainingRun`` (or anything exposing the same
    attributes).  Missing or empty fields are rendered as ``n/a`` and never
    raise, so a partially populated or failed run still produces a report.
    """
    lines: list[str] = [f"# Training report: {_run_name(run)}", ""]
    lines.extend(_error_section(run))
    lines.extend(_overview_section(run))
    lines.extend(_metrics_section(run))
    lines.extend(_cv_section(run))
    lines.extend(_confusion_section(run))
    lines.extend(_per_class_section(run))
    lines.extend(_feature_importance_section(run))
    lines.extend(_residual_section(run))
    lines.extend(_hyperparameters_section(run))
    lines.extend(_warnings_section(run))
    lines.append("---")
    lines.append("")
    lines.append("Generated by Furina ML.")
    return "\n".join(lines).rstrip() + "\n"

"""Turn raw form input into predictions from a trained model artifact."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd

from core.ml.io import json_safe

if TYPE_CHECKING:  # pragma: no cover - typing only
    from core.ml.trainer import ModelArtifact

BLANK_TEXTS = {"", "nan", "none", "null", "na", "n/a"}
"""Text a form may submit for an empty *number* field."""


class PredictionError(Exception):
    """Raised when a prediction request cannot be fulfilled by the model."""


def _stringify(value: Any) -> str:
    """Render a value the way the training preprocessor rendered categories."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    return str(value)


def _is_missing(value: Any) -> bool:
    """True for None, NaN and NaT."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _is_blank_text(value: Any) -> bool:
    """True for a missing value or text that holds nothing but spaces.

    Deliberately narrow: a category whose real training value is "NA" or
    "None" must survive untouched.
    """
    if _is_missing(value):
        return True
    return isinstance(value, str) and not value.strip()


def _is_blank_number(value: Any) -> bool:
    """True for a missing value or text that stands in for a missing number."""
    if _is_missing(value):
        return True
    return isinstance(value, str) and value.strip().lower() in BLANK_TEXTS


def _coerce_number(value: Any, column: str, row_number: int) -> float:
    """Coerce one incoming value to a float, using NaN for blanks."""
    if _is_blank_number(value):
        return float("nan")
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else float("nan")
    text = _stringify(value).strip()
    try:
        number = float(text)
    except ValueError:
        raise PredictionError(
            f"Value '{text}' for '{column}' in row {row_number} must be a number."
        ) from None
    return number if math.isfinite(number) else float("nan")


def _coerce_text(value: Any, options: Sequence[str] | None) -> str | None:
    """Coerce one incoming value to text, using None for blanks."""
    if _is_blank_text(value):
        return None
    text = _stringify(value).strip()
    if not options or text in options:
        return text
    # A number typed as "1" should still match a category recorded as "1.0".
    try:
        number = float(text)
    except ValueError:
        return text
    candidates = [_stringify(number)]
    if number.is_integer():
        candidates.append(_stringify(int(number)))
    for candidate in candidates:
        if candidate in options:
            return candidate
    return text


def _prediction_frame(
    artifact: "ModelArtifact", rows: Sequence[dict]
) -> pd.DataFrame:
    """Build a one-row-per-input DataFrame holding every training feature."""
    schema = {
        str(field.get("name")): field
        for field in (getattr(artifact, "input_schema", None) or [])
        if field.get("name") is not None
    }
    columns = [str(column) for column in (getattr(artifact, "feature_columns", None) or [])]
    if not columns:
        columns = list(schema)
    if not columns:
        raise PredictionError("This model does not record which columns it needs.")

    prepared: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise PredictionError(
                f"Row {index} must be an object of column names and values."
            )
        cleaned: dict[str, Any] = {}
        filled = 0
        for column in columns:
            field = schema.get(column) or {}
            kind = str(field.get("kind") or "categorical")
            raw = row.get(column)
            if kind == "numeric":
                value: Any = _coerce_number(raw, column, index)
                if not (isinstance(value, float) and math.isnan(value)):
                    filled += 1
            else:
                value = _coerce_text(raw, field.get("options"))
                if value is not None:
                    filled += 1
            cleaned[column] = value
        if filled == 0:
            raise PredictionError(
                f"Row {index} has no values. Fill in at least one field."
            )
        prepared.append(cleaned)

    data: dict[str, pd.Series] = {}
    for column in columns:
        kind = str((schema.get(column) or {}).get("kind") or "categorical")
        values = [item[column] for item in prepared]
        dtype = "float64" if kind == "numeric" else "object"
        data[column] = pd.Series(values, dtype=dtype)
    return pd.DataFrame(data, columns=columns)


def _class_labels(artifact: "ModelArtifact") -> list[str]:
    """The original training labels, stringified."""
    return [str(label) for label in (getattr(artifact, "classes", None) or [])]


def _decode_label(artifact: "ModelArtifact", value: Any, labels: Sequence[str]) -> str:
    """Turn an encoded class back into the original training label."""
    encoder = getattr(artifact, "label_encoder", None)
    if encoder is not None:
        try:
            return _stringify(encoder.inverse_transform(np.asarray([value]))[0])
        except (ValueError, TypeError, AttributeError):
            pass
    try:
        index = int(value)
    except (TypeError, ValueError):
        return _stringify(value)
    if 0 <= index < len(labels):
        return labels[index]
    return _stringify(value)


def predict(artifact: "ModelArtifact", rows: list[dict]) -> list[dict]:
    """Predict one result per input row.

    Raw values from the UI are coerced to the kinds recorded at training time;
    blank values become NaN so the pipeline's imputers fill them in.
    """
    if artifact is None or getattr(artifact, "pipeline", None) is None:
        raise PredictionError("This run does not have a trained model to predict with.")
    if not isinstance(rows, (list, tuple)):
        raise PredictionError("Provide a list of rows to predict.")
    if len(rows) == 0:
        raise PredictionError("Provide at least one row of values to predict.")

    frame = _prediction_frame(artifact, rows)
    pipeline = artifact.pipeline

    try:
        raw_predictions = np.asarray(pipeline.predict(frame))
    except Exception as exc:  # noqa: BLE001 - surfaced as a readable message
        raise PredictionError(
            f"The model could not make a prediction from these values: {exc}"
        ) from None

    is_classification = str(getattr(artifact, "task_type", "")) == "classification"
    labels = _class_labels(artifact)

    probabilities: np.ndarray | None = None
    codes: np.ndarray | None = None
    if is_classification:
        # Best-effort: an estimator with no usable probabilities degrades to
        # predictions only. The catch is broad because a library exception need
        # not derive from a builtin -- LightGBMError subclasses Exception
        # directly, so a narrow tuple would let it escape as a 500.
        try:
            probabilities = np.asarray(pipeline.predict_proba(frame), dtype=float)
        except Exception:  # noqa: BLE001 - absence of probabilities is not an error
            probabilities = None
        if probabilities is not None and probabilities.ndim == 2:
            model = getattr(pipeline, "named_steps", {}).get("model")
            model_classes = getattr(model, "classes_", None)
            if model_classes is not None and len(model_classes) == probabilities.shape[1]:
                codes = np.asarray(model_classes)
            else:
                codes = np.arange(probabilities.shape[1])
        else:
            probabilities = None

    results: list[dict] = []
    for position in range(len(frame)):
        value = raw_predictions[position] if position < len(raw_predictions) else None
        if is_classification:
            prediction: Any = _decode_label(artifact, value, labels)
            row_probabilities: list[dict] | None = None
            if probabilities is not None and codes is not None:
                row_probabilities = []
                for column, code in enumerate(codes):
                    label = _decode_label(artifact, code, labels)
                    score = float(probabilities[position, column])
                    if not math.isfinite(score):
                        score = 0.0
                    row_probabilities.append(
                        {"label": label, "probability": score}
                    )
                row_probabilities.sort(
                    key=lambda item: item["probability"], reverse=True
                )
        else:
            row_probabilities = None
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = float("nan")
            prediction = number if math.isfinite(number) else None
        results.append({"prediction": prediction, "probabilities": row_probabilities})

    return json_safe(results)

"""Model training, evaluation and artifact persistence for Furina ML."""

from __future__ import annotations

import io
import math
import time
from datetime import datetime, timezone
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_recall_fscore_support,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

from core.ml.io import json_safe
from core.ml.pipeline import build_pipeline, expanded_feature_names, usable_feature_kinds
from core.ml.profiling import column_kind, infer_task
from core.ml.registry import (
    AlgorithmNotFound,
    build_estimator,
    coerce_hyperparameters,
    get_spec,
)

try:  # scikit-learn >= 1.4
    from sklearn.metrics import root_mean_squared_error
except ImportError:  # pragma: no cover - older scikit-learn
    root_mean_squared_error = None

MAX_FEATURE_IMPORTANCE = 40
MAX_RESIDUAL_SAMPLE = 300
MAX_OPTION_VALUES = 200
MIN_TEST_SIZE = 0.05
MAX_TEST_SIZE = 0.95
MIN_CV_FOLDS = 2
MAX_CV_FOLDS = 10

CLASSIFICATION_METRIC_KEYS = (
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
REGRESSION_METRIC_KEYS = (
    "r2",
    "mae",
    "mse",
    "rmse",
    "mape",
    "explained_variance",
    "max_error",
)


class TrainingError(Exception):
    """Raised for any problem the user can fix. ``str(e)`` is shown to them."""


# --------------------------------------------------------------------------
# small conversion helpers (JSON safety by construction)
# --------------------------------------------------------------------------


def _finite(value: Any) -> float | None:
    """Return ``value`` as a plain float, or None when it is missing/infinite."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _number(value: Any, default: float = 0.0) -> float:
    """Return ``value`` as a plain finite float, falling back to ``default``."""
    number = _finite(value)
    return default if number is None else number


def _stringify(value: Any) -> str:
    """Render a label as text the way the preprocessor renders categories."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    return str(value)


def _unique_ordered(values: Sequence[Any]) -> list[Any]:
    """De-duplicate while keeping the original order."""
    seen: set[Any] = set()
    result: list[Any] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def classification_metrics(
    y_true: Any, y_pred: Any, y_proba: Any, labels: Sequence[int]
) -> dict:
    """Compute every classification metric of the contract.

    All keys are always present; a metric that cannot be computed is None.
    """
    true_values = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    label_list = [int(label) for label in labels]

    metrics: dict[str, float | None] = {
        "accuracy": _finite(accuracy_score(true_values, predicted)),
        "balanced_accuracy": _finite(balanced_accuracy_score(true_values, predicted)),
        "precision": _finite(
            precision_score(
                true_values,
                predicted,
                labels=label_list,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall": _finite(
            recall_score(
                true_values,
                predicted,
                labels=label_list,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1": _finite(
            f1_score(
                true_values,
                predicted,
                labels=label_list,
                average="weighted",
                zero_division=0,
            )
        ),
        "f2": _finite(
            fbeta_score(
                true_values,
                predicted,
                beta=2,
                labels=label_list,
                average="weighted",
                zero_division=0,
            )
        ),
        "roc_auc": None,
        "log_loss": None,
        "matthews_corrcoef": _finite(matthews_corrcoef(true_values, predicted)),
    }

    proba = _usable_proba(y_proba, true_values, label_list)
    if proba is not None:
        try:
            if len(label_list) == 2:
                metrics["roc_auc"] = _finite(roc_auc_score(true_values, proba[:, 1]))
            else:
                metrics["roc_auc"] = _finite(
                    roc_auc_score(
                        true_values,
                        proba,
                        multi_class="ovr",
                        average="weighted",
                        labels=label_list,
                    )
                )
        except (ValueError, IndexError):
            metrics["roc_auc"] = None
        try:
            metrics["log_loss"] = _finite(
                log_loss(true_values, proba, labels=label_list)
            )
        except (ValueError, IndexError):
            metrics["log_loss"] = None

    return metrics


def _usable_proba(
    y_proba: Any, y_true: np.ndarray, labels: Sequence[int]
) -> np.ndarray | None:
    """Return probabilities only when they line up with every known class."""
    if y_proba is None:
        return None
    proba = np.asarray(y_proba, dtype=float)
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    if proba.ndim != 2 or proba.shape[1] != len(labels):
        return None
    if len(np.unique(y_true)) < len(labels):
        return None
    if not np.isfinite(proba).all():
        return None
    return proba


def regression_metrics(y_true: Any, y_pred: Any) -> dict:
    """Compute every regression metric of the contract."""
    true_values = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    mse = float(mean_squared_error(true_values, predicted))
    if root_mean_squared_error is not None:
        rmse = float(root_mean_squared_error(true_values, predicted))
    else:  # pragma: no cover - older scikit-learn
        rmse = math.sqrt(mse) if mse >= 0 else float("nan")

    metrics: dict[str, float | None] = {
        "r2": _finite(r2_score(true_values, predicted)),
        "mae": _finite(mean_absolute_error(true_values, predicted)),
        "mse": _finite(mse),
        "rmse": _finite(rmse),
        "mape": None,
        "explained_variance": _finite(
            explained_variance_score(true_values, predicted)
        ),
        "max_error": _finite(max_error(true_values, predicted)),
    }
    if true_values.size and not bool(np.any(true_values == 0)):
        metrics["mape"] = _finite(
            mean_absolute_percentage_error(true_values, predicted)
        )
    return metrics


# --------------------------------------------------------------------------
# prediction input schema
# --------------------------------------------------------------------------


def build_input_schema(df: pd.DataFrame, feature_columns: Sequence[str]) -> list[dict]:
    """Describe every feature column so the UI can build a prediction form."""
    schema: list[dict] = []
    for column in feature_columns:
        if column not in df.columns:
            continue
        series = df[column]
        kind = column_kind(series)
        non_null = series.dropna()
        field: dict[str, Any] = {
            "name": str(column),
            "kind": kind,
            "required": bool(int(series.isna().sum()) == 0),
            "options": None,
            "min": None,
            "max": None,
            "default": None,
        }
        if kind == "numeric":
            numbers = pd.to_numeric(non_null, errors="coerce").dropna()
            if len(numbers):
                field["min"] = _finite(numbers.min())
                field["max"] = _finite(numbers.max())
                field["default"] = _finite(numbers.median())
        else:
            counts = non_null.astype("object").value_counts()
            values = [_stringify(value) for value in counts.index]
            options = _unique_ordered(values)
            if len(options) > MAX_OPTION_VALUES:
                options = options[:MAX_OPTION_VALUES]
            field["options"] = sorted(options)
            if values:
                field["default"] = values[0]
        schema.append(field)
    return schema


# --------------------------------------------------------------------------
# artifact
# --------------------------------------------------------------------------


class ModelArtifact:
    """Everything needed to reproduce a prediction later. Joblib-serializable."""

    def __init__(
        self,
        *,
        pipeline: Any,
        task_type: str,
        algorithm: str,
        target_column: str,
        feature_columns: Sequence[str],
        classes: Sequence[str] | None = None,
        label_encoder: Any = None,
        input_schema: Sequence[dict] | None = None,
        sklearn_version: str | None = None,
        created_at: str | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.task_type = str(task_type)
        self.algorithm = str(algorithm)
        self.target_column = str(target_column)
        self.feature_columns = [str(column) for column in feature_columns]
        self.classes = None if classes is None else [str(label) for label in classes]
        self.label_encoder = label_encoder
        self.input_schema = list(input_schema or [])
        self.sklearn_version = sklearn_version or sklearn.__version__
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ModelArtifact(algorithm={self.algorithm!r}, "
            f"task_type={self.task_type!r}, target={self.target_column!r})"
        )


def dump_artifact(artifact: ModelArtifact) -> bytes:
    """Serialize a model artifact to bytes with joblib."""
    buffer = io.BytesIO()
    joblib.dump(artifact, buffer)
    return buffer.getvalue()


def load_artifact(path_or_bytes: Any) -> ModelArtifact:
    """Load a model artifact from a path, raw bytes or an open file object."""
    source = path_or_bytes
    if isinstance(source, (bytes, bytearray, memoryview)):
        source = io.BytesIO(bytes(source))
    try:
        artifact = joblib.load(source)
    except Exception:  # noqa: BLE001 - any unpickling problem is user facing
        raise TrainingError(
            "The saved model file could not be read. It may be damaged or saved by a "
            "different version of Furina ML."
        ) from None
    if not isinstance(artifact, ModelArtifact):
        raise TrainingError("That file does not contain a Furina ML model.")
    return artifact


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def _resolve_feature_columns(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: Sequence[str] | None,
    warnings: list[str],
) -> list[str]:
    """Validate the requested feature columns, defaulting to everything else."""
    if not feature_columns:
        return [column for column in df.columns if column != target_column]

    requested = _unique_ordered([str(column) for column in feature_columns])
    missing = [column for column in requested if column not in df.columns]
    if missing:
        names = ", ".join(f"'{column}'" for column in missing)
        raise TrainingError(f"Column {names} is not in the dataset.")
    if target_column in requested:
        requested = [column for column in requested if column != target_column]
        warnings.append(
            f"The target column '{target_column}' was removed from the feature list."
        )
    return requested


def _encode_classification_target(series: pd.Series) -> tuple[np.ndarray, Any, list[str]]:
    """Label-encode a classification target and keep the original labels."""
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        raw = series.to_numpy()
    else:
        raw = np.asarray(
            [_stringify(value) for value in series.astype("object")], dtype=object
        )
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(raw).astype(int)
    classes = [_stringify(label) for label in encoder.classes_]
    return encoded, encoder, classes


def _numeric_regression_target(
    series: pd.Series, target_column: str, warnings: list[str]
) -> np.ndarray:
    """Turn a regression target into a float array or explain why it cannot be."""
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=float)
    if pd.api.types.is_datetime64_any_dtype(series):
        warnings.append(
            f"The target column '{target_column}' holds dates, so it was converted "
            "to seconds since 1970 for regression."
        )
        return series.astype("int64").to_numpy(dtype=float) / 1_000_000_000.0
    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy(dtype=float)
    converted = pd.to_numeric(series, errors="coerce")
    if bool(converted.isna().any()):
        raise TrainingError(
            f"Regression needs a numeric target, but column '{target_column}' "
            "contains values that are not numbers."
        )
    return converted.to_numpy(dtype=float)


def _feature_spread(pipe: Any, X_train: Any, width: int) -> np.ndarray | None:
    """Standard deviation of every expanded feature, or None when unavailable."""
    if X_train is None:
        return None
    step = pipe.named_steps.get("preprocess")
    if step is None:
        return None
    try:
        matrix = np.asarray(step.transform(X_train), dtype=float)
    except Exception:  # noqa: BLE001 - importance is a diagnostic, never fatal
        return None
    if matrix.ndim != 2 or matrix.shape[1] != width:
        return None
    spread = np.nanstd(matrix, axis=0)
    if not np.isfinite(spread).all() or not spread.any():
        return None
    return spread


def _feature_importance(
    pipe: Any,
    numeric: Sequence[str],
    categorical: Sequence[str],
    X_train: Any = None,
) -> list[dict] | None:
    """Rank the expanded features of a fitted pipeline, best first.

    Tree models report their own scale-free importances. Linear models only
    expose coefficients, which are per unit of the column they multiply and so
    cannot be compared with each other unless every column happens to be on the
    same scale. Each coefficient is therefore multiplied by the spread of its
    column, which leaves the numbers unchanged when the features were
    standardized and makes them comparable when they were not.
    """
    model = pipe.named_steps.get("model")
    values: np.ndarray | None = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float).ravel()
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim > 1:
            values = np.abs(coef).mean(axis=0).ravel()
        else:
            values = np.abs(coef).ravel()
        spread = _feature_spread(pipe, X_train, values.size)
        if spread is not None:
            values = values * spread
    if values is None or values.size == 0:
        return None

    names = expanded_feature_names(
        pipe.named_steps.get("preprocess"), numeric, categorical
    )
    if len(names) != values.size:
        names = [f"feature_{index}" for index in range(values.size)]

    ranked = [
        {"feature": str(name), "importance": _number(value)}
        for name, value in zip(names, values)
    ]
    ranked.sort(key=lambda item: item["importance"], reverse=True)
    return ranked[:MAX_FEATURE_IMPORTANCE]


def _residual_sample(y_true: Any, y_pred: Any) -> list[dict]:
    """Up to 300 actual/predicted pairs from the test split."""
    sample: list[dict] = []
    for actual, predicted in zip(np.asarray(y_true), np.asarray(y_pred)):
        actual_value = _finite(actual)
        predicted_value = _finite(predicted)
        if actual_value is None or predicted_value is None:
            continue
        sample.append({"actual": actual_value, "predicted": predicted_value})
        if len(sample) >= MAX_RESIDUAL_SAMPLE:
            break
    return sample


def _safe_proba(pipe: Any, X: pd.DataFrame) -> np.ndarray | None:
    """Return predict_proba output, or None when the estimator has none.

    The catch is deliberately broad. A library exception need not derive from a
    builtin: `LightGBMError` subclasses `Exception` directly, so the narrower
    `(AttributeError, ValueError, NotImplementedError)` let it escape and turn a
    missing-probabilities case into a 500. Every caller already treats `None` as
    "no probabilities", and ROC AUC and log loss are reported as null.
    """
    try:
        return np.asarray(pipe.predict_proba(X), dtype=float)
    except Exception:  # noqa: BLE001 - absence of probabilities is not an error
        return None


def _cross_validate(
    pipe: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    task_type: str,
    cv_folds: Any,
    random_state: Any,
    warnings: list[str],
) -> dict | None:
    """Run cross-validation over the whole usable dataset."""
    try:
        folds = int(cv_folds)
    except (TypeError, ValueError):
        warnings.append("Cross-validation was skipped because the fold count is not a number.")
        return None

    requested = folds
    folds = max(MIN_CV_FOLDS, min(MAX_CV_FOLDS, folds))
    if folds != requested:
        warnings.append(
            f"The number of cross-validation folds was changed from {requested} to {folds}."
        )
    n_samples = int(len(y))
    if folds > n_samples:
        folds = n_samples
        warnings.append(
            f"The number of cross-validation folds was reduced to {folds} to match the "
            "number of rows."
        )

    seed = random_state if isinstance(random_state, (int, np.integer)) else None
    if task_type == "classification":
        smallest = int(np.bincount(y).min()) if len(y) else 0
        if smallest >= MIN_CV_FOLDS:
            if folds > smallest:
                folds = smallest
                warnings.append(
                    f"The number of cross-validation folds was reduced to {folds} because "
                    "the smallest class does not have more rows."
                )
            splitter: Any = StratifiedKFold(
                n_splits=folds, shuffle=True, random_state=seed
            )
        else:
            warnings.append(
                "Cross-validation used plain K-fold splitting because at least one class "
                "has fewer than 2 rows."
            )
            splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        scoring = "accuracy"
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        scoring = "r2"

    if folds < MIN_CV_FOLDS:
        warnings.append(
            "Cross-validation was skipped because the dataset does not have enough rows."
        )
        return None

    try:
        raw_scores = cross_val_score(
            clone(pipe), X, y, cv=splitter, scoring=scoring, error_score="raise"
        )
    except Exception as exc:  # noqa: BLE001 - report, do not fail the run
        warnings.append(f"Cross-validation could not be completed: {exc}")
        return None

    scores = [_number(score) for score in np.asarray(raw_scores, dtype=float)]
    if not scores:
        return None
    return {
        "folds": int(folds),
        "scoring": scoring,
        "scores": scores,
        "mean": _number(np.mean(scores)),
        "std": _number(np.std(scores)),
    }


def train(
    df: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: Sequence[str] | None,
    algorithm: str,
    task_type: str,
    test_size: float = 0.2,
    random_state: int | None = 42,
    hyperparameters: dict | None = None,
    stratify: bool = True,
    scale_features: bool = True,
    cv_folds: int | None = None,
) -> dict:
    """Fit and evaluate one model, returning a JSON-safe result plus an artifact."""
    warnings: list[str] = []

    if df is None or not hasattr(df, "columns"):
        raise TrainingError("No dataset was provided for training.")
    if not target_column or target_column not in df.columns:
        raise TrainingError(f"Column '{target_column}' is not in the dataset.")

    features = _resolve_feature_columns(df, target_column, feature_columns, warnings)
    if not features:
        raise TrainingError("Select at least one feature column to train on.")

    frame = df.loc[:, [*features, target_column]].copy()
    total_rows = len(frame)
    frame = frame[frame[target_column].notna()]
    dropped = total_rows - len(frame)
    if dropped:
        warnings.append(
            f"Dropped {dropped} row(s) with a missing value in the target column "
            f"'{target_column}'."
        )
    frame = frame.reset_index(drop=True)
    if len(frame) < 2:
        raise TrainingError(
            "At least 2 rows with a value in the target column are needed to train a model."
        )

    resolved_task = str(task_type or "auto").strip().lower()
    if resolved_task == "auto":
        resolved_task = infer_task(frame[target_column])
    if resolved_task not in ("classification", "regression"):
        raise TrainingError(
            f"Task type '{task_type}' is not supported. Use 'classification' or 'regression'."
        )

    label_encoder = None
    classes: list[str] | None = None
    if resolved_task == "classification":
        y, label_encoder, classes = _encode_classification_target(frame[target_column])
        if len(classes) < 2:
            raise TrainingError(
                f"The target column '{target_column}' has only one distinct value, so a "
                "classification model cannot be trained."
            )
    else:
        y = _numeric_regression_target(frame[target_column], target_column, warnings)
        finite = np.isfinite(y)
        if not bool(finite.all()):
            removed = int((~finite).sum())
            frame = frame[finite].reset_index(drop=True)
            y = y[finite]
            warnings.append(
                f"Dropped {removed} row(s) whose target value is not a finite number."
            )
            if len(frame) < 2:
                raise TrainingError(
                    "At least 2 rows with a usable target value are needed to train a model."
                )

    numeric_columns, categorical_columns = usable_feature_kinds(frame, features)
    ignored = [
        column
        for column in features
        if column not in numeric_columns and column not in categorical_columns
    ]
    if ignored:
        names = ", ".join(f"'{column}'" for column in ignored)
        warnings.append(f"Column {names} is empty and was ignored.")
    if not numeric_columns and not categorical_columns:
        raise TrainingError(
            "None of the selected feature columns contain any values, so there is "
            "nothing to learn from."
        )

    size = _finite(test_size)
    if size is None:
        size = 0.2
    if size < MIN_TEST_SIZE or size > MAX_TEST_SIZE:
        size = min(MAX_TEST_SIZE, max(MIN_TEST_SIZE, size))
        warnings.append(f"The test size was adjusted to {size:g}.")

    n_samples = len(frame)
    n_test = int(math.ceil(size * n_samples))
    n_train = n_samples - n_test
    if n_train < 1 or n_test < 1:
        raise TrainingError(
            "The test size leaves no rows for training. Use a smaller test size or a "
            "bigger dataset."
        )

    stratify_values = None
    if resolved_task == "classification" and stratify:
        counts = np.bincount(y, minlength=len(classes or []))
        smallest = int(counts.min()) if counts.size else 0
        n_classes = len(classes or [])
        if smallest < 2:
            warnings.append(
                "Stratified splitting was turned off because at least one class has "
                "fewer than 2 rows."
            )
        elif n_test < n_classes or n_train < n_classes:
            warnings.append(
                "Stratified splitting was turned off because the split is too small to "
                "keep every class on both sides."
            )
        else:
            stratify_values = y

    X = frame.loc[:, features]

    try:
        spec = get_spec(algorithm)
    except AlgorithmNotFound:
        raise TrainingError(f"Algorithm '{algorithm}' is not available.") from None

    supported = [str(item) for item in (spec.get("tasks") or [])]
    if resolved_task not in supported:
        label = str(spec.get("label") or algorithm)
        raise TrainingError(f"{label} cannot be used for {resolved_task}.")

    clean_hyperparameters, hyperparameter_warnings = coerce_hyperparameters(
        algorithm, resolved_task, dict(hyperparameters or {})
    )
    warnings.extend(str(message) for message in hyperparameter_warnings or [])

    try:
        estimator = build_estimator(
            algorithm, resolved_task, dict(clean_hyperparameters), random_state
        )
    except AlgorithmNotFound:
        raise TrainingError(f"Algorithm '{algorithm}' is not available.") from None
    except (TypeError, ValueError) as exc:
        raise TrainingError(f"The algorithm settings are not valid: {exc}") from None

    # Report what the estimator actually holds, not what the form advertises:
    # the two differ whenever a setting means something different for the other
    # task (criterion, loss) or does not exist on this estimator at all
    # (class_weight on a regressor).
    task_spec = get_spec(algorithm, resolved_task)
    estimator_params = estimator.get_params(deep=False)
    effective: dict[str, Any] = {
        str(item.get("name")): estimator_params[str(item.get("name"))]
        for item in (task_spec.get("hyperparameters") or [])
        if item.get("name") and str(item.get("name")) in estimator_params
    }
    if "random_state" in estimator_params:
        effective["random_state"] = estimator_params["random_state"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_values,
    )

    pipe = build_pipeline(frame, features, estimator, bool(scale_features))

    started = time.perf_counter()
    try:
        pipe.fit(X_train, y_train)
    except Exception as exc:  # noqa: BLE001 - surfaced as a readable message
        raise TrainingError(f"The model could not be trained: {exc}") from None

    try:
        test_predictions = pipe.predict(X_test)
        train_predictions = pipe.predict(X_train)
    except Exception as exc:  # noqa: BLE001 - surfaced as a readable message
        raise TrainingError(f"The trained model could not make predictions: {exc}") from None

    confusion: dict | None = None
    per_class: list[dict] | None = None
    residual_sample: list[dict] | None = None

    if resolved_task == "classification":
        label_codes = list(range(len(classes or [])))
        test_proba = _safe_proba(pipe, X_test)
        train_proba = _safe_proba(pipe, X_train)
        metrics = classification_metrics(
            y_test, test_predictions, test_proba, label_codes
        )
        train_metrics = classification_metrics(
            y_train, train_predictions, train_proba, label_codes
        )
        matrix = confusion_matrix(y_test, test_predictions, labels=label_codes)
        confusion = {
            "labels": list(classes or []),
            "matrix": [[int(cell) for cell in row] for row in np.asarray(matrix)],
        }
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test,
            test_predictions,
            labels=label_codes,
            average=None,
            zero_division=0,
        )
        per_class = [
            {
                "label": (classes or [])[index],
                "precision": _number(precision[index]),
                "recall": _number(recall[index]),
                "f1": _number(f1[index]),
                "support": int(support[index]),
            }
            for index in range(len(classes or []))
        ]
    else:
        metrics = regression_metrics(y_test, test_predictions)
        train_metrics = regression_metrics(y_train, train_predictions)
        residual_sample = _residual_sample(y_test, test_predictions)

    cv = None
    if cv_folds is not None:
        cv = _cross_validate(
            pipe, X, y, resolved_task, cv_folds, random_state, warnings
        )

    duration_ms = int(round((time.perf_counter() - started) * 1000))

    importance = _feature_importance(pipe, numeric_columns, categorical_columns, X_train)
    input_schema = build_input_schema(frame, features)

    artifact = ModelArtifact(
        pipeline=pipe,
        task_type=resolved_task,
        algorithm=str(algorithm),
        target_column=str(target_column),
        feature_columns=features,
        classes=classes,
        label_encoder=label_encoder,
        input_schema=input_schema,
        sklearn_version=sklearn.__version__,
    )

    result = {
        "task_type": resolved_task,
        "algorithm": str(algorithm),
        "algorithm_label": str(spec.get("label") or algorithm),
        "feature_columns": [str(column) for column in features],
        "effective_hyperparameters": json_safe(effective),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "duration_ms": int(duration_ms),
        "metrics": json_safe(metrics),
        "train_metrics": json_safe(train_metrics),
        "cv": json_safe(cv),
        "classes": None if classes is None else [str(label) for label in classes],
        "confusion_matrix": json_safe(confusion),
        "per_class": json_safe(per_class),
        "feature_importance": json_safe(importance),
        "residual_sample": json_safe(residual_sample),
        "input_schema": json_safe(input_schema),
        "warnings": [str(message) for message in warnings],
    }
    result["artifact"] = artifact
    return result

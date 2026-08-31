"""DRF serializers for Furina ML.

Two families live here:

* output serializers (``DatasetSummarySerializer`` and friends) that render the
  exact JSON shapes described in the API contract, and
* request validators (``UploadSerializer`` and friends) that turn a raw request
  body into clean Python values with plain-sentence error messages.

Every value that leaves an output serializer goes through
:func:`core.ml.io.json_safe`, so NaN/Inf and numpy scalars can never reach the
JSON encoder.
"""

from __future__ import annotations

from typing import Any

from django.urls import reverse
from rest_framework import serializers

from core.ml.io import MAX_PREVIEW_ROWS, MAX_UPLOAD_BYTES, json_safe
from core.ml.preprocessing import STEP_OPS

DEFAULT_PREVIEW_ROWS = 10

#: Delimiters the upload endpoint accepts, mapped to the character to use.
DELIMITER_CHOICES: dict[str, str] = {
    "auto": "auto",
    ",": ",",
    ";": ";",
    "\t": "\t",
    "\\t": "\t",
    "tab": "\t",
    "|": "|",
    "comma": ",",
    "semicolon": ";",
    "pipe": "|",
}

ALLOWED_UPLOAD_SUFFIXES = (".csv", ".tsv", ".txt")

TASK_TYPES = ("classification", "regression")
TRAIN_TASK_CHOICES = ("classification", "regression", "auto")


def _absolute(request: Any, name: str, **kwargs: Any) -> str:
    """Build an absolute URL for a named route, falling back to the path."""
    try:
        path = reverse(name, kwargs=kwargs)
    except Exception:  # noqa: BLE001 - a missing route must not break rendering
        return ""
    if request is None:
        return path
    return request.build_absolute_uri(path)


def _iso(value: Any) -> str | None:
    """Render a datetime as ISO-8601, or None when it is not set yet."""
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else str(value)


# --------------------------------------------------------------------------
# Output serializers
# --------------------------------------------------------------------------


class DatasetSummarySerializer(serializers.Serializer):
    """DatasetSummary: the light-weight shape used by list endpoints."""

    def to_representation(self, instance: Any) -> dict:
        return json_safe(
            {
                "id": None if instance.id is None else str(instance.id),
                "name": str(instance.name or ""),
                "source": str(instance.source or "upload"),
                "n_rows": int(instance.n_rows or 0),
                "n_cols": int(instance.n_cols or 0),
                "created_at": _iso(instance.created_at),
            }
        )


class DatasetDetailSerializer(DatasetSummarySerializer):
    """DatasetSummary plus the column profile, a preview and a download URL.

    Context keys:
      ``request``      - used to build absolute URLs.
      ``preview_rows`` - how many preview rows to render (default 10).
      ``dataframe``    - optional in-memory frame used instead of re-reading
                         the stored file (needed for dry-run previews).
    """

    def to_representation(self, instance: Any) -> dict:
        from core.ml.io import DataError, dataframe_preview

        data = super().to_representation(instance)
        request = self.context.get("request")
        preview_rows = self.context.get("preview_rows", DEFAULT_PREVIEW_ROWS)

        frame = self.context.get("dataframe")
        preview: dict = {"headers": [], "rows": []}
        if frame is None and instance.id is not None and instance.file:
            try:
                frame = instance.load_dataframe()
            except (DataError, OSError, ValueError):
                frame = None
        if frame is not None:
            try:
                preview = dataframe_preview(frame, preview_rows)
            except Exception:  # noqa: BLE001 - a preview must never break the response
                preview = {"headers": [], "rows": []}

        download_url = ""
        if instance.id is not None and instance.file:
            download_url = _absolute(request, "core:dataset-download", pk=str(instance.id))

        parent_id = getattr(instance, "parent_id", None)

        data.update(
            {
                "parent": None if parent_id is None else str(parent_id),
                "size_bytes": int(instance.size_bytes or 0),
                "columns": list(instance.columns or []),
                "preview": preview,
                "download_url": download_url,
            }
        )
        return json_safe(data)


class TrainingRunSummarySerializer(serializers.Serializer):
    """TrainingRunSummary: everything a run list row needs.

    The full `metrics` and `train_metrics` blobs are included, not just the
    headline number. They are flat `{name: float}` dicts of roughly ten keys,
    so the cost is small, and carrying them here is what lets the dashboard
    rank and compare every run from one request instead of one request per row.
    `train_metrics` is what makes the train-versus-test gap -- the only
    overfitting signal there is -- visible in a list.
    """

    def to_representation(self, instance: Any) -> dict:
        return json_safe(
            {
                "id": str(instance.id),
                "name": str(instance.name or ""),
                "dataset_id": str(instance.dataset_id),
                "dataset_name": str(getattr(instance.dataset, "name", "") or ""),
                "algorithm": str(instance.algorithm or ""),
                "algorithm_label": str(instance.algorithm_label or ""),
                "task_type": str(instance.task_type or ""),
                "target_column": str(instance.target_column or ""),
                "status": str(instance.status or "success"),
                "primary_metric_name": instance.primary_metric_name,
                "primary_metric": instance.primary_metric,
                "metrics": dict(instance.metrics or {}),
                "train_metrics": dict(instance.train_metrics or {}),
                "n_train": int(instance.n_train or 0),
                "n_test": int(instance.n_test or 0),
                "duration_ms": int(instance.duration_ms or 0),
                "cv_folds": instance.cv_folds,
                "created_at": _iso(instance.created_at),
            }
        )


class TrainingRunDetailSerializer(TrainingRunSummarySerializer):
    """TrainingRunSummary plus every metric, artefact URL and diagnostic."""

    def to_representation(self, instance: Any) -> dict:
        data = super().to_representation(instance)
        request = self.context.get("request")
        run_id = str(instance.id)

        data.update(
            {
                "feature_columns": list(instance.feature_columns or []),
                "hyperparameters": dict(instance.hyperparameters or {}),
                "effective_hyperparameters": dict(instance.effective_hyperparameters or {}),
                "test_size": float(instance.test_size or 0.0),
                "random_state": (
                    None if instance.random_state is None else int(instance.random_state)
                ),
                "stratify": bool(instance.stratify),
                "scale_features": bool(instance.scale_features),
                "n_train": int(instance.n_train or 0),
                "n_test": int(instance.n_test or 0),
                "duration_ms": int(instance.duration_ms or 0),
                "error": instance.error or None,
                "metrics": dict(instance.metrics or {}),
                "train_metrics": dict(instance.train_metrics or {}),
                "cv": instance.cv,
                "classes": instance.classes,
                "confusion_matrix": instance.confusion_matrix,
                "per_class": instance.per_class,
                "feature_importance": instance.feature_importance,
                "residual_sample": instance.residual_sample,
                "input_schema": list(instance.input_schema or []),
                "warnings": list(instance.warnings or []),
                "model_download_url": (
                    _absolute(request, "core:run-download-model", pk=run_id)
                    if instance.artifact
                    else ""
                ),
                "report_download_url": _absolute(request, "core:run-report", pk=run_id),
            }
        )
        return json_safe(data)


# --------------------------------------------------------------------------
# Request validators
# --------------------------------------------------------------------------


class UploadSerializer(serializers.Serializer):
    """Validates a multipart CSV upload."""

    file = serializers.FileField()
    name = serializers.CharField(required=False, allow_blank=True, max_length=255)
    delimiter = serializers.CharField(required=False, allow_blank=True)

    def validate_file(self, value: Any) -> Any:
        filename = (getattr(value, "name", "") or "").lower()
        if not filename.endswith(ALLOWED_UPLOAD_SUFFIXES):
            allowed = ", ".join(ALLOWED_UPLOAD_SUFFIXES)
            raise serializers.ValidationError(
                f"Only {allowed} files can be uploaded. Save your data as a CSV and try again."
            )
        size = getattr(value, "size", None)
        if size is not None and int(size) > MAX_UPLOAD_BYTES:
            limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
            raise serializers.ValidationError(
                f"The file is larger than the {limit_mb} MB upload limit. "
                "Upload a smaller file or split it into parts."
            )
        return value

    def validate_delimiter(self, value: str) -> str:
        raw = value if value is not None else ""
        if raw.strip() == "" and raw != "\t":
            return "auto"
        resolved = DELIMITER_CHOICES.get(raw) or DELIMITER_CHOICES.get(raw.strip().lower())
        if resolved is None:
            raise serializers.ValidationError(
                "Choose one of the supported column separators: comma, semicolon, tab or pipe."
            )
        return resolved


class PreprocessStepSerializer(serializers.Serializer):
    """One entry of a preprocessing recipe."""

    op = serializers.CharField()
    columns = serializers.ListField(
        child=serializers.CharField(allow_blank=False),
        required=False,
        allow_null=True,
    )
    params = serializers.DictField(required=False, allow_null=True)

    def validate_op(self, value: str) -> str:
        op = (value or "").strip()
        if op not in STEP_OPS:
            supported = ", ".join(sorted(STEP_OPS))
            raise serializers.ValidationError(
                f"'{value}' is not a preprocessing step we know. "
                f"Supported steps are: {supported}."
            )
        return op


class PreprocessRequestSerializer(serializers.Serializer):
    """Body of POST /api/datasets/<id>/preprocess/."""

    steps = serializers.ListField(child=PreprocessStepSerializer(), allow_empty=True)
    dry_run = serializers.BooleanField(required=False, default=False)
    name = serializers.CharField(
        required=False, allow_blank=True, allow_null=True, max_length=255
    )


class TrainRequestSerializer(serializers.Serializer):
    """Body of POST /api/train/."""

    dataset_id = serializers.UUIDField()
    target_column = serializers.CharField(max_length=255)
    feature_columns = serializers.ListField(
        child=serializers.CharField(allow_blank=False),
        required=False,
        allow_null=True,
        allow_empty=True,
    )
    algorithm = serializers.CharField(max_length=64)
    task_type = serializers.ChoiceField(
        choices=TRAIN_TASK_CHOICES, required=False, default="auto"
    )
    test_size = serializers.FloatField(required=False, default=0.2, min_value=0.05, max_value=0.95)
    random_state = serializers.IntegerField(
        required=False, allow_null=True, default=42, min_value=-2147483648, max_value=2147483647
    )
    hyperparameters = serializers.DictField(required=False, allow_null=True, default=dict)
    stratify = serializers.BooleanField(required=False, default=True)
    scale_features = serializers.BooleanField(required=False, default=True)
    cv_folds = serializers.IntegerField(
        required=False, allow_null=True, default=None, min_value=2, max_value=10
    )
    name = serializers.CharField(
        required=False, allow_blank=True, allow_null=True, max_length=255
    )

    def validate_target_column(self, value: str) -> str:
        target = (value or "").strip()
        if not target:
            raise serializers.ValidationError("Choose the column you want the model to predict.")
        return target


class PredictRequestSerializer(serializers.Serializer):
    """Body of POST /api/runs/<id>/predict/."""

    rows = serializers.ListField(child=serializers.DictField(), allow_empty=False)


def clamp_preview_rows(raw: Any) -> int:
    """Turn a ``?preview_rows=`` query value into a safe row count."""
    try:
        rows = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_PREVIEW_ROWS
    return max(0, min(rows, MAX_PREVIEW_ROWS))

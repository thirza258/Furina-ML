"""Database models for Furina ML.

Two tables only: an uploaded (or derived) tabular Dataset, and a TrainingRun
that records everything needed to reproduce and explain a fitted model.

`core.ml` imports are deliberately function-local so that Django can start,
and `makemigrations` can run, without pulling pandas/scikit-learn into memory.
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING, Any

from django.db import models

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas


DATASET_SOURCES = [
    ("upload", "Uploaded"),
    ("preprocessed", "Preprocessed"),
]

RUN_STATUSES = [
    ("success", "Success"),
    ("failed", "Failed"),
]


class Dataset(models.Model):
    """A tabular dataset stored on disk as a normalized UTF-8 comma CSV."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to="datasets/")
    source = models.CharField(max_length=20, choices=DATASET_SOURCES, default="upload")
    parent = models.ForeignKey(
        "self",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="children",
    )
    n_rows = models.PositiveIntegerField(default=0)
    n_cols = models.PositiveIntegerField(default=0)
    columns = models.JSONField(default=list)
    size_bytes = models.PositiveBigIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.name} ({self.n_rows}x{self.n_cols})"

    @property
    def column_names(self) -> list[str]:
        """Column names taken from the stored profile, in dataset order."""
        return [str(col.get("name")) for col in (self.columns or []) if col.get("name") is not None]

    def load_dataframe(self) -> "pandas.DataFrame":
        """Read the stored CSV back into a dataframe."""
        from core.ml.io import read_csv

        path = self.file_path()
        if path is not None:
            return read_csv(path, delimiter=",")
        with self.file.open("rb") as handle:
            return read_csv(handle, delimiter=",")

    def file_path(self) -> str | None:
        """Absolute on-disk path of the stored CSV, or None when unavailable."""
        return _existing_path(self.file)


class TrainingRun(models.Model):
    """One fit-and-evaluate cycle over a Dataset."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255, blank=True)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name="runs")
    algorithm = models.CharField(max_length=64)
    algorithm_label = models.CharField(max_length=128, blank=True)
    task_type = models.CharField(max_length=20)
    target_column = models.CharField(max_length=255)
    feature_columns = models.JSONField(default=list)
    hyperparameters = models.JSONField(default=dict)
    effective_hyperparameters = models.JSONField(default=dict)
    test_size = models.FloatField(default=0.2)
    random_state = models.IntegerField(null=True, blank=True)
    stratify = models.BooleanField(default=True)
    scale_features = models.BooleanField(default=True)
    cv_folds = models.PositiveSmallIntegerField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=RUN_STATUSES, default="success")
    error = models.TextField(blank=True, default="")
    n_train = models.PositiveIntegerField(default=0)
    n_test = models.PositiveIntegerField(default=0)
    duration_ms = models.PositiveIntegerField(default=0)
    metrics = models.JSONField(default=dict)
    train_metrics = models.JSONField(default=dict)
    cv = models.JSONField(null=True, blank=True)
    classes = models.JSONField(null=True, blank=True)
    confusion_matrix = models.JSONField(null=True, blank=True)
    per_class = models.JSONField(null=True, blank=True)
    feature_importance = models.JSONField(null=True, blank=True)
    residual_sample = models.JSONField(null=True, blank=True)
    input_schema = models.JSONField(default=list)
    warnings = models.JSONField(default=list)
    artifact = models.FileField(upload_to="models/", null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.display_name} [{self.algorithm}]"

    @property
    def display_name(self) -> str:
        """Human label for the run, falling back to algorithm + target."""
        if self.name:
            return self.name
        return f"{self.algorithm_label or self.algorithm} on {self.target_column}"

    @property
    def primary_metric_name(self) -> str:
        """`accuracy` for classification runs, `r2` for regression runs."""
        return "r2" if self.task_type == "regression" else "accuracy"

    @property
    def primary_metric(self) -> float | None:
        """Value of the primary metric, or None when it is missing/not finite."""
        value = (self.metrics or {}).get(self.primary_metric_name)
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if number != number or number in (float("inf"), float("-inf")):
            return None
        return number

    def artifact_path(self) -> str | None:
        """Absolute on-disk path of the saved model, or None when unavailable."""
        return _existing_path(self.artifact)

    def load_artifact(self) -> Any:
        """Deserialize the saved model artifact."""
        from core.ml.trainer import load_artifact

        if not self.artifact:
            raise FileNotFoundError("This training run has no saved model file.")
        path = self.artifact_path()
        if path is not None:
            return load_artifact(path)
        with self.artifact.open("rb") as handle:
            return load_artifact(handle.read())


def _existing_path(field_file) -> str | None:
    """Return the storage path of a FileField value when it exists on disk."""
    if not field_file:
        return None
    try:
        path = field_file.path
    except (NotImplementedError, ValueError, AttributeError):
        return None
    return path if os.path.exists(path) else None

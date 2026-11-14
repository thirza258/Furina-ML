"""Experiment-lifecycle models for Furina ML.

The domain is a straight line:

    Dataset -> DatasetVersion -> Experiment -> ExperimentRun ->
    Model/ModelVersion -> Deployment -> MonitoringRecord

Everything here is metadata. The heavy lifting (CSV processing, fitting,
cross-validation, search) lives in ``core.ml`` and runs in the job worker,
not in an HTTP request. Two compatibility seams keep history safe:

* ``ExperimentRun.legacy_run`` points back at a pre-lifecycle
  ``core.TrainingRun`` so a migrated run keeps its old id and its old
  endpoints work unchanged;
* ``ModelVersion.legacy_run`` does the same for a model registered from a
  legacy run.

Model classes never import pandas or scikit-learn; a few ``core.ml`` imports
are function-local for the same reason ``core.models`` uses them, so
``makemigrations`` and a bare Django shell start without the ML stack.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from django.conf import settings
from django.db import models
from django.utils import timezone

#: How a version came to be. "derived" is a version built from another version
#: (feature engineering); "preprocessed" is a saved cleaning recipe.
VERSION_SOURCES = [
    ("upload", "Upload"),
    ("preprocessed", "Preprocessed"),
    ("derived", "Derived"),
]

RUN_STATUSES = [
    ("queued", "Queued"),
    ("running", "Running"),
    ("success", "Success"),
    ("failed", "Failed"),
    ("cancelled", "Cancelled"),
]

JOB_STATUSES = [
    ("queued", "Queued"),
    ("running", "Running"),
    ("completed", "Completed"),
    ("failed", "Failed"),
    ("cancelled", "Cancelled"),
]

JOB_KINDS = [
    ("train", "Train a single model"),
    ("hpo", "Hyperparameter search"),
    ("automl", "AutoML"),
    ("reproduce", "Reproduce a run"),
]

#: The registry's one-way lifecycle. A version walks down this list; the
#: order in the list *is* the only permitted direction.
MODEL_STATES = [
    ("candidate", "Candidate"),
    ("validated", "Validated"),
    ("staging", "Staging"),
    ("production", "Production"),
    ("archived", "Archived"),
]
_MODEL_STATE_INDEX = {state: index for index, (state, _label) in enumerate(MODEL_STATES)}

MONITORING_KINDS = [
    ("request", "Inference request"),
    ("drift", "Distribution drift"),
    ("performance", "Labelled performance"),
]


def content_hash_of(payload: bytes) -> str:
    """SHA-256 of raw file bytes -- the version's identity."""
    return hashlib.sha256(payload).hexdigest()


def schema_hash_of(columns: Any) -> str:
    """SHA-256 of a canonical column description.

    Two files that hold the same columns (name, dtype, kind) hash equal even
    when their rows differ, which is the "same shape" signal a drift or
    reproduction check wants. The canonical form sorts the columns, so the
    hash is independent of column order in the file.
    """
    canonical = [
        {
            "name": col.get("name"),
            "dtype": col.get("dtype"),
            "kind": col.get("kind"),
        }
        for col in (columns or [])
        if isinstance(col, dict)
    ]
    canonical.sort(key=lambda item: str(item.get("name")))
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def config_hash_of(config: dict[str, Any]) -> str:
    """SHA-256 of a training configuration, canonicalised for stable diffs."""
    blob = json.dumps(config or {}, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DatasetVersion(models.Model):
    """An immutable snapshot of a Dataset, with lineage to its parent version."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    dataset = models.ForeignKey(
        "core.Dataset", on_delete=models.CASCADE, related_name="versions"
    )
    version = models.PositiveSmallIntegerField()
    parent = models.ForeignKey(
        "self",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="children",
    )
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, default="")
    source_type = models.CharField(max_length=20, choices=VERSION_SOURCES, default="upload")
    file = models.FileField(upload_to="datasets/versions/", null=True, blank=True)
    content_hash = models.CharField(max_length=64, db_index=True)
    schema_hash = models.CharField(max_length=64, db_index=True)
    row_count = models.PositiveIntegerField(default=0)
    column_count = models.PositiveIntegerField(default=0)
    recipe = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="dataset_versions",
    )

    class Meta:
        ordering = ["dataset", "-version"]
        constraints = [
            models.UniqueConstraint(fields=["dataset", "version"], name="unique_dataset_version"),
        ]

    def __str__(self) -> str:
        return f"{self.dataset.name} v{self.version}"

    def load_dataframe(self) -> Any:
        """Read this version's stored CSV back, exactly as a run sees it."""
        from core.ml.io import read_csv

        dataset = self.dataset
        if self.file:
            import os

            path = getattr(self.file, "path", None)
            if path and os.path.exists(path):
                return read_csv(path, delimiter=",")
            with self.file.open("rb") as handle:
                return read_csv(handle, delimiter=",")
        # A version whose own file vanished still falls back to the dataset's.
        return dataset.load_dataframe()

    @property
    def lineage(self) -> list["DatasetVersion"]:
        """This version and every ancestor, oldest first."""
        chain: list[DatasetVersion] = [self]
        current = self.parent
        seen = {self.pk}
        while current and current.pk not in seen:
            chain.append(current)
            seen.add(current.pk)
            current = current.parent
        return list(reversed(chain))

    def duplicate_of(self) -> "DatasetVersion | None":
        """An earlier version of the same dataset with the same content, or None."""
        qs = DatasetVersion.objects.filter(
            dataset=self.dataset, content_hash=self.content_hash
        ).exclude(pk=self.pk)
        qs = qs.order_by("version")
        return qs.first()


class Experiment(models.Model):
    """A named set of runs over one dataset version, with one objective."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    dataset_version = models.ForeignKey(
        DatasetVersion, on_delete=models.PROTECT, related_name="experiments"
    )
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, default="")
    task_type = models.CharField(max_length=20, default="classification")
    objective_metric = models.CharField(max_length=64, default="accuracy")
    objective_direction = models.CharField(max_length=10, default="maximize")
    tags = models.JSONField(default=list)
    best_run = models.ForeignKey(
        "ExperimentRun",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="+",
    )
    best_metric_value = models.FloatField(null=True, blank=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="experiments",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.name

    def objective_value(self, run: "ExperimentRun") -> float | None:
        """This run's score on the experiment's objective, or None."""
        metrics = run.metrics or {}
        value = metrics.get(self.objective_metric)
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if number == number and number not in (float("inf"), float("-inf")) else None

    def consider_run(self, run: "ExperimentRun") -> bool:
        """Adopt a finished run as the experiment's best, keeping the record current."""
        if run.status != "success":
            return False
        value = self.objective_value(run)
        if value is None:
            return False
        best = self.best_run
        if best is None or self.best_metric_value is None:
            self.best_run, self.best_metric_value = run, value
            self.save(update_fields=["best_run", "best_metric_value"])
            return True
        better = value > self.best_metric_value if self.objective_direction == "maximize" else value < self.best_metric_value
        if better:
            self.best_run, self.best_metric_value = run, value
            self.save(update_fields=["best_run", "best_metric_value"])
            return True
        return False


class ExperimentRun(models.Model):
    """One fit-and-evaluate cycle, fully described so it can be reproduced."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    experiment = models.ForeignKey(
        Experiment, on_delete=models.CASCADE, related_name="runs"
    )
    dataset_version = models.ForeignKey(
        DatasetVersion, on_delete=models.PROTECT, related_name="runs"
    )
    legacy_run = models.ForeignKey(
        "core.TrainingRun",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="lifecycle_runs",
    )
    job = models.ForeignKey(
        "Job", null=True, blank=True, on_delete=models.SET_NULL, related_name="runs"
    )
    number = models.PositiveIntegerField(default=1)
    trial_index = models.PositiveIntegerField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=RUN_STATUSES, default="queued")
    error = models.TextField(blank=True, default="")

    # -- the dataset as this run saw it -----------------------------------
    dataset_hash = models.CharField(max_length=64, blank=True, default="")
    dataset_row_count = models.PositiveIntegerField(default=0)
    dataset_feature_count = models.PositiveIntegerField(default=0)

    # -- task & model -------------------------------------------------------
    algorithm = models.CharField(max_length=64)
    algorithm_label = models.CharField(max_length=128, blank=True, default="")
    task_type = models.CharField(max_length=20, default="classification")
    target_column = models.CharField(max_length=255)
    feature_columns = models.JSONField(default=list)

    # -- split & preprocessing --------------------------------------------
    test_size = models.FloatField(default=0.2)
    random_state = models.IntegerField(null=True, blank=True)
    stratify = models.BooleanField(default=True)
    cv_folds = models.PositiveSmallIntegerField(null=True, blank=True)
    scale_features = models.BooleanField(default=True)
    preprocessing = models.JSONField(default=dict, blank=True)

    # -- the model's own configuration ---------------------------------------
    hyperparameters = models.JSONField(default=dict, blank=True)
    effective_hyperparameters = models.JSONField(default=dict, blank=True)
    config_hash = models.CharField(max_length=64, blank=True, default="")

    # -- environment the run executed under ---------------------------------
    environment = models.JSONField(null=True, blank=True)

    # -- results ------------------------------------------------------------
    metrics = models.JSONField(default=dict, blank=True)
    train_metrics = models.JSONField(default=dict, blank=True)
    diagnostics = models.JSONField(null=True, blank=True)
    cv = models.JSONField(null=True, blank=True)
    classes = models.JSONField(null=True, blank=True)
    confusion_matrix = models.JSONField(null=True, blank=True)
    per_class = models.JSONField(null=True, blank=True)
    feature_importance = models.JSONField(null=True, blank=True)
    residual_sample = models.JSONField(null=True, blank=True)
    input_schema = models.JSONField(default=list, blank=True)
    warnings = models.JSONField(default=list, blank=True)
    n_train = models.PositiveIntegerField(default=0)
    n_test = models.PositiveIntegerField(default=0)
    duration_ms = models.PositiveIntegerField(default=0)
    model_size_bytes = models.PositiveBigIntegerField(default=0)
    artifact = models.FileField(upload_to="models/lifecycle/", null=True, blank=True)
    evaluation_report = models.FileField(upload_to="reports/", null=True, blank=True)

    created_at = models.DateTimeField(default=timezone.now)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["experiment", "-created_at"]),
            models.Index(fields=["status"]),
        ]

    def __str__(self) -> str:
        label = f"run #{self.number}"
        if self.trial_index is not None:
            label = f"trial {self.trial_index + 1}"
        return f"{self.experiment.name} {label} [{self.algorithm}]"

    @property
    def objective_metric(self) -> str:
        """The metric this experiment optimises; the run ranks by it."""
        return self.experiment.objective_metric

    @property
    def primary_metric_name(self) -> str:
        return "r2" if self.task_type == "regression" else "accuracy"

    def primary_metric(self) -> float | None:
        value = (self.metrics or {}).get(self.objective_metric)
        if value is None:
            value = (self.metrics or {}).get(self.primary_metric_name)
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if number == number and number not in (float("inf"), float("-inf")) else None

    @property
    def display_name(self) -> str:
        return str(self)

    def full_config(self) -> dict[str, Any]:
        """The whole reproducible configuration, for hashing and 'reproduce'."""
        return {
            "dataset_version": str(self.dataset_version_id),
            "dataset_hash": self.dataset_hash,
            "target_column": self.target_column,
            "feature_columns": list(self.feature_columns or []),
            "algorithm": self.algorithm,
            "task_type": self.task_type,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "stratify": self.stratify,
            "cv_folds": self.cv_folds,
            "scale_features": self.scale_features,
            "preprocessing": self.preprocessing or {},
            "hyperparameters": self.hyperparameters or {},
        }


class Job(models.Model):
    """An asynchronous unit of work: one training, search or AutoML job."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    kind = models.CharField(max_length=20, choices=JOB_KINDS)
    status = models.CharField(max_length=20, choices=JOB_STATUSES, default="queued")
    payload = models.JSONField(default=dict, blank=True)
    progress = models.FloatField(default=0.0)
    total_units = models.PositiveIntegerField(default=0)
    completed_units = models.PositiveIntegerField(default=0)
    experiment = models.ForeignKey(
        Experiment, null=True, blank=True, on_delete=models.SET_NULL, related_name="jobs"
    )
    best_run = models.ForeignKey(
        ExperimentRun, null=True, blank=True, on_delete=models.SET_NULL, related_name="+"
    )
    result_model_version = models.ForeignKey(
        "ModelVersion", null=True, blank=True, on_delete=models.SET_NULL, related_name="+"
    )
    cancel_requested = models.BooleanField(default=False)
    error = models.TextField(blank=True, default="")
    suggested_action = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    last_heartbeat = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["status"])]

    def __str__(self) -> str:
        return f"{self.kind}:{self.id}"

    @property
    def fraction_done(self) -> float:
        if self.total_units <= 0:
            return 1.0 if self.status in ("completed", "failed", "cancelled") else 0.0
        return min(1.0, self.completed_units / self.total_units)

    def mark_progress(self, completed: int, total: int, note_progress: float | None = None) -> None:
        """Update the counters and a 0-100 progress number for the UI."""
        self.total_units = total
        self.completed_units = completed
        if note_progress is not None:
            self.progress = float(min(100.0, max(0.0, note_progress)))
        else:
            self.progress = 100.0 * self.fraction_done
        self.save(update_fields=["total_units", "completed_units", "progress"])

    def request_cancel(self) -> bool:
        """Ask the worker to stop; returns True when the request was new."""
        if self.status not in ("queued", "running"):
            return False
        if not self.cancel_requested:
            self.cancel_requested = True
            self.save(update_fields=["cancel_requested"])
            return True
        return False

    def is_cancel_requested(self) -> bool:
        """Freshly re-read the flag so a cancel lands even across processes."""
        fresh = Job.objects.filter(pk=self.pk, cancel_requested=True).first()
        return bool(fresh)


class Model(models.Model):
    """A registered model: a name with a versioned history and one production pointer."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    task_type = models.CharField(max_length=20, default="classification")
    description = models.TextField(blank=True, default="")
    production_version = models.ForeignKey(
        "ModelVersion",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="+",
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="models",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [
            models.UniqueConstraint(fields=["name", "task_type"], name="unique_model_name_task"),
        ]

    def __str__(self) -> str:
        return self.name

    def next_version_number(self) -> int:
        latest = self.versions.order_by("-version").first()
        return (latest.version + 1) if latest else 1


class ModelVersion(models.Model):
    """One immutable, registerable artefact tied to the run that produced it."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    model = models.ForeignKey(Model, on_delete=models.CASCADE, related_name="versions")
    version = models.PositiveIntegerField()
    source_run = models.ForeignKey(
        ExperimentRun, null=True, blank=True, on_delete=models.SET_NULL, related_name="model_versions"
    )
    legacy_run = models.ForeignKey(
        "core.TrainingRun", null=True, blank=True, on_delete=models.SET_NULL, related_name="model_versions"
    )
    dataset_version = models.ForeignKey(
        DatasetVersion, null=True, blank=True, on_delete=models.SET_NULL, related_name="model_versions"
    )
    status = models.CharField(max_length=20, choices=MODEL_STATES, default="candidate")
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="model_versions",
    )
    signature = models.JSONField(default=dict, blank=True)
    metrics = models.JSONField(default=dict, blank=True)
    model_card = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["model", "-version"]
        constraints = [
            models.UniqueConstraint(fields=["model", "version"], name="unique_model_version"),
        ]

    def __str__(self) -> str:
        return f"{self.model.name} v{self.version} ({self.status})"

    @property
    def state_index(self) -> int:
        return _MODEL_STATE_INDEX.get(self.status, 0)

    def can_promote_to(self, target: str) -> tuple[bool, str]:
        """Whether this version may move to ``target``, and why not otherwise."""
        if target not in _MODEL_STATE_INDEX:
            return False, f"'{target}' is not a model state."
        if target == self.status:
            return False, f"Already in the '{self.get_status_display()}' state."
        forward = _MODEL_STATE_INDEX[target] > self.state_index
        if forward:
            return True, ""
        # The one allowed backwards move is archiving.
        if target == "archived":
            return True, ""
        return False, f"Model states only move forward (or to archived), not back to '{target}'."

    def load_artifact(self) -> Any:
        """The fitted pipeline that backs this version.

        A run's own lifecycle artefact is the first place to look (new-style
        runs dump their pipeline directly); the legacy twin is the fallback
        for runs registered from the old workbench, and finally a version
        registered straight from a legacy run.
        """
        from core.ml.trainer import load_artifact

        if self.source_run_id:
            run = ExperimentRun.objects.filter(pk=self.source_run_id).first()
            if run is not None:
                artifact = load_run_artifact(run)
                if artifact is not None:
                    return artifact
        if self.legacy_run_id:
            legacy = _legacy_run(self.legacy_run_id)
            if legacy is not None:
                artifact = _read_legacy_artifact(legacy)
                if artifact is not None:
                    return artifact
        raise FileNotFoundError(
            "This model version has no downloadable model artefact on this server."
        )


def load_run_artifact(run: ExperimentRun) -> Any:
    """The pipeline stored on a lifecycle run, or None when there is none."""
    from core.ml.trainer import load_artifact

    if run.artifact:
        path = _existing_path(run.artifact)
        if path is not None:
            return load_artifact(path)
        try:
            with run.artifact.open("rb") as handle:
                return load_artifact(handle.read())
        except (OSError, ValueError):
            pass
    if run.legacy_run_id:
        legacy = _legacy_run(run.legacy_run_id)
        if legacy is not None:
            return _read_legacy_artifact(legacy)
    return None


def _existing_path(field_file: Any) -> str | None:
    """On-disk path of a stored file, or None when it cannot be resolved."""
    import os

    if not field_file:
        return None
    try:
        path = field_file.path
    except (NotImplementedError, ValueError, AttributeError):
        return None
    return path if os.path.exists(path) else None


def _legacy_run(pk: Any) -> Any:
    from core.models import TrainingRun

    return TrainingRun.objects.filter(pk=pk).first()


def _read_legacy_artifact(legacy: Any) -> Any:
    from core.ml.trainer import load_artifact

    if legacy is None or not legacy.artifact:
        return None
    path = legacy.artifact_path()
    if path is not None:
        return load_artifact(path)
    try:
        with legacy.artifact.open("rb") as handle:
            return load_artifact(handle.read())
    except (OSError, ValueError):
        return None

    def artifact_path(self) -> str | None:
        """Best-effort on-disk path of the backing artefact, for downloads."""
        if self.legacy_run_id:
            from core.models import TrainingRun

            legacy = TrainingRun.objects.filter(pk=self.legacy_run_id).first()
            if legacy is not None:
                return legacy.artifact_path()
        return None


class Deployment(models.Model):
    """A live inference API for one model version."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    model_version = models.ForeignKey(
        ModelVersion, on_delete=models.PROTECT, related_name="deployments"
    )
    slug = models.SlugField(max_length=100, unique=True)
    is_active = models.BooleanField(default=True)
    #: Feature-distribution statistics recorded from the training data at
    #: deploy time; drift checks compare live inputs against these.
    baseline_stats = models.JSONField(default=dict, blank=True)
    total_requests = models.PositiveBigIntegerField(default=0)
    total_errors = models.PositiveBigIntegerField(default=0)
    last_predicted_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.slug} (v{self.model_version.version})"


class MonitoringRecord(models.Model):
    """One monitoring observation: a request, a drift check, or labelled performance."""

    deployment = models.ForeignKey(
        Deployment, on_delete=models.CASCADE, related_name="monitoring"
    )
    kind = models.CharField(max_length=20, choices=MONITORING_KINDS)
    payload = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["deployment", "kind", "-created_at"])]

    def __str__(self) -> str:
        return f"{self.kind} on {self.deployment.slug}"

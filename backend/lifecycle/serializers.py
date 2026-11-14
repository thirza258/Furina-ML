"""DRF serializers for the lifecycle API.

Two families, matching ``core/serializers.py``:

* output serializers that render the JSON shapes the frontend expects, and
* request validators that turn a raw body into clean values with
  plain-sentence error messages.

Every value that leaves an output serializer goes through
:func:`core.ml.io.json_safe`, so a numpy array that slipped into a metric
can never reach the JSON encoder.
"""

from __future__ import annotations

from typing import Any

from django.urls import reverse
from rest_framework import serializers

from core.ml.io import json_safe


def _absolute(request: Any, name: str, **kwargs: Any) -> str:
    try:
        path = reverse(name, kwargs=kwargs)
    except Exception:  # noqa: BLE001 - a missing route must not break rendering
        return ""
    if request is None:
        return path
    return request.build_absolute_uri(path)


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else str(value)


# --------------------------------------------------------------------------
# dataset versions
# --------------------------------------------------------------------------


class DatasetVersionSerializer(serializers.Serializer):
    """A dataset version: the immutable snapshot a run trains on."""

    def to_representation(self, instance: Any) -> dict:
        dataset = instance.dataset
        parent = instance.parent
        return json_safe(
            {
                "id": str(instance.id),
                "dataset_id": str(instance.dataset_id),
                "dataset_name": str(dataset.name or ""),
                "version": int(instance.version),
                "parent": None if parent is None else str(parent.id),
                "name": str(instance.name or ""),
                "description": str(instance.description or ""),
                "source_type": str(instance.source_type or "derived"),
                "content_hash": str(instance.content_hash or ""),
                "schema_hash": str(instance.schema_hash or ""),
                "row_count": int(instance.row_count or 0),
                "column_count": int(instance.column_count or 0),
                "recipe": instance.recipe,
                "created_at": _iso(instance.created_at),
            }
        )


class DatasetVersionLineageSerializer(serializers.Serializer):
    """A version plus the chain of versions that produced it, oldest first."""

    def to_representation(self, instance: Any) -> dict:
        chain = [
            {
                "id": str(item.id),
                "version": int(item.version),
                "name": str(item.name or ""),
                "source_type": str(item.source_type or ""),
                "created_at": _iso(item.created_at),
            }
            for item in instance.lineage
        ]
        return json_safe({"version": DatasetVersionSerializer(instance).data, "lineage": chain})


# --------------------------------------------------------------------------
# experiments & runs
# --------------------------------------------------------------------------


class ExperimentSerializer(serializers.Serializer):
    """An experiment: one dataset version, one objective, many runs."""

    def to_representation(self, instance: Any) -> dict:
        best = instance.best_run
        return json_safe(
            {
                "id": str(instance.id),
                "dataset_version_id": str(instance.dataset_version_id),
                "name": str(instance.name or ""),
                "description": str(instance.description or ""),
                "task_type": str(instance.task_type or ""),
                "objective_metric": str(instance.objective_metric or ""),
                "objective_direction": str(instance.objective_direction or ""),
                "tags": list(instance.tags or []),
                "best_run_id": None if best is None else str(best.id),
                "best_metric_value": instance.best_metric_value,
                "n_runs": instance.runs.count(),
                "created_at": _iso(instance.created_at),
            }
        )


class ExperimentDetailSerializer(ExperimentSerializer):
    """The experiment plus its dataset version's identity for the page header."""

    def to_representation(self, instance: Any) -> dict:
        data = super().to_representation(instance)
        version = instance.dataset_version
        dataset = version.dataset
        data.update(
            {
                "dataset_version": DatasetVersionSerializer(version).data,
                "dataset_name": str(dataset.name or ""),
                "objective_choices": _objective_choices(instance.task_type),
            }
        )
        return data


def _objective_choices(task_type: str) -> list[dict]:
    from lifecycle.services import objective_metric_choices

    return json_safe(objective_metric_choices(task_type))


class RunSerializer(serializers.Serializer):
    """A run's list row: enough to rank and compare without a detail call."""

    def to_representation(self, instance: Any) -> dict:
        return json_safe(
            {
                "id": str(instance.id),
                "experiment_id": str(instance.experiment_id),
                "experiment_name": str(instance.experiment.name or ""),
                "number": int(instance.number or 1),
                "trial_index": instance.trial_index,
                "algorithm": str(instance.algorithm or ""),
                "algorithm_label": str(instance.algorithm_label or ""),
                "task_type": str(instance.task_type or ""),
                "target_column": str(instance.target_column or ""),
                "status": str(instance.status or "queued"),
                "objective_metric": str(instance.objective_metric),
                "primary_metric": instance.primary_metric(),
                "metrics": dict(instance.metrics or {}),
                "train_metrics": dict(instance.train_metrics or {}),
                "duration_ms": int(instance.duration_ms or 0),
                "cv_folds": instance.cv_folds,
                "config_hash": str(instance.config_hash or ""),
                "n_train": int(instance.n_train or 0),
                "n_test": int(instance.n_test or 0),
                "created_at": _iso(instance.created_at),
            }
        )


class RunDetailSerializer(RunSerializer):
    """Everything a run-detail page renders, plus the reproduce affordance."""

    def to_representation(self, instance: Any) -> dict:
        data = super().to_representation(instance)
        request = self.context.get("request")
        run_id = str(instance.id)
        version = instance.dataset_version
        data.update(
            {
                "dataset_version_id": str(version.id),
                "dataset_hash": str(instance.dataset_hash or ""),
                "dataset_row_count": int(instance.dataset_row_count or 0),
                "feature_columns": list(instance.feature_columns or []),
                "hyperparameters": dict(instance.hyperparameters or {}),
                "effective_hyperparameters": dict(instance.effective_hyperparameters or {}),
                "test_size": float(instance.test_size or 0.0),
                "random_state": instance.random_state,
                "stratify": bool(instance.stratify),
                "scale_features": bool(instance.scale_features),
                "preprocessing": dict(instance.preprocessing or {}),
                "environment": instance.environment,
                "cv": instance.cv,
                "classes": instance.classes,
                "confusion_matrix": instance.confusion_matrix,
                "per_class": instance.per_class,
                "feature_importance": instance.feature_importance,
                "diagnostics": instance.diagnostics,
                "residual_sample": instance.residual_sample,
                "input_schema": list(instance.input_schema or []),
                "warnings": list(instance.warnings or []),
                "error": instance.error or None,
                "model_size_bytes": int(instance.model_size_bytes or 0),
                "model_download_url": (
                    _absolute(request, "lifecycle:run-download-model", pk=run_id)
                    if instance.artifact
                    else ""
                ),
                "reproduce_url": _absolute(request, "lifecycle:run-reproduce", pk=run_id),
            }
        )
        return data


# --------------------------------------------------------------------------
# run comparison
# --------------------------------------------------------------------------


def compare_runs(runs: list[Any]) -> dict:
    """A side-by-side table across runs, refusing to mix task types.

    The PRD's rule: never compare incompatible tasks on a shared objective.
    This groups the given runs by task type and returns one table per group,
    so the UI can render them separately with their own objective.
    """
    groups: dict[str, list[Any]] = {}
    for run in runs:
        groups.setdefault(str(run.task_type or ""), []).append(run)

    from core.ml.report import METRIC_LABELS

    from lifecycle.services import OBJECTIVE_METRICS

    tables: dict[str, Any] = {}
    for task, members in groups.items():
        available = [m["key"] for m in OBJECTIVE_METRICS.get(task, [])]
        metric_keys = available or list(members[0].metrics or {})
        rows = []
        for run in members:
            row = {
                "run_id": str(run.id),
                "label": run.display_name,
                "algorithm": run.algorithm,
                "status": run.status,
                "duration_ms": int(run.duration_ms or 0),
            }
            for key in metric_keys:
                row[f"metric:{key}"] = (run.metrics or {}).get(key)
            rows.append(row)
        tables[task] = {
            "task_type": task,
            "metrics": [
                {"key": key, "label": METRIC_LABELS.get(key, key)} for key in metric_keys
            ],
            "objective_metric": members[0].experiment.objective_metric,
            "objective_direction": members[0].experiment.objective_direction,
            "rows": rows,
        }
    return json_safe({"task_types": sorted(groups), "tables": tables})


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------


class JobSerializer(serializers.Serializer):
    """A job's state for the progress UI."""

    def to_representation(self, instance: Any) -> dict:
        best = instance.best_run
        return json_safe(
            {
                "id": str(instance.id),
                "kind": str(instance.kind or ""),
                "status": str(instance.status or "queued"),
                "progress": float(instance.progress or 0.0),
                "total_units": int(instance.total_units or 0),
                "completed_units": int(instance.completed_units or 0),
                "cancel_requested": bool(instance.cancel_requested),
                "error": instance.error or None,
                "suggested_action": instance.suggested_action or None,
                "best_run_id": None if best is None else str(best.id),
                "payload": instance.payload,
                "created_at": _iso(instance.created_at),
                "started_at": _iso(instance.started_at),
                "finished_at": _iso(instance.finished_at),
            }
        )


# --------------------------------------------------------------------------
# model registry
# --------------------------------------------------------------------------


class ModelSerializer(serializers.Serializer):
    """A registry model: the name plus its version history."""

    def to_representation(self, instance: Any) -> dict:
        production = instance.production_version
        return json_safe(
            {
                "id": str(instance.id),
                "name": str(instance.name or ""),
                "task_type": str(instance.task_type or ""),
                "description": str(instance.description or ""),
                "production_version": None if production is None else int(production.version),
                "n_versions": instance.versions.count(),
                "created_at": _iso(instance.created_at),
            }
        )


class ModelVersionSerializer(serializers.Serializer):
    """One registry version and where it sits in the lifecycle."""

    def to_representation(self, instance: Any) -> dict:
        return json_safe(
            {
                "id": str(instance.id),
                "model_id": str(instance.model_id),
                "model_name": str(instance.model.name or ""),
                "version": int(instance.version),
                "status": str(instance.status or "candidate"),
                "status_label": instance.get_status_display(),
                "source_run_id": None if instance.source_run_id is None else str(instance.source_run_id),
                "dataset_version_id": (
                    None if instance.dataset_version_id is None else str(instance.dataset_version_id)
                ),
                "metrics": dict(instance.metrics or {}),
                "model_card": instance.model_card,
                "signature": instance.signature,
                "created_at": _iso(instance.created_at),
                "states": [state for state, _label in _model_states()],
            }
        )


def _model_states() -> list[tuple[str, str]]:
    from lifecycle.models import MODEL_STATES

    return [item for item in MODEL_STATES]


# --------------------------------------------------------------------------
# deployments & monitoring
# --------------------------------------------------------------------------


class DeploymentSerializer(serializers.Serializer):
    """A live inference endpoint for one model version."""

    def to_representation(self, instance: Any) -> dict:
        request = self.context.get("request")
        version = instance.model_version
        return json_safe(
            {
                "id": str(instance.id),
                "slug": str(instance.slug or ""),
                "model_version_id": str(instance.model_version_id),
                "model_name": str(version.model.name or ""),
                "model_version": int(version.version),
                "status": str(version.status or "candidate"),
                "is_active": bool(instance.is_active),
                "total_requests": int(instance.total_requests or 0),
                "total_errors": int(instance.total_errors or 0),
                "last_predicted_at": _iso(instance.last_predicted_at),
                "signature": instance.model_version.signature,
                "predict_url": _absolute(request, "lifecycle:deployment-predict", slug=instance.slug),
                "batch_predict_url": _absolute(request, "lifecycle:deployment-batch-predict", slug=instance.slug),
                "openapi_url": _absolute(request, "lifecycle:deployment-openapi", slug=instance.slug),
                "client_snippets_url": _absolute(request, "lifecycle:deployment-snippets", slug=instance.slug),
                "created_at": _iso(instance.created_at),
            }
        )


class MonitoringSummarySerializer(serializers.Serializer):
    """The monitoring dashboard's at-a-glance numbers for one deployment."""

    def to_representation(self, instance: Any) -> dict:
        deployment = instance
        drift = deployment.monitoring.filter(kind="drift").order_by("-created_at")
        performance = deployment.monitoring.filter(kind="performance").order_by("-created_at")
        return json_safe(
            {
                "deployment": DeploymentSerializer(deployment, context=self.context).data,
                "total_requests": int(deployment.total_requests or 0),
                "total_errors": int(deployment.total_errors or 0),
                "error_rate": (
                    float(deployment.total_errors) / float(deployment.total_requests)
                    if deployment.total_requests
                    else 0.0
                ),
                "latest_drift": drift.first().payload if drift.exists() else None,
                "latest_performance": performance.first().payload if performance.exists() else None,
            }
        )


# --------------------------------------------------------------------------
# request validators
# --------------------------------------------------------------------------


class CreateExperimentSerializer(serializers.Serializer):
    """Body of POST /api/lifecycle/experiments/."""

    dataset_id = serializers.UUIDField()
    dataset_version_id = serializers.UUIDField(required=False, allow_null=True)
    name = serializers.CharField(max_length=255)
    description = serializers.CharField(required=False, allow_blank=True, max_length=2000)
    task_type = serializers.ChoiceField(
        choices=("classification", "regression", "auto"), required=False, default="auto"
    )
    objective_metric = serializers.CharField(required=False, allow_blank=True, max_length=64)
    objective_direction = serializers.ChoiceField(
        choices=("maximize", "minimize"), required=False, default="maximize"
    )
    tags = serializers.ListField(
        child=serializers.CharField(max_length=64), required=False, default=list
    )

    def validate_name(self, value: str) -> str:
        name = (value or "").strip()
        if not name:
            raise serializers.ValidationError("Give the experiment a name.")
        return name


class TrainJobSerializer(serializers.Serializer):
    """Body of POST /api/lifecycle/experiments/<id>/train/."""

    dataset_version_id = serializers.UUIDField(required=False, allow_null=True)
    target_column = serializers.CharField(max_length=255)
    feature_columns = serializers.ListField(
        child=serializers.CharField(allow_blank=False), required=False, allow_null=True
    )
    algorithm = serializers.CharField(max_length=64, default="logistic_regression")
    task_type = serializers.ChoiceField(
        choices=("classification", "regression", "auto"), required=False, default="auto"
    )
    test_size = serializers.FloatField(required=False, default=0.2, min_value=0.05, max_value=0.95)
    random_state = serializers.IntegerField(
        required=False, allow_null=True, default=42, min_value=-2147483648, max_value=2147483647
    )
    hyperparameters = serializers.DictField(required=False, allow_null=True, default=dict)
    stratify = serializers.BooleanField(required=False, default=True)
    cv_folds = serializers.IntegerField(
        required=False, allow_null=True, default=None, min_value=2, max_value=10
    )
    name = serializers.CharField(required=False, allow_blank=True, allow_null=True, max_length=255)
    run_now = serializers.BooleanField(required=False, default=False)

    def validate_target_column(self, value: str) -> str:
        target = (value or "").strip()
        if not target:
            raise serializers.ValidationError("Choose the column the model should predict.")
        return target


class HpoJobSerializer(serializers.Serializer):
    """Body of POST /api/lifecycle/experiments/<id>/search/."""

    dataset_version_id = serializers.UUIDField(required=False, allow_null=True)
    target_column = serializers.CharField(max_length=255)
    feature_columns = serializers.ListField(
        child=serializers.CharField(allow_blank=False), required=False, allow_null=True
    )
    algorithm = serializers.CharField(max_length=64, required=False, default="random_forest")
    task_type = serializers.ChoiceField(
        choices=("classification", "regression", "auto"), required=False, default="auto"
    )
    method = serializers.ChoiceField(
        choices=("random", "grid", "successive_halving"), required=False, default="random"
    )
    n_trials = serializers.IntegerField(required=False, default=20, min_value=1, max_value=500)
    cv_folds = serializers.IntegerField(
        required=False, allow_null=True, default=None, min_value=2, max_value=10
    )
    seed = serializers.IntegerField(required=False, default=42, min_value=0, max_value=2147483647)
    test_size = serializers.FloatField(required=False, default=0.2, min_value=0.05, max_value=0.95)
    stratify = serializers.BooleanField(required=False, default=True)
    param_space = serializers.DictField(required=False, allow_null=True)
    run_now = serializers.BooleanField(required=False, default=False)

    def validate_target_column(self, value: str) -> str:
        target = (value or "").strip()
        if not target:
            raise serializers.ValidationError("Choose the column the search should predict.")
        return target


class AutoMlJobSerializer(serializers.Serializer):
    """Body of POST /api/lifecycle/experiments/<id>/automl/."""

    dataset_version_id = serializers.UUIDField(required=False, allow_null=True)
    target_column = serializers.CharField(max_length=255)
    feature_columns = serializers.ListField(
        child=serializers.CharField(allow_blank=False), required=False, allow_null=True
    )
    task_type = serializers.ChoiceField(
        choices=("classification", "regression", "auto"), required=False, default="auto"
    )
    time_budget_seconds = serializers.FloatField(
        required=False, default=600.0, min_value=10.0, max_value=7200.0
    )
    max_models = serializers.IntegerField(required=False, default=6, min_value=1, max_value=10)
    cv_folds = serializers.IntegerField(required=False, default=3, min_value=2, max_value=10)
    seed = serializers.IntegerField(required=False, default=42, min_value=0, max_value=2147483647)
    test_size = serializers.FloatField(required=False, default=0.2, min_value=0.05, max_value=0.95)
    run_now = serializers.BooleanField(required=False, default=False)

    def validate_target_column(self, value: str) -> str:
        target = (value or "").strip()
        if not target:
            raise serializers.ValidationError("Choose the column AutoML should predict.")
        return target


class RegisterModelSerializer(serializers.Serializer):
    """Body of POST /api/lifecycle/runs/<id>/register/."""

    model_name = serializers.CharField(required=False, allow_blank=True, max_length=255)


class PromoteModelSerializer(serializers.Serializer):
    """Body of POST /api/lifecycle/models/<id>/versions/<v>/promote/."""

    target = serializers.ChoiceField(
        choices=("candidate", "validated", "staging", "production", "archived")
    )
    confirmed = serializers.BooleanField(required=False, default=False)


class CreateDeploymentSerializer(serializers.Serializer):
    """Body of POST /api/lifecycle/models/<id>/versions/<v>/deploy/."""

    slug = serializers.SlugField(required=False, allow_blank=True, max_length=100)


class PredictOneSerializer(serializers.Serializer):
    """Body of POST /api/lifecycle/deployments/<slug>/predict/."""

    data = serializers.DictField()


class BatchPredictSerializer(serializers.Serializer):
    """Body of POST /api/lifecycle/deployments/<slug>/batch-predict/ (multipart)."""

    file = serializers.FileField()


class DriftCheckSerializer(serializers.Serializer):
    """Body of POST /api/lifecycle/deployments/<slug>/drift-check/."""

    rows = serializers.ListField(child=serializers.DictField(), required=False, default=list)

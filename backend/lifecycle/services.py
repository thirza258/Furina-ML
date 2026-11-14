"""Orchestration for the experiment lifecycle.

Django only *orchestrates* here; every heavy call funnels into
``core.ml.trainer.train`` and the pure service modules in this package
(``leakage``, ``evaluation``, ``search``, ``signature``, ``env``). Nothing in
this module runs a fit itself -- it builds the run's recorded configuration,
calls the trainer, and stores the result, so a Redis/Celery worker or a local
thread pool can both drive it unchanged.
"""

from __future__ import annotations

import io
import time
import uuid
from typing import Any, Sequence

import numpy as np
import pandas as pd
from django.conf import settings
from django.core.files.base import ContentFile
from django.db import transaction
from django.utils import timezone

from core.ml.io import json_safe, write_csv
from core.ml.trainer import ModelArtifact, TrainingError, dump_artifact, train
from core.ml.profiling import infer_task

from lifecycle import env, evaluation, leakage, search, signature
from lifecycle.models import (
    DatasetVersion,
    Experiment,
    ExperimentRun,
    Job,
    Model,
    ModelVersion,
    config_hash_of,
    content_hash_of,
    schema_hash_of,
)

#: Objective metrics a user can rank an experiment by, per task type.
OBJECTIVE_METRICS: dict[str, list[dict[str, Any]]] = {
    "classification": [
        {"key": "accuracy", "label": "Accuracy", "direction": "maximize"},
        {"key": "f1", "label": "F1 (weighted)", "direction": "maximize"},
        {"key": "balanced_accuracy", "label": "Balanced accuracy", "direction": "maximize"},
        {"key": "precision", "label": "Precision (weighted)", "direction": "maximize"},
        {"key": "recall", "label": "Recall (weighted)", "direction": "maximize"},
        {"key": "roc_auc", "label": "ROC AUC", "direction": "maximize"},
        {"key": "log_loss", "label": "Log loss", "direction": "minimize"},
        {"key": "matthews_corrcoef", "label": "Matthews correlation", "direction": "maximize"},
    ],
    "regression": [
        {"key": "r2", "label": "R2", "direction": "maximize"},
        {"key": "mae", "label": "Mean absolute error", "direction": "minimize"},
        {"key": "rmse", "label": "Root mean squared error", "direction": "minimize"},
        {"key": "mse", "label": "Mean squared error", "direction": "minimize"},
        {"key": "mape", "label": "Mean absolute percentage error", "direction": "minimize"},
    ],
}


class LifecycleError(Exception):
    """A request the user can fix. ``str(e)`` reaches the API as a 400."""


def objective_metric_choices(task_type: str) -> list[dict[str, Any]]:
    return OBJECTIVE_METRICS.get(str(task_type), OBJECTIVE_METRICS["classification"])


# --------------------------------------------------------------------------
# dataset versions
# --------------------------------------------------------------------------


def _next_version(dataset_id: Any) -> int:
    latest = DatasetVersion.objects.filter(dataset_id=dataset_id).order_by("-version").first()
    return (latest.version + 1) if latest else 1


def create_dataset_version(
    dataset: Any,
    *,
    frame: pd.DataFrame | None = None,
    source_type: str = "derived",
    parent: DatasetVersion | None = None,
    name: str | None = None,
    description: str = "",
    recipe: list[dict] | None = None,
    user: Any = None,
) -> DatasetVersion:
    """Persist a dataset snapshot as a new immutable version.

    ``frame`` is the data that becomes the version; when omitted the
    dataset's own stored file is used verbatim (how a legacy dataset gets its
    version 1). The version's file is written fresh -- never the dataset's
    file object -- so a version stays correct even after its parent is gone.
    """
    from core.ml.profiling import profile_dataframe

    if frame is None:
        frame = dataset.load_dataframe()
    payload = io.BytesIO()
    write_csv(frame, payload)
    data = payload.getvalue()

    # Re-read exactly what was just written: the version must describe the
    # bytes in the file, not the frame in memory (the same round-trip rule
    # the upload path follows).
    stored = pd.read_csv(io.BytesIO(data))
    version = DatasetVersion(
        dataset=dataset,
        version=_next_version(dataset.id),
        parent=parent,
        name=str(name or dataset.name or "dataset"),
        description=description,
        source_type=source_type,
        row_count=int(len(stored)),
        column_count=int(stored.shape[1]),
        content_hash=content_hash_of(data),
        schema_hash=schema_hash_of(profile_dataframe(stored)),
        recipe=json_safe(recipe) if recipe else None,
        created_by=user,
    )
    version.file.save(f"v{version.version}_{uuid.uuid4().hex[:8]}.csv", ContentFile(data), save=False)
    version.save()
    return version


def ensure_dataset_version(dataset: Any, user: Any = None) -> DatasetVersion:
    """Version 1 for a dataset that predates versioning -- idempotent."""
    existing = DatasetVersion.objects.filter(dataset=dataset).order_by("version").first()
    if existing is not None:
        return existing
    return create_dataset_version(dataset, source_type="upload", user=user)


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------


def create_experiment(
    *,
    dataset_version: DatasetVersion,
    name: str,
    task_type: str = "auto",
    objective_metric: str | None = None,
    objective_direction: str = "maximize",
    description: str = "",
    tags: list[str] | None = None,
    user: Any = None,
) -> Experiment:
    if task_type == "auto":
        frame = dataset_version.load_dataframe()
        task_type = "classification"
        objective_metric = objective_metric or "accuracy"
    else:
        objective_metric = objective_metric or (
            "accuracy" if task_type == "classification" else "r2"
        )
    if objective_direction not in ("maximize", "minimize"):
        raise LifecycleError("Objective direction must be 'maximize' or 'minimize'.")
    if not any(m["key"] == objective_metric for m in objective_metric_choices(task_type)):
        known = ", ".join(m["key"] for m in objective_metric_choices(task_type))
        raise LifecycleError(f"'{objective_metric}' is not an objective for {task_type} tasks. Use one of: {known}.")

    return Experiment.objects.create(
        dataset_version=dataset_version,
        name=str(name).strip() or "Untitled experiment",
        description=description,
        task_type=task_type,
        objective_metric=objective_metric,
        objective_direction=objective_direction,
        tags=json_safe(tags or []),
        created_by=user,
    )


# --------------------------------------------------------------------------
# resource governance
# --------------------------------------------------------------------------

#: Per-algorithm-family multipliers on the raw data footprint. A data matrix
#: of `n` rows and `m` features costs roughly `n * m * 8` bytes per copy;
#: each family keeps a bounded number of such copies during a fit.
_FOOTPRINT_FACTOR: dict[str, float] = {
    "linear": 2.0,
    "tree": 3.0,
    "ensemble": 4.0,
    "svm": 8.0,
    "knn": 2.0,
    "neural": 6.0,
    "naive_bayes": 1.5,
    "baseline": 1.0,
}

#: The default "how much memory does the box have" -- used to grade risk.
DEFAULT_AVAILABLE_GB = 4.0


def _family_of(algorithm: str) -> str:
    key = str(algorithm)
    if any(token in key for token in ("tree", "forest", "boost", "gbdt", "xgb", "lgbm", "bag")):
        return "ensemble" if any(t in key for t in ("boost", "forest", "bag", "xgb", "lgbm")) else "tree"
    if any(token in key for token in ("svm", "svc", "svr")):
        return "svm"
    if "knn" in key or "centroid" in key:
        return "knn"
    if "mlp" in key or "neural" in key:
        return "neural"
    if "bayes" in key:
        return "naive_bayes"
    if "dummy" in key:
        return "baseline"
    return "linear"


def estimate_resources(
    frame: pd.DataFrame,
    algorithm: str,
    *,
    n_trials: int = 1,
    available_gb: float | None = None,
) -> dict[str, Any]:
    """A coarse memory estimate and a risk grade, shown before a job starts.

    The estimate is deliberately rough: it sizes the data matrix and applies
    the algorithm family's multiplier. Its job is to stop a 200-column,
    high-cardinality dataset from silently OOM-ing the worker, not to be
    exact.
    """
    rows, cols = frame.shape
    raw_gb = (rows * cols * 8 * 2) / 1e9  # two copies (X and the target block)
    estimated_gb = raw_gb * _FOOTPRINT_FACTOR[_family_of(algorithm)] * min(max(n_trials, 1), 8) / max(n_trials, 1)
    available = available_gb if available_gb is not None else DEFAULT_AVAILABLE_GB
    if estimated_gb > available:
        risk = "high"
    elif estimated_gb > available * 0.5:
        risk = "medium"
    else:
        risk = "low"
    return {
        "estimated_gb": round(estimated_gb, 2),
        "available_gb": round(available, 2),
        "risk": risk,
        "rows": int(rows),
        "columns": int(cols),
    }


def enforce_limits(frame: pd.DataFrame, n_trials: int = 1) -> None:
    """Hard caps a single job may use; exceeding them fails the job cleanly."""
    max_rows = getattr(settings, "FURINA_MAX_TRAIN_ROWS", 250_000)
    max_columns = getattr(settings, "FURINA_MAX_TRAIN_COLUMNS", 300)
    max_trials = getattr(settings, "FURINA_MAX_HPO_TRIALS", 500)
    rows, cols = frame.shape
    if rows > max_rows:
        raise LifecycleError(
            f"The dataset has {rows:,} rows, above the {max_rows:,} training limit. "
            "Reduce it (sample rows or drop unused columns) and try again."
        )
    if cols > max_columns:
        raise LifecycleError(
            f"The dataset has {cols} columns, above the {max_columns} training limit. Drop unused columns first."
        )
    if n_trials > max_trials:
        raise LifecycleError(
            f"{n_trials} search trials exceeds the {max_trials}-trial limit. Reduce the trial budget."
        )


# --------------------------------------------------------------------------
# running one training configuration
# --------------------------------------------------------------------------

#: How many test-set arrays we keep for diagnostics; enough to plot, bounded
#: so a run's stored JSON cannot bloat without bound.
MAX_DIAGNOSTIC_POINTS = 2_000


def _diagnostics_from_run_arrays(
    result: dict[str, Any],
    task_type: str,
    classes: list[str] | None,
    leakage_summary: dict[str, Any] | None,
) -> dict[str, Any] | None:
    y_test = result.get("y_test_arr")
    y_pred = result.get("y_pred_arr")
    if y_test is None or y_pred is None:
        return None
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    step = max(1, len(y_test) // MAX_DIAGNOSTIC_POINTS)
    y_test = y_test[::step]
    y_pred = y_pred[::step]
    y_proba = result.get("y_proba_arr")
    if y_proba is not None:
        y_proba = np.asarray(y_proba)[::step]
    return evaluation.build_diagnostics(
        task_type=task_type,
        y_true=y_test,
        y_proba=y_proba,
        y_pred=y_pred,
        classes=classes or [],
        metrics=result["metrics"],
        train_metrics=result["train_metrics"],
        leakage_summary=leakage_summary,
        residual_sample=result.get("residual_sample"),
    )


def run_training_config(
    *,
    experiment: Experiment,
    dataset_version: DatasetVersion,
    algorithm: str,
    target_column: str,
    feature_columns: list[str] | None = None,
    task_type: str = "auto",
    test_size: float = 0.2,
    random_state: int | None = 42,
    hyperparameters: dict[str, Any] | None = None,
    stratify: bool = True,
    scale_features: bool = True,
    cv_folds: int | None = None,
    preprocessing: dict[str, Any] | None = None,
    name: str = "",
    trial_index: int | None = None,
    user: Any = None,
    cancellation_check: Any = None,
    frame_override: pd.DataFrame | None = None,
) -> ExperimentRun:
    """Fit one model and record it as an ``ExperimentRun``.

    The run *also* writes a legacy ``core.TrainingRun`` row so the older
    endpoints (predict, download, report) keep serving it, and a joblib
    artifact of its own. ``cancellation_check`` is called between the slow
    phases; when it returns True the run is recorded as cancelled and no
    fit happens.
    """
    from core.models import TrainingRun

    if cancellation_check and cancellation_check():
        run = ExperimentRun.objects.create(
            experiment=experiment,
            dataset_version=dataset_version,
            status="cancelled",
            algorithm=algorithm,
            task_type=str(task_type),
            target_column=target_column,
            feature_columns=json_safe(feature_columns or []),
        )
        return run

    frame = frame_override if frame_override is not None else dataset_version.load_dataframe()
    enforce_limits(frame)

    # Pre-training leakage and quality checks; they never block, they record.
    leakage_summary = leakage.run_leakage_checks(frame, target_column=target_column, feature_columns=feature_columns or [])
    if leakage_summary["severity"] == "critical":
        # Record, do not block: the PRD's rule. The findings ride in the run.
        pass

    extra_warnings: list[str] = []
    resolved_task = task_type if task_type != "auto" else "auto"
    if resolved_task == "auto":
        target_series = frame[target_column] if target_column in frame.columns else None
        if target_series is not None:
            resolved_task = infer_task(target_series)

    started = time.perf_counter()
    try:
        result = train(
            frame,
            target_column=target_column,
            feature_columns=feature_columns,
            algorithm=algorithm,
            task_type=resolved_task,
            test_size=test_size,
            random_state=random_state,
            hyperparameters=hyperparameters,
            stratify=stratify,
            scale_features=scale_features,
            cv_folds=cv_folds,
        )
    except TrainingError as exc:
        run = ExperimentRun.objects.create(
            experiment=experiment,
            dataset_version=dataset_version,
            status="failed",
            error=str(exc),
            algorithm=algorithm,
            task_type=resolved_task,
            target_column=target_column,
            feature_columns=json_safe(feature_columns or []),
            completed_at=timezone.now(),
        )
        return run

    artifact: ModelArtifact = result.pop("artifact", None)
    diagnostics = _diagnostics_from_run_arrays(
        result, resolved_task, result.get("classes"), leakage_summary
    )

    config = {
        "dataset_version": str(dataset_version.id),
        "dataset_hash": dataset_version.content_hash,
        "target_column": target_column,
        "feature_columns": list(feature_columns or []),
        "algorithm": algorithm,
        "task_type": resolved_task,
        "test_size": float(test_size),
        "random_state": random_state,
        "stratify": bool(stratify),
        "cv_folds": cv_folds,
        "scale_features": bool(scale_features),
        "preprocessing": json_safe(preprocessing or {}),
        "hyperparameters": json_safe(hyperparameters or {}),
    }
    number = (
        experiment.runs.count() + 1
        if trial_index is None
        else 1  # trials number themselves; the run gets a trial index instead
    )

    run = ExperimentRun.objects.create(
        experiment=experiment,
        dataset_version=dataset_version,
        number=number,
        trial_index=trial_index,
        status="success",
        algorithm=result["algorithm"],
        algorithm_label=result["algorithm_label"],
        task_type=resolved_task,
        target_column=target_column,
        feature_columns=result["feature_columns"],
        test_size=float(test_size),
        random_state=random_state,
        stratify=bool(stratify),
        cv_folds=cv_folds,
        scale_features=bool(scale_features),
        preprocessing=json_safe(preprocessing or {}),
        hyperparameters=json_safe(hyperparameters or {}),
        effective_hyperparameters=result["effective_hyperparameters"],
        config_hash=config_hash_of(config),
        environment=env.capture_environment(),
        metrics=result["metrics"],
        train_metrics=result["train_metrics"],
        diagnostics=diagnostics,
        cv=result["cv"],
        classes=result["classes"],
        confusion_matrix=result["confusion_matrix"],
        per_class=result["per_class"],
        feature_importance=result["feature_importance"],
        residual_sample=result["residual_sample"],
        input_schema=result["input_schema"],
        warnings=list(result.get("warnings") or []) + extra_warnings,
        n_train=result["n_train"],
        n_test=result["n_test"],
        duration_ms=result["duration_ms"],
        dataset_hash=dataset_version.content_hash,
        dataset_row_count=dataset_version.row_count,
        dataset_feature_count=len(result["feature_columns"]),
        completed_at=timezone.now(),
    )

    if leakage_summary["count"]:
        run.warnings = list(run.warnings or []) + [
            f"Leakage/quality check ({leakage_summary['severity']}): "
            + "; ".join(f["message"] for f in leakage_summary["findings"][:3])
        ]
        run.save(update_fields=["warnings"])

    # Store the artifact on the lifecycle run and on a legacy twin, so both
    # /api/runs/<legacy>/download_model/ and the lifecycle download work.
    if artifact is not None:
        try:
            blob = dump_artifact(artifact)
        except Exception:  # noqa: BLE001 - a failed dump must not lose the run
            blob = None
        if blob:
            run.artifact.save(f"lifecycle_{uuid.uuid4().hex[:8]}.joblib", ContentFile(blob), save=False)
            run.model_size_bytes = len(blob)
            legacy = TrainingRun.objects.create(
                name=name or f"{experiment.name} trial" if trial_index is not None else name,
                dataset=dataset_version.dataset,
                algorithm=run.algorithm,
                algorithm_label=run.algorithm_label,
                task_type=run.task_type,
                target_column=target_column,
                feature_columns=run.feature_columns,
                hyperparameters=run.hyperparameters,
                effective_hyperparameters=run.effective_hyperparameters,
                test_size=run.test_size,
                random_state=run.random_state,
                stratify=run.stratify,
                scale_features=run.scale_features,
                cv_folds=run.cv_folds,
                status="success",
                n_train=run.n_train,
                n_test=run.n_test,
                duration_ms=run.duration_ms,
                metrics=run.metrics,
                train_metrics=run.train_metrics,
                cv=run.cv,
                classes=run.classes,
                confusion_matrix=run.confusion_matrix,
                per_class=run.per_class,
                feature_importance=run.feature_importance,
                residual_sample=run.residual_sample,
                input_schema=run.input_schema,
                warnings=run.warnings,
            )
            legacy.artifact.save(f"{run.algorithm}_{target_column}_{uuid.uuid4().hex[:6]}.joblib", ContentFile(blob), save=False)
            legacy.save()
            run.legacy_run = legacy
        run.save()

    experiment.consider_run(run)
    return run


# --------------------------------------------------------------------------
# job execution: the three kinds of job body
# --------------------------------------------------------------------------


def execute_job(job: Job) -> None:
    """Run a queued job to completion, updating progress and the cancel flag.

    A worker calls this with a claimed job. It never raises for a user-fixable
    problem: those are recorded on the job with a suggested action.
    """
    job.status = "running"
    job.started_at = timezone.now()
    job.last_heartbeat = job.started_at
    job.save(update_fields=["status", "started_at", "last_heartbeat"])

    try:
        payload = dict(job.payload or {})
        kind = job.kind
        if kind == "train":
            _execute_train(job, payload)
        elif kind == "hpo":
            _execute_hpo(job, payload)
        elif kind == "automl":
            _execute_automl(job, payload)
        elif kind == "reproduce":
            _execute_reproduce(job, payload)
        else:  # pragma: no cover - unknown kinds fail loudly
            raise LifecycleError(f"Unknown job kind '{kind}'.")
    except (LifecycleError, TrainingError, ValueError) as exc:
        job.status = "failed"
        job.error = str(exc)
        job.suggested_action = _suggested_action_for(exc, job.payload)
        job.progress = 100.0
        job.finished_at = timezone.now()
        job.save()
        return
    except Exception as exc:  # noqa: BLE001 - the worker's safety net
        job.status = "failed"
        job.error = "The job could not be completed."
        job.suggested_action = f"Server error ({exc.__class__.__name__}); see the server logs for details."
        job.finished_at = timezone.now()
        job.save()
        return

    if job.is_cancel_requested():
        job.status = "cancelled"
        job.finished_at = timezone.now()
        job.save(update_fields=["status", "finished_at"])
        return
    job.status = "completed"
    job.finished_at = timezone.now()
    job.progress = 100.0
    job.save()


def _suggested_action_for(exc: Exception, payload: dict) -> str:
    text = str(exc)
    if "limit" in text.lower() or "limit." in text:
        return "Reduce the dataset or the search budget and submit a smaller job."
    if "not available" in text.lower() or "does not support" in text.lower():
        return "Pick an algorithm that supports the task, or correct the objective metric."
    return "Check the job's configuration and try again with a smaller dataset or fewer trials."


def _cancellation(job: Job) -> Any:
    def check() -> bool:
        return job.is_cancel_requested()

    return check


def _save_search_report(
    job: Job,
    space: Any,
    plans: list[dict] | None,
    best: ExperimentRun | None,
    completed: int,
    *,
    automl: bool = False,
    candidates: int | None = None,
) -> None:
    """Record what a search did so 'what did Furina do?' has an honest answer."""
    report: dict[str, Any] = {
        "kind": "automl" if automl else "hpo",
        "completed_trials": completed,
        "best": {
            "run_id": str(best.id) if best else None,
            "algorithm": best.algorithm if best else None,
            "objective": best.experiment.objective_metric if best else None,
            "value": best.primary_metric() if best else None,
            "hyperparameters": best.hyperparameters if best else None,
        },
        "candidates_considered": candidates,
    }
    if space is not None and plans:
        report["algorithm"] = space.algorithm
        report["methods_available"] = ["random", "grid", "successive_halving"]
    job.payload = {**(job.payload or {}), "search_report": json_safe(report)}
    job.save(update_fields=["payload"])


def _resolve_experiment(payload: dict) -> Experiment:
    experiment_id = payload.get("experiment_id")
    if experiment_id:
        experiment = Experiment.objects.filter(pk=experiment_id).first()
        if experiment is None:
            raise LifecycleError("The experiment this job belongs to no longer exists.")
        return experiment
    raise LifecycleError("The job has no experiment to run trials for.")


def _resolve_version(payload: dict, experiment: Experiment) -> DatasetVersion:
    version_id = payload.get("dataset_version_id") or experiment.dataset_version_id
    version = DatasetVersion.objects.filter(pk=version_id).first()
    if version is None:
        raise LifecycleError("The dataset version this job trains on no longer exists.")
    return version


def _execute_train(job: Job, payload: dict) -> None:
    experiment = _resolve_experiment(payload)
    version = _resolve_version(payload, experiment)
    job.total_units = 1
    job.mark_progress(0, 1)
    run = run_training_config(
        experiment=experiment,
        dataset_version=version,
        algorithm=str(payload.get("algorithm", "logistic_regression")),
        target_column=str(payload.get("target_column", "")),
        feature_columns=payload.get("feature_columns") or None,
        task_type=str(payload.get("task_type") or "auto"),
        test_size=float(payload.get("test_size") or 0.2),
        random_state=payload.get("random_state", 42),
        hyperparameters=payload.get("hyperparameters") or {},
        stratify=bool(payload.get("stratify", True)),
        cv_folds=payload.get("cv_folds"),
        preprocessing=payload.get("preprocessing") or {},
        name=str(payload.get("name") or ""),
        cancellation_check=_cancellation(job),
    )
    job.best_run = run
    job.mark_progress(1, 1)
    job.save(update_fields=["best_run"])
    if not run.status == "success":
        raise LifecycleError(f"The model did not train: {run.error or 'it was cancelled.'}")


def _execute_hpo(job: Job, payload: dict) -> None:
    experiment = _resolve_experiment(payload)
    version = _resolve_version(payload, experiment)
    frame = version.load_dataframe()
    target = str(payload.get("target_column") or "")
    features = payload.get("feature_columns") or None
    enforce_limits(frame, int(payload.get("n_trials") or 20))

    space = search.ParamSpace.from_payload(payload.get("param_space") or {})
    method = str(payload.get("method") or "random").lower()
    n_trials = int(payload.get("n_trials") or 20)
    cv_folds = payload.get("cv_folds")
    objective = experiment.objective_metric
    direction = experiment.objective_direction
    seed = int(payload.get("seed") or 42)
    task_type = str(payload.get("task_type") or "auto")

    plans = search.plan_trials(space, method=method, n_trials=n_trials, cv_folds=cv_folds, seed=seed)
    job.total_units = len(plans)
    job.mark_progress(0, len(plans))

    cancel = _cancellation(job)
    best: ExperimentRun | None = None
    best_value: float | None = None
    completed = 0
    for index, plan in enumerate(plans, start=1):
        if cancel():
            break
        job.last_heartbeat = timezone.now()
        job.save(update_fields=["last_heartbeat"])
        run = run_training_config(
            experiment=experiment,
            dataset_version=version,
            algorithm=space.algorithm,
            target_column=target,
            feature_columns=features,
            task_type=task_type,
            test_size=float(payload.get("test_size") or 0.2),
            random_state=seed,
            hyperparameters=plan["hyperparameters"],
            stratify=bool(payload.get("stratify", True)),
            cv_folds=cv_folds,
            trial_index=index - 1,
            name=f"{space.algorithm} trial {index}",
            cancellation_check=cancel,
        )
        completed += 1
        job.mark_progress(completed, len(plans))
        value = experiment.objective_value(run)
        if run.status == "success" and value is not None:
            better = best_value is None or (
                value > best_value if direction == "maximize" else value < best_value
            )
            if better:
                best, best_value = run, value
    job.completed_units = completed
    job.mark_progress(completed, len(plans))
    if best is not None:
        job.best_run = best
    if cancel():
        raise LifecycleError("The search was cancelled before every trial ran.")
    if best is None:
        raise LifecycleError("No trial completed successfully, so the search has no best configuration.")
    _save_search_report(job, space, plans, best, completed)


def _execute_automl(job: Job, payload: dict) -> None:
    experiment = _resolve_experiment(payload)
    version = _resolve_version(payload, experiment)
    frame = version.load_dataframe()
    target = str(payload.get("target_column") or "")
    if target not in frame.columns:
        raise LifecycleError(f"Column '{target}' is not in the dataset.")
    task_type = str(payload.get("task_type") or infer_task(frame[target]))
    budget = float(payload.get("time_budget_seconds") or 600)
    max_models = int(payload.get("max_models") or 6)
    cv_folds = int(payload.get("cv_folds") or 3)
    seed = int(payload.get("seed") or 42)
    objective = experiment.objective_metric
    direction = experiment.objective_direction

    # AutoML caps its own data so one job cannot pin the box: sample the frame
    # when it is too large, with the seed so a rerun samples identically.
    max_rows = int(payload.get("max_rows") or getattr(settings, "FURINA_AUTOML_MAX_ROWS", 40_000))
    automl_frame = frame
    if len(frame) > max_rows:
        automl_frame = frame.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    candidates = search.default_candidates(task_type, max_models)
    deadline = time.monotonic() + budget
    cancel = _cancellation(job)

    job.total_units = len(candidates) + 1  # baselines + the top model's mini search
    job.mark_progress(0, job.total_units)

    trial_counter = 0
    best: ExperimentRun | None = None
    best_value: float | None = None
    completed = 0

    def consider(run: ExperimentRun) -> None:
        nonlocal best, best_value
        if run.status != "success":
            return
        value = experiment.objective_value(run)
        if value is None:
            return
        better = best_value is None or (
            value > best_value if direction == "maximize" else value < best_value
        )
        if better:
            best, best_value = run, value

    for index, (algorithm, hyperparameters) in enumerate(search.automl_schedule(candidates, seed), start=1):
        if cancel():
            break
        if time.monotonic() > deadline and index > 1:
            break  # let at least the baseline land, then honour the budget
        job.last_heartbeat = timezone.now()
        job.save(update_fields=["last_heartbeat"])
        run = run_training_config(
            experiment=experiment,
            dataset_version=version,
            algorithm=algorithm,
            target_column=target,
            feature_columns=payload.get("feature_columns") or None,
            task_type=task_type,
            test_size=float(payload.get("test_size") or 0.2),
            random_state=seed,
            hyperparameters=hyperparameters,
            cv_folds=cv_folds,
            trial_index=trial_counter,
            name=f"automl {algorithm}",
            cancellation_check=cancel,
            frame_override=automl_frame,
        )
        trial_counter += 1
        completed += 1
        job.mark_progress(completed, job.total_units)
        consider(run)

    # A short random search around the best candidate, still under budget.
    if best is not None and time.monotonic() < deadline:
        from core.ml.registry import get_spec

        suggestion = search.suggested_search_space(get_spec(best.algorithm, task_type))
        micro = search.plan_trials(
            search.ParamSpace.from_payload(suggestion),
            method="random",
            n_trials=4,
            cv_folds=cv_folds,
            seed=seed,
        )
        for plan in micro:
            if time.monotonic() > deadline or cancel():
                break
            run = run_training_config(
                experiment=experiment,
                dataset_version=version,
                algorithm=best.algorithm,
                target_column=target,
                feature_columns=payload.get("feature_columns") or None,
                task_type=task_type,
                test_size=float(payload.get("test_size") or 0.2),
                random_state=seed,
                hyperparameters=plan["hyperparameters"],
                cv_folds=cv_folds,
                trial_index=trial_counter,
                name=f"automl {best.algorithm} tune",
                cancellation_check=cancel,
                frame_override=automl_frame,
            )
            trial_counter += 1
            completed += 1
            consider(run)
        job.mark_progress(job.total_units, job.total_units)

    if best is None:
        raise LifecycleError("AutoML could not train any candidate model on this dataset.")
    job.best_run = best
    job.save(update_fields=["best_run"])
    _save_search_report(job, None, None, best, completed, automl=True, candidates=len(candidates))


def _execute_reproduce(job: Job, payload: dict) -> None:
    run_id = payload.get("run_id")
    original = ExperimentRun.objects.filter(pk=run_id).first()
    if original is None:
        raise LifecycleError("The run to reproduce no longer exists.")
    experiment = original.experiment
    version = original.dataset_version
    cancel = _cancellation(job)
    job.total_units = 1
    job.mark_progress(0, 1)
    rerun = run_training_config(
        experiment=experiment,
        dataset_version=version,
        algorithm=original.algorithm,
        target_column=original.target_column,
        feature_columns=original.feature_columns or None,
        task_type=original.task_type,
        test_size=original.test_size,
        random_state=original.random_state,
        hyperparameters=original.hyperparameters or {},
        stratify=original.stratify,
        scale_features=original.scale_features,
        cv_folds=original.cv_folds,
        preprocessing=original.preprocessing or {},
        name=f"reproduce {original.display_name}",
        cancellation_check=cancel,
    )
    diffs = env.environment_diffs(original.environment, env.capture_environment())
    if diffs:
        rerun.warnings = list(rerun.warnings or []) + [
            "This reproduction ran in a different environment: "
            + "; ".join(diffs)
            + ". Binary-exact results are not guaranteed across library versions."
        ]
        rerun.save(update_fields=["warnings"])
    job.best_run = rerun
    job.mark_progress(1, 1)
    job.save(update_fields=["best_run"])


# --------------------------------------------------------------------------
# model registry
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# job submission: build a validated, queue-ready Job
# --------------------------------------------------------------------------


def _experiment_from(payload: dict) -> Experiment:
    """Resolve the experiment a job trains against, or raise a 400 reason."""
    experiment = Experiment.objects.filter(pk=payload.get("experiment_id")).first()
    if experiment is None:
        raise LifecycleError("Choose an experiment to add this job to.")
    return experiment


def _version_from(experiment: Experiment, payload: dict) -> DatasetVersion:
    version = DatasetVersion.objects.filter(pk=payload.get("dataset_version_id")).first()
    if version is None:
        version = experiment.dataset_version
    if version.dataset_id != experiment.dataset_version.dataset_id:
        raise LifecycleError(
            "This job would train a dataset that does not belong to the experiment. "
            "Pick a dataset version from the same dataset."
        )
    return version


def create_training_job(payload: dict, *, user: Any = None, start: bool = False) -> Job:
    """Queue a single-training job against an experiment.

    ``start`` runs the job inline instead of queueing it -- the synchronous
    path a small deployment or a test takes. Queueing is the default, which is
    what keeps training out of the HTTP request in a real deployment.
    """
    experiment = _experiment_from(payload)
    version = _version_from(experiment, payload)
    frame = version.load_dataframe()
    target = str(payload.get("target_column") or "")
    enforce_limits(frame)

    job = Job.objects.create(
        kind="train",
        experiment=experiment,
        status="queued",
        payload=json_safe(
            {
                "experiment_id": str(experiment.id),
                "dataset_version_id": str(version.id),
                "algorithm": str(payload.get("algorithm") or "logistic_regression"),
                "target_column": target,
                "feature_columns": payload.get("feature_columns") or [],
                "task_type": str(payload.get("task_type") or experiment.task_type),
                "test_size": payload.get("test_size") or 0.2,
                "random_state": payload.get("random_state", 42),
                "hyperparameters": payload.get("hyperparameters") or {},
                "stratify": bool(payload.get("stratify", True)),
                "cv_folds": payload.get("cv_folds"),
                "name": str(payload.get("name") or ""),
            }
        ),
    )
    if start:
        execute_job(job)
    return job


def create_hpo_job(payload: dict, *, user: Any = None, start: bool = False) -> Job:
    """Queue a hyperparameter-search job.

    The search budget (trial count) and the algorithm's parameter space are
    validated up front so a queueing request never enqueues work that would
    immediately fail for a user-fixable reason.
    """
    experiment = _experiment_from(payload)
    version = _version_from(experiment, payload)
    frame = version.load_dataframe()
    target = str(payload.get("target_column") or "")
    if target and target not in frame.columns:
        raise LifecycleError(f"Column '{target}' is not in the dataset.")

    n_trials = int(payload.get("n_trials") or 20)
    enforce_limits(frame, n_trials)

    algorithm = str(payload.get("algorithm") or "random_forest")
    task_type = str(payload.get("task_type") or experiment.task_type)
    from core.ml.registry import get_spec

    try:
        spec = get_spec(algorithm, task_type if task_type != "auto" else None)
    except Exception:  # noqa: BLE001 - an unknown algorithm is a 400, not a 500
        raise LifecycleError(f"Algorithm '{algorithm}' is not available for this task.") from None

    # The caller may pass an explicit param space, otherwise derive a
    # sensible one from the registry's advertised form.
    param_space = payload.get("param_space") or search.suggested_search_space(spec)

    job = Job.objects.create(
        kind="hpo",
        experiment=experiment,
        status="queued",
        payload=json_safe(
            {
                "experiment_id": str(experiment.id),
                "dataset_version_id": str(version.id),
                "algorithm": algorithm,
                "target_column": target,
                "feature_columns": payload.get("feature_columns") or [],
                "task_type": task_type,
                "test_size": payload.get("test_size") or 0.2,
                "seed": payload.get("seed", 42),
                "n_trials": n_trials,
                "method": str(payload.get("method") or "random"),
                "cv_folds": payload.get("cv_folds"),
                "stratify": bool(payload.get("stratify", True)),
                "param_space": json_safe(param_space),
            }
        ),
    )
    if start:
        execute_job(job)
    return job


def create_automl_job(payload: dict, *, user: Any = None, start: bool = False) -> Job:
    """Queue an AutoML job: pick a candidate, then tune the best one."""
    experiment = _experiment_from(payload)
    version = _version_from(experiment, payload)
    frame = version.load_dataframe()
    target = str(payload.get("target_column") or "")
    if target not in frame.columns:
        raise LifecycleError(f"Column '{target}' is not in the dataset.")

    budget = float(payload.get("time_budget_seconds") or 600)
    enforce_limits(frame, int(payload.get("max_models") or 6))

    job = Job.objects.create(
        kind="automl",
        experiment=experiment,
        status="queued",
        payload=json_safe(
            {
                "experiment_id": str(experiment.id),
                "dataset_version_id": str(version.id),
                "target_column": target,
                "feature_columns": payload.get("feature_columns") or [],
                "task_type": str(payload.get("task_type") or "auto"),
                "time_budget_seconds": budget,
                "max_models": int(payload.get("max_models") or 6),
                "cv_folds": int(payload.get("cv_folds") or 3),
                "seed": int(payload.get("seed") or 42),
                "test_size": payload.get("test_size") or 0.2,
            }
        ),
    )
    if start:
        execute_job(job)
    return job


def request_reproduce_job(run: ExperimentRun, *, user: Any = None, start: bool = False) -> Job:
    """Queue a job that re-trains a run's exact configuration.

    Reproduction is honest about its limits: if the captured environment
    differs from the current one, the resulting run carries a warning saying
    so, instead of silently claiming byte-identical results.
    """
    experiment = run.experiment
    job = Job.objects.create(
        kind="reproduce",
        experiment=experiment,
        status="queued",
        payload=json_safe(
            {
                "experiment_id": str(experiment.id),
                "dataset_version_id": str(run.dataset_version_id),
                "run_id": str(run.id),
            }
        ),
    )
    if start:
        execute_job(job)
    return job


# --------------------------------------------------------------------------
# model registry
# --------------------------------------------------------------------------


def register_model(
    run: ExperimentRun,
    *,
    model_name: str | None = None,
    status: str = "candidate",
    user: Any = None,
) -> ModelVersion:
    """Register a trained run as a version in the registry.

    The version is immutable once created; moving it through the lifecycle
    happens through :func:`promote_model_version`, which never edits the
    version's data, only its ``status``.
    """
    if run.status != "success" or not run.metrics:
        raise LifecycleError("Only a successful run with recorded metrics can be registered.")
    name = str(model_name or f"{run.experiment.name} model")
    model, _created = Model.objects.get_or_create(
        name=name, task_type=run.task_type, defaults={"created_by": user}
    )
    version_number = model.next_version_number()
    version = ModelVersion.objects.create(
        model=model,
        version=version_number,
        source_run=run,
        legacy_run=run.legacy_run,
        dataset_version=run.dataset_version,
        status=status,
        signature=signature.build_signature(
            run.input_schema,
            task_type=run.task_type,
            target_column=run.target_column,
            classes=run.classes,
            feature_columns=run.feature_columns,
        ),
        metrics=run.metrics,
        model_card=build_model_card(model, version_number, run),
        created_by=getattr(user, "id", None),
    )
    return version


def promote_model_version(version: ModelVersion, target: str, *, confirmed: bool = False) -> ModelVersion:
    """Move a version to the next lifecycle state.

    Promoting to production needs ``confirmed`` and moves the incumbent
    production version to archived, keeping the registry's one-active-
    production invariant.
    """
    allowed, reason = version.can_promote_to(target)
    if not allowed:
        raise LifecycleError(reason)
    if target == "production" and not confirmed:
        raise LifecycleError(
            "Promoting to production is a live change. Confirm it before doing so."
        )

    with transaction.atomic():
        if target == "production":
            incumbent = version.model.production_version
            if incumbent is not None and incumbent.pk != version.pk and incumbent.status == "production":
                incumbent.status = "archived"
                incumbent.save(update_fields=["status"])
        version.status = target
        version.save(update_fields=["status"])
        if target == "production":
            version.model.production_version = version
            version.model.save(update_fields=["production_version"])
    return version


def build_model_card(model: Model, version_number: int, run: ExperimentRun) -> dict[str, Any]:
    """The auto-generated model card: the honest, human-readable summary."""
    metrics = run.metrics or {}
    dataset = run.dataset_version
    card = {
        "model": model.name,
        "version": version_number,
        "task": run.task_type,
        "algorithm": run.algorithm_label or run.algorithm,
        "target_column": run.target_column,
        "dataset": {
            "dataset_version": str(dataset.name or dataset.dataset.name),
            "rows": dataset.row_count,
            "columns": dataset.column_count,
        },
        "training_method": {
            "cv_folds": run.cv_folds,
            "test_size": run.test_size,
            "random_state": run.random_state,
            "stratified": run.stratify,
            "hyperparameters": run.effective_hyperparameters or run.hyperparameters,
        },
        "features": [str(column) for column in (run.feature_columns or [])],
        "metrics": {key: value for key, value in metrics.items() if value is not None},
        "validation_strategy": (
            f"{run.cv_folds}-fold cross-validation"
            if run.cv_folds
            else f"{run.test_size:.0%} holdout split"
        ),
        "known_limitations": [str(note) for note in (run.warnings or [])][:6],
        "intended_use": f"Predicting '{run.target_column}' on tabular data shaped like {dataset.name}.",
        "out_of_scope_use": [
            "Decisions on data that does not look like the training data",
            "As the sole basis for a high-impact decision without human review",
        ],
        "deployment_status": "Not deployed yet.",
        "training_timestamp": run.created_at.isoformat() if run.created_at else None,
    }
    if dataset and any("leakage" in str(note).lower() or "identif" in str(note).lower() for note in (run.warnings or [])):
        card["data_quality_note"] = (
            "The training data raised a data-quality or leakage flag; see the run's "
            "diagnostics before relying on this model."
        )
    return json_safe(card)

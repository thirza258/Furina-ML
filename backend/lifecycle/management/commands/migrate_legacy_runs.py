"""Backfill the lifecycle platform from the legacy workbench's records.

Every ``core.Dataset`` becomes a versioned dataset; every ``core.TrainingRun``
becomes an ``ExperimentRun`` inside an experiment, keeping a pointer back to
the legacy row so the old endpoints keep serving it unchanged.

The command is idempotent and safe to interrupt: each dataset is migrated in
its own transaction, and anything already migrated is skipped.

    python manage.py migrate_legacy_runs --dry-run
    python manage.py migrate_legacy_runs
"""

from __future__ import annotations

from typing import Any

from django.core.management.base import BaseCommand
from django.db import transaction

from core.ml.io import json_safe
from lifecycle.models import DatasetVersion, Experiment, ExperimentRun
from lifecycle.models import config_hash_of as config_hash_for


class Command(BaseCommand):
    help = "Migrate legacy core.Dataset / core.TrainingRun records into the lifecycle platform."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Report what would be created without writing anything.",
        )
        parser.add_argument(
            "--dataset-id",
            default=None,
            help="Migrate only this dataset (by id).",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        from core.models import Dataset, TrainingRun

        dry_run = bool(options.get("dry_run"))
        datasets = Dataset.objects.all().order_by("created_at")
        if options.get("dataset_id"):
            datasets = datasets.filter(pk=options["dataset_id"])
            if not datasets.exists():
                self.stderr.write(f"No dataset with id {options['dataset_id']}.")
                return

        counts = {"datasets": 0, "versions": 0, "experiments": 0, "runs": 0, "skipped": 0}
        for dataset in datasets:
            try:
                self._migrate_dataset(dataset, dry_run, counts)
            except Exception as exc:  # noqa: BLE001 - one bad dataset must not stop the rest
                counts["skipped"] += 1
                self.stderr.write(f"Skipped dataset {dataset.id} ({dataset.name}): {exc}")

        verb = "Would create" if dry_run else "Created"
        self.stdout.write(
            self.style.SUCCESS(
                f"{verb}: {counts['datasets']} dataset version(s), {counts['experiments']} experiment(s), "
                f"{counts['runs']} run(s). Skipped: {counts['skipped']}."
            )
        )

    def _migrate_dataset(self, dataset: Any, dry_run: bool, counts: dict) -> None:
        from core.models import TrainingRun

        runs = list(TrainingRun.objects.filter(dataset=dataset).order_by("created_at"))
        if not runs and not DatasetVersion.objects.filter(dataset=dataset).exists():
            counts["skipped"] += 1
            return

        existing = DatasetVersion.objects.filter(dataset=dataset).order_by("version").first()
        if existing is not None:
            version = existing
        else:
            version = self._build_version(dataset, dry_run)
            if version is not None and not dry_run:
                counts["datasets"] += 1
                counts["versions"] += 1

        if version is None:
            return

        for run in runs:
            if ExperimentRun.objects.filter(legacy_run=run).exists():
                continue
            task_type = str(run.task_type or "classification")
            experiment = self._find_or_create_experiment(version, dataset, task_type, dry_run)
            if experiment is None:
                continue
            if dry_run:
                counts["runs"] += 1
                continue
            self._copy_run(run, experiment, version)
            counts["runs"] += 1

        if not dry_run:
            self._refresh_best_run(experiment=None)

    def _build_version(self, dataset: Any, dry_run: bool):
        """Create version 1 for a legacy dataset from its stored file."""
        if dry_run:
            return DatasetVersion(dataset=dataset, version=1, name=str(dataset.name or ""))

        try:
            payload = dataset.file.read() if dataset.file else None
        except (OSError, ValueError):
            payload = None

        if payload is None:
            # A dataset whose file is gone still deserves a version row, so
            # its runs keep a resolvable dataset_version reference.
            version = DatasetVersion(
                dataset=dataset,
                version=1,
                name=str(dataset.name or ""),
                source_type="upload",
                row_count=0,
                column_count=0,
                description="Migrated from the legacy workbench; the original file was unavailable.",
            )
            version.save()
            return version

        import hashlib

        from django.core.files.base import ContentFile

        digest = hashlib.sha256(payload).hexdigest()
        version = DatasetVersion(
            dataset=dataset,
            version=1,
            name=str(dataset.name or ""),
            source_type=str(dataset.source or "upload"),
            content_hash=digest,
            row_count=int(getattr(dataset, "n_rows", 0) or 0),
            column_count=int(getattr(dataset, "n_cols", 0) or 0),
            description="Migrated from the legacy workbench.",
        )
        version.file.save(f"{dataset.id}-legacy.csv", ContentFile(payload), save=False)
        version.save()
        return version

    def _find_or_create_experiment(
        self, version: DatasetVersion, dataset: Any, task_type: str, dry_run: bool
    ):
        existing = Experiment.objects.filter(
            dataset_version=version, task_type=task_type
        ).first()
        if existing is not None:
            return existing
        if dry_run:
            return Experiment(dataset_version=version, name=str(dataset.name or ""))
        experiment = Experiment.objects.create(
            dataset_version=version,
            name=str(dataset.name or "Migrated experiment"),
            task_type=task_type,
            objective_metric="accuracy" if task_type == "classification" else "r2",
            objective_direction="maximize",
            description="Created by migrate_legacy_runs from the legacy workbench.",
            tags=["migrated"],
        )
        return experiment

    def _copy_run(self, legacy: Any, experiment: Experiment, version: DatasetVersion) -> None:
        """Copy a legacy run's recorded results into a lifecycle run."""
        metrics = dict(legacy.metrics or {})
        train_metrics = dict(legacy.train_metrics or {})
        hyperparameters = dict(legacy.hyperparameters or {})
        effective = dict(legacy.effective_hyperparameters or {})
        config = {
            "algorithm": legacy.algorithm,
            "task_type": legacy.task_type,
            "target_column": legacy.target_column,
            "feature_columns": list(legacy.feature_columns or []),
            "hyperparameters": hyperparameters,
            "test_size": float(legacy.test_size or 0.0),
            "random_state": legacy.random_state,
            "stratify": bool(legacy.stratify),
            "scale_features": bool(getattr(legacy, "scale_features", False)),
            "cv_folds": legacy.cv_folds,
        }
        run = ExperimentRun.objects.create(
            experiment=experiment,
            dataset_version=version,
            legacy_run=legacy,
            status=str(legacy.status or "success"),
            algorithm=str(legacy.algorithm or ""),
            algorithm_label=str(getattr(legacy, "algorithm_label", "") or ""),
            task_type=str(legacy.task_type or "classification"),
            target_column=str(legacy.target_column or ""),
            feature_columns=json_safe(list(legacy.feature_columns or [])),
            hyperparameters=json_safe(hyperparameters),
            effective_hyperparameters=json_safe(effective),
            metrics=json_safe(metrics),
            train_metrics=json_safe(train_safe(train_metrics)),
            cv=legacy.cv,
            classes=legacy.classes,
            confusion_matrix=legacy.confusion_matrix,
            per_class=legacy.per_class,
            feature_importance=legacy.feature_importance,
            residual_sample=legacy.residual_sample,
            input_schema=legacy.input_schema,
            warnings=json_safe(list(legacy.warnings or [])),
            error=str(legacy.error or ""),
            duration_ms=int(legacy.duration_ms or 0),
            test_size=float(legacy.test_size or 0.0),
            random_state=legacy.random_state,
            stratify=bool(legacy.stratify),
            scale_features=bool(getattr(legacy, "scale_features", True)),
            cv_folds=legacy.cv_folds,
            n_train=int(getattr(legacy, "n_train", 0) or 0),
            n_test=int(getattr(legacy, "n_test", 0) or 0),
            dataset_hash=str(version.content_hash or ""),
            dataset_row_count=int(version.row_count or 0),
            config_hash=config_hash_for(config),
            # The original run predates environment capture; saying so is
            # honest, while claiming a captured environment would not be.
            environment={"migrated": True},
            created_at=legacy.created_at,
            completed_at=legacy.created_at,
        )
        return run

    def _refresh_best_run(self, experiment: Experiment | None = None) -> None:
        """Point each experiment at its best run, by its own objective."""
        experiments = (
            Experiment.objects.all() if experiment is None else Experiment.objects.filter(pk=experiment.pk)
        )
        for item in experiments:
            best = None
            best_value = None
            for run in item.runs.all():
                value = run.primary_metric()
                if value is None:
                    continue
                if best_value is None:
                    best, best_value = run, value
                    continue
                better = (
                    value > best_value
                    if item.objective_direction == "maximize"
                    else value < best_value
                )
                if better:
                    best, best_value = run, value
            if best is not None and item.best_run_id != best.pk:
                item.best_run = best
                item.best_metric_value = best_value
                item.save(update_fields=["best_run", "best_metric_value"])


def train_safe(metrics: dict) -> dict:
    """Legacy train metrics may hold numpy types; make them JSON-safe."""
    return json_safe(metrics)

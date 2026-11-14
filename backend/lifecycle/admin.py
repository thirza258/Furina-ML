"""Admin views over the experiment-lifecycle records.

Read-mostly, matching ``core/admin.py``: operators browse the lineage, spot a
stuck job, and correct a lifecycle state, but the forms deliberately expose
only the fields a human should ever edit by hand.
"""

from django.contrib import admin

from lifecycle.models import (
    DatasetVersion,
    Deployment,
    Experiment,
    ExperimentRun,
    Job,
    Model,
    ModelVersion,
    MonitoringRecord,
)


@admin.register(DatasetVersion)
class DatasetVersionAdmin(admin.ModelAdmin):
    list_display = ("__str__", "source_type", "row_count", "column_count", "content_hash", "created_at")
    list_filter = ("source_type", "created_at")
    search_fields = ("name", "content_hash")
    readonly_fields = ("id", "content_hash", "schema_hash", "row_count", "column_count", "created_at")
    ordering = ("-created_at",)


@admin.register(Experiment)
class ExperimentAdmin(admin.ModelAdmin):
    list_display = ("name", "task_type", "objective_metric", "objective_direction", "best_run", "created_at")
    list_filter = ("task_type", "objective_direction", "created_at")
    search_fields = ("name", "tags")
    readonly_fields = ("id", "created_at", "best_metric_value")
    ordering = ("-created_at",)


@admin.register(ExperimentRun)
class ExperimentRunAdmin(admin.ModelAdmin):
    list_display = ("__str__", "status", "algorithm", "task_type", "objective_metric", "created_at")
    list_filter = ("status", "task_type", "created_at")
    search_fields = ("algorithm", "target_column")
    readonly_fields = (
        "id",
        "config_hash",
        "metrics",
        "train_metrics",
        "cv",
        "confusion_matrix",
        "per_class",
        "feature_importance",
        "residual_sample",
        "input_schema",
        "effective_hyperparameters",
        "environment",
        "created_at",
        "completed_at",
    )
    ordering = ("-created_at",)


@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = ("__str__", "kind", "status", "progress", "cancel_requested", "created_at", "finished_at")
    list_filter = ("kind", "status", "created_at")
    readonly_fields = (
        "id",
        "payload",
        "progress",
        "total_units",
        "completed_units",
        "error",
        "suggested_action",
        "created_at",
        "started_at",
        "finished_at",
    )
    ordering = ("-created_at",)


@admin.register(Model)
class ModelAdmin(admin.ModelAdmin):
    list_display = ("name", "task_type", "production_version", "created_at")
    list_filter = ("task_type", "created_at")
    search_fields = ("name",)
    readonly_fields = ("id", "created_at")
    ordering = ("-created_at",)


@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = ("__str__", "status", "created_at")
    list_filter = ("status", "created_at")
    search_fields = ("model__name",)
    readonly_fields = ("id", "signature", "metrics", "model_card", "created_at")
    ordering = ("-created_at",)


@admin.register(Deployment)
class DeploymentAdmin(admin.ModelAdmin):
    list_display = ("slug", "model_version", "is_active", "total_requests", "last_predicted_at", "created_at")
    list_filter = ("is_active", "created_at")
    search_fields = ("slug",)
    readonly_fields = ("id", "baseline_stats", "total_requests", "total_errors", "created_at")
    ordering = ("-created_at",)


@admin.register(MonitoringRecord)
class MonitoringRecordAdmin(admin.ModelAdmin):
    list_display = ("kind", "deployment", "created_at")
    list_filter = ("kind", "created_at")
    readonly_fields = ("payload", "created_at")
    ordering = ("-created_at",)

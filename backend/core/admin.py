"""Django admin registrations for browsing datasets and training runs."""

from django.contrib import admin

from core.models import Dataset, TrainingRun


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    """Read-mostly view over uploaded and derived datasets."""

    list_display = ("name", "source", "n_rows", "n_cols", "size_bytes", "created_at")
    list_filter = ("source", "created_at")
    search_fields = ("name",)
    readonly_fields = ("id", "created_at", "columns")
    ordering = ("-created_at",)


@admin.register(TrainingRun)
class TrainingRunAdmin(admin.ModelAdmin):
    """Read-mostly view over training runs and their recorded metrics."""

    list_display = (
        "display_name",
        "algorithm",
        "task_type",
        "target_column",
        "status",
        "created_at",
    )
    list_filter = ("task_type", "status", "algorithm", "created_at")
    search_fields = ("name", "algorithm", "target_column")
    readonly_fields = (
        "id",
        "created_at",
        "metrics",
        "train_metrics",
        "cv",
        "confusion_matrix",
        "per_class",
        "feature_importance",
        "residual_sample",
        "input_schema",
        "effective_hyperparameters",
    )
    ordering = ("-created_at",)

    @admin.display(description="Run")
    def display_name(self, obj: TrainingRun) -> str:
        """Fall back to algorithm + target when the run has no name."""
        return obj.display_name

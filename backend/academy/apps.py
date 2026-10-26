from django.apps import AppConfig


class AcademyConfig(AppConfig):
    """Accounts, saved course progress and the per-student notebook server."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "academy"
    verbose_name = "Academy"

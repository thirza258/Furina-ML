from django.apps import AppConfig


class LifecycleConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "lifecycle"
    verbose_name = "ML experiment lifecycle"

    def ready(self) -> None:
        """Start the in-process job worker, unless we should not.

        The poller is what keeps training off the HTTP request path on a
        deployment that has not added a Redis worker yet. It refuses to start
        under management commands that would misbehave with a background
        thread (``test`` is the big one, and ``migrate``/``makemigrations``
        have no business training anything), and a deployment can set
        ``FURINA_JOB_WORKERS=0`` to hand the queue to an external runner.
        """
        import os
        import sys

        if os.getenv("FURINA_DISABLE_JOB_POLLER") == "1":
            return
        command = sys.argv[1] if len(sys.argv) > 1 else ""
        if command and command not in ("runserver", "daphne", "serve"):
            # gunicorn's argv carries no subcommand, so a real server falls
            # through and starts the poller; `manage.py <anything>` does not.
            return
        try:
            from lifecycle.worker import start_poller

            start_poller()
        except Exception:  # noqa: BLE001 - a poller failure must never stop boot
            import logging

            logging.getLogger(__name__).exception("Job poller failed to start; jobs stay queued.")

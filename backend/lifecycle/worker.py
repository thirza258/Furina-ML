"""The job worker: who actually runs a queued ``Job``.

The PRD's target architecture puts a Redis queue in front of an ML worker;
this module is the seam where that queue plugs in. Until the deployment adds
one, the runner here keeps the platform's promise — *long training never runs
inside an HTTP request* — with no new infrastructure:

* :func:`start_poller` starts a daemon thread at boot. Each loop it claims a
  queued job and executes it, so the request handler that created the job
  already returned.
* :func:`claim_next_job` makes the claim atomic in the database
  (``select_for_update`` + status flip inside one transaction), so any number
  of pollers — gunicorn workers, a standalone runner, a future RQ worker —
  can run side by side without a job running twice.
* Cancellation is cooperative: the flag on the job is what a worker checks
  between trials, so a cancel takes effect at the next checkpoint even in a
  separate process.

Moving to Redis/RQ means replacing :func:`poll_once` with a queue consumer;
nothing above it (services, models, endpoints) changes.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from django.db import close_old_connections, transaction
from django.utils import timezone

logger = logging.getLogger(__name__)

_poller_thread: threading.Thread | None = None
_stop_event = threading.Event()

#: A job claimed but not finished this long is considered stuck and may be
#: reclaimed by another poller. Generously above the slowest legitimate
#: single trial; a real deployment tunes it per workload.
STALE_JOB_SECONDS = 1800

#: How long the poller sleeps between empty queue checks.
IDLE_SECONDS = 2.0


def start_poller() -> bool:
    """Start the in-process job poller. Returns True when it started now.

    A second call is a no-op. Starting under ``manage.py test`` is refused
    outright (apps.ready checks the command first), because tests drive jobs
    synchronously through ``run_now=True`` and a background thread racing
    them would flake the suite.
    """
    global _poller_thread
    if _poller_thread is not None and _poller_thread.is_alive():
        return False
    _stop_event.clear()
    _poller_thread = threading.Thread(
        target=_loop, name="furina-job-poller", daemon=True
    )
    _poller_thread.start()
    logger.info("Furina job poller started.")
    return True


def stop_poller() -> None:  # pragma: no cover - shutdown path
    """Signal the poller to stop and wait briefly for it to exit."""
    _stop_event.set()
    if _poller_thread is not None:
        _poller_thread.join(timeout=10)


def _loop() -> None:
    while not _stop_event.is_set():
        try:
            job = poll_once()
            if job is not None:
                _run_claimed(job)
        except Exception:  # noqa: BLE001 - the poller must survive any single failure
            logger.exception("Job poll loop failed; retrying.")
        _stop_event.wait(timeout=IDLE_SECONDS)


def poll_once() -> Any:
    """Claim and return the next queued job, or None when the queue is empty."""
    from lifecycle.models import Job

    return claim_next_job(Job)


def claim_next_job(job_model: Any) -> Any:
    """Atomically claim one queued job. Safe to call from many processes."""
    with transaction.atomic():
        candidate = (
            job_model.objects.select_for_update(skip_locked=True)
            .filter(status="queued")
            .order_by("created_at")
            .first()
        )
        if candidate is None:
            return None
        candidate.status = "running"
        candidate.started_at = timezone.now()
        candidate.last_heartbeat = candidate.started_at
        candidate.save(update_fields=["status", "started_at", "last_heartbeat"])
        return candidate


def _run_claimed(job: Any) -> None:
    """Execute a claimed job and record its outcome, whatever happens."""
    from lifecycle.services import execute_job

    try:
        close_old_connections()
        execute_job(job)
    except Exception:  # noqa: BLE001 - execute_job catches user errors; this is the net
        logger.exception("Job %s raised out of the worker.", job.id)
        try:
            job.status = "failed"
            job.error = "The job runner stopped unexpectedly."
            job.finished_at = timezone.now()
            job.save()
        except Exception:  # pragma: no cover - a second failure is logged, not raised
            logger.exception("Could not mark job %s failed.", job.id)


def reclaim_stale_jobs() -> int:
    """Put stuck running jobs back in the queue; returns how many were reclaimed.

    A worker that dies mid-job leaves a row in ``running`` forever; anything
    whose heartbeat is older than :data:`STALE_JOB_SECONDS` almost certainly
    belongs to a dead process.
    """
    from datetime import timedelta

    from lifecycle.models import Job

    cutoff = timezone.now() - timedelta(seconds=STALE_JOB_SECONDS)
    return Job.objects.filter(status="running", last_heartbeat__lt=cutoff).update(
        status="queued", started_at=None, progress=0.0
    )

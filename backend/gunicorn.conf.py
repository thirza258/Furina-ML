"""Gunicorn configuration for the Furina ML API.

Read from the environment so the same image can be tuned per deployment
without a rebuild.

Two settings here are not defaults for a reason:

* ``timeout`` -- fitting a scikit-learn model is a synchronous, CPU-bound
  request that routinely runs for minutes. Gunicorn's stock 30s would SIGKILL
  the worker mid-fit and hand the browser a 502. The frontend's axios client
  waits 300s, so the whole chain (nginx -> gunicorn -> axios) is set to match.
* ``threads`` with the ``gthread`` worker -- one in-flight training request
  occupies its handler for its whole duration. Without spare handlers the
  health check and the rest of the UI would queue behind it.
"""

import os


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


bind = os.getenv("GUNICORN_BIND", "0.0.0.0:8000")
workers = _int("GUNICORN_WORKERS", 2)
worker_class = os.getenv("GUNICORN_WORKER_CLASS", "gthread")
threads = _int("GUNICORN_THREADS", 4)

timeout = _int("GUNICORN_TIMEOUT", 300)
graceful_timeout = _int("GUNICORN_GRACEFUL_TIMEOUT", 30)
keepalive = _int("GUNICORN_KEEPALIVE", 5)

# nginx is the only client, and it buffers, so a large upload never trickles in.
max_requests = _int("GUNICORN_MAX_REQUESTS", 0)
max_requests_jitter = _int("GUNICORN_MAX_REQUESTS_JITTER", 0)

# Container logging: everything to stdout/stderr for `docker compose logs`.
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")

# nginx sets X-Forwarded-For / -Proto; trust it since nothing else can reach us.
forwarded_allow_ips = os.getenv("GUNICORN_FORWARDED_ALLOW_IPS", "*")

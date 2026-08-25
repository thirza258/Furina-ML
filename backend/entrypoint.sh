#!/bin/sh
#
# Container entrypoint for the Furina ML backend.
#
# Waits for the database to accept connections, applies migrations, optionally
# creates an admin user, then hands control to the command in CMD.
set -eu

log() { printf '[entrypoint] %s\n' "$*"; }

if [ "${SKIP_DB_WAIT:-False}" != "True" ]; then
    log "waiting for the database..."
    ATTEMPTS="${DB_WAIT_ATTEMPTS:-60}"
    python - "$ATTEMPTS" <<'PY'
import sys
import time

import django
from django.db import connections
from django.db.utils import OperationalError

django.setup()

attempts = int(sys.argv[1])
for attempt in range(1, attempts + 1):
    try:
        connections["default"].ensure_connection()
    except OperationalError as exc:
        if attempt == attempts:
            print(f"[entrypoint] database never became available: {exc}", file=sys.stderr)
            raise SystemExit(1)
        time.sleep(1)
    else:
        connections["default"].close()
        print(f"[entrypoint] database is up (after {attempt}s)")
        break
PY
fi

if [ "${RUN_MIGRATIONS:-True}" = "True" ]; then
    log "applying migrations"
    python manage.py migrate --noinput
fi

# Optional, and skipped silently when the variables are absent. `createsuperuser
# --noinput` exits non-zero when the user already exists, which is fine on the
# second boot.
if [ -n "${DJANGO_SUPERUSER_USERNAME:-}" ] && [ -n "${DJANGO_SUPERUSER_PASSWORD:-}" ]; then
    log "ensuring superuser '${DJANGO_SUPERUSER_USERNAME}' exists"
    python manage.py createsuperuser --noinput \
        --username "${DJANGO_SUPERUSER_USERNAME}" \
        --email "${DJANGO_SUPERUSER_EMAIL:-admin@example.com}" 2>/dev/null \
        || log "superuser already present, leaving it alone"
fi

log "starting: $*"
exec "$@"

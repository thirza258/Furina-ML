# syntax=docker/dockerfile:1
#
# Furina ML backend -- Django + scikit-learn behind gunicorn.
#
# 3.13 is the newest interpreter with prebuilt manylinux wheels for every pin in
# requirements.txt (pandas 3.0.5, scikit-learn 1.9.0, numpy 2.3.4, scipy
# 1.18.1, psycopg2-binary 2.9.11). On a version without them, pip would fall
# back to compiling numpy and scipy from source.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DJANGO_SETTINGS_MODULE=furinaml_settings.settings

WORKDIR /app

# libgomp1 is the OpenMP runtime that scikit-learn's and scipy's compiled
# extensions link against; the slim image does not ship it.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Dependencies first, so an application edit does not reinstall the ML stack.
# The cache mount lives outside the image, so re-running a build that failed
# part-way through does not re-download ~350MB of wheels; the generous retry
# and timeout settings are for exactly that kind of flaky link.
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 10 --timeout 180 -r requirements.txt

COPY . .
RUN chmod +x /app/entrypoint.sh

# Collected at build time, not boot time: STORAGES uses whitenoise's manifest
# storage, so the manifest has to exist before the first request. No database
# is needed -- settings.py gives `collectstatic` the dummy backend.
RUN DEVELOPMENT_MODE=False DEBUG=False SECRET_KEY=build-time-only \
    python manage.py collectstatic --noinput --clear

# Runs unprivileged. The mkdir/chown must happen here, before USER: a named
# volume mounted at /app/media inherits its ownership from the image's
# directory the first time it is created, and a root-owned one would make
# every upload fail with EACCES.
RUN groupadd --system furina \
 && useradd --system --gid furina --home-dir /app --shell /usr/sbin/nologin furina \
 && mkdir -p /app/media/datasets /app/media/models /app/staticfiles \
 && chown -R furina:furina /app
USER furina

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=10s --start-period=40s --retries=5 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/api/health/', timeout=5).read()" || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["gunicorn", "--config", "/app/gunicorn.conf.py", "furinaml_settings.wsgi:application"]

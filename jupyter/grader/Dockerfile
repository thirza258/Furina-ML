# syntax=docker/dockerfile:1
#
# Furina grader worker: claims submitted notebooks from Django and grades each
# one in a throwaway, network-less container of the student image. See
# furina_grader/worker.py.
#
# Built with the lesson templates as an extra build context:
#   docker build --build-context notebooks=../../notebooks .

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
 && rm /tmp/requirements.txt \
 && useradd --create-home --uid 1000 grader

# The same templates the student image is built with: a submission is graded
# against the template its lesson links to, never against the student's copy.
COPY --from=notebooks . /opt/furina/notebooks
RUN rm -rf /opt/furina/notebooks/tools /opt/furina/notebooks/README.md
COPY furina_grader /opt/furina/grader/furina_grader

USER grader
WORKDIR /opt/furina/grader
CMD ["python", "-m", "furina_grader.worker"]

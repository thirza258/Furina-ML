"""Automated grading of submitted lesson notebooks: what Django's side does.

Django only queues submissions and stores results. The grader worker
(``jupyter/grader``) does the grading: it claims a queued submission over the
internal API below, rebuilds the notebook from the course template with only
the student's exercise cells swapped in, runs it in a sandbox container with
no network, and posts the per-check results back.

The internal API is authenticated by a shared secret, ``GRADER_TOKEN``, and
nginx does not route it from outside: the worker calls Django directly.
"""

from __future__ import annotations

import hmac
import json

from django.conf import settings
from rest_framework.permissions import BasePermission

#: The most cells and source characters kept from one submitted notebook.
MAX_CELLS = 1000
MAX_CELL_CHARS = 100_000


def grading_enabled() -> bool:
    """True when a grader worker is configured to serve this site."""
    return bool(settings.GRADER_TOKEN)


class HasGraderToken(BasePermission):
    """The caller presents the grader worker's shared secret as a bearer token."""

    message = "A valid grader token is required."

    def has_permission(self, request, view) -> bool:
        expected = settings.GRADER_TOKEN
        header = request.headers.get("Authorization", "")
        scheme, _, token = header.partition(" ")
        if not expected or scheme.lower() != "bearer" or not token:
            return False
        return hmac.compare_digest(token.strip().encode(), expected.encode())


class InvalidNotebook(ValueError):
    """The upload is not a Jupyter notebook the grader can read."""


def code_cells(raw) -> dict:
    """Parse an uploaded ``.ipynb`` and keep only what grading needs.

    ``raw`` is the file's bytes or text, or an already-parsed notebook. The
    result is a minimal notebook of code cells (id and source), so outputs --
    plots, printed data -- are never stored.
    """
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise InvalidNotebook("The file is not UTF-8 text; upload the .ipynb file itself.") from None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raise InvalidNotebook("The file is not a Jupyter notebook (.ipynb).") from None
    if not isinstance(raw, dict) or not isinstance(raw.get("cells"), list):
        raise InvalidNotebook("The file is not a Jupyter notebook (.ipynb).")
    if raw.get("nbformat") != 4:
        raise InvalidNotebook("Only notebooks in the current format (nbformat 4) can be graded.")

    cells = []
    for cell in raw["cells"]:
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(part for part in source if isinstance(part, str))
        if not isinstance(source, str):
            raise InvalidNotebook("A code cell in the notebook is malformed.")
        if len(source) > MAX_CELL_CHARS:
            raise InvalidNotebook("One code cell is too long to grade.")
        cell_id = cell.get("id")
        cells.append(
            {
                "cell_type": "code",
                "id": cell_id if isinstance(cell_id, str) and len(cell_id) <= 64 else None,
                "source": source,
            }
        )
        if len(cells) > MAX_CELLS:
            raise InvalidNotebook("The notebook has too many cells to grade.")
    if not cells:
        raise InvalidNotebook("The notebook has no code cells.")
    return {"nbformat": 4, "nbformat_minor": 5, "cells": cells}

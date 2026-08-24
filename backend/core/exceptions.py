"""Project-wide DRF exception handling.

Guarantees the error envelope the API contract promises:
``400 -> {"detail": ..., "errors": {...}}``, ``404 -> {"detail": "Not found."}``
and ``500 -> {"detail": ...}`` -- never an HTML traceback, never a bare list.
"""

from __future__ import annotations

import logging
from typing import Any

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import exception_handler as drf_exception_handler

logger = logging.getLogger(__name__)

GENERIC_SERVER_ERROR = (
    "Something went wrong while handling this request. Try again, and if it keeps "
    "happening simplify the request or upload the file again."
)


def _flatten_detail(payload: Any) -> str:
    """Pick a single human sentence out of an arbitrary DRF error payload."""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("detail", "non_field_errors"):
            if key in payload:
                return _flatten_detail(payload[key])
        for value in payload.values():
            return _flatten_detail(value)
        return "The request could not be processed."
    if isinstance(payload, (list, tuple)):
        for item in payload:
            return _flatten_detail(item)
        return "The request could not be processed."
    return str(payload)


def _normalize_errors(payload: Any) -> dict | None:
    """Return a ``{field: [message]}`` map when the payload carries one."""
    if not isinstance(payload, dict):
        return None
    errors: dict[str, list[str]] = {}
    for field, value in payload.items():
        if field == "detail":
            continue
        if isinstance(value, (list, tuple)):
            errors[str(field)] = [str(item) for item in value]
        elif isinstance(value, dict):
            errors[str(field)] = [_flatten_detail(value)]
        else:
            errors[str(field)] = [str(value)]
    return errors or None


def api_exception_handler(exc: Exception, context: dict) -> Response:
    """Convert any exception raised inside a DRF view into the JSON envelope."""
    response = drf_exception_handler(exc, context)

    if response is None:
        logger.exception("Unhandled error in %s", context.get("view"))
        return Response(
            {"detail": GENERIC_SERVER_ERROR},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    if response.status_code == status.HTTP_404_NOT_FOUND:
        response.data = {"detail": "Not found."}
        return response

    payload: dict = {"detail": _flatten_detail(response.data)}
    errors = _normalize_errors(response.data)
    if errors:
        payload["errors"] = errors
    response.data = payload
    return response

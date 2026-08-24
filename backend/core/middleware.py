"""Project middleware that keeps API responses inside the JSON envelope."""

from __future__ import annotations

from typing import Any, Callable

from django.http import HttpRequest, HttpResponse, JsonResponse

API_PREFIX = "/api/"


class JsonApiNotFoundMiddleware:
    """Answer every unresolved ``/api/`` path with ``{"detail": "Not found."}``.

    A path that does not match any route -- a malformed uuid, a typo -- fails
    URL resolution before any view runs, so Django replies with its own HTML
    404 page and the client gets markup where it expects the error envelope.
    This turns those responses back into JSON; 404s raised inside a DRF view
    are already JSON and are passed through untouched.
    """

    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]) -> None:
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> Any:
        response = self.get_response(request)
        if response.status_code != 404 or not request.path.startswith(API_PREFIX):
            return response
        content_type = str(response.headers.get("Content-Type", ""))
        if "json" in content_type.lower():
            return response
        return JsonResponse({"detail": "Not found."}, status=404)

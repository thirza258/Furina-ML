"""Read datasets and training runs from the Furina ML workbench in a notebook.

The same module works in both places the course notebooks run:

* **JupyterLite**, where Python runs inside the browser (Pyodide). Requests go
  to the Furina API on the page's own origin, so they need no configuration.
* **The Jupyter server container**, where Python runs on the server. Requests
  go to ``FURINA_API_URL`` (the compose file points it at the frontend
  container) and, when ``FURINA_API_HOST`` is set, carry that ``Host`` header
  so a Django ``ALLOWED_HOSTS`` list that names only the public domain still
  accepts them.

Everything here is read-only: it lists and downloads, it never uploads,
trains or deletes. Network problems and API errors are raised as
:class:`FurinaError` with a sentence a learner can act on.

Example::

    import furina

    for dataset in furina.list_datasets():
        print(dataset["name"], dataset["n_rows"], "rows")
    frame = furina.load_dataset("houses")      # a dataset id or its exact name
"""

from __future__ import annotations

import io
import json
import os
import sys
from typing import Any
from urllib.parse import quote, urlencode

__all__ = [
    "FurinaError",
    "api_url",
    "configure",
    "get_run",
    "list_datasets",
    "list_runs",
    "load_dataset",
]

_IN_BROWSER = sys.platform == "emscripten"
_TIMEOUT_SECONDS = 60
_settings: dict[str, str | None] = {"api_url": None, "host": None}


class FurinaError(RuntimeError):
    """The workbench could not be reached, or it refused the request."""


def configure(api_url: str | None = None, host: str | None = None) -> None:
    """Point the helper at a different API, for example a split deployment.

    ``api_url`` is the address that ends in ``/api``. ``host`` overrides the
    ``Host`` header outside the browser; browsers do not allow that, so it is
    ignored in JupyterLite.
    """
    if api_url is not None:
        _settings["api_url"] = api_url.rstrip("/")
    if host is not None:
        _settings["host"] = host or None


def api_url() -> str:
    """The API address requests are sent to."""
    if _settings["api_url"]:
        return str(_settings["api_url"])
    configured = os.environ.get("FURINA_API_URL", "").strip()
    if configured:
        return configured.rstrip("/")
    if _IN_BROWSER:
        from js import location  # type: ignore[import-not-found]

        return f"{location.origin}/api"
    return "http://localhost:8000/api"


def _host_header() -> str | None:
    return _settings["host"] or os.environ.get("FURINA_API_HOST", "").strip() or None


def _get(path: str, params: dict[str, str] | None = None) -> tuple[str, str]:
    """GET one API path and return ``(content_type, body_text)``."""
    url = f"{api_url()}/{path.lstrip('/')}"
    if params:
        url = f"{url}?{urlencode(params)}"
    if _IN_BROWSER:
        return _get_in_browser(url)
    return _get_on_server(url)


def _get_in_browser(url: str) -> tuple[str, str]:
    # Pyodide runs the kernel in a web worker, where a synchronous request is
    # allowed. That keeps every function here an ordinary call instead of a
    # coroutine the learner would have to await.
    from js import XMLHttpRequest  # type: ignore[import-not-found]

    request = XMLHttpRequest.new()
    try:
        request.open("GET", url, False)
        request.send(None)
    except Exception as exc:  # noqa: BLE001 - the browser reports a JS error
        raise FurinaError(
            f"The Furina API at {url} could not be reached from this notebook: {exc}"
        ) from None
    status = int(request.status)
    body = str(request.responseText)
    if status == 0:
        raise FurinaError(
            f"The Furina API at {url} did not answer. Check that the workbench is running."
        )
    if status >= 400:
        raise FurinaError(_error_message(url, status, body))
    return str(request.getResponseHeader("Content-Type") or ""), body


def _get_on_server(url: str) -> tuple[str, str]:
    import urllib.error
    import urllib.request

    headers = {"Accept": "application/json, text/csv;q=0.9"}
    host = _host_header()
    if host:
        headers["Host"] = host
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.headers.get("Content-Type", ""), response.read().decode(charset)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise FurinaError(_error_message(url, exc.code, body)) from None
    except urllib.error.URLError as exc:
        raise FurinaError(
            f"The Furina API at {url} could not be reached: {exc.reason}. Set "
            "FURINA_API_URL to the address that ends in /api."
        ) from None


def _error_message(url: str, status: int, body: str) -> str:
    detail = ""
    try:
        detail = str(json.loads(body).get("detail") or "")
    except (ValueError, AttributeError):
        pass
    if status == 400 and "Invalid HTTP_HOST" in body:
        return (
            f"The Furina API at {url} rejected this server's Host header (HTTP 400). "
            "Set FURINA_API_HOST to a name listed in the backend's ALLOWED_HOSTS."
        )
    if status == 404:
        return detail or f"Nothing was found at {url} (HTTP 404)."
    return f"The Furina API answered HTTP {status} for {url}." + (f" {detail}" if detail else "")


def _get_json(path: str, params: dict[str, str] | None = None) -> Any:
    _, body = _get(path, params)
    try:
        return json.loads(body)
    except ValueError:
        raise FurinaError(
            f"{api_url()}/{path.lstrip('/')} did not return JSON. Check that "
            "FURINA_API_URL points at the Furina API and ends in /api."
        ) from None


def list_datasets() -> list[dict]:
    """Every dataset in the workbench, newest first.

    Each entry has ``id``, ``name``, ``n_rows``, ``n_cols``, ``source`` and
    ``created_at``.
    """
    return list(_get_json("datasets/"))


def _resolve_dataset_id(dataset: str) -> str:
    wanted = str(dataset).strip()
    datasets = list_datasets()
    for entry in datasets:
        if str(entry.get("id")) == wanted:
            return wanted
    matches = [entry for entry in datasets if str(entry.get("name", "")).lower() == wanted.lower()]
    if len(matches) == 1:
        return str(matches[0]["id"])
    if len(matches) > 1:
        ids = ", ".join(str(entry["id"]) for entry in matches)
        raise FurinaError(f"More than one dataset is called '{wanted}'. Use one of these ids: {ids}.")
    names = ", ".join(f"'{entry.get('name')}'" for entry in datasets) or "none yet"
    raise FurinaError(f"No dataset has the id or name '{wanted}'. Available datasets: {names}.")


def load_dataset(dataset: str):
    """Download one dataset as a pandas DataFrame, by id or by exact name.

    The CSV is the one stored in the workbench, so a cleaned dataset saved
    from Data Preprocessing loads with its cleaning already applied.
    """
    import pandas as pd

    dataset_id = _resolve_dataset_id(dataset)
    _, text = _get(f"datasets/{quote(dataset_id)}/download/")
    return pd.read_csv(io.StringIO(text))


def list_runs(dataset: str | None = None) -> list[dict]:
    """Training runs, newest first, optionally only those for one dataset.

    Each run carries its algorithm, task type, test metrics and train metrics,
    so a model built in a notebook can be compared with one from the workbench.
    """
    params = {"dataset_id": _resolve_dataset_id(dataset)} if dataset is not None else None
    return list(_get_json("runs/", params))


def get_run(run_id: str) -> dict:
    """The full evaluation of one training run."""
    return dict(_get_json(f"runs/{quote(str(run_id).strip())}/"))

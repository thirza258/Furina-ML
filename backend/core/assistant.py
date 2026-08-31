"""The in-app helper, proxied to OpenRouter.

Why this lives on the server at all: an API key shipped to the browser is a
published API key. Anything in the Vite bundle is readable by anyone who opens
the network tab, so the key stays here and the browser talks only to us.

What gets sent, and what does not:

* The *shape* of the user's data -- column names, inferred kinds, row and
  column counts, missing-value counts -- and the metrics of a finished run.
  This is what lets the helper answer "which column should I predict?" with
  reference to the actual dataset rather than in the abstract.
* Never a single row of the data itself. The helper is told about the columns,
  not what is in them.

Column names can still be sensitive, so the UI says plainly what leaves the
machine, and the whole feature is inert until an API key is configured.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from django.conf import settings

#: Hard ceilings, so a wide dataset cannot push a huge prompt to a paid API.
MAX_COLUMNS_IN_CONTEXT = 40
MAX_HISTORY_MESSAGES = 12
MAX_MESSAGE_CHARS = 4000


class AssistantError(Exception):
    """Anything the user should be told about in words rather than a traceback."""


class AssistantNotConfigured(AssistantError):
    """No API key, so the feature is switched off rather than broken."""


def is_configured() -> bool:
    """True when an OpenRouter key is present."""
    return bool(getattr(settings, "OPENROUTER_API_KEY", ""))


# --------------------------------------------------------------------------
# Prompt assembly
# --------------------------------------------------------------------------

_BASE_PROMPT = """You are the built-in helper for Furina ML, a web app for training \
machine-learning models on tabular CSV data without writing code.

How the app works, so you can answer accurately:
- Three steps, in order: Data Preprocessing (upload a CSV, read its profile, \
compose a cleaning recipe), Data Training (pick a target column, feature \
columns and an algorithm, set the train/test split), Model Evaluation (read the \
metrics, feature importances, and try live predictions). There is also a \
Dashboard that ranks past runs against each other.
- Models are real scikit-learn, XGBoost and LightGBM estimators fitted on the \
server. Nothing is simulated.
- Every run builds one Pipeline: a ColumnTransformer (median imputation and \
optional scaling for numeric columns; most-frequent imputation and one-hot \
encoding for categorical ones) followed by the chosen estimator. Because \
preprocessing lives inside the pipeline it is fitted on the training split \
only, so the test split never leaks into it.
- Metrics shown to the user are computed on the held-out test split. Train-split \
metrics are shown beside them so overfitting is visible.

How to answer:
- Be concise and concrete. Two or three short paragraphs at most, usually less.
- Use the context block below when it is relevant; refer to the user's actual \
column names and scores rather than generic examples.
- If the context does not contain what you need, say so and tell the user which \
page would show it. Never invent a number, a column name or a score.
- You cannot click anything or change the app. You can explain what a control \
does and what the user should choose, but tell them to do it themselves."""


def _dataset_context(dataset: Any) -> str:
    columns = list(dataset.columns or [])[:MAX_COLUMNS_IN_CONTEXT]
    lines = [
        f"Active dataset: {dataset.name}",
        f"  {dataset.n_rows} rows x {dataset.n_cols} columns (source: {dataset.source})",
    ]
    if columns:
        lines.append("  Columns (name / kind / missing / distinct):")
        for column in columns:
            lines.append(
                "    - {name} / {kind} / {missing} missing / {unique} distinct{target}".format(
                    name=column.get("name"),
                    kind=column.get("kind", "unknown"),
                    missing=column.get("missing", 0),
                    unique=column.get("unique", 0),
                    target="" if column.get("suitable_as_target", True) else " (not usable as a target)",
                )
            )
        hidden = len(dataset.columns or []) - len(columns)
        if hidden > 0:
            lines.append(f"    ... and {hidden} more columns not listed here")
    return "\n".join(lines)


def _run_context(run: Any) -> str:
    lines = [
        f"Most recent training run: {run.algorithm_label or run.algorithm}",
        f"  task: {run.task_type}, target column: {run.target_column}",
        f"  status: {run.status}",
    ]
    if run.status == "failed" and run.error:
        lines.append(f"  it failed with: {run.error}")
    metrics = dict(run.metrics or {})
    train_metrics = dict(run.train_metrics or {})
    if metrics:
        pairs = ", ".join(
            f"{key}={value:.4f}" for key, value in metrics.items()
            if isinstance(value, (int, float))
        )
        lines.append(f"  test-split metrics: {pairs}")
    if train_metrics:
        pairs = ", ".join(
            f"{key}={value:.4f}" for key, value in train_metrics.items()
            if isinstance(value, (int, float))
        )
        lines.append(f"  train-split metrics: {pairs}")
    if run.warnings:
        lines.append(f"  warnings raised: {'; '.join(str(w) for w in run.warnings[:5])}")
    return "\n".join(lines)


def build_system_prompt(*, tool: str | None, dataset: Any = None, run: Any = None) -> str:
    """The base prompt plus whatever the user is actually looking at."""
    blocks = [_BASE_PROMPT, "", "--- Current context ---"]
    blocks.append(f"The user is on the {tool or 'unknown'} page.")
    blocks.append(
        _dataset_context(dataset) if dataset is not None
        else "No dataset is loaded yet. The user has to upload a CSV on the Data Preprocessing page first."
    )
    blocks.append(
        _run_context(run) if run is not None
        else "No model has been trained yet in this workspace."
    )
    return "\n".join(blocks)


# --------------------------------------------------------------------------
# The call itself
# --------------------------------------------------------------------------


def sanitize_history(messages: Any) -> list[dict[str, str]]:
    """Keep the last few well-formed turns, and nothing else.

    The client sends the conversation back on every request, so this is an
    untrusted list: only `user` and `assistant` roles survive, which stops a
    crafted request from smuggling in its own system prompt.
    """
    clean: list[dict[str, str]] = []
    for message in list(messages or [])[-MAX_HISTORY_MESSAGES:]:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", ""))
        content = str(message.get("content", "")).strip()
        if role not in ("user", "assistant") or not content:
            continue
        clean.append({"role": role, "content": content[:MAX_MESSAGE_CHARS]})
    return clean


def ask(system_prompt: str, history: list[dict[str, str]]) -> str:
    """Send one completion request to OpenRouter and return the reply text."""
    if not is_configured():
        raise AssistantNotConfigured(
            "The assistant is not configured on this server. Set OPENROUTER_API_KEY "
            "to switch it on."
        )
    if not history:
        raise AssistantError("Ask a question and the assistant will answer it.")

    payload = {
        "model": settings.OPENROUTER_MODEL,
        "messages": [{"role": "system", "content": system_prompt}, *history],
        "max_tokens": settings.OPENROUTER_MAX_TOKENS,
        "temperature": 0.2,
    }
    request = urllib.request.Request(
        f"{settings.OPENROUTER_BASE_URL.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            # OpenRouter uses these for its own attribution listings.
            "HTTP-Referer": settings.SITE_URL,
            "X-Title": "Furina ML",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=settings.OPENROUTER_TIMEOUT) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = _describe_http_error(exc)
        raise AssistantError(detail) from None
    except urllib.error.URLError as exc:
        raise AssistantError(
            f"The assistant could not reach OpenRouter: {exc.reason}."
        ) from None
    except (TimeoutError, OSError):
        raise AssistantError(
            "The assistant took too long to answer. Try a shorter question."
        ) from None
    except json.JSONDecodeError:
        raise AssistantError("OpenRouter returned a response that could not be read.") from None

    try:
        return str(body["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError):
        raise AssistantError("OpenRouter returned no answer.") from None


def _describe_http_error(exc: urllib.error.HTTPError) -> str:
    """Turn OpenRouter's error body into one sentence, without leaking the key."""
    try:
        payload = json.loads(exc.read().decode("utf-8"))
        message = payload.get("error", {}).get("message") or payload.get("message")
    except Exception:  # noqa: BLE001 - the body is whatever it is
        message = None
    if exc.code == 401:
        return "OpenRouter rejected the API key configured on this server."
    if exc.code == 402:
        return "The OpenRouter account for this server is out of credit."
    if exc.code == 429:
        return "The assistant is being rate limited by OpenRouter. Try again in a moment."
    return f"OpenRouter returned an error ({exc.code}): {message or 'no detail given'}."

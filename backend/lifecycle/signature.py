"""The model signature: the inference contract a deployment validates against.

A signature is a JSON blob stored on a ``ModelVersion`` describing exactly
what input a prediction request must carry, and what preprocessing the model
expects. Deployment and batch endpoints use it to reject malformed payloads
*before* they spend CPU, and the generated client examples (curl/python/js)
render from it.
"""

from __future__ import annotations

from typing import Any

from core.ml.io import json_safe

#: How many category values we echo in the signature before it stops being
#: useful and starts being a data dump.
MAX_SIGNABLE_CATEGORIES = 100


def build_signature(
    input_schema: list[dict] | None,
    *,
    task_type: str,
    target_column: str,
    classes: list[str] | None = None,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Turn a run's input schema into a deployable signature.

    ``input_schema`` is the per-feature description the trainer already built
    (name, kind, required, options, min/max, default). This adds the envelope
    a client needs: the task, what it predicts, and the accepted request shape.
    """
    fields = [
        {
            "name": str(field.get("name")),
            "type": str(field.get("kind") or "categorical"),
            "required": bool(field.get("required", True)),
            "categories": None,
            "minimum": field.get("min"),
            "maximum": field.get("max"),
            "default": field.get("default"),
        }
        for field in (input_schema or [])
        if field.get("name")
    ]
    # Echo the categories a categorical field was trained on, capped.
    for field in fields:
        source = next((item for item in (input_schema or []) if item.get("name") == field["name"]), None)
        if source and source.get("options"):
            field["categories"] = [str(option) for option in source["options"]][:MAX_SIGNABLE_CATEGORIES]

    example: dict[str, Any] = {}
    for field in fields:
        value = field["default"]
        if value is None and field["type"] == "numeric":
            value = 0.0
        elif value is None and field["categories"]:
            value = field["categories"][0]
        example[field["name"]] = value

    signature = {
        "task_type": str(task_type),
        "target_column": str(target_column),
        "classes": [str(label) for label in (classes or [])] if task_type == "classification" else None,
        "features": fields,
        "example_request": example,
    }
    return json_safe(signature)


def validate_payload(signature: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    """Return human sentences for every problem with ``payload``.

    Empty means the payload matches the signature. Unknown columns are
    ignored (a client may send extras) but missing *required* columns and
    out-of-range numerics are reported, because those are the failures that
    would otherwise surface as a 500 mid-prediction.
    """
    problems: list[str] = []
    fields = {str(f.get("name")): f for f in (signature.get("features") or [])}
    for name, spec in fields.items():
        if name not in payload:
            if spec.get("required"):
                problems.append(f"Missing required field '{name}'.")
            continue
        value = payload[name]
        ftype = str(spec.get("type") or "categorical")
        if value is None or (isinstance(value, str) and not value.strip()):
            continue  # the pipeline's imputer fills blanks
        if ftype == "numeric":
            try:
                number = float(value)
            except (TypeError, ValueError):
                problems.append(f"Field '{name}' must be a number, got {value!r}.")
                continue
            minimum, maximum = spec.get("minimum"), spec.get("maximum")
            if minimum is not None and number < float(minimum):
                problems.append(f"Field '{name}' is below its training minimum ({float(minimum):g}).")
            if maximum is not None and number > float(maximum):
                problems.append(f"Field '{name}' is above its training maximum ({float(maximum):g}).")
        elif spec.get("categories") and isinstance(value, str):
            categories = {str(c) for c in spec["categories"]}
            if value not in categories:
                # Not fatal -- unseen categories are ignored at predict time --
                # but worth a note so a typo does not silently become a 'missing'.
                problems.append(
                    f"Field '{name}' was not a training category; it will be treated as unseen."
                )
    return problems


def client_snippet(signature: dict[str, Any], deployment: Any, language: str = "python") -> dict[str, str]:
    """cURL / Python / JavaScript examples for calling one deployment.

    ``deployment`` supplies the public URL under which the endpoint is reached
    (its ``predict_url``); the snippet is the request, not the answer.
    """
    base = str(getattr(deployment, "predict_url", "") or "")
    example = json_safe(signature.get("example_request") or {})
    import json as _json

    body = _json.dumps(example, indent=2)
    curl = f"curl -X POST {base} -H 'Content-Type: application/json' -d '{_json.dumps(example)}'"
    python = (
        "import requests\n\n"
        f"response = requests.post(\n    {base!r},\n    json={_json.dumps(example)},\n)\n"
        "response.raise_for_status()\nprint(response.json())"
    )
    javascript = (
        "const response = await fetch(\n  "
        f"{base!r},\n  {{\n    method: 'POST',\n"
        "    headers: { 'Content-Type': 'application/json' },\n"
        f"    body: JSON.stringify({_json.dumps(example)}),\n  }}\n);\n"
        "const result = await response.json();\nconsole.log(result);"
    )
    if language in ("all", "both"):
        return {"curl": curl, "python": python, "javascript": javascript}
    return {language: {"curl": curl, "python": python, "javascript": javascript}.get(language, python)}

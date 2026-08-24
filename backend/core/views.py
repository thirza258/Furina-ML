"""HTTP API for Furina ML.

Every endpoint is a plain ``APIView`` so DRF's page-number pagination stays out
of the way and list endpoints return bare JSON arrays. Anything the user can be
expected to fix (a bad CSV, an unknown column, an impossible target) comes back
as HTTP 400 with a plain sentence in ``detail``; nothing here is allowed to
escape as a traceback.
"""

from __future__ import annotations

import io
import re
import uuid
from typing import Any

import numpy
import pandas
import sklearn
from django.core.files.base import ContentFile
from django.http import FileResponse, HttpResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone
from drf_spectacular.utils import extend_schema
from rest_framework import status
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from core.models import Dataset, TrainingRun
from core.ml import registry
from core.ml.io import DataError, dataframe_preview, json_safe, read_csv, write_csv
from core.ml.predict import PredictionError, predict as run_prediction
from core.ml.preprocessing import PreprocessError, apply_steps
from core.ml.profiling import infer_task, profile_dataframe
from core.ml.report import markdown_report
from core.ml.trainer import TrainingError, dump_artifact, train
from core.serializers import (
    DatasetDetailSerializer,
    DatasetSummarySerializer,
    PredictRequestSerializer,
    PreprocessRequestSerializer,
    TrainRequestSerializer,
    TrainingRunDetailSerializer,
    TrainingRunSummarySerializer,
    UploadSerializer,
    clamp_preview_rows,
)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")

#: Data problems that are always the user's to fix, never a server fault.
USER_ERRORS = (DataError, TrainingError, PredictionError, PreprocessError)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def bad_request(detail: str, errors: dict | None = None) -> Response:
    """Build the contract's 400 envelope."""
    payload: dict[str, Any] = {"detail": str(detail)}
    if errors:
        payload["errors"] = {
            str(field): [str(message) for message in _as_list(messages)]
            for field, messages in errors.items()
        }
    return Response(payload, status=status.HTTP_400_BAD_REQUEST)


def _as_list(value: Any) -> list:
    """Wrap a scalar error message in a list."""
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, dict):
        return [str(value)]
    return [value]


def _error_leaves(value: Any, path: str = "") -> list[tuple[str, str]]:
    """Walk a serializer error payload down to its individual sentences.

    Nested list and dict errors (a bad step inside ``steps``, a blank name
    inside ``feature_columns``) are reported by DRF as nested structures whose
    ``str()`` is a Python repr. This returns ``(field path, message)`` pairs in
    document order instead, so ``{"steps": {0: {"op": [...]}}}`` becomes
    ``[("steps[0].op", "'bogus' is not a preprocessing step we know...")]``.
    """
    if isinstance(value, dict):
        leaves: list[tuple[str, str]] = []
        for key, item in value.items():
            if isinstance(key, int):
                child = f"{path}[{key}]"
            elif str(key) == "non_field_errors" and path:
                child = path
            else:
                child = str(key) if not path else f"{path}.{key}"
            leaves.extend(_error_leaves(item, child))
        return leaves
    if isinstance(value, (list, tuple)):
        leaves = []
        for index, item in enumerate(value):
            # A list of plain sentences belongs to the field itself; a list of
            # nested structures is indexed so the offending element is named.
            child = path if isinstance(item, str) else f"{path}[{index}]"
            leaves.extend(_error_leaves(item, child))
        return leaves
    return [(path, str(value))]


def invalid(serializer) -> Response:
    """Turn serializer errors into the 400 envelope with a readable summary."""
    leaves = _error_leaves(serializer.errors)
    detail = "The request could not be processed."
    if leaves:
        field, message = leaves[0]
        detail = message if field in ("", "non_field_errors") else f"{field}: {message}"
    errors: dict[str, list[str]] = {}
    for field, message in leaves:
        errors.setdefault(field or "non_field_errors", []).append(message)
    return bad_request(detail, errors)


def _safe_filename(name: str, suffix: str) -> str:
    """Build a collision-free storage filename from a user-supplied name."""
    stem = _SAFE_NAME.sub("_", (name or "dataset").strip()).strip("._-") or "dataset"
    for known in (".csv", ".tsv", ".txt"):
        if stem.lower().endswith(known):
            stem = stem[: -len(known)]
            break
    return f"{stem[:60]}_{uuid.uuid4().hex[:8]}{suffix}"


def _roundtrip(frame: "pandas.DataFrame") -> tuple["pandas.DataFrame", bytes]:
    """Serialize a frame the way it is stored and read it straight back.

    The profile and the preview have to describe the bytes that end up in the
    file, not the frame in memory. Writing a CSV and reading it again turns
    some text back into missing values ("NA", "null", an empty string) and can
    settle a column on a different dtype, so profiling the in-memory frame
    would describe data the file does not contain.
    """
    buffer = io.BytesIO()
    write_csv(frame, buffer)
    payload = buffer.getvalue()
    try:
        stored = read_csv(io.BytesIO(payload), delimiter=",")
    except Exception:  # noqa: BLE001 - an unreadable round trip falls back
        return frame, payload
    return stored, payload


def _roundtrip_notes(before: "pandas.DataFrame", after: "pandas.DataFrame") -> list[str]:
    """Explain, in the user's terms, anything saving the file changed.

    A CSV cannot tell an empty cell apart from a missing one, and a column that
    now holds only digits is read back as a number, so a recipe can lose work
    the moment its result is stored. Saying so beats letting the user find out
    from a model.
    """
    notes: list[str] = []
    for column in before.columns:
        if column not in after.columns:
            continue
        source, saved = before[column], after[column]
        lost = int(saved.isna().sum()) - int(source.isna().sum())
        if lost > 0:
            notes.append(
                f"Saving the file turned {lost} value(s) in column '{column}' back into "
                "missing values, because a CSV cannot tell an empty cell apart from a "
                "missing one."
            )
        elif not pandas.api.types.is_numeric_dtype(
            source
        ) and pandas.api.types.is_numeric_dtype(saved):
            notes.append(
                f"Column '{column}' now holds only numbers, so it is stored as a number "
                "column; any leading zeros (for example 01234) are gone."
            )
    if len(notes) > 3:
        extra = len(notes) - 3
        notes = notes[:3] + [f"{extra} other column(s) changed the same way when the file was saved."]
    return notes


def _store_dataframe(
    frame: "pandas.DataFrame",
    *,
    name: str,
    source: str,
    parent: Dataset | None = None,
) -> tuple[Dataset, "pandas.DataFrame"]:
    """Persist a dataframe as a normalized UTF-8 comma CSV plus its profile.

    Returns the saved dataset together with the frame as it will be read back,
    which is the one every later request sees.
    """
    stored, payload = _roundtrip(frame)
    dataset = Dataset(
        name=name,
        source=source,
        parent=parent,
        n_rows=int(len(stored)),
        n_cols=int(stored.shape[1]),
        columns=json_safe(profile_dataframe(stored)),
        size_bytes=len(payload),
    )
    dataset.file.save(_safe_filename(name, ".csv"), ContentFile(payload), save=False)
    dataset.save()
    return dataset, stored


def _dataset_detail(request, dataset: Dataset, frame=None) -> dict:
    """Render a DatasetDetail payload for one dataset."""
    context = {
        "request": request,
        "preview_rows": clamp_preview_rows(request.query_params.get("preview_rows")),
    }
    if frame is not None:
        context["dataframe"] = frame
    return DatasetDetailSerializer(dataset, context=context).data


def _load_frame(dataset: Dataset):
    """Read a stored dataset, translating storage problems into DataError."""
    try:
        return dataset.load_dataframe()
    except DataError:
        raise
    except FileNotFoundError:
        raise DataError("The data file for this dataset could not be found on the server.")
    except OSError:
        raise DataError("The data file for this dataset could not be read.")


# --------------------------------------------------------------------------
# 1. health
# --------------------------------------------------------------------------


class HealthView(APIView):
    """Liveness probe that also reports the installed scientific stack."""

    @extend_schema(responses=dict)
    def get(self, request) -> Response:
        return Response(
            {
                "status": "ok",
                "sklearn": sklearn.__version__,
                "pandas": pandas.__version__,
                "numpy": numpy.__version__,
            }
        )


# --------------------------------------------------------------------------
# 2-6. datasets
# --------------------------------------------------------------------------


class DatasetUploadView(APIView):
    """POST /api/datasets/upload/ -- accept a delimited file and profile it."""

    parser_classes = [MultiPartParser, FormParser]

    @extend_schema(request=UploadSerializer, responses=dict)
    def post(self, request) -> Response:
        serializer = UploadSerializer(data=request.data)
        if not serializer.is_valid():
            return invalid(serializer)

        upload = serializer.validated_data["file"]
        delimiter = serializer.validated_data.get("delimiter") or "auto"
        name = (serializer.validated_data.get("name") or "").strip() or getattr(
            upload, "name", "dataset.csv"
        )

        try:
            frame = read_csv(upload, delimiter=delimiter)
        except DataError as exc:
            return bad_request(str(exc))
        except Exception:  # noqa: BLE001 - anything pandas throws is a bad file
            return bad_request(
                "That file could not be read as a table. Check that it is a CSV with a "
                "header row and try again."
            )

        dataset, stored = _store_dataframe(frame, name=name, source="upload")
        return Response(
            _dataset_detail(request, dataset, stored),
            status=status.HTTP_201_CREATED,
        )


class DatasetListView(APIView):
    """GET /api/datasets/ -- newest first, as a bare array."""

    @extend_schema(responses=list)
    def get(self, request) -> Response:
        datasets = Dataset.objects.all()
        return Response(DatasetSummarySerializer(datasets, many=True).data)


class DatasetDetailView(APIView):
    """GET/DELETE /api/datasets/<uuid>/."""

    @extend_schema(responses=dict)
    def get(self, request, pk) -> Response:
        dataset = get_object_or_404(Dataset, pk=pk)
        return Response(_dataset_detail(request, dataset))

    @extend_schema(responses=None)
    def delete(self, request, pk) -> Response:
        dataset = get_object_or_404(Dataset, pk=pk)
        dataset.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class DatasetDownloadView(APIView):
    """GET /api/datasets/<uuid>/download/ -- the stored CSV as an attachment."""

    @extend_schema(responses=bytes)
    def get(self, request, pk):
        dataset = get_object_or_404(Dataset, pk=pk)
        if not dataset.file:
            return bad_request("This dataset has no stored file to download.")
        try:
            handle = dataset.file.open("rb")
        except (OSError, ValueError):
            return bad_request("The data file for this dataset could not be found on the server.")
        filename = _safe_filename(dataset.name, ".csv")
        response = FileResponse(handle, content_type="text/csv")
        response["Content-Disposition"] = f'attachment; filename="{filename}"'
        return response


class DatasetPreprocessView(APIView):
    """POST /api/datasets/<uuid>/preprocess/ -- run a recipe, optionally dry."""

    parser_classes = [JSONParser, FormParser, MultiPartParser]

    @extend_schema(request=PreprocessRequestSerializer, responses=dict)
    def post(self, request, pk) -> Response:
        dataset = get_object_or_404(Dataset, pk=pk)
        serializer = PreprocessRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return invalid(serializer)

        steps = serializer.validated_data["steps"]
        dry_run = bool(serializer.validated_data.get("dry_run", False))
        requested_name = (serializer.validated_data.get("name") or "").strip()

        try:
            frame = _load_frame(dataset)
            result, report = apply_steps(frame, [dict(step) for step in steps])
        except (DataError, PreprocessError) as exc:
            return bad_request(str(exc))

        report = json_safe(report)

        if dry_run:
            # Preview what saving would produce, so a dry run and the real run
            # describe the same data.
            preview, _ = _roundtrip(result)
            report["warnings"] = list(report.get("warnings") or []) + _roundtrip_notes(
                result, preview
            )
            preview_rows = clamp_preview_rows(request.query_params.get("preview_rows"))
            payload = {
                "id": None,
                "name": requested_name or f"{dataset.name} (preview)",
                "source": "preprocessed",
                "n_rows": int(len(preview)),
                "n_cols": int(preview.shape[1]),
                "created_at": timezone.now().isoformat(),
                "parent": str(dataset.id),
                "size_bytes": 0,
                "columns": json_safe(profile_dataframe(preview)),
                "preview": json_safe(dataframe_preview(preview, preview_rows)),
                "download_url": "",
            }
            return Response({"dataset": json_safe(payload), "report": report})

        new_dataset, stored = _store_dataframe(
            result,
            name=requested_name or f"{dataset.name} (processed)",
            source="preprocessed",
            parent=dataset,
        )
        report["warnings"] = list(report.get("warnings") or []) + _roundtrip_notes(
            result, stored
        )
        return Response(
            {
                "dataset": _dataset_detail(request, new_dataset, stored),
                "report": report,
            }
        )


# --------------------------------------------------------------------------
# 8. algorithms
# --------------------------------------------------------------------------


class AlgorithmListView(APIView):
    """GET /api/algorithms/ -- every algorithm spec, optionally task-filtered."""

    @extend_schema(responses=list)
    def get(self, request) -> Response:
        task = request.query_params.get("task")
        return Response(json_safe(registry.list_algorithms(task)))


# --------------------------------------------------------------------------
# 9. train
# --------------------------------------------------------------------------


class TrainView(APIView):
    """POST /api/train/ -- fit one model and record everything about it."""

    parser_classes = [JSONParser, FormParser, MultiPartParser]

    @extend_schema(request=TrainRequestSerializer, responses=dict)
    def post(self, request) -> Response:
        serializer = TrainRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return invalid(serializer)
        data = serializer.validated_data

        try:
            dataset = Dataset.objects.get(pk=data["dataset_id"])
        except Dataset.DoesNotExist:
            return bad_request("That dataset does not exist. Upload it again or pick another one.")

        try:
            frame = _load_frame(dataset)
        except DataError as exc:
            return bad_request(str(exc))

        target = data["target_column"]
        if target not in frame.columns:
            return bad_request(f"Column '{target}' is not in the dataset.")

        task_type = data.get("task_type") or "auto"
        extra_warnings: list[str] = []
        if task_type == "classification":
            series = frame[target]
            if pandas.api.types.is_numeric_dtype(series) and infer_task(series) == "regression":
                extra_warnings.append(
                    f"Column '{target}' holds many different numbers, so treating it as "
                    "classes may produce a model that is hard to use. Regression is usually "
                    "the better choice for this column."
                )

        raw_hyperparameters = dict(data.get("hyperparameters") or {})

        try:
            result = train(
                frame,
                target_column=target,
                feature_columns=data.get("feature_columns") or None,
                algorithm=data["algorithm"],
                task_type=task_type,
                test_size=float(data.get("test_size", 0.2)),
                random_state=data.get("random_state", 42),
                hyperparameters=raw_hyperparameters,
                stratify=bool(data.get("stratify", True)),
                scale_features=bool(data.get("scale_features", True)),
                cv_folds=data.get("cv_folds"),
            )
        except (TrainingError, registry.AlgorithmNotFound) as exc:
            return bad_request(_clean_message(exc))
        except ValueError as exc:
            return bad_request(_clean_message(exc))

        artifact = result.pop("artifact", None)
        warnings = extra_warnings + list(result.get("warnings") or [])

        run = TrainingRun(
            name=(data.get("name") or "").strip(),
            dataset=dataset,
            algorithm=result["algorithm"],
            algorithm_label=result["algorithm_label"],
            task_type=result["task_type"],
            target_column=target,
            feature_columns=result["feature_columns"],
            hyperparameters=json_safe(raw_hyperparameters),
            effective_hyperparameters=result["effective_hyperparameters"],
            test_size=float(data.get("test_size", 0.2)),
            random_state=data.get("random_state", 42),
            stratify=bool(data.get("stratify", True)),
            scale_features=bool(data.get("scale_features", True)),
            cv_folds=data.get("cv_folds"),
            status="success",
            error="",
            n_train=result["n_train"],
            n_test=result["n_test"],
            duration_ms=result["duration_ms"],
            metrics=result["metrics"],
            train_metrics=result["train_metrics"],
            cv=result["cv"],
            classes=result["classes"],
            confusion_matrix=result["confusion_matrix"],
            per_class=result["per_class"],
            feature_importance=result["feature_importance"],
            residual_sample=result["residual_sample"],
            input_schema=result["input_schema"],
            warnings=warnings,
        )

        if artifact is not None:
            try:
                blob = dump_artifact(artifact)
            except Exception:  # noqa: BLE001 - a failed dump must not lose the run
                blob = None
            if blob:
                run.artifact.save(
                    _safe_filename(f"{run.algorithm}_{target}", ".joblib"),
                    ContentFile(blob),
                    save=False,
                )
        run.save()

        return Response(
            TrainingRunDetailSerializer(run, context={"request": request}).data,
            status=status.HTTP_201_CREATED,
        )


def _clean_message(exc: Exception) -> str:
    """Render an exception as a plain sentence, without KeyError quoting."""
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    text = str(exc).strip()
    return text or "That request could not be completed."


# --------------------------------------------------------------------------
# 10-15. runs
# --------------------------------------------------------------------------


class RunListView(APIView):
    """GET /api/runs/ -- newest first, optionally filtered by dataset."""

    @extend_schema(responses=list)
    def get(self, request) -> Response:
        runs = TrainingRun.objects.select_related("dataset").all()
        dataset_id = request.query_params.get("dataset_id")
        if dataset_id:
            try:
                uuid.UUID(str(dataset_id))
            except (ValueError, AttributeError, TypeError):
                return bad_request("The dataset id in the filter is not a valid id.")
            runs = runs.filter(dataset_id=dataset_id)
        return Response(TrainingRunSummarySerializer(runs, many=True).data)


class RunDetailView(APIView):
    """GET/DELETE /api/runs/<uuid>/."""

    @extend_schema(responses=dict)
    def get(self, request, pk) -> Response:
        run = get_object_or_404(TrainingRun.objects.select_related("dataset"), pk=pk)
        return Response(TrainingRunDetailSerializer(run, context={"request": request}).data)

    @extend_schema(responses=None)
    def delete(self, request, pk) -> Response:
        run = get_object_or_404(TrainingRun, pk=pk)
        run.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class RunPredictView(APIView):
    """POST /api/runs/<uuid>/predict/ -- score raw rows with a saved model."""

    parser_classes = [JSONParser, FormParser, MultiPartParser]

    @extend_schema(request=PredictRequestSerializer, responses=dict)
    def post(self, request, pk) -> Response:
        run = get_object_or_404(TrainingRun, pk=pk)
        serializer = PredictRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return invalid(serializer)

        if not run.artifact:
            return bad_request(
                "This run has no saved model, so it cannot make predictions. Train it again."
            )
        try:
            artifact = run.load_artifact()
        except (TrainingError, FileNotFoundError, OSError) as exc:
            return bad_request(_clean_message(exc))

        try:
            predictions = run_prediction(artifact, list(serializer.validated_data["rows"]))
        except PredictionError as exc:
            return bad_request(str(exc))

        return Response({"predictions": json_safe(predictions)})


class RunDownloadModelView(APIView):
    """GET /api/runs/<uuid>/download_model/ -- the joblib artefact."""

    @extend_schema(responses=bytes)
    def get(self, request, pk):
        run = get_object_or_404(TrainingRun, pk=pk)
        if not run.artifact:
            return bad_request("This run has no saved model file to download.")
        try:
            handle = run.artifact.open("rb")
        except (OSError, ValueError):
            return bad_request("The saved model file could not be found on the server.")
        filename = _safe_filename(run.display_name or run.algorithm, ".joblib")
        response = FileResponse(handle, content_type="application/octet-stream")
        response["Content-Disposition"] = f'attachment; filename="{filename}"'
        return response


class RunReportView(APIView):
    """GET /api/runs/<uuid>/report/ -- the Markdown evaluation report."""

    @extend_schema(responses=str)
    def get(self, request, pk):
        run = get_object_or_404(TrainingRun.objects.select_related("dataset"), pk=pk)
        wanted = (request.query_params.get("format") or "md").strip().lower()
        if wanted not in ("md", "markdown", "json"):
            return bad_request("The report format must be either 'md' or 'json'.")

        markdown = markdown_report(run)
        if wanted == "json":
            return Response(
                {
                    "report": markdown,
                    "run": TrainingRunDetailSerializer(
                        run, context={"request": request}
                    ).data,
                }
            )

        filename = _safe_filename(run.display_name or run.algorithm, ".md")
        response = HttpResponse(markdown, content_type="text/markdown; charset=utf-8")
        response["Content-Disposition"] = f'attachment; filename="{filename}"'
        return response

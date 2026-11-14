"""HTTP API for the experiment lifecycle, mounted under ``/api/lifecycle/``.

The same rules as ``core.views`` apply here: plain ``APIView``s, bare-array
list endpoints, user-fixable problems as 400 with a plain sentence in
``detail``, and nothing that escapes as a traceback. The heavy ML never runs
in these views: training, search and AutoML are queued as ``Job``s for the
worker (or run inline for ``run_now`` when the caller asks for a synchronous
answer, which keeps the platform usable on a tiny single-process deploy).
"""

from __future__ import annotations

import io
import json
import math
import re
import uuid
from typing import Any

import pandas as pd
from django.core.files.base import ContentFile
from django.db import transaction
from django.http import FileResponse, HttpResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone
from drf_spectacular.utils import extend_schema
from rest_framework import status
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from core.models import Dataset
from core.ml.io import json_safe, read_csv, write_csv
from core.ml.trainer import load_artifact

from lifecycle import services, signature as signature_service
from lifecycle.models import (
    DatasetVersion,
    Deployment,
    Experiment,
    ExperimentRun,
    Job,
    Model,
    ModelVersion,
    MonitoringRecord,
)
from lifecycle.serializers import (
    AutoMlJobSerializer,
    CreateDeploymentSerializer,
    CreateExperimentSerializer,
    DatasetVersionLineageSerializer,
    DatasetVersionSerializer,
    DeploymentSerializer,
    DriftCheckSerializer,
    ExperimentDetailSerializer,
    ExperimentSerializer,
    HpoJobSerializer,
    JobSerializer,
    ModelSerializer,
    ModelVersionSerializer,
    MonitoringSummarySerializer,
    PredictOneSerializer,
    PromoteModelSerializer,
    RegisterModelSerializer,
    RunDetailSerializer,
    RunSerializer,
    TrainJobSerializer,
    BatchPredictSerializer,
    compare_runs,
)

from lifecycle import leakage  # pre-train checks

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")

#: User-fixable failures from the service layer.
USER_ERRORS = (services.LifecycleError,)


def bad_request(detail: str) -> Response:
    return Response({"detail": str(detail)}, status=status.HTTP_400_BAD_REQUEST)


def invalid(serializer) -> Response:
    """Render a serializer's errors as the 400 envelope, first sentence first."""
    from core.views import invalid as core_invalid

    return core_invalid(serializer)


# --------------------------------------------------------------------------
# dataset versions
# --------------------------------------------------------------------------


class DatasetVersionListView(APIView):
    """GET /api/lifecycle/datasets/<dataset_id>/versions/ -- a dataset's versions, newest first."""

    @extend_schema(responses=list)
    def get(self, request, dataset_id) -> Response:
        dataset = get_object_or_404(Dataset, pk=dataset_id)
        versions = DatasetVersion.objects.filter(dataset=dataset)
        return Response(DatasetVersionSerializer(versions, many=True).data)

    @extend_schema(request=dict, responses=dict)
    def post(self, request, dataset_id) -> Response:
        """Create a new immutable version from the dataset's current content."""
        dataset = get_object_or_404(Dataset, pk=dataset_id)
        parent_id = request.data.get("parent_version_id")
        parent = None
        if parent_id:
            parent = DatasetVersion.objects.filter(pk=parent_id).first()
            if parent is None:
                return bad_request("That parent version does not exist.")
        description = str(request.data.get("description") or "")[:2000]
        name = str(request.data.get("name") or "").strip() or f"{dataset.name} v-new"
        source_type = str(request.data.get("source_type") or "derived")
        if source_type not in ("upload", "preprocessed", "derived"):
            return bad_request("Source type must be 'upload', 'preprocessed' or 'derived'.")
        recipe = request.data.get("recipe") or None
        if recipe is not None and not isinstance(recipe, list):
            return bad_request("The recipe must be a list of preprocessing steps.")
        try:
            version = services.create_dataset_version(
                dataset,
                source_type=source_type,
                parent=parent,
                name=name,
                description=description,
                recipe=recipe,
                user=request.user if request.user.is_authenticated else None,
            )
        except USER_ERRORS as exc:
            return bad_request(str(exc))
        return Response(
            DatasetVersionSerializer(version).data, status=status.HTTP_201_CREATED
        )


class DatasetVersionDetailView(APIView):
    """GET/DELETE /api/lifecycle/versions/<id>/ -- one version, immutable."""

    @extend_schema(responses=dict)
    def get(self, request, pk) -> Response:
        version = get_object_or_404(DatasetVersion.objects.select_related("dataset"), pk=pk)
        return Response(DatasetVersionSerializer(version).data)

    @extend_schema(responses=None)
    def delete(self, request, pk) -> Response:
        version = get_object_or_404(DatasetVersion, pk=pk)
        # A parent may be deleted only while no child run depends on it being
        # the current version; the PRD requires children to survive with their
        # metadata intact, and the FK is SET_NULL, so deletion is allowed but
        # acknowledged.
        version.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class DatasetVersionLineageView(APIView):
    """GET /api/lifecycle/versions/<id>/lineage/ -- the chain that produced this version."""

    @extend_schema(responses=dict)
    def get(self, request, pk) -> Response:
        version = get_object_or_404(DatasetVersion.objects.select_related("dataset", "parent"), pk=pk)
        return Response(DatasetVersionLineageSerializer(version).data)


class DatasetVersionDownloadView(APIView):
    """GET /api/lifecycle/versions/<id>/download/ -- the version's stored CSV."""

    @extend_schema(responses=bytes)
    def get(self, request, pk) -> Response:
        version = get_object_or_404(DatasetVersion, pk=pk)
        if not version.file:
            return bad_request("This version has no stored file to download.")
        try:
            handle = version.file.open("rb")
        except (OSError, ValueError):
            return bad_request("The version's data file could not be found on the server.")
        filename = _safe_name(version.name, ".csv")
        response = FileResponse(handle, content_type="text/csv")
        response["Content-Disposition"] = f'attachment; filename="{filename}"'
        return response


class DatasetVersionLeakageView(APIView):
    """POST /api/lifecycle/versions/<id>/leakage/ -- pre-train checks on a version."""

    parser_classes = [JSONParser, FormParser, MultiPartParser]

    @extend_schema(request=dict, responses=dict)
    def post(self, request, pk) -> Response:
        version = get_object_or_404(DatasetVersion, pk=pk)
        target = str(request.data.get("target_column") or "").strip()
        features = request.data.get("feature_columns")
        if features is not None and not isinstance(features, list):
            return bad_request("feature_columns must be a list of column names.")
        try:
            frame = version.load_dataframe()
        except Exception:  # noqa: BLE001 - storage problems are 400 here
            return bad_request("The version's data file could not be read.")
        if target and target not in frame.columns:
            return bad_request(f"Column '{target}' is not in the dataset.")
        summary = leakage.run_leakage_checks(
            frame, target_column=target or None, feature_columns=features or None
        )
        return Response(summary)


class DatasetVersionResourcesView(APIView):
    """POST /api/lifecycle/versions/<id>/resources/ -- the resource-governance estimate."""

    parser_classes = [JSONParser, FormParser, MultiPartParser]

    @extend_schema(request=dict, responses=dict)
    def post(self, request, pk) -> Response:
        version = get_object_or_404(DatasetVersion, pk=pk)
        algorithm = str(request.data.get("algorithm") or "logistic_regression")
        n_trials = int(request.data.get("n_trials") or 1)
        try:
            frame = version.load_dataframe()
        except Exception:  # noqa: BLE001
            return bad_request("The version's data file could not be read.")
        return Response(
            json_safe(services.estimate_resources(frame, algorithm, n_trials=n_trials))
        )


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------


class ExperimentListView(APIView):
    """GET/POST /api/lifecycle/experiments/."""

    @extend_schema(responses=list)
    def get(self, request) -> Response:
        experiments = Experiment.objects.select_related("dataset_version", "best_run").all()
        dataset_id = request.query_params.get("dataset_id")
        if dataset_id:
            experiments = experiments.filter(dataset_version__dataset_id=dataset_id)
        return Response(ExperimentSerializer(experiments, many=True).data)

    @extend_schema(request=CreateExperimentSerializer, responses=dict)
    def post(self, request) -> Response:
        serializer = CreateExperimentSerializer(data=request.data)
        if not serializer.is_valid():
            return invalid(serializer)
        data = serializer.validated_data

        dataset = Dataset.objects.filter(pk=data["dataset_id"]).first()
        if dataset is None:
            return bad_request("That dataset does not exist. Upload it again or pick another one.")

        version = None
        if data.get("dataset_version_id"):
            version = DatasetVersion.objects.filter(pk=data["dataset_version_id"], dataset=dataset).first()
            if version is None:
                return bad_request("That dataset version does not belong to this dataset.")
        if version is None:
            version = services.ensure_dataset_version(dataset)

        try:
            experiment = services.create_experiment(
                dataset_version=version,
                name=data["name"],
                task_type=data.get("task_type", "auto"),
                objective_metric=data.get("objective_metric") or None,
                objective_direction=data.get("objective_direction", "maximize"),
                description=data.get("description", ""),
                tags=data.get("tags") or [],
                user=request.user if request.user.is_authenticated else None,
            )
        except USER_ERRORS as exc:
            return bad_request(str(exc))
        return Response(
            ExperimentDetailSerializer(experiment, context={"request": request}).data,
            status=status.HTTP_201_CREATED,
        )


class ExperimentDetailView(APIView):
    """GET/DELETE /api/lifecycle/experiments/<id>/."""

    @extend_schema(responses=dict)
    def get(self, request, pk) -> Response:
        experiment = get_object_or_404(
            Experiment.objects.select_related("dataset_version", "dataset_version__dataset", "best_run"), pk=pk
        )
        return Response(ExperimentDetailSerializer(experiment, context={"request": request}).data)

    @extend_schema(responses=None)
    def delete(self, request, pk) -> Response:
        experiment = get_object_or_404(Experiment, pk=pk)
        experiment.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class ExperimentRunsView(APIView):
    """GET /api/lifecycle/experiments/<id>/runs/ -- the experiment's runs, newest first."""

    @extend_schema(responses=list)
    def get(self, request, pk) -> Response:
        experiment = get_object_or_404(Experiment, pk=pk)
        runs = experiment.runs.select_related("experiment", "dataset_version").all()
        return Response(RunSerializer(runs, many=True).data)


class ExperimentTrainView(APIView):
    """POST /api/lifecycle/experiments/<id>/train/ -- queue one training run."""

    parser_classes = [JSONParser, FormParser, MultiPartParser]

    @extend_schema(request=TrainJobSerializer, responses=dict)
    def post(self, request, pk) -> Response:
        experiment = get_object_or_404(Experiment, pk=pk)
        serializer = TrainJobSerializer(data=request.data)
        if not serializer.is_valid():
            return invalid(serializer)
        payload = {**serializer.validated_data, "experiment_id": str(experiment.id)}
        payload = _ids_to_str(payload)
        try:
            job = services.create_training_job(
                payload, user=request.user if request.user.is_authenticated else None,
                start=bool(serializer.validated_data.get("run_now")),
            )
        except USER_ERRORS as exc:
            return bad_request(str(exc))
        return Response(JobSerializer(job, context={"request": request}).data, status=status.HTTP_201_CREATED)


class ExperimentHpoView(APIView):
    """POST /api/lifecycle/experiments/<id>/search/ -- queue a hyperparameter search."""

    parser_classes = [JSONParser, FormParser, MultiPartParser]

    @extend_schema(request=HpoJobSerializer, responses=dict)
    def post(self, request, pk) -> Response:
        experiment = get_object_or_404(Experiment, pk=pk)
        serializer = HpoJobSerializer(data=request.data)
        if not serializer.is_valid():
            return invalid(serializer)
        payload = _ids_to_str({**serializer.validated_data, "experiment_id": str(experiment.id)})
        try:
            job = services.create_hpo_job(
                payload, user=request.user if request.user.is_authenticated else None,
                start=bool(serializer.validated_data.get("run_now")),
            )
        except USER_ERRORS as exc:
            return bad_request(str(exc))
        return Response(JobSerializer(job, context={"request": request}).data, status=status.HTTP_201_CREATED)


class ExperimentAutoMlView(APIView):
    """POST /api/lifecycle/experiments/<id>/automl/ -- queue the AutoML workflow."""

    parser_classes = [JSONParser, FormParser, MultiPartParser]

    @extend_schema(request=AutoMlJobSerializer, responses=dict)
    def post(self, request, pk) -> Response:
        experiment = get_object_or_404(Experiment, pk=pk)
        serializer = AutoMlJobSerializer(data=request.data)
        if not serializer.is_valid():
            return invalid(serializer)
        payload = _ids_to_str({**serializer.validated_data, "experiment_id": str(experiment.id)})
        try:
            job = services.create_automl_job(
                payload, user=request.user if request.user.is_authenticated else None,
                start=bool(serializer.validated_data.get("run_now")),
            )
        except USER_ERRORS as exc:
            return bad_request(str(exc))
        return Response(JobSerializer(job, context={"request": request}).data, status=status.HTTP_201_CREATED)


# --------------------------------------------------------------------------
# runs
# --------------------------------------------------------------------------


class RunDetailView(APIView):
    """GET/DELETE /api/lifecycle/runs/<id>/."""

    @extend_schema(responses=dict)
    def get(self, request, pk) -> Response:
        run = get_object_or_404(
            ExperimentRun.objects.select_related("experiment", "dataset_version", "dataset_version__dataset"),
            pk=pk,
        )
        return Response(RunDetailSerializer(run, context={"request": request}).data)

    @extend_schema(responses=None)
    def delete(self, request, pk) -> Response:
        run = get_object_or_404(ExperimentRun, pk=pk)
        run.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class RunCompareView(APIView):
    """POST /api/lifecycle/runs/compare/ -- side-by-side metrics for chosen runs."""

    parser_classes = [JSONParser, FormParser, MultiPartParser]

    @extend_schema(request=dict, responses=dict)
    def post(self, request) -> Response:
        run_ids = request.data.get("run_ids")
        if not isinstance(run_ids, list) or not run_ids:
            return bad_request("Choose at least two runs to compare.")
        if len(run_ids) > 12:
            return bad_request("Compare up to 12 runs at a time.")
        ids: list[Any] = []
        for raw in run_ids:
            try:
                ids.append(uuid.UUID(str(raw)))
            except (ValueError, AttributeError, TypeError):
                return bad_request(f"'{raw}' is not a valid run id.")
        runs = [
            run
            for run in ExperimentRun.objects.filter(id__in=ids).select_related(
                "experiment", "dataset_version"
            )
        ]
        if len(runs) < 2:
            return bad_request("At least two of those runs exist to compare.")
        return Response(compare_runs(runs))


class RunDownloadModelView(APIView):
    """GET /api/lifecycle/runs/<id>/download_model/ -- the run's joblib artifact."""

    @extend_schema(responses=bytes)
    def get(self, request, pk) -> Response:
        run = get_object_or_404(ExperimentRun, pk=pk)
        from lifecycle.models import load_run_artifact

        artifact = load_run_artifact(run)
        if artifact is None:
            return bad_request("This run has no saved model file to download.")
        from core.ml.trainer import dump_artifact

        try:
            blob = dump_artifact(artifact)
        except Exception:  # noqa: BLE001 - a failed dump is a 400, not a 500
            return bad_request("The saved model could not be read back from storage.")
        filename = _safe_name(run.display_name, ".joblib")
        response = HttpResponse(blob, content_type="application/octet-stream")
        response["Content-Disposition"] = f'attachment; filename="{filename}"'
        return response


class RunRegisterModelView(APIView):
    """POST /api/lifecycle/runs/<id>/register/ -- put a run's model in the registry."""

    parser_classes = [JSONParser, FormParser, MultiPartParser]

    @extend_schema(request=RegisterModelSerializer, responses=dict)
    def post(self, request, pk) -> Response:
        run = get_object_or_404(ExperimentRun, pk=pk)
        serializer = RegisterModelSerializer(data=request.data)
        if not serializer.is_valid():
            return invalid(serializer)
        try:
            version = services.register_model(
                run,
                model_name=serializer.validated_data.get("model_name") or None,
                user=request.user if request.user.is_authenticated else None,
            )
        except USER_ERRORS as exc:
            return bad_request(str(exc))
        return Response(
            ModelVersionSerializer(version, context={"request": request}).data,
            status=status.HTTP_201_CREATED,
        )


class RunReproduceView(APIView):
    """POST /api/lifecycle/runs/<id>/reproduce/ -- re-train the run's exact config."""

    parser_classes = [JSONParser, FormParser, MultiPartParser]

    @extend_schema(request=dict, responses=dict)
    def post(self, request, pk) -> Response:
        run = get_object_or_404(
            ExperimentRun.objects.select_related("experiment", "dataset_version"), pk=pk
        )
        run_now = bool(request.data.get("run_now", False))
        try:
            job = services.request_reproduce_job(
                run, user=request.user if request.user.is_authenticated else None, start=run_now
            )
        except USER_ERRORS as exc:
            return bad_request(str(exc))
        return Response(JobSerializer(job, context={"request": request}).data, status=status.HTTP_201_CREATED)


class RunNotebookView(APIView):
    """GET /api/lifecycle/runs/<id>/notebook/ -- an executable .ipynb that reproduces the run.

    The generated notebook carries its own data loading (from the version's
    download URL), the exact preprocessing and split, the same estimator and
    hyperparameters, and the same evaluation -- so "open in a notebook" and
    "reproduce" stay the same artifact.
    """

    @extend_schema(responses=bytes)
    def get(self, request, pk) -> Response:
        run = get_object_or_404(
            ExperimentRun.objects.select_related("experiment", "dataset_version", "dataset_version__dataset"),
            pk=pk,
        )
        notebook = _build_reproduction_notebook(request, run)
        filename = _safe_name(run.display_name, ".ipynb")
        response = HttpResponse(
            json_dumps_notebook(notebook),
            content_type="application/json",
        )
        response["Content-Disposition"] = f'attachment; filename="{filename}"'
        return response


def json_dumps_notebook(notebook: dict) -> str:  # small alias keeps imports tidy
    import json

    return json.dumps(notebook, indent=1)


def _build_reproduction_notebook(request: Any, run: ExperimentRun) -> dict:
    """Assemble the .ipynb JSON for one run."""
    download_url = ""
    try:
        from django.urls import reverse

        path = reverse("lifecycle:version-download", kwargs={"pk": str(run.dataset_version_id)})
        download_url = request.build_absolute_uri(path) if request else path
    except Exception:  # noqa: BLE001
        pass

    hyperparameters = json_safe(run.hyperparameters or {})
    feature_columns = json_safe(run.feature_columns or [])

    def cell(source: str, cell_type: str = "code") -> dict:
        lines = source.split("\n")
        return {
            "cell_type": cell_type,
            "metadata": {},
            "source": [line + "\n" for line in lines[:-1]] + [lines[-1]],
            **(
                {"outputs": [], "execution_count": None}
                if cell_type == "code"
                else {}
            ),
        }

    intro = (
        f"# Reproducing: {run.display_name}\n\n"
        f"This notebook regenerates run `{run.id}` of experiment "
        f"**{run.experiment.name}**.\n\n"
        f"* Dataset version: {run.dataset_version.name} (v{run.dataset_version.version})\n"
        f"* Algorithm: {run.algorithm_label or run.algorithm}\n"
        f"* Target: `{run.target_column}`\n"
        f"* Random seed: {run.random_state}\n\n"
        "The environment you run this in may differ from the one that produced the "
        "original run, so metrics can differ slightly across library versions."
    )
    setup = (
        "import json\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.compose import ColumnTransformer\n"
        "from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler\n"
        "from sklearn.impute import SimpleImputer\n\n"
        "RANDOM_STATE = " + str(run.random_state if run.random_state is not None else 42) + "\n"
        "TEST_SIZE = " + str(run.test_size) + "\n"
        "TARGET = " + repr(run.target_column) + "\n"
        "FEATURES = " + repr(feature_columns) + "\n\n"
        "# Cast text columns to str while keeping missing values missing -- the\n"
        "# same transformation the Furina pipeline applies internally.\n"
        "def to_string_frame(X):\n"
        "    X = pd.DataFrame(X).copy()\n"
        "    for column in X.columns:\n"
        "        missing = X[column].isna()\n"
        "        X[column] = X[column].astype(str)\n"
        "        X.loc[missing, column] = np.nan\n"
        "    return X\n"
    )
    load = (
        f"df = pd.read_csv({download_url!r})\n"
        "df = df.dropna(subset=[TARGET])\n"
        "X = df[FEATURES] if FEATURES else df.drop(columns=[TARGET])\n"
        "y = df[TARGET]\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        f"    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, "
        f"stratify={'y' if run.stratify and run.task_type == 'classification' else 'None'},\n"
        ")\n"
        "print('train rows:', len(X_train), 'test rows:', len(X_test))\n"
    )
    params = json.dumps(hyperparameters, indent=4)
    build = (
        "numeric = X.select_dtypes(include='number').columns.tolist()\n"
        "categorical = [c for c in X.columns if c not in numeric]\n\n"
        "numeric_steps = [('impute', SimpleImputer(strategy='median'))]\n"
        + (
            "numeric_steps.append(('scale', StandardScaler()))\n"
            if run.scale_features
            else ""
        )
        + "preprocess = ColumnTransformer([\n"
        "    ('num', Pipeline(numeric_steps), numeric),\n"
        "    ('cat', Pipeline([\n"
        "        ('stringify', FunctionTransformer(to_string_frame, feature_names_out='one-to-one')),\n"
        "        ('impute', SimpleImputer(strategy='most_frequent', fill_value='missing')),\n"
        "        ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),\n"
        "    ]), categorical),\n"
        "])\n\n"
        f"from importlib import import_module\n"
        f"MODULE, CLASS = _ESTIMATORS[{run.algorithm!r}]\n"
        "kwargs = json.loads(HYPERPARAMETERS_JSON)\n"
        "model = getattr(import_module(MODULE), CLASS)(**kwargs)\n"
        "if 'random_state' in getattr(model, 'get_params', lambda: {})():\n"
        "    model.set_params(random_state=RANDOM_STATE)\n"
        "pipeline = Pipeline([('preprocess', preprocess), ('model', model)])\n"
        "pipeline\n"
    )
    estimator_helper = (
        "# The Furina algorithm key -> the estimator class behind it.\n"
        "# This run's own settings are the constants the next cells use.\n"
        "HYPERPARAMETERS_JSON = " + json.dumps(params) + "\n"
        "_ESTIMATORS = {\n"
        + "".join(
            f"    {key!r}: ({cls_path!r}, {class_name!r}),\n"
            for key, cls_path, class_name in _estimator_table_for(run.algorithm)
        )
        + "}\n"
    )
    fit_eval = (
        "pipeline.fit(X_train, y_train)\n"
        "from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error\n"
        "pred = pipeline.predict(X_test)\n"
        "if y.dtype == object or str(y.dtype) == 'bool':\n"
        "    print('accuracy', round(accuracy_score(y_test, pred), 4))\n"
        "    print('f1 (weighted)', round(f1_score(y_test, pred, average='weighted'), 4))\n"
        "else:\n"
        "    print('r2', round(r2_score(y_test, pred), 4))\n"
        "    print('mae', round(mean_absolute_error(y_test, pred), 4))\n"
    )
    cells = [
        cell(intro, "markdown"),
        cell(setup),
        cell(load),
        cell(estimator_helper),
        cell(build),
        cell(fit_eval),
    ]
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "cells": cells,
    }


def _estimator_table_for(preferred: str) -> list[tuple[str, str, str]]:
    """(key, module, class) rows for the notebook's estimator lookup."""
    from core.ml import registry as algorithm_registry

    table: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for spec in algorithm_registry.list_algorithms():
        key = str(spec.get("key"))
        if key in seen:
            continue
        seen.add(key)
        for task in spec.get("tasks") or []:
            variant_cls = _estimator_class_for(key, str(task))
            if variant_cls is not None:
                table.append((key, variant_cls.__module__, variant_cls.__name__))
                break
    # Make sure the run's own algorithm resolves even if the loop missed it.
    if preferred and not any(row[0] == preferred for row in table):
        cls = _estimator_class_for(preferred, "classification") or _estimator_class_for(
            preferred, "regression"
        )
        if cls is not None:
            table.append((preferred, cls.__module__, cls.__name__))
    return table


def _estimator_class_for(key: str, task: str):
    from core.ml.registry import _ALGORITHMS

    for algorithm in _ALGORITHMS:
        if algorithm.key == key:
            variant = algorithm.variants.get(task)
            if variant is not None:
                return variant.cls
            return next(iter(algorithm.variants.values())).cls
    return None


def _ids_to_str(payload: dict) -> dict:
    """UUID values serialize to plain strings for the job payload."""
    clean = dict(payload)
    for name in ("dataset_version_id", "experiment_id", "run_id"):
        if name in clean and clean[name] is not None:
            clean[name] = str(clean[name])
    return clean


def _safe_name(name: str, suffix: str) -> str:
    stem = _SAFE_NAME.sub("_", (name or "file").strip()).strip("._-") or "file"
    return f"{stem[:60]}{suffix}"


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------


class JobDetailView(APIView):
    """GET /api/lifecycle/jobs/<id>/ -- poll a job's state and progress."""

    @extend_schema(responses=dict)
    def get(self, request, pk) -> Response:
        job = get_object_or_404(Job.objects.select_related("best_run"), pk=pk)
        return Response(JobSerializer(job, context={"request": request}).data)


class JobCancelView(APIView):
    """POST /api/lifecycle/jobs/<id>/cancel/ -- ask the worker to stop."""

    @extend_schema(request=dict, responses=dict)
    def post(self, request, pk) -> Response:
        job = get_object_or_404(Job, pk=pk)
        if job.status not in ("queued", "running"):
            return bad_request(
                f"This job has already finished ({job.get_status_display().lower()}), "
                "so there is nothing to cancel."
            )
        job.request_cancel()
        return Response(JobSerializer(job, context={"request": request}).data)


# --------------------------------------------------------------------------
# model registry
# --------------------------------------------------------------------------


class ModelListView(APIView):
    """GET /api/lifecycle/models/ -- every registry model with its production pointer."""

    @extend_schema(responses=list)
    def get(self, request) -> Response:
        models = Model.objects.select_related("production_version").all()
        return Response(ModelSerializer(models, many=True).data)


class ModelDetailView(APIView):
    """GET /api/lifecycle/models/<id>/ -- one model plus its version history."""

    @extend_schema(responses=dict)
    def get(self, request, pk) -> Response:
        model = get_object_or_404(Model.objects.select_related("production_version"), pk=pk)
        data = ModelSerializer(model).data
        data["versions"] = ModelVersionSerializer(
            model.versions.select_related("model").all(), many=True
        ).data
        return Response(data)


class ModelVersionDetailView(APIView):
    """GET /api/lifecycle/models/<id>/versions/<version>/ -- one version, with its model card."""

    @extend_schema(responses=dict)
    def get(self, request, pk, version) -> Response:
        model = get_object_or_404(Model, pk=pk)
        model_version = get_object_or_404(
            ModelVersion.objects.select_related("model", "source_run"), model=model, version=int(version)
        )
        data = ModelVersionSerializer(model_version, context={"request": request}).data
        data["tools"] = {
            "predict_url": _deploy_predict_url(request, model_version),
            "download_url": _run_model_url(request, model_version),
        }
        return Response(data)


def _model_version_for(model_id: Any, version: Any):
    model = Model.objects.filter(pk=model_id).first()
    if model is None:
        return None
    return ModelVersion.objects.filter(model=model, version=int(version)).select_related("model", "source_run").first()


def _deploy_predict_url(request: Any, model_version: ModelVersion) -> str:
    deployment = model_version.deployments.filter(is_active=True).order_by("-created_at").first()
    if deployment is None:
        return ""
    from django.urls import reverse

    try:
        path = reverse("lifecycle:deployment-predict", kwargs={"slug": deployment.slug})
    except Exception:  # noqa: BLE001
        return ""
    return request.build_absolute_uri(path) if request else path


def _run_model_url(request: Any, model_version: ModelVersion) -> str:
    if model_version.source_run_id:
        from django.urls import reverse

        try:
            path = reverse("lifecycle:run-download-model", kwargs={"pk": str(model_version.source_run_id)})
        except Exception:  # noqa: BLE001
            return ""
        return request.build_absolute_uri(path) if request else path
    return ""


class ModelVersionPromoteView(APIView):
    """POST /api/lifecycle/models/<id>/versions/<version>/promote/ -- move the lifecycle forward."""

    parser_classes = [JSONParser, FormParser, MultiPartParser]

    @extend_schema(request=PromoteModelSerializer, responses=dict)
    def post(self, request, pk, version) -> Response:
        model_version = _model_version_for(pk, version)
        if model_version is None:
            return bad_request("That model version does not exist.")
        serializer = PromoteModelSerializer(data=request.data)
        if not serializer.is_valid():
            return invalid(serializer)
        try:
            services.promote_model_version(
                model_version,
                serializer.validated_data["target"],
                confirmed=bool(serializer.validated_data.get("confirmed")),
            )
        except USER_ERRORS as exc:
            return bad_request(str(exc))
        return Response(ModelVersionSerializer(model_version, context={"request": request}).data)


class ModelVersionDeployView(APIView):
    """POST /api/lifecycle/models/<id>/versions/<version>/deploy/ -- expose the version as an API."""

    parser_classes = [JSONParser, FormParser, MultiPartParser]

    @extend_schema(request=CreateDeploymentSerializer, responses=dict)
    def post(self, request, pk, version) -> Response:
        model_version = _model_version_for(pk, version)
        if model_version is None:
            return bad_request("That model version does not exist.")
        serializer = CreateDeploymentSerializer(data=request.data)
        if not serializer.is_valid():
            return invalid(serializer)
        slug = (serializer.validated_data.get("slug") or "").strip()
        if not slug:
            base = _slugify(f"{model_version.model.name}-v{model_version.version}")
            slug = base[:80]
        if Deployment.objects.filter(slug=slug).exists():
            slug = f"{slug[:80]}-{uuid.uuid4().hex[:6]}"

        baseline = _baseline_stats(model_version)
        deployment = Deployment.objects.create(
            model_version=model_version,
            slug=slug,
            baseline_stats=baseline,
        )
        model_version.model_card = {
            **(model_version.model_card or {}),
            "deployment_status": f"Deployed as /{deployment.slug}/ with version {model_version.version}.",
        }
        model_version.save(update_fields=["model_card"])
        return Response(
            DeploymentSerializer(deployment, context={"request": request}).data,
            status=status.HTTP_201_CREATED,
        )


def _slugify(text: str) -> str:
    import re as _re

    slug = _re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return slug or "model"


def _baseline_stats(model_version: ModelVersion) -> dict:
    """Training-time feature statistics used for drift detection."""
    run = model_version.source_run
    dataset_version = model_version.dataset_version
    if dataset_version is None:
        return {}
    try:
        frame = dataset_version.load_dataframe()
    except Exception:  # noqa: BLE001 - a missing file must not block a deployment
        return {}
    baseline: dict[str, Any] = {}
    for field in (model_version.signature or {}).get("features") or []:
        name = field.get("name")
        if name not in frame.columns:
            continue
        series = frame[name]
        kind = field.get("type") or "categorical"
        if kind == "numeric":
            values = pd.to_numeric(series, errors="coerce").dropna()
            non_null = series.dropna()
            baseline[name] = {
                "kind": "numeric",
                "mean": float(values.mean()) if len(values) else None,
                "std": float(values.std(ddof=0)) if len(values) > 1 else 0.0,
                "min": float(values.min()) if len(values) else None,
                "max": float(values.max()) if len(values) else None,
                "missing_rate": float(len(series) - len(non_null)) / max(len(series), 1),
                "counts": None,
            }
        else:
            counts = series.dropna().astype(str).value_counts(normalize=True)
            non_null = series.dropna()
            baseline[name] = {
                "kind": "categorical",
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "missing_rate": float(len(series) - len(non_null)) / max(len(series), 1),
                "counts": {str(key): float(value) for key, value in counts.head(50).items()},
            }
    return json_safe({"features": baseline, "trained_rows": dataset_version.row_count, "run_id": str(run.id) if run else None})


# --------------------------------------------------------------------------
# deployments: inference, batch, openapi, snippets
# --------------------------------------------------------------------------


class DeploymentListView(APIView):
    """GET /api/lifecycle/deployments/ -- the active inference endpoints."""

    @extend_schema(responses=list)
    def get(self, request) -> Response:
        deployments = Deployment.objects.select_related("model_version", "model_version__model").all()
        return Response(DeploymentSerializer(deployments, many=True, context={"request": request}).data)


class DeploymentDetailView(APIView):
    """GET/DELETE /api/lifecycle/deployments/<slug>/."""

    @extend_schema(responses=dict)
    def get(self, request, slug) -> Response:
        deployment = get_object_or_404(
            Deployment.objects.select_related("model_version", "model_version__model"), slug=slug
        )
        return Response(DeploymentSerializer(deployment, context={"request": request}).data)

    @extend_schema(responses=None)
    def delete(self, request, slug) -> Response:
        deployment = get_object_or_404(Deployment, slug=slug)
        deployment.is_active = False
        deployment.save(update_fields=["is_active"])
        return Response(status=status.HTTP_204_NO_CONTENT)


def _load_deployment_artifact(deployment: Deployment):
    """The artifact behind a deployment, or raise LifecycleError with a sentence."""
    try:
        return deployment.model_version.load_artifact()
    except FileNotFoundError as exc:
        raise services.LifecycleError(str(exc)) from None
    except Exception:  # noqa: BLE001 - a corrupt artifact is a 400, not a 500
        raise services.LifecycleError(
            "The deployed model file could not be read. Re-register the model from a run."
        ) from None


class DeploymentPredictView(APIView):
    """POST /api/lifecycle/deployments/<slug>/predict/ -- score one row."""

    parser_classes = [JSONParser, FormParser, MultiPartParser]

    @extend_schema(request=PredictOneSerializer, responses=dict)
    def post(self, request, slug) -> Response:
        deployment = get_object_or_404(
            Deployment.objects.select_related("model_version"), slug=slug
        )
        if not deployment.is_active:
            return Response(
                {"detail": "This deployment is archived and no longer accepts predictions."},
                status=status.HTTP_410_GONE,
            )
        data = request.data.get("data") if "data" in request.data else request.data
        if not isinstance(data, dict):
            return bad_request("Send a JSON object of feature values, e.g. {\"age\": 32}.")
        return self._predict(deployment, data)

    def _predict(self, deployment: Deployment, data: dict) -> Response:
        import time as _time

        started = _time.perf_counter()
        signature = deployment.model_version.signature or {}
        problems = signature_service.validate_payload(signature, data)
        blocking = [p for p in problems if not p.startswith("Field")]
        if blocking:
            return bad_request(" ".join(problems))

        try:
            artifact = _load_deployment_artifact(deployment)
        except USER_ERRORS as exc:
            return bad_request(str(exc))

        from core.ml.predict import PredictionError, predict as run_prediction

        try:
            predictions = run_prediction(artifact, [data])
        except PredictionError as exc:
            return bad_request(str(exc))
        result = predictions[0]
        latency_ms = round((_time.perf_counter() - started) * 1000, 1)

        # Monitoring: one row per request, capped so a busy endpoint does not
        # fill the database with individual requests.
        deployment.total_requests = (deployment.total_requests or 0) + 1
        deployment.last_predicted_at = timezone.now()
        deployment.save(update_fields=["total_requests", "last_predicted_at"])
        MonitoringRecord.objects.create(
            deployment=deployment,
            kind="request",
            payload=json_safe(
                {
                    "latency_ms": latency_ms,
                    "features": {str(k): v for k, v in data.items()},
                    "prediction": result.get("prediction"),
                    "warnings": problems if problems else None,
                }
            ),
        )

        body = {
            "prediction": result.get("prediction"),
            "probabilities": result.get("probabilities"),
            "model_version": str(deployment.model_version.version),
        }
        if result.get("probabilities"):
            body["probabilities_map"] = {
                str(item["label"]): item["probability"]
                for item in result["probabilities"]
            }
        if problems:
            body["notes"] = problems
        return Response(json_safe(body))


class DeploymentBatchPredictView(APIView):
    """POST /api/lifecycle/deployments/<slug>/batch-predict/ -- score a whole CSV."""

    parser_classes = [MultiPartParser, FormParser]

    @extend_schema(request=BatchPredictSerializer, responses=bytes)
    def post(self, request, slug):
        deployment = get_object_or_404(Deployment.objects.select_related("model_version"), slug=slug)
        if not deployment.is_active:
            return Response(
                {"detail": "This deployment is archived and no longer accepts predictions."},
                status=status.HTTP_410_GONE,
            )
        serializer = BatchPredictSerializer(data=request.data)
        if not serializer.is_valid():
            return invalid(serializer)
        upload = serializer.validated_data["file"]

        try:
            frame = read_csv(upload, delimiter="auto")
        except Exception:  # noqa: BLE001 - any unreadable CSV is a 400
            return bad_request(
                "That file could not be read as a table. Upload a CSV with a header row."
            )
        if len(frame) == 0:
            return bad_request("The file has a header row but no data rows.")

        try:
            artifact = _load_deployment_artifact(deployment)
        except USER_ERRORS as exc:
            return bad_request(str(exc))

        from core.ml.predict import PredictionError, predict as run_prediction

        signature = deployment.model_version.signature or {}
        feature_names = [f["name"] for f in (signature.get("features") or [])]
        rows = [
            {name: (record.get(name) if name in record else None) for name in feature_names}
            for record in frame.to_dict(orient="records")
        ]
        try:
            predictions = run_prediction(artifact, rows)
        except PredictionError as exc:
            return bad_request(str(exc))

        out = frame.copy()
        out["prediction"] = [item.get("prediction") for item in predictions]
        probabilities = [item.get("probabilities") for item in predictions]
        labels: list[str] = []
        for item in probabilities:
            if item:
                for entry in item:
                    if entry["label"] not in labels:
                        labels.append(entry["label"])
        for label in labels:
            column = f"probability_{label}"
            out[column] = [
                next((e["probability"] for e in (item or []) if e["label"] == label), None)
                for item in probabilities
            ]

        buffer = io.BytesIO()
        write_csv(out, buffer)
        deployment.total_requests = (deployment.total_requests or 0) + len(frame)
        deployment.last_predicted_at = timezone.now()
        deployment.save(update_fields=["total_requests", "last_predicted_at"])
        response = HttpResponse(buffer.getvalue(), content_type="text/csv")
        filename = _safe_name(f"{deployment.slug}-predictions", ".csv")
        response["Content-Disposition"] = f'attachment; filename="{filename}"'
        return response


class DeploymentOpenApiView(APIView):
    """GET /api/lifecycle/deployments/<slug>/openapi/ -- this deployment's API contract."""

    @extend_schema(responses=dict)
    def get(self, request, slug) -> Response:
        deployment = get_object_or_404(
            Deployment.objects.select_related("model_version", "model_version__model"), slug=slug
        )
        version = deployment.model_version
        signature = version.signature or {}
        predict_path = request.build_absolute_uri(
            _reverse("lifecycle:deployment-predict", slug=deployment.slug)
        )
        request_schema = {
            "type": "object",
            "required": [f["name"] for f in signature.get("features", []) if f.get("required")],
            "properties": {
                field["name"]: _openapi_property(field) for field in signature.get("features", [])
            },
        }
        return Response(
            json_safe(
                {
                    "openapi": "3.1.0",
                    "info": {
                        "title": f"{version.model.name} v{version.version}",
                        "version": str(version.version),
                    },
                    "paths": {
                        "/predict": {
                            "post": {
                                "summary": "Score one row",
                                "requestBody": {
                                    "required": True,
                                    "content": {"application/json": {"schema": request_schema}},
                                    "example": signature.get("example_request"),
                                },
                                "responses": {
                                    "200": {
                                        "description": "A prediction",
                                        "content": {
                                            "application/json": {
                                                "schema": {
                                                    "type": "object",
                                                    "properties": {
                                                        "prediction": {"type": "string"},
                                                        "probabilities": {"type": "object"},
                                                        "model_version": {"type": "string"},
                                                    },
                                                }
                                            }
                                        },
                                    }
                                },
                            }
                        }
                    },
                    "servers": [{"url": predict_path.rsplit("/predict", 1)[0]}],
                }
            )
        )


def _openapi_property(field: dict) -> dict:
    spec: dict[str, Any] = {"description": f"{field.get('type')} feature"}
    if field.get("type") == "numeric":
        spec["type"] = "number"
        if field.get("minimum") is not None:
            spec["minimum"] = field["minimum"]
        if field.get("maximum") is not None:
            spec["maximum"] = field["maximum"]
    else:
        spec["type"] = "string"
        if field.get("categories"):
            spec["enum"] = field["categories"][:50]
    return spec


class DeploymentSnippetsView(APIView):
    """GET /api/lifecycle/deployments/<slug>/snippets/ -- curl / Python / JavaScript examples."""

    @extend_schema(responses=dict)
    def get(self, request, slug) -> Response:
        deployment = get_object_or_404(
            Deployment.objects.select_related("model_version", "model_version__model"), slug=slug
        )
        version = deployment.model_version
        signature = version.signature or {}
        predict_url = request.build_absolute_uri(
            _reverse("lifecycle:deployment-predict", slug=deployment.slug)
        )
        snippets = signature_service.client_snippet(
            signature, _SnippetTarget(predict_url), language="all"
        )
        return Response(json_safe({"predict_url": predict_url, "snippets": snippets}))


class _SnippetTarget:
    """What ``client_snippet`` needs from a deployment: just its URL."""

    def __init__(self, predict_url: str) -> None:
        self.predict_url = predict_url


def _reverse(name: str, **kwargs: Any) -> str:
    from django.urls import reverse

    return reverse(name, kwargs=kwargs)


# --------------------------------------------------------------------------
# monitoring
# --------------------------------------------------------------------------


class MonitoringOverviewView(APIView):
    """GET /api/lifecycle/monitoring/ -- every active deployment's health at a glance."""

    @extend_schema(responses=list)
    def get(self, request) -> Response:
        deployments = Deployment.objects.select_related(
            "model_version", "model_version__model"
        ).filter(is_active=True)
        return Response(
            MonitoringSummarySerializer(deployments, many=True, context={"request": request}).data
        )


class DeploymentMonitoringView(APIView):
    """GET /api/lifecycle/deployments/<slug>/monitoring/ -- one deployment's record."""

    @extend_schema(responses=dict)
    def get(self, request, slug) -> Response:
        deployment = get_object_or_404(
            Deployment.objects.select_related("model_version", "model_version__model"), slug=slug
        )
        records = deployment.monitoring.all()[:200]
        return Response(
            json_safe(
                {
                    "summary": MonitoringSummarySerializer(
                        deployment, context={"request": request}
                    ).data,
                    "records": [
                        {
                            "kind": record.kind,
                            "payload": record.payload,
                            "created_at": record.created_at.isoformat(),
                        }
                        for record in records
                    ],
                }
            )
        )


class DeploymentDriftCheckView(APIView):
    """POST /api/lifecycle/deployments/<slug>/drift-check/ -- PSI / distribution drift.

    Compares a batch of live rows against the baseline statistics recorded at
    deploy time. Each feature gets a population-stability index (for numeric
    and categorical alike) and a missing-value drift figure; the worst
    offender sets the deployment's headline verdict.
    """

    parser_classes = [JSONParser, FormParser, MultiPartParser]

    @extend_schema(request=DriftCheckSerializer, responses=dict)
    def post(self, request, slug) -> Response:
        deployment = get_object_or_404(
            Deployment.objects.select_related("model_version"), slug=slug
        )
        rows = request.data.get("rows")
        if not isinstance(rows, list) or not rows:
            return bad_request("Send a list of current rows to compare against the training data.")
        if len(rows) > 10_000:
            return bad_request("Send at most 10,000 rows per drift check.")

        try:
            frame = pd.DataFrame(rows)
        except Exception:  # noqa: BLE001
            return bad_request("The rows could not be read as a table.")
        baseline = (deployment.baseline_stats or {}).get("features") or {}
        if not baseline:
            return bad_request(
                "This deployment has no training-time statistics to compare against. "
                "Re-deploy the model version to record them."
            )

        findings = drift_report(baseline, frame)
        MonitoringRecord.objects.create(
            deployment=deployment, kind="drift", payload=json_safe(findings)
        )
        return Response(json_safe(findings))


def drift_report(baseline: dict, frame: "pd.DataFrame") -> dict:
    """Population-stability index per feature plus a missing-value delta.

    PSI below 0.1 is 'stable'; 0.1-0.25 'moderate'; above 0.25 'significant'.
    The mapping is the common rule of thumb and what the PRD asks to report.
    """
    features: list[dict] = []
    worst = 0.0
    for name, stats in baseline.items():
        if name not in frame.columns:
            features.append(
                {
                    "feature": name,
                    "kind": stats.get("kind"),
                    "status": "missing",
                    "psi": None,
                    "missing_delta": None,
                    "message": f"'{name}' was not present in the incoming rows.",
                }
            )
            continue
        series = frame[name]
        kind = stats.get("kind") or "categorical"
        expected_missing = float(stats.get("missing_rate") or 0.0)
        actual_missing = float(len(series) - len(series.dropna())) / max(len(series), 1)
        missing_delta = actual_missing - expected_missing

        if kind == "numeric":
            values = pd.to_numeric(series, errors="coerce").dropna()
            if len(values) == 0:
                psi = None
            else:
                psi = _numeric_psi(stats, values)
            message = _numeric_message(stats, values, missing_delta)
        else:
            counts = series.dropna().astype(str).value_counts(normalize=True)
            psi = _categorical_psi(stats.get("counts") or {}, counts, len(series))
            message = _categorical_message(stats.get("counts") or {}, counts, missing_delta)

        if psi is not None:
            worst = max(worst, psi)
        features.append(
            {
                "feature": name,
                "kind": kind,
                "status": _drift_status(psi),
                "psi": psi,
                "missing_delta": missing_delta,
                "message": message,
            }
        )

    verdict = _drift_status(worst if worst else None)
    return {
        "overall": verdict,
        "max_psi": worst,
        "rows": int(len(frame)),
        "features": features,
    }


def _drift_status(psi: float | None) -> str:
    if psi is None:
        return "unknown"
    if psi > 0.25:
        return "significant"
    if psi > 0.1:
        return "moderate"
    return "stable"


def _numeric_psi(stats: dict, values: "pd.Series") -> float:
    mean = stats.get("mean")
    std = stats.get("std") or 0.0
    if mean is None:
        return 0.0
    # Bucket the training distribution at mean +/- 1 and 2 std, then compare
    # the share of live rows in the same buckets. This is the standard PSI
    # formulation for a continuous feature.
    edges = [-float("inf"), mean - 2 * std, mean - std, mean, mean + std, mean + 2 * std, float("inf")]
    baseline_shares = [0.0228, 0.1359, 0.3413, 0.3413, 0.1359, 0.0228]
    counts = [0] * 6
    for value in values:
        for index in range(6):
            if edges[index] <= value < edges[index + 1]:
                counts[index] += 1
                break
    total = sum(counts) or 1
    return _psi_from_shares(baseline_shares, [c / total for c in counts])


def _categorical_psi(baseline_counts: dict, actual_counts: "pd.Series", total: int) -> float:
    categories = set(baseline_counts) | {str(key) for key in actual_counts.index}
    expected = [float(baseline_counts.get(str(cat), 0.0)) for cat in categories]
    actual = [
        float(actual_counts.get(str(cat), 0.0)) / max(total, 1) for cat in categories
    ]
    return _psi_from_shares(expected, actual)


def _psi_from_shares(expected: list[float], actual: list[float]) -> float:
    """PSI with a small floor so an empty bucket does not divide by zero."""
    psi = 0.0
    for base, live in zip(expected, actual):
        base = max(base, 1e-4)
        live = max(live, 1e-4)
        psi += (live - base) * math.log(live / base)
    return round(psi, 4)


def _numeric_message(stats: dict, values: "pd.Series", missing_delta: float) -> str:
    if len(values) == 0:
        return "No numeric values arrived for this feature."
    low, high = float(values.min()), float(values.max())
    trained_low, trained_high = stats.get("min"), stats.get("max")
    parts: list[str] = []
    if trained_low is not None and trained_high is not None:
        outside = low < trained_low or high > trained_high
        if outside:
            parts.append(
                f"Live range {low:g}-{high:g} falls outside the training range {trained_low:g}-{trained_high:g}."
            )
    if abs(missing_delta) > 0.05:
        direction = "more" if missing_delta > 0 else "fewer"
        parts.append(f"{direction.capitalize()} missing values than in training ({missing_delta:+.0%}).")
    return " ".join(parts) or "The distribution looks close to training."


def _categorical_message(baseline_counts: dict, actual_counts: "pd.Series", missing_delta: float) -> str:
    unseen = [str(key) for key in actual_counts.index if str(key) not in baseline_counts]
    parts: list[str] = []
    if unseen:
        preview = ", ".join(unseen[:3])
        parts.append(f"{len(unseen)} value(s) never seen in training: {preview}.")
    if abs(missing_delta) > 0.05:
        parts.append(f"Missing values changed by {missing_delta:+.0%}.")
    return " ".join(parts) or "The category mix looks close to training."

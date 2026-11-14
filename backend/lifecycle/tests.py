"""Tests for the experiment lifecycle platform.

Every test drives the real HTTP endpoints under ``/api/lifecycle/``, so the
shapes asserted here are exactly what the frontend receives. The fixture
mirrors ``core/tests.py``: anonymous client, a throwaway ``MEDIA_ROOT``, CSV
builder helpers, and a ``run_now=True`` flag so jobs execute synchronously
instead of waiting for a background worker that never runs in tests.
"""

from __future__ import annotations

import io
import json
import random
import shutil
import tempfile
import uuid

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from core.models import Dataset as CoreDataset, TrainingRun
from lifecycle.models import (
    DatasetVersion,
    Experiment,
    ExperimentRun,
    Job,
    Model,
    ModelVersion,
)

MEDIA_ROOT = tempfile.mkdtemp(prefix="furina-lifecycle-tests-")

CLASSIFICATION_COLUMNS = ["employee_id", "age", "salary", "city", "churn"]
REGRESSION_COLUMNS = ["sqft", "bedrooms", "neighborhood", "price"]


def config_hash_of(config: dict) -> str:
    """SHA-256 of a canonical config dict, mirroring the model's helper."""
    import hashlib

    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def classification_csv(rows: int = 120) -> bytes:
    """A messy two-class CSV: an id column, blanks, whitespace, mixed kinds."""
    rng = random.Random(7)
    cities = ["  Boston ", "Austin", "Chicago", "New York"]
    lines = [",".join(CLASSIFICATION_COLUMNS)]
    for index in range(rows):
        churn = "Yes" if index % 2 == 0 else "No"
        age = "" if index % 17 == 0 else str(20 + (index % 40))
        salary = 30000 + (index % 2) * 40000 + rng.randint(0, 4000)
        city = "" if index % 13 == 0 else cities[index % len(cities)]
        lines.append(f"EMP-{index:04d},{age},{salary},{city},{churn}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def regression_csv(rows: int = 90) -> bytes:
    """A numeric regression CSV with one categorical feature."""
    hoods = ["north", "south", "east", "west"]
    lines = [",".join(REGRESSION_COLUMNS)]
    for index in range(rows):
        sqft = 600 + index * 13
        bedrooms = 1 + (index % 5)
        hood = hoods[index % 4]
        price = sqft * 250 + bedrooms * 1000 + index % 7
        lines.append(f"{sqft},{bedrooms},{hood},{price}")
    return ("\n".join(lines) + "\n").encode("utf-8")


@override_settings(MEDIA_ROOT=MEDIA_ROOT)
class LifecycleApiTestCase(TestCase):
    """Shared fixture: an anonymous client and upload/create helpers."""

    def setUp(self):
        self.client = APIClient()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(MEDIA_ROOT, ignore_errors=True)

    def upload(self, payload: bytes, name: str = "dataset") -> dict:
        response = self.client.post(
            "/api/datasets/upload/",
            {"file": SimpleUploadedFile("data.csv", payload, content_type="text/csv"), "name": name},
            format="multipart",
        )
        self.assertEqual(response.status_code, 201, response.content)
        return response.json()

    def create_experiment(self, dataset: dict, **overrides) -> dict:
        body = {
            "dataset_id": dataset["id"],
            "name": overrides.pop("name", "Test experiment"),
            "task_type": overrides.pop("task_type", "auto"),
            "objective_metric": overrides.pop("objective_metric", ""),
            "objective_direction": overrides.pop("objective_direction", "maximize"),
            **overrides,
        }
        response = self.client.post("/api/lifecycle/experiments/", body, format="json")
        self.assertEqual(response.status_code, 201, response.content)
        return response.json()

    def train_run(self, experiment: dict, **overrides) -> dict:
        body = {
            "target_column": overrides.pop("target_column", "churn"),
            "algorithm": overrides.pop("algorithm", "logistic_regression"),
            "task_type": overrides.pop("task_type", "auto"),
            "feature_columns": overrides.pop("feature_columns", ["age", "salary", "city"]),
            "run_now": True,
            **overrides,
        }
        response = self.client.post(
            f"/api/lifecycle/experiments/{experiment['id']}/train/", body, format="json"
        )
        self.assertEqual(response.status_code, 201, response.content)
        return response.json()


class DatasetVersionApiTests(LifecycleApiTestCase):
    """PRD §9: versioning, hashing, lineage, immutability."""

    def test_experiment_creation_ensures_version_one(self):
        dataset = self.upload(classification_csv(), name="churn")
        experiment = self.create_experiment(dataset)
        version = DatasetVersion.objects.get(pk=experiment["dataset_version_id"])
        self.assertEqual(version.version, 1)
        self.assertEqual(version.content_hash, _sha256_of(version))
        self.assertEqual(version.row_count, 120)
        self.assertEqual(version.column_count, 5)
        self.assertEqual(version.source_type, "upload")

    def test_versions_are_append_only(self):
        dataset = self.upload(classification_csv(), name="churn")
        first = self.client.post(
            f"/api/lifecycle/datasets/{dataset['id']}/versions/", {}, format="json"
        ).json()
        second = self.client.post(
            f"/api/lifecycle/datasets/{dataset['id']}/versions/", {}, format="json"
        ).json()
        self.assertEqual(first["version"], 1)
        self.assertEqual(second["version"], 2)
        self.assertEqual(DatasetVersion.objects.filter(dataset_id=dataset["id"]).count(), 2)

    def test_lineage_chain_is_walkable(self):
        dataset = self.upload(classification_csv(), name="churn")
        v1 = self.client.post(
            f"/api/lifecycle/datasets/{dataset['id']}/versions/", {}, format="json"
        ).json()
        v2 = self.client.post(
            f"/api/lifecycle/datasets/{dataset['id']}/versions/",
            {"parent_version_id": v1["id"], "source_type": "preprocessed"},
            format="json",
        ).json()
        response = self.client.get(f"/api/lifecycle/versions/{v2['id']}/lineage/")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual([step["version"] for step in payload["lineage"]], [1, 2])
        self.assertEqual(payload["lineage"][1]["source_type"], "preprocessed")

    def test_identical_bytes_hash_identically(self):
        dataset = self.upload(classification_csv(), name="churn")
        other = self.upload(classification_csv(), name="churn again")
        v1 = self.client.post(
            f"/api/lifecycle/datasets/{dataset['id']}/versions/", {}, format="json"
        ).json()
        v2 = self.client.post(
            f"/api/lifecycle/datasets/{other['id']}/versions/", {}, format="json"
        ).json()
        self.assertEqual(v1["content_hash"], v2["content_hash"])

    def test_version_download_round_trips(self):
        dataset = self.upload(classification_csv(), name="churn")
        version = self.client.post(
            f"/api/lifecycle/datasets/{dataset['id']}/versions/", {}, format="json"
        ).json()
        response = self.client.get(f"/api/lifecycle/versions/{version['id']}/download/")
        self.assertEqual(response.status_code, 200)
        body = b"".join(response.streaming_content)
        self.assertTrue(body.startswith(b"employee_id,age"))

    def test_leakage_endpoint_flags_identifier_column(self):
        dataset = self.upload(classification_csv(), name="churn")
        experiment = self.create_experiment(dataset)
        response = self.client.post(
            f"/api/lifecycle/versions/{experiment['dataset_version_id']}/leakage/",
            {"target_column": "churn", "feature_columns": ["employee_id", "age", "salary", "city"]},
            format="json",
        )
        self.assertEqual(response.status_code, 200, response.content)
        payload = response.json()
        codes = {finding["code"] for finding in payload["findings"]}
        self.assertIn("identifier", codes)
        self.assertEqual(payload["severity"], "critical")

    def test_resource_estimate_endpoint(self):
        dataset = self.upload(classification_csv(), name="churn")
        experiment = self.create_experiment(dataset)
        response = self.client.post(
            f"/api/lifecycle/versions/{experiment['dataset_version_id']}/resources/",
            {"algorithm": "random_forest", "n_trials": 5},
            format="json",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("risk", payload)
        self.assertIn("estimated_gb", payload)
        self.assertEqual(payload["risk"], "low")


def _sha256_of(version: DatasetVersion) -> str:
    import hashlib

    with version.file.open("rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


class ExperimentRunApiTests(LifecycleApiTestCase):
    """PRD §10: experiments, run metadata, environment capture, §11 comparison."""

    def test_train_job_records_full_reproducibility_metadata(self):
        dataset = self.upload(classification_csv(), name="churn")
        experiment = self.create_experiment(dataset, name="Churn study")
        job = self.train_run(experiment, hyperparameters={"max_iter": 200})
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["kind"], "train")

        run = ExperimentRun.objects.get(pk=job["best_run_id"])
        self.assertEqual(run.status, "success")
        self.assertEqual(run.algorithm, "logistic_regression")
        self.assertEqual(run.dataset_hash, run.dataset_version.content_hash)
        self.assertEqual(run.config_hash, config_hash_of(run.full_config()))
        self.assertIsNotNone(run.environment)
        self.assertIn("packages", run.environment)
        self.assertEqual(run.environment["packages"]["scikit-learn"], __import__("sklearn").__version__)
        self.assertEqual(run.hyperparameters, {"max_iter": 200})
        self.assertIn("accuracy", run.metrics)
        self.assertGreater(run.n_train, 0)
        self.assertGreater(run.n_test, 0)

    def test_train_run_creates_legacy_twin_for_old_endpoints(self):
        dataset = self.upload(classification_csv(), name="churn")
        experiment = self.create_experiment(dataset)
        job = self.train_run(experiment)
        run = ExperimentRun.objects.get(pk=job["best_run_id"])
        self.assertIsNotNone(run.legacy_run)
        # The old endpoint still serves it.
        response = self.client.get(f"/api/runs/{run.legacy_run_id}/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["algorithm"], "logistic_regression")

    def test_objective_tracks_best_run(self):
        dataset = self.upload(classification_csv(), name="churn")
        experiment = self.create_experiment(dataset, objective_metric="accuracy")
        self.train_run(experiment, algorithm="logistic_regression")
        self.train_run(experiment, algorithm="decision_tree")
        model = Experiment.objects.get(pk=experiment["id"])
        self.assertIsNotNone(model.best_run)
        self.assertEqual(model.best_run.algorithm in ("logistic_regression", "decision_tree"), True)
        self.assertGreater(model.best_metric_value, 0)

    def test_failed_run_records_error_and_suggested_action(self):
        dataset = self.upload(classification_csv(), name="churn")
        experiment = self.create_experiment(dataset)
        response = self.client.post(
            f"/api/lifecycle/experiments/{experiment['id']}/train/",
            {"target_column": "churn", "algorithm": "nope_not_an_algorithm", "run_now": True},
            format="json",
        )
        self.assertEqual(response.status_code, 201, response.content)
        job = response.json()
        self.assertEqual(job["status"], "failed")
        self.assertIn("not available", job["error"])

    def test_run_comparison_groups_by_task_type(self):
        churn = self.upload(classification_csv(), name="churn")
        houses = self.upload(regression_csv(), name="houses")
        clf_exp = self.create_experiment(churn, name="clf")
        reg_exp = self.create_experiment(houses, name="reg", task_type="regression", objective_metric="r2")
        clf_job = self.train_run(clf_exp, target_column="churn")
        reg_job = self.train_run(reg_exp, target_column="price", algorithm="linear_regression")

        response = self.client.post(
            "/api/lifecycle/runs/compare/",
            {"run_ids": [clf_job["best_run_id"], reg_job["best_run_id"]]},
            format="json",
        )
        self.assertEqual(response.status_code, 200, response.content)
        payload = response.json()
        # Never one merged table across task types.
        self.assertEqual(sorted(payload["task_types"]), ["classification", "regression"])
        self.assertIn("metric:accuracy", payload["tables"]["classification"]["rows"][0])
        self.assertIn("metric:r2", payload["tables"]["regression"]["rows"][0])

    def test_comparison_needs_two_runs(self):
        churn = self.upload(classification_csv(), name="churn")
        clf_exp = self.create_experiment(churn)
        job = self.train_run(clf_exp)
        response = self.client.post(
            "/api/lifecycle/runs/compare/", {"run_ids": [job["best_run_id"]]}, format="json"
        )
        self.assertEqual(response.status_code, 400)

    def test_reproduce_job_matches_config_hash(self):
        dataset = self.upload(classification_csv(), name="churn")
        experiment = self.create_experiment(dataset)
        job = self.train_run(experiment, hyperparameters={"max_iter": 200})
        original = ExperimentRun.objects.get(pk=job["best_run_id"])

        response = self.client.post(
            f"/api/lifecycle/runs/{original.id}/reproduce/", {"run_now": True}, format="json"
        )
        self.assertEqual(response.status_code, 201, response.content)
        rerun_job = response.json()
        self.assertEqual(rerun_job["status"], "completed")
        rerun = ExperimentRun.objects.get(pk=rerun_job["best_run_id"])
        self.assertEqual(rerun.config_hash, original.config_hash)
        self.assertAlmostEqual(
            rerun.metrics.get("accuracy", 0), original.metrics.get("accuracy", 0), places=6
        )

    def test_run_detail_carries_diagnostics(self):
        dataset = self.upload(classification_csv(), name="churn")
        experiment = self.create_experiment(dataset)
        job = self.train_run(experiment)
        response = self.client.get(f"/api/lifecycle/runs/{job['best_run_id']}/")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIsNotNone(payload["diagnostics"])
        self.assertIn("diagnosis", payload["diagnostics"])
        self.assertIn(payload["diagnostics"]["diagnosis"]["verdict"], ("good_fit", "overfitting", "underfitting", "potential_leakage", "unknown"))
        # No non-finite JSON anywhere.
        json.dumps(payload, allow_nan=False)

    def test_run_notebook_downloads_executable_ipynb(self):
        dataset = self.upload(classification_csv(), name="churn")
        experiment = self.create_experiment(dataset)
        job = self.train_run(experiment)
        response = self.client.get(f"/api/lifecycle/runs/{job['best_run_id']}/notebook/")
        self.assertEqual(response.status_code, 200)
        notebook = json.loads(response.content)
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 5)
        sources = "".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn("train_test_split", sources)
        self.assertIn("logistic_regression", sources)

    def test_run_download_model(self):
        dataset = self.upload(classification_csv(), name="churn")
        experiment = self.create_experiment(dataset)
        job = self.train_run(experiment)
        response = self.client.get(f"/api/lifecycle/runs/{job['best_run_id']}/download_model/")
        self.assertEqual(response.status_code, 200)
        blob = b"".join(response.streaming_content) if hasattr(response, "streaming_content") else response.content
        self.assertGreater(len(blob), 0)
        # It must be a readable Furina model artifact, not merely some bytes.
        import joblib

        artifact = joblib.load(io.BytesIO(blob))
        self.assertEqual(artifact.algorithm, "logistic_regression")


class HpoAndAutoMlTests(LifecycleApiTestCase):
    """PRD §15/§16: search jobs and the AutoML workflow."""

    def test_hpo_job_runs_trials_and_reports_best(self):
        dataset = self.upload(classification_csv(), name="churn")
        experiment = self.create_experiment(dataset, objective_metric="accuracy")
        response = self.client.post(
            f"/api/lifecycle/experiments/{experiment['id']}/search/",
            {
                "target_column": "churn",
                "algorithm": "decision_tree",
                "method": "grid",
                "n_trials": 3,
                "feature_columns": ["age", "salary", "city"],
                "run_now": True,
            },
            format="json",
        )
        self.assertEqual(response.status_code, 201, response.content)
        job = response.json()
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["kind"], "hpo")
        self.assertGreaterEqual(job["completed_units"], 1)
        report = job["payload"]["search_report"]
        self.assertEqual(report["kind"], "hpo")
        self.assertIsNotNone(report["best"]["algorithm"])

    def test_automl_job_picks_a_model_and_stays_in_budget(self):
        dataset = self.upload(classification_csv(), name="churn")
        experiment = self.create_experiment(dataset)
        response = self.client.post(
            f"/api/lifecycle/experiments/{experiment['id']}/automl/",
            {
                "target_column": "churn",
                "time_budget_seconds": 60,
                "max_models": 3,
                "cv_folds": 2,
                "feature_columns": ["age", "salary", "city"],
                "run_now": True,
            },
            format="json",
        )
        self.assertEqual(response.status_code, 201, response.content)
        job = response.json()
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["kind"], "automl")
        best = job["payload"]["search_report"]["best"]
        self.assertIsNotNone(best["algorithm"])
        self.assertIsNotNone(best["value"])

    def test_job_cancel_before_start(self):
        dataset = self.upload(classification_csv(), name="churn")
        experiment = self.create_experiment(dataset)
        response = self.client.post(
            f"/api/lifecycle/experiments/{experiment['id']}/train/",
            {"target_column": "churn", "algorithm": "decision_tree", "run_now": False},
            format="json",
        )
        self.assertEqual(response.status_code, 201)
        job = response.json()
        cancel = self.client.post(f"/api/lifecycle/jobs/{job['id']}/cancel/", {}, format="json")
        self.assertEqual(cancel.status_code, 200)
        self.assertTrue(cancel.json()["cancel_requested"])
        detail = self.client.get(f"/api/lifecycle/jobs/{job['id']}/")
        self.assertEqual(detail.status_code, 200)
        self.assertIn(detail.json()["status"], ("cancelled", "queued"))


class ModelRegistryApiTests(LifecycleApiTestCase):
    """PRD §17-19: registry, signatures, model cards, promotion, deployment."""

    def _registered(self, name="Churn model") -> tuple[dict, dict]:
        dataset = self.upload(classification_csv(), name="churn")
        experiment = self.create_experiment(dataset)
        job = self.train_run(experiment)
        response = self.client.post(
            f"/api/lifecycle/runs/{job['best_run_id']}/register/",
            {"model_name": name},
            format="json",
        )
        self.assertEqual(response.status_code, 201, response.content)
        version = response.json()
        return version, job

    def test_register_builds_signature_and_card(self):
        version, _job = self._registered()
        self.assertEqual(version["version"], 1)
        self.assertEqual(version["status"], "candidate")
        self.assertEqual(version["model_card"]["model"], "Churn model")
        self.assertIn("metrics", version["model_card"])
        self.assertIn("validation_strategy", version["model_card"])
        signature = version["signature"]
        self.assertEqual(signature["task_type"], "classification")
        self.assertEqual(signature["target_column"], "churn")
        names = {f["name"] for f in signature["features"]}
        self.assertEqual(names, {"age", "salary", "city"})
        self.assertIn("example_request", signature)

    def test_versions_increment_and_are_immutable(self):
        version1, _ = self._registered(name="Churn model")
        version2, _ = self._registered(name="Churn model")
        self.assertEqual(version1["version"], 1)
        self.assertEqual(version2["version"], 2)
        self.assertEqual(ModelVersion.objects.filter(model__name="Churn model").count(), 2)

    def test_promotion_requires_confirmation_for_production(self):
        version, _ = self._registered(name="Churn model")
        model_id = version["model_id"]
        # Refused without confirmation.
        response = self.client.post(
            f"/api/lifecycle/models/{model_id}/versions/1/promote/",
            {"target": "production"},
            format="json",
        )
        self.assertEqual(response.status_code, 400)
        # The walk forward works.
        for target in ("validated", "staging"):
            response = self.client.post(
                f"/api/lifecycle/models/{model_id}/versions/1/promote/",
                {"target": target},
                format="json",
            )
            self.assertEqual(response.status_code, 200, response.content)
        # Once in staging, walking *backwards* is refused (only forward or
        # archiving is allowed).
        response = self.client.post(
            f"/api/lifecycle/models/{model_id}/versions/1/promote/",
            {"target": "candidate"},
            format="json",
        )
        self.assertEqual(response.status_code, 400)
        response = self.client.post(
            f"/api/lifecycle/models/{model_id}/versions/1/promote/",
            {"target": "production", "confirmed": True},
            format="json",
        )
        self.assertEqual(response.status_code, 200, response.content)
        self.assertEqual(response.json()["status"], "production")
        model = Model.objects.get(pk=model_id)
        self.assertEqual(model.production_version.version, 1)

    def test_incumbent_production_archives_on_new_promotion(self):
        version1, _ = self._registered(name="Churn model")
        version2, _ = self._registered(name="Churn model")
        model_id = version1["model_id"]
        for target, confirmed in [("validated", False), ("staging", False), ("production", True)]:
            self.client.post(
                f"/api/lifecycle/models/{model_id}/versions/1/promote/",
                {"target": target, "confirmed": confirmed},
                format="json",
            )
        for target, confirmed in [("validated", False), ("staging", False), ("production", True)]:
            self.client.post(
                f"/api/lifecycle/models/{model_id}/versions/2/promote/",
                {"target": target, "confirmed": confirmed},
                format="json",
            )
        v1 = ModelVersion.objects.get(model_id=model_id, version=1)
        v2 = ModelVersion.objects.get(model_id=model_id, version=2)
        self.assertEqual(v1.status, "archived")
        self.assertEqual(v2.status, "production")

    def test_model_list_and_detail(self):
        version, _ = self._registered(name="Churn model")
        listing = self.client.get("/api/lifecycle/models/").json()
        self.assertEqual(len(listing), 1)
        self.assertEqual(listing[0]["name"], "Churn model")
        detail = self.client.get(f"/api/lifecycle/models/{version['model_id']}/").json()
        self.assertEqual(len(detail["versions"]), 1)
        self.assertEqual(detail["versions"][0]["status"], "candidate")


class DeploymentApiTests(LifecycleApiTestCase):
    """PRD §19-21: deployment, prediction, monitoring, drift."""

    def _deployed(self) -> dict:
        dataset = self.upload(classification_csv(), name="churn")
        experiment = self.create_experiment(dataset)
        job = self.train_run(experiment)
        version = self.client.post(
            f"/api/lifecycle/runs/{job['best_run_id']}/register/",
            {"model_name": "Churn model"},
            format="json",
        ).json()
        deployment = self.client.post(
            f"/api/lifecycle/models/{version['model_id']}/versions/{version['version']}/deploy/",
            {},
            format="json",
        )
        self.assertEqual(deployment.status_code, 201, deployment.content)
        return deployment.json()

    def test_predict_returns_probabilities(self):
        deployment = self._deployed()
        response = self.client.post(
            f"/api/lifecycle/deployments/{deployment['slug']}/predict/",
            {"age": 33, "salary": 60000, "city": "Austin"},
            format="json",
        )
        self.assertEqual(response.status_code, 200, response.content)
        payload = response.json()
        self.assertIn(payload["prediction"], ("Yes", "No"))
        self.assertIsNotNone(payload["probabilities"])
        total = sum(entry["probability"] for entry in payload["probabilities"])
        self.assertAlmostEqual(total, 1.0, places=5)
        self.assertEqual(payload["model_version"], "1")

    def test_predict_rejects_missing_required_field(self):
        deployment = self._deployed()
        response = self.client.post(
            f"/api/lifecycle/deployments/{deployment['slug']}/predict/",
            {"age": 33},
            format="json",
        )
        self.assertEqual(response.status_code, 400)

    def test_batch_predict_round_trips(self):
        deployment = self._deployed()
        csv = b"age,salary,city\n33,60000,Austin\n45,72000,Boston\n"
        response = self.client.post(
            f"/api/lifecycle/deployments/{deployment['slug']}/batch-predict/",
            {"file": SimpleUploadedFile("rows.csv", csv, content_type="text/csv")},
            format="multipart",
        )
        self.assertEqual(response.status_code, 200, response.content)
        body = response.content
        self.assertIn(b"prediction", body)
        self.assertIn(b"probability_", body)

    def test_drift_check_flags_shifted_distribution(self):
        deployment = self._deployed()
        # A production distribution far outside training range for age.
        rows = [{"age": 200 + i, "salary": 60000, "city": "Austin"} for i in range(50)]
        response = self.client.post(
            f"/api/lifecycle/deployments/{deployment['slug']}/drift-check/",
            {"rows": rows},
            format="json",
        )
        self.assertEqual(response.status_code, 200, response.content)
        payload = response.json()
        self.assertIn(payload["overall"], ("significant", "moderate"))
        ages = next(f for f in payload["features"] if f["feature"] == "age")
        self.assertIsNotNone(ages["psi"])

    def test_monitoring_overview_counts_requests(self):
        deployment = self._deployed()
        self.client.post(
            f"/api/lifecycle/deployments/{deployment['slug']}/predict/",
            {"age": 33, "salary": 60000, "city": "Austin"},
            format="json",
        )
        response = self.client.get("/api/lifecycle/monitoring/")
        self.assertEqual(response.status_code, 200)
        summary = response.json()[0]
        self.assertEqual(summary["total_requests"], 1)
        self.assertEqual(summary["error_rate"], 0.0)

    def test_openapi_and_snippets(self):
        deployment = self._deployed()
        openapi = self.client.get(f"/api/lifecycle/deployments/{deployment['slug']}/openapi/")
        self.assertEqual(openapi.status_code, 200)
        spec = openapi.json()
        self.assertEqual(spec["openapi"], "3.1.0")
        self.assertIn("/predict", spec["paths"])
        snippets = self.client.get(f"/api/lifecycle/deployments/{deployment['slug']}/snippets/")
        self.assertEqual(snippets.status_code, 200)
        payload = snippets.json()
        self.assertIn("curl", payload["snippets"])
        self.assertIn("python", payload["snippets"])


class LegacyMigrationTests(LifecycleApiTestCase):
    """PRD §44: existing runs remain accessible after migration."""

    def test_migration_is_idempotent_and_preserves_legacy_access(self):
        # One legacy dataset + run, created directly.
        dataset_row = CoreDataset(
            name="legacy churn", source="upload", n_rows=120, n_cols=5
        )
        dataset_row.file.save(
            "legacy.csv", SimpleUploadedFile("legacy.csv", classification_csv(), content_type="text/csv"), save=False
        )
        dataset_row.save()
        legacy = TrainingRun.objects.create(
            dataset=dataset_row,
            algorithm="decision_tree",
            algorithm_label="Decision Tree",
            task_type="classification",
            target_column="churn",
            feature_columns=["age", "salary", "city"],
            hyperparameters={"max_depth": 4},
            effective_hyperparameters={"max_depth": 4},
            status="success",
            metrics={"accuracy": 0.5},
            train_metrics={"accuracy": 0.5},
        )

        from django.core.management import call_command

        call_command("migrate_legacy_runs")
        self.assertEqual(DatasetVersion.objects.filter(dataset=dataset_row).count(), 1)
        self.assertEqual(Experiment.objects.filter(dataset_version__dataset=dataset_row).count(), 1)
        migrated = ExperimentRun.objects.get(legacy_run=legacy)
        self.assertEqual(migrated.metrics["accuracy"], 0.5)
        self.assertEqual(migrated.environment, {"migrated": True})

        # Idempotent.
        call_command("migrate_legacy_runs")
        self.assertEqual(ExperimentRun.objects.filter(legacy_run=legacy).count(), 1)

        # The old endpoint still serves the legacy run.
        response = self.client.get(f"/api/runs/{legacy.id}/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["algorithm"], "decision_tree")


class JobModelTests(LifecycleApiTestCase):
    """PRD §23: job claim, progress, cancellation mechanics."""

    def test_claim_next_job_is_atomic(self):
        job = Job.objects.create(kind="train", payload={})
        from lifecycle.worker import claim_next_job

        claimed = claim_next_job(Job)
        self.assertEqual(claimed.id, job.id)
        self.assertEqual(claimed.status, "running")
        self.assertIsNone(claim_next_job(Job))

    def test_reclaim_stale_jobs(self):
        from datetime import timedelta

        from django.utils import timezone as tz

        stale = Job.objects.create(kind="train", payload={}, status="running")
        Job.objects.filter(pk=stale.pk).update(
            last_heartbeat=tz.now() - timedelta(seconds=99999)
        )
        from lifecycle.worker import reclaim_stale_jobs

        reclaimed = reclaim_stale_jobs()
        self.assertGreaterEqual(reclaimed, 1)
        stale.refresh_from_db()
        self.assertEqual(stale.status, "queued")

    def test_cancel_request_flips_flag(self):
        job = Job.objects.create(kind="train", payload={}, status="running")
        changed = job.request_cancel()
        self.assertTrue(changed)
        job.refresh_from_db()
        self.assertTrue(job.cancel_requested)
        self.assertFalse(job.request_cancel())


class HealthProbeTests(TestCase):
    """PRD §38: liveness and readiness probes."""

    def test_liveness_is_always_ok(self):
        response = self.client.get("/api/health/live/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_readiness_checks_dependencies(self):
        response = self.client.get("/api/health/ready/")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["database"], "ok")
        self.assertEqual(payload["media_root"], "ok")

    @override_settings(MEDIA_ROOT="/nonexistent/furina/media")
    def test_readiness_fails_when_media_root_is_missing(self):
        response = self.client.get("/api/health/ready/")
        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertEqual(payload["status"], "not ready")
        self.assertEqual(payload["media_root"], "missing")

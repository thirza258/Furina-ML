"""End-to-end tests for the Furina ML HTTP API.

Every test drives the real endpoints through DRF's test client, so the shapes
asserted here are exactly what the frontend receives. Uploaded files and model
artefacts are written to a throwaway ``MEDIA_ROOT`` that is removed afterwards.
"""

from __future__ import annotations

import io
import json
import random
import shutil
import tempfile

import joblib
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from core.models import Dataset, TrainingRun

MEDIA_ROOT = tempfile.mkdtemp(prefix="furina-tests-")

CLASSIFICATION_COLUMNS = ["employee_id", "age", "salary", "city", "churn"]
REGRESSION_COLUMNS = ["sqft", "bedrooms", "neighborhood", "price"]


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
    rng = random.Random(11)
    hoods = ["north", "south", "east", "west"]
    lines = [",".join(REGRESSION_COLUMNS)]
    for index in range(rows):
        sqft = 600 + index * 13
        bedrooms = 1 + (index % 5)
        hood = hoods[index % len(hoods)]
        price = 40000 + sqft * 180 + bedrooms * 5000 + rng.randint(-3000, 3000)
        lines.append(f"{sqft},{bedrooms},{hood},{price:.2f}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def multiclass_csv(rows: int = 120) -> bytes:
    """An integer-coded three-class target -- the classification trap."""
    rng = random.Random(3)
    lines = ["f1,f2,f3,grade"]
    for index in range(rows):
        grade = index % 3
        base = grade * 10
        lines.append(
            f"{base + rng.random():.4f},{base * 2 + rng.random():.4f},"
            f"{rng.random():.4f},{grade}"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def semicolon_latin1_csv(rows: int = 60) -> bytes:
    """A semicolon-separated latin-1 file, to prove decoding and normalizing."""
    lines = ["a;b;café_name;label"]
    for index in range(rows):
        label = "oui" if index % 2 == 0 else "non"
        lines.append(f"{index};{index * 2};Café n°{index};{label}")
    return ("\n".join(lines) + "\n").encode("latin-1")


@override_settings(MEDIA_ROOT=MEDIA_ROOT)
class ApiTestCase(TestCase):
    """Shared client plus small helpers for uploading and training."""

    maxDiff = None

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(MEDIA_ROOT, ignore_errors=True)
        super().tearDownClass()

    def setUp(self) -> None:
        self.client = APIClient()

    def upload(self, payload: bytes, filename: str = "data.csv", **extra) -> dict:
        """Upload a CSV and return the parsed DatasetDetail."""
        body = {"file": SimpleUploadedFile(filename, payload, content_type="text/csv")}
        body.update(extra)
        response = self.client.post("/api/datasets/upload/", body, format="multipart")
        self.assertEqual(response.status_code, 201, response.content)
        return response.json()

    def train(self, **body) -> tuple[int, dict]:
        """POST /api/train/ and return (status_code, payload)."""
        response = self.client.post("/api/train/", body, format="json")
        return response.status_code, response.json()


class HealthTests(ApiTestCase):
    def test_health_reports_the_stack(self) -> None:
        response = self.client.get("/api/health/")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        for key in ("sklearn", "pandas", "numpy"):
            self.assertIsInstance(payload[key], str)
            self.assertTrue(payload[key])


class UploadTests(ApiTestCase):
    def test_upload_profiles_the_dataset(self) -> None:
        detail = self.upload(classification_csv(), name="Churn")
        self.assertEqual(detail["source"], "upload")
        self.assertEqual(detail["name"], "Churn")
        self.assertEqual(detail["n_rows"], 120)
        self.assertEqual(detail["n_cols"], len(CLASSIFICATION_COLUMNS))
        self.assertGreater(detail["size_bytes"], 0)
        self.assertIsNone(detail["parent"])
        self.assertIn("/download/", detail["download_url"])
        self.assertTrue(detail["download_url"].startswith("http"))

        by_name = {column["name"]: column for column in detail["columns"]}
        self.assertEqual(set(by_name), set(CLASSIFICATION_COLUMNS))
        self.assertEqual(by_name["churn"]["suggested_task"], "classification")
        self.assertTrue(by_name["churn"]["suitable_as_target"])
        self.assertFalse(by_name["employee_id"]["suitable_as_target"])
        self.assertEqual(by_name["age"]["kind"], "numeric")
        self.assertGreater(by_name["age"]["missing"], 0)

        self.assertEqual(detail["preview"]["headers"], CLASSIFICATION_COLUMNS)
        self.assertEqual(len(detail["preview"]["rows"]), 10)

    def test_upload_normalizes_semicolon_latin1_to_comma_utf8(self) -> None:
        detail = self.upload(semicolon_latin1_csv(), filename="euro.csv")
        self.assertEqual(detail["n_cols"], 4)
        self.assertEqual(detail["preview"]["headers"], ["a", "b", "café_name", "label"])

        stored = Dataset.objects.get(pk=detail["id"])
        with stored.file.open("rb") as handle:
            first_line = handle.readline().decode("utf-8")
        self.assertEqual(first_line.strip(), "a,b,café_name,label")

        # The normalized copy must still be readable by every later endpoint.
        frame = stored.load_dataframe()
        self.assertEqual(list(frame.columns), ["a", "b", "café_name", "label"])

    def test_upload_respects_an_explicit_delimiter(self) -> None:
        detail = self.upload(semicolon_latin1_csv(), delimiter=";")
        self.assertEqual(detail["n_cols"], 4)

    def test_garbage_upload_is_rejected_with_400(self) -> None:
        junk = bytes(range(256)) * 40
        response = self.client.post(
            "/api/datasets/upload/",
            {"file": SimpleUploadedFile("junk.csv", junk, content_type="text/csv")},
            format="multipart",
        )
        self.assertEqual(response.status_code, 400, response.content)
        self.assertIn("detail", response.json())

    def test_empty_upload_is_rejected(self) -> None:
        response = self.client.post(
            "/api/datasets/upload/",
            {"file": SimpleUploadedFile("empty.csv", b"", content_type="text/csv")},
            format="multipart",
        )
        self.assertEqual(response.status_code, 400)

    def test_wrong_extension_is_rejected(self) -> None:
        response = self.client.post(
            "/api/datasets/upload/",
            {"file": SimpleUploadedFile("model.xlsx", b"a,b\n1,2\n")},
            format="multipart",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("errors", response.json())

    def test_missing_file_is_rejected(self) -> None:
        response = self.client.post("/api/datasets/upload/", {}, format="multipart")
        self.assertEqual(response.status_code, 400)


class DatasetTests(ApiTestCase):
    def test_list_is_a_bare_array_newest_first(self) -> None:
        self.upload(classification_csv(), name="first")
        self.upload(regression_csv(), name="second")
        response = self.client.get("/api/datasets/")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIsInstance(payload, list)
        self.assertEqual([item["name"] for item in payload], ["second", "first"])
        self.assertEqual(
            set(payload[0]),
            {"id", "name", "source", "n_rows", "n_cols", "created_at"},
        )

    def test_detail_honours_preview_rows(self) -> None:
        detail = self.upload(classification_csv())
        response = self.client.get(f"/api/datasets/{detail['id']}/?preview_rows=3")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["preview"]["rows"]), 3)

    def test_preview_rows_is_clamped(self) -> None:
        detail = self.upload(classification_csv())
        response = self.client.get(f"/api/datasets/{detail['id']}/?preview_rows=9999")
        self.assertLessEqual(len(response.json()["preview"]["rows"]), 200)

    def test_missing_dataset_is_404_json(self) -> None:
        response = self.client.get("/api/datasets/00000000-0000-4000-8000-000000000000/")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Not found."})

    def test_delete_removes_the_dataset(self) -> None:
        detail = self.upload(classification_csv())
        response = self.client.delete(f"/api/datasets/{detail['id']}/")
        self.assertEqual(response.status_code, 204)
        self.assertFalse(Dataset.objects.filter(pk=detail["id"]).exists())

    def test_download_returns_csv(self) -> None:
        detail = self.upload(classification_csv())
        response = self.client.get(f"/api/datasets/{detail['id']}/download/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/csv", response["Content-Type"])
        self.assertIn("attachment;", response["Content-Disposition"])
        body = b"".join(response.streaming_content)
        self.assertTrue(body.startswith(b"employee_id,age"))


class PreprocessTests(ApiTestCase):
    def recipe(self) -> list[dict]:
        return [
            {"op": "remove_duplicates", "columns": None, "params": None},
            {"op": "strip_whitespace", "columns": None, "params": None},
            {"op": "fill_mean", "columns": ["age"], "params": None},
            {"op": "fill_mode", "columns": ["city"], "params": None},
            {"op": "lowercase", "columns": ["city"], "params": None},
            {"op": "one_hot_encode", "columns": ["city"], "params": None},
        ]

    def test_recipe_creates_a_child_dataset(self) -> None:
        parent = self.upload(classification_csv(), name="raw")
        response = self.client.post(
            f"/api/datasets/{parent['id']}/preprocess/",
            {"steps": self.recipe(), "name": "clean"},
            format="json",
        )
        self.assertEqual(response.status_code, 200, response.content)
        payload = response.json()
        child, report = payload["dataset"], payload["report"]

        self.assertEqual(child["source"], "preprocessed")
        self.assertEqual(child["parent"], parent["id"])
        self.assertEqual(child["name"], "clean")
        self.assertGreater(child["n_cols"], parent["n_cols"])
        self.assertIsNotNone(child["id"])

        self.assertEqual(report["rows_before"], parent["n_rows"])
        self.assertEqual(report["cols_before"], parent["n_cols"])
        self.assertEqual(report["cols_after"], child["n_cols"])
        self.assertEqual(len(report["steps"]), len(self.recipe()))
        for entry in report["steps"]:
            self.assertEqual(
                set(entry),
                {"op", "columns", "message", "skipped", "rows_after", "cols_after"},
            )
        self.assertFalse(any(entry["skipped"] for entry in report["steps"]))

        # The child is a real dataset that can be trained on afterwards.
        stored = Dataset.objects.get(pk=child["id"])
        self.assertEqual(len(stored.load_dataframe()), child["n_rows"])

    def test_dry_run_does_not_persist(self) -> None:
        parent = self.upload(classification_csv())
        before = Dataset.objects.count()
        response = self.client.post(
            f"/api/datasets/{parent['id']}/preprocess/",
            {"steps": [{"op": "drop_columns", "columns": ["employee_id"]}], "dry_run": True},
            format="json",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIsNone(payload["dataset"]["id"])
        self.assertEqual(payload["dataset"]["n_cols"], parent["n_cols"] - 1)
        self.assertEqual(payload["dataset"]["parent"], parent["id"])
        self.assertEqual(Dataset.objects.count(), before)

    def test_unknown_op_is_400(self) -> None:
        parent = self.upload(classification_csv())
        response = self.client.post(
            f"/api/datasets/{parent['id']}/preprocess/",
            {"steps": [{"op": "make_coffee"}]},
            format="json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("make_coffee", response.json()["detail"])

    def test_impossible_step_is_skipped_not_fatal(self) -> None:
        parent = self.upload(classification_csv())
        response = self.client.post(
            f"/api/datasets/{parent['id']}/preprocess/",
            {"steps": [{"op": "fill_mean", "columns": ["city"]}]},
            format="json",
        )
        self.assertEqual(response.status_code, 200)
        report = response.json()["report"]
        self.assertTrue(report["steps"][0]["skipped"])
        self.assertTrue(report["steps"][0]["message"])


class AlgorithmTests(ApiTestCase):
    def test_list_is_a_bare_array_of_specs(self) -> None:
        response = self.client.get("/api/algorithms/")
        self.assertEqual(response.status_code, 200)
        specs = response.json()
        self.assertIsInstance(specs, list)
        keys = {spec["key"] for spec in specs}
        for required in (
            "logistic_regression",
            "ridge_classifier",
            "linear_regression",
            "ridge_regression",
            "lasso_regression",
            "elastic_net",
            "svm",
            "sgd",
            "decision_tree",
            "random_forest",
            "extra_trees",
            "gradient_boosting",
            "hist_gradient_boosting",
            "ada_boost",
            "knn",
            "naive_bayes",
            "mlp",
        ):
            self.assertIn(required, keys)
        for spec in specs:
            self.assertEqual(
                set(spec),
                {
                    "key",
                    "label",
                    "family",
                    "tasks",
                    "description",
                    "supports_proba",
                    "supports_feature_importance",
                    "hyperparameters",
                },
            )
            for param in spec["hyperparameters"]:
                self.assertEqual(
                    set(param),
                    {
                        "name",
                        "label",
                        "type",
                        "default",
                        "min",
                        "max",
                        "step",
                        "options",
                        "description",
                        "advanced",
                    },
                )

    def test_task_filter(self) -> None:
        regression = self.client.get("/api/algorithms/?task=regression").json()
        self.assertTrue(regression)
        for spec in regression:
            self.assertIn("regression", spec["tasks"])
        self.assertNotIn("naive_bayes", {spec["key"] for spec in regression})


class TrainingTests(ApiTestCase):
    def test_classification_run_is_complete(self) -> None:
        dataset = self.upload(classification_csv(), name="churn data")
        code, run = self.train(
            dataset_id=dataset["id"],
            target_column="churn",
            feature_columns=["age", "salary", "city"],
            algorithm="random_forest",
            cv_folds=3,
            name="baseline",
        )
        self.assertEqual(code, 201, run)
        self.assertEqual(run["task_type"], "classification")
        self.assertEqual(run["status"], "success")
        self.assertEqual(run["primary_metric_name"], "accuracy")
        self.assertEqual(run["dataset_id"], dataset["id"])
        self.assertEqual(run["dataset_name"], "churn data")
        self.assertGreater(run["metrics"]["accuracy"], 0.5)
        self.assertEqual(run["primary_metric"], run["metrics"]["accuracy"])
        self.assertEqual(
            set(run["metrics"]),
            {
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "f2",
                "roc_auc",
                "log_loss",
                "matthews_corrcoef",
            },
        )
        self.assertEqual(run["confusion_matrix"]["labels"], ["No", "Yes"])
        self.assertEqual(len(run["confusion_matrix"]["matrix"]), 2)
        self.assertEqual(len(run["confusion_matrix"]["matrix"][0]), 2)
        self.assertEqual(len(run["per_class"]), 2)
        self.assertTrue(run["feature_importance"])
        self.assertEqual(len(run["input_schema"]), 3)
        self.assertIsNone(run["residual_sample"])
        self.assertEqual(run["cv"]["folds"], 3)
        self.assertEqual(len(run["cv"]["scores"]), 3)
        self.assertGreater(run["n_train"], 0)
        self.assertGreater(run["n_test"], 0)
        self.assertIsInstance(run["duration_ms"], int)
        self.assertIn("/download_model/", run["model_download_url"])
        self.assertIn("/report/", run["report_download_url"])
        self.assertIsNone(run["error"])

        # Nothing non-finite may reach the client.
        json.dumps(run, allow_nan=False)

    def test_regression_run_has_no_classification_keys(self) -> None:
        dataset = self.upload(regression_csv(), name="houses")
        code, run = self.train(
            dataset_id=dataset["id"],
            target_column="price",
            algorithm="linear_regression",
        )
        self.assertEqual(code, 201, run)
        self.assertEqual(run["task_type"], "regression")
        self.assertEqual(run["primary_metric_name"], "r2")
        self.assertIn("r2", run["metrics"])
        self.assertNotIn("accuracy", run["metrics"])
        self.assertIsNone(run["classes"])
        self.assertIsNone(run["confusion_matrix"])
        self.assertIsNone(run["per_class"])
        self.assertTrue(run["residual_sample"])
        self.assertEqual(
            set(run["residual_sample"][0]),
            {"actual", "predicted"},
        )

    def test_integer_target_is_treated_as_classification(self) -> None:
        dataset = self.upload(multiclass_csv(), name="grades")
        code, run = self.train(
            dataset_id=dataset["id"],
            target_column="grade",
            algorithm="logistic_regression",
        )
        self.assertEqual(code, 201, run)
        self.assertEqual(run["task_type"], "classification")
        self.assertEqual(run["classes"], ["0", "1", "2"])
        self.assertEqual(len(run["confusion_matrix"]["matrix"]), 3)

    def test_explicit_task_type_overrides_inference(self) -> None:
        dataset = self.upload(multiclass_csv())
        code, run = self.train(
            dataset_id=dataset["id"],
            target_column="grade",
            algorithm="linear_regression",
            task_type="regression",
        )
        self.assertEqual(code, 201, run)
        self.assertEqual(run["task_type"], "regression")
        self.assertIn("r2", run["metrics"])

    def test_forcing_classification_on_a_continuous_target_warns(self) -> None:
        dataset = self.upload(regression_csv())
        code, run = self.train(
            dataset_id=dataset["id"],
            target_column="bedrooms",
            algorithm="decision_tree",
            task_type="classification",
        )
        self.assertEqual(code, 201, run)
        code, run = self.train(
            dataset_id=dataset["id"],
            target_column="sqft",
            algorithm="decision_tree",
            task_type="classification",
        )
        self.assertEqual(code, 201, run)
        self.assertTrue(run["warnings"])
        self.assertTrue(any("sqft" in message for message in run["warnings"]))

    def test_forcing_regression_on_a_text_target_is_400(self) -> None:
        dataset = self.upload(classification_csv())
        code, payload = self.train(
            dataset_id=dataset["id"],
            target_column="churn",
            algorithm="linear_regression",
            task_type="regression",
        )
        self.assertEqual(code, 400, payload)
        self.assertIn("detail", payload)

    def test_unknown_column_is_400(self) -> None:
        dataset = self.upload(classification_csv())
        code, payload = self.train(
            dataset_id=dataset["id"],
            target_column="does_not_exist",
            algorithm="random_forest",
        )
        self.assertEqual(code, 400)
        self.assertIn("does_not_exist", payload["detail"])

        code, payload = self.train(
            dataset_id=dataset["id"],
            target_column="churn",
            feature_columns=["age", "nope"],
            algorithm="random_forest",
        )
        self.assertEqual(code, 400)
        self.assertIn("nope", payload["detail"])

    def test_unknown_algorithm_is_400(self) -> None:
        dataset = self.upload(classification_csv())
        code, payload = self.train(
            dataset_id=dataset["id"], target_column="churn", algorithm="magic"
        )
        self.assertEqual(code, 400)
        self.assertNotIn("Traceback", payload["detail"])

    def test_algorithm_that_cannot_do_the_task_is_400(self) -> None:
        dataset = self.upload(regression_csv())
        code, payload = self.train(
            dataset_id=dataset["id"],
            target_column="price",
            algorithm="naive_bayes",
            task_type="regression",
        )
        self.assertEqual(code, 400)

    def test_unknown_dataset_is_400(self) -> None:
        code, payload = self.train(
            dataset_id="00000000-0000-4000-8000-000000000000",
            target_column="churn",
            algorithm="random_forest",
        )
        self.assertEqual(code, 400)

    def test_bad_request_body_is_400_with_field_errors(self) -> None:
        response = self.client.post("/api/train/", {}, format="json")
        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertIn("errors", payload)
        self.assertIn("dataset_id", payload["errors"])

    def test_hyperparameters_are_recorded_raw_and_effective(self) -> None:
        dataset = self.upload(classification_csv())
        code, run = self.train(
            dataset_id=dataset["id"],
            target_column="churn",
            algorithm="random_forest",
            hyperparameters={"n_estimators": 25, "bogus_setting": 4},
        )
        self.assertEqual(code, 201, run)
        self.assertEqual(run["hyperparameters"], {"n_estimators": 25, "bogus_setting": 4})
        self.assertEqual(run["effective_hyperparameters"]["n_estimators"], 25)
        self.assertNotIn("bogus_setting", run["effective_hyperparameters"])
        self.assertTrue(any("bogus_setting" in note for note in run["warnings"]))
        json.dumps(run["effective_hyperparameters"], allow_nan=False)


class RunEndpointTests(ApiTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.dataset = self.upload(classification_csv(), name="churn data")
        code, self.run = self.train(
            dataset_id=self.dataset["id"],
            target_column="churn",
            feature_columns=["age", "salary", "city"],
            algorithm="random_forest",
            cv_folds=3,
        )
        self.assertEqual(code, 201, self.run)

    def test_run_list_is_a_bare_array(self) -> None:
        response = self.client.get("/api/runs/")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIsInstance(payload, list)
        self.assertEqual(len(payload), 1)
        self.assertEqual(
            set(payload[0]),
            {
                "id",
                "name",
                "dataset_id",
                "dataset_name",
                "algorithm",
                "algorithm_label",
                "task_type",
                "target_column",
                "status",
                "primary_metric_name",
                "primary_metric",
                "created_at",
            },
        )

    def test_run_list_filters_by_dataset(self) -> None:
        other = self.upload(regression_csv())
        self.train(
            dataset_id=other["id"],
            target_column="price",
            algorithm="linear_regression",
        )
        all_runs = self.client.get("/api/runs/").json()
        self.assertEqual(len(all_runs), 2)
        filtered = self.client.get(f"/api/runs/?dataset_id={other['id']}").json()
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["dataset_id"], other["id"])

    def test_run_detail_round_trips(self) -> None:
        response = self.client.get(f"/api/runs/{self.run['id']}/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], self.run["id"])

    def test_run_delete(self) -> None:
        response = self.client.delete(f"/api/runs/{self.run['id']}/")
        self.assertEqual(response.status_code, 204)
        self.assertFalse(TrainingRun.objects.filter(pk=self.run["id"]).exists())

    def test_predict_with_blank_and_unseen_values(self) -> None:
        response = self.client.post(
            f"/api/runs/{self.run['id']}/predict/",
            {
                "rows": [
                    {"age": "41", "salary": "", "city": "Atlantis"},
                    {"age": 29, "salary": 51000, "city": "Austin"},
                ]
            },
            format="json",
        )
        self.assertEqual(response.status_code, 200, response.content)
        predictions = response.json()["predictions"]
        self.assertEqual(len(predictions), 2)
        for item in predictions:
            self.assertIn(item["prediction"], self.run["classes"])
            self.assertEqual(len(item["probabilities"]), 2)
            total = sum(entry["probability"] for entry in item["probabilities"])
            self.assertAlmostEqual(total, 1.0, places=5)

    def test_predict_rejects_an_empty_row_list(self) -> None:
        response = self.client.post(
            f"/api/runs/{self.run['id']}/predict/", {"rows": []}, format="json"
        )
        self.assertEqual(response.status_code, 400)

    def test_predict_rejects_a_non_numeric_number(self) -> None:
        response = self.client.post(
            f"/api/runs/{self.run['id']}/predict/",
            {"rows": [{"age": "not-a-number", "salary": 1000, "city": "Austin"}]},
            format="json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("age", response.json()["detail"])

    def test_download_model_is_a_usable_artifact(self) -> None:
        response = self.client.get(f"/api/runs/{self.run['id']}/download_model/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/octet-stream")
        blob = b"".join(response.streaming_content)
        self.assertGreater(len(blob), 0)

        artifact = joblib.load(io.BytesIO(blob))
        self.assertEqual(artifact.task_type, "classification")
        self.assertEqual(artifact.target_column, "churn")
        self.assertEqual(artifact.feature_columns, ["age", "salary", "city"])

        from core.ml.predict import predict as offline_predict

        offline = offline_predict(artifact, [{"age": 33, "salary": 60000, "city": "Austin"}])
        self.assertIn(offline[0]["prediction"], self.run["classes"])

    def test_report_markdown(self) -> None:
        response = self.client.get(f"/api/runs/{self.run['id']}/report/?format=md")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/markdown", response["Content-Type"])
        body = response.content.decode("utf-8")
        self.assertTrue(body.startswith("# Training report:"))
        self.assertIn("## Metrics", body)
        self.assertIn("## Confusion matrix", body)

    def test_report_defaults_to_markdown(self) -> None:
        response = self.client.get(f"/api/runs/{self.run['id']}/report/")
        self.assertIn("text/markdown", response["Content-Type"])

    def test_report_json(self) -> None:
        response = self.client.get(f"/api/runs/{self.run['id']}/report/?format=json")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("report", payload)
        self.assertIn("run", payload)
        self.assertEqual(payload["run"]["id"], self.run["id"])

    def test_report_rejects_an_unknown_format(self) -> None:
        response = self.client.get(f"/api/runs/{self.run['id']}/report/?format=pdf")
        self.assertEqual(response.status_code, 400)

    def test_missing_run_is_404(self) -> None:
        response = self.client.get("/api/runs/00000000-0000-4000-8000-000000000000/")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Not found."})


class EndToEndTests(ApiTestCase):
    def test_upload_preprocess_train_predict(self) -> None:
        raw = self.upload(classification_csv(), name="raw")
        processed = self.client.post(
            f"/api/datasets/{raw['id']}/preprocess/",
            {
                "steps": [
                    {"op": "drop_columns", "columns": ["employee_id"]},
                    {"op": "strip_whitespace", "columns": None},
                    {"op": "fill_mean", "columns": ["age"]},
                    {"op": "fill_mode", "columns": ["city"]},
                ],
                "name": "clean",
            },
            format="json",
        ).json()["dataset"]

        code, run = self.train(
            dataset_id=processed["id"],
            target_column="churn",
            algorithm="gradient_boosting",
            cv_folds=3,
        )
        self.assertEqual(code, 201, run)
        self.assertGreater(run["metrics"]["accuracy"], 0.5)

        row = {field["name"]: field["default"] for field in run["input_schema"]}
        response = self.client.post(
            f"/api/runs/{run['id']}/predict/", {"rows": [row]}, format="json"
        )
        self.assertEqual(response.status_code, 200, response.content)
        self.assertIn(
            response.json()["predictions"][0]["prediction"], run["classes"]
        )

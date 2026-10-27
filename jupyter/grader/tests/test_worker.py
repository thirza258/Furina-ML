"""Tests for the grader worker, with a stand-in for Docker.

    cd jupyter/grader && python -m unittest discover -s tests

The fake container unpacks what the worker uploads into a temporary folder
and runs the notebook there with this Python, so the whole path -- template
lookup, bundle, run, read-back, scoring -- is exercised without Docker. The
sandbox's isolation itself (no network, limits) is Docker's job and is not
tested here. Needs the docker SDK, nbclient and ipykernel; skipped otherwise.
"""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

NEEDS = ("docker", "nbclient", "ipykernel")
AVAILABLE = all(importlib.util.find_spec(name) for name in NEEDS)

if AVAILABLE:
    from docker.errors import NotFound

    from furina_grader import worker
    from test_grading import CORRECT, TEMPLATE, submission


class FakeContainer:
    def __init__(self, root: Path, entrypoint: list[str], command: list[str], **options) -> None:
        self.root, self.options, self.id = root, options, "fake0123456789"
        self.argv = [part.replace("/tmp", str(root)) for part in entrypoint + command]
        self.removed = False

    def put_archive(self, path: str, data: bytes) -> bool:
        assert path == "/tmp"
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            names = tar.getnames()
            assert all(name == "grade" or name.startswith("grade/") for name in names), names
            assert {member.uid for member in tar.getmembers()} == {worker.SANDBOX_UID}
            tar.extractall(self.root, filter="data")
        return True

    def start(self) -> None:
        argv = [sys.executable if self.argv[0] == "python" else self.argv[0], *self.argv[1:]]
        subprocess.run(argv, cwd=self.root / "grade", capture_output=True, timeout=300, check=True)

    def wait(self, timeout: int) -> dict:
        return {"StatusCode": 0}

    def kill(self) -> None:
        pass

    def get_archive(self, path: str):
        target = Path(path.replace("/tmp", str(self.root)))
        if not target.exists():
            raise NotFound("no such file")
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            tar.add(target, arcname=target.name)
        return iter([buffer.getvalue()]), {}

    def remove(self, force: bool = False) -> None:
        self.removed = True


class FakeClient:
    def __init__(self, root: Path) -> None:
        self.root, self.created = root, []
        self.containers = self

    def create(self, image: str, *, entrypoint, command, **options) -> FakeContainer:
        container = FakeContainer(self.root, entrypoint, command, image=image, **options)
        self.created.append(container)
        return container


@unittest.skipUnless(AVAILABLE, f"needs {', '.join(NEEDS)}")
class WorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        root = Path(self.folder.name)
        (root / "templates" / "demo-course").mkdir(parents=True)
        (root / "templates" / "demo-course" / "01_lesson.ipynb").write_text(json.dumps(TEMPLATE))
        (root / "templates" / "demo-course" / "helper.py").write_text("VALUE = 1\n")
        (root / "sandbox").mkdir()
        self.settings = worker.Settings({
            "GRADER_TOKEN": "token", "GRADER_TEMPLATES": str(root / "templates"),
            "GRADER_CELL_TIMEOUT": "10",
        })
        self.client = FakeClient(root / "sandbox")

    def tearDown(self) -> None:
        self.folder.cleanup()

    def grade(self, path: str, notebook: dict) -> dict:
        return worker.grade(self.settings, self.client, {"id": "s1", "path": path, "notebook": notebook})

    def test_grades_a_submission_in_a_locked_down_container(self) -> None:
        outcome = self.grade("demo-course/01_lesson.ipynb", submission(**CORRECT))
        self.assertEqual((outcome["status"], outcome["passed"], outcome["total"]), ("graded", 3, 3))
        self.assertEqual(outcome["score"], 1.0)
        self.assertEqual(len(outcome["result"]["items"]), 3)

        container = self.client.created[0]
        self.assertTrue(container.removed)
        options = container.options
        self.assertEqual(options["network_mode"], "none")
        self.assertEqual(options["cap_drop"], ["ALL"])
        self.assertIn("no-new-privileges", options["security_opt"])
        self.assertEqual(options["user"], str(worker.SANDBOX_UID))
        self.assertEqual(options["mem_limit"], options["memswap_limit"])
        # The lesson folder's helper modules travel with the notebook.
        self.assertTrue((self.client.root / "grade" / "helper.py").exists())

    def test_unknown_or_malicious_paths_are_errors(self) -> None:
        for path in ("demo-course/02_missing.ipynb", "../templates/demo-course/01_lesson.ipynb",
                     "demo-course/../../etc/passwd.ipynb", "/etc/passwd"):
            outcome = self.grade(path, submission(**CORRECT))
            self.assertEqual(outcome["status"], "error", path)
        self.assertEqual(self.client.created, [])

    def test_a_notebook_that_never_finishes_writing_is_an_error(self) -> None:
        original = FakeContainer.start
        FakeContainer.start = lambda self: None  # the runner never runs
        try:
            outcome = self.grade("demo-course/01_lesson.ipynb", submission(**CORRECT))
        finally:
            FakeContainer.start = original
        self.assertEqual(outcome["status"], "error")
        self.assertIn("stopped before", outcome["error"])
        self.assertTrue(self.client.created[0].removed)


if __name__ == "__main__":
    unittest.main()

"""Tests for building and scoring grading notebooks.

    cd jupyter/grader && python -m unittest discover -s tests

The execution tests run real kernels and need nbclient and ipykernel (the
student image's environment has both); without them they are skipped. To
check the grader against every course notebook, run the course notebooks
through it with filled-in and unfinished exercises.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from furina_grader.build import build_grading_notebook, score

EXECUTE = Path(__file__).resolve().parents[1] / "furina_grader" / "execute.py"
CAN_EXECUTE = all(importlib.util.find_spec(name) for name in ("nbclient", "ipykernel"))

SETUP = '''def practice(name, yours, reference, cases):
    try:
        for args in cases:
            if yours(*args) != reference(*args):
                print(f"❌ {name}")
                return reference
    except NotImplementedError:
        print(f"✍️ {name}: not written yet")
        return reference
    except Exception as error:
        print(f"❌ {name} raised {error!r}")
        return reference
    print(f"✅ {name}")
    return yours


def check(name, function, cases):
    for args, expected in cases:
        try:
            result = function(*args)
        except NotImplementedError:
            print(f"{name}: not implemented yet")
            return
        assert result == expected, f"{name}{args} returned {result!r}, expected {expected!r}"
    print(f"{name}: passed")'''


def code(cell_id: str, source: str, tags: tuple[str, ...] = ()) -> dict:
    metadata = {"tags": list(tags)} if tags else {}
    return {"cell_type": "code", "id": cell_id, "metadata": metadata, "source": source,
            "execution_count": None, "outputs": []}


TEMPLATE = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {"kernelspec": {"name": "python", "display_name": "Python"}},
    "cells": [
        {"cell_type": "markdown", "id": "intro", "metadata": {}, "source": "# A lesson"},
        code("setup", SETUP),
        code("refs", "def double_reference(x):\n    return 2 * x\n\n\n"
                     "def square_reference(x):\n    return x * x", tags=("reference",)),
        code("ex-double", "def double(x):\n    raise NotImplementedError"),
        code("check-double", 'double = practice("double", double, double_reference, cases=[(1,), (2,)])'),
        code("ex-square", "def square(x):\n    raise NotImplementedError"),
        code("check-square", 'check("square", square, [((3,), 9)])\n'
                             'check("square of a negative", square, [((-2,), 4)])'),
        # A learner's own experiment after the check: never graded.
        code("play", 'practice("double", lambda x: 0, double_reference, cases=[(1,)])'),
    ],
}


def submission(**sources: str) -> dict:
    notebook = copy.deepcopy(TEMPLATE)
    for cell in notebook["cells"]:
        if cell["id"] in sources:
            cell["source"] = sources[cell["id"]]
    return notebook


CORRECT = {"ex-double": "def double(x):\n    return x + x",
           "ex-square": "def square(x):\n    return x ** 2"}


class PlanTests(unittest.TestCase):
    def test_graded_checks_come_from_the_template(self) -> None:
        plan = build_grading_notebook(TEMPLATE, submission(**CORRECT))
        self.assertEqual(
            [(item.cell_id, item.names) for item in plan.items],
            [("check-double", ["double"]), ("check-square", ["square", "square of a negative"])],
        )
        self.assertEqual((plan.answered, plan.unanswered), (["ex-double", "ex-square"], []))

    def test_edited_or_deleted_checks_change_nothing(self) -> None:
        tampered = submission(**CORRECT, **{"check-square": "pass"})
        tampered["cells"] = [cell for cell in tampered["cells"] if cell["id"] != "check-double"]
        plan = build_grading_notebook(TEMPLATE, tampered)
        sources = {cell["id"]: "".join(cell["source"]) for cell in plan.notebook["cells"]}
        self.assertIn('practice("double", double', sources["check-double"])
        self.assertIn('check("square", square', sources["check-square"])
        self.assertEqual(sources["ex-double"], CORRECT["ex-double"])

    def test_only_exercise_cells_are_taken_from_the_submission(self) -> None:
        plan = build_grading_notebook(TEMPLATE, submission(refs="def double_reference(x):\n    return 0"))
        sources = {cell["id"]: cell["source"] for cell in plan.notebook["cells"]}
        self.assertIn("return 2 * x", sources["refs"])

    def test_cells_without_ids_are_matched_by_name(self) -> None:
        notebook = submission(**CORRECT)
        for cell in notebook["cells"]:
            del cell["id"]
        plan = build_grading_notebook(TEMPLATE, notebook)
        sources = {cell["id"]: cell["source"] for cell in plan.notebook["cells"]}
        self.assertEqual(sources["ex-square"], CORRECT["ex-square"])

    def test_missing_report_is_incomplete(self) -> None:
        plan = build_grading_notebook(TEMPLATE, TEMPLATE)
        result = score(plan, {"cells": []})
        self.assertFalse(result["complete"])
        self.assertEqual({item["status"] for item in result["items"]}, {"not-run"})


@unittest.skipUnless(CAN_EXECUTE, "needs nbclient and ipykernel")
class ExecutionTests(unittest.TestCase):
    def grade(self, notebook: dict) -> dict:
        plan = build_grading_notebook(TEMPLATE, notebook)
        with tempfile.TemporaryDirectory() as folder:
            grading, executed = Path(folder, "grading.ipynb"), Path(folder, "executed.ipynb")
            grading.write_text(json.dumps(plan.notebook))
            subprocess.run([sys.executable, str(EXECUTE), str(grading), str(executed), "10"],
                           check=True, capture_output=True, timeout=120)
            result = score(plan, json.loads(executed.read_text()))
        self.assertTrue(result["complete"])
        return result

    def statuses(self, result: dict) -> dict:
        return {item["name"]: item["status"] for item in result["items"]}

    def test_unfinished_notebook_scores_nothing(self) -> None:
        result = self.grade(TEMPLATE)
        self.assertEqual((result["passed"], result["total"], result["unanswered"]), (0, 3, 2))

    def test_correct_notebook_scores_everything(self) -> None:
        result = self.grade(submission(**CORRECT))
        self.assertEqual((result["score"], result["passed"]), (1.0, 3))
        self.assertIn("✅ double", result["items"][0]["message"])

    def test_each_check_is_scored_on_its_own(self) -> None:
        result = self.grade(submission(**{**CORRECT, "ex-square": "def square(x):\n    return x * 2"}))
        self.assertEqual(self.statuses(result), {
            "double": "passed", "square": "failed", "square of a negative": "failed",
        })
        # The first wrong check did not stop the second one from running.
        self.assertIn("returned 6", result["items"][1]["message"])
        self.assertIn("returned -4", result["items"][2]["message"])

    def test_redefining_the_checker_does_not_help(self) -> None:
        cheat = "def double(x):\n    raise NotImplementedError\n\n" \
                "practice = lambda name, yours, *args, **kwargs: yours\ncheck = lambda *a, **k: None"
        result = self.grade(submission(**{**CORRECT, "ex-double": cheat}))
        self.assertEqual(self.statuses(result), {
            "double": "failed", "square": "passed", "square of a negative": "passed",
        })

    def test_a_broken_cell_costs_only_its_own_checks(self) -> None:
        for broken in ("def double(x:\n    return 2 * x", "while True:\n    pass"):
            with self.subTest(broken=broken.splitlines()[0]):
                result = self.grade(submission(**{**CORRECT, "ex-double": broken}))
                statuses = self.statuses(result)
                self.assertNotEqual(statuses.pop("double"), "passed")
                self.assertEqual(set(statuses.values()), {"passed"})
                self.assertIn("Your ✍️ cell stopped with", result["items"][0]["message"])


if __name__ == "__main__":
    unittest.main()

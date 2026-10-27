"""Grade every course notebook as several kinds of student would hand it in.

    cd jupyter/grader && python tests/check_course_notebooks.py [variant ...] [--only text]

Run it after changing a lesson notebook or the grader. It needs the student
image's libraries plus nbclient and ipykernel, like
``notebooks/tools/check_notebooks.py --execute``. Variants (default: all):

    shipped           the template as is: every check must fail
    filled            every exercise answered: every check must pass
    noids             filled, with cell ids removed (as some editors save it):
                      every check must pass
    tamper-practice   one exercise left unfinished, with practice()/check()
                      redefined to always pass: that exercise must still fail
                      and every other check pass
    tamper-reference  one exercise wrong, with its reference replaced by it:
                      that exercise must fail (later checks may fail too;
                      that only costs the student)
    syntax, loop      one exercise cell with a syntax error / an endless loop:
                      that exercise fails, every other check passes

"Filled" answers delegate to the notebook's own reference solutions, which is
a valid answer here: the grader restores the references before each check.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from furina_grader.build import (  # noqa: E402
    _source,
    _tests_stub,
    build_grading_notebook,
    checker_calls,
    defined_names,
    is_stub,
    score,
)

NOTEBOOKS = HERE.parents[2] / "notebooks"
EXECUTE = HERE.parent / "furina_grader" / "execute.py"
VARIANTS = ["shipped", "filled", "noids", "tamper-practice", "tamper-reference", "syntax", "loop"]


def _solutions(template: dict) -> dict[str, str]:
    """Math notebooks keep their solutions as ``<name>_solution`` at the end."""
    found = {}
    for cell in template["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = _source(cell)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.endswith("_solution"):
                text = ast.get_source_segment(source, node)
                found[node.name[: -len("_solution")]] = re.sub(r"\b(\w+)_solution\b", r"\1", text)
    return found


def _answer(source: str, solved: dict[str, str], template_text: str) -> str:
    """An exercise cell answered with the notebook's own solutions."""
    tree = ast.parse(source)
    parts = []
    for node in tree.body:
        text = ast.get_source_segment(source, node)
        unfinished = any(
            isinstance(inner, ast.Raise) and "NotImplementedError" in ast.unparse(inner)
            for inner in ast.walk(node)
        )
        if isinstance(node, ast.FunctionDef) and unfinished and node.name in solved:
            text = solved[node.name]
        elif isinstance(node, ast.FunctionDef) and unfinished:
            arguments = node.args
            call = [a.arg for a in arguments.posonlyargs + arguments.args]
            call += [f"*{arguments.vararg.arg}"] if arguments.vararg else []
            call += [f"{a.arg}={a.arg}" for a in arguments.kwonlyargs]
            call += [f"**{arguments.kwarg.arg}"] if arguments.kwarg else []
            text = (f"def {node.name}({ast.unparse(arguments)}):\n"
                    f"    return {node.name}_reference({', '.join(call)})")
        elif isinstance(node, ast.ClassDef) and unfinished:
            camel = f"{node.name}Reference"
            reference = camel if f"class {camel}" in template_text else f"{node.name}_reference"
            text = f"{node.name} = {reference}"
        parts.append(text)
    return "\n\n".join(parts)


def filled(template: dict) -> dict:
    notebook = copy.deepcopy(template)
    solved, text = _solutions(template), json.dumps(template)
    for cell in notebook["cells"]:
        if is_stub(cell):
            cell["source"] = _answer(_source(cell), solved, text)
    return notebook


def check_exercises(template: dict) -> dict[str, str]:
    """check name -> the exercise it tests."""
    stub_names: set[str] = set()
    for cell in template["cells"]:
        if is_stub(cell):
            stub_names |= defined_names(_source(cell))
    mapping: dict[str, str] = {}
    for cell in template["cells"]:
        if cell["cell_type"] == "code" and not is_stub(cell):
            for _, name, yours in checker_calls(_source(cell)):
                exercise = _tests_stub(name, yours, stub_names)
                if exercise:
                    mapping.setdefault(name, exercise)
    return mapping


def _first_exercise(template: dict) -> dict | None:
    tested = set(check_exercises(template).values())
    return next(
        (cell for cell in template["cells"] if is_stub(cell) and defined_names(_source(cell)) & tested),
        None,
    )


def submission_for(variant: str, template: dict) -> dict:
    if variant == "shipped":
        return template
    notebook = filled(template)
    if variant == "noids":
        for cell in notebook["cells"]:
            cell.pop("id", None)
        return notebook
    if variant == "filled":
        return notebook
    first = _first_exercise(template)
    if first is None:
        return notebook
    target = next(cell for cell in notebook["cells"] if cell.get("id") == first.get("id"))
    names = sorted(defined_names(_source(first)))
    target["source"] = {
        "tamper-practice": _source(first)
        + "\n\npractice = lambda name, yours, *a, **k: yours\ncheck = lambda *a, **k: True\n",
        "tamper-reference": "".join(
            f"def {name}(*args, **kwargs):\n    return 0\n\n{name}_reference = {name}\n\n"
            for name in names
        ),
        "syntax": "def broken(:\n    pass\n",
        "loop": "while True:\n    pass\n",
    }[variant]
    return notebook


def verdict(variant: str, template: dict, result: dict) -> str:
    """'' when the result is as expected, otherwise what is wrong."""
    if not result["complete"]:
        return "the notebook did not run to the end"
    failed = {item["name"] for item in result["items"] if item["status"] != "passed"}
    if variant == "shipped":
        passed = {item["name"] for item in result["items"]} - failed
        return f"passed while unfinished: {sorted(passed)}" if passed else ""
    if variant in ("filled", "noids"):
        return "" if not failed else f"failed: {sorted(failed)}"
    first = _first_exercise(template)
    if first is None:
        return ""
    broken = set(defined_names(_source(first)))
    own = {name for name, exercise in check_exercises(template).items() if exercise in broken}
    if not own <= failed:
        return f"broken exercise passed: {sorted(own - failed)}"
    collateral = failed - own
    if collateral and variant != "tamper-reference":
        return f"other checks failed: {sorted(collateral)}"
    return ""


def grade(path: Path, variant: str, cell_timeout: int) -> tuple[Path, str, dict | None, str]:
    template = json.loads(path.read_text(encoding="utf-8"))
    plan = build_grading_notebook(template, submission_for(variant, template))
    with tempfile.TemporaryDirectory() as folder:
        work = Path(folder) / "grade"
        shutil.copytree(path.parent, work, ignore=shutil.ignore_patterns(
            "*.ipynb", "__pycache__", ".ipynb_checkpoints"))
        (work / "grading.ipynb").write_text(json.dumps(plan.notebook))
        run = subprocess.run(
            [sys.executable, str(EXECUTE), str(work / "grading.ipynb"),
             str(work / "executed.ipynb"), str(cell_timeout)],
            capture_output=True, text=True, timeout=1800,
        )
        if not (work / "executed.ipynb").exists():
            return path, variant, None, run.stderr.strip()[-500:] or "no executed notebook"
        executed = json.loads((work / "executed.ipynb").read_text())
    result = score(plan, executed)
    return path, variant, result, verdict(variant, template, result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("variants", nargs="*", help=f"any of {', '.join(VARIANTS)} (default: all)")
    parser.add_argument("--only", help="grade only notebooks whose path contains this")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--cell-timeout", type=int, default=20)
    args = parser.parse_args()

    variants = args.variants or VARIANTS
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        parser.error(f"unknown variant(s): {', '.join(sorted(unknown))}")
    paths = [
        path for path in sorted(NOTEBOOKS.rglob("*.ipynb"))
        if ".ipynb_checkpoints" not in path.parts and "tools" not in path.parts
        and (not args.only or args.only in str(path))
    ]
    jobs = [(path, variant) for path in paths for variant in variants]
    problems = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for path, variant, result, problem in pool.map(
            lambda job: grade(*job, args.cell_timeout), jobs
        ):
            name = path.relative_to(NOTEBOOKS)
            if result is None:
                problems += 1
                print(f"FAIL {variant:16s} {name}: {problem}", flush=True)
                continue
            mark = "FAIL" if problem else "ok  "
            problems += bool(problem)
            print(f"{mark} {variant:16s} {name}: {result['passed']}/{result['total']}"
                  + (f"  {problem}" if problem else ""), flush=True)
    print(f"{len(jobs)} grading run(s), {problems} problem(s).")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

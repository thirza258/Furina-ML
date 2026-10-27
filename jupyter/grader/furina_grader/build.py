"""Turn a template notebook and a student's submission into a grading notebook.

The grading notebook is the *template*, with only the student's ✍️ exercise
cells swapped in. Everything else -- the setup, the hidden references, the data
and the `practice` / `check` calls -- comes from the template, so deleting or
editing a check in the submitted file changes nothing.

Before every graded check cell a guard re-defines the trusted checkers
(`practice*`, `check`) and the `*_reference` functions from the template, then
wraps the checkers so that each call made by that cell is recorded. A student
cell that redefines `practice` to always pass is therefore undone before it can
matter, and calls made from any other cell are not recorded at all. A final
cell prints the recorded results.

What this does not stop: code in a student's function can still monkeypatch
libraries (for example `numpy.allclose`), reach the recorder, or tamper with
the process that runs the notebook. Like other notebook autograders, it resists
accidental and casual tampering, not a determined cheat; instructors review
submissions.
"""

from __future__ import annotations

import ast
import copy
import json
from dataclasses import dataclass, field

REPORT_CELL_ID = "furina-grader-report"
REPORT_MARKER = "FURINA_REPORT "
GRADED_TAG = "furina-graded"
#: Longest feedback kept for one check, in characters.
MESSAGE_LIMIT = 2000

_PRELUDE = '''# Furina grader: records the outcome of every practice()/check() call made by a graded cell.
def __furina_install():
    import contextlib
    import io
    import sys

    state = {"cell": None}
    results = {}

    class Tee(io.TextIOBase):
        def __init__(self, *streams):
            self.streams = streams

        def write(self, text):
            for stream in self.streams:
                stream.write(text)
            return len(text)

        def flush(self):
            for stream in self.streams:
                stream.flush()

    def wrap(function_name, original):
        def wrapped(*args, **kwargs):
            cell = state["cell"]
            if cell is None:
                return original(*args, **kwargs)
            strings = [i for i, value in enumerate(args) if isinstance(value, str)]
            name = str(args[strings[0]]) if strings else function_name
            index = strings[0] + 1 if strings else len(args)
            yours = args[index] if index < len(args) else None
            unfinished = []
            if function_name == "check" and callable(yours):
                # check() returns None both when it passes and while the exercise
                # is unfinished, so watch the learner's function directly.
                target = yours

                def yours(*inner_args, **inner_kwargs):
                    try:
                        return target(*inner_args, **inner_kwargs)
                    except NotImplementedError:
                        unfinished.append(True)
                        raise

                args = args[:index] + (yours,) + args[index + 1:]
            captured = io.StringIO()
            record = [name, False, ""]
            results.setdefault(cell, []).append(record)
            try:
                with contextlib.redirect_stdout(Tee(sys.stdout, captured)):
                    result = original(*args, **kwargs)
            except AssertionError as error:
                if function_name != "check":
                    record[2] = captured.getvalue() + f"AssertionError: {error}"
                    raise
                # A wrong answer must not stop the other checks in the cell.
                print(f"❌ {error}")
                record[2] = captured.getvalue() + f"❌ {error}"
                return False
            except BaseException as error:
                record[2] = captured.getvalue() + f"{type(error).__name__}: {error}"
                raise
            if function_name == "check":
                passed = not unfinished and result is not False
            else:
                # practice*() hands back the learner's function only when it matches.
                passed = yours is not None and result is yours
            record[1] = bool(passed)
            record[2] = captured.getvalue()
            return result

        wrapped.__furina_original__ = original
        return wrapped

    def enter(cell):
        state["cell"] = cell
        if cell is None:
            return
        namespace = globals()
        for key, value in list(namespace.items()):
            if (key.startswith("practice") or key == "check") and callable(value):
                namespace[key] = wrap(key, getattr(value, "__furina_original__", value))

    def stand_in(names):
        # Runs just before an exercise cell. If that cell fails (a syntax
        # error, a time-out), its names stay bound to these stand-ins, which
        # behave like the unfinished exercise: the checks fall back to the
        # reference, as they would for the student. Without them, a name
        # defined earlier in the notebook (a scaffold of the same name) would
        # be graded in the student's place.
        namespace = globals()
        for name in names:
            def unfinished(*args, _name=name, **kwargs):
                raise NotImplementedError(f"{_name} is not defined: its cell did not run")

            namespace[name] = unfinished

    return enter, stand_in, lambda: results


__furina_enter, __furina_stand_in, __furina_results = __furina_install()
del __furina_install'''

_REPORT = f'''__furina_enter(None)
import json as __furina_json
print({REPORT_MARKER!r} + __furina_json.dumps(__furina_results()))'''


def _source(cell) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _tags(cell) -> set[str]:
    return set(cell.get("metadata", {}).get("tags", []))


def is_reference(cell) -> bool:
    return "reference" in _tags(cell)


def is_stub(cell) -> bool:
    """A cell the learner types into: code that still raises NotImplementedError."""
    return (
        cell.get("cell_type") == "code"
        and not is_reference(cell)
        and "raise NotImplementedError" in _source(cell)
        and "_reference" not in _source(cell)
    )


def _parse(source: str):
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def defined_names(source: str) -> set[str]:
    """Top-level functions and classes a cell defines (empty when it does not parse)."""
    tree = _parse(source)
    if tree is None:
        return set()
    return {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def checker_calls(source: str) -> list[tuple[str, str, str | None]]:
    """(checker, check name, identifier passed as "yours") for each practice*/check call."""
    tree = _parse(source)
    if tree is None:
        return []
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        checker = node.func.id
        if not (checker.startswith("practice") or checker == "check"):
            continue
        positions = [
            i for i, arg in enumerate(node.args)
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        ]
        if not positions:
            continue
        name = node.args[positions[0]].value
        after = node.args[positions[0] + 1] if len(node.args) > positions[0] + 1 else None
        yours = after.id if isinstance(after, ast.Name) else None
        calls.append((checker, name, yours))
    return calls


def _tests_stub(name: str, yours: str | None, stub_names: set[str]) -> str | None:
    """The exercise a check call tests, or None when it tests something else.

    Usually the learner's function is passed directly (`practice("relu", relu,
    ...)`). Some checks pass a wrapper around it instead (`yours_with_seed`),
    and then the check's name is the exercise's name.
    """
    if yours in stub_names:
        return yours
    if yours is None or yours.endswith(("_solution", "_reference")):
        return None
    if name in stub_names:
        return name
    return next((stub for stub in sorted(stub_names) if yours.startswith(stub)), None)


def _definition_source(source: str, node) -> str | None:
    if node.decorator_list:
        lines = source.splitlines()
        return "\n".join(lines[node.decorator_list[0].lineno - 1:node.end_lineno])
    return ast.get_source_segment(source, node)


def _trusted_definitions(cell, first_code: bool) -> list[str]:
    """Function definitions to re-run before each check: checkers and references.

    The first code cell sets the notebook up, so all of its functions count. A
    cell that defines a checker brings the helpers beside it (`_close`,
    `_short`, ...). A reference cell brings its `*_reference` functions.
    """
    source = _source(cell)
    tree = _parse(source)
    if tree is None:
        return []
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    defines_checker = any(
        node.name.startswith("practice") or node.name == "check" for node in functions
    )
    if first_code or (defines_checker and not is_reference(cell)):
        keep = functions
    elif is_reference(cell):
        keep = [node for node in functions if node.name.endswith("_reference")]
    else:
        keep = []
    return [segment for node in keep if (segment := _definition_source(source, node))]


@dataclass
class GradedItem:
    cell_id: str
    names: list[str]
    #: Check name -> the id of the exercise cell it tests.
    exercises: dict[str, str] = field(default_factory=dict)


@dataclass
class GradingPlan:
    notebook: dict
    items: list[GradedItem] = field(default_factory=list)
    #: Exercise cells the student changed, and those still as shipped.
    answered: list[str] = field(default_factory=list)
    unanswered: list[str] = field(default_factory=list)

    @property
    def check_names(self) -> list[str]:
        return [name for item in self.items for name in item.names]


def bound_names(source: str) -> set[str]:
    """Names a cell binds at top level: definitions and plain assignments."""
    tree = _parse(source)
    if tree is None:
        return set()
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
    return names


def _match_student_cells(stub_cells, student_cells, template_sources):
    """Pair each exercise cell with the student's version of it, or None.

    A cell with the same id is the student's version. Notebooks downloaded from
    some editors (Colab, older Jupyter) lose cell ids; then the student's
    version is the next cell, in notebook order, that binds every name the
    exercise defines -- skipping cells left exactly as the template has them,
    such as a setup cell that defines a function of the same name.
    """
    code = [cell for cell in student_cells if cell.get("cell_type") == "code"]
    by_id = {cell.get("id"): cell for cell in code if cell.get("id")}
    matches = {}
    cursor = 0
    for stub in stub_cells:
        stub_id = stub.get("id")
        if stub_id in by_id:
            matches[stub_id] = by_id[stub_id]
            cursor = code.index(by_id[stub_id]) + 1
            continue
        wanted = defined_names(_source(stub))
        matches[stub_id] = None
        if not wanted:
            continue
        for position in range(cursor, len(code)):
            cell = code[position]
            source = _source(cell).strip()
            if source in template_sources or not wanted <= bound_names(source):
                continue
            matches[stub_id] = cell
            cursor = position + 1
            break
    return matches


def _code_cell(cell_id: str, source: str) -> dict:
    return {"cell_type": "code", "id": cell_id, "metadata": {}, "execution_count": None,
            "outputs": [], "source": source}


def build_grading_notebook(template: dict, submission: dict) -> GradingPlan:
    """Assemble the grading notebook and the list of graded checks."""
    notebook = copy.deepcopy(template)
    student_cells = submission.get("cells", []) if isinstance(submission, dict) else []
    student_cells = [cell for cell in student_cells if isinstance(cell, dict)]

    stub_cells = [cell for cell in notebook["cells"] if is_stub(cell)]
    stub_names: set[str] = set()
    for cell in stub_cells:
        stub_names |= defined_names(_source(cell))
    template_sources = {
        _source(cell).strip()
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code" and not is_stub(cell)
    }
    matches = _match_student_cells(stub_cells, student_cells, template_sources)
    exercise_cells = {
        name: str(cell.get("id")) for cell in stub_cells for name in defined_names(_source(cell))
    }

    plan = GradingPlan(notebook=notebook)
    tested: set[str] = set()
    trusted: list[str] = []
    stubs: list[tuple[dict, bool]] = []
    prelude_added = False
    cells = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            cells.append(cell)
            continue
        if is_stub(cell):
            if prelude_added and defined_names(_source(cell)):
                names = sorted(defined_names(_source(cell)))
                cells.append(_code_cell(f"furina-grader-stand-in-{len(cells)}", f"__furina_stand_in({names!r})"))
            student = matches.get(cell.get("id"))
            answered = student is not None and _source(student).strip() != _source(cell).strip()
            stubs.append((cell, answered))
            if answered:
                cell = dict(cell, source=_source(student))
            cell = dict(cell, outputs=[], execution_count=None)
        else:
            definitions = _trusted_definitions(cell, first_code=not prelude_added)
            graded, exercises = [], {}
            for _, name, yours in checker_calls(_source(cell)):
                exercise = _tests_stub(name, yours, stub_names)
                if exercise is not None and name not in graded:
                    graded.append(name)
                    exercises[name] = exercise_cells[exercise]
                    tested.add(exercise)
            if graded and prelude_added:
                cell_id = str(cell.get("id"))
                guard = "\n\n\n".join(trusted + [f"__furina_enter({cell_id!r})"])
                cell = dict(cell, source=f"{guard}\n\n# --- the check, from the template ---\n{_source(cell)}")
                cell["metadata"] = dict(cell.get("metadata", {}), tags=sorted(_tags(cell) | {GRADED_TAG}))
                plan.items.append(GradedItem(cell_id=cell_id, names=graded, exercises=exercises))
            trusted += definitions
        cells.append(cell)
        if GRADED_TAG in _tags(cell):
            # Stop recording: later cells' calls (a learner's own test, a demo)
            # must not count towards this cell's checks.
            cells.append(_code_cell(f"furina-grader-leave-{len(cells)}", "__furina_enter(None)"))
        if not prelude_added:
            prelude_added = True
            cells.append(_code_cell("furina-grader-prelude", _PRELUDE))
    cells.append(_code_cell(REPORT_CELL_ID, _REPORT))
    notebook["cells"] = cells

    for cell, answered in stubs:
        if defined_names(_source(cell)) & tested:
            (plan.answered if answered else plan.unanswered).append(str(cell.get("id")))
    return plan


def _cell_text(cell) -> str:
    """Printed output and errors of an executed cell, as plain text."""
    parts = []
    for output in cell.get("outputs", []):
        kind = output.get("output_type")
        if kind == "stream":
            text = output.get("text", "")
            parts.append("".join(text) if isinstance(text, list) else text)
        elif kind == "error":
            parts.append(f"{output.get('ename', 'Error')}: {output.get('evalue', '')}\n")
    return "".join(parts)


def _cell_error(cell) -> str | None:
    """The error an executed cell stopped with, if any."""
    for output in cell.get("outputs", []):
        if output.get("output_type") == "error":
            return f"{output.get('ename', 'Error')}: {output.get('evalue', '')}"
    return None


def read_report(executed: dict) -> dict | None:
    """The results the report cell printed, keyed by cell id; None when it printed none."""
    for cell in executed.get("cells", []):
        if cell.get("id") != REPORT_CELL_ID:
            continue
        for line in reversed(_cell_text(cell).splitlines()):
            if line.startswith(REPORT_MARKER):
                try:
                    report = json.loads(line[len(REPORT_MARKER):])
                except ValueError:
                    return None
                return report if isinstance(report, dict) else None
    return None


def score(plan: GradingPlan, executed: dict) -> dict:
    """Score an executed grading notebook against its plan.

    Each check passes when every call it made passed. A check whose cell never
    reached it (an earlier error, a time-out) did not run and counts as failed.
    """
    report = read_report(executed)
    found = report is not None
    report = report or {}
    outputs = {str(cell.get("id")): _cell_text(cell) for cell in executed.get("cells", [])}
    errors = {str(cell.get("id")): _cell_error(cell) for cell in executed.get("cells", [])}
    items = []
    for item in plan.items:
        records = report.get(item.cell_id, [])
        for name in item.names:
            calls = [record for record in records if isinstance(record, list) and record[:1] == [name]]
            if not calls:
                status = "not-run"
                message = outputs.get(item.cell_id, "").strip() or "This check did not run."
            else:
                status = "passed" if all(record[1] is True for record in calls) else "failed"
                # Calls nest (practice_with calls practice); the first, outermost
                # one captured everything the inner ones printed.
                message = str(calls[0][2]).strip()
            exercise_error = errors.get(item.exercises.get(name, ""))
            if status != "passed" and exercise_error:
                # Otherwise the check would only say "not written yet".
                message = f"Your ✍️ cell stopped with {exercise_error}\n\n{message}"
            items.append({"name": name, "status": status, "message": message[-MESSAGE_LIMIT:]})
    passed = sum(item["status"] == "passed" for item in items)
    total = len(items)
    return {
        "score": round(passed / total, 4) if total else None,
        "passed": passed,
        "total": total,
        "items": items,
        "answered": len(plan.answered),
        "unanswered": len(plan.unanswered),
        # False when the notebook stopped before the end (the kernel died or the
        # time limit ran out): the result is then incomplete.
        "complete": found,
    }

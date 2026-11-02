"""Check the course notebooks before they ship.

Usage (from the repository root)::

    python notebooks/tools/check_notebooks.py              # structure and dependency tiers
    python notebooks/tools/check_notebooks.py --execute    # also run each one from a clean kernel
    python notebooks/tools/check_notebooks.py --strip      # clear outputs in place, then check

Every notebook is a template that is copied into each student's workspace, so
it must ship clean (no outputs, no execution counts), declare the ``python``
kernel that both JupyterLite and the JupyterHub image provide, and declare its
dependency tier in ``metadata.furina``:

    "furina": {"course": "math-foundations", "tier": 0}

Tier 0 allows the Python standard library only; tier 1 adds NumPy and
Matplotlib; tier 2 adds pandas, SciPy and scikit-learn; tier 3 adds PyTorch.
A cell tagged ``comparison``, ``visualization`` or ``data`` may import a
library above the notebook's tier: to check a from-scratch result against a
reference implementation, to draw a plot, or to load a dataset, in a clearly
separated cell.

A notebook that compiles native code also declares its toolchain, "c", "c++"
or "cuda", in ``metadata.furina.toolchain``. Its course must declare a
``toolchain`` in ``frontend/src/data/courses`` as well, because the site
offers such notebooks no in-browser (JupyterLite) runtime. A "cuda" notebook
needs only a C++ compiler: it uses nvcc and an NVIDIA GPU when both are
present and a CPU emulator otherwise.

The import check is static and best-effort. It keeps course content honest; it
is not a security boundary, and it does not stop code that is determined to
load a module some other way.

``--execute`` runs every notebook top to bottom in a fresh kernel, from the
notebook's own folder, without saving the outputs. It needs nbclient, nbformat
and ipykernel; jupyter_client starts its standard ``python3`` kernel for the
``python`` name. Notebooks with a toolchain also need a C and a C++ compiler
(gcc and g++, or clang) on the PATH.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import sys
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parents[1]
REPO = NOTEBOOKS.parent
COURSE_DATA = REPO / "frontend" / "src" / "data" / "courses"

#: Libraries a learner's code may import at each tier, beyond the standard library.
TIER_LIBRARIES: dict[int, frozenset[str]] = {
    0: frozenset(),
    1: frozenset({"numpy", "matplotlib"}),
    2: frozenset({"numpy", "matplotlib", "pandas", "scipy", "sklearn"}),
    3: frozenset({"numpy", "matplotlib", "pandas", "scipy", "sklearn", "torch"}),
}
#: Third-party libraries the tier check knows about. Anything else that is not
#: part of the standard library is reported as unknown.
KNOWN_THIRD_PARTY = frozenset(
    {
        "numpy", "matplotlib", "pandas", "scipy", "sklearn", "torch", "tensorflow",
        "jax", "seaborn", "plotly", "statsmodels", "xgboost", "lightgbm", "sympy",
    }
)
EXEMPT_TAGS = frozenset({"comparison", "visualization", "data"})
KERNEL_NAME = "python"
#: Native toolchains a notebook may declare in metadata.furina.toolchain.
TOOLCHAINS = frozenset({"c", "c++", "cuda"})


def _source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _python_lines(source: str) -> tuple[str, list[str]]:
    """Drop IPython magics and shell escapes, returning them separately."""
    kept: list[str] = []
    escapes: list[str] = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("%", "!")):
            escapes.append(stripped)
            kept.append("")
        else:
            kept.append(line)
    return "\n".join(kept), escapes


def imported_modules(source: str) -> set[str]:
    """Top-level names imported by a block of Python source."""
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and node.args:
            func = node.func
            called = (
                func.id if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute)
                else ""
            )
            first = node.args[0]
            if called in ("__import__", "import_module") and isinstance(first, ast.Constant):
                if isinstance(first.value, str):
                    names.add(first.value.split(".")[0])
    return names


def _check_imports(
    where: str, modules: set[str], tier: int, local: set[str], problems: list[str]
) -> None:
    allowed = TIER_LIBRARIES[tier]
    for module in sorted(modules):
        if module in local or module in sys.stdlib_module_names:
            continue
        if module in KNOWN_THIRD_PARTY:
            if module not in allowed:
                problems.append(
                    f"{where}: imports '{module}', which tier {tier} does not allow outside "
                    "a cell tagged 'comparison' or 'visualization'"
                )
        else:
            problems.append(f"{where}: imports '{module}', which is not a known library")


def check_notebook(path: Path, problems: list[str]) -> dict | None:
    where = str(path.relative_to(REPO))
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        problems.append(f"{where}: not readable as JSON ({exc})")
        return None
    if notebook.get("nbformat") != 4:
        problems.append(f"{where}: nbformat must be 4")
    metadata = notebook.get("metadata", {})
    if metadata.get("kernelspec", {}).get("name") != KERNEL_NAME:
        problems.append(f"{where}: kernelspec name must be '{KERNEL_NAME}'")
    furina = metadata.get("furina", {})
    tier = furina.get("tier")
    if tier not in TIER_LIBRARIES:
        problems.append(f"{where}: metadata.furina.tier must be 0, 1, 2 or 3")
        tier = None
    if furina.get("course") != path.parent.name:
        problems.append(f"{where}: metadata.furina.course must be '{path.parent.name}'")
    toolchain = furina.get("toolchain")
    if toolchain is not None and toolchain not in TOOLCHAINS:
        problems.append(f"{where}: metadata.furina.toolchain must be one of {sorted(TOOLCHAINS)}")
    elif (toolchain is not None) != (path.parent.name in courses_with_toolchain()):
        problems.append(
            f"{where}: metadata.furina.toolchain and the course's `toolchain` in "
            f"frontend/src/data/courses must be declared together"
        )

    # A helper module beside the notebook is importable, so it is held to the
    # notebook's tier as well; otherwise it would be a way around the check.
    local = {module.stem for module in path.parent.glob("*.py")}
    if tier is not None:
        for module in sorted(path.parent.glob("*.py")):
            try:
                modules = imported_modules(module.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                problems.append(f"{module.relative_to(REPO)}: syntax error ({exc})")
                continue
            if module.stem == "furina":
                # The workbench helper reads datasets with pandas; it may only
                # sit beside tier 2 or 3 notebooks.
                modules = modules if tier < 2 else set()
            _check_imports(str(module.relative_to(REPO)), modules, tier, local, problems)

    seen_ids: set[str] = set()
    for index, cell in enumerate(notebook.get("cells", []), start=1):
        cell_where = f"{where} cell {index}"
        cell_id = cell.get("id")
        if not cell_id:
            problems.append(f"{cell_where}: missing a cell id")
        elif cell_id in seen_ids:
            problems.append(f"{cell_where}: duplicate cell id '{cell_id}'")
        seen_ids.add(str(cell_id))
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs") or cell.get("execution_count") is not None:
            problems.append(f"{cell_where}: has outputs; templates ship clean (use --strip)")
        python, escapes = _python_lines(_source(cell))
        tags = set(cell.get("metadata", {}).get("tags", []))
        for escape in escapes:
            if escape.startswith("!") or escape.split()[0] in ("%pip", "%conda"):
                problems.append(
                    f"{cell_where}: '{escape.split()[0]}' installs or shells out; the course "
                    "environments are fixed and student servers have no internet access"
                )
        try:
            modules = imported_modules(python)
        except SyntaxError as exc:
            problems.append(f"{cell_where}: syntax error ({exc.msg}, line {exc.lineno})")
            continue
        if tier is not None and not (tags & EXEMPT_TAGS):
            _check_imports(cell_where, modules, tier, local, problems)
    return notebook


def strip(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def execute(path: Path, timeout: int, problems: list[str]) -> None:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    notebook = nbformat.read(path, as_version=4)
    toolchain = notebook.metadata.get("furina", {}).get("toolchain")
    compilers = (("cc", "gcc", "clang"), ("c++", "g++", "clang++"))
    if toolchain and not all(any(shutil.which(name) for name in group) for group in compilers):
        problems.append(
            f"{path.relative_to(REPO)}: compiles '{toolchain}' code; install a C and a C++ compiler "
            "(gcc and g++, or clang) to run it"
        )
        return
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=KERNEL_NAME,
        resources={"metadata": {"path": str(path.parent)}},
    )
    try:
        client.execute()
    except CellExecutionError as exc:
        summary = str(exc).strip().splitlines()
        problems.append(f"{path.relative_to(REPO)}: failed when run ({summary[-1] if summary else exc})")


_TOOLCHAIN_COURSES: set[str] | None = None


def courses_with_toolchain() -> set[str]:
    """Ids of the courses whose definition declares a `toolchain`.

    Static and simple: a course file that mentions `toolchain:` is taken to
    define one course, whose id is the first `id: "..."` in the file.
    """
    global _TOOLCHAIN_COURSES
    if _TOOLCHAIN_COURSES is None:
        found = set()
        for data in COURSE_DATA.glob("*.ts"):
            text = data.read_text(encoding="utf-8")
            if re.search(r"^\s*toolchain:", text, re.M):
                match = re.search(r'\bid:\s*"([^"]+)"', text)
                if match:
                    found.add(match.group(1))
        _TOOLCHAIN_COURSES = found
    return _TOOLCHAIN_COURSES


def check_course_links(problems: list[str]) -> int:
    """Every notebook a lesson links to has to exist."""
    pattern = re.compile(r'path:\s*"([^"]+\.ipynb)"')
    count = 0
    for data in sorted(COURSE_DATA.glob("*.ts")):
        for match in pattern.finditer(data.read_text(encoding="utf-8")):
            count += 1
            if not (NOTEBOOKS / match.group(1)).is_file():
                problems.append(
                    f"{data.relative_to(REPO)}: links to notebooks/{match.group(1)}, which does not exist"
                )
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", type=Path, help="notebooks to check (default: all)")
    parser.add_argument("--execute", action="store_true", help="run each notebook in a clean kernel")
    parser.add_argument("--strip", action="store_true", help="clear outputs in place first")
    parser.add_argument("--timeout", type=int, default=180, help="seconds allowed per cell")
    args = parser.parse_args()

    paths = [path.resolve() for path in args.paths] or sorted(
        path for path in NOTEBOOKS.rglob("*.ipynb") if ".ipynb_checkpoints" not in path.parts
    )
    problems: list[str] = []
    for path in paths:
        if args.strip:
            strip(path)
        if check_notebook(path, problems) is not None and args.execute:
            execute(path, args.timeout, problems)
    links = check_course_links(problems)

    for problem in problems:
        print(f"FAIL {problem}")
    verb = "checked and ran" if args.execute else "checked"
    print(f"{len(paths)} notebook(s) {verb}; {links} lesson link(s) verified; {len(problems)} problem(s).")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

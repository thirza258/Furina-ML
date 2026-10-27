"""Run a grading notebook top to bottom in a fresh kernel and save the result.

This file runs inside the sandbox container, from the student image, so it
uses only what that image provides (nbclient, nbformat) and nothing else from
this package. The worker copies it in next to the notebook.

    python execute.py grading.ipynb executed.ipynb [seconds per cell]

A cell that fails or runs out of time does not stop the run: the next cells
still run, so one broken exercise does not cost the others. If the kernel
dies, the cells that did run are still saved.
"""

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> int:
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    cell_timeout = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    notebook = nbformat.read(source, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=cell_timeout,
        kernel_name="python",
        allow_errors=True,
        interrupt_on_timeout=True,
        resources={"metadata": {"path": str(source.parent)}},
    )
    try:
        client.execute()
    except Exception as error:  # the kernel died; keep what ran
        print(f"execution stopped: {type(error).__name__}: {error}", file=sys.stderr)
    nbformat.write(notebook, target)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Course notebooks

Every lesson notebook in Furina ML lives here, one folder per course
(`math-foundations/`, …). The same files are used in three places:

| Where | How it gets them | Who runs the code |
|-------|------------------|-------------------|
| **JupyterHub** (signed-in students, when the site runs the `hub` profile) | Baked into the student image at `/opt/furina/notebooks`; each student's server copies the files they do not have yet into their own workspace on start | The student's own isolated container |
| **JupyterLite** preview at `/notebooks/` | Built into the frontend image by `jupyter lite build` | The visitor's browser (Pyodide) |
| **Download** from a lesson | Served from the JupyterLite build at `/notebooks/files/<path>` | Wherever the learner opens it |

A lesson points at its notebook with `notebook: { path, title }` in
`frontend/src/data/courses/*.ts`; the path is relative to this folder.

## Rules every notebook follows

`tools/check_notebooks.py` enforces the first four.

1. **Ships clean.** No outputs, no execution counts. Run the checker with
   `--strip` to clear them.
2. **Declares the `python` kernel.** That is the name of JupyterLite's Pyodide
   kernel. On the JupyterHub servers (and in Colab) jupyter_client treats
   `python` as an alias of the standard `python3` kernel, so one template opens
   without a "select kernel" prompt everywhere.
3. **Declares its course and dependency tier** in the notebook metadata:

   ```json
   "furina": {"course": "math-foundations", "tier": 0}
   ```

   | Tier | Core cells may import |
   |------|-----------------------|
   | 0 | The Python standard library only |
   | 1 | Tier 0 plus NumPy and Matplotlib |
   | 2 | Tier 1 plus pandas, SciPy and scikit-learn |
   | 3 | Tier 2 plus PyTorch |

   A cell tagged `comparison`, `visualization` or `data` may import a
   higher-tier library: to check a from-scratch result, draw a plot, or load
   a dataset. Nothing may depend on a `comparison` or `visualization` cell.
   The check is static and keeps the content honest; it is not a security
   boundary.

   A notebook that compiles native code also declares its toolchain, `"c"`,
   `"c++"` or `"cuda"`, and its course declares a `toolchain` string in its
   `frontend/src/data/courses` file. The checker requires both or neither,
   because the site never opens such a notebook in JupyterLite, which cannot
   run a compiler:

   ```json
   "furina": {"course": "cuda-base-models", "tier": 0, "toolchain": "cuda"}
   ```
4. **Runs top to bottom in a clean kernel** in well under a minute, with no
   network access, no `%pip` and no shell escapes (student servers have no
   internet access). Native-code notebooks run the compiler and their check
   programs through `subprocess`, never through `!`.
5. **Teaches in a fixed order:** learning objectives, dependencies, the
   maths, step-by-step implementation with tests and expected behaviour,
   exercises, common errors, summary. Exercises are stubs that raise
   `NotImplementedError` and a `check(...)` cell that reports "not implemented
   yet" until the learner fills them in. Reference solutions sit in a final
   section under different names and pass the same checks.

## Checking

From the repository root, in an environment with the student image's
packages (`jupyter/singleuser/requirements.txt`) plus `nbclient`:

```sh
python notebooks/tools/check_notebooks.py --execute
```

Without `--execute` it checks structure, metadata, tiers and that every
notebook a lesson links to exists. Notebooks with a toolchain also need a C
and a C++ compiler (gcc and g++, or clang) to run.

## Native code: C, C++ and CUDA

The notebooks in `cuda-base-models/` keep Python at tier 0 and put the model
in C, C++ and CUDA. Each notebook is self-contained, so it works when it is
downloaded or opened in Colab on its own:

- An exercise is a Python function that returns source code. Its stub starts
  with `raise NotImplementedError`, which the learner deletes, so the grader
  treats it like any other exercise cell. Keep the text `_reference` out of
  exercise cells.
- `practice_native(name, yours, reference, program, language=...)` wraps each
  source in a complete check program (`program(source)` adds a `main` that
  prints labelled numbers), compiles it, runs it in its own process and
  compares the printouts within a tolerance. A crash or an endless loop in
  the learner's code fails the check instead of the kernel. Code that has
  passed its check is loaded with `ctypes` for the training runs.
- `mark(source, label)` adds a `#line` directive, so compiler messages name
  the learner's function instead of a line of the generated program.
- Pass every buffer to C from a variable: `address(array(...))` on a
  temporary hands C memory that Python may already have freed.
- CUDA notebooks pick a backend in their second cell: `nvcc` and an NVIDIA
  GPU when both are present (Colab's GPU runtime, a local GPU), otherwise a
  CPU emulator, `furina_cuda_cpu.h`, embedded in the cell. The notebook
  rewrites `kernel<<<grid, block>>>(args)` for the emulator, which runs each
  block's threads one after another, or as one operating-system thread per
  CUDA thread when the kernel calls `__syncthreads()`. Keep such blocks small
  (8 x 8): student servers and the grader limit a container to 256 processes
  and threads. Set `FURINA_CUDA_BACKEND=emulator` or `gpu` to force a backend.
- The emulator checks results, not speed, and cannot reveal every race.
  Cells that time kernels label emulator numbers as CPU timings.

## Grading

Handed-in notebooks are graded against the template (see
[docs/academy.md](../docs/academy.md#automated-grading)). Only exercise cells
-- code cells that raise `NotImplementedError` in the template -- come from the
student. A check is graded when a `practice*(...)` or `check(...)` call in
another cell passes the learner's function (or a wrapper of it) right after
the check's name. Keep everything an exercise needs inside its own cell, and
after changing a notebook, check that grading still works:

```sh
cd jupyter/grader && python tests/check_course_notebooks.py --only <course-folder>
```

## The workbench helper

`workbench/furina.py` reads datasets and runs from the Furina workbench API.
It works in JupyterLite, where the API is on the page's own origin. Student
JupyterHub servers have no network route to the API by design, so notebooks
that run there cannot use it.

## Updating a notebook students already have

A student's copy is never overwritten. New students, and students who delete
their copy, get the current template the next time their server starts.

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
4. **Runs top to bottom in a clean kernel** in well under a minute, with no
   network access, no `%pip` and no shell escapes (student servers have no
   internet access).
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
notebook a lesson links to exists.

## The workbench helper

`workbench/furina.py` reads datasets and runs from the Furina workbench API.
It works in JupyterLite, where the API is on the page's own origin. Student
JupyterHub servers have no network route to the API by design, so notebooks
that run there cannot use it.

## Updating a notebook students already have

A student's copy is never overwritten. New students, and students who delete
their copy, get the current template the next time their server starts.

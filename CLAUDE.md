# Furina ML

Educational ML Platform (Django REST Framework + React TypeScript Vite + Docker + PostgreSQL)

## Essential Commands
- **Full stack**: `docker compose up -d --build` (runs at http://localhost:8080)
- **Backend dev**: `docker compose up -d db` && `python backend/manage.py runserver`
- **Backend tests**: `python backend/manage.py test [app.tests]` (e.g. `academy.tests`, `core.tests`)
- **Backend migrations**: `python backend/manage.py makemigrations && python backend/manage.py migrate`
- **Frontend dev**: `npm run dev --prefix frontend` (Vite dev server)
- **Frontend build & lint**: `npm run build --prefix frontend` / `npm run lint --prefix frontend`

## Project Structure
- `backend/`: Django 5.2 backend, DRF, PostgreSQL, auth, APIs.
  - `academy/`: Courses, lessons, quizzes, progress tracking, JupyterHub, notebook grading.
  - `core/`: The workbench — dataset upload, training, evaluation, prediction, health,
    OAuth2. Its ML engine (`core/ml/`) is Django-free and reusable.
  - `lifecycle/`: The ML lifecycle platform — dataset versions, experiments, runs,
    jobs, model registry, deployments, monitoring. See `docs/lifecycle.md`.
  - `furinaml_settings/`: Django settings, ASGI/WSGI configuration.
- `frontend/`: React 18, TypeScript, Tailwind CSS, Vite frontend.
  - `src/interface/`: API shape mirrors (`MlTypes.ts`, `LifecycleTypes.ts`) — the single
    source of truth for API types. Never declare a local copy of an API shape.
  - `src/services/services.ts`: the only HTTP layer; pages never build requests themselves.
- `notebooks/` & `jupyter/`: Jupyter interactive materials and Hub configs.

## Architectural Rule
**Django orchestrates ML; it does not execute it.** Views validate and enqueue a
`lifecycle.Job`; `lifecycle/worker.py` runs it via `lifecycle.services`. Fitting logic
lives only in `core/ml/`. Adding an endpoint must not add a fit to a request handler.

`/api/lifecycle/` is additive. The legacy `/api/datasets/`, `/api/train/`, `/api/runs/`
response shapes are frozen contracts asserted by `core/tests.py` and consumed by the
shipping frontend — extend, never mutate.

## Efficiency Rules
- **Targeted scope**: Inspect and modify only relevant app files.
- **Targeted tests**: Run specific test cases rather than the entire test suite
  (`manage.py test lifecycle`, `core.tests.TrainingTests`, …).
- **Minimal output**: Use `git diff --stat` and targeted excerpts; return concise summaries.
- **Verify**: `manage.py test lifecycle core academy` and `npm run build --prefix frontend`
  must both pass before reporting work done.

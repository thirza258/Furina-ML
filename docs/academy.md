# Accounts, notebooks and the per-student Jupyter server

This document describes the interactive-learning platform: accounts, saved
course progress, lesson notebooks, an isolated JupyterHub server per student,
and automated grading of handed-in notebooks. It records the architecture, the
security model and its assumptions, how to deploy and operate it, and what is
not built yet.

The product requirements were written for a Go backend. Furina ML's backend is
Django, so the same responsibilities are implemented there: Django handles
accounts, roles, progress and notebook authorization, and JupyterHub manages
the notebook servers. The course catalogue itself is still the static data in
`frontend/src/data/courses/`.

## What a student can do

1. Create an account or sign in (`/register`, `/login`). The workbench and all
   lesson text stay open to anonymous visitors.
2. Pass knowledge checks. Signed out, progress is kept in the browser; signed
   in, it is saved to the account. Progress made while signed out is uploaded
   once at sign-in and then removed from the browser, so it cannot be credited
   to the next person who signs in on that browser.
3. Learn in notebooks, Colab-style. In notebook courses the notebook *is* the
   lesson: it opens at the top of the page, in Jupyter's single-document view,
   and the written lesson follows as notes. The learner picks where the code
   runs once; the choice is remembered, so every later lesson opens straight
   away:
   - **This browser**: JupyterLite, with no account. Python (Pyodide) runs on
     the learner's own device and edits stay in that browser.
   - **My saved copy**: when the site runs the hub and the learner is signed
     in. The notebook comes from the learner's own server and workspace, and
     stays there between visits.
   - **Open in Colab** (when `NOTEBOOKS_GITHUB` is set), **Download .ipynb**,
     **New tab**, **Full screen**, and **JupyterLab** for the full IDE.
4. Build the models themselves. Each building block follows the same
   pattern: a short explanation, a tiny demo, then a **✍️ Your turn** cell
   where the learner types the function, and a `practice(...)` check. The check
   compares the learner's function with a reference on test inputs and, once
   it matches, the rest of the notebook runs on the learner's own code. Until
   then it falls back to the reference, so "Run all" always works. The
   references sit in collapsed cells, to open only when stuck.
5. Hand a finished notebook in for grading (signed in, when the site runs the
   grader). The learner downloads their copy from wherever they worked on it
   (this browser, their server, Colab) and uploads it under the notebook. The
   page shows every check as passed, not yet, or did not run, with what the
   check printed. See [Automated grading](#automated-grading).

## Architecture

```
browser ──► nginx (frontend container) ──┬─► SPA, JupyterLite site (/notebooks/)
  │                                       ├─► Django (/api/, /admin/)      ──► PostgreSQL
  │                                       └─► JupyterHub proxy (hub host and *.hub host)
  │                                                │
  │                                                ├─► Hub ──► docker-proxy ──► Docker
  │                                                │     └─► Django (OAuth token + user info, internal network)
  └────────────────────────────────────────────────┴─► u<id>: one container per student
                                                          (network "furina-students": hub only)

grader worker ──► Django (/api/grader/, internal network, shared token)
      └─────────► docker-proxy ──► one throwaway sandbox per submission (no network)
```

| Component | Responsibility |
|---|---|
| Django `academy` app | Accounts, roles (student, instructor, admin), saved progress, notebook launch authorization and launch records, OAuth2 provider for the hub |
| JupyterHub 6.1.1 | Signs students in through Django, starts and stops their servers, routes their traffic, culls idle servers |
| DockerSpawner | One container per student, from the `furina-notebook` image, with a named volume for their files |
| `docker-proxy` | The hub's only access to Docker: container and image calls, nothing else |
| Student image | Python 3.13, the pinned course environment, and the notebook templates |
| JupyterLite | The no-account preview, built into the frontend image |
| Grader worker (`jupyter/grader`) | Claims submitted notebooks from Django, runs each in a sandbox container of the student image, posts per-check results back |

### Sign-in to the hub

Django is an OAuth2 provider (django-oauth-toolkit, authorization code with
PKCE). The hub's client is registered at every backend start by
`manage.py ensure_jupyterhub_client` from `JUPYTERHUB_OAUTH_CLIENT_ID` and
`JUPYTERHUB_OAUTH_CLIENT_SECRET`, and is marked first-party, so a signed-in
student never sees a consent page.

1. The lesson page calls `POST /api/notebooks/launch/`. Django checks the
   session and the path, records the launch, and returns
   `<hub>/hub/user-redirect/lab/tree/<path>`.
2. The hub sends the browser to `<site>/api/oauth/authorize/`. A student who is
   signed in to Furina is sent straight back with a code; one who is not is
   sent to `/login?next=…` first.
3. The hub exchanges the code at `http://backend:8000/api/oauth/token/` and
   reads `http://backend:8000/api/auth/hub-user/` on the internal network.
   The hub username is `u<user id>`: stable, a valid DNS label, and free of
   personal data.
4. The hub starts the student's container if it is stopped and opens the file.

The hub keeps the access token (encrypted with `JUPYTERHUB_CRYPT_KEY`) and
re-checks it every `JUPYTERHUB_AUTH_REFRESH_SECONDS` (300) and before every
start. Signing out of Furina deletes the student's OAuth tokens and then sends
the browser through the hub's logout. That logout deletes every token issued
to the browser session, including those of the student's own server, stops
the server, and redirects back to the site.

### Notebook templates and student copies

Templates live in `notebooks/` (see `notebooks/README.md`). The student image
contains them at `/opt/furina/notebooks`. Each time a student's server starts,
it copies the templates the student does not have into `~/work`. It never
overwrites a file the student already has, so a template update never
silently changes someone's work. Deleting a file restores the current
template on the next start.

Notebook files persist in the volume `furina-nb-u<id>`. Kernel memory does
not: an idle server is stopped after `NOTEBOOK_IDLE_SECONDS` (default 1800),
and idle kernels are culled after 30 minutes.

## Security model

Running student code is running untrusted code. The controls below were
verified on a running stack. Verification method: `docker inspect`, cgroup
files and connection attempts from inside a student container.

| Control | How | Verified |
|---|---|---|
| One server per student | DockerSpawner, `furina-nb-u<id>` | Yes |
| Non-root | User `jovyan` (uid 1000) | Yes |
| No privileges | All capabilities dropped, `no-new-privileges`, not privileged | Yes |
| CPU, memory, processes | 1 CPU, 1 GiB and 256 processes by default (`NOTEBOOK_*`) | Yes, in the container's cgroup |
| No route to internal services | Students are only on `furina-students`, an internal network shared with the hub alone | Yes: backend, db, frontend, docker-proxy unreachable |
| No internet, host or metadata access | The internal network has no gateway | Yes: 1.1.1.1, pypi.org, 169.254.169.254 and the host unreachable |
| No Docker socket in student containers | Only `docker-proxy` mounts it | Yes |
| Cross-student isolation | Hub authorization per server, and a subdomain per student (`subdomain_host`) | Yes: a second student opening the first one's URL gets "No access" |
| Idle shutdown | `jupyterhub-idle-culler` service | Configured; the service runs |
| Framing | Hub and student pages may only be framed by the site (`frame-ancestors`); the embed is sandboxed so it cannot navigate the page | Yes, the header is present |

Assumptions and limits you should know before exposing this:

- **The hub is privileged.** `docker-proxy` blocks exec, builds and swarm calls,
  but it must allow container creation. Anyone who controls the hub can start
  a container that mounts the host. Keep the hub patched and its secrets
  private.
- **Containers are not a complete security boundary** for a large public
  service that runs arbitrary code. For that scale, use gVisor or Kata
  containers, or run student servers in separate virtual machines.
- **Per-student subdomains are required.** They need a wildcard DNS record and,
  behind TLS, a wildcard certificate for `*.<hub host>`. Without them, one
  student's server could serve pages that act as another student.
- **No per-student disk quota.** Docker's local volumes cannot enforce one.
  Monitor volume sizes, or use a storage driver with quotas.
- **JupyterLite runs on the site's own origin.** Code that a visitor runs in the
  preview can call the Furina API as that visitor, with the visitor's session.
  Students only run notebooks they open themselves. An administrator should
  still not run untrusted notebooks in the preview while signed in. Moving the
  preview to its own origin is follow-up work.
- **The JupyterLite preview loads Pyodide from the jsDelivr CDN.** Offline
  deployments get the hub, but not the preview.
- **Sign-in throttling** counts attempts per client address taken from
  `X-Forwarded-For`. Set `NUM_PROXIES` to the number of proxies in front of
  Django.

## Automated grading

A signed-in student uploads their `.ipynb` for a lesson
(`POST /api/submissions/`). Django keeps only the code cells' ids and source,
never outputs, and queues it. The grader worker claims it over Django's
internal grader API and grades it:

1. **Rebuild from the template.** The grading notebook is the lesson's own
   template from the grader image. Only the **✍️ Your turn** cells (code cells
   that raise `NotImplementedError` in the template) are taken from the
   submission. They are matched by cell id, or by the functions they define
   when an editor dropped the ids (Colab). Setup, data, hidden references and
   every `practice(...)`/`check(...)` call come from the template. Deleting or
   editing a check in the submitted file therefore changes nothing.
2. **Guard each check.** Before every check cell that tests an exercise, the
   trusted checkers (`practice*`, `check`) and `*_reference` functions are
   re-defined from the template and wrapped to record each call's outcome. A
   student cell that redefines `practice` to always pass is undone before it
   matters, and calls made from any other cell are not recorded. Before each
   exercise cell, its names are bound to stand-ins that act like the
   unfinished exercise. A cell that fails (syntax error, time-out) then costs
   only its own checks, never a scaffold of the same name defined earlier.
3. **Run in a sandbox.** The notebook runs top to bottom in a fresh kernel,
   in a new container of the student image. One failing or timed-out cell
   does not stop the run.
4. **Score.** Each named check passes when every call it made passed. The
   score is passed / total. A check that never ran (an earlier error, a dead
   kernel) counts as failed and is marked "did not run". When an exercise cell
   itself failed, its error is shown with the check's feedback.

All 46 course notebooks are graded under seven kinds of submission by
`jupyter/grader/tests/check_course_notebooks.py`: unfinished, fully answered
(299/299 checks), answered without cell ids, two kinds of tampering, a syntax
error and an endless loop. Every run gives the expected result.

**What it does not stop.** The anti-tampering scope is the same as other
notebook autograders: it resists accidental and casual tampering, not a
determined cheat. Code in a student's function runs in the same process as
the checks, so it can monkeypatch libraries (`numpy.allclose`), reach the
recorder, or forge the report. A student can also call the reference
solution, which sits in the notebook by design. Instructors review
submissions in the Django admin (Academy → Submissions), which shows each
student's code cells and per-check results. They can also queue submissions
to be graded again.

| Sandbox control | How |
|---|---|
| A new container per submission, removed afterwards | Labelled `furina.grader=sandbox`; leftovers are removed when the worker starts |
| No network at all | `network_mode: none` |
| Non-root, no privileges | uid 1000, all capabilities dropped, `no-new-privileges` |
| CPU, memory, processes | `GRADER_CPU_LIMIT`, `GRADER_MEM_LIMIT` (no swap), 256 processes |
| Time | `GRADER_CELL_TIMEOUT` per cell (the cell is interrupted, the run goes on) and `GRADER_TIMEOUT` for the whole notebook (the container is killed) |
| Worker's reach | Docker only through `docker-proxy`; Django only through `/api/grader/`, which needs `GRADER_TOKEN` and which nginx answers with 404 from outside |
| Queue limits | One submission per student and notebook at a time, `SUBMISSION_RATE_LIMIT` per account, `MAX_SUBMISSION_BYTES` per upload |

The container options are checked by the worker tests. The worker's full path
(bundle → run → read back → score → report to Django) was run against a live
Django with a stand-in for Docker, including through the browser. **Not yet
verified on a running Docker stack:** that the socket proxy passes the archive
calls the worker uses (`PUT`/`GET /containers/{id}/archive`), and the
sandbox's network and limit settings as `docker inspect` shows them. Do both
before exposing grading.

## Deploying

Everything is opt-in through `.env`. The hub needs its own host name, and
every student gets a subdomain of it:

```sh
COMPOSE_PROFILES=hub
SITE_URL=https://furina.example.com
JUPYTERHUB_URL=https://notebooks.furina.example.com      # *.notebooks.furina.example.com → this server
JUPYTERHUB_OAUTH_CLIENT_ID=furina-jupyterhub
JUPYTERHUB_OAUTH_CLIENT_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
JUPYTERHUB_CRYPT_KEY=$(openssl rand -hex 32)
```

Keep the hub a subdomain of the site's domain. The embedded notebook relies on
cookies that browsers only send within the same site. Then:

```sh
docker compose up -d --build
```

The `hub` profile adds three services: `jupyterhub`, `docker-proxy`, and
`notebook-image`. The last one only builds the student image and exits
straight away. nginx in the frontend container answers for the hub's host and
every subdomain under it once `JUPYTERHUB_URL` is set.

Grading is a separate profile, `grader`, and works with or without the hub.
It adds the `grader` worker (and `docker-proxy` and `notebook-image` if the hub
profile is off). Set one shared secret for Django and the worker:

```sh
COMPOSE_PROFILES=hub,grader
GRADER_TOKEN=$(openssl rand -hex 32)
```

To try it on one machine, use the `*.localhost` names. Browsers resolve them
to the local machine, and they are a secure context, so JupyterLite's service
worker works:

```sh
SITE_URL=http://furina.localhost:8080
JUPYTERHUB_URL=http://notebooks.furina.localhost:8080
```

### Environment reference

| Variable | Default | Purpose |
|---|---|---|
| `ACCOUNT_REGISTRATION` | `True` | Allow sign-up; `False` means admin-created accounts only |
| `AUTH_RATE_LIMIT` | `10/minute` | Sign-in and sign-up attempts per client address |
| `NUM_PROXIES` | `1` | Proxies in front of Django that append to `X-Forwarded-For` |
| `JUPYTERHUB_URL` | empty | The hub's public address; empty disables the hub everywhere |
| `JUPYTERHUB_OAUTH_CLIENT_ID` / `_SECRET` | empty | The hub's OAuth client |
| `JUPYTERHUB_CRYPT_KEY` | empty | Encrypts stored tokens (required with the hub) |
| `JUPYTERHUB_AUTH_REFRESH_SECONDS` | `300` | How often the hub re-checks a student's token |
| `NOTEBOOK_MEM_LIMIT` / `_CPU_LIMIT` / `_PIDS_LIMIT` | `1G` / `1` / `256` | Per-student limits |
| `NOTEBOOK_IDLE_SECONDS` | `1800` | Idle server shutdown |
| `NOTEBOOKS_GITHUB` | empty | `owner/repo/blob/branch/notebooks`: shows "Open in Colab" (public repositories only) |
| `GRADER_TOKEN` | empty | Shared secret between Django and the grader worker; empty turns grading off. Set it only where the worker runs: without one, submissions stay queued, and a student cannot hand the same notebook in again until it is graded |
| `SUBMISSION_RATE_LIMIT` | `30/hour` | Notebooks handed in per student account |
| `MAX_SUBMISSION_BYTES` | 8 MiB | Largest upload (outputs are discarded, so this is generous) |
| `GRADING_STALE_SECONDS` | `900` | A submission still running after this long is queued again (three attempts) |
| `GRADER_CELL_TIMEOUT` / `GRADER_TIMEOUT` | `60` / `300` | Seconds per cell / per notebook in the sandbox |
| `GRADER_MEM_LIMIT` / `GRADER_CPU_LIMIT` | `1g` / `1` | Sandbox limits |

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/auth/session/` | Who is signed in, whether the hub and grading are enabled; sets the CSRF cookie |
| `POST` | `/api/auth/register/` | Create a student account and sign in (CSRF checked) |
| `POST` | `/api/auth/login/` | Sign in (CSRF checked, rate limited) |
| `POST` | `/api/auth/logout/` | Sign out and revoke the hub's tokens |
| `GET` | `/api/progress/` | Passed knowledge checks (signed in) |
| `POST` | `/api/progress/` | Mark `course/lesson` keys complete; idempotent |
| `POST` | `/api/notebooks/launch/` | Authorize and record a notebook launch; returns the hub URL |
| `GET` | `/api/submissions/?path=` | The signed-in student's latest submissions (of one notebook) |
| `POST` | `/api/submissions/` | Hand in a notebook: `path` and `notebook` (multipart file or JSON); rate limited |
| `GET` | `/api/submissions/{id}/` | One of the student's submissions, with per-check results |
| `POST` | `/api/grader/claim/` | Grader worker only (`Authorization: Bearer <GRADER_TOKEN>`): take the oldest queued submission |
| `POST` | `/api/grader/submissions/{id}/result/` | Grader worker only: report a result |
| `GET` | `/api/auth/hub-user/` | OAuth user info for the hub (bearer token, `profile` scope) |
| `GET` | `/api/oauth/authorize/`, `POST` `/api/oauth/token/` | OAuth endpoints used by the hub |

Once a session is signed in, every unsafe request needs the `X-CSRFToken`
header. The SPA's API client sends it.

## Python environments

| Where | Python | Libraries |
|---|---|---|
| Student server | 3.13 | `jupyter/singleuser/requirements.txt`: JupyterLab 4.6.4, NumPy 2.3.4, pandas 3.0.5, SciPy 1.18.1, scikit-learn 1.9.0, Matplotlib 3.11.2 |
| JupyterLite preview | 3.14 (Pyodide 314.0.6) | Pyodide's own builds, loaded on import; at the time of writing NumPy 2.4, pandas 3.0, scikit-learn 1.8, Matplotlib 3.10 |

Notebooks declare a dependency tier (0 = standard library only, up to 3 =
PyTorch). The checker enforces each notebook's tier on its imports. Because
the preview's library versions differ from the server's, notebooks compare
results within tolerances and never against exact library output.

## Authoring notebooks

See `notebooks/README.md`. In short: put the notebook in its course folder,
give it the `python` kernelspec and `metadata.furina`, link it from the lesson
with `notebook: { path, title }`, and run:

```sh
python notebooks/tools/check_notebooks.py --execute
```

Rebuild the student image (`docker compose build notebook-image`), the
frontend and the grader (`docker compose build grader`) to publish it. The
grader grades against its own copy of the templates, so it must be rebuilt
whenever a notebook changes.

For grading, a check counts when a `practice*(...)` or `check(...)` call in a
non-exercise cell names the exercise: its first string argument is the check's
name, and the argument after it is the learner's function (or a wrapper whose
check name is the exercise's name). After changing a notebook or the grader,
run the grader against every course notebook (slow; it runs each notebook
seven times):

```sh
cd jupyter/grader && python tests/check_course_notebooks.py
python -m unittest discover -s tests    # the grader's own tests
```

## Backup and restore

| Volume | Holds |
|---|---|
| `db_data` | Accounts, roles, saved progress, launch records, OAuth clients (plus datasets and runs) |
| `media_data` | Uploaded CSVs and trained models |
| `hub_data` | The hub's database and cookie secret (rebuildable; losing it signs everyone out of the hub) |
| `furina-nb-u<id>` | Each student's notebook files |

```sh
# Database
docker compose exec db pg_dump -U furinaml furinaml > furina.sql
# One student's files (repeat per volume, or loop over `docker volume ls`)
docker run --rm -v furina-nb-u12:/data -v "$PWD":/backup alpine \
    tar czf /backup/furina-nb-u12.tgz -C /data .
# Restore a student's files into a fresh volume
docker run --rm -v furina-nb-u12:/data -v "$PWD":/backup alpine \
    tar xzf /backup/furina-nb-u12.tgz -C /data
```

## Course content

Nine courses are notebook courses, with 46 notebooks in `notebooks/` (one
folder per course): Mathematics for ML from First Principles, Neural Networks
from Scratch, Build Your Own Autograd, and the six deep-learning courses (Base
Model, Intermediate, Transformers, LLMs, Vision-Language, Voice), which were
rewritten so learners build each model in NumPy. Every notebook runs top to
bottom in a clean kernel, with its exercise stubs unfilled, on the student
image (Python 3.13) and in CPython 3.14, where `check_notebooks.py --execute`
runs them; each was also run with every stub filled from its reference.
The deep-learning courses share one small reverse-mode tensor engine,
which learners write in Base Model Training and which later courses include
in a collapsed recap cell.

Two things to know:

- In Google Colab the collapsed reference cells are shown expanded: Colab does
  not read Jupyter's `source_hidden` metadata.
- Runtimes were measured in CPython. In the browser preview (Pyodide) the
  training notebooks take several times longer.

## Not built yet

This slice delivers the PRD's core loop. Still to come, roughly in the PRD's
order:

- Database-backed courses, modules and versioned notebook templates, with an
  instructor course builder (courses are still static frontend data).
- Enrollment and course access rules. Today every account may open every
  notebook.
- Assignments on top of grading: due dates, required notebooks per course,
  and an instructor gradebook in the app (today instructors review
  submissions in the Django admin, which needs staff access).
- Handing in straight from the student's hub workspace, without the
  download-and-upload step.
- Analytics views over saved progress and launch records, and a student
  dashboard.
- The rest of the from-scratch path in the PRD addendum (numerical computing
  and classical ML algorithms from scratch, with dependency-tier comparisons),
  PyTorch and GPU environments, and the marketing-focused courses.
- Moving the JupyterLite preview to its own origin, and per-student disk
  quotas.

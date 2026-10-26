# Accounts, notebooks and the per-student Jupyter server

This document describes the first slice of the interactive-learning platform
(the "core loop"): accounts, saved course progress, lesson notebooks, and an
isolated JupyterHub server per student. It records the architecture, the
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
```

| Component | Responsibility |
|---|---|
| Django `academy` app | Accounts, roles (student, instructor, admin), saved progress, notebook launch authorization and launch records, OAuth2 provider for the hub |
| JupyterHub 6.1.1 | Signs students in through Django, starts and stops their servers, routes their traffic, culls idle servers |
| DockerSpawner | One container per student, from the `furina-notebook` image, with a named volume for their files |
| `docker-proxy` | The hub's only access to Docker: container and image calls, nothing else |
| Student image | Python 3.13, the pinned course environment, and the notebook templates |
| JupyterLite | The no-account preview, built into the frontend image |

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

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/auth/session/` | Who is signed in, whether the hub is enabled; sets the CSRF cookie |
| `POST` | `/api/auth/register/` | Create a student account and sign in (CSRF checked) |
| `POST` | `/api/auth/login/` | Sign in (CSRF checked, rate limited) |
| `POST` | `/api/auth/logout/` | Sign out and revoke the hub's tokens |
| `GET` | `/api/progress/` | Passed knowledge checks (signed in) |
| `POST` | `/api/progress/` | Mark `course/lesson` keys complete; idempotent |
| `POST` | `/api/notebooks/launch/` | Authorize and record a notebook launch; returns the hub URL |
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

Rebuild the student image (`docker compose build notebook-image`) and the
frontend to publish it.

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

## Not built yet

This slice delivers the PRD's core loop. Still to come, roughly in the PRD's
order:

- Database-backed courses, modules and versioned notebook templates, with an
  instructor course builder (courses are still static frontend data).
- Enrollment and course access rules. Today every account may open every
  notebook.
- Assignments, submissions and an isolated grading worker. The notebooks'
  self-checks already test behaviour, not source text, so they can become
  graders.
- Analytics views over saved progress and launch records, and a student
  dashboard.
- PyTorch and GPU environments, more of the from-scratch learning path
  (numerical computing, ML algorithms, neural networks, autodiff), and the
  marketing-focused courses.
- Moving the JupyterLite preview to its own origin, and per-student disk
  quotas.

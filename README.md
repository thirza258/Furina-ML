# Furina ML

A web application for training real scikit-learn models on your own CSV files.
Upload a dataset, see what is actually in it, clean it, pick a model, tune its
hyperparameters, train it, read the evaluation, and make live predictions —
without writing any code.

Workbench evaluation uses real training results: every number on its evaluation
page comes from a model that was fitted on the data you uploaded.

## What it does

### 1. Data Preprocessing
- Upload a CSV (auto-detected delimiter, UTF-8 / UTF-8-BOM / Latin-1 encodings)
- A real column profile: inferred kind, dtype, missing counts, unique counts,
  min/mean/max, most common values, and whether a column is usable as a target
- A preview of the first rows with missing values shown as missing
- A cleaning recipe you compose step by step — deduplicate, fill missing values,
  strip/lowercase text, drop outliers, scale, encode, PCA / LDA / SelectKBest —
  with a dry-run preview before anything is applied
- Cleaned data is saved as a new derived dataset, so the original is never lost

### 2. Model Training
- Choose the target column and the feature columns from your real data
- Task type is inferred (classification vs regression) and can be overridden —
  the override matters for integer-coded targets like `0/1/2`, which look
  numeric but are really classes
- 28 algorithms across both tasks — 26 from scikit-learn plus XGBoost and
  LightGBM — each with a hyperparameter form generated from the algorithm's
  own parameter schema:

  | Family | Algorithms |
  |--------|------------|
  | Baselines | Dummy Baseline |
  | Linear models | Logistic Regression, Ridge Classifier, Linear / Ridge / Lasso Regression, Elastic Net, Huber Regression, Bayesian Ridge Regression, SGD |
  | Discriminant analysis | Linear and Quadratic Discriminant Analysis |
  | Support vector machines | SVC / SVR, Linear Support Vector Machine |
  | Trees and ensembles | Decision Tree, Random Forest, Extra Trees, Bagging, Gradient Boosting, Histogram Gradient Boosting, AdaBoost, XGBoost, LightGBM |
  | Nearest neighbours | K-Nearest Neighbours, Nearest Centroid |
  | Naive Bayes | Gaussian and Bernoulli Naive Bayes |
  | Neural networks | Multilayer perceptron (MLP) |

- Adjustable train/test split, random seed, stratification, feature scaling,
  and optional k-fold cross-validation
- Untouched hyperparameters are left to scikit-learn's own defaults, apart
  from a few deliberate overrides that the form shows as its defaults — for
  example, SVC estimates probabilities, and Quadratic Discriminant Analysis
  uses Ledoit-Wolf shrinkage because one-hot columns make its plain per-class
  covariance singular

### 3. Model Evaluation
- Classification: accuracy, balanced accuracy, weighted precision / recall /
  F1 / F2, ROC AUC, log loss, Matthews correlation, a full N×N confusion matrix,
  and per-class precision / recall / F1 / support
- Regression: R², MAE, MSE, RMSE, MAPE, explained variance, max error, and an
  actual-vs-predicted plot
- Train-set metrics alongside test-set metrics, so overfitting is visible
- Feature importances (or coefficient magnitudes) over the expanded feature names
- "Try it yourself" — a prediction form built from the model's own input schema,
  with class probabilities
- Download the fitted model as a `.joblib` file, or the evaluation as Markdown

### 4. Dashboard
- Every run this instance has trained, ranked by whichever metric you pick
- Ranking honours the metric's own direction, so for log loss or MAE the
  shortest bar leads
- Train-split scores beside test-split scores, so the overfitting gap is
  visible across runs rather than one run at a time
- Metrics are only compared within one task type: accuracy and R² are not on
  the same scale, so the chart says so instead of drawing them together

### 5. Courses
- Sixteen courses with **122 lessons**, from beginner to advanced. Ten are
  **notebook courses** (50 Jupyter notebooks): learners build each model
  themselves, typing every building block into a "✍️ Your turn" cell that is
  checked against a reference, after which the rest of the notebook runs on
  their own code:

  | Course | Lessons | Practical work |
  |--------|---------|----------------|
  | Mathematics for ML from First Principles | 5 | Pure-Python notebooks: floats, linear algebra, statistics, calculus, gradient descent |
  | Model Training | 4 | Target selection, splitting and training in the workbench |
  | Model Evaluation | 4 | Metrics, overfitting and run comparisons |
  | Data Preparation | 5 | Reading the column profile, safe cleaning, and preprocessing without leaking the test split |
  | Choosing a Model | 8 | Baselines, every estimator family in the workbench, and a fair, documented model comparison |
  | Time-Series Forecasting | 8 | Chronological backtesting, lag features, multi-step forecasts and calibrated prediction intervals (Python lab) |
  | Neural Networks from Scratch | 12 | NumPy notebooks: neurons, layers, losses, backpropagation, training, real data, regularisation, a framework comparison and a final project |
  | Build Your Own Autograd | 4 | Pure-Python notebooks: computational graphs, a scalar autodiff engine, reverse mode, and a network trained with it |
  | Base Model Training | 8 | NumPy notebooks: a tensor autograd engine, data pipelines, Adam, diagnostics, exact-resume checkpoints and evaluation |
  | Base Model Training in C, C++ and CUDA | 8 | No NumPy: backpropagation by hand in C, an autograd engine with RAII in C++, CUDA kernels with shared-memory tiles, and a training loop, checkpoints and capstone that stay on the GPU |
  | Intermediate Model Training | 8 | NumPy notebooks: imbalance and thresholds, schedules, clipping and accumulation, transfer, mixed precision, search and data parallelism |
  | Transformer Models | 8 | NumPy notebooks: tokenizers, attention and masks, transformer blocks, and a tiny transformer trained and sampled |
  | Large Language Model Training | 8 | NumPy notebooks: deduplication, pretraining, response-only tuning, LoRA, quantisation, DPO, retrieval, evaluation and release |
  | Vision-Language Models (VLMs) | 8 | NumPy notebooks: patch encoders, contrastive alignment, a connector for visual question answering, and grounding checks |
  | Voice & Speech Models | 8 | NumPy notebooks: a hand-written DFT and log-mel features, CTC trained from scratch, augmentation, audio tokens and error rates |
  | Designing Machine Learning Systems | 16 | Architecture, data contracts, serving, monitoring, rollout and a complete design workbook |

- Notebooks open beside the lesson: in the browser with JupyterLite (no
  account or installation), as the learner's own saved copy on a JupyterHub
  server, in Google Colab (optional), or as a download. The C, C++ and CUDA
  notebooks compile code, which JupyterLite cannot do, so they skip the
  browser option: their CUDA kernels run on an NVIDIA GPU where one is
  present, such as a Colab GPU runtime, and on a CPU emulator built into the
  notebooks everywhere else
- Search course and lesson topics; filter by level or specialization
- Course overviews with prerequisites, linked learning paths, syllabuses,
  notebooks, primary references, downloadable labs and capstone review criteria
- Eight self-contained Python scripts under `frontend/public/course-labs/`,
  with setup instructions. The forecasting lab uses pinned NumPy and
  scikit-learn, and the system-design lab only Python's standard library. The
  six PyTorch labs are kept as optional versions of the deep-learning courses'
  experiments; the courses themselves no longer need them
- Interactive examples for splits, classification/regression metrics, attention
  masks, training memory and batch budgets, audio timing, CTC decoding,
  system architecture, serving capacity and request-based error budgets
- The system-design lab replays a canary rollout, delayed labels, fallback and
  rollback. Its request timings and dependency incidents are explicitly
  simulated; it does not start a prediction server or measure real throughput.
  A downloadable workbook includes a worked design and capstone rubric
- A knowledge check in every lesson, with explanations and unlimited retries
- Knowledge-check progress is saved in this browser, or to the account when
  signed in; courses can be read and completed without a dataset or a running
  backend
- Notebook checks, labs and capstones are reviewed separately from quiz
  completion. The web training tools continue to train tabular scikit-learn
  models; the workbench shuffles its train/test split, so time-series practice
  happens in the forecasting lab
- Open **Courses** from the sidebar, home page, or either model tool. Individual
  lessons have shareable URLs under `/app/courses/{course}/{lesson}`
- Every course declares a dependency tier, from tier 0 (the Python standard
  library only) to tier 3 (PyTorch). A course whose notebooks compile native
  code also declares its toolchain

### 6. Accounts and Jupyter notebooks
- Optional accounts (`/register`, `/login`) with student, instructor and admin
  roles. The workbench and lesson text stay open without one
- Signed-in course progress is saved to the account instead of the browser
- Lessons can carry a Jupyter notebook, opened three ways: as the student's
  own copy on a personal JupyterHub server (embedded beside the lesson, saved
  between visits), as an in-browser JupyterLite preview that needs no account,
  or as a download
- One isolated notebook container per student: non-root, CPU, memory and
  process limits, idle shutdown, a persistent volume, and no network route to
  the database, the API or the internet. The image includes gcc and g++ for
  the C, C++ and CUDA course; it has no GPU, so that course's kernels run on
  its CPU emulator there. JupyterHub signs students in through
  Furina's own OAuth provider, so there is no second login
- Automated grading: a signed-in student hands in a finished notebook and the
  grader re-runs the lesson's own checks on their exercise cells, in a
  throwaway container with no network, then lists each check as passed or not
  yet, with its feedback
- The hub and the grader are opt-in (`COMPOSE_PROFILES=hub,grader`). [docs/academy.md](docs/academy.md)
  covers the architecture, the security model, deployment, backups and what
  is not built yet; [notebooks/README.md](notebooks/README.md) covers writing
  notebooks

## How the model is built

Every run assembles one scikit-learn `Pipeline`:

```
ColumnTransformer
├── numeric columns      -> SimpleImputer(median) [-> StandardScaler]
└── categorical columns  -> SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown="ignore")
└── estimator (the algorithm you chose)
```

Because imputation, scaling and encoding all live *inside* the pipeline, they
are fitted on the training split only — the test split never leaks into them.
The same pipeline is what gets pickled, so a downloaded model already knows how
to preprocess raw input rows.

## Getting Started

### Docker (the whole app in one command)

Requires Docker with Compose v2.

```bash
cp .env.example .env      # optional for a local run, required before deploying
docker compose up -d --build
```

The app is then at **http://localhost:8080** — the SPA, the API under `/api/`,
the interactive docs at `/api/docs/` and the Django admin at `/admin/`, all on
one origin.

| Path | What is there |
|------|---------------|
| `/` | The front page: what the app does, the method, the estimator list |
| `/app` | The workbench — overview and the resume panel |
| `/app/preprocessing`, `/app/training`, `/app/evaluation` | The three tools, each deep-linkable |
| `/app/courses` | Searchable ML and AI course library with saved lesson progress |
| `/app/courses/{course}` | Course syllabus, prerequisites, labs and project criteria |
| `/app/courses/{course}/{lesson}` | A lesson, its exercises and knowledge check |
| `/about` | How the pipeline is assembled, in detail |
| `/api/docs/` | Interactive API documentation (assets self-hosted, no CDN) |
| `/admin/` | Django admin |

Three containers come up: `db` (PostgreSQL 17), `backend` (Django on gunicorn)
and `frontend` (the built SPA on nginx, which also reverse-proxies `/api` to
the backend — which is why there is no CORS configuration to get wrong).
Migrations run automatically on every boot.

```bash
docker compose logs -f backend        # follow the API log
docker compose exec backend python manage.py test core academy   # run the test suite
docker compose down                   # stop; datasets, models and the DB survive
docker compose down -v                # stop and erase all of it
```

Uploaded CSVs and fitted `.joblib` models live in the `media_data` volume and
the database in `db_data`, so both survive `down`/`up`.

Everything worth changing is an environment variable in `.env` — the host port,
the Postgres password, `SECRET_KEY`, `ALLOWED_HOSTS`, the gunicorn worker count
and the request timeout. Two of them ship with insecure defaults on purpose and
must be set before this is exposed to a network you do not control:

| Variable | Why |
|----------|-----|
| `SECRET_KEY` | Signs sessions and the admin login |
| `POSTGRES_PASSWORD` | The database password |

Set `DJANGO_SUPERUSER_USERNAME` and `DJANGO_SUPERUSER_PASSWORD` to have an
`/admin/` login created on first boot.

To point the SPA at a backend somewhere else, set `API_BASE_URL` — the frontend
image reads it at start-up and writes it into `/config.js`, so no rebuild is
needed.

#### The in-app helper

An optional "Ask for help" button on every workbench page, answered by a model
through OpenRouter. It is off until a key is set:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=qwen/qwen3.7-flash   # any slug from openrouter.ai/models
```

The key stays on the server — the browser only ever calls `/api/assistant/`, so
the key is never in the JavaScript bundle. When someone asks a question the
model is told which page they are on, their column *names*, types and
missing-value counts, and the scores of their last run. It is never sent a row
of their data.

#### Search engines

`robots.txt` and `sitemap.xml` are served from the site root, and the canonical
URL, Open Graph tags and sitemap entries all come from one `SITE_URL`. The
frontend container rewrites them at start-up, so deploying to a different host
needs no rebuild:

```bash
SITE_URL=https://furina.nevatal.id
```

All public pages (`/`, `/about`), workbench tools (`/app`, `/app/preprocessing`,
`/app/training`, `/app/evaluation`, `/app/dashboard`), and every course and
lesson (`/app/courses/*`) are listed in `sitemap.xml` and allowed in
`robots.txt` for indexing. The sitemap is generated from the course catalogue;
run `npm run sitemap` in `frontend/` after adding a course or lesson.

#### Per-student notebook servers (JupyterHub)

Off by default. It needs a host name for the hub and a wildcard DNS record
for the per-student subdomains under it; see the notebooks section of
`.env.example` and [docs/academy.md](docs/academy.md). To try it on one
machine:

```bash
COMPOSE_PROFILES=hub
SITE_URL=http://furina.localhost:8080
JUPYTERHUB_URL=http://notebooks.furina.localhost:8080
JUPYTERHUB_OAUTH_CLIENT_ID=furina-jupyterhub
JUPYTERHUB_OAUTH_CLIENT_SECRET=<python -c "import secrets; print(secrets.token_urlsafe(32))">
JUPYTERHUB_CRYPT_KEY=<openssl rand -hex 32>
```

Then open `http://furina.localhost:8080`, create an account and open any
lesson notebook. The in-browser JupyterLite preview at `/notebooks/` is always
built into the frontend image and needs none of this.

#### Automated grading

Also off by default, and independent of the hub. Add the `grader` profile and
one shared secret:

```bash
COMPOSE_PROFILES=hub,grader          # or just grader
GRADER_TOKEN=<openssl rand -hex 32>
```

Each lesson notebook then shows **Hand in for grading** to signed-in
students. The grader's own tests and a check of every course notebook are in
`jupyter/grader/tests/`; see [docs/academy.md](docs/academy.md#automated-grading).

#### Behind HTTPS

The `frontend` container speaks plain http on purpose, so you can put it behind
whatever already terminates TLS for you (Caddy, Traefik, a cloud load balancer).
Point that at `${APP_PORT}` and tell Django it is there:

```bash
ALLOWED_HOSTS=furina.example.com
CSRF_TRUSTED_ORIGINS=https://furina.example.com
USE_X_FORWARDED_PROTO=True
```

### Backend (Django + scikit-learn)

Requires Python 3.11+ (verified on 3.14).

```bash
cd backend
python3 -m venv env
source env/bin/activate          # Windows: env\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

The API is then at `http://localhost:8000/api/` and interactive docs at
`http://localhost:8000/api/docs/`.

### Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

The app is at `http://localhost:5173`. To point it at a different backend,
copy `.env.example` to `.env` and set `VITE_API_BASE_URL`.

To use the JupyterLite notebook preview in development, build it once into
`public/notebooks/` (needs Python 3.10+), and again after editing a notebook:

```bash
npm run notebooks
```

### Running the backend tests

```bash
cd backend
python manage.py test core academy
```

The course notebooks have their own checker (structure, dependency tiers, and
a clean run of every notebook):

```bash
python notebooks/tools/check_notebooks.py --execute
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/health/` | Service check with library versions |
| `GET` | `/api/assistant/` | Whether the in-app helper is configured |
| `POST` | `/api/assistant/` | Ask the in-app helper a question |
| `POST` | `/api/datasets/upload/` | Upload a CSV, get its profile back |
| `GET` | `/api/datasets/` | List datasets |
| `GET` | `/api/datasets/{id}/` | Profile + preview |
| `DELETE` | `/api/datasets/{id}/` | Delete a dataset |
| `GET` | `/api/datasets/{id}/download/` | Download the stored CSV |
| `POST` | `/api/datasets/{id}/preprocess/` | Apply (or dry-run) a cleaning recipe |
| `GET` | `/api/algorithms/` | Algorithms and their hyperparameter schemas |
| `POST` | `/api/train/` | Fit and evaluate a model |
| `GET` | `/api/runs/` | List training runs |
| `GET` | `/api/runs/{id}/` | Full evaluation for one run |
| `DELETE` | `/api/runs/{id}/` | Delete a run |
| `POST` | `/api/runs/{id}/predict/` | Predict from feature values |
| `GET` | `/api/runs/{id}/download_model/` | Download the fitted `.joblib` pipeline |
| `GET` | `/api/runs/{id}/report/` | Markdown or JSON evaluation report |
| `GET` | `/api/auth/session/` | Who is signed in and whether the notebook hub is enabled |
| `POST` | `/api/auth/register/`, `/api/auth/login/`, `/api/auth/logout/` | Accounts |
| `GET`, `POST` | `/api/progress/` | Saved course progress (signed in) |
| `POST` | `/api/notebooks/launch/` | Open a lesson notebook on the student's JupyterHub server |

List endpoints return bare JSON arrays. Once a visitor is signed in, unsafe
requests need the `X-CSRFToken` header; [docs/academy.md](docs/academy.md)
lists the OAuth endpoints JupyterHub uses.

## Notes

- Uploaded CSVs and fitted models are written under `backend/media/`. The
  development setup uses SQLite; set `DEVELOPMENT_MODE=False` and `DATABASE_URL`
  to use PostgreSQL (which is what `docker compose` does).
- The neural-network option is scikit-learn's `MLPClassifier` / `MLPRegressor`
  rather than TensorFlow, so the whole backend stays on one ML library.
- Training is a long, synchronous, CPU-bound request. The timeouts are lined up
  end to end — axios 300s, nginx `proxy_read_timeout` 300s, gunicorn `--timeout`
  300s — so a slow fit finishes instead of dying halfway. Raise all three
  together (`PROXY_TIMEOUT`, `GUNICORN_TIMEOUT`) if you train something bigger.
- The 50MB upload ceiling is enforced in `core/ml/io.py` and again by nginx's
  `client_max_body_size`. Both read one `MAX_UPLOAD_BYTES` value, so raising it
  in `.env` moves the whole chain at once.

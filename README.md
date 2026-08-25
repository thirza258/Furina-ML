# Furina ML

A web application for training real scikit-learn models on your own CSV files.
Upload a dataset, see what is actually in it, clean it, pick a model, tune its
hyperparameters, train it, read the evaluation, and make live predictions —
without writing any code.

Nothing is simulated: every number on the evaluation page comes from a model
that was fitted on the data you uploaded.

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
- 17 scikit-learn algorithms across both tasks, each with a hyperparameter form
  generated from the algorithm's own parameter schema
- Adjustable train/test split, random seed, stratification, feature scaling,
  and optional k-fold cross-validation
- Untouched hyperparameters are left to scikit-learn's own defaults

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

Three containers come up: `db` (PostgreSQL 17), `backend` (Django on gunicorn)
and `frontend` (the built SPA on nginx, which also reverse-proxies `/api` to
the backend — which is why there is no CORS configuration to get wrong).
Migrations run automatically on every boot.

```bash
docker compose logs -f backend        # follow the API log
docker compose exec backend python manage.py test core   # run the test suite
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

### Running the backend tests

```bash
cd backend
python manage.py test core
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/health/` | Service check with library versions |
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

List endpoints return bare JSON arrays.

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

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
  to use PostgreSQL.
- The neural-network option is scikit-learn's `MLPClassifier` / `MLPRegressor`
  rather than TensorFlow, so the whole backend stays on one ML library.
- There is no `docker-compose.yml` in this repository yet; use the local setup
  above.

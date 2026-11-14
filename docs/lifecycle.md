# The ML Lifecycle Platform

Furina ML's `lifecycle` app turns the workbench from a *training tool* into a
*reproducible experimentation platform*. This document describes what it is,
where it lives, and the rules a change to it has to keep.

## What problem it solves

The original workbench answers "how did this model do?". It could not answer:

- What data produced this model, exactly?
- Which run was best, and why?
- Can I reproduce this result?
- Is this model actually ready to use?
- Is the deployed model still behaving?

The lifecycle platform answers those, by making the relationships between
datasets, runs, models and deployments explicit and recorded.

```
Dataset → DatasetVersion → Experiment → ExperimentRun → ModelVersion → Deployment → Monitoring
```

## Where the code lives

| Concern | Module |
|---|---|
| Domain models | `lifecycle/models.py` |
| Orchestration (the only place that fits models) | `lifecycle/services.py` |
| HTTP API | `lifecycle/views.py`, `lifecycle/urls.py`, `lifecycle/serializers.py` |
| Pre-train data checks | `lifecycle/leakage.py` |
| Post-train diagnostics | `lifecycle/evaluation.py` |
| HPO / AutoML planning | `lifecycle/search.py` |
| Inference contract | `lifecycle/signature.py` |
| Environment capture | `lifecycle/env.py` |
| Storage abstraction | `lifecycle/artifacts.py` |
| Background job runner | `lifecycle/worker.py` |
| Legacy backfill | `lifecycle/management/commands/migrate_legacy_runs.py` |

## The architectural rule

> **Django orchestrates ML; it does not execute it.**

`lifecycle/services.py` builds a configuration, calls `core.ml.trainer.train`,
and stores the result. It contains no fitting logic of its own. Every
expensive operation goes through a `Job`, which the worker executes outside
the HTTP request. This is what makes the eventual move to Redis/RQ a
swap of the worker's claim loop rather than a rewrite.

Concretely:

- an HTTP view validates input, creates a `Job`, and returns its id;
- `lifecycle/worker.py` claims the job (`select_for_update` + status flip, so
  two pollers cannot double-run it) and calls `services.execute_job`;
- `execute_job` dispatches to `_execute_train` / `_execute_hpo` /
  `_execute_automl` / `_execute_reproduce`, updating progress and checking the
  cancel flag between trials;
- endpoints with `run_now=True` execute inline, which keeps small deployments
  and the test suite working without a worker.

## Reproducibility

Every `ExperimentRun` records, in full:

- **dataset**: version id, content hash, row/feature counts;
- **split**: test size, random seed, stratification, CV folds;
- **preprocessing**: the recipe that produced the version;
- **model**: algorithm, raw hyperparameters, effective hyperparameters;
- **environment**: Python, OS, Furina version, git commit, and the versions of
  numpy/pandas/scipy/scikit-learn/xgboost/lightgbm/joblib;
- **a `config_hash`**: SHA-256 of the whole configuration, so two runs are
  provably identical or provably not.

`POST /runs/<id>/reproduce/` re-trains from that record. It compares the
captured environment with the current one and, when they differ, attaches a
warning naming the difference — because binary-exact reproduction across
library versions is not something the platform can promise, and pretending
otherwise would be worse than saying so.

## Dataset versions are immutable

`DatasetVersion` rows are append-only. Creating a version writes its own file
and its own `content_hash`; nothing ever edits an existing one. This is what
makes "which data produced this model?" answerable years later.

`content_hash` is SHA-256 of the file bytes, so two uploads of identical data
hash identically and duplicates are detectable. `schema_hash` is SHA-256 of
the canonical (sorted) column description, so "same shape, different rows" is
also detectable — which is what drift and reproduction checks key off.

## Registry lifecycle

```
candidate → validated → staging → production → archived
```

States move **forward**, except that archiving is allowed from anywhere (the
one legitimate backwards move). Promoting to `production` requires
`confirmed=true`, because it changes what live traffic receives; promoting a
new production version automatically archives the incumbent, so there is
always exactly one production version per model.

A registered `ModelVersion` is immutable. Its `signature` is the inference
contract; its `model_card` is the auto-generated, honest summary — including
`known_limitations` taken from the run's own warnings, and a
`data_quality_note` when leakage or quality flags fired.

## Monitoring and drift

Deploying a version records **baseline statistics** for every signature
feature (numeric mean/std/range; categorical value shares; missing rate),
computed from the version's own data. `POST /deployments/<slug>/drift-check/`
compares incoming rows against those baselines and reports a
population-stability index per feature:

| PSI | Status |
|---|---|
| < 0.1 | stable |
| 0.1 – 0.25 | moderate |
| > 0.25 | significant |

Every prediction also writes a `MonitoringRecord`, so request counts, latency
and error rates are real numbers rather than estimates.

## Resource governance

Before any fit, `services.enforce_limits` refuses a job that exceeds the
configured ceilings, and `services.estimate_resources` grades the likely
memory use so an obviously-too-large job is visible before it starts. The
ceilings are settings (`FURINA_MAX_TRAIN_ROWS`, `FURINA_MAX_TRAIN_COLUMNS`,
`FURINA_MAX_HPO_TRIALS`, `FURINA_AUTOML_MAX_ROWS`), all environment-tunable.
AutoML additionally subsamples above `FURINA_AUTOML_MAX_ROWS` and enforces its
own wall-clock budget.

## Backwards compatibility

The existing `/api/datasets/`, `/api/train/`, `/api/runs/` endpoints are
untouched and their response shapes are frozen — `core/tests.py` asserts them
and the shipping frontend consumes them. History is preserved by
`ExperimentRun.legacy_run`: a lifecycle run keeps a pointer to the
`core.TrainingRun` it corresponds to, and registration carries that pointer
onto the `ModelVersion`. `manage.py migrate_legacy_runs` backfills existing
installs idempotently.

## Testing

`lifecycle/tests.py` drives the real HTTP endpoints with `run_now=True`, in a
throwaway `MEDIA_ROOT`, mirroring `core/tests.py`'s fixture pattern. Run it
with:

```bash
cd backend
DEVELOPMENT_MODE=True python manage.py test lifecycle
```

The worker refuses to start under `manage.py test` by design (a background
thread racing the synchronous path would flake the suite), and can be disabled
in any deployment with `FURINA_DISABLE_JOB_POLLER=1`.

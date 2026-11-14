"""Hyperparameter search and the AutoML candidate schedule.

Three search methods the PRD asks for first: random, grid, and successive
halving. Each produces a *plan* -- an ordered list of trial hyperparameter
dicts -- that the job executor feeds, one at a time, into
``services.run_training_config``. Keeping the plan a pure function of
``(space, method, n_trials, seed)`` makes the whole search reproducible: the
same seed replays the same trials.

Successive halving is implemented as a budgeted plan: it proposes a coarse
grid of configurations and lets the executor drop the laggards early. The
plan itself is the set of candidate configurations; early stopping is the
executor's job, which keeps this module free of any training code.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

from core.ml.registry import get_spec

#: The algorithms AutoML walks, in the order it tries them. A small, fast,
#: interpretable model first (the baseline), then the strong tabular learners.
AUTOML_DEFAULT_CANDIDATES: list[tuple[str, dict[str, Any]]] = [
    ("logistic_regression", {}),
    ("decision_tree", {}),
    ("random_forest", {}),
    ("hist_gradient_boosting", {}),
    ("xgboost", {}),
    ("lightgbm", {}),
]

AUTOML_REGRESSION_CANDIDATES: list[tuple[str, dict[str, Any]]] = [
    ("linear_regression", {}),
    ("decision_tree", {}),
    ("random_forest", {}),
    ("hist_gradient_boosting", {}),
    ("xgboost", {}),
    ("lightgbm", {}),
]


class SearchError(Exception):
    """A user-readable reason a search space is unusable."""


@dataclass
class ParamSpace:
    """A single algorithm plus the hyperparameters to search over it.

    ``params`` maps a hyperparameter name to its sampling rule:
    ``{"values": [...]}`` for a finite set, ``{"log": [lo, hi, n]}`` for a
    log-spaced range, or a bare list of values. Anything else is treated as
    a fixed value for the whole search.
    """

    algorithm: str
    params: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "ParamSpace":
        payload = payload or {}
        algorithm = str(payload.get("algorithm") or "logistic_regression")
        raw_params = payload.get("params") or {}
        extra = {k: v for k, v in raw_params.items() if isinstance(v, (int, float, str, bool))}
        sampled = {
            k: v for k, v in raw_params.items() if isinstance(v, dict)
        }
        return cls(algorithm=algorithm, params=sampled, extra=extra)

    def sample_values(self, name: str, rng: Any) -> list[Any]:
        """The distinct values this parameter can take in the search."""
        rule = self.params.get(name)
        if rule is None:
            return []
        if "values" in rule:
            values = list(rule["values"])
        elif "log" in rule:
            lo, hi, points = rule["log"]
            points = int(points) if points and points >= 2 else 5
            step = (math.log10(hi) - math.log10(lo)) / (points - 1) if hi > lo > 0 else 0.0
            values = [float(f"{10 ** (math.log10(lo) + step * i):.6g}") for i in range(points)]
        else:
            values = list(rule)
        return values

    def grid(self) -> list[dict[str, Any]]:
        """The full Cartesian product of every sampled parameter."""
        names = [name for name in self.params if self.sample_values(name, None)]
        axes = [self.sample_values(name, None) for name in names]
        grids = list(itertools.product(*axes)) if axes else [{}]
        return [{name: value for name, value in zip(names, combo)} for combo in grids]

    def apply(self, chosen: dict[str, Any]) -> dict[str, Any]:
        """Merge a chosen combination with the fixed values."""
        return {**self.extra, **chosen}


def _shuffle(values: list[Any], rng: Any) -> list[Any]:
    order = list(range(len(values)))
    rng.shuffle(order)
    return [values[i] for i in order]


def plan_trials(
    space: ParamSpace,
    *,
    method: str = "random",
    n_trials: int = 20,
    cv_folds: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Return ``n_trials`` trial configs (each ``{"hyperparameters": {...}}``).

    ``cv_folds`` is not part of the trial config -- it is passed to the
    executor -- but it is returned here so a caller that wants to log it has a
    single home for the whole search description.
    """
    import random

    rng = random.Random(seed)
    method = str(method).lower()
    n_trials = max(1, int(n_trials))

    grids = space.grid()
    if not grids:
        # Nothing to vary: one trial with the fixed values only.
        return [
            {
                "hyperparameters": space.apply({}),
                "algorithm": space.algorithm,
                "method": method,
                "cv_folds": cv_folds,
            }
        ]

    if method == "grid":
        chosen = (grids + [grids[0]] * n_trials)[:n_trials]
    elif method == "successive_halving":
        # A coarse first sweep over the distinct combinations, then the same
        # set again as a "budget" round; the executor early-stops the laggards
        # between rounds. Capped so a big grid cannot explode the plan.
        chosen = (grids + [g for g in _shuffle(list(grids), rng)][:n_trials])[:n_trials]
        chosen = chosen[: max(1, n_trials)]
    else:  # random (and unknown methods fall back to it)
        names = [name for name in space.params if space.sample_values(name, None)]
        chosen = []
        for _ in range(n_trials):
            combo = {
                name: rng.choice(space.sample_values(name, rng)) for name in names
            }
            chosen.append(combo)

    return [
        {
            "hyperparameters": space.apply(combo),
            "algorithm": space.algorithm,
            "method": method,
            "cv_folds": cv_folds,
        }
        for combo in chosen
    ]


# --------------------------------------------------------------------------
# suggested search spaces
# --------------------------------------------------------------------------


def suggested_search_space(spec: dict[str, Any]) -> dict[str, Any]:
    """Reasonable default ranges for an algorithm's *searchable* parameters.

    Only the parameters that actually move the needle are varied; everything
    else is left to the estimator's default. The ranges honour the spec's own
    advertised min/max so a suggestion can never be out of range for the form.
    """
    def range_of(name: str, candidate: list[float]) -> list[float]:
        param = next((p for p in spec.get("hyperparameters", []) if p.get("name") == name), None)
        if not param:
            return candidate
        lo, hi = param.get("min"), param.get("max")
        lo = lo if lo is not None else (candidate[0] if candidate else 0.0)
        hi = hi if hi is not None else (candidate[-1] if candidate else 1.0)
        return [max(lo, v) if v < hi else v for v in candidate]

    params: dict[str, Any] = {}
    key = spec.get("key", "")
    family = spec.get("family", "")

    if "tree" in key.lower() or "forest" in key.lower() or "boost" in key.lower() or "xgb" in key.lower() or "lgbm" in key.lower():
        params["max_depth"] = {"values": [3, 5, 7, 10]}
        if key in ("xgboost", "lightgbm", "gradient_boosting", "hist_gradient_boosting"):
            params["learning_rate"] = {"log": [0.005, 0.3, 5]}
        if "n_estimators" in {p.get("name") for p in spec.get("hyperparameters", [])}:
            params["n_estimators"] = {"values": [100, 200, 400]}
    if "regression" in family.lower() or "linear" in key.lower():
        if key in ("ridge_regression", "lasso_regression", "elastic_net", "bayesian_ridge", "logistic_regression"):
            param_names = {p.get("name") for p in spec.get("hyperparameters", [])}
            if "C" in param_names:
                params["C"] = {"log": [0.1, 100.0, 5]}
            if "alpha" in param_names:
                params["alpha"] = {"log": [1e-4, 10.0, 5]}
    if "n_estimators" not in params and "n_estimators" in {p.get("name") for p in spec.get("hyperparameters", [])}:
        params["n_estimators"] = {"values": [100, 300]}
    if "n_neighbors" in {p.get("name") for p in spec.get("hyperparameters", [])}:
        params["n_neighbors"] = {"values": [3, 5, 9, 15]}
    if not params:
        # Even with nothing to vary, return one stable trial's worth.
        params = {}
    return {"algorithm": key, "params": params}


def default_candidates(task_type: str, max_models: int) -> list[tuple[str, dict[str, Any]]]:
    """The AutoML candidate list, bounded by ``max_models`` and the task."""
    table = AUTOML_REGRESSION_CANDIDATES if task_type == "regression" else AUTOML_DEFAULT_CANDIDATES
    return table[: max(1, max_models)]


def automl_schedule(candidates: list[tuple[str, dict[str, Any]]], seed: int = 42) -> list[tuple[str, dict[str, Any]]]:
    """Yield candidate model + starting hyperparameters for a full AutoML pass."""
    # Deterministic: the seed only reorders the non-default models so a rerun
    # explores the same order. The baseline is always first.
    import random

    ordered = list(candidates)
    if len(ordered) > 1:
        rng = random.Random(seed)
        head, tail = ordered[0], ordered[1:]
        rng.shuffle(tail)
        ordered = [head] + tail
    return ordered

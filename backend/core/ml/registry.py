"""Catalogue of the scikit-learn algorithms Furina ML exposes.

This module owns three things:

* the `AlgorithmSpec` payload the frontend renders (labels, families and the
  hyperparameter form for every algorithm),
* `coerce_hyperparameters`, which turns untrusted form input into keyword
  arguments scikit-learn will accept (it never raises, it warns),
* `build_estimator`, which constructs the unfitted estimator, passing only the
  keyword arguments that particular estimator class actually accepts.

No Django and no pandas imports: everything here is plain Python plus
scikit-learn, so it can be imported and tested on its own.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Mapping, Sequence

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# The two non-scikit-learn entries in the catalogue. Both ship a scikit-learn
# compatible wrapper, so they drop into the same Pipeline, answer to the same
# `clone()` that cross-validation relies on, and pickle through joblib exactly
# like every other estimator here. They are pinned in requirements.txt; a hard
# import means a missing dependency fails at startup rather than silently
# shrinking the catalogue.
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

__all__ = [
    "ALGORITHM_KEYS",
    "AlgorithmNotFound",
    "build_estimator",
    "coerce_hyperparameters",
    "get_spec",
    "list_algorithms",
    "supports",
]

CLASSIFICATION = "classification"
REGRESSION = "regression"
TASK_TYPES = (CLASSIFICATION, REGRESSION)

#: Largest hidden layer and deepest network the MLP form will accept. Anything
#: bigger is a typo far more often than it is a real request, and it would hang
#: the request thread for minutes.
MAX_LAYER_UNITS = 1024
MAX_LAYERS = 5


class AlgorithmNotFound(KeyError):
    """Raised when an algorithm key is not in the registry."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else "Unknown algorithm."


# Sentinel meaning "leave this hyperparameter out and let scikit-learn use its
# own default". Never returned to callers.
class _Omit:
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<use default>"


_OMIT = _Omit()

# A cast maps a validated option string to the value scikit-learn expects and
# may report warnings; it raises ValueError (with a plain sentence) on input it
# cannot use.
Cast = Callable[[str], "tuple[Any, list[str]]"]

#: Estimators that accept `n_jobs` but no longer use it. Passing it there only
#: produces a deprecation warning in the user's training log.
_IGNORES_N_JOBS = frozenset({"LogisticRegression"})


def _cast_text(value: str) -> tuple[Any, list[str]]:
    """Keep the option value as-is."""
    return value, []


def _cast_default_when_none(value: str) -> tuple[Any, list[str]]:
    """Map the literal option "none" to "use the scikit-learn default"."""
    return (_OMIT if value == "none" else value), []


def _cast_max_features(value: str) -> tuple[Any, list[str]]:
    """Map "1.0" (all features) to a float, keep "sqrt"/"log2" as strings."""
    if value in ("sqrt", "log2"):
        return value, []
    return float(value), []


def _cast_bool_or_auto(value: str) -> tuple[Any, list[str]]:
    """Map "auto"/"true"/"false" to "auto"/True/False."""
    if value == "auto":
        return "auto", []
    return value == "true", []


def _cast_hidden_layers(value: str) -> tuple[Any, list[str]]:
    """Parse "100,50" into the tuple (100, 50) scikit-learn wants."""
    warnings: list[str] = []
    text = str(value).strip().strip("()[]")
    parts = [part.strip() for part in text.split(",")]
    while parts and parts[-1] == "":
        parts.pop()
    if not parts or any(part == "" for part in parts):
        raise ValueError(
            "Layer sizes must be whole numbers separated by commas, for example '100,50'."
        )
    sizes: list[int] = []
    for part in parts:
        try:
            size = int(float(part))
        except (TypeError, ValueError):
            raise ValueError(
                "Layer sizes must be whole numbers separated by commas, for example '100,50'."
            ) from None
        if size < 1:
            raise ValueError("Every hidden layer needs at least one neuron.")
        if size > MAX_LAYER_UNITS:
            warnings.append(
                f"A hidden layer of {size} neurons was reduced to {MAX_LAYER_UNITS}, "
                "the largest size this app allows."
            )
            size = MAX_LAYER_UNITS
        sizes.append(size)
    if len(sizes) > MAX_LAYERS:
        warnings.append(
            f"Only the first {MAX_LAYERS} hidden layers were kept; the rest were ignored."
        )
        sizes = sizes[:MAX_LAYERS]
    return tuple(sizes), warnings


@dataclass(frozen=True)
class _Param:
    """One row of the hyperparameter form for one algorithm."""

    name: str
    label: str
    type: str
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    options: tuple[tuple[str, str], ...] | None = None
    description: str = ""
    advanced: bool = False
    cast: Cast = _cast_text
    free_text: bool = False

    def to_spec(self) -> dict[str, Any]:
        """Render this parameter as a contract `HyperparamSpec` dict."""
        return {
            "name": self.name,
            "label": self.label,
            "type": self.type,
            "default": self.default,
            "min": None if self.minimum is None else float(self.minimum),
            "max": None if self.maximum is None else float(self.maximum),
            "step": None if self.step is None else float(self.step),
            "options": (
                [{"value": value, "label": label} for value, label in self.options]
                if self.options
                else None
            ),
            "description": self.description,
            "advanced": self.advanced,
        }


@dataclass(frozen=True)
class _Variant:
    """The estimator class and form used for one task type."""

    cls: type
    params: tuple[_Param, ...]
    overrides: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _Algorithm:
    """One catalogue entry, covering classification, regression or both."""

    key: str
    label: str
    family: str
    description: str
    supports_proba: bool
    supports_feature_importance: bool
    variants: Mapping[str, _Variant]

    @property
    def short_label(self) -> str:
        """The label without the "Scikit-learn — " prefix, for messages."""
        return self.label.split("—")[-1].strip()


# --------------------------------------------------------------------------
# Reusable parameter builders
# --------------------------------------------------------------------------


def _int_param(
    name: str,
    label: str,
    default: int | None,
    minimum: float,
    maximum: float,
    description: str,
    *,
    step: float = 1,
    advanced: bool = False,
) -> _Param:
    return _Param(
        name=name,
        label=label,
        type="int",
        default=default,
        minimum=minimum,
        maximum=maximum,
        step=step,
        description=description,
        advanced=advanced,
    )


def _float_param(
    name: str,
    label: str,
    default: float | None,
    minimum: float,
    maximum: float,
    description: str,
    *,
    step: float = 0.01,
    advanced: bool = False,
) -> _Param:
    return _Param(
        name=name,
        label=label,
        type="float",
        default=default,
        minimum=minimum,
        maximum=maximum,
        step=step,
        description=description,
        advanced=advanced,
    )


def _bool_param(
    name: str, label: str, default: bool, description: str, *, advanced: bool = False
) -> _Param:
    return _Param(
        name=name,
        label=label,
        type="bool",
        default=default,
        description=description,
        advanced=advanced,
    )


def _select_param(
    name: str,
    label: str,
    default: str,
    options: Sequence[tuple[str, str]],
    description: str,
    *,
    advanced: bool = False,
    cast: Cast = _cast_text,
    free_text: bool = False,
) -> _Param:
    return _Param(
        name=name,
        label=label,
        type="select",
        default=default,
        options=tuple(options),
        description=description,
        advanced=advanced,
        cast=cast,
        free_text=free_text,
    )


def _class_weight_param(*, subsample: bool = False) -> _Param:
    options = [("none", "Treat every class equally"), ("balanced", "Balanced")]
    if subsample:
        options.append(("balanced_subsample", "Balanced per tree"))
    return _select_param(
        "class_weight",
        "Class weighting",
        "none",
        options,
        "Choose 'Balanced' when one class is much rarer than the others so it is not ignored.",
        cast=_cast_default_when_none,
    )


def _max_depth_param(*, advanced: bool = False) -> _Param:
    return _int_param(
        "max_depth",
        "Maximum tree depth",
        None,
        1,
        50,
        "How many questions deep a tree may go. Smaller trees are simpler and overfit less; "
        "leave it blank to let the tree grow as far as it needs to.",
        advanced=advanced,
    )


def _min_samples_split_param() -> _Param:
    return _int_param(
        "min_samples_split",
        "Minimum rows to split a node",
        2,
        2,
        200,
        "A branch is only split further when it still holds at least this many rows.",
    )


def _min_samples_leaf_param(default: int = 1, *, maximum: int = 200) -> _Param:
    return _int_param(
        "min_samples_leaf",
        "Minimum rows per leaf",
        default,
        1,
        maximum,
        "Every final group must keep at least this many rows, which smooths out noisy splits.",
    )


def _tree_max_features_param(default: str, *, advanced: bool = False) -> _Param:
    return _select_param(
        "max_features",
        "Features considered per split",
        default,
        [
            ("sqrt", "Square root of the feature count"),
            ("log2", "Log2 of the feature count"),
            ("1.0", "All features"),
        ],
        "How many features each split may look at. Looking at fewer features makes the trees "
        "more varied and usually generalizes better.",
        advanced=advanced,
        cast=_cast_max_features,
    )


def _n_estimators_param(default: int, description: str) -> _Param:
    return _int_param(
        "n_estimators",
        "Number of trees",
        default,
        10,
        1000,
        description,
        step=10,
    )


def _tol_param(default: float, *, minimum: float = 1e-06, maximum: float = 0.1) -> _Param:
    return _float_param(
        "tol",
        "Stopping tolerance",
        default,
        minimum,
        maximum,
        "How small the improvement per pass has to get before training stops.",
        step=1e-06,
        advanced=True,
    )


def _fit_intercept_param() -> _Param:
    return _bool_param(
        "fit_intercept",
        "Fit an intercept",
        True,
        "Let the model add a constant offset. Leave this on unless your data is already centred.",
        advanced=True,
    )


def _positive_param() -> _Param:
    return _bool_param(
        "positive",
        "Force positive coefficients",
        False,
        "Require every coefficient to be zero or positive, for cases where a negative effect "
        "makes no sense.",
        advanced=True,
    )


def _alpha_param(
    default: float = 1.0, *, label: str = "Regularization strength (alpha)"
) -> _Param:
    return _float_param(
        "alpha",
        label,
        default,
        0.0001,
        1000.0,
        "Higher values shrink the coefficients more, which fights overfitting but can "
        "underfit if pushed too far.",
        step=0.01,
    )


def _selection_param() -> _Param:
    return _select_param(
        "selection",
        "Coefficient update order",
        "cyclic",
        [("cyclic", "In order"), ("random", "Random order")],
        "Updating coefficients in a random order often converges faster on wide datasets.",
        advanced=True,
    )


def _linear_max_iter_param(default: int | None, *, maximum: int = 10000) -> _Param:
    return _int_param(
        "max_iter",
        "Maximum training passes",
        default,
        50,
        maximum,
        "The most passes over the data before fitting stops. Raise it if you see a "
        "convergence warning.",
        step=50,
    )


# --------------------------------------------------------------------------
# The catalogue
# --------------------------------------------------------------------------

_SGD_SHARED: tuple[_Param, ...] = (
    _float_param(
        "alpha",
        "Regularization strength (alpha)",
        0.0001,
        1e-07,
        1.0,
        "Higher values keep the weights small, which reduces overfitting.",
        step=0.0001,
    ),
    _select_param(
        "penalty",
        "Penalty type",
        "l2",
        [("l2", "L2 (ridge)"), ("l1", "L1 (lasso)"), ("elasticnet", "Elastic net (both)")],
        "L2 shrinks every weight, L1 drives weak features to exactly zero, and elastic net "
        "mixes the two.",
    ),
    _linear_max_iter_param(1000),
    _float_param(
        "l1_ratio",
        "Elastic net mix",
        0.15,
        0.0,
        1.0,
        "How much of the elastic-net penalty is L1 rather than L2. Only used when the "
        "penalty is elastic net.",
        step=0.05,
        advanced=True,
    ),
    _float_param(
        "eta0",
        "Starting learning rate",
        0.01,
        0.0001,
        10.0,
        "The step size the model starts with. Too large and training becomes unstable, "
        "too small and it never gets there.",
        step=0.001,
        advanced=True,
    ),
    _tol_param(0.001),
)

_TREE_SHARED: tuple[_Param, ...] = (
    _max_depth_param(),
    _min_samples_split_param(),
    _min_samples_leaf_param(),
    _int_param(
        "max_leaf_nodes",
        "Maximum leaves",
        None,
        2,
        1000,
        "Cap the number of final groups the tree may end up with. Leave it blank for no cap.",
        advanced=True,
    ),
    _float_param(
        "ccp_alpha",
        "Pruning strength",
        0.0,
        0.0,
        1.0,
        "Prunes branches that add very little accuracy. Larger values give a smaller, "
        "simpler tree.",
        step=0.001,
        advanced=True,
    ),
)

_FOREST_SHARED: tuple[_Param, ...] = (
    _max_depth_param(),
    _min_samples_split_param(),
    _min_samples_leaf_param(),
)

_GB_SHARED: tuple[_Param, ...] = (
    _n_estimators_param(100, "How many small trees are added one after another."),
    _float_param(
        "learning_rate",
        "Learning rate",
        0.1,
        0.001,
        1.0,
        "How much each new tree is allowed to correct. Lower values need more trees but "
        "usually score better.",
        step=0.01,
    ),
    _int_param(
        "max_depth",
        "Maximum tree depth",
        3,
        1,
        20,
        "Depth of each individual tree. Boosting works best with shallow trees.",
    ),
    _min_samples_split_param(),
    _min_samples_leaf_param(),
    _float_param(
        "subsample",
        "Row sample per tree",
        1.0,
        0.1,
        1.0,
        "Fraction of the rows each tree is trained on. Below 1.0 adds randomness that can "
        "reduce overfitting.",
        step=0.05,
        advanced=True,
    ),
    _tree_max_features_param("1.0", advanced=True),
)

# --------------------------------------------------------------------------
# Gradient boosting libraries outside scikit-learn
#
# Defaults below are each library's own defaults, not opinions of ours -- the
# app's contract is that a hyperparameter you do not touch is left where the
# library put it. XGBoost's learning rate really is 0.3 and LightGBM's really
# is 0.1; they are not typos for each other.
# --------------------------------------------------------------------------

_XGB_SHARED: tuple[_Param, ...] = (
    _n_estimators_param(100, "How many boosting rounds are run, one tree per round."),
    _float_param(
        "learning_rate",
        "Learning rate",
        0.3,
        0.001,
        1.0,
        "How much each new tree is allowed to correct. Lower values need more rounds "
        "but usually score better.",
        step=0.01,
    ),
    _int_param(
        "max_depth",
        "Maximum tree depth",
        6,
        1,
        16,
        "The main size control for this model: deeper trees capture more interaction "
        "between features and overfit sooner.",
    ),
    _float_param(
        "subsample",
        "Row sampling per round",
        1.0,
        0.1,
        1.0,
        "The fraction of rows each tree is fitted on. Below 1.0 this adds randomness "
        "that usually helps generalisation.",
        step=0.05,
    ),
    _float_param(
        "colsample_bytree",
        "Feature sampling per tree",
        1.0,
        0.1,
        1.0,
        "The fraction of features each tree may choose from.",
        step=0.05,
    ),
    _float_param(
        "min_child_weight",
        "Minimum child weight",
        1.0,
        0.0,
        20.0,
        "How much evidence a split must have before it is kept. Higher values make "
        "the model more conservative.",
        step=0.1,
        advanced=True,
    ),
    _float_param(
        "gamma",
        "Minimum split gain",
        0.0,
        0.0,
        10.0,
        "A split has to improve the loss by at least this much to be made at all.",
        step=0.01,
        advanced=True,
    ),
    _float_param(
        "reg_lambda",
        "L2 regularization",
        1.0,
        0.0,
        10.0,
        "Penalty on large leaf values, which smooths the model.",
        step=0.01,
        advanced=True,
    ),
    _float_param(
        "reg_alpha",
        "L1 regularization",
        0.0,
        0.0,
        10.0,
        "Penalty that pushes leaf values to exactly zero, which simplifies the model.",
        step=0.01,
        advanced=True,
    ),
)

_LGBM_SHARED: tuple[_Param, ...] = (
    _n_estimators_param(100, "How many boosting rounds are run, one tree per round."),
    _float_param(
        "learning_rate",
        "Learning rate",
        0.1,
        0.001,
        1.0,
        "How much each new tree is allowed to correct. Lower values need more rounds "
        "but usually score better.",
        step=0.01,
    ),
    _int_param(
        "num_leaves",
        "Maximum leaves per tree",
        31,
        2,
        255,
        "The main size control for this model: LightGBM grows leaf-by-leaf rather than "
        "level-by-level, so this matters more here than tree depth does.",
    ),
    _int_param(
        "min_child_samples",
        "Minimum rows per leaf",
        20,
        1,
        200,
        "A leaf has to hold at least this many rows, which stops the model carving out "
        "single-row special cases.",
    ),
    _int_param(
        "max_depth",
        "Maximum tree depth",
        -1,
        -1,
        32,
        "An extra ceiling on depth on top of the leaf count. -1 means no limit, which "
        "is the usual choice for LightGBM.",
        advanced=True,
    ),
    _float_param(
        "colsample_bytree",
        "Feature sampling per tree",
        1.0,
        0.1,
        1.0,
        "The fraction of features each tree may choose from.",
        step=0.05,
        advanced=True,
    ),
    _float_param(
        "reg_lambda",
        "L2 regularization",
        0.0,
        0.0,
        10.0,
        "Penalty on large leaf values, which smooths the model.",
        step=0.01,
        advanced=True,
    ),
    _float_param(
        "reg_alpha",
        "L1 regularization",
        0.0,
        0.0,
        10.0,
        "Penalty that pushes leaf values to exactly zero, which simplifies the model.",
        step=0.01,
        advanced=True,
    ),
    # LightGBM's `subsample` is deliberately absent: it does nothing unless
    # `subsample_freq` is also set above 0, and a control that silently does
    # nothing is worse than no control at all.
)

_HGB_SHARED: tuple[_Param, ...] = (
    _int_param(
        "max_iter",
        "Number of boosting rounds",
        100,
        10,
        1000,
        "How many trees are added one after another.",
        step=10,
    ),
    _float_param(
        "learning_rate",
        "Learning rate",
        0.1,
        0.001,
        1.0,
        "How much each new tree is allowed to correct. Lower values need more rounds but "
        "usually score better.",
        step=0.01,
    ),
    _int_param(
        "max_leaf_nodes",
        "Maximum leaves per tree",
        31,
        2,
        255,
        "The main size control for this model: more leaves capture more detail and "
        "overfit sooner.",
    ),
    _min_samples_leaf_param(20),
    _max_depth_param(advanced=True),
    _float_param(
        "l2_regularization",
        "L2 regularization",
        0.0,
        0.0,
        10.0,
        "Extra penalty on large leaf values, which smooths the model.",
        step=0.01,
        advanced=True,
    ),
    _int_param(
        "max_bins",
        "Histogram bins",
        255,
        2,
        255,
        "How finely numeric features are bucketed before splitting. Fewer bins train faster "
        "but lose detail.",
        advanced=True,
    ),
    _select_param(
        "early_stopping",
        "Early stopping",
        "auto",
        [("auto", "Automatic"), ("true", "Always on"), ("false", "Off")],
        "Stop adding trees once a held-out slice of the data stops improving. 'Automatic' "
        "turns it on for large datasets only.",
        advanced=True,
        cast=_cast_bool_or_auto,
    ),
)

_KNN_SHARED: tuple[_Param, ...] = (
    _int_param(
        "n_neighbors",
        "Number of neighbours (k)",
        5,
        1,
        100,
        "How many similar rows vote on each prediction. Larger values give smoother, "
        "less jumpy predictions.",
    ),
    _select_param(
        "weights",
        "Neighbour weighting",
        "uniform",
        [
            ("uniform", "Every neighbour counts equally"),
            ("distance", "Closer neighbours count more"),
        ],
        "Whether nearby rows should have more say than far-away ones.",
    ),
    _select_param(
        "metric",
        "Distance measure",
        "minkowski",
        [
            ("minkowski", "Minkowski"),
            ("euclidean", "Euclidean (straight line)"),
            ("manhattan", "Manhattan (city block)"),
            ("chebyshev", "Chebyshev"),
            ("cosine", "Cosine"),
        ],
        "How the distance between two rows is measured.",
        advanced=True,
    ),
    _int_param(
        "p",
        "Minkowski power",
        2,
        1,
        5,
        "Shape of the Minkowski distance: 1 is city-block, 2 is straight-line. Ignored by "
        "the other distance measures.",
        advanced=True,
    ),
    _select_param(
        "algorithm",
        "Neighbour search method",
        "auto",
        [
            ("auto", "Choose automatically"),
            ("ball_tree", "Ball tree"),
            ("kd_tree", "KD tree"),
            ("brute", "Brute force"),
        ],
        "The lookup structure used to find neighbours. 'Choose automatically' is almost "
        "always right.",
        advanced=True,
    ),
    _int_param(
        "leaf_size",
        "Search tree leaf size",
        30,
        1,
        200,
        "Tuning knob for the ball-tree and KD-tree searches. It changes speed, not accuracy.",
        advanced=True,
    ),
)

_MLP_SHARED: tuple[_Param, ...] = (
    _select_param(
        "hidden_layer_sizes",
        "Hidden layers",
        "100",
        [
            ("50", "One layer of 50"),
            ("100", "One layer of 100"),
            ("200", "One layer of 200"),
            ("100,50", "Two layers: 100 then 50"),
            ("64,32", "Two layers: 64 then 32"),
            ("64,32,16", "Three layers: 64, 32 then 16"),
        ],
        "The neurons in each hidden layer, separated by commas. '100,50' means a layer of "
        "100 followed by a layer of 50.",
        cast=_cast_hidden_layers,
        free_text=True,
    ),
    _select_param(
        "activation",
        "Activation function",
        "relu",
        [
            ("relu", "ReLU"),
            ("tanh", "Tanh"),
            ("logistic", "Logistic (sigmoid)"),
            ("identity", "Identity (no activation)"),
        ],
        "The non-linearity applied inside each neuron. ReLU is the usual first choice.",
    ),
    _float_param(
        "alpha",
        "Regularization strength (alpha)",
        0.0001,
        1e-07,
        1.0,
        "Penalty on large weights. Raise it if the network memorises the training data.",
        step=0.0001,
    ),
    _float_param(
        "learning_rate_init",
        "Initial learning rate",
        0.001,
        1e-05,
        1.0,
        "How big the first weight updates are. Too large and training bounces around, too "
        "small and it crawls.",
        step=0.0001,
    ),
    _int_param(
        "max_iter",
        "Maximum training passes",
        200,
        50,
        2000,
        "The most passes over the training data. Raise it if you see a convergence warning.",
        step=50,
    ),
    _bool_param(
        "early_stopping",
        "Early stopping",
        False,
        "Hold back part of the training data and stop as soon as it stops improving.",
    ),
    _select_param(
        "solver",
        "Optimizer",
        "adam",
        [("adam", "Adam"), ("lbfgs", "L-BFGS"), ("sgd", "Plain gradient descent")],
        "How the weights are updated. 'Adam' suits most datasets; 'L-BFGS' can be better on "
        "small ones.",
        advanced=True,
    ),
    _select_param(
        "learning_rate",
        "Learning rate schedule",
        "constant",
        [
            ("constant", "Constant"),
            ("invscaling", "Gradually decreasing"),
            ("adaptive", "Adaptive"),
        ],
        "How the learning rate changes while training. Only used by the plain gradient "
        "descent optimizer.",
        advanced=True,
    ),
)


_ALGORITHMS: tuple[_Algorithm, ...] = (
    _Algorithm(
        key="logistic_regression",
        label="Scikit-learn — Logistic Regression",
        family="Linear Models",
        description=(
            "A fast, dependable baseline classifier. It weighs each feature and turns the "
            "total into a probability for every class."
        ),
        supports_proba=True,
        supports_feature_importance=True,
        variants={
            CLASSIFICATION: _Variant(
                cls=LogisticRegression,
                overrides={"max_iter": 1000},
                params=(
                    _float_param(
                        "C",
                        "Regularization strength (C)",
                        1.0,
                        0.001,
                        1000.0,
                        "Smaller values regularize harder and keep the model simple; larger "
                        "values let it fit the training data more closely.",
                        step=0.01,
                    ),
                    _linear_max_iter_param(1000),
                    _class_weight_param(),
                    _float_param(
                        "l1_ratio",
                        "Feature sparsity (L1 ratio)",
                        0.0,
                        0.0,
                        1.0,
                        "0 shrinks every feature a little; 1 removes weak features entirely. "
                        "Anything above 0 needs the 'saga' solver, which is selected for you.",
                        step=0.05,
                        advanced=True,
                    ),
                    _select_param(
                        "solver",
                        "Solver",
                        "lbfgs",
                        [
                            ("lbfgs", "L-BFGS"),
                            ("saga", "SAGA"),
                            ("newton-cholesky", "Newton-Cholesky"),
                        ],
                        "The optimization routine used to fit the model. 'L-BFGS' works well "
                        "for most datasets.",
                        advanced=True,
                    ),
                    _tol_param(0.0001),
                    _fit_intercept_param(),
                ),
            )
        },
    ),
    _Algorithm(
        key="ridge_classifier",
        label="Scikit-learn — Ridge Classifier",
        family="Linear Models",
        description=(
            "A very fast linear classifier that keeps its coefficients small. It reports a "
            "class but no probabilities."
        ),
        supports_proba=False,
        supports_feature_importance=True,
        variants={
            CLASSIFICATION: _Variant(
                cls=RidgeClassifier,
                params=(
                    _alpha_param(1.0),
                    _class_weight_param(),
                    _select_param(
                        "solver",
                        "Solver",
                        "auto",
                        [
                            ("auto", "Choose automatically"),
                            ("svd", "SVD"),
                            ("cholesky", "Cholesky"),
                            ("lsqr", "LSQR"),
                            ("sparse_cg", "Sparse CG"),
                            ("sag", "SAG"),
                            ("saga", "SAGA"),
                        ],
                        "The numerical routine used to solve the model. 'Choose automatically' "
                        "is almost always right.",
                        advanced=True,
                    ),
                    _linear_max_iter_param(None),
                    _tol_param(0.0001),
                    _fit_intercept_param(),
                ),
            )
        },
    ),
    _Algorithm(
        key="linear_regression",
        label="Scikit-learn — Linear Regression",
        family="Linear Models",
        description=(
            "The classic straight-line fit. It has almost nothing to tune, which makes it a "
            "good first model and an honest baseline."
        ),
        supports_proba=False,
        supports_feature_importance=True,
        variants={
            REGRESSION: _Variant(
                cls=LinearRegression,
                params=(
                    _bool_param(
                        "fit_intercept",
                        "Fit an intercept",
                        True,
                        "Let the line start somewhere other than zero. Leave this on unless "
                        "your data is already centred.",
                    ),
                    _positive_param(),
                    _float_param(
                        "tol",
                        "Stopping tolerance",
                        1e-06,
                        1e-10,
                        0.1,
                        "How precise the numerical solution has to be before fitting stops.",
                        step=1e-06,
                        advanced=True,
                    ),
                ),
            )
        },
    ),
    _Algorithm(
        key="ridge_regression",
        label="Scikit-learn — Ridge Regression",
        family="Linear Models",
        description=(
            "Linear regression with a penalty that keeps every coefficient small. It is the "
            "safe choice when features overlap or repeat each other."
        ),
        supports_proba=False,
        supports_feature_importance=True,
        variants={
            REGRESSION: _Variant(
                cls=Ridge,
                params=(
                    _alpha_param(1.0),
                    _select_param(
                        "solver",
                        "Solver",
                        "auto",
                        [
                            ("auto", "Choose automatically"),
                            ("svd", "SVD"),
                            ("cholesky", "Cholesky"),
                            ("lsqr", "LSQR"),
                            ("sparse_cg", "Sparse CG"),
                            ("sag", "SAG"),
                            ("saga", "SAGA"),
                        ],
                        "The numerical routine used to solve the model. 'Choose automatically' "
                        "is almost always right.",
                        advanced=True,
                    ),
                    _linear_max_iter_param(None),
                    _tol_param(0.0001),
                    _fit_intercept_param(),
                    _positive_param(),
                ),
            )
        },
    ),
    _Algorithm(
        key="lasso_regression",
        label="Scikit-learn — Lasso Regression",
        family="Linear Models",
        description=(
            "Linear regression that pushes unhelpful coefficients all the way to zero, so "
            "you end up with a shorter, easier-to-read model."
        ),
        supports_proba=False,
        supports_feature_importance=True,
        variants={
            REGRESSION: _Variant(
                cls=Lasso,
                params=(
                    _alpha_param(1.0),
                    _linear_max_iter_param(1000, maximum=100000),
                    _tol_param(0.0001),
                    _selection_param(),
                    _fit_intercept_param(),
                    _positive_param(),
                ),
            )
        },
    ),
    _Algorithm(
        key="elastic_net",
        label="Scikit-learn — Elastic Net",
        family="Linear Models",
        description=(
            "A blend of ridge and lasso regression: it shrinks all coefficients and can still "
            "drop the useless ones entirely."
        ),
        supports_proba=False,
        supports_feature_importance=True,
        variants={
            REGRESSION: _Variant(
                cls=ElasticNet,
                params=(
                    _alpha_param(1.0),
                    _float_param(
                        "l1_ratio",
                        "Lasso share (L1 ratio)",
                        0.5,
                        0.0,
                        1.0,
                        "0 behaves like ridge regression, 1 like lasso, and values in between "
                        "mix the two.",
                        step=0.05,
                    ),
                    _linear_max_iter_param(1000, maximum=100000),
                    _tol_param(0.0001),
                    _selection_param(),
                    _fit_intercept_param(),
                    _positive_param(),
                ),
            )
        },
    ),
    _Algorithm(
        key="svm",
        label="Scikit-learn — Support Vector Machine",
        family="Support Vector Machines",
        description=(
            "Draws the widest possible boundary between the groups in your data. Strong on "
            "small and medium datasets, slow on very large ones."
        ),
        supports_proba=True,
        supports_feature_importance=False,
        variants={
            CLASSIFICATION: _Variant(
                cls=SVC,
                overrides={"probability": True},
                params=(
                    _float_param(
                        "C",
                        "Error penalty (C)",
                        1.0,
                        0.001,
                        1000.0,
                        "How heavily mistakes on the training rows are punished. Larger values "
                        "fit the training data harder and risk overfitting.",
                        step=0.01,
                    ),
                    _select_param(
                        "kernel",
                        "Kernel",
                        "rbf",
                        [
                            ("rbf", "RBF (curved)"),
                            ("linear", "Linear"),
                            ("poly", "Polynomial"),
                            ("sigmoid", "Sigmoid"),
                        ],
                        "The shape of boundary the model may draw. 'RBF' handles curved "
                        "boundaries, 'Linear' is the fastest.",
                    ),
                    _bool_param(
                        "probability",
                        "Estimate probabilities",
                        True,
                        "Also work out how confident the model is in each class. Turning this "
                        "off trains faster but hides probabilities, ROC AUC and log loss.",
                    ),
                    _class_weight_param(),
                    _select_param(
                        "gamma",
                        "Kernel reach (gamma)",
                        "scale",
                        [("scale", "Scaled to the data"), ("auto", "1 / number of features")],
                        "How far the influence of a single training row reaches when using a "
                        "curved kernel.",
                        advanced=True,
                    ),
                    _int_param(
                        "degree",
                        "Polynomial degree",
                        3,
                        2,
                        6,
                        "How curved the polynomial kernel is. Ignored by the other kernels.",
                        advanced=True,
                    ),
                    _tol_param(0.001),
                    _int_param(
                        "max_iter",
                        "Maximum solver steps",
                        -1,
                        -1,
                        1000000,
                        "A hard limit on solver work. -1 means keep going until the model "
                        "settles.",
                        step=100,
                        advanced=True,
                    ),
                ),
            ),
            REGRESSION: _Variant(
                cls=SVR,
                params=(
                    _float_param(
                        "C",
                        "Error penalty (C)",
                        1.0,
                        0.001,
                        1000.0,
                        "How heavily mistakes on the training rows are punished. Larger values "
                        "fit the training data harder and risk overfitting.",
                        step=0.01,
                    ),
                    _select_param(
                        "kernel",
                        "Kernel",
                        "rbf",
                        [
                            ("rbf", "RBF (curved)"),
                            ("linear", "Linear"),
                            ("poly", "Polynomial"),
                            ("sigmoid", "Sigmoid"),
                        ],
                        "The shape of curve the model may fit. 'RBF' handles curved "
                        "relationships, 'Linear' is the fastest.",
                    ),
                    _float_param(
                        "epsilon",
                        "Ignored error band",
                        0.1,
                        0.0,
                        10.0,
                        "Errors smaller than this are treated as no error at all, so the model "
                        "ignores tiny deviations.",
                        step=0.01,
                    ),
                    _select_param(
                        "gamma",
                        "Kernel reach (gamma)",
                        "scale",
                        [("scale", "Scaled to the data"), ("auto", "1 / number of features")],
                        "How far the influence of a single training row reaches when using a "
                        "curved kernel.",
                        advanced=True,
                    ),
                    _int_param(
                        "degree",
                        "Polynomial degree",
                        3,
                        2,
                        6,
                        "How curved the polynomial kernel is. Ignored by the other kernels.",
                        advanced=True,
                    ),
                    _tol_param(0.001),
                    _int_param(
                        "max_iter",
                        "Maximum solver steps",
                        -1,
                        -1,
                        1000000,
                        "A hard limit on solver work. -1 means keep going until the model "
                        "settles.",
                        step=100,
                        advanced=True,
                    ),
                ),
            ),
        },
    ),
    _Algorithm(
        key="sgd",
        label="Scikit-learn — SGD (Stochastic Gradient Descent)",
        family="Linear Models",
        description=(
            "Fits a linear model one small batch at a time, which makes it the fastest option "
            "on very large tables."
        ),
        supports_proba=True,
        supports_feature_importance=True,
        variants={
            CLASSIFICATION: _Variant(
                cls=SGDClassifier,
                overrides={"loss": "log_loss"},
                params=(
                    _select_param(
                        "loss",
                        "Loss function",
                        "log_loss",
                        [
                            ("log_loss", "Log loss (logistic regression)"),
                            ("hinge", "Hinge (linear SVM)"),
                            ("modified_huber", "Modified Huber"),
                            ("squared_hinge", "Squared hinge"),
                            ("perceptron", "Perceptron"),
                        ],
                        "What the model tries to minimise. 'Log loss' behaves like logistic "
                        "regression and is the only common choice that gives probabilities.",
                    ),
                    *_SGD_SHARED,
                    _class_weight_param(),
                    _select_param(
                        "learning_rate",
                        "Learning rate schedule",
                        "optimal",
                        [
                            ("optimal", "Optimal"),
                            ("constant", "Constant"),
                            ("invscaling", "Gradually decreasing"),
                            ("adaptive", "Adaptive"),
                        ],
                        "How the step size changes while training.",
                        advanced=True,
                    ),
                ),
            ),
            REGRESSION: _Variant(
                cls=SGDRegressor,
                params=(
                    _select_param(
                        "loss",
                        "Loss function",
                        "squared_error",
                        [
                            ("squared_error", "Squared error"),
                            ("huber", "Huber (outlier tolerant)"),
                            ("epsilon_insensitive", "Epsilon insensitive"),
                            ("squared_epsilon_insensitive", "Squared epsilon insensitive"),
                        ],
                        "What the model tries to minimise. 'Huber' is the safer pick when the "
                        "target has outliers.",
                    ),
                    *_SGD_SHARED,
                    _float_param(
                        "epsilon",
                        "Ignored error band",
                        0.1,
                        0.0,
                        10.0,
                        "The size of the error band the Huber and epsilon-insensitive losses "
                        "treat as no error.",
                        step=0.01,
                        advanced=True,
                    ),
                    _select_param(
                        "learning_rate",
                        "Learning rate schedule",
                        "invscaling",
                        [
                            ("invscaling", "Gradually decreasing"),
                            ("constant", "Constant"),
                            ("optimal", "Optimal"),
                            ("adaptive", "Adaptive"),
                        ],
                        "How the step size changes while training.",
                        advanced=True,
                    ),
                ),
            ),
        },
    ),
    _Algorithm(
        key="decision_tree",
        label="Scikit-learn — Decision Tree",
        family="Trees",
        description=(
            "A flowchart of yes/no questions learned from the data. Easy to explain to "
            "anyone, but it overfits unless you limit its size."
        ),
        supports_proba=True,
        supports_feature_importance=True,
        variants={
            CLASSIFICATION: _Variant(
                cls=DecisionTreeClassifier,
                params=(
                    _select_param(
                        "criterion",
                        "Split quality measure",
                        "gini",
                        [
                            ("gini", "Gini impurity"),
                            ("entropy", "Entropy (information gain)"),
                            ("log_loss", "Log loss"),
                        ],
                        "How the tree judges whether a split separated the classes well.",
                    ),
                    *_TREE_SHARED,
                    _tree_max_features_param("1.0"),
                    _class_weight_param(),
                ),
            ),
            REGRESSION: _Variant(
                cls=DecisionTreeRegressor,
                params=(
                    _select_param(
                        "criterion",
                        "Split quality measure",
                        "squared_error",
                        [
                            ("squared_error", "Squared error"),
                            ("absolute_error", "Absolute error"),
                        ],
                        "How the tree judges whether a split reduced the error. 'Absolute "
                        "error' copes better with outliers but is slower.",
                    ),
                    *_TREE_SHARED,
                    _tree_max_features_param("1.0"),
                ),
            ),
        },
    ),
    _Algorithm(
        key="random_forest",
        label="Scikit-learn — Random Forest",
        family="Ensembles",
        description=(
            "Hundreds of different decision trees voting together. It is accurate out of the "
            "box and very hard to break, which makes it a great default."
        ),
        supports_proba=True,
        supports_feature_importance=True,
        variants={
            CLASSIFICATION: _Variant(
                cls=RandomForestClassifier,
                params=(
                    _n_estimators_param(
                        100, "How many trees vote. More trees are steadier but slower."
                    ),
                    _select_param(
                        "criterion",
                        "Split quality measure",
                        "gini",
                        [
                            ("gini", "Gini impurity"),
                            ("entropy", "Entropy (information gain)"),
                            ("log_loss", "Log loss"),
                        ],
                        "How each tree judges whether a split separated the classes well.",
                    ),
                    *_FOREST_SHARED,
                    _tree_max_features_param("sqrt"),
                    _class_weight_param(subsample=True),
                    _bool_param(
                        "bootstrap",
                        "Sample rows with replacement",
                        True,
                        "Give each tree its own random sample of the rows so the trees differ "
                        "from one another.",
                        advanced=True,
                    ),
                ),
            ),
            REGRESSION: _Variant(
                cls=RandomForestRegressor,
                params=(
                    _n_estimators_param(
                        100, "How many trees are averaged. More trees are steadier but slower."
                    ),
                    _select_param(
                        "criterion",
                        "Split quality measure",
                        "squared_error",
                        [
                            ("squared_error", "Squared error"),
                            ("absolute_error", "Absolute error"),
                        ],
                        "How each tree judges whether a split reduced the error. 'Absolute "
                        "error' copes better with outliers but is slower.",
                    ),
                    *_FOREST_SHARED,
                    _tree_max_features_param("1.0"),
                    _bool_param(
                        "bootstrap",
                        "Sample rows with replacement",
                        True,
                        "Give each tree its own random sample of the rows so the trees differ "
                        "from one another.",
                        advanced=True,
                    ),
                ),
            ),
        },
    ),
    _Algorithm(
        key="extra_trees",
        label="Scikit-learn — Extra Trees",
        family="Ensembles",
        description=(
            "Like a random forest, but each split point is picked at random. That extra "
            "randomness trains faster and often generalizes just as well."
        ),
        supports_proba=True,
        supports_feature_importance=True,
        variants={
            CLASSIFICATION: _Variant(
                cls=ExtraTreesClassifier,
                params=(
                    _n_estimators_param(
                        100, "How many trees vote. More trees are steadier but slower."
                    ),
                    _select_param(
                        "criterion",
                        "Split quality measure",
                        "gini",
                        [
                            ("gini", "Gini impurity"),
                            ("entropy", "Entropy (information gain)"),
                            ("log_loss", "Log loss"),
                        ],
                        "How each tree judges whether a split separated the classes well.",
                    ),
                    *_FOREST_SHARED,
                    _tree_max_features_param("sqrt"),
                    _class_weight_param(subsample=True),
                    _bool_param(
                        "bootstrap",
                        "Sample rows with replacement",
                        False,
                        "Give each tree its own random sample of the rows. Extra Trees "
                        "normally uses all rows instead.",
                        advanced=True,
                    ),
                ),
            ),
            REGRESSION: _Variant(
                cls=ExtraTreesRegressor,
                params=(
                    _n_estimators_param(
                        100, "How many trees are averaged. More trees are steadier but slower."
                    ),
                    _select_param(
                        "criterion",
                        "Split quality measure",
                        "squared_error",
                        [
                            ("squared_error", "Squared error"),
                            ("absolute_error", "Absolute error"),
                        ],
                        "How each tree judges whether a split reduced the error. 'Absolute "
                        "error' copes better with outliers but is slower.",
                    ),
                    *_FOREST_SHARED,
                    _tree_max_features_param("1.0"),
                    _bool_param(
                        "bootstrap",
                        "Sample rows with replacement",
                        False,
                        "Give each tree its own random sample of the rows. Extra Trees "
                        "normally uses all rows instead.",
                        advanced=True,
                    ),
                ),
            ),
        },
    ),
    _Algorithm(
        key="gradient_boosting",
        label="Scikit-learn — Gradient Boosting",
        family="Ensembles",
        description=(
            "Builds small trees one after another, each one fixing the mistakes of the last. "
            "Often the most accurate model on tabular data, but slower to train."
        ),
        supports_proba=True,
        supports_feature_importance=True,
        variants={
            CLASSIFICATION: _Variant(cls=GradientBoostingClassifier, params=_GB_SHARED),
            REGRESSION: _Variant(
                cls=GradientBoostingRegressor,
                params=(
                    *_GB_SHARED,
                    _select_param(
                        "loss",
                        "Loss function",
                        "squared_error",
                        [
                            ("squared_error", "Squared error"),
                            ("absolute_error", "Absolute error"),
                            ("huber", "Huber (outlier tolerant)"),
                            ("quantile", "Quantile"),
                        ],
                        "What the model tries to minimise. 'Huber' and 'Absolute error' cope "
                        "better with outliers in the target.",
                        advanced=True,
                    ),
                ),
            ),
        },
    ),
    _Algorithm(
        key="hist_gradient_boosting",
        label="Scikit-learn — Histogram Gradient Boosting",
        family="Ensembles",
        description=(
            "A much faster gradient boosting implementation that buckets numeric features "
            "first. The best starting point for datasets with many rows."
        ),
        supports_proba=True,
        # Histogram gradient boosting exposes neither feature_importances_ nor
        # coef_ (verified on scikit-learn 1.9); importances need a permutation
        # run, which the trainer does not do.
        supports_feature_importance=False,
        variants={
            CLASSIFICATION: _Variant(
                cls=HistGradientBoostingClassifier,
                params=(*_HGB_SHARED, _class_weight_param()),
            ),
            REGRESSION: _Variant(cls=HistGradientBoostingRegressor, params=_HGB_SHARED),
        },
    ),
    _Algorithm(
        key="xgboost",
        label="XGBoost",
        family="Ensembles",
        description=(
            "Gradient boosting from the XGBoost library rather than scikit-learn. Level-wise "
            "trees with strong regularization; often the first thing to reach for on "
            "tabular data."
        ),
        supports_proba=True,
        supports_feature_importance=True,
        variants={
            # verbosity=0 keeps XGBoost's own warnings out of the server log;
            # anything the user needs to know is reported through the run's
            # warnings list instead.
            CLASSIFICATION: _Variant(
                cls=XGBClassifier,
                params=_XGB_SHARED,
                overrides={"verbosity": 0},
            ),
            # XGBClassifier has no `class_weight`; it balances with
            # `scale_pos_weight`, which only applies to two-class problems, so
            # no weighting control is offered rather than one that half works.
            REGRESSION: _Variant(
                cls=XGBRegressor,
                params=_XGB_SHARED,
                overrides={"verbosity": 0},
            ),
        },
    ),
    _Algorithm(
        key="lightgbm",
        label="LightGBM",
        family="Ensembles",
        description=(
            "Gradient boosting from the LightGBM library. Grows trees leaf-by-leaf instead "
            "of level-by-level, which is what makes it fast on wide datasets with many rows."
        ),
        supports_proba=True,
        supports_feature_importance=True,
        variants={
            # verbose=-1 silences the per-fit "[LightGBM] [Info] ..." banner,
            # which otherwise lands in the gunicorn log on every training run
            # and on every cross-validation fold.
            CLASSIFICATION: _Variant(
                cls=LGBMClassifier,
                params=(*_LGBM_SHARED, _class_weight_param()),
                overrides={"verbose": -1},
            ),
            REGRESSION: _Variant(
                cls=LGBMRegressor,
                params=_LGBM_SHARED,
                overrides={"verbose": -1},
            ),
        },
    ),
    _Algorithm(
        key="ada_boost",
        label="Scikit-learn — AdaBoost",
        family="Ensembles",
        description=(
            "Trains a sequence of very simple trees, giving more weight to the rows the "
            "previous ones got wrong."
        ),
        supports_proba=True,
        supports_feature_importance=True,
        variants={
            CLASSIFICATION: _Variant(
                cls=AdaBoostClassifier,
                params=(
                    _n_estimators_param(
                        50, "How many simple trees are trained one after another."
                    ),
                    _float_param(
                        "learning_rate",
                        "Learning rate",
                        1.0,
                        0.01,
                        10.0,
                        "How much each new tree is allowed to change the result. Lower values "
                        "need more trees.",
                        step=0.01,
                    ),
                ),
            ),
            REGRESSION: _Variant(
                cls=AdaBoostRegressor,
                params=(
                    _n_estimators_param(
                        50, "How many simple trees are trained one after another."
                    ),
                    _float_param(
                        "learning_rate",
                        "Learning rate",
                        1.0,
                        0.01,
                        10.0,
                        "How much each new tree is allowed to change the result. Lower values "
                        "need more trees.",
                        step=0.01,
                    ),
                    _select_param(
                        "loss",
                        "Loss function",
                        "linear",
                        [
                            ("linear", "Linear"),
                            ("square", "Square"),
                            ("exponential", "Exponential"),
                        ],
                        "How strongly big mistakes are punished when the rows are re-weighted.",
                        advanced=True,
                    ),
                ),
            ),
        },
    ),
    _Algorithm(
        key="knn",
        label="Scikit-learn — K-Nearest Neighbours",
        family="Nearest Neighbours",
        description=(
            "Predicts each row by looking at the most similar rows in the training data. "
            "Simple and surprisingly strong when the features are on comparable scales."
        ),
        supports_proba=True,
        supports_feature_importance=False,
        variants={
            CLASSIFICATION: _Variant(cls=KNeighborsClassifier, params=_KNN_SHARED),
            REGRESSION: _Variant(cls=KNeighborsRegressor, params=_KNN_SHARED),
        },
    ),
    _Algorithm(
        key="naive_bayes",
        label="Scikit-learn — Gaussian Naive Bayes",
        family="Naive Bayes",
        description=(
            "Assumes every feature tells its own independent story. It trains almost "
            "instantly and is a solid reference point, even when that assumption is wrong."
        ),
        supports_proba=True,
        supports_feature_importance=False,
        variants={
            CLASSIFICATION: _Variant(
                cls=GaussianNB,
                params=(
                    _float_param(
                        "var_smoothing",
                        "Variance smoothing",
                        1e-09,
                        1e-12,
                        1.0,
                        "A tiny amount added to each feature's spread to keep the maths "
                        "stable. Raise it if the model looks over-confident.",
                        step=1e-09,
                    ),
                ),
            )
        },
    ),
    _Algorithm(
        key="mlp",
        label="Scikit-learn — Neural Network (MLP)",
        family="Neural Networks",
        description=(
            "A small neural network of fully connected layers. It can learn complex patterns "
            "but needs scaled features and more training time than the other models."
        ),
        supports_proba=True,
        supports_feature_importance=False,
        variants={
            CLASSIFICATION: _Variant(cls=MLPClassifier, params=_MLP_SHARED),
            REGRESSION: _Variant(cls=MLPRegressor, params=_MLP_SHARED),
        },
    ),
)

_BY_KEY: dict[str, _Algorithm] = {algorithm.key: algorithm for algorithm in _ALGORITHMS}

#: Every algorithm key in the registry, sorted alphabetically.
ALGORITHM_KEYS: tuple[str, ...] = tuple(sorted(_BY_KEY))


# --------------------------------------------------------------------------
# Value coercion helpers
# --------------------------------------------------------------------------


def _normalize_task(task: Any) -> str:
    """Lower-case and trim a task name without validating it."""
    return str(task).strip().lower() if task is not None else ""


def _lookup(key: Any) -> _Algorithm:
    """Return the algorithm for `key` or raise AlgorithmNotFound."""
    name = str(key).strip().lower() if key is not None else ""
    algorithm = _BY_KEY.get(name)
    if algorithm is None:
        raise AlgorithmNotFound(
            f"'{key}' is not an available algorithm. Choose one of: "
            f"{', '.join(ALGORITHM_KEYS)}."
        )
    return algorithm


def _format_number(value: Any) -> str:
    """Render a number for a user-facing message without exponent noise."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e16:
            return str(int(value))
        return f"{value:g}"
    return str(value)


def _is_blank(value: Any) -> bool:
    """True when the value means "leave this setting at its default"."""
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, tuple, dict, set)) and len(value) == 0:
        return True
    return False


def _to_number(value: Any) -> float | None:
    """Best-effort conversion to a finite float, or None when impossible."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            number = float(value.strip())
        except (TypeError, ValueError):
            return None
    else:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _to_bool(value: Any) -> bool | None:
    """Best-effort conversion to a bool, or None when impossible."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "t", "yes", "y", "on", "1"):
            return True
        if text in ("false", "f", "no", "n", "off", "0"):
            return False
    return None


def _stringify(value: Any) -> str:
    """Render a value the way select option values are written."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(_stringify(item) for item in value)
    return str(value).strip()


def _coerce_numeric(param: _Param, value: Any) -> tuple[Any, list[str]]:
    """Coerce, round and clamp a numeric hyperparameter."""
    warnings: list[str] = []
    number = _to_number(value)
    if number is None:
        warnings.append(
            f"'{_stringify(value)}' is not a valid number for '{param.label}', "
            "so the default was used."
        )
        return _OMIT, warnings
    if param.type == "int":
        rounded = int(round(number))
        if rounded != number:
            warnings.append(
                f"'{param.label}' has to be a whole number, so {_format_number(number)} "
                f"was rounded to {rounded}."
            )
        number = rounded
    else:
        number = float(number)
    low, high = param.minimum, param.maximum
    if low is not None and number < low:
        clamped = int(low) if param.type == "int" else float(low)
        warnings.append(
            f"'{param.label}' has to be between {_format_number(low)} and "
            f"{_format_number(high)}, so {_format_number(number)} was raised to "
            f"{_format_number(clamped)}."
        )
        number = clamped
    elif high is not None and number > high:
        clamped = int(high) if param.type == "int" else float(high)
        warnings.append(
            f"'{param.label}' has to be between {_format_number(low)} and "
            f"{_format_number(high)}, so {_format_number(number)} was lowered to "
            f"{_format_number(clamped)}."
        )
        number = clamped
    return number, warnings


def _coerce_select(param: _Param, value: Any) -> tuple[Any, list[str]]:
    """Match a value against the option list, or parse it when free text."""
    warnings: list[str] = []
    text = _stringify(value)
    if not text:
        return _OMIT, warnings
    options = param.options or ()
    matches = {option.lower(): option for option, _ in options}
    match = matches.get(text.lower())
    if match is not None:
        return param.cast(match)
    if param.free_text:
        try:
            return param.cast(text)
        except ValueError as exc:
            warnings.append(f"{exc} The default for '{param.label}' was used instead.")
            return _OMIT, warnings
    if text.lower() in ("auto", "default", "none", "null"):
        return _OMIT, warnings
    choices = ", ".join(f"'{option}'" for option, _ in options)
    warnings.append(
        f"'{text}' is not one of the choices for '{param.label}', so the default was used. "
        f"Valid choices are {choices}."
    )
    return _OMIT, warnings


def _coerce_value(param: _Param, value: Any) -> tuple[Any, list[str]]:
    """Coerce one user-supplied value for one hyperparameter."""
    if _is_blank(value) and not (param.type == "select" and isinstance(value, (list, tuple))):
        return _OMIT, []
    if param.type == "bool":
        coerced = _to_bool(value)
        if coerced is None:
            return _OMIT, [
                f"'{_stringify(value)}' is not a yes/no value for '{param.label}', "
                "so the default was used."
            ]
        return coerced, []
    if param.type == "select":
        return _coerce_select(param, value)
    return _coerce_numeric(param, value)


def _params_for(algorithm: _Algorithm, task: str) -> tuple[_Param, ...]:
    """Parameters for one task, falling back to the union of all tasks."""
    variant = algorithm.variants.get(task)
    if variant is not None:
        return variant.params
    return _union_params(algorithm)


def _union_params(algorithm: _Algorithm) -> tuple[_Param, ...]:
    """Every parameter across all supported tasks, classification first."""
    merged: dict[str, _Param] = {}
    for task in TASK_TYPES:
        variant = algorithm.variants.get(task)
        if variant is None:
            continue
        for param in variant.params:
            merged.setdefault(param.name, param)
    return tuple(merged.values())


def _defined_for_other_task(algorithm: _Algorithm, name: str, task: str) -> str | None:
    """The other task that does define `name`, when this one does not."""
    for other, variant in algorithm.variants.items():
        if other == task:
            continue
        if any(param.name == name for param in variant.params):
            return other
    return None


def _apply_compatibility(key: str, clean: dict[str, Any]) -> list[str]:
    """Fix combinations of settings scikit-learn would reject outright."""
    warnings: list[str] = []
    if key == "logistic_regression":
        l1_ratio = clean.get("l1_ratio")
        solver = clean.get("solver", "lbfgs")
        if isinstance(l1_ratio, (int, float)) and l1_ratio > 0 and solver != "saga":
            clean["solver"] = "saga"
            warnings.append(
                f"The '{solver}' solver cannot apply L1 regularization, so the solver "
                "was changed to 'saga'."
            )
    elif key == "ridge_regression":
        if clean.get("positive") is True and clean.get("solver", "auto") not in ("auto", "lbfgs"):
            solver = clean["solver"]
            clean["solver"] = "auto"
            warnings.append(
                f"The '{solver}' solver cannot force positive coefficients, so the solver "
                "was set back to 'Choose automatically'."
            )
    elif key == "knn":
        if clean.get("metric") == "cosine" and clean.get("algorithm") in ("kd_tree", "ball_tree"):
            clean["algorithm"] = "brute"
            warnings.append(
                "The cosine distance only works with the brute-force neighbour search, so "
                "that search method was used."
            )
    return warnings


@lru_cache(maxsize=None)
def _accepted_params(cls: type) -> frozenset[str]:
    """Keyword arguments the estimator class actually accepts.

    Reading the signature is enough for every scikit-learn estimator, which
    names each hyperparameter explicitly. It is *not* enough for XGBoost:
    `XGBClassifier.__init__` is `(self, *, objective=..., **kwargs)`, so the
    signature advertises exactly one name and the other forty arrive through
    `**kwargs`. Filtering on the signature alone would silently drop every
    hyperparameter the user set, along with `random_state` -- the run would
    still succeed, just unconfigured and unreproducible, with nothing said.

    So when the constructor takes `**kwargs`, the signature is treated as
    incomplete and `get_params()` -- the scikit-learn estimator contract, which
    XGBoost does implement fully -- is folded in.
    """
    names: set[str] = set()
    takes_var_keyword = False
    try:
        signature = inspect.signature(cls)
        for name, parameter in signature.parameters.items():
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                names.add(name)
            elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
                takes_var_keyword = True
    except (TypeError, ValueError):  # pragma: no cover - defensive
        takes_var_keyword = True

    if names and not takes_var_keyword:
        return frozenset(names)

    try:
        names |= set(cls().get_params())
    except Exception:  # pragma: no cover - defensive
        pass
    return frozenset(names)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def _spec_for(algorithm: _Algorithm, task: str | None = None) -> dict[str, Any]:
    """Render an algorithm as a contract `AlgorithmSpec` dict.

    When `task` names one of the tasks the algorithm supports, the published
    hyperparameter form is the one that actually applies to that task, so the
    defaults and options shown are the ones the estimator really uses. Without
    a task the form is the union across tasks (classification wins a clash).
    """
    params = _params_for(algorithm, task) if task else _union_params(algorithm)
    return {
        "key": algorithm.key,
        "label": algorithm.label,
        "family": algorithm.family,
        "tasks": [task for task in TASK_TYPES if task in algorithm.variants],
        "description": algorithm.description,
        "supports_proba": algorithm.supports_proba,
        "supports_feature_importance": algorithm.supports_feature_importance,
        "hyperparameters": [param.to_spec() for param in params],
    }


def list_algorithms(task: str | None = None) -> list[dict]:
    """Return every algorithm spec, optionally filtered to one task type.

    Specs are sorted by family and then by label. An unrecognised task name
    simply matches nothing and returns an empty list.
    """
    wanted = _normalize_task(task) or None
    specs = [
        _spec_for(algorithm, wanted)
        for algorithm in _ALGORITHMS
        if wanted is None or wanted in algorithm.variants
    ]
    specs.sort(key=lambda spec: (spec["family"], spec["label"]))
    return specs


def get_spec(key: str, task: str | None = None) -> dict:
    """Return one algorithm spec.

    Without a task the hyperparameter list is the union across the tasks the
    algorithm supports; where classification and regression define the same
    setting differently, the classification version is shown. Pass a task to
    get exactly the settings that apply to that task.

    Raises:
        AlgorithmNotFound: when the key is not in the registry.
    """
    algorithm = _lookup(key)
    wanted = _normalize_task(task) or None
    if wanted is not None and wanted not in algorithm.variants:
        wanted = None
    return _spec_for(algorithm, wanted)


def supports(key: str, task: str) -> bool:
    """True when the algorithm can be used for that task type."""
    try:
        algorithm = _lookup(key)
    except AlgorithmNotFound:
        return False
    return _normalize_task(task) in algorithm.variants


def coerce_hyperparameters(key: str, task: str, raw: dict) -> tuple[dict, list[str]]:
    """Turn user-supplied hyperparameters into safe scikit-learn keyword arguments.

    Unknown settings are dropped with a warning, numbers outside the allowed
    range are clamped with a warning, and blank values fall back to the
    estimator default by being left out entirely. Values are returned ready to
    pass to the estimator (for example `hidden_layer_sizes` becomes a tuple).
    This function never raises on bad values; it returns
    `(clean_kwargs, warnings)`.

    Raises:
        AlgorithmNotFound: when the key is not in the registry.
    """
    algorithm = _lookup(key)
    task_name = _normalize_task(task)
    warnings: list[str] = []
    if task_name not in algorithm.variants:
        supported = " or ".join(task for task in TASK_TYPES if task in algorithm.variants)
        warnings.append(
            f"{algorithm.short_label} does not support {task_name or 'that'} problems; "
            f"it can only be used for {supported}."
        )
    params = _params_for(algorithm, task_name)
    by_name = {param.name: param for param in params}

    if raw is None:
        return {}, warnings
    if not isinstance(raw, Mapping):
        warnings.append(
            "Model settings have to be sent as a set of name and value pairs, so the "
            "defaults were used instead."
        )
        return {}, warnings

    clean: dict[str, Any] = {}
    for raw_name, value in raw.items():
        name = str(raw_name).strip()
        param = by_name.get(name)
        if param is None:
            other = _defined_for_other_task(algorithm, name, task_name)
            if other is not None:
                warnings.append(
                    f"'{name}' does not apply to {task_name} models, so it was ignored."
                )
            else:
                warnings.append(
                    f"'{name}' is not a setting for {algorithm.short_label}, so it was ignored."
                )
            continue
        coerced, messages = _coerce_value(param, value)
        warnings.extend(messages)
        if isinstance(coerced, _Omit):
            continue
        clean[param.name] = coerced

    warnings.extend(_apply_compatibility(algorithm.key, clean))
    return clean, warnings


def build_estimator(
    key: str,
    task: str,
    hyperparameters: Mapping[str, Any] | None = None,
    random_state: int | None = None,
) -> BaseEstimator:
    """Create the unfitted estimator for an algorithm and task.

    User hyperparameters are coerced first, then layered over the registry's own
    defaults, then filtered down to the keyword arguments this estimator class
    accepts, so settings that only apply to the other task are dropped rather
    than raising. `random_state` is passed only when the estimator accepts it
    and is not None, and `n_jobs=-1` only when the estimator accepts it and
    still acts on it.

    Raises:
        AlgorithmNotFound: when the key is not in the registry.
        ValueError: when the algorithm does not support the task type.
    """
    algorithm = _lookup(key)
    task_name = _normalize_task(task)
    if task_name not in TASK_TYPES:
        raise ValueError(
            f"'{task}' is not a supported problem type. Use 'classification' or 'regression'."
        )
    variant = algorithm.variants.get(task_name)
    if variant is None:
        supported = " or ".join(item for item in TASK_TYPES if item in algorithm.variants)
        raise ValueError(
            f"{algorithm.short_label} cannot be used for {task_name} problems; "
            f"it only supports {supported}."
        )

    clean, _ = coerce_hyperparameters(algorithm.key, task_name, dict(hyperparameters or {}))
    merged: dict[str, Any] = {**variant.overrides, **clean}

    # The registry's own overrides are always accepted. The filter exists to
    # drop *user* settings meant for the other task, not to second-guess this
    # module: LightGBM's `verbose` is a passthrough keyword absent from its
    # `get_params()`, so filtering it out would leave every fit -- and every
    # cross-validation fold -- printing an "[LightGBM] [Info] ..." banner into
    # the server log.
    accepted = _accepted_params(variant.cls) | set(variant.overrides)
    kwargs = {name: value for name, value in merged.items() if name in accepted}
    if "random_state" in accepted and random_state is not None:
        try:
            kwargs.setdefault("random_state", int(random_state))
        except (TypeError, ValueError):
            pass
    if "n_jobs" in accepted and variant.cls.__name__ not in _IGNORES_N_JOBS:
        kwargs.setdefault("n_jobs", -1)

    try:
        return variant.cls(**kwargs)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"{algorithm.short_label} could not be set up with those settings: {exc}"
        ) from exc

"""Column profiling and task-type inference.

Turns a pandas DataFrame into the ``ColumnProfile`` dictionaries the API
returns, and implements the task-type inference rule the trainer relies on.
Every dictionary produced here is already JSON-safe.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from .io import json_safe

ColumnKind = Literal["numeric", "categorical", "boolean", "datetime", "text"]
TaskType = Literal["classification", "regression"]

#: How many distinct values are echoed back as ``sample_values``.
MAX_SAMPLE_VALUES: int = 5
#: Maximum length of any stringified sample / top value.
MAX_VALUE_CHARS: int = 60
#: How many entries ``top_values`` holds.
MAX_TOP_VALUES: int = 10
#: Kinds that get a ``top_values`` list instead of ``null``.
TOP_VALUE_KINDS: tuple[str, ...] = ("categorical", "boolean", "text")

_BOOLEAN_TOKEN_SETS: tuple[set[str], ...] = ({"0", "1"}, {"true", "false"})


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _as_series(series: Any) -> pd.Series:
    """Coerce ``series`` into a pandas Series without copying when possible."""
    if isinstance(series, pd.Series):
        return series
    if isinstance(series, pd.DataFrame):
        if series.shape[1] == 0:
            return pd.Series([], dtype="object")
        return series.iloc[:, 0]
    if isinstance(series, pd.Index):
        return pd.Series(series, name=series.name)
    return pd.Series(series)


def _unique_values(non_null: pd.Series) -> list[Any]:
    """Distinct values of an already null-free series, in order of appearance."""
    try:
        return list(pd.unique(non_null))
    except (TypeError, ValueError):
        try:
            return list(dict.fromkeys(non_null.tolist()))
        except TypeError:
            return [repr(value) for value in non_null.tolist()]


def _normalise_token(value: Any) -> str:
    """Lower-case text form of a value, used only for boolean detection."""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
        return str(number)
    return str(value).strip().lower()


def _looks_boolean(uniques: list[Any]) -> bool:
    """True when the two distinct values are 0/1 or true/false."""
    try:
        tokens = {_normalise_token(value) for value in uniques}
    except TypeError:
        return False
    return any(tokens == candidate for candidate in _BOOLEAN_TOKEN_SETS)


def _is_categorical(series: pd.Series) -> bool:
    return isinstance(series.dtype, pd.CategoricalDtype)


def _kind_from(series: pd.Series, non_null: pd.Series, uniques: list[Any]) -> ColumnKind:
    """Shared implementation of :func:`column_kind` with values already computed."""
    if ptypes.is_bool_dtype(series):
        return "boolean"
    if len(uniques) == 2 and _looks_boolean(uniques):
        return "boolean"
    if ptypes.is_datetime64_any_dtype(series):
        return "datetime"
    if ptypes.is_numeric_dtype(series):
        return "numeric"
    if ptypes.is_string_dtype(series) and not _is_categorical(series):
        if len(uniques) > max(50, 0.5 * len(non_null)):
            return "text"
    return "categorical"


def _display(value: Any) -> str:
    """Stringify a value for display, truncated to :data:`MAX_VALUE_CHARS`."""
    safe = json_safe(value)
    if safe is None:
        return ""
    text = safe if isinstance(safe, str) else str(safe)
    return text[:MAX_VALUE_CHARS]


def _numeric_stats(non_null: pd.Series) -> dict[str, float | None]:
    """min/max/mean/std of a numeric column; every entry is None when unknown."""
    empty: dict[str, float | None] = {"min": None, "max": None, "mean": None, "std": None}
    if len(non_null) == 0:
        return empty
    try:
        values = np.asarray(non_null.to_numpy(dtype="float64", na_value=np.nan), dtype="float64")
    except (TypeError, ValueError):
        return empty
    values = values[np.isfinite(values)]
    if values.size == 0:
        return empty
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else None,
    }


def _top_values(non_null: pd.Series) -> list[dict[str, Any]]:
    """The most common values of a column, most frequent first."""
    try:
        counts = non_null.value_counts(dropna=True).head(MAX_TOP_VALUES)
    except (TypeError, ValueError):
        return []
    return [
        {"value": _display(value), "count": int(count)}
        for value, count in counts.items()
    ]


def _all_integral(non_null: pd.Series) -> bool:
    """True when every non-null value is a whole number."""
    if len(non_null) == 0:
        return False
    if ptypes.is_integer_dtype(non_null) or ptypes.is_bool_dtype(non_null):
        return True
    try:
        values = np.asarray(non_null.to_numpy(dtype="float64", na_value=np.nan), dtype="float64")
    except (TypeError, ValueError):
        return False
    if values.size == 0 or not bool(np.isfinite(values).all()):
        return False
    return bool(np.all(values == np.floor(values)))


def _suitable(non_null: pd.Series, uniques: list[Any], kind: str) -> bool:
    """Shared implementation of :func:`suitable_as_target`."""
    n_non_null = len(non_null)
    if n_non_null == 0:
        return False
    n_unique = len(uniques)
    if n_unique < 2:
        return False
    # An id-like column: free text where every single value is different.
    if kind == "text" and n_unique == n_non_null:
        return False
    return True


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def column_kind(series: Any) -> ColumnKind:
    """Classify a column as numeric, categorical, boolean, datetime or text.

    boolean -> bool dtype, or exactly two distinct values in {0,1}/{true,false};
    datetime -> a datetime64 dtype; numeric -> any numeric dtype;
    text -> a string column with more than ``max(50, 0.5 * non-null)`` distinct
    values; everything else is categorical.
    """
    column = _as_series(series)
    non_null = column.dropna()
    return _kind_from(column, non_null, _unique_values(non_null))


def infer_task(series: Any) -> TaskType:
    """Infer whether a column should be modelled as classification or regression.

    Booleans, strings and categoricals are classification; datetimes are
    regression. A numeric column is classification when every value is a whole
    number and it has few enough distinct values, or when it has at most two
    distinct values; otherwise it is regression.
    """
    column = _as_series(series)

    if ptypes.is_bool_dtype(column):
        return "classification"
    if _is_categorical(column) or ptypes.is_string_dtype(column):
        return "classification"
    if ptypes.is_datetime64_any_dtype(column):
        return "regression"

    if ptypes.is_numeric_dtype(column):
        non_null = column.dropna()
        n_rows = len(column)
        try:
            n_unique = int(non_null.nunique())
        except (TypeError, ValueError):
            n_unique = len(_unique_values(non_null))
        if _all_integral(non_null) and n_unique <= max(20, 0.05 * n_rows) and n_unique <= 50:
            return "classification"
        if n_unique <= 2:
            return "classification"
        return "regression"

    # Anything else (object, timedelta, mixed) is treated as a label column.
    return "classification"


def suitable_as_target(series: Any) -> bool:
    """True when a column can reasonably be used as a training target.

    False for all-null columns, columns with fewer than two distinct values,
    and id-like text columns where every non-null value is distinct.
    """
    column = _as_series(series)
    non_null = column.dropna()
    uniques = _unique_values(non_null)
    return _suitable(non_null, uniques, _kind_from(column, non_null, uniques))


def profile_column(series: Any) -> dict:
    """Build the ColumnProfile dictionary for one column. Always JSON-safe."""
    column = _as_series(series)
    non_null = column.dropna()
    uniques = _unique_values(non_null)
    kind = _kind_from(column, non_null, uniques)

    n_rows = len(column)
    missing = n_rows - len(non_null)
    missing_pct = round(missing / n_rows * 100, 2) if n_rows else 0.0

    profile: dict[str, Any] = {
        "name": "" if column.name is None else str(column.name),
        "dtype": str(column.dtype),
        "kind": kind,
        "missing": int(missing),
        "missing_pct": missing_pct,
        "unique": int(len(uniques)),
        "sample_values": [_display(value) for value in uniques[:MAX_SAMPLE_VALUES]],
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "top_values": None,
        "suitable_as_target": False,
        "suggested_task": None,
    }

    if kind == "numeric":
        profile.update(_numeric_stats(non_null))
    if kind in TOP_VALUE_KINDS:
        profile["top_values"] = _top_values(non_null)

    usable = _suitable(non_null, uniques, kind)
    profile["suitable_as_target"] = usable
    profile["suggested_task"] = infer_task(column) if usable else None

    return json_safe(profile)


def profile_dataframe(df: Any) -> list[dict]:
    """Profile every column of a DataFrame, left to right."""
    if df is None or not hasattr(df, "shape"):
        return []
    if isinstance(df, pd.Series):
        return [profile_column(df)]
    # Positional access so duplicate column names still yield one profile each.
    return [profile_column(df.iloc[:, index]) for index in range(df.shape[1])]

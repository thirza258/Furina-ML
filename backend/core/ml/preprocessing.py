"""Dataset preprocessing steps for Furina ML.

A preprocessing request is a list of steps, each one ``{"op", "columns",
"params"}``.  Steps are applied in order to a copy of the incoming dataframe;
the original is never mutated.  A step that cannot be applied to the data it is
pointed at is recorded as ``skipped`` together with a plain-English reason
instead of raising, so a long recipe never fails half-way.  Only an unknown
``op`` is a hard error.
"""

from __future__ import annotations

import math
import re
import string
import warnings as warnings_module
from typing import Any, Callable, NamedTuple

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


class PreprocessError(Exception):
    """Raised when a preprocessing request itself is unusable.

    ``str(exc)`` is shown to the user, so the message must be a plain sentence.
    """


#: Supported operations mapped to their human readable label.
STEP_OPS: dict[str, str] = {
    "remove_duplicates": "Remove duplicate rows",
    "drop_columns": "Drop columns",
    "drop_missing_rows": "Drop rows with missing values",
    "fill_mean": "Fill missing values with the mean",
    "fill_median": "Fill missing values with the median",
    "fill_mode": "Fill missing values with the most frequent value",
    "fill_constant": "Fill missing values with a constant",
    "fill_empty_string": "Fill missing text with an empty string",
    "strip_whitespace": "Trim surrounding whitespace",
    "lowercase": "Convert text to lowercase",
    "remove_punctuation": "Remove punctuation from text",
    "remove_outliers_iqr": "Remove outliers (1.5x IQR)",
    "remove_outliers_zscore": "Remove outliers (z-score)",
    "minmax_scale": "Scale to the 0-1 range",
    "standard_scale": "Standardize (mean 0, standard deviation 1)",
    "robust_scale": "Robust scale (median and IQR)",
    "one_hot_encode": "One-hot encode",
    "label_encode": "Label encode",
    "pca": "Principal component analysis",
    "select_k_best": "Select the k best features",
    "lda": "Linear discriminant analysis",
}

#: A single source column may not explode into more than this many dummies.
MAX_ONE_HOT_COLUMNS: int = 50
DEFAULT_ZSCORE_THRESHOLD: float = 3.0
IQR_MULTIPLIER: float = 1.5

_PUNCTUATION_RE = re.compile("[%s]" % re.escape(string.punctuation))


class _StepOutcome(NamedTuple):
    """What a single step handler produced."""

    df: pd.DataFrame
    columns: list[str]
    message: str
    skipped: bool


# ---------------------------------------------------------------------------
# small formatting / dtype helpers
# ---------------------------------------------------------------------------


def _format_number(value: Any) -> str:
    """Format a number with four significant digits and no exponent."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    if number == 0:
        return "0"
    magnitude = math.floor(math.log10(abs(number)))
    if -5 <= magnitude < 15:
        decimals = max(0, 3 - magnitude)
        text = f"{number:,.{decimals}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text
    return f"{number:.4g}"


def _describe_value(value: Any) -> str:
    """Render a fill value for a message."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return _format_number(value)
    return str(value)


def _quote(names: list[str]) -> str:
    """Render column names as a readable quoted list."""
    return ", ".join(f"'{name}'" for name in names)


def _plain(names: list[str]) -> str:
    """Render column names as a plain comma separated list."""
    return ", ".join(str(name) for name in names)


def _clean_error(exc: Exception) -> str:
    """Turn any exception into a single readable sentence."""
    text = " ".join(str(exc).split())
    if not text:
        text = exc.__class__.__name__
    if len(text) > 200:
        text = text[:197] + "..."
    if not text.endswith("."):
        text += "."
    return text


def _is_string_like(series: pd.Series) -> bool:
    """True for text-ish columns (string, object or categorical of strings)."""
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        categories = dtype.categories
        return not (
            is_numeric_dtype(categories) or is_datetime64_any_dtype(categories)
        )
    if is_numeric_dtype(series) or is_datetime64_any_dtype(series):
        return False
    return is_string_dtype(series) or is_object_dtype(series)


def _as_text(series: pd.Series) -> pd.Series:
    """Return a string-dtype view of a text-ish column, keeping nulls null."""
    if isinstance(series.dtype, pd.CategoricalDtype) or is_object_dtype(series):
        return series.astype("string").astype("str")
    if is_string_dtype(series):
        return series
    return series.astype("string").astype("str")


def _numeric_auto(df: pd.DataFrame) -> list[str]:
    """Numeric columns picked automatically (booleans are not included)."""
    return [
        str(c) for c in df.columns
        if is_numeric_dtype(df[c]) and not is_bool_dtype(df[c])
    ]


def _string_auto(df: pd.DataFrame) -> list[str]:
    """Text-ish columns picked automatically."""
    return [str(c) for c in df.columns if _is_string_like(df[c])]


def _encodable_auto(df: pd.DataFrame) -> list[str]:
    """Columns one-hot / label encoding picks automatically (non numeric)."""
    return [str(c) for c in df.columns if not is_numeric_dtype(df[c])]


def _numeric_allowed(series: pd.Series) -> bool:
    """Explicitly named columns may also be boolean; they cast cleanly."""
    return is_numeric_dtype(series) or is_bool_dtype(series)


def _anything(series: pd.Series) -> bool:
    """Predicate for ops that accept any column."""
    return True


def _float_matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Return the named columns as a float64 matrix with NaN for nulls."""
    parts = []
    for name in columns:
        series = df[name]
        if is_bool_dtype(series):
            numeric = series.astype("float64")
        else:
            numeric = pd.to_numeric(series, errors="coerce")
        parts.append(np.asarray(numeric.to_numpy(dtype="float64", na_value=np.nan)))
    if not parts:
        return np.empty((len(df), 0), dtype="float64")
    return np.column_stack(parts)


def _float_series(series: pd.Series) -> pd.Series:
    """Return a float64 copy of a numeric/boolean column."""
    if is_bool_dtype(series):
        return series.astype("float64")
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _target_task(series: pd.Series) -> str:
    """Classification or regression for a target column.

    Uses ``core.ml.profiling.infer_task`` when that module is importable so the
    rule stays defined in exactly one place, with a conservative dtype-based
    fallback otherwise.
    """
    try:
        from .profiling import infer_task  # local import: keeps this module standalone

        return str(infer_task(series))
    except Exception:
        if is_bool_dtype(series) or not is_numeric_dtype(series):
            return "classification"
        return "regression"


# ---------------------------------------------------------------------------
# column resolution
# ---------------------------------------------------------------------------


def _resolve_columns(
    df: pd.DataFrame,
    requested: list[str] | None,
    auto: list[str],
    allowed: Callable[[pd.Series], bool],
    requirement: str,
    notes: list[str],
    auto_empty: str = "",
) -> tuple[list[str], str]:
    """Work out which columns a step should touch.

    Returns ``(columns, refusal)``.  ``refusal`` is a non-empty sentence when
    no usable column is left, in which case the step is recorded as skipped.
    Partial problems (one bad name out of three) are appended to ``notes``.
    """
    if requested is None:
        if not auto:
            return [], auto_empty or (
                f"The dataset has no {requirement} columns, so there was nothing to do."
            )
        return list(auto), ""

    usable: list[str] = []
    missing: list[str] = []
    wrong: list[str] = []
    for name in requested:
        if name not in df.columns:
            missing.append(name)
        elif allowed(df[name]):
            usable.append(name)
        else:
            wrong.append(name)

    problems: list[str] = []
    if missing:
        word = "Column" if len(missing) == 1 else "Columns"
        verb = "is" if len(missing) == 1 else "are"
        problems.append(f"{word} {_quote(missing)} {verb} not in the dataset.")
    if wrong:
        word = "Column" if len(wrong) == 1 else "Columns"
        verb = "is" if len(wrong) == 1 else "are"
        problems.append(f"{word} {_quote(wrong)} {verb} not {requirement}.")

    if not usable:
        return [], " ".join(problems) or "No columns were selected."
    notes.extend(problems)
    return usable, ""


# ---------------------------------------------------------------------------
# row / column shape steps
# ---------------------------------------------------------------------------


def _op_remove_duplicates(
    df: pd.DataFrame, columns: list[str] | None, params: dict, notes: list[str]
) -> _StepOutcome:
    cols, refusal = _resolve_columns(
        df,
        columns,
        list(df.columns),
        _anything,
        "in the dataset",
        notes,
        "The dataset has no columns, so there was nothing to do.",
    )
    if refusal:
        return _StepOutcome(df, columns or [], refusal, True)
    subset = cols if columns is not None else None
    out = df.drop_duplicates(subset=subset).reset_index(drop=True)
    removed = int(len(df) - len(out))
    scope = f" (comparing {_plain(cols)})" if subset else ""
    if removed == 0:
        return _StepOutcome(df, cols, f"No duplicate rows were found{scope}.", False)
    plural = "row" if removed == 1 else "rows"
    return _StepOutcome(out, cols, f"Removed {removed} duplicate {plural}{scope}.", False)


def _op_drop_columns(
    df: pd.DataFrame, columns: list[str] | None, params: dict, notes: list[str]
) -> _StepOutcome:
    if columns is None:
        return _StepOutcome(
            df, [], "No columns were selected, so nothing was dropped.", True
        )
    cols, refusal = _resolve_columns(
        df,
        columns,
        list(df.columns),
        _anything,
        "in the dataset",
        notes,
        "The dataset has no columns, so there was nothing to do.",
    )
    if refusal:
        return _StepOutcome(df, columns, refusal, True)
    if len(cols) >= len(df.columns):
        return _StepOutcome(
            df,
            cols,
            "Dropping every column would leave an empty dataset, so nothing was dropped.",
            True,
        )
    out = df.drop(columns=cols)
    plural = "column" if len(cols) == 1 else "columns"
    return _StepOutcome(out, cols, f"Dropped {len(cols)} {plural}: {_plain(cols)}.", False)


def _op_drop_missing_rows(
    df: pd.DataFrame, columns: list[str] | None, params: dict, notes: list[str]
) -> _StepOutcome:
    cols, refusal = _resolve_columns(
        df,
        columns,
        list(df.columns),
        _anything,
        "in the dataset",
        notes,
        "The dataset has no columns, so there was nothing to do.",
    )
    if refusal:
        return _StepOutcome(df, columns or [], refusal, True)
    out = df.dropna(subset=cols, how="any")
    removed = int(len(df) - len(out))
    scope = "any column" if columns is None else _plain(cols)
    if removed == 0:
        return _StepOutcome(df, cols, f"No rows had missing values in {scope}.", False)
    if len(out) == 0:
        return _StepOutcome(
            df,
            cols,
            f"Every row has a missing value in {scope}, so no rows were dropped.",
            True,
        )
    plural = "row" if removed == 1 else "rows"
    return _StepOutcome(
        out.reset_index(drop=True),
        cols,
        f"Dropped {removed} {plural} with missing values in {scope}.",
        False,
    )


# ---------------------------------------------------------------------------
# missing value steps
# ---------------------------------------------------------------------------


def _fill_with(
    df: pd.DataFrame,
    columns: list[str] | None,
    notes: list[str],
    auto: list[str],
    allowed: Callable[[pd.Series], bool],
    requirement: str,
    statistic: Callable[[pd.Series], Any],
    label: str,
    auto_empty: str = "",
) -> _StepOutcome:
    """Shared implementation for the fill_* family."""
    cols, refusal = _resolve_columns(
        df, columns, auto, allowed, requirement, notes, auto_empty
    )
    if refusal:
        return _StepOutcome(df, columns or [], refusal, True)

    out = df.copy()
    filled_total = 0
    details: list[str] = []
    used: list[str] = []
    for name in cols:
        series = out[name]
        missing = int(series.isna().sum())
        value = statistic(series)
        if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
            notes.append(
                f"Column '{name}' has no usable {label}, so it was left unchanged."
            )
            continue
        used.append(name)
        if missing == 0:
            continue
        out[name] = series.fillna(value)
        filled_total += missing
        details.append(f"{name}: {_describe_value(value)}")

    if not used:
        return _StepOutcome(
            df,
            cols,
            f"None of the selected columns has a usable {label}, so nothing was filled.",
            True,
        )
    if filled_total == 0:
        return _StepOutcome(
            df, used, f"There were no missing values to fill in {_plain(used)}.", False
        )
    plural = "value" if filled_total == 1 else "values"
    return _StepOutcome(
        out,
        used,
        f"Filled {filled_total} missing {plural} using the {label} ({'; '.join(details)}).",
        False,
    )


def _op_fill_mean(
    df: pd.DataFrame, columns: list[str] | None, params: dict, notes: list[str]
) -> _StepOutcome:
    return _fill_with(
        df,
        columns,
        notes,
        _numeric_auto(df),
        _numeric_allowed,
        "numeric",
        lambda s: _float_series(s).mean(),
        "mean",
    )


def _op_fill_median(
    df: pd.DataFrame, columns: list[str] | None, params: dict, notes: list[str]
) -> _StepOutcome:
    return _fill_with(
        df,
        columns,
        notes,
        _numeric_auto(df),
        _numeric_allowed,
        "numeric",
        lambda s: _float_series(s).median(),
        "median",
    )


def _mode_of(series: pd.Series) -> Any:
    modes = series.mode(dropna=True)
    if len(modes) == 0:
        return None
    return modes.iloc[0]


def _op_fill_mode(
    df: pd.DataFrame, columns: list[str] | None, params: dict, notes: list[str]
) -> _StepOutcome:
    return _fill_with(
        df,
        columns,
        notes,
        list(df.columns),
        _anything,
        "in the dataset",
        _mode_of,
        "most frequent value",
        "The dataset has no columns to fill.",
    )


def _coerce_fill_value(series: pd.Series, value: Any) -> tuple[Any, bool]:
    """Coerce a user supplied constant to the column's dtype."""
    if value is None:
        return None, False
    if is_bool_dtype(series):
        text = str(value).strip().lower()
        if text in {"true", "1", "yes"}:
            return True, True
        if text in {"false", "0", "no"}:
            return False, True
        return None, False
    if is_numeric_dtype(series):
        number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(number):
            return None, False
        return float(number), True
    if is_datetime64_any_dtype(series):
        stamp = pd.to_datetime(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(stamp):
            return None, False
        return stamp, True
    return str(value), True


def _op_fill_constant(
    df: pd.DataFrame, columns: list[str] | None, params: dict, notes: list[str]
) -> _StepOutcome:
    if "value" not in params:
        return _StepOutcome(
            df,
            columns or [],
            "This step needs a 'value' parameter saying what to fill missing values with.",
            True,
        )
    raw_value = params.get("value")
    cols, refusal = _resolve_columns(
        df,
        columns,
        list(df.columns),
        _anything,
        "in the dataset",
        notes,
        "The dataset has no columns to fill.",
    )
    if refusal:
        return _StepOutcome(df, columns or [], refusal, True)

    out = df.copy()
    filled_total = 0
    used: list[str] = []
    for name in cols:
        series = out[name]
        value, ok = _coerce_fill_value(series, raw_value)
        if not ok:
            notes.append(
                f"The value '{raw_value}' does not fit column '{name}' ({series.dtype}), "
                "so that column was left unchanged."
            )
            continue
        used.append(name)
        missing = int(series.isna().sum())
        if missing == 0:
            continue
        out[name] = series.fillna(value)
        filled_total += missing

    if not used:
        return _StepOutcome(
            df,
            cols,
            f"The value '{raw_value}' does not fit any of the selected columns, so nothing was filled.",
            True,
        )
    if filled_total == 0:
        return _StepOutcome(
            df, used, f"There were no missing values to fill in {_plain(used)}.", False
        )
    plural = "value" if filled_total == 1 else "values"
    return _StepOutcome(
        out,
        used,
        f"Filled {filled_total} missing {plural} with '{raw_value}' in {_plain(used)}.",
        False,
    )


def _op_fill_empty_string(
    df: pd.DataFrame, columns: list[str] | None, params: dict, notes: list[str]
) -> _StepOutcome:
    cols, refusal = _resolve_columns(
        df, columns, _string_auto(df), _is_string_like, "text", notes
    )
    if refusal:
        return _StepOutcome(df, columns or [], refusal, True)
    out = df.copy()
    filled_total = 0
    for name in cols:
        series = _as_text(out[name])
        missing = int(series.isna().sum())
        out[name] = series.fillna("")
        filled_total += missing
    if filled_total == 0:
        return _StepOutcome(
            out, cols, f"There were no missing values to fill in {_plain(cols)}.", False
        )
    plural = "value" if filled_total == 1 else "values"
    return _StepOutcome(
        out,
        cols,
        f"Filled {filled_total} missing {plural} with an empty string in {_plain(cols)}.",
        False,
    )


# ---------------------------------------------------------------------------
# text steps
# ---------------------------------------------------------------------------


def _text_step(
    df: pd.DataFrame,
    columns: list[str] | None,
    notes: list[str],
    transform: Callable[[pd.Series], pd.Series],
    verb: str,
) -> _StepOutcome:
    """Shared implementation for the text cleaning family."""
    cols, refusal = _resolve_columns(
        df, columns, _string_auto(df), _is_string_like, "text", notes
    )
    if refusal:
        return _StepOutcome(df, columns or [], refusal, True)
    out = df.copy()
    changed_total = 0
    for name in cols:
        original = _as_text(out[name])
        updated = transform(original)
        changed_total += int((original.ne(updated) & original.notna()).sum())
        out[name] = updated
    plural = "value" if changed_total == 1 else "values"
    return _StepOutcome(
        out,
        cols,
        f"{verb} in {_plain(cols)} ({changed_total} {plural} changed).",
        False,
    )


def _op_strip_whitespace(
    df: pd.DataFrame, columns: list[str] | None, params: dict, notes: list[str]
) -> _StepOutcome:
    return _text_step(
        df, columns, notes, lambda s: s.str.strip(), "Trimmed surrounding whitespace"
    )


def _op_lowercase(
    df: pd.DataFrame, columns: list[str] | None, params: dict, notes: list[str]
) -> _StepOutcome:
    return _text_step(df, columns, notes, lambda s: s.str.lower(), "Converted text to lowercase")


def _op_remove_punctuation(
    df: pd.DataFrame, columns: list[str] | None, params: dict, notes: list[str]
) -> _StepOutcome:
    return _text_step(
        df,
        columns,
        notes,
        lambda s: s.str.replace(_PUNCTUATION_RE, "", regex=True),
        "Removed punctuation",
    )


# ---------------------------------------------------------------------------
# outlier steps
# ---------------------------------------------------------------------------


def _op_remove_outliers_iqr(
    df: pd.DataFrame, columns: list[str] | None, params: dict, notes: list[str]
) -> _StepOutcome:
    cols, refusal = _resolve_columns(
        df, columns, _numeric_auto(df), _numeric_allowed, "numeric", notes
    )
    if refusal:
        return _StepOutcome(df, columns or [], refusal, True)

    mask = pd.Series(False, index=df.index)
    for name in cols:
        values = _float_series(df[name])
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        if pd.isna(q1) or pd.isna(q3):
            notes.append(f"Column '{name}' has no usable values, so it was ignored.")
            continue
        spread = float(q3) - float(q1)
        if spread == 0:
            notes.append(
                f"Column '{name}' has an interquartile range of 0, so it has no outliers."
            )
            continue
        low = float(q1) - IQR_MULTIPLIER * spread
        high = float(q3) + IQR_MULTIPLIER * spread
        mask = mask | ((values < low) | (values > high)).fillna(False)

    removed = int(mask.sum())
    if removed == 0:
        return _StepOutcome(
            df, cols, f"No outliers were found in {_plain(cols)} using the 1.5x IQR rule.", False
        )
    if removed >= len(df):
        return _StepOutcome(
            df, cols, "Every row looks like an outlier, so no rows were removed.", True
        )
    out = df.loc[~mask].reset_index(drop=True)
    plural = "row" if removed == 1 else "rows"
    return _StepOutcome(
        out,
        cols,
        f"Removed {removed} outlier {plural} in {_plain(cols)} using the 1.5x IQR rule.",
        False,
    )


def _op_remove_outliers_zscore(
    df: pd.DataFrame, columns: list[str] | None, params: dict, notes: list[str]
) -> _StepOutcome:
    threshold, note = _positive_float(params, "threshold", DEFAULT_ZSCORE_THRESHOLD)
    if note:
        notes.append(note)
    cols, refusal = _resolve_columns(
        df, columns, _numeric_auto(df), _numeric_allowed, "numeric", notes
    )
    if refusal:
        return _StepOutcome(df, columns or [], refusal, True)

    mask = pd.Series(False, index=df.index)
    for name in cols:
        values = _float_series(df[name])
        mean = values.mean()
        std = values.std(ddof=0)
        if pd.isna(mean) or pd.isna(std) or float(std) == 0:
            notes.append(
                f"Column '{name}' does not vary, so it cannot contain z-score outliers."
            )
            continue
        scores = (values - float(mean)).abs() / float(std)
        mask = mask | (scores > threshold).fillna(False)

    removed = int(mask.sum())
    shown = _format_number(threshold)
    if removed == 0:
        return _StepOutcome(
            df, cols, f"No values in {_plain(cols)} had a z-score above {shown}.", False
        )
    if removed >= len(df):
        return _StepOutcome(
            df, cols, "Every row looks like an outlier, so no rows were removed.", True
        )
    out = df.loc[~mask].reset_index(drop=True)
    plural = "row" if removed == 1 else "rows"
    return _StepOutcome(
        out,
        cols,
        f"Removed {removed} {plural} with a z-score above {shown} in {_plain(cols)}.",
        False,
    )


# ---------------------------------------------------------------------------
# scaling steps
# ---------------------------------------------------------------------------


def _scale_step(
    df: pd.DataFrame,
    columns: list[str] | None,
    notes: list[str],
    scaler_factory: Callable[[], Any],
    description: str,
) -> _StepOutcome:
    cols, refusal = _resolve_columns(
        df, columns, _numeric_auto(df), _numeric_allowed, "numeric", notes
    )
    if refusal:
        return _StepOutcome(df, columns or [], refusal, True)

    usable: list[str] = []
    for name in cols:
        values = _float_series(df[name])
        if int(values.notna().sum()) == 0:
            notes.append(f"Column '{name}' is empty, so it was left unchanged.")
            continue
        if int(values.nunique(dropna=True)) <= 1:
            notes.append(
                f"Column '{name}' has a single distinct value, so scaling maps it to 0."
            )
        usable.append(name)
    if not usable:
        return _StepOutcome(
            df, cols, "None of the selected columns has any values to scale.", True
        )

    matrix = _float_matrix(df, usable)
    scaled = scaler_factory().fit_transform(matrix)
    out = df.copy()
    for index, name in enumerate(usable):
        out[name] = pd.Series(scaled[:, index], index=df.index, dtype="float64")
    plural = "column" if len(usable) == 1 else "columns"
    return _StepOutcome(
        out, usable, f"{description} for {len(usable)} {plural}: {_plain(usable)}.", False
    )


def _op_minmax_scale(
    df: pd.DataFrame, columns: list[str] | None, params: dict, notes: list[str]
) -> _StepOutcome:
    return _scale_step(df, columns, notes, MinMaxScaler, "Scaled values to the 0-1 range")


def _op_standard_scale(
    df: pd.DataFrame, columns: list[str] | None, params: dict, notes: list[str]
) -> _StepOutcome:
    return _scale_step(
        df, columns, notes, StandardScaler, "Standardized values to mean 0 and standard deviation 1"
    )


def _op_robust_scale(
    df: pd.DataFrame, columns: list[str] | None, params: dict, notes: list[str]
) -> _StepOutcome:
    return _scale_step(
        df, columns, notes, RobustScaler, "Rescaled values using the median and interquartile range"
    )


# ---------------------------------------------------------------------------
# encoding steps
# ---------------------------------------------------------------------------


def _unique_name(name: str, taken: set[str]) -> str:
    """Return ``name`` or the first free ``name_2``, ``name_3`` variant."""
    if name not in taken:
        return name
    suffix = 2
    while f"{name}_{suffix}" in taken:
        suffix += 1
    return f"{name}_{suffix}"


def _op_one_hot_encode(
    df: pd.DataFrame, columns: list[str] | None, params: dict, notes: list[str]
) -> _StepOutcome:
    cols, refusal = _resolve_columns(
        df,
        columns,
        _encodable_auto(df),
        _anything,
        "in the dataset",
        notes,
        "The dataset has no non-numeric columns to encode, so there was nothing to do.",
    )
    if refusal:
        return _StepOutcome(df, columns or [], refusal, True)

    out = df.copy()
    encoded: list[str] = []
    reasons: list[str] = []
    created = 0
    for name in cols:
        source = out[name]
        distinct = int(source.nunique(dropna=True))
        if distinct == 0:
            reasons.append(f"Column '{name}' is empty, so there was nothing to encode.")
            notes.append(reasons[-1])
            continue
        if distinct > MAX_ONE_HOT_COLUMNS:
            reasons.append(
                f"Column '{name}' has {distinct} distinct values, which is more than the "
                f"limit of {MAX_ONE_HOT_COLUMNS} new columns, so it was left as it is."
            )
            notes.append(reasons[-1])
            continue
        if isinstance(source.dtype, pd.CategoricalDtype):
            source = source.astype("str")
        dummies = pd.get_dummies(source, prefix=name, prefix_sep="_", dtype="int64")
        taken = set(map(str, out.columns)) - {name}
        renamed = {}
        for column in dummies.columns:
            # Reading the stored file back strips whitespace from headers, so a
            # name is stripped here instead: otherwise the saved column and the
            # profiled one would have different names and the feature would not
            # be selectable.
            cleaned = str(column).strip() or f"{name}_value"
            safe = _unique_name(cleaned, taken)
            if cleaned != str(column):
                notes.append(
                    f"The value '{str(column)[len(name) + 1:]}' in column '{name}' has "
                    f"spaces around it, so its new column was named '{safe}'."
                )
            elif safe != cleaned:
                notes.append(
                    f"New column '{cleaned}' already existed, so it was named '{safe}'."
                )
            taken.add(safe)
            renamed[column] = safe
        dummies = dummies.rename(columns=renamed)
        position = list(out.columns).index(name)
        order = list(out.columns)
        order[position : position + 1] = list(dummies.columns)
        out = pd.concat([out.drop(columns=[name]), dummies], axis=1)
        out = out[order]
        encoded.append(name)
        created += int(dummies.shape[1])

    if not encoded:
        message = "None of the selected columns could be one-hot encoded."
        return _StepOutcome(df, cols, " ".join([message] + reasons), True)
    plural = "column" if len(encoded) == 1 else "columns"
    return _StepOutcome(
        out,
        encoded,
        f"One-hot encoded {len(encoded)} {plural} into {created} new columns: {_plain(encoded)}.",
        False,
    )


def _op_label_encode(
    df: pd.DataFrame, columns: list[str] | None, params: dict, notes: list[str]
) -> _StepOutcome:
    cols, refusal = _resolve_columns(
        df,
        columns,
        _encodable_auto(df),
        _anything,
        "in the dataset",
        notes,
        "The dataset has no non-numeric columns to encode, so there was nothing to do.",
    )
    if refusal:
        return _StepOutcome(df, columns or [], refusal, True)

    out = df.copy()
    encoded: list[str] = []
    details: list[str] = []
    for name in cols:
        source = out[name]
        if isinstance(source.dtype, pd.CategoricalDtype):
            source = source.astype("str")
        distinct = list(pd.Series(source.dropna().unique()))
        if not distinct:
            notes.append(f"Column '{name}' is empty, so there was nothing to encode.")
            continue
        try:
            ordered = sorted(distinct)
        except TypeError:
            ordered = sorted(distinct, key=str)
        mapping = {value: index for index, value in enumerate(ordered)}
        codes = source.map(mapping)
        numeric = pd.to_numeric(pd.Series(codes, index=out.index), errors="coerce")
        if bool(numeric.isna().any()):
            out[name] = numeric.astype("Int64")
        else:
            out[name] = numeric.astype("int64")
        encoded.append(name)
        details.append(f"{name}: {len(ordered)} codes (0-{len(ordered) - 1})")

    if not encoded:
        return _StepOutcome(
            df, cols, "None of the selected columns had any values to label encode.", True
        )
    plural = "column" if len(encoded) == 1 else "columns"
    return _StepOutcome(
        out,
        encoded,
        f"Label encoded {len(encoded)} {plural} into whole numbers ({'; '.join(details)}).",
        False,
    )


# ---------------------------------------------------------------------------
# feature engineering steps
# ---------------------------------------------------------------------------


def _positive_int(params: dict, key: str, default: int) -> tuple[int, str]:
    """Read an int parameter, falling back to ``default`` with a note."""
    raw = params.get(key, default)
    if raw is None or raw == "":
        return default, ""
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return default, f"'{raw}' is not a whole number for '{key}', so {default} was used."
    if value < 1:
        return default, f"'{key}' must be at least 1, so {default} was used."
    return value, ""


def _positive_float(params: dict, key: str, default: float) -> tuple[float, str]:
    """Read a positive float parameter, falling back to ``default`` with a note."""
    raw = params.get(key, default)
    if raw is None or raw == "":
        return default, ""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default, f"'{raw}' is not a number for '{key}', so {_format_number(default)} was used."
    if not math.isfinite(value) or value <= 0:
        return default, f"'{key}' must be greater than 0, so {_format_number(default)} was used."
    return value, ""


def _feature_setup(
    df: pd.DataFrame,
    columns: list[str] | None,
    params: dict,
    notes: list[str],
    op_label: str,
    target_required: bool,
) -> tuple[list[str], np.ndarray, str | None, str]:
    """Resolve the numeric feature matrix (and target) for pca/select/lda.

    Returns ``(features, matrix, target, refusal)``.
    """
    target = params.get("target")
    if isinstance(target, str):
        target = target.strip() or None
    elif target is not None:
        target = str(target)

    if target_required and not target:
        return [], np.empty((0, 0)), None, f"{op_label} needs a 'target' column to work with."
    if target and target not in df.columns:
        return [], np.empty((0, 0)), None, f"Column '{target}' is not in the dataset."

    auto = [name for name in _numeric_auto(df) if name != target]
    requested = None if columns is None else [c for c in columns if c != target]
    features, refusal = _resolve_columns(df, requested, auto, _numeric_allowed, "numeric", notes)
    if refusal:
        return [], np.empty((0, 0)), target, refusal
    features = [name for name in features if name != target]
    if not features:
        return [], np.empty((0, 0)), target, "There are no numeric feature columns left to work with."
    if len(df) < 2:
        return [], np.empty((0, 0)), target, "At least 2 rows are needed for this step."

    matrix = _float_matrix(df, features)
    bad = [features[i] for i in range(matrix.shape[1]) if not np.isfinite(matrix[:, i]).all()]
    if bad:
        word = "Column" if len(bad) == 1 else "Columns"
        verb = "still contains" if len(bad) == 1 else "still contain"
        return (
            [],
            np.empty((0, 0)),
            target,
            f"{word} {_quote(bad)} {verb} missing or infinite values. "
            "Add a fill step (for example fill_mean) before this one.",
        )
    return features, matrix, target, ""


def _replace_features(
    df: pd.DataFrame, features: list[str], values: np.ndarray, prefix: str
) -> pd.DataFrame:
    """Drop the used feature columns and append the new components."""
    out = df.drop(columns=features)
    for index in range(values.shape[1]):
        out[f"{prefix}_{index + 1}"] = pd.Series(
            values[:, index], index=df.index, dtype="float64"
        )
    return out


def _op_pca(
    df: pd.DataFrame, columns: list[str] | None, params: dict, notes: list[str]
) -> _StepOutcome:
    features, matrix, target, refusal = _feature_setup(
        df, columns, params, notes, "Principal component analysis", target_required=False
    )
    if refusal:
        return _StepOutcome(df, columns or [], refusal, True)

    requested, note = _positive_int(params, "n_components", 2)
    if note:
        notes.append(note)
    limit = min(len(features), len(df))
    n_components = max(1, min(requested, limit))
    if n_components != requested:
        notes.append(
            f"n_components was reduced from {requested} to {n_components}, which is the most "
            f"this data supports ({len(features)} numeric columns, {len(df)} rows)."
        )

    # Standardize first: principal components follow raw variance, so without
    # this a column measured in large units (salary) becomes pc_1 on its own
    # and every other column is thrown away.
    model = SkPipeline(
        [("scale", StandardScaler()), ("pca", PCA(n_components=n_components))]
    )
    with warnings_module.catch_warnings():
        warnings_module.simplefilter("ignore")
        components = model.fit_transform(matrix)
    ratios = np.asarray(
        model.named_steps["pca"].explained_variance_ratio_, dtype="float64"
    )
    out = _replace_features(df, features, components, "pc")
    explained = float(np.nansum(ratios) * 100)
    kept = ", ".join(f"pc_{i + 1}" for i in range(n_components))
    plural = "column" if len(features) == 1 else "columns"
    return _StepOutcome(
        out,
        features,
        f"Reduced {len(features)} numeric {plural} to {kept}, keeping "
        f"{_format_number(explained)}% of the variance. The columns were put on the "
        "same scale first, so each one counted equally.",
        False,
    )


def _op_select_k_best(
    df: pd.DataFrame, columns: list[str] | None, params: dict, notes: list[str]
) -> _StepOutcome:
    features, matrix, target, refusal = _feature_setup(
        df, columns, params, notes, "Select the k best features", target_required=True
    )
    if refusal:
        return _StepOutcome(df, columns or [], refusal, True)

    labels = df[target]
    keep = labels.notna().to_numpy()
    if int(keep.sum()) < 2:
        return _StepOutcome(
            df, features, f"Column '{target}' has fewer than 2 usable values.", True
        )
    task = _target_task(labels[keep])
    score_func = f_classif if task == "classification" else f_regression
    if task == "classification" and int(labels[keep].nunique()) < 2:
        return _StepOutcome(
            df, features, f"Column '{target}' has only one distinct value.", True
        )

    requested, note = _positive_int(params, "k", 5)
    if note:
        notes.append(note)
    k = max(1, min(requested, len(features)))
    if k != requested:
        notes.append(
            f"k was reduced from {requested} to {k}, which is the number of numeric "
            "feature columns available."
        )
    if k == len(features):
        return _StepOutcome(
            df,
            features,
            f"All {len(features)} numeric feature columns were kept because k is not "
            "smaller than the number of features.",
            False,
        )

    y = _target_values(labels[keep], task)
    with warnings_module.catch_warnings():
        warnings_module.simplefilter("ignore")
        with np.errstate(divide="ignore", invalid="ignore"):
            selector = SelectKBest(score_func=score_func, k=k).fit(matrix[keep], y)
    chosen = [features[i] for i in selector.get_support(indices=True)]
    dropped = [name for name in features if name not in chosen]
    out = df.drop(columns=dropped)
    method = "f_classif" if task == "classification" else "f_regression"
    return _StepOutcome(
        out,
        features,
        f"Kept the {len(chosen)} numeric columns most related to '{target}' ({method}): "
        f"{_plain(chosen)}. Dropped {_plain(dropped)}.",
        False,
    )


def _target_values(labels: pd.Series, task: str) -> np.ndarray:
    """Return an sklearn friendly array for the target column."""
    if task == "classification" and not (
        is_numeric_dtype(labels) or is_bool_dtype(labels)
    ):
        return labels.astype("object").to_numpy()
    numeric = pd.to_numeric(labels, errors="coerce")
    return np.asarray(numeric.to_numpy(dtype="float64", na_value=np.nan))


def _op_lda(
    df: pd.DataFrame, columns: list[str] | None, params: dict, notes: list[str]
) -> _StepOutcome:
    features, matrix, target, refusal = _feature_setup(
        df, columns, params, notes, "Linear discriminant analysis", target_required=True
    )
    if refusal:
        return _StepOutcome(df, columns or [], refusal, True)

    labels = df[target]
    keep = labels.notna().to_numpy()
    if int(keep.sum()) < 2:
        return _StepOutcome(
            df, features, f"Column '{target}' has fewer than 2 usable values.", True
        )
    task = _target_task(labels[keep])
    if task != "classification":
        return _StepOutcome(
            df,
            features,
            f"Linear discriminant analysis needs a target with classes, and '{target}' "
            "looks like a continuous number.",
            True,
        )
    n_classes = int(labels[keep].nunique())
    if n_classes < 2:
        return _StepOutcome(
            df,
            features,
            f"Column '{target}' has only one distinct value, so there is nothing to separate.",
            True,
        )

    requested, note = _positive_int(params, "n_components", 1)
    if note:
        notes.append(note)
    limit = min(len(features), n_classes - 1)
    n_components = max(1, min(requested, limit))
    if n_components != requested:
        notes.append(
            f"n_components was reduced from {requested} to {n_components}, which is the most "
            f"linear discriminant analysis can produce for {n_classes} classes and "
            f"{len(features)} numeric columns."
        )

    y = _target_values(labels[keep], task)
    model = LinearDiscriminantAnalysis(n_components=n_components)
    with warnings_module.catch_warnings():
        warnings_module.simplefilter("ignore")
        model.fit(matrix[keep], y)
        projected = np.asarray(model.transform(matrix), dtype="float64")
    out = _replace_features(df, features, projected, "ld")
    kept = ", ".join(f"ld_{i + 1}" for i in range(n_components))
    plural = "column" if len(features) == 1 else "columns"
    return _StepOutcome(
        out,
        features,
        f"Reduced {len(features)} numeric {plural} to {kept} using the {n_classes} classes "
        f"of '{target}'.",
        False,
    )


_HANDLERS: dict[str, Callable[..., _StepOutcome]] = {
    "remove_duplicates": _op_remove_duplicates,
    "drop_columns": _op_drop_columns,
    "drop_missing_rows": _op_drop_missing_rows,
    "fill_mean": _op_fill_mean,
    "fill_median": _op_fill_median,
    "fill_mode": _op_fill_mode,
    "fill_constant": _op_fill_constant,
    "fill_empty_string": _op_fill_empty_string,
    "strip_whitespace": _op_strip_whitespace,
    "lowercase": _op_lowercase,
    "remove_punctuation": _op_remove_punctuation,
    "remove_outliers_iqr": _op_remove_outliers_iqr,
    "remove_outliers_zscore": _op_remove_outliers_zscore,
    "minmax_scale": _op_minmax_scale,
    "standard_scale": _op_standard_scale,
    "robust_scale": _op_robust_scale,
    "one_hot_encode": _op_one_hot_encode,
    "label_encode": _op_label_encode,
    "pca": _op_pca,
    "select_k_best": _op_select_k_best,
    "lda": _op_lda,
}


def _parse_step(step: Any, position: int) -> tuple[str, list[str] | None, dict, str]:
    """Validate one raw step. Returns ``(op, columns, params, problem)``."""
    if not isinstance(step, dict):
        raise PreprocessError(
            f"Step {position} is not valid. Each step must be an object with an 'op' field."
        )
    raw_op = step.get("op")
    op = raw_op.strip().lower() if isinstance(raw_op, str) else raw_op
    if not op or op not in STEP_OPS:
        supported = ", ".join(sorted(STEP_OPS))
        raise PreprocessError(
            f"'{raw_op}' is not a preprocessing step we know. Supported steps are: {supported}."
        )

    params = step.get("params")
    if not isinstance(params, dict):
        params = {}

    raw_columns = step.get("columns")
    problem = ""
    columns: list[str] | None
    if raw_columns is None:
        columns = None
    elif isinstance(raw_columns, str):
        columns = [raw_columns]
    elif isinstance(raw_columns, (list, tuple)):
        columns = []
        for item in raw_columns:
            if item is None:
                continue
            name = str(item)
            if name not in columns:
                columns.append(name)
        if not columns:
            columns = None
    else:
        columns = None
        problem = "The 'columns' value must be a list of column names."
    return op, columns, params, problem


def apply_steps(df: pd.DataFrame, steps: list[dict]) -> tuple[pd.DataFrame, dict]:
    """Apply preprocessing steps in order.

    Returns ``(new_dataframe, report)`` where the report matches the
    ``PreprocessReport`` shape: ``rows_before``, ``rows_after``, ``cols_before``,
    ``cols_after``, ``steps`` and ``warnings``.  A step that cannot be applied is
    recorded with ``skipped: true`` and an explanatory message.  Only an unknown
    ``op`` raises :class:`PreprocessError`.
    """
    if df is None:
        raise PreprocessError("There is no dataset to preprocess.")
    if steps is None:
        steps = []
    if not isinstance(steps, (list, tuple)):
        raise PreprocessError("The preprocessing steps must be given as a list.")

    current = df.copy()
    rows_before = int(len(current))
    cols_before = int(current.shape[1])
    entries: list[dict] = []
    warnings: list[str] = []

    for position, raw_step in enumerate(steps, start=1):
        op, columns, params, problem = _parse_step(raw_step, position)
        label = STEP_OPS[op]
        notes: list[str] = []
        if problem:
            outcome = _StepOutcome(current, [], problem, True)
        else:
            handler = _HANDLERS[op]
            try:
                outcome = handler(current.copy(), columns, params, notes)
            except PreprocessError:
                raise
            except Exception as exc:  # noqa: BLE001 - surfaced as a skipped step
                outcome = _StepOutcome(
                    current,
                    list(columns or []),
                    f"This step could not be applied: {_clean_error(exc)}",
                    True,
                )
        current = outcome.df
        for note in notes:
            warnings.append(f"{label}: {note}")
        entries.append(
            {
                "op": op,
                "columns": [str(name) for name in outcome.columns],
                "message": outcome.message,
                "skipped": bool(outcome.skipped),
                "rows_after": int(len(current)),
                "cols_after": int(current.shape[1]),
            }
        )

    report = {
        "rows_before": rows_before,
        "rows_after": int(len(current)),
        "cols_before": cols_before,
        "cols_after": int(current.shape[1]),
        "steps": entries,
        "warnings": warnings,
    }
    return current, report

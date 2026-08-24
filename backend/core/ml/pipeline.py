"""Feature preprocessing pipelines for Furina ML.

Everything here must stay picklable: the fitted pipeline is joblib-dumped as
part of a model artifact, so no lambdas or closures may end up inside an
estimator's parameters.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from core.ml.profiling import column_kind

NUMERIC_BLOCK = "num"
CATEGORICAL_BLOCK = "cat"
_BLOCK_PREFIXES = (f"{NUMERIC_BLOCK}__", f"{CATEGORICAL_BLOCK}__")


def to_string_frame(X: Any) -> pd.DataFrame:
    """Cast every column to plain text, keeping missing values as NaN.

    Defined at module level (never a lambda) so the fitted pipeline can be
    joblib-dumped and re-loaded in another process.
    """
    frame = pd.DataFrame(X)
    data = {}
    for column in frame.columns:
        series = frame[column]
        missing = series.isna().to_numpy(dtype=bool)
        text = series.astype("object").astype(str).to_numpy(dtype=object)
        text[missing] = np.nan
        data[column] = text
    return pd.DataFrame(data, index=frame.index, columns=frame.columns)


def split_feature_kinds(
    df: pd.DataFrame, feature_columns: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Split feature columns into (numeric, categorical).

    Datetime, text and boolean columns are folded into the categorical group
    because they are one-hot encoded.
    """
    numeric: list[str] = []
    categorical: list[str] = []
    for column in feature_columns:
        if column not in df.columns:
            continue
        if column_kind(df[column]) == "numeric":
            numeric.append(column)
        else:
            categorical.append(column)
    return numeric, categorical


def usable_feature_kinds(
    df: pd.DataFrame, feature_columns: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Like :func:`split_feature_kinds` but drops columns that hold no values.

    A column that is entirely empty carries no information and would make the
    imputers emit warnings, so it is left out of the preprocessor.
    """
    numeric, categorical = split_feature_kinds(df, feature_columns)
    numeric = [column for column in numeric if bool(df[column].notna().any())]
    categorical = [column for column in categorical if bool(df[column].notna().any())]
    return numeric, categorical


def build_preprocessor(
    df: pd.DataFrame, feature_columns: Sequence[str], scale: bool
) -> ColumnTransformer:
    """Build the column transformer that turns raw columns into a float matrix.

    Tolerates zero numeric columns, zero categorical columns and columns that
    are entirely empty. Unseen categories at predict time are ignored.
    """
    numeric, categorical = usable_feature_kinds(df, feature_columns)
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric:
        numeric_steps: list[tuple[str, Any]] = [
            ("impute", SimpleImputer(strategy="median")),
        ]
        if scale:
            numeric_steps.append(("scale", StandardScaler()))
        transformers.append((NUMERIC_BLOCK, Pipeline(numeric_steps), numeric))

    if categorical:
        categorical_steps: list[tuple[str, Any]] = [
            (
                "stringify",
                FunctionTransformer(to_string_frame, feature_names_out="one-to-one"),
            ),
            (
                "impute",
                SimpleImputer(strategy="most_frequent", fill_value="missing"),
            ),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
        transformers.append(
            (CATEGORICAL_BLOCK, Pipeline(categorical_steps), categorical)
        )

    return ColumnTransformer(transformers, remainder="drop")


def _strip_block_prefix(name: str) -> str:
    """Remove the ColumnTransformer block prefix from an expanded name."""
    text = str(name)
    for prefix in _BLOCK_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def expanded_feature_names(
    fitted_preprocessor: Any,
    numeric: Sequence[str],
    categorical: Sequence[str],
) -> list[str]:
    """Return the names of the columns the fitted preprocessor produces."""
    try:
        return [
            _strip_block_prefix(name)
            for name in fitted_preprocessor.get_feature_names_out()
        ]
    except Exception:
        pass

    fallback: list[str] = []
    for _name, transformer, columns in getattr(
        fitted_preprocessor, "transformers_", []
    ):
        if transformer == "drop" or isinstance(columns, str):
            continue
        try:
            fallback.extend(str(column) for column in columns)
        except TypeError:
            continue
    if fallback:
        return fallback
    return [str(column) for column in list(numeric) + list(categorical)]


def build_pipeline(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    estimator: Any,
    scale: bool,
) -> Pipeline:
    """Wrap the preprocessor and an estimator into a single fittable pipeline."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor(df, feature_columns, scale)),
            ("model", estimator),
        ]
    )

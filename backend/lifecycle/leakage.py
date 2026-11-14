"""Data-leakage and quality checks that run before training.

Each check returns a :class:`LeakageFinding`; :func:`run_leakage_checks`
bundles them all, and :func:`severity` folds a list down to the single
severity the UI badges with. Checks are deliberately non-blocking: they
always return findings, never raise, and a check that cannot decide says so
in its message instead of guessing.

This module is pure pandas and has no Django import so the same rules can
run in the in-process HTTP path *and* in a future worker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core.ml.io import json_safe
from core.ml.profiling import column_kind

SEVERITIES = ("info", "warning", "critical")
_SEVERITY_RANK = {severity: index for index, severity in enumerate(SEVERITIES)}


@dataclass
class LeakageFinding:
    """One check's verdict, ready for JSON."""

    code: str
    severity: str
    message: str
    columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "code": self.code,
                "severity": self.severity,
                "message": self.message,
                "columns": list(self.columns),
            }
        )


def severity(findings: list[LeakageFinding] | list[dict]) -> str:
    """The highest severity present, defaulting to ``info`` when nothing fired."""
    ranks = [0]
    for finding in findings:
        label = finding["severity"] if isinstance(finding, dict) else finding.severity
        ranks.append(_SEVERITY_RANK.get(str(label).lower(), 0))
    return SEVERITIES[max(ranks)] if ranks else "info"


# --------------------------------------------------------------------------
# individual checks
# --------------------------------------------------------------------------


def _n_unique(series: pd.Series) -> int:
    try:
        return int(series.nunique(dropna=True))
    except (TypeError, ValueError):
        return int(series.dropna().apply(lambda value: hash(value)).nunique())


def target_leakage(frame: pd.DataFrame, target_column: str, feature_columns: list[str]) -> list[LeakageFinding]:
    """Features that are suspiciously decisive about the target."""
    findings: list[LeakageFinding] = []
    if target_column not in frame.columns:
        return findings

    target = frame[target_column]
    target_is_text = not (
        pd.api.types.is_numeric_dtype(target) or pd.api.types.is_bool_dtype(target)
    )
    target_unique = _n_unique(target)
    if target_unique < 2:
        return findings

    for column in feature_columns:
        if column not in frame.columns:
            continue
        series = frame[column]
        col_kind = column_kind(series)

        # An identifier-like text column can never explain a target: it just
        # *names* rows.
        if col_kind == "text":
            findings.append(
                LeakageFinding(
                    code="identifier",
                    severity="critical",
                    message=(
                        f"Column '{column}' holds nearly-unique text and looks like an identifier. "
                        "An identifier does not cause the outcome and must not be used as a feature."
                    ),
                    columns=[column],
                )
            )
            continue

        # A categorical column that is almost as unique as the target is
        # likely to *encode* it (loan_status, account_number, ...).
        if col_kind in ("categorical", "boolean"):
            if _n_unique(series) == 1:
                continue
            col_unique = _n_unique(series)
            # The column distinguishes at least as many cases as the target
            # has, so knowing it can pin the target down row by row.
            if col_unique >= target_unique * 0.5 and col_unique >= 5:
                findings.append(
                    LeakageFinding(
                        code="target",
                        severity="critical",
                        message=(
                            f"Column '{column}' is very high-cardinality categorical, "
                            "almost as unique as the target itself. A model can memorise the "
                            "target through it, which is leakage, not signal."
                        ),
                        columns=[column],
                    )
                )
        # A numeric column that is a near-perfect proxy for the target.
        elif pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null) < 2:
                continue
            corr = _safe_correlation(non_null, target)
            if corr is not None and abs(corr) >= 0.99:
                findings.append(
                    LeakageFinding(
                        code="target",
                        severity="critical",
                        message=(
                            f"Column '{column}' is almost perfectly correlated with the target "
                            f"(|r|={abs(corr):.2f}). It likely encodes the outcome rather than causing it."
                        ),
                        columns=[column],
                    )
                )
    return findings


def _safe_correlation(a: pd.Series, b: pd.Series) -> float | None:
    """Correlation between two possibly-mixed series, NaN-safe, or None."""
    aligned = pd.concat([pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)], axis=1).dropna()
    if len(aligned) < 3:
        return None
    x, y = aligned.iloc[:, 0], aligned.iloc[:, 1]
    if not (pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y)):
        # Both sides are text: map to codes so we can still catch a proxy.
        x = pd.Series(pd.factorize(x)[0])
        y = pd.Series(pd.factorize(y)[0])
    try:
        return float(
            np.corrcoef(
                np.asarray(x, dtype=float), np.asarray(y, dtype=float)
            )[0, 1]
        )
    except (ValueError, TypeError):
        return None


def identifier_leakage(frame: pd.DataFrame, feature_columns: list[str]) -> list[LeakageFinding]:
    """Near-unique columns whose names or shape scream 'row identifier'."""
    findings: list[LeakageFinding] = []
    total_rows = max(len(frame), 1)
    name_hits = ("id", "uuid", "guid", "index", "key", "code", "number", "ref", "uuid")
    for column in feature_columns:
        if column not in frame.columns:
            continue
        series = frame[column]
        non_null = series.dropna()
        if len(non_null) == 0:
            continue
        ratio = _n_unique(series) / total_rows
        if ratio < 0.9:
            continue
        # A datetime column being unique is normal, not an identifier.
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        is_text = column_kind(series) in ("text", "categorical")
        looks_like_id_name = any(token in str(column).lower() for token in name_hits)
        # Numeric columns are only flagged when the name itself says identifier:
        # a 50-row dataset where every age differs is data, not leakage.
        if not is_text and not looks_like_id_name:
            continue
        if not (looks_like_id_name or ratio >= 0.98):
            continue
        findings.append(
            LeakageFinding(
                code="identifier",
                severity="critical" if ratio >= 0.98 else "warning",
                message=(
                    f"Column '{column}' is unique in {ratio:.0%} of rows, which is what an "
                    "identifier looks like. Drop it: it only teaches the model to copy rows."
                ),
                columns=[column],
            )
        )
    return findings


def train_test_overlap(
    train: pd.DataFrame, test: pd.DataFrame, key_columns: list[str] | None = None
) -> list[LeakageFinding]:
    """Duplicate rows that appear in both the train and the test split.

    If a row is scored in training *and* in evaluation, the metric is
    inflated; the model literally saw the answer.
    """
    findings: list[LeakageFinding] = []
    if train.empty or test.empty:
        return findings
    key = [key for key in (key_columns or train.columns) if key in test.columns]
    if not key:
        key = [column for column in train.columns if column in test.columns]
    if not key:
        return findings
    train_rows = train[key].copy()
    test_rows = test[key].copy()
    both = train_rows.drop_duplicates().merge(test_rows.drop_duplicates(), on=key, how="inner")
    overlap = int(len(both))
    if overlap:
        share = overlap / max(len(test_rows.drop_duplicates()), 1)
        findings.append(
            LeakageFinding(
                code="duplication",
                severity="critical" if share >= 0.5 else "warning",
                message=(
                    f"{overlap} row(s) appear in both the training and test sets "
                    f"({share:.0%} of the test set). Their scores are not independent, so the "
                    "metric overstates performance. Check for split keys or duplicate rows."
                ),
                columns=key,
            )
        )
    return findings


def temporal_leakage(frame: pd.DataFrame, target_column: str) -> list[LeakageFinding]:
    """Warn when a datetime column exists and a random split is likely to cross time."""
    findings: list[LeakageFinding] = []
    date_columns = [
        column
        for column in frame.columns
        if column != target_column and pd.api.types.is_datetime64_any_dtype(frame[column])
    ]
    if date_columns:
        findings.append(
            LeakageFinding(
                code="temporal",
                severity="warning",
                message=(
                    "This dataset has a date column. A random train/test split can put later "
                    "rows in training and earlier rows in the test, leaking the future. For data "
                    "that changes over time, evaluate on a chronological hold-out instead."
                ),
                columns=date_columns,
            )
        )
    return findings


def constant_features(frame: pd.DataFrame, feature_columns: list[str]) -> list[LeakageFinding]:
    """Columns with no spread: useless and, when they survive, a hazard."""
    findings: list[LeakageFinding] = []
    for column in feature_columns:
        if column not in frame.columns:
            continue
        series = frame[column]
        if series.dropna().nunique(dropna=True) <= 1:
            findings.append(
                LeakageFinding(
                    code="constant",
                    severity="warning",
                    message=(
                        f"Column '{column}' is (nearly) constant, so it carries no information "
                        "and can only add noise. Drop it before training."
                    ),
                    columns=[column],
                )
            )
    return findings


def high_cardinality(frame: pd.DataFrame, feature_columns: list[str]) -> list[LeakageFinding]:
    """Categorical columns whose one-hot expansion would explode the feature space."""
    findings: list[LeakageFinding] = []
    threshold = max(50, int(len(frame) * 0.05))
    for column in feature_columns:
        if column not in frame.columns:
            continue
        series = frame[column]
        if not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            uniques = _n_unique(series)
            if uniques > threshold:
                findings.append(
                    LeakageFinding(
                        code="high_cardinality",
                        severity="info",
                        message=(
                            f"Column '{column}' has {uniques} distinct values. One-hot encoding "
                            "this would create that many columns; consider label-encoding, "
                            "frequency-encoding, or dropping it."
                        ),
                        columns=[column],
                    )
                )
    return findings


def run_leakage_checks(
    frame: pd.DataFrame,
    *,
    target_column: str | None = None,
    feature_columns: list[str] | None = None,
    train: pd.DataFrame | None = None,
    test: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run every check and return a JSON-safe summary the API renders.

    ``severity`` is the highest severity found; ``findings`` is the full list.
    Checks that need a train/test split only run when both frames are given.
    """
    features = [c for c in (feature_columns or []) if c in frame.columns]
    findings: list[LeakageFinding] = []

    if target_column and target_column in frame.columns:
        findings.extend(target_leakage(frame, target_column, features))
        findings.extend(temporal_leakage(frame, target_column))
    findings.extend(identifier_leakage(frame, features))
    findings.extend(constant_features(frame, features))
    findings.extend(high_cardinality(frame, features))
    if train is not None and test is not None:
        findings.extend(train_test_overlap(train, test))

    payload: dict[str, Any] = {
        "severity": severity(findings),
        "count": len(findings),
        "findings": [finding.to_dict() for finding in findings],
    }
    return json_safe(payload)

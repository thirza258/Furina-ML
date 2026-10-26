"""Furina ML: forecast a generated daily series with chronological validation.

Run: python time_series_forecasting.py --horizon 14 --output-dir time_series_results
Uses generated teaching data, not a real-world performance benchmark.

The script generates three years of daily parcel volume for one imaginary
depot. The series has a trend, a slowly wandering level, a weekly cycle, a
yearly cycle, scheduled promotions, public holidays and autocorrelated noise.
The training, validation and test partitions are consecutive periods of that
one series; rows are never shuffled.

Lag and rolling-mean features are built only from days before the forecast
day. A last-value baseline and a seasonal-naive baseline are compared with
Ridge regression and HistGradientBoostingRegressor. Candidates are selected
with expanding-window (rolling-origin) backtests inside the validation period.
The selected model is refitted on training plus validation days and evaluated
once on the test period with MAE, RMSE and MASE.

Tree models cannot extrapolate: once a feature exceeds the largest value seen
in training, increasing it further does not change their prediction. One
boosting candidate therefore uses raw levels, and another expresses lags,
means and the target relative to the mean of the previous seven days. The
report also records how many test days have a recent level above anything in
the fitted data, a warning sign that needs no test labels.

Multi-step forecasts are recursive: one model predicts the next day, and that
prediction becomes an input for the day after. Only one model is needed, but
an early error is fed back as an input, so errors can accumulate along the
horizon. A direct strategy trains a separate model for each horizon step and
avoids that feedback, at the cost of more models and fewer usable rows for
the longer steps.

Two quantile models (loss="quantile") form a central prediction interval.
They are fitted on training days and calibrated on validation days with a
conformal adjustment, so they are deliberately not refitted on validation
days. Empirical coverage is then measured once on the test period.

Requires numpy and scikit-learn; joblib is installed as a scikit-learn
dependency. The script makes no network requests. model.joblib is a pickle
file, so only load model files you trust.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import time
from datetime import date, timedelta
from pathlib import Path

# These fits are small, so extra OpenMP threads add overhead rather than speed.
# The variable must be set before NumPy and scikit-learn load; export your own
# OMP_NUM_THREADS value to override it.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, root_mean_squared_error  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

DATA_SEED = 2024  # Fixed, so the training seed never changes the series or the split.
START_DATE = date(2023, 1, 2)  # A Monday.
TRAIN_DAYS, VALIDATION_DAYS, TEST_DAYS = 756, 168, 168  # 108, 24 and 24 weeks.
VALIDATION_START = TRAIN_DAYS
TEST_START = TRAIN_DAYS + VALIDATION_DAYS
TOTAL_DAYS = TEST_START + TEST_DAYS
SEASON = 7  # Weekly seasonality in daily data.
MAX_HORIZON = 28
LAGS = (1, 2, 3, 7, 14, 21, 28)
WINDOWS = (7, 28)
HISTORY = max(LAGS + WINDOWS)  # Days needed before the first usable feature row.
WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
WEEKLY_EFFECT = np.array([18.0, 12.0, 8.0, 10.0, 22.0, -28.0, -42.0])
HOLIDAYS = ((1, 1), (5, 1), (12, 25), (12, 26))
# Known-in-advance inputs: the calendar and the published promotion schedule.
KNOWN_NAMES = [f"weekday_{name}" for name in WEEKDAYS] + [
    "year_sin", "year_cos", "holiday", "promotion", "days_since_start"]
FEATURE_NAMES = ([f"lag_{lag}" for lag in LAGS]
                 + [f"mean_previous_{window}" for window in WINDOWS] + KNOWN_NAMES)
LEVEL_COLUMN = FEATURE_NAMES.index("mean_previous_7")
HISTORY_COLUMNS = [index for index, name in enumerate(FEATURE_NAMES)
                   if name.startswith(("lag_", "mean_previous_"))]


def make_series(trend_per_day):
    """Return dates, daily parcel counts and the known-in-advance calendar inputs."""
    rng = np.random.default_rng(DATA_SEED)
    days = np.arange(TOTAL_DAYS)
    dates = [START_DATE + timedelta(days=int(day)) for day in days]
    weekday = np.array([day.weekday() for day in dates])
    day_of_year = np.array([day.timetuple().tm_yday for day in dates])
    holiday = np.array([(day.month, day.day) in HOLIDAYS for day in dates], dtype=float)
    promotion = np.zeros(TOTAL_DAYS)
    day = int(rng.integers(5, 20))
    while day < TOTAL_DAYS:  # Promotions are scheduled, and published, in advance.
        promotion[day] = 0.0 if holiday[day] else 1.0
        day += int(rng.integers(18, 41))
    level = 220.0 + trend_per_day * days + np.cumsum(rng.normal(0.0, 0.8, TOTAL_DAYS))
    yearly = 14.0 * np.cos(2 * np.pi * (day_of_year - 350) / 365.25)
    expected = (level + WEEKLY_EFFECT[weekday] + yearly) * np.where(holiday == 1, 0.35, 1.0)
    expected *= np.where(promotion == 1, 1.3, 1.0)
    shocks = rng.normal(0.0, 6.0, TOTAL_DAYS)
    noise = np.zeros(TOTAL_DAYS)
    noise[0] = shocks[0] / math.sqrt(1 - 0.5 ** 2)
    for index in range(1, TOTAL_DAYS):  # AR(1) noise: part of each surprise persists.
        noise[index] = 0.5 * noise[index - 1] + shocks[index]
    values = np.maximum(np.round(expected + noise), 0.0)
    angle = 2 * np.pi * day_of_year / 365.25
    known = np.column_stack([np.eye(7)[weekday], np.sin(angle), np.cos(angle),
                             holiday, promotion, days.astype(float)])
    return dates, values, known


def windows_before(values, days):
    """For each target day t, return values[t - HISTORY:t]: strictly earlier days."""
    days = np.asarray(days)
    if days.min() < HISTORY or days.max() > len(values):
        raise ValueError("Every target day needs HISTORY earlier days.")
    view = np.lib.stride_tricks.sliding_window_view(values, HISTORY)
    return np.ascontiguousarray(view[days - HISTORY])


def features(windows, known_rows):
    """Build model inputs. The same function serves training, backtests and forecasts."""
    lags = windows[:, [-lag for lag in LAGS]]
    means = np.column_stack([windows[:, -window:].mean(axis=1) for window in WINDOWS])
    return np.column_stack([lags, means, known_rows])


def relative(rows):
    """Express every lag and mean relative to the previous 7-day mean."""
    shifted = rows.copy()
    shifted[:, HISTORY_COLUMNS] -= rows[:, [LEVEL_COLUMN]]
    return shifted


def fit(spec, inputs, target):
    """Fit a forecaster: a scikit-learn estimator plus the framing of its inputs and target."""
    if spec["framing"] == "relative":
        inputs, target = relative(inputs), target - inputs[:, LEVEL_COLUMN]
    return {"estimator": clone(spec["estimator"]).fit(inputs, target), "framing": spec["framing"]}


def predict(model, rows):
    if model["framing"] == "relative":
        return model["estimator"].predict(relative(rows)) + rows[:, LEVEL_COLUMN]
    return model["estimator"].predict(rows)


def recursive_forecast(model, values, known, origins, horizon, interval_models=(), inputs=None):
    """Forecast `horizon` days from each origin, reading only values before the origin.

    Forecast step s targets day origin + s. Lags that fall on or after the origin
    use earlier forecasts, so a mistake at step 0 becomes an input at step 1.
    """
    origins = np.asarray(origins)
    window = windows_before(values, origins)
    forecast = np.empty((len(origins), horizon))
    bounds = [np.empty((len(origins), horizon)) for _ in interval_models]
    for step in range(horizon):
        rows = features(window, known[origins + step])
        if inputs is not None:
            inputs.append(rows)
        forecast[:, step] = predict(model, rows)
        for output, interval_model in zip(bounds, interval_models):
            output[:, step] = predict(interval_model, rows)
        window = np.concatenate([window[:, 1:], forecast[:, step:step + 1]], axis=1)
    return forecast, bounds


def baseline_forecasts(values, origins, horizon):
    """Last-value and seasonal-naive forecasts that read only observed days."""
    origins = np.asarray(origins)[:, None]
    steps = np.arange(horizon)[None, :]
    last_index = np.repeat(origins - 1, horizon, axis=1)
    # The latest observed day with the target's weekday. For steps of 7 or more,
    # origin + step - 7 would be on or after the origin, which is not yet observed.
    seasonal_index = origins - SEASON + steps % SEASON
    assert (last_index < origins).all() and (seasonal_index < origins).all()
    assert ((origins + steps - seasonal_index) % SEASON == 0).all()
    return {"seasonal_naive": values[seasonal_index], "last_value": values[last_index]}


def actuals(values, origins, horizon):
    return values[np.asarray(origins)[:, None] + np.arange(horizon)[None, :]]


def seasonal_naive_scale(values, stop):
    """MASE scale: in-sample MAE of one-step seasonal-naive forecasts before `stop`."""
    history = values[:stop]
    return float(np.mean(np.abs(history[SEASON:] - history[:-SEASON])))


def point_metrics(actual, forecast, scale):
    actual, forecast = actual.ravel(), forecast.ravel()
    mae = mean_absolute_error(actual, forecast)
    return {"mae": float(mae), "rmse": float(root_mean_squared_error(actual, forecast)),
            "mase": float(mae / scale), "mean_error": float(np.mean(actual - forecast)),
            "forecasts": int(actual.size)}


def boosting(seed, **settings):
    # early_stopping=False: the automatic early-stopping holdout is a random, not
    # chronological, split. max_features < 1 samples features per split, so --seed matters.
    return HistGradientBoostingRegressor(early_stopping=False, max_features=0.8,
                                         random_state=seed, **settings)


def candidates(seed):
    """Fixed candidate list. Scaling sits inside the Ridge pipeline, so every fit
    estimates feature means and scales from its own training window only."""
    tree_settings = {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 15,
                     "l2_regularization": 1.0}
    return [
        ("ridge_alpha_1", {"model": "Ridge", "alpha": 1.0, "framing": "levels"},
         {"estimator": make_pipeline(StandardScaler(), Ridge(alpha=1.0)), "framing": "levels"}),
        ("ridge_alpha_30", {"model": "Ridge", "alpha": 30.0, "framing": "levels"},
         {"estimator": make_pipeline(StandardScaler(), Ridge(alpha=30.0)), "framing": "levels"}),
        ("boosting_levels", {"model": "HistGradientBoostingRegressor", **tree_settings, "framing": "levels"},
         {"estimator": boosting(seed, **tree_settings), "framing": "levels"}),
        ("boosting_relative", {"model": "HistGradientBoostingRegressor", **tree_settings,
                               "framing": "relative to the previous 7-day mean"},
         {"estimator": boosting(seed, **tree_settings), "framing": "relative"}),
    ]


def quantile_spec(quantile, seed):
    return {"framing": "relative", "estimator": boosting(
        seed, loss="quantile", quantile=quantile, learning_rate=0.05, max_iter=200,
        max_leaf_nodes=7, min_samples_leaf=40, l2_regularization=1.0)}


def fold_plan(folds, horizon):
    """Expanding-window folds whose evaluation blocks tile the validation period.

    Each fold trains on every day before its block. Origins advance one day at a
    time, and each forecast window stays inside its own block, so no fold scores
    a day from a later block or from the test period.
    """
    splitter = TimeSeriesSplit(n_splits=folds, test_size=VALIDATION_DAYS // folds)
    plan = []
    for train_index, block in splitter.split(np.zeros((TEST_START, 1))):
        cutoff, block_end = int(block[0]), int(block[-1]) + 1
        assert train_index[0] == 0 and train_index[-1] == cutoff - 1
        plan.append({"cutoff": cutoff, "block_end": block_end,
                     "origins": np.arange(cutoff, block_end - horizon + 1)})
    tiled = np.concatenate([np.arange(fold["cutoff"], fold["block_end"]) for fold in plan])
    assert np.array_equal(tiled, np.arange(VALIDATION_START, TEST_START))
    return plan


def check_partitions(dates):
    bounds = [(0, VALIDATION_START), (VALIDATION_START, TEST_START), (TEST_START, TOTAL_DAYS)]
    assert all(left[1] == right[0] for left, right in zip(bounds, bounds[1:]))
    assert all(later - earlier == timedelta(days=1) for earlier, later in zip(dates, dates[1:]))
    return True


def check_features_use_only_past(values, known, all_features):
    """Hide every value from day t onward; the features for day t must not change."""
    for day in range(HISTORY, TOTAL_DAYS):
        hidden = values.copy()
        hidden[day:] = np.nan  # Any read of day t or later would put NaN in the row.
        rebuilt = features(windows_before(hidden, [day]), known[[day]])
        if not np.array_equal(rebuilt[0], all_features[day - HISTORY]):
            raise AssertionError(f"Features for day {day} read a value from that day or later.")
    return True


def check_forecasts_ignore_future(model, values, known, origins, horizon):
    """Hide values from the origin onward; the inputs and forecasts must not change."""
    for origin in origins:
        hidden = values.copy()
        hidden[origin:] = np.nan
        seen, blind = [], []
        expected, _ = recursive_forecast(model, values, known, [origin], horizon, inputs=seen)
        produced, _ = recursive_forecast(model, hidden, known, [origin], horizon, inputs=blind)
        if not (np.array_equal(expected, produced)
                and all(np.array_equal(a, b) for a, b in zip(seen, blind))):
            raise AssertionError(f"The forecast from day {origin} read a value from that day or later.")
    return True


def conformal_margin(lower, upper, actual, coverage):
    """Split-conformal adjustment for a quantile interval, computed from held-out days.

    Each score is how far the actual value fell outside the interval (negative when
    inside). Widening both bounds by the chosen score aims for the nominal coverage
    when held-out and future errors behave alike, which drifting series can violate.
    """
    scores = np.maximum(lower - actual, actual - upper)
    rank = min(math.ceil((len(scores) + 1) * coverage), len(scores))
    return float(np.sort(scores)[rank - 1])


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=7, help=(
        "Training randomness for gradient boosting feature subsampling. The series uses "
        "a separate fixed seed, so the data and split stay the same (default: %(default)s)."))
    parser.add_argument("--horizon", type=int, default=14, help=(
        "Days forecast from each origin, from 1 to 28 (default: %(default)s)."))
    parser.add_argument("--folds", type=int, default=6, choices=(2, 3, 4, 6), help=(
        "Expanding-window backtest folds tiling the 168 validation days (default: %(default)s)."))
    parser.add_argument("--interval", type=float, default=0.8, help=(
        "Nominal central interval coverage from 0.5 to 0.95; 0.8 fits the 10%% and 90%% "
        "quantiles (default: %(default)s)."))
    parser.add_argument("--trend-per-day", type=float, default=0.1, help=(
        "Generated trend in parcels per day, from -0.1 to 0.5. It changes all three "
        "partitions (default: %(default)s)."))
    parser.add_argument("--output-dir", type=Path, default=Path("time_series_results"), help=(
        "Folder for model.joblib, report.json and forecasts.csv (default: %(default)s)."))
    args = parser.parse_args()
    block = VALIDATION_DAYS // args.folds
    if not 1 <= args.horizon <= min(MAX_HORIZON, block):
        parser.error(f"horizon must be from 1 to {min(MAX_HORIZON, block)} with {args.folds} folds")
    if not (math.isfinite(args.interval) and 0.5 <= args.interval <= 0.95):
        parser.error("interval must be from 0.5 to 0.95")
    if not (math.isfinite(args.trend_per_day) and -0.1 <= args.trend_per_day <= 0.5):
        parser.error("trend-per-day must be from -0.1 to 0.5")

    dates, values, known = make_series(args.trend_per_day)
    series_sha256 = hashlib.sha256(values.tobytes()).hexdigest()
    all_features = features(windows_before(values, np.arange(HISTORY, TOTAL_DAYS)), known[HISTORY:])
    checks = {"partitions_consecutive_and_disjoint": check_partitions(dates),
              "features_use_only_earlier_days": check_features_use_only_past(values, known, all_features)}

    def rows(start, stop):
        days = np.arange(max(start, HISTORY), stop)
        return all_features[days - HISTORY], values[days]

    # Model selection uses validation backtests only; no test day is fitted or scored here.
    plan = fold_plan(args.folds, args.horizon)
    assert max(fold["block_end"] for fold in plan) <= TEST_START
    fold_scales = [seasonal_naive_scale(values, fold["cutoff"]) for fold in plan]
    validation_actual = [actuals(values, fold["origins"], args.horizon) for fold in plan]
    validation = {name: [baseline_forecasts(values, fold["origins"], args.horizon)[name] for fold in plan]
                  for name in ("seasonal_naive", "last_value")}
    settings, specs = {}, {}
    for name, description, spec in candidates(args.seed):
        settings[name], specs[name] = description, spec
        validation[name] = [recursive_forecast(fit(spec, *rows(0, fold["cutoff"])), values, known,
                                               fold["origins"], args.horizon)[0] for fold in plan]
    summary = {}
    for name, forecasts in validation.items():
        errors = np.concatenate([np.abs(a - f).ravel() for a, f in zip(validation_actual, forecasts)])
        fold_mae = [float(np.mean(np.abs(a - f))) for a, f in zip(validation_actual, forecasts)]
        summary[name] = {"validation_mae": float(np.mean(errors)), "fold_mae": fold_mae,
                         "fold_mase": [mae / scale for mae, scale in zip(fold_mae, fold_scales)]}
    model_names = list(settings)
    selected = min(model_names, key=lambda name: (summary[name]["validation_mae"], model_names.index(name)))
    beats_baseline = summary[selected]["validation_mae"] < summary["seasonal_naive"]["validation_mae"]

    # Intervals: fit on training days, then calibrate one-step intervals on validation days.
    quantiles = ((1 - args.interval) / 2, (1 + args.interval) / 2)
    interval_models = [fit(quantile_spec(q, args.seed), *rows(0, VALIDATION_START)) for q in quantiles]
    validation_x, validation_y = rows(VALIDATION_START, TEST_START)
    raw_lower, raw_upper = (predict(model, validation_x) for model in interval_models)
    raw_low, raw_high = np.minimum(raw_lower, raw_upper), np.maximum(raw_lower, raw_upper)
    margin = conformal_margin(raw_low, raw_high, validation_y, args.interval)
    validation_interval = {
        "uncalibrated_coverage": float(np.mean((validation_y >= raw_low) & (validation_y <= raw_high))),
        "uncalibrated_mean_width": float(np.mean(raw_high - raw_low)),
        "crossings": int(np.sum(raw_lower > raw_upper)), "conformal_margin": margin}

    # Refit the selected model once on training plus validation days; then score test once.
    final_model = fit(specs[selected], *rows(0, TEST_START))
    test_origins = np.arange(TEST_START, TOTAL_DAYS - args.horizon + 1)
    first_inputs = []
    test_forecast, test_bounds = recursive_forecast(final_model, values, known, test_origins,
                                                    args.horizon, interval_models, first_inputs)
    checks["forecast_inputs_match_training_features"] = bool(
        np.array_equal(first_inputs[0], all_features[test_origins - HISTORY]))
    assert checks["forecast_inputs_match_training_features"]
    probes = sorted({int(test_origins[0]), int(test_origins[len(test_origins) // 2]), int(test_origins[-1])})
    checks["forecasts_ignore_values_from_origin_onward"] = check_forecasts_ignore_future(
        final_model, values, known, probes, args.horizon)
    test_actual = actuals(values, test_origins, args.horizon)
    test_baselines = baseline_forecasts(values, test_origins, args.horizon)
    checks["baselines_use_only_observed_days"] = True  # Asserted inside baseline_forecasts.
    scale = seasonal_naive_scale(values, TEST_START)
    test_results = {"selected_model": point_metrics(test_actual, test_forecast, scale)}
    test_results.update({name: point_metrics(test_actual, forecast, scale)
                         for name, forecast in test_baselines.items()})
    mae_by_step = {name: np.mean(np.abs(test_actual - forecast), axis=0).tolist() for name, forecast in
                   (("selected_model", test_forecast), *test_baselines.items())}

    test_x, test_y = rows(TEST_START, TOTAL_DAYS)
    one_step = [predict(model, test_x) for model in interval_models]
    low, high = np.minimum(*one_step) - margin, np.maximum(*one_step) + margin
    test_interval = {
        "empirical_coverage": float(np.mean((test_y >= low) & (test_y <= high))),
        "mean_width": float(np.mean(high - low)), "crossings": int(np.sum(one_step[0] > one_step[1])),
        "pinball_loss_lower": float(mean_pinball_loss(test_y, low, alpha=quantiles[0])),
        "pinball_loss_upper": float(mean_pinball_loss(test_y, high, alpha=quantiles[1])),
        "days": int(test_y.size)}
    path_low = np.minimum(*test_bounds) - margin
    path_high = np.maximum(*test_bounds) + margin
    coverage_by_step = np.mean((test_actual >= path_low) & (test_actual <= path_high), axis=0).tolist()
    # A label-free monitoring signal: inputs beyond the fitted range, where trees go flat.
    fitted_level_max = float(all_features[:TEST_START - HISTORY, LEVEL_COLUMN].max())
    above_range = float(np.mean(test_x[:, LEVEL_COLUMN] > fitted_level_max))

    # Save, reload and rebuild the forecasts from the artifact and regenerated raw data.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output_dir / "model.joblib"
    joblib.dump({"format_version": 1, "point_model": final_model, "interval_models": interval_models,
                 "quantiles": quantiles, "conformal_margin": margin, "selected_candidate": selected,
                 "feature_names": FEATURE_NAMES, "lags": LAGS, "windows": WINDOWS,
                 "history_days": HISTORY, "horizon": args.horizon,
                 "point_model_fitted_through": dates[TEST_START - 1].isoformat(),
                 "interval_models_fitted_through": dates[VALIDATION_START - 1].isoformat(),
                 "data": {"data_seed": DATA_SEED, "trend_per_day": args.trend_per_day,
                          "series_sha256": series_sha256}}, artifact_path)
    artifact = joblib.load(artifact_path)  # A pickle: only load files you trust.
    if (artifact["feature_names"] != FEATURE_NAMES or tuple(artifact["lags"]) != LAGS
            or tuple(artifact["windows"]) != WINDOWS):
        raise ValueError("The saved model expects a different feature recipe.")
    _, regenerated, regenerated_known = make_series(artifact["data"]["trend_per_day"])
    assert hashlib.sha256(regenerated.tobytes()).hexdigest() == artifact["data"]["series_sha256"]
    reloaded_forecast, reloaded_bounds = recursive_forecast(
        artifact["point_model"], regenerated, regenerated_known, test_origins,
        artifact["horizon"], artifact["interval_models"])
    assert np.array_equal(reloaded_forecast, test_forecast)
    assert all(np.array_equal(a, b) for a, b in zip(reloaded_bounds, test_bounds))
    assert artifact["conformal_margin"] == margin
    checks["model_reload_identical_forecasts"] = True

    with (args.output_dir / "forecasts.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["origin_date", "target_date", "step", "target_holiday", "target_promotion",
                         "actual", "forecast", "interval_lower", "interval_upper", "seasonal_naive",
                         "last_value"])
        holiday_column, promotion_column = KNOWN_NAMES.index("holiday"), KNOWN_NAMES.index("promotion")
        for i, origin in enumerate(test_origins):
            for step in range(args.horizon):
                target = origin + step
                writer.writerow([dates[origin].isoformat(), dates[target].isoformat(), step + 1,
                                 int(known[target, holiday_column]), int(known[target, promotion_column]),
                                 test_actual[i, step], round(test_forecast[i, step], 3),
                                 round(path_low[i, step], 3), round(path_high[i, step], 3),
                                 test_baselines["seasonal_naive"][i, step],
                                 test_baselines["last_value"][i, step]])

    def partition(start, stop):
        holidays = known[start:stop, KNOWN_NAMES.index("holiday")]
        promotions = known[start:stop, KNOWN_NAMES.index("promotion")]
        return {"first_date": dates[start].isoformat(), "last_date": dates[stop - 1].isoformat(),
                "days": stop - start, "holidays": int(holidays.sum()), "promotions": int(promotions.sum())}

    report = {
        "task": "Generated daily parcel volume; chronological forecasting teaching lab.",
        "scope": "Real model fitting on generated data. Results describe this teaching series only.",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
        "seed": args.seed, "data_seed": DATA_SEED, "series_sha256": series_sha256,
        "settings": {"horizon_days": args.horizon, "folds": args.folds, "origin_spacing_days": 1,
                     "seasonal_period_days": SEASON, "interval": args.interval,
                     "quantiles": list(quantiles), "trend_per_day": args.trend_per_day,
                     "feature_names": FEATURE_NAMES},
        "partitions": {"training": partition(0, VALIDATION_START),
                       "validation": partition(VALIDATION_START, TEST_START),
                       "test": partition(TEST_START, TOTAL_DAYS)},
        "selection": {
            "metric": "Pooled MAE over every validation origin and horizon step; lower is better; "
                      "ties keep the earlier candidate.",
            "folds": [{"fit_through": dates[fold["cutoff"] - 1].isoformat(),
                       "scored": [dates[fold["cutoff"]].isoformat(), dates[fold["block_end"] - 1].isoformat()],
                       "origins": int(len(fold["origins"]))} for fold in plan],
            "candidates": {name: {**settings[name], **summary[name]} for name in model_names},
            "baselines": {name: summary[name] for name in ("seasonal_naive", "last_value")},
            "selected": selected, "selected_beats_seasonal_naive": bool(beats_baseline),
        },
        "test": {
            "protocol": "The selected model was refitted once on training and validation days and "
                        "then frozen. Origins advance one day at a time through the test period, and "
                        "each forecast reads only values observed before its origin. Validation folds "
                        "refitted at every block, so the frozen test model grows staler than they did.",
            "origins": int(len(test_origins)), "mase_scale": scale,
            "metrics": test_results, "mae_by_step": mae_by_step,
        },
        "intervals": {
            "method": "Two HistGradientBoostingRegressor models with loss='quantile', fitted on "
                      "training days and widened by a conformal margin from validation days.",
            "validation_one_step": validation_interval, "test_one_step": test_interval,
            "test_coverage_by_step": coverage_by_step,
            "note": "Steps after the first feed recursive point forecasts into one-step quantile "
                    "models. That ignores accumulated error, so the by-step coverage is a diagnostic, "
                    "not a calibrated multi-step interval.",
        },
        "monitoring": {
            "largest_previous_7_day_mean_in_fitted_days": fitted_level_max,
            "share_of_test_days_above_that_level": above_range,
            "note": "Uses inputs that exist at forecast time and no test labels. Tree models "
                    "predict a flat continuation beyond the largest input value they were fitted on.",
        },
        "checks": checks,
        "runtime_seconds": round(time.perf_counter() - started, 2),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    printed = {
        "selected_model": selected,
        "validation_mae": {name: round(summary[name]["validation_mae"], 3) for name in summary},
        "selected_beats_seasonal_naive_on_validation": bool(beats_baseline),
        "test": {name: {key: round(value, 3) for key, value in result.items() if key != "forecasts"}
                 for name, result in test_results.items()},
        "test_mae_first_and_last_step": [round(mae_by_step["selected_model"][0], 3),
                                         round(mae_by_step["selected_model"][-1], 3)],
        "interval_coverage": {"nominal": args.interval,
                              "validation_uncalibrated": round(validation_interval["uncalibrated_coverage"], 3),
                              "test_one_step": round(test_interval["empirical_coverage"], 3),
                              "test_last_step": round(coverage_by_step[-1], 3)},
        "share_of_test_days_above_fitted_level": round(above_range, 3),
        "checks_passed": all(checks.values()), "runtime_seconds": report["runtime_seconds"],
    }
    print(json.dumps(printed, indent=2))
    print(f"Artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

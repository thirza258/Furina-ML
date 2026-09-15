#!/usr/bin/env python3
"""A local ML-system teaching lab. Python 3.10+, standard library only.

Model fitting and predictions are real. Request latency and infrastructure
failures are simulated; this is not a web server or a throughput benchmark.
All event timestamps use synthetic minutes. Latency fields use milliseconds.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics

FEATURE_VERSION = "delivery-features-v1"
FEATURE_NAMES = ["distance_km", "congestion", "rain"]
MAX_FEATURE_AGE = 30
MAX_LABEL_DELAY = 540
REQUEST_DEADLINE_MS = 100
CANDIDATE_BUDGET_MS = 80
TRAIN = slice(0, 900)
VALIDATION = slice(1000, 1250)
TEST = slice(1350, 1600)
LIVE = slice(1700, 2100)


@dataclass(frozen=True)
class Feature:
    entity: int
    event_time: int
    available_at: int
    congestion: float
    version: int


@dataclass(frozen=True)
class Delivery:
    request_id: str
    entity: int
    prediction_time: int
    distance_km: float
    rain: int
    label: int
    label_available_at: int


def sigmoid(value):
    value = max(-40.0, min(40.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def as_of(history, entity, prediction_time, max_age=MAX_FEATURE_AGE):
    eligible = [
        row for row in history.get(entity, [])
        if row.event_time <= prediction_time
        and row.available_at <= prediction_time
        and prediction_time - row.event_time <= max_age
    ]
    return max(eligible, key=lambda row: (
        row.event_time, row.available_at, row.version
    ), default=None)


def make_data(distance_shift=0.0):
    """Separate fixed data RNG makes comparisons use the same event identities."""
    rng = random.Random(42)
    deliveries, history = [], {}
    for index in range(2100):
        timestamp, entity = index * 10, index % 24
        distance = rng.uniform(1.0, 30.0)
        if index >= LIVE.start:
            distance += distance_shift
        congestion, rain = rng.random(), int(rng.random() < 0.25)
        probability = sigmoid(-4.5 + 0.11 * distance + 3.0 * congestion + 0.8 * rain)
        label = int(rng.random() < probability)
        delay = 180 + rng.randrange(120) + 240 * label
        deliveries.append(Delivery(
            f"delivery-{index}", entity, timestamp, distance, rain,
            label, timestamp + delay,
        ))
        # Some current values arrive too late for this request.
        available_at = timestamp + 8 if index % 17 == 0 else timestamp - 1
        history.setdefault(entity, []).extend([
            Feature(entity, timestamp - 4, available_at, congestion, 1),
            # A later correction must not change this historical prediction.
            Feature(entity, timestamp - 2, timestamp + 20, 1.0 - congestion, 2),
        ])
    return deliveries, history


def raw_features(delivery, feature):
    return [delivery.distance_km, feature.congestion, float(delivery.rain)]


def transform(values, model):
    return [(value - mean) / scale for value, mean, scale
            in zip(values, model["means"], model["scales"])]


def predict(values, model):
    return sigmoid(model["bias"] + sum(
        weight * value for weight, value in zip(model["weights"], transform(values, model))
    ))


def fit(rows, history, steps, seed):
    pairs = [(row, as_of(history, row.entity, row.prediction_time)) for row in rows]
    pairs = [(row, feature) for row, feature in pairs if feature is not None]
    values = [raw_features(row, feature) for row, feature in pairs]
    columns = list(zip(*values))
    model = {
        "version": "candidate-logistic-v1",
        "means": [statistics.mean(column) for column in columns],
        "scales": [max(statistics.pstdev(column), 1e-8) for column in columns],
        "weights": [0.0] * len(FEATURE_NAMES),
        "bias": 0.0,
    }
    normalized = [transform(value, model) for value in values]
    rng = random.Random(seed)
    for _ in range(steps):
        indices = rng.sample(range(len(pairs)), min(64, len(pairs)))
        errors = [
            sigmoid(model["bias"] + sum(w * x for w, x in zip(model["weights"], normalized[i])))
            - pairs[i][0].label for i in indices
        ]
        for column in range(len(FEATURE_NAMES)):
            gradient = sum(error * normalized[i][column] for error, i in zip(errors, indices)) / len(indices)
            model["weights"][column] -= 0.08 * (gradient + 0.001 * model["weights"][column])
        model["bias"] -= 0.08 * statistics.mean(errors)
    return model, len(pairs)


def metrics(labels, scores, decisions):
    if not labels:
        return {"count": 0, "note": "No eligible labels; quality is not observable."}
    tp = sum(y == 1 and decision == 1 for y, decision in zip(labels, decisions))
    fp = sum(y == 0 and decision == 1 for y, decision in zip(labels, decisions))
    fn = sum(y == 1 and decision == 0 for y, decision in zip(labels, decisions))
    tn = len(labels) - tp - fp - fn
    loss = -statistics.mean(
        y * math.log(max(score, 1e-12)) + (1 - y) * math.log(max(1 - score, 1e-12))
        for y, score in zip(labels, scores)
    )
    return {
        "count": len(labels), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "log_loss": loss, "cost_per_delivery": (fp + 4 * fn) / len(labels),
    }


def scores_for(rows, history, model, prior):
    result = []
    for row in rows:
        feature = as_of(history, row.entity, row.prediction_time)
        result.append(predict(raw_features(row, feature), model) if feature else prior)
    return result


def evaluate_policy(rows, history, package, candidate):
    prior = package["baseline"]["probability"]
    baseline_threshold = package["baseline"]["threshold"]
    if not candidate:
        scores = [prior] * len(rows)
        decisions = [int(prior >= baseline_threshold)] * len(rows)
    else:
        scores = scores_for(rows, history, package["candidate"], prior)
        decisions = []
        for row, score in zip(rows, scores):
            feature = as_of(history, row.entity, row.prediction_time)
            threshold = package["candidate"]["threshold"] if feature else baseline_threshold
            decisions.append(int(score >= threshold))
    return metrics([row.label for row in rows], scores, decisions)


def choose_threshold(labels, scores):
    # Declared validation objective: false positive cost 1, false negative cost 4.
    # Higher threshold is the declared tie breaker.
    return min(
        [i / 100 for i in range(101)],
        key=lambda threshold: (
            sum((1 if y == 0 else 4) for y, score in zip(labels, scores)
                if int(score >= threshold) != y),
            -threshold,
        ),
    )


def bucket(request_id):
    # Stable across Python processes; built-in hash() would not be.
    return int(hashlib.sha256(request_id.encode()).hexdigest()[:8], 16) % 100


def verify_contracts():
    history = {1: [
        Feature(1, 90, 95, 0.2, 1),
        Feature(1, 98, 105, 0.9, 2),  # past event, future availability
        Feature(1, 101, 99, 0.8, 3),  # future event, earlier availability
    ]}
    assert as_of(history, 1, 100).congestion == 0.2
    assert as_of(history, 1, 100, max_age=5) is None
    assert as_of(history, 2, 100) is None
    assert as_of(history, 1, 105).version == 3
    return {"availability_and_event_time": True, "stale_and_missing_features": True}


def verify_partition_timing(deliveries):
    train, validation, test, live = [deliveries[part] for part in (TRAIN, VALIDATION, TEST, LIVE)]
    assert max(row.label_available_at for row in train) < validation[0].prediction_time
    assert max(row.label_available_at for row in validation) < test[0].prediction_time
    assert max(row.label_available_at for row in test) < live[0].prediction_time


def content_version(prefix, values):
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return prefix + "-" + hashlib.sha256(encoded).hexdigest()[:12]


def build_package(deliveries, history, steps, seed):
    train, validation = deliveries[TRAIN], deliveries[VALIDATION]
    model, usable = fit(train, history, steps, seed)
    prior = statistics.mean(row.label for row in train)
    baseline_threshold = choose_threshold([row.label for row in validation], [prior] * len(validation))
    usable_validation = [
        row for row in validation if as_of(history, row.entity, row.prediction_time) is not None
    ]
    model["threshold"] = choose_threshold(
        [row.label for row in usable_validation],
        scores_for(usable_validation, history, model, prior),
    )
    model["version"] = content_version("logistic", {
        key: value for key, value in model.items() if key != "version"
    })
    package = {
        "format_version": 1, "feature_version": FEATURE_VERSION,
        "feature_names": FEATURE_NAMES, "max_feature_age_minutes": MAX_FEATURE_AGE,
        "candidate": model,
        "baseline": {"version": content_version("constant-baseline", [prior, baseline_threshold]),
                     "probability": prior, "threshold": baseline_threshold},
        "training": {
            "seed": seed, "steps": steps, "data_seed": 42,
            "training_rows": len(train), "rows_with_usable_features": usable,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    baseline_result = evaluate_policy(validation, history, package, False)
    candidate_result = evaluate_policy(validation, history, package, True)
    package["validation"] = {"baseline": baseline_result, "candidate": candidate_result}
    package["candidate_eligible"] = (
        candidate_result["cost_per_delivery"] <= baseline_result["cost_per_delivery"] - 0.005
    )
    package["release_rule"] = "Candidate validation cost improves by at least 0.005 per delivery; test data never select."
    return package


def validate_package(package):
    if (package.get("format_version") != 1 or package.get("feature_version") != FEATURE_VERSION
            or package.get("feature_names") != FEATURE_NAMES
            or package.get("max_feature_age_minutes") != MAX_FEATURE_AGE):
        raise ValueError("Saved package is incompatible with this lab's feature contract.")
    model = package["candidate"]
    for name in ("means", "scales", "weights"):
        if len(model[name]) != len(FEATURE_NAMES) or not all(math.isfinite(x) for x in model[name]):
            raise ValueError(f"Invalid model {name}.")
    if any(scale <= 0 for scale in model["scales"]):
        raise ValueError("Feature scales must be positive.")
    if not math.isfinite(model["bias"]):
        raise ValueError("Model bias must be finite.")
    if not all(0 <= value <= 1 for value in (
        model["threshold"], package["baseline"]["probability"], package["baseline"]["threshold"]
    )):
        raise ValueError("Probabilities and thresholds must be in [0, 1].")
    if not isinstance(package["candidate_eligible"], bool):
        raise ValueError("Candidate eligibility must be a boolean.")


def replay(rows, history, package, args):
    rng = random.Random(2026)
    log, rollbacks = [], []
    active = package["candidate_eligible"]
    attempted, timeouts = 0, 0
    prior = package["baseline"]["probability"]
    for index, row in enumerate(rows):
        assigned = active and bucket(row.request_id) < args.canary_percent
        feature, reason, candidate_work_ms = None, None, None
        score, threshold = prior, package["baseline"]["threshold"]
        served_version, response_ms = package["baseline"]["version"], 5.0
        if assigned:
            attempted += 1
            outage = args.feature_outage and len(rows) // 3 <= index < 2 * len(rows) // 3
            feature = None if outage else as_of(history, row.entity, row.prediction_time)
            if feature is None:
                reason = "feature_dependency_unavailable" if outage else "missing_or_stale_feature"
                response_ms = 15.0
            else:
                candidate_work_ms = rng.uniform(25.0, 65.0) + args.candidate_latency_ms
                if candidate_work_ms > CANDIDATE_BUDGET_MS:
                    timeouts += 1
                    reason, response_ms = "candidate_timeout", CANDIDATE_BUDGET_MS + 5.0
                else:
                    score = predict(raw_features(row, feature), package["candidate"])
                    threshold = package["candidate"]["threshold"]
                    served_version, response_ms = package["candidate"]["version"], candidate_work_ms
        log.append({
            "prediction_id": row.request_id + "/dispatch-v1",
            "request_id": row.request_id, "entity": row.entity,
            "prediction_time_minutes": row.prediction_time,
            "distance_km": row.distance_km, "rain": row.rain,
            "feature_version": FEATURE_VERSION,
            "feature": asdict(feature) if feature else None,
            "feature_age_minutes": row.prediction_time - feature.event_time if feature else None,
            "assigned_version": package["candidate"]["version"] if assigned else package["baseline"]["version"],
            "served_version": served_version,
            "score": score, "threshold": threshold, "action": int(score >= threshold),
            "fallback_reason": reason,
            "simulated_candidate_work_ms": candidate_work_ms,
            "simulated_response_ms": response_ms,
        })
        # A technical demonstration gate, not a statistical quality test.
        if active and attempted >= 10 and timeouts / attempted > 0.10:
            active = False
            rollbacks.append({
                "after_request": row.request_id, "candidate_attempts": attempted,
                "timeouts": timeouts, "reason": "Candidate timeout fraction exceeds 10% after at least 10 attempts.",
            })
    return log, rollbacks


def monitoring(rows, log, package, training_distances):
    by_id = {row.request_id: row for row in rows}
    cutoff = rows[-1].prediction_time + 120
    observed = [entry for entry in log if by_id[entry["request_id"]].label_available_at <= cutoff]
    mature = [entry for entry in log if entry["prediction_time_minutes"] <= cutoff - MAX_LABEL_DELAY]
    assert all(by_id[entry["request_id"]].label_available_at <= cutoff for entry in mature)

    def quality(entries):
        return metrics(
            [by_id[entry["request_id"]].label for entry in entries],
            [entry["score"] for entry in entries], [entry["action"] for entry in entries],
        )

    reasons = Counter(entry["fallback_reason"] for entry in log if entry["fallback_reason"])
    latencies = sorted(entry["simulated_response_ms"] for entry in log)
    # Nearest-rank p95 of this synthetic replay, not measured production latency.
    p95 = latencies[math.ceil(0.95 * len(latencies)) - 1]
    shift = ((statistics.mean(row.distance_km for row in rows) - statistics.mean(training_distances))
             / max(statistics.pstdev(training_distances), 1e-8))
    return {
        "requests": len(log),
        "assigned_by_version": dict(Counter(entry["assigned_version"] for entry in log)),
        "served_by_version": dict(Counter(entry["served_version"] for entry in log)),
        "fallback_reasons": dict(reasons), "fallback_rate": sum(reasons.values()) / len(log),
        "simulated_response_p95_ms": p95,
        "simulated_response_deadline_misses": sum(value > REQUEST_DEADLINE_MS for value in latencies),
        "request_deadline_ms": REQUEST_DEADLINE_MS, "candidate_budget_ms": CANDIDATE_BUDGET_MS,
        "labels": {
            "as_of_minutes": cutoff, "observed": len(observed), "eligible_predictions": len(log),
            "coverage": len(observed) / len(log), "maximum_delay_minutes": MAX_LABEL_DELAY,
            "observed_subset_quality": quality(observed),
            "mature_cohort_quality": quality(mature),
            "mature_quality_by_served_version": {
                version: quality([entry for entry in mature if entry["served_version"] == version])
                for version in (package["baseline"]["version"], package["candidate"]["version"])
            },
            "mature_quality_by_assigned_version": {
                version: quality([entry for entry in mature if entry["assigned_version"] == version])
                for version in (package["baseline"]["version"], package["candidate"]["version"])
            },
            "note": "Recent observed labels can be biased toward fast outcomes. Served-version groups are also selected by fallback behavior; inspect assigned groups and cohort sizes. These are descriptive metrics, not a causal product experiment.",
        },
        "data_signal": {
            "distance_mean_shift_in_training_stddevs": shift,
            "investigate": abs(shift) > 1.0,
            "note": "An illustrative diagnostic threshold, not proof of concept drift or quality loss.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400, help="Training updates when creating a package.")
    parser.add_argument("--seed", type=int, default=17, help="Training minibatch seed; source data use a fixed seed.")
    parser.add_argument("--canary-percent", type=int, default=20)
    parser.add_argument("--candidate-latency-ms", type=float, default=0.0, help="Extra simulated candidate work.")
    parser.add_argument("--feature-outage", action="store_true", help="Simulate an unavailable dependency in the middle third.")
    parser.add_argument("--distance-shift-km", type=float, default=0.0, help="Add distance to live deliveries only, including their generated outcomes.")
    parser.add_argument("--load-model", type=Path, help="Replay a saved JSON package without training or threshold selection.")
    parser.add_argument("--output-dir", type=Path, default=Path("ml_system_design_run"))
    args = parser.parse_args()
    if args.steps < 1 or not 0 <= args.canary_percent <= 100:
        parser.error("steps must be positive and canary-percent must be in [0, 100]")
    if any(not math.isfinite(value) or value < 0 for value in (args.candidate_latency_ms, args.distance_shift_km)):
        parser.error("latency and distance shift must be finite and nonnegative")

    checks = verify_contracts()
    deliveries, history = make_data(args.distance_shift_km)
    verify_partition_timing(deliveries)
    package = (json.loads(args.load_model.read_text()) if args.load_model
               else build_package(deliveries, history, args.steps, args.seed))
    validate_package(package)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.json"
    model_path.write_text(json.dumps(package, indent=2) + "\n")
    restored = json.loads(model_path.read_text())
    validate_package(restored)
    fixtures = [[2.0, 0.1, 0.0], [15.0, 0.8, 1.0], [29.0, 0.4, 0.0]]
    assert all(predict(values, package["candidate"]) == predict(values, restored["candidate"]) for values in fixtures)
    checks["package_prediction_round_trip"] = True
    checks["training_validation_and_test_labels_mature_before_next_split"] = True
    log, rollbacks = replay(deliveries[LIVE], history, restored, args)
    assert len({entry["prediction_id"] for entry in log}) == len(log)
    for entry in log:
        feature = entry["feature"]
        if feature:
            assert feature["event_time"] <= entry["prediction_time_minutes"]
            assert feature["available_at"] <= entry["prediction_time_minutes"]
        if entry["served_version"] == restored["candidate"]["version"]:
            values = [entry["distance_km"], feature["congestion"], float(entry["rain"])]
            assert math.isclose(entry["score"], predict(values, restored["candidate"]), abs_tol=1e-12)
    checks["prediction_ids_and_serving_replay"] = True
    if not restored["candidate_eligible"]:
        assert all(entry["assigned_version"] == restored["baseline"]["version"] for entry in log)
    if rollbacks:
        stop = next(i for i, entry in enumerate(log) if entry["request_id"] == rollbacks[0]["after_request"])
        assert all(entry["assigned_version"] == restored["baseline"]["version"] for entry in log[stop + 1:])
    checks["eligibility_and_rollback_stop_exposure"] = True
    report = {
        "scope": "Real local logistic training and predictions; synthetic data, timing and dependency incidents. No real serving benchmark or intervention experiment.",
        "python_version": platform.python_version(), "loaded_existing_package": args.load_model is not None,
        "candidate_eligible": restored["candidate_eligible"], "release_rule": restored["release_rule"],
        "validation": restored["validation"],
        "independent_test": {
            "baseline": evaluate_policy(deliveries[TEST], history, restored, False),
            "candidate": evaluate_policy(deliveries[TEST], history, restored, True),
        },
        "monitoring": monitoring(deliveries[LIVE], log, restored, [row.distance_km for row in deliveries[TRAIN]]),
        "rollbacks": rollbacks, "checks": checks,
    }
    manifest = {
        "data_generator": "delivery-events-v1", "data_seed": 42,
        "timestamp_unit": "synthetic minutes", "feature_version": FEATURE_VERSION,
        "label": "Generated late-delivery outcome; intervention does not alter labels in this lab.",
        "partitions": {
            name: {"start_inclusive": part.start, "stop_exclusive": part.stop,
                   "prediction_time_range": [deliveries[part][0].prediction_time, deliveries[part][-1].prediction_time]}
            for name, part in (("training", TRAIN), ("validation", VALIDATION), ("test", TEST), ("replay", LIVE))
        },
        "scenario": {"canary_percent": args.canary_percent, "candidate_latency_ms": args.candidate_latency_ms,
                     "feature_outage": args.feature_outage, "distance_shift_km": args.distance_shift_km},
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "replay_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_model": str(args.load_model) if args.load_model else None,
    }
    for name, payload in (("report.json", report), ("manifest.json", manifest)):
        (args.output_dir / name).write_text(json.dumps(payload, indent=2) + "\n")
    (args.output_dir / "predictions.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in log))
    print(json.dumps(report, indent=2))
    print(f"Artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

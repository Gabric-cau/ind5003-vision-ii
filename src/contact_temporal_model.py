"""Fixed-protocol contact localization on automatic, label-free video features."""
from collections import Counter
import re

import numpy as np


LAGS_SECONDS = (-.4, -.2, 0., .2, .4)
POSITIVE_RADIUS_SECONDS = .15
NEGATIVE_RADIUS_SECONDS = .35
SMOOTHING_SECONDS = .1
MINIMUM_SUPPORT_SAMPLES = 3
REGULARIZATION_C = .1
SEED = 20261026
EXPECTED_GROUPS = {"col_eng": 4, "cro_den": 2, "esp_rus": 2, "rus_cro": 2}
EVIDENCE_FEATURES = ("pose_visible", "ball_visible", "seed_available")


def validate_record(record):
    provenance = record.get("provenance", {})
    if (provenance.get("human_labels_used") is not False
            or provenance.get("old_contact_used") is not False):
        raise ValueError("Feature provenance must exclude human labels and old contact times")
    if any(key in record for key in ("reference_contact_seconds", "targets", "labels")):
        raise ValueError("Reference labels do not belong in feature records")
    names = record["feature_names"]
    forbidden = re.compile(r"(^|_)(time|timestamp|frame|duration|source|match|event|clip|core|label|target|reference|contact_time|progress)(_|$)")
    if (not names or len(set(names)) != len(names)
            or any(not isinstance(name, str) or forbidden.search(name.lower()) for name in names)):
        raise ValueError("Feature names must exclude absolute clocks, identifiers, and labels")
    if not set(EVIDENCE_FEATURES) <= set(names):
        raise ValueError("Feature schema requires explicit pose, ball, and seed availability masks")
    x = np.asarray(record["features"], dtype=float)
    times = np.asarray(record["time_seconds"], dtype=float)
    frames = np.asarray(record["frame_ids"])
    fps = record["fps"]
    if (not isinstance(fps, (int, float)) or isinstance(fps, bool)
            or not np.isfinite(fps) or fps <= 0):
        raise ValueError("Invalid source FPS")
    if (x.ndim != 2 or x.shape != (len(times), len(names)) or len(times) < 2
            or frames.shape != times.shape or not np.isfinite(x).all()
            or not np.isfinite(times).all() or not np.issubdtype(frames.dtype, np.integer)
            or np.any(np.diff(times) <= 0) or np.any(np.diff(frames) <= 0)
            or not np.allclose(times, frames / fps, rtol=0, atol=1e-6)):
        raise ValueError("Features must have finite aligned source frames and times")
    window = record["window"]
    start, end = window["start_frame"], window["end_frame_exclusive"]
    if not 0 <= start <= frames[0] <= frames[-1] < end:
        raise ValueError("Feature frames lie outside their automatic window")
    for name, value in window.items():
        if any(word in name.lower() for word in ("label", "manual", "reference", "old_contact")) and value is True:
            raise ValueError("Automatic window cannot use annotation or old contact evidence")
    if not re.fullmatch(r"[a-fA-F0-9]{64}", record["video_sha256"]):
        raise ValueError("Missing source video SHA256")
    return x, times


def evidence_mask(record):
    x = np.asarray(record["features"], dtype=float)
    indices = {name: record["feature_names"].index(name) for name in EVIDENCE_FEATURES}
    return ((x[:, indices["pose_visible"]] > .5)
            & ((x[:, indices["ball_visible"]] > .5) | (x[:, indices["seed_available"]] > .5)))


def build_temporal_features(record):
    x, times = validate_record(record)
    step = float(np.median(np.diff(times)))
    tolerance = step * .51 + 1e-8
    parts, masks = [], []
    for lag in LAGS_SECONDS:
        requested = times + lag
        right = np.searchsorted(times, requested).clip(0, len(times) - 1)
        left = (right - 1).clip(0, len(times) - 1)
        indices = np.where(abs(times[left] - requested) <= abs(times[right] - requested), left, right)
        available = ((requested >= times[0] - 1e-8) & (requested <= times[-1] + 1e-8)
                     & (abs(times[indices] - requested) <= tolerance))
        parts.append(np.where(available[:, None], x[indices], 0.))
        masks.append(available)
    available = np.stack(masks, axis=1)
    eligible = available.all(axis=1) & evidence_mask(record)
    supported = np.zeros(len(times), dtype=bool)
    run = []
    for index in np.flatnonzero(eligible):
        if run and (index != run[-1] + 1 or times[index] - times[run[-1]] > 1.5 * step + 1e-9):
            if len(run) >= MINIMUM_SUPPORT_SAMPLES:
                supported[run] = True
            run = []
        run.append(index)
    if len(run) >= MINIMUM_SUPPORT_SAMPLES:
        supported[run] = True
    return np.concatenate([*parts, available.astype(float)], axis=1), supported


def validate_reference(record, reference):
    if (record["core_id"] != reference["core_id"]
            or record["video_sha256"].lower() != reference["video_sha256"].lower()
            or record["clip_id"] != reference["clip_id"]):
        raise ValueError("Contact reference is not bound to the same video and event")
    time = reference["reference_contact_seconds"]
    if not isinstance(time, (int, float)) or isinstance(time, bool) or not np.isfinite(time) or time < 0:
        raise ValueError("A finite nonnegative contact reference is required")
    return float(time)


def visual_bindings(record):
    provenance = record.get("provenance", {})
    result = {"visual_model_sha256": provenance.get("model_sha256", {}),
              "visual_settings": record.get("settings", {}),
              "extractor_code_sha256": provenance.get("code_sha256", {})}
    if any(not isinstance(value, dict) for value in result.values()):
        raise ValueError("Visual model, settings, and code bindings must be mappings")
    for field in ("visual_model_sha256", "extractor_code_sha256"):
        if any(not isinstance(value, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", value)
               for value in result[field].values()):
            raise ValueError("Visual model and extractor code bindings require SHA256 values")
    return result


def validate_prediction_bindings(record, metadata):
    if record["feature_names"] != metadata["feature_names"]:
        raise ValueError("Prediction feature schema differs from training")
    for name, value in visual_bindings(record).items():
        if name not in metadata or value != metadata[name]:
            raise ValueError(f"Prediction visual binding differs from training: {name}")
    if metadata["lags_seconds"] != list(LAGS_SECONDS) or metadata["smoothing_seconds"] != SMOOTHING_SECONDS:
        raise ValueError("Prediction temporal protocol differs from training")
    if metadata.get("minimum_support_samples") != MINIMUM_SUPPORT_SAMPLES:
        raise ValueError("Prediction evidence support policy differs from training")


def validate_cohort(records, references, strict=False):
    if not records:
        raise ValueError("Empty feature cohort")
    ids, hashes = set(), set()
    names = records[0]["feature_names"]
    bindings = visual_bindings(records[0])
    for record in records:
        validate_record(record)
        if record["core_id"] in ids or record["video_sha256"] in hashes:
            raise ValueError("Duplicate event or video in feature cohort")
        ids.add(record["core_id"])
        hashes.add(record["video_sha256"])
        if record["feature_names"] != names:
            raise ValueError("Feature schema differs between events")
        if visual_bindings(record) != bindings:
            raise ValueError("Visual model, settings, or extractor code differs between events")
        if record["match_group"] not in EXPECTED_GROUPS:
            raise ValueError("Unexpected match, including held-out external footage")
        if record["core_id"] not in references:
            raise ValueError("Missing contact reference")
        validate_reference(record, references[record["core_id"]])
    if strict and dict(Counter(row["match_group"] for row in records)) != EXPECTED_GROUPS:
        raise ValueError("The fixed protocol requires exactly ten core events from four matches")


def leave_one_match_out(records):
    groups = sorted({row["match_group"] for row in records})
    if len(groups) < 2:
        raise ValueError("At least two matches are required")
    for group in groups:
        yield group, [row for row in records if row["match_group"] != group], [row for row in records if row["match_group"] == group]


def fit_contact_head(records, references):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    validate_cohort(records, references)
    features, labels, weights, audit = [], [], [], []
    for record in records:
        x, eligible = build_temporal_features(record)
        reference = validate_reference(record, references[record["core_id"]])
        distance = abs(np.asarray(record["time_seconds"]) - reference)
        positive = eligible & (distance <= POSITIVE_RADIUS_SECONDS + 1e-9)
        negative = eligible & (distance >= NEGATIVE_RADIUS_SECONDS - 1e-9)
        row = {"core_id": record["core_id"], "positive_frames": int(positive.sum()),
               "negative_frames": int(negative.sum()), "used": bool(positive.any() and negative.any())}
        if not row["used"]:
            row["reason"] = "no_positive_or_negative_frames_in_automatic_context_window"
            audit.append(row)
            continue
        keep = positive | negative
        event_weights = np.where(positive[keep], .5 / positive.sum(), .5 / negative.sum())
        features.append(x[keep])
        labels.append(positive[keep].astype(int))
        weights.append(event_weights)
        row["sample_weight_sum"] = float(event_weights.sum())
        audit.append(row)
    if not features:
        raise ValueError("No training event has both contact and non-contact evidence")
    x, y, weight = np.concatenate(features), np.concatenate(labels), np.concatenate(weights)
    # Center first so weighted variance of constant mask columns stays exactly zero.
    origin = x[0].copy()
    scaler = StandardScaler().fit(x - origin, sample_weight=weight)
    scaler.mean_ += origin
    head = LogisticRegression(C=REGULARIZATION_C, solver="lbfgs", max_iter=2000, random_state=SEED)
    head.fit(scaler.transform(x), y, sample_weight=weight)
    return scaler, head, audit


def export_parameters(scaler, head):
    if head.classes_.tolist() != [0, 1]:
        raise ValueError("Contact classifier must contain classes zero and one")
    return {"scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
            "coef": head.coef_, "intercept": head.intercept_}


def predict_probabilities(features, parameters):
    x = (np.atleast_2d(features) - parameters["scaler_mean"]) / parameters["scaler_scale"]
    logits = (x @ parameters["coef"].T + parameters["intercept"]).reshape(-1)
    return np.exp(-np.logaddexp(0., -logits))


def predict_contact(record, parameters):
    x, eligible = build_temporal_features(record)
    times = np.asarray(record["time_seconds"])
    probability = predict_probabilities(x, parameters)
    smoothed = probability.copy()
    # An incomplete boundary context can never win the event-level decision.
    for index in np.flatnonzero(eligible):
        neighbors = eligible & (abs(times - times[index]) <= SMOOTHING_SECONDS / 2 + 1e-9)
        smoothed[index] = probability[neighbors].mean()
    indices = np.flatnonzero(eligible)
    if not len(indices):
        return {"time_seconds": None, "frame": None,
                "reason": "evidence_missing" if not evidence_mask(record).any() else "insufficient_consecutive_evidence_or_temporal_context",
                "eligible_frames": 0, "probabilities_calibrated": False}
    selected = int(indices[np.argmax(smoothed[indices])])
    return {"time_seconds": float(times[selected]), "frame": int(record["frame_ids"][selected]),
            "probability_uncalibrated": float(smoothed[selected]), "eligible_frames": int(len(indices)),
            "probabilities_calibrated": False, "reason": "fixed_smoothed_argmax"}


def error_metrics(rows, method, allowed_ids=None):
    selected = [row for row in rows if allowed_ids is None or row["core_id"] in allowed_ids]
    available = [row for row in selected if row["predictions"][method]["time_seconds"] is not None]
    errors = np.array([abs(row["predictions"][method]["time_seconds"] - row["reference_contact_seconds"])
                       for row in available])
    result = {"events": len(selected), "available": len(available),
              "coverage": len(available) / len(selected) if selected else None,
              "mae_seconds": float(errors.mean()) if len(errors) else None,
              "median_error_seconds": float(np.median(errors)) if len(errors) else None,
              "max_error_seconds": float(errors.max()) if len(errors) else None}
    for tolerance in (.1, .2, .5):
        result[f"within_{tolerance:.1f}_seconds"] = float(np.mean(errors <= tolerance + 1e-9)) if len(errors) else None
        result[f"within_{tolerance:.1f}_seconds_over_all_events"] = float(np.count_nonzero(errors <= tolerance + 1e-9) / len(selected)) if selected else None
    return result


def compare_predictions(rows):
    methods = ("trained_contact", "existing_rule", "ankle_proximity", "training_median")
    common = {row["core_id"] for row in rows if all(row["predictions"][method]["time_seconds"] is not None for method in methods)}
    trained = {row["core_id"] for row in rows if row["predictions"]["trained_contact"]["time_seconds"] is not None}
    return {"per_method_available_events": {method: error_metrics(rows, method) for method in methods},
            "common_available_core_ids": sorted(common),
            "all_methods_on_same_events": {method: error_metrics(rows, method, common) for method in methods},
            "pairwise_same_events": {method: {
                "trained_contact": error_metrics(rows, "trained_contact", trained & {r["core_id"] for r in rows if r["predictions"][method]["time_seconds"] is not None}),
                method: error_metrics(rows, method, trained & {r["core_id"] for r in rows if r["predictions"][method]["time_seconds"] is not None})}
                for method in methods[1:]}}

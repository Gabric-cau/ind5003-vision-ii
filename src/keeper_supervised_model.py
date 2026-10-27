"""Small supervised keeper heads over frozen, camera-stabilized observations."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np


COORDINATES = "source_image_x_right"
FEATURE_NAMES = (
    "late_lateral_body", "peak_signed_lateral_body", "peak_absolute_lateral_body",
    "late_vertical_body", "minimum_vertical_body", "maximum_vertical_body",
    "median_lateral_speed_body_s", "median_vertical_speed_body_s", "peak_speed_body_s",
    "minimum_height_ratio", "median_height_ratio", "torso_horizontal_fraction",
    "shoulder_ankle_horizontal_fraction", "pose_vectors_available",
)
HEADS = {
    "direction": {"classes": ("left", "center", "right"),
                  "features": (0, 1, 2, 6, 11, 12, 13)},
    "height": {"classes": ("low", "middle"), "features": (3, 4, 5, 7, 8, 9, 10)},
}


@dataclass(frozen=True)
class KeeperConfig:
    minimum_observations: int = 5
    minimum_span_seconds: float = 0.10
    minimum_coverage: float = 0.15
    maximum_velocity_gap_seconds: float = 0.20
    regularization_c: float = 0.10
    seed: int = 20261026


def _trajectory(value, cfg):
    try:
        rows = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if (rows.ndim != 2 or rows.shape[1] != 4 or len(rows) < cfg.minimum_observations
            or not np.isfinite(rows).all() or np.any(rows[:, 3] <= 0)):
        return None
    if np.any(np.diff(rows[:, 0]) <= 0) or rows[-1, 0] - rows[0, 0] < cfg.minimum_span_seconds:
        return None
    return rows


def _lean(samples, key):
    vectors = []
    for sample in samples:
        try:
            vector = np.asarray(sample.get(key), dtype=float)
        except (TypeError, ValueError):
            continue
        if vector.shape == (2,) and np.isfinite(vector).all() and np.linalg.norm(vector) > 1e-8:
            vectors.append(float(vector[0] / np.linalg.norm(vector)))
    if len(vectors) < 3:
        return 0.0, False
    # Most lateral observations describe the dive, independent of frame labels.
    values = np.asarray(vectors)
    lateral = np.abs(values) >= np.quantile(np.abs(values), 2 / 3)
    return float(np.median(values[lateral])), True


def extract_features(keeper, config=None):
    """Use geometry only. Missing identity/stabilization evidence cannot be imputed."""
    cfg = config or KeeperConfig()
    candidate = keeper.get("selected_candidate") or {}
    reasons = []
    if keeper.get("status") != "ok" or candidate.get("qualifies") is not True:
        reasons.append("no_qualified_keeper_track")
    if candidate.get("is_selected_kicker") is True or keeper.get("role", {}).get("ambiguous_identity") is True:
        reasons.append("ambiguous_or_kicker_identity")
    if keeper.get("direction_coordinate_system") != COORDINATES:
        reasons.append("unsupported_direction_coordinates")
    if not all(value is True for value in (
            candidate.get("camera_reference_chain_complete"),
            keeper.get("gk_prekick_stabilization_complete"), keeper.get("dive_stabilization_complete"))):
        reasons.append("incomplete_camera_stabilization")
    for field in ("pre_coverage", "post_coverage"):
        value = candidate.get(field)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or not cfg.minimum_coverage <= value <= 1:
            reasons.append(f"insufficient_{field}")
    pre = _trajectory(candidate.get("stabilized_pre_trajectory"), cfg)
    post = _trajectory(candidate.get("stabilized_post_trajectory"), cfg)
    if pre is None or post is None:
        reasons.append("missing_or_invalid_trajectories")
    if reasons:
        return {"available": False, "x": None, "reasons": reasons}
    if pre[-1, 0] > post[0, 0] + 1e-8:
        return {"available": False, "x": None, "reasons": ["overlapping_pre_post_clocks"]}
    scale = float(np.median(pre[:, 3]))
    baseline = np.median(pre[-5:, 1:3], axis=0)
    displacement = (post[:, 1:3] - baseline) / scale
    late = np.median(displacement[-3:], axis=0)
    lateral_bounds = np.quantile(displacement[:, 0], [0.1, 0.9])
    dt = np.diff(post[:, 0])
    valid = dt <= cfg.maximum_velocity_gap_seconds + 1e-8
    velocity = np.diff(post[:, 1:3], axis=0)[valid] / dt[valid, None] / scale
    if not len(velocity):
        return {"available": False, "x": None, "reasons": ["no_contiguous_post_motion"]}
    median_velocity = np.median(velocity, axis=0)
    height = post[:, 3] / scale
    samples = keeper.get("pose_direction_evidence", {}).get("pose_samples", [])
    torso, has_torso = _lean(samples, "torso_vector")
    body, has_body = _lean(samples, "shoulder_to_ankle_vector")
    x = np.array([
        late[0], lateral_bounds[np.argmax(np.abs(lateral_bounds))],
        np.quantile(np.abs(displacement[:, 0]), 0.9), late[1],
        np.quantile(displacement[:, 1], 0.1), np.quantile(displacement[:, 1], 0.9),
        median_velocity[0], median_velocity[1], np.quantile(np.linalg.norm(velocity, axis=1), 0.9),
        np.quantile(height, 0.1), np.median(height), torso, body, float(has_torso and has_body),
    ])
    if not np.isfinite(x).all():
        return {"available": False, "x": None, "reasons": ["nonfinite_features"]}
    return {"available": True, "x": x.tolist(), "reasons": [],
            "pre_observations": len(pre), "post_observations": len(post),
            "post_span_seconds": float(post[-1, 0] - post[0, 0]),
            "post_coverage": candidate["post_coverage"], "pose_vectors_available": bool(has_torso and has_body)}


def fit_models(events, held_out_match=None, config=None):
    """Fit each fold's scaler and fixed heads on its training matches only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    cfg = config or KeeperConfig()
    training = [e for e in events if e["match_group"] != held_out_match and e["features"]["available"]]
    if not training:
        raise ValueError("No usable training keeper observations")
    model = {"schema_version": "keeper-supervised-v1", "config": asdict(cfg),
             "feature_names": list(FEATURE_NAMES), "direction_coordinate_system": COORDINATES,
             "held_out_match": held_out_match, "training_ids": [e["clip_id"] for e in training],
             "height_semantics": "human_qualitative_image_plane_low_or_middle_only", "heads": {}}
    for head, spec in HEADS.items():
        labels = [e[head] for e in training]
        if not set(labels).issubset(spec["classes"]):
            raise ValueError(f"Unsupported {head} training label")
        x = np.asarray([e["features"]["x"] for e in training])[:, spec["features"]]
        scaler = StandardScaler().fit(x)
        output = {"feature_indices": list(spec["features"]), "mean": scaler.mean_.tolist(),
                  "scale": scaler.scale_.tolist(), "label_counts": dict(Counter(labels))}
        if len(set(labels)) == 1:
            output.update({"classes": [labels[0]], "constant": labels[0]})
        else:
            estimator = LogisticRegression(C=cfg.regularization_c, class_weight="balanced", max_iter=2000,
                                           random_state=cfg.seed).fit(scaler.transform(x), labels)
            output.update({"classes": estimator.classes_.tolist(), "coef": estimator.coef_.tolist(),
                           "intercept": estimator.intercept_.tolist()})
        model["heads"][head] = output
    return model


def predict_features(features, model):
    result = {"status": "abstain", "dive_direction": "unknown", "dive_height": "unknown",
              "direction": "unknown", "height": "unknown", "scores_uncalibrated": {},
              "reasons": features["reasons"], "height_high_class_supported": False}
    if not features["available"]:
        return result
    if model.get("feature_names") != list(FEATURE_NAMES) or model.get("direction_coordinate_system") != COORDINATES:
        raise ValueError("Incompatible keeper model feature contract")
    x = np.asarray(features["x"], dtype=float)
    if x.shape != (len(FEATURE_NAMES),) or not np.isfinite(x).all():
        raise ValueError("Invalid keeper feature vector")
    for name in HEADS:
        head = model["heads"][name]
        classes = head["classes"]
        if "constant" in head:
            probabilities = np.ones(1)
        else:
            normalized = (x[head["feature_indices"]] - head["mean"]) / head["scale"]
            logits = np.asarray(head["coef"]) @ normalized + head["intercept"]
            if len(classes) == 2:
                logits = np.array([0.0, logits[0]])
            probabilities = np.exp(logits - np.max(logits))
            probabilities /= probabilities.sum()
        result[name] = classes[int(np.argmax(probabilities))]
        result["scores_uncalibrated"][name] = dict(zip(classes, probabilities.tolist()))
    result.update({"status": "candidate", "dive_direction": {
        "left": "image_left", "center": "center", "right": "image_right"}[result["direction"]],
        "dive_height": result["height"]})
    return result


def predict_keeper(keeper, model):
    """Inference interface: automatic keeper dictionary plus a JSON model; no labels."""
    return predict_features(extract_features(keeper, KeeperConfig(**model["config"])), model)


def classification_metrics(truths, predictions, classes):
    from sklearn.metrics import confusion_matrix, f1_score

    if len(truths) != len(predictions) or not set(truths).issubset(classes):
        raise ValueError("Invalid classification evaluation records")
    predictions = [p if p in classes else "unknown" for p in predictions]
    covered = [i for i, p in enumerate(predictions) if p in classes]
    correct = sum(a == b for a, b in zip(truths, predictions))
    return {"events": len(truths), "outputs": len(covered), "correct": correct,
            "coverage": len(covered) / len(truths) if truths else None,
            "all_event_accuracy": correct / len(truths) if truths else None,
            "covered_accuracy": correct / len(covered) if covered else None,
            "all_event_macro_f1": float(f1_score(truths, predictions, labels=list(classes),
                                                  average="macro", zero_division=0)) if truths else None,
            "covered_macro_f1": float(f1_score([truths[i] for i in covered], [predictions[i] for i in covered],
                                                labels=list(classes), average="macro", zero_division=0)) if covered else None,
            "class_order": list(classes) + ["unknown"],
            "confusion_rows_true_columns_predicted": confusion_matrix(
                truths, predictions, labels=list(classes) + ["unknown"]).tolist() if truths
                else np.zeros((len(classes) + 1, len(classes) + 1), dtype=int).tolist()}

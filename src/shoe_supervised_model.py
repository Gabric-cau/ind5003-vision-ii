"""Small supervised heads over frozen, source-bound shoe pose evidence."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.shoe_landing_events import _project, _read_records, _transform_map


SIDES = ("left", "right")
JOINTS = {"left": (23, 25, 27, 29, 31), "right": (24, 26, 28, 30, 32)}
LAGS_SECONDS = (-0.12, -0.06, 0.0, 0.06, 0.12)
BASE_FEATURE_NAMES = (
    "ankle_hip_x", "ankle_hip_y", "heel_hip_x", "heel_hip_y",
    "toe_hip_x", "toe_hip_y", "knee_cosine", "heel_toe_distance",
    "shoe_bottom_clearance", "pose_confidence", "mirror_error_body",
    "shoe_speed_before", "shoe_speed_after", "relative_speed_before",
    "relative_speed_after", "vertical_speed_before", "vertical_speed_after",
    "hip_speed_before", "hip_speed_after",
)


@dataclass(frozen=True)
class ShoeConfig:
    confidence_min: float = 0.35
    maximum_mirror_error_body: float = 0.15
    minimum_body_height: float = 30.0
    positive_seconds: float = 0.05
    negative_seconds: float = 0.13
    minimum_peak_separation_seconds: float = 0.20
    landing_score_threshold: float = 0.5
    minimum_temporal_observations: int = 3
    foot_before_seconds: float = 0.4
    foot_after_seconds: float = 0.25
    seed: int = 20261026


def _cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0


def extract_evidence(inputs, records, config=None):
    """Read automatic input windows only; never read a human label document."""
    cfg = config or ShoeConfig()
    fps = float(inputs["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid source fps")
    first, last = inputs["inference_interval_frames"]
    parsed = _read_records(records, first, last)
    transforms = _transform_map(inputs["original_camera_motion"])
    hashes = inputs["decoded_frame_hashes"]
    states = {side: {} for side in SIDES}
    used_hashes = set()
    for frame in range(first, last + 1):
        pair = [parsed.get((frame, reflected)) for reflected in (False, True)]
        digest = hashes.get(str(frame), hashes.get(frame))
        matrix = transforms.get(frame)
        if (not digest or digest in used_hashes or matrix is None
                or any(row is None or not row["accepted"] for row in pair)):
            continue
        used_hashes.add(digest)
        if any(row["pixel_hash"] != digest for row in pair):
            raise ValueError("Pose and decoded pixel hashes disagree")
        scale = float(np.median([r["box"][3] - r["box"][1] for r in pair]))
        if scale < cfg.minimum_body_height:
            continue
        projected = [_project(r["points"], matrix) for r in pair]
        if any(p is None for p in projected):
            continue
        projected_box = [_project(np.array([[r["box"][0], r["box"][1]],
                                           [r["box"][0], r["box"][3]]]), matrix) for r in pair]
        if any(p is None for p in projected_box):
            continue
        stable_scale = float(np.median([np.linalg.norm(p[1] - p[0]) for p in projected_box]))
        if stable_scale <= 1e-6:
            continue
        points = np.mean(projected, axis=0)
        hip = points[[23, 24]].mean(axis=0)
        for side, joints in JOINTS.items():
            indices = list(joints) + [23, 24]
            confidence = float(min(np.min(r["confidence"][indices]) for r in pair))
            mirror = float(np.max(np.linalg.norm(pair[0]["points"][indices]
                                                  - pair[1]["points"][indices], axis=1)) / scale)
            if confidence < cfg.confidence_min or mirror > cfg.maximum_mirror_error_body:
                continue
            hip_side, knee, ankle, heel, toe = points[list(joints)]
            relative = (points[list(joints)[2:]] - hip) / stable_scale
            shoe = (heel + toe) / 2
            clearance = np.mean([(r["box"][3] - np.mean(r["points"][list(joints)[3:], 1])) / scale
                                 for r in pair])
            geometry = np.r_[relative.ravel(), _cosine(hip_side - knee, ankle - knee),
                             np.linalg.norm(heel - toe) / stable_scale,
                             clearance, confidence, mirror]
            states[side][frame] = {"geometry": geometry, "shoe": shoe, "hip": hip,
                                   "scale": stable_scale, "relative": (shoe - hip) / stable_scale}
    features = {side: {} for side in SIDES}
    for side in SIDES:
        for frame, state in states[side].items():
            motion = []
            for sign in (-1, 1):
                target = frame + sign * max(1, round(0.1 * fps))
                candidates = [f for f in states[side] if sign * (f - frame) > 0
                              and abs(f - target) / fps <= 0.06]
                if not candidates:
                    motion.append([0.0] * 4)
                    continue
                other_frame = min(candidates, key=lambda f: abs(f - target))
                other = states[side][other_frame]
                dt = (other_frame - frame) / fps
                mean_scale = (state["scale"] + other["scale"]) / 2
                velocity = (other["shoe"] - state["shoe"]) / dt / mean_scale
                rel_velocity = (other["relative"] - state["relative"]) / dt
                motion.append([np.linalg.norm(velocity), np.linalg.norm(rel_velocity),
                               velocity[1], np.linalg.norm(other["hip"] - state["hip"]) / abs(dt) / mean_scale])
            arranged = np.asarray(motion).T.ravel()
            features[side][frame] = np.clip(np.r_[state["geometry"], arranged], -20, 20)
    landing_rows = []
    start, end = inputs["interval_frames"]
    for side in SIDES:
        for frame in sorted(features[side]):
            if not start <= frame <= end:
                continue
            blocks, masks = [], []
            for lag in LAGS_SECONDS:
                target = frame + round(lag * fps)
                block = features[side].get(target)
                blocks.append(block if block is not None else np.zeros(len(BASE_FEATURE_NAMES)))
                masks.append(float(block is not None))
            if sum(masks) >= cfg.minimum_temporal_observations:
                landing_rows.append({"frame": frame, "foot": side,
                                     "x": np.r_[np.concatenate(blocks), masks]})
    bracket = inputs.get("contact_bracket_frames")
    foot_vector = None
    if isinstance(bracket, list) and len(bracket) == 2:
        summaries = []
        for side in SIDES:
            values = [value for frame, value in features[side].items()
                      if bracket[0] - cfg.foot_before_seconds * fps <= frame
                      <= bracket[1] + cfg.foot_after_seconds * fps]
            if len(values) < 3:
                break
            summaries.append(np.quantile(values, [0.25, 0.5, 0.75], axis=0).ravel())
        if len(summaries) == 2:
            foot_vector = summaries[1] - summaries[0]
    return {"landing_rows": landing_rows, "foot_vector": foot_vector,
            "supported_side_frames": {s: len(features[s]) for s in SIDES}}


def landing_training_rows(evidence, references, fps, config=None):
    cfg = config or ShoeConfig()
    rows, labels = [], []
    for row in evidence["landing_rows"]:
        frames = [ref["frame"] for ref in references if ref["foot"] == row["foot"]]
        distance = min((abs(row["frame"] - f) / fps for f in frames), default=float("inf"))
        if distance <= cfg.positive_seconds or distance >= cfg.negative_seconds:
            rows.append(row["x"])
            labels.append(int(distance <= cfg.positive_seconds))
    if len(set(labels)) != 2:
        return None
    labels = np.asarray(labels)
    weights = np.array([0.5 / np.sum(labels == label) for label in labels])
    return np.asarray(rows), labels, weights


def fit_models(events, held_out_match=None, config=None):
    cfg = config or ShoeConfig()
    training = [e for e in events if e["match_group"] != held_out_match]
    landing_parts, foot_x, foot_y = [], [], []
    landing_ids, foot_ids = [], []
    for event in training:
        part = landing_training_rows(event["evidence"], event["landings"], event["fps"], cfg)
        if part is not None:
            landing_parts.append(part)
            landing_ids.append(event["clip_id"])
        vector = event["evidence"]["foot_vector"]
        if vector is not None and event["foot"] in SIDES:
            label = int(event["foot"] == "right")
            foot_x.extend((vector, -vector))
            foot_y.extend((label, 1 - label))
            foot_ids.append(event["clip_id"])
    landing_model = None
    if landing_parts:
        x = np.concatenate([part[0] for part in landing_parts])
        y = np.concatenate([part[1] for part in landing_parts])
        weights = np.concatenate([part[2] for part in landing_parts])
        weights *= len(weights) / weights.sum()
        landing_model = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=7,
            max_depth=3, min_samples_leaf=20, l2_regularization=5.0, random_state=cfg.seed)
        landing_model.fit(x, y, sample_weight=weights)
    foot_model, foot_scale = None, None
    if foot_x:
        foot_scale = StandardScaler(with_mean=False).fit(foot_x)
        foot_model = LogisticRegression(C=0.1, fit_intercept=False, max_iter=2000,
                                        random_state=cfg.seed).fit(foot_scale.transform(foot_x), foot_y)
    return {"landing_model": landing_model, "foot_model": foot_model, "foot_scale": foot_scale,
            "config": asdict(cfg), "held_out_match": held_out_match,
            "training_landing_ids": landing_ids, "training_foot_ids": foot_ids}


def predict_event(model, evidence, fps):
    cfg = ShoeConfig(**model["config"])
    rows = evidence["landing_rows"]
    landings = None
    if rows and model["landing_model"] is not None:
        scores = model["landing_model"].predict_proba(np.asarray([r["x"] for r in rows]))[:, 1]
        candidates = []
        for row, score in zip(rows, scores):
            neighbors = [float(s) for r, s in zip(rows, scores) if r["foot"] == row["foot"]
                         and abs(r["frame"] - row["frame"]) / fps <= 0.05]
            candidates.append({"frame": row["frame"], "foot": row["foot"], "score": float(np.mean(neighbors))})
        landings = []
        for candidate in sorted(candidates, key=lambda r: (-r["score"], r["frame"])):
            if candidate["score"] < cfg.landing_score_threshold:
                continue
            if any(p["foot"] == candidate["foot"] and
                   abs(p["frame"] - candidate["frame"]) / fps < cfg.minimum_peak_separation_seconds for p in landings):
                continue
            landings.append(candidate)
        landings.sort(key=lambda r: (r["frame"], r["foot"]))
    foot, right_score = "unknown", None
    vector = evidence["foot_vector"]
    if vector is not None and model["foot_model"] is not None:
        right_score = float(model["foot_model"].predict_proba(model["foot_scale"].transform([vector]))[0, 1])
        foot = "right" if right_score >= 0.5 else "left"
    return {"landings": landings, "candidate_count": len(landings) if landings is not None else None,
            "striking_foot": foot, "right_score_uncalibrated": right_score}

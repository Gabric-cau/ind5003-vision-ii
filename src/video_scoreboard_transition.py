"""Infer a single-attempt outcome from a visible before/after icon transition."""
from pathlib import Path

import cv2
import numpy as np

from src.broadcast_scoreboard import extract_penalty_scoreboard, stable_scoreboard_runs
from src.runtime import file_sha256


def outcome_from_transition(runs, contact_frame, fps):
    result = {"rule_outcome": "unknown", "reason": "missing_stable_before_after_panels",
              "attempt_ordinal_supplied": False, "human_labels_used": False,
              "ball_goal_plane_crossing_verified": False}
    if contact_frame is None:
        return {**result, "reason": "automatic_contact_unavailable"}
    before = []
    for run in runs:
        ids = [f for f in run["frame_ids"] if f <= contact_frame]
        if len(ids) >= 3 and (ids[-1] - ids[0]) / fps >= .4:
            before.append((ids[-1], run))
    if not before:
        return result
    baseline = max(before, key=lambda item: item[0])[1]
    witnesses, invalid = [], []
    for run in runs:
        if run["start_frame"] <= contact_frame:
            continue
        changes = [(side, i, old, new) for side in ("left", "right")
                   for i, (old, new) in enumerate(zip(baseline["states"][side], run["states"][side])) if old != new]
        if not changes:
            continue
        if len(changes) != 1 or changes[0][2] != "pending" or changes[0][3] not in ("green", "red"):
            invalid.append(run["start_frame"])
            continue
        side, slot, _, state = changes[0]
        witnesses.append({"side": side, "slot": slot, "state": state, "frames": run["frame_ids"]})
    if invalid:
        return {**result, "reason": "multiple_attempts_or_contradictory_panel_transition", "invalid_frames": invalid}
    if not witnesses:
        return result
    if len({(w["side"], w["slot"], w["state"]) for w in witnesses}) != 1:
        return {**result, "reason": "inconsistent_transition_witnesses"}
    return {**result, "rule_outcome": "goal_likely" if witnesses[0]["state"] == "green" else "no_goal_likely",
            "reason": "one_pending_icon_resolved_after_automatic_contact", "witnesses": witnesses,
            "baseline_frames": baseline["frame_ids"]}


def analyze_video_transition(video, contact_seconds, *, stride_seconds=.2):
    if not np.isfinite(stride_seconds) or not 0 < stride_seconds <= 1:
        raise ValueError("Invalid sampling stride")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError("Cannot open video")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("Invalid FPS")
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps * stride_seconds))
    observations = []
    try:
        for frame_id in range(expected):
            ok, frame = cap.read()
            if not ok:
                raise ValueError("Video decode ended early")
            if frame_id % step == 0 or frame_id == expected - 1:
                observations.append(extract_penalty_scoreboard(frame, frame_id=frame_id))
    finally:
        cap.release()
    runs = stable_scoreboard_runs(observations, fps=fps)
    result = outcome_from_transition(runs, contact_seconds * fps if contact_seconds is not None else None, fps)
    return {"source_sha256": file_sha256(video), "fps": fps, "contact_seconds": contact_seconds,
            "decision": result, "stable_runs": runs, "sample_count": len(observations),
            "supported_panel_samples": sum(r["status"] == "candidate_scoreboard" for r in observations),
            "scope": "Known red five-circle broadcast design; unsupported designs remain unknown",
            "code_sha256": file_sha256(Path(__file__))}

"""Reject candidate actors whose stationary-ball evidence exists only much earlier or later."""
from copy import deepcopy
from pathlib import Path

import numpy as np

from src.contact_temporal_model import build_temporal_features, predict_contact
from src.contact_video_features import geometric_features, rule_baseline
from src.runtime import file_sha256

# Use the existing contact scorer's 0.65s approach horizon, without tuning on labels.
CAUSAL_SETTINGS = {"causal_actor_recovery_version": 1, "precontact_seed_horizon_seconds": .65}
CODE_PATH = "src/contact_causal_actor.py"


def recent_stationary_support(candidate, fps):
    frame = candidate["contact_frame"]
    previous = [int(o["frame"]) for o in candidate.get("stationary_ball_observations", [])
                if int(o["frame"]) < frame]
    lag = (frame - max(previous)) / fps if previous else None
    return {"supported": lag is not None and lag <= CAUSAL_SETTINGS["precontact_seed_horizon_seconds"],
            "last_precontact_stationary_frame": max(previous) if previous else None,
            "gap_seconds": lag}


def recover_causal_actor(record, evidence, frame_height):
    if evidence["video_sha256"] != record["video_sha256"]:
        raise ValueError("Causal actor evidence source mismatch")
    result, visual = deepcopy(record), deepcopy(evidence)
    candidates = evidence.get("candidate_contacts", [])
    receipt = {"used": False, "status": "unchanged", "human_labels_used": False,
               "policy": "stationary_ball_observed_before_candidate_within_existing_approach_horizon",
               "candidate_checks": [{"rank": i, "actor_id": c["contact_associated_track_id"],
                   **recent_stationary_support(c, record["fps"])} for i, c in enumerate(candidates)]}
    if candidates and not receipt["candidate_checks"][0]["supported"]:
        receipt["status"] = "abstain"
        transforms = {int(k): np.asarray(v) for k, v in evidence["transforms"].items()}
        for index, candidate in enumerate(candidates):
            check = receipt["candidate_checks"][index]
            if not check["supported"] or not candidate.get("ankle_approach_supported") or not candidate.get("ball_departure_frames"):
                continue
            ordered = [candidate] + [c for i, c in enumerate(candidates) if i != index]
            features, actor, proximity = geometric_features(record["frame_ids"], record["fps"], frame_height,
                evidence["tracks"], evidence["balls"], transforms, ordered, evidence["camera"])
            trial = {**record, "features": features}
            _, supported = build_temporal_features(trial)
            start, end = candidate["interval_frames"]
            in_interval = supported & (np.asarray(record["frame_ids"]) >= start) & (np.asarray(record["frame_ids"]) <= end)
            check["eligible_frames_in_candidate_interval"] = int(in_interval.sum())
            if in_interval.any():
                result["features"] = features
                result["diagnostics"]["actor"] = actor
                result["baselines"] = {"existing_rule": rule_baseline(ordered, evidence["camera"], record["window"]["end_frame_exclusive"]),
                                       "ankle_proximity": proximity}
                visual["pre_causal_candidate_contacts"] = deepcopy(candidates)
                visual["candidate_contacts"] = ordered
                receipt.update(used=True, status="recovered", selected_candidate_rank=index, selected_actor_id=actor["actor_id"])
                break
    result["settings"].update(CAUSAL_SETTINGS)
    result["provenance"]["code_sha256"][CODE_PATH] = file_sha256(Path(__file__).parents[1] / CODE_PATH)
    result["diagnostics"]["causal_actor_recovery"] = receipt
    visual["causal_actor_recovery"] = receipt
    return result, visual


def predict_causal_contact(record, parameters):
    if record["diagnostics"]["causal_actor_recovery"]["status"] == "abstain":
        return {"time_seconds": None, "frame": None, "eligible_frames": 0,
                "probabilities_calibrated": False, "reason": "no_causal_precontact_stationary_ball_support"}
    return predict_contact(record, parameters)

"""Keep causal actor recovery's final decision inside its automatic candidate window."""
from copy import deepcopy
from pathlib import Path

import numpy as np

from src.contact_causal_actor import predict_causal_contact
from src.contact_temporal_model import build_temporal_features, predict_probabilities, SMOOTHING_SECONDS
from src.runtime import file_sha256

WINDOW_SETTINGS = {"recovered_candidate_window_version": 1}
CODE_PATH = "src/contact_candidate_window.py"


def bind_candidate_window(record, evidence, frame_height=None):
    if record["video_sha256"] != evidence["video_sha256"]:
        raise ValueError("Candidate window evidence source mismatch")
    result, visual = deepcopy(record), deepcopy(evidence)
    receipt = {"used": False, "human_labels_used": False, "probability_context": "full_automatic_shot"}
    if record["diagnostics"]["causal_actor_recovery"]["used"]:
        candidate = evidence["candidate_contacts"][0]
        if candidate["contact_associated_track_id"] != record["diagnostics"]["actor"]["actor_id"]:
            raise ValueError("Candidate interval and recovered actor disagree")
        receipt.update(used=True, candidate_interval_frames=list(candidate["interval_frames"]),
                       source="existing_automatic_ball_ankle_departure_candidate")
    result["settings"].update(WINDOW_SETTINGS)
    result["provenance"]["code_sha256"][CODE_PATH] = file_sha256(Path(__file__).parents[1] / CODE_PATH)
    result["diagnostics"]["candidate_window_recovery"] = receipt
    visual["candidate_window_recovery"] = receipt
    return result, visual


def predict_candidate_contact(record, parameters):
    constraint = record["diagnostics"]["candidate_window_recovery"]
    if not constraint["used"]:
        return predict_causal_contact(record, parameters)
    x, eligible = build_temporal_features(record)
    times = np.asarray(record["time_seconds"])
    probability = predict_probabilities(x, parameters)
    start, end = constraint["candidate_interval_frames"]
    within = eligible & (np.asarray(record["frame_ids"]) >= start) & (np.asarray(record["frame_ids"]) <= end)
    indices = np.flatnonzero(within)
    if not len(indices):
        return {"time_seconds": None, "frame": None, "eligible_frames": 0, "probabilities_calibrated": False,
                "reason": "no_temporal_support_inside_automatic_candidate_interval"}
    smoothed = np.array([probability[eligible & (abs(times - times[i]) <= SMOOTHING_SECONDS / 2 + 1e-9)].mean()
                         for i in indices])
    best = int(np.argmax(smoothed))
    index = int(indices[best])
    return {"time_seconds": float(times[index]), "frame": int(record["frame_ids"][index]),
            "probability_uncalibrated": float(smoothed[best]), "eligible_frames": int(len(indices)),
            "probabilities_calibrated": False, "reason": "smoothed_argmax_inside_automatic_causal_candidate"}

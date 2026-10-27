"""Recover an automatic contact actor only when its temporal support is empty."""
from copy import deepcopy
from pathlib import Path

import numpy as np

from src.contact_temporal_model import build_temporal_features
from src.contact_video_features import geometric_features, rule_baseline
from src.runtime import file_sha256

RECOVERY_SETTINGS = {"actor_support_recovery_version": 1}
CODE_PATH = "src/contact_actor_support.py"


def recover_actor_support(record, evidence, frame_height):
    if evidence["video_sha256"] != record["video_sha256"]:
        raise ValueError("Actor recovery evidence source mismatch")
    result, visual = deepcopy(record), deepcopy(evidence)
    _, original_support = build_temporal_features(record)
    receipt = {"used": False, "initial_eligible_frames": int(original_support.sum()),
               "policy": "first_ranked_automatic_candidate_with_temporal_support_in_its_contact_interval",
               "human_labels_used": False, "candidate_trials": []}
    if not original_support.any():
        transforms = {int(k): np.asarray(v) for k, v in evidence["transforms"].items()}
        for index, candidate in enumerate(evidence["candidate_contacts"]):
            ordered = [candidate] + [c for i, c in enumerate(evidence["candidate_contacts"]) if i != index]
            features, actor, proximity = geometric_features(record["frame_ids"], record["fps"], frame_height,
                evidence["tracks"], evidence["balls"], transforms, ordered, evidence["camera"])
            trial = deepcopy(record)
            trial["features"] = features
            _, supported = build_temporal_features(trial)
            start, end = candidate["interval_frames"]
            candidate_support = supported & (np.asarray(record["frame_ids"]) >= start) & (np.asarray(record["frame_ids"]) <= end)
            receipt["candidate_trials"].append({"candidate_rank": index, "actor_id": actor["actor_id"],
                "eligible_frames": int(supported.sum()), "eligible_frames_in_candidate_interval": int(candidate_support.sum())})
            if candidate_support.any():
                result["features"] = features
                result["diagnostics"]["actor"] = actor
                result["baselines"] = {"existing_rule": rule_baseline(ordered, evidence["camera"], record["window"]["end_frame_exclusive"]),
                                       "ankle_proximity": proximity}
                visual["original_candidate_contacts"] = deepcopy(evidence["candidate_contacts"])
                visual["candidate_contacts"] = ordered
                receipt.update(used=True, selected_candidate_rank=index, selected_actor_id=actor["actor_id"])
                break
    result["settings"].update(RECOVERY_SETTINGS)
    result["provenance"]["code_sha256"][CODE_PATH] = file_sha256(Path(__file__).parents[1] / CODE_PATH)
    result["diagnostics"]["actor_support_recovery"] = receipt
    visual["actor_support_recovery"] = receipt
    return result, visual

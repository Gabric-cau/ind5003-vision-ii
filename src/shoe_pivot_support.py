"""Development supplement for heel or toe pivots during visible stance.

The same anatomical landmark must support both views throughout a stance.
This never switches from heel to toe inside a support run or reads labels.
"""
from copy import deepcopy

from src.shoe_landing_events import (
    LandingConfig, SHOES, _propose_view, _read_records, _series, _transform_map,
    propose_shoe_landings,
)


def propose_pivot_landings(records, fps, interval_frames, transforms, decoded_frame_hashes, config=None):
    cfg = config or LandingConfig()
    baseline = propose_shoe_landings(records, fps, interval_frames, transforms, decoded_frame_hashes, cfg)
    result = deepcopy(baseline)
    result.update(method_version="paired_fixed_heel_or_toe_pivot_development_v1",
                  baseline_candidate_count=baseline["candidate_count"], pivot_added_count=0)
    if baseline["landings"] is None:
        return result
    first, last = interval_frames
    matrices = _transform_map(transforms)
    views, _ = _series(_read_records(records, first, last), first, last, matrices, decoded_frame_hashes, cfg)
    extra = []
    for side, indices in SHOES.items():
        for index, label in zip(indices, ("heel", "toe")):
            normal, _, _ = _propose_view(views[False], side, matrices, fps, cfg, [index])
            mirrored, _, _ = _propose_view(views[True], side, matrices, fps, cfg, [index])
            used = set()
            for event in normal:
                candidates = []
                for mirror_id, mirror in enumerate(mirrored):
                    if mirror_id in used or abs(event["frame"] - mirror["frame"]) / fps > cfg.maximum_pair_gap_seconds:
                        continue
                    overlap = sorted(set(event["stance_frames"]) & set(mirror["stance_frames"]))
                    if (len(overlap) >= cfg.minimum_stance_observations
                            and (overlap[-1] - overlap[0]) / fps >= cfg.minimum_stance_seconds
                            and all((b - a) / fps <= cfg.maximum_pair_gap_seconds for a, b in zip(overlap, overlap[1:]))):
                        candidates.append((abs(event["frame"] - mirror["frame"]), mirror_id, mirror, overlap))
                if not candidates:
                    continue
                _, mirror_id, mirror, overlap = min(candidates, key=lambda item: (item[0], item[1]))
                used.add(mirror_id)
                extra.append(dict(frame=max(event["frame"], mirror["frame"]), foot=side,
                                  support_landmark=label, support_landmark_index=index,
                                  stance_frames=overlap, normal=event, mirrored_inverse_mapped=mirror,
                                  distinct_decoded_stance_observations=len(overlap),
                                  common_stance_seconds=(overlap[-1] - overlap[0]) / fps,
                                  evidence_type="paired_fixed_landmark_pivot_candidate"))
    # Preserve frozen baseline events and merge anatomically identical support
    # phases using the pre-existing same-foot separation, without label tuning.
    accepted = deepcopy(baseline["landings"])
    for event in sorted(extra, key=lambda row: (row["frame"], row["foot"], row["support_landmark_index"])):
        if any(old["foot"] == event["foot"] and abs(old["frame"] - event["frame"]) / fps < cfg.minimum_same_foot_separation_seconds
               for old in accepted):
            continue
        accepted.append(event)
        result["pivot_added_count"] += 1
    accepted.sort(key=lambda row: (row["frame"], row["foot"]))
    result.update(landings=accepted, candidate_count=len(accepted), observed_proposal_count=len(accepted),
                  status="paired_visible_transition_proposals" if accepted else "no_paired_visible_transition_proposal")
    return result

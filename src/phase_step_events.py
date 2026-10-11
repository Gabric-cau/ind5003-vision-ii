"""Gap-aware ankle-phase extrema for review, not verified foot contacts.

Unlike the historical counter, this proposes events across the whole visible
pre-contact track, leaving temporal classification to penalty_phases. Three
fixed prominence settings expose sensitivity; none is fitted to user counts.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks


def propose_step_events(frames, fps, hips, left_ankles, right_ankles, heights,
                        confidences, *, kick_frame=None, event_end_frame=None):
    raw_frames = np.asarray(frames)
    if raw_frames.ndim != 1 or not np.all(np.isfinite(raw_frames)) or not np.all(raw_frames == np.floor(raw_frames)):
        raise ValueError("Frames must be finite integer IDs")
    frames = raw_frames.astype(int)
    hips, left, right = (np.asarray(a, float) for a in (hips, left_ankles, right_ankles))
    heights, conf = np.asarray(heights, float), np.asarray(confidences, float)
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError("fps must be positive")
    n = len(frames)
    if any(a.shape != (n, 2) for a in (hips, left, right)) or conf.shape != (n, 3) or heights.shape != (n,):
        raise ValueError("Aligned Nx2 positions, Nx3 confidences and N heights required")
    if n and (frames[0] < 0 or np.any(np.diff(frames) <= 0)):
        raise ValueError("Frames must be unique, increasing and nonnegative")
    result = {"method": "relative_ankle_phase_extrema_sensitivity_v1", "events": [],
              "variants": {}, "human_verified": False, "counts_are_contacts": False,
              "prominence_units": "median_person_box_height", "warnings": [],
              "thresholds": {"sensitive": .025, "central": .04, "strict": .06},
              "direction": None, "direction_source": None}
    valid = (np.all(np.isfinite(hips), axis=1) & np.all(np.isfinite(left), axis=1)
             & np.all(np.isfinite(right), axis=1) & np.all(np.isfinite(conf), axis=1)
             & (np.min(conf, axis=1) >= .35) & np.isfinite(heights) & (heights > 0))
    limit = event_end_frame if event_end_frame is not None else kick_frame
    if limit is not None:
        valid &= frames <= limit
    if np.count_nonzero(valid) < 8:
        result["warnings"].append("insufficient_valid_pose")
        return result
    scale = float(np.median(heights[valid]))
    relative_difference = (left - right) / scale
    anchor = kick_frame if kick_frame is not None else frames[valid][-1]
    tail = valid & (frames >= anchor - round(fps)) & (frames <= anchor)
    # Ankle-difference PCA avoids treating camera pan as approach direction.
    # Its sign cannot identify an anatomical landing foot, so extrema polarity
    # is retained and the foot is deliberately left unspecified.
    samples = relative_difference[tail if np.count_nonzero(tail) >= 8 else valid]
    centred = samples - np.median(samples, axis=0)
    _, singular, axes = np.linalg.svd(centred, full_matrices=False)
    if len(singular) < 1 or singular[0] < .02:
        result["warnings"].append("ankle_phase_signal_too_small")
        return result
    axis = axes[0]
    if axis[np.argmax(abs(axis))] < 0:
        axis = -axis
    result.update({"direction": axis.tolist(), "direction_source": "precontact_ankle_difference_PCA",
                   "median_box_height_px": scale,
                   "axis_energy_fraction": float(singular[0] ** 2 / max(1e-12, np.sum(singular ** 2)))})
    if kick_frame is None:
        result["direction_source"] = "unanchored_track_tail_ankle_difference_PCA"
        result["warnings"].append("contact_anchor_unavailable_scope_not_known_precontact")
    if result["axis_energy_fraction"] < .65:
        result["warnings"].append("ankle_phase_axis_ambiguous")
    result["warnings"].append("precontact_axis_may_include_kicking_swing")
    grid = np.arange(frames[0], frames[-1] + 1)
    signal = np.full(len(grid), np.nan)
    known = np.flatnonzero(valid)
    signal[frames[known] - grid[0]] = relative_difference[known] @ axis
    max_gap = max(1, round(.12 * fps))
    for a, b in zip(known[:-1], known[1:]):
        gap = frames[b] - frames[a]
        if 1 < gap <= max_gap:
            lo, hi = frames[a] - grid[0], frames[b] - grid[0]
            signal[lo:hi + 1] = np.linspace(signal[lo], signal[hi], hi - lo + 1)
    bounds = np.diff(np.r_[False, np.isfinite(signal), False].astype(int))
    runs = zip(np.flatnonzero(bounds == 1), np.flatnonzero(bounds == -1))
    runs = list(runs)
    smoothing = max(3, round(.10 * fps) | 1)
    for lo, hi in runs:
        signal[lo:hi] = median_filter(signal[lo:hi], size=smoothing, mode="nearest")
    for name, prominence in result["thresholds"].items():
        events = []
        for lo, hi in runs:
            if hi - lo < 6:
                continue
            candidates = []
            for polarity in (1, -1):
                indices, props = find_peaks(polarity * signal[lo:hi], prominence=prominence,
                                           distance=max(1, round(.22 * fps)))
                candidates.extend({"frame": int(grid[lo + i]), "polarity": polarity,
                                   "prominence": float(p), "foot": "unknown"}
                                  for i, p in zip(indices, props["prominences"]))
            kept = []
            for event in sorted(candidates, key=lambda e: (-e["prominence"], e["frame"])):
                if all(abs(event["frame"] - old["frame"]) >= max(1, round(.12 * fps)) for old in kept):
                    kept.append(event)
            events.extend(sorted(kept, key=lambda e: e["frame"]))
        result["variants"][name] = events
    result["events"] = result["variants"]["central"]
    result["candidate_count_sensitivity"] = {name: len(v) for name, v in result["variants"].items()}
    result["signal"] = {"frames": grid.tolist(), "values": [float(x) if np.isfinite(x) else None for x in signal]}
    result["warnings"].extend(["extrema_are_not_verified_landing_contacts", "anatomical_landing_foot_unspecified",
                               "kick_swing_near_contact_requires_review", "same_count_across_thresholds_does_not_validate_count"])
    return result

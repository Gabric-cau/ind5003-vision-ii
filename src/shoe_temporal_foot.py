"""Contact-aligned anatomical-foot features preserving motion order."""
from copy import deepcopy

import numpy as np

from src.shoe_supervised_model import BASE_FEATURE_NAMES, extract_evidence


OFFSETS_SECONDS = (-.24, -.12, 0., .12, .24)
SUPPORT_RADIUS_SECONDS = .06


def temporal_foot_vector(inputs, records):
    """Use only measured paired poses and an automatically inferred contact bracket."""
    expanded = deepcopy(inputs)
    expanded["interval_frames"] = list(inputs["inference_interval_frames"])
    evidence = extract_evidence(expanded, records)
    return vector_from_rows(evidence["landing_rows"], inputs["contact_bracket_frames"], inputs["fps"])


def vector_from_rows(rows, bracket, fps):
    if not bracket or len(bracket) != 2:
        return None
    center = float(np.mean(bracket))
    dimensions = len(BASE_FEATURE_NAMES)
    blocks, supported = [], 0
    for seconds in OFFSETS_SECONDS:
        target = center + seconds * fps
        values = []
        for side in ("left", "right"):
            nearby = [np.asarray(row["x"])[2 * dimensions:3 * dimensions]
                      for row in rows if row["foot"] == side
                      and abs(row["frame"] - target) <= SUPPORT_RADIUS_SECONDS * fps + 1e-8]
            values.append(np.mean(nearby, axis=0) if nearby else None)
        if any(value is None for value in values):
            blocks.append(np.zeros(dimensions))
        else:
            blocks.append(values[1] - values[0])
            supported += 1
    return np.concatenate(blocks) if supported >= 3 else None

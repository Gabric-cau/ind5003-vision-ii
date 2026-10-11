"""Source-only choice between independently produced contact candidates."""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ContactSelectionPolicy:
    minimum_common_stationary_frames: int = 3
    minimum_spot_tolerance_px: float = 6.
    spot_tolerance_diameters: float = 1.
    final_tie_preference: str = 'm'


def select_contact_producer(candidates, policy=None):
    """Inputs contain contact evidence only, with no outcome or reference labels."""
    cfg = policy or ContactSelectionPolicy()
    available = {label: row for label, row in candidates.items() if row is not None}
    result = {'selected_label': None, 'reason': 'neither_producer_supports_contact',
              'policy': asdict(cfg), 'compatible': None,
              'outcome_fields_used_for_selection': False, 'reference_labels_used_for_selection': False}
    if len(available) == 1:
        result.update(selected_label=next(iter(available)), reason='only_one_source_bound_contact_candidate')
        return result
    if not available:
        return result
    if len(available) != 2:
        raise ValueError('Contact selection requires exactly two independent producers')
    first, second = available.values()
    if first['contact_associated_track_id'] != second['contact_associated_track_id']:
        result.update(reason='contact_actor_disagreement', compatible=False)
        return result
    a, b = first['contact_bracket_frames'], second['contact_bracket_frames']
    overlap = [max(a[0], b[0]), min(a[1], b[1])]
    result['bracket_overlap_frames'] = overlap
    if overlap[0] > overlap[1]:
        result.update(reason='canonical_contact_brackets_do_not_overlap', compatible=False)
        return result
    maps = [{row['frame']: row for row in candidate['stationary_ball_observations']} for candidate in (first, second)]
    common = sorted(set(maps[0]) & set(maps[1]))
    result['common_stationary_frames'] = common
    if len(common) < cfg.minimum_common_stationary_frames:
        result.update(reason='insufficient_shared_stationary_source_frames', compatible=False)
        return result
    separations = [float(np.linalg.norm(np.asarray(maps[0][frame]['xy'])-maps[1][frame]['xy'])) for frame in common]
    diameters = [row.get('diameter', (row['bbox'][2]-row['bbox'][0]+row['bbox'][3]-row['bbox'][1])/2)
                 for rows in maps for frame in common for row in (rows[frame],)]
    separation = float(np.median(separations))
    tolerance = max(cfg.minimum_spot_tolerance_px, cfg.spot_tolerance_diameters*float(np.median(diameters)))
    result.update(common_stationary_center_median_distance_px=separation,
                  common_stationary_center_tolerance_px=tolerance)
    if separation > tolerance:
        result.update(reason='stationary_ball_spot_disagreement', compatible=False)
        return result
    def rank(item):
        label, contact = item
        bracket = contact['contact_bracket_frames']
        return (contact['score'], -(bracket[1]-bracket[0]),
                min(row['confidence'] for row in contact['departure_ball_observations']),
                label == cfg.final_tie_preference, label)
    result.update(selected_label=max(available.items(), key=rank)[0],
                  reason='compatible_source_evidence_score_then_interval_then_departure_confidence', compatible=True)
    return result

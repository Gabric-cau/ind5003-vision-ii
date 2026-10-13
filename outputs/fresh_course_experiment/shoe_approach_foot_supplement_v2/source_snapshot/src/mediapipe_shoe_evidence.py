"""Source-coordinate geometry for MediaPipe's 33 anatomical landmarks."""
import numpy as np


MIRROR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12),
                (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
                (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
COCO_MAP = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
SHOE_PAIRS = {'left': [29, 31], 'right': [30, 32]}


def restore_landmarks(normalized_xy, confidence, crop_xyxy, reflected=False):
    points, confidence, crop = np.asarray(normalized_xy, float), np.asarray(confidence, float), np.asarray(crop_xyxy, float)
    if (points.shape != (33, 2) or confidence.shape != (33,) or crop.shape != (4,)
            or not np.isfinite(points).all() or not np.isfinite(confidence).all()
            or not np.isfinite(crop).all() or np.any(confidence < 0) or np.any(confidence > 1)
            or crop[2] <= crop[0] or crop[3] <= crop[1]):
        raise ValueError('Landmarks require finite 33-point xy/confidence and a positive source crop')
    points = points * (crop[2:] - crop[:2])
    confidence = confidence.copy()
    if reflected:
        points[:, 0] = crop[2] - crop[0] - 1 - points[:, 0]
        for left, right in MIRROR_PAIRS:
            points[[left, right]] = points[[right, left]]
            confidence[[left, right]] = confidence[[right, left]]
    inside = ((points[:, 0] >= 0) & (points[:, 0] <= crop[2] - crop[0] - 1)
              & (points[:, 1] >= 0) & (points[:, 1] <= crop[3] - crop[1] - 1))
    confidence[~inside] = 0.
    return points + crop[:2], confidence


def segment_distance(point, first, second):
    point, first, second = np.asarray(point, float), np.asarray(first, float), np.asarray(second, float)
    if any(value.shape != (2,) or not np.isfinite(value).all() for value in (point, first, second)):
        raise ValueError('Segment distance requires finite xy points')
    direction = second - first
    length = float(direction @ direction)
    fraction = float(np.clip((point - first) @ direction / length, 0, 1)) if length > 1e-12 else 0.
    return float(np.linalg.norm(point - (first + fraction * direction)))


def coco_samples(rows, *, shoe_centroids=False):
    output = []
    for row in rows:
        if row.get('landmarks_source_xy') is None:
            continue
        all_points = np.asarray(row['landmarks_source_xy'], float)
        all_confidence = np.asarray(row['landmark_confidence'], float)
        points, confidence = all_points[COCO_MAP].copy(), all_confidence[COCO_MAP].copy()
        if shoe_centroids:
            for index, side in enumerate(('left', 'right')):
                points[15 + index] = all_points[SHOE_PAIRS[side]].mean(axis=0)
                confidence[15 + index] = all_confidence[SHOE_PAIRS[side]].min()
        output.append(dict(frame=row['frame'], bbox=row['anchor_bbox'],
                           keypoints=points.tolist(), confidence=confidence.tolist()))
    return output


def shoe_ball_distances(rows, target, scale, confidence_min=.5):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Observed preparation scale must be finite and positive')
    output = []
    for row in rows:
        if row.get('landmarks_source_xy') is None:
            continue
        points, confidence = np.asarray(row['landmarks_source_xy']), np.asarray(row['landmark_confidence'])
        values = dict(frame=row['frame'])
        for side, indices in SHOE_PAIRS.items():
            values[side] = dict(confident=bool(confidence[indices].min() >= confidence_min),
                heel_to_foot_index_source_xy=points[indices].tolist(),
                minimum_confidence=float(confidence[indices].min()),
                stationary_point_distance_source_px=segment_distance(target, *points[indices]),
                stationary_point_distance_body=segment_distance(target, *points[indices]) / scale)
        output.append(values)
    return output


def source_body_box(points, confidence, confidence_min=.5):
    points, confidence = np.asarray(points, float), np.asarray(confidence, float)
    indices = [0, 11, 12, 23, 24, 25, 26, 27, 28]
    if min(confidence[[11, 12, 23, 24]]) < confidence_min:
        return None
    visible = [index for index in indices if confidence[index] >= confidence_min]
    if len(visible) < 6:
        return None
    return np.r_[points[visible].min(axis=0), points[visible].max(axis=0)].tolist()

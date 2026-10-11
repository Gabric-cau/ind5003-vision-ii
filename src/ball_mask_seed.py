"""Experimental pixel masks for a ball start; mask support is not identity truth.

Only pre-kick neural observations with measured bright/low-saturation pixels
are eligible. Flight detections remain intact: a white mask can miss a blurred
or shadowed ball and must not delete later real observations.
"""
from dataclasses import dataclass, asdict
import types
import cv2
import numpy as np
from src.event_ball_tracking import select_stationary_seed
from src.ball_continuity import track_ball_continuity


@dataclass(frozen=True)
class MaskSeedConfig:
    maximum_saturation: int = 105
    minimum_value: int = 145
    minimum_white_pixels: int = 3
    minimum_white_fraction: float = .10
    tight_foot_radius_px: float = 35.
    maximum_green_excess: float | None = None
    maximum_seed_motion_gray: float | None = None
    seed_after_kick_frames: int = 3


def white_mask(frame, config=None):
    cfg = config or MaskSeedConfig()
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError('Expected uint8 BGR frame')
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, cfg.minimum_value), (179, cfg.maximum_saturation, 255))
    if cfg.maximum_green_excess is not None:
        b, g, r = frame.astype(float).transpose(2, 0, 1)
        mask[g - (b + r) / 2 > cfg.maximum_green_excess] = 0
    return mask


def measure_mask_support(mask, detection, config=None):
    cfg = config or MaskSeedConfig()
    h, w = mask.shape
    x, y = detection['x'], detection['y']
    bw, bh = detection['width'], detection['height']
    xa, xb = max(0, int(np.floor(x - bw / 2))), min(w, int(np.ceil(x + bw / 2)))
    ya, yb = max(0, int(np.floor(y - bh / 2))), min(h, int(np.ceil(y + bh / 2)))
    patch = mask[ya:yb, xa:xb]
    pixels = int(np.count_nonzero(patch))
    fraction = pixels / patch.size if patch.size else 0.
    return {'white_pixels': pixels, 'white_fraction': round(fraction, 6),
        'mask_support': bool(pixels >= cfg.minimum_white_pixels and fraction >= cfg.minimum_white_fraction),
        'football_identity_established': False}


def choose_seed(detections, kick, anchors, mask_evidence, *, use_mask, radius, config=None):
    # Filtering affects seed selection only, not the later candidate pool.
    cfg = config or MaskSeedConfig()
    selected = {f: [d for d in ds if d['frame'] <= kick + cfg.seed_after_kick_frames and
                    (cfg.maximum_seed_motion_gray is None or d.get('motion_mean_abs_gray') is None or
                     d['motion_mean_abs_gray'] <= cfg.maximum_seed_motion_gray) and (not use_mask or
                       mask_evidence.get(d['detection_id'], {}).get('mask_support', False))]
                for f, ds in detections.items()}
    seed, diag = select_stationary_seed(selected, kick, anchors, radius=radius)
    return seed, {**diag, 'pixel_mask_used_for_seed': use_mask, 'foot_radius_px': radius,
                  'mask_candidate_identity_verified': False, 'manual_seed_input': False}


def associate_with_seed(detections, kick, anchors, goal, seed, seed_diag, *, tracker=None, **kwargs):
    """Keep the production association unchanged, replacing only its seed hook.

    A private function namespace avoids mutating shared production code/global
    state. The supplied seed must be measured members of the current input.
    All bootstrap hypotheses and later association rules remain the original.
    """
    ids = {(d['frame'], d['detection_id']) for ds in detections.values() for d in ds}
    if any((d['frame'], d['detection_id']) not in ids for d in seed):
        raise ValueError('Seed is not an input observation')
    original = tracker or track_ball_continuity
    namespace = {**original.__globals__, 'select_stationary_seed': lambda *_a, **_k: (seed, seed_diag)}
    private = types.FunctionType(original.__code__, namespace, original.__name__, original.__defaults__)
    private.__kwdefaults__ = dict(original.__kwdefaults__ or {})
    path, diag = private(detections, kick, anchors, goal, initial_path=None, **kwargs)
    diag['seed_method'] = 'white_pixel_mask' if seed_diag['pixel_mask_used_for_seed'] else 'radius_control'
    return path, diag

import numpy as np
import cv2
import pytest

from src.annotation_provenance import is_human_verified, require_verified
from src.field_masks import build_field_mask, foot_region_support, background_feature_mask, white_ball_proposals, pitch_line_candidates
from src.homography import PitchHomographyTransformer
from src.kicker_analysis import KickerFeatureExtractor


def scene():
    frame = np.zeros((240, 400, 3), np.uint8)
    frame[70:] = (40, 140, 40)
    cv2.rectangle(frame, (160, 100), (190, 200), (0, 0, 220), -1)
    cv2.circle(frame, (230, 190), 4, (255, 255, 255), -1)
    return frame


def test_field_fills_player_hole_and_preserves_off_pitch_boundary():
    field = build_field_mask(scene())
    assert field.valid and field.region[150, 175] == 255
    assert field.region[20, 175] == 0
    assert foot_region_support(field, (160, 100, 190, 200)) > .9
    assert foot_region_support(field, (10, 5, 30, 40)) == 0


def test_unusable_mask_does_not_reject_unknown_foot_position():
    field = build_field_mask(np.zeros((100, 100, 3), np.uint8))
    assert not field.valid
    assert foot_region_support(field, (0, 0, 10, 10)) is None


def test_background_excludes_players_and_preserves_field():
    mask, stats = background_feature_mask(scene(), [(160, 100, 190, 200)])
    assert mask[150, 175] == 0 and mask[220, 300] == 255
    assert stats["field_used"]


def test_white_candidates_are_not_verified_ball_detections():
    _, proposals = white_ball_proposals(scene())
    assert any(abs(p["x"] - 230) < 2 for p in proposals)
    assert all(p["source"] == "hsv_white_candidate_not_verified" for p in proposals)


def test_legacy_manual_and_ai_tags_are_not_human_verification():
    assert not is_human_verified({"annotation_method": "manual_frame_review", "confidence": "high"})
    assert not is_human_verified({"annotator_type": "ai_assisted", "human_verified": "true"})
    with pytest.raises(ValueError, match="unverified annotations"):
        require_verified([{"clip_id": "p1"}], "steps")
    assert is_human_verified({"annotator_type": "human", "human_verified": "true",
                              "reviewer_id": "reviewer_1", "reviewed_at": "2026-10-08"})


def test_homography_rejects_degenerate_and_tracks_unit():
    with pytest.raises(ValueError, match="collinear"):
        PitchHomographyTransformer(np.array([[0, 0], [1, 0], [2, 0], [3, 0]]),
                                   np.array([[0, 0], [1, 0], [2, 0], [3, 0]]))
    src = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    transformer = PitchHomographyTransformer(src, src / 10, destination_unit="m")
    assert np.allclose(transformer.transform_trajectory([(50, 50)]), [(5, 5)])
    assert transformer.metric_calibrated
    assert transformer.diagnostics["independently_validated"] is False


def test_uncalibrated_pixels_are_not_reported_as_metres():
    result = KickerFeatureExtractor().analyze_runup_trajectory([(0, 0, 0), (1, 30, 40)])
    assert result["runup_distance_m"] is None
    assert result["runup_speed_mps"] is None
    assert result["coordinate_unit"] == "px" and result["runup_displacement"] == 50


def test_masked_edges_find_white_line_without_creating_roi_border():
    frame = scene()
    cv2.line(frame, (20, 220), (360, 220), (255, 255, 255), 3)
    edges, lines = pitch_line_candidates(frame, [(160, 100, 190, 200)])
    assert any(abs(y1 - 220) < 5 and abs(y2 - 220) < 5 for x1, y1, x2, y2 in lines)
    assert not np.any(edges[:65])

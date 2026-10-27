import numpy as np
import pytest

from scripts import evaluate_worldcup_holdout_once as evaluation


def test_failed_height_predictions_remain_in_denominator():
    result = evaluation.category_metrics(["low", "middle", "high"], ["low", "unknown", "low"],
                                         ["low", "middle", "high"])
    assert result["reference_weight"] == 3
    assert result["accuracy_all_eligible"] == pytest.approx(1 / 3)
    assert result["output_coverage"] == pytest.approx(2 / 3)
    assert result["accuracy_covered"] == .5


def test_camera_scores_time_overlap_not_window_count():
    references = [{"filename": "a.mp4", "class": "behind_kicker", "start_seconds": 0., "end_seconds_exclusive": 4.},
                  {"filename": "a.mp4", "class": "goal_area", "start_seconds": 4., "end_seconds_exclusive": 6.}]
    predictions = {"a.mp4": [{"class": "behind_kicker", "start_seconds": 0., "end_seconds_exclusive": 3.},
                             {"class": "goal_area", "start_seconds": 3., "end_seconds_exclusive": 6.}]}
    result = evaluation.score_camera(references, predictions)
    assert result["time_weighted"]["accuracy_all_eligible"] == pytest.approx(5 / 6)
    assert result["reference_interval_majority"]["accuracy_all_eligible"] == 1
    assert result["oracle_cropped_inference"] is False


def test_partial_camera_reference_does_not_score_unlabelled_time():
    references = [{"filename": "a.mp4", "class": "behind_kicker", "start_seconds": 13., "end_seconds_exclusive": 19.36}]
    predictions = {"a.mp4": [{"class": "other_side", "start_seconds": 0., "end_seconds_exclusive": 13.},
                             {"class": "behind_kicker", "start_seconds": 13., "end_seconds_exclusive": 20.}]}
    result = evaluation.score_camera(references, predictions)
    assert result["time_weighted"]["reference_weight"] == pytest.approx(6.36)
    assert result["time_weighted"]["accuracy_all_eligible"] == 1
    assert result["masked_localization"]["per_class"]["other_side"]["prediction_runs"] == 0


def test_missing_camera_predictions_count_as_unknown():
    references = [{"filename": "a.mp4", "class": "goal_area", "start_seconds": 2., "end_seconds_exclusive": 5.}]
    result = evaluation.score_camera(references, {})
    assert result["time_weighted"]["accuracy_all_eligible"] == 0
    assert result["time_weighted"]["output_coverage"] == 0
    assert result["reference_interval_majority"]["accuracy_all_eligible"] == 0


def test_overlapping_prediction_windows_are_rejected():
    references = [{"filename": "a.mp4", "class": "goal_area", "start_seconds": 0., "end_seconds_exclusive": 5.}]
    predictions = {"a.mp4": [{"class": "goal_area", "start_seconds": 0., "end_seconds_exclusive": 5.}] * 2}
    with pytest.raises(ValueError, match="double-count"):
        evaluation.score_camera(references, predictions)


def test_reference_mapping_is_frozen_and_inexact_partial_is_excluded():
    label = next(key for key, value in evaluation.camera.LABEL_MAP.items() if value == "behind_kicker")
    base = {"export_index": 17, "video_file": "videos/match_17.mp4", "original_shot_id": "a",
            "camera_label": label, "camera_annotation_source": "human_user_observation_in_chat",
            "camera_label_start_seconds": 13., "camera_label_end_seconds_exclusive": 19.36,
            "camera_label_scope": "partial_shot"}
    rows = [base, {**base, "original_shot_id": "b", "camera_label": "unmapped"},
            {**base, "original_shot_id": "c", "camera_label_start_seconds": None}]
    admitted, excluded = evaluation.camera_references({"shot_timeline": rows}, {17})
    assert len(admitted) == 1 and len(excluded) == 2
    assert admitted[0]["start_seconds"] == 13.


def test_localization_merges_adjacent_same_class_and_matches_one_to_one():
    references = [{"filename": "a.mp4", "class": "goal_area", "start_seconds": 0., "end_seconds_exclusive": 2.},
                  {"filename": "a.mp4", "class": "goal_area", "start_seconds": 2., "end_seconds_exclusive": 4.}]
    predictions = {"a.mp4": [{"class": "goal_area", "start_seconds": 0., "end_seconds_exclusive": 4.}]}
    result = evaluation.masked_localization(references, predictions)
    assert result["per_class"]["goal_area"]["reference_runs"] == 1
    assert result["per_class"]["goal_area"]["matched_runs"] == 1
    assert result["boundary_errors"] is None

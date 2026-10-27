import json

import pytest

from scripts import evaluate_offline_video_batch as evaluation


def test_unknown_prediction_is_miss_but_not_in_conditional_denominator():
    rows = [{"core_id": "a", "reference": "left", "prediction": "left"},
            {"core_id": "b", "reference": "right", "prediction": None}]
    score = evaluation.categorical_metrics(rows, ("left", "right"))
    assert score["coverage_on_comparable_references"] == 0.5
    assert score["strict_accuracy_unknown_as_incorrect"] == 0.5
    assert score["conditional_accuracy_on_available"] == 1.0
    assert score["per_class"]["right"]["false_negative_including_unknown"] == 1


def test_mismatched_coordinate_convention_is_not_scored_as_agreement():
    rows = [{"core_id": "a", "reference": "left", "prediction": "left", "comparable": False,
             "reason": "image_vs_behind_kicker"}]
    score = evaluation.categorical_metrics(rows, evaluation.DIRECTIONS)
    assert score["strict_accuracy_unknown_as_incorrect"] is None
    assert score["correct_events"] == 0
    assert score["comparable_reference_events"] == 0


def test_structured_partial_outputs_do_not_claim_terminal_or_strict_geometry():
    assert evaluation.final_position_value({"event_final_position": None,
        "last_observed": {"xy": [3, 4]}}) is None
    assert evaluation.strict_height_value({"value": "low",
        "strict_mask_pose_category": "unknown"}) == "unknown"


def test_coarse_timing_keeps_unknowns_in_tolerance_denominator():
    rows = [{"core_id": "a", "reference": 1.0, "prediction": 1.1},
            {"core_id": "b", "reference": 2.0, "prediction": None}]
    score = evaluation.numeric_metrics(rows, (0.2,))
    assert score["mae_on_available"] == pytest.approx(0.1)
    assert score["mean_signed_prediction_minus_reference"] == pytest.approx(0.1)
    assert score["tolerances"]["0.2"]["strict_hit_fraction_unknown_as_miss"] == 0.5
    assert score["tolerances"]["0.2"]["conditional_hit_fraction"] == 1.0


def test_full_steps_and_visible_window_counts_remain_unaligned():
    reference = {"clip_id": "event", "video_sha256": "a" * 64, "foot": "right",
        "keeper_direction": "left", "direction_basis": "source_image_x_right", "outcome": "goal",
        "contact_seconds": 1.0, "contact_precision": "coarse", "full_runup_steps": 5,
        "first_recorded_camera_cut_seconds": 2.0, "keeper_height": "low", "ball_height": "low",
        "ball_direction": "left", "ball_height_text_conflict": False, "keeper_movement_direction": "left",
        "final_position_text": "left goal"}
    prediction = {"required_outputs": {"kicking_foot": "right", "visible_runup_steps": 5,
        "keeper_dive_direction": "image_left", "goal_outcome": "goal_likely"},
        "source": {"fps": 30, "slice_end_sec": 3}, "trained_contact": {"prediction": {"time_seconds": 1.0}},
        "auxiliary_models": {"trained_shoes": {"count_complete": False, "full_approach_visible": False},
                             "trained_ball": {"predictions": {"ball_direction": {"labels": ["left"]}}}}}
    report = evaluation.evaluate({"core_01": prediction}, {"core_01": reference})
    steps = report["numeric"]["full_runup_steps"]
    assert steps["mae_on_available"] is None
    assert steps["comparable_reference_events"] == 0
    assert steps["rows"][0]["unaligned_difference_diagnostic_only"] == 0
    assert report["categorical"]["ball_direction_auxiliary"]["correct_events"] == 0
    assert report["output_coverage"]["observed_window_step_count"]["fraction"] == 1.0


@pytest.fixture
def completed_batch(tmp_path):
    video_bytes = b"exact input video bytes"
    pipeline = tmp_path / "events/core_01/attempt_001/pipeline"
    source_dir = pipeline / "source"
    source_dir.mkdir(parents=True)
    video = source_dir / "unseen_penalty.mp4"
    video.write_bytes(video_bytes)
    video_sha = evaluation.sha(video)
    analysis = {"input_sha256": video_sha, "all_stages_executed_successfully": True,
        "manual_annotations_used": False, "required_outputs": {},
        "source": {"clip_filename": video.name, "clip_sha256": video_sha}}
    analysis_path = pipeline / "analysis.json"
    analysis_path.write_text(json.dumps(analysis))
    manifest_path = tmp_path / "batch_manifest.json"
    manifest_path.write_text(json.dumps({"bindings": {"samples": [{"core_id": "core_01", "video_sha256": video_sha}]}}))
    digest = evaluation.sha(manifest_path)
    summary = {"complete": True, "failed_events": 0, "batch_bindings_sha256": digest,
        "expected_events": 1, "events": [{"core_id": "core_01", "status": "success",
            "video_sha256": video_sha, "batch_bindings_sha256": digest,
            "analysis_path": str(analysis_path.relative_to(tmp_path)), "analysis_sha256": evaluation.sha(analysis_path)}]}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    return tmp_path, video, analysis_path


def test_incomplete_batch_refuses_evaluation_before_any_reference_is_opened(completed_batch):
    directory, _, _ = completed_batch
    summary = evaluation.read(directory / "summary.json")
    summary["complete"] = False
    (directory / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="every requested prediction"):
        evaluation.load_completed_batch(directory)


@pytest.mark.parametrize("target", ["video", "prediction"])
def test_changed_video_or_prediction_rejected_before_reference_comparison(completed_batch, target):
    directory, video, prediction = completed_batch
    path = video if target == "video" else prediction
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="differs|changed"):
        evaluation.load_completed_batch(directory)


def test_valid_batch_binds_event_and_video_before_reference_comparison(completed_batch):
    directory, _, _ = completed_batch
    predictions, samples, binding = evaluation.load_completed_batch(directory)
    assert set(predictions) == set(samples) == {"core_01"}
    assert binding["events"][0]["video_sha256"] == samples["core_01"]["video_sha256"]

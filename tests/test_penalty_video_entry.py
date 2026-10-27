import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.analyze_penalty_video import prepare_source, stage_commands, summarize, endpoint_output, contact_stage_command
from scripts import analyze_penalty_video as entry


def test_inference_isolates_user_caches_for_all_stage_processes(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    for key in ("MPLCONFIGDIR", "YOLO_CONFIG_DIR", "TORCH_HOME"):
        monkeypatch.setenv(key, "unusable-user-directory")
    env = entry.stage_environment(2)
    for key, folder in (("MPLCONFIGDIR", "matplotlib"), ("YOLO_CONFIG_DIR", "ultralytics"),
                        ("TORCH_HOME", "torch")):
        assert env[key] == str(tmp_path / ".runtime-cache" / folder)
        assert os.environ[key] == "unusable-user-directory"
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == "2"


def test_new_video_manifest_does_not_supply_answers_or_invent_shootout_order(tmp_path):
    video = tmp_path / "anonymous.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30., (64, 48))
    for _ in range(30):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    source = prepare_source(video, tmp_path / "source")
    assert source["penalty_number"] is None
    assert source["kicker"] == source["team"] == "unknown"
    assert source["original_slice_end_frame_in_clip"] == 30
    assert source["source_start_frame"] == 0
    manifest = json.loads((tmp_path / "source/source_manifest.json").read_text())
    assert manifest["manual_annotations_used"] is False
    assert (tmp_path / "source" / source["clip_filename"]).read_bytes() == video.read_bytes()


def test_invalid_video_fails_before_manifest_is_written(tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="Cannot open"):
        prepare_source(broken, tmp_path / "source")
    assert not (tmp_path / "source").exists()


@pytest.mark.parametrize("extension", [".mov", ".mkv", ".webm"])
def test_non_mp4_is_rejected_before_decode_or_directory_creation(tmp_path, monkeypatch, extension):
    def unexpected_decode(*args, **kwargs):
        pytest.fail("Unsupported extension must fail before video decoding")

    monkeypatch.setattr(cv2, "VideoCapture", unexpected_decode)
    folder = tmp_path / "new_output" / "source"
    with pytest.raises(ValueError, match="requires an MP4"):
        prepare_source(tmp_path / ("video" + extension), folder)
    assert not folder.parent.exists()


def test_all_empirical_stages_are_called_without_reference_files(tmp_path):
    commands = stage_commands(tmp_path / "source", tmp_path, Path("pose.pt"), Path("ball.pt"), "cpu")
    assert [name for name, _, _ in commands] == ["pose", "ball", "foot", "phase", "pitch", "camera", "height", "endpoint"]
    for _, module, arguments in commands:
        assert module != "src.pipeline"
        assert "--clips" in arguments and "unseen_penalty" in arguments
        assert not any("annotation" in str(value) or "labels" in str(value) for value in arguments)


def test_specialist_only_supplements_ball_stage_and_keeps_coco_for_people(tmp_path):
    commands = stage_commands(tmp_path / "source", tmp_path, Path("pose.pt"), Path("coco.pt"), "cpu",
                              Path("trained_ball.pt"), "tiles", 1920)
    for name, _, arguments in commands:
        if name == "ball":
            assert arguments[arguments.index("--model") + 1] == Path("coco.pt")
            assert arguments[arguments.index("--ball-specialist") + 1] == Path("trained_ball.pt")
            assert arguments[arguments.index("--specialist-mode") + 1] == "tiles"
            assert arguments[arguments.index("--specialist-imgsz") + 1] == 1920
        else:
            assert "--ball-specialist" not in arguments


def test_low_memory_batch_setting_reaches_every_batched_visual_stage(tmp_path):
    commands = stage_commands(tmp_path / "source", tmp_path, Path("pose.pt"), Path("ball.pt"),
                              "cuda:0", batch_size=1)
    for name, _, arguments in commands:
        if name in {"pose", "ball", "foot"}:
            assert arguments[arguments.index("--batch-size") + 1] == "1"
    with pytest.raises(ValueError, match="batch_size"):
        stage_commands(tmp_path / "source", tmp_path, Path("pose.pt"), Path("ball.pt"),
                       "cuda:0", batch_size=0)


@pytest.mark.parametrize("maximum,expected", [(12., "scripts.predict_video_contact"),
                                               (60., "scripts.predict_video_contact_fullshot")])
def test_contact_stage_uses_the_protocol_bound_to_its_model(tmp_path, maximum, expected):
    (tmp_path / "deployment_metadata.json").write_text(json.dumps({"visual_settings": {"max_shot_sec": maximum}}))
    name, module, args = contact_stage_command(Path("video.mp4"), tmp_path, Path("pose.pt"),
                                             Path("ball.pt"), tmp_path, "cpu", 2)
    assert name == "trained_contact"
    assert module == expected
    assert ("--single-output" in args) == (maximum == 60.)
    assert "--allow-window-transfer" not in args


def test_unbound_contact_protocol_is_rejected(tmp_path):
    (tmp_path / "deployment_metadata.json").write_text(json.dumps({"visual_settings": {"max_shot_sec": 100.}}))
    with pytest.raises(ValueError, match="Unsupported contact"):
        contact_stage_command(Path("video.mp4"), tmp_path, Path("pose.pt"), Path("ball.pt"), tmp_path, "cpu", 2)


def test_failed_or_absent_producers_cannot_become_completed_outputs(tmp_path):
    folder = tmp_path / "source"
    folder.mkdir()
    source = dict(clip_id="unseen_penalty", event_id=1, match_id="unassigned",
        kicker="unknown", team="unknown", clip_sha256="a" * 64, fps=30., source_start_frame=0)
    (folder / "source_manifest.json").write_text(json.dumps({"clips": [source]}))
    result = summarize(source, tmp_path, [{"stage": "pose", "exit_code": 1, "seconds": 0.1}])
    assert result["all_stages_executed_successfully"] is False
    assert result["all_teacher_outputs_established"] is False
    assert result["required_outputs"]["kicking_foot"] == "unknown"
    assert result["required_outputs"]["ball_final_position"]["event_final_position"] is None
    assert result["feature_table"]["external_outcome_reference"] is None


def test_censored_last_ball_observation_does_not_become_an_event_endpoint():
    endpoint = {"status": "camera_cut_censored", "observed_ball_identity_candidate": True,
        "last_observed_position": [100., 200.], "last_observed_frame": 45,
        "trajectory": [{"frame": 45, "x": 100., "y": 200., "detection_id": "45:2"}]}
    result = endpoint_output(endpoint, 30.)
    assert result["last_observed"]["seconds"] == 1.5
    assert result["last_observed"]["detection_id"] == "45:2"
    assert result["event_final_position"] is None
    assert result["event_end_observed"] is False
    endpoint["observed_ball_identity_candidate"] = False
    assert endpoint_output(endpoint, 30.)["last_observed"] is None


def summary_source(tmp_path):
    source = dict(clip_id="unseen_penalty", event_id=1, match_id="unassigned",
        kicker="unknown", team="unknown", clip_sha256="a" * 64, fps=30., source_start_frame=0)
    (tmp_path / "source").mkdir()
    (tmp_path / "source/source_manifest.json").write_text(json.dumps({"clips": [source]}))
    return source


def test_runup_points_exclude_contact_postkick_and_unclassified_ground_points(tmp_path):
    source = summary_source(tmp_path)
    points = [{"frame_id": frame, "phase": phase} for frame, phase in enumerate(
        ("pre_kick_candidate", "contact_interval_candidate", "post_kick_candidate", None))]
    (tmp_path / "pitch").mkdir()
    (tmp_path / "pitch/unseen_penalty.json").write_text(json.dumps({
        "clip_id": source["clip_id"], "source_sha256": source["clip_sha256"],
        "human_verified": False, "sparse_ground_contact_candidates": points}))
    result = summarize(source, tmp_path, [])
    trajectory = result["required_outputs"]["runup_top_down_trajectory"]
    assert trajectory["points"] == points[:1]
    assert trajectory["complete_trajectory_verified"] is False
    assert trajectory["coordinate_unit"] == "m_hypothesis"
    assert result["final_feature_table"]["runup_projected_candidate_points"] == 1
    assert result["components"]["pitch"]["sparse_ground_contact_candidates"] == points


def test_selected_scalar_outputs_match_auxiliary_overrides_without_rewriting_legacy(tmp_path):
    source = summary_source(tmp_path)
    predictions = {
        "trained_shoes": {"striking_foot": "right", "candidate_count": 4},
        "trained_keeper": {"prediction": {"dive_direction": "image_left",
            "direction_method": "learned_missing_rule_fallback", "height_model_candidate": "low"}},
        "scoreboard_transition": {"decision": {"rule_outcome": "goal_likely"}},
    }
    for name, prediction in predictions.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "prediction.json").write_text(json.dumps(
            {**prediction, "source_sha256": source["clip_sha256"]}))
    result = summarize(source, tmp_path, [])
    final = result["final_feature_table"]
    assert final["kicking_foot"] == "right"
    assert final["visible_runup_steps"] == 4
    assert final["keeper_dive_direction"] == "image_left"
    assert final["keeper_dive_height"] == "low"
    assert final["goal_outcome"] == "goal_likely"
    for key in ("kicking_foot", "visible_runup_steps", "keeper_dive_direction", "goal_outcome"):
        assert final[key] == result["required_outputs"][key]
    assert result["feature_table"]["kicking_foot_candidate"] == "unknown"
    assert result["feature_table"]["steps_phase_candidate"] is None
    assert result["feature_table"]["combined_rule_outcome"] == "unknown"
    assert result["feature_table_provenance"]["auxiliary_overrides_applied"] is False
    assert result["final_feature_table_provenance"]["source"] == "required_outputs_after_auxiliary_selection"

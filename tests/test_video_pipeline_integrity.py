import json

import pytest

from scripts import analyze_penalty_video as pipeline


@pytest.fixture
def summary_fixture(tmp_path, monkeypatch):
    source = {"clip_id": "unseen_penalty", "clip_sha256": "a" * 64}
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "source_manifest.json").write_text(json.dumps({"clips": [source]}))
    monkeypatch.setattr(pipeline, "bind_stage", lambda *args: None)
    monkeypatch.setattr(pipeline, "feature_row", lambda *args: {
        "kicking_foot_candidate": "unknown", "steps_phase_candidate": None,
        "keeper_pre_movement_candidate": "unknown", "keeper_direction_image_candidate": "unknown",
        "keeper_strict_height_candidate": "unknown", "combined_rule_outcome": "unknown",
        "ball_rule_outcome": "unknown"})
    monkeypatch.setattr(pipeline, "compact_stage", lambda *args: {})
    receipts = []
    for stage, pattern in pipeline.STAGE_PATHS.items():
        if stage == "scoreboard":
            continue
        target = tmp_path / stage / pattern.format(clip_id=source["clip_id"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"present": True}))
        receipts.append({"stage": stage, "exit_code": 0})
    contact = tmp_path / "trained_contact/prediction.json"
    contact.parent.mkdir()
    contact.write_text(json.dumps({"video_sha256": source["clip_sha256"]}))
    receipts.append({"stage": "trained_contact", "exit_code": 0})
    return tmp_path, source, receipts


def test_partial_receipts_cannot_claim_all_stages_executed(summary_fixture):
    directory, source, receipts = summary_fixture
    result = pipeline.summarize(source, directory, receipts[:1])
    assert result["all_stages_executed_successfully"] is False


@pytest.mark.parametrize("missing", ["camera/clips/unseen_penalty.json", "trained_contact/prediction.json"])
def test_missing_artifact_cannot_be_reported_as_a_successful_complete_run(summary_fixture, missing):
    directory, source, receipts = summary_fixture
    (directory / missing).unlink()
    result = pipeline.summarize(source, directory, receipts)
    assert result["all_stages_executed_successfully"] is False


def test_contact_prediction_from_another_video_is_rejected(summary_fixture):
    directory, source, receipts = summary_fixture
    (directory / "trained_contact/prediction.json").write_text(json.dumps({"video_sha256": "b" * 64}))
    with pytest.raises(ValueError, match="another source video"):
        pipeline.summarize(source, directory, receipts)


def test_keeper_rejection_overrides_an_unqualified_old_direction(summary_fixture, monkeypatch):
    directory, source, receipts = summary_fixture
    old_feature_row = pipeline.feature_row
    monkeypatch.setattr(pipeline, "feature_row", lambda *args: {
        **old_feature_row(*args), "keeper_direction_image_candidate": "image_left"})
    prediction = directory / "trained_keeper/prediction.json"
    prediction.parent.mkdir()
    prediction.write_text(json.dumps({"video_sha256": source["clip_sha256"],
        "prediction": {"dive_direction": "unknown", "direction_method": "unavailable"}}))
    receipts.append({"stage": "trained_keeper", "exit_code": 0})
    result = pipeline.summarize(source, directory, receipts)
    assert result["required_outputs"]["keeper_dive_direction"] == "unknown"
    assert result["feature_table"]["keeper_direction_image_candidate"] == "image_left"
    assert result["selected_output_sources"]["keeper_dive_direction"] == "unavailable"


def test_qualitative_height_remains_separate_from_strict_geometry(summary_fixture):
    directory, source, receipts = summary_fixture
    prediction = directory / "trained_keeper/prediction.json"
    prediction.parent.mkdir()
    prediction.write_text(json.dumps({"video_sha256": source["clip_sha256"],
        "prediction": {"dive_direction": "image_left", "direction_method": "existing_supported_rule",
                       "height_model_candidate": "low"}}))
    receipts.append({"stage": "trained_keeper", "exit_code": 0})
    height = pipeline.summarize(source, directory, receipts)["required_outputs"]["keeper_dive_height"]
    assert height["value"] == "low"
    assert height["strict_mask_pose_category"] == "unknown"
    assert height["high_supported"] is False
    assert height["physical_height_metres"] is None

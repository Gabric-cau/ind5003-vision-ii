import hashlib
import json
from pathlib import Path
from urllib.parse import unquote

import pytest

from scripts.build_shoe_reference_review import build_viewer


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(tmp_path):
    clips = tmp_path / "source clips"
    clips.mkdir()
    video = clips / "event one.mp4"
    video.write_bytes(b"original test video bytes")
    clip_hash = hashlib.sha256(video.read_bytes()).hexdigest()
    analysis = tmp_path / "analysis"
    source = {"clip_id": "event_1", "match_id": "col_eng", "event_id": 1,
              "kicker": "player", "team": "COL", "penalty_number": 1,
              "clip_filename": video.name, "clip_sha256": clip_hash,
              "fps": 30.0, "frame_count": 120, "source_start_frame": 300,
              "source_end_frame_exclusive": 420, "source_filename": "col_eng_pen.mp4",
              "source_video_sha256": "a" * 64}
    prediction = analysis / "events/event_1/landing_result.json"
    write_json(prediction, {"clip_id": "event_1", "source_sha256": clip_hash, "fps": 30.0,
                            "frame_count": 120, "source_mapping": source,
                            "candidate_count": 987654321, "striking_foot": "private_model_label",
                            "observable_window": {"start_frame": 7654321, "end_frame": 8765432},
                            "landings": [{"frame": 6543210, "foot": "private_model_label"}]})
    freeze = analysis / "prediction_freeze.json"
    write_json(freeze, {"schema_version": "shoe-landing-prediction-freeze-v1", "predictions": [
        {"clip_id": "event_1", "path": "events/event_1/landing_result.json",
         "sha256": hashlib.sha256(prediction.read_bytes()).hexdigest()}]})
    references = analysis / "human_reference_template.json"
    pending = {"reviewer_id": "", "reviewed_at": None, "annotator_type": None,
               "human_verified": False, "review_status": "pending", "striking_foot": "unknown",
               "verified_landings": None, "full_approach_visible": None}
    write_json(references, {"schema_version": "shoe-landing-reference-v1",
                           "prediction_freeze_sha256": hashlib.sha256(freeze.read_bytes()).hexdigest(),
                           "events": [{"clip_id": "event_1", "source_sha256": clip_hash, "fps": 30.0,
                                       "frame_count": 120, "frame_clock": "clip_local_zero_based",
                                       "reviews": [pending]}]})
    return freeze, clips, references, analysis / "blind_review", prediction, video


def test_blind_viewer_admits_only_identity_metadata_and_unmodified_video_urls(tmp_path):
    freeze, clips, references, output, _, video = fixture(tmp_path)
    receipt = build_viewer(freeze, clips, references, output, expected_events=1)
    html = (output / "index.html").read_text()
    for forbidden in ("987654321", "7654321", "8765432", "6543210", "private_model_label", "candidate_count", '"landings"'):
        assert forbidden not in html
        assert forbidden not in json.dumps(receipt)
    row = receipt["sources"][0]
    assert row["video_url"].startswith("../")
    assert "%20" in row["video_url"]
    assert (output / unquote(row["video_url"])).resolve() == video.resolve()
    assert (output / "human_reference_template.json").read_bytes() == references.read_bytes()
    assert receipt["human_reviewed_events"] == 0
    assert receipt["held_out_test"] is False
    assert receipt["media_copied_or_uploaded"] is False
    assert "requestVideoFrameCallback" in html
    assert "Original clip unavailable" in html


def test_blind_viewer_refuses_changed_prediction_bytes(tmp_path):
    freeze, clips, references, output, prediction, _ = fixture(tmp_path)
    prediction.write_text(prediction.read_text() + " ")
    with pytest.raises(ValueError, match="changed after freezing"):
        build_viewer(freeze, clips, references, output, expected_events=1)
    assert not output.exists()


def test_blind_viewer_refuses_changed_original_video(tmp_path):
    freeze, clips, references, output, _, video = fixture(tmp_path)
    video.write_bytes(b"replaced source video")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_viewer(freeze, clips, references, output, expected_events=1)


def test_blind_viewer_cannot_publish_completed_labels_as_pending(tmp_path):
    freeze, clips, references, output, _, _ = fixture(tmp_path)
    value = json.loads(references.read_text())
    value["events"][0]["reviews"][0]["verified_landings"] = [{"frame": 20, "foot": "left"}]
    write_json(references, value)
    with pytest.raises(ValueError, match="untouched pending"):
        build_viewer(freeze, clips, references, output, expected_events=1)


def test_blind_viewer_refuses_reference_clock_mismatch(tmp_path):
    freeze, clips, references, output, _, _ = fixture(tmp_path)
    value = json.loads(references.read_text())
    value["events"][0]["frame_clock"] = "original_recording_frames"
    write_json(references, value)
    with pytest.raises(ValueError, match="reference source identity differs"):
        build_viewer(freeze, clips, references, output, expected_events=1)


def test_default_viewer_requires_the_complete_39_event_cohort(tmp_path):
    freeze, clips, references, output, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="expected 39"):
        build_viewer(freeze, clips, references, output)


def test_existing_viewer_is_not_overwritten(tmp_path):
    freeze, clips, references, output, _, _ = fixture(tmp_path)
    output.mkdir()
    (output / "index.html").write_text("existing viewer")
    with pytest.raises(FileExistsError, match="new blind review"):
        build_viewer(freeze, clips, references, output, expected_events=1)
    assert (output / "index.html").read_text() == "existing viewer"

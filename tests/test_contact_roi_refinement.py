from copy import deepcopy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from src import contact_roi_refinement as roi


def observation(frame, box):
    return {"frame": frame, "bbox": list(box), "keypoints": [[25.0, 35.0]] * 17,
            "keypoint_confidences": [0.9] * 17, "confidence": 0.9}


class Capture:
    def __init__(self, *args):
        self.frame = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return {3: 200, 4: 100, 5: 30, 7: 6, 1: self.frame}.get(prop, self.frame / 30 * 1000)

    def set(self, prop, value):
        self.frame = value
        return True

    def read(self):
        self.frame += 1
        return True, np.zeros((100, 200, 3), dtype=np.uint8)

    def release(self):
        self.released = True


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"controlled-video-content")
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    record = {"video_sha256": digest, "fps": 30, "frame_ids": [0, 2, 4],
              "features": [[1.0]], "baselines": {"existing_rule": {"time_seconds": None}},
              "diagnostics": {"actor": {"actor_visible_samples": 2}}, "settings": {},
              "provenance": {"code_sha256": {}}, "runtime": {}, "window": {"end_frame_exclusive": 6}}
    evidence = {"video_sha256": digest, "tracks": [{"track_id": 0, "observations": [
        observation(0, [30, 20, 50, 60]), observation(4, [30, 20, 50, 60])]}],
        "transforms": {str(f): np.eye(3).tolist() for f in (0, 2, 4)}, "balls": [], "camera": {}}
    capture = Capture()
    monkeypatch.setattr(roi.cv2, "VideoCapture", lambda *args: capture)
    seen = []

    def recover(current, **kwargs):
        seen.append(deepcopy(current))
        count = sum(len(t["observations"]) for t in current["tracks"])
        return {"tracks": current["tracks"], "feature_tracks": current["tracks"],
                "feature_candidates": [], "candidates": [],
                "recovery_diagnostics": {"actor_measured_samples": count},
                "pose_roi_requests": [{"frame": 2, "bbox": [10, 10, 60, 80], "track_id": 0,
                                       "measured_anchor_frames": [0, 4]}] if len(seen) == 1 else []}

    monkeypatch.setattr(roi, "recover_contact_evidence", recover)
    monkeypatch.setattr(roi, "geometric_features", lambda *args: (
        [[9.0]], {"actor_visible_samples": 3}, {"time_seconds": None}))
    monkeypatch.setattr(roi, "rule_baseline", lambda *args: {"time_seconds": 2 / 30})
    monkeypatch.setattr(roi, "_extract_result_detections", lambda result, frame: (
        [observation(frame, [20, 10, 40, 50])], []))
    model = SimpleNamespace(predict=lambda images, **kwargs: [object() for _ in images])
    return SimpleNamespace(video=video, record=record, evidence=evidence, model=model, seen=seen, capture=capture)


def test_crop_coordinates_are_restored_and_only_new_neural_observation_is_added(fixture):
    original_record, original_evidence = deepcopy(fixture.record), deepcopy(fixture.evidence)
    updated, augmented = roi.refine_contact_record(fixture.record, fixture.evidence, fixture.video, fixture.model)
    added = fixture.seen[1]["tracks"][-1]["observations"][0]
    assert added["frame"] == 2
    assert added["bbox"] == [30.0, 20.0, 50.0, 60.0]
    assert added["keypoints"][0] == [35.0, 45.0]
    assert added["observation_source"] == "new_neural_pose_on_automatic_gap_roi"
    assert updated["features"] == [[9.0]]
    assert updated["diagnostics"]["roi_refinement"]["interpolated_pose_used"] is False
    assert fixture.record == original_record
    assert fixture.evidence == original_evidence
    assert fixture.capture.released


def test_empty_detection_never_becomes_interpolated_pose(fixture, monkeypatch):
    monkeypatch.setattr(roi, "_extract_result_detections", lambda *args: ([], []))
    updated, augmented = roi.refine_contact_record(fixture.record, fixture.evidence, fixture.video, fixture.model)
    assert updated["features"] == fixture.record["features"]
    assert augmented["tracks"] == fixture.evidence["tracks"]
    assert updated["diagnostics"]["roi_refinement"]["accepted_neural_observations"] == 0
    assert updated["diagnostics"]["roi_refinement"]["applied"] is False


def test_existing_contact_features_are_not_overwritten_by_gap_observation(fixture):
    fixture.record["baselines"]["existing_rule"]["time_seconds"] = 0.1
    updated, _ = roi.refine_contact_record(fixture.record, fixture.evidence, fixture.video, fixture.model)
    assert updated["features"] == fixture.record["features"]
    assert updated["baselines"] == fixture.record["baselines"]
    assert updated["diagnostics"]["roi_refinement"]["accepted_neural_observations"] == 1
    assert updated["diagnostics"]["roi_refinement"]["applied"] is False


@pytest.mark.parametrize("mismatch", ["video", "evidence"])
def test_source_hash_mismatch_rejected_before_inference(fixture, mismatch):
    if mismatch == "video":
        fixture.video.write_bytes(b"different-video")
    else:
        fixture.evidence["video_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        roi.refine_contact_record(fixture.record, fixture.evidence, fixture.video, fixture.model)
    assert fixture.seen == []


def test_nonlinear_camera_motion_uses_stabilized_identity_anchor(fixture):
    fixture.evidence["tracks"][0]["observations"][0]["bbox"] = [20, 20, 40, 60]
    fixture.evidence["tracks"][0]["observations"][1]["bbox"] = [120, 20, 140, 60]
    fixture.evidence["transforms"]["2"][0][2] = -10
    fixture.evidence["transforms"]["4"][0][2] = -100
    updated, _ = roi.refine_contact_record(fixture.record, fixture.evidence, fixture.video, fixture.model)
    assert updated["diagnostics"]["roi_refinement"]["accepted_neural_observations"] == 1
    assert updated["diagnostics"]["roi_refinement"]["applied"] is True


def test_pose_batch_cannot_silently_drop_a_requested_frame(fixture):
    fixture.model.predict = lambda images, **kwargs: []
    with pytest.raises(ValueError, match="[Bb]atch|count|every requested ROI"):
        roi.refine_contact_record(fixture.record, fixture.evidence, fixture.video, fixture.model)


def test_missing_joint_sentinel_is_not_translated_into_a_real_point(fixture, monkeypatch):
    def detect(result, frame):
        row = observation(frame, [20, 10, 40, 50])
        row["keypoints"][15] = [0.0, 0.0]
        row["keypoint_confidences"][15] = 0.3
        return [row], []
    monkeypatch.setattr(roi, "_extract_result_detections", detect)
    updated, _ = roi.refine_contact_record(fixture.record, fixture.evidence, fixture.video, fixture.model)
    added = fixture.seen[1]["tracks"][-1]["observations"][0]
    assert added["keypoints"][15] == [0.0, 0.0]
    assert added["keypoint_confidences"][15] == 0.0
    assert added["keypoints"][16] == [35.0, 45.0]

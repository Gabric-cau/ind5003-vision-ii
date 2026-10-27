from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src import shoe_camera_segments as segments


@pytest.fixture
def sample():
    camera = {"source_sha256": "a" * 64,
        "segment_ids": {str(i): int(i >= 2) for i in range(10)},
        "transforms": {str(i): np.eye(3).tolist() for i in range(10)},
        "camera_diagnostics": {"failed_pairs": [[1, 2]]}}
    inputs = {"source_mapping": {"clip_sha256": "a" * 64}, "fps": 10.,
        "inference_interval_frames": [0, 9], "interval_frames": [0, 5],
        "contact_bracket_frames": [5, 7],
        "original_camera_motion": {"camera_diagnostics": {"failed_pairs": [[1, 2]]}}}
    prediction = {"striking_foot": "unknown", "right_score_uncalibrated": None,
                  "landings": [{"frame": 1, "foot": "right", "score": .9}], "candidate_count": 1}
    return inputs, camera, prediction


def test_existing_foot_decision_never_invokes_camera_or_feature_work(sample, monkeypatch):
    inputs, camera, prediction = sample
    prediction.update(striking_foot="left", right_score_uncalibrated=.1)
    monkeypatch.setattr(segments, "measure_segment_camera", lambda *a: pytest.fail("unnecessary remeasurement"))
    monkeypatch.setattr(segments, "extract_evidence", lambda *a: pytest.fail("unnecessary feature replay"))
    result, receipt, measured = segments.recover_missing_foot(inputs, [], {}, prediction)
    assert result == prediction
    assert receipt["reason"] == "existing_foot_decision"
    assert measured is None


def test_unknown_without_camera_failure_is_preserved(sample):
    inputs, camera, prediction = sample
    inputs["original_camera_motion"] = {"camera_diagnostics": {"failed_pairs": []}}
    result, receipt, measured = segments.recover_missing_foot(inputs, [], {}, prediction)
    assert result == prediction
    assert receipt["reason"] == "no_recorded_camera_failure"


def test_contact_segment_excludes_earlier_coordinates_and_keeps_landings(sample, monkeypatch):
    inputs, camera, prediction = sample
    original = deepcopy(inputs)
    captured = {}
    def extract(scoped, records):
        captured.update(scoped)
        return {"foot_vector": np.ones(3), "supported_side_frames": {"left": 6, "right": 6}}
    monkeypatch.setattr(segments, "extract_evidence", extract)
    monkeypatch.setattr(segments, "predict_event", lambda *a: {
        "striking_foot": "right", "right_score_uncalibrated": .8,
        "landings": [{"frame": 3}], "candidate_count": 99})
    result, receipt, measured = segments.recover_missing_foot(inputs, [], {}, prediction, camera=camera)
    assert captured["inference_interval_frames"] == [2, 9]
    assert set(captured["original_camera_motion"]["transforms"]) == {str(i) for i in range(2, 10)}
    assert result["striking_foot"] == "right"
    assert result["landings"] == prediction["landings"]
    assert result["candidate_count"] == 1
    assert receipt["used"] is True
    assert inputs == original


def test_contact_bracket_crossing_failed_link_stays_unknown(sample, monkeypatch):
    inputs, camera, prediction = sample
    inputs["contact_bracket_frames"] = [1, 3]
    monkeypatch.setattr(segments, "extract_evidence", lambda *a: pytest.fail("Cannot bridge camera failure"))
    result, receipt, _ = segments.recover_missing_foot(inputs, [], {}, prediction, camera=camera)
    assert result == prediction
    assert receipt["reason"] == "contact_crosses_camera_failure"


def test_failed_camera_recovery_cannot_invent_missing_paired_pose(sample, monkeypatch):
    inputs, camera, prediction = sample
    monkeypatch.setattr(segments, "extract_evidence", lambda *a: {
        "supported_side_frames": {"left": 0, "right": 0}})
    monkeypatch.setattr(segments, "predict_event", lambda *a: prediction)
    result, receipt, _ = segments.recover_missing_foot(inputs, [], {}, prediction, camera=camera)
    assert result == prediction
    assert receipt["reason"] == "segment_paired_pose_still_unavailable"


def test_wrong_source_camera_is_rejected(sample):
    inputs, camera, prediction = sample
    camera["source_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="does not bind"):
        segments.recover_missing_foot(inputs, [], {}, prediction, camera=camera)


def test_noncontiguous_segment_is_rejected(sample):
    inputs, camera, prediction = sample
    camera["segment_ids"]["3"] = 9
    with pytest.raises(ValueError, match="not contiguous"):
        segments.recover_missing_foot(inputs, [], {}, prediction, camera=camera)


def test_stream_restarts_camera_reference_instead_of_propagating_failure(tmp_path, monkeypatch):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"source")
    digest = segments.file_sha256(video)
    source = {"clip_sha256": digest, "frame_count": 5}
    images = [np.full((3, 4, 3), i, np.uint8) for i in range(5)]
    inputs = {"source_mapping": source, "fps": 10., "scene_interval_frames": [0, 5],
        "decoded_frame_hashes": {str(i): hashlib.sha256(image.tobytes()).hexdigest()
                                 for i, image in enumerate(images)}}
    pose = {"source_mapping": source, "inference_identity": {"source_sha256": digest},
            "frame_height": 3, "frame_width": 4}
    class Capture:
        frame = 0
        def isOpened(self):
            return True
        def get(self, key):
            return 10. if key == segments.cv2.CAP_PROP_FPS else 5
        def read(self):
            image = images[self.frame]
            self.frame += 1
            return True, image
        def release(self):
            pass
    monkeypatch.setattr(segments.cv2, "VideoCapture", lambda *a: Capture())
    translation = np.eye(3)
    translation[0, 2] = 10
    def pair(images, frame_ids, pose):
        return [np.eye(3), translation], {"failed_pairs": [frame_ids] if frame_ids == [1, 2] else [],
                                             "median_inlier_ratio": 1.}
    monkeypatch.setattr(segments, "_camera_transforms", pair)
    result = segments.measure_segment_camera(video, inputs, pose)
    assert result["camera_diagnostics"]["failed_pairs"] == [[1, 2]]
    assert [result["transforms"][str(i)][0][2] for i in range(5)] == [0, 10, 0, 10, 20]
    assert list(result["segment_ids"].values()) == [0, 0, 1, 1, 1]
    assert result["cross_segment_motion_used"] is False
    inputs["decoded_frame_hashes"]["3"] = "wrong"
    with pytest.raises(ValueError, match="decoded pixels differ"):
        segments.measure_segment_camera(video, inputs, pose)


def test_video_adapter_exposes_recovery_and_preserves_landing_output(tmp_path, monkeypatch):
    from scripts import predict_video_shoes as adapter
    analysis = tmp_path / "analysis"
    (analysis / "source").mkdir(parents=True)
    (analysis / "pose").mkdir()
    source = {"clip_id": "unseen", "clip_filename": "unseen.mp4", "clip_sha256": "a" * 64, "fps": 10.}
    (analysis / "source/source_manifest.json").write_text(json.dumps({
        "clips": [source], "manual_annotations_used": False}))
    (analysis / "pose/unseen.json").write_text("{}")
    pose_model = tmp_path / "pose.task"
    pose_model.write_bytes(b"pose")
    bound = {key: None for key in ("source_mapping", "scene_interval_frames", "contact_bracket_frames",
        "actor_track_id", "original_pose_selected_track_id", "original_camera_motion", "original_phase_result", "contact")}
    bound.update(video="source.mp4", plan=[], fps=10., interval_frames=[0, 3], inference_interval_frames=[0, 9])
    monkeypatch.setattr(adapter, "load_bound_model", lambda *a: ({}, {"model_sha256": "b" * 64}))
    monkeypatch.setattr(adapter, "bind_event", lambda *a: (bound, {}))
    monkeypatch.setattr(adapter, "infer_event", lambda *a: ([], {}, {}))
    monkeypatch.setattr(adapter, "extract_evidence", lambda *a: {"supported_side_frames": {"left": 1, "right": 1}})
    monkeypatch.setattr(adapter, "predict_event", lambda *a: {
        "striking_foot": "unknown", "right_score_uncalibrated": None, "landings": None, "candidate_count": None})
    def recovered(inputs, records, model, prediction, **kwargs):
        return {**prediction, "striking_foot": "right", "right_score_uncalibrated": .8}, {
            "used": True, "landing_output_preserved": True}, {"transforms": {}}
    monkeypatch.setattr(adapter, "recover_missing_foot", recovered)
    output = tmp_path / "result"
    result = adapter.run(SimpleNamespace(analysis_dir=analysis, output=output,
        model=tmp_path / "model", pose33_model=pose_model, threads=1, recover_camera_segments=True))
    assert result["status"] == "automatic_candidates"
    assert result["striking_foot"] == "right"
    assert result["candidate_count"] is None
    assert result["foot_camera_recovery"]["used"] is True
    assert result["foot_camera_recovery"]["camera_sha256"] == adapter.sha(output / "camera_segments.json")

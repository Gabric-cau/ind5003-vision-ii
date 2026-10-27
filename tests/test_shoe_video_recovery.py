from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src import shoe_video_recovery as recovery


@pytest.fixture
def evidence():
    source = {"clip_id": "unseen", "clip_filename": "unseen.mp4", "clip_sha256": "a" * 64,
              "fps": 10., "frame_count": 20}
    observation = lambda frame: {"frame": frame, "bbox": [10. + frame / 10, 10., 40. + frame / 10, 90.]}
    tracks = [{"track_id": 2, "observations": [observation(frame) for frame in range(18) if frame != 5]}]
    pose = {"source_mapping": source, "inference_identity": {"source_sha256": source["clip_sha256"]},
        "reference_labels_used_for_prediction": False, "new_neural_inference": True,
        "source_frame_count": 20, "fps": 10., "shot_start_frame": 2, "shot_end_frame": 18,
        "frame_width": 100, "frame_height": 100, "selected_track_id": 2, "tracks": tracks}
    window = {"start_frame": 0, "end_frame_exclusive": 20, "automatic": True}
    diagnostics = {"actor": {"actor_id": 8}}
    contact = {"video_sha256": source["clip_sha256"], "prediction": {"frame": 10, "time_seconds": 1.},
        "manual_annotations_required": False, "automatic_window": window, "diagnostics": diagnostics}
    measured = {"video_sha256": source["clip_sha256"], "tracks": [
        {"track_id": 8, "observations": [observation(frame) for frame in range(0, 20, 2)]}],
        "transforms": {str(frame): np.eye(3).tolist() for frame in range(0, 20, 2)}}
    features = {"video_sha256": source["clip_sha256"], "fps": 10., "window": window,
        "diagnostics": diagnostics, "frame_ids": list(range(0, 20, 2)),
        "provenance": {"human_labels_used": False, "old_contact_used": False,
                       "manual_camera_boundaries_used": False}}
    return source, Path("source.mp4"), pose, contact, measured, features


def test_recovery_reuses_actual_boxes_without_filling_gaps_or_reencoding(evidence):
    bound, reason = recovery.measured_bound(*evidence)
    assert reason is None
    assert bound["scene_interval_frames"] == [2, 18]
    assert bound["inference_interval_frames"] == [2, 17]
    assert bound["interval_frames"] == [2, 9]
    assert bound["contact_bracket_frames"] == [9, 11]
    assert bound["dense_actor_association"]["selected_old_track_id"] == 2
    plan = {row["frame"]: row for row in bound["plan"]}
    assert plan[5]["anchor_bbox"] is None
    assert plan[4]["anchor_bbox"] == evidence[4]["tracks"][0]["observations"][2]["bbox"]
    assert bound["clock_mapping"]["exported_intermediate_video"] is False
    assert bound["clock_mapping"]["mapping_kind"] == "original_video_no_reencoding"
    assert "all_pixels_verified_equal" not in bound["clock_mapping"]


@pytest.mark.parametrize("which", [3, 4, 5])
def test_recovery_rejects_evidence_from_different_video(evidence, which):
    changed = deepcopy(evidence)
    changed[which]["video_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="another source"):
        recovery.measured_bound(*changed)


def test_recovery_rejects_manual_or_interpolated_evidence(evidence):
    changed = deepcopy(evidence)
    changed[5]["provenance"]["human_labels_used"] = True
    with pytest.raises(ValueError, match="manual labels"):
        recovery.measured_bound(*changed)
    changed = deepcopy(evidence)
    changed[4]["tracks"][0]["observations"][0]["interpolated"] = True
    with pytest.raises(ValueError, match="Interpolated poses"):
        recovery.measured_bound(*changed)


def test_contact_outside_shared_shot_does_not_get_clamped(evidence):
    evidence[3]["prediction"].update(frame=18, time_seconds=1.8)
    bound, reason = recovery.measured_bound(*evidence)
    assert bound is None
    assert reason == "contact_outside_shared_automatic_shot"


def test_missing_actor_does_not_create_fake_pose(evidence):
    evidence[4]["tracks"] = []
    bound, reason = recovery.measured_bound(*evidence)
    assert bound is None
    assert reason == "contact_actor_missing_from_measured_tracks"


def test_contact_pose_context_recovers_measured_actor_without_qualifying_goal(evidence):
    source, video, original, contact, measured, features = evidence
    original = {**original, "selected_track_id": None, "selected_track_qualified": False,
                "shot_start_frame": None, "shot_end_frame": None, "tracks": []}
    context, reason = recovery.contact_pose_context(source, original, features, measured)
    assert reason is None
    assert context["selected_track_qualified"] is False
    assert context["tracks"] == measured["tracks"]
    bound, reason = recovery.measured_bound(source, video, context, contact, measured, features)
    assert reason is None
    assert bound["actor_track_id"] == 8
    assert bound["original_pose_selected_track_id"] is None
    assert bound["dense_actor_association"]["selected_old_track_id"] is None
    assert bound["clock_mapping"]["pose_evidence_source"] == "trained_contact_measured_tracks"
    assert bound["scene_interval_frames"] == [0, 20]
    assert bound["plan"][5]["anchor_bbox"] is None


def test_contact_pose_context_rejects_mismatched_original_source(evidence):
    source, _, original, _, measured, features = evidence
    original["source_frame_count"] += 1
    with pytest.raises(ValueError, match="source provenance"):
        recovery.contact_pose_context(source, original, features, measured)


def test_contact_pose_context_rejects_interpolated_measurement(evidence):
    source, video, original, contact, measured, features = evidence
    measured["tracks"][0]["observations"][0]["interpolated"] = True
    context, _ = recovery.contact_pose_context(source, original, features, measured)
    with pytest.raises(ValueError, match="Interpolated poses"):
        recovery.measured_bound(source, video, context, contact, measured, features)


def test_actual_source_hash_checked_before_camera_or_neural_work(tmp_path, evidence, monkeypatch):
    source, _, pose, contact, measured, features = evidence
    artifacts = {"source/source_manifest.json": {"manual_annotations_used": False, "clips": [source]},
                 "pose/unseen.json": pose, "trained_contact/prediction.json": contact,
                 "trained_contact/evidence.json": measured, "trained_contact/features.json": features}
    for name, value in artifacts.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    (tmp_path / "source/unseen.mp4").write_bytes(b"mismatched source bytes")
    monkeypatch.setattr(recovery, "measure_dense_camera", lambda *args: pytest.fail("No camera work after bad source"))
    with pytest.raises(ValueError, match="source video or manifest hash"):
        recovery.bind_recovered_event(tmp_path, source, recovery.file_sha256(tmp_path / "source/source_manifest.json"))


@pytest.mark.parametrize("pose_status", ["selected_person_not_qualified", "selected_person_is_goal_line_occupant"])
def test_adapter_uses_contact_pose_only_for_missing_qualification(tmp_path, evidence, monkeypatch, pose_status):
    source, _, pose, contact, measured, features = deepcopy(evidence)
    video = tmp_path / "source/unseen.mp4"
    video.parent.mkdir()
    video.write_bytes(b"bound source bytes")
    source["clip_sha256"] = recovery.file_sha256(video)
    pose["inference_identity"]["source_sha256"] = source["clip_sha256"]
    pose.update(selected_track_id=None, selected_track_qualified=False, shot_start_frame=None,
                shot_end_frame=None, tracks=[])
    for data in (contact, measured, features):
        data["video_sha256"] = source["clip_sha256"]
    artifacts = {"source/source_manifest.json": {"manual_annotations_used": False, "clips": [source]},
                 "pose/unseen.json": pose, "trained_contact/prediction.json": contact,
                 "trained_contact/evidence.json": measured, "trained_contact/features.json": features}
    for name, value in artifacts.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    monkeypatch.setattr(recovery, "_pose_input", lambda *a: (None, {"status": pose_status}))
    def camera(bound, context):
        assert context["pose_evidence_source"] == "trained_contact_measured_tracks"
        return {"camera_diagnostics": {"failed_pairs": []}, "transforms": {}}
    monkeypatch.setattr(recovery, "measure_dense_camera", camera)
    bound, receipt = recovery.bind_recovered_event(tmp_path, source,
        recovery.file_sha256(tmp_path / "source/source_manifest.json"))
    if pose_status == "selected_person_not_qualified":
        assert bound["actor_track_id"] == 8
        assert receipt["used"] is True
        assert receipt["selected_pose_evidence"] == "trained_contact_measured_tracks"
        assert receipt["goal_geometry_required_for_shoe_sampling"] is False
    else:
        assert bound is None
        assert receipt["unavailable_reason"] == pose_status


def test_camera_failure_invalidates_following_frames_without_interpolation(monkeypatch):
    class Capture:
        def isOpened(self):
            return True
        def get(self, key):
            return 10. if key == recovery.cv2.CAP_PROP_FPS else 4
        def read(self):
            return True, np.zeros((2, 2, 3), dtype=np.uint8)
        def release(self):
            pass
    monkeypatch.setattr(recovery.cv2, "VideoCapture", lambda *args: Capture())
    monkeypatch.setattr(recovery, "_camera_transforms", lambda *args: (
        [np.eye(3) for _ in range(4)], {"failed_pairs": [[1, 2]], "reliable": False}))
    result = recovery.measure_dense_camera({"scene_interval_frames": [0, 4], "video": "source.mp4",
        "fps": 10., "width": 2, "height": 2, "source_mapping": {"frame_count": 4}}, {})
    assert result["transforms"]["1"] == np.eye(3).tolist()
    assert result["transforms"]["2"] is result["transforms"]["3"] is None
    assert result["interpolated_transforms_used"] is False


def test_successful_original_binding_never_calls_fallback(tmp_path, monkeypatch):
    from scripts import predict_video_shoes as adapter
    source = tmp_path / "analysis/source"
    source.mkdir(parents=True)
    (source / "source_manifest.json").write_text(json.dumps({"manual_annotations_used": False,
        "clips": [{"clip_id": "unseen", "clip_filename": "unseen.mp4", "clip_sha256": "a" * 64, "fps": 30.}]}))
    checkpoint = tmp_path / "pose.task"
    checkpoint.write_bytes(b"pose model")
    monkeypatch.setattr(adapter, "load_bound_model", lambda *args: ({}, {"model_sha256": "b" * 64}))
    monkeypatch.setattr(adapter, "bind_event", lambda *args: ({"plan": []}, {"original": True}))
    monkeypatch.setattr(adapter, "bind_recovered_event", lambda *args: pytest.fail("Original successful binding must remain unchanged"))
    class ReachedOriginalInference(Exception):
        pass
    def infer(*args):
        raise ReachedOriginalInference
    monkeypatch.setattr(adapter, "infer_event", infer)
    with pytest.raises(ReachedOriginalInference):
        adapter.run(SimpleNamespace(analysis_dir=tmp_path / "analysis", output=tmp_path / "prediction",
            model=tmp_path / "model", pose33_model=checkpoint, threads=2))


def test_shoe_adapter_imports_without_detector_training_dependencies():
    code = """
import importlib.abc
import sys
class DenyDetectorImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'ultralytics'}:
            raise AssertionError('Unexpected shoe dependency: ' + fullname)
sys.meta_path.insert(0, DenyDetectorImports())
import scripts.predict_video_shoes
import src.shoe_video_recovery
assert not {'torch', 'torchvision', 'ultralytics'} & set(sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

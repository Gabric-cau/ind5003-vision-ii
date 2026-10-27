import hashlib
import json
from pathlib import Path

import numpy as np

from scripts import predict_video_shoes as target


def cache_fixture(tmp_path, monkeypatch):
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    directory = tmp_path / "cache"
    directory.mkdir()
    model = tmp_path / "pose.task"
    model.write_bytes(b"model")
    bound = {key: None for key in target.BOUND_INPUT_KEYS}
    bound.update(plan=[{"frame": 0}], actor_track_id=1, original_camera_motion={"transforms": {"0": "matrix"}})
    poses = directory / "poses33.json"
    poses.write_text("[]")
    inputs = {key: bound[key] for key in target.BOUND_INPUT_KEYS}
    inputs.update(poses33_sha256=target.sha(poses), model_sha256=target.sha(model),
        source_manifest_sha256="manifest", reference_labels_read=False, decoded_frame_hashes={"0": digest})
    (directory / "inputs.json").write_text(json.dumps(inputs))
    (directory / "roi_plan.json").write_text(json.dumps(bound["plan"]))
    prediction = {"source_code_sha256": {"scripts/run_fresh_shoe_approach_experiment.py":
        target.sha(target.ROOT / "scripts/run_fresh_shoe_approach_experiment.py")}, "actual_inference_calls": 2}
    prediction.update({key: 1 for key in ("decoded_interval_frames", "distinct_decoded_pixel_frames",
        "repeated_decoded_pixel_frames", "accepted_roi_pose_calls", "unique_pixel_accepted_roi_pose_calls",
        "roi_identity_status_counts", "original_actor_missing_frames")})
    (directory / "prediction.json").write_text(json.dumps(prediction))
    monkeypatch.setattr(target, "decode_interval", lambda value: [(0, image, 0)])
    return directory, bound, model, image


def test_exact_binding_reuses_without_counting_cached_calls_as_fresh(tmp_path, monkeypatch):
    directory, bound, model, _ = cache_fixture(tmp_path, monkeypatch)
    evidence, receipt = target.bound_pose_cache(directory, bound, model, "manifest")
    assert receipt["used"]
    assert evidence[2]["actual_inference_calls"] == 0
    assert evidence[2]["cached_original_inference_calls"] == 2


def test_changed_actor_or_pixels_requires_fresh_inference(tmp_path, monkeypatch):
    directory, bound, model, image = cache_fixture(tmp_path, monkeypatch)
    bound["actor_track_id"] = 2
    assert target.bound_pose_cache(directory, bound, model, "manifest")[0] is None
    bound["actor_track_id"] = 1
    image[0, 0, 0] = 1
    evidence, receipt = target.bound_pose_cache(directory, bound, model, "manifest")
    assert evidence is None and receipt["reason"] == "decoded_pixels_differ"

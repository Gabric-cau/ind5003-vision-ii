from types import SimpleNamespace

import numpy as np
import pytest

from src import active_keeper_detector as active


class Tensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


def detection(boxes=(), scores=(), classes=None):
    return SimpleNamespace(boxes=SimpleNamespace(xyxy=Tensor(list(boxes)), conf=Tensor(list(scores)),
                           cls=Tensor([0] * len(boxes) if classes is None else classes)))


def model(result):
    class Model:
        names = {0: "active_keeper"}
        task = "detect"
        predictor = SimpleNamespace(device=SimpleNamespace(type="cpu"))

        def predict(self, images, **kwargs):
            assert len(images) == 1
            assert kwargs["classes"] == [0] and kwargs["imgsz"] == 1280
            return [result]

    return Model()


@pytest.mark.parametrize("names,task", [({0: "person"}, "detect"), ({0: "ball"}, "detect"),
                                       ({0: "active_keeper", 1: "person"}, "detect"),
                                       ({0: "active_keeper"}, "pose")])
def test_wrong_model_rejected(names, task):
    with pytest.raises(ValueError, match="single-class"):
        active.validate_model(SimpleNamespace(names=names, task=task))


def test_select_top_model_confidence_without_reference():
    candidates, selected = active.select_detection(detection([[1, 2, 8, 9], [10, 2, 18, 9]], [.6, .9]), 20, 10)
    assert selected == candidates[0]
    assert selected["bbox_xyxy"] == [10, 2, 18, 9]
    assert selected["candidate_id"] == 1
    assert active.select_detection(detection(), 20, 10) == ([], None)
    assert active.select_detection(detection([[1, 2, 8, 9]], [.09]), 20, 10) == ([], None)


def test_invalid_box_and_confidence_are_not_silently_changed():
    with pytest.raises(ValueError, match="outside"):
        active.select_detection(detection([[1, 2, 22, 9]], [.5]), 20, 10)
    with pytest.raises(ValueError, match="Invalid"):
        active.select_detection(detection([[1, 2, 8, 9]], [float("nan")]), 20, 10)


def test_frame_sampling_includes_zero_and_does_not_shift_at_end():
    assert [i for i in range(12) if active.sampled(i, 5)] == [0, 5, 10]
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            active.sampled(0, invalid)


def test_roi_and_pose_mapping_keep_unknown_null():
    seed = [20, 30, 60, 70]
    roi = active.roi_window(seed, 100, 100)
    assert roi == [10, 20, 70, 80]
    points = [[15., 15.]] * 17
    points[2] = [-20., 15.]
    scores = [.9] * 17
    scores[1] = .01
    result = SimpleNamespace(boxes=SimpleNamespace(xyxy=Tensor([[10, 10, 50, 50]]), conf=Tensor([.9])),
                             keypoints=SimpleNamespace(xy=Tensor([points]), conf=Tensor([scores])))
    mapped = active.map_pose(result, roi, seed, 100, 100)
    assert mapped["keypoints_xy"][0] == [25., 35.]
    assert mapped["keypoints_xy"][1:3] == [None, None]
    assert mapped["keypoint_observed_mask"][:3] == [True, False, False]
    assert mapped["pose_box_xyxy"] == seed
    assert mapped["unknown_points_synthesized"] is False


def test_no_detection_never_calls_pose():
    result = active.predict_frame(np.zeros((10, 20, 3), dtype=np.uint8), model(detection()), "cpu", object())
    assert result["selected_candidate"] is None and result["pose"] is None
    assert result["status"] == "no_detection"


def test_local_paths_and_nonoverwriting_output(tmp_path):
    video, weights = tmp_path / "input.mp4", tmp_path / "weights.pt"
    video.write_bytes(b"video")
    weights.write_bytes(b"weights")
    resolved = active.validate_paths(video, weights, tmp_path / "out")
    assert resolved[:2] == (video, weights)
    with pytest.raises(FileNotFoundError):
        active.validate_paths(video, tmp_path / "remote.pt", tmp_path / "out")
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        active.validate_paths(video, weights, tmp_path / "out")


def test_sampled_video_keeps_unsampled_frames_in_overlay(tmp_path):
    import cv2

    video, weights, output = tmp_path / "input.mp4", tmp_path / "weights.pt", tmp_path / "out"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10., (64, 48))
    if not writer.isOpened():
        pytest.skip("Local OpenCV has no MP4 writer")
    for i in range(12):
        writer.write(np.full((48, 64, 3), 30 + i * 5, dtype=np.uint8))
    writer.release()
    weights.write_bytes(b"fake weights for injected test model")
    report = active.run_video(video, output, model(detection()), weights=weights, stride=5)
    assert [r["frame_zero_based"] for r in report["frames"]] == [0, 5, 10]
    assert report["sampled_frame_count"] == 3 and report["frames_with_detection"] == 0
    assert report["overlay"]["decoded_frames"] == report["source_frame_count"] == 12
    assert report["manual_annotations_used"] is False and report["accuracy_measured"] is False
    assert (output / "prediction.json").is_file()

"""Label-free active-keeper detection and optional pose evidence on a video."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

from src.runtime import file_sha256

CONFIDENCE = .10
NMS_IOU = .70
IMAGE_SIZE = 1280
KEYPOINT_CONFIDENCE = .15


def validate_model(model, *, pose=False):
    names = model.names
    if not isinstance(names, (dict, list, tuple)):
        raise ValueError("Model class names are unavailable")
    try:
        names = {int(k): v for k, v in names.items()} if isinstance(names, dict) else dict(enumerate(names))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid model class names") from exc
    if pose:
        shape = getattr(getattr(model, "model", None), "yaml", {}).get("kpt_shape")
        if model.task != "pose" or names != {0: "person"} or list(shape or []) != [17, 3]:
            raise ValueError("Optional pose weights must be a COCO17 person pose model")
    elif model.task != "detect" or names != {0: "active_keeper"}:
        raise ValueError("Weights must be a single-class active_keeper detector")


def validate_paths(video, weights, output, pose_weights=None):
    video, weights, output = (Path(p).expanduser().resolve() for p in (video, weights, output))
    pose_weights = Path(pose_weights).expanduser().resolve() if pose_weights is not None else None
    if not video.is_file() or video.stat().st_size == 0:
        raise FileNotFoundError("A nonempty local source video is required")
    for path in [weights] + ([pose_weights] if pose_weights is not None else []):
        if not path.is_file() or path.suffix.lower() != ".pt" or path.stat().st_size == 0:
            raise FileNotFoundError("Weights must be a nonempty local .pt file; downloads are not requested")
        if path == video:
            raise ValueError("Video and model paths must differ")
    if output.exists():
        raise FileExistsError("Choose a new output directory")
    return video, weights, output, pose_weights


def sampled(frame_number, stride):
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("Frame stride must be a positive integer")
    if isinstance(frame_number, bool) or not isinstance(frame_number, int) or frame_number < 0:
        raise ValueError("Frame numbers must be nonnegative integers")
    return frame_number % stride == 0


def _box(box, width, height):
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        raise ValueError("Finite xyxy coordinates are required")
    x1, y1, x2, y2 = map(float, box)
    if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
        raise ValueError("Prediction box is outside the source image")
    return [x1, y1, x2, y2]


def select_detection(result, width, height):
    boxes = result.boxes.xyxy.cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()
    classes = result.boxes.cls.cpu().tolist()
    if not len(boxes) == len(confidences) == len(classes):
        raise ValueError("Unaligned detector outputs")
    candidates = []
    for index, (box, confidence, class_id) in enumerate(zip(boxes, confidences, classes)):
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1 or class_id != 0:
            raise ValueError("Invalid active-keeper prediction")
        bbox = _box(box, width, height)
        if confidence >= CONFIDENCE:
            candidates.append({"candidate_id": index, "bbox_xyxy": bbox, "confidence": confidence})
    candidates.sort(key=lambda row: (-row["confidence"], row["candidate_id"]))
    return candidates, candidates[0] if candidates else None


def roi_window(box, width, height):
    x1, y1, x2, y2 = _box(box, width, height)
    dx, dy = .25 * (x2 - x1), .25 * (y2 - y1)
    return [max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy)),
            min(width, math.ceil(x2 + dx)), min(height, math.ceil(y2 + dy))]


def map_pose(result, roi, seed_box, width, height):
    from scripts.optimize_manual_pose import box_iou, persons_from_result

    people = persons_from_result(result, roi[:2])
    eligible = [p for p in people if box_iou(p["bbox"], seed_box) >= .3]
    chosen = max(eligible, key=lambda row: row["confidence"]) if eligible else None
    if chosen is None:
        return None
    mapped, visible = [], []
    for point, confidence in zip(chosen["keypoints"], chosen["keypoint_confidences"]):
        x, y = map(float, point)
        confidence = float(confidence)
        usable = (math.isfinite(confidence) and confidence >= KEYPOINT_CONFIDENCE and
                  math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height)
        mapped.append([x, y] if usable else None)
        visible.append(usable)
    return {"roi_xyxy": list(roi), "pose_box_xyxy": chosen["bbox"],
            "pose_confidence": chosen["confidence"], "keypoints_xy": mapped,
            "keypoint_confidences": [float(c) if math.isfinite(float(c)) else None
                                      for c in chosen["keypoint_confidences"]],
            "keypoint_observed_mask": visible, "keypoint_order": "COCO17_anatomical_left_right",
            "coordinate_system": "original_image_pixels_x_right_y_down",
            "unknown_points_synthesized": False}


def _verify_device(model, device):
    actual = getattr(getattr(model, "predictor", None), "device", None)
    expected = "mps" if device == "mps" else "cpu" if device == "cpu" else "cuda"
    if getattr(actual, "type", None) != expected:
        raise RuntimeError("Inference did not use the requested device")


def predict_frame(image, detector, device, pose_model=None):
    height, width = image.shape[:2]
    outputs = detector.predict([image], imgsz=IMAGE_SIZE, conf=CONFIDENCE, iou=NMS_IOU,
                               max_det=20, classes=[0], device=device, verbose=False)
    if len(outputs) != 1:
        raise RuntimeError("Detector output count differs from input")
    _verify_device(detector, device)
    candidates, selected = select_detection(outputs[0], width, height)
    pose = None
    if selected is not None and pose_model is not None:
        window = roi_window(selected["bbox_xyxy"], width, height)
        x1, y1, x2, y2 = window
        outputs = pose_model.predict([image[y1:y2, x1:x2]], imgsz=640, conf=.10,
                                     iou=.70, max_det=10, device=device, verbose=False)
        if len(outputs) != 1:
            raise RuntimeError("Pose output count differs from input")
        _verify_device(pose_model, device)
        pose = map_pose(outputs[0], window, selected["bbox_xyxy"], width, height)
    return {"candidates": candidates, "selected_candidate": selected, "pose": pose,
            "status": "candidate" if selected else "no_detection", "identity_verified": False}


def draw_overlay(image, record, frame_number):
    import cv2

    canvas = image.copy()
    label = f"frame {frame_number}: not sampled"
    if record is not None:
        label = f"frame {frame_number}: {record['status']}"
        selected = record["selected_candidate"]
        if selected is not None:
            x1, y1, x2, y2 = map(round, selected["bbox_xyxy"])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (30, 210, 240), 2)
            label += f" {selected['confidence']:.3f}"
        if record["pose"] is not None:
            for point in record["pose"]["keypoints_xy"]:
                if point is not None:
                    cv2.circle(canvas, tuple(map(round, point)), 3, (70, 230, 80), -1)
    cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .6,
                (245, 245, 245), 2, cv2.LINE_AA)
    return canvas


def run_video(video, output, detector, *, weights, device="cpu", stride=5,
              pose_model=None, pose_weights=None):
    import cv2

    video, weights, output, pose_weights = validate_paths(video, weights, output, pose_weights)
    sampled(0, stride)
    validate_model(detector)
    if (pose_model is None) != (pose_weights is None):
        raise ValueError("Pose model and pose weights must be supplied together")
    if pose_model is not None:
        validate_model(pose_model, pose=True)
    source_hash, weights_hash = file_sha256(video), file_sha256(weights)
    pose_hash = file_sha256(pose_weights) if pose_weights is not None else None
    started = time.perf_counter()
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise ValueError("Cannot decode source video")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width, height = (int(capture.get(k)) for k in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0 or min(width, height, expected_frames) < 1:
        capture.release()
        raise ValueError("Invalid source video metadata")
    output.mkdir(parents=True)
    overlay = output / "overlay.mp4"
    writer = cv2.VideoWriter(str(overlay), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise RuntimeError("Cannot create MP4 overlay")
    rows, frame_number, previous_pts = [], 0, None
    inference_seconds = 0.
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if image.shape[:2] != (height, width):
                raise ValueError("Video frame dimensions changed")
            raw_pts = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.
            valid_pts = math.isfinite(raw_pts) and raw_pts >= 0 and (previous_pts is None or raw_pts > previous_pts)
            pts = raw_pts if valid_pts else None
            previous_pts = raw_pts if math.isfinite(raw_pts) else previous_pts
            record = None
            if sampled(frame_number, stride):
                begin = time.perf_counter()
                record = predict_frame(image, detector, device, pose_model)
                elapsed = time.perf_counter() - begin
                inference_seconds += elapsed
                record.update(frame_zero_based=frame_number, decoded_pts_seconds=pts,
                              nominal_time_seconds=frame_number / fps, inference_seconds=elapsed)
                rows.append(record)
            writer.write(draw_overlay(image, record, frame_number))
            frame_number += 1
    finally:
        capture.release()
        writer.release()
    if frame_number != expected_frames:
        raise RuntimeError(f"Source decode incomplete: {frame_number}/{expected_frames}")
    check = cv2.VideoCapture(str(overlay))
    overlay_frames = 0
    try:
        while check.isOpened():
            ok, _ = check.read()
            if not ok:
                break
            overlay_frames += 1
    finally:
        check.release()
    if overlay_frames != frame_number:
        raise RuntimeError("Overlay does not contain every decoded input frame")
    if file_sha256(video) != source_hash or file_sha256(weights) != weights_hash:
        raise RuntimeError("Source video or detector weights changed during inference")
    if pose_weights is not None and file_sha256(pose_weights) != pose_hash:
        raise RuntimeError("Pose weights changed during inference")
    report = {"schema_version": "active-keeper-video-candidates-v1", "source_video": str(video),
              "source_sha256": source_hash, "weights": str(weights), "weights_sha256": weights_hash,
              "pose_weights_sha256": pose_hash, "device": device, "width": width, "height": height,
              "fps": fps, "source_frame_count": frame_number, "frame_stride": stride,
              "sampled_frame_count": len(rows), "frames_with_detection": sum(r["selected_candidate"] is not None for r in rows),
              "frames_with_pose": sum(r["pose"] is not None for r in rows),
              "settings": {"imgsz": IMAGE_SIZE, "confidence": CONFIDENCE, "nms_iou": NMS_IOU,
                           "selection": "highest_model_confidence_per_sampled_frame", "roi_margin": .25,
                           "pose_imgsz": 640, "keypoint_confidence": KEYPOINT_CONFIDENCE},
              "frames": rows, "manual_annotations_used": False, "manual_roi_used": False,
              "reference_labels_used_for_prediction": False, "identity_verified": False,
              "accuracy_measured": False, "all_teacher_outputs_established": False,
              "scope": "independent_active_keeper_detection_and_optional_pose_candidates",
              "overlay": {"file": overlay.name, "sha256": file_sha256(overlay),
                          "decoded_frames": overlay_frames, "preserves_all_source_frames": True,
                          "unsampled_frames_use_previous_boxes": False},
              "inference_seconds": inference_seconds, "end_to_end_seconds": time.perf_counter() - started,
              "limitations": ["No temporal identity tracking or complete keeper event classification.",
                              "Detection availability is not correctness accuracy.",
                              "Unknown pose coordinates remain null; no missing joints are invented.",
                              "Stride sampling can miss brief contact or movement transitions."]}
    (output / "prediction.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report

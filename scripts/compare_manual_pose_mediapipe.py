"""Static MediaPipe heavy inference on frozen model-only person crops."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

try:
    from scripts.optimize_manual_pose import digest, evaluate, load_images, roi_window, save
except ModuleNotFoundError:
    from optimize_manual_pose import digest, evaluate, load_images, roi_window, save

COCO_TO_MEDIAPIPE = {0: 0, 1: 2, 2: 5, 3: 7, 4: 8, 5: 11, 6: 12, 7: 13, 8: 14,
                     9: 15, 10: 16, 11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28}


def run(args):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    samples, manifest = load_images(args.manifest)
    frozen = json.loads(args.person_predictions.read_text())
    if frozen.get("reference_coordinates_used_for_prediction") is not False or frozen.get("mode") != "full":
        raise ValueError("Person proposals must be frozen reference-free full-frame predictions")
    proposal_lookup = {row["frame_id"]: row for row in frozen["rows"]}
    if set(proposal_lookup) != {sample["frame_id"] for sample in samples}:
        raise ValueError("Frozen person proposal inventory mismatch")
    protocol = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "mediapipe": mp.__version__,
                "manifest_sha256": digest(args.manifest), "model_sha256": digest(args.model),
                "person_proposals_sha256": digest(args.person_predictions), "script_sha256": digest(__file__),
                "running_mode": "IMAGE", "time_context": "static frames; no inferred temporal continuity",
                "roi_policy": "each frozen YOLO person bbox expanded20percent; no reference joint in prediction",
                "detection_confidence": .5, "pose_presence_confidence": .5,
                "coco_to_mediapipe": COCO_TO_MEDIAPIPE,
                "primary_metric": "136 human visible points; within10px; missing pose or joint counts failure",
                "validation_video": "19-08", "independent_test": False}
    save(args.output / "protocol.json", protocol)
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.model), delegate=python.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.IMAGE, num_poses=1,
        min_pose_detection_confidence=.5, min_pose_presence_confidence=.5)
    rows = []
    started = time.perf_counter()
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for sample in samples:
            image = cv2.imread(str(sample["image"]))
            if image is None:
                raise ValueError("Cannot read source image")
            height, width = image.shape[:2]
            proposals = proposal_lookup[sample["frame_id"]]["people"]
            people, raw = [], []
            for proposal in proposals:
                x1, y1, x2, y2 = roi_window(proposal["bbox"], width, height)
                crop = image[y1:y2, x1:x2]
                if min(crop.shape[:2]) < 16:
                    continue
                rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                detected = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                record = {"candidate_id": proposal["candidate_id"], "crop_xyxy": [x1, y1, x2, y2],
                          "pose_count": len(detected.pose_landmarks), "landmarks33": []}
                if detected.pose_landmarks:
                    landmarks = detected.pose_landmarks[0]
                    points = [[p.x * (x2 - x1) + x1, p.y * (y2 - y1) + y1] for p in landmarks]
                    confidence = [min(float(p.visibility), float(p.presence)) for p in landmarks]
                    record["landmarks33"] = [{"xy": p, "confidence": c} for p, c in zip(points, confidence)]
                    people.append({"candidate_id": proposal["candidate_id"], "bbox": proposal["bbox"],
                                   "confidence": proposal["confidence"],
                                   "keypoints": [points[COCO_TO_MEDIAPIPE[j]] for j in range(17)],
                                   "keypoint_confidences": [confidence[COCO_TO_MEDIAPIPE[j]] for j in range(17)],
                                   "source": "mediapipe_heavy_static_auto_person_roi"})
                raw.append(record)
            rows.append({"frame_id": sample["frame_id"], "video_id": sample["video_id"],
                         "frame_zero_based": sample["frame_zero_based"], "width": width, "height": height,
                         "people": people, "automatic_rois": raw})
    predictions = {"mode": "mediapipe_heavy_static_auto_person_roi", "rows": rows,
                   "reference_coordinates_used_for_prediction": False,
                   "seconds": time.perf_counter() - started}
    save(args.output / "predictions.json", predictions)
    target_path = args.manifest.parent / "frame_targets.json"
    if digest(target_path) != manifest["artifact_sha256"]["frame_targets.json"]:
        raise ValueError("Manual target hash mismatch")
    targets = json.loads(target_path.read_text())["frames"]
    score = evaluate(predictions, targets)
    save(args.output / "evaluation.json", score)
    print(json.dumps({"all": score["all"], "validation": score["validation_video"],
                      "by_video": score["by_video"], "seconds": predictions["seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--person-predictions", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())

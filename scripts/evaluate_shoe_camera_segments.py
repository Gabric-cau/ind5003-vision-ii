"""Replay a fixed shoe head with camera-segment recovery; no labels are read."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import cv2
import joblib
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.runtime import file_sha256
from src.shoe_camera_segments import recover_missing_foot
from src.shoe_supervised_model import extract_evidence, predict_event


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", action="append", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output; preserve prior evidence")
    if args.threads < 1:
        raise ValueError("threads must be positive")
    metadata = read(args.model / "deployment_metadata.json")
    model_path = args.model / "deployment.joblib"
    if (file_sha256(model_path) != metadata["model_sha256"]
            or file_sha256(ROOT / "src/shoe_supervised_model.py") != metadata["evidence_contract"]["feature_code_sha256"]
            or version("scikit-learn") != metadata["dependencies"]["scikit-learn"]):
        raise ValueError("Frozen model, features or runtime changed")
    model = joblib.load(model_path)
    cv2.setNumThreads(args.threads)
    started = time.perf_counter()
    rows = []
    paths = sorted({path for root in args.evidence_root for path in root.rglob("inputs.json")
                    if (path.parent / "poses33.json").is_file()})
    for index, path in enumerate(paths):
        inputs = read(path)
        poses_path = path.parent / "poses33.json"
        if inputs["poses33_sha256"] != file_sha256(poses_path):
            raise ValueError("Cached shoe poses changed")
        if inputs["model_sha256"] not in metadata["evidence_contract"]["pose_model_sha256"]:
            raise ValueError("Pose model differs from the frozen shoe contract")
        if inputs.get("reference_labels_read") is not False:
            raise ValueError("Pose cache must be independent of reference labels")
        records = read(poses_path)
        analysis = path.parent.parent
        source = inputs["source_mapping"]
        video = analysis / "source" / source["clip_filename"]
        pose_path = analysis / "pose" / (source["clip_id"] + ".json")
        pose = read(pose_path) if pose_path.is_file() and video.is_file() else None
        with threadpool_limits(limits=args.threads):
            evidence = extract_evidence(inputs, records)
            before = predict_event(model, evidence, inputs["fps"])
            after, receipt, camera = recover_missing_foot(inputs, records, model, before,
                video=video if pose else None, pose=pose)
        label = next((parent.name for parent in path.parents if parent.name.startswith("wc_")),
                     source["clip_id"])
        row = {"source_id": label, "inputs_path": str(path), "inputs_sha256": file_sha256(path),
               "poses_sha256": file_sha256(poses_path), "video_sha256": source["clip_sha256"],
               "before": before, "after": after, "camera_recovery": receipt,
               "existing_foot_changed": before["striking_foot"] in ("left", "right")
                   and before["striking_foot"] != after["striking_foot"],
               "landings_changed": before["landings"] != after["landings"]}
        if camera:
            camera_path = args.output / "camera" / f"{index:03d}_{label}.json"
            write(camera_path, camera)
            row["camera_path"] = str(camera_path)
            row["camera_sha256"] = file_sha256(camera_path)
        rows.append(row)
        print(json.dumps({"source_id": label, "before": before["striking_foot"],
                          "after": after["striking_foot"], "recovery": receipt["reason"]}), flush=True)
    report = {"schema_version": "shoe-contact-camera-segment-replay-v1", "events": rows,
        "model_sha256": metadata["model_sha256"], "event_count": len(rows),
        "before_available": sum(row["before"]["striking_foot"] in ("left", "right") for row in rows),
        "after_available": sum(row["after"]["striking_foot"] in ("left", "right") for row in rows),
        "existing_foot_changed": sum(row["existing_foot_changed"] for row in rows),
        "landings_changed": sum(row["landings_changed"] for row in rows),
        "human_labels_read": False, "new_neural_inference": False, "weights_updated": False,
        "evaluation_scope": "source-only feature recovery and frozen-weight regression, not independent accuracy",
        "seconds": time.perf_counter() - started,
        "code_sha256": {name: file_sha256(ROOT / name) for name in (
            "src/shoe_camera_segments.py", "src/shoe_supervised_model.py", "src/kick_contact.py")}}
    write(args.output / "report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key not in ("events", "code_sha256")}))


if __name__ == "__main__":
    main()

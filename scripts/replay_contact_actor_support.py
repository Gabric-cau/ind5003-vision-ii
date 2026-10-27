"""Prepare and predict the support-gated actor protocol on bound visual caches."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from scripts.predict_video_contact_fullshot import read_json, write_json
from src.contact_actor_support import recover_actor_support
from src.contact_causal_actor import recover_causal_actor, predict_causal_contact
from src.contact_candidate_window import bind_candidate_window, predict_candidate_contact
from src.contact_temporal_model import predict_contact, validate_prediction_bindings
from src.runtime import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "predict"])
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--per-event", action="store_true")
    parser.add_argument("--recovery", choices=["support", "causal", "bounded"], default="support")
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--source-pipelines", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior actor recovery experiments")
    metadata = read_json(args.model / "deployment_metadata.json")
    weights = args.model / "deployment_weights.npz"
    if file_sha256(weights) != metadata["weights_sha256"]:
        raise ValueError("Contact model weights hash mismatch")
    for path, checksum in metadata["extractor_code_sha256"].items():
        if file_sha256(ROOT / path) != checksum:
            raise ValueError("Contact feature code hash mismatch")
    samples = read_json(args.inputs)["samples"]
    for sample in samples:
        if any("reference" in k or "contact_seconds" in k for k in sample):
            raise ValueError("Reference information cannot enter actor recovery")
        event = sample["core_id"]
        feature_path = args.features / event / "features.json" if args.per_event else args.features / (event + ".json")
        evidence_path = args.features / event / "evidence.json" if args.per_event else args.features / "evidence" / (event + ".json")
        record, evidence = read_json(feature_path), read_json(evidence_path)
        validate_prediction_bindings(record, metadata)
        if record["video_sha256"] != sample["video_sha256"]:
            raise ValueError("Manifest and features refer to different source videos")
        if args.command == "prepare":
            video = args.video_root / (event + ".mp4") if args.video_root else Path(sample["video_path"])
            if args.source_pipelines:
                video = args.source_pipelines / event / "attempt_001/pipeline/source/unseen_penalty.mp4"
            if file_sha256(video) != record["video_sha256"]:
                raise ValueError("Actor recovery source hash mismatch")
            cap = cv2.VideoCapture(str(video))
            try:
                height, fps = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
                if height <= 0 or fps != record["fps"]:
                    raise ValueError("Source video dimensions or FPS invalid")
            finally:
                cap.release()
            recover = {"support": recover_actor_support, "causal": recover_causal_actor, "bounded": bind_candidate_window}[args.recovery]
            record, evidence = recover(record, evidence, height)
            write_json(args.output / (event + ".json"), record)
            write_json(args.output / "evidence" / (event + ".json"), evidence)
            receipt_key = {"support": "actor_support_recovery", "causal": "causal_actor_recovery", "bounded": "candidate_window_recovery"}[args.recovery]
            print(event, record["diagnostics"][receipt_key], flush=True)
        else:
            started = time.perf_counter()
            with np.load(weights, allow_pickle=False) as parameters:
                predict = (predict_candidate_contact if record["settings"].get("recovered_candidate_window_version") else
                           predict_causal_contact if record["settings"].get("causal_actor_recovery_version") else predict_contact)
                prediction = predict(record, parameters)
            output = {"video_sha256": record["video_sha256"], "prediction": prediction,
                "selected_method": "temporal_classifier", "temporal_prediction": prediction, "calibrated_prediction": None,
                "manual_annotations_required": False, "automatic_window": record["window"], "baselines": record["baselines"],
                "diagnostics": record["diagnostics"], "feature_runtime": record["runtime"],
                "head_prediction_seconds": time.perf_counter() - started,
                "contact_model_metadata_sha256": file_sha256(args.model / "deployment_metadata.json"),
                "model_bindings": {"mode": "matched_training_protocol"},
                "scope": "first automatic shot up to 60s with support-gated actor recovery", "promoted_to_default_pipeline": False}
            for name, data in (("prediction.json", output), ("features.json", record), ("evidence.json", evidence)):
                write_json(args.output / event / name, data)
            print(event, prediction, flush=True)


if __name__ == "__main__":
    main()

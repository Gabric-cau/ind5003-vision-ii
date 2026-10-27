"""Replay the optional keeper fallback on saved automatic analyses, without labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.predict_video_keeper import predict_analysis
from src.runtime import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new keeper recovery output directory")
    import cv2
    cv2.setNumThreads(2)
    inputs = sorted((args.batch / "events").glob("*/attempt_001/pipeline"))
    if not inputs:
        raise ValueError("No completed pipeline analyses found")
    args.output.mkdir(parents=True)
    protocol = {
        "schema_version": "keeper-contact-recovery-replay-v1",
        "model_sha256": file_sha256(args.model),
        "code_sha256": {name: file_sha256(ROOT / name) for name in (
            "scripts/predict_video_keeper.py", "src/keeper_contact_recovery.py",
            "src/keeper_supervised_model.py", "src/goalkeeper_pipeline.py",
            "src/goalkeeper_observation_guard.py", "src/goalkeeper_role_association.py")},
        "source_analyses": [str(path.resolve()) for path in inputs],
        "reference_labels_read_during_inference": False,
        "neural_training_performed": False, "new_neural_inference": False,
        "scope": "previously examined development videos; not a new held-out test",
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    rows = []
    for directory in inputs:
        event = directory.parents[1].name
        old = predict_analysis(directory, args.model)
        new = predict_analysis(directory, args.model, recover_from_contact=True)
        (args.output / f"{event}.json").write_text(json.dumps(new, indent=2, allow_nan=False) + "\n")
        recovery = new["keeper_contact_recovery"]
        evidence = recovery.get("keeper") or {}
        row = {
            "event": event, "video_sha256": new["video_sha256"],
            "before": old["prediction"], "after": new["prediction"],
            "recovery_applied": recovery["applied"],
            "recovery_reason": evidence.get("reason"),
            "contact_frame": evidence.get("kick_frame"),
            "role_reason": (evidence.get("role") or {}).get("reason"),
            "head_reasons": (recovery.get("learned_prediction") or {}).get("reasons"),
            "warnings": evidence.get("qa_warnings"),
            "runtime_seconds": new["head_runtime_seconds_including_hash_checks"],
            "prediction_sha256": file_sha256(args.output / f"{event}.json"),
        }
        rows.append(row)
        (args.output / "summary.json").write_text(json.dumps({
            "protocol": protocol, "events": len(rows), "rows": rows,
            "reference_labels_read_during_inference": False,
        }, indent=2, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()

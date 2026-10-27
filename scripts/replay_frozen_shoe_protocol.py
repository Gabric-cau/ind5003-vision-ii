"""Replay default shoe inference with new automatic contact and verified cache reuse."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.predict_video_shoes import run
from scripts.run_fresh_shoe_approach_experiment import read, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "contact-root", "model", "pose33-model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh replay output")
    rows = read(args.inputs)
    if not isinstance(rows, list) or len({row["event_id"] for row in rows}) != len(rows):
        raise ValueError("Input manifest must contain unique events")
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    results = []
    for row in rows:
        event = row["event_id"]
        if Path(event).name != event or event in (".", ".."):
            raise ValueError("Event ID must be a basename")
        original = Path(row["analysis_dir"]).resolve()
        contact = (args.contact_root / event).resolve()
        analysis = args.output / event / "analysis"
        analysis.mkdir(parents=True)
        for name in ("source", "pose", "ball", "phase"):
            target = original / name
            if not target.is_dir():
                raise FileNotFoundError(target)
            (analysis / name).symlink_to(target, target_is_directory=True)
        if not all((contact / name).is_file() for name in ("prediction.json", "features.json", "evidence.json")):
            raise FileNotFoundError(contact)
        (analysis / "trained_contact").symlink_to(contact, target_is_directory=True)
        prediction = run(SimpleNamespace(analysis_dir=analysis, output=args.output / event / "shoes",
            model=args.model, pose33_model=args.pose33_model, threads=args.threads,
            recover_camera_segments=False, pose_cache=Path(row["pose_cache"]) if row.get("pose_cache") else None))
        result = {"event_id": event, "source_sha256": prediction["source_sha256"],
            "upstream_analysis_dir": str(original), "contact_directory": str(contact),
            "contact_sha256": {name: sha(contact / name) for name in ("prediction.json", "features.json", "evidence.json")},
            "prediction_sha256": sha(args.output / event / "shoes/prediction.json"),
            "striking_foot": prediction["striking_foot"], "binding_method": prediction["binding_method"],
            "actual_inference_calls": prediction["actual_inference_calls"],
            "pose_cache": prediction.get("pose_cache"), "status": prediction["status"],
            "seconds": prediction["runtime_seconds_including_model_load_and_pose_inference"]}
        results.append(result)
        write(args.output / "progress.json", results)
        print(result, flush=True)
    report = {"schema_version": "default-shoe-source-only-protocol-replay-v1", "events": results,
        "inputs_sha256": sha(args.inputs), "human_labels_read": False, "weights_updated": False,
        "camera_segment_recovery_enabled": False, "local_crop_ablation_enabled": False,
        "seconds": time.perf_counter() - started, "evaluation_scope": "development replay, not a new blind test"}
    write(args.output / "report.json", report)


if __name__ == "__main__":
    main()

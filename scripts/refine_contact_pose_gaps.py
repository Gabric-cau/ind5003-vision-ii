"""Recover label-free pose evidence using source-video ROI inference."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.contact_roi_refinement import refine_contact_record
from src.runtime import file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--video-root", type=Path, required=True)
    p.add_argument("--pose-model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("New output directory required")
    from ultralytics import YOLO
    import torch
    import cv2
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    model = YOLO(str(args.pose_model))
    for path in sorted(args.features.glob("core_*.json")):
        record = json.loads(path.read_text())
        if record["provenance"]["model_sha256"].get(args.pose_model.name) != file_sha256(args.pose_model):
            raise ValueError("Pose model differs from cached evidence")
        evidence = json.loads((args.features / "evidence" / path.name).read_text())
        output, evidence = refine_contact_record(record, evidence,
            args.video_root / (path.stem + ".mp4"), model, device=args.device)
        for folder, data in ((args.output, output), (args.output / "evidence", evidence)):
            folder.mkdir(parents=True, exist_ok=True)
            (folder / path.name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        d = output["diagnostics"]["roi_refinement"]
        print(json.dumps({"core": path.stem, "crops": d["requested_crops"],
            "accepted": d["accepted_neural_observations"], "applied": d["applied"],
            "actor": output["diagnostics"]["actor"], "baseline": output["baselines"]["existing_rule"]}), flush=True)


if __name__ == "__main__":
    main()

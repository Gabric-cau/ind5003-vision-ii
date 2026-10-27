"""Evaluate already selected ball checkpoints without training or test selection."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.optimize_ball_detector import load_exports, evaluate_checkpoint
from scripts.train_ball_frame_detector import digest, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--development-retest", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Frozen evaluations must use a new directory")
    candidate = json.loads(args.candidate.read_text())
    if candidate.get("selection_uses_test") is not False:
        raise ValueError("Candidate lacks development-only selection receipt")
    if candidate.get("test_results_seen_before_training") and not args.development_retest:
        raise ValueError("A follow-up after viewing earlier results requires explicit development-retest scope")
    weights = Path(candidate["weights"])
    if digest(weights) != candidate["weights_sha256"]:
        raise ValueError("Selected weights have changed")
    items = load_exports(args.manifest)
    if {row["split"] for row in items} != {"test"}:
        raise ValueError("Only evaluation-only samples belong in this comparison")
    if {row["video_id"] for row in items} != {"core_05", "core_06", "core_09"}:
        raise ValueError("Unexpected evaluation videos")
    models = [("pretrained", args.pretrained, "full", 1280),
              ("previous_pilot", args.pilot, "full", 1280),
              ("development_selected", weights, candidate["mode"], candidate["image_size"])]
    args.output.mkdir(parents=True)
    protocol = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "manifest_sha256": digest(args.manifest), "candidate_receipt_sha256": digest(args.candidate),
                "script_sha256": digest(__file__), "candidate_selection_uses_test": False,
                "development_retest_after_previous_evaluation": args.development_retest,
                "models": [{"name": name, "weights_sha256": digest(path), "mode": mode, "image_size": size}
                           for name, path, mode, size in models],
                "scope": "new ball detector module; these videos were exposed to older contact/height components",
                "primary_metric": "top-ranked box IoU >= 0.5 over duplicate-adjusted annotated positive frames",
                "unknown_predictions_count_as_misses": True,
                "limitations": ["No exhaustive background negative annotations.",
                                "References are human localized plus automatic bright-spot refinement.",
                                "Boxes after a ball re-enters do not verify shot identity or final resting position.",
                                "This is not whole-project accuracy or an untouched final system test."]}
    save_json(args.output / "protocol.json", protocol)
    results = {}
    for name, path, mode, size in models:
        results[name] = evaluate_checkpoint(path, items, args.output / f"{name}.json", mode, size, args.device, 2)
        print(json.dumps({"name": name, "metrics": results[name]["metrics"]}), flush=True)
    save_json(args.output / "report.json", {"protocol": protocol, "results": results,
              "retrained_on_evaluation": False, "default_pipeline_changed": False})


if __name__ == "__main__":
    main()

"""Score frozen keeper recovery predictions against event-level references."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.keeper_supervised_model import classification_metrics
from src.runtime import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Evaluation output already exists")
    summary = json.loads((args.predictions / "summary.json").read_text())
    rows = summary["rows"]
    for row in rows:
        if file_sha256(args.predictions / (row["event"] + ".json")) != row["prediction_sha256"]:
            raise ValueError("Frozen keeper prediction hash mismatch")
    reference = json.loads(args.references.read_text())
    indexed = {row["core_id"]: row for row in reference["samples"]}
    if set(indexed) != {row["event"] for row in rows}:
        raise ValueError("Keeper reference/prediction events differ")
    truth = []
    for row in rows:
        source = indexed[row["event"]]
        if source["video_sha256"] != row["video_sha256"]:
            raise ValueError("Keeper reference/prediction source mismatch")
        truth.append(source["keeper_ordinal_height"])
    metrics = {name: classification_metrics(truth, [row[name]["height_model_candidate"] for row in rows],
                                           ("low", "middle", "high"))
               for name in ("before", "after")}
    output = {
        "schema_version": "keeper-recovery-development-evaluation-v1",
        "predictions_sha256": file_sha256(args.predictions / "summary.json"),
        "reference_sha256": file_sha256(args.references), "metrics": {"keeper_ordinal_height": metrics},
        "direction_accuracy_computed": False,
        "direction_reason": "Reference direction coordinate system was not established",
        "events": len(rows), "unknown_counts_as_incorrect": True,
        "independent_final_test": False,
        "scope": "previously examined World Cup development regression; no new unseen-match claim",
        "rows": [{"event": row["event"], "reference_height": expected,
                  "before": row["before"]["height_model_candidate"],
                  "after": row["after"]["height_model_candidate"]} for row, expected in zip(rows, truth)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output["metrics"], indent=2))


if __name__ == "__main__":
    main()

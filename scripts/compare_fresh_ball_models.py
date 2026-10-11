"""Compare two frozen detectors on identical course sources and temporal rules."""
import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.final_results import sha256
from src.fresh_results import read_json, write_json


def compare_event(first, second):
    a, b = first["inference_identity"], second["inference_identity"]
    for field in ("source_sha256", "pose_sha256", "source_manifest_sha256", "settings", "code_sha256"):
        if a.get(field) != b.get(field):
            raise ValueError(f"Uncontrolled detector comparison: {field}")
    if a.get("model_sha256") == b.get("model_sha256"):
        raise ValueError("Detector comparison requires distinct model bytes")
    if any(row.get("reference_labels_used_for_prediction") is not False for row in (first, second)):
        raise ValueError("Reference-dependent predictions cannot be compared")
    values = []
    for data in (first, second):
        candidate = (data.get("independent_contact_search") or {}).get("selected_candidate")
        values.append({"contact": candidate is not None,
                       "contact_frame": candidate.get("contact_frame") if candidate else None,
                       "actor": candidate.get("contact_associated_track_id") if candidate else None,
                       "bracket": candidate.get("contact_bracket_frames") if candidate else None,
                       "outcome": (data.get("rule_decision") or {}).get("rule_outcome", "unknown"),
                       "observations": len(data.get("trajectory", [])),
                       "elapsed_sec": data.get("elapsed_sec")})
    first_value, second_value = values
    return {"clip_id": first["clip_id"], "n": first_value, "m": second_value,
            "contact_pair": f'{int(first_value["contact"])}{int(second_value["contact"])}',
            "actor_difference": first_value["actor"] != second_value["actor"] if all(v["contact"] for v in values) else None,
            "contact_frame_difference": second_value["contact_frame"] - first_value["contact_frame"] if all(v["contact"] for v in values) else None,
            "outcome_difference": first_value["outcome"] != second_value["outcome"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--n-dir", type=Path, required=True)
    parser.add_argument("--m-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Preserve the frozen comparison in a new output directory")
    source = read_json(args.source_manifest)
    if len(source["clips"]) != 39 or len({r["clip_id"] for r in source["clips"]}) != 39:
        raise ValueError("Exactly 39 distinct canonical events are required")
    rows, bindings = [], []
    for record in source["clips"]:
        cid = record["clip_id"]
        paths = [directory / "results" / f"{cid}.json" for directory in (args.n_dir, args.m_dir)]
        predictions = [read_json(path) for path in paths]
        if any(p["clip_id"] != cid or p["inference_identity"]["source_sha256"] != record["clip_sha256"]
               or p["inference_identity"]["source_manifest_sha256"] != sha256(args.source_manifest) for p in predictions):
            raise ValueError("Comparison result is not bound to the canonical source")
        rows.append(compare_event(*predictions))
        bindings.append({"clip_id": cid, "results_sha256": [sha256(path) for path in paths]})
    summary = {"events": len(rows), "contact_pairs_n_m": dict(Counter(r["contact_pair"] for r in rows)),
               "contact_candidates": {label: sum(r[label]["contact"] for r in rows) for label in ("n", "m")},
               "outcomes": {label: dict(Counter(r[label]["outcome"] for r in rows)) for label in ("n", "m")},
               "new_neural_inference": False, "reference_labels_used": False, "human_verified": False,
               "interpretation": "Controlled development diagnostics; availability is not extraction accuracy"}
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "comparison.json", {"summary": summary, "rows": rows,
        "source_manifest_sha256": sha256(args.source_manifest), "result_bindings": bindings,
        "script_sha256": sha256(Path(__file__))})
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    counts = summary["contact_candidates"]
    bars = axes[0].bar(["YOLO11n", "YOLO11m"], list(counts.values()), color=["#287e78", "#c15b48"])
    axes[0].bar_label(bars, fmt="%d / 39"); axes[0].set_ylim(0, 44)
    axes[0].set_ylabel("Contact candidates"); axes[0].set_title("Same sources and temporal rules")
    paired = [r for r in rows if r["contact_pair"] == "11"]
    axes[1].scatter([r["n"]["observations"] for r in paired], [r["m"]["observations"] for r in paired], color="#287e78")
    limit = max([1] + [max(r[label]["observations"] for label in ("n", "m")) for r in paired]) + 1
    axes[1].plot([0, limit], [0, limit], ":", color="#777777")
    axes[1].set_xlim(0, limit); axes[1].set_ylim(0, limit)
    axes[1].set_xlabel("Nano ball observations"); axes[1].set_ylabel("Medium ball observations")
    axes[1].set_title(f"Both establish contact: {len(paired)} events")
    fig.text(.1, .015, "No outcome references are used; more detections do not establish a correct ball identity.", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1)); fig.savefig(args.output_dir / "model_comparison.png", dpi=170)
    plt.close(fig)
    print(summary)


if __name__ == "__main__":
    main()

"""Render already-scored manual pose errors; never used by neural inference."""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def render(data, experiment, output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    targets = {r["id"]: r for r in read(data / "frame_targets.json")["frames"]}
    names = ["baseline_full_1920", "trained_last_full_1280"]
    predictions = {name: {r["frame_id"]: r for r in read(experiment / f"{name}_predictions.json")["rows"]} for name in names}
    evaluations = {name: read(experiment / f"{name}_evaluation.json") for name in names}
    associations = {name: {r["frame_id"]: r for r in evaluations[name]["associations"]} for name in names}
    score = {}
    for evaluation in evaluations.values():
        for row in evaluation["points"]:
            score[row["frame_id"]] = max(score.get(row["frame_id"], 0), row["error_px"] or 0)
    selected = sorted(score, key=lambda key: -score[key])[:5]
    panels, records = [], []
    colors = [(50, 220, 50), (230, 150, 30), (50, 70, 240)]
    for frame_id in selected:
        target = targets[frame_id]
        original = cv2.imread(str(data / target["image"]))
        people = {}
        for name in names:
            candidate = associations[name][frame_id].get("candidate_id")
            people[name] = next((r for r in predictions[name][frame_id]["people"] if r["candidate_id"] == candidate), None)
        boxes = [p["bbox"] for p in people.values() if p]
        x1 = max(0, math.floor(min(box[0] for box in boxes) - 30))
        y1 = max(0, math.floor(min(box[1] for box in boxes) - 30))
        x2 = min(original.shape[1], math.ceil(max(box[2] for box in boxes) + 30))
        y2 = min(original.shape[0], math.ceil(max(box[3] for box in boxes) + 30))
        crop = original[y1:y2, x1:x2].copy()
        scale = min(640 / crop.shape[1], 540 / crop.shape[0])
        crop = cv2.resize(crop, None, fx=scale, fy=scale)
        panel = np.zeros((620, 700, 3), dtype=np.uint8)
        panel[60:60 + crop.shape[0], 30:30 + crop.shape[1]] = crop
        cv2.putText(panel, frame_id, (15, 25), 0, .65, (255, 255, 255), 1, cv2.LINE_AA)
        for joint, use in enumerate(target["coordinate_loss_mask"]):
            if not use:
                continue
            positions = [target["joint_xy_px"][joint]] + [people[name]["keypoints"][joint] if people[name] else None for name in names]
            for source, xy in enumerate(positions):
                if xy is None:
                    continue
                xy = (round((xy[0] - x1) * scale) + 30, round((xy[1] - y1) * scale) + 60)
                cv2.circle(panel, xy, 6, colors[source], 2, cv2.LINE_AA)
                cv2.putText(panel, str(joint), (xy[0] + 6, xy[1] - 5), 0, .45, colors[source], 1, cv2.LINE_AA)
        for index, label in enumerate(["Green: manual", "Blue: pretrained 1920", "Red: trained 1280"]):
            cv2.putText(panel, label, (15 + index * 220, 600), 0, .45, colors[index], 1, cv2.LINE_AA)
        cv2.imwrite(str(output / f"{frame_id}.png"), panel)
        records.append({"frame_id": frame_id, "max_error_px": score[frame_id], "display_crop_xyxy": [x1, y1, x2, y2],
                        "crop_for_diagnostic_display_only": True,
                        "associations": {name: associations[name][frame_id] for name in names}})
        panels.append(panel)
    cv2.imwrite(str(output / "largest_errors.png"), np.vstack(panels))
    (output / "selection.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.data, args.experiment, args.output)

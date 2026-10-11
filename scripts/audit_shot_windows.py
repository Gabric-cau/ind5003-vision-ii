"""Diagnose first-shot failures and propose full-field windows, never ground truth."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from src.step_pipeline import _histogram_for_cut
from src.field_masks import build_field_mask


def inspect(source, out):
    cap = cv2.VideoCapture(source["video_path"])
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25)
    limit = int(math.ceil(12 * fps))
    cuts, samples = [0], []
    previous = None
    frames = {}
    index = 0
    while index < limit:
        ok, frame = cap.read()
        if not ok:
            break
        hist = _histogram_for_cut(frame)
        if previous is not None and index - cuts[-1] >= max(1, round(.5 * fps)):
            if cv2.compareHist(previous, hist, cv2.HISTCMP_CHISQR) > 30:
                cuts.append(index)
        if index % max(1, round(.2 * fps)) == 0:
            small = cv2.resize(frame, (640, 360))
            mask = build_field_mask(small)
            samples.append((index, mask.valid, mask.coverage))
            frames[index] = small
        previous = hist
        index += 1
    cap.release()
    cuts.append(index)
    windows, panels = [], []
    for start, end in zip(cuts, cuts[1:]):
        part = [s for s in samples if start <= s[0] < end]
        fraction = float(np.mean([s[1] for s in part])) if part else 0.
        eligible = (end - start) / fps >= 1.5 and fraction >= .7
        windows.append({"start_frame": start, "end_frame_exclusive": end,
                        "duration_sec": (end - start) / fps, "field_mask_fraction": fraction,
                        "candidate_full_field_window": eligible})
        if part:
            mid = min(part, key=lambda s: abs(s[0] - (start + end) / 2))[0]
            tile = cv2.copyMakeBorder(frames[mid], 30, 0, 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(tile, f"f{start}:{end} | field-mask {fraction:.2f} | proposal={eligible}",
                        (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            panels.append(tile)
    if panels:
        success, encoded = cv2.imencode(".jpg", np.vstack(panels))
        if not success:
            raise ValueError("Cannot encode contact sheet")
        (out / f"{source['clip_id']}_shots.jpg").write_bytes(encoded.tobytes())
    return {"clip_id": source["clip_id"], "old_shot_end_frame": source["shot_end_frame"],
            "windows": windows, "recommended_window": next((w for w in windows if w["candidate_full_field_window"]), None),
            "human_verified": False, "note": "Search-window proposal only; must check action completeness and replay boundaries."}


def main():
    out = ROOT / "outputs/mask_optimization_2026-10-08/shot_audit"
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads((ROOT / "outputs/step_diagnostics.json").read_text(encoding="utf8"))
    rows = []
    for source in data:
        if source["status"] == "failed":
            print(source["clip_id"], flush=True)
            rows.append(inspect(source, out))
    (out / "windows.json").write_text(json.dumps(rows, indent=2), encoding="utf8")
    print(json.dumps([{k: row[k] for k in ("clip_id", "old_shot_end_frame", "recommended_window")} for row in rows], indent=2))


if __name__ == "__main__":
    main()

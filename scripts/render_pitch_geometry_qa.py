"""Render mask/Canny/Hough evidence without inventing homography landmarks."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from src.field_masks import pitch_line_candidates


def main():
    out = ROOT / "outputs/mask_optimization_2026-10-08/pitch_geometry"
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads((ROOT / "outputs/step_diagnostics.json").read_text(encoding="utf8"))
    audit_file = out.parent / "shot_audit/windows.json"
    audits = {r["clip_id"]: r for r in json.loads(audit_file.read_text(encoding="utf8"))} if audit_file.exists() else {}
    records = []
    for source in data:
        proposal = audits.get(source["clip_id"], {}).get("recommended_window")
        frame_id = max(0, int(source.get("kick_frame") or source["shot_end_frame"]) - 5)
        if proposal:
            frame_id = int(proposal["start_frame"]) + 15
        cap = cv2.VideoCapture(source["video_path"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise ValueError(f"Cannot read {source['clip_id']}")
        frame = cv2.resize(frame, (640, 360))
        edges, lines = pitch_line_candidates(frame)
        drawn = frame.copy()
        for x1, y1, x2, y2 in lines:
            cv2.line(drawn, (x1, y1), (x2, y2), (0, 0, 255), 1)
        views = []
        for image, title in [(frame, f"Original frame {frame_id}"),
                             (cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), "HSV region + Canny"),
                             (drawn, f"Hough candidates: {len(lines)} | NOT calibration")]:
            tile = cv2.copyMakeBorder(image, 32, 0, 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(tile, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            views.append(tile)
        ok, encoded = cv2.imencode(".jpg", np.hstack(views))
        if not ok:
            raise ValueError("Cannot encode preview")
        (out / f"{source['clip_id']}.jpg").write_bytes(encoded.tobytes())
        records.append({"clip_id": source["clip_id"], "frame": frame_id,
                        "line_candidate_count": len(lines), "calibration_completed": False})
    (out / "index.json").write_text(json.dumps(records, indent=2), encoding="utf8")
    print(f"Wrote {len(records)} geometry previews; no metric calibration asserted.")


if __name__ == "__main__":
    main()

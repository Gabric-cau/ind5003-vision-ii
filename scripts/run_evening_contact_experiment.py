"""One bounded contact-density ablation on the eight baseline abstentions."""
from pathlib import Path
import sys
import json
import hashlib
import time
import os
import argparse
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / ".venv/Lib/site-packages"))
RUN = ROOT / "outputs/optimization_evening_2026-10-09"


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["legacy", "stationary_transition"], default="legacy")
    parser.add_argument("--cohort", choices=["unresolved", "all"], default="unresolved")
    parser.add_argument("--directory", default="contact_dense")
    parser.add_argument("--roi", action="store_true")
    args = parser.parse_args()
    dest = RUN / args.directory
    dest.mkdir(parents=True, exist_ok=True)
    lock = dest / "run.lock"
    with lock.open("x", encoding="utf-8") as f:
        f.write(json.dumps({"pid": os.getpid(), "started": datetime.now(timezone.utc).isoformat()}))
    started = time.monotonic()
    try:
        import torch
        import cv2
        torch.set_num_threads(3)
        cv2.setNumThreads(3)
        from ultralytics import YOLO
        from src.kick_contact import estimate_kick_contact
        from scripts.probe_kick_contact import render_contact_qa
        summary = json.loads((ROOT / "outputs/style_aware_steps_2026-10-09/classified/summary.json").read_text())
        clips = [r["clip_id"] for r in summary["rows"] if args.cohort == "all" or r["candidate_step_count"] is None]
        weights = ROOT / "models/weights/yolo11n_object_detection.pt"
        provenance = {"source_sha256": sha(ROOT / "src/kick_contact.py"), "weights_sha256": sha(weights),
                      "stride": 3, "imgsz": 1280, "score_thresholds_unchanged": True,
                      "contact_window_policy": args.policy, "cohort": args.cohort,
                      "roi_enabled":args.roi,
                      "python": sys.version, "torch": torch.__version__, "opencv": cv2.__version__}
        save(dest / "protocol.json", {"clips": clips, **provenance})
        model = YOLO(str(weights))
        rows = []
        for cid in clips:
            if time.monotonic() - started > 1650:
                print("Stopping before next clip: budget", flush=True)
                break
            output = dest / f"{cid}.json"
            if output.exists():
                raise FileExistsError(output)
            source = ROOT / "outputs/shot_repair_2026-10-09/all_clips_final" / f"{cid}.json"
            rescued = ROOT / "outputs/phase_optimization_2026-10-09/shot_rescue/poses" / f"{cid}.json"
            if rescued.exists():
                source = rescued
            pose = json.loads(source.read_text(encoding="utf-8"))
            used_model = model
            roi = None
            if args.roi:
                from src.ball_roi_probe import propose_ball_corridor, ROIPredictor
                previous = json.loads((RUN/"contact_transition"/f"{cid}.json").read_text())
                roi = propose_ball_corridor(pose,previous)
                if roi:
                    used_model = ROIPredictor(model,roi["bbox"])
            print("START", cid, flush=True)
            t = time.monotonic()
            result = estimate_kick_contact(pose["video_path"], pose, used_model, sample_stride=3, imgsz=1280,
                                           contact_window_policy=args.policy)
            result["experiment"] = {**provenance, "pose_sha256": sha(source), "elapsed_seconds": time.monotonic()-t}
            result["experiment"]["roi"] = roi
            save(output, result)
            render_contact_qa(pose, result, dest / f"{cid}_qa.jpg")
            baseline = json.loads((ROOT / "outputs/phase_optimization_2026-10-09/contact_probe" / f"{cid}.json").read_text())
            rescued_contact = ROOT / "outputs/phase_optimization_2026-10-09/shot_rescue/contacts" / f"{cid}.json"
            if rescued_contact.exists():
                baseline = json.loads(rescued_contact.read_text())
            row = {"clip_id": cid, "baseline_status": baseline["status"], "new_status": result["status"],
                   "baseline_contact": baseline.get("contact_frame"), "new_contact": result.get("contact_frame"),
                   "new_track": result.get("contact_associated_track_id"), "warnings": result["warnings"],
                   "elapsed_seconds": time.monotonic()-t}
            rows.append(row)
            save(dest / "comparison.json", rows)
            print(json.dumps(row), flush=True)
        save(dest / "completion.json", {"status": "complete" if len(rows)==len(clips) else "budget_stopped", "clips": len(rows),
                                       "elapsed_seconds": time.monotonic()-started, "human_verified": False, "accuracy_validated": False})
    except BaseException as exc:
        save(dest / "failure.json", {"type": type(exc).__name__, "error": str(exc), "elapsed_seconds": time.monotonic()-started})
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

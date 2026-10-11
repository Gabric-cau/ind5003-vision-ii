"""Paired no-seed GK ablation; input kick times remain explicitly AI-unverified."""
from pathlib import Path
import sys, json, csv, time, hashlib, os
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / "outputs/optimization_evening_2026-10-09/goalkeeper_no_seeds"


def save(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    import cv2
    cv2.setNumThreads(2)
    from src.goalkeeper_pipeline import analyze_goalkeeper_result, _manifest_metadata
    RUN.mkdir(parents=True, exist_ok=True)
    lock = RUN / "run.lock"
    with lock.open("x") as h:
        json.dump({"pid":os.getpid(),"started":datetime.now(timezone.utc).isoformat()},h)
    started = time.monotonic()
    try:
        by_id = {}
        inputs = [ROOT/"outputs/step_diagnostics.json", ROOT/"outputs/goalkeeper_pose_fallback.json"]
        for p in inputs:
            by_id.update({v["clip_id"]:v for v in json.loads(p.read_text(encoding="utf-8"))})
        kicks = {v["clip_id"]:v for v in csv.DictReader((ROOT/"data/annotations/penalty_kick_frames.csv").open(encoding="utf-8"))}
        ids = list(_manifest_metadata(ROOT/"configs/penalty_slicing_manifest.yaml"))
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        save(RUN/"protocol.json", {"clips":ids,"no_seed_labels_used":True,
             "old_kick_times_are_ai_unverified":True,"score_thresholds_unchanged":True,
             "code_sha256":sha(ROOT/"src/goalkeeper_pipeline.py"),
             "pose_sha256":{p.name:sha(p) for p in inputs},"python":sys.version,"opencv":cv2.__version__})
        rows=[]
        for cid in ids:
            if time.monotonic()-started > 1150:
                break
            record={}
            for policy in ("legacy","bounded"):
                dest=RUN/policy/f"{cid}.json"
                if dest.exists():
                    raise FileExistsError(dest)
                out=analyze_goalkeeper_result(by_id[cid],int(kicks[cid]["kick_frame"]),
                     kick_frame_source="legacy_AI_estimate_not_human_verified",
                     kick_frame_confidence="unverified",seed_track_id=None,tracking_policy=policy)
                out.update(human_verified=False,accuracy_validated=False,no_seed_labels_used=True)
                save(dest,out)
                record[policy]={k:out.get(k) for k in ("goalkeeper_track_id","auto_accepted","review_required",
                         "dive_direction","dive_height","gk_prekick_movement_level","qa_warnings")}
            rows.append({"clip_id":cid,**record})
            save(RUN/"comparison.json",rows)
            print(json.dumps(rows[-1]),flush=True)
        save(RUN/"completion.json",{"status":"complete" if len(rows)==len(ids) else "budget_stopped",
             "clips":len(rows),"elapsed_seconds":time.monotonic()-started,
             "legacy_no_seed_qa_accepted":sum(v["legacy"]["auto_accepted"] for v in rows),
             "bounded_no_seed_qa_accepted":sum(v["bounded"]["auto_accepted"] for v in rows),
             "human_verified":False,"accuracy_validated":False})
    finally:
        lock.unlink(missing_ok=True)


if __name__=="__main__":
    main()

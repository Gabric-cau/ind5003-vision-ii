"""Anchor/window sensitivity for foot predictions; not an accuracy estimate."""
from pathlib import Path
from collections import Counter
import sys,json,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.step_pipeline import _track_pose_sequence
from src.kicker_analysis import KickerFeatureExtractor
RUN=ROOT/"outputs/optimization_evening_2026-10-09"
OUT=RUN/"foot_sensitivity";OUT.mkdir(parents=True,exist_ok=True)
def load(p):return json.loads(p.read_text(encoding="utf-8"))
rows=[]
for path in sorted((RUN/"steps_v4/classified").glob("*.json")):
    if path.name=="summary.json":continue
    r=load(path);posepath=Path(r["terminal_provenance"]["selected_pose_path"]);pose=load(posepath)
    track=next(t for t in pose["tracks"] if t["track_id"]==r["contact"]["contact_associated_track_id"])
    seq=_track_pose_sequence(track,.35);variants=[]
    for anchor in sorted(set([r["contact"]["contact_frame"],*r["contact"]["interval_frames"]])):
        for window in (3,5):
            v=KickerFeatureExtractor().classify_kicking_foot_from_motion(
                **{k:seq[k] for k in ("frames","left_ankles","right_ankles","left_hips","right_hips")},kick_frame=anchor,window=window)
            ratio=v.get("motion_ratio",0)
            variants.append({"anchor":anchor,"window":window,"label":v["kicking_foot"],
                             "motion_ratio":ratio if math.isfinite(ratio) else None})
    labels={v["label"] for v in variants}
    stable=len(labels)==1 and "unknown" not in labels and all(v["motion_ratio"] is not None and v["motion_ratio"]>=1.25 for v in variants)
    row={"clip_id":r["clip_id"],"nominal_candidate":r["foot_analysis"]["result"]["kicking_foot"],
         "stable_candidate":next(iter(labels)) if stable else "unknown","parameter_stable":stable,
         "variants":variants,"human_verified":False,"accuracy_validated":False}
    (OUT/path.name).write_text(json.dumps(row,indent=2),encoding="utf-8");rows.append(row)
(OUT/"summary.json").write_text(json.dumps({"clips":len(rows),"stable_candidates":sum(r["parameter_stable"] for r in rows),
           "nominal_labels":dict(Counter(r["nominal_candidate"] for r in rows)),"rows":rows,
           "meaning":"Consistency across fixed anchors and windows, not correctness or footedness ground truth"},indent=2),encoding="utf-8")

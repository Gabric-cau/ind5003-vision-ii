"""Paired tracking ablation reusing the fresh goal-region pose detections."""
from pathlib import Path
import sys,json,hashlib,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from src.goalkeeper_pipeline import analyze_goalkeeper_result
from scripts.render_goalkeeper_qa import render_row
cv2.setNumThreads(2)
parser=argparse.ArgumentParser();parser.add_argument("--directory",default="goal_appearance");args=parser.parse_args()
RUN=ROOT/"outputs/optimization_evening_2026-10-09";SOURCE=RUN/"goal_pose_all";OUT=RUN/args.directory
OUT.mkdir(parents=True,exist_ok=True)
def load(p):return json.loads(p.read_text(encoding="utf-8"))
rows=[];panels=[]
for row in load(SOURCE/"summary.json")["rows"]:
    cid=row["clip_id"];dest=OUT/(cid+".json")
    if dest.exists():raise FileExistsError(dest)
    old=load(SOURCE/(cid+".json"));posepath=SOURCE/(cid+"_pose.json")
    evidence=old.get("goal_identity_evidence",{});seed=evidence.get("seed_track_id")
    if posepath.exists() and seed is not None:
        pose=load(posepath)
        pose["goal_geometry_bounds"]=evidence["goal"]["selected"]["bbox"]
        out=analyze_goalkeeper_result(pose,old["kick_frame"],seed_track_id=seed,tracking_policy="appearance_bounded",
             kick_frame_source=old["kick_frame_source"],kick_frame_confidence=old["kick_frame_confidence"])
        out["goal_identity_evidence"]=evidence
        panel=render_row(out);cv2.imwrite(str(OUT/(cid+"_qa.jpg")),panel);panels.append(panel)
    else:out=dict(old)
    out.update(human_verified=False,accuracy_validated=False,source_pose_path=str(posepath),
               code_sha256={n:hashlib.sha256((ROOT/"src"/n).read_bytes()).hexdigest() for n in ("goalkeeper_pipeline.py","person_appearance.py")})
    dest.write_text(json.dumps(out,indent=2,allow_nan=False),encoding="utf-8")
    rows.append({"clip_id":cid,"baseline_qa":old["auto_accepted"],"baseline_track":old["goalkeeper_track_id"],
         **{k:out.get(k) for k in ("goalkeeper_track_id","auto_accepted","review_required","dive_direction","dive_height","qa_warnings")}})
    (OUT/"summary.json").write_text(json.dumps({"clips":len(rows),"rows":rows,
         "qa_accepted":sum(r["auto_accepted"] for r in rows),"human_verified":False,"accuracy_validated":False},indent=2),encoding="utf-8")
    print(json.dumps(rows[-1]),flush=True)
for i in range(0,len(panels),6):cv2.imwrite(str(OUT/f"montage_{i//6+1}.jpg"),np.vstack(panels[i:i+6]))

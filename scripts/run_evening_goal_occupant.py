"""Goal geometry association instead of manually/AI seeded keeper identities."""
from pathlib import Path
import sys,json,csv,time,hashlib,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from src.goal_geometry import select_goal_occupant
from src.goalkeeper_pipeline import analyze_goalkeeper_result,_manifest_metadata
from scripts.render_goalkeeper_qa import render_row
cv2.setNumThreads(2)
parser=argparse.ArgumentParser()
parser.add_argument("--directory",default="goalkeeper_geometry")
args=parser.parse_args()
RUN=ROOT/"outputs/optimization_evening_2026-10-09"/args.directory
RUN.mkdir(parents=True,exist_ok=True)
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf-8")
byid={}
for name in ("step_diagnostics","goalkeeper_pose_fallback"):
    byid.update({r["clip_id"]:r for r in json.loads((ROOT/"outputs"/(name+".json")).read_text(encoding="utf-8"))})
kicks={r["clip_id"]:r for r in csv.DictReader((ROOT/"data/annotations/penalty_kick_frames.csv").open(encoding="utf-8"))}
protocol={"uses_identity_annotations":False,"timing_source":"legacy_AI_kick_frame_unverified",
          "all_canonical_clips":True,"human_verified":False,"accuracy_validated":False,
          "source_sha256":{n:hashlib.sha256((ROOT/"src"/n).read_bytes()).hexdigest() for n in ("goal_geometry.py","goalkeeper_pipeline.py")}}
save(RUN/"protocol.json",protocol)
rows=[]
for cid in _manifest_metadata(ROOT/"configs/penalty_slicing_manifest.yaml"):
    dest=RUN/(cid+".json")
    if dest.exists():raise FileExistsError(dest)
    pose=byid[cid];kick=int(kicks[cid]["kick_frame"]);fid=max(0,kick-round(.35*pose["fps"]))
    cap=cv2.VideoCapture(pose["video_path"]);cap.set(1,fid);ok,frame=cap.read();cap.release()
    evidence=select_goal_occupant(frame,pose["tracks"],fid) if ok else {"status":"abstain","seed_track_id":None}
    if evidence["seed_track_id"] is not None:
        result=analyze_goalkeeper_result(pose,kick,seed_track_id=evidence["seed_track_id"],tracking_policy="bounded",
                kick_frame_source="legacy_AI_estimate_not_human_verified",kick_frame_confidence="unverified")
        result["identity_source"]="algorithmic_goal_frame_occupancy_not_manual"
        cv2.imwrite(str(RUN/(cid+"_qa.jpg")),render_row(result))
    else:
        result={"clip_id":cid,"goalkeeper_track_id":None,"auto_accepted":False,"review_required":True,
                "dive_direction":"unknown","dive_height":"unknown","gk_prekick_movement_level":"unknown",
                "qa_warnings":["goal_frame_identity_unresolved"]}
    result.update(human_verified=False,accuracy_validated=False,goal_identity_evidence=evidence)
    save(dest,result)
    if ok and evidence.get("goal",{}).get("selected"):
        cv2.polylines(frame,[np.array(evidence["goal"]["selected"]["corners"],np.int32)],True,(0,255,0),2)
        for c in evidence.get("candidates",[]):
            x1,y1,x2,y2=map(int,c["bbox"]);cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
            cv2.putText(frame,str(c["track_id"]),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
        cv2.imwrite(str(RUN/(cid+"_identity.jpg")),cv2.resize(frame,(960,540)))
    rows.append({"clip_id":cid,"seed":evidence["seed_track_id"],**{k:result.get(k) for k in
        ("goalkeeper_track_id","auto_accepted","review_required","dive_direction","dive_height","qa_warnings")}})
    save(RUN/"summary.json",{"rows":rows,"clips":len(rows),"identity_candidates":sum(r["seed"] is not None for r in rows),
                            "qa_accepted":sum(r["auto_accepted"] for r in rows),"human_verified":False,"accuracy_validated":False})
    print(json.dumps(rows[-1]),flush=True)

"""Fresh magnified goal-region pose with independently computed contact anchors."""
from pathlib import Path
import sys,json,argparse,time,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.append(str(ROOT/".venv/Lib/site-packages"))
import cv2,numpy as np,torch
cv2.setNumThreads(3);torch.set_num_threads(3)
from ultralytics import YOLO
from src.goal_geometry import detect_goal_frame,select_goal_occupant
from src.ball_roi_probe import ROIPredictor
from src.step_pipeline import _run_pose_in_batches,_build_tracks
from src.goalkeeper_pipeline import analyze_goalkeeper_result
from scripts.render_goalkeeper_qa import render_row
RUN=ROOT/"outputs/optimization_evening_2026-10-09"
parser=argparse.ArgumentParser();parser.add_argument("--all",action="store_true");parser.add_argument("--directory",default="goal_pose_pilot")
args=parser.parse_args();OUT=RUN/args.directory;OUT.mkdir(parents=True,exist_ok=True)
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False),encoding="utf-8")
ids=[r["clip_id"] for r in load(RUN/"steps_v4/classified/summary.json")["rows"]] if args.all else [
 "col_eng_p01_falcao_col9","col_eng_p02_kane_eng9","col_eng_p04_rashford_eng19","cro_den_p06_modric_cro10"]
save(OUT/"protocol.json",{"clips":ids,"no_identity_or_kick_annotations_read":True,"pre_sec":1.5,"post_sec":1.2,
    "pose_model":"yolo11n_pose.pt","image_size":960,"human_verified":False,"accuracy_validated":False,
    "source_sha256":{n:hashlib.sha256((ROOT/"src"/n).read_bytes()).hexdigest() for n in
                     ("goal_geometry.py","goalkeeper_pipeline.py","step_pipeline.py","ball_roi_probe.py")}})
model=YOLO(str(ROOT/"models/weights/yolo11n_pose.pt"));rows=[]
for cid in ids:
    if (OUT/(cid+".json")).exists():raise FileExistsError(cid)
    r=load(RUN/"steps_v4/classified"/(cid+".json"));fps=r["fps"];kick=r["contact"]["contact_frame"]
    fid=max(r["shot_start_frame"],kick-round(.35*fps))
    cap=cv2.VideoCapture(r["video_path"]);cap.set(1,fid);ok,frame=cap.read();cap.release()
    geometry=detect_goal_frame(frame) if ok else {}
    if geometry.get("selected") is None:
        out={"clip_id":cid,"status":"abstain_no_goal_region","goalkeeper_track_id":None,"auto_accepted":False,
             "review_required":True,"goal_geometry":geometry,"human_verified":False,"accuracy_validated":False}
    else:
        x1,y1,x2,y2=geometry["selected"]["bbox"];w=x2-x1;h=y2-y1;fh,fw=frame.shape[:2]
        roi=[max(0,int(x1-.45*w)),max(0,int(y1-.35*h)),min(fw,int(x2+.45*w)),min(fh,int(y2+.65*h))]
        start=max(r["shot_start_frame"],kick-round(1.5*fps));end=min(r["shot_end_frame_exclusive"],kick+round(1.2*fps)+1)
        started=time.monotonic()
        pf,warnings=_run_pose_in_batches(r["video_path"],end,ROIPredictor(model,roi),batch_size=8,pose_conf=.08,
                           predict_kwargs={"imgsz":960,"device":"cpu"},roi_inference_mode="full",shot_start_frame=start)
        tracks=_build_tracks(pf)
        pose={"clip_id":cid,"video_path":r["video_path"],"fps":fps,"frame_width":fw,"frame_height":fh,
              "selected_track_id":None,"tracks":list(tracks.values()),"pose_frames":pf}
        save(OUT/(cid+"_pose.json"),pose)
        identity=select_goal_occupant(frame,pose["tracks"],fid)
        if identity["seed_track_id"] is not None:
            out=analyze_goalkeeper_result(pose,kick,seed_track_id=identity["seed_track_id"],tracking_policy="bounded",
                  post_window_sec=1.0,kick_frame_source="independent_contact_candidate",kick_frame_confidence=r["contact"]["status"])
            cv2.imwrite(str(OUT/(cid+"_qa.jpg")),render_row(out))
        else:
            out={"clip_id":cid,"status":"abstain_no_goal_occupant","goalkeeper_track_id":None,"auto_accepted":False,"review_required":True}
        out.update(goal_identity_evidence=identity,roi=roi,pose_window=[start,end],elapsed_seconds=time.monotonic()-started)
    out.update(human_verified=False,accuracy_validated=False,kick_source_path=str(RUN/"steps_v4/classified"/(cid+".json")))
    save(OUT/(cid+".json"),out)
    rows.append({"clip_id":cid,**{k:out.get(k) for k in ("status","goalkeeper_track_id","auto_accepted","review_required","dive_direction","dive_height","qa_warnings")}})
    save(OUT/"summary.json",{"clips":len(rows),"rows":rows,"candidate_identity_available":sum(v["goalkeeper_track_id"] is not None for v in rows),
        "qa_accepted":sum(v["auto_accepted"] for v in rows),"human_verified":False,"accuracy_validated":False})
    print(json.dumps(rows[-1]),flush=True)

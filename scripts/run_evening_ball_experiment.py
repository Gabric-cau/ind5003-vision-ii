"""Paired shot-bounded ball windows; no outcome CSV is read by prediction."""
from pathlib import Path
from collections import Counter
import sys,json,csv,argparse,time,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.append(str(ROOT/".venv/Lib/site-packages"))
import cv2,numpy as np,torch
cv2.setNumThreads(3);torch.set_num_threads(3)
from ultralytics import YOLO
from src.ball_pipeline import _read_window,_extract_detections,select_ball_track
from src.ball_outcome import BallOutcomeExtractor
from src.goal_geometry import detect_goal_frame
from src.ball_roi_probe import ROIPredictor,propose_ball_corridor
parser=argparse.ArgumentParser();parser.add_argument("--limit",type=int,default=1);parser.add_argument("--directory",default="ball_pilot")
args=parser.parse_args();RUN=ROOT/"outputs/optimization_evening_2026-10-09";dest=RUN/args.directory
dest.mkdir(parents=True,exist_ok=True)
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False),encoding="utf-8")
baseline=list(csv.DictReader((ROOT/"data/processed/penalty_ball_outcome_features.csv").open(encoding="utf-8")))
clips=[r["clip_id"] for r in baseline if r["ball_track_status"]!="auto_accepted"][:args.limit]
save(dest/"protocol.json",{"clips":clips,"anchor_policy":"contact_roi_if_available_else_transition",
    "factorial_windows":[32,60],"detector_modes":["full","automatic_corridor"],"fixed_model":"yolo11n",
    "no_outcome_labels_read":True,"human_verified":False,"accuracy_validated":False,
    "code_sha256":{n:hashlib.sha256((ROOT/"src"/n).read_bytes()).hexdigest() for n in
                   ("ball_pipeline.py","ball_roi_probe.py","goal_geometry.py")}})
model=YOLO(str(ROOT/"models/weights/yolo11n_object_detection.pt"));rows=[]
for cid in clips:
    if (dest/(cid+".json")).exists():raise FileExistsError(cid)
    pose=load(ROOT/"outputs/shot_repair_2026-10-09/all_clips_final"/(cid+".json"))
    cp=RUN/"contact_roi"/(cid+".json")
    if not cp.exists():cp=RUN/"contact_transition"/(cid+".json")
    contact=load(cp)
    rescue=ROOT/"outputs/phase_optimization_2026-10-09/shot_rescue/poses"/(cid+".json")
    if rescue.exists():pose=load(rescue)
    kick=contact.get("contact_frame")
    if kick is None:
        record={"clip_id":cid,"status":"abstain_no_contact","variants":{},"human_verified":False}
        save(dest/(cid+".json"),record);rows.append(record);continue
    track=next((t for t in pose["tracks"] if t["track_id"]==contact["contact_associated_track_id"]),{})
    anchors=[o["keypoints"][j] for o in track.get("observations",[]) if abs(o["frame"]-kick)<=2
             for j in (15,16) if o["keypoint_confidences"][j]>=.25]
    cap=cv2.VideoCapture(pose["video_path"]);cap.set(1,max(0,kick-round(.35*pose["fps"])))
    ok,frame=cap.read();cap.release()
    geometry=detect_goal_frame(frame) if ok else {}
    goal=geometry.get("selected");goal_center=tuple(np.mean(goal["corners"],axis=0)) if goal else None
    roi=propose_ball_corridor(pose,contact)
    end=min(pose["shot_end_frame"]-1,kick+60)
    fids,frames,fps,w,h=_read_window(pose["video_path"],max(pose["shot_start_frame"],kick-3),end)
    variants={};started=time.monotonic()
    for mode,predictor in (("full",model),("corridor",ROIPredictor(model,roi["bbox"]) if roi else None)):
        if predictor is None:continue
        detections={}
        for start in range(0,len(frames),8):
            batch=frames[start:start+8]
            preds=predictor.predict(batch,imgsz=1280,conf=.01,classes=[32],device="cpu",verbose=False)
            detections.update(_extract_detections(fids[start:start+8],preds,batch))
        for window in (32,60):
            sample={f:d for f,d in detections.items() if f<=kick+window}
            path,sel=select_ball_track(sample,kick,anchors,goal_center=goal_center,preserve_completed=True)
            features=BallOutcomeExtractor().extract_trajectory_and_speed(
                [(r["frame"]/fps,r["x"],r["y"]) for r in path],fps=fps,frame_width=w,frame_height=h,
                expected_frames=path[-1]["frame"]-path[0]["frame"]+1 if path else len(sample),
                confidences=[r["confidence"] for r in path])
            qa=bool(path and anchors and not features["ball_review_required"] and not sel.get("contact_anchor_fallback")
                    and goal_center is not None and sel.get("progress_to_goal_px",-999)>=45 and sel.get("goal_alignment",-1)>=.45)
            max_gap=max((b["frame"]-a["frame"] for a,b in zip(path,path[1:])),default=0)
            continuity_supported=bool(len(path)>=3 and max_gap<=round(.25*fps))
            variants[f"{mode}_{window}"]={"trajectory":path,"selection":sel,"features":features,"qa_accepted":qa,
                  "continuity_supported":continuity_supported,"max_observed_gap_frames":max_gap,
                  "continuity_qa_accepted":bool(qa and continuity_supported),
                  "last_observed_frame":path[-1]["frame"] if path else None,"requested_post_frames":window,
                  "actual_window_end":max(sample) if sample else None,"endpoint_is_event_final_position":False}
            cap=cv2.VideoCapture(pose["video_path"]);cap.set(1,kick);ok,img=cap.read();cap.release()
            if ok:
                pts=np.array([[r["x"],r["y"]] for r in path],np.int32)
                if len(pts)>1:cv2.polylines(img,[pts],False,(0,255,255),2)
                for point in pts:cv2.circle(img,tuple(point),3,(0,0,255),-1)
                if goal:cv2.polylines(img,[np.array(goal["corners"],np.int32)],True,(0,255,0),2)
                cv2.putText(img,f'{mode} {window} points={len(path)} QA={qa} not accuracy',(20,40),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,255),2)
                cv2.imwrite(str(dest/f"{cid}_{mode}_{window}.jpg"),cv2.resize(img,(960,540)))
    record={"clip_id":cid,"status":"experimental","contact_path":str(cp),"contact":contact["contact_frame"],
            "goal_geometry":geometry,"roi":roi,"variants":variants,"elapsed_seconds":time.monotonic()-started,
            "human_verified":False,"accuracy_validated":False,"outcome_prediction":"unknown"}
    save(dest/(cid+".json"),record)
    rows.append({"clip_id":cid,"status":"experimental","variants":{k:{"qa_accepted":v["qa_accepted"],
              "points":len(v["trajectory"]),"last_frame":v["last_observed_frame"],
              "max_gap":v["max_observed_gap_frames"],"continuity_qa":v["continuity_qa_accepted"]} for k,v in variants.items()}})
    save(dest/"summary.json",{"rows":rows,"clips":len(rows),"no_outcome_labels_read":True})
    print(json.dumps(rows[-1]),flush=True)
save(dest/"summary.json",{"rows":rows,"clips":len(rows),"no_outcome_labels_read":True,"human_verified":False,"accuracy_validated":False})

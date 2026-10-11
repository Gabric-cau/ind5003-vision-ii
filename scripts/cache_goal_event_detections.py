"""Bounded source-only ball event evidence: dense full-frame plus goal ROI.

Keeps all detections rather than only a preselected outward-going trajectory.
No outcome table or external annotations are loaded.
"""
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import argparse,hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.append(str(ROOT/".venv/Lib/site-packages"))
import cv2,numpy as np,torch
from ultralytics import YOLO
from src.ball_roi_probe import ROIPredictor
from src.ball_pipeline import _read_window,_extract_detections
from src.goal_geometry import detect_goal_frame
from src.goal_plane_rectification import project_image_point
from src.motion_stabilization import estimate_background_motion
cv2.setNumThreads(3);torch.set_num_threads(3)
BASE=ROOT/"outputs/optimization_evening_2026-10-09"
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--clips",nargs="+",default=["col_eng_p03_cuadrado_col11","col_eng_p06_henderson_eng8"])
    parser.add_argument("--output-dir",type=Path,default=ROOT/"outputs/goal_event_evidence_2026-10-10/detections")
    parser.add_argument("--batch-size",type=int,default=8)
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    weights=ROOT/"models/weights/yolo11n_object_detection.pt";model=YOLO(str(weights))
    for cid in args.clips:
        dest=out/(cid+".json")
        if dest.exists():raise FileExistsError(dest)
        started=time.monotonic();step_path=BASE/"steps_v4/classified"/(cid+".json");step=load(step_path)
        contact_path=BASE/"phase_v4/contact_probe"/(cid+".json");contact=load(contact_path)
        pose_path=Path(step["terminal_provenance"]["selected_pose_path"]);pose=load(pose_path)
        video=Path(step["video_path"]);video_hash=sha(video);fps=step["fps"];kick=step["contact"]["contact_frame"]
        start=max(step["shot_start_frame"],kick-round(.40*fps));end=min(step["shot_end_frame_exclusive"],kick+round(2.30*fps)+1)
        ids,frames,actual_fps,w,h=_read_window(str(video),start,end-1)
        if len(ids)!=end-start:raise ValueError("Incomplete input decode")
        geometries={fid:detect_goal_frame(fr) for fid,fr in zip(ids,frames)}
        goal_options=[(abs(fid-(kick-6)),g["selected"]) for fid,g in geometries.items() if g.get("selected")]
        if not goal_options:
            save(dest,{"clip_id":cid,"status":"abstain_no_goal","outcome":"unknown"});continue
        goal=min(goal_options,key=lambda x:x[0])[1];gx0,gy0,gx1,gy1=goal["bbox"]
        gw,gh=gx1-gx0,gy1-gy0
        roi=[max(0,int(gx0-.60*gw)),max(0,int(gy0-.30*gh)),min(w,int(gx1+.60*gw)),min(h,int(gy1+.80*gh))]
        all_dets={fid:[] for fid in ids}
        for mode,predictor,size in [("full",model,1280),("goal_roi",ROIPredictor(model,roi),1280)]:
            for j in range(0,len(frames),args.batch_size):
                batch=frames[j:j+args.batch_size];pred=predictor.predict(batch,imgsz=size,conf=.01,classes=[32],device="cpu",verbose=False)
                det=_extract_detections(ids[j:j+args.batch_size],pred,batch)
                for fid,items in det.items():
                    for d in items:all_dets[fid].append({**d,"sources":[mode]})
            print(f"{cid} {mode} decoded/detected {len(frames)} frames",flush=True)
        boxes=defaultdict(list)
        for track in pose.get("tracks",[]):
            for obs in track.get("observations",[]):
                if start<=obs["frame"]<end:boxes[obs["frame"]].append(obs["bbox"])
        camera=estimate_background_motion(str(video),start,end,boxes_by_frame=boxes)
        all_gray={fid:cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY) for fid,fr in zip(ids,frames)}
        for fid in ids:
            # Merge near-identical full-frame/ROI predictions; retain detector sources.
            merged=[]
            for d in sorted(all_dets[fid],key=lambda r:-r["confidence"]):
                hit=next((v for v in merged if np.hypot(d["x"]-v["x"],d["y"]-v["y"])<max(4,.40*min(d["width"],v["width"]))),None)
                if hit is not None:hit["sources"]=sorted(set(hit["sources"]+d["sources"]))
                else:merged.append(d)
            tf=camera["transforms"].get(str(fid))
            previous_tf=camera["transforms"].get(str(fid-1));diff=None
            if tf is not None and previous_tf is not None and fid-1 in all_gray:
                relative=np.linalg.inv(np.array(tf))@np.array(previous_tf)
                prev=cv2.warpPerspective(all_gray[fid-1],relative,(w,h))
                diff=cv2.absdiff(all_gray[fid],prev)
            for index,d in enumerate(merged):
                d["detection_id"]=f"{fid}:{index}"
                d["stabilized_xy"]=project_image_point(tf,[d["x"],d["y"]]) if tf is not None else None
                if diff is not None:
                    radius=max(3,int(min(d["width"],d["height"])/2))
                    x,y=round(d["x"]),round(d["y"]);patch=diff[max(0,y-radius):min(h,y+radius+1),max(0,x-radius):min(w,x+radius+1)]
                    d["motion_mean_abs_gray"]=round(float(np.mean(patch)),3) if patch.size else None
                else:d["motion_mean_abs_gray"]=None
            all_dets[fid]=merged
        result={"clip_id":cid,"status":"cached_candidates","video_path":str(video),"source_sha256":video_hash,
            "start_frame":start,"end_frame_exclusive":end,"fps":actual_fps,"width":w,"height":h,"kick_frame":kick,
            "contact":contact,"goal_roi":roi,"goal_proposals":geometries,"camera":camera,"detections":all_dets,
            "source_paths":{"step":str(step_path),"pose":str(pose_path),"contact":str(contact_path)},
            "protocol":{"model":str(weights),"weights_sha256":sha(weights),"detector_confidence":.01,
                        "image_size":1280,"batch_size":args.batch_size,"modes":["full","goal_roi"],"labels_read":False,
                        "created_at":datetime.now(timezone.utc).isoformat(),"code_sha256":sha(Path(__file__))},
            "outcome":"unknown","human_verified":False,"accuracy_validated":False,"elapsed_seconds":time.monotonic()-started}
        if sha(video)!=video_hash:raise AssertionError("Source video changed")
        save(dest,result)
        print(json.dumps({"clip":cid,"frames":len(ids),"detections":sum(map(len,all_dets.values())),"camera":camera["status"],"seconds":result["elapsed_seconds"]}),flush=True)


if __name__=="__main__":main()

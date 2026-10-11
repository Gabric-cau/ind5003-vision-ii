"""Experimental independent contact probe; never overwrites official outputs."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def render_contact_qa(pose, result, path):
    """Inspection-only filmstrip; colored boxes are predictions, not labels."""
    import cv2
    import numpy as np
    candidate = result.get("contact_frame")
    if candidate is None and result.get("candidates"):
        candidate=result["candidates"][0]["contact_frame"]
    if candidate is None:
        return
    cap=cv2.VideoCapture(pose["video_path"])
    tiles=[]
    for delta in (-12,-6,0,6,12):
        frame_id=max(pose["shot_start_frame"],min(pose["shot_end_frame"]-1,candidate+delta))
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_id)
        ok,frame=cap.read()
        if not ok:
            continue
        for track in pose.get("tracks",[]):
            observations=[o for o in track.get("observations",[]) if o["frame"]==frame_id]
            if not observations:
                continue
            x1,y1,x2,y2=map(int,observations[0]["bbox"])
            chosen=track["track_id"]==result.get("contact_associated_track_id")
            color=(0,255,255) if chosen else (180,180,180)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2 if chosen else 1)
            cv2.putText(frame,f'T{track["track_id"]}',(x1,y1-4),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
        for ball in result.get("ball_detections",[]):
            if ball["frame"]==frame_id:
                x,y=map(int,ball["xy"])
                cv2.circle(frame,(x,y),8,(0,0,255),2)
        frame=cv2.resize(frame,(640,360))
        cv2.putText(frame,f'{frame_id} / {frame_id/pose["fps"]:.2f}s | {result["status"]}',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
        tiles.append(frame)
    cap.release()
    if tiles:
        cv2.imwrite(str(path),np.vstack(tiles))


def render_qa_overviews(directory):
    """Four complete filmstrips per contact sheet for AI/user inspection."""
    import cv2
    import numpy as np
    files=sorted(directory.glob("*_contact_qa.jpg"))
    for start in range(0,len(files),4):
        columns=[]
        for path in files[start:start+4]:
            img=cv2.imread(str(path))
            if img is None:
                continue
            img=cv2.resize(img,(400,1125))
            header=np.zeros((48,400,3),np.uint8)
            label=path.name.replace("_contact_qa.jpg","")
            cv2.putText(header,label,(4,30),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
            columns.append(np.vstack((header,img)))
        if columns:
            cv2.imwrite(str(directory/f"qa_overview_{start//4+1:02d}.jpg"),np.hstack(columns))

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip",action="append",default=[])
    parser.add_argument("--all",action="store_true",help="Probe all per-clip pose files in the input folder")
    parser.add_argument("--pose-dir",type=Path,default=ROOT/"outputs/shot_repair_2026-10-09/all_clips_final")
    parser.add_argument("--out-dir",type=Path,default=ROOT/"outputs/phase_optimization_2026-10-09/contact_probe")
    parser.add_argument("--weights",type=Path,default=ROOT/"models/weights/yolo11n_object_detection.pt")
    parser.add_argument("--imgsz",type=int,default=1280)
    parser.add_argument("--stride",type=int,default=6)
    parser.add_argument("--threads",type=int,default=3)
    args=parser.parse_args()
    if args.all:
        args.clip=sorted(path.stem for path in args.pose_dir.glob("*.json") if (ROOT/"data/raw_clips"/(path.stem+".mp4")).exists())
    if not args.clip:
        parser.error("Provide --clip or --all")
    import torch
    torch.set_num_threads(args.threads)
    import cv2
    cv2.setNumThreads(args.threads)
    from ultralytics import YOLO
    from src.kick_contact import estimate_kick_contact
    model=YOLO(str(args.weights))
    if model.names.get(32) != "sports ball":
        raise ValueError("Expected COCO sports-ball class 32, but model class names differ")
    args.out_dir.mkdir(parents=True,exist_ok=True)
    summary=[]
    for clip_id in args.clip:
        pose=json.loads((args.pose_dir/f"{clip_id}.json").read_text(encoding="utf-8"))
        started=time.perf_counter()
        result=estimate_kick_contact(pose["video_path"],pose,model,sample_stride=args.stride,imgsz=args.imgsz)
        result["elapsed_sec"]=round(time.perf_counter()-started,2)
        result["object_weights"]=str(args.weights.resolve())
        result["object_class"]="sports ball (COCO 32)"
        # Existing estimate is added after inference solely for comparison.
        result["previous_pose_kick_frame_for_comparison_only"]=pose.get("kick_frame")
        (args.out_dir/f"{clip_id}.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        render_contact_qa(pose,result,args.out_dir/f"{clip_id}_contact_qa.jpg")
        row={key:result[key] for key in ("clip_id","status","contact_frame","contact_time_sec","confidence","sparse_frames","refined_frames","elapsed_sec")}
        row["previous_pose_kick_frame"]=pose.get("kick_frame")
        row["ball_detection_count"]=len(result["ball_detections"])
        summary.append(row)
        print(json.dumps(row),flush=True)
    # A second selected-clips run adds to the audit instead of hiding earlier runs.
    complete=[]
    for path in sorted(args.out_dir.glob("*.json")):
        if path.name=="summary.json":
            continue
        item=json.loads(path.read_text(encoding="utf-8"))
        if "method" not in item:
            continue
        complete.append({key:item.get(key) for key in ("clip_id","status","contact_frame","contact_time_sec","contact_associated_track_id","confidence","sparse_frames","refined_frames","elapsed_sec","previous_pose_kick_frame_for_comparison_only")})
    (args.out_dir/"summary.json").write_text(json.dumps(complete,ensure_ascii=False,indent=2),encoding="utf-8")
    render_qa_overviews(args.out_dir)

if __name__=="__main__":
    main()

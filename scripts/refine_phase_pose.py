"""Run optional medium-pose crops on automatically ball-associated actors."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

REPRESENTATIVE = ["col_eng_p03_cuadrado_col11", "esp_rus_p04_ignashevich_rus4",
                  "rus_cro_p02_brozovic_cro11", "rus_cro_p06_modric_cro10",
                  "esp_rus_p07_ramos_esp15", "col_eng_p07_uribe_col15"]


def render_qa(original, refined, destination):
    import cv2
    import numpy as np
    from src.pose_refinement import crop_bounds
    track_id = refined.get("selected_track_id")
    original_track = next((t for t in original.get("tracks",[]) if t["track_id"]==track_id),{})
    old = {o["frame"]:o for o in original_track.get("observations",[])}
    new = {o["frame"]:o for t in refined.get("tracks",[]) for o in t["observations"]}
    if not new:
        return
    ids = sorted(new)
    contact = int(refined["contact_frame"])
    targets = sorted(set([ids[0],ids[len(ids)//3],ids[len(ids)*2//3],*[
        min(ids,key=lambda f:abs(f-target)) for target in (contact-6,contact,contact+6)]]))
    capture=cv2.VideoCapture(original["video_path"])
    rows=[]
    for frame_id in targets:
        capture.set(cv2.CAP_PROP_POS_FRAMES,frame_id)
        ok,frame=capture.read()
        if not ok:
            continue
        box=(old.get(frame_id) or new[frame_id])["bbox"]
        x1,y1,x2,y2=crop_bounds(box,frame.shape[1],frame.shape[0],.6)
        tiles=[]
        for label,record in (("ORIGINAL",old.get(frame_id)),("REFINED",new.get(frame_id))):
            crop=frame[y1:y2,x1:x2].copy()
            if record:
                for j in (11,12,13,14,15,16):
                    if record["keypoint_confidences"][j] >= .25:
                        point=tuple(round(v-offset) for v,offset in zip(record["keypoints"][j],(x1,y1)))
                        cv2.circle(crop,point,2,(0,255,255),-1)
            scale=min(320/crop.shape[1],320/crop.shape[0])
            crop=cv2.resize(crop,(round(crop.shape[1]*scale),round(crop.shape[0]*scale)))
            tile=np.zeros((360,360,3),np.uint8)
            tile[35:35+crop.shape[0],:crop.shape[1]]=crop
            cv2.putText(tile,f"{label} T{track_id} f{frame_id}",(5,23),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
            tiles.append(tile)
        rows.append(np.hstack(tiles))
    capture.release()
    if rows:
        ok,image=cv2.imencode(".jpg",np.vstack(rows))
        if not ok:
            raise ValueError("Cannot encode refinement QA")
        destination.write_bytes(image.tobytes())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-dir",type=Path,default=ROOT/"outputs/shot_repair_2026-10-09/all_clips_final")
    parser.add_argument("--contact-dir",type=Path,default=ROOT/"outputs/phase_optimization_2026-10-09/contact_probe")
    parser.add_argument("--out-dir",type=Path,default=ROOT/"outputs/phase_optimization_2026-10-09/refined_pose")
    parser.add_argument("--weights",type=Path,default=ROOT/"models/weights/yolo11m_pose.pt")
    parser.add_argument("--clip",action="append")
    parser.add_argument("--all",action="store_true")
    parser.add_argument("--reuse",action="store_true",help="Read existing outputs; otherwise existing result files are not overwritten")
    parser.add_argument("--imgsz",type=int,default=384)
    parser.add_argument("--threads",type=int,default=3)
    args=parser.parse_args()
    if args.all and args.clip:
        parser.error("Use --all or --clip, not both")
    clips=args.clip or REPRESENTATIVE
    if args.all:
        clips=sorted(p.stem for p in args.contact_dir.glob("*.json") if (args.pose_dir/p.name).exists())
    import torch
    import cv2
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    from ultralytics import YOLO
    from src.pose_refinement import refine_contact_track
    model=None
    args.out_dir.mkdir(parents=True,exist_ok=True)
    weights_hash=hashlib.sha256(args.weights.read_bytes()).hexdigest()
    summary=[]
    for clip_id in clips:
        source=args.pose_dir/f"{clip_id}.json"
        contact_source=args.contact_dir/f"{clip_id}.json"
        if not source.exists() or not contact_source.exists():
            print(json.dumps({"clip_id":clip_id,"status":"skipped_missing_input"}),flush=True)
            continue
        destination=args.out_dir/f"{clip_id}.json"
        if destination.exists() and not args.reuse:
            raise FileExistsError(f"Refusing to overwrite {destination}; use --reuse or a new output directory")
        original=json.loads(source.read_text(encoding="utf-8"))
        if args.reuse and destination.exists():
            result=json.loads(destination.read_text(encoding="utf-8"))
        else:
            contact=json.loads(contact_source.read_text(encoding="utf-8"))
            if model is None and contact.get("status") in {"supported","tentative"}:
                model=YOLO(str(args.weights))
            started=time.perf_counter()
            result=refine_contact_track(original["video_path"],original,contact,model,imgsz=args.imgsz)
            result["elapsed_sec"]=round(time.perf_counter()-started,3)
            result["provenance"].update({"pose_input":str(source.resolve()),"pose_input_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
                "contact_input":str(contact_source.resolve()),"contact_input_sha256":hashlib.sha256(contact_source.read_bytes()).hexdigest(),
                "weights":str(args.weights.resolve()),"weights_sha256":weights_hash,
                "refinement_source_sha256":hashlib.sha256((ROOT/"src/pose_refinement.py").read_bytes()).hexdigest(),
                "torch_version":torch.__version__,"opencv_version":cv2.__version__,"threads":args.threads})
            destination.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
            render_qa(original,result,args.out_dir/f"{clip_id}_qa.jpg")
        row={key:result.get(key) for key in ("clip_id","status","reason","selected_track_id","contact_frame","contact_status","elapsed_sec")}
        row["original_quality"]=result.get("diagnostics",{}).get("original_associated_track")
        row["refined_quality"]=result.get("diagnostics",{}).get("refined_track")
        row["valid_frame_delta"]=result.get("diagnostics",{}).get("valid_frame_delta")
        summary.append(row)
        print(json.dumps(row),flush=True)
    (args.out_dir/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


if __name__=="__main__":
    main()

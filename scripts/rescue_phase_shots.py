"""Bounded later-shot rescue validated by independent contact evidence.

Writes separate experimental inputs. Never modifies old pose/contact outputs,
never uses old kick frames or hand-selected video times, and never claims labels
are human-verified. An unsuccessful probe is explicitly retained as no_rescue.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip",default="rus_cro_p04_kovacic_cro8")
    parser.add_argument("--max-later-shots",type=int,default=3)
    parser.add_argument("--threads",type=int,default=3)
    parser.add_argument("--out-dir",type=Path,default=ROOT/"outputs/phase_optimization_2026-10-09/shot_rescue")
    args=parser.parse_args()
    import torch
    import cv2
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    from ultralytics import YOLO
    from src.step_pipeline import _scan_camera_shots, _process_shot_steps
    from src.kick_contact import estimate_kick_contact
    from scripts.probe_kick_contact import render_contact_qa
    old=json.loads((ROOT/"outputs/shot_repair_2026-10-09/all_clips_final"/f"{args.clip}.json").read_text(encoding="utf-8"))
    video=str((ROOT/"data/raw_clips"/f"{args.clip}.mp4").resolve())
    shots=_scan_camera_shots(video,8,30)
    eligible=[s for s in shots if s["shot_start_frame"]>=old["shot_end_frame"]
              and s["shot_end_frame"]-s["shot_start_frame"]>=int(1.0*s["fps"])]
    print(json.dumps({"camera_shots":shots,"eligible_later_shots":len(eligible)}),flush=True)
    for subfolder in ("poses","contacts","attempts"):
        (args.out_dir/subfolder).mkdir(parents=True,exist_ok=True)
    pose_model=YOLO(str(ROOT/"models/weights/yolo11n_pose.pt"))
    object_model=YOLO(str(ROOT/"models/weights/yolo11n_object_detection.pt"))
    attempts=[]
    selected=None
    for shot in eligible[:args.max_later_shots]:
        started=time.perf_counter()
        pose=_process_shot_steps(video,pose_model,shot=shot,clip_id=args.clip,batch_size=12,
                                 roi_mode="auto",use_field_mask=False,
                                 predict_kwargs={"imgsz":960,"device":"cpu","verbose":False})
        contact=estimate_kick_contact(video,pose,object_model,imgsz=1280)
        candidate=contact["candidates"][0] if contact["candidates"] else {}
        rawtrack=next((t for t in pose.get("tracks",[]) if t["track_id"]==contact.get("contact_associated_track_id")),{})
        preceding=[o for o in rawtrack.get("observations",[]) if o["frame"]<=int(contact.get("contact_frame") or -1)]
        pre_span=(preceding[-1]["frame"]-preceding[0]["frame"])/pose["fps"] if preceding else 0
        acceptable=bool(contact["status"]=="supported" and candidate.get("ankle_approach_supported") and len(preceding)>=15 and pre_span>=.4)
        row={"shot_index":shot["shot_index"],"shot_start_frame":shot["shot_start_frame"],"shot_end_frame":shot["shot_end_frame"],
             "pose_status":pose["status"],"contact_status":contact["status"],"contact_frame":contact["contact_frame"],
             "contact_track_id":contact["contact_associated_track_id"],"precontact_observations":len(preceding),
             "precontact_span_sec":pre_span,"accepted_as_experimental_rescue":acceptable,"elapsed_sec":time.perf_counter()-started}
        attempts.append(row)
        prefix=f'{args.clip}_shot{shot["shot_index"]}'
        (args.out_dir/"attempts"/f"{prefix}_pose.json").write_text(json.dumps(pose,ensure_ascii=False,indent=2),encoding="utf-8")
        (args.out_dir/"attempts"/f"{prefix}_contact.json").write_text(json.dumps(contact,ensure_ascii=False,indent=2),encoding="utf-8")
        render_contact_qa(pose,contact,args.out_dir/"attempts"/f"{prefix}_qa.jpg")
        print(json.dumps(row),flush=True)
        if acceptable:
            pose["shot_rescue"]={"method":"earliest_later_shot_with_independent_ball_approach_departure_support",
                                 "previous_shot_start_frame":old["shot_start_frame"],"previous_shot_end_frame":old["shot_end_frame"],
                                 "human_verified":False,"uses_prior_kick_frame":False,"attempts":attempts,
                                 "contact_associated_track_id":contact["contact_associated_track_id"],
                                 "contact_frame":contact["contact_frame"],"contact_interval_frames":contact["interval_frames"]}
            pose["review_required"]=True
            pose["auto_accepted"]=False
            pose["qa_warnings"].append("independent_contact_shot_rescue_requires_review")
            (args.out_dir/"poses"/f"{args.clip}.json").write_text(json.dumps(pose,ensure_ascii=False,indent=2),encoding="utf-8")
            (args.out_dir/"contacts"/f"{args.clip}.json").write_text(json.dumps(contact,ensure_ascii=False,indent=2),encoding="utf-8")
            render_contact_qa(pose,contact,args.out_dir/f"{args.clip}_rescued_contact_qa.jpg")
            selected=row
            break
    report={"clip_id":args.clip,"status":"experimental_rescue" if selected else "no_rescue",
            "human_verified":False,"uses_prior_kick_frame":False,"camera_shots":shots,
            "max_later_shots":args.max_later_shots,"attempts":attempts,"selected":selected,
            "note":"No replay/live ground-truth labels. Earliest later independently supported event is a provisional rescue, requiring visual and eventual human review."}
    (args.out_dir/f"{args.clip}_rescue_summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"status":report["status"],"selected":selected}),flush=True)


if __name__=="__main__":
    main()

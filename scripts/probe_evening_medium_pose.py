"""One fixed-shot full-frame medium-pose experiment for fragmented identity."""
from pathlib import Path
import sys,json,time,hashlib,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.append(str(ROOT/".venv/Lib/site-packages"))
import cv2,torch
cv2.setNumThreads(3);torch.set_num_threads(3)
from ultralytics import YOLO
from src.step_pipeline import _process_shot_steps
from src.kick_contact import estimate_kick_contact
from scripts.probe_kick_contact import render_contact_qa
cid="rus_cro_p02_brozovic_cro11"
parser=argparse.ArgumentParser()
parser.add_argument("--roi",default="full",choices=["full","lower_tiles"])
parser.add_argument("--directory",default="medium_pose")
parser.add_argument("--cached-pose",type=Path)
parser.add_argument("--contact-policy",default="stationary_transition",choices=["stationary_transition","causal_first_departure"])
args=parser.parse_args()
RUN=ROOT/"outputs/optimization_evening_2026-10-09"/args.directory
RUN.mkdir(parents=True,exist_ok=True)
source=ROOT/"outputs/shot_repair_2026-10-09/all_clips_final"/(cid+".json")
old=json.loads(source.read_text(encoding="utf-8"))
shot=dict(old)
shot["shot_index"]=old.get("shot_index",0)
if (RUN/(cid+"_pose.json")).exists():raise FileExistsError(cid)
started=time.monotonic()
pose=(json.loads(args.cached_pose.read_text(encoding="utf-8")) if args.cached_pose else
      _process_shot_steps(old["video_path"],YOLO(str(ROOT/"models/weights/yolo11m_pose.pt")),shot=shot,
            clip_id=cid,roi_mode=args.roi,batch_size=6,predict_kwargs={"imgsz":960,"device":"cpu","verbose":False}))
pose["experiment"]={"weights_sha256":hashlib.sha256((ROOT/"models/weights/yolo11m_pose.pt").read_bytes()).hexdigest(),
                    "input_pose_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"same_shot_as_baseline":True,
                    "human_verified":False,"roi":args.roi,"cached_pose":str(args.cached_pose) if args.cached_pose else None,
                    "elapsed_seconds":time.monotonic()-started}
(RUN/(cid+"_pose.json")).write_text(json.dumps(pose,indent=2),encoding="utf-8")
contact=estimate_kick_contact(old["video_path"],pose,YOLO(str(ROOT/"models/weights/yolo11n_object_detection.pt")),
                              sample_stride=3,contact_window_policy=args.contact_policy)
(RUN/(cid+"_contact.json")).write_text(json.dumps(contact,indent=2),encoding="utf-8")
render_contact_qa(pose,contact,RUN/(cid+"_qa.jpg"))
print(json.dumps({"clip":cid,"contact_status":contact["status"],"frame":contact["contact_frame"],
                  "track":contact["contact_associated_track_id"],"elapsed":time.monotonic()-started}),flush=True)

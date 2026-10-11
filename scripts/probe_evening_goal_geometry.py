from pathlib import Path
import sys,json,hashlib,argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
from src.goal_geometry import detect_goal_frame
cv2.setNumThreads(2)
parser=argparse.ArgumentParser()
parser.add_argument("--directory",default="goal_geometry")
args=parser.parse_args()
RUN=ROOT/"outputs/optimization_evening_2026-10-09"/args.directory
RUN.mkdir(parents=True,exist_ok=True)
poses=ROOT/"outputs/shot_repair_2026-10-09/all_clips_final"
rows=[]
for p in sorted(poses.glob("*.json")):
    if not p.stem.startswith(("col_eng","cro_den","esp_rus","rus_cro")):continue
    pose=json.loads(p.read_text(encoding="utf-8"))
    if (RUN/(p.stem+".json")).exists(): raise FileExistsError(p.stem)
    frame_id=int(pose["shot_start_frame"])+10
    cap=cv2.VideoCapture(pose["video_path"]);cap.set(cv2.CAP_PROP_POS_FRAMES,frame_id)
    ok,frame=cap.read();cap.release()
    if not ok:continue
    result=detect_goal_frame(frame)
    result.update(clip_id=p.stem,frame=frame_id,video_path=pose["video_path"],
                  code_sha256=hashlib.sha256((ROOT/"src/goal_geometry.py").read_bytes()).hexdigest())
    (RUN/(p.stem+".json")).write_text(json.dumps(result,indent=2),encoding="utf-8")
    for i,c in enumerate(result["candidates"][:4]):
        import numpy as np
        cv2.polylines(frame,[np.array(c["corners"],np.int32)],True,(0,255,0) if i==0 else (0,0,255),2)
        cv2.putText(frame,f'{i}: {c["score"]:.2f}',tuple(np.array(c["corners"][0],int)),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
    cv2.putText(frame,result["status"],(20,40),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,255),2)
    cv2.imwrite(str(RUN/(p.stem+".jpg")),cv2.resize(frame,(960,540)))
    rows.append({"clip_id":p.stem,"status":result["status"],"candidates":len(result["candidates"])})
    print(json.dumps(rows[-1]),flush=True)
(RUN/"summary.json").write_text(json.dumps({"rows":rows,"supported":sum(r["status"]=="candidate_supported" for r in rows),
                                           "human_verified":False,"accuracy_validated":False},indent=2),encoding="utf-8")

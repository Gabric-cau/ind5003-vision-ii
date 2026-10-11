"""Label-blind feasibility thumbnails and scoreboard-region image-change cues."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from src.goalkeeper_pipeline import _manifest_metadata
cv2.setNumThreads(2)
OUT=ROOT/"outputs/optimization_evening_2026-10-09/outcome_visibility";OUT.mkdir(parents=True,exist_ok=True)
ids=list(_manifest_metadata(ROOT/"configs/penalty_slicing_manifest.yaml"))
selected=[]
for match in ("col_eng","cro_den","esp_rus","rus_cro"):
    cohort=[c for c in ids if c.startswith(match)]
    selected.extend([cohort[0],cohort[-1]])
rows=[];tiles=[]
for cid in selected:
    cap=cv2.VideoCapture(str(ROOT/"data/raw_clips"/(cid+".mp4")))
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));frames=[];fids=[0,int(.75*(count-1)),count-1]
    panels=[]
    for fid in fids:
        cap.set(1,fid);ok,f=cap.read()
        if not ok:raise ValueError((cid,fid))
        frames.append(f)
        tile=cv2.resize(f,(400,225));tile=cv2.copyMakeBorder(tile,26,0,0,0,cv2.BORDER_CONSTANT)
        cv2.putText(tile,f'{cid} f{fid}',(5,18),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1)
        panels.append(tile)
    cap.release()
    top=[cv2.resize(f[:int(.22*f.shape[0]),:int(.48*f.shape[1])],(240,90)) for f in frames]
    changes=[float(np.mean(cv2.absdiff(top[0],t))) for t in top[1:]]
    rows.append({"clip_id":cid,"sampled_frames":fids,"scoreboard_roi_mean_absolute_changes":changes,
                 "outcome_prediction":"unknown","reason":"Pixel changes do not decode a score or establish an event result",
                 "labels_used_for_prediction":False})
    tiles.append(np.hstack(panels))
for i in range(0,len(tiles),4):cv2.imwrite(str(OUT/f"visibility_{i//4+1}.jpg"),np.vstack(tiles[i:i+4]))
(OUT/"results.json").write_text(json.dumps({"scope":"eight stratified feasibility clips, three frames per clip",
       "clips":8,"rows":rows,"no_outcome_annotation_file_read":True,"human_verified":False,"accuracy_validated":False},indent=2),encoding="utf-8")

"""Local object-detector crop ablation; outputs original-video coordinates."""
from types import SimpleNamespace
import numpy as np
import cv2
from src.goal_geometry import detect_goal_frame


def propose_ball_corridor(pose, previous_probe):
    candidates=previous_probe.get("candidates") or []
    # Only approaching-foot candidates nominate a crop; this is not acceptance.
    candidates=[c for c in candidates if c.get("ankle_approach_supported")]
    if not candidates:return None
    c=max(candidates,key=lambda c:c["score"]);fid=c["contact_frame"]
    cap=cv2.VideoCapture(pose["video_path"]);cap.set(1,max(0,fid-round(.35*pose["fps"])))
    ok,frame=cap.read();cap.release()
    if not ok:return None
    goal=detect_goal_frame(frame)
    if goal.get("selected") is None:return None
    track=next((t for t in pose["tracks"] if t["track_id"]==c["contact_associated_track_id"]),{})
    boxes=[o["bbox"] for o in track.get("observations",[]) if fid-.7*pose["fps"]<=o["frame"]<=fid+.5*pose["fps"]]
    if not boxes:return None
    boxes.append(goal["selected"]["bbox"]);boxes=np.array(boxes)
    pad=max(35,.075*frame.shape[0]);h,w=frame.shape[:2]
    roi=[max(0,int(boxes[:,0].min()-pad)),max(0,int(boxes[:,1].min()-pad)),
         min(w,int(boxes[:,2].max()+pad)),min(h,int(boxes[:,3].max()+pad))]
    if (roi[2]-roi[0])*(roi[3]-roi[1])>.85*w*h:return None
    return {"bbox":roi,"source_contact_candidate":fid,"source_track_candidate":c["contact_associated_track_id"],
            "method":"automatic_goal_to_approaching_actor_corridor","human_verified":False}


class ROIPredictor:
    def __init__(self,model,roi):
        self.model=model;self.roi=roi

    def predict(self,frames,**kwargs):
        x1,y1,x2,y2=self.roi
        crops=[f[y1:y2,x1:x2] for f in frames]
        outputs=self.model.predict(crops,**kwargs)
        result=[]
        for p in outputs:
            if p.boxes is None:result.append(SimpleNamespace(boxes=None));continue
            xyxy=p.boxes.xyxy.clone();xyxy[:,[0,2]]+=x1;xyxy[:,[1,3]]+=y1
            item=SimpleNamespace(boxes=SimpleNamespace(xyxy=xyxy,conf=p.boxes.conf))
            if getattr(p,"keypoints",None) is not None:
                xy=p.keypoints.xy.clone();xy[:,:,0]+=x1;xy[:,:,1]+=y1
                item.keypoints=SimpleNamespace(xy=xy,conf=p.keypoints.conf)
            result.append(item)
        return result

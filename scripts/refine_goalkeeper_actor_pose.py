"""Tight automatic actor ROI ablation after a goal-line role is selected."""
from pathlib import Path
import sys,argparse,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_goalkeeper_direction_update import read,save,sha
from scripts.run_goalkeeper_role_experiment import compute,geometries_for
from src.goalkeeper_role_association import deduplicate_pose_people,unrotate_pose


def main():
    p=argparse.ArgumentParser();p.add_argument('--clip',default='col_eng_p03_cuadrado_col11');p.add_argument('--pilot',action='store_true')
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/goalkeeper_role_association_2026-10-11/experiment_actor_v3');a=p.parse_args()
    out=a.output_dir;out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise FileExistsError('Use a fresh output directory')
    base=ROOT/'outputs/goalkeeper_role_association_2026-10-11/experiment_v1';pose=read(base/f'{a.clip}_fresh_pose.json')
    earlier=read(base/f'{a.clip}_fresh_role_lock.json');baseline=read(ROOT/'outputs/goalkeeper_pose_direction_2026-10-11/results_v3_final/clips'/f'{a.clip}.json')
    kick=baseline['kick_frame'];seed=earlier['role']['seed_track_id']
    if seed is None:raise ValueError('No automatic pre-kick role candidate')
    pre=[o for t in pose['tracks'] if t['track_id']==seed for o in t['observations'] if kick-10<=o['frame']<kick]
    boxes=np.asarray([o['bbox'] for o in pre]);b=np.median(boxes,axis=0);scale=float(b[3]-b[1]);cx=float((b[0]+b[2])/2)
    roi=[max(0,int(cx-1.8*scale)),max(0,int(b[1]-.25*scale)),min(pose['frame_width'],int(cx+1.8*scale)),min(pose['frame_height'],int(b[3]+.55*scale))]
    protocol=dict(seed_source='automatic multi-frame goal-line role, not manual identity',coarse_pose_sha256=sha(base/f'{a.clip}_fresh_pose.json'),
                  roi=roi,rotations=[0,90,270],pilot=a.pilot,code_sha256={n:sha(ROOT/n) for n in ['scripts/refine_goalkeeper_actor_pose.py','src/goalkeeper_role_association.py','scripts/run_goalkeeper_role_experiment.py']},
                  human_verified=False,accuracy_validated=False)
    save(out/'protocol.json',protocol)
    sys.path.append(str(ROOT/'.venv/Lib/site-packages'));import torch
    torch.set_num_threads(3);cv2.setNumThreads(2);from ultralytics import YOLO
    from src.step_pipeline import _build_tracks
    model=YOLO(str(ROOT/'models/weights/yolo11m_pose.pt'));cap=cv2.VideoCapture(pose['video_path'])
    frames=[kick+12,kick+18,kick+24] if a.pilot else list(range(kick+8,kick+round(pose['fps'])+1))
    extra={};started=time.monotonic()
    for f in frames:
        cap.set(1,f);ok,img=cap.read()
        if not ok:raise ValueError(f)
        crop=img[roi[1]:roi[3],roi[0]:roi[2]];h,w=crop.shape[:2]
        views=[crop,cv2.rotate(crop,cv2.ROTATE_90_CLOCKWISE),cv2.rotate(crop,cv2.ROTATE_90_COUNTERCLOCKWISE)]
        preds=model.predict(views,imgsz=960,conf=.08,iou=.8,max_det=20,device='cpu',verbose=False);detections=[]
        for angle,r in zip([0,90,270],preds):
            if r.boxes is None or r.keypoints is None:continue
            boxes,kp=unrotate_pose(r.boxes.xyxy.cpu().numpy(),r.keypoints.xy.cpu().numpy(),w,h,angle)
            conf=r.keypoints.conf.cpu().numpy();boxes[:,[0,2]]+=roi[0];boxes[:,[1,3]]+=roi[1];kp[:,:,0]+=roi[0];kp[:,:,1]+=roi[1]
            for j in range(len(boxes)):
                detections.append(dict(frame=f,bbox=boxes[j].tolist(),keypoints=kp[j].tolist(),keypoint_confidences=conf[j].tolist(),
                                       box_confidence=float(r.boxes.conf[j]),source_roi=f'auto_actor_roi_rotation_{angle}'))
        extra[f]=detections
        print(f'actor ROI f{f}, detections={len(detections)}, elapsed={time.monotonic()-started:.1f}s',flush=True)
    cap.release();save(out/'extra_detections.json',extra)
    for row in pose['pose_frames']:
        if row['frame'] in extra:row['detections']=deduplicate_pose_people(row['detections']+extra[row['frame']])
    tracks=_build_tracks(pose['pose_frames']);pose['tracks']=list(tracks.values())
    save(out/f'{a.clip}_pose.json',pose)
    result=compute(pose,kick,geometries_for(pose,kick),baseline);save(out/f'{a.clip}_role_lock.json',result)
    print(dict(direction=result['experimental_direction'],seed=result['role']['seed_track_id'],
               last_frame=result['association'].get('last_observed_frame'),reasons=result['pose_direction_evidence']['reasons']))


if __name__=='__main__':main()

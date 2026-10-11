"""Fresh high-resolution pose and mirrored-view ablation on the user's camera cohort.

Reviewer labels are intentionally not loaded here. Evaluation is a separate step.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.append(str(ROOT/'.venv/Lib/site-packages'))
from src.kicking_foot_redetection import undo_mirror,transform_points,foot_evidence,approach_candidates,fuse_evidence


def load(p): return json.loads(p.read_text(encoding='utf-8'))
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def select_person(result,crop,expected,mirrored):
    if result.keypoints is None or not len(result.keypoints): return None
    boxes=result.boxes.xyxy.cpu().numpy(); xy=result.keypoints.xy.cpu().numpy(); conf=result.keypoints.conf.cpu().numpy()
    x0,y0,x1,y1=crop; w=x1-x0
    target=np.asarray(expected,float);centre=(target[:2]+target[2:])/2;h=max(1,target[3]-target[1])
    candidates=[]
    for b,k,c in zip(boxes,xy,conf):
        if mirrored:
            k,c=undo_mirror(k,c,w);b=np.array([w-1-b[2],b[1],w-1-b[0],b[3]])
        b=b+np.array([x0,y0,x0,y0]);k=k+np.array([x0,y0])
        dist=np.linalg.norm((b[:2]+b[2:])/2-centre)/h
        scale=abs(np.log(max(b[3]-b[1],1)/h))
        if dist<=.8 and scale<=.7:
            candidates.append((dist+.3*scale,k,c,b))
    if not candidates:return None
    _,k,c,b=min(candidates,key=lambda r:r[0])
    return dict(keypoints=k.tolist(),confidence=c.tolist(),bbox=b.tolist())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'outputs/goal_right_foot_redetection_2026-10-11');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    out=args.output.resolve()
    if out.exists() and not args.resume:raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    selection=load(ROOT/'outputs/goal_right_camera_review_2026-10-11/selection.json')['selected']
    ids=[r['clip_id'] for r in selection];weights=ROOT/'models/weights/yolo11m_pose.pt'
    protocol=dict(cohort=ids,scope='10 camera-group development clips; user labels not read by inference',
        new_neural_inference=True,weights=str(weights),weights_sha256=sha(weights),imgsz=640,
        configurations=['normal ROI','horizontally reflected ROI with inverse pixel AND channel mapping'],
        handwritten_foot_labels_used=False,approach_heuristic_not_anatomical_guarantee=True,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/kicking_foot_redetection.py']})
    save(out/'protocol.json',protocol)
    for p in [Path(__file__),ROOT/'src/kicking_foot_redetection.py']:
        (out/'source_snapshot').mkdir(exist_ok=True);(out/'source_snapshot'/p.name).write_bytes(p.read_bytes())
    import torch
    torch.set_num_threads(3);cv2.setNumThreads(3)
    from ultralytics import YOLO
    model=YOLO(str(weights)); rows=[]
    for index,cid in enumerate(ids,1):
        target=out/'results'/f'{cid}.json'
        if args.resume and target.exists():rows.append(load(target));continue
        started=time.monotonic();print(f'START {index}/10 {cid}',flush=True)
        oldpath=ROOT/'outputs/optimization_evening_2026-10-09/steps_v4/classified'/f'{cid}.json';old=load(oldpath)
        posepath=Path(old['terminal_provenance']['selected_pose_path']);pose=load(posepath)
        track=next(t for t in pose['tracks'] if t['track_id']==old['contact']['contact_associated_track_id'])
        observed={o['frame']:o for o in track['observations']}
        ballpath=ROOT/'outputs/ball_mask_review_ready_2026-10-11/results'/f'{cid}.json';ball=load(ballpath)['variants']['mask_and_radius_35']
        seed=ball['seed'];reference=np.median([p['stabilized_xy'] for p in seed],axis=0)
        moving=[p for p in ball['trajectory'] if np.linalg.norm(np.asarray(p['stabilized_xy'])-reference)>20]
        if not moving:raise ValueError(f'No measurable departure: {cid}')
        departure=moving[0]['frame'];last_seed=max(p['frame'] for p in seed if p['frame']<departure)
        first=max(0,last_seed-12);last=departure+7
        camerapath=ROOT/'outputs/ball_detection_all_2026-10-11/detections'/f'{cid}.json';camera=load(camerapath)
        transforms=camera['camera'].get('transforms',{})
        goalpath=ROOT/'outputs/optimization_evening_2026-10-09/goal_geometry_v3'/f'{cid}.json';geometry=load(goalpath).get('selected')
        goal=np.mean(np.asarray(geometry['corners'])[[2,3]],axis=0) if geometry else None
        vp=ROOT/'data/raw_clips'/f'{cid}.mp4';cap=cv2.VideoCapture(str(vp));cap.set(1,first)
        images=[];specs=[];source_frames={}
        for frame in range(first,last+1):
            ok,img=cap.read()
            if not ok:raise ValueError(f'Source decode {cid}:{frame}')
            nearest=min(observed,key=lambda f:abs(f-frame))
            if abs(nearest-frame)>2:continue
            b=np.asarray(observed[nearest]['bbox']);h=b[3]-b[1]
            crop=[max(0,int(b[0]-.45*h)),max(0,int(b[1]-.2*h)),min(img.shape[1],int(b[2]+.45*h)),min(img.shape[0],int(b[3]+.35*h))]
            tile=img[crop[1]:crop[3],crop[0]:crop[2]]
            source_frames[frame]=img
            for reflected in (False,True):
                images.append(np.ascontiguousarray(tile[:,::-1]) if reflected else tile.copy());specs.append((frame,crop,b,reflected))
        cap.release();samples={'normal':[],'mirrored':[]};raw=[]
        for start in range(0,len(images),8):
            predictions=model.predict(images[start:start+8],imgsz=640,conf=.15,device='cpu',verbose=False)
            for prediction,(frame,crop,b,reflected) in zip(predictions,specs[start:start+8]):
                p=select_person(prediction,crop,b,reflected)
                if p is None:continue
                mat=transforms.get(str(frame))
                raw.append(dict(frame=frame,reflected=reflected,crop=crop,**p,camera_transform_available=mat is not None))
                if mat is None:continue
                points=transform_points(p['keypoints'],mat)
                sample=dict(frame=frame,keypoints=points.tolist(),confidence=p['confidence'])
                samples['mirrored' if reflected else 'normal'].append(sample)
        # Inference records are frozen before any reviewer labels are loaded.
        normal=foot_evidence(samples['normal'],reference,last_seed,departure)
        mirrored=foot_evidence(samples['mirrored'],reference,last_seed,departure)
        approach=approach_candidates(samples['normal'],reference,goal,last_seed)
        result=dict(clip_id=cid,automatic_candidate=fuse_evidence(normal,mirrored),normal_evidence=normal,
            mirrored_evidence=mirrored,approach_heuristics=approach,source_video=str(vp),source_video_sha256=sha(vp),
            contact_bracket_frames=[last_seed,departure],old_contact_frame=old['contact']['contact_frame'],
            stationary_ball_reference=reference.tolist(),goal_ground_midpoint_uncalibrated=goal.tolist() if goal is not None else None,
            observed_original_pose_frames=len(samples['normal']),observed_mirrored_pose_frames=len(samples['mirrored']),
            raw_pose_file=f'poses/{cid}.json',reviewer_labels_used=False,accuracy_estimate=None,
            input_sha256={str(p):sha(p) for p in [oldpath,posepath,ballpath,camerapath,goalpath]},elapsed_seconds=time.monotonic()-started)
        save(out/'poses'/f'{cid}.json',raw);save(target,result);rows.append(result)
        print(json.dumps(dict(clip=cid,automatic=result['automatic_candidate']['candidate'],motion=normal['motion_candidate'],contact=normal['contact_candidate'],approach=approach,elapsed=round(result['elapsed_seconds'],1))),flush=True)
        # Original-resolution crop evidence; green labels are candidate anatomical channels.
        panels=[]
        for frame in range(last_seed-3,departure+3):
            if frame not in source_frames:continue
            observations=[p for p in raw if p['frame']==frame and not p['reflected']]
            if not observations:continue
            p=observations[0];img=source_frames[frame].copy();kp=np.asarray(p['keypoints']);cc=np.asarray(p['confidence'])
            for a,bidx in [(11,13),(13,15),(12,14),(14,16),(11,12)]:
                if min(cc[a],cc[bidx])>=.4:cv2.line(img,tuple(kp[a].astype(int)),tuple(kp[bidx].astype(int)),(0,255,0),2)
            for idx,label,color in [(15,'L',(255,80,20)),(16,'R',(20,80,255))]:
                if cc[idx]>=.4:
                    pos=tuple(kp[idx].astype(int));cv2.circle(img,pos,4,color,-1);cv2.putText(img,label,pos,cv2.FONT_HERSHEY_SIMPLEX,.5,color,2)
            ballraw=transform_points([reference],np.linalg.inv(np.asarray(transforms[str(frame)])))[0]
            cv2.circle(img,tuple(ballraw.astype(int)),7,(0,255,255),1)
            x0,y0,x1,y1=p['crop'];tile=cv2.resize(img[y0:y1,x0:x1],(360,300));cv2.putText(tile,f'frame {frame}',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2);panels.append(tile)
        if panels:
            count=len(panels);sheet=np.zeros((300*((count+3)//4),1440,3),np.uint8)
            for i,panel in enumerate(panels):y,x=divmod(i,4);sheet[y*300:(y+1)*300,x*360:(x+1)*360]=panel
            (out/'evidence').mkdir(exist_ok=True);cv2.imwrite(str(out/'evidence'/f'{cid}.jpg'),sheet)
        save(out/'progress.json',dict(processed=len(rows),total=len(ids)))
    save(out/'summary.json',dict(processed=len(rows),rows=rows,reviewer_labels_used_for_inference=False,accuracy_estimate=None))


if __name__=='__main__':main()

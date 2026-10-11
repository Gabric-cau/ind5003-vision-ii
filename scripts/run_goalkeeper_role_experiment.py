"""Paired cached/fresh role-lock experiment without manual identity seeds."""
from pathlib import Path
import argparse,sys,json,time,hashlib
from copy import deepcopy
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_goalkeeper_direction_update import read,save,sha,RUN
from src.goal_geometry import detect_goal_frame
from src.goalkeeper_role_association import select_goalkeeper_role,associate_locked_goalkeeper,deduplicate_pose_people,unrotate_pose
from src.person_appearance import torso_histogram
from src.goalkeeper_pipeline import _candidate_summary
from src.goalkeeper_pose_direction import pose_direction_evidence


def geometries_for(pose,kick):
    fps=pose['fps'];frames=sorted(set(kick-round(x*fps) for x in (.4,.33,.27,.2,.13,.067)))
    cap=cv2.VideoCapture(pose['video_path']);geometry={}
    for f in frames:
        if f<0:continue
        cap.set(1,f);ok,img=cap.read()
        if not ok:raise ValueError(f'Decode failed/{f}')
        g=detect_goal_frame(img)
        if g.get('selected'):geometry[f]=g
    cap.release();return geometry


def fresh_pose(base,kick,geometry,dest,model_path,imgsz,rotations):
    import torch
    torch.set_num_threads(3)
    sys.path.append(str(ROOT/'.venv/Lib/site-packages'))
    from ultralytics import YOLO
    from src.step_pipeline import _build_tracks
    if not model_path.is_file():raise FileNotFoundError(model_path)
    model=YOLO(str(model_path));fps=base['fps'];first=max(0,kick-round(.5*fps));end=kick+round(1.2*fps)+1
    corners=[g['selected']['bbox'] for g in geometry.values()]
    if not corners:raise ValueError('No supported pre-kick goal geometry')
    b=np.asarray(corners);x0,y0=b[:,:2].min(0);x1,y1=b[:,2:].max(0);gw=x1-x0;gh=y1-y0
    roi=[max(0,int(x0-.35*gw)),max(0,int(y0-.25*gh)),min(base['frame_width'],int(x1+.45*gw)),min(base['frame_height'],int(y1+.75*gh))]
    cap=cv2.VideoCapture(base['video_path']);cap.set(1,first);pf=[];started=time.monotonic()
    for fstart in range(first,end,6):
        frames=[];ids=[]
        for f in range(fstart,min(end,fstart+6)):
            ok,img=cap.read()
            if not ok:raise ValueError(f'Fresh pose source decode failed/{f}')
            frames.append(img[roi[1]:roi[3],roi[0]:roi[2]]);ids.append(f)
        views=[];contexts=[];byframe={f:[] for f in ids}
        for f,img in zip(ids,frames):
            for angle in rotations:
                view=img if angle==0 else cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE if angle==90 else cv2.ROTATE_90_COUNTERCLOCKWISE)
                views.append(view);contexts.append((f,angle,img.shape[1],img.shape[0]))
        predictions=model.predict(views,imgsz=imgsz,device='cpu',conf=.08,iou=.80,max_det=40,verbose=False)
        for (f,angle,cw,ch),p in zip(contexts,predictions):
            detections=byframe[f]
            if p.boxes is not None and p.keypoints is not None:
                boxes=p.boxes.xyxy.cpu().numpy();kp=p.keypoints.xy.cpu().numpy();cf=p.keypoints.conf.cpu().numpy()
                boxes,kp=unrotate_pose(boxes,kp,cw,ch,angle)
                boxes[:,[0,2]]+=roi[0];boxes[:,[1,3]]+=roi[1]
                kp[:,:,0]+=roi[0];kp[:,:,1]+=roi[1]
                for j in range(len(boxes)):
                    detections.append(dict(frame=f,bbox=boxes[j].tolist(),box_confidence=float(p.boxes.conf[j]),
                                           keypoints=kp[j].tolist(),keypoint_confidences=cf[j].tolist(),source_roi=f'auto_goal_role_crop_rotation_{angle}'))
            # No bbox-only overlap suppression: nearby distinct pose instances
            # remain separate. The model's explicit .80 NMS is recorded.
        for f in ids:pf.append(dict(frame=f,detections=deduplicate_pose_people(byframe[f])))
        print(f'{base["clip_id"]} fresh frame {ids[-1]}/{end-1} elapsed {time.monotonic()-started:.1f}s',flush=True)
    cap.release();tracks=_build_tracks(pf)
    pose={k:base[k] for k in ('clip_id','video_path','fps','frame_width','frame_height')}
    pose.update(tracks=list(tracks.values()),pose_frames=pf,selected_track_id=None,
                inference=dict(model_path=str(model_path),model_sha256=sha(model_path),imgsz=imgsz,conf=.08,nms_iou=.80,
                               roi=roi,rotations=rotations,frame_window=[first,end],elapsed_seconds=time.monotonic()-started))
    save(dest,pose);return pose


def compute(pose,kick,geometry,baseline):
    role=select_goalkeeper_role(pose,geometry,kick)
    hist={};cap=cv2.VideoCapture(pose['video_path'])
    byframe={}
    for t in pose['tracks']:
        for o in t['observations']:
            if kick-round(.5*pose['fps'])<=o['frame']<=kick+round(pose['fps']):
                byframe.setdefault(o['frame'],[]).append((t['track_id'],o))
    frames=sorted(byframe)
    if frames:
        cap.set(1,frames[0])
        for f in range(frames[0],frames[-1]+1):
            ok,img=cap.read()
            if not ok:raise ValueError(f'Histogram decode failed/{f}')
            for tid,o in byframe.get(f,[]):hist[(f,tid)]=torso_histogram(img,o)
    cap.release()
    lock=associate_locked_goalkeeper(pose,role,hist,kick,pre_sec=.5)
    # A declared composite association is separate from original tracker IDs.
    # Each original source ID and each continuity decision is retained in lock.
    virtual='role_locked_candidate';vp={k:pose[k] for k in ('clip_id','fps','frame_width','frame_height','video_path')}
    vp.update(tracks=[dict(track_id=virtual,observations=lock['observations'])])
    scratch=deepcopy(baseline);scratch['goal_identity_evidence']=role
    selected=_candidate_summary(dict(track_id=0,observations=lock['observations']),kick_frame=kick,fps=pose['fps'],
                frame_width=pose['frame_width'],frame_height=pose['frame_height'],pre_window_sec=.5,
                post_window_sec=1.,kicker_centre=None,is_selected_kicker=False)
    selected.update(source_track_ids=[virtual],association_source_track_ids=lock['source_track_ids'])
    scratch['selected_candidate']=selected
    e=pose_direction_evidence(scratch,vp)
    e.update(policy='role_locked_upright_to_lateral_pose_v1',association_policy=lock['policy'],
             supporting_source_track_ids={str(o['frame']):o['association_source_track_id'] for o in lock['observations'] if o['frame'] in e.get('supporting_frames',[])},
             identity_verified=False)
    return dict(clip_id=pose['clip_id'],role=role,association=lock,pose_direction_evidence=e,
                experimental_direction=e['direction'] if e['supported'] else 'unknown',
                baseline_direction=baseline['dive_direction'],height_unchanged=baseline['dive_height'],
                human_verified=False,accuracy_validated=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--clip',default='col_eng_p03_cuadrado_col11')
    p.add_argument('--fresh',action='store_true');p.add_argument('--all-roles',action='store_true')
    p.add_argument('--model',default='yolo11m_pose.pt');p.add_argument('--imgsz',type=int,default=960)
    p.add_argument('--rotations',nargs='+',type=int,choices=[0,90,270],default=[0])
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/goalkeeper_role_association_2026-10-11/experiment_v1')
    a=p.parse_args();cv2.setNumThreads(2);out=a.output_dir;out.mkdir(parents=True,exist_ok=True)
    protocol=dict(code_sha256={n:sha(ROOT/n) for n in ['src/goalkeeper_role_association.py','src/goalkeeper_pose_direction.py',
                             'scripts/run_goalkeeper_role_experiment.py','src/goal_geometry.py','src/person_appearance.py']},
                  selected_clip=a.clip,fresh=a.fresh,all_roles=a.all_roles,model=a.model,imgsz=a.imgsz,rotations=a.rotations,
                  no_identity_annotations_or_outcome_labels_read=True,identity_verified=False,accuracy_estimate=None)
    if (out/'protocol.json').exists():raise FileExistsError('Use a new experiment output directory')
    save(out/'protocol.json',protocol);baseroot=ROOT/'outputs/goalkeeper_pose_direction_2026-10-11/results_v3_final'
    summary=read(baseroot/'summary.json');roles=[];inputs=[]
    for row in summary['rows']:
        cid=row['clip_id']
        if cid!=a.clip and not a.all_roles:continue
        baseline=read(baseroot/row['result_file']);source=read(RUN/'goal_appearance_v2'/f'{cid}.json');pp=Path(source.get('source_pose_path','missing'))
        if not pp.is_file():
            roles.append(dict(clip_id=cid,status='abstain',reason='missing_cached_pose'));continue
        pose=read(pp);kick=baseline['kick_frame'];g=geometries_for(pose,kick)
        role=select_goalkeeper_role(pose,g,kick);save(out/'pre_roles'/f'{cid}.json',role)
        roles.append(dict(clip_id=cid,status=role['status'],new_seed=role['seed_track_id'],
                          old_seed=source.get('goal_identity_evidence',{}).get('seed_track_id'),
                          excluded_pre_person_ids=role['excluded_pre_person_ids'],reason=role['reason']))
        inputs.append(dict(clip_id=cid,pose_file=str(pp),pose_sha256=sha(pp),baseline_file=str(baseroot/row['result_file']),baseline_sha256=sha(baseroot/row['result_file'])))
        if cid!=a.clip:continue
        cached=compute(pose,kick,g,baseline);save(out/f'{cid}_cached_role_lock.json',cached)
        if a.fresh:
            fresh=fresh_pose(pose,kick,g,out/f'{cid}_fresh_pose.json',ROOT/'models/weights'/a.model,a.imgsz,a.rotations)
            fresh_result=compute(fresh,kick,g,baseline);save(out/f'{cid}_fresh_role_lock.json',fresh_result)
        print(json.dumps(dict(clip_id=cid,cached_direction=cached['experimental_direction'],
                    fresh_direction=fresh_result['experimental_direction'] if a.fresh else None),ensure_ascii=False),flush=True)
    save(out/'roles_summary.json',dict(rows=roles,clips=len(roles),identity_verified=False))
    save(out/'input_hashes.json',inputs)
    print(json.dumps(dict(clips=len(roles),role_candidates=sum(r['status']=='candidate_supported' for r in roles),
         changed_pre_seed=[r['clip_id'] for r in roles if r.get('new_seed') is not None and r.get('new_seed')!=r.get('old_seed')]),indent=2))


if __name__=='__main__':main()

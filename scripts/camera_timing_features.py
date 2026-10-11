"""Cached automatic camera evidence. No manual annotations or outcomes read."""
from pathlib import Path
import hashlib,json,sys
import cv2,numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.camera_timing_v2 import jersey_hist,hist_distance,goal_camera_hypotheses,pose_action_feature,playback_evidence
from src.goal_geometry import detect_goal_frame

VERSION='pose_kit_geometry_v1'
BASE=ROOT/'outputs/optimization_evening_2026-10-09'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model():
    # Keep the active interpreter's torch/numpy ahead of project additions.
    sys.path.append(str(ROOT/'.venv/Lib/site-packages'))
    import torch
    torch.set_num_threads(2)
    from ultralytics import YOLO
    return YOLO(str(ROOT/'models/weights/yolo11n_pose.pt'))


def infer(model,frame):
    result=model.predict(frame,imgsz=960,conf=.25,verbose=False,device='cpu')[0]
    if result.keypoints is None:return []
    boxes=result.boxes.xyxy.cpu().numpy();kp=result.keypoints.xy.cpu().numpy();kc=result.keypoints.conf.cpu().numpy()
    h,w=frame.shape[:2];rows=[]
    for b,k,c in zip(boxes,kp,kc):
        rows.append({'bbox':b.tolist(),'keypoints':k.tolist(),'keypoint_confidences':c.tolist(),
            'height_fraction':float((b[3]-b[1])/h),'area_fraction':float((b[2]-b[0])*(b[3]-b[1])/(w*h)),
            'face_visible_fraction':float(np.mean(c[:5]>.4)),
            'jersey_hist':jersey_hist(frame,b,k,c)})
    return rows


def get_frame(cap,fid):
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(fid));ok,frame=cap.read()
    if not ok:raise ValueError(f'Failed frame {fid}')
    return frame


def build_references(scan,cap,model):
    """Use prior automatic role candidates, with hashed provenance, not labels."""
    cid=scan['clip_id'];step_path=BASE/'steps_v4/classified'/f'{cid}.json'
    step=read(step_path);pose_path=Path(step['terminal_provenance']['selected_pose_path']);pose=read(pose_path)
    contact=step['contact'].get('contact_frame');track_id=step['contact'].get('contact_associated_track_id')
    track=next((t for t in pose['tracks'] if t['track_id']==track_id),None)
    if track is None:
        track=next((t for t in pose['tracks'] if t['track_id']==pose.get('selected_track_id')),None)
    rows=[o for o in (track or {}).get('observations',[]) if contact is not None and contact-35<=o['frame']<=contact-3]
    selected=[rows[i] for i in sorted(set([0,len(rows)//2,len(rows)-1]))] if rows else []
    kicker=[]
    for o in selected:
        frame=get_frame(cap,o['frame']);hist=jersey_hist(frame,o['bbox'],o['keypoints'],o['keypoint_confidences'])
        if hist is not None:kicker.append(hist)
    keeper=[];keeper_samples=[];gk_path=BASE/'goal_appearance_v2'/f'{cid}.json'
    gk=read(gk_path);pre=(gk.get('selected_candidate') or {}).get('pre_trajectory',[])
    declared_gk_pose=Path(gk['source_pose_path']);gk_pose_path=declared_gk_pose if declared_gk_pose.exists() else pose_path
    gk_pose=read(gk_pose_path)
    from src.goalkeeper_pipeline import _centre_and_scale
    for index in sorted(set([len(pre)//4,len(pre)//2,3*len(pre)//4])) if pre else []:
        t,x,y,scale=pre[index];fid=round(t*scan['fps']);frame=get_frame(cap,fid)
        # Upstream trajectories use HIP centers when available, not bbox centers.
        # Associate back to the exact retained source observation, avoiding
        # different detections from a new full-frame low-resolution pass.
        people=[o for tr in gk_pose['tracks'] for o in tr['observations'] if o['frame']==fid]
        ranked=sorted(people,key=lambda p:np.linalg.norm(np.array(_centre_and_scale(p)[:2])-[x,y]))
        if not ranked:continue
        p=ranked[0];dist=float(np.linalg.norm(np.array(_centre_and_scale(p)[:2])-[x,y]))
        hist=jersey_hist(frame,p['bbox'],p['keypoints'],p['keypoint_confidences'])
        if dist>max(2,.05*scale) or hist is None:continue
        keeper.append(hist);keeper_samples.append({'frame':fid,'bbox':p['bbox'],'association_distance_px':dist,'center_definition':'upstream hips or bbox fallback'})
    return {'kicker_histograms':kicker,'keeper_histograms':keeper,'contact_frame_candidate':contact,
            'kicker_track_id_candidate':track.get('track_id') if track else None,
            'kicker_reference_frames':[o['frame'] for o in selected],'keeper_reference_samples':keeper_samples,
            'human_verified':False,'identity_method':'upstream automatic role + torso HSV appearance; same-kit people unresolved',
            'reference_policy':'exact_source_observation_hip_association_v2',
            'declared_keeper_pose_available':declared_gk_pose.exists(),
            'reference_warning':None if declared_gk_pose.exists() else 'Declared keeper pose missing; exact-match search in retained kicker pose only; may abstain.',
            'upstream_sha256':{str(p.relative_to(ROOT)):digest(p) for p in (step_path,pose_path,gk_path,gk_pose_path)}}


def assign_roles(people,refs):
    for p in people:
        scores={role:min((hist_distance(p['jersey_hist'],h) for h in refs[f'{role}_histograms'] if p['jersey_hist'] is not None),default=1.) for role in ('kicker','keeper')}
        best=min(scores,key=scores.get);other='keeper' if best=='kicker' else 'kicker'
        p['appearance_distances']=scores
        p['role']=best+'_candidate' if scores[best]<.50 and scores[other]-scores[best]>.12 else 'unknown'
        p['identity_verified']=False
        p['identity_warning']='Kit appearance does not distinguish teammates or prove cross-shot identity.'
    return people


def extract_features(scan,out,model,resume=False):
    path=out/'features'/f"{scan['clip_id']}.json"
    signature={'version':VERSION,'source_sha256':scan['source_sha256'],'scan_sha256':digest(out/'scan'/f"{scan['clip_id']}.json"),
               'weights_sha256':digest(ROOT/'models/weights/yolo11n_pose.pt'),'feature_script_sha256':digest(__file__)}
    if path.exists():
        if not resume:raise FileExistsError(path)
        result=read(path)
        if result['cache_signature']!=signature:raise ValueError('Feature cache signature mismatch; use a new output directory')
        for name,value in result['references']['upstream_sha256'].items():
            if digest(ROOT/name)!=value:raise ValueError('Upstream candidate changed')
        return result
    cap=cv2.VideoCapture(scan['video_path']);refs=build_references(scan,cap,model);segments=[]
    for segment in scan['segments']:
        samples=[]
        for fid in segment['sample_frame_ids']:
            frame=get_frame(cap,fid);people=assign_roles(infer(model,frame),refs);goal=detect_goal_frame(frame)
            selected=goal.get('selected');calibration=goal_camera_hypotheses(selected['corners'],scan['width'],scan['height']) if selected else {'status':'no_supported_goal'}
            samples.append({'frame':fid,'width':scan['width'],'height':scan['height'],'people':people,'goal':selected,
                            'goal_proposal_status':goal['status'],'goal_camera':calibration})
        segments.append({'segment_id':segment['segment_id'],'samples':samples})
    cap.release()
    return {'clip_id':scan['clip_id'],'cache_signature':signature,'references':refs,'segments':segments,
            'human_verified':False,'accuracy_validated':False,'source_outcome_labels_read':False}


def extract_action(scan,features,model,out,resume=False):
    """Bounded 8-Hz pose alignment; unknown is normal when no matching action."""
    from src.camera_timing_v2 import relabel_cached_features,assign_kit_roles
    refs_path=out/'references_v2'/f"{scan['clip_id']}.json"
    if refs_path.exists():features={**features,'references':read(refs_path)}
    context_path=out/'context_v2'/f"{scan['clip_id']}.json"
    if context_path.exists():
        from camera_timing_context import merge
        features=merge(features,read(context_path))
    features=relabel_cached_features(features)
    path=out/'action_v3'/f"{scan['clip_id']}.json"
    signature={'features_sha256':digest(out/'features'/f"{scan['clip_id']}.json"),'version':VERSION,
               'feature_script_sha256':digest(__file__),'classifier_sha256':digest(ROOT/'src/camera_timing_v2.py'),
               'reference_update_sha256':digest(refs_path) if refs_path.exists() else None,
               'context_update_sha256':digest(context_path) if context_path.exists() else None}
    if path.exists():
        if not resume:raise FileExistsError(path)
        result=read(path)
        if result['cache_signature']!=signature:raise ValueError('Action cache mismatch')
        return result
    from src.camera_timing_v2 import classify_view
    cf=features['references']['contact_frame_candidate']
    ref=next((s for s in scan['segments'] if cf is not None and s['start_frame']<=cf<s['end_frame_exclusive']),None)
    empty={'clip_id':scan['clip_id'],'cache_signature':signature,'reference_segment_id':ref['segment_id'] if ref else None,
           'sequences':{},'evidence':{},'human_verified':False}
    if ref is None:return empty
    # Only spend dense inference on potential replay views. Other shots retain
    # unknown playback, not normal-speed labels.
    choices=[]
    for f,s in zip(features['segments'],scan['segments']):
        view=classify_view(f['samples'])['view_candidate']
        if s['segment_id']!=ref['segment_id'] and s['start_frame']>cf and view in ('kicker_closeup','back_of_goalkeeper','back_of_kicker','angle_45_right','other_oblique_left','oblique_side_unresolved'):
            choices.append(s)
    if not choices:return empty
    cap=cv2.VideoCapture(scan['video_path']);sequences={};fps=scan['fps'];refs=features['references'];reused=[]
    # Reuse measurements, never old decisions. These inputs and the pose feature
    # definition are unchanged; only the spatial vote rule was revised in v3.
    prior_path=out/'action_v2'/f"{scan['clip_id']}.json";prior={}
    if prior_path.exists():
        candidate=read(prior_path)
        keys=('features_sha256','reference_update_sha256','context_update_sha256','version')
        if all(candidate['cache_signature'].get(k)==signature.get(k) for k in keys):prior=candidate.get('sequences',{})
    for s in [ref]+choices:
        if str(s['segment_id']) in prior:
            sequences[str(s['segment_id'])]=prior[str(s['segment_id'])];reused.append(s['segment_id']);continue
        if s==ref:a=max(s['start_frame'],cf-round(1.4*fps));b=min(s['end_frame_exclusive'],cf+round(1.0*fps))
        else:a=s['start_frame']+2;b=min(s['end_frame_exclusive']-2,a+round(8*fps))
        seq={'kicker':[],'keeper':[]}
        for fid in range(a,b,max(1,round(fps/8))):
            frame=get_frame(cap,fid);people=assign_kit_roles(infer(model,frame),refs)
            for role in seq:
                candidates=[p for p in people if p['role']==role+'_candidate']
                # Multiple same-kit people are not reliably identifiable.
                if len(candidates)!=1:continue
                p=candidates[0];value=pose_action_feature(p)
                if value is not None:seq[role].append({'frame':fid,'time_sec':fid/fps,'feature':value})
        sequences[str(s['segment_id'])]=seq
    cap.release();evidence={};reference=sequences[str(ref['segment_id'])]
    for s in choices:
        sid=str(s['segment_id']);role_results={}
        for role in ('kicker','keeper'):
            a=reference[role];b=sequences[sid][role]
            result=playback_evidence([r['time_sec'] for r in a],[r['feature'] for r in a],[r['time_sec'] for r in b],[r['feature'] for r in b])
            result.update(role_candidate=role,identity_verified=False,viewpoint_invariant=False,
                          alignment_warning='Same-kit identity and viewpoint changes may cause mismatches; review required.')
            role_results[role]=result
        supported=[r for r in role_results.values() if r['status']=='slow_motion_candidate']
        evidence[sid]={'status':'slow_motion_candidate' if supported else 'unknown','role_results':role_results,
                       'reference_segment_id':ref['segment_id'],'reference_speed_verified':False,'review_required':True,
                       'reason':'relative_action_time_stretch' if supported else 'no_reliable_same_action_slowdown_evidence'}
    return {**empty,'sequences':sequences,'evidence':evidence,'reused_sequence_segment_ids':reused,
            'reused_sequence_source_sha256':digest(prior_path) if reused else None,
            'reused_sequence_policy':'v2 same input hashes, kit policy and six-angle definition; decisions recomputed'}

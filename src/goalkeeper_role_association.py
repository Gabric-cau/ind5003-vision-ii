"""Pre-kick goal-line role selection and explicit other-person exclusions.

Weights are engineering scores, not probabilities. Hands are wrist keypoints,
not glove recognition. Goal-line distances are image-space, not metric ground
distances. No player names, outcomes, human role labels or post-kick actions
are used to select the pre-kick role.
"""
from collections import defaultdict
import math
from statistics import median
import numpy as np


WEIGHTS={'foot_goal_line':.60,'body_goal_mouth':.20,'wrist_ready_pose':.10,'pre_visibility':.10}


def deduplicate_pose_people(detections):
    """Fuse multi-view duplicates only when body keypoints also coincide."""
    kept=[]
    for d in sorted(detections,key=lambda o:-float(o.get('box_confidence',0))):
        duplicate=False
        for other in kept:
            a=np.asarray(d['bbox']);b=np.asarray(other['bbox'])
            intersection=float(np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))))
            area_a=float(np.prod(a[2:]-a[:2]));area_b=float(np.prod(b[2:]-b[:2]))
            iou=intersection/max(1,area_a+area_b-intersection)
            if iou<.45:continue
            p=_points(d,(5,6,11,12,15,16));q=_points(other,(5,6,11,12,15,16))
            if p is None or q is None:continue
            # Augmented detectors can swap left/right labels. Pair means retain
            # the body geometry without mistaking label swaps for a new person.
            p=p.reshape(3,2,2).mean(axis=1);q=q.reshape(3,2,2).mean(axis=1)
            scale=max(8,math.sqrt(max(area_a,area_b)))
            if np.sqrt(np.mean(np.sum((p-q)**2,axis=1)))/scale<.12:
                duplicate=True;break
        if not duplicate:kept.append(d)
    return kept


def unrotate_pose(boxes,keypoints,crop_width,crop_height,angle):
    """Map augmented detections into the unrotated crop; no invented pixels."""
    boxes=np.asarray(boxes,float).copy();kp=np.asarray(keypoints,float).copy()
    if angle==0:return boxes,kp
    if angle not in (90,270):raise ValueError('Unsupported rotation')
    pts=kp.copy();corners=np.stack([boxes[:,[0,1]],boxes[:,[2,1]],boxes[:,[2,3]],boxes[:,[0,3]]],axis=1)
    if angle==90:
        kp[...,0]=pts[...,1];kp[...,1]=crop_height-1-pts[...,0]
        old=corners.copy();corners[...,0]=old[...,1];corners[...,1]=crop_height-old[...,0]
    else:
        kp[...,0]=crop_width-1-pts[...,1];kp[...,1]=pts[...,0]
        old=corners.copy();corners[...,0]=crop_width-old[...,1];corners[...,1]=old[...,0]
    boxes[:,:2]=corners.min(axis=1);boxes[:,2:]=corners.max(axis=1)
    return boxes,kp


def _points(o,indices,min_conf=.40):
    try:
        pts=np.asarray(o['keypoints'],float);conf=np.asarray(o['keypoint_confidences'],float)
        if pts.shape!=(17,2) or conf.shape!=(17,) or not np.isfinite(pts[list(indices)]).all():return None
        if any(not math.isfinite(conf[j]) or conf[j]<min_conf for j in indices):return None
        return pts[list(indices)]
    except (KeyError,ValueError,TypeError):return None


def line_features(o,corners):
    corners=np.asarray(corners,float)
    if corners.shape!=(4,2) or not np.isfinite(corners).all():return None
    bl,br=corners[3],corners[2];line=br-bl;length=float(np.linalg.norm(line))
    gh=float(np.mean([np.linalg.norm(corners[0]-bl),np.linalg.norm(corners[1]-br)]))
    if length<10 or gh<10:return None
    ankles=_points(o,(15,16));torso=_points(o,(5,6,11,12))
    if ankles is None or torso is None:return None
    b=o['bbox'];body_h=b[3]-b[1]
    projections=((ankles-bl)@line/(length*length))
    distances=np.abs(line[0]*(ankles[:,1]-bl[1])-line[1]*(ankles[:,0]-bl[0]))/length/gh
    # Either planted foot may be on the line while the other takes a small step.
    distance=float(np.min(distances));projection=float(np.mean(projections))
    shoulder=torso[:2].mean(0);hip=torso[2:].mean(0)
    body_middle=(shoulder+hip)/2
    import cv2
    inside=cv2.pointPolygonTest(corners.astype(np.float32),tuple(map(float,body_middle)),False)>=0
    wrists=_points(o,(9,10));hand_score=None
    if wrists is not None:
        # Weak readiness cue: wrists around the waist/chest, potentially spread.
        body_length=max(1,float(np.linalg.norm(shoulder-hip)))
        vertical=(wrists[:,1]>=shoulder[1]-.5*body_length)&(wrists[:,1]<=hip[1]+body_length)
        spread=np.clip(np.abs(wrists[:,0]-hip[0])/body_length,0,1)
        hand_score=float(np.mean(vertical*(.5+.5*spread)))
    reasons=[]
    if not -.05<=projection<=1.05:reasons.append('feet_outside_goal_span')
    if distance>.28:reasons.append('feet_far_from_goal_line')
    if not .30<=body_h/gh<=1.60:reasons.append('body_scale_inconsistent_with_goal')
    if not inside:reasons.append('torso_outside_goal_mouth')
    return dict(foot_distance_goal_height=distance,ankle_distances_goal_height=distances.tolist(),
                goal_line_projection=projection,body_goal_mouth=float(inside),
                wrist_ready_pose=hand_score,body_height_goal_height=body_h/gh,
                reasons=reasons,eligible=not reasons)


def _same_person(a,b):
    """Deduplicate the same pose, never just overlapping nearby boxes."""
    aa={o['frame']:o for o in a['observations']};bb={o['frame']:o for o in b['observations']}
    distances=[]
    for f in aa.keys()&bb.keys():
        p=_points(aa[f],(5,6,11,12,15,16));q=_points(bb[f],(5,6,11,12,15,16))
        if p is None or q is None:continue
        scale=max(8,aa[f]['bbox'][3]-aa[f]['bbox'][1],bb[f]['bbox'][3]-bb[f]['bbox'][1])
        p=p.reshape(3,2,2).mean(axis=1);q=q.reshape(3,2,2).mean(axis=1)
        distances.append(float(np.sqrt(np.mean(np.sum((p-q)**2,axis=1)))/scale))
    return len(distances)>=2 and median(distances)<.10


def select_goalkeeper_role(pose,geometries,kick_frame,*,min_samples=3,margin=.12):
    fps=float(pose['fps'])
    if not math.isfinite(fps) or fps<=0:raise ValueError('Invalid fps')
    sample_frames=sorted(f for f in geometries if kick_frame-round(.5*fps)<=f<kick_frame)
    out=dict(policy='pre_kick_ankle_goal_line_weighted_role_v1',status='abstain',seed_track_id=None,
             candidates=[],excluded_pre_person_ids=[],weights=WEIGHTS.copy(),
             geometry_sample_frames=sample_frames,human_verified=False,identity_verified=False,
             calibrated_probability=False,reason='no_supported_goal_geometry')
    if not sample_frames:return out
    tracks=pose.get('tracks',[]);candidates=[]
    for t in tracks:
        observations={o['frame']:o for o in t.get('observations',[])};samples=[]
        for f in sample_frames:
            o=observations.get(f);g=geometries[f]
            if o is None or not g.get('selected'):continue
            feature=line_features(o,g['selected']['corners'])
            if feature:samples.append(dict(frame=f,**feature))
        if len(samples)<min_samples:continue
        good=[s for s in samples if s['eligible']]
        distance=median(s['foot_distance_goal_height'] for s in samples)
        components={'foot_goal_line':math.exp(-((distance/.15)**2)),
                    'body_goal_mouth':sum(s['body_goal_mouth'] for s in samples)/len(samples),
                    'pre_visibility':min(1,len(samples)/len(sample_frames))}
        wrists=[s['wrist_ready_pose'] for s in samples if s['wrist_ready_pose'] is not None]
        if wrists:components['wrist_ready_pose']=median(wrists)
        # Missing wrists give no bonus and cannot defeat strong feet/goal evidence.
        score=sum(WEIGHTS[k]*v for k,v in components.items())
        reasons=sorted({r for s in samples for r in s['reasons']})
        c=dict(track_id=t['track_id'],score=score,eligible=len(good)>=min_samples and len(good)/len(samples)>=.6,
               components=components,samples=samples,median_foot_distance_goal_height=distance,
               rejection_reasons=reasons,same_person_source_ids=[t['track_id']])
        candidates.append(c)
    candidates.sort(key=lambda c:(-c['score'],str(c['track_id'])))
    eligible=[c for c in candidates if c['eligible']]
    distinct=[];by_id={t['track_id']:t for t in tracks}
    for c in eligible:
        same=next((d for d in distinct if _same_person(by_id[c['track_id']],by_id[d['track_id']])),None)
        if same:same['same_person_source_ids'].append(c['track_id'])
        else:distinct.append(c)
    out.update(candidates=candidates,distinct_goal_person_candidates=distinct)
    if not distinct:out['reason']='no_person_with_supported_pre_kick_feet_on_goal_line';return out
    top=distinct[0];gap=top['score']-distinct[1]['score'] if len(distinct)>1 else None
    out['runner_up_margin']=gap
    if gap is not None and gap<margin:out['reason']='ambiguous_goal_line_role';return out
    own=set(top['same_person_source_ids'])
    excluded=[c['track_id'] for c in candidates if c['track_id'] not in own
              and not any(_same_person(by_id[c['track_id']],by_id[x]) for x in own)]
    out.update(status='candidate_supported',seed_track_id=top['track_id'],
               same_person_seed_ids=sorted(own),excluded_pre_person_ids=excluded,
               reason='multi_frame_feet_on_goal_line_with_auxiliary_pose',ambiguous_identity=False)
    return out


def associate_locked_goalkeeper(pose,role,histograms,kick_frame,*,pre_sec=1.5,post_sec=1.,max_gap_sec=.20):
    """Follow the pre-kick selected person; excluded nearby people cannot take over."""
    from src.person_appearance import similarity
    from src.goalkeeper_pipeline import _centre_and_scale
    out=dict(policy='pre_role_excluded_person_lock_v1',observations=[],decisions=[],status='abstain',
             source_track_ids=[],identity_verified=False,max_gap_sec=max_gap_sec)
    if role.get('seed_track_id') is None:return out
    fps=float(pose['fps']);seed=role['seed_track_id'];excluded=set(role['excluded_pre_person_ids'])
    own=set(role.get('same_person_seed_ids',[seed]));pool=defaultdict(list)
    for t in pose['tracks']:
        for o in t['observations']:
            if kick_frame-round(pre_sec*fps)<=o['frame']<=kick_frame+round(post_sec*fps):
                pool[o['frame']].append((t['track_id'],o))
    pre=[]
    for f,rows in pool.items():
        if f>kick_frame:continue
        choices=[(tid,o) for tid,o in rows if tid in own]
        if choices:
            tid,o=max(choices,key=lambda x:(x[0]==seed,float(x[1].get('box_confidence',0))))
            pre.append((f,tid,o))
    pre.sort(key=lambda r:r[0])
    if not pre:return out
    anchor=pre[-1];anchor_frame=anchor[0]
    if kick_frame-anchor_frame>round(.10*fps):out['reason']='missing_near_kick_seed';return out
    templates=[histograms[(f,tid)] for f,tid,o in pre[-10:] if histograms.get((f,tid)) is not None]
    if not templates:out['reason']='missing_pre_kick_appearance';return out
    template=np.mean(templates,axis=0);template/=max(1e-9,float(template.sum()))
    pre_obs=[dict(o,association_source_track_id=tid) for f,tid,o in pre]
    previous=anchor[2];previous_id=anchor[1];last_frame=anchor_frame;chosen=pre_obs[:]
    base_scale=median(_centre_and_scale(o)[2] for _,_,o in pre[-10:])
    for f in range(anchor_frame+1,kick_frame+round(post_sec*fps)+1):
        if f-last_frame>round(max_gap_sec*fps):out['stop_reason']='long_observation_gap';break
        ranked=[];rejected=[];prev=np.asarray(_centre_and_scale(previous)[:2]);dt=(f-last_frame)/fps
        for tid,o in pool.get(f,[]):
            if tid in excluded:
                rejected.append(dict(source_track_id=tid,reason='excluded_pre_kick_other_person'));continue
            body=_points(o,(5,6,11,12),min_conf=.25)
            if body is None:
                rejected.append(dict(source_track_id=tid,reason='weak_body_keypoints'));continue
            xy=np.asarray(_centre_and_scale(o)[:2]);distance=float(np.linalg.norm(xy-prev)/base_scale)
            if distance>.35+3*dt:
                rejected.append(dict(source_track_id=tid,reason='disconnected_body_motion'));continue
            sim=similarity(template,histograms.get((f,tid)))
            if sim is None or sim<.35:
                rejected.append(dict(source_track_id=tid,reason='appearance_inconsistent',similarity=sim));continue
            # Body HEIGHT collapses during a dive; do not penalise that as identity
            # change. Camera-independent limb/body extent is considered instead.
            cost=.60*distance+.40*(1-sim)-(.08 if tid==previous_id else 0)
            ranked.append((cost,tid,o,sim,distance))
        ranked.sort(key=lambda r:r[0]);decision=dict(frame=f,rejected=rejected)
        if ranked:
            # Distinct equally plausible people are not silently merged.
            if len(ranked)>1 and ranked[1][0]-ranked[0][0]<.035:
                a,b=ranked[0],ranked[1]
                p=_points(a[2],(5,6,11,12));q=_points(b[2],(5,6,11,12))
                if p is None or q is None or np.linalg.norm(p-q)/base_scale>.2:
                    decision['reason']='ambiguous_post_kick_person';out['decisions'].append(decision);continue
            cost,tid,o,sim,distance=ranked[0]
            decision.update(selected_source_track_id=tid,appearance_similarity=sim,motion_distance_body=distance,
                            id_switch=tid!=previous_id)
            chosen.append(dict(o,association_source_track_id=tid));previous=o;previous_id=tid;last_frame=f
        else:decision['reason']='no_role_consistent_person_detection'
        out['decisions'].append(decision)
    out.update(observations=chosen,status='candidate_track' if chosen else 'abstain',
               source_track_ids=sorted({o['association_source_track_id'] for o in chosen}),
               last_observed_frame=last_frame,excluded_pre_person_ids=sorted(excluded),
               appearance_threshold=.35,association_cost_weights={'motion':.60,'appearance':.40},
               matched_reference_count=len(templates),pre_body_scale_px=base_scale)
    return out

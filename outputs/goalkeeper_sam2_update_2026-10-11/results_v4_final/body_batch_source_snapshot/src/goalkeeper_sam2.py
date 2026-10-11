"""SAM2 mask QC and anatomical direction; engineering evidence, not accuracy.

SAM2 supplies an unsigned visible silhouette. Direction sign must come from
visible head/shoulder/hip pose, never PCA alone or mask centroid displacement.
"""
import cv2
import numpy as np


def mask_geometry(mask):
    y,x=np.nonzero(mask)
    if len(x)<30:return None
    pts=np.column_stack([x,y]).astype(float);center=pts.mean(0)
    vals,vec=np.linalg.eigh(np.cov(pts.T));axis=vec[:,-1]
    projection=(pts-center)@axis
    return dict(center=center.tolist(),axis=axis.tolist(),
                elongation=float(np.sqrt(vals[-1]/max(vals[0],1))),
                half_length=float((np.percentile(projection,98)-np.percentile(projection,2))/2),
                pixels=len(x),bbox=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)])


def points_inside(mask, points, radius=3):
    pts=np.asarray(points,float)
    allowed=cv2.dilate(mask.astype(np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8))
    out=np.zeros(len(pts),bool);finite=np.isfinite(pts).all(1)
    xy=np.zeros((len(pts),2),int);xy[finite]=np.rint(pts[finite]).astype(int)
    valid=finite&(xy[:,0]>=0)&(xy[:,1]>=0)&(xy[:,0]<mask.shape[1])&(xy[:,1]<mask.shape[0])
    out[valid]=allowed[xy[valid,1],xy[valid,0]]>0
    return out


def mask_qc(mask, previous, scale, initial_area, excluded_torso=None):
    g=mask_geometry(mask)
    if g is None:return dict(valid=False,reason='empty_or_tiny_mask')
    if not .08<=g['pixels']/max(initial_area,1)<=4:
        return dict(valid=False,reason='mask_area_discontinuity',geometry=g)
    if previous is not None:
        prev=mask_geometry(previous)
        if prev and np.linalg.norm(np.array(g['center'])-prev['center'])>max(8,.5*scale):
            return dict(valid=False,reason='mask_coordinate_jump',geometry=g)
    if excluded_torso is not None and excluded_torso.any():
        overlap=float(np.count_nonzero(mask & excluded_torso)/np.count_nonzero(excluded_torso))
        if overlap>.20:return dict(valid=False,reason='mask_on_excluded_person_torso',geometry=g,excluded_torso_overlap=overlap)
    return dict(valid=True,reason='candidate_mask_only',geometry=g,identity_verified=False)


def anatomy_sample(observation, mask, offset=(0,0)):
    """Sign from COCO head and torso, shape corroboration from SAM mask."""
    g=mask_geometry(mask)
    if g is None:return None
    try:
        kp=np.asarray(observation['keypoints'],float)-offset
        conf=np.asarray(observation['keypoint_confidences'],float)
        if kp.shape!=(17,2) or conf.shape!=(17,):return None
        inside=points_inside(mask,kp)
        body=[5,6,11,12]
        if not all(np.isfinite(conf[j]) and conf[j]>=.5 and inside[j] for j in body):return None
        # Nose is mandatory: averaging weak/occluded inferred eyes cannot create a head.
        if not np.isfinite(conf[0]) or conf[0]<.5 or not inside[0]:return None
        shoulder=kp[[5,6]].mean(0);hip=kp[[11,12]].mean(0);torso=shoulder-hip
        if not np.isfinite(torso).all() or np.linalg.norm(torso)<4:return None
        head=kp[0];center=np.asarray(g['center']);axis=np.asarray(g['axis']);head_vec=head-center
        signed_axis=axis if head_vec@axis>=0 else -axis
        head_projection=float(abs(head_vec@axis)/max(g['half_length'],1))
        lateral=(abs(torso[0])>abs(torso[1]) and abs(axis[0])>abs(axis[1])
                 and torso[0]*signed_axis[0]>0 and head_vec[0]*torso[0]>0
                 and g['elongation']>=1.8 and head_projection>=.25)
        upright=(abs(torso[0])<abs(torso[1]) and torso[1]<0)
        return dict(frame=int(observation['frame']),torso_vector=torso.tolist(),
            head_point=head.tolist(),head_relative_to_mask_center=head_vec.tolist(),
            mask_axis_signed_by_visible_head=signed_axis.tolist(),head_endpoint_fraction=head_projection,
            mask_elongation=g['elongation'],upright=bool(upright),
            sign=(1 if torso[0]>0 else -1) if lateral else 0,
            min_anatomical_confidence=float(conf[[0,5,6,11,12]].min()),
            source_pose=observation.get('source_pose','unknown'))
    except (KeyError,TypeError,ValueError):return None


def sam2_direction_evidence(samples, qc, kick, fps):
    out=dict(policy='sam2_visible_head_torso_transition_v1',supported=False,direction='unknown',
        reasons=[],identity_verified=False,accuracy_validated=False,
        coordinate_system='source_image_x_right',uses_mask_centroid_direction=False,
        uses_unsigned_PCA_direction=False,anatomical_confidence_min=.5,
        min_sustained_samples=3, min_sustained_span_sec=.06)
    def reject(reason):out['reasons'].append(reason);return out
    pre=[s for s in samples if kick-round(.5*fps)<=s['frame']<kick and s['upright']]
    if len(pre)<3 or kick-pre[-1]['frame']>round(.10*fps):return reject('missing_near_kick_upright_anatomy')
    if any(b['frame']-a['frame']>round(.20*fps) for a,b in zip(pre[-5:],pre[-5:][1:])):
        return reject('disconnected_upright_reference')
    window=sorted(f for f in qc if kick<=f<=kick+round(fps))
    if not window or window[0]-kick>round(.10*fps):return reject('missing_mask_onset')
    good=[];last=None
    for f in window:
        if not qc[f]['valid']:
            out['mask_prefix_stop_reason']=qc[f]['reason'];break
        if last is not None and f-last>round(.20*fps):break
        good.append(f);last=f
    byframe={s['frame']:s for s in samples};runs=[];run=[]
    for f in good:
        s=byframe.get(f);sign=s['sign'] if s else 0
        if not sign or run and (sign!=run[-1]['sign'] or f-run[-1]['frame']>round(.10*fps)):
            if run:runs.append(run)
            run=[]
        if sign:run.append(s)
    if run:runs.append(run)
    out.update(pre_reference_frames=[s['frame'] for s in pre[-5:]],observed_mask_prefix_frames=good,
               anatomical_samples=samples)
    supported=[r for r in runs if len(r)>=3 and (r[-1]['frame']-r[0]['frame'])/fps>=.06-1e-9]
    if not supported:return reject('no_sustained_visible_head_torso_transition')
    if len({r[0]['sign'] for r in supported})>1:return reject('conflicting_anatomical_directions')
    run=supported[0];out.update(supported=True,direction='image_right' if run[0]['sign']>0 else 'image_left',
        supporting_frames=[s['frame'] for s in run],transition_frame=run[0]['frame'])
    return out

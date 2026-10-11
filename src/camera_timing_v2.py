"""Evidence-separated camera timing candidates, never human ground truth.

Shot boundaries, camera view, replay identity, and playback speed are distinct.
The legacy grass/edge classifier is deliberately not used for semantic labels.
"""
from collections import Counter
from copy import deepcopy
import math
import cv2
import numpy as np

TASK_CLASSES=('back_of_kicker','angle_45_right','slowmo_kicker_closeup','slowmo_gk_back')
SCAN_VERSION='adaptive_mad_histogram_v3_soft_transition'


def choose_boundaries(metrics,fps):
    """Local-change normalization + histogram corroboration + peak suppression.

    Project-specific heuristic, inspired by adaptive shot-change detectors,
    not copied from PySceneDetect. Never claims boundary accuracy without labels.
    """
    candidates=[];radius=max(4,round(fps*.25))
    for i,m in enumerate(metrics):
        if i==0:continue
        neighbours=[v['mad'] for j,v in enumerate(metrics[max(1,i-radius):i+radius+1],max(1,i-radius))
                    if abs(j-i)>1]
        baseline=max(2.,float(np.median(neighbours))) if neighbours else 2.
        ratio=m['mad']/baseline
        strong=(m['mad']>18 and ratio>3 and m['hist_distance']>.18)
        hard=(m['mad']>35 and m['hist_distance']>.45)
        # A broadcast dissolve can spread the pixel change over several frames.
        # Require corroborating histogram change on an adjacent frame, not a
        # lone low-amplitude change. Its timestamp is approximate, not a hard cut.
        soft=(m['mad']>4 and m['hist_distance']>.30 and
              any(metrics[j]['hist_distance']>.25 for j in (i-1,i+1) if 0<=j<len(metrics)))
        if strong or hard or soft:
            candidates.append({'frame':i,'mad':m['mad'],'hist_distance':m['hist_distance'],
                               'adaptive_ratio':ratio,'score':ratio+3*m['hist_distance'],
                               'transition_kind':'hard_candidate' if strong or hard else 'gradual_candidate',
                               'timing_uncertainty_sec':1/fps if strong or hard else .25})
    selected=[];gap=max(2,round(.24*fps))
    for c in sorted(candidates,key=lambda c:-c['score']):
        if c['frame']<gap or c['frame']>=len(metrics)-gap:continue
        if all(abs(c['frame']-v['frame'])>=gap for v in selected):selected.append(c)
    return sorted(selected,key=lambda c:c['frame'])


def scan_video(path):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise ValueError(f'Cannot open {path}')
    fps=float(cap.get(cv2.CAP_PROP_FPS));expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps<=0:raise ValueError('Missing valid source FPS')
    width,height=int(cap.get(3)),int(cap.get(4));metrics=[];previous=None;previous_hist=None
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            small=cv2.resize(frame,(192,108),interpolation=cv2.INTER_AREA)
            gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
            hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
            hist=cv2.calcHist([hsv],[0,1],None,[18,8],[0,180,0,256]);hist/=max(1,hist.sum())
            mad=float(cv2.absdiff(gray,previous).mean()) if previous is not None else 0.
            hd=float(cv2.compareHist(hist,previous_hist,cv2.HISTCMP_BHATTACHARYYA)) if previous_hist is not None else 0.
            metrics.append({'mad':round(mad,4),'hist_distance':round(hd,4),'near_duplicate':bool(previous is not None and mad<.35)})
            previous=gray;previous_hist=hist
    finally:cap.release()
    if len(metrics)!=expected or expected==0:raise ValueError(f'Incomplete decode: {len(metrics)}/{expected}')
    peaks=choose_boundaries(metrics,fps);bounds=[0]+[p['frame'] for p in peaks]+[expected]
    segments=[]
    for i,(a,b) in enumerate(zip(bounds,bounds[1:])):
        # Interior samples avoid transition overlays at exact cut frames.
        sample=sorted(set(min(b-1,a+int((b-a)*q)) for q in (.15,.50,.85)))
        dup=sum(m['near_duplicate'] for m in metrics[a+1:b])/max(1,b-a-1)
        segments.append({'segment_id':i,'start_frame':a,'end_frame_exclusive':b,
            'start_time_sec':a/fps,'end_time_sec':b/fps,'sample_frame_ids':sample,
            'duplicate_fraction':round(dup,4),'boundary_verified':False})
    return {'fps':fps,'width':width,'height':height,'frame_count':expected,'decoded_frames':expected,
            'segments':segments,'cut_candidates':peaks,'frame_metrics':metrics,
            'boundary_method':SCAN_VERSION,'timeline_coverage':1.,'accuracy_validated':False}


def jersey_hist(frame,bbox,keypoints=None,confidence=None):
    """HSV torso appearance, not identity recognition."""
    x0,y0,x1,y1=map(float,bbox);w,h=x1-x0,y1-y0
    if keypoints is not None and confidence is not None and min(confidence[j] for j in (5,6,11,12))>.4:
        points=np.array([keypoints[j] for j in (5,6,11,12)],float)
        x0,x1=float(points[:,0].min()),float(points[:,0].max())
        y0,y1=float(points[:2,1].mean()),float(points[2:,1].mean());w,h=x1-x0,y1-y0
        x0+=.2*w;x1-=.2*w;y0+=.15*h;y1-=.15*h
    else:x0+=.28*w;x1-=.28*w;y0+=.22*h;y1=y0+.33*h
    patch=frame[max(0,round(y0)):min(frame.shape[0],round(y1)),max(0,round(x0)):min(frame.shape[1],round(x1))]
    if patch.size<60:return None
    hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
    hist=cv2.calcHist([hsv],[0,1,2],None,[12,3,3],[0,180,0,256,0,256]).astype(float).ravel()
    return (hist/max(1,hist.sum())).tolist()


def hist_distance(a,b):
    if a is None or b is None:return None
    return float(np.sqrt(max(0,1-np.sqrt(np.asarray(a)*np.asarray(b)).sum())))


ROLE_POLICY='circular_hue_brightness_marginal_v2'


def kit_hist_distance(a,b):
    """Smooth circular hue; marginalize exposure for saturated kit colours.

    Red spans both HSV hue endpoints. Black/white need brightness but not hue.
    This is an appearance cue, not face or jersey-number identification.
    """
    if a is None or b is None:return 1.
    def descriptor(value):
        h=np.asarray(value,float).reshape(12,3,3)
        colour=h[:,1:,:].sum(axis=(1,2))
        colour=.5*colour+.25*np.roll(colour,1)+.25*np.roll(colour,-1)
        grey=h[:,0,:].sum(axis=0)
        d=np.r_[colour,grey];return d/max(1e-9,d.sum())
    return hist_distance(descriptor(a),descriptor(b))


def assign_kit_roles(people,refs):
    for p in people:
        scores={role:min((kit_hist_distance(p.get('jersey_hist'),h) for h in refs[f'{role}_histograms']),default=1.) for role in ('kicker','keeper')}
        best=min(scores,key=scores.get);other='keeper' if best=='kicker' else 'kicker'
        p['appearance_distances']=scores
        p['role']=best+'_candidate' if scores[best]<.52 and scores[other]-scores[best]>.12 else 'unknown'
        p['role_policy']=ROLE_POLICY;p['identity_verified']=False
        p['identity_warning']='Kit appearance does not distinguish teammates or prove cross-shot identity.'
    return people


def relabel_cached_features(features):
    result=deepcopy(features)
    for s in result['segments']:
        for frame in s['samples']:assign_kit_roles(frame['people'],result['references'])
    result['role_reanalysis_policy']=ROLE_POLICY
    return result


def goal_camera_hypotheses(corners,width,height):
    """Sensitivity probe, NOT validated camera calibration.

    World X is kicker-right facing goal, Y points from penalty spot into goal,
    Z points up. Image TL/TR are assigned front-side posts; behind-goal views
    must not use this front-side assumption. Intrinsics are unknown, so sweep
    focal lengths and principal points and expose the resulting angle range.
    """
    world=np.array([[-3.66,0,2.44],[3.66,0,2.44],[3.66,0,0],[-3.66,0,0]],np.float64)
    image=np.asarray(corners,np.float64);rows=[]
    if image.shape!=(4,2) or not np.isfinite(image).all():return {'status':'invalid_corners'}
    for ratio in (.7,1.,1.5,2.,3.):
        for dx in (-.03,0,.03):
            k=np.array([[width*ratio,0,width*(.5+dx)],[0,width*ratio,height*.5],[0,0,1.]],float)
            try:ok,r,t=cv2.solvePnP(world,image,k,None,flags=cv2.SOLVEPNP_ITERATIVE)
            except cv2.error:continue
            if not ok:continue
            rot=cv2.Rodrigues(r)[0];camera=(-rot.T@t).ravel();projected=cv2.projectPoints(world,r,t,k,None)[0].reshape(-1,2)
            error=float(np.sqrt(np.mean(np.sum((projected-image)**2,axis=1))))
            if camera[1]>=-1 or not .5<camera[2]<60 or error>3:continue
            angle=math.degrees(math.atan2(camera[0],-camera[1]))
            rows.append({'angle':angle,'reprojection_error_px':error,'focal_over_width':ratio})
    if len(rows)<3:return {'status':'abstain_intrinsics_or_corners','solutions':len(rows)}
    angles=[r['angle'] for r in rows]
    return {'status':'sensitivity_candidate','median_degrees':float(np.median(angles)),
            'min_degrees':min(angles),'max_degrees':max(angles),'solutions':len(rows),
            'right_sign_consistent':min(angles)>10,'left_sign_consistent':max(angles)<-10,
            'calibration_verified':False,'assumptions':['goal front side','standard rectangular goal','unknown intrinsics swept','no lens distortion']}


def classify_view(samples):
    """Roles + framing + geometry. No playback-speed inference from one image."""
    votes=[];reasons=[]
    # A run-up moves the kicker sideways in the image, not the camera. Establish
    # front-side context at one sample, then compare goal pose across the shot.
    # Otherwise an off-centre run-up incorrectly rejects a behind-kicker camera.
    front_context=any(f.get('goal') and any(p.get('role')=='kicker_candidate' and
        p['bbox'][3]>np.mean(np.array(f['goal']['corners'])[2:,1])+.035*f['height']
        for p in f.get('people',[])) for f in samples)
    for frame in samples:
        people=frame.get('people',[]);goal=frame.get('goal');w,h=frame['width'],frame['height']
        kickers=[p for p in people if p.get('role')=='kicker_candidate'];keepers=[p for p in people if p.get('role')=='keeper_candidate']
        kicker=max(kickers,key=lambda p:p['area_fraction'],default=None)
        keeper=max(keepers,key=lambda p:p['area_fraction'],default=None)
        vote='unknown';reason='insufficient_role_or_geometry_evidence'
        if kicker and kicker['height_fraction']>.58 and kicker['area_fraction']>.10:
            vote='kicker_closeup';reason='kicker_kit_match_and_large_person'
        elif keeper and keeper['height_fraction']>.22 and ((kicker and keeper['height_fraction']>1.25*kicker['height_fraction']) or frame.get('foreground_net_cue',{}).get('supported')):
            # Goalkeeper in foreground, kicker beyond. The missing face cue is
            # supporting evidence only; inferred skeletons are not ground truth.
            beyond=bool(kicker and keeper['bbox'][3]>kicker['bbox'][3]+.08*h)
            net=frame.get('foreground_net_cue',{}).get('supported')
            # A goalkeeper can turn the head while the camera remains behind.
            # Near-net framing supplies a separate spatial cue; full face-on
            # detections still abstain. This remains a candidate, not proof.
            if (beyond or net) and keeper.get('face_visible_fraction',1)<(.7 if net else .4):
                vote='back_of_goalkeeper';reason='keeper_foreground_kicker_beyond_and_back_facing_pose_cue'
                if not beyond:reason='keeper_kit_back_facing_pose_with_near_net_side_lines; distant_kicker_unresolved'
        elif goal and front_context:
            calibration=frame.get('goal_camera',{});angle=calibration.get('median_degrees')
            gx0,gy0,gx1,gy1=goal['bbox'];offset=abs((kicker['bbox'][0]+kicker['bbox'][2])/2-(gx0+gx1)/2)/max(1,gx1-gx0) if kicker else float('inf')
            if angle is not None and calibration.get('right_sign_consistent') and 20<=angle<=70:
                vote='angle_45_right';reason='front_goal_hypotheses_consistently_kicker_right_oblique_not_exact_45'
            elif angle is not None and calibration.get('left_sign_consistent') and abs(angle)>20:
                vote='other_oblique_left';reason='camera_side_hypotheses_left_not_requested_right'
            elif angle is not None and max(abs(calibration.get('min_degrees',angle)),abs(calibration.get('max_degrees',angle)))<22:
                vote='back_of_kicker';reason='front_side_context_and_near_frontal_goal_camera_hypotheses'
            elif angle is None and offset<.35 and kicker.get('face_visible_fraction',1)<.4:
                vote='back_of_kicker';reason='foreground_back_facing_kicker_aligned_with_goal; calibration_missing'
            else:vote='oblique_side_unresolved';reason='kicker_goal_layout_oblique_side_not_supported'
        elif people and max(p['height_fraction'] for p in people)>.55:
            vote='person_closeup_identity_unresolved';reason='closeup_not_enough_to_establish_kicker_identity'
        votes.append(vote);reasons.append(reason)
    counts=Counter(votes);best,n=counts.most_common(1)[0] if counts else ('unknown',0)
    consistent=n>=max(2,math.ceil(len(samples)*.6))
    return {'view_candidate':best if consistent else 'unknown','sample_votes':votes,'sample_reasons':reasons,
            'consistent_samples':n,'sample_count':len(samples),'vote_fraction':n/max(1,len(samples)),
            'confidence_calibrated':False,'view_verified':False,'review_required':True}


def combine_task_class(view,speed):
    if view in ('back_of_kicker','angle_45_right'):return view
    if speed.get('status')=='slow_motion_candidate':
        if view=='kicker_closeup':return 'slowmo_kicker_closeup'
        if view=='back_of_goalkeeper':return 'slowmo_gk_back'
    return 'unknown'


def playback_evidence(reference_times,reference_features,query_times,query_features):
    """Same-action subsequence alignment with explicit unknown reference speed.

    Ordered nearest matches need broad temporal support and non-static motion.
    Relative dilation supports only a slow-motion *candidate*, never a fact.
    Input features must use the same representation and subject across shots.
    """
    result={'status':'unknown','reference_speed_verified':False,'absolute_speed_factor':None,
            'reason':'insufficient_same_action_alignment','review_required':True}
    a=np.asarray(reference_features,float);b=np.asarray(query_features,float)
    if len(a)<10 or len(b)<10 or a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1]:return result
    if not np.isfinite(a).all() or not np.isfinite(b).all():return result
    # Static or nearly static gestures cannot establish time dilation.
    if np.mean(np.std(a,axis=0))<.06 or np.mean(np.std(b,axis=0))<.06:return {**result,'reason':'insufficient_action_variation'}
    dist=np.sqrt(np.mean((b[:,None,:]-a[None,:,:])**2,axis=2))
    index=dist.argmin(axis=1);errors=dist[np.arange(len(b)),index]
    good=errors<.16
    if good.sum()<10:return result
    rt=np.asarray(reference_times)[index[good]];qt=np.asarray(query_times)[good]
    increasing=np.mean(np.diff(rt)>=-.10)
    if increasing<.85 or np.ptp(rt)<.5 or np.ptp(qt)<.8:return result
    slope,intercept=np.polyfit(rt,qt,1);residual=float(np.median(np.abs(qt-(slope*rt+intercept))))
    if slope<=0 or residual>.22:return result
    return {**result,'status':'slow_motion_candidate' if slope>=1.45 else 'no_slowdown_evidence',
            'relative_time_stretch':float(slope),'median_alignment_error_sec':residual,
            'matched_samples':int(good.sum()),'reference_span_sec':float(np.ptp(rt)),
            'reason':'ordered_pose_action_alignment; reference normal speed is an unverified assumption'}


def pose_action_feature(person):
    """Six unsigned joint angles; less scale-sensitive, not viewpoint invariant.

    Limb visibility is mandatory. Features cannot establish subject identity.
    """
    k=np.asarray(person.get('keypoints',[]),float)
    c=np.asarray(person.get('keypoint_confidences',[]),float)
    triples=((5,7,9),(6,8,10),(5,11,13),(6,12,14),(11,13,15),(12,14,16))
    if k.shape!=(17,2) or c.shape!=(17,):return None
    if min(c[j] for j in range(5,17))<.4:return None
    values=[]
    for a,b,d in triples:
        u=k[a]-k[b];v=k[d]-k[b];norm=np.linalg.norm(u)*np.linalg.norm(v)
        if norm<1:return None
        values.append(float(np.arccos(np.clip(np.dot(u,v)/norm,-1,1))/np.pi))
    return values

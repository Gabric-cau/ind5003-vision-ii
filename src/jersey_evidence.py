"""Local, target-blind jersey-text evidence and within-shot crop retrieval.

Inspired by tracklet/legibility/pose-ROI/consolidation decomposition in
https://github.com/mkoshkina/jersey-number-pipeline . This is a lightweight
project-written baseline, NOT their trained model or a SoccerNet reproduction.
No number, name, kit match or tracklet is automatically human ground truth.
"""
from collections import defaultdict
import re
import unicodedata
import cv2
import numpy as np


def iou(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    area=float(np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))))
    union=float(np.prod(np.maximum(0,a[2:]-a[:2]))+np.prod(np.maximum(0,b[2:]-b[:2]))-area)
    return area/max(1.,union)


def build_tracklets(frames,fps,max_gap_sec=.4,min_iou=.3,ambiguity_margin=.1):
    """Mutual-best IoU association; refuse ambiguous joins and long gaps.

    Caller passes ONE shot. Colour, expected jersey and OCR are not inputs.
    """
    if fps<=0 or any(b['frame']<=a['frame'] for a,b in zip(frames,frames[1:])):
        raise ValueError('Frames must be ordered and FPS positive')
    tracks=[]
    for frame in frames:
        fid=frame['frame']
        people=[(i,p) for i,p in enumerate(frame['people']) if p['bbox'][3]-p['bbox'][1]>=160
                and p.get('area_fraction',0)>=.025]
        active=[t for t in tracks if 0<(fid-t['observations'][-1]['frame'])/fps<=max_gap_sec]
        score=np.array([[iou(t['observations'][-1]['person']['bbox'],p['bbox']) for _,p in people] for t in active])
        assignments={}
        if active and people:
            for j,(_,p) in enumerate(people):
                i=int(score[:,j].argmax());row=score[i];col=score[:,j]
                if row[j]<min_iou or int(row.argmax())!=j:continue
                row_alt=max(np.delete(row,j),default=0.);col_alt=max(np.delete(col,i),default=0.)
                if row[j]-row_alt<ambiguity_margin or col[i]-col_alt<ambiguity_margin:continue
                assignments[j]=active[i]
        for j,(person_index,p) in enumerate(people):
            t=assignments.get(j)
            if t is None:
                t={'tracklet_id':len(tracks),'observations':[]};tracks.append(t)
            t['observations'].append({'frame':fid,'time_sec':fid/fps,'person_index':person_index,'person':p})
    return tracks


def torso_crop(frame,person):
    """Axis-aligned torso ROI. No synthetic completion or super-resolution."""
    h,w=frame.shape[:2];b=np.asarray(person['bbox'],float);bw,bh=b[2]-b[0],b[3]-b[1]
    kp=np.asarray(person.get('keypoints',[]),float);conf=np.asarray(person.get('keypoint_confidences',[]),float)
    if kp.shape==(17,2) and conf.shape==(17,) and min(conf[[5,6,11,12]])>=.4 and kp[[11,12],1].mean()>kp[[5,6],1].mean()+20:
        pts=kp[[5,6,11,12]];lo=pts.min(axis=0);hi=pts.max(axis=0);size=hi-lo
        box=[lo[0]-.12*size[0],lo[1]-.06*size[1],hi[0]+.12*size[0],hi[1]+.03*size[1]]
        mode='pose_torso_bounds'
    else:
        box=[b[0]+.12*bw,b[1]+.20*bh,b[2]-.12*bw,b[3]];mode='bbox_torso_fallback'
    x0,y0,x1,y1=map(int,np.rint(box))
    x0=max(0,int(b[0]),x0);y0=max(0,int(b[1]),y0)
    x1=min(w,int(b[2]),x1);y1=min(h,int(b[3]),y1)
    if x1-x0<24 or y1-y0<30:return None,{'reason':'roi_too_small'}
    crop=frame[y0:y1,x0:x1].copy()
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    sharpness=float(cv2.Laplacian(gray,cv2.CV_64F).var())
    return crop,{'roi':[x0,y0,x1,y1],'roi_method':mode,'sharpness_laplacian':sharpness,
                 'source_width':x1-x0,'source_height':y1-y0,
                 'quality':float(np.log1p(sharpness)*np.sqrt((x1-x0)*(y1-y0)))}


def select_keyframes(observations,max_frames=6,min_spacing_sec=.2):
    """Pick highest-quality frame per temporal bin, then greedy spacing.

    Does not know the query number/name, or inspect OCR outputs.
    """
    if not observations:return []
    ordered=sorted(observations,key=lambda o:o['frame'])
    bins=np.array_split(np.arange(len(ordered)),min(max_frames,len(ordered)))
    chosen=[]
    for bucket in bins:
        for i in sorted(bucket,key=lambda i:-ordered[i]['crop_info']['quality']):
            row=ordered[int(i)]
            if all(abs(row['time_sec']-v['time_sec'])>=min_spacing_sec-1e-9 for v in chosen):
                chosen.append(row);break
    return sorted(chosen,key=lambda o:o['frame'])


def normalize_name(value):
    value=unicodedata.normalize('NFKD',value)
    return ''.join(c for c in value.upper() if 'A'<=c<='Z')


def parse_number(text):
    text=str(text).strip()
    if not re.fullmatch(r'[1-9][0-9]?',text):return None
    return int(text)


def consolidate(observations,min_confidence=.85,min_samples=2,min_span_sec=.2,consensus=.75):
    """Target-blind vote. Duplicate same-frame crops count at most once.

    More than one number on the same torso is ambiguous, not a chance to select
    whichever matches the requested player. Numeric conflicts remain explicit.
    """
    frame_numbers=defaultdict(dict);names=defaultdict(dict)
    for obs in observations:
        for token in obs.get('tokens',[]):
            if token['confidence']<min_confidence or not token.get('within_torso',True):continue
            number=parse_number(token['text'])
            if number is not None:
                frame_numbers[obs['frame']][number]=max(frame_numbers[obs['frame']].get(number,0),token['confidence'])
            elif len(normalize_name(token['text']))>=4:
                name=normalize_name(token['text']);names[name][obs['frame']]=obs['time_sec']
    frame_times={o['frame']:o['time_sec'] for o in observations};votes=defaultdict(list);ambiguous=[]
    for fid,numbers in frame_numbers.items():
        if len(numbers)!=1:ambiguous.append(fid);continue
        num,score=next(iter(numbers.items()));votes[num].append({'frame':fid,'time_sec':frame_times[fid],'confidence':score})
    ranked=sorted(votes,key=lambda n:(-len(votes[n]),-sum(r['confidence'] for r in votes[n]),n))
    count=sum(len(v) for v in votes.values());winner=ranked[0] if ranked else None
    selected=votes[winner] if winner is not None else []
    fraction=len(selected)/max(1,count);span=max((r['time_sec'] for r in selected),default=0)-min((r['time_sec'] for r in selected),default=0)
    supported=len(selected)>=min_samples and span>=min_span_sec-1e-9 and fraction>=consensus
    return {'status':'number_candidate' if supported else 'unknown','number':winner if supported else None,
       'ranked_number_votes':[{'number':n,'samples':votes[n]} for n in ranked],
       'numeric_frames':count,'ambiguous_numeric_frames':ambiguous,'consensus_fraction':fraction,'support_span_sec':span,
       'name_tokens':[{'text':name,'frames':sorted(frames),'span_sec':max(frames.values())-min(frames.values())} for name,frames in names.items()],
       'human_verified':False,'identity_verified':False,'accuracy_validated':False,
       'warning':'Numbers can repeat across teams and spectators; OCR is not proof of kicker identity.'}


def query_evidence(evidence,number,surname):
    expected=normalize_name(surname);name_hits=[n for n in evidence['name_tokens'] if n['text']==expected]
    repeated_name=any(len(n['frames'])>=2 and n['span_sec']>=.2-1e-9 for n in name_hits)
    n=evidence['number'];has_number=evidence['status']=='number_candidate'
    if has_number and n!=number:status='number_conflict'
    elif has_number and n==number:status='number_and_name_candidate' if repeated_name else 'number_match_candidate'
    elif repeated_name:status='name_match_candidate'
    else:status='unknown'
    return {'status':status,'expected_number_from_manifest':number,'expected_surname_from_manifest':surname,
            'identity_verified':False,'name_hit_frames':sorted({f for n in name_hits for f in n['frames']}),
            'expected_values_used_only_after_target_blind_recognition':True}

"""Diagnostic within-shot optical-flow pilot; no production candidate override."""
from pathlib import Path
import sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_camera_partial_pose_experiment import DEFAULT,read,save,sha
from src.goal_temporal_tracking import track_goal_pair
from src.camera_timing_v2 import goal_camera_hypotheses,classify_view
from src.camera_view_intervals import supported_intervals


def main():
    cid='col_eng_p03_cuadrado_col11';sid=2
    source=DEFAULT/'dense_views_v1'/'clips'/f'{cid}.json'
    data=read(source);segment=next(s for s in data['segments'] if s['segment_id']==sid)
    sampled={s['frame']:s for s in segment['samples']};ids=sorted(sampled)
    out=DEFAULT/'goal_tracking_pilot_v1';out.mkdir(exist_ok=False)
    save(out/'protocol.json',{'clip_id':cid,'segment_id':sid,'selection':'diagnostic known failure; not independent test',
          'step_frames':3,'maximum_independent_detection_age_sec':1.2,'max_detection_disagreement_goal_width':.025,
          'code_sha256':sha(ROOT/'src/goal_temporal_tracking.py'),'runner_sha256':sha(__file__),
          'source_feature_sha256':sha(source),'source':'https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html'})
    video=ROOT/'data/raw_clips'/f'{cid}.mp4'
    if sha(video)!=data['source_sha256']:raise ValueError('Source changed')
    cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS)
    current_corners=None;previous=None;last_detection=None;rows=[];new_samples=[];tiles=[]
    for fid in range(ids[0],ids[-1]+1,3):
        cap.set(1,fid);ok,frame=cap.read()
        if not ok:raise RuntimeError(fid)
        sample=sampled.get(fid);direct=(sample or {}).get('goal')
        flow=None;origin='missing'
        if current_corners is not None:
            flow=track_goal_pair(previous,frame,current_corners)
            if flow['status']=='tracked_candidate':current_corners=flow['corners'];origin='flow'
            else:current_corners=None
        if direct:
            if current_corners is None:
                current_corners=direct['corners'];last_detection=fid;origin='direct_seed'
            else:
                width=max(1,float(np.ptp(np.array(current_corners)[:,0])))
                deviation=float(np.max(np.linalg.norm(np.array(current_corners)-direct['corners'],axis=1))/width)
                if deviation<=.025:last_detection=fid;origin='flow_agrees_with_direct'
        if current_corners is not None and (fid-last_detection)/fps>1.2:
            current_corners=None;origin='independent_detection_too_old'
        row={'frame':fid,'origin':origin,'flow':flow,'direct_goal':direct,
             'tracked_corners':current_corners,'last_independent_detection_frame':last_detection}
        rows.append(row)
        if sample:
            updated={**sample};c=np.array(current_corners) if current_corners is not None else None
            goal={'corners':c.tolist(),'bbox':[float(c[:,0].min()),float(c[:,1].min()),float(c[:,0].max()),float(c[:,1].max())]} if c is not None else None
            updated['goal']=goal
            updated['goal_camera']=goal_camera_hypotheses(c,sample['width'],sample['height']) if c is not None else {'status':'no_supported_goal'}
            new_samples.append(updated)
            if direct:cv2.polylines(frame,[np.array(direct['corners'],np.int32)],True,(0,0,255),3)
            if c is not None:cv2.polylines(frame,[c.astype(np.int32)],True,(0,255,0),3)
            thumb=cv2.resize(frame,(480,270))
            cv2.putText(thumb,f'{fid/fps:.2f}s {origin}',(10,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,0),2)
            tiles.append(thumb)
        previous=frame.copy() if sample is None else None
        # Do not feed the drawn overlay into optical flow.
        if sample:
            cap.set(1,fid);ok,previous=cap.read()
    cap.release();view=classify_view(new_samples)
    intervals=supported_intervals(ids,view['sample_votes'],ids[0],ids[-1]+1,fps)
    summary={'clip_id':cid,'segment_id':sid,'frames_processed':len(rows),
         'dense_samples':len(ids),'direct_goal_samples':sum(bool(s.get('goal')) for s in segment['samples']),
         'tracked_goal_samples':sum(bool(s.get('goal')) for s in new_samples),
         'old_votes':segment['dense_classification']['sample_votes'],'new_votes':view['sample_votes'],
         'candidate_intervals':intervals,'human_verified':False,'accuracy_validated':False,'release_replaced':False}
    save(out/'tracking.json',rows);save(out/'samples.json',new_samples);save(out/'summary.json',summary)
    canvas=np.zeros((int(np.ceil(len(tiles)/3))*270,1440,3),np.uint8)
    for i,tile in enumerate(tiles):canvas[i//3*270:(i//3+1)*270,i%3*480:(i%3+1)*480]=tile
    cv2.imencode('.jpg',canvas,[cv2.IMWRITE_JPEG_QUALITY,90])[1].tofile(str(out/'comparison.jpg'))
    print(summary)


if __name__=='__main__':main()

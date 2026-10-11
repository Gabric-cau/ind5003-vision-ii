"""Guard the spatial candidates without changing prior measurements/releases."""
from pathlib import Path
import sys
from collections import Counter
import cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_camera_partial_pose_experiment import DEFAULT,BASE,read,save,sha
from src.camera_view_guard import classify_guarded,closeup_context
from src.camera_view_intervals import supported_intervals


def main():
    source=DEFAULT/'dense_views_v1';out=DEFAULT/'guarded_views_v2';out.mkdir(exist_ok=False)
    save(out/'protocol.json',{'guard_sha256':sha(ROOT/'src/camera_view_guard.py'),
        'runner_sha256':sha(__file__),'source_protocol_sha256':sha(source/'protocol.json'),
        'selection':'54 interval contact sheets AI inspected before these engineering guards; development experiment, not holdout',
        'guard':'comparable same-kit competitors >= .4 target area; foreground count >=4; green outside all people >=.12 full frame',
        'note':'Masks cannot prove footballer identity; valid tight closeups may be rejected. Speed always unknown.'})
    baseline={c['clip_id']:c for c in read(BASE/'camera_timings_candidates.json')['clips']}
    rows=[];comparisons=[];reasons=Counter();geofix=0;sample_count=0;inherited={}
    for path in sorted((source/'clips').glob('*.json')):
        d=read(path);cid=d['clip_id'];clip=baseline[cid]
        cap=cv2.VideoCapture(str(ROOT/'data/raw_clips'/f'{cid}.mp4'));segments=[];intervals=[]
        if sha(ROOT/'data/raw_clips'/f'{cid}.mp4')!=d['source_sha256']:raise ValueError('Source changed')
        for s in d['segments']:
            for sample in s['samples']:
                cap.set(1,sample['frame']);ok,frame=cap.read()
                if not ok:raise ValueError('Decode failed')
                sample['closeup_context']=closeup_context(frame,sample['people']);sample_count+=1
            view=classify_guarded(s['samples']);src=clip['segments'][s['segment_id']]
            for old,new,why,g in zip(view['old_sample_votes'],view['sample_votes'],view['sample_reasons'],view['closeup_guards']):
                if old!=new:
                    reasons[why.split(';')[0]]+=1
                    if why.startswith('goal_geometry_fallback'):geofix+=1
            found=supported_intervals([f['frame'] for f in s['samples']],view['sample_votes'],
                     src['start_frame'],src['end_frame_exclusive'],clip['fps'])
            intervals.extend({**r,'clip_id':cid,'segment_id':s['segment_id'],
                'old_whole_shot_view':src['view_candidate'],'identity_verified':False} for r in found)
            segments.append({**s,'guarded_classification':view})
            vote_lookup=dict(zip((f['frame'] for f in s['samples']),view['sample_votes']))
            for old in d['intervals']:
                if old['segment_id']!=s['segment_id']:continue
                keep=sum(vote_lookup.get(f)==old['view_candidate'] for f in old['support_frames'])
                comparisons.append({'clip_id':cid,'segment_id':old['segment_id'],'old_start_frame':old['start_frame'],
                    'old_view':old['view_candidate'],'support_before':len(old['support_frames']),'support_after_guard':keep,
                    'all_original_support_survives':keep==len(old['support_frames'])})
        cap.release();rows.extend(intervals);inherited[cid]=sha(path)
        save(out/'clips'/f'{cid}.json',{'clip_id':cid,'source_sha256':d['source_sha256'],
             'source_dense_sha256':sha(path),'segments':segments,'intervals':intervals})
    summary={'clips_indexed':len(baseline),'dense_samples_checked':sample_count,'intervals_before':54,
        'intervals_after':len(rows),'views':dict(Counter(r['view_candidate'] for r in rows)),
        'sample_vote_change_reasons':dict(reasons),'geometry_fallback_samples':geofix,
        'old_intervals_with_any_guard_rejection':sum(r['support_after_guard']<r['support_before'] for r in comparisons),
        'old_intervals_all_support_rejected':sum(r['support_after_guard']==0 for r in comparisons),
        'human_ground_truth_records':0,'accuracy_estimate':None,'independent_holdout':False,'release_replaced':False,
        'intervals':rows,'comparisons':comparisons}
    save(out/'summary.json',summary);print({k:v for k,v in summary.items() if k not in ('intervals','comparisons')})


if __name__=='__main__':main()

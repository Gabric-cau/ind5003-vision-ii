"""Re-use measured poses, add denser goal cues, report only supported subintervals."""
from pathlib import Path
from collections import Counter
import argparse,json,sys
import cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_camera_partial_pose_experiment import DEFAULT,BASE,read,save,sha
from scripts.camera_timing_context import near_net
from src.goal_geometry import detect_goal_frame
from src.camera_timing_v2 import goal_camera_hypotheses,classify_view
from src.camera_view_intervals import supported_intervals


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=DEFAULT/'dense_views_v1')
    p.add_argument('--measurements',type=Path,default=DEFAULT/'measurements');p.add_argument('--resume',action='store_true')
    a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    cfg={'dense_stride':3,'min_samples':4,'min_duration_sec':.6,'max_gap_sec':.35,
         'script_sha256':sha(__file__),'view_code_sha256':sha(ROOT/'src/camera_timing_v2.py'),
         'interval_code_sha256':sha(ROOT/'src/camera_view_intervals.py'),
         'goal_code_sha256':sha(ROOT/'src/goal_geometry.py'),'net_code_sha256':sha(ROOT/'scripts/camera_timing_context.py'),
         'warning':'Only previously selected experimental pose windows are sampled; not complete video coverage. No slow-motion claims.'}
    if (out/'protocol.json').exists():
        if read(out/'protocol.json')!=cfg:raise ValueError('Protocol changed')
    else:save(out/'protocol.json',cfg)
    package=read(BASE/'camera_timings_candidates.json');rows=[];cv2.setNumThreads(2)
    for clip in package['clips']:
        cid=clip['clip_id'];target=out/'clips'/f'{cid}.json';measurement_path=a.measurements/f'{cid}.json'
        m=read(measurement_path)
        if target.exists():
            if not a.resume:raise FileExistsError(target)
            cached=read(target)
            if cached['measurement_sha256']!=sha(measurement_path):raise ValueError('Measurements changed')
            rows.extend(cached['intervals']);continue
        path=Path(clip['raw_video_path'])
        if sha(path)!=clip['source_sha256']:raise ValueError('Source changed')
        cap=cv2.VideoCapture(str(path));segments=[];intervals=[]
        for s in m['segments']:
            source=clip['segments'][s['segment_id']];samples=[]
            for observation in s['frames'][::cfg['dense_stride']]:
                fid=observation['frame'];cap.set(1,fid);ok,frame=cap.read()
                if not ok:raise ValueError('Decode failed')
                h,w=frame.shape[:2];proposal=detect_goal_frame(frame,max_goal_width_fraction=.98,max_post_height_fraction=.85)
                goal=proposal.get('selected')
                samples.append({'frame':fid,'width':w,'height':h,'people':observation['people'],'goal':goal,
                    'goal_camera':goal_camera_hypotheses(goal['corners'],w,h) if goal else {'status':'no_supported_goal'},
                    'foreground_net_cue':near_net(frame)})
            view=classify_view(samples)
            found=supported_intervals([f['frame'] for f in samples],view['sample_votes'],
                source['start_frame'],source['end_frame_exclusive'],clip['fps'],min_samples=cfg['min_samples'],
                min_duration=cfg['min_duration_sec'],max_gap_sec=cfg['max_gap_sec'])
            intervals.extend({**i,'clip_id':cid,'segment_id':s['segment_id'],'old_whole_shot_view':source['view_candidate'],
                              'old_whole_shot_start_sec':source['start_time_sec'],'old_whole_shot_end_sec':source['end_time_sec']} for i in found)
            segments.append({'segment_id':s['segment_id'],'old_view':source['view_candidate'],'sampled_window':[s['start_frame'],s['end_frame_exclusive']],
                             'dense_classification':view,'samples':samples})
        cap.release()
        result={'clip_id':cid,'measurement_sha256':sha(measurement_path),'source_sha256':clip['source_sha256'],
                'segments':segments,'intervals':intervals,'human_verified':False,'accuracy_validated':False}
        save(target,result);rows.extend(intervals)
        print(json.dumps({'clip':cid,'spatial_intervals':len(intervals),'dense_samples':sum(len(s['samples']) for s in segments)}),flush=True)
    summary={'canonical_clips':39,'clips_with_intervals':len(set(r['clip_id'] for r in rows)),
             'candidate_subintervals':len(rows),'views':dict(Counter(r['view_candidate'] for r in rows)),
             'new_spatial_intervals_from_unknown_shots':sum(r['old_whole_shot_view']=='unknown' for r in rows),
             'intervals':rows,'human_ground_truth_records':0,'accuracy_estimate':None,'complete_video_coverage':False,'release_replaced':False}
    save(out/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='intervals'}),flush=True)

if __name__=='__main__':main()

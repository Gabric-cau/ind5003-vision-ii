"""New, isolated dense-pose measurements; retain rejection causes and partial bodies.

No old outputs, human annotations or outcome labels are modified/read.
"""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import argparse, hashlib, json, sys
import cv2
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.camera_timing_features import load_model,infer,get_frame
from src.camera_timing_v2 import assign_kit_roles

BASE=ROOT/'outputs/camera_timing_retest_2026-10-10_01'
DEFAULT=ROOT/'outputs/camera_partial_alignment_2026-10-11'
ELIGIBLE={'kicker_closeup','back_of_goalkeeper','back_of_kicker','angle_45_right','other_oblique_left','oblique_side_unresolved'}
PILOT=('col_eng_p01_falcao_col9','col_eng_p02_kane_eng9','col_eng_p03_cuadrado_col11',
       'esp_rus_p02_smolov_rus10','col_eng_p08_trippier_eng12','rus_cro_p03_dzagoev_rus9')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT)
    parser.add_argument('--cohort',choices=('pilot','remaining','all'),default='all')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();out=args.output_dir.resolve()
    if out==BASE or out.is_relative_to(BASE):raise ValueError('Use a separate experiment directory')
    package=read(BASE/'camera_timings_candidates.json')
    ids=[c['clip_id'] for c in package['clips']]
    selected=[c for c in package['clips'] if args.cohort=='all' or ((c['clip_id'] in PILOT)==(args.cohort=='pilot'))]
    protocol={'sampling_fps_target':10,'reference_window_sec':[-1.4,1.0],'query_max_seconds':8,
      'query_selection':'after_contact_and_at_least_one_existing_sample_vote_eligible; not a new verified camera label',
      'pose_confidence':.4,'role_policy':'unchanged torso kit matching; exactly one role candidate per frame',
      'identity_verified':False,'pilot_clips':PILOT,'remaining_is_independent_holdout':False,
      'canonical_clips':ids,'outcome_labels_read':False,'human_labels_created':False,
      'extraction_script_sha256':sha(__file__),'baseline_json_sha256':sha(BASE/'camera_timings_candidates.json'),
      'weights_sha256':sha(ROOT/'models/weights/yolo11n_pose.pt'),
      'inference_helper_sha256':sha(ROOT/'scripts/camera_timing_features.py'),
      'role_code_sha256':sha(ROOT/'src/camera_timing_v2.py')}
    if (out/'extraction_protocol.json').exists():
        if read(out/'extraction_protocol.json')!=json.loads(json.dumps(protocol)):raise ValueError('Protocol changed')
    else:save(out/'extraction_protocol.json',protocol)
    cv2.setNumThreads(2);model=None
    for clip in selected:
        cid=clip['clip_id'];target=out/'measurements'/f'{cid}.json'
        if target.exists():
            if not args.resume:raise FileExistsError(target)
            cached=read(target)
            if cached['protocol_sha256']!=sha(out/'extraction_protocol.json'):raise ValueError('Cache mismatch')
            continue
        features=read(BASE/'features'/f'{cid}.json');refs=features['references'];cf=refs['contact_frame_candidate']
        for name,value in refs['upstream_sha256'].items():
            if sha(ROOT/name)!=value:raise ValueError('Upstream source changed: '+name)
        original=Path(clip['raw_video_path'])
        if sha(original)!=clip['source_sha256']:raise ValueError('Original video changed')
        reference=next((s for s in clip['segments'] if s['start_frame']<=cf<s['end_frame_exclusive']),None)
        choices=[s for s in clip['segments'] if s['start_frame']>cf and any(v in ELIGIBLE for v in s['sample_votes'])]
        data={'clip_id':cid,'protocol_sha256':sha(out/'extraction_protocol.json'),'source_sha256':sha(original),
              'features_sha256':sha(BASE/'features'/f'{cid}.json'),'contact_frame_candidate':cf,
              'fps':clip['fps'],'reference_segment_id':reference['segment_id'] if reference else None,
              'query_segment_ids':[s['segment_id'] for s in choices], 'segments':[],
              'human_verified':False,'identity_verified':False,'accuracy_validated':False}
        if reference and choices:
            if model is None:model=load_model()
            cap=cv2.VideoCapture(str(original));fps=clip['fps']
            for shot in [reference]+choices:
                if shot==reference:
                    start=max(shot['start_frame'],cf-round(1.4*fps));end=min(shot['end_frame_exclusive'],cf+round(fps))
                else:
                    start=shot['start_frame']+2;end=min(shot['end_frame_exclusive']-2,start+round(8*fps))
                frames=[]
                for fid in range(start,end,max(1,round(fps/10))):
                    frame=get_frame(cap,fid);people=assign_kit_roles(infer(model,frame),refs)
                    # Retain raw observations so feature experiments need no new inference.
                    frames.append({'frame':fid,'time_sec':fid/fps,'people':people,
                      'role_counts':{role:sum(p['role']==role+'_candidate' for p in people) for role in ('kicker','keeper')}})
                data['segments'].append({'segment_id':shot['segment_id'],'start_frame':start,'end_frame_exclusive':end,
                      'view_candidate_unchanged':shot['view_candidate'],'baseline_sample_votes':shot['sample_votes'],
                      'baseline_action_tested':str(shot['segment_id']) in read(BASE/'action_v3'/f'{cid}.json')['evidence'],
                      'frames':frames})
            cap.release()
        if sha(original)!=data['source_sha256']:raise ValueError('Original changed during inference')
        save(target,data)
        print(json.dumps({'clip':cid,'queries':len(choices),'pose_frames':sum(len(s['frames']) for s in data['segments'])}),flush=True)

if __name__=='__main__':main()

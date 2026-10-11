"""Export interval candidates, explicit unknowns, and a read-only review package."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.camera_timing_v2 import classify_view,combine_task_class,TASK_CLASSES,relabel_cached_features,ROLE_POLICY


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def export(out,ids):
    clips=[];flat=[]
    for cid in ids:
        scan=read(out/'scan'/f'{cid}.json');features=read(out/'features'/f'{cid}.json')
        refs_path=out/'references_v2'/f'{cid}.json'
        if refs_path.exists():features={**features,'references':read(refs_path)}
        context_path=out/'context_v2'/f'{cid}.json'
        if context_path.exists():
            from scripts.camera_timing_context import merge
            features=merge(features,read(context_path))
        features=relabel_cached_features(features)
        action_path=out/'action_v3'/f'{cid}.json';action=read(action_path) if action_path.exists() else {'evidence':{}}
        if action_path.exists() and action['cache_signature']['classifier_sha256']!=sha(ROOT/'src/camera_timing_v2.py'):
            raise ValueError('Action evidence is stale relative to classification code')
        segments=[];previous=0
        for s,f in zip(scan['segments'],features['segments'],strict=True):
            assert s['segment_id']==f['segment_id'] and s['start_frame']==previous
            previous=s['end_frame_exclusive'];view=classify_view(f['samples'])
            speed=action['evidence'].get(str(s['segment_id']),{'status':'unknown','reason':'no_same_action_alignment_test',
                'reference_speed_verified':False,'review_required':True})
            cls=combine_task_class(view['view_candidate'],speed)
            row={**s,**view,'clip_id':cid,'playback_evidence':speed,'task_class_candidate':cls,'verified_task_class':'unknown',
                'human_verified':False,'source_match_start_sec':scan['manifest']['start_sec']+s['start_time_sec'],
                'source_match_end_sec':scan['manifest']['start_sec']+s['end_time_sec'],
                'source_time_basis':'manifest offset + local frame/FPS; offset not independently verified',
                'identity_warning':'Torso kit matching can confuse teammates; all identities are unverified.',
                'boundary_start_kind':next((c['transition_kind'] for c in scan['cut_candidates'] if c['frame']==s['start_frame']),'clip_start'),
                'boundary_start_uncertainty_sec':next((c['timing_uncertainty_sec'] for c in scan['cut_candidates'] if c['frame']==s['start_frame']),0.)}
            segments.append(row)
            flat.append({k:row[k] for k in ('clip_id','segment_id','start_frame','end_frame_exclusive','start_time_sec','end_time_sec',
                'source_match_start_sec','source_match_end_sec','view_candidate','task_class_candidate','verified_task_class',
                'human_verified','review_required','boundary_start_kind','boundary_start_uncertainty_sec')}|{
                'playback_status':speed['status'],'source_sha256':scan['source_sha256'],
                'evidence_reasons':' | '.join(row['sample_reasons'])})
        assert previous==scan['frame_count']
        presence={c:{'status':'detected_candidate' if any(s['task_class_candidate']==c for s in segments) else 'not_detected',
                     'segment_ids':[s['segment_id'] for s in segments if s['task_class_candidate']==c],
                     'absence_verified':False} for c in TASK_CLASSES}
        clips.append({'clip_id':cid,'fps':scan['fps'],'frame_count':scan['frame_count'],'duration_sec':scan['frame_count']/scan['fps'],
            'video':'videos/'+cid+'.mp4','raw_video_path':scan['video_path'],'source_sha256':scan['source_sha256'],
            'source_match_video':scan['manifest']['source_video_filename'],'manifest_offset_sec':scan['manifest']['start_sec'],
            'segments':segments,'task_category_presence':presence,'source_outcome_labels_read':False,'human_verified':False})
    summary={'clips_processed':len(clips),'segments':len(flat),'source_frames_decoded':sum(c['frame_count'] for c in clips),
        'view_candidates':dict(Counter(r['view_candidate'] for r in flat)),
        'task_class_candidates':dict(Counter(r['task_class_candidate'] for r in flat)),
        'playback_status':dict(Counter(r['playback_status'] for r in flat)),
        'timeline_coverage':1.,'human_verified_segments':0,'boundary_accuracy':None,'class_accuracy':None,
        'accuracy_warning':'Complete timeline coverage is not camera-classification accuracy.',
        'source_outcome_labels_read':False,'role_reanalysis_policy':ROLE_POLICY,
        'action_stage_clips':sum((out/'action_v3'/f'{cid}.json').exists() for cid in ids)}
    signature=sha(ROOT/'src/camera_timing_v2.py')
    package={'schema_version':'camera_timing_candidates_v2','classifier_sha256':signature,'summary':summary,'clips':clips,
        'task_classes':list(TASK_CLASSES),'notes':['End frame/time is exclusive.','All automated labels are candidates, not human ground truth.',
        'not_detected does not mean absent.','Camera-right is kicker-right when facing the goal, not image-right.',
        'No exact 45-degree calibration or absolute slow-motion speed has been verified.']}
    write(out/'camera_timings_candidates.json',package);write(out/'summary.json',summary)
    with (out/'camera_timings_candidates.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    review=out/'review';review.mkdir(exist_ok=True)
    # JSON is script data, never executable user input; escape HTML delimiters.
    encoded=json.dumps(package,ensure_ascii=False,allow_nan=False).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    (review/'data.js').write_text('window.CAMERA_DATA = '+encoded+';\n',encoding='utf-8')
    for name in ('index.html','app.js','style.css'):shutil.copyfile(ROOT/'assets/camera_review'/name,review/name)
    shutil.copyfile(out/'camera_timings_candidates.csv',review/'camera_timings_candidates.csv')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/camera_timing_2026-10-10/release');a=p.parse_args()
    export(a.output_dir,[f.stem for f in sorted((a.output_dir/'features').glob('*.json'))])

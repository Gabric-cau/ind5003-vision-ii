"""Recompute cached goalkeeper evidence with independent direction/height gates."""
from pathlib import Path
from collections import Counter
import sys,json,hashlib,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.goalkeeper_observation_guard import guard_postkick_observations
RUN=ROOT/'outputs/optimization_evening_2026-10-09'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/goalkeeper_pose_direction_2026-10-11/results_v3_final')
    p.add_argument('--policy',choices=['separate_direction_v2','legacy_full_window_v1','pose_direction_v3'],default='pose_direction_v3')
    p.add_argument('--resume',action='store_true');a=p.parse_args();out=a.output_dir
    source=RUN/'goal_appearance_v2';old_dir=RUN/'goal_observation_guard';rows=[];changes=[];hashes=[]
    protocol={'policy':a.policy,'source':'existing appearance-bounded trajectories; no new model inference',
       'source_directory':str(source),'old_directory':str(old_dir),
       'code_sha256':{n:sha(ROOT/n) for n in ['src/goalkeeper_observation_guard.py','src/goalkeeper_direction_evidence.py',
                                             'src/goalkeeper_pose_direction.py','scripts/run_goalkeeper_direction_update.py']},
       'human_ground_truth_read':False,'coordinate_system':'source_image_x_right',
       'height_rule':'original full 1s window: >=80% samples, <=.2s gaps, <=.1s endpoint loss',
       'direction_rule':'qualified non-kicker track, unambiguous seed, onset-connected >=6 points spanning >=.2s, >=50% local density, sustained >=.3 body lateral motion, >=75% directional increments; must agree with prior side candidate',
       'center_rule':'short static prefix never proves center; requires original full-window support',
       'whole_record_QA_not_promoted_from_direction_only':True}
    if a.policy=='pose_direction_v3':
        protocol.update(pose_direction_rule='Primary when supported: connected selected source track, upright pre-kick reference, >=3 lateral torso and shoulder-to-ankle poses spanning >=.06s; no box-displacement minimum, no long gap or identity switch bridging; otherwise retain v2 motion fallback',
                        pose_input='cached exact source-track keypoints; no new inference or user-label inputs')
    if (out/'protocol.json').exists():
        if not a.resume:raise FileExistsError(out/'protocol.json')
        if read(out/'protocol.json')!=protocol:raise ValueError('Cached protocol changed')
    else:save(out/'protocol.json',protocol)
    for path in sorted(source.glob('*.json')):
        if path.name=='summary.json':continue
        original=read(path);old=read(old_dir/path.name)
        if 'kick_frame' not in original:
            step=read(RUN/'steps_v4/classified'/path.name)
            original.update(kick_frame=step['contact']['contact_frame'],fps=step['fps'],video_path=step['video_path'],
                            kick_frame_source='independent_contact_candidate')
        legacy=guard_postkick_observations(original,policy='legacy_full_window_v1')
        for key in ['dive_direction','dive_height','auto_accepted','dive_observability']:
            if legacy[key]!=old[key]:raise ValueError(f'Legacy replay differs: {path.name}/{key}')
        pose_path=original.get('source_pose_path')
        pose=read(pose_path) if a.policy=='pose_direction_v3' and pose_path and Path(pose_path).is_file() else None
        result=guard_postkick_observations(original,policy=a.policy,pose=pose)
        if pose is not None:
            result.update(source_pose_path=pose_path,source_pose_sha256=sha(pose_path))
        result.update(source_candidate_path=str(path),source_candidate_sha256=sha(path),
                      previous_guard_path=str(old_dir/path.name),previous_guard_sha256=sha(old_dir/path.name))
        target=out/'clips'/path.name
        if target.exists():
            if not a.resume or read(target)!=result:raise FileExistsError(target)
        else:save(target,result)
        row={'clip_id':result['clip_id'],'previous_direction':old['dive_direction'],
          'v2_direction':guard_postkick_observations(original,policy='separate_direction_v2')['dive_direction'],
          'dive_direction':result['dive_direction'],'dive_height':result['dive_height'],
          'auto_accepted':result['auto_accepted'],'direction_support_status':result.get('dive_direction_support_status'),
          'direction_evidence':result.get('dive_direction_evidence'),
          'pose_direction_evidence':result.get('pose_direction_evidence'),
          'pose_vs_motion_conflict':result.get('pose_vs_motion_conflict',False),
          'full_window_supported':result['dive_observability']['window_supported'],
          'whole_record_review_required':result['review_required'],'human_verified':False,'identity_verified':False,
          'result_file':f'clips/{path.name}'}
        rows.append(row)
        if old['dive_direction']!=result['dive_direction']:changes.append(row)
        if old['dive_height']!=result['dive_height'] or old['auto_accepted']!=result['auto_accepted']:
            raise ValueError('Direction change unexpectedly alters height or whole-record QA')
        hashes.append({'file':str(path),'sha256':sha(path),'old_result':str(old_dir/path.name),'old_sha256':sha(old_dir/path.name)})
        if pose is not None:hashes[-1].update(pose_file=pose_path,pose_sha256=sha(pose_path))
    if len(rows)!=39:raise ValueError('Canonical cohort must have 39 clips')
    summary={'clips':len(rows),'old_unknown':sum(r['previous_direction']=='unknown' for r in rows),
         'new_unknown':sum(r['dive_direction']=='unknown' for r in rows),'direction_candidates':sum(r['dive_direction']!='unknown' for r in rows),
         'directions':dict(Counter(r['dive_direction'] for r in rows)),
         'recovered_direction_candidates':sum(r['previous_direction']=='unknown' and r['dive_direction']!='unknown' for r in rows),
         'changed_direction_labels':len(changes),
         'v2_unknown':sum(r['v2_direction']=='unknown' for r in rows),
         'changes_since_v2':[r for r in rows if r['v2_direction']!=r['dive_direction']],
         'unchanged_height_labels':39,
         'whole_record_QA_accepted':sum(r['auto_accepted'] for r in rows),'legacy_replays_match':39,
         'changes':changes,'rows':rows,'human_ground_truth_records':0,'accuracy_estimate':None,
         'note':'Recovering an observed direction does not establish person identity or complete dive height; all partial candidates remain review-required.'}
    save(out/'summary.json',summary);save(out/'goalkeeper_results_39.json',{'policy':a.policy,'rows':rows})
    save(out/'input_hashes.json',hashes)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['changes','rows','changes_since_v2']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

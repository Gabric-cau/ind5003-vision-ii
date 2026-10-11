"""Reassociate an unreviewed package twice after a global departure refinement.

Raw neural/pixel candidates, windows and outcome-independent inputs are retained.
Videos are regenerated whenever selected observations or forecasts change.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from scripts.run_ball_mask_continuity_review import (
    ROOT, load, save, sha, associate_with_seed, collision_geometry,
    render, make_report, ContinuityConfig, track_ball_continuity, motion_forecasts,
    MaskSeedConfig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();out=args.directory.resolve()
    if (out/'visual_review.json').exists():
        raise ValueError('Reassociation requires a package without published visual notes')
    protocol=load(out/'protocol.json');summary=load(out/'summary.json')
    validation=load(out/'validation.json')
    cfg=ContinuityConfig();mask_cfg=MaskSeedConfig(**protocol['mask_config'])
    for rel,h in protocol['code_sha256'].items():
        assert sha(out/'source_snapshot'/Path(rel).name)==h
        if Path(rel).as_posix()!='src/ball_continuity_v4.py':assert sha(ROOT/rel)==h
    before=out/'protocol_before_refinement.json'
    if not before.exists():
        before.write_bytes((out/'protocol.json').read_bytes())
        (out/'tests_before_refinement.json').write_bytes((out/'tests.json').read_bytes())
        (out/'source_snapshot/ball_continuity_v4_before_refinement.py').write_bytes(
            (out/'source_snapshot/ball_continuity_v4.py').read_bytes())
    else:
        assert load(before)==protocol,'Only an interrupted refinement can resume'
    refinements=[];rows=[]
    for rr in summary['rows']:
        cid=rr['clip_id'];r=load(out/'results'/f'{cid}.json')
        c=load(out/'inputs'/f'{cid}.json')
        neural=load(ROOT/'outputs/ball_detection_all_2026-10-11/detections'/f'{cid}.json')
        baseline=load(ROOT/'outputs/ball_detection_all_2026-10-11/results'/f'{cid}.json')
        assert sha(neural['video_path'])==c['source_sha256']==r['source_sha256']
        assert sha(ROOT/'outputs/ball_detection_all_2026-10-11/detections'/f'{cid}.json')==c['raw_neural_cache_sha256']
        v=r['variants']['mask_and_radius_35'];pool=c['candidates']
        pool_hash=hashlib.sha256(json.dumps(pool,sort_keys=True).encode()).hexdigest()
        assert pool_hash==c['candidate_hash']==r['candidate_hash']
        meta={f:{'coordinate_supported':c['camera']['transforms'].get(f) is not None} for f in pool}
        kw=dict(tracker=track_ball_continuity,config=cfg,frame_metadata=meta,
                collision_regions=collision_geometry(neural,{}))
        p,d=associate_with_seed(pool,neural['kick_frame'],c['anchors'],c['goal_center'],
                                v['seed'],v['seed_diagnostics'],**kw)
        repeat,rd=associate_with_seed(pool,neural['kick_frame'],c['anchors'],c['goal_center'],
                                v['seed'],v['seed_diagnostics'],**kw)
        assert p==repeat and d==rd
        assert hashlib.sha256(json.dumps(pool,sort_keys=True).encode()).hexdigest()==pool_hash
        allowed={(q['frame'],q['detection_id']) for ds in pool.values() for q in ds}
        assert all((q['frame'],q['detection_id']) in allowed for q in p)
        forecasts=motion_forecasts(p,neural['start_frame'],neural['end_frame_exclusive'],
                                  config=cfg,model_states=d.get('motion_states'))
        changed=p!=v['trajectory'] or forecasts!=v['forecasts']
        old_ids=[q['detection_id'] for q in v['trajectory']]
        v.update(trajectory=p,tracking=d,forecasts=forecasts,
                 effective_stationary_seed=[q for q in v['seed'] if q['frame']<=p[0]['frame']]
                 if p and d['tracking_status']=='candidate' else v['seed'])
        if changed:
            staging=out/'refinement_render'/str(uuid.uuid4())
            (staging/'media').mkdir(parents=True)
            media=render(neural,baseline,r,staging,mask_config=mask_cfg)
            for file in (staging/'media').iterdir():
                assert file.is_file() and file.name.startswith(cid)
                shutil.copy2(file,out/'media'/file.name)
            r['media']=media
        save(out/'results'/f'{cid}.json',r)
        newrow={**rr,'path_n':len(p),'end':p[-1]['frame'] if p else None,
                'status':d['tracking_status'],**r['media']}
        rows.append(newrow)
        for q in validation['rows']:
            if q['clip_id']==cid:q['decoded_video_frames']=r['media']['decoded_frames']
        refinements.append({'clip_id':cid,'two_association_runs_equal':True,
                            'candidate_pool_unchanged':True,'media_regenerated':changed,
                            'old_selected_ids':old_ids,
                            'new_selected_ids':[q['detection_id'] for q in p]})
        print(f'{cid}: {len(p)} candidates; regenerated={changed}',flush=True)
    protocol.update(continuity_config=asdict(cfg),
        association_refinement={'created_at':datetime.now(timezone.utc).isoformat(),
          'previous_protocol':'protocol_before_refinement.json',
          'record':'association_refinement.json',
          'scope':'all 39 clips twice; same raw candidates and source windows',
          'changed_setting':'global departure minimum speed 10 to 15 pixels per original source frame'})
    protocol['code_sha256']={rel:sha(ROOT/rel) for rel in protocol['code_sha256']}
    for rel in protocol['code_sha256']:(out/'source_snapshot'/Path(rel).name).write_bytes((ROOT/rel).read_bytes())
    save(out/'protocol.json',protocol);save(out/'association_refinement.json',refinements)
    save(out/'validation.json',validation)
    (out/'tests.json').write_bytes((ROOT/'tmp/ball_mask_final_tests.json').read_bytes())
    make_report(out,rows,complete=True)


if __name__=='__main__':main()

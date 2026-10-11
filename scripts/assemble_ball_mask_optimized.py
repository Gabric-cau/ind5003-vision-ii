"""Combine two fully replay-tested current seed hypotheses on identical inputs."""
from scripts.run_ball_mask_seed_experiment import *
from scripts.run_ball_mask_continuity_review import make_report
from src.ball_hypothesis_selection import select_early_flight,SelectionConfig
import copy


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    a=ROOT/'outputs/ball_mask_continuity_2026-10-11'
    b=ROOT/'outputs/ball_mask_continuity_v2_2026-10-11'
    inp=ROOT/'outputs/ball_detection_all_2026-10-11';out=args.output_dir.resolve()
    if out.exists():raise FileExistsError(out)
    for folder in ['results','inputs','media','source_snapshot']:(out/folder).mkdir(parents=True,exist_ok=True)
    protocol=load(b/'protocol.json');protocol['created_at']=datetime.now(timezone.utc).isoformat()
    protocol['current_hypothesis_runs']=[str(a),str(b)]
    protocol['selection_config']=asdict(SelectionConfig())
    protocol['selection_old_baseline_trajectory_used']=False
    protocol['association_recomputed_in_assembly']=False
    protocol['hypothesis_selection_note']='Two current source-only mask/departure runs, same frozen candidate pool; early interval only, no labels or historical baseline paths.'
    for f in [Path(__file__),ROOT/'src/ball_hypothesis_selection.py',ROOT/'scripts/build_ball_mask_review_evidence.py']:
        (out/'source_snapshot'/f.name).write_bytes(f.read_bytes())
        protocol['code_sha256'][str(f.relative_to(ROOT))]=sha(f)
    for f in (b/'source_snapshot').iterdir():(out/'source_snapshot'/f.name).write_bytes(f.read_bytes())
    save(out/'protocol.json',protocol)
    cfg=MaskSeedConfig(**protocol['mask_config']);rows=[];valid=[]
    for i,cid in enumerate(load(b/'summary.json')['rows'],1):
        cid=cid['clip_id'];print(f'[{i}/39] {cid}',flush=True)
        r0=load(a/'results'/f'{cid}.json');r1=load(b/'results'/f'{cid}.json')
        assert r0['candidate_hash']==r1['candidate_hash']
        assert r0['source_sha256']==r1['source_sha256']
        assert r0['deterministic_replay'] and r1['deterministic_replay']
        c=load(inp/'detections'/f'{cid}.json');baseline=load(inp/'results'/f'{cid}.json')
        proposals={'seed_before_contact':r0['variants']['mask_and_radius_35'],
          'seed_including_contact_neighborhood':r1['variants']['mask_and_radius_35']}
        name,diag=select_early_flight(proposals,c['kick_frame'],baseline['goal_reference_center'],'seed_including_contact_neighborhood')
        repeat,rd=select_early_flight(proposals,c['kick_frame'],baseline['goal_reference_center'],'seed_including_contact_neighborhood')
        assert name==repeat and diag==rd
        chosen=copy.deepcopy(proposals[name]);path=chosen['trajectory'];seed=chosen['seed']
        chosen['effective_stationary_seed']=[q for q in seed if q['frame']<=path[0]['frame']] if path and chosen['tracking']['tracking_status']=='candidate' else seed
        r=copy.deepcopy(r1);r['variants']['mask_and_radius_35']=chosen
        r['current_seed_hypotheses']=proposals;r['hypothesis_selection']=diag
        r['manual_ball_points_used']=False;r['old_ball_path_used_for_inference']=False
        cache=load(b/'inputs'/f'{cid}.json');pool=cache['candidates']
        ids={(q['frame'],q['detection_id']) for ds in pool.values() for q in ds}
        assert all((q['frame'],q['detection_id']) in ids for q in path)
        assert sha(c['video_path'])==c['source_sha256']
        (out/'inputs'/f'{cid}.json').write_bytes((b/'inputs'/f'{cid}.json').read_bytes())
        r['media']=render(c,baseline,r,out,mask_config=cfg)
        save(out/'results'/f'{cid}.json',r)
        old=baseline['variants']['detector_and_motion']['trajectory']
        row={'clip_id':cid,'seed_n':len(chosen['effective_stationary_seed']),
          'old_seed_n':len(baseline['stationary_seed']),'path_n':len(path),'end':path[-1]['frame'] if path else None,
          'old_end':old[-1]['frame'] if old else None,'status':chosen['tracking']['tracking_status'],
          'chosen_hypothesis':name,**r['media']}
        rows.append(row);valid.append({'clip_id':cid,'source_hash_matches':True,
          'identical_pool_across_hypotheses':True,'candidate_membership':True,
          'deterministic_association_replays':True,'deterministic_selection':True,
          'expected_video_frames':c['end_frame_exclusive']-c['start_frame'],
          'decoded_video_frames':r['media']['decoded_frames']})
        save(out/'validation.json',{'processed_clips':len(rows),'rows':valid,'accuracy_validated':False})
        make_report(out,rows,complete=len(rows)==39)
    print('Completed 39 hypotheses and decoded comparison videos',flush=True)


if __name__=='__main__':main()

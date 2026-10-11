"""Reproducible artifact checks, not a real-video accuracy evaluation."""
from pathlib import Path
import hashlib,json,sys,unittest,importlib.util
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'outputs/goal_event_evidence_2026-10-10'


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    suite=unittest.TestSuite()
    files=['test_goal_event_tracker.py','test_goal_plane_rectification.py','test_goal_geometry.py',
           'test_goalkeeper_bounded.py','test_goalkeeper_observation_guard.py','test_goalkeeper_qa_gap.py',
           'test_ball_expired_paths.py','test_ball_outcome.py','test_motion_stabilization.py']
    for filename in files:
        path=ROOT/'tests'/filename;spec=importlib.util.spec_from_file_location(path.stem,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():raise AssertionError('Regression tests failed')
    rows=[]
    for group,folder in [('development',BASE),('additional_frozen_probe',BASE/'additional_probe')]:
        for file in sorted((folder/'release_v6').glob('*.json')):
            if file.name=='summary.json':continue
            record=load(file);cache=load(Path(record['cache_path']));v=record['variants']['detector_motion_turn']
            assert sha(Path(record['video_path']))==cache['source_sha256']
            assert not record['outcome_labels_read'] and not record['human_verified'] and not record['accuracy_validated']
            path=v['trajectory'];assert len({p['frame'] for p in path})==len(path)
            assert all(cache['start_frame']<=p['frame']<cache['end_frame_exclusive'] for p in path)
            assert all(p.get('stabilized_xy') is not None for p in path)
            assert all(not p.get('duplicate_frame',False) for p in path[1:])
            assert v['events']['outcome']=='unknown' and not v['events']['whole_ball_crossing_established']
            for rel,digest in record['code_sha256'].items():assert sha(ROOT/rel)==digest
            replay=None
            if group=='development':
                previous=load(BASE/'tracking_v6'/file.name)['variants']['detector_motion_turn']['trajectory']
                replay=[(d['frame'],d['detection_id']) for d in path]==[(d['frame'],d['detection_id']) for d in previous]
                assert replay
            rows.append({'clip_id':record['clip_id'],'group':group,'source_sha256_verified':True,
                'camera_status':cache['camera']['status'],'decoded_detection_window_frames':cache['end_frame_exclusive']-cache['start_frame'],
                'observations':len(path),'incoming':v['tracking'].get('incoming_observations'),
                'outgoing':v['tracking'].get('outgoing_observations'),'event_cue':v['events'],
                'development_replay_identical':replay,'accuracy_validated':False})
    videos=load(BASE/'videos_release/summary.json')
    for video in videos['rows']:
        p=ROOT/Path(video['output'])
        assert sha(p)==video['output_sha256'] and video['decoded_frames']==82
    review=load(BASE/'assistant_visual_review.json')
    report={'tests_passed':result.testsRun,'test_modules':files,'clips':rows,
        'video_full_decode_frames':sum(r['decoded_frames'] for r in videos['rows']),
        'video_outputs':[str((ROOT/Path(r['output'])).resolve()) for r in videos['rows']],
        'visual_review':review,'human_ground_truth_available':False,'automatic_binary_outcomes_completed':0,
        'baseline_replaced':False,'requirements_changed':False,
        'runtime_notes':['A high-memory run exited without useful output; batch size 1 completed the additional detections.',
                         'A concurrent test import hit an OpenBLAS allocation error; single-threaded isolated rerun passed.'],
        'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/render_goal_event_video.py',
            ROOT/'scripts/cache_goal_event_detections.py',ROOT/'scripts/cache_goal_event_people.py',
            ROOT/'src/goal_event_tracker_v2.py',ROOT/'src/keeper_kit_evidence.py',ROOT/'scripts/probe_goal_event_tracking.py']}}
    (BASE/'experiment_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'tests_passed':result.testsRun,'clips':len(rows),'source_hashes_verified':len(rows),
                     'video_full_decode_frames':report['video_full_decode_frames'],'automatic_binary_outcomes_completed':0}))


if __name__=='__main__':main()

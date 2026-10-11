"""Structural/media integrity and targeted regression checks, not accuracy."""
from pathlib import Path
import argparse,csv,hashlib,io,json,sys,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def audit(out):
    data=read(out/'camera_timings_candidates.json');media=read(out/'media_audit.json');records={r['clip_id']:r for r in media['records']}
    errors=[];total=0;frames=0
    for c in data['clips']:
        cid=c['clip_id'];end=0
        for s in c['segments']:
            assert s['start_frame']==end and s['end_frame_exclusive']>end
            assert abs(s['start_time_sec']-s['start_frame']/c['fps'])<1e-7
            assert abs(s['end_time_sec']-s['end_frame_exclusive']/c['fps'])<1e-7
            assert s['verified_task_class']=='unknown' and not s['human_verified'] and s['review_required']
            assert abs(s['source_match_start_sec']-c['manifest_offset_sec']-s['start_time_sec'])<1e-7
            assert (out/'review'/s['thumbnail']).exists()
            end=s['end_frame_exclusive'];total+=1
        assert end==c['frame_count'];frames+=end
        assert sha(Path(c['raw_video_path']))==c['source_sha256']
        r=records[cid];assert r['source_sha256']==c['source_sha256'] and r['decoded_frames']==end and r['full_stream_decode_pass']
        assert sha(out/'review'/c['video'])==r['output_sha256']
        for category,p in c['task_category_presence'].items():
            assert p['absence_verified'] is False
            assert p['segment_ids']==[s['segment_id'] for s in c['segments'] if s['task_class_candidate']==category]
    with (out/'camera_timings_candidates.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==total==data['summary']['segments']
    for r in rows:assert r['human_verified']=='False'
    patterns=('test_camera_timing_v2.py','test_goal_geometry.py','test_goal_plane_rectification.py','test_goal_event_tracker.py')
    suite=unittest.TestSuite(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=p) for p in patterns)
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (out/'targeted_tests.log').write_text(stream.getvalue(),encoding='utf-8')
    assert result.wasSuccessful(),stream.getvalue()
    action=[read(p) for p in (out/'action_v3').glob('*.json')]
    sources=['src/camera_timing_v2.py','src/goal_geometry.py','scripts/run_camera_timing_analysis.py','scripts/camera_timing_features.py',
             'scripts/camera_timing_context.py','scripts/export_camera_timings.py','scripts/prepare_camera_review_media.py',
             'assets/camera_review/index.html','assets/camera_review/app.js','assets/camera_review/style.css']
    report={'scope':'canonical manifest clips only; no independently verified camera labels',
        'clips':len(data['clips']),'segments':total,'original_frames':frames,'original_hashes_unchanged':len(data['clips']),
        'review_videos_full_decode_pass':len(records),'contiguous_exclusive_intervals':True,'csv_json_consistent':True,
        'targeted_tests_passed':result.testsRun,'targeted_test_patterns':patterns,
        'full_suite_status':'Full unittest discovery attempted: 184 passed, one import error because pytest unavailable before final extra test. Not a clean full-suite pass.',
        'slow_motion_alignment_segments_tested':sum(len(d['evidence']) for d in action),
        'slow_motion_supported_segments':sum(v['status']=='slow_motion_candidate' for d in action for v in d['evidence'].values()),
        'human_verified_count':0,'accuracy_estimated':False,'source_outcome_labels_read':False,
        'source_code_sha256':{name:sha(ROOT/name) for name in sources},
        'known_limits':['Role appearance can confuse teammates.','Partial goal visibility and changing framing cause missed camera labels.',
            'Six 2D joint angles are not viewpoint invariant; cropped limbs reject slow-motion alignment.',
            'The existing clips can end during replays; not_detected is not confirmed absent.',
            'One declared goalkeeper pose cache (Kramaric) is missing; no replacement identity label invented.',
            'Unknown camera intrinsics: angle_45_right is a broad right-oblique candidate, not a measured 45 degrees.']}
    (out/'artifact_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/camera_timing_2026-10-10/release');a=p.parse_args();audit(a.output_dir.resolve())

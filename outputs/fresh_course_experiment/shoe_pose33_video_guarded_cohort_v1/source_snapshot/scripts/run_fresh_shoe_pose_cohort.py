"""Run the fixed 33-point VIDEO diagnostic on the 39 course-event cohort."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_fresh_shoe_pose_experiment import main as run_event, read, sha, write


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('clips-dir', 'pose-dir', 'foot-dir', 'ball-dir', 'model', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use an empty cohort output directory')
    manifest_path = args.clips_dir / 'source_manifest.json'
    sources = read(manifest_path)['clips']
    if len(sources) != 39 or len({row['clip_id'] for row in sources}) != 39:
        raise ValueError('Cohort requires the exact distinct 39-event source manifest')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_names = ('scripts/run_fresh_shoe_pose_cohort.py', 'scripts/run_fresh_shoe_pose_experiment.py',
        'src/mediapipe_shoe_evidence.py', 'src/kicking_foot_strike_window.py',
        'src/kicking_foot_evidence.py', 'src/kicking_foot_evidence_v2.py',
        'src/pose_refinement.py', 'scripts/run_fresh_foot_experiment.py',
        'docs/FRESH_SHOE_POSE33_PROTOCOL_2026-10-13.md', 'requirements-pose33.txt')
    code_hashes = {}
    for name in snapshot_names:
        dest = args.output_dir / 'source_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
        code_hashes[name] = sha(dest)
    protocol = dict(schema_version='fresh-shoe-pose33-cohort-v1', events=39,
        source_manifest_sha256=sha(manifest_path), model_sha256=sha(args.model), code_sha256=code_hashes,
        running_mode='VIDEO', method_registered_before_cohort_results=True,
        development_pilot_previously_inspected=True, development_data_previously_inspected=True,
        reference_labels_read=False, human_verified=False, accuracy_validated=False, held_out_test=False,
        prediction_scope='Source-only shoe-swing/contact-region diagnostic; no direct shoe-contact truth')
    write(args.output_dir / 'protocol.json', protocol)
    rows = []
    started = time.monotonic()
    for source in sources:
        cid = source['clip_id']
        original = read(args.foot_dir / 'results' / f'{cid}.json')
        event_dir = args.output_dir / 'events' / cid
        row = dict(clip_id=cid, candidate='unknown', shoe_region_candidate='unknown',
                   status='independent_ball_contact_unavailable', actual_inference_calls=0,
                   human_verified=False, accuracy_validated=False)
        if original.get('new_neural_inference'):
            result = run_event(['--clips-dir', str(args.clips_dir), '--foot-dir', str(args.foot_dir),
                '--pose-dir', str(args.pose_dir),
                '--ball-dir', str(args.ball_dir), '--model', str(args.model), '--clip-id', cid,
                '--running-mode', 'VIDEO', '--output-dir', str(event_dir)])
            row.update(candidate=result['source_coordinate_shoe_swing_candidate'],
                       shoe_region_candidate=result['source_coordinate_shoe_contact_region_candidate'],
                       status='ambiguous_contact_ball_identity' if result['local_ball_identity_ambiguous_frames']
                              else result['evidence']['shoe_centroid_motion']['status'],
                       actual_inference_calls=result['actual_inference_calls'],
                       accepted_roi_pose_calls=result['accepted_roi_pose_calls'],
                       result_sha256=sha(event_dir / 'result.json'))
        else:
            event_dir.mkdir(parents=True, exist_ok=True)
            write(event_dir / 'unavailable.json', dict(row, source_mapping=source,
                upstream_neural_producer_sha256=sha(args.foot_dir / 'results' / f'{cid}.json')))
        rows.append(row)
        write(args.output_dir / 'progress.json', dict(processed=len(rows), rows=rows))
    summary = dict(events=39, rows=rows, candidate_counts=dict(Counter(row['candidate'] for row in rows)),
        shoe_region_candidate_counts=dict(Counter(row['shoe_region_candidate'] for row in rows)),
        status_counts=dict(Counter(row['status'] for row in rows)),
        actual_inference_calls=sum(row['actual_inference_calls'] for row in rows),
        neural_events=sum(row['actual_inference_calls'] > 0 for row in rows),
        elapsed_seconds=time.monotonic() - started, reference_labels_read=False,
        human_verified=False, accuracy_validated=False, held_out_test=False,
        interpretation='Development diagnostic availability, not anatomical/contact accuracy')
    write(args.output_dir / 'summary.json', summary)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    main()

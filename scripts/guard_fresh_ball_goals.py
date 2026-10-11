"""Apply keeper-occlusion guard to selected fresh producers, without new NN."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_fresh_ball_experiment import producer_record, replay_provenance
from scripts.run_fresh_ball_experiment import load, save
from scripts.run_fresh_phase_experiment import _bound_contact
from scripts.select_fresh_ball_producers import copy_unchanged
from src.final_results import sha256
from src.fresh_cache_provenance import validate_cached_neural_producer
from src.fresh_goal_guard import guard_keeper_occluded_goal


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clips-dir', type=Path, required=True)
    parser.add_argument('--pose-dir', type=Path, required=True)
    parser.add_argument('--ball-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--clips', nargs='+')
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError('Use a new goal-guard analysis directory')
    manifest_path = args.clips_dir/'source_manifest.json'
    manifest_hash = sha256(manifest_path)
    records = {row['clip_id']: row for row in load(manifest_path)['clips']}
    ids = args.clips or list(records)
    if len(ids) != len(set(ids)) or any(cid not in records for cid in ids) or (not args.clips and len(ids) != 39):
        raise ValueError('Goal guard needs a unique source-bound cohort of 39 or explicit clips')
    files = ('scripts/guard_fresh_ball_goals.py', 'src/fresh_goal_guard.py',
        'src/fresh_cache_provenance.py', 'scripts/replay_fresh_ball_experiment.py',
        'scripts/select_fresh_ball_producers.py', 'scripts/run_fresh_phase_experiment.py')
    hashes = {name: sha256(ROOT/name) for name in files}
    save(args.output_dir/'protocol.json', {'source_manifest_sha256': manifest_hash, 'cohort_clip_ids': ids,
        'code_sha256': hashes, 'new_neural_inference': False, 'new_neural_inference_this_run': False,
        'goal_rule_reanalysis_this_run': True, 'trajectory_reanalysis_this_run': False,
        'contact_artifacts_policy': 'byte_exact_original_selected_neural_producer',
        'contact_file_new_neural_inference_field': 'original_neural_producer_receipt',
        'reference_labels_used_for_prediction': False, 'human_verified': False, 'accuracy_validated': False})
    if (args.ball_dir/'origins').is_dir():
        shutil.copytree(args.ball_dir/'origins', args.output_dir/'origins')
    rows = []
    for cid in ids:
        source = records[cid]
        filename = Path(source['clip_filename'])
        if filename.name != str(filename) or sha256(args.clips_dir/filename) != source['clip_sha256']:
            raise ValueError('Goal guard source video hash mismatch')
        pose_path = args.pose_dir/f'{cid}.json'
        pose = load(pose_path)
        contact, binding = _bound_contact(args.ball_dir, source, pose_path, pose, manifest_hash)
        result_path = args.ball_dir/'results'/f'{cid}.json'
        original = load(result_path)
        if contact['contact_frame'] is None:
            result = deepcopy(original)
            result.update(new_neural_inference=False, new_neural_inference_this_run=False,
                          goal_rule_reanalysis_this_run=False, trajectory_reanalysis_this_run=False)
        else:
            raw_path = args.ball_dir/'detections'/f'{cid}.json'
            producer = producer_record(result_path, raw_path, original)
            identity = {**deepcopy(original['inference_identity']),
                        'settings': {'analysis_mode': 'source_only_goal_keeper_occlusion_guard_v4'},
                        'code_sha256': hashes, 'source_cache_sha256': producer['detections_sha256'],
                        'producer_receipt_sha256': producer['receipt_sha256']}
            metadata = replay_provenance(identity, producer)
            copy_unchanged(result_path, args.output_dir/producer['receipt_file'])
            copy_unchanged(raw_path, args.output_dir/producer['detections_file'])
            validate_cached_neural_producer({**metadata, 'clip_id': cid}, args.output_dir)
            result = deepcopy(original)
            result['rule_decision'], result['goal_occlusion_guard'] = guard_keeper_occluded_goal(original, pose)
            search_path = args.output_dir/'contact_search'/f'{cid}.json'
            copy_unchanged(args.ball_dir/'contact_search'/f'{cid}.json', search_path)
            event = load(args.ball_dir/'event_windows'/f'{cid}.json')
            event.update(**metadata)
            event_path = args.output_dir/'event_windows'/f'{cid}.json'
            save(event_path, event)
            result.update(**metadata, contact_search_sha256=sha256(search_path),
                event_cache_sha256=sha256(event_path), goal_rule_reanalysis_this_run=True,
                trajectory_reanalysis_this_run=False, contact_binding=binding)
            save(args.output_dir/'analysis_contacts'/f'{cid}.json', {**metadata, 'clip_id': cid,
                'source_sha256': source['clip_sha256'], 'pose_sha256': sha256(pose_path),
                'contact': result['independent_contact_search']['selected_candidate'],
                'independent_contact_search': result['independent_contact_search'],
                'ball_departure': result.get('ball_departure'), 'source_mapping': source, 'fps': source['fps']})
        for folder, suffix in (('contacts', '.json'), ('inputs', '.json'), ('media', '.jpg'), ('selection_records', '.json')):
            path = args.ball_dir/folder/f'{cid}{suffix}'
            if path.is_file():
                copy_unchanged(path, args.output_dir/folder/path.name)
        save(args.output_dir/'results'/f'{cid}.json', result)
        row = {'clip_id': cid, 'status': result['status'],
               'original_rule_outcome': original['rule_decision']['rule_outcome'],
               'rule_outcome': result['rule_decision']['rule_outcome'],
               'keeper_overlap_frame': (result.get('goal_occlusion_guard') or {}).get('first_keeper_overlap_frame'),
               'changed': original['rule_decision']['rule_outcome'] != result['rule_decision']['rule_outcome'],
               'new_neural_inference': False}
        rows.append(row)
        print(json.dumps(row), flush=True)
    save(args.output_dir/'summary.json', {'rows': rows, 'processed_clips': len(rows), 'requested_clips': len(ids),
        'rule_outcome_counts': dict(Counter(row['rule_outcome'] for row in rows)),
        'changed_clips': sum(row['changed'] for row in rows),
        'new_neural_inference_clips': 0, 'all_requested_clips_processed': True,
        'human_verified': False, 'accuracy_validated': False})


if __name__ == '__main__':
    main()

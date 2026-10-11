"""Select original n/m contact artifacts without rerunning neural inference."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_fresh_ball_experiment import load, save
from scripts.run_fresh_phase_experiment import _bound_contact
from src.final_results import sha256
from src.fresh_contact_selection import ContactSelectionPolicy, select_contact_producer
from src.fresh_event_analysis import contact_window


def verified_producer(directory, source, pose_path, pose, manifest_hash):
    cid = source['clip_id']
    result_path = directory/'results'/f'{cid}.json'
    result = load(result_path)
    identity = result.get('inference_identity') or {}
    if (result.get('clip_id') != cid or identity.get('source_sha256') != source['clip_sha256'] or
            identity.get('pose_sha256') != sha256(pose_path) or
            identity.get('source_manifest_sha256') != manifest_hash or
            result.get('reference_labels_used_for_prediction') is not False):
        raise ValueError(f'Producer source/pose/manifest provenance mismatch: {cid}')
    files = {'results': result_path}
    if result.get('new_neural_inference') is not True:
        interval, reason = contact_window(pose, source)
        if interval is not None or result.get('status') != 'abstain_contact' or result.get('reason') != reason:
            raise ValueError(f'Missing neural cache is not explained by a verified scene rejection: {cid}')
    else:
        bindings = {'detections': 'source_cache_sha256', 'contact_search': 'contact_search_sha256'}
        if result.get('status') == 'candidate_event':
            bindings['event_windows'] = 'event_cache_sha256'
        for folder, field in bindings.items():
            path = directory/folder/f'{cid}.json'
            if not path.is_file() or sha256(path) != result.get(field):
                raise ValueError(f'Producer artifact missing or changed: {cid}: {folder}')
            files[folder] = path
        raw = load(files['detections'])
        if (raw.get('inference_identity') != identity or raw.get('fresh_model_inference') is not True
                or raw.get('source_sha256') != source['clip_sha256']):
            raise ValueError('Producer raw observations do not belong to the original fresh inference')
    bounded, binding = _bound_contact(directory, source, pose_path, pose, manifest_hash)
    candidate = load(directory/'contact_search'/f'{cid}.json')['selected_candidate'] if bounded['contact_frame'] is not None else None
    for folder in ('contacts', 'inputs', 'media'):
        path = directory/folder/f'{cid}.jpg' if folder == 'media' else directory/folder/f'{cid}.json'
        if path.is_file():
            files[folder] = path
    return {'result': result, 'candidate': candidate, 'binding': binding, 'files': files}


def copy_unchanged(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    digest = sha256(source)
    if sha256(target) != digest:
        raise AssertionError('Selected producer artifact bytes changed')
    return digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clips-dir', type=Path, required=True)
    parser.add_argument('--pose-dir', type=Path, required=True)
    parser.add_argument('--n-dir', type=Path, required=True)
    parser.add_argument('--m-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--clips', nargs='+')
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError('Use a new contact-selection bundle directory')
    manifest_path = args.clips_dir/'source_manifest.json'
    manifest = load(manifest_path)
    manifest_hash = sha256(manifest_path)
    records = {row['clip_id']: row for row in manifest['clips']}
    ids = args.clips or list(records)
    if len(records) != len(manifest['clips']) or len(ids) != len(set(ids)) or any(cid not in records for cid in ids):
        raise ValueError('Source manifest and selected IDs must be unique')
    if not args.clips and len(ids) != 39:
        raise ValueError('A complete selection bundle requires exactly 39 course clips')
    producers = {'n': args.n_dir, 'm': args.m_dir}
    policy = ContactSelectionPolicy()
    verified = {}
    for cid in ids:
        source = records[cid]
        filename = Path(source['clip_filename'])
        if filename.name != str(filename) or sha256(args.clips_dir/filename) != source['clip_sha256']:
            raise ValueError('Source clip hash or local filename mismatch')
        pose_path = args.pose_dir/f'{cid}.json'
        pose = load(pose_path)
        verified[cid] = {label: verified_producer(directory, source, pose_path, pose, manifest_hash)
                         for label, directory in producers.items()}
    files = ('scripts/select_fresh_ball_producers.py', 'src/fresh_contact_selection.py',
             'scripts/run_fresh_phase_experiment.py', 'src/fresh_cache_provenance.py')
    code_hashes = {name: sha256(ROOT/name) for name in files}
    save(args.output_dir/'protocol.json', {
        'cohort_clip_ids': ids, 'source_manifest_sha256': manifest_hash,
        'policy': policy.__dict__, 'code_sha256': code_hashes,
        'new_neural_inference_this_run': False, 'new_neural_inference': False,
        'copied_existing_inference': True,
        'source_result_new_neural_inference_field': 'unchanged_original_neural_producer_receipt',
        'outcome_fields_used_for_selection': False, 'reference_labels_used_for_prediction': False,
        'human_verified': False, 'accuracy_validated': False})
    for label, directory in producers.items():
        copy_unchanged(directory/'protocol.json', args.output_dir/'origins'/label/'protocol.json')
    rows = []
    for cid in ids:
        alternatives = verified[cid]
        decision = select_contact_producer({label: row['candidate'] for label, row in alternatives.items()}, policy)
        origins = {}
        for label, row in alternatives.items():
            original = args.output_dir/'origins'/label/'results'/f'{cid}.json'
            original_hash = copy_unchanged(row['files']['results'], original)
            origins[label] = {'receipt_file': str(original.relative_to(args.output_dir)),
                'receipt_sha256': original_hash,
                'inference_identity': row['result']['inference_identity'],
                'original_new_neural_inference': row['result']['new_neural_inference'],
                'source_cache_sha256': row['result'].get('source_cache_sha256'),
                'contact_candidate_supported': row['candidate'] is not None,
                'contact_binding': row['binding']}
        selected = decision['selected_label']
        hashes = {}
        if selected:
            for folder, source_path in alternatives[selected]['files'].items():
                target = args.output_dir/folder/source_path.name
                hashes[str(target.relative_to(args.output_dir))] = copy_unchanged(source_path, target)
        else:
            source, identity = records[cid], alternatives['n']['result']['inference_identity']
            rejected = {'clip_id': cid, 'status': 'abstain_contact_selection', 'reason': decision['reason'],
                'inference_identity': {key: identity[key] for key in ('source_sha256', 'pose_sha256', 'source_manifest_sha256')},
                'source_mapping': source, 'new_neural_inference': False,
                'new_neural_inference_this_run': False, 'copied_existing_inference': False,
                'rule_decision': {'rule_outcome': 'unknown'}, 'trajectory': [], 'final_position': None,
                'keeper': {'status': 'abstain', 'dive_direction': 'unknown', 'dive_height': 'unknown'},
                'reference_labels_used_for_prediction': False, 'human_verified': False, 'accuracy_validated': False}
            save(args.output_dir/'results'/f'{cid}.json', rejected)
            save(args.output_dir/'contacts'/f'{cid}.json', {**rejected,
                'source_sha256': identity['source_sha256'], 'pose_sha256': identity['pose_sha256'],
                'contact': None, 'independent_contact_search': {'selected_candidate': None}})
        selection = {**decision, 'clip_id': cid, 'original_producers': origins,
            'selected_artifact_sha256': hashes, 'copied_existing_inference': selected is not None,
            'new_neural_inference_this_run': False, 'human_verified': False, 'accuracy_validated': False}
        save(args.output_dir/'selection_records'/f'{cid}.json', selection)
        row = {'clip_id': cid, 'selected_label': selected, 'reason': decision['reason'],
               'contact_frame': alternatives[selected]['candidate']['contact_frame'] if selected else None,
               'contact_bracket_frames': alternatives[selected]['candidate']['contact_bracket_frames'] if selected else None,
               'contact_associated_track_id': alternatives[selected]['candidate']['contact_associated_track_id'] if selected else None}
        rows.append(row)
        print(json.dumps(row), flush=True)
    save(args.output_dir/'summary.json', {'rows': rows, 'processed_clips': len(rows), 'requested_clips': len(ids),
        'selected_producer_counts': dict(Counter(row['selected_label'] or 'unknown' for row in rows)),
        'reason_counts': dict(Counter(row['reason'] for row in rows)),
        'new_neural_inference_clips_this_run': 0, 'all_requested_clips_processed': True,
        'human_verified': False, 'accuracy_validated': False})


if __name__ == '__main__':
    main()

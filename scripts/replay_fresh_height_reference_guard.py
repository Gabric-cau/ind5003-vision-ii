"""Reject invalid height references in frozen measurements without inference."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.final_results import sha256
from src.fresh_keeper_height import guard_observed_height_reference


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def guard_cached_height(data, source, manifest_sha):
    identity = data.get('inference_identity') or {}
    if (data.get('clip_id') != source['clip_id']
            or identity.get('source_clip_sha256') != source['clip_sha256']
            or identity.get('source_manifest_sha256') != manifest_sha):
        raise ValueError('Cached height source or manifest mismatch')
    if any(data.get(flag) is not False for flag in
           ('human_verified', 'accuracy_validated', 'reference_labels_read')):
        raise ValueError('Replay requires unverified source-only height measurements')
    if data.get('physical_height_metres') is not None:
        raise ValueError('Image-plane replay cannot verify physical height')
    result = deepcopy(data)
    signal = data.get('observed_image_plane_signal')
    candidate = data.get('observed_image_plane_height_candidate', 'unknown')
    if signal is None:
        if candidate != 'unknown':
            raise ValueError('Height category has no supporting observed signal')
        return result
    if signal.get('candidate') != candidate:
        raise ValueError('Cached height category disagrees with observed signal')
    result['observed_image_plane_signal'] = guard_observed_height_reference(signal)
    result['observed_image_plane_height_candidate'] = result['observed_image_plane_signal']['candidate']
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--height-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError('Use a new directory to preserve frozen height measurements')
    manifest_sha = sha256(args.source_manifest)
    sources = read(args.source_manifest)['clips']
    ids = [source['clip_id'] for source in sources]
    if len(ids) != 39 or len(set(ids)) != 39:
        raise ValueError('Height replay needs exactly 39 canonical source events')
    code_names = ('src/fresh_keeper_height.py', 'scripts/replay_fresh_height_reference_guard.py')
    code_hashes = {name: sha256(ROOT/name) for name in code_names}
    prepared = []
    for source in sources:
        clip_id = source['clip_id']
        if Path(clip_id).name != clip_id:
            raise ValueError('Canonical clip ID must be a basename')
        input_path = args.height_dir/(clip_id+'.json')
        input_sha = sha256(input_path)
        data = read(input_path)
        result = guard_cached_height(data, source, manifest_sha)
        if sha256(input_path) != input_sha:
            raise ValueError('Frozen height input changed during replay')
        result['original_height_input'] = dict(path=str(input_path.resolve()), sha256=input_sha,
                                              inference_identity=data['inference_identity'])
        result['inference_identity'] = dict(data['inference_identity'], code_sha256=code_hashes,
                                            original_height_sha256=input_sha)
        result['reference_guard_replay'] = dict(schema_version='fresh-height-reference-guard-v2',
            measured_values_recomputed=False, original_pose_phase_recomputed=False,
            original_strict_height_recomputed=False, new_neural_inference=False,
            reference_labels_read=False, human_verified=False, accuracy_validated=False)
        prepared.append((source, input_path, input_sha, data, result))
    if sha256(args.source_manifest) != manifest_sha:
        raise ValueError('Source manifest changed during replay')
    args.output_dir.mkdir(parents=True)
    rows, inputs = [], []
    for source, input_path, input_sha, data, result in prepared:
        clip_id = source['clip_id']
        write(args.output_dir/(clip_id+'.json'), result)
        before = data.get('observed_image_plane_height_candidate', 'unknown')
        after = result.get('observed_image_plane_height_candidate', 'unknown')
        signal = result.get('observed_image_plane_signal') or {}
        rows.append(dict(clip_id=clip_id, before=before, after=after, category_withdrawn=before != after,
            reference_depth_assumption_failed=signal.get('reference_depth_assumption_failed'),
            observed_maximum_torso_clearance_body=signal.get('observed_maximum_torso_clearance_body'),
            strict_height_candidate=result.get('strict_height_candidate', 'unknown'),
            ordinal_pose_height_candidate=result.get('ordinal_pose_height_candidate', 'unknown')))
        inputs.append(dict(clip_id=clip_id, path=str(input_path.resolve()), sha256=input_sha))
    protocol = dict(schema_version='fresh-height-reference-guard-v2', cohort_clip_ids=ids,
        source_manifest_sha256=manifest_sha, code_sha256=code_hashes, input_height_artifacts=inputs,
        scope='Reference-validity guard on frozen measured signals; no new height measurement',
        new_neural_inference=False, reference_labels_read=False, human_verified=False,
        accuracy_validated=False, original_frozen_outputs_overwritten=False)
    write(args.output_dir/'protocol.json', protocol)
    summary = dict(events=len(rows), rows=rows,
        observed_candidate_counts_before=dict(Counter(row['before'] for row in rows)),
        observed_candidate_counts_after=dict(Counter(row['after'] for row in rows)),
        withdrawn_category_count=sum(row['category_withdrawn'] for row in rows),
        new_category_count=sum(row['before'] == 'unknown' and row['after'] != 'unknown' for row in rows),
        invalid_reference_category_count_after=sum(row['after'] != 'unknown'
            and row['reference_depth_assumption_failed'] is True for row in rows),
        strict_candidate_counts=dict(Counter(row['strict_height_candidate'] for row in rows)),
        ordinal_candidate_counts=dict(Counter(row['ordinal_pose_height_candidate'] for row in rows)),
        all_measured_values_preserved=True, new_neural_inference=False, human_verified=False,
        accuracy_validated=False, physical_height_metres_estimated=False)
    write(args.output_dir/'summary.json', summary)
    print(json.dumps({key: value for key, value in summary.items() if key != 'rows'}))
    return summary


if __name__ == '__main__':
    main()

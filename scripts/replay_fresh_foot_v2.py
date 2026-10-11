"""Replay existing paired neural poses; no model inference and no label lookup."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_fresh_foot_experiment import contact_payload, selected_ball_observations
from scripts.run_fresh_phase_experiment import _bound_contact
from scripts.run_fresh_pitch_experiment import _hash, _pose_input, _write
from src.kicking_foot_evidence import kicking_foot_evidence
from src.kicking_foot_evidence_v2 import kicking_foot_evidence_v2


def _equivalent(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equivalent(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_equivalent(a, b) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)):
        return bool(np.isclose(left, right, rtol=1e-10, atol=1e-10))
    return left == right


def cached_samples(records):
    normal, mirror = [], []
    seen = set()
    for row in records:
        key = (row['frame'], row['reflected'])
        if key in seen:
            raise ValueError('Duplicate neural source-frame/view record')
        seen.add(key)
        selected = row.get('selected')
        if selected is None:
            continue
        if selected.get('frame') != row['frame'] or row.get('selection_reason') != 'accepted':
            raise ValueError('Cached pose selection/source frame mismatch')
        sample = dict(frame=row['frame'], bbox=selected['bbox'], keypoints=selected['keypoints'],
                      confidence=selected['keypoint_confidences'])
        (mirror if row['reflected'] else normal).append(sample)
    return normal, mirror


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('clips-dir', 'pose-dir', 'ball-dir', 'foot-dir', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use a new empty output directory to preserve experiment artifacts')
    if any(args.output_dir.resolve() == path.resolve() for path in
           (args.clips_dir, args.pose_dir, args.ball_dir, args.foot_dir)):
        parser.error('Output must be distinct from all inference inputs')
    manifest_path = args.clips_dir/'source_manifest.json'
    source_manifest_sha = _hash(manifest_path)
    sources = json.loads(manifest_path.read_text())['clips']
    if len(sources) != 39 or len({row['clip_id'] for row in sources}) != 39:
        raise ValueError('Fresh foot replay needs the exact 39 distinct course events')
    paths = ('src/kicking_foot_evidence_v2.py', 'scripts/replay_fresh_foot_v2.py',
             'docs/FRESH_FOOT_V2_PROTOCOL_2026-10-12.md', 'src/kicking_foot_evidence.py',
             'scripts/run_fresh_phase_experiment.py', 'scripts/run_fresh_pitch_experiment.py')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'results').mkdir()
    hashes = {}
    for name in paths:
        target = args.output_dir/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
        hashes[name] = _hash(target)
    protocol = dict(schema_version='fresh-foot-cpu-replay-v2', source_manifest_sha256=source_manifest_sha,
        code_sha256=hashes, method_registered_before_results=True,
        development_data_previously_inspected=True, new_neural_inference=False,
        reference_labels_read=False, human_verified=False, accuracy_validated=False, held_out_test=False,
        upstream_foot_protocol_sha256=_hash(args.foot_dir/'protocol.json'))
    _write(args.output_dir/'protocol.json', protocol)
    rows = []
    for source in sources:
        cid = source['clip_id']
        producer_path = args.foot_dir/'results'/(cid+'.json')
        producer = json.loads(producer_path.read_text())
        pose_path = args.pose_dir/(cid+'.json')
        video = args.clips_dir/source['clip_filename']
        if _hash(video) != source['clip_sha256']:
            raise ValueError('Fresh foot replay source video SHA mismatch')
        original = producer['inference_identity']
        expected = dict(pose_sha256=_hash(pose_path), source_sha256=source['clip_sha256'],
                        source_manifest_sha256=source_manifest_sha)
        if any(original.get(key) != value for key, value in expected.items()):
            raise ValueError('Foot neural producer/source/pose identity mismatch')
        result = dict(clip_id=cid, kicking_foot_candidate='unknown', status=producer['status'],
                      new_neural_inference=False, human_verified=False, accuracy_validated=False,
                      held_out_test=False, reference_labels_used_for_prediction=False,
                      shoe_ball_contact_verified=False,
                      original_neural_producer=dict(path=str(producer_path.resolve()),
                        sha256=_hash(producer_path), new_neural_inference=producer['new_neural_inference'],
                        inference_identity=original), source_mapping=source,
                      inference_identity=dict(**expected, code_sha256=hashes),
                      v1_kicking_foot_candidate=producer['kicking_foot_candidate'])
        if producer.get('new_neural_inference'):
            if (producer.get('upstream_scene_rejections') != []
                    or producer.get('model_task_validation', {}).get('actual_task') != 'pose'
                    or producer.get('reference_labels_used_for_prediction') is not False):
                raise ValueError('Cached foot inference lacks qualified scene/pose-model/no-label guard')
            pose, _ = _pose_input(pose_path, source)
            if pose is None:
                raise ValueError('Cached foot scene no longer passes source guards')
            contact, _ = _bound_contact(args.ball_dir, source, pose_path, pose, source_manifest_sha)
            if contact['status'] != 'tentative':
                raise ValueError('Cached foot has no source-bound canonical departure')
            contact_path = args.ball_dir/'contacts'/(cid+'.json')
            if _hash(contact_path) != original['contact_sha256']:
                raise ValueError('Foot producer actual contact cache SHA mismatch')
            contact_record = json.loads(contact_path.read_text())
            payload = contact_payload(contact_record)
            crop_path = args.foot_dir/'poses'/(cid+'.json')
            normal, mirror = cached_samples(json.loads(crop_path.read_text()))
            bracket = producer['timing']['contact_bracket_frames']
            if bracket != payload.get('contact_bracket_frames'):
                raise ValueError('Cached foot bracket differs from the actual ball producer')
            if _hash(ROOT/'src/kicking_foot_evidence.py') != original['module_sha256']:
                raise ValueError('V1 evidence module changed: replay its frozen source instead')
            v1 = kicking_foot_evidence(normal, mirror, bracket, pose['fps'],
                                       ball_observations=selected_ball_observations(contact_record))
            if not _equivalent(v1, producer['temporal_evidence']):
                raise ValueError('Cached pose bytes do not reproduce their stored V1 evidence')
            evidence = kicking_foot_evidence_v2(normal, mirror, bracket, pose['fps'],
                stationary_ball_observations=payload.get('stationary_ball_observations', []),
                departure_ball_observations=payload.get('departure_ball_observations', []))
            result.update(temporal_evidence=evidence, status=evidence['status'],
                          kicking_foot_candidate=evidence['candidate'], v1_evidence_reproduced=True,
                          cropped_pose_sha256=_hash(crop_path), actual_contact_sha256=_hash(contact_path))
            if producer.get('local_seed_identity_ambiguous_frames'):
                result.update(status='ambiguous_contact_ball_identity', kicking_foot_candidate='unknown',
                    local_seed_identity_ambiguous_frames=producer['local_seed_identity_ambiguous_frames'])
        _write(args.output_dir/'results'/(cid+'.json'), result)
        rows.append(dict(clip_id=cid, candidate=result['kicking_foot_candidate'], status=result['status'],
                         v1_candidate=result['v1_kicking_foot_candidate']))
        print(json.dumps(rows[-1]), flush=True)
    summary = dict(events=39, rows=rows, candidate_counts=dict(Counter(row['candidate'] for row in rows)),
        status_counts=dict(Counter(row['status'] for row in rows)), new_neural_inference_events=0,
        original_neural_events=sum(json.loads((args.foot_dir/'results'/(row['clip_id']+'.json')).read_text())['new_neural_inference'] for row in sources),
        reference_labels_read=False, human_verified=False, accuracy_validated=False, held_out_test=False)
    _write(args.output_dir/'summary.json', summary)
    return summary


if __name__ == '__main__':
    main()

"""Replay the strike-window foot diagnostic over cached source-only poses."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.replay_fresh_foot_v2 import _equivalent, cached_samples
from scripts.run_fresh_foot_experiment import contact_payload, selected_ball_observations
from scripts.run_fresh_pitch_experiment import _hash, _pose_input, _write
from src.kicking_foot_evidence import kicking_foot_evidence
from src.kicking_foot_strike_window import kicking_foot_strike_window


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clips-dir', type=Path, required=True)
    parser.add_argument('--pose-dir', type=Path, required=True)
    parser.add_argument('--ball-dir', type=Path, required=True)
    parser.add_argument('--foot-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use a new empty output directory to preserve prior replays')
    sources = read(args.clips_dir / 'source_manifest.json')['clips']
    if len(sources) != 39 or len({row['clip_id'] for row in sources}) != 39:
        raise ValueError('Replay requires exact distinct 39-event course manifest')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'results').mkdir()
    snapshot_names = ('src/kicking_foot_strike_window.py',
                      'scripts/replay_fresh_foot_strike_window.py',
                      'src/kicking_foot_evidence.py',
                      'src/kicking_foot_evidence_v2.py',
                      'scripts/replay_fresh_foot_v2.py',
                      'scripts/run_fresh_foot_experiment.py',
                      'scripts/run_fresh_pitch_experiment.py',
                      'docs/FRESH_STRIKE_WINDOW_PROTOCOL_2026-10-13.md')
    hashes = {}
    for name in snapshot_names:
        dest = args.output_dir / 'source_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
        hashes[name] = _hash(dest)
    manifest_sha = _hash(args.clips_dir / 'source_manifest.json')
    protocol = dict(schema_version='fresh-foot-strike-window-v1',
        source_manifest_sha256=manifest_sha, code_sha256=hashes,
        method_registered_before_results=True, new_neural_inference=False,
        development_data_previously_inspected=True,
        reference_labels_read=False, human_verified=False, accuracy_validated=False,
        held_out_test=False, upstream_foot_protocol_sha256=_hash(args.foot_dir / 'protocol.json'))
    _write(args.output_dir / 'protocol.json', protocol)
    rows = []
    for source in sources:
        cid = source['clip_id']
        producer_path = args.foot_dir / 'results' / f'{cid}.json'
        producer = read(producer_path)
        pose_path = args.pose_dir / f'{cid}.json'
        expected = dict(pose_sha256=_hash(pose_path), source_sha256=source['clip_sha256'],
                        source_manifest_sha256=manifest_sha)
        if any(producer['inference_identity'].get(name) != value for name, value in expected.items()):
            raise ValueError(f'Cached neural producer/source/pose identity mismatch: {cid}')
        if _hash(args.clips_dir / source['clip_filename']) != source['clip_sha256']:
            raise ValueError(f'Actual source video SHA mismatch: {cid}')
        result = dict(clip_id=cid, candidate='unknown', status='upstream_neural_event_unavailable',
            new_neural_inference=False, human_verified=False, accuracy_validated=False,
            held_out_test=False, reference_labels_used_for_prediction=False,
            shoe_ball_contact_verified=False, source_mapping=source,
            original_neural_producer=dict(path=str(producer_path.resolve()), sha256=_hash(producer_path),
                new_neural_inference=producer.get('new_neural_inference', False),
                inference_identity=producer.get('inference_identity')),
            inference_identity=dict(pose_sha256=_hash(pose_path), source_sha256=source['clip_sha256'],
                source_manifest_sha256=manifest_sha, code_sha256=hashes))
        if producer.get('new_neural_inference'):
            if (producer.get('upstream_scene_rejections') != []
                    or producer.get('model_task_validation', {}).get('actual_task') != 'pose'
                    or producer.get('reference_labels_used_for_prediction') is not False):
                raise ValueError(f'Cached foot inference lacks scene/pose-model/no-label guard: {cid}')
            pose, _ = _pose_input(pose_path, source)
            contact_path = args.ball_dir / 'contacts' / f'{cid}.json'
            contact = read(contact_path)
            if _hash(contact_path) != producer['inference_identity']['contact_sha256']:
                raise ValueError(f'Contact bytes differ from cached neural inference: {cid}')
            normal, mirror = cached_samples(read(args.foot_dir / 'poses' / f'{cid}.json'))
            bracket = producer['timing']['contact_bracket_frames']
            payload = contact_payload(contact)
            original_evidence = kicking_foot_evidence(normal, mirror, bracket, pose['fps'],
                                                     ball_observations=selected_ball_observations(contact))
            if not _equivalent(original_evidence, producer['temporal_evidence']):
                raise ValueError(f'Original neural evidence does not reproduce from cached poses: {cid}')
            evidence = kicking_foot_strike_window(
                normal, mirror, bracket, pose['fps'],
                stationary_ball_observations=payload.get('stationary_ball_observations', []),
                departure_ball_observations=payload.get('departure_ball_observations', []))
            result.update(candidate=evidence['candidate'], status=evidence['status'],
                temporal_evidence=evidence, actual_contact_sha256=_hash(contact_path),
                cached_pose_sha256=_hash(args.foot_dir / 'poses' / f'{cid}.json'),
                original_neural_evidence_reproduced=True)
            if producer.get('local_seed_identity_ambiguous_frames'):
                result.update(candidate='unknown', status='ambiguous_contact_ball_identity')
        rows.append(dict(clip_id=cid, candidate=result['candidate'], status=result['status']))
        _write(args.output_dir / 'results' / f'{cid}.json', result)
    summary = dict(events=39, rows=rows,
        candidate_counts=dict(Counter(row['candidate'] for row in rows)),
        status_counts=dict(Counter(row['status'] for row in rows)),
        new_neural_inference_events=0, reference_labels_read=False,
        human_verified=False, accuracy_validated=False, held_out_test=False,
        interpretation='Source-only strike-window diagnostic; not a shoe-contact label or held-out accuracy test')
    _write(args.output_dir / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    main()

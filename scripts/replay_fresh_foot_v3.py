"""Replay source-only stationary-spot kinematics from cached V1 pose bytes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.replay_fresh_foot_v2 import cached_samples, _equivalent
from scripts.run_fresh_foot_experiment import contact_payload, selected_ball_observations
from scripts.run_fresh_phase_experiment import _bound_contact
from scripts.run_fresh_pitch_experiment import _hash, _pose_input, _write
from src.kicking_foot_evidence import kicking_foot_evidence
from src.kicking_foot_evidence_v3 import kicking_foot_evidence_v3


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('clips-dir', 'pose-dir', 'ball-dir', 'foot-dir', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use a new empty output directory to preserve prior replays')
    manifest_path = args.clips_dir/'source_manifest.json'
    source_manifest_sha = _hash(manifest_path)
    sources = json.loads(manifest_path.read_text())['clips']
    if len(sources) != 39 or len({row['clip_id'] for row in sources}) != 39:
        raise ValueError('Replay requires exact distinct 39-event course manifest')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'results').mkdir()
    snapshot = ('src/kicking_foot_evidence_v3.py', 'scripts/replay_fresh_foot_v3.py',
                'src/kicking_foot_evidence.py', 'scripts/replay_fresh_foot_v2.py')
    hashes = {}
    for name in snapshot:
        dest = args.output_dir/'source_snapshot'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dest)
        hashes[name] = _hash(dest)
    protocol = dict(schema_version='fresh-foot-kinematic-replay-v3',
        source_manifest_sha256=source_manifest_sha, code_sha256=hashes,
        method_registered_before_results=True, new_neural_inference=False,
        reference_labels_read=False, human_verified=False, accuracy_validated=False, held_out_test=False)
    _write(args.output_dir/'protocol.json', protocol)
    rows=[]
    for source in sources:
        cid=source['clip_id']; producer_path=args.foot_dir/'results'/(cid+'.json')
        producer=json.loads(producer_path.read_text()); pose_path=args.pose_dir/(cid+'.json')
        result=dict(clip_id=cid, candidate='unknown', status='upstream_neural_event_unavailable',
            new_neural_inference=False, human_verified=False, accuracy_validated=False,
            held_out_test=False, reference_labels_used_for_prediction=False,
            shoe_ball_contact_verified=False, source_mapping=source,
            original_neural_producer=dict(path=str(producer_path.resolve()),sha256=_hash(producer_path),
                new_neural_inference=producer.get('new_neural_inference',False),
                inference_identity=producer.get('inference_identity')),
            inference_identity=dict(pose_sha256=_hash(pose_path), source_sha256=source['clip_sha256'],
                source_manifest_sha256=source_manifest_sha, code_sha256=hashes))
        if producer.get('new_neural_inference'):
            pose,_=_pose_input(pose_path,source)
            contact_path=args.ball_dir/'contacts'/(cid+'.json')
            contact=json.loads(contact_path.read_text())
            payload=contact_payload(contact)
            crop_path=args.foot_dir/'poses'/(cid+'.json')
            normal,mirror=cached_samples(json.loads(crop_path.read_text()))
            bracket=producer['timing']['contact_bracket_frames']
            v1=kicking_foot_evidence(normal,mirror,bracket,pose['fps'],
                                     ball_observations=selected_ball_observations(contact))
            if not _equivalent(v1,producer['temporal_evidence']):
                raise ValueError(f'V1 cached evidence does not reproduce for {cid}')
            evidence=kicking_foot_evidence_v3(normal,mirror,bracket,pose['fps'],
                stationary_ball_observations=payload.get('stationary_ball_observations',[]),
                departure_ball_observations=payload.get('departure_ball_observations',[]))
            result.update(candidate=evidence['candidate'],status=evidence['status'],
                temporal_evidence=evidence,v1_evidence_reproduced=True,
                cropped_pose_sha256=_hash(crop_path),actual_contact_sha256=_hash(contact_path))
            if producer.get('local_seed_identity_ambiguous_frames'):
                result.update(candidate='unknown',status='ambiguous_contact_ball_identity')
        _write(args.output_dir/'results'/(cid+'.json'),result)
        rows.append(dict(clip_id=cid,candidate=result['candidate'],status=result['status']))
        print(json.dumps(rows[-1]),flush=True)
    summary=dict(events=39,rows=rows,candidate_counts=dict(Counter(row['candidate'] for row in rows)),
        status_counts=dict(Counter(row['status'] for row in rows)),new_neural_inference_events=0,
        reference_labels_read=False,human_verified=False,accuracy_validated=False,held_out_test=False)
    _write(args.output_dir/'summary.json',summary)
    return summary


if __name__=='__main__':
    main()

"""Deterministic reanalysis of fresh pose; no user review labels loaded."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.kicking_foot_redetection import transform_points,followthrough_evidence
RUN=ROOT/'outputs/goal_right_foot_redetection_2026-10-11/run_v2'


def main():
    rows=[]
    for r in json.loads((RUN/'summary.json').read_text(encoding='utf-8'))['rows']:
        cid=r['clip_id'];path=RUN/'poses'/f'{cid}.json';raw=json.loads(path.read_text(encoding='utf-8'))
        camera=json.loads((ROOT/'outputs/ball_detection_all_2026-10-11/detections'/f'{cid}.json').read_text(encoding='utf-8'))['camera']['transforms']
        ball=json.loads((ROOT/'outputs/ball_mask_review_ready_2026-10-11/results'/f'{cid}.json').read_text(encoding='utf-8'))['variants']['mask_and_radius_35']
        reference=np.asarray(r['stationary_ball_reference']);departure=r['contact_bracket_frames'][1]
        flight=[np.asarray(p['stabilized_xy']) for p in ball['trajectory'] if departure<=p['frame']<=departure+6]
        if r['goal_ground_midpoint_uncalibrated'] is not None:
            axis=np.asarray(r['goal_ground_midpoint_uncalibrated'])-reference;axis_source='automatic_goal_ground_midpoint'
        elif flight:
            axis=np.median(flight,axis=0)-reference;axis_source='observed_early_ball_flight_not_ground_calibrated'
        else:axis=np.zeros(2);axis_source='unavailable'
        variants={}
        for reflected in (False,True):
            samples=[dict(frame=p['frame'],keypoints=transform_points(p['keypoints'],camera[str(p['frame'])]),confidence=p['confidence']) for p in raw if p['reflected']==reflected and str(p['frame']) in camera]
            variants['mirrored' if reflected else 'normal']=followthrough_evidence(samples,axis,departure)
        labels=[d['candidate'] for d in variants.values()]
        foot=labels[0] if labels[0]==labels[1] and labels[0]!='unknown' else 'unknown'
        approach=r['approach_heuristics']['goal_axis_candidate']
        # A cross-check can reject but must not silently reverse anatomy.
        fused=foot if foot!='unknown' and approach==foot else 'unknown'
        row=dict(clip_id=cid,axis=axis.tolist(),axis_source=axis_source,variants=variants,
                 followthrough_consensus_candidate=foot,approach_crosschecked_candidate=fused,
                 goal_axis_approach_candidate=approach,reviewer_labels_used=False,
                 fresh_pose_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        rows.append(row);print(cid,labels,approach,foot,fused)
    (RUN/'followthrough_analysis.json').write_text(json.dumps(dict(rows=rows,reviewer_labels_used=False,accuracy_estimate=None,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'src/kicking_foot_redetection.py']}),indent=2),encoding='utf-8')


if __name__=='__main__':main()

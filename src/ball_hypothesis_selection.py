"""Select current seed hypotheses using only the early shot interval.

Later clothing/net candidate density cannot make a weak release hypothesis win.
Scores express support for review, never an estimated probability of correctness.
"""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class SelectionConfig:
    early_frames: int = 12
    minimum_progress: float = .15
    maximum_progress: float = .70
    alternative_margin: float = .05


def select_early_flight(proposals, kick, goal_center, preferred, config=None):
    cfg=config or SelectionConfig()
    scores={};evidence={}
    for name,v in proposals.items():
        seed=v['seed'];score=0.;ids=[]
        if seed:
            origin=np.median([q['stabilized_xy'] for q in seed],axis=0)
            direction=np.asarray(goal_center,float)-origin
            reach=float(np.linalg.norm(direction))
            if reach>1e-6:
                direction/=reach
                for q in v['trajectory']:
                    if q['frame']>kick+cfg.early_frames:continue
                    progress=float(np.dot(np.asarray(q['stabilized_xy'])-origin,direction)/reach)
                    if not cfg.minimum_progress<=progress<=cfg.maximum_progress:continue
                    score += .1 if 'motion_blob' in q['sources'] else .2+min(.5,q.get('confidence',0.))
                    ids.append(q['detection_id'])
        scores[name]=float(score);evidence[name]=ids
    chosen=preferred
    best=max(scores,key=scores.get)
    if scores[best]>scores[preferred]+cfg.alternative_margin:chosen=best
    return chosen,{'method':'early_flight_support_across_current_seed_hypotheses',
      'config':asdict(cfg),'scores':scores,'evidence_ids':evidence,'chosen':chosen,
      'later_tail_used_for_selection':False,'scores_are_accuracy_probabilities':False}

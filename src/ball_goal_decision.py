"""Uncalibrated short-window goal suggestions, separate from visual review.

Image-plane mouth occupancy is a cue, never proof of whole-ball crossing.
Large turns after supported entry are not converted into a miss. No outcome
labels, player names, or manually chosen ball coordinates are inputs.
"""
from dataclasses import dataclass,asdict
import numpy as np


@dataclass(frozen=True)
class GoalDecisionConfig:
    minimum_inside_points: int = 3
    minimum_outside_points: int = 2
    maximum_gap_frames: int = 4
    mouth_margin_fraction: float = .04
    arrival_window_before: int = 1
    arrival_window_after: int = 7


def classify_goal_evidence(points,boundary,rebound,*,config=None):
    cfg=config or GoalDecisionConfig()
    base={'rule_outcome':'unknown','is_suggestion':True,'calibrated_probability':None,
          'whole_ball_crossing_established':False,'human_verified':False,
          'config':asdict(cfg),'evidence_frames':[],'post_event_tail_used':False,
          'all_large_turns_mean_no_goal':False}
    good=[];seen=set()
    for p in sorted(points,key=lambda q:q['frame']):
        if not p.get('identity_support') or not p.get('geometry_supported'):continue
        if not boundary-cfg.arrival_window_before<=p['frame']<=boundary+cfg.arrival_window_after:continue
        key=p.get('independent_image_key',p['frame'])
        if key in seen:continue
        seen.add(key);good.append(p)
    inside=[p for p in good if p['inside_mouth'] and p.get('arrival_progress_supported')]
    outside=[p for p in good if p.get('clearly_outside_mouth') and p.get('arrival_progress_supported')]
    # Require actual consecutive independent images near arrival. A finite
    # ball footprint straddling a post is not a whole-ball outside cue.
    runs=[];run=[]
    for p in inside:
        if run and p['frame']-run[-1]['frame']>cfg.maximum_gap_frames:
            runs.append(run);run=[]
        run.append(p)
    runs.append(run);best=max(runs,key=len)
    entry=best[cfg.minimum_inside_points-1]['frame'] if len(best)>=cfg.minimum_inside_points else None
    base['projected_entry_support_frame']=entry
    impact=(rebound.get('contact_prediction') or {}).get('frame_float')
    if rebound.get('turn_confirmed') and impact is not None:
        if entry is not None and entry<impact:
            return {**base,'rule_outcome':'goal_likely','reason':'mouth_sequence_before_later_turn',
                    'evidence_frames':[p['frame'] for p in best],
                    'later_turn_not_used_as_miss':True}
        return {**base,'rule_outcome':'no_goal_likely','reason':'supported_short_keeper_rebound',
                'evidence_frames':rebound.get('support_frames',[]),
                'contact_time_is_prediction':True,'depth_ambiguity_remaining':True}
    if len(outside)>=cfg.minimum_outside_points and not inside:
        return {**base,'rule_outcome':'no_goal_likely','reason':'finite_ball_outside_mouth_near_arrival',
                'evidence_frames':[p['frame'] for p in outside]}
    if entry is not None:
        return {**base,'rule_outcome':'goal_likely','reason':'continuous_projected_mouth_sequence',
                'evidence_frames':[p['frame'] for p in best],'depth_ambiguity_remaining':True}
    return {**base,'reason':'insufficient_ball_identity_or_arrival_evidence',
            'evidence_frames':[p['frame'] for p in good]}

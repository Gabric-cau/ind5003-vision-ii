"""Bound the shot analysis without turning late net motion into a miss label.

This is a phase boundary on unverified 2-D observations, not goal-line depth
estimation. The full input track is retained separately by the caller.
"""
from dataclasses import asdict, dataclass
import math
import numpy as np


@dataclass(frozen=True)
class ShotFocusConfig:
    large_turn_degrees: float = 55.0
    outgoing_support_points: int = 3
    outgoing_heading_tolerance_degrees: float = 25.0
    max_gap_frames: int = 6
    measurement_slack_px: float = 10.0
    reachable_speed_ratio: float = 1.6
    contact_context_seconds: float = .20
    maximum_analysis_seconds: float = .85
    minimum_departure_px: float = 60.0


def xy(d):
    return np.asarray(d['stabilized_xy'],float)


def ball_extent(d):
    """Measured candidate footprint; blur/false identity can enlarge this box."""
    width=max(1.,float(d.get('width',8.)))
    height=max(1.,float(d.get('height',8.)))
    return {'half_width_px':width/2,'half_height_px':height/2,
            'radius_bound_px':math.hypot(width,height)/2,
            'whole_ball_crossing_established':False,
            'source':'candidate_bbox_not_calibrated_ball_volume'}


def angle(a,b):
    na,nb=np.linalg.norm(a),np.linalg.norm(b)
    if min(na,nb)<1e-9:return None
    return float(np.degrees(np.arccos(np.clip(np.dot(a,b)/(na*nb),-1,1))))


def focus_shot(path,seed,goal_center,kick_frame,fps,window_end_exclusive,*,goal_polygon=None,config=None):
    """Causal incoming boundary plus at most a short contact observation context.

    A goal-area projection marks a review boundary, never a scored goal. Large
    turns before it are miss/deflection hints; identity errors remain possible.
    Turns after arrival never become a no-goal hint. No outcome labels are read.
    """
    cfg=config or ShotFocusConfig()
    if fps<=0 or cfg.outgoing_support_points<2 or cfg.max_gap_frames<1:
        raise ValueError('Positive fps/gap and at least two outgoing points required')
    pts=list(path)
    if any(b['frame']<=a['frame'] for a,b in zip(pts,pts[1:])):
        raise ValueError('Observations require increasing source frames')
    if any(xy(d).shape!=(2,) or not np.isfinite(xy(d)).all() for d in pts):
        raise ValueError('Finite camera-stabilized observations required')
    limit=min(window_end_exclusive-1,kick_frame+math.ceil(cfg.maximum_analysis_seconds*fps))
    base={'policy':'incoming_shot_then_short_contact_context','config':asdict(cfg),
          'outcome':'unknown','whole_ball_crossing_established':False,
          'final_position':None,'human_verified':False,'accuracy_validated':False,
          'source_track_is_verified':False,'time_basis':'original_source_frames',
          'no_goal_hint':False,'supported_large_turn':None,
          'tail_used_for_outcome':False,'physical_contact_established':False,
          'input_observation_count':len(pts)}
    if not pts or not seed:
        return [],{**base,'reason':'missing_track_or_seed','main_end_frame':None,
                   'focus_end_frame':max(kick_frame,min(limit,window_end_exclusive-1)),
                   'excluded_observation_count':len(pts),'context_observation_ids':[]}
    origin=np.median([xy(d) for d in seed],axis=0)
    goal=np.asarray(goal_center,float)
    gvec=goal-origin;reach=float(np.linalg.norm(gvec))
    if reach<cfg.minimum_departure_px or not np.isfinite(goal).all():
        raise ValueError('Finite, spatially separate goal reference required')
    axis=gvec/reach
    gate=reach
    if goal_polygon is not None:
        polygon=np.asarray(goal_polygon,float)
        if polygon.shape!=(4,2) or not np.isfinite(polygon).all():
            raise ValueError('Four finite camera-stabilized goal corners required')
        gate=float(np.min((polygon-origin)@axis))
        if gate<=cfg.minimum_departure_px:raise ValueError('Goal region overlaps departure area')
    main_end=None;reason='bounded_horizon_without_established_event'
    event=None;arrival=None;stop_before=None
    # Wait for departure before treating a stationary seed or boot as a flight.
    flying=[q for q in pts if np.linalg.norm(xy(q)-origin)>=cfg.minimum_departure_px]
    previous_velocity=None
    for i,d in enumerate(flying):
        f=d['frame']
        if f>limit:break
        extent=ball_extent(d)
        progress=float(np.dot(xy(d)-origin,axis))
        # Center-plane progress plus the candidate's finite extent is a useful
        # visual boundary even for a wide/high shot. It is not depth/crossing.
        near_goal=progress>=gate-extent['radius_bound_px']-cfg.measurement_slack_px
        if i:
            prev=flying[i-1];dt=f-prev['frame'];delta=xy(d)-xy(prev)
            if dt>cfg.max_gap_frames:
                main_end=prev['frame'];reason='continuity_lost';stop_before=f;break
            velocity=delta/dt
            if previous_velocity is not None:
                reach_bound=cfg.reachable_speed_ratio*np.linalg.norm(previous_velocity)*dt+cfg.measurement_slack_px+extent['radius_bound_px']
                if np.linalg.norm(delta)>reach_bound:
                    main_end=prev['frame'];reason='candidate_outside_reachable_range';stop_before=f;break
        if near_goal:
            arrival={'frame':f,'detection_id':d['detection_id'],
                     'progress_px':progress,'reference_goal_distance_px':reach,
                     'goal_area_gate_progress_px':gate,
                     'candidate_extent':extent,'projection_only':True}
            main_end=f;reason='goal_reference_area_reached';break
        if i==0:continue
        if previous_velocity is not None:
            theta=angle(previous_velocity,velocity)
            if theta is not None and theta>=cfg.large_turn_degrees and np.linalg.norm(delta)>cfg.measurement_slack_px:
                # A single jump does not establish physical deflection.
                outgoing=flying[i:i+cfg.outgoing_support_points]
                supported=(len(outgoing)==cfg.outgoing_support_points and
                           outgoing[-1]['frame']<=limit and
                           outgoing[-1]['frame']-prev['frame']<=math.ceil(cfg.contact_context_seconds*fps))
                support_angles=[]
                for a,b in zip([prev]+outgoing[:-1],outgoing):
                    gap=b['frame']-a['frame']
                    v=(xy(b)-xy(a))/gap
                    heading=angle(velocity,v)
                    supported=supported and gap<=cfg.max_gap_frames and heading is not None and heading<=cfg.outgoing_heading_tolerance_degrees
                    support_angles.append(heading)
                if supported:
                    event={'turn_frame':f,'last_incoming_frame':prev['frame'],
                           'angle_degrees':theta,'support_ids':[q['detection_id'] for q in outgoing],
                           'support_frames':[q['frame'] for q in outgoing],
                           'support_heading_differences_degrees':support_angles,
                           'relative_to_goal_area':'before_goal_reference_area',
                           'identity_switch_cannot_be_excluded':True,
                           'before_physical_goal_line_established':False}
                    main_end=prev['frame'];reason='supported_large_turn_before_goal_area'
                    break
        if np.linalg.norm(delta)>cfg.measurement_slack_px:
            previous_velocity=velocity if previous_velocity is None else .35*previous_velocity+.65*velocity
    if main_end is None:
        main_end=min(limit,pts[-1]['frame'])
        if pts[-1]['frame']<limit:reason='track_ended_without_established_event'
    context_frames=max(0,math.ceil(cfg.contact_context_seconds*fps))
    end=min(limit,main_end+context_frames,window_end_exclusive-1)
    if stop_before is not None:end=min(end,stop_before-1)
    kept=[q for q in pts if q['frame']<=end]
    incoming=[q for q in kept if q['frame']<=main_end]
    context=[q for q in kept if q['frame']>main_end]
    return kept,{**base,'reason':reason,'main_end_frame':main_end,'focus_end_frame':end,
      'arrival_projection':arrival,'supported_large_turn':event,'no_goal_hint':event is not None,
      'no_goal_hint_status':'possible_miss_or_identity_switch' if event else 'not_established',
      'incoming_observation_ids':[q['detection_id'] for q in incoming],
      'context_observation_ids':[q['detection_id'] for q in context],
      'excluded_observation_count':len(pts)-len(kept),
      'goal_line_depth_status':'not_established_from_2d_track',
      'goal_area_gate_progress_px':gate,'goal_polygon_used':goal_polygon is not None,
      'endpoint_status':'analysis_boundary_not_verified_ball_final_position'}

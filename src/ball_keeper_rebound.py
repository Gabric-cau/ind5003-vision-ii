"""Short offline contact association, with observations distinct from forecasts.

A turn is admitted only when its timed trajectory meets the incoming forecast
near independently tracked keeper geometry. Goal-area projection alone cannot
unlock it. This is 2-D association evidence, not a goal-line decision.
"""
from dataclasses import dataclass, asdict
from itertools import combinations, product
import numpy as np
from src.ball_continuity_v4 import _inside_convex, _segment_distance


@dataclass(frozen=True)
class ReboundConfig:
    contact_horizon_frames: int = 6
    support_window_frames: int = 7
    minimum_support_points: int = 3
    contact_margin_px: float = 24.
    timed_intersection_slack_px: float = 12.
    minimum_turn_degrees: float = 55.
    minimum_outgoing_speed: float = 3.
    maximum_rebound_speed_ratio: float = 1.6
    fit_slack_px: float = 6.
    first_observation_time_slack: float = .5
    maximum_gap_frames: int = 4
    rebound_context_seconds: float = .45


def xy(q):
    return np.asarray(q['stabilized_xy'], float)


def incoming_velocity(path):
    # Do not include the disputed continuation in the incoming estimate.
    tail = path[-4:]
    vs = [(xy(b)-xy(a))/(b['frame']-a['frame']) for a,b in zip(tail,tail[1:])]
    return np.median(vs, axis=0)


def timed_intersection(last, velocity, first, outgoing_velocity):
    vi, vo = np.asarray(velocity), np.asarray(outgoing_velocity)
    d = xy(last)-vi*last['frame']-xy(first)+vo*first['frame']
    relative = vi-vo
    if np.dot(relative,relative) < 1e-8:
        return None
    t = -float(np.dot(d,relative)/np.dot(relative,relative))
    p = xy(last)+vi*(t-last['frame'])
    residual = float(np.linalg.norm(p-(xy(first)+vo*(t-first['frame']))))
    return {'frame_float': t, 'stabilized_xy': p.tolist(), 'residual_px': residual,
            'is_observation': False, 'physical_contact_established': False}


def keeper_distance(position, polygons, time):
    matches=[]
    for f in (int(np.floor(time)), int(np.ceil(time))):
        for region in polygons.get(str(f), polygons.get(f, [])):
            if region.get('kind') != 'keeper_candidate':
                continue
            poly=region['polygon']
            distance=0. if _inside_convex(position,poly) else min(
                _segment_distance(position,position,a,b)
                for a,b in zip(poly,poly[1:]+poly[:1]))
            matches.append((distance, f, region))
    return min(matches,key=lambda q:q[0]) if matches else None


def inside_keeper(q, regions):
    near=keeper_distance(xy(q),regions,float(q['frame']))
    # Include the ball radius around held/detected body boxes, so a glove or
    # boot on the box edge cannot masquerade as a separated outgoing ball.
    return near is not None and near[0] <= 8.


def associate_keeper_rebound(incoming, candidates, keeper_regions, fps, *,
                             frame_fingerprints=None, config=None):
    cfg=config or ReboundConfig()
    if fps<=0 or len(incoming)<3 or cfg.minimum_support_points<3:
        raise ValueError('Positive fps and at least three incoming/support points required')
    if any(b['frame']<=a['frame'] for a,b in zip(incoming,incoming[1:])):
        raise ValueError('Incoming frames must increase')
    if any(xy(q).shape!=(2,) or not np.isfinite(xy(q)).all() for q in incoming):
        raise ValueError('Finite stabilized observations required')
    last=incoming[-1];vi=incoming_velocity(incoming);speed=float(np.linalg.norm(vi))
    base={'config':asdict(cfg),'outcome':'unknown','physical_contact_established':False,
          'whole_ball_crossing_established':False,'manual_ball_points_used':False,
          'incoming_prefix_preserved':True,'turn_confirmed':False,
          'incoming_velocity':vi.tolist(),'time_basis':'original_source_frames',
          'goal_projection_unlocks_turn':False,'discarded_after_boundary':[],
          'rejected_timed_intersections':0,'independent_support_required':cfg.minimum_support_points}
    if speed<1e-8:return list(incoming),{**base,'reason':'incoming_direction_unavailable'}
    # Keeper permission is based on the incoming forecast before testing any
    # destination. A destination on a person cannot create its own permission.
    contact_frames=[]
    for f in range(last['frame']+1,last['frame']+cfg.contact_horizon_frames+1):
        p=xy(last)+vi*(f-last['frame'])
        near=keeper_distance(p,keeper_regions,float(f))
        if near and near[0]<=cfg.contact_margin_px:contact_frames.append(f)
    base['incoming_forecast_near_keeper_frames']=contact_frames
    if not contact_frames:return list(incoming),{**base,'reason':'no_independent_keeper_proximity'}
    pool={}
    seen_fingerprints=set()
    fingerprints=frame_fingerprints or {}
    limit=last['frame']+cfg.contact_horizon_frames+cfg.support_window_frames
    for f in range(last['frame']+1,limit+1):
        fingerprint=fingerprints.get(str(f),fingerprints.get(f))
        if fingerprint is not None and fingerprint in seen_fingerprints:continue
        if fingerprint is not None:seen_fingerprints.add(fingerprint)
        qs=[]
        for q in candidates.get(str(f),candidates.get(f,[])):
            if 'motion_blob' in q.get('sources',[]) or not np.isfinite(xy(q)).all():continue
            if q.get('rebound_current_white_support') is False:continue
            if min(q.get('width',10),q.get('height',10))>32:continue
            if np.linalg.norm(xy(q)-xy(last))>speed*cfg.contact_horizon_frames+100:continue
            if any(np.linalg.norm(xy(q)-xy(a))<2 for a in qs):continue
            qs.append(q)
        if qs:pool[f]=qs
    hypotheses=[]
    for fs in combinations(sorted(pool),cfg.minimum_support_points):
        if fs[-1]-fs[0]>cfg.support_window_frames or fs[0]>max(contact_frames)+3:continue
        if any(b-a>cfg.maximum_gap_frames for a,b in zip(fs,fs[1:])):continue
        for qs in product(*(pool[f] for f in fs)):
            # The first point may overlap the contact region. The following
            # two must separate from the keeper; repeated glove/body hits do
            # not prove an outgoing ball trajectory.
            if any(inside_keeper(q,keeper_regions) for q in qs[1:]):continue
            ts=np.asarray(fs,float);pts=np.array([xy(q) for q in qs]);center=ts.mean()
            vo=((ts-center)[:,None]*(pts-pts.mean(axis=0))).sum(axis=0)/((ts-center)**2).sum()
            ospeed=float(np.linalg.norm(vo))
            if not cfg.minimum_outgoing_speed<=ospeed<=cfg.maximum_rebound_speed_ratio*speed:continue
            theta=float(np.degrees(np.arccos(np.clip(np.dot(vi,vo)/(speed*ospeed),-1,1))))
            if theta<cfg.minimum_turn_degrees:continue
            fitted=pts.mean(axis=0)+(ts-center)[:,None]*vo
            error=float(np.sqrt(np.mean(np.sum((pts-fitted)**2,axis=1))))
            if error>cfg.fit_slack_px:continue
            first={**qs[0],'stabilized_xy':fitted[0].tolist()}
            impact=timed_intersection(last,vi,first,vo)
            if (impact is None or impact['residual_px']>cfg.timed_intersection_slack_px
                or not last['frame']<impact['frame_float']<=last['frame']+cfg.contact_horizon_frames
                or fs[0]<impact['frame_float']-cfg.first_observation_time_slack):
                base['rejected_timed_intersections']+=1;continue
            near=keeper_distance(impact['stabilized_xy'],keeper_regions,impact['frame_float'])
            if not near or near[0]>cfg.contact_margin_px:continue
            # Prefer low residual, early evidence. Confidence cannot override
            # temporal reachability or turn support (boots can score highly).
            score=error+impact['residual_px']*.7+(fs[0]-last['frame'])*.25
            score-=.3*sum(min(.5,q.get('confidence',0)) for q in qs)
            dense=[]
            for f in sorted(pool):
                if not fs[0]<=f<=fs[0]+cfg.support_window_frames:continue
                target=fitted[0]+vo*(f-fs[0])
                eligible=[q for q in pool[f] if (f==fs[0] or not inside_keeper(q,keeper_regions))
                          and np.linalg.norm(xy(q)-target)<=cfg.fit_slack_px]
                if eligible:dense.append(min(eligible,key=lambda q:np.linalg.norm(xy(q)-target)))
            # Dense refitting can discard an original support point. Never
            # leave a confirmed turn with fewer than three independent actual
            # observations or a newly introduced unsupported frame gap.
            if len(dense)<cfg.minimum_support_points:continue
            if any(b['frame']-a['frame']>cfg.maximum_gap_frames for a,b in zip(dense,dense[1:])):continue
            score-=1.5*len(dense)
            qs=tuple(dense)
            hypotheses.append((score,qs,vo,impact,theta,error,near))
    if not hypotheses:return list(incoming),{**base,'reason':'no_supported_timed_rebound'}
    score,support,vo,impact,theta,error,near=min(hypotheses,key=lambda q:q[0])
    outgoing=list(support)
    # Rebuild the narrow lane around the supported outgoing direction, then
    # stop after a short proof window. No late tail is used for goal labels.
    end=int(np.floor(impact['frame_float']+fps*cfg.rebound_context_seconds))
    for f in sorted(pool):
        if f<=outgoing[-1]['frame'] or f>end:continue
        dt=f-outgoing[-1]['frame']
        if dt>cfg.maximum_gap_frames:break
        prediction=xy(outgoing[-1])+vo*dt
        options=[]
        for q in pool[f]:
            if inside_keeper(q,keeper_regions):continue
            residual=float(np.linalg.norm(xy(q)-prediction))
            if residual<=cfg.fit_slack_px+dt*1.5:
                options.append((residual-.5*min(.5,q.get('confidence',0)),q))
        if options:
            q=min(options,key=lambda q:q[0])[1]
            measured=(xy(q)-xy(outgoing[-1]))/dt
            vo=.4*vo+.6*measured
            outgoing.append(q)
    return list(incoming)+outgoing,{**base,'reason':'supported_keeper_rebound',
        'turn_confirmed':True,'turn_angle_degrees':theta,'contact_prediction':impact,
        'keeper_geometry_frame':near[1],'keeper_distance_px':near[0],
        'support_ids':[q['detection_id'] for q in support],
        'support_frames':[q['frame'] for q in support],
        'confirmation_frame':support[-1]['frame'],'offline_confirmation_uses_later_observations':True,
        'outgoing_observation_ids':[q['detection_id'] for q in outgoing],
        'context_end_frame':end,'last_observed_frame':outgoing[-1]['frame'],
        'hypothesis_count':len(hypotheses),'selected_score':score,'outgoing_fit_error_px':error}

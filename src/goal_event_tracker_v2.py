"""Conservative experimental 2-D ball tracker with a departure bootstrap.

Coordinates remain source observations, not reconstructed 3-D positions.
Temporal indices may omit near-duplicate encoded frames; reported timestamps
always remain the original frame numbers. This is NOT a calibrated classifier.
"""
import math
import numpy as np
from src.event_ball_tracking import point, distance, select_stationary_seed


def tick(d):
    return float(d.get("motion_time", d["frame"]))


def cosine(a, b):
    return float(np.dot(a, b) / max(1e-9, np.linalg.norm(a) * np.linalg.norm(b)))


def quality(d):
    # Motion components use measured area, not a made-up detector probability.
    if "motion_blob" in d.get("sources", []):
        return .45 + min(.7, d.get("blob_area", 0) / 300)
    return .7 + min(.65, 2 * d.get("confidence", 0))


def usable(d):
    return (d.get("stabilized_xy") is not None and not d.get("duplicate_frame", False)
            and d.get("width", 8) >= 5 and d.get("height", 8) >= 4
            and (d.get("motion_mean_abs_gray") is None or d["motion_mean_abs_gray"] >= 8))


def bootstrap(seed, observations, goal, max_tick_gap=2):
    """Require two moving points whose backward line meets the stationary ball.

    A first jump directly from the penalty spot to keeper gloves is excluded by
    the departure-region bound. No externally supplied outcome enters selection.
    """
    last = seed[-1]
    origin = np.median([point(p) for p in seed], axis=0)
    direction = goal - origin
    reach = np.linalg.norm(direction)
    starts = []
    for i, a in enumerate(observations):
        if not last["frame"] < a["frame"] <= last["frame"] + 12:
            continue
        offset = point(a) - origin
        length = np.linalg.norm(offset)
        if not 35 <= length <= min(180, .30 * reach) or cosine(offset, direction) < .7:
            continue
        for b in observations[i + 1:]:
            gap = tick(b) - tick(a)
            if gap <= 0 or gap > max_tick_gap:
                continue
            velocity = (point(b) - point(a)) / gap
            speed = np.linalg.norm(velocity)
            if not 10 <= speed <= 160 or cosine(velocity, direction) < .7:
                continue
            back_time = np.dot(offset, velocity) / (speed * speed)
            miss = np.linalg.norm(offset - back_time * velocity)
            onset = tick(a) - back_time
            if miss > 22 or back_time < .2 or not tick(last) - 1 <= onset <= tick(a):
                continue
            if cosine(velocity, offset) < .92:
                continue
            starts.append({"path": [last, a, b], "velocity": velocity,
                           "score": quality(a) + quality(b) - miss / 20,
                           "turns": [], "onset_motion_time": float(onset)})
    return starts


def track_goal_event(detections, kick, anchors, goal_center, *, allow_turn=True,
                     beam_width=100, max_gap=2, goal_width=300):
    seed, seed_info = select_stationary_seed(detections, kick, anchors)
    base = {**seed_info, "outcome": "unknown", "human_verified": False,
            "path_is_verified": False, "radial_envelope_applied": False,
            "tracker_version": "departure_motion_time_v2"}
    if not seed:
        return [], {**base, "tracking_status": "abstain_no_seed"}
    obs = sorted([d for items in detections.values() for d in items if usable(d)],
                 key=lambda d: (d["frame"], d["detection_id"]))
    goal = np.asarray(goal_center, float)
    starts = bootstrap(seed, obs, goal)
    if not starts:
        return seed, {**base, "tracking_status": "abstain_no_supported_departure"}
    by_time = {}
    for d in obs:
        by_time.setdefault(tick(d), []).append(d)
    pending = {}
    origin = np.median([point(p) for p in seed], axis=0)
    goal_distance = float(np.linalg.norm(goal - origin))
    for s in starts:
        pending.setdefault(tick(s["path"][-1]), []).append(s)
    states, finished = [], []
    for current in sorted(by_time):
        expanded = pending.get(current, [])[:]
        for state in states:
            last = state["path"][-1]
            progress = float(np.dot(point(last)-origin, (goal-origin)/goal_distance))
            # Clothing can form a long smooth path. It is not a shot if it never
            # leaves the kicker's vicinity within the initial flight interval.
            if current-state["onset_motion_time"] > 10 and progress < .40*goal_distance:
                continue
            gap = current - tick(last)
            if gap <= 0:
                expanded.append(state)
                continue
            if gap > max_gap:
                finished.append(state)
                continue
            expanded.append({**state, "score": state["score"] - .16})
            for d in by_time[current]:
                # A ball can overlap a keeper at impact. A long run of points
                # stuck on a person cannot establish a separately tracked ball.
                if d.get("person_overlap") and all(q.get("person_overlap") for q in state["path"][-2:]):
                    continue
                v = state["velocity"]
                instantaneous = (point(d) - point(last)) / gap
                speed, old_speed = np.linalg.norm(instantaneous), np.linalg.norm(v)
                if speed > max(25, old_speed * 1.35):
                    continue
                residual = np.linalg.norm(point(d) - point(last) - v * gap)
                tolerance = 10 + .22 * old_speed * gap
                turn = False
                if residual > tolerance:
                    near_goal = (progress > .40*goal_distance and
                                 np.linalg.norm(point(last) - goal) < max(160, .95 * goal_width))
                    if not (allow_turn and near_goal and len(state["turns"]) < 2 and gap <= 2):
                        continue
                    # Impact may redirect or slow the ball, not teleport it.
                    if np.linalg.norm(point(d) - point(last)) > max(40, old_speed * gap * 1.2):
                        continue
                    turn = True
                    cost = 1.2 + min(.7, residual / 150)
                else:
                    cost = .7 * residual / max(10, tolerance)
                reward = quality(d) - (.45 if speed < 2 else 0) - (.5 if d.get("person_overlap") else 0)
                turns = list(state["turns"])
                if turn:
                    turns.append(d["frame"])
                expanded.append({**state, "path": state["path"] + [d],
                                 "velocity": instantaneous if turn else .2 * v + .8 * instantaneous,
                                 "turns": turns, "score": state["score"] + reward - cost})
        expanded.sort(key=lambda s: s["score"], reverse=True)
        states, seen = [], set()
        for s in expanded:
            signature = (s["path"][-1]["detection_id"], tuple(np.rint(s["velocity"] / 8).astype(int)), len(s["turns"]))
            if signature in seen:
                continue
            seen.add(signature)
            states.append(s)
            if len(states) >= beam_width:
                break
    finished += states
    viable = [s for s in finished if len(s["path"]) >= 5 and
              max(float(np.dot(point(d)-origin,(goal-origin)/goal_distance)) for d in s["path"]) > .40*goal_distance]
    if not viable:
        return seed, {**base, "tracking_status": "abstain_short_flight"}
    best = max(viable, key=lambda s: s["score"])
    path = best["path"]
    return path, {**base, "tracking_status": "candidate", "turn_frames": best["turns"],
                  "point_count": len(path), "beam_score": round(best["score"], 3),
                  "bootstrap_hypotheses": len(starts),
                  "first_departure_observation_frame": path[1]["frame"],
                  "max_gap_frames": max(b["frame"]-a["frame"] for a,b in zip(path,path[1:])),
                  "whole_ball_crossing_established": False}


def track_event_segments(detections, kick, anchors, goal_center, goal_width=300):
    """Freeze supported incoming flight, then test a short outgoing segment.

    A long glove/clothing track cannot retroactively replace a better incoming
    flight. The post-event horizon is deliberately bounded (8 distinct frames),
    not a claim to follow the ball until play ends.
    """
    incoming, diag = track_goal_event(detections,kick,anchors,goal_center,
                                     allow_turn=False,goal_width=goal_width)
    result={**diag,"segment_policy":"fixed_incoming_then_bounded_outgoing",
            "outgoing_status":"unavailable","incoming_observations":len(incoming),
            "outgoing_observations":0,"outcome":"unknown"}
    if diag.get('tracking_status')!='candidate' or len(incoming)<4:
        return incoming,result
    last=incoming[-1];previous=incoming[-3]
    vin=(point(last)-point(previous))/(tick(last)-tick(previous))
    speed_in=np.linalg.norm(vin)
    obs=sorted([d for ds in detections.values() for d in ds if usable(d) and
                0<tick(d)-tick(last)<=8 and d.get('kit_neighborhood_fraction',0)<.045
                and ("motion_blob" in d.get('sources',[]) or d.get('white_ratio',.5)>=.30)],
               key=lambda d:(tick(d),d['detection_id']))
    states=[]
    for i,a in enumerate(obs):
        dt=tick(a)-tick(last)
        if dt>3 or distance(a,last)>max(40,speed_in*dt*1.2):continue
        for b in obs[i+1:]:
            gap=tick(b)-tick(a)
            if not 0<gap<=2:continue
            v=(point(b)-point(a))/gap;speed=np.linalg.norm(v)
            if not 5<speed<max(30,speed_in*1.3) or cosine(v,vin)>.70:continue
            # Avoid sudden unseen excursions and then a convenient reappearance.
            if np.linalg.norm(point(a)-point(last))>max(70,1.8*speed*dt):continue
            states.append({'path':[a,b],'v':v,'score':quality(a)+quality(b)-.15*(dt-1)})
    complete=[]
    for s in states:
        beam=[s]
        for t in sorted(set(tick(d) for d in obs if tick(d)>tick(s['path'][-1]))):
            extended=[]
            for state in beam:
                p=state['path'][-1];gap=t-tick(p)
                if gap>2:complete.append(state);continue
                extended.append({**state,'score':state['score']-.2})
                for d in obs:
                    if tick(d)!=t:continue
                    delta=point(d)-point(p);speed=np.linalg.norm(state['v'])
                    residual=np.linalg.norm(delta-state['v']*gap)
                    tolerance=7+.18*speed*gap
                    if residual>tolerance:continue
                    extended.append({'path':state['path']+[d],'v':.2*state['v']+.8*delta/gap,
                                     'score':state['score']+quality(d)-residual/tolerance})
            extended.sort(key=lambda q:-q['score']);beam=extended[:12]
        complete+=beam
    viable=[s for s in complete if len(s['path'])>=4 and distance(s['path'][0],s['path'][-1])>=30]
    if not viable:return incoming,{**result,'outgoing_status':'abstain_no_separate_consistent_segment'}
    best=max(viable,key=lambda q:q['score']);post=best['path']
    vout=(point(post[-1])-point(post[0]))/(tick(post[-1])-tick(post[0]))
    angle=math.degrees(math.acos(float(np.clip(cosine(vin,vout),-1,1))))
    return incoming+post,{**result,'outgoing_status':'candidate','outgoing_observations':len(post),
        'outgoing_start_frame':post[0]['frame'],'outgoing_end_frame':post[-1]['frame'],
        'outgoing_score':round(best['score'],3),'turn_angle_degrees':round(angle,2),
        'turn_frames':[post[0]['frame']],'point_count':len(incoming)+len(post),
        'post_event_horizon_distinct_frames':8,'tracker_version':'fixed_segments_v6',
        'max_gap_frames':max(b['frame']-a['frame'] for a,b in zip(incoming+post,(incoming+post)[1:]))}


def summarize_segment_cues(path, diagnostics, keeper_frames=None):
    """Categorize 2-D cues while keeping physical contact/crossing unproved."""
    cue={'event_candidate':'insufficient_track','outcome':'unknown',
         'whole_ball_crossing_established':False,'keeper_contact_established':False,
         'human_verified':False,'review_required':True,'confidence_calibrated':False}
    if diagnostics.get('tracking_status')!='candidate':return cue
    cue['event_candidate']='incoming_flight_ending_unresolved'
    if diagnostics.get('outgoing_status')!='candidate':return cue
    n=diagnostics['incoming_observations'];before,after=path[n-1],path[n]
    distances=[]
    for d in (before,after):
        keeper=(keeper_frames or {}).get(str(d['frame']))
        if not keeper:continue
        x0,y0,x1,y1=keeper['bbox'];x,y=d['x'],d['y']
        distances.append(math.hypot(max(x0-x,0,x-x1),max(y0-y,0,y-y1)))
    near=min(distances) if distances else None
    cue.update(event_candidate='direction_change_candidate',
        change_interval_frames=[before['frame'],after['frame']],
        turn_angle_image_degrees=diagnostics['turn_angle_degrees'],
        closest_kit_bbox_distance_px=round(near,2) if near is not None else None,
        keeper_proximity_supported=near is not None and near<=25,
        possible_causes=['keeper_deflection','post_or_net_rebound','tracking_error'],
        reason='Image-plane turn and kit proximity do not prove contact or whole-ball crossing')
    if cue['keeper_proximity_supported']:cue['event_candidate']='keeper_proximal_deflection_candidate'
    return cue

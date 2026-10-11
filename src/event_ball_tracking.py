"""Experimental source-observation tracker which retains post-impact turns.

No rule here asserts 3-D goal-line crossing. Rebounds and goal-mouth projection
are event cues; even a keeper contact does not logically imply no goal.
"""
from collections import Counter
import math
import numpy as np


def point(d):return np.asarray(d["stabilized_xy"],float)
def distance(a,b):return float(np.linalg.norm(point(a)-point(b)))


def select_stationary_seed(detections,kick,anchors,*,radius=85,max_spread=9):
    """A near-foot cluster supported on several pre-kick frames, camera-corrected."""
    pre=[d for f,items in detections.items() if kick-12<=int(f)<=kick+3 for d in items
         if d.get("stabilized_xy") is not None and "motion_blob" not in d.get("sources",[])]
    candidates=[]
    for d in pre:
        foot=min((np.linalg.norm(point(d)-np.asarray(a,float)) for a in anchors),default=float("inf"))
        if foot>radius:continue
        near=[q for q in pre if distance(q,d)<=max_spread]
        by_frame={}
        for q in near:
            if q["frame"] not in by_frame or q["confidence"]>by_frame[q["frame"]]["confidence"]:by_frame[q["frame"]]=q
        support=sorted(by_frame.values(),key=lambda q:q["frame"])
        if len(support)<3:continue
        xs=np.array([point(q) for q in support]);spread=float(np.median(np.linalg.norm(xs-np.median(xs,axis=0),axis=1)))
        score=len(support)+3*np.mean([q["confidence"] for q in support])-foot/45-spread/6
        candidates.append((score,support))
    if not candidates:return [],{"status":"abstain_no_multiframe_ball_seed"}
    candidates.sort(key=lambda q:-q[0]);support=candidates[0][1]
    return support,{"status":"stationary_seed_candidate","support_frames":[q["frame"] for q in support],
                    "score":float(candidates[0][0]),"independently_verified":False}


def track_ball_event(detections,kick,anchors,goal_center,*,max_gap=3,beam_width=60,allow_turn=True):
    seed,info=select_stationary_seed(detections,kick,anchors)
    if not seed:return [],info
    first=seed[-1];goal=np.asarray(goal_center,float)
    # No synthetic point is inserted for a missing ball.
    states=[{"path":[first],"score":0.,"velocity":None,"turns":[],"misses":0}];finished=[]
    for fid in range(first["frame"]+1,max(map(int,detections))+1):
        expanded=[]
        for s in states:
            last=s["path"][-1];gap=fid-last["frame"]
            if gap<=max_gap:expanded.append({**s,"score":s["score"]-.13,"misses":gap})
            else:finished.append(s);continue
            for d in detections.get(str(fid),detections.get(fid,[])):
                if d.get("stabilized_xy") is None:continue
                movement=point(d)-point(last);speed=float(np.linalg.norm(movement)/gap)
                if speed>180:continue
                # Very small stationary image artifacts are not flight evidence.
                if d.get("motion_mean_abs_gray") is not None and d["motion_mean_abs_gray"]<8 and speed>5:continue
                v=s["velocity"];turn=False;residual=0.
                if v is None:
                    if speed<4 and fid<=kick+8:
                        cost=.5
                    else:
                        vector=goal-point(last)
                        alignment=float(np.dot(movement,vector)/max(1e-9,np.linalg.norm(movement)*np.linalg.norm(vector)))
                        if speed<4 or alignment<.65:continue
                        cost=(1-alignment)*2
                else:
                    prediction=point(last)+v*gap
                    residual=float(np.linalg.norm(point(d)-prediction))
                    tolerance=10+.27*float(np.linalg.norm(v))*gap
                    if residual>tolerance:
                        near_goal=np.linalg.norm(point(last)-goal)<220
                        if not (allow_turn and near_goal and not s["turns"] and gap<=2 and speed<100):continue
                        turn=True;cost=2.5+min(2.,residual/100)
                    else:cost=residual/max(12,tolerance)
                instantaneous=movement/gap
                new_v=(None if speed<4 and v is None else instantaneous if v is None or turn else .30*v+.70*instantaneous)
                # Small motion-only blobs assist the detector; they are never labels.
                blob="motion_blob" in d.get("sources",[])
                reward=.65+min(.8,3*d.get("confidence",0))+(0 if blob else .2)
                if d.get("motion_mean_abs_gray") is not None:reward+=min(.25,d["motion_mean_abs_gray"]/160)
                if speed<1 and fid>kick+8:reward-=.5
                expanded.append({"path":s["path"]+[d],"score":s["score"]+reward-cost,"velocity":new_v,
                                 "turns":s["turns"]+[fid] if turn else list(s["turns"]),"misses":0})
        def score(s):
            progress=distance(s["path"][0],s["path"][-1])
            return s["score"]+min(4,progress/100)
        expanded.sort(key=score,reverse=True);states=[];seen=set()
        for s in expanded:
            d=s["path"][-1];velocity=s["velocity"]
            sig=(d["detection_id"],tuple(np.rint(velocity/12).astype(int)) if velocity is not None else None,bool(s["turns"]))
            if sig in seen:continue
            seen.add(sig);states.append(s)
            if len(states)>=beam_width:break
        if not states:break
    finished+=states
    viable=[s for s in finished if len(s["path"])>=3 and distance(s["path"][0],s["path"][-1])>80]
    if not viable:return seed,{**info,"tracking_status":"abstain_no_flight","path_is_verified":False}
    viable.sort(key=lambda s:s["score"]+min(4,distance(s["path"][0],s["path"][-1])/100),reverse=True)
    best=viable[0];path=best["path"]
    # Crucially: retain both incoming and outgoing observed points. No radial envelope.
    return path,{**info,"tracking_status":"candidate","turn_frames":best["turns"],
                 "beam_score":round(best["score"],3),"point_count":len(path),
                 "max_gap_frames":max(b["frame"]-a["frame"] for a,b in zip(path,path[1:])),
                 "outcome":"unknown","path_is_verified":False,"radial_envelope_applied":False,
                 "observation_sources":dict(Counter("blob" if "motion_blob" in d.get("sources",[]) else "detector" for d in path))}


def summarize_event_cues(path,goal_center,goal_width,keeper_positions=None):
    """A direction-change cue is not a verified save or a no-goal classification."""
    output={"event_candidate":"insufficient_track","outcome":"unknown","crossing_established":False,
            "keeper_contact_established":False,"review_required":True,"human_verified":False}
    if len(path)<6:return output
    xy=np.array([point(p) for p in path]);goal=np.asarray(goal_center,float)
    nearest=int(np.argmin(np.linalg.norm(xy-goal,axis=1)))
    output["closest_goal_projection_frame"]=path[nearest]["frame"]
    if 2<=nearest<len(path)-3:
        vin=xy[nearest]-xy[max(0,nearest-3)];vout=xy[min(len(path)-1,nearest+5)]-xy[nearest]
        angle=math.degrees(math.acos(float(np.clip(np.dot(vin,vout)/max(1e-9,np.linalg.norm(vin)*np.linalg.norm(vout)),-1,1))))
        increase=float(np.linalg.norm(xy[-1]-goal)-np.linalg.norm(xy[nearest]-goal))
        output.update(turn_angle_degrees=round(angle,2),post_turn_distance_increase_px=round(increase,2))
        if angle>65 and increase>max(35,.12*goal_width):
            output["event_candidate"]="rebound_or_deflection_candidate"
            output["reason"]="Observed 2-D turn; save, post or net rebound are not disambiguated"
            return output
    output["event_candidate"]="goalward_flight_observed_ending_unresolved"
    output["reason"]="No valid automatic inference of whole-ball crossing from this image-plane track"
    return output

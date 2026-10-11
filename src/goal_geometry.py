"""Classical goal-frame proposals; never a pitch homography or ground truth.

White-mask Hough segments propose two upright posts and a connecting crossbar.
Conservative evidence gates and runner-up separation allow explicit abstention.
Post coordinates describe a vertical goal plane, NOT the ground plane.
"""
from itertools import combinations
import cv2
import numpy as np


def _line_support(distance, a, b, radius):
    points = np.linspace(a,b,80)
    x=np.clip(np.rint(points[:,0]).astype(int),0,distance.shape[1]-1)
    y=np.clip(np.rint(points[:,1]).astype(int),0,distance.shape[0]-1)
    return float(np.mean(distance[y,x] <= radius))


def detect_goal_frame(frame, *, max_goal_width_fraction=.68, max_post_height_fraction=.45):
    h,w=frame.shape[:2]
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    white=cv2.inRange(hsv,np.array([0,0,170],np.uint8),np.array([179,85,255],np.uint8))
    white[:int(.12*h)]=0
    edges=cv2.Canny(white,50,150)
    lines=cv2.HoughLinesP(edges,1,np.pi/360,threshold=max(25,int(.03*h)),
                         minLineLength=max(30,int(.07*h)),maxLineGap=max(6,int(.018*h)))
    # The dense net can interrupt Canny post edges. A vertical opening retains
    # the solid upright itself while removing most lettering and net diagonals.
    posts=cv2.morphologyEx(white,cv2.MORPH_OPEN,np.ones((max(15,int(.04*h)),2),np.uint8))
    post_lines=cv2.HoughLinesP(posts,1,np.pi/360,threshold=max(20,int(.03*h)),
                              minLineLength=max(30,int(.07*h)),maxLineGap=max(6,int(.018*h)))
    if post_lines is not None:
        lines=post_lines if lines is None else np.concatenate((lines,post_lines),axis=0)
    if lines is None:
        return {"status":"abstain","candidates":[],"reason":"no_white_segments"}
    vertical=[]
    segments=sorted(lines[:,0,:].astype(float),key=lambda v:-abs(v[3]-v[1]))
    for x1,y1,x2,y2 in segments:
        if y1>y2: x1,y1,x2,y2=x2,y2,x1,y1
        length=y2-y1
        if length < .075*h or abs(x2-x1)>.22*length or length>max_post_height_fraction*h:
            continue
        match=next((i for i,v in enumerate(vertical) if abs((x1+x2)/2-(v[0]+v[2])/2)<.005*w and
                   min(y2,v[3])-max(y1,v[1])>-.015*h),None)
        if match is None:
            vertical.append((x1,y1,x2,y2))
        else:
            v=vertical[match]
            top=(x1,y1) if y1<v[1] else v[:2]
            bottom=(x2,y2) if y2>v[3] else v[2:]
            vertical[match]=(*top,*bottom)
    distance=cv2.distanceTransform(255-white,cv2.DIST_L2,3)
    green=(hsv[...,0]>=28)&(hsv[...,0]<=95)&(hsv[...,1]>=45)
    candidates=[]
    for a,b in combinations(vertical,2):
        if a[0]>b[0]: a,b=b,a
        width=b[0]-a[0]; heights=[a[3]-a[1],b[3]-b[1]]
        if not .09*w<=width<=max_goal_width_fraction*w or not 1.1<=width/np.mean(heights)<=6:
            continue
        if min(heights)/max(heights)<.50 or abs(a[1]-b[1])>.45*width:
            continue
        crossbar=_line_support(distance,a[:2],b[:2],max(3,.005*h))
        if crossbar<.65:
            continue
        bottom_green=[]
        for post in (a,b):
            x,y=int(post[2]),int(post[3]); rad=max(6,int(.02*h))
            patch=green[max(0,y):min(h,y+rad),max(0,x-rad):min(w,x+rad)]
            bottom_green.append(float(np.mean(patch)) if patch.size else 0)
        ground_support=min(bottom_green)
        if ground_support<.18:
            continue
        # Net-like white texture inside helps reject unrelated field markings.
        x0,x1=int(a[0]+.05*width),int(b[0]-.05*width)
        y0,y1=int(max(a[1],b[1])+5),int(min(a[3],b[3])-5)
        if x1<=x0 or y1<=y0: continue
        density=float(np.mean(white[y0:y1,x0:x1]>0))
        score=.55*crossbar+.25*ground_support+.20*min(1,density/.12)
        candidates.append({"corners":[list(a[:2]),list(b[:2]),list(b[2:]),list(a[2:])],
                           "bbox":[min(a[0],a[2]),min(a[1],b[1]),max(b[0],b[2]),max(a[3],b[3])],
                           "score":round(score,4),"crossbar_support":crossbar,
                           "ground_support":ground_support,"interior_white_density":density})
    candidates.sort(key=lambda c:-c["score"])
    # Near-identical line-edge proposals are one region, not distinct rivals.
    distinct=[]
    for c in candidates:
        if not any(np.linalg.norm(np.array(c["bbox"])-d["bbox"])/w<.05 for d in distinct):
            distinct.append(c)
    top=distinct[0] if distinct else None
    ambiguous=len(distinct)>1 and distinct[1]["score"]>=distinct[0]["score"]-.08
    supported=bool(top and top["score"]>=.72 and not ambiguous)
    return {"status":"candidate_supported" if supported else "abstain",
            "selected":top if supported else None,"candidates":distinct[:12],
            "reason":"supported_geometry" if supported else "insufficient_or_ambiguous_geometry",
            "human_verified":False,"metric_calibration":False}


def select_goal_occupant(frame, tracks, frame_id, max_frame_gap=2):
    """Associate a standing person with an image goal mouth before the kick.

    This does not assert goalkeeper identity. All candidate scores and geometry
    are retained; no manual identity labels or post-kick outcomes are consulted.
    """
    goal=detect_goal_frame(frame)
    if goal["status"]!="candidate_supported":
        return {"status":"abstain","seed_track_id":None,"goal":goal,"candidates":[]}
    corners=np.array(goal["selected"]["corners"],np.float32)
    bl,br=corners[3],corners[2]
    gh=float(np.mean([np.linalg.norm(corners[0]-bl),np.linalg.norm(corners[1]-br)]))
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    green=(hsv[...,0]>=28)&(hsv[...,0]<=95)&(hsv[...,1]>=45)
    candidates=[]
    for t in tracks:
        near=[o for o in t.get("observations",[]) if abs(o["frame"]-frame_id)<=max_frame_gap]
        if not near:continue
        o=min(near,key=lambda v:abs(v["frame"]-frame_id))
        box=np.array(o["bbox"],float);bh=box[3]-box[1]
        cx=float((box[0]+box[2])/2);cy=float(box[1]+.60*bh)
        if not .35*gh<=bh<=1.5*gh or cv2.pointPolygonTest(corners,(cx,cy),False)<0:
            continue
        gy=float(bl[1]+(br[1]-bl[1])*(cx-bl[0])/max(1,br[0]-bl[0]))
        foot_distance=abs(box[3]-gy)/max(1,gh)
        if foot_distance>.32:continue
        rad=max(5,int(.18*bh));x,y=int(cx),int(box[3])
        patch=green[max(0,y):min(frame.shape[0],y+rad),max(0,x-rad):min(frame.shape[1],x+rad)]
        support=float(np.mean(patch)) if patch.size else 0
        if support<.25:continue
        score=.6*max(0,1-foot_distance/.32)+.25*support+.15*float(o.get("box_confidence",0))
        candidates.append({"track_id":t["track_id"],"score":float(score),"bbox":box.tolist(),
                           "observed_frame":o["frame"],"foot_to_goal_line_height_ratio":float(foot_distance),
                           "field_support":support})
    candidates.sort(key=lambda c:-c["score"])
    distinct=[]
    for c in candidates:
        def overlap(other):
            a,b=np.array(c["bbox"]),np.array(other["bbox"])
            area=float(np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))))
            return area/max(1,min(float(np.prod(a[2:]-a[:2])),float(np.prod(b[2:]-b[:2]))))
        if not any(overlap(d)>.70 for d in distinct):distinct.append(c)
    ambiguous=len(distinct)>1 and distinct[1]["score"]>=distinct[0]["score"]-.12
    selected=distinct[0] if distinct and not ambiguous else None
    return {"status":"candidate_supported" if selected else "abstain",
            "seed_track_id":selected["track_id"] if selected else None,"goal":goal,
            "candidates":candidates,"distinct_candidates":distinct,"ambiguous_identity":ambiguous,
            "human_verified":False,"accuracy_validated":False}

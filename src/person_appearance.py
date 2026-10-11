"""Torso-only HSV signatures for conservative within-shot identity continuity."""
from collections import defaultdict
import cv2
import numpy as np


def torso_histogram(frame, observation):
    h,w=frame.shape[:2];box=observation.get("bbox",[])
    if len(box)!=4:return None
    x1,y1,x2,y2=box
    x0,y0=max(0,int(x1)),max(0,int(y1));xr,yr=min(w,int(x2)+1),min(h,int(y2)+1)
    if xr-x0<4 or yr-y0<4:return None
    crop=frame[y0:yr,x0:xr];mask=np.zeros(crop.shape[:2],np.uint8)
    points=np.array(observation.get("keypoints",[]));conf=np.array(observation.get("keypoint_confidences",[]))
    indices=[5,6,12,11]
    if points.shape==(17,2) and conf.shape==(17,) and np.min(conf[indices])>=.25:
        polygon=np.round(points[indices]-[x0,y0]).astype(np.int32)
        hull=cv2.convexHull(polygon)
        if cv2.contourArea(hull)>=8:cv2.fillConvexPoly(mask,hull,255)
    if np.count_nonzero(mask)<8:
        # Fallback stays central and away from grass at the bounding-box edges.
        hh,ww=mask.shape;mask[int(.18*hh):max(int(.18*hh)+1,int(.58*hh)),int(.25*ww):max(int(.25*ww)+1,int(.75*ww))]=255
    if np.count_nonzero(mask)<8:return None
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    hist=cv2.calcHist([hsv],[0,1,2],mask,[12,6,6],[0,180,0,256,0,256])
    # Soft bins avoid treating small illumination/pose changes as a new shirt.
    from scipy.ndimage import gaussian_filter
    hist=gaussian_filter(hist,sigma=(.50,.55,.55),mode=("wrap","nearest","nearest")).ravel()
    return hist/max(float(hist.sum()),1)


def similarity(a,b):
    if a is None or b is None:return None
    return float(np.clip(np.sum(np.sqrt(np.asarray(a)*np.asarray(b))),0,1))


def appearance_scores(result,seed_id,anchor_frame,start,end):
    observations=defaultdict(list)
    for t in result.get("tracks",[]):
        for o in t.get("observations",[]):
            if start<=o["frame"]<=end:observations[o["frame"]].append((t["track_id"],o))
    histograms={};cap=cv2.VideoCapture(str(result.get("video_path","")))
    cap.set(1,max(0,start))
    for frame_id in range(max(0,start),end+1):
        ok,frame=cap.read()
        if not ok:break
        for tid,o in observations.get(frame_id,[]):histograms[(frame_id,tid)]=torso_histogram(frame,o)
    cap.release()
    templates=[v for (f,tid),v in histograms.items() if tid==seed_id and anchor_frame-10<=f<=anchor_frame and v is not None]
    if not templates:return {},{"status":"unavailable","seed_id":seed_id}
    template=np.mean(templates,axis=0);template/=max(float(template.sum()),1e-9)
    scores={key:similarity(template,hist) for key,hist in histograms.items()}
    return scores,{"status":"available","method":"soft_bin_torso_HSV_Bhattacharyya_coefficient",
                   "seed_id":seed_id,"template_observations":len(templates),"minimum_similarity":.50,
                   "human_verified":False,"calibrated_probability":False}

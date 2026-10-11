"""Small-sample, unsupervised HSV kit mask for ball/keeper ambiguity cues.

An existing pre-kick keeper candidate seeds the colour profile. The mask is
neither segmentation ground truth nor evidence of physical ball contact.
"""
import cv2
import numpy as np


def learn_kit_profile(frame, bbox):
    x0,y0,x1,y1=map(float,bbox);w,h=x1-x0,y1-y0
    crop=frame[round(y0+.25*h):round(y0+.60*h),round(x0+.28*w):round(x1-.28*w)]
    if crop.size==0:return None
    pixels=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV).reshape(-1,3).astype(float)
    pixels=pixels[(pixels[:,1]>70)&(pixels[:,2]>80)]
    if len(pixels)<20:return None
    hist,_=np.histogram(pixels[:,0],bins=np.arange(0,181,6),weights=pixels[:,1])
    peak=int(hist.argmax())*6+3
    delta=np.minimum(np.abs(pixels[:,0]-peak),180-np.abs(pixels[:,0]-peak))
    selected=pixels[delta<=10]
    return {"hue":float(np.median(selected[:,0])),"hue_tolerance":11.,
            "saturation_min":max(70.,float(np.percentile(selected[:,1],10))-30),
            "value_min":max(70.,float(np.percentile(selected[:,2],10))-30),
            "sample_pixels":len(selected),"human_verified":False}


def kit_mask(frame, profile, roi):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV).astype(float)
    delta=np.abs(hsv[...,0]-profile['hue']);delta=np.minimum(delta,180-delta)
    mask=((delta<=profile['hue_tolerance'])&(hsv[...,1]>=profile['saturation_min'])&
          (hsv[...,2]>=profile['value_min'])).astype(np.uint8)*255
    x0,y0,x1,y1=map(int,roi);region=np.zeros_like(mask);region[y0:y1,x0:x1]=255
    mask=cv2.bitwise_and(mask,region)
    return cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))


def annotate_kit_ambiguity(frame, profile, roi, detections, previous_center):
    mask=kit_mask(frame,profile,roi)
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8)
    options=[i for i in range(1,n) if stats[i,4]>=40 and stats[i,4]<7000
             and np.linalg.norm(centers[i]-previous_center)<100]
    if not options:return None
    chosen=min(options,key=lambda i:np.linalg.norm(centers[i]-previous_center)-min(20,stats[i,4]/50))
    component=(labels==chosen).astype(np.uint8)
    for d in detections:
        x,y=round(d['x']),round(d['y']);radius=20
        patch=component[max(0,y-radius):y+radius+1,max(0,x-radius):x+radius+1]
        fraction=float(np.mean(patch)) if patch.size else 0.
        d['kit_neighborhood_fraction']=fraction
        if fraction>.045:d['person_overlap']=True
    x,y,w,h,area=map(int,stats[chosen]);center=centers[chosen].tolist()
    return {'center':center,'bbox':[x,y,x+w,y+h],'mask_area':area,
            'identity_verified':False,'source':'HSV_seeded_kit_component'}

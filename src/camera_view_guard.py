"""Experimental camera guards, separate from the frozen v2 classifier.

Field colour and kit similarity remain weak cues, NOT identity recognition.
Unknown is preferable to labelling a supporter or teammate as the kicker.
"""
from copy import deepcopy
from collections import Counter
import math
import cv2
import numpy as np
from src.camera_timing_v2 import classify_view


def closeup_context(frame, people):
    h,w=frame.shape[:2]
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    green=(hsv[:,:,0]>=28)&(hsv[:,:,0]<=95)&(hsv[:,:,1]>=45)&(hsv[:,:,2]>=35)
    outside=np.ones((h,w),bool)
    for p in people:
        x0,y0,x1,y1=map(int,p['bbox'])
        outside[max(0,y0):min(h,y1+1),max(0,x0):min(w,x1+1)]=False
    # Denominator is full frame: tiny visible patches cannot supply strong context.
    return {'green_outside_people_fraction':float(np.mean(green&outside)),
            'field_mask_is_not_semantic_segmentation':True}


def closeup_guard(people, context):
    candidates=[p for p in people if p.get('role')=='kicker_candidate'
                and p['height_fraction']>.58 and p['area_fraction']>.10]
    if not candidates:return {'allowed':False,'reason':'no_large_kit_candidate'}
    target=max(candidates,key=lambda p:p['area_fraction'])
    competitors=[p for p in people if p is not target and p.get('role')=='kicker_candidate'
                 and p['area_fraction']>=.4*target['area_fraction'] and p['height_fraction']>.35]
    foreground=[p for p in people if p['area_fraction']>=.06 and p['height_fraction']>.35]
    if competitors:return {'allowed':False,'reason':'multiple_comparable_same_kit_people'}
    if len(foreground)>=4:return {'allowed':False,'reason':'crowded_foreground'}
    if context.get('green_outside_people_fraction',0)<.12:
        return {'allowed':False,'reason':'insufficient_field_context_not_proof_of_spectator'}
    return {'allowed':True,'reason':'isolated_kit_candidate_with_field_context_identity_still_unverified'}


def classify_guarded(samples):
    original=classify_view(samples)
    # The old elif entered a keeper-size branch even when its back-view test
    # subsequently failed. That prevented the goal-geometry branch from running.
    # This second pass explicitly asks only for the geometry alternative. It
    # never substitutes another person's identity, or marks the keeper absent.
    geometry_pass=deepcopy(samples)
    for s in geometry_pass:
        for p in s['people']:
            if p.get('role')=='keeper_candidate':p['role']='unknown'
    geometry=classify_view(geometry_pass)
    votes=[];reasons=[];guards=[]
    for i,s in enumerate(samples):
        vote=original['sample_votes'][i];reason=original['sample_reasons'][i];guard=None
        if vote=='unknown' and s.get('goal') and geometry['sample_votes'][i] in {
                'back_of_kicker','angle_45_right','other_oblique_left','oblique_side_unresolved'}:
            vote=geometry['sample_votes'][i]
            reason='goal_geometry_fallback_after_keeper_back_view_not_supported; '+geometry['sample_reasons'][i]
        if vote=='kicker_closeup':
            guard=closeup_guard(s['people'],s.get('closeup_context',{}))
            if not guard['allowed']:vote='person_closeup_identity_unresolved'
            reason=guard['reason']
        votes.append(vote);reasons.append(reason);guards.append(guard)
    counts=Counter(votes);best,n=counts.most_common(1)[0] if counts else ('unknown',0)
    return {'view_candidate':best if n>=max(2,math.ceil(len(samples)*.6)) else 'unknown',
            'sample_votes':votes,'sample_reasons':reasons,'closeup_guards':guards,
            'old_sample_votes':original['sample_votes'],'identity_verified':False,
            'accuracy_validated':False,'human_verified':False,'review_required':True}

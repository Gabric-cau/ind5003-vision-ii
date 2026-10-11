"""Visible hip height inside independently observed silhouette phase candidates.

The contact-bracket ground path is still an uncalibrated 2-D assumption. This
is stronger anatomical evidence than a mask-centroid proxy, not human truth.
"""
import numpy as np


def contact_bracket_hip_height(samples,silhouette,phase,reference,fps,pose_landing_frame=None):
    out=dict(policy='visible_hip_contact_bracket_height_v1',supported=False,height_candidate='unknown',
             reasons=[],requires_review=True,physical_height_metres=None,accuracy_estimate=None,
             ground_path_assumption='linear image-plane ground intercept between silhouette contact candidates',
             body_location_source='confident observed shoulders and hips inside keeper mask',
             phase_boundary_source='provisional silhouette stage, not true contact annotation')
    def reject(reason):out['reasons'].append(reason);return out
    if not phase.get('complete_first_dive_candidate') or not phase.get('extension_frames'):
        return reject('missing_complete_extension_and_contact_bracket')
    out['independent_pose_landing_frame']=pose_landing_frame
    if pose_landing_frame is not None and abs(phase['landing_frame']-pose_landing_frame)/fps > .15:
        return reject('pose_and_silhouette_contact_boundaries_disagree')
    anchors=phase.get('contact_anchor_frames')
    if not anchors or not reference:return reject('missing_reference')
    byframe={s['frame']:s for s in silhouette};a,b=(byframe.get(f) for f in anchors)
    if a is None or b is None:return reject('missing_contact_anchor_pixels')
    slope=reference['ground_line'][0];scale=reference['body_scale_px']
    intercept_a=a['lower_edge'][1]-slope*a['lower_edge'][0]
    intercept_b=b['lower_edge'][1]-slope*b['lower_edge'][0]
    if abs(intercept_b-intercept_a)/scale > .65:return reject('large_depth_change_ground_path_unreliable')
    frames=phase['extension_frames'];selected=sorted([s for s in samples if s['frame'] in set(frames)],key=lambda s:s['frame'])
    out.update(observed_pose_frames=[s['frame'] for s in selected],expected_extension_frames=frames)
    if len(selected)<3:return reject('too_few_visible_extension_hips')
    # No claiming a peak across an unobserved stretch or a missed start/end.
    gaps=[selected[0]['frame']-frames[0],frames[-1]-selected[-1]['frame']]
    gaps += [y['frame']-x['frame'] for x,y in zip(selected,selected[1:])]
    if max(gaps)/fps > .15:return reject('extension_peak_pose_gap')
    if len(selected)/len(frames) < .4:return reject('low_extension_pose_coverage')
    measured=[]
    for s in selected:
        fraction=(s['frame']-a['frame'])/max(1,b['frame']-a['frame'])
        intercept=intercept_a+fraction*(intercept_b-intercept_a)
        height=(slope*s['hip'][0]+intercept-s['hip'][1])/scale
        if not -.05 <= height <= 1.6:return reject('implausible_corrected_hip_height')
        measured.append(dict(frame=s['frame'],hip_clearance_body=float(height),min_confidence=s['min_confidence']))
    peaks=[(float(np.median([s['hip_clearance_body'] for s in measured[i:i+3]])),
            [s['frame'] for s in measured[i:i+3]]) for i in range(len(measured)-2)
           if (measured[i+2]['frame']-measured[i]['frame'])/fps <= .15]
    if not peaks:return reject('no_connected_three_sample_peak')
    peak,support=max(peaks,key=lambda p:p[0]);out.update(peak_hip_clearance_body=peak,supporting_frames=support,measurements=measured)
    if any(abs(peak-threshold)<.04 for threshold in [.35,.75]):return reject('height_near_category_boundary')
    out.update(supported=True,height_candidate='low' if peak<.35 else 'high' if peak>.75 else 'mid')
    return out

"""Observed early lateral motion can support direction before a full dive ends.

Coordinates stay in the image frame. Identity, physical contact, camera motion,
and goalkeeper-relative left/right are not established by this cached evidence.
All thresholds are engineering rules, not calibrated accuracy probabilities.
"""
import math
from statistics import median


def early_direction_evidence(record, *, min_samples=6, min_span_sec=.20,
                             min_density=.50, lateral_threshold=.30,
                             max_gap_sec=.20, onset_tolerance_sec=.10):
    """Use one onset-connected prefix; never bridge a long gap to a later person."""
    fps=float(record.get('fps') or 30)
    if not math.isfinite(fps) or fps<=0:raise ValueError('FPS must be positive and finite')
    kick=int(record['kick_frame']);selected=record.get('selected_candidate') or {}
    out={'policy':'onset_connected_lateral_motion_v1','direction':'unknown','supported':False,
         'reasons':[],'identity_verified':False,'camera_motion_compensated':False,
         'coordinate_system':'source_image_x_right','min_samples':min_samples,
         'min_span_sec':min_span_sec,'min_density':min_density,'lateral_threshold_body':lateral_threshold,
         'max_gap_sec':max_gap_sec,'onset_tolerance_sec':onset_tolerance_sec}
    def reject(reason):
        out['reasons'].append(reason);return out
    if selected.get('qualifies') is not True or selected.get('is_selected_kicker') is not False:
        return reject('no_qualified_non_kicker_track')
    if record.get('goal_identity_evidence',{}).get('ambiguous_identity') is True:
        return reject('ambiguous_goalkeeper_identity')
    if selected.get('pre_coverage',0)<.50 or selected.get('pose_quality',0)<.50:
        return reject('weak_pre_kick_track_support')
    def clean(rows):
        frames={};conflicts=[]
        for row in rows:
            if len(row)<4:continue
            t,x,y,s=map(float,row[:4])
            if not all(map(math.isfinite,(t,x,y,s))) or s<=1:continue
            frame=round(t*fps);point=(frame,x,y,s)
            if frame in frames and frames[frame]!=point:conflicts.append(frame)
            frames[frame]=point
        return [frames[f] for f in sorted(frames)],conflicts
    pre,pre_conflicts=clean(selected.get('pre_trajectory',[]))
    pre=[p for p in pre if kick-round(.25*fps)<=p[0]<=kick][-3:]
    if len(pre)<3:return reject('missing_near_kick_baseline')
    if pre_conflicts:return reject('conflicting_duplicate_pre_frame')
    baseline_x=median(p[1] for p in pre);scale=median(p[3] for p in pre)
    if max(p[0] for p in pre)-min(p[0] for p in pre)>round(.20*fps):
        return reject('disconnected_pre_kick_baseline')
    post,conflicts=clean(selected.get('post_trajectory',[]))
    post=[p for p in post if kick<=p[0]<=kick+round(fps)]
    if conflicts:return reject('conflicting_duplicate_post_frame')
    if not post:return reject('no_postkick_observations')
    if post[0][0]-kick>round(onset_tolerance_sec*fps):return reject('missing_dive_onset')
    prefix=[];stopped=None
    for point in post:
        if prefix:
            dt=(point[0]-prefix[-1][0])/fps
            if dt>max_gap_sec+1e-9:stopped='gap';break
            jump=math.hypot(point[1]-prefix[-1][1],point[2]-prefix[-1][2])/scale
            if jump>.35+4*dt:stopped='coordinate_jump';break
        if not .25<=point[3]/scale<=2.:stopped='scale_discontinuity';break
        prefix.append(point)
    out.update(observed_frames=[p[0] for p in prefix],prefix_stop_reason=stopped,
               baseline_x=baseline_x,baseline_body_scale_px=scale,
               sample_count=len(prefix),kick_anchor_status=record.get('kick_frame_confidence','unknown'))
    if len(prefix)<min_samples:return reject('insufficient_onset_connected_samples')
    span=(prefix[-1][0]-prefix[0][0])/fps
    density=len(prefix)/(prefix[-1][0]-prefix[0][0]+1)
    out.update(span_sec=span,sample_density=density)
    if span<min_span_sec-1e-9:return reject('short_motion_span')
    if density<min_density:return reject('sparse_motion_prefix')
    dx=[(p[1]-baseline_x)/scale for p in prefix]
    tail=dx[-3:];lateral=median(tail);sign=1 if lateral>0 else -1
    out['sustained_lateral_displacement_body']=lateral
    if abs(lateral)<lateral_threshold or min(sign*x for x in tail)<.22:
        return reject('no_sustained_lateral_motion')
    if (prefix[-1][0]-prefix[-3][0])/fps<.06-1e-9:
        return reject('tail_support_too_brief')
    changes=[b-a for a,b in zip(dx,dx[1:])]
    substantial=[c for c in changes if abs(c)>=.02]
    agreement=sum(sign*c>0 for c in substantial)/len(substantial) if substantial else 0
    variation=sum(abs(c) for c in changes)
    efficiency=abs(dx[-1]-dx[0])/max(variation,1e-9)
    out.update(directional_increment_agreement=agreement,net_to_total_lateral_motion=efficiency)
    if agreement<.75 or efficiency<.65:return reject('oscillating_or_reversing_motion')
    # Moving to both sides strongly cannot establish one early dive direction.
    if min(sign*x for x in dx)<-.30:return reject('opposite_lateral_excursion')
    out.update(direction='image_right' if sign>0 else 'image_left',supported=True)
    return out

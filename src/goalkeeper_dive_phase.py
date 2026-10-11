"""Image-plane first-dive phases and pre-landing height, not metric 3-D height.

Explicit engineering thresholds; no user labels or outcome inputs. A low final
position cannot label an earlier high extension. All times are clip-relative.
"""
import math
import numpy as np
from src.goalkeeper_sam2 import points_inside

POLICY = dict(min_pose_confidence=.4, onset_shift_body=.15, onset_lean_degrees=30,
              landing_hip_clearance_body=.30, landing_bottom_clearance_body=.05,
              min_sustained_samples=3, max_sample_gap_sec=.15,
              low_height_below_body=.35, high_height_above_body=.75,
              category_boundary_margin_body=.04)


def pose_measurement(observation, mask, offset, inverse_camera, scale, ground_line):
    """Only observed confident torso joints inside the locked body mask."""
    kp = np.asarray(observation.get('keypoints', []), float)
    cf = np.asarray(observation.get('keypoint_confidences', []), float)
    if kp.shape != (17, 2) or cf.shape != (17,):
        return None
    required = [5, 6, 11, 12]
    if not np.isfinite(kp[required]).all() or not np.isfinite(cf[required]).all():
        return None
    inside = points_inside(mask, kp - offset)
    if any(cf[j] < POLICY['min_pose_confidence'] or not inside[j] for j in required):
        return None
    homo = np.c_[kp, np.ones(17)] @ np.asarray(inverse_camera).T
    stable = homo[:, :2] / homo[:, 2:3]
    shoulder = stable[[5, 6]].mean(0)
    hip = stable[[11, 12]].mean(0)
    torso = shoulder - hip
    if np.linalg.norm(torso) < max(3, .08 * scale):
        return None
    slope, intercept = ground_line
    height = (slope * hip[0] + intercept - hip[1]) / scale
    hands = [j for j in [9, 10] if cf[j] >= .4 and inside[j] and np.isfinite(stable[j]).all()]
    wrists = [(slope * stable[j, 0] + intercept - stable[j, 1]) / scale for j in hands]
    return dict(frame=int(observation['frame']), hip=hip.tolist(), shoulder=shoulder.tolist(),
                hip_clearance_body=float(height),
                torso_lean_degrees=float(math.degrees(math.atan2(abs(torso[0]), abs(torso[1])))),
                wrist_clearance_body=max(wrists) if wrists else None,
                min_confidence=float(cf[required].min()), source=observation.get('source_pose', 'cached'))


def sustained_start(samples, predicate, fps):
    run = []
    for s in samples:
        if not predicate(s) or run and (s['frame'] - run[-1]['frame']) / fps > POLICY['max_sample_gap_sec']:
            run = []
        if predicate(s):
            run.append(s)
            if len(run) >= POLICY['min_sustained_samples'] and (run[-1]['frame']-run[0]['frame'])/fps >= .06:
                return run[0]['frame']
    return None


def analyse_phases(samples, kick, fps, scale, *, tracking_stop_reason=None, tracking_last=None,
                   source_censored=False, camera_supported=True):
    samples = sorted(samples, key=lambda s: s['frame'])
    out = dict(policy='first_dive_prelanding_hip_clearance_v1', height_candidate='unknown',
               height_supported=False, complete_first_dive_observed=False,
               physical_height_metres=None, coordinate_system='camera_stabilized_image_plane',
               identity_verified=False, accuracy_estimate=None, reasons=[], phases=[],
               thresholds=POLICY.copy(), samples=samples, tracking_stop_reason=tracking_stop_reason,
               source_censored=source_censored)
    def reject(reason):
        out['reasons'].append(reason)
        return out
    near_pre = [s for s in samples if kick-round(.5*fps) <= s['frame'] < kick]
    pre = [s for s in near_pre if s['frame'] < kick-round(.2*fps)]
    if len(pre) < 3: pre = near_pre
    if len(pre) < 3 or not near_pre or kick-near_pre[-1]['frame'] > round(.15*fps):
        return reject('missing_near_kick_torso_reference')
    ref = np.median([s['hip'] for s in pre[:5]], axis=0)
    event_start = max(pre[-1]['frame']+1,kick-round(.25*fps))
    post = [s for s in samples if s['frame'] >= event_start]
    if not post or post[0]['frame']-event_start > round(.15*fps):
        return reject('missing_postkick_onset')
    # Stop at a pose gap: no jumping to a later roll/recovery or different person.
    connected = []
    prev = event_start
    for s in post:
        if (s['frame']-prev)/fps > POLICY['max_sample_gap_sec']:
            out['pose_prefix_stop_frame'] = s['frame']
            break
        connected.append(s)
        prev = s['frame']
    out['connected_pose_frames'] = [s['frame'] for s in connected]
    onset = sustained_start(connected, lambda s: s['torso_lean_degrees'] >= POLICY['onset_lean_degrees']
                            or np.linalg.norm(np.asarray(s['hip'])-ref)/scale >= POLICY['onset_shift_body'], fps)
    if onset is None:
        return reject('no_sustained_first_dive_onset')
    active = [s for s in connected if s['frame'] >= onset]
    landing = sustained_start(active, lambda s: s['frame'] >= onset+round(.1*fps)
        and s['torso_lean_degrees'] >= 45
        and s['hip_clearance_body'] <= POLICY['landing_hip_clearance_body']
        and s.get('bottom_clearance_body', 1) <= POLICY['landing_bottom_clearance_body'], fps)
    out['phases'] = [dict(name='preparation', start_frame=pre[0]['frame'], end_frame=kick-1),
                     dict(name='push_off_or_lunge', start_frame=onset, end_frame=onset+round(.1*fps)-1),
                     dict(name='extension_before_ground_contact', start_frame=onset,
                          end_frame=(landing-1 if landing is not None else prev))]
    if landing is not None:
        out['phases'].append(dict(name='ground_contact_and_landing', start_frame=landing, end_frame=prev))
        out['complete_first_dive_observed'] = True
    else:
        out['reasons'].append('landing_not_observed_or_not_supported')
    ref_height = float(np.median([s['hip_clearance_body'] for s in pre[:5]]))
    extension = [s for s in active if (landing is None or s['frame'] < landing)
                 and (s['torso_lean_degrees'] >= 45 or s['hip_clearance_body'] > ref_height+.15)]
    out.update(onset_frame=onset, landing_frame=landing,
               extension_frames=[s['frame'] for s in extension], landing_frames_excluded_from_height=True)
    if not camera_supported:
        return reject('camera_compensation_not_supported')
    if landing is None:
        return reject('extension_peak_may_be_unobserved')
    if len(extension) < 3 or (extension[-1]['frame']-extension[0]['frame'])/fps < .06:
        return reject('too_few_prelanding_samples')
    if any(s.get('bottom_clearance_body',0) < -.2 for s in extension):
        return reject('depth_motion_invalidates_fixed_ground_reference')
    # Use the maximum three-sample median, not a lone possibly hallucinated peak.
    peaks = [(float(np.median([p['hip_clearance_body'] for p in extension[i:i+3]])),
              [p['frame'] for p in extension[i:i+3]]) for i in range(len(extension)-2)]
    height, frames = max(peaks, key=lambda p: p[0])
    out.update(peak_hip_clearance_body=height, height_supporting_frames=frames,
               visible_wrist_peak_body=max((s['wrist_clearance_body'] for s in extension
                                           if s.get('wrist_clearance_body') is not None), default=None))
    if not -.05 <= height <= 1.6:
        return reject('implausible_image_plane_clearance')
    if any(abs(height-b) < POLICY['category_boundary_margin_body'] for b in [.35, .75]):
        return reject('height_near_category_boundary')
    out.update(height_candidate='low' if height < .35 else 'high' if height > .75 else 'mid',
               height_supported=True)
    return out


def mask_phase_candidates(rows,kick,fps,scale):
    """Provisional silhouette phases and contact-bracketed 2-D height proxy.

    Compact sideways silhouette plus lower-edge settling suggests landing,
    not proof of ankle contact. Ground path between contact anchors is assumed
    linear in the stabilized image. Never report this as metric 3-D height.
    """
    out = dict(policy='silhouette_settling_contact_bracket_v1',supported=False,
               height_proxy_candidate='unknown',phases=[],reasons=[],
               physical_ground_plane_calibrated=False,complete_first_dive_candidate=False,
               source='SAM2 visible silhouette, not ankle contact ground truth',
               height_definition='pre-landing silhouette centroid clearance / pre-kick body height',
               ground_path_assumption='linear projected contact path between last upright and settled silhouette',
               identity_verified=False,accuracy_estimate=None,requires_review=True)
    pre = [s for s in rows if s['frame'] < kick-round(.2*fps) and s['axis_lean'] < 35]
    if len(pre) < 3:
        out['reasons'].append('no_upright_silhouette_reference'); return out
    ref = np.median([s['center'] for s in pre[:5]],axis=0)
    event = [s for s in rows if s['frame'] >= kick-round(.25*fps)]
    onset = sustained_start(event,lambda s:s['axis_lean'] >= 35 or abs(s['center'][0]-ref[0])/scale >= .15,fps)
    if onset is None:
        out['reasons'].append('no_sustained_silhouette_onset'); return out
    active = [s for s in rows if s['frame'] >= onset]
    prefoot = np.median([s['lower_edge'][1] for s in pre[-5:]])
    landing = None
    for i in range(len(active)-4):
        run = active[i:i+5]
        if run[0]['frame'] < onset+round(.15*fps): continue
        if any(b['frame']-a['frame'] > 1 for a,b in zip(run,run[1:])): continue
        compact = all(s['axis_lean'] >= 45 and s['vertical_extent_body'] < .65 for s in run)
        yy = [s['lower_edge'][1] for s in run]
        settled = np.ptp(yy)/scale < .08 and np.median(yy) > prefoot+.08*scale
        if compact and settled:
            landing = run[0]['frame']; break
    out.update(onset_frame=onset,landing_frame=landing,
               phases=[dict(name='push_off_or_lunge_candidate',start_frame=onset,
                            end_frame=landing-1 if landing else active[-1]['frame'])])
    if landing is None:
        out['reasons'].append('no_sustained_compact_lower_edge_settling'); return out
    out['phases'].append(dict(name='landing_candidate',start_frame=landing,end_frame=active[-1]['frame']))
    out['complete_first_dive_candidate'] = True
    anchors = [s for s in rows if s['frame'] < onset and s['axis_lean'] < 35]
    if not anchors:
        out['reasons'].append('no_last_upright_contact_anchor'); return out
    anchor = anchors[-1]; landrow = next(s for s in rows if s['frame']==landing)
    extension = [s for s in active if s['frame'] < landing and s['axis_lean'] >= 45 and s['elongation'] >= 1.6]
    if len(extension)<3:
        out['reasons'].append('too_few_prelanding_sideways_silhouettes'); return out
    measured = []
    for s in extension:
        fraction = (s['frame']-anchor['frame'])/max(1,landing-anchor['frame'])
        ground = (1-fraction)*anchor['lower_edge'][1]+fraction*landrow['lower_edge'][1]
        measured.append(dict(frame=s['frame'],centroid_clearance_body=float((ground-s['center'][1])/scale)))
    peaks = [(float(np.median([s['centroid_clearance_body'] for s in measured[i:i+3]])),
              [s['frame'] for s in measured[i:i+3]]) for i in range(len(measured)-2)
             if measured[i+2]['frame']-measured[i]['frame']==2]
    if not peaks:
        out['reasons'].append('no_contiguous_peak_support'); return out
    height,frames = max(peaks,key=lambda p:p[0])
    out.update(proxy_peak_body=height,supporting_frames=frames,extension_frames=[s['frame'] for s in measured],
               contact_anchor_frames=[anchor['frame'],landing],measurements=measured)
    if not 0 <= height <= 1.6 or any(abs(height-b)<.04 for b in [.35,.75]):
        out['reasons'].append('implausible_or_boundary_proxy'); return out
    out.update(supported=True,height_proxy_candidate='low' if height<.35 else 'high' if height>.75 else 'mid')
    return out

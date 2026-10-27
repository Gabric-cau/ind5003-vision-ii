"""Direction from a sustained upright-to-lateral body transition, not box travel.

This is image-space pose evidence, not calibrated accuracy or verified identity.
Only exact observations of the selected source track are read. A long gap or a
track-id switch ends the onset-connected sequence; later lying poses cannot
independently establish a new dive.
"""
import math
from statistics import median
from src.goalkeeper_direction_evidence import early_direction_evidence


def pose_direction_evidence(record, pose):
    out = dict(policy='upright_to_lateral_pose_v1', supported=False,
               direction='unknown', reasons=[], observed_frames=[],
               identity_verified=False, coordinate_system='source_image_x_right',
               uses_box_displacement_threshold=False,
               min_keypoint_confidence=.50, min_sustained_samples=3,
               min_sustained_span_sec=.06, angular_rule='horizontal component exceeds vertical on both torso and shoulder-to-ankle axes')
    def reject(reason):
        out['reasons'].append(reason)
        return out
    if not pose:
        return reject('missing_cached_pose')
    # Reuse track qualification and the conservative onset-connected prefix.
    motion = early_direction_evidence(record)
    frames = motion.get('observed_frames', [])
    if not frames:
        return reject('no_qualified_onset_connected_track')
    selected = record['selected_candidate']
    ids = set(selected.get('source_track_ids', []))
    source_by_frame = selected.get('source_track_id_by_frame')
    if source_by_frame is not None:
        if not isinstance(source_by_frame, dict) or any(
                isinstance(tid, bool) or not isinstance(tid, int) or tid not in ids
                for tid in source_by_frame.values()):
            return reject('invalid_selected_source_frame_binding')
    fps = float(record['fps']); kick = int(record['kick_frame'])
    if pose.get('fps') is not None and not math.isclose(float(pose['fps']),fps,rel_tol=1e-6):
        return reject('pose_fps_mismatch')
    if pose.get('clip_id') and record.get('clip_id') and pose['clip_id']!=record['clip_id']:
        return reject('pose_clip_mismatch')
    by_frame = {}
    for track in pose.get('tracks', []):
        if track.get('track_id') not in ids:
            continue
        for o in track.get('observations', []):
            f = o['frame']
            # The stitched trajectory selects one source observation per frame.
            # Other detector views of that person are not extra selected people.
            if source_by_frame is not None and source_by_frame.get(str(f)) != track['track_id']:
                continue
            if kick-round(.5*fps) <= f <= frames[-1]:
                by_frame.setdefault(f, []).append((track['track_id'], o))
    def axes(o):
        try:
            kp = o['keypoints']; conf = o['keypoint_confidences']
            js = (5, 6, 11, 12, 15, 16)
            if any(not math.isfinite(float(conf[j])) or conf[j] < .50 for j in js):
                return None
            if any(not math.isfinite(float(v)) for j in js for v in kp[j][:2]):
                return None
            means = [tuple((kp[a][k]+kp[b][k])/2 for k in (0, 1))
                     for a, b in ((5,6), (11,12), (15,16))]
            shoulder, hip, ankle = means
            torso = tuple(a-b for a,b in zip(shoulder,hip))
            body = tuple(a-b for a,b in zip(shoulder,ankle))
            if math.hypot(*torso)<4 or math.hypot(*body)<8:
                return None
            return torso, body
        except (KeyError, IndexError, TypeError, ValueError):
            return None
    if not by_frame.get(frames[0]):
        return reject('missing_onset_pose')
    if len(by_frame[frames[0]]) != 1:
        return reject('overlapping_selected_source_tracks')
    first_id = by_frame[frames[0]][0][0]
    pre = []
    for f, entries in by_frame.items():
        if f >= kick or len(entries)!=1 or entries[0][0]!=first_id:
            continue
        axis = axes(entries[0][1])
        if axis:
            t,b = axis
            pre.append((f, t, b))
    pre.sort()
    pre = pre[-5:]
    if len(pre)<3 or kick-pre[-1][0]>round(.10*fps):
        return reject('missing_connected_upright_reference')
    if any(b[0]-a[0]>round(.20*fps) for a,b in zip(pre,pre[1:])):
        return reject('disconnected_upright_reference')
    upright = [abs(t[0])<abs(t[1]) and abs(b[0])<abs(b[1]) and t[1]<0 and b[1]<0 for _,t,b in pre]
    if sum(upright)<3:
        return reject('pre_kick_body_not_upright')
    out.update(source_track_id=first_id, pre_reference_frames=[r[0] for r in pre],
               pre_torso_horizontal_fraction=median(abs(t[0])/math.hypot(*t) for _,t,_ in pre))
    sequence=[]; stop=None; last=None
    for f in frames:
        entries=by_frame.get(f, [])
        if len(entries)>1:
            return reject('overlapping_selected_source_tracks')
        if not entries:
            if last is not None and f-last>round(.20*fps):
                stop='pose_gap';break
            continue
        tid,o=entries[0]
        if tid!=first_id:
            stop='source_track_switch';break
        if last is not None and f-last>round(.20*fps):
            stop='pose_gap';break
        axis=axes(o)
        if not axis:
            sequence.append((f,0,None));continue
        last=f;t,b=axis
        lateral=abs(t[0])>abs(t[1]) and abs(b[0])>abs(b[1]) and t[0]*b[0]>0
        sign=(1 if t[0]>0 else -1) if lateral else 0
        evidence=dict(frame=f,torso_vector=list(t),shoulder_to_ankle_vector=list(b),
                      direction='image_right' if sign>0 else 'image_left' if sign<0 else 'not_lateral')
        sequence.append((f,sign,evidence))
    out.update(observed_frames=[f for f,_,_ in sequence], prefix_stop_reason=stop or motion.get('prefix_stop_reason'),
               pose_samples=[e for _,_,e in sequence if e], kick_anchor_status=record.get('kick_frame_confidence','unknown'))
    runs=[];current=[]
    for item in sequence:
        f,sign,_=item
        if not sign or current and (sign!=current[-1][1] or f-current[-1][0]>round(.10*fps)):
            if current:runs.append(current)
            current=[]
        if sign:current.append(item)
    if current:runs.append(current)
    supported=[r for r in runs if len(r)>=3 and (r[-1][0]-r[0][0])/fps>=.06-1e-9]
    if not supported:return reject('no_sustained_lateral_body_transition')
    if len({r[0][1] for r in supported})>1:
        return reject('conflicting_lateral_body_directions')
    run=supported[0]
    out.update(supported=True,direction='image_right' if run[0][1]>0 else 'image_left',
               supporting_frames=[r[0] for r in run], transition_frame=run[0][0])
    return out

"""Extract complete camera intervals and view/speed evidence from fresh clips."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from scripts.camera_timing_context import near_net
from scripts.camera_timing_features import get_frame, infer
from src.camera_timing_v2 import (assign_kit_roles, combine_task_class,
    goal_camera_hypotheses, jersey_hist, scan_video)
from src.camera_partial_alignment import align, partial_angles, usable
from src.camera_view_guard import classify_guarded, closeup_context
from src.final_results import sha256
from src.fresh_cache_provenance import validate_cached_neural_producer
from src.fresh_event_analysis import contact_pose
from src.goal_geometry import detect_goal_frame, select_goal_occupant
from src.goalkeeper_role_association import _same_person
from src.runtime import inference_provenance, resolve_device


REPLAY_VIEWS = {"kicker_closeup", "back_of_goalkeeper", "back_of_kicker", "angle_45_right",
                "other_oblique_left", "oblique_side_unresolved"}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def qualified_scene(pose):
    selection = pose.get('shot_selection', {})
    selected = next((row for row in selection.get('attempts', [])
                     if row.get('shot_index') == selection.get('selected_shot_index')), {})
    actor = next((row for row in pose.get('track_candidates', [])
                  if row.get('track_id') == pose.get('selected_track_id')), {})
    return bool(pose.get('selected_track_id') is not None and pose.get('selected_track_qualified') is True
                and selection.get('found_qualified_shot') is True and selected.get('qualifies') is True
                and selected.get('scene_rejections') == []
                and (selected.get('goal_context') or {}).get('supported') is True
                and actor.get('field_support') is not None and actor['field_support'] >= .20)


def validate_pose_model(model):
    configuration = getattr(getattr(model, 'model', None), 'yaml', {})
    shape = configuration.get('kpt_shape', [])
    if model.task != 'pose' or list(shape) != [17, 3]:
        raise ValueError('Fresh camera inference requires a COCO pose model with kpt_shape=[17, 3]')
    return {'task': 'pose', 'kpt_shape': [17, 3], 'validated': True}


def contact_anchor(pose, contact_path=None):
    result = {'frame': None, 'source': 'scene_or_contact_unknown', 'human_verified': False,
              'pose_proposal_for_comparison_only': pose.get('kick_frame')}
    if not qualified_scene(pose):
        return result
    if contact_path is None:
        frame = pose.get('kick_frame')
        if frame is not None and pose['shot_start_frame'] <= frame < pose['shot_end_frame']:
            result.update(frame=int(frame), source='automatic_pose_proposal_unverified')
        return result
    if not contact_path.is_file():
        return {**result, 'source': 'fresh_ball_contact_file_missing'}
    contact = read(contact_path)
    identity = contact.get('inference_identity', {})
    if contact.get('clip_id') != pose['clip_id'] or identity.get('source_sha256') != pose['inference_identity']['source_sha256']:
        raise ValueError('Fresh ball contact source/clip mismatch')
    if contact.get('reference_labels_used_for_prediction') is not False:
        raise ValueError('Fresh contact is not source-only evidence')
    if not validate_cached_neural_producer(contact, contact_path.parent.parent):
        return {**result, 'source': 'fresh_ball_contact_abstained', 'contact_file_sha256': sha256(contact_path)}
    if identity.get('pose_sha256') != pose.get('_pose_file_sha256'):
        raise ValueError('Fresh ball contact was generated from a different pose result')
    selected = (contact.get('independent_contact_search') or {}).get('selected_candidate')
    if not selected or selected.get('contact_associated_track_id') is None:
        return {**result, 'source': 'fresh_ball_contact_abstained', 'contact_file_sha256': sha256(contact_path)}
    actor_pose = contact_pose(pose, selected)
    bracket = selected.get('contact_bracket_frames')
    frame = selected.get('contact_frame')
    if (selected.get('contact_timing_status') != 'bounded_stationary_departure_candidate' or
            not isinstance(bracket, list) or len(bracket) != 2 or
            not all(isinstance(value, int) for value in bracket) or
            frame is None or not bracket[0] <= frame <= bracket[1]):
        return {**result, 'source': 'fresh_ball_contact_timing_unknown', 'contact_file_sha256': sha256(contact_path)}
    if frame is not None and pose['shot_start_frame'] <= frame < pose['shot_end_frame']:
        result.update(frame=int(round(frame)), source='fresh_ball_stationary_seed_and_departure_candidate',
                      contact_file_sha256=sha256(contact_path), contact_bracket_frames=bracket,
                      contact_frame_is_estimated=True, exact_contact_frame_established=False,
                      contact_associated_track_id=actor_pose['selected_track_id'],
                      original_pose_selected_track_id=pose['selected_track_id'],
                      actor_reselected_by_ball_evidence=actor_pose['actor_reselected_by_ball_evidence'])
    else:
        result.update(source='fresh_ball_contact_abstained', contact_file_sha256=sha256(contact_path))
    return result


def appearance_references(cap, pose, contact=None):
    references = {"kicker_histograms": [], "keeper_histograms": [], "identity_verified": False,
                  "samples": [], "warning": "Same-kit people are not independently identified"}
    if not qualified_scene(pose):
        references['rejection'] = 'selected_scene_or_actor_not_qualified'
        return references
    kick = contact if contact is not None else pose.get("kick_frame")
    references['contact_frame_candidate'] = kick
    track = next((track for track in pose.get("tracks", [])
                  if track["track_id"] == pose.get("selected_track_id")), {})
    fps = float(pose['fps'])
    observations = sorted([row for row in track.get("observations", [])
                    if kick is not None and max(pose['shot_start_frame'], kick-round(1.6*fps)) <= row["frame"] <= kick-max(1, round(.15*fps))],
                    key=lambda row: row['frame'])
    if not observations:
        return references
    selected = [observations[index] for index in sorted({0, len(observations) // 2, len(observations) - 1})]
    excluded_ids = {pose['selected_track_id']}
    for other in pose.get('tracks', []):
        if other['track_id'] != track.get('track_id') and _same_person(track, other):
            excluded_ids.add(other['track_id'])
    references['excluded_same_actor_ids'] = sorted(excluded_ids, key=str)
    keeper_tracks = [other for other in pose.get('tracks', []) if other['track_id'] not in excluded_ids]
    for observation in selected:
        frame = get_frame(cap, observation["frame"])
        if frame is None or not frame.size:
            references['samples'].append({'frame': observation['frame'], 'rejection': 'empty_source_frame'})
            continue
        hist = jersey_hist(frame, observation["bbox"], observation["keypoints"],
                           observation["keypoint_confidences"])
        if hist is not None:
            references["kicker_histograms"].append(hist)
        occupant = select_goal_occupant(frame, keeper_tracks, observation["frame"])
        keeper = next((other for other in keeper_tracks
                       if other['track_id'] == occupant.get('seed_track_id')), {})
        rows = keeper.get("observations", [])
        if rows:
            row = min(rows, key=lambda row: abs(row["frame"] - observation["frame"]))
            if abs(row['frame']-observation['frame']) <= 2:
                keeper_frame = get_frame(cap, row["frame"])
                if keeper_frame is not None and keeper_frame.size:
                    hist = jersey_hist(keeper_frame, row["bbox"], row["keypoints"], row["keypoint_confidences"])
                    if hist is not None:
                        references["keeper_histograms"].append(hist)
        references["samples"].append({"frame": observation["frame"],
            "kicker_track_id": pose["selected_track_id"], "keeper_association": occupant})
    return references


def action_sequence(cap, model, references, start, end, fps, device):
    sequence = {"kicker": [], "keeper": []}
    raw = []
    if start >= end or not any(references.get(role+'_histograms') for role in sequence):
        return {'roles': sequence, 'raw_pose_observations': [], 'sampling_fps_target': 8,
                'status': 'unknown', 'reason': 'empty_or_unidentified_action_window'}
    for frame_id in range(start, end, max(1, round(fps / 8))):
        frame = get_frame(cap, frame_id)
        if frame is None or not frame.size:
            raw.append({'frame': frame_id, 'people': [], 'reason': 'empty_source_frame'})
            continue
        people = assign_kit_roles(infer(model, frame, device), references)
        raw.append({"frame": frame_id, "people": people})
        for role in sequence:
            candidates = [person for person in people if person["role"] == role + "_candidate"]
            if len(candidates) != 1:
                continue
            feature = partial_angles(candidates[0])
            if usable({'feature': feature}):
                sequence[role].append({"frame": frame_id, "time_sec": frame_id / fps, "feature": feature})
    return {"roles": sequence, "raw_pose_observations": raw, "sampling_fps_target": 8}


def speed_evidence(reference, query, view):
    result = {'status': 'unknown', 'reason': 'insufficient_role_specific_action_alignment',
              'identity_verified': False, 'reference_speed_verified': False,
              'absolute_speed_verified': False, 'absolute_speed_factor': None}
    if not reference or not query:
        return result
    roles = {role: align(reference['roles'][role], query['roles'][role]) for role in ('kicker', 'keeper')}
    expected_role = {'kicker_closeup': 'kicker', 'back_of_goalkeeper': 'keeper'}.get(view)
    result.update(role_results=roles, target_role=expected_role,
                  method='partial_lower_body_ordered_affine_time_alignment_with_internal_nulls')
    if expected_role and roles[expected_role]['status'] == 'relative_slowdown_candidate':
        result.update(status='slow_motion_candidate', reason='target_role_relative_slowdown_candidate_only')
    return result


def validate_timeline(scan, record):
    if scan['frame_count'] != record['frame_count'] or scan['decoded_frames'] != record['frame_count']:
        raise ValueError('Camera source did not fully decode the manifest timeline')
    if not scan['segments'] or scan['segments'][0]['start_frame'] != 0 or scan['segments'][-1]['end_frame_exclusive'] != scan['frame_count']:
        raise ValueError('Camera timeline does not span the complete source clip')
    for previous, current in zip(scan['segments'], scan['segments'][1:]):
        if previous['end_frame_exclusive'] != current['start_frame']:
            raise ValueError('Camera intervals contain a gap or overlap')
    for shot in scan['segments']:
        if not shot['start_frame'] < shot['end_frame_exclusive']:
            raise ValueError('Invalid camera interval')
        if any(not shot['start_frame'] <= fid < shot['end_frame_exclusive'] for fid in shot['sample_frame_ids']):
            raise ValueError('Camera sample lies outside its source interval')


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument('--contact-dir', type=Path, help='Fresh independent ball-contact results; never reference labels')
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clips", nargs="+")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError("Use a new experiment directory")
    from ultralytics import YOLO
    device = resolve_device(args.device)
    model = YOLO(str(args.model))
    model_check = validate_pose_model(model)
    model._visionii_device = device
    source = read(args.clips_dir / "source_manifest.json")
    records = {row["clip_id"]: row for row in source["clips"]}
    if len(records) != len(source['clips']):
        raise ValueError('Duplicate source clip IDs')
    ids = args.clips or list(records)
    if len(ids) != len(set(ids)) or any(cid not in records for cid in ids):
        raise ValueError("Clip IDs must be distinct source-manifest members")
    if not args.clips and len(ids) != 39:
        raise ValueError('The complete camera cohort must contain exactly 39 clips')
    protocol = {"runtime": inference_provenance(args.device, device, [args.model]),
        'model_task_validation': model_check,
        "cohort_clip_ids": ids, "source_manifest_sha256": sha256(args.clips_dir / "source_manifest.json"),
        "runner_sha256": sha256(Path(__file__)), "reference_labels_used_for_prediction": False,
        "old_caches_used": False, "human_verified": False, "held_out_test": False,
        'contact_dir': str(args.contact_dir) if args.contact_dir else None,
        'settings': {'imgsz': 960, 'confidence': .25, 'action_sampling_fps_target': 8},
        "module_sha256": {name: sha256(ROOT / name) for name in (
            "src/camera_timing_v2.py", "src/camera_view_guard.py", "src/goal_geometry.py",
            'src/camera_partial_alignment.py', 'src/goalkeeper_role_association.py',
            'src/fresh_event_analysis.py', 'src/fresh_cache_provenance.py',
            "scripts/camera_timing_context.py", "scripts/camera_timing_features.py")}}
    save(args.output_dir / "protocol.json", protocol)
    clips = []
    cv2.setNumThreads(2)
    for index, cid in enumerate(ids):
        started = time.monotonic()
        record = records[cid]
        if Path(record['clip_filename']).name != record['clip_filename']:
            raise ValueError('Source manifest requires local clip basenames')
        video = args.clips_dir / record["clip_filename"]
        pose_path = args.pose_dir / f"{cid}.json"
        pose = read(pose_path)
        if sha256(video) != record["clip_sha256"] or pose["source_mapping"]["clip_sha256"] != record["clip_sha256"]:
            raise ValueError("Fresh camera video/pose hash mismatch")
        if pose.get('clip_id') != cid or pose.get('new_neural_inference') is not True or pose.get('reference_labels_used_for_prediction') is not False:
            raise ValueError('Camera requires fresh source-only pose results')
        if pose.get('inference_identity', {}).get('source_sha256') != record['clip_sha256']:
            raise ValueError('Camera pose identity does not match source video')
        pose['_pose_file_sha256'] = sha256(pose_path)
        scan = scan_video(video)
        validate_timeline(scan, record)
        cap = cv2.VideoCapture(str(video))
        anchor = contact_anchor(pose, args.contact_dir/f'{cid}.json' if args.contact_dir else None)
        contact = anchor['frame']
        reference_pose = {**pose, 'selected_track_id': anchor.get('contact_associated_track_id', pose['selected_track_id'])}
        references = appearance_references(cap, reference_pose, contact) if contact is not None else {
            'kicker_histograms': [], 'keeper_histograms': [], 'identity_verified': False,
            'samples': [], 'rejection': 'no_supported_contact_reference_window'}
        reference = next((shot for shot in scan["segments"] if contact is not None
                          and shot["start_frame"] <= contact < shot["end_frame_exclusive"]), None)
        reference_action = action_sequence(cap, model, references,
            max(reference["start_frame"], contact - round(1.4 * scan["fps"])),
            min(reference["end_frame_exclusive"], contact + round(scan["fps"])),
            scan["fps"], device) if reference else None
        segments = []
        for shot in scan["segments"]:
            samples = []
            for frame_id in shot["sample_frame_ids"]:
                frame = get_frame(cap, frame_id)
                if frame is None or not frame.size:
                    samples.append({'frame': frame_id, 'width': scan['width'], 'height': scan['height'],
                                    'people': [], 'goal': None, 'goal_camera': {},
                                    'reason': 'empty_source_frame'})
                    continue
                people = assign_kit_roles(infer(model, frame, device), references)
                goal = detect_goal_frame(frame, max_goal_width_fraction=.98, max_post_height_fraction=.85)
                selected = goal.get("selected")
                samples.append({"frame": frame_id, "width": scan["width"], "height": scan["height"],
                    "people": people, "goal": selected, "goal_camera": goal_camera_hypotheses(
                        selected["corners"], scan["width"], scan["height"]) if selected else {},
                    "foreground_net_cue": near_net(frame), "closeup_context": closeup_context(frame, people)})
            view = classify_guarded(samples)
            speed = {"status": "unknown", "reason": "same_action_alignment_not_attempted",
                     "identity_verified": False, "reference_speed_verified": False}
            query = None
            if reference and shot["start_frame"] > contact and view["view_candidate"] in REPLAY_VIEWS:
                query = action_sequence(cap, model, references, shot["start_frame"] + 2,
                    min(shot["end_frame_exclusive"] - 2, shot["start_frame"] + round(8 * scan["fps"])),
                    scan["fps"], device)
                speed = speed_evidence(reference_action, query, view['view_candidate'])
            source_start = record['source_start_frame'] + shot['start_frame']
            source_end = record['source_start_frame'] + shot['end_frame_exclusive']
            segments.append({**shot, 'course_source_start_frame': source_start,
                'course_source_end_frame_exclusive': source_end,
                'course_source_start_sec': source_start/record['fps'],
                'course_source_end_sec_exclusive': source_end/record['fps'],
                "samples": samples, "view_evidence": view,
                "view_candidate": view["view_candidate"], "playback_evidence": speed,
                "task_class_candidate": combine_task_class(view["view_candidate"], speed),
                "query_action": query, "human_verified": False, "accuracy_validated": False})
        cap.release()
        if sha256(video) != record['clip_sha256'] or sha256(pose_path) != pose['_pose_file_sha256']:
            raise AssertionError('Camera source video or pose changed during inference')
        result = {"clip_id": cid, "source_mapping": record, "pose_sha256": sha256(pose_path),
            "fps": scan["fps"], "source_frames": scan["frame_count"], "timeline_coverage": scan["timeline_coverage"],
            "camera_scan": scan, "references": references, "reference_action": reference_action, 'contact_anchor': anchor,
            "segments": segments, "new_neural_inference": True, "human_verified": False,
            "accuracy_validated": False, "elapsed_sec": round(time.monotonic() - started, 3)}
        save(args.output_dir / "clips" / f"{cid}.json", result)
        clips.append(result)
        print(json.dumps({"completed": index + 1, "total": len(ids), "clip_id": cid,
                          "segments": len(segments), "elapsed_sec": result["elapsed_sec"]}), flush=True)
    segments = [segment for clip in clips for segment in clip["segments"]]
    save(args.output_dir / "summary.json", {"events": len(clips), "segments": len(segments),
        "view_candidates": dict(Counter(row["view_candidate"] for row in segments)),
        "task_class_candidates": dict(Counter(row["task_class_candidate"] for row in segments)),
        "playback_candidates": dict(Counter(row["playback_evidence"]["status"] for row in segments)),
        "timeline_coverage_all": all(clip["timeline_coverage"] == 1 for clip in clips),
        "human_verified": False, "accuracy_validated": False, "held_out_test": False})


if __name__ == "__main__":
    main()

"""Prepare event-level ball classification candidates without changing source labels."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/external_ball_training_2026-10-25/manifest.json"
SPLITS = {
    "german_cup_2022": "train",
    "coppa_italia_2025": "validation",
    "world_cup_2022": "heldout",
}
DIRECTION_LABELS = {"\u5de6": "left", "\u4e2d": "center", "\u53f3": "right"}
HEIGHT_LABELS = {"\u4f4e": "low", "\u4e2d": "middle", "\u9ad8": "high"}
BEHIND_KICKER_CAMERAS = {
    "\u70b9\u7403\u8005\u540e\u4fa7\u89c6\u89d2",
    "\u70b9\u7403\u8005\u80cc\u540e\u8fdc\u666f",
}
PRIMARY_ACTION_CAMERAS = BEHIND_KICKER_CAMERAS | {
    "\u5de6\u4fa7\u9ad8\u673a\u4f4d\u8fdc\u666f",
    "\u5e38\u89c4\u89c6\u89d2\uff08\u5de6\u4fa7\u9ad8\u673a\u4f4d\u8fdc\u666f\uff09",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_media(root, relative):
    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Media path escapes dataset root: {relative}")
    return candidate


def candidate_segments(clip, timeline, policy):
    if policy not in {"behind_kicker", "primary_action"}:
        raise ValueError(f"Unsupported view policy: {policy}")
    fps = float(clip["fps"])
    duration = float(clip["duration_seconds"])
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Video timing must be positive and finite")
    allowed = BEHIND_KICKER_CAMERAS if policy == "behind_kicker" else PRIMARY_ACTION_CAMERAS
    selected = []
    for shot in timeline:
        if shot.get("video_file") != clip["file"]:
            continue
        if not shot.get("included_in_confirmed_event_range"):
            continue
        if shot.get("camera_annotation_source") != "human_user_observation_in_chat":
            continue
        if shot.get("camera_label") not in allowed or shot.get("occlusion_label") is True:
            continue
        start = shot.get("camera_label_start_seconds")
        end = shot.get("camera_label_end_seconds_exclusive")
        if start is None or end is None:
            continue
        start, end = float(start), float(end)
        shot_start = float(shot["segment_start_seconds"])
        shot_end = float(shot["segment_end_seconds_exclusive"])
        if not all(math.isfinite(x) for x in (start, end, shot_start, shot_end)):
            raise ValueError("Non-finite segment boundary")
        if not (0 <= shot_start <= start < end <= shot_end <= duration + 1e-6):
            raise ValueError(f"Invalid annotated interval: {shot['original_shot_id']}")
        # Primary-action candidates are confined to the manually named opening shot.
        if policy == "primary_action" and (shot_start != 0 or shot.get("replay_label") is True):
            continue
        first_frame = math.ceil(start * fps - 1e-6)
        end_frame = math.ceil(end * fps - 1e-6)
        if end_frame > int(clip["expected_video_frames"]):
            raise ValueError("Annotated segment extends beyond video frames")
        selected.append({
            "shot_id": shot["original_shot_id"],
            "start_seconds": start,
            "end_seconds_exclusive": end,
            "frame_start_zero_based": first_frame,
            "frame_end_exclusive": end_frame,
            "source_frame_start_zero_based": int(clip["source_start_frame_zero_based"]) + first_frame,
            "source_frame_end_exclusive": int(clip["source_start_frame_zero_based"]) + end_frame,
            "source_segment_start_seconds": shot_start,
            "source_segment_end_seconds_exclusive": shot_end,
            "fps": fps,
            "camera_label": shot["camera_label"],
            "camera_label_scope": shot.get("camera_label_scope"),
            "camera_annotation_source": shot["camera_annotation_source"],
            "replay_label": shot.get("replay_label"),
            "slow_motion_label": shot.get("slow_motion_label"),
            "occlusion_label": shot.get("occlusion_label"),
            "human_note": shot.get("human_note"),
            "selection_status": "candidate_from_human_camera_annotation",
            "ball_visibility_verified": False,
            "complete_action_verified": False,
            "event_label_is_not_frame_label": True,
        })
    return sorted(selected, key=lambda item: (item["start_seconds"], item["shot_id"]))


def build_manifest(document, annotation_path, *, view_policy="primary_action", verify_media=False,
                   augment_training_behind_kicker=False):
    annotation_path = Path(annotation_path).resolve()
    dataset_root = annotation_path.parent.parent
    timeline = document.get("shot_timeline", [])
    sources = {source["dataset_id"]: source for source in document.get("sources", [])}
    samples, excluded, inventory = [], [], []
    seen = set()
    for clip in document["clips"]:
        dataset_id = clip["dataset_id"]
        event_id = f"{dataset_id}:{clip['original_event_id']}"
        if event_id in seen:
            raise ValueError(f"Duplicate event group: {event_id}")
        seen.add(event_id)
        split = SPLITS.get(dataset_id)
        roster = {"event_id": event_id, "dataset_id": dataset_id,
                  "split": split, "filename": clip["filename"]}
        inventory.append(roster)
        if split == "heldout":
            excluded.append({**roster, "reason": "heldout_locked_no_training_validation_or_selection"})
            continue
        if split is None:
            excluded.append({**roster, "reason": "dataset_not_in_frozen_split_policy"})
            continue
        if clip.get("outcome_conflict") or clip.get("training_goal_label") is None:
            excluded.append({**roster, "reason": "entire_event_quarantined_pending_result_and_identity_review"})
            continue
        features = clip.get("human_features", {})
        if features.get("provenance") != "human_user_observation_in_chat":
            excluded.append({**roster, "reason": "human_event_label_provenance_missing"})
            continue
        if features.get("direction_reference_frame") != "behind_kicker_view":
            excluded.append({**roster, "reason": "direction_reference_unconfirmed"})
            continue
        direction = DIRECTION_LABELS.get(features.get("ball_direction_label"))
        height = HEIGHT_LABELS.get(features.get("ball_height_label"))
        if direction is None or height is None:
            excluded.append({**roster, "reason": "unknown_or_unsupported_target_label"})
            continue
        candidates = {policy: candidate_segments(clip, timeline, policy)
                      for policy in ("behind_kicker", "primary_action")}
        selected = list(candidates[view_policy])
        if not selected:
            excluded.append({**roster, "reason": f"no_eligible_{view_policy}_camera_candidate"})
            continue
        if augment_training_behind_kicker and split == "train":
            seen_shots = {segment["shot_id"] for segment in selected}
            selected.extend(segment for segment in candidates["behind_kicker"]
                            if segment["shot_id"] not in seen_shots)
        selected = [{**segment, "event_segment_weight": 1 / len(selected)} for segment in selected]
        path = resolve_media(dataset_root, clip["file"])
        declared_sha = clip.get("media_check", {}).get("sha256")
        actual_sha = sha256(path) if verify_media else None
        if verify_media and (not declared_sha or actual_sha != declared_sha):
            raise ValueError(f"Video SHA-256 mismatch: {clip['file']}")
        flags = ["event_level_qualitative_single_observer_labels",
                 "ball_visibility_and_terminal_frame_not_manually_verified",
                 "no_ball_boxes_trajectory_or_metric_height_ground_truth",
                 "no_horizontal_flip_augmentation_without_target_transform"]
        if any(segment["replay_label"] is True for segment in selected):
            flags.append("selected_view_contains_declared_replay_of_same_event")
        if any(segment["slow_motion_label"] is None for segment in selected):
            flags.append("selected_view_playback_speed_unconfirmed")
        if any(segment["occlusion_label"] is None for segment in selected):
            flags.append("selected_view_occlusion_unconfirmed")
        if clip["filename"] == "match_08.mp4":
            flags.append("short_3_32_second_clip_complete_action_unconfirmed")
        samples.append({
            **roster,
            "event_group": event_id,
            "player_name": clip.get("player_name"),
            "video_path": str(path),
            "video_relative_path": clip["file"],
            "video_sha256": actual_sha if verify_media else declared_sha,
            "video_sha256_declared": declared_sha,
            "video_sha256_verified": actual_sha,
            "video_present": path.is_file(),
            "source_video_sha256_declared": sources.get(dataset_id, {}).get("sha256"),
            "source_url": clip.get("source_url"),
            "fps": float(clip["fps"]),
            "video_frame_count": int(clip["expected_video_frames"]),
            "duration_seconds": float(clip["duration_seconds"]),
            "targets": {"ball_direction": direction, "ball_height_category": height},
            "direction_reference_frame": "behind_kicker_view",
            "human_already_normalized_direction": True,
            "mirror_again": False,
            "raw_human_observations": features.get("raw_user_observations", []),
            "target_provenance": features["provenance"],
            "candidates": candidates,
            "selected_segments": selected,
            "audit_flags": flags,
        })
    counts = Counter(sample["split"] for sample in samples)
    return {
        "schema_version": "external-ball-event-training-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "event_level_ball_direction_and_height_classification_prototype",
        "annotation_source": {"path": str(annotation_path), "sha256": sha256(annotation_path)},
        "dataset_root": str(dataset_root),
        "view_policy": view_policy,
        "augment_training_behind_kicker": augment_training_behind_kicker,
        "split_policy": SPLITS,
        "heldout_locked": True,
        "heldout_labels_exported": False,
        "official_results_are_not_ball_labels": True,
        "original_metadata_modified": False,
        "class_order": {"ball_direction": ["left", "center", "right"],
                        "ball_height_category": ["low", "middle", "high"]},
        "grouping_policy": "All clips, views, replays and sampled frames from an event stay in its match split.",
        "unknown_policy": "Missing direction references and unknown target classes are excluded, never negative examples.",
        "out_of_goal_policy": "Preserve this release's qualitative placement labels; do not mix with the old 39-event outside-goal unknown policy.",
        "selection_note": "Camera annotations select candidate intervals, not clean ball evidence or frame-level labels.",
        "samples": samples,
        "excluded_events": excluded,
        "event_inventory": inventory,
        "summary": {
            "source_events": len(inventory),
            "selected_events": len(samples),
            "train_events": counts["train"],
            "validation_events": counts["validation"],
            "heldout_events_locked": sum(row["split"] == "heldout" for row in inventory),
            "selected_segments": sum(len(sample["selected_segments"]) for sample in samples),
            "exclusions": dict(Counter(row["reason"] for row in excluded)),
            "target_counts": {split: {target: dict(Counter(sample["targets"][target] for sample in samples
                                                                 if sample["split"] == split))
                                      for target in ("ball_direction", "ball_height_category")}
                              for split in ("train", "validation")},
        },
        "limitations": [
            "One tiny training match and one validation match cannot establish generalization.",
            "Qualitative event labels do not supervise ball detection, physical height or full trajectories.",
            "Camera, scoreboards, player identity and scene appearance can provide shortcut signals.",
            "Validation metrics are development results; World Cup images, targets and segments are excluded from samples.",
            "No proposed AI visual correction is promoted to a human label; kicking-foot labels are not targets.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--view-policy", choices=("behind_kicker", "primary_action"), default="primary_action")
    parser.add_argument("--verify-media-sha256", action="store_true")
    parser.add_argument("--augment-training-behind-kicker", action="store_true")
    args = parser.parse_args()
    document = json.loads(args.annotations.read_text(encoding="utf-8-sig"))
    manifest = build_manifest(document, args.annotations, view_policy=args.view_policy,
                              verify_media=args.verify_media_sha256,
                              augment_training_behind_kicker=args.augment_training_behind_kicker)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **manifest["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

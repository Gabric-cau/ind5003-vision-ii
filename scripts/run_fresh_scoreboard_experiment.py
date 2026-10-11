"""Extract source-only penalty-icon outcome witnesses from fresh course clips."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.broadcast_scoreboard import (extract_penalty_scoreboard, infer_first_shooting_side,
                                      outcome_from_ordinal, render_scoreboard_evidence,
                                      stable_scoreboard_runs)


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _decode_sampled(video, frame_ids, fps):
    """Decode once sequentially; random seeking made the feasibility scan needlessly slow."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened() or not np.isclose(capture.get(cv2.CAP_PROP_FPS), fps):
        capture.release()
        raise ValueError("Source decode/fps mismatch")
    wanted = iter(frame_ids)
    target = next(wanted, None)
    frame_id = target if target is not None else 0
    if target is not None and target:
        capture.set(cv2.CAP_PROP_POS_FRAMES, target)
    try:
        while target is not None:
            ok, image = capture.read()
            if not ok:
                raise ValueError("Decoded source frame ended before requested sample")
            if int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1 != frame_id:
                raise ValueError("Decoded source frame identity mismatch")
            if frame_id == target:
                yield target, image
                target = next(wanted, None)
            frame_id += 1
    finally:
        capture.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clips", nargs="+")
    parser.add_argument("--sample-stride-seconds", type=float, default=.2)
    parser.add_argument("--original-videos-dir", type=Path,
                        help="Search slice_end to next slice_start in SHA-verified full source videos")
    args = parser.parse_args(argv)
    if not np.isfinite(args.sample_stride_seconds) or not 0 < args.sample_stride_seconds <= 1.:
        parser.error("Sampling stride must be positive and at most one second")
    if args.output_dir.resolve() == args.clips_dir.resolve():
        parser.error("Output must not overwrite source clips")
    manifest_path = args.clips_dir / "source_manifest.json"
    manifest_hash = _hash(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    sources = {row["clip_id"]: row for row in manifest["clips"]}
    ids = args.clips or list(sources)
    if len(sources) != len(manifest["clips"]) or len(set(ids)) != len(ids) or any(cid not in sources for cid in ids):
        raise ValueError("Clip IDs must be distinct source manifest members")
    code_hashes = {name: _hash(ROOT / name) for name in ("src/broadcast_scoreboard.py", "scripts/run_fresh_scoreboard_experiment.py")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {"schema_version": "source-scoreboard-protocol-v1", "frozen_before_extraction": True,
                "source_scope": "original_full_video" if args.original_videos_dir else "fresh_context_clip",
                "window": "canonical slice_end inclusive to next canonical slice_start exclusive; final attempt to source end"
                    if args.original_videos_dir else "entire context clip, outcome witnesses after canonical slice_end",
                "sampling_stride_seconds": args.sample_stride_seconds,
                "geometry": "long lower red bar plus two complete regularly spaced five-circle groups",
                "stability": "at least three consecutive equal complete panels spanning at least 0.4s; absent samples break runs",
                "association": "both icon rows have resolved contiguous prefixes; exact resolved total equals current attempt ordinal",
                "shooting_order": "infer only from unique same-source-video odd-total prefix majority; even ordinal otherwise unknown",
                "outcome": "current ordinal circle green/red only; score digits and human/external outcomes are not inputs",
                "next_attempt": "search ends before next canonical slice start; any panel total differing from current ordinal is unusable",
                "bounds_are_true_contact_times": False, "code_sha256": code_hashes,
                "human_verified": False, "accuracy_validated": False}
    protocol_path = args.output_dir / "scoreboard_protocol.json"
    _write(protocol_path, protocol)
    original_metadata = {}
    if args.original_videos_dir is not None:
        for source in sources.values():
            name = source["source_filename"]
            if name in original_metadata:
                if original_metadata[name]["sha256"] != source["source_video_sha256"]:
                    raise ValueError("Conflicting full source SHA in manifest")
                continue
            video = args.original_videos_dir / name
            video_hash = _hash(video)
            if video_hash != source["source_video_sha256"]:
                raise ValueError("Original full source hash mismatch")
            cap = cv2.VideoCapture(str(video))
            try:
                if not cap.isOpened() or not np.isclose(cap.get(cv2.CAP_PROP_FPS), source["fps"]):
                    raise ValueError("Original source fps/decode mismatch")
                original_metadata[name] = {"sha256": video_hash, "frame_count": int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))}
            finally:
                cap.release()
    cv2.setNumThreads(2)
    clips = {}
    match_runs = defaultdict(list)
    for cid in ids:
        source = sources[cid]
        context_video = args.clips_dir / source["clip_filename"]
        if _hash(context_video) != source["clip_sha256"]:
            raise ValueError("Source clip hash mismatch")
        if args.original_videos_dir is None:
            video = context_video
            expected_video_hash = source["clip_sha256"]
            frame_origin = source["source_start_frame"]
            start, end = 0, source["frame_count"]
            after = source["original_slice_end_frame_in_clip"]
        else:
            video = args.original_videos_dir / source["source_filename"]
            expected_video_hash = source["source_video_sha256"]
            frame_origin = 0
            start = int(round(source["slice_end_sec"] * source["fps"]))
            next_source = next((row for row in sources.values() if row["match_id"] == source["match_id"]
                               and row["penalty_number"] == source["penalty_number"] + 1), None)
            end = (int(round(next_source["slice_start_sec"] * source["fps"])) if next_source
                   else original_metadata[source["source_filename"]]["frame_count"])
            after = start
            if not 0 <= start < end <= original_metadata[source["source_filename"]]["frame_count"]:
                raise ValueError("Canonical post-event window outside original source")
        stride = max(1, int(round(source["fps"] * args.sample_stride_seconds)))
        sampled_frames = sorted(set(range(start, end, stride)) | {end - 1})
        observations = []
        for fid, image in _decode_sampled(video, sampled_frames, source["fps"]):
            parsed = extract_penalty_scoreboard(image, frame_id=fid)
            parsed.update(source_frame_id=frame_origin + fid,
                          decoded_pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest())
            observations.append(parsed)
        if _hash(video) != expected_video_hash:
            raise ValueError("Source video changed during extraction")
        runs = stable_scoreboard_runs(observations, fps=source["fps"])
        clips[cid] = {"source": source, "video": video, "observations": observations, "runs": runs,
                      "sampling_frame_stride": stride, "frame_origin": frame_origin, "after": after,
                      "search_interval": [start, end], "video_sha256": expected_video_hash}
        match_runs[source["match_id"]].extend(run for run in runs if run["resolved_total"] == source["penalty_number"])
        print(cid, len(observations), "sampled", sum(row["status"] == "candidate_scoreboard" for row in observations),
              "complete panels", len(runs), "stable runs", flush=True)
    first_sides = {match: infer_first_shooting_side(runs) for match, runs in match_runs.items()}
    rows = []
    for cid in ids:
        clip = clips[cid]
        source = clip["source"]
        # Course slices define conservative sampling bounds, not true event endpoints.
        after = clip["after"]
        decision = outcome_from_ordinal(clip["runs"], attempt_ordinal=source["penalty_number"], after_frame=after,
                                       first_shooting_side=first_sides[source["match_id"]])
        frame_map = {row["frame_id"]: row for row in clip["observations"]}
        witnesses = []
        witness_ids = sorted({fid for run in clip["runs"] for fid in
                              (run["frame_ids"][0], run["frame_ids"][len(run["frame_ids"]) // 2], run["frame_ids"][-1])})
        for fid, image in _decode_sampled(clip["video"], witness_ids, source["fps"]):
            if hashlib.sha256(image.tobytes()).hexdigest() != frame_map[fid]["decoded_pixels_sha256"]:
                raise ValueError("Witness decoded pixels do not match discovery frame")
            overlay_path = args.output_dir / f"{cid}_f{fid:06d}_scoreboard.jpg"
            if not cv2.imwrite(str(overlay_path), render_scoreboard_evidence(image, frame_map[fid])):
                raise OSError("Cannot write scoreboard witness overlay")
            witnesses.append({"frame_id": fid, "source_frame_id": clip["frame_origin"] + fid,
                              "path": overlay_path.name, "sha256": _hash(overlay_path),
                              "decoded_pixels_sha256": frame_map[fid]["decoded_pixels_sha256"]})
        result = {"schema_version": "fresh-scoreboard-experiment-v1", "clip_id": cid, "event_id": source["event_id"],
                  "status": decision["status"], "scoreboard_outcome": decision,
                  "observations": clip["observations"], "stable_runs": clip["runs"], "witness_images": witnesses,
                  "sampling_frame_stride": clip["sampling_frame_stride"], "source_mapping": source,
                  "frame_index_basis": "original_full_video" if args.original_videos_dir else "context_clip",
                  "source_video_path": str(clip["video"].resolve()), "source_video_sha256": clip["video_sha256"],
                  "search_interval_frames": clip["search_interval"],
                  "after_frame_sampling_bound": after,
                  "after_bound_is_semantic_event_end_truth": False,
                  "first_shooting_side": first_sides[source["match_id"]],
                  "inference_identity": {"source_sha256": source["clip_sha256"], "source_manifest_sha256": manifest_hash,
                                         "code_sha256": code_hashes, "opencv_version": cv2.__version__},
                  "scoreboard_protocol_sha256": _hash(protocol_path),
                  "numeric_scores_read": False, "outcome_labels_read": False,
                  "reference_labels_used_for_prediction": False, "human_verified": False, "accuracy_validated": False,
                  "new_neural_inference": False, "source_only_image_analysis": True,
                  "scope": "Stable ten-circle broadcast scoreboard witnesses within bounded source window; ordinal is source metadata"}
        _write(args.output_dir / (cid + ".json"), result)
        rows.append({"clip_id": cid, "event_id": source["event_id"], "status": result["status"],
                     "rule_outcome": decision["rule_outcome"], "reason": decision["reason"],
                     "complete_panel_frames": sum(row["status"] == "candidate_scoreboard" for row in clip["observations"]),
                     "stable_runs": len(clip["runs"])})
    summary = {"cohort_size": len(ids), "cohort_clip_ids": ids, "rows": rows,
               "status_counts": dict(Counter(row["status"] for row in rows)),
               "outcome_counts": dict(Counter(row["rule_outcome"] for row in rows)),
               "first_shooting_side_from_odd_cumulative_panels": first_sides,
               "source_manifest_sha256": manifest_hash, "code_sha256": code_hashes,
               "scoreboard_protocol_sha256": _hash(protocol_path),
               "human_verified": False, "accuracy_validated": False, "outcome_labels_read": False,
               "reference_labels_used_for_prediction": False, "scope": "Candidate coverage, not measured accuracy"}
    _write(args.output_dir / "scoreboard_experiment_summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("status_counts", "outcome_counts", "first_shooting_side_from_odd_cumulative_panels")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

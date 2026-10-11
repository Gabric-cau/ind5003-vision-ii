"""Replay four cached clips with motion association; preserve earlier releases."""
from pathlib import Path
import argparse
import hashlib
import html
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np

from src.ball_continuity import ContinuityConfig, motion_forecasts, search_predictions, track_ball_continuity
from src.goal_plane_rectification import project_image_point
from scripts.probe_goal_event_tracking import motion_blobs, source_time_indices

cv2.setNumThreads(2)
BASE = ROOT / "outputs/goal_event_evidence_2026-10-10"
FFMPEG = ROOT / "outputs/review_tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe"


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def save(p, value):
    Path(p).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def label(im, text, y, color=(245, 245, 245)):
    cv2.putText(im, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)


def collision_geometry(cache, earlier):
    """Project collider candidates into the same camera-stabilized coordinates.

    Existing keeper geometry is an unverified candidate, not contact truth.
    A goal-mouth polygon cannot establish net depth and cannot unlock a turn.
    """
    regions = {}
    keeper_frames = (earlier.get("keeper_kit_evidence") or {}).get("frames", {})
    for f in range(cache["start_frame"], cache["end_frame_exclusive"]):
        key = str(f)
        transform = cache["camera"]["transforms"].get(key)
        regions[key] = []
        if transform is None:
            continue
        proposal = cache["goal_proposals"].get(key, {}).get("selected")
        if proposal:
            corners = [project_image_point(transform, p) for p in proposal["corners"]]
            if all(p is not None for p in corners):
                for kind, i, j in (("left_post", 0, 3), ("right_post", 1, 2), ("crossbar", 0, 1)):
                    regions[key].append({"kind": kind, "segment": [corners[i], corners[j]],
                                         "source": "unverified_automatic_goal_lines"})
                regions[key].append({"kind": "goal_projection", "polygon": corners,
                                     "strength": "weak", "source": "goal_mouth_depth_unresolved"})
        keeper = keeper_frames.get(key)
        if keeper:
            x0, y0, x1, y1 = keeper["bbox"]
            polygon = [project_image_point(transform, p) for p in
                       ([x0, y0], [x1, y0], [x1, y1], [x0, y1])]
            if all(p is not None for p in polygon):
                regions[key].append({"kind": "keeper_candidate", "polygon": polygon,
                                     "source": "previous_unverified_keeper_geometry_only"})
    return regions


def render(cache, path, predictions, old_path, dest, regression_path=None):
    source = Path(cache["video_path"])
    video = dest / (cache["clip_id"] + ".mp4")
    command = [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
               "-pix_fmt", "bgr24", "-s", "960x620", "-r", str(cache["fps"] / 3),
               "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2", "-crf", "20",
               "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    cap = cv2.VideoCapture(str(source))
    cap.set(cv2.CAP_PROP_POS_FRAMES, cache["start_frame"])
    observed = {p["frame"]: p for p in path}
    predicted = {p["frame"]: p for p in predictions}
    previous = {p["frame"]: p for p in old_path}
    regression = {p["frame"]: p for p in (regression_path or [])}
    scale = np.array([960 / cache["width"], 540 / cache["height"]])
    samples = {cache["kick_frame"] + n for n in (5, 10, 15, 20, 30, 45)}
    tiles = []
    try:
        for f in range(cache["start_frame"], cache["end_frame_exclusive"]):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Source decode failed at {f}")
            im = np.full((620, 960, 3), 25, dtype=np.uint8)
            scene = cv2.resize(frame, (960, 540))
            old, current = previous.get(f), observed.get(f)
            rejected = regression.get(f)
            if rejected:
                xy = tuple(np.rint(np.array([rejected["x"], rejected["y"]]) * scale).astype(int))
                cv2.drawMarker(scene, xy, (80, 80, 255), cv2.MARKER_TILTED_CROSS, 20, 2)
            if old:
                xy = tuple(np.rint(np.array([old["x"], old["y"]]) * scale).astype(int))
                cv2.circle(scene, xy, 13, (255, 160, 50), 1, cv2.LINE_AA)
            if current:
                xy = tuple(np.rint(np.array([current["x"], current["y"]]) * scale).astype(int))
                cv2.circle(scene, xy, 8, (70, 255, 70), 2, cv2.LINE_AA)
            transform = cache["camera"]["transforms"].get(str(f))
            proposal = predicted.get(f)
            if proposal and transform is not None:
                inv = np.linalg.inv(np.array(transform))
                xy = project_image_point(inv, proposal["predicted_stabilized_xy"])
                if xy is not None:
                    # Map the entire reference search ellipse into the image.
                    radius = proposal["search_radius_px"]
                    reference = np.array(proposal["predicted_stabilized_xy"])
                    points = [project_image_point(inv, reference + radius * np.array([np.cos(a), np.sin(a)]))
                              for a in np.linspace(0, 2 * np.pi, 48)]
                    if all(p is not None for p in points):
                        polygon = np.rint(np.array(points) * scale).astype(np.int32)
                        for i in range(0, 46, 4):
                            cv2.polylines(scene, [polygon[i:i + 3]], False, (20, 180, 255), 1)
                    q = tuple(np.rint(np.array(xy) * scale).astype(int))
                    cv2.drawMarker(scene, q, (20, 180, 255), cv2.MARKER_CROSS, 10, 1)
            im[30:570] = scene
            label(im, f"{cache['clip_id']} | source frame {f} | playback 1/3", 21)
            label(im, "Green: preserved + continued | Blue: prior flight | Red x: previous failed version", 591)
            label(im, "Orange: prediction only | source observations preserved | outcome unknown", 612)
            proc.stdin.write(im.tobytes())
            if f in samples:
                tiles.append(im.copy())
                cv2.imwrite(str(dest / f"{cache['clip_id']}_frame_{f}.jpg"), im)
    finally:
        cap.release()
        proc.stdin.close()
    error = proc.stderr.read().decode(errors="replace")
    if proc.wait():
        raise RuntimeError(error)
    cap = cv2.VideoCapture(str(video))
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape != (620, 960, 3):
            raise AssertionError("Unexpected rendered dimensions")
        count += 1
    cap.release()
    if count != cache["end_frame_exclusive"] - cache["start_frame"]:
        raise AssertionError("Output full decode failed")
    sheet = dest / (cache["clip_id"] + "_contact_sheet.jpg")
    cv2.imwrite(str(sheet), np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))
    return {"video": video.name, "contact_sheet": sheet.name, "decoded_frames": count,
            "video_sha256": sha(video)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/ball_continuity_2026-10-11")
    parser.add_argument("--fixed-incoming", action="store_true",
                        help="Preserve the entire prior incoming flight; only associate beyond its end")
    parser.add_argument("--collision-gated", action="store_true",
                        help="Follow local flight trends; abrupt turns require prior-ball proximity")
    parser.add_argument("--prefix-points", type=int,
                        help="Freeze only this many early incoming observations, before impacts")
    parser.add_argument("--comparison-dir", type=Path,
                        help="Optional previous continuity experiment for the blue overlay")
    parser.add_argument("--regression-dir", type=Path,
                        help="Optional failed experiment for the red cross overlay")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    snapshot = args.output_dir / "source_snapshot"
    snapshot.mkdir()
    for source in (Path(__file__), ROOT / "src/ball_continuity.py", ROOT / "tests/test_ball_continuity.py"):
        (snapshot / source.name).write_bytes(source.read_bytes())
    cfg = ContinuityConfig(require_collision_for_turn=args.collision_gated,
                           lock_flight_direction=args.collision_gated)
    # Freeze parameters before reading these four results. None are independent
    # held-out tests; their earlier failures are already known.
    save(args.output_dir / "protocol.json", {"config": cfg.__dict__,
         "samples": "two previous development clips and two previously inspected probes",
         "held_out": False, "outcome_labels_read": False, "semantic_masks_used": False,
         "fixed_previous_incoming": args.fixed_incoming, "prefix_points": args.prefix_points,
         "collision_gated": args.collision_gated,
         "incoming_policy": ("complete_prior_flight_preserved" if args.fixed_incoming and
                             args.prefix_points is None else "early_prefix_ablation_or_bootstrap"),
         "regression_dir": str(args.regression_dir) if args.regression_dir else None,
         "comparison_dir": str(args.comparison_dir) if args.comparison_dir else "previous_v6"})
    rows = []
    for folder in (BASE, BASE / "additional_probe"):
        for file in sorted((folder / "release_v6").glob("*.json")):
            if file.name == "summary.json":
                continue
            earlier = load(file)
            cache = load(ROOT / earlier["cache_path"])
            source = Path(cache["video_path"])
            before = sha(source)
            if before != cache["source_sha256"]:
                raise AssertionError("Source hash differs from detection cache")
            timing = source_time_indices(cache)
            blobs = motion_blobs(cache)
            candidates = {f: [{**d, **timing[int(f)]} for d in ds + blobs.get(f, [])]
                          for f, ds in cache["detections"].items()}
            metadata = {str(f): {"coordinate_supported": cache["camera"]["transforms"].get(str(f)) is not None}
                        for f in timing}
            inputs_hash = hashlib.sha256(json.dumps(candidates, sort_keys=True).encode()).hexdigest()
            old_variant = earlier["variants"]["detector_motion_turn"]
            initial_path = (old_variant["trajectory"][:old_variant["tracking"]["incoming_observations"]]
                            if args.fixed_incoming else None)
            if args.prefix_points is not None:
                if not args.fixed_incoming or args.prefix_points < 3:
                    raise ValueError("--prefix-points requires --fixed-incoming and at least 3 points")
                initial_path = initial_path[:args.prefix_points]
            regions = collision_geometry(cache, earlier) if args.collision_gated else None
            path, diagnostics = track_ball_continuity(candidates, cache["kick_frame"], earlier["anchors"],
                earlier["goal_reference_center"], frame_metadata=metadata, config=cfg, initial_path=initial_path,
                collision_regions=regions)
            replay, replay_diag = track_ball_continuity(candidates, cache["kick_frame"], earlier["anchors"],
                earlier["goal_reference_center"], frame_metadata=metadata, config=cfg, initial_path=initial_path,
                collision_regions=regions)
            if path != replay or diagnostics != replay_diag:
                raise AssertionError("Replay differs")
            preserved = initial_path is None or all(
                (a["frame"], a["detection_id"], a["x"], a["y"]) ==
                (b["frame"], b["detection_id"], b["x"], b["y"])
                for a, b in zip(initial_path, path[:len(initial_path)])) and len(path) >= len(initial_path)
            if not preserved:
                raise AssertionError("Existing incoming flight was replaced")
            if hashlib.sha256(json.dumps(candidates, sort_keys=True).encode()).hexdigest() != inputs_hash:
                raise AssertionError("Input candidates modified")
            ids = {(d["frame"], d["detection_id"]) for ds in candidates.values() for d in ds}
            assert all((p["frame"], p["detection_id"]) in ids for p in path)
            assert len({p["frame"] for p in path}) == len(path)
            forecasts = [p for p in motion_forecasts(path, cache["start_frame"], cache["end_frame_exclusive"],
                         config=cfg, model_states=diagnostics.get("motion_states"))
                           if metadata[str(p["frame"])]["coordinate_supported"]]
            predictions = [p for p in forecasts if not p["observation_available"]]
            old = (load(args.comparison_dir / file.name)["trajectory"] if args.comparison_dir
                   else earlier["variants"]["detector_motion_turn"]["trajectory"])
            failed = load(args.regression_dir / file.name)["trajectory"] if args.regression_dir else None
            media = render(cache, path, forecasts, old, args.output_dir, regression_path=failed)
            if sha(source) != before:
                raise AssertionError("Source changed")
            result = {"clip_id": cache["clip_id"], "cache_path": earlier["cache_path"],
                      "trajectory": path, "tracking": diagnostics, "predictions": predictions,
                      "forecasts": forecasts, "collision_region_candidates": regions,
                      "prior_incoming_observations_preserved": preserved,
                      "preserved_incoming_count": len(initial_path) if initial_path else 0,
                      "source_sha256_verified": True, "deterministic_replay": True,
                      "outcome_labels_read": False, "accuracy_validated": False, "human_verified": False}
            save(args.output_dir / (cache["clip_id"] + ".json"), result)
            row = {"clip_id": cache["clip_id"], "old_points": len(old), "new_points": len(path),
                   "old_end": old[-1]["frame"] if old else None, "new_end": path[-1]["frame"] if path else None,
                   "max_gap": diagnostics.get("max_gap_frames"), "turn_frames": diagnostics.get("turn_frames", []),
                   "prediction_count": len(predictions), "coordinate_break": diagnostics.get("coordinate_break_frame"),
                   "forecast_count": len(forecasts),
                   "rejected_turns_without_collision_cue": diagnostics.get("rejected_turns_without_collision_cue", 0),
                   "collision_turn_candidates": diagnostics.get("collision_turn_candidates", []),
                   "tracking_status": diagnostics["tracking_status"], **media}
            row["preserved_incoming_count"] = len(initial_path) if initial_path else 0
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = {"rows": rows, "config": cfg.__dict__, "accuracy_validated": False,
              "baseline_replaced": False, "outcome": "unknown", "visual_review": "pending",
              "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                              (Path(__file__), ROOT / "src/ball_continuity.py")}}
    save(args.output_dir / "summary.json", report)
    body = "<meta charset='utf-8'><title>球轨迹预测与碰撞约束实验</title><style>body{max-width:1080px;margin:30px auto;background:#15191f;color:#ddd;font:16px sans-serif}video,img{width:100%}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #666}a{color:#8bd}</style>"
    body += "<h1>球轨迹连续性修正</h1><p>保留此前已经跟上的完整飞行段，只在末端续接；后段衣物或网纹候选不能反过来删除这段观测。绿色圈：保留及续接的观测；蓝色圈：此前连续轨迹；红色叉：上一版错误轨迹；橙色虚线：观测前预测。预测不计为检测或越线证据。</p>"
    if args.collision_gated:
        body += "<p>根据最近的观测更新运动趋势，允许图像轨迹逐渐弯曲；连续飞行仍有候选支持时优先续接。只有此前球的位置靠近碰撞候选时才考虑突然转向，新候选落在门将身上不能为自己制造碰撞理由。新增观测在2个原始帧后固定；漏检预测最多6帧。几何接近仍不能证明物理接触。</p>"
    body += "<p>参数事先固定，四条样本均曾查看；本页没有独立准确率。跟得更久可能也跟错，须查看原画面。</p>"
    body += "<table><tr><th>样本</th><th>旧/新观测点</th><th>旧/新结束帧</th><th>最长观测间隔</th></tr>"
    for row in rows:
        body += f"<tr><td>{html.escape(row['clip_id'])}</td><td>{row['old_points']} / {row['new_points']}</td><td>{row['old_end']} / {row['new_end']}</td><td>{row['max_gap']}</td></tr>"
    body += "</table>"
    for row in rows:
        body += f"<h2>{html.escape(row['clip_id'])}</h2><video controls preload='metadata' src='{row['video']}'></video><p><a href='{row['clip_id']}.json'>观测与预测数据</a></p><img src='{row['contact_sheet']}'>"
    (args.output_dir / "report.html").write_text(body, encoding="utf-8")


if __name__ == "__main__":
    main()

"""Fresh, source-only ball detection and current association on canonical clips.

Automatic contact/shot/pose proposals define the window and foot anchors. No
old ball trajectory, outcome label, glove/boot/net mask, or human point seeds
enter inference. Visual review is separate from candidate/QA counts.
"""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import html
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from src.ball_pipeline import _read_window, _extract_detections
from src.ball_roi_probe import ROIPredictor
from src.ball_continuity import ContinuityConfig, motion_forecasts, track_ball_continuity
from src.event_ball_tracking import select_stationary_seed
from src.goal_geometry import detect_goal_frame
from src.goal_plane_rectification import project_image_point
from src.motion_stabilization import estimate_background_motion
from src.project_paths import resolve_project_file, resolve_project_video
from src.runtime import add_runtime_arguments, resolve_device, resolve_ffmpeg, inference_provenance
from scripts.probe_goal_event_tracking import motion_blobs, source_time_indices
from scripts.run_ball_continuity_probe import collision_geometry

BASE = ROOT / "outputs/optimization_evening_2026-10-09"
WEIGHTS = ROOT / "models/weights/yolo11n_object_detection.pt"
cv2.setNumThreads(3)
torch.set_num_threads(3)


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def save(p, data):
    Path(p).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def canonical_ids():
    manifest = yaml.safe_load((ROOT / "configs/penalty_slicing_manifest.yaml").read_text(encoding="utf-8"))
    return [f"{match}_p{r['number']:02d}_{r['kicker']}" for match, info in manifest.items()
            for r in info["penalties"]]


def detect_clip(cid, model, batch_size, device='cpu',source_dir=None,steps_dir=None,image_size=1280,model_path=WEIGHTS):
    step_path = (steps_dir or BASE / "steps_v4/classified") / (cid + ".json")
    step = load(step_path)
    contact = step["contact"]
    kick, fps = contact["contact_frame"], step["fps"]
    pose_path = resolve_project_file(ROOT,step["terminal_provenance"]["selected_pose_path"])
    pose = load(pose_path)
    source = resolve_project_video(ROOT,step["video_path"],cid,source_dir)
    before = sha(source)
    start = max(step["shot_start_frame"], kick - round(.4 * fps))
    end = min(step["shot_end_frame_exclusive"], kick + round(2.3 * fps) + 1)
    ids, frames, actual_fps, w, h = _read_window(str(source), start, end - 1)
    if len(ids) != end - start:
        raise ValueError("Incomplete source decode")
    geometries = {fid: detect_goal_frame(fr) for fid, fr in zip(ids, frames)}
    goal_options = [(abs(fid - (kick - 6)), g["selected"]) for fid, g in geometries.items()
                    if g.get("selected")]
    roi = None
    if goal_options:
        goal = min(goal_options, key=lambda t: t[0])[1]
        x0, y0, x1, y1 = goal["bbox"]
        gw, gh = x1 - x0, y1 - y0
        roi = [max(0, int(x0 - .6 * gw)), max(0, int(y0 - .3 * gh)),
               min(w, int(x1 + .6 * gw)), min(h, int(y1 + .8 * gh))]
    all_dets = {fid: [] for fid in ids}
    modes = [("full", model)] + ([("goal_roi", ROIPredictor(model, roi))] if roi else [])
    for mode, predictor in modes:
        for j in range(0, len(frames), batch_size):
            batch = frames[j:j + batch_size]
            pred = predictor.predict(batch, imgsz=image_size, conf=.01, classes=[32],
                                     device=device, verbose=False)
            det = _extract_detections(ids[j:j + batch_size], pred, batch)
            for fid, ds in det.items():
                all_dets[fid].extend({**d, "sources": [mode]} for d in ds)
    boxes = defaultdict(list)
    for track in pose.get("tracks", []):
        for obs in track.get("observations", []):
            if start <= obs["frame"] < end:
                boxes[obs["frame"]].append(obs["bbox"])
    camera = estimate_background_motion(str(source), start, end, boxes_by_frame=boxes)
    gray = {fid: cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) for fid, fr in zip(ids, frames)}
    for fid in ids:
        merged = []
        for d in sorted(all_dets[fid], key=lambda q: -q["confidence"]):
            hit = next((v for v in merged if np.hypot(d["x"] - v["x"], d["y"] - v["y"]) <
                        max(4., .4 * min(d["width"], v["width"]))), None)
            if hit is not None:
                hit["sources"] = sorted(set(hit["sources"] + d["sources"]))
            else:
                merged.append(d)
        tf, previous_tf = camera["transforms"].get(str(fid)), camera["transforms"].get(str(fid - 1))
        diff = None
        if tf is not None and previous_tf is not None and fid - 1 in gray:
            previous = cv2.warpPerspective(gray[fid - 1], np.linalg.inv(np.array(tf)) @ np.array(previous_tf), (w, h))
            diff = cv2.absdiff(gray[fid], previous)
        for i, d in enumerate(merged):
            d["detection_id"] = f"{fid}:{i}"
            d["stabilized_xy"] = project_image_point(tf, [d["x"], d["y"]]) if tf is not None else None
            d["motion_mean_abs_gray"] = None
            if diff is not None:
                radius = max(3, int(min(d["width"], d["height"]) / 2))
                x, y = round(d["x"]), round(d["y"])
                patch = diff[max(0, y - radius):min(h, y + radius + 1), max(0, x - radius):min(w, x + radius + 1)]
                d["motion_mean_abs_gray"] = round(float(np.mean(patch)), 3) if patch.size else None
        all_dets[fid] = merged
    if sha(source) != before:
        raise AssertionError("Source changed during detection")
    return {"clip_id": cid, "status": "cached_candidates", "video_path": str(source),
            "source_sha256": before, "start_frame": start, "end_frame_exclusive": end,
            "kick_frame": kick, "fps": actual_fps, "width": w, "height": h, "contact": contact,
            "goal_roi": roi, "goal_proposals": geometries, "camera": camera, "detections": all_dets,
            "source_paths": {"step": str(step_path), "pose": str(pose_path)},
            "fresh_model_inference": True, "old_ball_trajectory_input": False,
            "runtime":inference_provenance(device,device,[model_path],[source],torch),
            "input_sha256":{str(p):sha(p) for p in (step_path,pose_path)},
            "modes": [m for m, _ in modes]}


def anchors_and_goal(cache):
    pose = load(resolve_project_file(ROOT,cache["source_paths"]["pose"]))
    track = next((t for t in pose["tracks"] if t["track_id"] == cache["contact"]["contact_associated_track_id"]), {})
    anchors = []
    for obs in track.get("observations", []):
        tf = cache["camera"]["transforms"].get(str(obs["frame"]))
        if abs(obs["frame"] - cache["kick_frame"]) <= 3 and tf is not None:
            for j in (15, 16):
                if obs["keypoint_confidences"][j] >= .25:
                    p = project_image_point(tf, obs["keypoints"][j])
                    if p is not None:
                        anchors.append(p)
    options = [(abs(int(f) - cache["kick_frame"]), int(f), g["selected"])
               for f, g in cache["goal_proposals"].items() if g.get("selected") and
               cache["camera"]["transforms"].get(str(f)) is not None]
    if not options:
        return anchors, None
    _, f, g = min(options, key=lambda q: q[0])
    return anchors, project_image_point(cache["camera"]["transforms"][str(f)], np.mean(g["corners"], axis=0))


def associate(cache):
    timing = source_time_indices(cache)
    blobs = motion_blobs(cache)
    detector = {str(f): [{**d, **timing[int(f)]} for d in ds]
                for f, ds in cache["detections"].items()}
    mixed = {f: ds + [{**d, **timing[int(f)]} for d in blobs.get(f, [])] for f, ds in detector.items()}
    anchors, goal = anchors_and_goal(cache)
    metadata = {str(f): {"coordinate_supported": cache["camera"]["transforms"].get(str(f)) is not None}
                for f in timing}
    regions = collision_geometry(cache, {})  # Goal geometry only; no prior keeper or ball path.
    cfg = ContinuityConfig()
    seed, seed_diag = select_stationary_seed(detector, cache["kick_frame"], anchors)
    variants = {}
    for name, candidates in (("detector_only", detector), ("detector_and_motion", mixed)):
        if goal is None:
            path, diag = [], {"tracking_status": "abstain_no_goal_geometry", "outcome": "unknown"}
        else:
            before = hashlib.sha256(json.dumps(candidates, sort_keys=True).encode()).hexdigest()
            path, diag = track_ball_continuity(candidates, cache["kick_frame"], anchors, goal,
                config=cfg, frame_metadata=metadata, collision_regions=regions, initial_path=None)
            replay, repeat = track_ball_continuity(candidates, cache["kick_frame"], anchors, goal,
                config=cfg, frame_metadata=metadata, collision_regions=regions, initial_path=None)
            if path != replay or diag != repeat:
                raise AssertionError("Non-deterministic association replay")
            if hashlib.sha256(json.dumps(candidates, sort_keys=True).encode()).hexdigest() != before:
                raise AssertionError("Association modified candidates")
            valid = {(d["frame"], d["detection_id"]) for ds in candidates.values() for d in ds}
            assert all((d["frame"], d["detection_id"]) in valid for d in path)
        forecasts = [q for q in motion_forecasts(path, cache["start_frame"], cache["end_frame_exclusive"],
                     config=cfg, model_states=diag.get("motion_states"))
                     if metadata[str(q["frame"])]["coordinate_supported"]]
        variants[name] = {"trajectory": path, "tracking": diag, "forecasts": forecasts,
                          "old_trajectory_supplied": False, "deterministic_replay": goal is not None}
    return {"clip_id": cache["clip_id"], "anchors": anchors, "goal_reference_center": goal,
            "stationary_seed": seed, "seed_diagnostics": seed_diag, "variants": variants,
            "human_verified": False, "accuracy_validated": False, "old_ball_trajectory_input": False,
            "candidate_detector_count": sum(map(len, detector.values())),
            "candidate_motion_count": sum(map(len, blobs.values()))}


def text(im, value, xy, color=(240, 240, 240), scale=.45):
    cv2.putText(im, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def crop_zoom(frame, xy, size=180):
    x, y = map(round, xy)
    padded = cv2.copyMakeBorder(frame, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    crop = padded[max(0, y):max(0, y) + 80, max(0, x):max(0, x) + 80]
    if crop.size == 0:
        return np.full((size, size, 3), 30, np.uint8)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_NEAREST)


def render(cache, result, dest,ffmpeg=None):
    cid = cache["clip_id"]
    path = result["variants"]["detector_and_motion"]["trajectory"]
    pure = result["variants"]["detector_only"]["trajectory"]
    observed, detected = ({p["frame"]: p for p in rows} for rows in (path, pure))
    seed = {p["frame"]: p for p in result["stationary_seed"]}
    predicted = {p["frame"]: p for p in result["variants"]["detector_and_motion"]["forecasts"]}
    samples = [min(cache["end_frame_exclusive"] - 1, max(cache["start_frame"], cache["kick_frame"] + n))
               for n in (-5, 0, 5, 10, 20, 40)]
    video = dest / (cid + ".mp4")
    cmd = [resolve_ffmpeg(ffmpeg,ROOT), "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
           "-pix_fmt", "bgr24", "-s", "900x460", "-r", str(cache["fps"] / 3), "-i", "pipe:0",
           "-an", "-c:v", "libx264", "-threads", "2", "-crf", "21", "-preset", "fast",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    cap = cv2.VideoCapture(cache["video_path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, cache["start_frame"])
    tiles = {}
    scale = np.array([640 / cache["width"], 360 / cache["height"]])
    try:
        for f in range(cache["start_frame"], cache["end_frame_exclusive"]):
            ok, original = cap.read()
            if not ok:
                raise ValueError("Media source decode failed")
            im = np.full((460, 900, 3), 25, np.uint8)
            scene = cv2.resize(original, (640, 360))
            # Show raw neural detections separately from association decisions.
            for d in sorted(cache["detections"].get(str(f), cache["detections"].get(f, [])), key=lambda q: -q["confidence"])[:12]:
                if d["confidence"] >= .04:
                    xy = tuple(np.rint(np.array([d["x"], d["y"]]) * scale).astype(int))
                    cv2.circle(scene, xy, 5, (150, 150, 150), 1)
            direct, current, pre = detected.get(f), observed.get(f), seed.get(f)
            for d, color, radius in ((pre, (60, 220, 255), 8), (direct, (255, 155, 40), 11),
                                     (current, (70, 255, 70), 7)):
                if d:
                    xy = tuple(np.rint(np.array([d["x"], d["y"]]) * scale).astype(int))
                    cv2.circle(scene, xy, radius, color, 2)
            focus = current or direct or pre
            forecast = predicted.get(f)
            if forecast and cache["camera"]["transforms"].get(str(f)) is not None:
                xy = project_image_point(np.linalg.inv(np.array(cache["camera"]["transforms"][str(f)])),
                                         forecast["predicted_stabilized_xy"])
                if xy is not None:
                    q = tuple(np.rint(np.array(xy) * scale).astype(int))
                    cv2.drawMarker(scene, q, (40, 170, 255), cv2.MARKER_CROSS, 10, 1)
                    if focus is None:
                        focus = {"x": xy[0], "y": xy[1], "sources": ["prediction_only"]}
            im[35:395, :640] = scene
            text(im, f"{cid} | frame {f} | kick candidate {cache['kick_frame']}", (12, 21))
            text(im, "Raw zoom of selected target", (657, 55))
            if focus:
                im[70:250, 675:855] = crop_zoom(original, [focus["x"], focus["y"]])
                kind = "prediction only" if focus["sources"] == ["prediction_only"] else (
                       "motion component" if "motion_blob" in focus["sources"] else "neural detection")
                text(im, kind, (665, 273), (120, 230, 230))
                text(im, f"x={focus['x']:.1f} y={focus['y']:.1f}", (660, 292))
            else:
                text(im, "No associated observation", (653, 140))
            raw = sorted(cache["detections"].get(str(f), cache["detections"].get(f, [])), key=lambda q: -q["confidence"])
            if raw:
                top = raw[0]
                im[315:395, 650:730] = crop_zoom(original, [top["x"], top["y"]], 80)
                text(im, f"Top raw conf {top['confidence']:.2f}", (739, 353), scale=.4)
            text(im, "Green: detector + motion | Blue: detector-only association | Yellow: pre-kick seed", (12, 416))
            text(im, "Gray: raw detector candidates | Orange +: prediction only | Fresh inference, no old ball path", (12, 439))
            proc.stdin.write(im.tobytes())
            if f in samples:
                tiles[f] = im.copy()
                cv2.imwrite(str(dest / f"{cid}_f{f}.jpg"), im)
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
        assert frame.shape == (460, 900, 3)
        count += 1
    cap.release()
    assert count == cache["end_frame_exclusive"] - cache["start_frame"]
    selected = [tiles[f] for f in samples]
    sheet = np.vstack([np.hstack(selected[i:i + 2]) for i in range(0, 6, 2)])
    cv2.imwrite(str(dest / (cid + "_sheet.jpg")), sheet)
    return {"video": video.name, "contact_sheet": cid + "_sheet.jpg", "sample_frames": samples,
            "decoded_frames": count, "video_sha256": sha(video)}


def write_report(out, rows, errors):
    fields = ["clip_id", "seed_points", "detector_only_points", "mixed_points", "mixed_end_frame",
              "detector_candidates", "motion_candidates", "tracking_status", "camera_status", "decoded_frames"]
    with (out / "candidate_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    save(out / "summary.json", {"processed_clips": len(rows), "errors": errors, "rows": rows,
        "accuracy_validated": False, "human_verified": False, "visual_review": "pending",
        "old_ball_trajectory_input": False, "fresh_model_inference": True})
    body = """<meta charset='utf-8'><title>39条视频球识别复测</title><style>
body{max-width:1150px;margin:28px auto;background:#15191f;color:#ddd;font:16px sans-serif}
video,img{width:100%}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #555}
a{color:#8bd}details{margin:20px 0;border:1px solid #555;padding:12px}summary{cursor:pointer}
</style><h1>全部39条标准点球：球识别复测</h1>
<p>重新运行本地YOLO11n球检测：逐帧全图＋可用的球门区域。既有自动触球/机位/姿态提议限定窗口，旧球轨迹没有输入关联。绿色是检测＋运动候选关联，蓝色是仅检测关联，黄色是触球前多帧球候选；右侧显示对应原像素局部放大。橙色仅预测，不计为识别。</p>
<p>检测候选数、关联点数和程序检查不能证明识别成功。以下回放需要逐条查看真实球与误检；本轮不自动判进球。</p>"""
    body += f"<p>完成 {len(rows)} / 39 条；处理错误 {len(errors)} 条。视觉核对结果尚待写入。</p>"
    body += "<table><tr><th>片段</th><th>静止候选</th><th>纯检测/含运动关联点</th><th>结束帧</th><th>状态</th></tr>"
    for r in rows:
        body += f"<tr><td><a href='#{r['clip_id']}'>{r['clip_id']}</a></td><td>{r['seed_points']}</td><td>{r['detector_only_points']} / {r['mixed_points']}</td><td>{r['mixed_end_frame']}</td><td>{r['tracking_status']}</td></tr>"
    body += "</table>"
    for r in rows:
        body += f"<details id='{r['clip_id']}'><summary>{html.escape(r['clip_id'])}</summary><video controls preload='none' src='media/{r['video']}'></video><img loading='lazy' src='media/{r['contact_sheet']}'><a href='results/{r['clip_id']}.json'>逐帧候选与关联数据</a></details>"
    (out / "report.html").write_text(body, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    add_runtime_arguments(parser)
    parser.add_argument("--source-dir",type=Path,default=ROOT/"data/raw_clips")
    parser.add_argument("--steps-dir",type=Path,default=BASE/"steps_v4/classified")
    parser.add_argument("--model",type=Path,default=WEIGHTS)
    parser.add_argument("--image-size",type=int,default=1280)
    args = parser.parse_args()
    if args.image_size <= 0 or args.batch_size <= 0:
        parser.error("Image size and batch size must be positive")
    device=resolve_device(args.device,torch);ffmpeg=resolve_ffmpeg(args.ffmpeg,ROOT)
    out = args.output_dir
    if out.exists() and not args.resume:
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    for folder in ("detections", "results", "media", "source_snapshot"):
        (out / folder).mkdir(exist_ok=True)
    ids = canonical_ids()
    assert len(ids) == len(set(ids)) == 39
    assert all((args.steps_dir / (cid + ".json")).exists() for cid in ids)
    protocol = {"created_at": datetime.now(timezone.utc).isoformat(), "canonical_clips": ids,
        "weights": str(args.model), "weights_sha256": sha(args.model), "fresh_model_inference": True,
        "detector_class": 32, "conf": .01, "imgsz": args.image_size, "device": device, "threads": 3,
        "runtime":inference_provenance(args.device,device,[args.model],torch_module=torch),"ffmpeg":ffmpeg,
        "steps_dir":str(args.steps_dir.resolve()),"source_dir":str(args.source_dir.resolve()),
        "old_ball_trajectory_input": False, "outcome_labels_read": False, "manual_ball_seed_input": False,
        "semantic_masks_used": False, "automatic_contact_and_pose_proposals_used": True,
        "window_policy": "automatic kick minus0.4s plus2.3s bounded by primary shot",
        "held_out": False, "independent_accuracy_measured": False,
        "config": ContinuityConfig().__dict__}
    sources = [Path(__file__), ROOT / "src/ball_continuity.py", ROOT / "src/event_ball_tracking.py",
               ROOT / "src/goal_event_tracker_v2.py", ROOT / "src/ball_pipeline.py",
               ROOT / "scripts/probe_goal_event_tracking.py",ROOT/"src/runtime.py",ROOT/"src/project_paths.py"]
    protocol["code_sha256"] = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    if not args.resume:
        save(out / "protocol.json", protocol)
        for p in sources:
            (out / "source_snapshot" / p.name).write_bytes(p.read_bytes())
    else:
        previous = load(out / "protocol.json")
        keys=("code_sha256","weights_sha256","imgsz","device","steps_dir","source_dir")
        if any(previous.get(key)!=protocol[key] for key in keys):
            raise ValueError("Resume code/weights/runtime changed")
    model = YOLO(str(args.model))
    rows, errors = [], []
    for index, cid in enumerate(ids, 1):
        begun = time.monotonic()
        cache_path = out / "detections" / (cid + ".json")
        result_path = out / "results" / (cid + ".json")
        try:
            if args.resume and result_path.exists():
                r = load(result_path)
                source=resolve_project_video(ROOT,'',cid,args.source_dir)
                if sha(source)!=r["source_sha256"]:
                    raise ValueError("Resumed result belongs to different source video")
                rows.append(r["summary"])
                continue
            print(f"[{index}/39] {cid} fresh detection start", flush=True)
            cache = load(cache_path) if args.resume and cache_path.exists() else detect_clip(cid, model, args.batch_size,
                                      device,args.source_dir,args.steps_dir,args.image_size,args.model)
            cache["video_path"]=str(resolve_project_video(ROOT,cache["video_path"],cid,args.source_dir))
            if sha(cache["video_path"])!=cache["source_sha256"]:
                raise ValueError("Detection cache belongs to different source video")
            for recorded,digest in cache.get("input_sha256",{}).items():
                if sha(resolve_project_file(ROOT,recorded))!=digest:
                    raise ValueError("Cached pose/contact evidence changed")
            save(cache_path, cache)
            cache = load(cache_path)  # Normalize source-frame dictionary keys for replay.
            result = associate(cache)
            media = render(cache, result, out / "media",ffmpeg)
            assert sha(Path(cache["video_path"])) == cache["source_sha256"]
            a, b = result["variants"]["detector_only"], result["variants"]["detector_and_motion"]
            row = {"clip_id": cid, "seed_points": len(result["stationary_seed"]),
                   "detector_only_points": len(a["trajectory"]), "mixed_points": len(b["trajectory"]),
                   "mixed_end_frame": b["trajectory"][-1]["frame"] if b["trajectory"] else None,
                   "detector_candidates": result["candidate_detector_count"],
                   "motion_candidates": result["candidate_motion_count"],
                   "tracking_status": b["tracking"]["tracking_status"], "camera_status": cache["camera"]["status"],
                   "start_frame": cache["start_frame"], "end_frame_exclusive": cache["end_frame_exclusive"],
                   "kick_frame": cache["kick_frame"], "elapsed_seconds": round(time.monotonic() - begun, 2), **media}
            result.update(cache_path=str(cache_path), source_sha256=cache['source_sha256'],
                          runtime=cache.get('runtime'),input_sha256=cache.get('input_sha256',{}),source_sha256_verified=True,
                          fresh_model_inference=True, summary=row)
            save(result_path, result)
            rows.append(row)
            print(json.dumps(row), flush=True)
        except Exception as exc:
            errors.append({"clip_id": cid, "error_type": type(exc).__name__, "error": str(exc)})
            print(json.dumps(errors[-1]), flush=True)
        write_report(out, rows, errors)
    write_report(out, rows, errors)
    if errors or len(rows) != 39:
        raise SystemExit("Retest incomplete; inspect errors")


if __name__ == "__main__":
    main()

"""Package the phase experiment with complete browser-compatible local videos."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_review_video import inspect_video


def load(path):
    return json.loads(path.read_text(encoding="utf8"))


def export_media(result, directory, ffmpeg):
    source = Path(result["video_path"])
    target = directory / f"{result['clip_id']}.mp4"
    validation = target.with_suffix(".validation.json")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if validation.exists() and target.exists():
        report = load(validation)
        if report.get("source_sha256") == source_hash and report.get("playback_sha256") == hashlib.sha256(target.read_bytes()).hexdigest():
            return report
        raise ValueError(f"Existing media provenance mismatch; refusing overwrite: {target}")
    if target.exists():
        raise FileExistsError(f"Unvalidated playback file exists; inspect before retry: {target}")
    original = inspect_video(source)
    subprocess.run([str(ffmpeg), "-hide_banner", "-loglevel", "error", "-n", "-i", str(source),
                    "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-threads", "2",
                    "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-movflags", "+faststart", str(target)], check=True)
    converted = inspect_video(target)
    for key in ("decoded_frames", "fps", "width", "height"):
        if original[key] != converted[key]:
            raise ValueError(f"Playback changed {key}: {target}")
    report = {"clip_id": result["clip_id"], "source": str(source), "playback_copy": str(target),
              "source_sha256": source_hash, "playback_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
              "original": original, "converted": converted, "full_duration_preserved": True,
              "annotations_added_to_video": False, "human_verified": False}
    validation.write_text(json.dumps(report, indent=2), encoding="utf8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    run = ROOT / "outputs/phase_optimization_2026-10-09"
    parser.add_argument("--classified-dir", type=Path, default=run / "classified")
    parser.add_argument("--refined-classified-dir", type=Path, default=run / "classified_refined")
    parser.add_argument("--output-dir", type=Path, default=run / "review")
    parser.add_argument("--ffmpeg", type=Path, default=ROOT / "outputs/review_tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe")
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--media-prefix", default="media/", help="Relative media path within the local review server")
    parser.add_argument("--comparison-path", type=Path, help="Optional version-specific model comparison output")
    args = parser.parse_args()
    clips = [load(p) for p in sorted(args.classified_dir.glob("*.json")) if p.stem.startswith(("col_eng_p", "cro_den_p", "esp_rus_p", "rus_cro_p"))]
    refined = [load(p) for p in sorted(args.refined_classified_dir.glob("*.json")) if p.stem.startswith(("col_eng_p", "cro_den_p", "esp_rus_p", "rus_cro_p"))]
    refined = [r for r in refined if r["pose_source"].startswith("yolo11m")]
    pairs = []
    refined_by_id = {r["clip_id"]: r for r in refined}
    for clip in clips:
        counterpart = refined_by_id.get(clip["clip_id"])
        if not counterpart:
            continue
        key = "approach_candidates_including_precontact_strike_window"
        a = clip["count_sensitivity"].get("central", {}).get(key)
        b = counterpart["count_sensitivity"].get("central", {}).get(key)
        start_a, start_b = clip["phase_result"]["approach_start_frame"], counterpart["phase_result"]["approach_start_frame"]
        tags_a = sorted({t["tag"] for t in clip["phase_result"]["style_tags"]})
        tags_b = sorted({t["tag"] for t in counterpart["phase_result"]["style_tags"]})
        pair = {"clip_id": clip["clip_id"], "original_nominal_candidates": a, "refined_nominal_candidates": b,
                "candidate_count_difference": b - a if a is not None and b is not None else None,
                "onset_difference_frames": start_b - start_a if start_a is not None and start_b is not None else None,
                "style_tags_match": tags_a == tags_b, "human_verified": False}
        if "footwork_result" in clip and "footwork_result" in counterpart:
            fa, fb = clip["footwork_result"]["terminal"], counterpart["footwork_result"]["terminal"]
            pair["terminal_status_and_tags_match"] = (fa["status"], sorted(fa["tags"])) == (fb["status"], sorted(fb["tags"]))
            pair["terminal_boundary_difference_frames"] = fb["start_frame"] - fa["start_frame"] if fa["start_frame"] is not None and fb["start_frame"] is not None else None
        if "step_analysis" in clip and "step_analysis" in counterpart:
            pair["attempt_style_match"] = clip["step_analysis"]["attempt_style"] == counterpart["step_analysis"]["attempt_style"]
        clip["pose_ablation_comparison"] = counterpart["pose_ablation_comparison"] = pair
        pairs.append(pair)
    comparison = {"actual_refined_comparisons": len(pairs),
        "count_changed": sum(p["candidate_count_difference"] not in (None, 0) for p in pairs),
        "count_difference_at_least_two": sum(abs(p["candidate_count_difference"] or 0) >= 2 for p in pairs),
        "onset_difference_more_than_four_frames": sum(abs(p["onset_difference_frames"] or 0) > 4 for p in pairs),
        "style_tag_set_changed": sum(not p["style_tags_match"] for p in pairs),
        "pairs": pairs, "accuracy_improvement_demonstrated": False, "human_verified": False}
    comparison["terminal_status_or_tags_changed"] = sum(p.get("terminal_status_and_tags_match") is False for p in pairs)
    comparison["attempt_style_changed"] = sum(p.get("attempt_style_match") is False for p in pairs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = args.comparison_path or args.output_dir.parent / "model_comparison.json"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf8")
    if ":" in args.media_prefix or args.media_prefix.startswith(("/", "\\")):
        parser.error("Media must be a relative local path, not an external URL")
    media = (args.output_dir / args.media_prefix).resolve()
    if args.media_prefix != "media/" and not args.data_only:
        parser.error("External sibling media can only be reused with --data-only")
    media.mkdir(exist_ok=True)
    if not args.data_only:
        with ThreadPoolExecutor(max_workers=2) as executor:
            reports = list(executor.map(lambda r: export_media(r, media, args.ffmpeg), clips))
        (args.output_dir / "media_validation.json").write_text(json.dumps({"clips": len(reports),
            "full_decode_passed": True, "source_duration_and_geometry_preserved": True,
            "human_verified": False, "reports": reports}, indent=2), encoding="utf8")
    else:
        # Reuse only hash-matched, previously fully decoded playback copies.
        reports = []
        for clip in clips:
            target = media / f"{clip['clip_id']}.mp4"
            report = load(target.with_suffix(".validation.json"))
            if (report.get("source_sha256") != hashlib.sha256(Path(clip["video_path"]).read_bytes()).hexdigest()
                    or report.get("playback_sha256") != hashlib.sha256(target.read_bytes()).hexdigest()
                    or not report.get("full_duration_preserved")):
                raise ValueError(f"Reused playback evidence mismatch: {target}")
            reports.append(report)
        (args.output_dir / "media_validation.json").write_text(json.dumps({"clips": len(reports),
            "full_decode_passed": True, "decode_provenance": "prior_full_decode_current_source_and_copy_hashes_rechecked",
            "human_verified": False, "reports": reports}, indent=2), encoding="utf8")
    payload = {"summary": load(args.classified_dir / "summary.json"), "clips": clips,
               "refined_clips": refined, "media_prefix": args.media_prefix}
    (args.output_dir / "review-data.js").write_text("window.PENALTY_PHASE_REVIEW = " +
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + ";\n", encoding="utf8")
    template = ROOT / "scripts/templates/penalty_phase_review.html"
    if template.exists():
        shutil.copyfile(template, args.output_dir / "index.html")
    print(json.dumps({"clips": len(clips), "refined_comparisons": len(refined), "output": str(args.output_dir),
                      "index_exists": (args.output_dir / "index.html").exists()}, ensure_ascii=False))


if __name__ == "__main__":
    main()

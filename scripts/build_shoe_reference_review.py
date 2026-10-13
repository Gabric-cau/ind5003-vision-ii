"""Build an offline original-footage viewer without displaying predictions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
COURSE_SOURCES = {"col_eng_pen.mp4", "cro_den_pen.mp4", "esp_rus_pen.mp4", "rus_cro_pen.mp4"}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def source_records(freeze: Path, clips_dir: Path, reference_template: Path, output_dir: Path, expected_events=39):
    manifest = read(freeze)
    if manifest.get("schema_version") != "shoe-landing-prediction-freeze-v1":
        raise ValueError("unsupported prediction freeze")
    entries = manifest.get("predictions", [])
    if len(entries) != expected_events:
        raise ValueError(f"expected {expected_events} original-course events")
    template = read(reference_template)
    if (template.get("schema_version") != "shoe-landing-reference-v1" or
        template.get("prediction_freeze_sha256") != sha256(freeze)):
        raise ValueError("reference template does not bind this prediction freeze")
    events = template.get("events", [])
    reference_ids = [event.get("clip_id") for event in events]
    if len(reference_ids) != expected_events or len(set(reference_ids)) != expected_events:
        raise ValueError("reference template must contain the same distinct events")
    for event in events:
        if not event.get("reviews"):
            raise ValueError("reference event has no pending review")
        for review in event["reviews"]:
            if (review.get("review_status") != "pending" or review.get("reviewer_id") != "" or
                review.get("reviewed_at") is not None or review.get("human_verified") is not False or
                review.get("annotator_type") is not None or review.get("striking_foot") != "unknown" or
                review.get("verified_landings") is not None or review.get("full_approach_visible") is not None):
                raise ValueError("blind viewer needs an untouched pending reference template")
    references = {event["clip_id"]: event for event in events}
    rows, clip_ids, clip_hashes, parent_hashes = [], set(), set(), {}
    for entry in entries:
        path = (freeze.parent / entry["path"]).resolve()
        if sha256(path) != entry["sha256"]:
            raise ValueError("prediction bytes changed after freezing")
        # Only source identity fields are admitted to the viewer payload.
        document = read(path)
        source = document["source_mapping"]
        clip_id = source.get("clip_id")
        if clip_id != entry["clip_id"] or document.get("clip_id") != clip_id or clip_id in clip_ids:
            raise ValueError("mismatched or duplicate source identity")
        if source.get("source_filename") not in COURSE_SOURCES:
            raise ValueError("source is outside the four original course videos")
        filename = source.get("clip_filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("clip_filename must be a local basename")
        fps = source.get("fps")
        start, end, frame_count = source.get("source_start_frame"), source.get("source_end_frame_exclusive"), source.get("frame_count")
        if (isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0 or
            not all(_integer(value) for value in (start, end, frame_count)) or frame_count == 0 or
            end - start != frame_count):
            raise ValueError("source frame interval and clock do not agree")
        if (document.get("source_sha256") != source.get("clip_sha256") or
            document.get("fps") != fps or document.get("frame_count") != frame_count):
            raise ValueError("prediction and original clip metadata differ")
        reference = references.get(clip_id)
        if (reference is None or reference.get("source_sha256") != source.get("clip_sha256") or
            reference.get("fps") != fps or reference.get("frame_count") != frame_count or
            reference.get("frame_clock") != "clip_local_zero_based"):
            raise ValueError("reference source identity differs")
        video = (clips_dir / filename).resolve()
        actual_hash = sha256(video)
        if actual_hash != source.get("clip_sha256"):
            raise ValueError("original clip SHA-256 mismatch")
        if actual_hash in clip_hashes:
            raise ValueError("duplicate original clip bytes would repeat one event")
        parent = source.get("source_video_sha256")
        if not isinstance(parent, str) or len(parent) != 64:
            raise ValueError("original course-recording hash is missing")
        if source["source_filename"] in parent_hashes and parent_hashes[source["source_filename"]] != parent:
            raise ValueError("one original recording has conflicting source hashes")
        parent_hashes[source["source_filename"]] = parent
        clip_ids.add(clip_id)
        clip_hashes.add(actual_hash)
        relative_path = Path(os.path.relpath(video, output_dir.resolve())).as_posix()
        rows.append({"clip_id": clip_id, "match_id": source.get("match_id"),
                     "event_id": source.get("event_id"), "kicker": source.get("kicker"),
                     "team": source.get("team"), "penalty_number": source.get("penalty_number"),
                     "clip_filename": filename, "video_url": quote(relative_path, safe="/"),
                     "source_sha256": actual_hash, "clip_bytes": video.stat().st_size,
                     "fps": float(fps), "frame_count": frame_count,
                     "source_filename": source["source_filename"], "source_video_sha256": parent,
                     "source_start_frame": start, "source_end_frame_exclusive": end,
                     "frame_clock": "clip_local_zero_based"})
    if set(reference_ids) != clip_ids:
        raise ValueError("reference and source cohorts differ")
    if expected_events == 39 and set(parent_hashes) != COURSE_SOURCES:
        raise ValueError("39-event cohort must include all four original course videos")
    return sorted(rows, key=lambda row: (str(row["match_id"]), row["penalty_number"], row["clip_id"]))


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Original footage | IND5003</title>
<style>
:root{font-family:Arial,Helvetica,sans-serif;color:#202725;background:#f5f7f7;letter-spacing:0}
*{box-sizing:border-box}body{margin:0}button,input,select{font:inherit}a{color:#23649a;text-decoration:none}a:hover{text-decoration:underline}
header{background:#fff;border-bottom:1px solid #d7dfde;padding:18px 24px;display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap}
h1{font-size:21px;margin:0;font-weight:600}header p{font-size:13px;color:#626d69;margin:6px 0 0}nav{display:flex;gap:20px;font-size:13px;flex-wrap:wrap}
main{max-width:1160px;margin:auto;padding:20px 24px 30px}.selection{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:16px;align-items:end;margin-bottom:16px}
label,.label{display:block;font-size:12px;color:#626d69;margin-bottom:6px}select,input{color:#202725;background:white;border:1px solid #b9c7c2;border-radius:4px;min-height:38px;padding:7px 10px}select{width:100%;min-width:0}
.state{color:#2f7353;background:#edf5ef;border:1px solid #cddfd2;border-radius:4px;font-size:12px;padding:11px 14px;white-space:nowrap}
.video-surface{position:relative;background:#171a19;aspect-ratio:16/9;display:flex;align-items:center;justify-content:center;overflow:hidden}video{display:block;width:100%;height:100%;object-fit:contain;background:#171a19}
.missing{position:absolute;inset:0;background:#171a19;color:#fff;padding:24px;display:flex;justify-content:center;align-items:center;text-align:center;flex-direction:column;gap:10px}.missing[hidden]{display:none}.missing strong{font-size:18px}.missing span{color:#c1cac6;font-size:13px;overflow-wrap:anywhere}
.controls{padding:14px 0;display:flex;gap:14px;align-items:end;flex-wrap:wrap;border-bottom:1px solid #d7dfde}.frame-control{width:150px}.frame-control input{width:100%}.rate-control{width:150px}.rate-control select{width:100%}.times{display:flex;gap:28px;flex:1;justify-content:flex-end;font-variant-numeric:tabular-nums;flex-wrap:wrap}.times .value{font-size:15px;color:#263b33;min-width:92px}
.metadata{display:grid;grid-template-columns:1fr 1fr;gap:18px;padding:20px 0;font-size:13px;line-height:1.5}.metadata .value{overflow-wrap:anywhere}.hash{font-family:monospace;font-size:12px}.metadata .wide{grid-column:1/-1}
@media(max-width:640px){header{padding:16px}h1{font-size:19px}main{padding:16px}.selection{grid-template-columns:1fr}.state{justify-self:start;padding:7px 10px}.controls{gap:12px}.frame-control,.rate-control{width:calc(50% - 6px)}.times{justify-content:flex-start;gap:18px;flex-basis:100%;padding-top:4px}.metadata{grid-template-columns:1fr;gap:14px}.metadata .wide{grid-column:auto}nav{gap:16px}}
</style>
</head>
<body>
<header><div><h1>Original footage</h1><p>IND5003 · Development cohort · __EVENT_COUNT__ events</p></div><nav><a href="human_reference_template.json" download>Reference JSON</a><a href="source_receipt.json" download>Source receipt</a></nav></header>
<main>
<div class="selection"><div><label for="clipSelect">Penalty</label><select id="clipSelect" aria-label="Penalty"></select></div><span class="state">Review pending</span></div>
<div class="video-surface"><video id="video" controls preload="metadata" playsinline></video><div id="missing" class="missing" hidden><strong>Original clip unavailable</strong><span id="missingFile"></span></div></div>
<div class="controls"><div class="frame-control"><label for="frameInput">Nominal clip frame</label><input id="frameInput" type="number" min="0" step="1" value="0"></div><div class="rate-control"><label for="rateSelect">Playback rate</label><select id="rateSelect"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option></select></div><div class="times"><div><span class="label">Clip time</span><span id="clipTime" class="value">0.000 s</span></div><div><span class="label">Source time</span><span id="sourceTime" class="value">0.000 s</span></div><div><span class="label">Nominal source frame</span><span id="sourceFrame" class="value">0</span></div></div></div>
<div class="metadata"><div><span class="label">Clip</span><div id="clipName" class="value"></div></div><div><span class="label">Original recording</span><div id="sourceName" class="value"></div></div><div><span class="label">Frame rate</span><div id="fps" class="value"></div></div><div><span class="label">Clip frame interval</span><div id="interval" class="value"></div></div><div class="wide"><span class="label">Verified clip SHA-256</span><div id="hash" class="value hash"></div></div></div>
</main>
<script>
const clips=__SOURCE_DATA__;
const video=document.getElementById('video'),clipSelect=document.getElementById('clipSelect'),frameInput=document.getElementById('frameInput'),rateSelect=document.getElementById('rateSelect'),missing=document.getElementById('missing');
let current=clips[0],lastMediaTime=0;
for(const [index,clip] of clips.entries()){const option=document.createElement('option');option.value=index;option.textContent=clip.clip_id;clipSelect.appendChild(option);}
function updateClock(mediaTime){lastMediaTime=Number.isFinite(mediaTime)?mediaTime:0;const frame=Math.max(0,Math.min(current.frame_count-1,Math.round(lastMediaTime*current.fps)));if(document.activeElement!==frameInput)frameInput.value=frame;document.getElementById('clipTime').textContent=lastMediaTime.toFixed(3)+' s';document.getElementById('sourceTime').textContent=(current.source_start_frame/current.fps+lastMediaTime).toFixed(3)+' s';document.getElementById('sourceFrame').textContent=current.source_start_frame+frame;}
function selectClip(index){video.pause();current=clips[index];lastMediaTime=0;missing.hidden=true;frameInput.value=0;frameInput.max=current.frame_count-1;document.getElementById('clipName').textContent=current.clip_filename;document.getElementById('sourceName').textContent=current.source_filename;document.getElementById('fps').textContent=current.fps+' fps';document.getElementById('interval').textContent='0–'+(current.frame_count-1);document.getElementById('hash').textContent=current.source_sha256;document.getElementById('missingFile').textContent=current.clip_filename;video.src=current.video_url;video.load();updateClock(0);}
clipSelect.addEventListener('change',()=>selectClip(Number(clipSelect.value)));
rateSelect.addEventListener('change',()=>video.playbackRate=Number(rateSelect.value));
frameInput.addEventListener('change',()=>{const requested=Number(frameInput.value);if(!Number.isFinite(requested))return;const frame=Math.max(0,Math.min(current.frame_count-1,Math.round(requested)));frameInput.value=frame;video.pause();video.currentTime=Math.min((frame+0.25)/current.fps,Math.max(0,video.duration-0.001));});
video.addEventListener('loadedmetadata',()=>{missing.hidden=true;video.playbackRate=Number(rateSelect.value);updateClock(0);});
video.addEventListener('error',()=>{missing.hidden=false;});
video.addEventListener('seeked',()=>updateClock(video.currentTime));
video.addEventListener('timeupdate',()=>{if(!video.requestVideoFrameCallback)updateClock(video.currentTime);});
if(video.requestVideoFrameCallback){const tick=(now,metadata)=>{updateClock(metadata.mediaTime);video.requestVideoFrameCallback(tick);};video.requestVideoFrameCallback(tick);}
selectClip(0);
</script>
</body></html>
'''


def build_viewer(freeze: Path, clips_dir: Path, reference_template: Path, output_dir: Path, expected_events=39):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("use a new blind review directory")
    rows = source_records(freeze, clips_dir, reference_template, output_dir, expected_events)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rows, ensure_ascii=True).replace("</", "<\\/")
    html = HTML.replace("__EVENT_COUNT__", str(len(rows))).replace("__SOURCE_DATA__", payload)
    (output_dir / "index.html").write_text(html, encoding="utf-8")
    shutil.copyfile(reference_template, output_dir / "human_reference_template.json")
    receipt = {"schema_version": "shoe-blind-original-review-v1",
               "created_at": datetime.now(timezone.utc).isoformat(),
               "builder_sha256": sha256(Path(__file__)), "prediction_freeze_sha256": sha256(freeze),
               "reference_template_sha256": sha256(reference_template),
               "html_sha256": sha256(output_dir / "index.html"), "events": len(rows),
               "original_course_videos_only": True, "additional_video_used": False,
               "predicted_labels_displayed": False, "predicted_counts_displayed": False,
               "candidate_frames_displayed": False, "overlays_displayed": False,
               "human_reviewed_events": 0, "held_out_test": False, "development_cohort": True,
               "media_copied_or_uploaded": False,
               "frame_display_scope": "Nominal clip/source frame calculated from browser media time and clip FPS; not a decoded-frame export.",
               "sources": rows}
    (output_dir / "source_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--clips-dir", type=Path, default=ROOT / "data/course_context_clips_2026-10-12")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.analysis_dir / "blind_review"
    result = build_viewer(args.analysis_dir / "prediction_freeze.json", args.clips_dir,
                          args.analysis_dir / "human_reference_template.json", output_dir)
    print(json.dumps({"output": str(output_dir / "index.html"), "events": result["events"],
                      "predicted_labels_displayed": False, "human_reviewed_events": 0}, indent=2))


if __name__ == "__main__":
    main()

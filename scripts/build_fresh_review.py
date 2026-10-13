"""Build a portable, offline viewer from the frozen 39-event evidence bundle."""
import argparse
import csv
import html
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.final_results import sha256
from src.fresh_eda import load_bundle
from src.fresh_results import read_json, write_json


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IND5003 | Penalty evidence</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f5f6;color:#202124;font:14px Arial,sans-serif;letter-spacing:0}
header{padding:17px 22px;background:#fff;border-bottom:1px solid #d8dcdf;display:flex;align-items:center;justify-content:space-between;gap:12px}
h1{font-size:19px;margin:0}header span{font-size:12px;color:#656a70}.layout{display:grid;grid-template-columns:280px minmax(0,1fr);min-height:calc(100vh - 59px)}
aside{border-right:1px solid #d8dcdf;background:#fff;min-width:0;padding:16px 12px}.filters{display:grid;gap:10px}select,input{font:inherit;min-width:0;width:100%;padding:8px;background:#fff;border:1px solid #c5c9ce;border-radius:4px}
label{display:grid;gap:5px;font-size:12px;color:#656a70}#events{margin-top:15px;display:grid;gap:1px;max-height:calc(100vh - 233px);overflow:auto}
.event{border:0;border-bottom:1px solid #e8eaec;border-radius:0;padding:11px 9px;display:flex;justify-content:space-between;text-align:left;font:inherit;background:white;color:#202124;cursor:pointer;min-height:42px}
.event[aria-selected=true]{background:#e6f3ef;color:#145447;border-left:3px solid #237867}.event small{color:#70767c}
main{min-width:0;padding:19px 24px}.heading{display:flex;align-items:start;justify-content:space-between;gap:12px;margin-bottom:14px}h2{font-size:20px;margin:0 0 5px}.muted{font-size:12px;color:#687078;overflow-wrap:anywhere}
.tabs{display:flex;flex-wrap:wrap;gap:16px;border-bottom:1px solid #ccd1d5;margin-bottom:16px}.tabs button{border:0;border-bottom:3px solid transparent;background:transparent;padding:9px 3px;font:inherit;cursor:pointer}.tabs button[aria-selected=true]{border-bottom-color:#237867;color:#145447}
.content{display:grid;grid-template-columns:minmax(0,1fr) 250px;gap:22px}.media{position:relative;background:#17191c;aspect-ratio:16/9;overflow:hidden;border-radius:4px}.media video{width:100%;height:100%;object-fit:contain}.media canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.details{display:grid;gap:0;align-content:start}.metric{border-bottom:1px solid #d8dcdf;padding:9px 0;display:flex;justify-content:space-between;align-items:start;gap:12px}.metric span{color:#666d74;font-size:12px;flex:1;min-width:0}.metric strong{font-size:13px;font-weight:500;text-align:right;flex-shrink:0;max-width:50%;overflow-wrap:anywhere}
.links{display:flex;gap:18px;flex-wrap:wrap;margin:14px 0 0}a{color:#176b5e;font-size:12px}.image{width:100%;height:auto;display:block;border:1px solid #d8dcdf;background:white}.image[hidden]{display:none}.panel{display:none}.panel.active{display:block}.caption{padding:9px 0;color:#687078;font-size:12px}.clock{display:flex;gap:18px;margin-top:10px;font-variant-numeric:tabular-nums;font-size:12px;color:#687078}details{margin-top:16px}summary{cursor:pointer;color:#52616a;font-size:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px;line-height:1.5;background:#fff;padding:12px;border:1px solid #d8dcdf;max-height:340px;overflow:auto}
@media(max-width:1050px){.layout{grid-template-columns:225px minmax(0,1fr)}.content{grid-template-columns:minmax(0,1fr)}.details{grid-template-columns:repeat(2,minmax(0,1fr));column-gap:20px}}
@media(max-width:650px){header{padding:14px;align-items:start}header span{max-width:100px;text-align:right}.layout{display:block}aside{border-right:0;border-bottom:1px solid #d8dcdf;padding:12px}.filters{grid-template-columns:repeat(2,minmax(0,1fr))}#events{max-height:135px;margin-top:10px}.event{min-height:39px}main{padding:16px 12px}.content{gap:12px}h2{font-size:18px}.details{column-gap:16px}.metric strong{max-width:110px}.tabs{gap:20px}}
</style></head><body>
<header><h1>IND5003 / Penalty Evidence</h1><span>39 attempts<br>Automatic candidates</span></header>
<div class="layout"><aside><div class="filters"><label>Match<select id="match"><option value="">All matches</option><option>col_eng</option><option>cro_den</option><option>esp_rus</option><option>rus_cro</option></select></label><label>Player<input id="search" type="search" autocomplete="off"></label></div><div id="events" role="listbox" aria-label="Penalty attempts"></div></aside>
<main><div class="heading"><div><h2 id="title"></h2><div class="muted" id="identity"></div></div><span class="muted" id="ordinal"></span></div>
<div class="tabs" role="tablist"><button role="tab" data-tab="video" aria-selected="true">Source Video</button><button role="tab" data-tab="ground" aria-selected="false">Ground Projection</button><button role="tab" data-tab="evidence" aria-selected="false">Frame Evidence</button><button role="tab" data-tab="shoe" aria-selected="false">Foot Evidence</button></div>
<div class="content"><div><div id="video-panel" class="panel active"><div class="media"><video id="video" controls preload="metadata" playsinline></video><canvas id="overlay" width="1280" height="720"></canvas></div><div class="clock"><span id="frame"></span><span id="source-clock"></span></div></div>
<div id="ground-panel" class="panel"><img id="ground" class="image" alt="Same-frame ground contact candidates"><div class="caption" id="ground-caption"></div></div>
<div id="evidence-panel" class="panel"><img id="evidence" class="image" alt="Source-frame detector evidence"><div class="caption" id="evidence-caption"></div></div>
<div id="shoe-panel" class="panel"><img id="shoe" class="image" alt="Original pixels, normal shoe landmarks and inverse-mirror shoe landmarks"><div class="caption" id="shoe-caption"></div></div>
<div class="links"><a href="__BUNDLE__/features_39.csv" download>Feature Table</a><a href="__BUNDLE__/features_39.json" download>Structured Evidence</a><a href="__BUNDLE__/summary.json">Cohort Summary</a></div>
<details><summary>Source and Component Records</summary><pre id="record"></pre></details></div><div class="details" id="metrics"></div></div></main></div>
<script type="application/json" id="data">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent), video=document.getElementById('video'), canvas=document.getElementById('overlay'),ctx=canvas.getContext('2d');let selected;
const text=(id,value)=>document.getElementById(id).textContent=value;
const value=x=>x===null||x===undefined?'Unknown':String(x).replaceAll('_',' ');
function choose(id){selected=data.find(row=>row.clip_id===id);const r=selected.features;text('title',selected.name);text('identity',r.match_id+' / '+r.team+' / '+r.clip_id);text('ordinal',String(r.event_id).padStart(2,'0')+' / 39');
video.src=selected.video;video.onloadedmetadata=()=>{video.currentTime=r.contact_frame_candidate===null?0:Math.max(0,r.contact_frame_candidate/r.source_fps-.7)};
const metrics=[['Contact frame',r.contact_frame_candidate],['Contact actor',r.contact_actor_candidate],['Striking foot',r.kicking_foot_candidate],['Approach steps',r.steps_phase_candidate],['Sensitivity range',r.steps_sensitivity_min===null?null:r.steps_sensitivity_min+' - '+r.steps_sensitivity_max],['Keeper movement',r.keeper_pre_movement_candidate],['Keeper direction (image)',r.keeper_direction_image_candidate],['Height (strict ordinal)',r.keeper_strict_height_candidate],['Ball observations',r.ball_trajectory_observations],['Ball rule',r.ball_rule_outcome],['Endpoint visibility',r.ball_endpoint_status],['Ground pre-contact',r.ground_pre_contact_points],['Camera intervals',r.camera_intervals],['External outcome reference',r.external_outcome_reference]];
const target=document.getElementById('metrics');target.replaceChildren();for(const [label,v]of metrics){const cell=document.createElement('div');cell.className='metric';const name=document.createElement('span'), val=document.createElement('strong');name.textContent=label;val.textContent=value(v);cell.append(name,val);target.append(cell)}
const ground=document.getElementById('ground');ground.hidden=!selected.ground;ground.src=selected.ground||'';text('ground-caption',selected.ground?'Sparse source-frame sole candidates; gaps retained.':'No supported ground-contact observations.');
const evidence=document.getElementById('evidence');evidence.hidden=!selected.evidence;evidence.src=selected.evidence||'';text('evidence-caption',selected.evidence?'Source observations and automatic detection candidates.':'No contact-bound detector sheet.');
const shoe=document.getElementById('shoe');shoe.hidden=!selected.shoe;shoe.src=selected.shoe||'';text('shoe-caption',selected.shoe?'Paired shoe-region model candidate; physical contact and anatomical accuracy remain unverified.':'No additional paired shoe-region candidate.');text('record',JSON.stringify(selected.record,null,2));renderList();draw()}
function renderList(){const target=document.getElementById('events'),match=document.getElementById('match').value,search=document.getElementById('search').value.toLowerCase();target.replaceChildren();for(const row of data.filter(r=>(!match||r.features.match_id===match)&&r.name.toLowerCase().includes(search))){const b=document.createElement('button');b.className='event';b.type='button';b.setAttribute('role','option');b.setAttribute('aria-selected',String(row===selected));const name=document.createElement('span'), n=document.createElement('small');name.textContent=row.name;n.textContent=String(row.features.event_id).padStart(2,'0');b.append(name,n);b.onclick=()=>choose(row.clip_id);target.append(b)}}
function draw(){ctx.clearRect(0,0,1280,720);if(!selected)return;const frame=Math.floor(video.currentTime*selected.features.source_fps);text('frame','Clip frame '+frame);text('source-clock','Original '+((selected.source_start_frame+frame)/selected.features.source_fps).toFixed(3)+' s');
const ball=selected.ball.find(p=>p.frame===frame);if(ball){ctx.beginPath();ctx.arc(ball.x,ball.y,8,0,2*Math.PI);ctx.strokeStyle='#ffcf35';ctx.lineWidth=3;ctx.stroke()}const actor=selected.actor.find(p=>p.frame===frame);if(actor){ctx.strokeStyle='#52e1b9';ctx.lineWidth=2;const[x0,y0,x1,y1]=actor.bbox;ctx.strokeRect(x0,y0,x1-x0,y1-y0)}}
video.addEventListener('timeupdate',draw);video.addEventListener('seeked',draw);if(video.requestVideoFrameCallback){function tick(){draw();video.requestVideoFrameCallback(tick)}video.requestVideoFrameCallback(tick)}
document.getElementById('match').onchange=renderList;document.getElementById('search').oninput=renderList;for(const tab of document.querySelectorAll('[data-tab]'))tab.onclick=()=>{for(const b of document.querySelectorAll('[data-tab]'))b.setAttribute('aria-selected',String(b===tab));for(const panel of document.querySelectorAll('.panel'))panel.classList.toggle('active',panel.id===tab.dataset.tab+'-panel')};choose(data[0].clip_id);
</script></body></html>'''


def build(bundle, clips, output, include_video=False, shoe_review=None):
    rows, structured, _ = load_bundle(bundle)
    shoe_records = {}
    if shoe_review is not None:
        for record in read_json(shoe_review / "receipt.json")["records"]:
            if record["clip_id"] in shoe_records:
                raise ValueError("Duplicate shoe-review event")
            shoe_records[record["clip_id"]] = record
    if output.exists():
        raise FileExistsError("Use a new review directory to retain the evidence version")
    output.mkdir(parents=True)
    artifacts, data = {}, []
    for row, event in zip(rows, structured["events"], strict=True):
        cid, source = row["clip_id"], event["source_mapping"]
        video = clips / source["clip_filename"]
        if sha256(video) != source["clip_sha256"]:
            raise ValueError("Review source clip hash mismatch")
        if include_video:
            target = output / "video" / video.name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(video, target)
            if sha256(target) != source["clip_sha256"]:
                raise ValueError("Packaged review video bytes changed")
            artifacts[str(target.relative_to(output))] = sha256(target)
            video = target
        bindings = event["artifact_bindings"]
        pose_path = Path(bindings["pose"]["path"])
        if sha256(pose_path) != bindings["pose"]["sha256"]:
            raise ValueError("Review pose hash mismatch")
        pose = read_json(pose_path)
        actor_id = row["contact_actor_candidate"] if row["contact_actor_candidate"] is not None else row["pose_actor_baseline"]
        actor = next((track["observations"] for track in pose.get("tracks", []) if track["track_id"] == actor_id), [])
        entry = {"clip_id": cid, "name": source["kicker"].replace("_", " ").title(), "features": row,
                 "video": os.path.relpath(video, output), "source_start_frame": source["source_start_frame"],
                 "actor": [{"frame": p["frame"], "bbox": p["bbox"]} for p in actor],
                 "ball": event["components"]["ball"].get("trajectory", []),
                 "record": {"source": source, "component_bindings": bindings},
                 "ground": None, "evidence": None, "shoe": None}
        if cid in shoe_records:
            record = shoe_records[cid]
            image = shoe_review / f"{cid}.png"
            if (record["source_sha256"] != source["clip_sha256"]
                    or record["image"] != image.name or sha256(image) != record["image_sha256"]):
                raise ValueError("Shoe-review source or evidence image hash mismatch")
            target = output / "evidence" / f"{cid}_shoe.png"
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(image, target)
            artifacts[str(target.relative_to(output))] = sha256(target)
            entry["shoe"] = str(target.relative_to(output))
            entry["record"]["shoe_review"] = record
        for stage, kind in (("pitch", "ground"), ("ball", "evidence")):
            if stage not in bindings:
                continue
            path = Path(bindings[stage]["path"])
            if sha256(path) != bindings[stage]["sha256"]:
                raise ValueError("Review component hash mismatch")
            result = read_json(path)
            if stage == "pitch":
                record = result.get("runup_plot") or {}
                image = path.parent / record.get("path", "missing")
                if image.is_file() and sha256(image) != record.get("sha256"):
                    raise ValueError("Ground evidence image hash mismatch")
            else:
                image = path.parent.parent / "media" / f"{cid}.jpg"
            if image.is_file():
                target = output / "evidence" / f"{cid}_{kind}{image.suffix}"
                target.parent.mkdir(exist_ok=True)
                shutil.copyfile(image, target)
                artifacts[str(target.relative_to(output))] = sha256(target)
                entry[kind] = str(target.relative_to(output))
        data.append(entry)
    payload = json.dumps(data, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    bundle_link = html.escape(os.path.relpath(bundle, output), quote=True)
    (output / "index.html").write_text(PAGE.replace("__DATA__", payload).replace("__BUNDLE__", bundle_link), encoding="utf-8")
    artifacts["index.html"] = sha256(output / "index.html")
    write_json(output / "manifest.json", {"events": 39, "bundle_manifest_sha256": sha256(bundle / "manifest.json"),
        "artifacts": artifacts, "includes_source_video": include_video, "human_verified": False,
        "builder_sha256": sha256(Path(__file__))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-video", action="store_true")
    parser.add_argument("--shoe-review-dir", type=Path)
    args = parser.parse_args()
    build(args.bundle_dir, args.clips_dir, args.output_dir, args.include_video, args.shoe_review_dir)
    print(args.output_dir / "index.html")


if __name__ == "__main__":
    main()

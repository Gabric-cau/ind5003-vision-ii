"""Complete canonical shot intervals and explicitly unverified view candidates.

No frame is discarded merely because the segment is short. Slow-motion and
anatomical camera-side labels require evidence absent from grass/edge features.
"""
from pathlib import Path
from collections import Counter
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from src.step_pipeline import _scan_camera_shots
from src.camera_angles import CameraAngleClassifier
from src.goalkeeper_pipeline import _manifest_metadata
cv2.setNumThreads(2)
OUT=ROOT/"outputs/optimization_evening_2026-10-09/camera_canonical"
OUT.mkdir(parents=True,exist_ok=True)
classifier=CameraAngleClassifier();records=[];rows=[]
for cid in _manifest_metadata(ROOT/"configs/penalty_slicing_manifest.yaml"):
    video=ROOT/"data/raw_clips"/(cid+".mp4")
    dest=OUT/(cid+".json")
    if dest.exists():raise FileExistsError(dest)
    shots=_scan_camera_shots(str(video),3600,30)
    alternate=_scan_camera_shots(str(video),3600,45)
    cap=cv2.VideoCapture(str(video));segments=[]
    for shot in shots:
        start,end=shot["shot_start_frame"],shot["camera_shot_end_frame"]
        fids=sorted(set([start,(start+end-1)//2,end-1]));frames=[]
        for fid in fids:
            cap.set(1,fid);ok,f=cap.read()
            if ok:frames.append(f)
        features=[classifier.extract_shot_features(f) for f in frames]
        votes=[classifier.classify_shot([f]) for f in frames]
        label=classifier.classify_shot(frames)
        segment={"clip_id":cid,"start_frame":start,"end_frame_exclusive":end,
                 "start_time_sec":start/shot["fps"],"end_time_sec":end/shot["fps"],
                 "duration_sec":(end-start)/shot["fps"],"legacy_camera_candidate":label,
                 "sample_frame_ids":fids,"sample_votes":votes,"sample_features":features,
                 "candidate_vote_consistent":bool(votes and len(set(votes))==1),
                 "verified_camera_class":"unknown","slow_motion_status":"unknown",
                 "reason":"Grass and line angle do not establish playback speed or exact camera side",
                 "human_verified":False,"accuracy_validated":False}
        segments.append(segment);records.append(segment)
    cap.release()
    assert segments[0]["start_frame"]==0 and segments[-1]["end_frame_exclusive"]==shots[0]["source_frame_count"]
    assert all(a["end_frame_exclusive"]==b["start_frame"] for a,b in zip(segments,segments[1:]))
    boundaries=[s["start_frame"] for s in segments[1:]]
    alt=[s["shot_start_frame"] for s in alternate[1:]]
    stable=sum(any(abs(b-a)<=3 for a in alt) for b in boundaries)
    result={"clip_id":cid,"segments":segments,"fps":shots[0]["fps"],"source_frames":shots[0]["source_frame_count"],
            "timeline_coverage":1.0,"alternative_threshold_boundaries":alt,
            "boundary_sensitivity":{"default_cut_count":len(boundaries),"alternate_cut_count":len(alt),"matched_within_3_frames":stable},
            "human_verified":False,"accuracy_validated":False}
    dest.write_text(json.dumps(result,indent=2),encoding="utf-8")
    rows.append({"clip_id":cid,"segments":len(segments),"total_frames":result["source_frames"],"coverage":1.0,**result["boundary_sensitivity"]})
    print(json.dumps(rows[-1]),flush=True)
(OUT/"camera_timings_candidates.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
(OUT/"summary.json").write_text(json.dumps({"clips":len(rows),"segments":len(records),"rows":rows,
    "legacy_candidate_classes":dict(Counter(r["legacy_camera_candidate"] for r in records)),
    "verified_camera_classes":0,"canonical_unique":len({r["clip_id"] for r in rows})==39,
    "interpretation":"100% frame-interval coverage is not shot-boundary or camera-class accuracy",
    "human_verified":False,"accuracy_validated":False,
    "source_sha256":{n:hashlib.sha256((ROOT/"src"/n).read_bytes()).hexdigest() for n in ("step_pipeline.py","camera_angles.py")}},indent=2),encoding="utf-8")

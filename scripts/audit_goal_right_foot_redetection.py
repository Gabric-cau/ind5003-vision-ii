"""Hash/link and deterministic replay checks; diagnostic parameter sensitivity."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlparse
import json,sys,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.kicking_foot_redetection import transform_points,approach_candidates
RUN=ROOT/'outputs/goal_right_foot_redetection_2026-10-11/run_v2'
GALLERY=ROOT/'outputs/goal_right_camera_review_2026-10-11'


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        self.links.extend(v for k,v in attrs if k in ('href','src','poster'))


def main():
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    rows=json.loads((RUN/'summary.json').read_text(encoding='utf-8'))['rows']
    sensitivity=[]
    for r in rows:
        cid=r['clip_id'];assert sha(Path(r['source_video']))==r['source_video_sha256']
        for p,v in r['input_sha256'].items():assert sha(Path(p))==v
        raw=json.loads((RUN/'poses'/f'{cid}.json').read_text(encoding='utf-8'))
        cam=json.loads((ROOT/'outputs/ball_detection_all_2026-10-11/detections'/f'{cid}.json').read_text(encoding='utf-8'))['camera']['transforms']
        samples=[dict(frame=p['frame'],keypoints=transform_points(p['keypoints'],cam[str(p['frame'])]),confidence=p['confidence']) for p in raw if not p['reflected'] and str(p['frame']) in cam]
        base=approach_candidates(samples,r['stationary_ball_reference'],r['goal_ground_midpoint_uncalibrated'],r['contact_bracket_frames'][0])
        assert base==r['approach_heuristics']
        variants=[]
        for delta in (-2,0,2):
            a=approach_candidates(samples,r['stationary_ball_reference'],r['goal_ground_midpoint_uncalibrated'],r['contact_bracket_frames'][0]+delta)
            variants.append(dict(contact_shift_frames=delta,screen=a['screen_x_candidate'],axis=a['goal_axis_candidate']))
        geometry_variants=[]
        goal=r['goal_ground_midpoint_uncalibrated']
        if goal is not None:
            for dx,dy in [(0,0),(-20,0),(20,0),(0,-20),(0,20)]:
                a=approach_candidates(samples,r['stationary_ball_reference'],np.asarray(goal)+[dx,dy],r['contact_bracket_frames'][0])
                geometry_variants.append(dict(goal_offset_px=[dx,dy],axis=a['goal_axis_candidate']))
        sensitivity.append(dict(clip_id=cid,contact_variants=variants,goal_variants=geometry_variants,
            contact_axis_labels=list(sorted({v['axis'] for v in variants})),goal_axis_labels=list(sorted({v['axis'] for v in geometry_variants}))))
    audits=[]
    for base in [RUN,GALLERY]:
        parser=Links();parser.feed((base/'index.html').read_text(encoding='utf-8'));missing=[]
        for link in parser.links:
            u=urlparse(link)
            if u.scheme or not u.path:continue
            p=(base/unquote(u.path)).resolve()
            if not p.is_file():missing.append(link)
            if base==GALLERY:assert p.is_relative_to(GALLERY)
        assert not missing,missing
        audits.append(dict(file=str(base/'index.html'),links_checked=len(parser.links),missing=missing))
    protocol=json.loads((RUN/'protocol.json').read_text())
    snapshots={}
    for p,v in protocol['source_sha256'].items():
        snap=RUN/'source_snapshot'/Path(p).name;assert sha(snap)==v;snapshots[p]=v
    result=dict(source_clips_hash_checked=len(rows),input_hashes_checked=sum(len(r['input_sha256']) for r in rows),
        deterministic_approach_replays=len(rows),html_link_audits=audits,inference_source_snapshots_verified=snapshots,
        changed_contact_axis_labels=[s['clip_id'] for s in sensitivity if len(s['contact_axis_labels'])>1],
        changed_goal_axis_labels=[s['clip_id'] for s in sensitivity if len(s['goal_axis_labels'])>1],
        accuracy_estimate=None,sensitivity_is_not_confidence_interval=True)
    (RUN/'parameter_sensitivity.json').write_text(json.dumps(sensitivity,indent=2),encoding='utf-8')
    (RUN/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()

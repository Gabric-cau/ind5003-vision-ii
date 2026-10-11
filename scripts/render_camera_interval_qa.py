"""Render start/middle/end samples for every candidate; NOT human labels."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_camera_partial_pose_experiment import DEFAULT,read,save


def main():
    source=DEFAULT/'dense_views_v1';out=DEFAULT/'visual_qa';out.mkdir(exist_ok=True)
    rows=read(source/'summary.json')['intervals'];pages=[]
    for offset in range(0,len(rows),6):
        canvas=np.full((6*290,1440,3),245,np.uint8)
        for ri,r in enumerate(rows[offset:offset+6]):
            cid=r['clip_id'];cap=cv2.VideoCapture(str(ROOT/'data/raw_clips'/f'{cid}.mp4'))
            positions=[r['support_frames'][0],r['support_frames'][len(r['support_frames'])//2],r['support_frames'][-1]]
            clip=read(source/'clips'/f'{cid}.json')
            seg=next(s for s in clip['segments'] if s['segment_id']==r['segment_id'])
            samples={s['frame']:s for s in seg['samples']}
            label=f"#{offset+ri:02} {cid} / s{r['segment_id']} / {r['view_candidate']} / old={r['old_whole_shot_view']}"
            cv2.putText(canvas,label,(12,ri*290+22),cv2.FONT_HERSHEY_SIMPLEX,.55,(10,10,10),1,cv2.LINE_AA)
            for ci,fid in enumerate(positions):
                cap.set(1,fid);ok,frame=cap.read()
                if not ok:raise RuntimeError(f'Decode {cid} {fid}')
                s=samples[fid]
                for person in s['people']:
                    role=person.get('role','unknown')
                    if role not in ('kicker_candidate','keeper_candidate'):continue
                    color=(0,210,255) if role=='kicker_candidate' else (255,150,0)
                    x0,y0,x1,y1=map(int,person['bbox'])
                    cv2.rectangle(frame,(x0,y0),(x1,y1),color,3)
                if s.get('goal'):
                    cv2.polylines(frame,[np.array(s['goal']['corners'],np.int32)],True,(0,255,0),3)
                thumb=cv2.resize(frame,(480,240))
                cv2.putText(thumb,f'f{fid} {fid/30:.2f}s',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
                canvas[ri*290+35:ri*290+275,ci*480:(ci+1)*480]=thumb
            cap.release()
        name=f'intervals_{offset:02}_{min(offset+5,len(rows)-1):02}.jpg'
        cv2.imencode('.jpg',canvas,[cv2.IMWRITE_JPEG_QUALITY,90])[1].tofile(str(out/name));pages.append(name)
    save(out/'index.json',{'images':pages,'sampled_intervals':len(rows),'frames_per_interval':3,
                        'reviewer':'not yet reviewed','human_ground_truth':False,
                        'warning':'Static samples cannot verify speed, motion continuity, or exact boundaries.'})
    print(json.dumps(pages))

if __name__=='__main__':main()

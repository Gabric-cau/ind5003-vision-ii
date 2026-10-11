"""Decode identity evidence around close goalkeeper/referee interactions."""
from pathlib import Path
import argparse,sys,json
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_goalkeeper_direction_update import read,RUN,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--clip',default='col_eng_p03_cuadrado_col11');a=p.parse_args()
    r=read(RUN/'goal_appearance_v2'/f'{a.clip}.json');pose=read(r['source_pose_path'])
    kick=r['kick_frame'];frames=[kick-10,kick+9,kick+18,kick+30]
    cap=cv2.VideoCapture(r['video_path']);tiles=[];records=[]
    bounds=r['goal_identity_evidence']['goal']['selected']['bbox'];gh=bounds[3]-bounds[1]
    x0=max(0,int(bounds[0]-100));x1=min(pose['frame_width'],int(bounds[2]+180))
    y0=max(0,int(bounds[1]-90));y1=min(pose['frame_height'],int(bounds[3]+150))
    for f in frames:
        cap.set(1,f);ok,img=cap.read()
        if not ok:raise ValueError(f)
        ids=[]
        for t in pose['tracks']:
            for o in t['observations']:
                if o['frame']!=f:continue
                b=o['bbox'];cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2
                if not x0<cx<x1 or not y0<cy<y1 or b[3]-b[1]<.2*gh:continue
                ids.append({'track_id':t['track_id'],'bbox':b})
                cv2.rectangle(img,tuple(map(round,b[:2])),tuple(map(round,b[2:])),(0,220,255),1)
                cv2.putText(img,str(t['track_id']),tuple(map(round,b[:2])),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1)
                for j in [9,10,15,16]:
                    if o['keypoint_confidences'][j]>=.5:cv2.circle(img,tuple(map(round,o['keypoints'][j])),3,(255,0,255),-1)
        crop=img[y0:y1,x0:x1];crop=cv2.resize(crop,(650,450))
        cv2.putText(crop,f'f{f} {f/pose["fps"]:.2f}s',(8,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
        tiles.append(crop);records.append(dict(frame=f,candidates=ids))
    cap.release();out=ROOT/'outputs/goalkeeper_role_association_2026-10-11';out.mkdir(exist_ok=True)
    cv2.imwrite(str(out/f'{a.clip}_source_tracks.jpg'),np.vstack([np.hstack(tiles[:2]),np.hstack(tiles[2:])]))
    save(out/f'{a.clip}_source_tracks.json',records);print(json.dumps(records,indent=2))


if __name__=='__main__':main()

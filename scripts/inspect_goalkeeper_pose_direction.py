"""Inspect available goalkeeper pose evidence around the actual dive."""
from pathlib import Path
import sys,json,argparse
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_goalkeeper_direction_update import read,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--clip',default='col_eng_p02_kane_eng9');a=p.parse_args()
    r=read(ROOT/'outputs/optimization_evening_2026-10-09/goal_appearance_v2'/f'{a.clip}.json')
    pose=read(r['source_pose_path']);ids=set(r['selected_candidate']['source_track_ids'])
    obs=[o for t in pose['tracks'] if t['track_id'] in ids for o in t['observations']]
    obs={o['frame']:o for o in obs};kick=r['kick_frame'];cap=cv2.VideoCapture(r['video_path'])
    frames=[kick+i for i in [0,6,12,15,18,21,24,27,30]];tiles=[];records=[]
    for frame in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,img=cap.read()
        if not ok:raise ValueError(frame)
        o=obs.get(frame)
        if o:
            b=o['bbox'];kp=np.array(o['keypoints']);conf=np.array(o['keypoint_confidences'])
            cv2.rectangle(img,tuple(map(round,b[:2])),tuple(map(round,b[2:])),(0,220,255),2)
            for j in [0,5,6,11,12,15,16]:
                if conf[j]>=.35:cv2.circle(img,tuple(map(round,kp[j])),4,(0,0,255),-1)
            shoulder=kp[[5,6]].mean(axis=0);hip=kp[[11,12]].mean(axis=0);ankle=kp[[15,16]].mean(axis=0)
            records.append({'frame':frame,'bbox':b,'shoulder_minus_hip':(shoulder-hip).tolist(),
                            'shoulder_minus_ankle':(shoulder-ankle).tolist(),'kp_conf':conf[[5,6,11,12,15,16]].tolist()})
        img=img[170:470,210:650];img=cv2.resize(img,(440,300))
        cv2.putText(img,f'f{frame} {frame/r["fps"]:.2f}s '+('POSE' if o else 'NO POSE'),(8,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
        tiles.append(img)
    cap.release();out=ROOT/'outputs/goalkeeper_pose_direction_2026-10-11';out.mkdir(exist_ok=True)
    cv2.imwrite(str(out/f'{a.clip}_diagnostic.jpg'),np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,9,3)]))
    save(out/f'{a.clip}_pose_diagnostic.json',records)
    print(json.dumps(records,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
